from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from math import isfinite
from typing import Mapping

from ..contracts import (
    PlannedTrajectory,
    RobotModel,
    TrajectoryEvent,
    TrajectoryKnot,
    TrajectoryPhase,
    TrajectoryPolynomialSection,
    TrajectorySegment,
    TrajectoryStage,
)
from .errors import BackendOperationError, BackendUnavailableError


@dataclass(frozen=True)
class GeometricWaypoint:
    stage: TrajectoryStage
    positions_rad: tuple[float, ...]
    actuator_channels: dict[str, float]
    stop: bool = False
    minimum_duration_s: float = 0.0
    motion_scale: float = 1.0


@dataclass(frozen=True)
class RuckigConfig:
    version: str

    def validate(self) -> None:
        if not self.version:
            raise ValueError("Ruckig version must be explicit")


class RuckigRouteError(BackendOperationError):
    def __init__(self, stage: TrajectoryStage, message: str):
        super().__init__(message)
        self.stage = stage


def parse_ruckig_config(payload: Mapping[str, object]) -> RuckigConfig:
    version = payload.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("Ruckig.version must be a non-empty string")
    result = RuckigConfig(version)
    result.validate()
    return result


class RuckigTimeParameterizer:
    def __init__(self, robot_model: RobotModel, config: RuckigConfig):
        config.validate()
        try:
            import ruckig
        except (ImportError, OSError) as exc:
            raise BackendUnavailableError(f"Ruckig import failed: {exc}") from exc
        try:
            installed = metadata.version("ruckig")
        except metadata.PackageNotFoundError as exc:
            raise BackendUnavailableError("Ruckig package metadata is unavailable") from exc
        if installed != config.version:
            raise BackendUnavailableError(
                f"Ruckig version {installed} does not match required {config.version}"
            )
        self._ruckig = ruckig
        self._model = robot_model
        self._config = config
        self.version = installed

    def parameterize(
        self,
        waypoints: tuple[GeometricWaypoint, ...],
        events: tuple[tuple[str, TrajectoryStage, int, Mapping[str, float]], ...],
    ) -> PlannedTrajectory:
        if len(waypoints) < 2:
            raise BackendOperationError("Ruckig requires at least two geometric waypoints")
        dof = len(self._model.joint_names)
        if any(len(item.positions_rad) != dof for item in waypoints):
            raise BackendOperationError("geometric waypoint size does not match RobotModel")
        if any(
            not isfinite(item.motion_scale) or not 0.0 < item.motion_scale <= 1.0
            for item in waypoints
        ):
            raise BackendOperationError("geometric waypoint motion scales must lie in (0, 1]")
        zero_velocity = (0.0,) * dof
        knots: list[TrajectoryKnot] = [
            TrajectoryKnot(
                stage=waypoints[0].stage,
                time_s=0.0,
                positions_rad=waypoints[0].positions_rad,
                velocities_rad_s=zero_velocity,
                accelerations_rad_s2=(0.0,) * dof,
                actuator_channels=dict(waypoints[0].actuator_channels),
            )
        ]
        segments: list[TrajectorySegment] = []
        geometric_times = [0.0]
        current_time = 0.0
        for index, (left, right) in enumerate(zip(waypoints, waypoints[1:])):
            if set(left.actuator_channels) != set(right.actuator_channels):
                raise BackendOperationError(
                    "actuator channel names must remain stable across trajectory"
                )
            delta = tuple(
                target - source
                for source, target in zip(
                    left.positions_rad,
                    right.positions_rad,
                    strict=True,
                )
            )
            motion_scale = right.motion_scale
            moving = tuple(abs(value) > 1e-12 for value in delta)
            if not any(moving):
                duration = right.minimum_duration_s
                if duration <= 0.0:
                    raise RuckigRouteError(
                        right.stage,
                        f"zero-motion geometric segment {index} has no duration",
                    )
                sections = (
                    TrajectoryPolynomialSection(
                        0.0,
                        duration,
                        tuple(
                            (value, 0.0, 0.0, 0.0)
                            for value in left.positions_rad
                        ),
                    ),
                )
            else:
                generator = self._ruckig.Ruckig(1)
                parameters = self._ruckig.InputParameter(1)
                trajectory = self._ruckig.Trajectory(1)
                parameters.current_position = [0.0]
                parameters.current_velocity = [0.0]
                parameters.current_acceleration = [0.0]
                parameters.target_position = [1.0]
                parameters.target_velocity = [0.0]
                parameters.target_acceleration = [0.0]
                parameters.max_position = [1.0]
                parameters.min_position = [0.0]
                parameters.max_velocity = [
                    min(
                        motion_scale * limit.maximum_velocity_rad_s / abs(value)
                        for value, limit, active in zip(
                            delta, self._model.joint_limits, moving, strict=True
                        )
                        if active
                    )
                ]
                parameters.max_acceleration = [
                    min(
                        motion_scale**2
                        * limit.maximum_acceleration_rad_s2
                        / abs(value)
                        for value, limit, active in zip(
                            delta, self._model.joint_limits, moving, strict=True
                        )
                        if active
                    )
                ]
                parameters.max_jerk = [
                    min(
                        motion_scale**3 * limit.maximum_jerk_rad_s3 / abs(value)
                        for value, limit, active in zip(
                            delta, self._model.joint_limits, moving, strict=True
                        )
                        if active
                    )
                ]
                if right.minimum_duration_s > 0.0:
                    parameters.minimum_duration = right.minimum_duration_s
                status = generator.calculate(parameters, trajectory)
                if status != self._ruckig.Result.Working:
                    raise RuckigRouteError(
                        right.stage,
                        f"Ruckig rejected geometric segment {index}: {status}",
                    )
                duration = float(trajectory.duration)
                if duration <= 0.0:
                    raise RuckigRouteError(
                        right.stage,
                        f"Ruckig produced a non-positive duration for segment {index}",
                    )
                sections = _path_progress_sections(
                    trajectory,
                    left.positions_rad,
                    delta,
                )
            segments.append(
                TrajectorySegment(
                    stage=right.stage,
                    start_time_s=current_time,
                    duration_s=duration,
                    start_knot=index,
                    end_knot=index + 1,
                    sections=sections,
                    start_actuator_channels=dict(left.actuator_channels),
                    end_actuator_channels=dict(right.actuator_channels),
                )
            )
            current_time += duration
            geometric_times.append(current_time)
            knots.append(
                TrajectoryKnot(
                    stage=right.stage,
                    time_s=current_time,
                    positions_rad=right.positions_rad,
                    velocities_rad_s=zero_velocity,
                    accelerations_rad_s2=(0.0,) * dof,
                    actuator_channels=dict(right.actuator_channels),
                )
            )
        phases = _build_phases(knots, segments)
        trajectory_events = tuple(
            TrajectoryEvent(event, stage, geometric_times[waypoint_index], dict(channels))
            for event, stage, waypoint_index, channels in events
        )
        return PlannedTrajectory(
            joint_names=self._model.joint_names,
            knots=tuple(knots),
            segments=tuple(segments),
            phases=phases,
            events=trajectory_events,
        )


def _path_progress_sections(
    trajectory,
    source: tuple[float, ...],
    delta: tuple[float, ...],
) -> tuple[TrajectoryPolynomialSection, ...]:
    duration = float(trajectory.duration)
    boundaries = {0.0, duration}
    profiles = trajectory.profiles
    if len(profiles) != 1 or len(profiles[0]) != 1:
        raise BackendOperationError("Ruckig returned an unsupported path-progress profile")
    profile = profiles[0][0]
    elapsed = 0.0
    for profile_part in (profile.brake.t, profile.t, profile.accel.t):
        for interval in profile_part:
            elapsed += float(interval)
            if 0.0 < elapsed < duration:
                boundaries.add(elapsed)
    if abs(elapsed - duration) > 1e-7:
        raise BackendOperationError("Ruckig profile duration does not match trajectory duration")

    sections = []
    ordered = sorted(boundaries)
    for start, end in zip(ordered, ordered[1:]):
        section_duration = end - start
        if section_duration <= 1e-12:
            continue
        start_position, start_velocity, start_acceleration = trajectory.at_time(start)
        _, _, end_acceleration = trajectory.at_time(end)
        progress = float(start_position[0])
        progress_velocity = float(start_velocity[0])
        progress_acceleration = float(start_acceleration[0])
        progress_jerk = (
            float(end_acceleration[0]) - progress_acceleration
        ) / section_duration
        coefficients = tuple(
            (
                origin + joint_delta * progress,
                joint_delta * progress_velocity,
                0.5 * joint_delta * progress_acceleration,
                joint_delta * progress_jerk / 6.0,
            )
            for origin, joint_delta in zip(source, delta, strict=True)
        )
        sections.append(
            TrajectoryPolynomialSection(start, section_duration, coefficients)
        )
    if not sections:
        raise BackendOperationError("Ruckig returned no path-progress trajectory sections")
    return tuple(sections)


def _build_phases(
    knots: list[TrajectoryKnot], segments: list[TrajectorySegment]
) -> tuple[TrajectoryPhase, ...]:
    phases = []
    for stage in TrajectoryStage:
        indices = [index for index, segment in enumerate(segments) if segment.stage is stage]
        if not indices:
            raise BackendOperationError(f"time-parameterized trajectory is missing {stage.value}")
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise BackendOperationError(f"time-parameterized {stage.value} phase is not contiguous")
        first = segments[indices[0]]
        last = segments[indices[-1]]
        phases.append(
            TrajectoryPhase(
                stage,
                first.start_knot,
                last.end_knot,
                indices[0],
                indices[-1],
                first.start_time_s,
                last.start_time_s + last.duration_s,
            )
        )
    return tuple(phases)
