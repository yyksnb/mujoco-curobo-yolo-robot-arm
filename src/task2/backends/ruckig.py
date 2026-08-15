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
        velocities = _waypoint_velocities(waypoints, self._model)
        knots: list[TrajectoryKnot] = [
            TrajectoryKnot(
                stage=waypoints[0].stage,
                time_s=0.0,
                positions_rad=waypoints[0].positions_rad,
                velocities_rad_s=velocities[0],
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
            generator = self._ruckig.Ruckig(dof)
            parameters = self._ruckig.InputParameter(dof)
            trajectory = self._ruckig.Trajectory(dof)
            motion_scale = right.motion_scale
            parameters.current_position = list(left.positions_rad)
            parameters.current_velocity = list(velocities[index])
            parameters.current_acceleration = [0.0] * dof
            parameters.target_position = list(right.positions_rad)
            parameters.target_velocity = list(velocities[index + 1])
            parameters.target_acceleration = [0.0] * dof
            parameters.max_position = [
                item.maximum_position_rad for item in self._model.joint_limits
            ]
            parameters.min_position = [
                item.minimum_position_rad for item in self._model.joint_limits
            ]
            parameters.max_velocity = [
                motion_scale * item.maximum_velocity_rad_s
                for item in self._model.joint_limits
            ]
            parameters.max_acceleration = [
                motion_scale**2 * item.maximum_acceleration_rad_s2
                for item in self._model.joint_limits
            ]
            parameters.max_jerk = [
                motion_scale**3 * item.maximum_jerk_rad_s3
                for item in self._model.joint_limits
            ]
            parameters.synchronization = self._ruckig.Synchronization.Phase
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
            sections = _continuous_sections(trajectory, dof)
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
                    velocities_rad_s=velocities[index + 1],
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


def _waypoint_velocities(
    waypoints: tuple[GeometricWaypoint, ...],
    robot_model: RobotModel,
) -> tuple[tuple[float, ...], ...]:
    zero = (0.0,) * len(robot_model.joint_names)
    velocities = []
    for index, waypoint in enumerate(waypoints):
        if index == 0 or index == len(waypoints) - 1 or waypoint.stop:
            velocities.append(zero)
            continue
        tangent = tuple(
            after - before
            for before, after in zip(
                waypoints[index - 1].positions_rad,
                waypoints[index + 1].positions_rad,
            )
        )
        moving = tuple(
            (abs(value), limit.maximum_velocity_rad_s)
            for value, limit in zip(tangent, robot_model.joint_limits)
            if abs(value) > 1e-12
        )
        if not moving:
            raise RuckigRouteError(
                waypoint.stage,
                f"Ruckig cannot construct a non-stop tangent at waypoint {index}",
            )
        boundary_motion_scale = min(
            waypoint.motion_scale,
            waypoints[index + 1].motion_scale,
        )
        scale = boundary_motion_scale * min(
            0.05 * limit / value for value, limit in moving
        )
        velocities.append(tuple(scale * value for value in tangent))
    return tuple(velocities)


def _continuous_sections(trajectory, dof: int) -> tuple[TrajectoryPolynomialSection, ...]:
    duration = float(trajectory.duration)
    boundaries = {0.0, duration}
    profiles = trajectory.profiles
    if len(profiles) != 1 or len(profiles[0]) != dof:
        raise BackendOperationError("Ruckig returned an unsupported multi-section profile")
    for profile in profiles[0]:
        elapsed = 0.0
        for profile_part in (profile.brake.t, profile.t, profile.accel.t):
            for interval in profile_part:
                elapsed += float(interval)
                if 0.0 < elapsed < duration:
                    boundaries.add(elapsed)
        if abs(elapsed - duration) > 1e-7:
            raise BackendOperationError("Ruckig profile duration does not match trajectory duration")
    ordered = sorted(boundaries)
    sections = []
    for start, end in zip(ordered, ordered[1:]):
        section_duration = end - start
        if section_duration <= 1e-12:
            continue
        start_position, start_velocity, start_acceleration = trajectory.at_time(start)
        end_position, end_velocity, end_acceleration = trajectory.at_time(end)
        coefficients = []
        for joint in range(dof):
            acceleration = float(start_acceleration[joint])
            jerk = (float(end_acceleration[joint]) - acceleration) / section_duration
            values = (
                float(start_position[joint]),
                float(start_velocity[joint]),
                0.5 * acceleration,
                jerk / 6.0,
            )
            coefficients.append(values)
            predicted_position = values[0] + section_duration * (
                values[1] + section_duration * (values[2] + section_duration * values[3])
            )
            predicted_velocity = values[1] + section_duration * (
                2.0 * values[2] + section_duration * 3.0 * values[3]
            )
            if (
                abs(predicted_position - float(end_position[joint])) > 1e-7
                or abs(predicted_velocity - float(end_velocity[joint])) > 1e-7
            ):
                raise BackendOperationError("Ruckig profile could not be represented continuously")
        sections.append(
            TrajectoryPolynomialSection(start, section_duration, tuple(coefficients))
        )
    if not sections:
        raise BackendOperationError("Ruckig returned no continuous trajectory sections")
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
