from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Mapping, Protocol

from .geometry import RigidTransform, Vector3
from .planning import PlanningResult, TrajectoryStage


class ExecutionErrorCode(str, Enum):
    INVALID_REQUEST = "invalid_request"
    ADAPTER_UNAVAILABLE = "adapter_unavailable"
    MODEL_LOAD_FAILED = "model_load_failed"
    CONTROL_INTERFACE_FAILED = "control_interface_failed"
    NUMERICAL_SIMULATION_FAILED = "numerical_simulation_failed"
    LOG_IO_FAILED = "log_io_failed"
    ADAPTER_ERROR = "adapter_error"


class ExecutionStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class SceneObjectBinding:
    object_id: str
    object_type: str
    T_world_object: RigidTransform

    def validate(self, world_frame: str) -> None:
        if not self.object_id or not self.object_type:
            raise ValueError("Execution scene object ID and type must be non-empty")
        self.T_world_object.validate(label=f"Execution object {self.object_id} pose")
        if (
            self.T_world_object.parent_frame != world_frame
            or self.T_world_object.child_frame != self.object_id
        ):
            raise ValueError("Execution scene object pose frames do not match its binding")


@dataclass(frozen=True)
class DynamicsDiagnosticsConfig:
    world_frame: str
    source_support_geom_names: tuple[str, ...]
    minimum_departure_height_m: float
    target_position_world_m: Vector3
    target_position_tolerance_m: float
    post_release_observation_s: float
    settled_linear_speed_m_s: float
    settled_angular_speed_rad_s: float

    def validate(self) -> None:
        if not self.world_frame or not self.source_support_geom_names or any(
            not name for name in self.source_support_geom_names
        ):
            raise ValueError("dynamics diagnostics world frame and source support must be explicit")
        if len(self.source_support_geom_names) != len(set(self.source_support_geom_names)):
            raise ValueError("dynamics source support geom names must be unique")
        if len(self.target_position_world_m) != 3 or not all(
            isfinite(value) for value in self.target_position_world_m
        ):
            raise ValueError("dynamics target position must be a finite xyz vector")
        positive = (
            self.minimum_departure_height_m,
            self.target_position_tolerance_m,
            self.settled_linear_speed_m_s,
            self.settled_angular_speed_rad_s,
        )
        if not all(isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("dynamics diagnostic thresholds must be finite and positive")
        if not isfinite(self.post_release_observation_s) or self.post_release_observation_s < 0.0:
            raise ValueError("post-release observation duration must be finite and non-negative")


@dataclass(frozen=True)
class ContactRecord:
    first_geom: str
    second_geom: str
    maximum_penetration_m: float
    maximum_normal_force_n: float
    maximum_impact_impulse_ns: float

    def normalized_pair(self) -> tuple[str, str]:
        return tuple(sorted((self.first_geom, self.second_geom)))  # type: ignore[return-value]

    def to_dict(self) -> dict[str, object]:
        return {
            "first_geom": self.first_geom,
            "second_geom": self.second_geom,
            "maximum_penetration_m": self.maximum_penetration_m,
            "maximum_normal_force_n": self.maximum_normal_force_n,
            "maximum_impact_impulse_ns": self.maximum_impact_impulse_ns,
        }


@dataclass(frozen=True)
class BodyKinematicState:
    T_world_body: RigidTransform
    linear_velocity_m_s: Vector3
    angular_velocity_rad_s: Vector3

    def to_dict(self) -> dict[str, object]:
        return {
            "T_world_body": _transform_to_dict(self.T_world_body),
            "linear_velocity_m_s": list(self.linear_velocity_m_s),
            "angular_velocity_rad_s": list(self.angular_velocity_rad_s),
        }


@dataclass(frozen=True)
class RuntimeObservation:
    runtime_time_s: float
    joint_positions_rad: Mapping[str, float]
    actuator_states: Mapping[str, float]
    T_world_tcp: RigidTransform
    target_object_state: BodyKinematicState | None
    contacts: tuple[ContactRecord, ...]


@dataclass(frozen=True)
class ControlCommand:
    stage: TrajectoryStage
    planned_time_s: float
    joint_names: tuple[str, ...]
    positions_rad: tuple[float, ...]
    velocities_rad_s: tuple[float, ...]
    accelerations_rad_s2: tuple[float, ...]
    actuator_channels: Mapping[str, float]
    events: tuple[str, ...] = ()


class ExecutorAdapter(Protocol):
    adapter_id: str

    def initialize(
        self,
        scene_objects: tuple[SceneObjectBinding, ...],
        target_object_id: str,
        initial_command: ControlCommand,
    ) -> RuntimeObservation: ...

    def apply_control(self, command: ControlCommand) -> None: ...

    def advance(self, duration_s: float) -> RuntimeObservation: ...

    def observe(self) -> RuntimeObservation: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class ExecutionRequest:
    request_id: str
    planning_result: PlanningResult
    adapter: ExecutorAdapter
    scene_objects: tuple[SceneObjectBinding, ...]
    target_object_id: str
    control_frequency_hz: float
    control_log_frequency_hz: float
    diagnostic_log_frequency_hz: float
    control_log_path: str
    diagnostic_log_path: str
    diagnostics_config: DynamicsDiagnosticsConfig

    def validate(self) -> None:
        if not self.request_id or not self.target_object_id:
            raise ValueError("Execution request and target object IDs must be explicit")
        if not self.planning_result.success or self.planning_result.trajectory is None:
            raise ValueError("Execution requires a successful PlanningResult with a trajectory")
        trajectory = self.planning_result.trajectory
        if not trajectory.joint_names or len(trajectory.joint_names) != len(set(trajectory.joint_names)):
            raise ValueError("Execution trajectory joint names must be unique and non-empty")
        if not trajectory.segments or not trajectory.knots:
            raise ValueError("Execution trajectory must contain knots and segments")
        if tuple(phase.stage for phase in trajectory.phases) != tuple(TrajectoryStage):
            raise ValueError("Execution trajectory must contain the ordered six Planning phases")
        segment_stages = tuple(dict.fromkeys(segment.stage for segment in trajectory.segments))
        if segment_stages != tuple(TrajectoryStage):
            raise ValueError("Execution trajectory segments must cover the ordered six Planning phases")
        expected_start = 0.0
        for segment in trajectory.segments:
            if abs(segment.start_time_s - expected_start) > 1e-8 or segment.duration_s <= 0.0:
                raise ValueError("Execution trajectory segments must be positive and contiguous")
            expected_start = segment.start_time_s + segment.duration_s
        release_events = [event for event in trajectory.events if event.event == "release"]
        if len(release_events) != 1 or abs(release_events[0].time_s - expected_start) > 1e-8:
            raise ValueError("Execution trajectory requires one release event at its endpoint")
        if not getattr(self.adapter, "adapter_id", ""):
            raise ValueError("Execution adapter must expose a non-empty adapter_id")
        if not self.scene_objects:
            raise ValueError("Execution requires scene object bindings")
        object_ids = [item.object_id for item in self.scene_objects]
        if len(object_ids) != len(set(object_ids)) or self.target_object_id not in object_ids:
            raise ValueError("Execution scene object IDs must be unique and include the target")
        self.diagnostics_config.validate()
        for item in self.scene_objects:
            item.validate(self.diagnostics_config.world_frame)
        if not all(
            isfinite(value) and value > 0.0
            for value in (
                self.control_frequency_hz,
                self.control_log_frequency_hz,
                self.diagnostic_log_frequency_hz,
            )
        ):
            raise ValueError("Execution control and log frequencies must be positive")
        if self.control_log_frequency_hz > self.control_frequency_hz:
            raise ValueError(
                "Execution control log frequency cannot exceed the control frequency"
            )
        if not self.control_log_path or not self.diagnostic_log_path:
            raise ValueError("Execution control and diagnostic log paths must be explicit")
        if self.control_log_path == self.diagnostic_log_path:
            raise ValueError("Execution control and diagnostic logs must use separate paths")


@dataclass(frozen=True)
class DynamicsDiagnostics:
    evaluated: bool
    success: bool
    target_object_present: bool
    departed_source_support: bool
    pre_release_state: BodyKinematicState | None
    post_release_state: BodyKinematicState | None
    stable_after_release: bool
    target_position_error_m: float | None
    contact_pairs: tuple[ContactRecord, ...]
    maximum_penetration_m: float
    maximum_normal_force_n: float
    maximum_impact_impulse_ns: float
    failure_reasons: tuple[str, ...]

    @classmethod
    def not_evaluated(cls, reason: str) -> DynamicsDiagnostics:
        return cls(
            False,
            False,
            False,
            False,
            None,
            None,
            False,
            None,
            (),
            0.0,
            0.0,
            0.0,
            (reason,),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "evaluated": self.evaluated,
            "success": self.success,
            "target_object_present": self.target_object_present,
            "departed_source_support": self.departed_source_support,
            "pre_release_state": (
                self.pre_release_state.to_dict() if self.pre_release_state is not None else None
            ),
            "post_release_state": (
                self.post_release_state.to_dict() if self.post_release_state is not None else None
            ),
            "stable_after_release": self.stable_after_release,
            "target_position_error_m": self.target_position_error_m,
            "contact_pairs": [item.to_dict() for item in self.contact_pairs],
            "maximum_penetration_m": self.maximum_penetration_m,
            "maximum_normal_force_n": self.maximum_normal_force_n,
            "maximum_impact_impulse_ns": self.maximum_impact_impulse_ns,
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass(frozen=True)
class ExecutionResult:
    execution_status: ExecutionStatus
    error_code: ExecutionErrorCode | None
    message: str
    dynamics_diagnostics: DynamicsDiagnostics
    planned_duration_s: float
    executed_planned_duration_s: float
    control_sample_count: int
    control_log_sample_count: int
    diagnostic_log_sample_count: int
    elapsed_s: float

    @property
    def success(self) -> bool:
        return self.execution_status is ExecutionStatus.COMPLETED

    def to_dict(self) -> dict[str, object]:
        return {
            "success": self.success,
            "execution_status": self.execution_status.value,
            "error_code": self.error_code.value if self.error_code is not None else None,
            "message": self.message,
            "dynamics_diagnostics": self.dynamics_diagnostics.to_dict(),
            "planned_duration_s": self.planned_duration_s,
            "executed_planned_duration_s": self.executed_planned_duration_s,
            "control_sample_count": self.control_sample_count,
            "control_log_sample_count": self.control_log_sample_count,
            "diagnostic_log_sample_count": self.diagnostic_log_sample_count,
            "elapsed_s": self.elapsed_s,
        }


def _transform_to_dict(transform: RigidTransform) -> dict[str, object]:
    return {
        "parent_frame": transform.parent_frame,
        "child_frame": transform.child_frame,
        "matrix": [list(row) for row in transform.matrix],
    }
