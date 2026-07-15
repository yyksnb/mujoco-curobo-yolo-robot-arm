from __future__ import annotations

import importlib.util
import math
import time
from copy import deepcopy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from robot_arm_pipeline.planning.curobo_conversions import transform_to_curobo_pose
from robot_arm_pipeline.types import (
    PlannedTrajectory,
    PlanningRequest,
    PlanningResult,
    RobotState,
    TrajectoryWaypoint,
)

DEFAULT_ROBOT_CONFIG = Path("configs/curobo/gen3/robot.yml")
DEFAULT_WORLD_CONFIG = Path("configs/curobo/gen3/world.yml")
DEFAULT_GRAPH_CONFIG = Path("configs/curobo/gen3/graph.yml")
MOTION_PLAN_RESULT_SCHEMA = "motion_plan_result"
_LEGACY_CAMERA_ROUTE_SCHEMA = "camera_route_plan"
MotionPlanningStrategy = Literal["direct_pose", "cartesian_continuation"]


def filter_feasible_graph_goals(
    graph_planner: Any,
    start_position: Any,
    goal_positions: Any,
) -> tuple[Any, Any] | None:
    start = start_position.reshape(1, goal_positions.shape[-1])
    if not _all_true(graph_planner.check_samples_feasibility(start)):
        return None
    goals = goal_positions[graph_planner.check_samples_feasibility(goal_positions)]
    if goals.shape[0] == 0:
        return None
    return start[[0] * goals.shape[0]], goals


def _all_true(values: Any) -> bool:
    result = values.all()
    return bool(result.item() if hasattr(result, "item") else result)


def create_collision_filtered_motion_planner(config: Any) -> Any:
    from curobo._src.graph_planner.graph_planner_prm import TrajInterpolationType
    from curobo.motion_planner import MotionPlanner

    class CollisionFilteredMotionPlanner(MotionPlanner):
        def _get_graph_seed_trajectories(self, current_state: Any, seed_config: Any) -> Any:
            dof = self.trajopt_solver.action_dim
            graph_goals = seed_config.reshape(-1, dof)
            queries = filter_feasible_graph_goals(
                self.graph_planner,
                current_state.position,
                graph_goals,
            )
            if queries is None:
                return None
            graph_starts, graph_goals = queries
            result = self.graph_planner.find_path(
                graph_starts.clone(),
                graph_goals.clone(),
                interpolate_waypoints=True,
                interpolation_steps=self.trajopt_solver.action_horizon,
                interpolation_type=TrajInterpolationType.LINEAR,
                validate_interpolated_trajectory=False,
            )
            if not bool(result.success.any().item()):
                return None
            return result.interpolated_waypoints[result.success, :, :].unsqueeze(0)

    return CollisionFilteredMotionPlanner(config)


def _require_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be an object with string keys")
    return value


def _require_keys(value: Mapping[str, object], expected: set[str], context: str) -> None:
    missing = expected - value.keys()
    unknown = value.keys() - expected
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing={sorted(missing)}")
        if unknown:
            details.append(f"unknown={sorted(unknown)}")
        raise ValueError(f"{context} has invalid fields: {', '.join(details)}")


def _require_string(value: object, context: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{context} must be a{' possibly empty' if allow_empty else ' non-empty'} string")
    return value


def _require_bool(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{context} must be a boolean")
    return value


def _require_int(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{context} must be an integer >= {minimum}")
    return value


def _require_float(value: object, context: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        suffix = f" >= {minimum}" if minimum is not None else ""
        raise ValueError(f"{context} must be a finite number{suffix}")
    return result


def _require_optional_float(
    value: object, context: str, *, minimum: float | None = None
) -> float | None:
    return None if value is None else _require_float(value, context, minimum=minimum)


def _require_sequence(value: object, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{context} must be an array")
    return value


def _float_tuple(value: object, context: str) -> tuple[float, ...]:
    return tuple(
        _require_float(item, f"{context}[{index}]")
        for index, item in enumerate(_require_sequence(value, context))
    )


def _float_matrix(value: object, context: str) -> tuple[tuple[float, ...], ...]:
    return tuple(
        _float_tuple(row, f"{context}[{index}]")
        for index, row in enumerate(_require_sequence(value, context))
    )


@dataclass(frozen=True)
class PoseTarget:
    """A tool pose expressed in the configured cuRobo robot base frame."""

    target_id: str
    target_position: tuple[float, float, float]
    target_quaternion_wxyz: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        _require_string(self.target_id, "pose route target_id")
        if len(self.target_position) != 3:
            raise ValueError("pose route target_position must contain three values")
        if len(self.target_quaternion_wxyz) != 4:
            raise ValueError("pose route target_quaternion_wxyz must contain four values")
        position = tuple(
            _require_float(value, f"pose route target_position[{index}]")
            for index, value in enumerate(self.target_position)
        )
        quaternion = tuple(
            _require_float(value, f"pose route target_quaternion_wxyz[{index}]")
            for index, value in enumerate(self.target_quaternion_wxyz)
        )
        if not math.isclose(sum(value * value for value in quaternion), 1.0, abs_tol=1e-5):
            raise ValueError("pose route target quaternion must be normalized")
        object.__setattr__(self, "target_position", position)
        object.__setattr__(self, "target_quaternion_wxyz", quaternion)


@dataclass(frozen=True)
class MotionPlanningPolicy:
    max_attempts: int
    enable_graph_attempt: int
    num_ik_seeds: int
    num_trajopt_seeds: int
    random_seed: int
    position_tolerance: float
    orientation_tolerance: float
    enable_cartesian_continuation: bool = False
    continuation_offset_m: float = 0.10
    continuation_step_m: float = 0.005
    continuation_edge_sample_count: int = 11
    continuation_ik_solution_count: int = 8
    continuation_finetune_attempts: int = 3
    continuation_joint_tolerance_rad: float = 0.001
    continuation_stop_velocity_tolerance_rad_s: float = 0.01

    def __post_init__(self) -> None:
        _require_int(self.max_attempts, "pose route max_attempts", minimum=1)
        _require_int(
            self.enable_graph_attempt,
            "pose route enable_graph_attempt",
        )
        _require_int(self.num_ik_seeds, "pose route num_ik_seeds", minimum=1)
        _require_int(
            self.num_trajopt_seeds,
            "pose route num_trajopt_seeds",
            minimum=1,
        )
        _require_int(self.random_seed, "pose route random_seed")
        object.__setattr__(
            self,
            "position_tolerance",
            _require_float(
                self.position_tolerance,
                "pose route position_tolerance",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "orientation_tolerance",
            _require_float(
                self.orientation_tolerance,
                "pose route orientation_tolerance",
                minimum=0.0,
            ),
        )
        if self.position_tolerance == 0.0 or self.orientation_tolerance == 0.0:
            raise ValueError("pose route planning tolerances must be positive")
        if not isinstance(self.enable_cartesian_continuation, bool):
            raise ValueError("pose route enable_cartesian_continuation must be a boolean")
        object.__setattr__(
            self,
            "continuation_offset_m",
            _require_float(
                self.continuation_offset_m,
                "pose route continuation_offset_m",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "continuation_step_m",
            _require_float(
                self.continuation_step_m,
                "pose route continuation_step_m",
                minimum=0.0,
            ),
        )
        _require_int(
            self.continuation_edge_sample_count,
            "pose route continuation_edge_sample_count",
            minimum=2,
        )
        _require_int(
            self.continuation_ik_solution_count,
            "pose route continuation_ik_solution_count",
            minimum=1,
        )
        _require_int(
            self.continuation_finetune_attempts,
            "pose route continuation_finetune_attempts",
            minimum=1,
        )
        object.__setattr__(
            self,
            "continuation_joint_tolerance_rad",
            _require_float(
                self.continuation_joint_tolerance_rad,
                "pose route continuation_joint_tolerance_rad",
                minimum=0.0,
            ),
        )
        if self.continuation_offset_m == 0.0 or self.continuation_step_m == 0.0:
            raise ValueError("Cartesian continuation distances must be positive")
        if self.continuation_joint_tolerance_rad == 0.0:
            raise ValueError("pose route continuation joint tolerance must be positive")
        object.__setattr__(
            self,
            "continuation_stop_velocity_tolerance_rad_s",
            _require_float(
                self.continuation_stop_velocity_tolerance_rad_s,
                "pose route continuation_stop_velocity_tolerance_rad_s",
                minimum=0.0,
            ),
        )
        if self.continuation_stop_velocity_tolerance_rad_s == 0.0:
            raise ValueError("pose route continuation stop velocity tolerance must be positive")


@dataclass(frozen=True)
class IKSolution:
    target_id: str
    joint_positions: tuple[float, ...]

    def __post_init__(self) -> None:
        _require_string(self.target_id, "pose target IK target_id")
        if not self.joint_positions:
            raise ValueError("pose target IK joint_positions must not be empty")
        object.__setattr__(
            self,
            "joint_positions",
            tuple(
                _require_float(value, f"pose target IK joint_positions[{index}]")
                for index, value in enumerate(self.joint_positions)
            ),
        )


@dataclass(frozen=True)
class MotionPlanSegment:
    target_id: str
    success: bool
    message: str
    planning_time_s: float
    waypoint_count: int
    trajectory: tuple[tuple[float, ...], ...] = ()
    trajectory_time_s: tuple[float, ...] | None = None
    trajectory_velocity: tuple[tuple[float, ...], ...] | None = None
    target_position_error_m: float | None = None
    target_orientation_error_rad: float | None = None
    planning_strategy: str = "direct_pose"
    continuation_offset_m: float | None = None

    def __post_init__(self) -> None:
        _require_string(self.target_id, "pose route segment target_id")
        if not isinstance(self.success, bool):
            raise ValueError("pose route segment success must be a boolean")
        _require_string(self.message, "pose route segment message", allow_empty=True)
        _require_float(self.planning_time_s, "pose route segment planning_time_s", minimum=0.0)
        _require_int(self.waypoint_count, "pose route segment waypoint_count")
        if self.waypoint_count != len(self.trajectory):
            raise ValueError("pose route segment waypoint_count must match trajectory length")
        widths = {len(row) for row in self.trajectory}
        if len(widths) > 1 or (self.trajectory and 0 in widths):
            raise ValueError("pose route segment trajectory rows must have one non-zero width")
        for row_index, row in enumerate(self.trajectory):
            for value_index, value in enumerate(row):
                _require_float(value, f"pose route segment trajectory[{row_index}][{value_index}]")
        if self.trajectory_time_s is not None:
            if len(self.trajectory_time_s) != self.waypoint_count:
                raise ValueError("pose route segment trajectory_time_s must match trajectory length")
            previous = -math.inf
            for index, value in enumerate(self.trajectory_time_s):
                current = _require_float(value, f"trajectory_time_s[{index}]", minimum=0.0)
                if current <= previous:
                    raise ValueError("pose route segment trajectory_time_s must be strictly increasing")
                previous = current
        if self.trajectory_velocity is not None:
            if len(self.trajectory_velocity) != self.waypoint_count:
                raise ValueError("pose route segment trajectory_velocity must match trajectory length")
            expected_width = next(iter(widths), 0)
            for index, row in enumerate(self.trajectory_velocity):
                if len(row) != expected_width:
                    raise ValueError("pose route segment velocity rows must match trajectory width")
                for column, value in enumerate(row):
                    _require_float(value, f"trajectory_velocity[{index}][{column}]")
        _require_optional_float(
            self.target_position_error_m, "target_position_error_m", minimum=0.0
        )
        _require_optional_float(
            self.target_orientation_error_rad, "target_orientation_error_rad", minimum=0.0
        )
        _require_string(self.planning_strategy, "pose route segment planning_strategy")
        if self.planning_strategy not in {"direct_pose", "cartesian_continuation"}:
            raise ValueError("pose route segment planning_strategy is unsupported")
        _require_optional_float(self.continuation_offset_m, "continuation_offset_m", minimum=0.0)
        if self.planning_strategy == "direct_pose" and self.continuation_offset_m is not None:
            raise ValueError("direct pose route segment must not contain continuation_offset_m")
        if self.planning_strategy == "cartesian_continuation" and self.continuation_offset_m is None:
            raise ValueError(
                "cartesian continuation segment must contain continuation_offset_m"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "success": self.success,
            "message": self.message,
            "planning_time_s": self.planning_time_s,
            "waypoint_count": self.waypoint_count,
            "trajectory": [list(waypoint) for waypoint in self.trajectory],
            "trajectory_time_s": (
                list(self.trajectory_time_s) if self.trajectory_time_s is not None else None
            ),
            "trajectory_velocity": (
                [list(waypoint) for waypoint in self.trajectory_velocity]
                if self.trajectory_velocity is not None
                else None
            ),
            "target_position_error_m": self.target_position_error_m,
            "target_orientation_error_rad": self.target_orientation_error_rad,
            "planning_strategy": self.planning_strategy,
            "continuation_offset_m": self.continuation_offset_m,
        }

    @classmethod
    def from_dict(cls, payload: object) -> MotionPlanSegment:
        value = _require_mapping(payload, "pose route segment")
        required_fields = {
            "target_id", "success", "message", "planning_time_s", "waypoint_count",
            "trajectory", "trajectory_time_s", "trajectory_velocity",
            "target_position_error_m", "target_orientation_error_rad",
        }
        optional_fields = {
            "planning_strategy",
            "continuation_offset_m",
            "portal_offset_m",
        }
        missing = required_fields - value.keys()
        unknown = value.keys() - required_fields - optional_fields
        if missing or unknown:
            details = []
            if missing:
                details.append(f"missing={sorted(missing)}")
            if unknown:
                details.append(f"unknown={sorted(unknown)}")
            raise ValueError(
                f"pose route segment has invalid fields: {', '.join(details)}"
            )
        trajectory = _float_matrix(value["trajectory"], "trajectory")
        raw_time = value["trajectory_time_s"]
        raw_velocity = value["trajectory_velocity"]
        raw_strategy = _require_string(
            value.get("planning_strategy", "direct_pose"),
            "planning_strategy",
        )
        planning_strategy = (
            "cartesian_continuation"
            if raw_strategy == "portal_continuation"
            else raw_strategy
        )
        if (
            "continuation_offset_m" in value
            and "portal_offset_m" in value
        ):
            raise ValueError(
                "pose route segment cannot contain both continuation offset fields"
            )
        raw_continuation_offset = value.get(
            "continuation_offset_m", value.get("portal_offset_m")
        )
        return cls(
            target_id=_require_string(value["target_id"], "target_id"),
            success=_require_bool(value["success"], "success"),
            message=_require_string(value["message"], "message", allow_empty=True),
            planning_time_s=_require_float(value["planning_time_s"], "planning_time_s", minimum=0.0),
            waypoint_count=_require_int(value["waypoint_count"], "waypoint_count"),
            trajectory=trajectory,
            trajectory_time_s=None if raw_time is None else _float_tuple(raw_time, "trajectory_time_s"),
            trajectory_velocity=None if raw_velocity is None else _float_matrix(raw_velocity, "trajectory_velocity"),
            target_position_error_m=_require_optional_float(value["target_position_error_m"], "target_position_error_m", minimum=0.0),
            target_orientation_error_rad=_require_optional_float(value["target_orientation_error_rad"], "target_orientation_error_rad", minimum=0.0),
            planning_strategy=planning_strategy,
            continuation_offset_m=_require_optional_float(
                raw_continuation_offset,
                "continuation_offset_m",
                minimum=0.0,
            ),
        )


@dataclass(frozen=True)
class MotionPlanResult:
    success: bool
    planner_name: str
    joint_names: tuple[str, ...]
    segments: tuple[MotionPlanSegment, ...]
    failed_target_id: str | None
    message: str
    reached_target_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise ValueError("pose route plan success must be a boolean")
        _require_string(self.planner_name, "pose route planner_name")
        _require_string(self.message, "pose route plan message", allow_empty=True)
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("pose route plan joint_names must be non-empty and unique")
        for index, name in enumerate(self.joint_names):
            _require_string(name, f"joint_names[{index}]")
        target_ids = [segment.target_id for segment in self.segments]
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("pose route plan segment target_ids must be unique")
        for segment in self.segments:
            if any(len(row) != len(self.joint_names) for row in segment.trajectory):
                raise ValueError("pose route segment trajectory width must match joint_names")
        if self.failed_target_id is not None:
            _require_string(self.failed_target_id, "failed_target_id")
        if len(set(self.reached_target_ids)) != len(self.reached_target_ids):
            raise ValueError("pose route plan reached_target_ids must be unique")
        if any(target_id not in target_ids for target_id in self.reached_target_ids):
            raise ValueError("pose route plan reached_target_ids must reference segments")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": MOTION_PLAN_RESULT_SCHEMA,
            "success": self.success,
            "planner_name": self.planner_name,
            "joint_names": list(self.joint_names),
            "segments": [segment.to_dict() for segment in self.segments],
            "failed_target_id": self.failed_target_id,
            "message": self.message,
            "reached_target_ids": list(self.reached_target_ids),
        }

    @classmethod
    def from_dict(cls, payload: object) -> MotionPlanResult:
        value = _require_mapping(payload, "pose route plan")
        _require_keys(
            value,
            {"schema", "success", "planner_name", "joint_names", "segments", "failed_target_id", "message", "reached_target_ids"},
            "pose route plan",
        )
        if value["schema"] not in {
            MOTION_PLAN_RESULT_SCHEMA,
            _LEGACY_CAMERA_ROUTE_SCHEMA,
        }:
            raise ValueError(f"unsupported pose route plan schema: {value['schema']!r}")
        joint_names = tuple(
            _require_string(item, f"joint_names[{index}]")
            for index, item in enumerate(_require_sequence(value["joint_names"], "joint_names"))
        )
        segments = tuple(
            MotionPlanSegment.from_dict(item)
            for item in _require_sequence(value["segments"], "segments")
        )
        failed = value["failed_target_id"]
        reached = tuple(
            _require_string(item, f"reached_target_ids[{index}]")
            for index, item in enumerate(_require_sequence(value["reached_target_ids"], "reached_target_ids"))
        )
        return cls(
            success=_require_bool(value["success"], "success"),
            planner_name=_require_string(value["planner_name"], "planner_name"),
            joint_names=joint_names,
            segments=segments,
            failed_target_id=None if failed is None else _require_string(failed, "failed_target_id"),
            message=_require_string(value["message"], "message", allow_empty=True),
            reached_target_ids=reached,
        )


class PoseRoutePlanner(Protocol):
    planner_name: str

    @property
    def joint_names(self) -> tuple[str, ...]: ...

    def plan_pose_route(
        self,
        targets: tuple[PoseTarget, ...],
        start_state: RobotState,
        strategy: MotionPlanningStrategy = "direct_pose",
    ) -> MotionPlanResult: ...

    def find_collision_free_ik(
        self,
        targets: tuple[PoseTarget, ...],
        start_state: RobotState,
    ) -> tuple[IKSolution, ...]: ...


def plan_pose_route(
    planner: PoseRoutePlanner,
    targets: tuple[PoseTarget, ...],
    start_state: RobotState,
    *,
    strategy: MotionPlanningStrategy = "direct_pose",
) -> MotionPlanResult:
    if not targets:
        raise ValueError("pose route requires at least one target")
    return planner.plan_pose_route(
        targets, start_state, strategy=strategy
    )


def offset_pose_target_along_local_z(
    target: PoseTarget, distance_m: float
) -> PoseTarget:
    distance = _require_float(distance_m, "pose target offset distance", minimum=0.0)
    local_z = _quaternion_local_z_axis(target.target_quaternion_wxyz)
    return PoseTarget(
        target_id=target.target_id,
        target_position=tuple(
            position + distance * axis
            for position, axis in zip(target.target_position, local_z)
        ),
        target_quaternion_wxyz=target.target_quaternion_wxyz,
    )


def _pose_errors(actual_pose: Any, target_pose: Any) -> tuple[float, float]:
    import torch

    actual_position = actual_pose.position.reshape(-1, 3)[0]
    target_position = target_pose.position.reshape(-1, 3)[0]
    position_error = float(torch.linalg.vector_norm(actual_position - target_position).item())
    actual_quaternion = actual_pose.quaternion.reshape(-1, 4)[0]
    target_quaternion = target_pose.quaternion.reshape(-1, 4)[0]
    quaternion_dot = torch.abs(torch.sum(actual_quaternion * target_quaternion)).clamp(0.0, 1.0)
    orientation_error = float((2.0 * torch.acos(quaternion_dot)).item())
    return position_error, orientation_error


def _planning_result_succeeded(result: Any) -> bool:
    return bool(
        result is not None
        and result.success is not None
        and bool(result.success.any().item())
    )


def _validated_interpolated_data(
    result: Any,
    *,
    graph_planner: Any,
    expected_start: Any,
    expected_end: Any | None,
    joint_tolerance: float,
    joint_count: int,
) -> tuple[Any, tuple[float, ...], tuple[tuple[float, ...], ...]] | None:
    import torch

    if not _planning_result_succeeded(result):
        return None
    interpolated = result.get_interpolated_plan()
    if interpolated is None:
        return None
    positions = interpolated.position.reshape(-1, joint_count).clone()
    if len(positions) < 2 or not bool(torch.isfinite(positions).all().item()):
        return None
    trajectory_time = _optional_interpolated_time(interpolated, len(positions))
    trajectory_velocity = _optional_interpolated_matrix(
        interpolated, "velocity", len(positions), joint_count
    )
    if trajectory_time is None or trajectory_velocity is None:
        return None
    if not _joint_positions_match(
        positions[0], expected_start, joint_tolerance
    ) or (
        expected_end is not None
        and not _joint_positions_match(positions[-1], expected_end, joint_tolerance)
    ):
        return None
    if not _all_true(graph_planner.check_samples_feasibility(positions)):
        return None
    return positions, trajectory_time, trajectory_velocity


def _quaternion_local_z_axis(
    quaternion_wxyz: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    w, x, y, z = quaternion_wxyz
    return (
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    )


def _joint_edge_is_feasible(
    graph_planner: Any,
    start: Any,
    goal: Any,
    sample_count: int,
) -> bool:
    samples = _joint_edge_samples(start, goal, sample_count)
    return _all_true(graph_planner.check_samples_feasibility(samples))


def _joint_edge_samples(start: Any, goal: Any, sample_count: int) -> Any:
    import torch

    interpolation = torch.linspace(
        0.0,
        1.0,
        sample_count,
        device=start.device,
        dtype=start.dtype,
    ).reshape(-1, 1)
    samples = start.reshape(1, -1) + interpolation * (
        goal.reshape(1, -1) - start.reshape(1, -1)
    )
    return samples


def _joint_positions_match(actual: Any, expected: Any, tolerance: float) -> bool:
    import torch

    return bool(torch.max(torch.abs(actual - expected)).item() <= tolerance)


def _trajectory_endpoint_is_stopped(
    velocity: tuple[float, ...], tolerance: float
) -> bool:
    return max((abs(value) for value in velocity), default=math.inf) <= tolerance


class CuroboPlanner:
    name = "curobo_planner"
    planner_name = name

    def __init__(
        self,
        *,
        repo_root: Path,
        robot_config_path: Path = DEFAULT_ROBOT_CONFIG,
        world_config_path: Path = DEFAULT_WORLD_CONFIG,
        graph_config_path: Path = DEFAULT_GRAPH_CONFIG,
        ik_batch_size: int | None = None,
        ik_solutions_per_target: int | None = None,
        planning_policy: MotionPlanningPolicy | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.robot_config_path = self._resolve(robot_config_path)
        self.world_config_path = self._resolve(world_config_path)
        self.graph_config_path = self._resolve(graph_config_path)
        self._planner: Any | None = None
        self._ik_solver: Any | None = None
        self._joint_names: tuple[str, ...] = ()
        if (ik_batch_size is None) != (ik_solutions_per_target is None):
            raise ValueError(
                "pose route batch IK size and solution count must be configured together"
            )
        if ik_batch_size is not None and ik_batch_size <= 0:
            raise ValueError("pose route batch IK size must be positive")
        if ik_solutions_per_target is not None and ik_solutions_per_target <= 0:
            raise ValueError("pose route IK solutions per target must be positive")
        self._ik_batch_size = ik_batch_size
        self._ik_solutions_per_target = ik_solutions_per_target
        self._planning_policy = planning_policy

    @staticmethod
    def is_available() -> bool:
        return (
            importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("curobo") is not None
        )

    @property
    def joint_names(self) -> tuple[str, ...]:
        self._get_planner()
        return self._joint_names

    def plan(self, request: PlanningRequest) -> PlanningResult:
        """Plan the canonical pregrasp pose against the configured world."""
        if request.object_pose.object_id != request.grasp_target.object_id:
            return PlanningResult(
                success=False,
                trajectory=None,
                message="PlanningRequest object and grasp target IDs do not match.",
            )
        transform = request.grasp_target.T_world_pregrasp
        if transform is None:
            return PlanningResult(
                success=False,
                trajectory=None,
                message="PlanningRequest.grasp_target.T_world_pregrasp is required.",
            )
        try:
            pose = transform_to_curobo_pose(transform)
            result = self.plan_pose_route(
                (
                    PoseTarget(
                        target_id=request.grasp_target.object_id,
                        target_position=(pose[0], pose[1], pose[2]),
                        target_quaternion_wxyz=(pose[3], pose[4], pose[5], pose[6]),
                    ),
                ),
                request.robot_state,
            )
            trajectory = self._single_target_trajectory(
                result,
                target_object_id=request.grasp_target.object_id,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            return PlanningResult(success=False, trajectory=None, message=str(exc))
        if result.success and trajectory is None:
            return PlanningResult(
                success=False,
                trajectory=None,
                message="cuRobo planning did not produce one complete timed trajectory.",
            )
        return PlanningResult(
            success=result.success,
            trajectory=trajectory,
            message=result.message,
        )

    def plan_pose_route(
        self,
        targets: tuple[PoseTarget, ...],
        start_state: RobotState,
        strategy: MotionPlanningStrategy = "direct_pose",
    ) -> MotionPlanResult:
        if not targets:
            raise ValueError("pose route requires at least one target")
        planner = self._get_planner()
        joint_names = self._joint_names
        start_joint_positions = self._validate_start_state(start_state)
        if strategy not in {"direct_pose", "cartesian_continuation"}:
            raise ValueError(f"unsupported pose route planning strategy: {strategy}")
        if strategy == "cartesian_continuation" and not self._enable_cartesian_continuation:
            raise ValueError("Cartesian continuation is disabled by policy")
        import torch
        from curobo.types import GoalToolPose, JointState, Pose

        current = JointState.from_position(
            torch.tensor([start_joint_positions], device="cuda", dtype=torch.float32),
            joint_names=list(joint_names),
        )
        segments: list[MotionPlanSegment] = []

        for index, target in enumerate(targets):
            target_pose = Pose(
                position=torch.tensor(
                    [target.target_position], device="cuda", dtype=torch.float32
                ),
                quaternion=torch.tensor(
                    [target.target_quaternion_wxyz], device="cuda", dtype=torch.float32
                ),
            )
            goal = GoalToolPose.from_poses(
                {planner.tool_frames[0]: target_pose},
                ordered_tool_frames=planner.tool_frames,
            )
            started = time.perf_counter()
            if strategy == "cartesian_continuation":
                continuation_segment = self._plan_via_cartesian_continuation(
                    planner=planner,
                    target=target,
                    target_pose=target_pose,
                    current=current,
                    planning_started=started,
                )
                segments.append(continuation_segment)
                if continuation_segment.success and continuation_segment.trajectory:
                    current = JointState.from_position(
                        torch.tensor(
                            [continuation_segment.trajectory[-1]],
                            device="cuda",
                            dtype=torch.float32,
                        ),
                        joint_names=list(joint_names),
                    )
                    continue
                return MotionPlanResult(
                    success=False,
                    planner_name=self.planner_name,
                    joint_names=joint_names,
                    segments=tuple(segments),
                    failed_target_id=target.target_id,
                    message=f"cuRobo failed at required pose target {target.target_id}.",
                    reached_target_ids=tuple(item.target_id for item in targets[:index]),
                )

            result = planner.plan_pose(
                goal_tool_poses=goal,
                current_state=current,
                max_attempts=self._max_attempts,
                enable_graph_attempt=self._graph_attempt,
            )
            planning_time = round(time.perf_counter() - started, 6)
            if result is None or result.success is None or not bool(result.success.any().item()):
                segments.append(
                    MotionPlanSegment(
                        target_id=target.target_id,
                        success=False,
                        message=(
                            "cuRobo plan_pose did not find a collision-free "
                            "target-pose trajectory."
                        ),
                        planning_time_s=planning_time,
                        waypoint_count=0,
                    )
                )
                return MotionPlanResult(
                    success=False,
                    planner_name=self.planner_name,
                    joint_names=joint_names,
                    segments=tuple(segments),
                    failed_target_id=target.target_id,
                    message=f"cuRobo failed at required pose target {target.target_id}.",
                    reached_target_ids=tuple(item.target_id for item in targets[:index]),
                )

            interpolated = result.get_interpolated_plan()
            if interpolated is None:
                segments.append(
                    MotionPlanSegment(
                        target_id=target.target_id,
                        success=False,
                        message="cuRobo did not return an interpolated target-pose trajectory.",
                        planning_time_s=planning_time,
                        waypoint_count=0,
                    )
                )
                return MotionPlanResult(
                    success=False,
                    planner_name=self.planner_name,
                    joint_names=joint_names,
                    segments=tuple(segments),
                    failed_target_id=target.target_id,
                    message=f"cuRobo returned no executable trajectory at {target.target_id}.",
                    reached_target_ids=tuple(item.target_id for item in targets[:index]),
                )
            positions_cuda = interpolated.position.reshape(-1, len(joint_names))
            current = JointState.from_position(
                positions_cuda[-1].reshape(1, len(joint_names)),
                joint_names=list(joint_names),
            )
            actual_pose = planner.kinematics.compute_kinematics(current).tool_poses
            position_error, orientation_error = _pose_errors(actual_pose, target_pose)
            positions = positions_cuda.detach().cpu()
            trajectory = tuple(tuple(float(value) for value in row.tolist()) for row in positions)
            trajectory_velocity = _optional_interpolated_matrix(
                interpolated, "velocity", len(trajectory), len(joint_names)
            )
            trajectory_time_s = _optional_interpolated_time(interpolated, len(trajectory))
            trajectory_complete = (
                trajectory_time_s is not None and trajectory_velocity is not None
            )
            within_tolerance = (
                position_error <= self._position_tolerance_m
                and orientation_error <= self._orientation_tolerance_rad
            )
            segment_success = within_tolerance and trajectory_complete
            segments.append(
                MotionPlanSegment(
                    target_id=target.target_id,
                    success=segment_success,
                    message=(
                        "cuRobo target-pose planning and terminal-pose validation succeeded."
                        if segment_success
                        else "cuRobo returned incomplete trajectory timing or velocity data."
                        if within_tolerance
                        else "cuRobo trajectory exceeded the configured terminal target-pose tolerance."
                    ),
                    planning_time_s=planning_time,
                    waypoint_count=len(trajectory),
                    trajectory=trajectory,
                    trajectory_time_s=trajectory_time_s,
                    trajectory_velocity=trajectory_velocity,
                    target_position_error_m=position_error,
                    target_orientation_error_rad=orientation_error,
                )
            )
            if not segment_success:
                return MotionPlanResult(
                    success=False,
                    planner_name=self.planner_name,
                    joint_names=joint_names,
                    segments=tuple(segments),
                    failed_target_id=target.target_id,
                    message=(
                        f"cuRobo trajectory data was incomplete at {target.target_id}."
                        if within_tolerance
                        else f"cuRobo terminal pose was outside tolerance at {target.target_id}."
                    ),
                    reached_target_ids=tuple(item.target_id for item in targets[:index]),
                )

        return MotionPlanResult(
            success=True,
            planner_name=self.planner_name,
            joint_names=joint_names,
            segments=tuple(segments),
            failed_target_id=None,
            message=f"cuRobo planned and validated all {len(targets)} pose targets.",
            reached_target_ids=tuple(target.target_id for target in targets),
        )

    def _plan_via_cartesian_continuation(
        self,
        *,
        planner: Any,
        target: PoseTarget,
        target_pose: Any,
        current: Any,
        planning_started: float,
    ) -> MotionPlanSegment:
        import torch
        import torch.nn.functional as torch_functional
        from curobo.types import GoalToolPose, JointState, Pose

        if self._ik_solver is None or planner.graph_planner is None:
            return self._continuation_failure_segment(
                target,
                "Cartesian continuation requires collision-aware IK and graph planning.",
                planning_started,
            )
        staging_target = offset_pose_target_along_local_z(
            target, self._continuation_offset_m
        )
        staging_pose = Pose(
            position=torch.tensor(
                [staging_target.target_position], device="cuda", dtype=torch.float32
            ),
            quaternion=torch.tensor(
                [staging_target.target_quaternion_wxyz],
                device="cuda",
                dtype=torch.float32,
            ),
        )
        staging_goal = GoalToolPose.from_poses(
            {planner.tool_frames[0]: staging_pose},
            ordered_tool_frames=planner.tool_frames,
        )
        approach = planner.plan_pose(
            goal_tool_poses=staging_goal,
            current_state=current,
            max_attempts=self._max_attempts,
            enable_graph_attempt=self._graph_attempt,
        )
        approach_data = _validated_interpolated_data(
            approach,
            graph_planner=planner.graph_planner,
            expected_start=current.position.reshape(-1),
            expected_end=None,
            joint_tolerance=self._continuation_joint_tolerance_rad,
            joint_count=len(self._joint_names),
        )
        if approach_data is None:
            return self._continuation_failure_segment(
                target,
                "cuRobo could not reach the configured continuation staging pose.",
                planning_started,
            )
        approach_positions, approach_time, approach_velocity = approach_data
        if not _trajectory_endpoint_is_stopped(
            approach_velocity[-1],
            self._continuation_stop_velocity_tolerance_rad_s,
        ):
            return self._continuation_failure_segment(
                target,
                "The staging approach did not terminate at rest.",
                planning_started,
            )
        staging_terminal = JointState.from_position(
            approach_positions[-1].reshape(1, len(self._joint_names)),
            joint_names=list(self._joint_names),
        )
        staging_actual = planner.kinematics.compute_kinematics(
            staging_terminal
        ).tool_poses
        staging_position_error, staging_orientation_error = _pose_errors(
            staging_actual, staging_pose
        )
        if (
            staging_position_error > self._position_tolerance_m
            or staging_orientation_error > self._orientation_tolerance_rad
        ):
            return self._continuation_failure_segment(
                target,
                "The staging approach exceeded the configured terminal pose tolerance.",
                planning_started,
            )

        corridor = self._trace_ik_corridor_from_staging(
            planner=planner,
            target=target,
            staging_joint_position=approach_positions[-1],
        )
        if corridor is None:
            return self._continuation_failure_segment(
                target,
                "The reachable staging IK branch was not collision-continuous to the target.",
                planning_started,
            )

        with self._create_independent_motion_planner() as corridor_planner:
            if corridor_planner.graph_planner is None:
                return self._continuation_failure_segment(
                    target,
                    "Cartesian continuation requires a configured collision graph planner.",
                    planning_started,
                )
            staging_position = corridor[0]
            final_position = corridor[-1]
            staging_state = JointState.from_position(
                staging_position.reshape(1, len(self._joint_names)),
                joint_names=list(self._joint_names),
            )
            final_state = JointState.from_position(
                final_position.reshape(1, len(self._joint_names)),
                joint_names=list(self._joint_names),
            )
            corridor_path = torch.stack(corridor)
            resampled = torch_functional.interpolate(
                corridor_path.transpose(0, 1).unsqueeze(0),
                size=corridor_planner.trajopt_solver.action_horizon,
                mode="linear",
                align_corners=True,
            ).transpose(1, 2)
            seed_trajectory = resampled.unsqueeze(1).repeat(
                1, self._num_trajopt_seeds, 1, 1
            )
            continuation = corridor_planner.trajopt_solver.solve_cspace(
                final_state,
                staging_state,
                seed_traj=seed_trajectory,
                return_seeds=1,
                num_seeds=self._num_trajopt_seeds,
                finetune_attempts=self._continuation_finetune_attempts,
            )
            continuation_data = _validated_interpolated_data(
                continuation,
                graph_planner=corridor_planner.graph_planner,
                expected_start=staging_position,
                expected_end=final_position,
                joint_tolerance=self._continuation_joint_tolerance_rad,
                joint_count=len(self._joint_names),
            )
            if continuation_data is None:
                return self._continuation_failure_segment(
                    target,
                    "The staging IK corridor did not produce a validated cuRobo trajectory.",
                    planning_started,
                )
            continuation_positions, continuation_time, continuation_velocity = (
                continuation_data
            )
            if not _trajectory_endpoint_is_stopped(
                continuation_velocity[0],
                self._continuation_stop_velocity_tolerance_rad_s,
            ) or not _joint_edge_is_feasible(
                corridor_planner.graph_planner,
                approach_positions[-1],
                continuation_positions[0],
                self._continuation_edge_sample_count,
            ):
                return self._continuation_failure_segment(
                    target,
                    "The staging approach and local continuation were not safely continuous.",
                    planning_started,
                )

            terminal = JointState.from_position(
                continuation_positions[-1].reshape(1, len(self._joint_names)),
                joint_names=list(self._joint_names),
            )
            actual_pose = corridor_planner.kinematics.compute_kinematics(
                terminal
            ).tool_poses
            position_error, orientation_error = _pose_errors(actual_pose, target_pose)
            if (
                position_error > self._position_tolerance_m
                or orientation_error > self._orientation_tolerance_rad
            ):
                return self._continuation_failure_segment(
                    target,
                    "The Cartesian continuation exceeded the terminal target-pose tolerance.",
                    planning_started,
                )

            combined_positions = torch.cat(
                (approach_positions, continuation_positions[1:]), dim=0
            )
            trajectory = _tensor_matrix_to_tuple(combined_positions)
            return MotionPlanSegment(
                target_id=target.target_id,
                success=True,
                message=(
                    f"cuRobo reached a staging pose {self._continuation_offset_m:.3f} m "
                    "from the target and preserved its reachable IK branch through the "
                    "collision-checked local continuation."
                ),
                planning_time_s=round(time.perf_counter() - planning_started, 6),
                waypoint_count=len(trajectory),
                trajectory=trajectory,
                trajectory_time_s=_join_trajectory_times(
                    approach_time, continuation_time
                ),
                trajectory_velocity=approach_velocity + continuation_velocity[1:],
                target_position_error_m=position_error,
                target_orientation_error_rad=orientation_error,
                planning_strategy="cartesian_continuation",
                continuation_offset_m=self._continuation_offset_m,
            )

    def _trace_ik_corridor_from_staging(
        self,
        *,
        planner: Any,
        target: PoseTarget,
        staging_joint_position: Any,
    ) -> tuple[Any, ...] | None:
        import torch
        from curobo.types import GoalToolPose, JointState, Pose

        if self._ik_solver is None or planner.graph_planner is None:
            return None
        local_z = torch.tensor(
            _quaternion_local_z_axis(target.target_quaternion_wxyz),
            device="cuda",
            dtype=torch.float32,
        )
        target_position = torch.tensor(
            target.target_position, device="cuda", dtype=torch.float32
        )
        target_quaternion = torch.tensor(
            [target.target_quaternion_wxyz], device="cuda", dtype=torch.float32
        )
        step_count = max(1, math.ceil(self._continuation_offset_m / self._continuation_step_m))
        actual_step = self._continuation_offset_m / step_count
        previous = staging_joint_position.detach().clone()
        states = [previous]
        for step_index in range(1, step_count + 1):
            remaining_offset = max(0.0, self._continuation_offset_m - actual_step * step_index)
            step_pose = Pose(
                position=(target_position + local_z * remaining_offset).reshape(1, 3),
                quaternion=target_quaternion,
            )
            step_goal = GoalToolPose.from_poses(
                {planner.tool_frames[0]: step_pose},
                ordered_tool_frames=planner.tool_frames,
            )
            step_current = JointState.from_position(
                previous.reshape(1, len(self._joint_names)),
                joint_names=list(self._joint_names),
            )
            step_ik = self._ik_solver.solve_pose(
                step_goal,
                current_state=step_current,
                return_seeds=self._continuation_ik_solution_count,
            )
            if step_ik is None or step_ik.success is None or step_ik.solution is None:
                return None
            valid_positions = sorted(
                (
                    step_ik.solution[0, solution_index]
                    for solution_index in range(self._continuation_ik_solution_count)
                    if bool(step_ik.success[0, solution_index].item())
                ),
                key=lambda position: float(
                    torch.linalg.vector_norm(position - previous).item()
                ),
            )
            next_position = next(
                (
                    position.detach().clone()
                    for position in valid_positions
                    if _joint_edge_is_feasible(
                        planner.graph_planner,
                        previous,
                        position,
                        self._continuation_edge_sample_count,
                    )
                ),
                None,
            )
            if next_position is None:
                return None
            states.append(next_position)
            previous = next_position
        return tuple(states)

    def _continuation_failure_segment(
        self,
        target: PoseTarget,
        message: str,
        planning_started: float,
    ) -> MotionPlanSegment:
        return MotionPlanSegment(
            target_id=target.target_id,
            success=False,
            message=message,
            planning_time_s=round(time.perf_counter() - planning_started, 6),
            waypoint_count=0,
            planning_strategy="cartesian_continuation",
            continuation_offset_m=self._continuation_offset_m,
        )

    def find_collision_free_ik(
        self,
        targets: tuple[PoseTarget, ...],
        start_state: RobotState,
    ) -> tuple[IKSolution, ...]:
        """Solve target poses in collision-aware batches without trajectory optimization."""
        if not targets:
            return ()
        if self._ik_batch_size is None or self._ik_solutions_per_target is None:
            raise RuntimeError("pose route batch IK was not configured for this planner")

        planner = self._get_planner()
        start_joint_positions = self._validate_start_state(start_state)
        import torch
        from curobo.types import GoalToolPose, JointState, Pose

        solutions: list[IKSolution] = []
        for offset in range(0, len(targets), self._ik_batch_size):
            chunk = targets[offset : offset + self._ik_batch_size]
            batch_size = len(chunk)
            current = JointState.from_position(
                torch.tensor(
                    [start_joint_positions] * batch_size,
                    device="cuda",
                    dtype=torch.float32,
                ),
                joint_names=list(self._joint_names),
            )
            target_pose = Pose(
                position=torch.tensor(
                    [target.target_position for target in chunk],
                    device="cuda",
                    dtype=torch.float32,
                ),
                quaternion=torch.tensor(
                    [target.target_quaternion_wxyz for target in chunk],
                    device="cuda",
                    dtype=torch.float32,
                ),
            )
            goal = GoalToolPose.from_poses(
                {planner.tool_frames[0]: target_pose},
                ordered_tool_frames=planner.tool_frames,
                num_goalset=1,
            )
            result = self._ik_solver.solve_pose(
                goal,
                current_state=current,
                return_seeds=self._ik_solutions_per_target,
            )
            if result is None or result.success is None or result.solution is None:
                continue
            for target_index, target in enumerate(chunk):
                for solution_index in range(self._ik_solutions_per_target):
                    if not bool(result.success[target_index, solution_index].item()):
                        continue
                    position = result.solution[target_index, solution_index]
                    solutions.append(
                        IKSolution(
                            target.target_id,
                            tuple(float(value) for value in position.detach().cpu().tolist()),
                        )
                    )
        return tuple(solutions)

    def _get_planner(self) -> Any:
        if self._planner is not None:
            return self._planner
        self._validate_config_paths()
        self._validate_environment()
        import yaml
        robot = yaml.safe_load(self.robot_config_path.read_text(encoding="utf-8"))
        robot_payload = robot.get("robot_cfg", robot)
        robot_kinematics = robot_payload["kinematics"]
        urdf_path = Path(str(robot_kinematics["urdf_path"]))
        robot_kinematics["urdf_path"] = str(
            urdf_path if urdf_path.is_absolute() else self.repo_root / urdf_path
        )
        robot_kinematics["asset_root_path"] = str(self.repo_root)
        robot_kinematics.pop("load_collision_spheres", None)
        robot_kinematics.pop("num_envs", None)

        world = yaml.safe_load(self.world_config_path.read_text(encoding="utf-8"))
        for mesh in world.get("mesh", {}).values():
            mesh_path = Path(str(mesh["file_path"]))
            mesh["file_path"] = str(
                mesh_path if mesh_path.is_absolute() else self.repo_root / mesh_path
            )
        graph = yaml.safe_load(self.graph_config_path.read_text(encoding="utf-8"))
        if self._planning_policy is None:
            route_policy = graph.get(
                "motion_planning_policy",
                graph.get("camera_route_policy", {}),
            )
            self._max_attempts = int(route_policy.get("max_attempts", 0))
            self._graph_attempt = int(route_policy.get("enable_graph_attempt", -1))
            num_ik_seeds = int(route_policy.get("num_ik_seeds", 0))
            num_trajopt_seeds = int(route_policy.get("num_trajopt_seeds", 0))
            random_seed = int(route_policy.get("random_seed", -1))
            self._position_tolerance_m = float(
                route_policy.get("position_tolerance_m", 0.0)
            )
            self._orientation_tolerance_rad = float(
                route_policy.get("orientation_tolerance_rad", 0.0)
            )
            self._enable_cartesian_continuation = False
            self._continuation_offset_m = 0.10
            self._continuation_step_m = 0.005
            self._continuation_edge_sample_count = 11
            self._continuation_ik_solution_count = 8
            self._continuation_finetune_attempts = 3
            self._continuation_joint_tolerance_rad = 0.001
            self._continuation_stop_velocity_tolerance_rad_s = 0.01
        else:
            self._max_attempts = self._planning_policy.max_attempts
            self._graph_attempt = self._planning_policy.enable_graph_attempt
            num_ik_seeds = self._planning_policy.num_ik_seeds
            num_trajopt_seeds = self._planning_policy.num_trajopt_seeds
            random_seed = self._planning_policy.random_seed
            self._position_tolerance_m = self._planning_policy.position_tolerance
            self._orientation_tolerance_rad = self._planning_policy.orientation_tolerance
            self._enable_cartesian_continuation = (
                self._planning_policy.enable_cartesian_continuation
            )
            self._continuation_offset_m = self._planning_policy.continuation_offset_m
            self._continuation_step_m = self._planning_policy.continuation_step_m
            self._continuation_edge_sample_count = (
                self._planning_policy.continuation_edge_sample_count
            )
            self._continuation_ik_solution_count = (
                self._planning_policy.continuation_ik_solution_count
            )
            self._continuation_finetune_attempts = (
                self._planning_policy.continuation_finetune_attempts
            )
            self._continuation_joint_tolerance_rad = (
                self._planning_policy.continuation_joint_tolerance_rad
            )
            self._continuation_stop_velocity_tolerance_rad_s = (
                self._planning_policy.continuation_stop_velocity_tolerance_rad_s
            )
        if (
            self._max_attempts <= 0
            or self._graph_attempt < 0
            or num_ik_seeds <= 0
            or num_trajopt_seeds <= 0
            or random_seed < 0
            or self._position_tolerance_m <= 0.0
            or self._orientation_tolerance_rad <= 0.0
        ):
            raise ValueError("cuRobo pose route planning policy is invalid")
        if self._continuation_ik_solution_count > num_ik_seeds:
            raise ValueError(
                "Cartesian continuation IK solution count cannot exceed num_ik_seeds"
            )

        self._num_ik_seeds = num_ik_seeds
        self._num_trajopt_seeds = num_trajopt_seeds
        self._random_seed = random_seed
        self._robot_payload = robot
        self._world_payload = world
        self._graph_payload = graph
        config = self._create_motion_planner_config()
        self._planner = create_collision_filtered_motion_planner(config)
        raw_joint_names = getattr(self._planner, "joint_names", None)
        if (
            isinstance(raw_joint_names, (str, bytes))
            or not isinstance(raw_joint_names, Sequence)
            or not raw_joint_names
        ):
            raise RuntimeError("cuRobo planner did not expose an ordered joint list")
        self._joint_names = tuple(str(name) for name in raw_joint_names)
        if len(set(self._joint_names)) != len(self._joint_names):
            raise RuntimeError("cuRobo planner joint names must be unique")
        needs_standalone_ik = (
            self._ik_batch_size is not None
            or self._enable_cartesian_continuation
        )
        if needs_standalone_ik:
            if (
                self._ik_solutions_per_target is not None
                and self._ik_solutions_per_target > num_ik_seeds
            ):
                raise ValueError(
                    "pose route IK solutions per target cannot exceed num_ik_seeds"
                )
            from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg

            ik_config = InverseKinematicsCfg.create(
                robot=deepcopy(robot),
                scene_model=deepcopy(world),
                num_seeds=num_ik_seeds,
                position_tolerance=self._position_tolerance_m,
                orientation_tolerance=self._orientation_tolerance_rad,
                use_cuda_graph=True,
                random_seed=random_seed,
                max_batch_size=self._ik_batch_size or 1,
                multi_env=False,
                max_goalset=1,
            )
            self._ik_solver = InverseKinematics(
                ik_config, self._planner.scene_collision_checker
            )
        return self._planner

    def _create_motion_planner_config(self) -> Any:
        from curobo.motion_planner import MotionPlannerCfg

        return MotionPlannerCfg.create(
            robot=deepcopy(self._robot_payload),
            scene_model=deepcopy(self._world_payload),
            graph_planner_config=deepcopy(self._graph_payload),
            num_ik_seeds=self._num_ik_seeds,
            num_trajopt_seeds=self._num_trajopt_seeds,
            position_tolerance=self._position_tolerance_m,
            orientation_tolerance=self._orientation_tolerance_rad,
            use_cuda_graph=True,
            random_seed=self._random_seed,
        )

    def _create_independent_motion_planner(self) -> Any:
        self._get_planner()
        return create_collision_filtered_motion_planner(
            self._create_motion_planner_config()
        )

    def _single_target_trajectory(
        self,
        result: MotionPlanResult,
        *,
        target_object_id: str,
    ) -> PlannedTrajectory | None:
        if not result.success or len(result.segments) != 1:
            return None
        segment = result.segments[0]
        if not segment.success or segment.trajectory_time_s is None:
            return None
        return PlannedTrajectory(
            joint_names=result.joint_names,
            waypoints=tuple(
                TrajectoryWaypoint(time_s=time_s, joint_positions=positions)
                for time_s, positions in zip(
                    segment.trajectory_time_s,
                    segment.trajectory,
                    strict=True,
                )
            ),
            planner_name=self.planner_name,
            target_object_id=target_object_id,
        )

    def _validate_start_state(self, start_state: RobotState) -> tuple[float, ...]:
        if start_state.joint_names != self._joint_names:
            raise ValueError(
                "planner start_state.joint_names must exactly match the cuRobo robot "
                f"joint order: expected={self._joint_names}, "
                f"received={start_state.joint_names}"
            )
        if not all(math.isfinite(value) for value in start_state.joint_positions):
            raise ValueError("planner start_state joint positions must be finite")
        return start_state.joint_positions

    def _resolve(self, path: Path) -> Path:
        resolved = path if path.is_absolute() else self.repo_root / path
        return resolved.resolve()

    def _validate_config_paths(self) -> None:
        for name, path in (
            ("robot", self.robot_config_path),
            ("world", self.world_config_path),
            ("graph", self.graph_config_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(
                    f"cuRobo {name} configuration does not exist: {path}"
                )

    @staticmethod
    def _validate_environment() -> None:
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("curobo") is None:
            raise RuntimeError("cuRobo pose route planning requires torch and cuRobo")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("cuRobo pose route planning requires torch CUDA")


def _optional_interpolated_matrix(
    interpolated: Any,
    attribute: str,
    waypoint_count: int,
    joint_count: int,
) -> tuple[tuple[float, ...], ...] | None:
    tensor = getattr(interpolated, attribute, None)
    if tensor is None:
        return None
    values = tensor.detach().cpu().reshape(-1, joint_count)
    if len(values) != waypoint_count:
        return None
    return tuple(tuple(float(value) for value in row.tolist()) for row in values)


def _tensor_matrix_to_tuple(values: Any) -> tuple[tuple[float, ...], ...]:
    matrix = values.detach().cpu()
    return tuple(tuple(float(value) for value in row.tolist()) for row in matrix)


def _optional_interpolated_time(
    interpolated: Any, waypoint_count: int
) -> tuple[float, ...] | None:
    dt_tensor = getattr(interpolated, "dt", None)
    if dt_tensor is None:
        return None
    values = dt_tensor.detach().cpu().reshape(-1)
    if not len(values):
        return None
    dt = float(values[0])
    if not math.isfinite(dt) or dt <= 0.0:
        return None
    return tuple(round(index * dt, 9) for index in range(waypoint_count))


def _join_trajectory_times(
    first: tuple[float, ...], second: tuple[float, ...]
) -> tuple[float, ...]:
    offset = first[-1]
    return first + tuple(round(offset + value, 9) for value in second[1:])
