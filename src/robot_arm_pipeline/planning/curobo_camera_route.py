from __future__ import annotations

import importlib.util
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

DEFAULT_ROBOT_CONFIG = Path("configs/curobo/gen3/robot.yml")
DEFAULT_WORLD_CONFIG = Path("configs/curobo/gen3/world.yml")
DEFAULT_GRAPH_CONFIG = Path("configs/curobo/gen3/graph.yml")
GEN3_JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7")
DEFAULT_START_JOINT_POSITIONS = (0.0, 0.26179939, 3.14159265, -2.26892803, 0.0, 0.95993109, 1.57079633)
CAMERA_ROUTE_PLAN_SCHEMA = "camera_route_plan"


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
class CameraRouteTarget:
    target_id: str
    target_position: tuple[float, float, float]
    target_quaternion_wxyz: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        _require_string(self.target_id, "camera route target_id")
        if len(self.target_position) != 3:
            raise ValueError("camera route target_position must contain three values")
        if len(self.target_quaternion_wxyz) != 4:
            raise ValueError("camera route target_quaternion_wxyz must contain four values")
        position = tuple(
            _require_float(value, f"camera route target_position[{index}]")
            for index, value in enumerate(self.target_position)
        )
        quaternion = tuple(
            _require_float(value, f"camera route target_quaternion_wxyz[{index}]")
            for index, value in enumerate(self.target_quaternion_wxyz)
        )
        if not math.isclose(sum(value * value for value in quaternion), 1.0, abs_tol=1e-5):
            raise ValueError("camera route target quaternion must be normalized")
        object.__setattr__(self, "target_position", position)
        object.__setattr__(self, "target_quaternion_wxyz", quaternion)


@dataclass(frozen=True)
class CameraRouteSegment:
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

    def __post_init__(self) -> None:
        _require_string(self.target_id, "camera route segment target_id")
        if not isinstance(self.success, bool):
            raise ValueError("camera route segment success must be a boolean")
        _require_string(self.message, "camera route segment message", allow_empty=True)
        _require_float(self.planning_time_s, "camera route segment planning_time_s", minimum=0.0)
        _require_int(self.waypoint_count, "camera route segment waypoint_count")
        if self.waypoint_count != len(self.trajectory):
            raise ValueError("camera route segment waypoint_count must match trajectory length")
        widths = {len(row) for row in self.trajectory}
        if len(widths) > 1 or (self.trajectory and 0 in widths):
            raise ValueError("camera route segment trajectory rows must have one non-zero width")
        for row_index, row in enumerate(self.trajectory):
            for value_index, value in enumerate(row):
                _require_float(value, f"camera route segment trajectory[{row_index}][{value_index}]")
        if self.trajectory_time_s is not None:
            if len(self.trajectory_time_s) != self.waypoint_count:
                raise ValueError("camera route segment trajectory_time_s must match trajectory length")
            previous = -math.inf
            for index, value in enumerate(self.trajectory_time_s):
                current = _require_float(value, f"trajectory_time_s[{index}]", minimum=0.0)
                if current <= previous:
                    raise ValueError("camera route segment trajectory_time_s must be strictly increasing")
                previous = current
        if self.trajectory_velocity is not None:
            if len(self.trajectory_velocity) != self.waypoint_count:
                raise ValueError("camera route segment trajectory_velocity must match trajectory length")
            expected_width = next(iter(widths), 0)
            for index, row in enumerate(self.trajectory_velocity):
                if len(row) != expected_width:
                    raise ValueError("camera route segment velocity rows must match trajectory width")
                for column, value in enumerate(row):
                    _require_float(value, f"trajectory_velocity[{index}][{column}]")
        _require_optional_float(
            self.target_position_error_m, "target_position_error_m", minimum=0.0
        )
        _require_optional_float(
            self.target_orientation_error_rad, "target_orientation_error_rad", minimum=0.0
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
        }

    @classmethod
    def from_dict(cls, payload: object) -> CameraRouteSegment:
        value = _require_mapping(payload, "camera route segment")
        _require_keys(
            value,
            {
                "target_id", "success", "message", "planning_time_s", "waypoint_count",
                "trajectory", "trajectory_time_s", "trajectory_velocity",
                "target_position_error_m", "target_orientation_error_rad",
            },
            "camera route segment",
        )
        trajectory = _float_matrix(value["trajectory"], "trajectory")
        raw_time = value["trajectory_time_s"]
        raw_velocity = value["trajectory_velocity"]
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
        )


@dataclass(frozen=True)
class CameraRoutePlan:
    success: bool
    planner_name: str
    joint_names: tuple[str, ...]
    segments: tuple[CameraRouteSegment, ...]
    failed_target_id: str | None
    message: str
    reached_target_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise ValueError("camera route plan success must be a boolean")
        _require_string(self.planner_name, "camera route planner_name")
        _require_string(self.message, "camera route plan message", allow_empty=True)
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("camera route plan joint_names must be non-empty and unique")
        for index, name in enumerate(self.joint_names):
            _require_string(name, f"joint_names[{index}]")
        target_ids = [segment.target_id for segment in self.segments]
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("camera route plan segment target_ids must be unique")
        for segment in self.segments:
            if any(len(row) != len(self.joint_names) for row in segment.trajectory):
                raise ValueError("camera route segment trajectory width must match joint_names")
        if self.failed_target_id is not None:
            _require_string(self.failed_target_id, "failed_target_id")
        if len(set(self.reached_target_ids)) != len(self.reached_target_ids):
            raise ValueError("camera route plan reached_target_ids must be unique")
        if any(target_id not in target_ids for target_id in self.reached_target_ids):
            raise ValueError("camera route plan reached_target_ids must reference segments")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": CAMERA_ROUTE_PLAN_SCHEMA,
            "success": self.success,
            "planner_name": self.planner_name,
            "joint_names": list(self.joint_names),
            "segments": [segment.to_dict() for segment in self.segments],
            "failed_target_id": self.failed_target_id,
            "message": self.message,
            "reached_target_ids": list(self.reached_target_ids),
        }

    @classmethod
    def from_dict(cls, payload: object) -> CameraRoutePlan:
        value = _require_mapping(payload, "camera route plan")
        _require_keys(
            value,
            {"schema", "success", "planner_name", "joint_names", "segments", "failed_target_id", "message", "reached_target_ids"},
            "camera route plan",
        )
        if value["schema"] != CAMERA_ROUTE_PLAN_SCHEMA:
            raise ValueError(f"unsupported camera route plan schema: {value['schema']!r}")
        joint_names = tuple(
            _require_string(item, f"joint_names[{index}]")
            for index, item in enumerate(_require_sequence(value["joint_names"], "joint_names"))
        )
        segments = tuple(
            CameraRouteSegment.from_dict(item)
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


class CameraRoutePlanner(Protocol):
    planner_name: str

    def plan_camera_pose_route(
        self,
        targets: tuple[CameraRouteTarget, ...],
        start_joint_positions: tuple[float, ...],
    ) -> CameraRoutePlan: ...


def plan_camera_route(
    planner: CameraRoutePlanner,
    targets: tuple[CameraRouteTarget, ...],
    start_joint_positions: tuple[float, ...] = DEFAULT_START_JOINT_POSITIONS,
) -> CameraRoutePlan:
    if not targets:
        raise ValueError("camera route requires at least one target")
    return planner.plan_camera_pose_route(targets, start_joint_positions)


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


class CuroboCameraRoutePlanner:
    planner_name = "curobo_camera_route_planner"

    def __init__(
        self,
        *,
        repo_root: Path,
        robot_config_path: Path = DEFAULT_ROBOT_CONFIG,
        world_config_path: Path = DEFAULT_WORLD_CONFIG,
        graph_config_path: Path = DEFAULT_GRAPH_CONFIG,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.robot_config_path = self._resolve(robot_config_path)
        self.world_config_path = self._resolve(world_config_path)
        self.graph_config_path = self._resolve(graph_config_path)
        self._planner: Any | None = None

    def plan_camera_pose_route(
        self,
        targets: tuple[CameraRouteTarget, ...],
        start_joint_positions: tuple[float, ...] = DEFAULT_START_JOINT_POSITIONS,
    ) -> CameraRoutePlan:
        if len(start_joint_positions) != len(GEN3_JOINT_NAMES):
            raise ValueError("camera route start state must contain all seven Gen3 joints")
        planner = self._get_planner()
        import torch
        from curobo.types import GoalToolPose, JointState, Pose

        current = JointState.from_position(
            torch.tensor([start_joint_positions], device="cuda", dtype=torch.float32),
            joint_names=list(GEN3_JOINT_NAMES),
        )
        segments: list[CameraRouteSegment] = []

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
            result = planner.plan_pose(
                goal_tool_poses=goal,
                current_state=current,
                max_attempts=self._max_attempts,
                enable_graph_attempt=self._graph_attempt,
            )
            planning_time = round(time.perf_counter() - started, 6)
            if result is None or result.success is None or not bool(result.success.any().item()):
                segments.append(
                    CameraRouteSegment(
                        target_id=target.target_id,
                        success=False,
                        message="cuRobo plan_pose did not find a collision-free camera-pose trajectory.",
                        planning_time_s=planning_time,
                        waypoint_count=0,
                    )
                )
                return CameraRoutePlan(
                    success=False,
                    planner_name=self.planner_name,
                    joint_names=GEN3_JOINT_NAMES,
                    segments=tuple(segments),
                    failed_target_id=target.target_id,
                    message=f"cuRobo failed at required camera target {target.target_id}.",
                    reached_target_ids=tuple(item.target_id for item in targets[:index]),
                )

            interpolated = result.get_interpolated_plan()
            positions_cuda = interpolated.position.reshape(-1, len(GEN3_JOINT_NAMES))
            current = JointState.from_position(
                positions_cuda[-1].reshape(1, len(GEN3_JOINT_NAMES)),
                joint_names=list(GEN3_JOINT_NAMES),
            )
            actual_pose = planner.kinematics.compute_kinematics(current).tool_poses
            position_error, orientation_error = _pose_errors(actual_pose, target_pose)
            positions = positions_cuda.detach().cpu()
            trajectory = tuple(tuple(float(value) for value in row.tolist()) for row in positions)
            trajectory_velocity = _optional_interpolated_matrix(
                interpolated, "velocity", len(trajectory), len(GEN3_JOINT_NAMES)
            )
            trajectory_time_s = _optional_interpolated_time(interpolated, len(trajectory))
            within_tolerance = (
                position_error <= self._position_tolerance_m
                and orientation_error <= self._orientation_tolerance_rad
            )
            segments.append(
                CameraRouteSegment(
                    target_id=target.target_id,
                    success=within_tolerance,
                    message=(
                        "cuRobo camera-pose planning and terminal-pose validation succeeded."
                        if within_tolerance
                        else "cuRobo trajectory exceeded the configured terminal camera-pose tolerance."
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
            if not within_tolerance:
                return CameraRoutePlan(
                    success=False,
                    planner_name=self.planner_name,
                    joint_names=GEN3_JOINT_NAMES,
                    segments=tuple(segments),
                    failed_target_id=target.target_id,
                    message=f"cuRobo terminal camera pose was outside tolerance at {target.target_id}.",
                    reached_target_ids=tuple(item.target_id for item in targets[:index]),
                )

        return CameraRoutePlan(
            success=True,
            planner_name=self.planner_name,
            joint_names=GEN3_JOINT_NAMES,
            segments=tuple(segments),
            failed_target_id=None,
            message=f"cuRobo planned and validated all {len(targets)} camera targets.",
            reached_target_ids=tuple(target.target_id for target in targets),
        )
    def _get_planner(self) -> Any:
        if self._planner is not None:
            return self._planner
        self._validate_environment()
        import yaml
        from curobo.motion_planner import MotionPlannerCfg

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
        route_policy = graph.get("camera_route_policy", {})
        self._max_attempts = int(route_policy.get("max_attempts", 0))
        self._graph_attempt = int(route_policy.get("enable_graph_attempt", -1))
        num_ik_seeds = int(route_policy.get("num_ik_seeds", 0))
        num_trajopt_seeds = int(route_policy.get("num_trajopt_seeds", 0))
        random_seed = int(route_policy.get("random_seed", -1))
        self._position_tolerance_m = float(route_policy.get("position_tolerance_m", 0.0))
        self._orientation_tolerance_rad = float(route_policy.get("orientation_tolerance_rad", 0.0))
        if (
            self._max_attempts <= 0
            or self._graph_attempt < 0
            or num_ik_seeds <= 0
            or num_trajopt_seeds <= 0
            or random_seed < 0
            or self._position_tolerance_m <= 0.0
            or self._orientation_tolerance_rad <= 0.0
        ):
            raise ValueError("cuRobo camera route planning policy is invalid")

        config = MotionPlannerCfg.create(
            robot=robot,
            scene_model=world,
            graph_planner_config=graph,
            num_ik_seeds=num_ik_seeds,
            num_trajopt_seeds=num_trajopt_seeds,
            position_tolerance=self._position_tolerance_m,
            orientation_tolerance=self._orientation_tolerance_rad,
            use_cuda_graph=True,
            random_seed=random_seed,
        )
        self._planner = create_collision_filtered_motion_planner(config)
        return self._planner

    def _resolve(self, path: Path) -> Path:
        resolved = path if path.is_absolute() else self.repo_root / path
        if not resolved.exists():
            raise FileNotFoundError(f"cuRobo camera route configuration does not exist: {resolved}")
        return resolved

    @staticmethod
    def _validate_environment() -> None:
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("curobo") is None:
            raise RuntimeError("cuRobo camera route planning requires torch and cuRobo")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("cuRobo camera route planning requires torch CUDA")


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
