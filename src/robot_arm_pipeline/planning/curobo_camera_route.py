from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


DEFAULT_ROBOT_CONFIG = Path("configs/curobo/gen3/robot.yml")
DEFAULT_WORLD_CONFIG = Path("configs/curobo/gen3/world.yml")
DEFAULT_GRAPH_CONFIG = Path("configs/curobo/gen3/graph.yml")
GEN3_JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7")
DEFAULT_START_JOINT_POSITIONS = (0.0, 0.26179939, 3.14159265, -2.26892803, 0.0, 0.95993109, 1.57079633)


@dataclass(frozen=True)
class CameraRouteTarget:
    target_id: str
    reference_joint_positions: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.target_id:
            raise ValueError("camera route target_id must not be empty")
        if len(self.reference_joint_positions) != len(GEN3_JOINT_NAMES):
            raise ValueError("camera route target must contain all seven Gen3 joint positions")


@dataclass(frozen=True)
class CameraRouteSegment:
    target_id: str
    success: bool
    message: str
    planning_time_s: float
    waypoint_count: int
    trajectory: tuple[tuple[float, ...], ...] = ()
    target_position_error_m: float | None = None
    target_orientation_error_rad: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "success": self.success,
            "message": self.message,
            "planning_time_s": self.planning_time_s,
            "waypoint_count": self.waypoint_count,
            "trajectory": [list(waypoint) for waypoint in self.trajectory],
            "target_position_error_m": self.target_position_error_m,
            "target_orientation_error_rad": self.target_orientation_error_rad,
        }


@dataclass(frozen=True)
class CameraRoutePlan:
    success: bool
    planner_name: str
    joint_names: tuple[str, ...]
    segments: tuple[CameraRouteSegment, ...]
    failed_target_id: str | None
    message: str
    reached_target_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "success": self.success,
            "planner_name": self.planner_name,
            "joint_names": list(self.joint_names),
            "segments": [segment.to_dict() for segment in self.segments],
            "failed_target_id": self.failed_target_id,
            "message": self.message,
            "reached_target_ids": list(self.reached_target_ids),
        }


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
    planner_name = "curobo_camera_route_planner_v1"

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
        source_states = JointState.from_position(
            torch.tensor(
                [target.reference_joint_positions for target in targets],
                device="cuda",
                dtype=torch.float32,
            ),
            joint_names=list(GEN3_JOINT_NAMES),
        )
        target_poses = planner.kinematics.compute_kinematics(source_states).tool_poses
        segments: list[CameraRouteSegment] = []

        for index, target in enumerate(targets):
            target_pose = Pose(
                position=target_poses.position[index].reshape(1, 3),
                quaternion=target_poses.quaternion[index].reshape(1, 4),
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

            positions_cuda = result.get_interpolated_plan().position.reshape(-1, len(GEN3_JOINT_NAMES))
            current = JointState.from_position(
                positions_cuda[-1].reshape(1, len(GEN3_JOINT_NAMES)),
                joint_names=list(GEN3_JOINT_NAMES),
            )
            actual_pose = planner.kinematics.compute_kinematics(current).tool_poses
            position_error, orientation_error = _pose_errors(actual_pose, target_pose)
            positions = positions_cuda.detach().cpu()
            trajectory = tuple(tuple(float(value) for value in row.tolist()) for row in positions)
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
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

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
        self._planner = MotionPlanner(config)
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
