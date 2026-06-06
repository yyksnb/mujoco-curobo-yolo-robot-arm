from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CUROBO_ROOT = Path("/home/yyk/projects/curobo")


@dataclass(frozen=True)
class MotionGenSmokeConfig:
    demo_name: str
    robot: str
    scene_model: str | None
    goal_pose: tuple[float, float, float, float, float, float, float]
    max_attempts: int = 1
    num_ik_seeds: int = 16
    num_trajopt_seeds: int = 2
    use_cuda_graph: bool = False
    enable_graph_attempt: int = 10
    is_demo_config: bool = True
    notes: str = ""


@dataclass
class MotionGenSmokeResult:
    success: bool
    message: str
    failure_category: str | None
    motiongen_api_called: bool
    trajectory_available: bool
    trajectory: dict[str, Any] | None = None
    planning_time_sec: float = 0.0
    robot: str | None = None
    scene_model: str | None = None
    joint_names: list[str] = field(default_factory=list)
    tool_frames: list[str] = field(default_factory=list)
    status: str | None = None

    def to_report(self, trajectory_path: Path | None = None) -> dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "failure_category": self.failure_category,
            "motiongen_api_called": self.motiongen_api_called,
            "trajectory_available": self.trajectory_available,
            "trajectory_path": str(trajectory_path) if trajectory_path else None,
            "planning_time_sec": self.planning_time_sec,
            "robot": self.robot,
            "scene_model": self.scene_model,
            "joint_names": self.joint_names,
            "tool_frames": self.tool_frames,
            "status": self.status,
            "is_demo_config": True,
        }


def run_motiongen_smoke(config: MotionGenSmokeConfig) -> MotionGenSmokeResult:
    env_message = _validate_environment()
    if env_message:
        return MotionGenSmokeResult(
            success=False,
            message=env_message,
            failure_category="cuda_runtime" if "CUDA" in env_message else "environment",
            motiongen_api_called=False,
            trajectory_available=False,
            robot=config.robot,
            scene_model=config.scene_model,
        )

    try:
        return _run_motion_planner(config)
    except ValueError as exc:
        return MotionGenSmokeResult(
            success=False,
            message=str(exc),
            failure_category=_categorize_exception(exc),
            motiongen_api_called=False,
            trajectory_available=False,
            robot=config.robot,
            scene_model=config.scene_model,
        )
    except Exception as exc:  # pragma: no cover - protects against cuRobo runtime/API differences.
        return MotionGenSmokeResult(
            success=False,
            message=f"{type(exc).__name__}: {exc}",
            failure_category=_categorize_exception(exc),
            motiongen_api_called=True,
            trajectory_available=False,
            robot=config.robot,
            scene_model=config.scene_model,
        )


def _run_motion_planner(config: MotionGenSmokeConfig) -> MotionGenSmokeResult:
    import torch
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import GoalToolPose, JointState, Pose

    robot_arg = _resolve_curobo_config_arg(config.robot, "robot")
    scene_arg = _resolve_curobo_config_arg(config.scene_model, "scene") if config.scene_model else None
    _validate_pose(config.goal_pose)

    cfg = MotionPlannerCfg.create(
        robot=robot_arg,
        scene_model=scene_arg,
        num_ik_seeds=config.num_ik_seeds,
        num_trajopt_seeds=config.num_trajopt_seeds,
        use_cuda_graph=config.use_cuda_graph,
        max_goalset=1,
    )
    planner = MotionPlanner(cfg)
    q_start = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0),
        joint_names=planner.joint_names,
    )
    x, y, z, qw, qx, qy, qz = config.goal_pose
    pose = Pose(
        position=torch.tensor([[x, y, z]], device="cuda", dtype=torch.float32),
        quaternion=torch.tensor([[qw, qx, qy, qz]], device="cuda", dtype=torch.float32),
    )
    goal = GoalToolPose.from_poses({planner.tool_frames[0]: pose}, ordered_tool_frames=planner.tool_frames)

    started = time.perf_counter()
    result = planner.plan_pose(
        goal_tool_poses=goal,
        current_state=q_start,
        max_attempts=config.max_attempts,
        enable_graph_attempt=config.enable_graph_attempt,
    )
    planning_time_sec = round(time.perf_counter() - started, 6)

    if result is None:
        return MotionGenSmokeResult(
            success=False,
            message="MotionPlanner.plan_pose returned None.",
            failure_category="goal_pose",
            motiongen_api_called=True,
            trajectory_available=False,
            planning_time_sec=planning_time_sec,
            robot=config.robot,
            scene_model=config.scene_model,
            joint_names=list(planner.joint_names),
            tool_frames=list(planner.tool_frames),
        )

    success = bool(result.success is not None and result.success.any().item())
    status = str(getattr(result, "status", None))
    if not success:
        return MotionGenSmokeResult(
            success=False,
            message=f"MotionPlanner.plan_pose did not find a valid plan. status={status}",
            failure_category="goal_pose",
            motiongen_api_called=True,
            trajectory_available=False,
            planning_time_sec=planning_time_sec,
            robot=config.robot,
            scene_model=config.scene_model,
            joint_names=list(planner.joint_names),
            tool_frames=list(planner.tool_frames),
            status=status,
        )

    interpolated = result.get_interpolated_plan()
    trajectory = _trajectory_to_dict(interpolated, planner.joint_names, config)
    return MotionGenSmokeResult(
        success=True,
        message="MotionPlanner.plan_pose succeeded.",
        failure_category=None,
        motiongen_api_called=True,
        trajectory_available=True,
        trajectory=trajectory,
        planning_time_sec=planning_time_sec,
        robot=config.robot,
        scene_model=config.scene_model,
        joint_names=list(planner.joint_names),
        tool_frames=list(planner.tool_frames),
        status=status,
    )


def _trajectory_to_dict(interpolated: Any, joint_names: list[str], config: MotionGenSmokeConfig) -> dict[str, Any]:
    positions = interpolated.position.detach().cpu().reshape(-1, interpolated.position.shape[-1])
    dt_value = 0.025
    if getattr(interpolated, "dt", None) is not None:
        dt_tensor = interpolated.dt.detach().cpu().reshape(-1)
        if len(dt_tensor):
            dt_value = float(dt_tensor[0])
    waypoints = []
    for index, row in enumerate(positions):
        waypoints.append(
            {
                "time_s": round(index * dt_value, 6),
                "joint_positions": [float(value) for value in row.tolist()],
            }
        )
    return {
        "planner_name": "curobo_motion_planner_v2_demo",
        "target_object_id": "stage5_1_motiongen_smoke_goal",
        "joint_names": list(joint_names),
        "waypoints": waypoints,
        "metadata": {
            "demo_name": config.demo_name,
            "robot": config.robot,
            "scene_model": config.scene_model,
            "is_demo_config": config.is_demo_config,
            "notes": config.notes,
        },
    }


def _validate_environment() -> str | None:
    if importlib.util.find_spec("torch") is None:
        return "torch is not installed."
    if importlib.util.find_spec("curobo") is None:
        return "cuRobo is not installed."
    try:
        import torch
    except Exception as exc:
        return f"torch import failed: {exc}"
    if not torch.cuda.is_available():
        return "torch CUDA is not available."
    return None


def _resolve_curobo_config_arg(value: str | None, category: str) -> str:
    if not value:
        raise ValueError(f"{category} config is required")
    path = Path(value)
    if path.exists():
        return str(path)
    if path.is_absolute():
        raise ValueError(f"{category} config does not exist: {value}")
    subdir = "robot" if category == "robot" else "scene"
    candidate = CUROBO_ROOT / "curobo" / "content" / "configs" / subdir / path.name
    if candidate.exists():
        return path.name
    raise ValueError(f"{category} config does not exist: {value}")


def _validate_pose(values: tuple[float, float, float, float, float, float, float]) -> None:
    if len(values) != 7:
        raise ValueError("goal_pose must be [x, y, z, qw, qx, qy, qz]")


def _categorize_exception(exc: Exception) -> str:
    text = str(exc).lower()
    if "robot" in text:
        return "robot_config"
    if "scene" in text or "world" in text:
        return "world_config"
    if "joint" in text:
        return "joint_order"
    if "pose" in text or "goal" in text:
        return "goal_pose"
    if "cuda" in text or "core" in text:
        return "cuda_runtime"
    if "motionplanner" in text or "plan_pose" in text or "motion" in text:
        return "motiongen_api"
    return "unknown"
