from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path

add_src_to_path()

from robot_arm_pipeline.planning.curobo_planner import CuroboPlanner
from robot_arm_pipeline.types import GraspTarget, ObjectPose, PlanningRequest, Pose3D, RobotState


DEFAULT_CUROBO_ROOT = Path("/home/yyk/projects/curobo")
STAGE_POSES = (
    ("start_to_pregrasp", "pregrasp_pose_world"),
    ("pregrasp_to_grasp", "grasp_pose_world"),
    ("grasp_to_lift", "lift_pose_world"),
    ("lift_to_retreat", "retreat_pose_world"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Stage 5 cuRobo pick-lift planning demo scaffold.")
    parser.add_argument("--config", type=Path, default=Path("configs/curobo/example_pick_lift_demo.json"))
    parser.add_argument("--curobo-root", type=Path, default=DEFAULT_CUROBO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    report = run_demo(args.config, args.curobo_root)
    report_path = save_report(args.output_dir, report)
    print(
        f"success={report['success']} failure_category={report['failure_category']} "
        f"stages={len(report['stages'])} report={report_path}"
    )


def run_demo(config_path: Path, curobo_root: Path = DEFAULT_CUROBO_ROOT) -> dict[str, Any]:
    started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    robot_path = _resolve_curobo_config(config.get("robot_config_path"), curobo_root, "robot")
    world_path = _resolve_curobo_config(config.get("world_config_path"), curobo_root, "scene")

    base_report: dict[str, Any] = {
        "success": False,
        "demo_name": config.get("demo_name"),
        "config_path": str(config_path),
        "robot_config_path": config.get("robot_config_path"),
        "resolved_robot_config_path": str(robot_path) if robot_path else None,
        "world_config_path": config.get("world_config_path"),
        "resolved_world_config_path": str(world_path) if world_path else None,
        "is_demo_config": bool(config.get("is_demo_config", True)),
        "motiongen_entered": False,
        "stages": [],
        "failure_category": None,
        "message": "",
        "duration_s": 0.0,
    }

    env_message = _environment_message()
    if env_message:
        return _finish(base_report, "environment", env_message, started)
    if robot_path is None or not robot_path.exists():
        return _finish(base_report, "robot_config", f"robot_config_path does not exist: {config.get('robot_config_path')}", started)
    if world_path is None or not world_path.exists():
        return _finish(base_report, "world_config", f"world_config_path does not exist: {config.get('world_config_path')}", started)

    stages: list[dict[str, Any]] = []
    current_state = tuple(float(value) for value in config["start_joint_state"])
    for stage_name, pose_key in STAGE_POSES:
        stage_started = time.perf_counter()
        try:
            request = _build_request(config, pose_key, current_state)
            planner = CuroboPlanner(
                robot_config_path=robot_path,
                world_config_path=world_path,
                ee_link=config.get("ee_link"),
                base_link=config.get("base_link"),
                joint_names=tuple(config.get("joint_names", ())),
                use_cuda=True,
            )
            result = planner.plan(request)
            category = None if result.success else _categorize_message(result.message)
            stages.append(
                {
                    "stage_name": stage_name,
                    "success": result.success,
                    "message": result.message,
                    "failure_category": category,
                    "trajectory_available": result.trajectory is not None,
                    "planning_time_sec": round(time.perf_counter() - stage_started, 6),
                }
            )
            if not result.success:
                break
        except ValueError as exc:
            stages.append(_failed_stage(stage_name, "pose_conversion", str(exc), stage_started))
            break
        except Exception as exc:  # pragma: no cover - protects against cuRobo API/runtime differences.
            stages.append(_failed_stage(stage_name, "motiongen_api", f"{type(exc).__name__}: {exc}", stage_started))
            break

    base_report["stages"] = stages
    success = bool(stages) and all(bool(stage["success"]) for stage in stages)
    base_report["success"] = success
    first_failure = next((stage for stage in stages if not stage["success"]), None)
    if first_failure:
        base_report["failure_category"] = first_failure["failure_category"]
        base_report["message"] = first_failure["message"]
    else:
        base_report["failure_category"] = None
        base_report["message"] = "All demo planning stages completed."
    base_report["duration_s"] = round(time.perf_counter() - started, 6)
    return base_report


def save_report(output_dir: Path, report: dict[str, Any]) -> Path:
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "curobo_pick_lift_demo_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def _environment_message() -> str | None:
    if importlib.util.find_spec("torch") is None:
        return "torch is not installed."
    if importlib.util.find_spec("curobo") is None:
        return "cuRobo is not installed."
    import torch

    if not torch.cuda.is_available():
        return "torch CUDA is not available."
    return None


def _resolve_curobo_config(value: Any, curobo_root: Path, category: str) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.exists():
        return path
    if path.is_absolute():
        return path
    subdir = "robot" if category == "robot" else "scene"
    candidate = curobo_root / "curobo" / "content" / "configs" / subdir / path.name
    return candidate


def _build_request(config: dict[str, Any], pose_key: str, joint_positions: tuple[float, ...]) -> PlanningRequest:
    joint_names = tuple(str(name) for name in config["joint_names"])
    if len(joint_names) != len(joint_positions):
        raise ValueError("joint_names and start_joint_state must have the same length")

    pose_values = tuple(float(value) for value in config[pose_key])
    object_values = tuple(float(value) for value in config["object_pose_world"])
    pose = _pose_from_curobo_pose(pose_values)
    transform = _pose_to_transform(pose_values)
    object_pose = _pose_from_curobo_pose(object_values)
    object_transform = _pose_to_transform(object_values)
    return PlanningRequest(
        object_pose=ObjectPose(
            object_id="stage5_demo_cube",
            label="demo_cube",
            pose=object_pose,
            T_world_object=object_transform,
        ),
        grasp_target=GraspTarget(
            object_id="stage5_demo_cube",
            pose=pose,
            approach_vector=(0.0, 0.0, -1.0),
            gripper_width_m=0.04,
            T_world_pregrasp=transform,
            T_world_grasp=transform,
        ),
        robot_state=RobotState(joint_names=joint_names, joint_positions=joint_positions),
    )


def _pose_from_curobo_pose(values: tuple[float, ...]) -> Pose3D:
    if len(values) != 7:
        raise ValueError("pose must be [x, y, z, qw, qx, qy, qz]")
    x, y, z, qw, qx, qy, qz = values
    return Pose3D(position=(x, y, z), orientation_xyzw=(qx, qy, qz, qw))


def _pose_to_transform(values: tuple[float, ...]) -> tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]:
    if len(values) != 7:
        raise ValueError("pose must be [x, y, z, qw, qx, qy, qz]")
    x, y, z, qw, qx, qy, qz = values
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), x),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), y),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), z),
        (0.0, 0.0, 0.0, 1.0),
    )


def _categorize_message(message: str) -> str:
    lowered = message.lower()
    if "real cuda motion planning is not implemented" in lowered:
        return "motiongen_api"
    if "cuda" in lowered or "installed" in lowered:
        return "environment"
    if "robot_config" in lowered:
        return "robot_config"
    if "world_config" in lowered:
        return "world_config"
    if "joint" in lowered:
        return "joint_order"
    if "pose" in lowered or "transform" in lowered:
        return "pose_conversion"
    return "unknown"


def _failed_stage(stage_name: str, category: str, message: str, started: float) -> dict[str, Any]:
    return {
        "stage_name": stage_name,
        "success": False,
        "message": message,
        "failure_category": category,
        "trajectory_available": False,
        "planning_time_sec": round(time.perf_counter() - started, 6),
    }


def _finish(report: dict[str, Any], category: str, message: str, started: float) -> dict[str, Any]:
    report["failure_category"] = category
    report["message"] = message
    report["duration_s"] = round(time.perf_counter() - started, 6)
    return report


if __name__ == "__main__":
    main()
