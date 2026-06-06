from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path

add_src_to_path()

from robot_arm_pipeline.planning.curobo_motiongen_smoke import (
    MotionGenSmokeConfig,
    run_motiongen_smoke,
)


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
    for stage_name, pose_key in STAGE_POSES:
        stage_started = time.perf_counter()
        try:
            smoke_config = MotionGenSmokeConfig(
                demo_name=f"{config.get('demo_name')}_{stage_name}",
                robot=str(config.get("robot_config_path")),
                scene_model=str(config.get("world_config_path")),
                goal_pose=_as_pose(config[pose_key]),
                max_attempts=int(config.get("max_attempts", 1)),
                num_ik_seeds=int(config.get("num_ik_seeds", 16)),
                num_trajopt_seeds=int(config.get("num_trajopt_seeds", 2)),
                use_cuda_graph=bool(config.get("use_cuda_graph", False)),
                enable_graph_attempt=int(config.get("enable_graph_attempt", 10)),
                is_demo_config=bool(config.get("is_demo_config", True)),
                notes=str(config.get("notes", "")),
            )
            result = run_motiongen_smoke(smoke_config)
            stages.append(
                {
                    "stage_name": stage_name,
                    "success": result.success,
                    "message": result.message,
                    "failure_category": result.failure_category,
                    "trajectory_available": result.trajectory_available,
                    "motiongen_api_called": result.motiongen_api_called,
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
    base_report["motiongen_entered"] = any(bool(stage.get("motiongen_api_called")) for stage in stages)
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


def _as_pose(values: Any) -> tuple[float, float, float, float, float, float, float]:
    pose = tuple(float(value) for value in values)
    if len(pose) != 7:
        raise ValueError("pose must be [x, y, z, qw, qx, qy, qz]")
    return pose


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
