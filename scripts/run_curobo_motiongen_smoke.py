from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path

add_src_to_path()

from robot_arm_pipeline.planning.curobo_motiongen_smoke import (
    MotionGenSmokeConfig,
    run_motiongen_smoke,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real cuRobo V2 MotionPlanner smoke test.")
    parser.add_argument("--config", type=Path, default=Path("configs/curobo/official_motiongen_smoke_test.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    config = _load_config(args.config)
    result = run_motiongen_smoke(config)
    trajectory_path = _save_trajectory(args.output_dir, result.trajectory) if result.trajectory else None
    report_path = _save_report(args.output_dir, args.config, config, result.to_report(trajectory_path))
    print(
        f"success={result.success} motiongen_api_called={result.motiongen_api_called} "
        f"failure_category={result.failure_category} trajectory_available={result.trajectory_available} "
        f"report={report_path}"
    )


def _load_config(path: Path) -> MotionGenSmokeConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return MotionGenSmokeConfig(
        demo_name=str(payload["demo_name"]),
        robot=str(payload["robot"]),
        scene_model=payload.get("scene_model"),
        goal_pose=tuple(float(value) for value in payload["goal_pose"]),
        max_attempts=int(payload.get("max_attempts", 1)),
        num_ik_seeds=int(payload.get("num_ik_seeds", 16)),
        num_trajopt_seeds=int(payload.get("num_trajopt_seeds", 2)),
        use_cuda_graph=bool(payload.get("use_cuda_graph", False)),
        enable_graph_attempt=int(payload.get("enable_graph_attempt", 10)),
        is_demo_config=bool(payload.get("is_demo_config", True)),
        notes=str(payload.get("notes", "")),
    )


def _save_trajectory(output_dir: Path, trajectory: dict[str, Any]) -> Path:
    trajectory_dir = output_dir / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    path = trajectory_dir / "curobo_motiongen_smoke_trajectory.json"
    path.write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
    return path


def _save_report(output_dir: Path, config_path: Path, config: MotionGenSmokeConfig, report: dict[str, Any]) -> Path:
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "curobo_motiongen_smoke_report.json"
    payload = {
        **report,
        "config_path": str(config_path),
        "demo_name": config.demo_name,
        "notes": config.notes,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
