from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 5 integrated pick-lift demo scaffold.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    report = run_full_demo(args.output_dir)
    report_path = save_report(args.output_dir, report)
    print(f"overall_status={report['overall_status']} report={report_path}")


def run_full_demo(output_dir: Path = Path("outputs")) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    _run_script(repo_root / "scripts" / "run_curobo_pick_lift_demo.py", output_dir)
    _run_script(repo_root / "scripts" / "run_mujoco_physical_grasp_demo.py", output_dir)

    planning_report = _load_json(output_dir / "reports" / "curobo_pick_lift_demo_report.json")
    physics_report = _load_json(output_dir / "reports" / "mujoco_physical_grasp_demo_report.json")
    planning_success = bool(planning_report.get("success"))
    physics_success = bool(physics_report.get("success"))
    planning_failure_category = planning_report.get("failure_category")

    if planning_success and physics_success:
        overall_status = "success"
        next_step = "Bridge cuRobo trajectory output into MuJoCo execution with matching robot and gripper joints."
    elif physics_success and not planning_success:
        overall_status = "partial_success_physics_ready_planning_needs_robot_config"
        next_step = "Prepare real cuRobo robot/world config and then connect planned trajectories to MuJoCo."
    elif not planning_success and bool(physics_report.get("simulation_ran")):
        overall_status = "partial_success_simulation_runs_planning_needs_robot_config_physics_needs_tuning"
        next_step = "Prepare real cuRobo configs and tune toy gripper actuator/friction/contact parameters."
    else:
        overall_status = "needs_environment_or_demo_model_debugging"
        next_step = "Inspect planning and physics reports before attempting integration."

    return {
        "planning_demo_success": planning_success,
        "planning_failure_category": planning_failure_category,
        "physics_demo_success": physics_success,
        "physics_contact_detected": bool(physics_report.get("contact_detected")),
        "physics_lifted_distance": physics_report.get("lifted_distance"),
        "overall_status": overall_status,
        "next_step": next_step,
        "planning_report_path": str(output_dir / "reports" / "curobo_pick_lift_demo_report.json"),
        "physics_report_path": str(output_dir / "reports" / "mujoco_physical_grasp_demo_report.json"),
    }


def save_report(output_dir: Path, report: dict[str, Any]) -> Path:
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "pick_lift_full_demo_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def _run_script(script: Path, output_dir: Path) -> None:
    subprocess.run(
        [sys.executable, str(script), "--output-dir", str(output_dir)],
        cwd=script.resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
