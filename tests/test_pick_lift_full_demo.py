import json
import subprocess
import sys
from pathlib import Path

from robot_arm_pipeline.stage3_pipeline import run_stage3_pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
YOLO_EXAMPLE = REPO_ROOT / "examples" / "yolo_detection.json"
BODEX_EXAMPLE = REPO_ROOT / "examples" / "bodex_grasp_target.json"
MUJOCO_MODEL = REPO_ROOT / "examples" / "mujoco" / "minimal_six_joint_arm.xml"


def test_curobo_pick_lift_demo_gracefully_generates_report(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_curobo_pick_lift_demo.py"),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report_path = tmp_path / "reports" / "curobo_pick_lift_demo_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert "failure_category=" in result.stdout
    assert report_path.exists()
    assert "success" in report
    assert "failure_category" in report
    assert "stages" in report
    assert report["is_demo_config"] is True


def test_pick_lift_full_demo_generates_summary_report(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_pick_lift_full_demo.py"),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report_path = tmp_path / "reports" / "pick_lift_full_demo_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert "overall_status=" in result.stdout
    assert report_path.exists()
    assert "planning_demo_success" in report
    assert "physics_demo_success" in report
    assert report["overall_status"]


def test_stage3_pipeline_still_runs_with_stage5_demos_present(tmp_path: Path) -> None:
    result = run_stage3_pipeline(
        yolo_detection_path=YOLO_EXAMPLE,
        bodex_grasp_path=BODEX_EXAMPLE,
        mujoco_model_path=MUJOCO_MODEL,
        output_dir=tmp_path,
    )

    assert result.success is True
    assert (tmp_path / "reports" / "stage3_evaluation_report.json").exists()
