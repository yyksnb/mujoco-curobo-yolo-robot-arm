import json
import subprocess
import sys
from pathlib import Path

from robot_arm_pipeline.stage3_pipeline import run_stage3_pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
YOLO_EXAMPLE = REPO_ROOT / "examples" / "yolo_detection.json"
BODEX_EXAMPLE = REPO_ROOT / "examples" / "bodex_grasp_target.json"
MUJOCO_MODEL = REPO_ROOT / "examples" / "mujoco" / "minimal_six_joint_arm.xml"


def test_run_curobo_motiongen_demo_gracefully_fails_until_real_robot_config_exists(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_curobo_motiongen_demo.py"),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report_path = tmp_path / "reports" / "curobo_motiongen_demo_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert "success=False" in result.stdout
    assert report["success"] is False
    assert report["trajectory_available"] is False
    assert report["message"]
    expected_messages = (
        "configuration does not exist",
        "requires torch and cuRobo",
    )
    assert any(message in report["message"] for message in expected_messages)


def test_stage3_pipeline_still_runs_with_motiongen_demo_present(tmp_path: Path) -> None:
    result = run_stage3_pipeline(
        yolo_detection_path=YOLO_EXAMPLE,
        bodex_grasp_path=BODEX_EXAMPLE,
        mujoco_model_path=MUJOCO_MODEL,
        output_dir=tmp_path,
    )

    assert result.success is True
    assert (tmp_path / "reports" / "stage3_evaluation_report.json").exists()
