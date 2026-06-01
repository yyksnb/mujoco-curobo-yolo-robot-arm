import json
import subprocess
import sys
from pathlib import Path

import pytest

from robot_arm_pipeline.perception import load_bodex_grasp_target, load_yolo_detection
from robot_arm_pipeline.stage3_pipeline import run_stage3_pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
YOLO_EXAMPLE = REPO_ROOT / "examples" / "yolo_detection.json"
BODEX_EXAMPLE = REPO_ROOT / "examples" / "bodex_grasp_target.json"
MUJOCO_MODEL = REPO_ROOT / "examples" / "mujoco" / "minimal_six_joint_arm.xml"


def test_yolo_json_can_be_loaded() -> None:
    detection = load_yolo_detection(YOLO_EXAMPLE)

    assert detection.object_id == "object_001"
    assert detection.class_name == "mock_cube"
    assert detection.T_world_object[2][3] == 0.08


def test_bodex_json_can_be_loaded() -> None:
    grasp_target = load_bodex_grasp_target(BODEX_EXAMPLE)

    assert grasp_target.object_id == "object_001"
    assert grasp_target.pose.position == (0.45, 0.05, 0.2)
    assert grasp_target.hand_joint_goal == (0.02, 0.02, 0.02, 0.02)


def test_stage3_pipeline_fails_on_object_id_mismatch(tmp_path: Path) -> None:
    mismatched_bodex = tmp_path / "mismatched_bodex.json"
    payload = json.loads(BODEX_EXAMPLE.read_text(encoding="utf-8"))
    payload["object_id"] = "other_object"
    mismatched_bodex.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="object_id does not match"):
        run_stage3_pipeline(
            yolo_detection_path=YOLO_EXAMPLE,
            bodex_grasp_path=mismatched_bodex,
            mujoco_model_path=MUJOCO_MODEL,
            output_dir=tmp_path,
        )


def test_yolo_detection_without_world_pose_has_clear_error(tmp_path: Path) -> None:
    missing_pose = tmp_path / "missing_pose_yolo.json"
    payload = json.loads(YOLO_EXAMPLE.read_text(encoding="utf-8"))
    payload.pop("T_world_object")
    missing_pose.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not estimate 6D pose from 2D bbox_xyxy"):
        load_yolo_detection(missing_pose)


def test_run_stage3_pipeline_script_generates_final_report(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_stage3_pipeline.py"),
            "--yolo-detection",
            str(YOLO_EXAMPLE),
            "--bodex-grasp",
            str(BODEX_EXAMPLE),
            "--model",
            str(MUJOCO_MODEL),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report_path = tmp_path / "reports" / "stage3_evaluation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert "success=" in result.stdout
    assert report["object_id"] == "object_001"
    assert report["planning_success"] is True
    assert (tmp_path / "trajectories" / "object_001_trajectory.json").exists()
