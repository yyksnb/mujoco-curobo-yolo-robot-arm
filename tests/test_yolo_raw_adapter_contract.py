import json
import subprocess
import sys
from pathlib import Path

from robot_arm_pipeline.perception import load_yolo_detection


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_EXAMPLE = REPO_ROOT / "examples" / "yolo_raw_output_sample.json"
CONVERTER = REPO_ROOT / "scripts" / "convert_yolo_raw_to_stage3_contract.py"


def test_raw_yolo_sample_can_be_parsed() -> None:
    payload = json.loads(RAW_EXAMPLE.read_text(encoding="utf-8"))

    assert payload["camera_name"] == "wrist_camera"
    assert payload["detections"][0]["class_name"] == "mock_cube"
    assert payload["detections"][0]["bbox_xyxy"] == [120.0, 80.0, 220.0, 180.0]


def test_converter_generates_stage3_contract_with_mock_pose(tmp_path: Path) -> None:
    output = tmp_path / "converted_yolo_detection.json"

    subprocess.run(
        [
            sys.executable,
            str(CONVERTER),
            "--raw",
            str(RAW_EXAMPLE),
            "--output",
            str(output),
            "--allow-mock-pose",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    detection = load_yolo_detection(output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert detection.object_id == "object_001"
    assert detection.class_name == "mock_cube"
    assert detection.T_world_object[0][3] == 0.45
    assert payload["pose_source"] == "mock_pose_for_adapter_test_only"


def test_converter_fails_without_pose_or_mock_flag(tmp_path: Path) -> None:
    output = tmp_path / "should_not_exist.json"

    result = subprocess.run(
        [
            sys.executable,
            str(CONVERTER),
            "--raw",
            str(RAW_EXAMPLE),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "T_world_object is required by the Stage 3 planning contract" in result.stderr
    assert not output.exists()


def test_converter_keeps_existing_pose_when_present(tmp_path: Path) -> None:
    raw_with_pose = tmp_path / "raw_with_pose.json"
    payload = json.loads(RAW_EXAMPLE.read_text(encoding="utf-8"))
    payload["detections"][0]["T_world_object"] = [
        [1.0, 0.0, 0.0, 0.50],
        [0.0, 1.0, 0.0, 0.10],
        [0.0, 0.0, 1.0, 0.15],
        [0.0, 0.0, 0.0, 1.0],
    ]
    raw_with_pose.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "converted.json"

    subprocess.run(
        [sys.executable, str(CONVERTER), "--raw", str(raw_with_pose), "--output", str(output)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    detection = load_yolo_detection(output)
    converted = json.loads(output.read_text(encoding="utf-8"))

    assert detection.T_world_object[0][3] == 0.50
    assert "pose_source" not in converted
