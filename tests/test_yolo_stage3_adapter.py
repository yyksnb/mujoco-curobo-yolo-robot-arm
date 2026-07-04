import json
from pathlib import Path

import pytest

from robot_arm_pipeline.perception import (
    build_yolo_raw_payload,
    convert_yolo_raw_payload_to_stage3_detection,
    load_stage3_yolo_profile,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YOLO_PROFILE = REPO_ROOT / "configs" / "yolo" / "stage3_default.yaml"


def test_stage3_yolo_profile_maps_teammate_classes_to_stage3_names() -> None:
    profile = load_stage3_yolo_profile(DEFAULT_YOLO_PROFILE)

    assert profile.profile_name == "stage3_default_yolo"
    assert profile.model_path == REPO_ROOT / "outputs" / "yolo_models" / "current" / "best.pt"
    assert profile.dataset_yaml_path == REPO_ROOT / "outputs" / "yolo_models" / "current" / "dataset.yaml"
    assert profile.mapping_for_yolo_id(5).stage3_class_name == "notebook"
    assert profile.mapping_for_name("内六角扳手").default_object_id == "target_hex_key"


def test_stage3_yolo_raw_detection_converts_to_stage3_contract_with_mock_pose() -> None:
    profile = load_stage3_yolo_profile(DEFAULT_YOLO_PROFILE)
    raw_payload = {
        "image_path": "outputs/images/scan_0001.png",
        "camera_name": "wrist",
        "detections": [
            {
                "class_id": 5,
                "confidence": 0.91,
                "bbox_xyxy": [120.0, 80.0, 220.0, 180.0],
            }
        ],
    }

    detection = convert_yolo_raw_payload_to_stage3_detection(
        raw_payload,
        profile,
        allow_mock_fields=True,
    )

    assert detection["object_id"] == "target_notebook"
    assert detection["class_name"] == "notebook"
    assert detection["confidence"] == 0.91
    assert detection["bbox_xyxy"] == [120.0, 80.0, 220.0, 180.0]
    assert detection["T_world_object"][2][3] == 0.03
    assert detection["pose_source"] == "mock_pose_for_stage3_yolo_profile_placeholder"


def test_local_yolo_raw_payload_uses_profile_paths_and_detection_schema() -> None:
    profile = load_stage3_yolo_profile(DEFAULT_YOLO_PROFILE)

    payload = build_yolo_raw_payload(
        image_path=REPO_ROOT / "outputs" / "images" / "scan_0001.png",
        camera_name="wrist",
        profile=profile,
        detections=[
            {
                "class_id": 5,
                "confidence": 0.91,
                "bbox_xyxy": [120, 80, 220, 180],
            }
        ],
        inference={"image_size": 640, "confidence_threshold": 0.25},
    )

    assert payload["schema_version"] == "yolo_raw_detections_v1"
    assert payload["image_path"] == "outputs/images/scan_0001.png"
    assert payload["model_path"] == "outputs/yolo_models/current/best.pt"
    assert payload["detections"][0] == {
        "class_id": 5,
        "class_name": "本子",
        "confidence": 0.91,
        "bbox_xyxy": [120.0, 80.0, 220.0, 180.0],
    }


def test_stage3_yolo_raw_detection_converts_coco_bbox_and_chinese_name() -> None:
    profile = load_stage3_yolo_profile(DEFAULT_YOLO_PROFILE)
    raw_payload = {
        "detections": [
            {
                "class_name": "纸胶带",
                "score": 0.7,
                "bbox": [10.0, 20.0, 30.0, 40.0],
            }
        ],
    }

    detection = convert_yolo_raw_payload_to_stage3_detection(
        raw_payload,
        profile,
        allow_mock_fields=True,
    )

    assert detection["object_id"] == "target_tape"
    assert detection["class_name"] == "tape"
    assert detection["bbox_xyxy"] == [10.0, 20.0, 40.0, 60.0]


def test_stage3_yolo_conversion_requires_pose_without_mock_fields() -> None:
    profile = load_stage3_yolo_profile(DEFAULT_YOLO_PROFILE)
    raw_payload = {
        "detections": [
            {
                "class_id": 6,
                "confidence": 0.8,
                "bbox_xyxy": [1.0, 2.0, 3.0, 4.0],
            }
        ],
    }

    with pytest.raises(ValueError, match="T_world_object is required"):
        convert_yolo_raw_payload_to_stage3_detection(raw_payload, profile)


def test_yolo_profile_rejects_paths_outside_project(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_YOLO_PROFILE.read_text(encoding="utf-8"))
    payload["artifact"]["model_path"] = "../external_yolo_result/best.pt"
    config = tmp_path / "bad_profile.yaml"
    config.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="must stay inside the project"):
        load_stage3_yolo_profile(config)
