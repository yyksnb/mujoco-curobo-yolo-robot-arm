from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MISSING_POSE_MESSAGE = (
    "T_world_object is required by the Stage 3 planning contract. "
    "bbox_xyxy alone is not enough for cuRobo planning."
)

DEFAULT_MOCK_T_WORLD_OBJECT = [
    [1.0, 0.0, 0.0, 0.45],
    [0.0, 1.0, 0.0, 0.05],
    [0.0, 0.0, 1.0, 0.08],
    [0.0, 0.0, 0.0, 1.0],
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert YOLO raw output to the Stage 3 detection contract.")
    parser.add_argument("--raw", required=True, type=Path, help="Path to raw YOLO output JSON.")
    parser.add_argument("--output", required=True, type=Path, help="Output path for Stage 3 detection JSON.")
    parser.add_argument(
        "--allow-mock-pose",
        action="store_true",
        help="Fill a clearly marked mock T_world_object when raw YOLO has only bbox_xyxy.",
    )
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing output file: {args.output}")

    stage3_detection = convert_raw_yolo_to_stage3(args.raw, allow_mock_pose=args.allow_mock_pose)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stage3_detection, indent=2), encoding="utf-8")
    print(f"wrote={args.output}")


def convert_raw_yolo_to_stage3(raw_path: Path, *, allow_mock_pose: bool) -> dict[str, Any]:
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    detections = payload.get("detections")
    if not isinstance(detections, list) or not detections:
        raise ValueError("raw YOLO output must contain a non-empty detections list")

    detection = detections[0]
    if not isinstance(detection, dict):
        raise ValueError("raw YOLO detection must be an object")

    class_name = _required_str(detection, "class_name")
    confidence = float(_required(detection, "confidence"))
    bbox_xyxy = _float_list(detection, "bbox_xyxy", 4)
    object_id = str(detection.get("object_id") or f"{class_name}_001")

    t_world_object = detection.get("T_world_object")
    mock_pose_used = False
    if t_world_object is None:
        if not allow_mock_pose:
            raise ValueError(MISSING_POSE_MESSAGE)
        t_world_object = DEFAULT_MOCK_T_WORLD_OBJECT
        mock_pose_used = True

    result: dict[str, Any] = {
        "object_id": object_id,
        "class_name": class_name,
        "confidence": confidence,
        "bbox_xyxy": bbox_xyxy,
        "T_world_object": _transform_matrix(t_world_object),
    }
    if mock_pose_used:
        result["pose_source"] = "mock_pose_for_adapter_test_only"
        result["warning"] = "Mock T_world_object was inserted because raw YOLO output had only bbox_xyxy."
    return result


def _required(payload: dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError(f"raw YOLO detection is missing required field: {key}")
    return payload[key]


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = _required(payload, key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"raw YOLO detection field {key} must be a non-empty string")
    return value


def _float_list(payload: dict[str, Any], key: str, length: int) -> list[float]:
    value = _required(payload, key)
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"raw YOLO detection field {key} must be a list of {length} numbers")
    return [float(item) for item in value]


def _transform_matrix(value: Any) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("T_world_object must be a 4x4 matrix")
    rows: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError("T_world_object must be a 4x4 matrix")
        rows.append([float(item) for item in row])
    if rows[3] != [0.0, 0.0, 0.0, 1.0]:
        raise ValueError("T_world_object bottom row must be [0, 0, 0, 1]")
    return rows


if __name__ == "__main__":
    main()
