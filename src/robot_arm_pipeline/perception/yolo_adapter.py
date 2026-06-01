from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from robot_arm_pipeline.types import ObjectDetection, ObjectPose, Pose3D, TransformMatrix


def fake_yolo_object_poses() -> tuple[ObjectPose, ...]:
    """Return deterministic Stage 1 object poses without calling YOLO."""
    return (
        ObjectPose(
            object_id="object_001",
            label="mock_cube",
            pose=Pose3D(position=(0.45, 0.05, 0.08)),
        ),
    )


def load_yolo_detection(path: Path | str) -> ObjectDetection:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "T_world_object" not in payload:
        raise ValueError(
            "YOLO detection is missing T_world_object. Stage 3 does not estimate "
            "6D pose from 2D bbox_xyxy; upstream must provide T_world_object."
        )
    return ObjectDetection(
        object_id=_required_str(payload, "object_id"),
        class_name=_required_str(payload, "class_name"),
        confidence=float(_required(payload, "confidence")),
        bbox_xyxy=_float_tuple(payload, "bbox_xyxy", 4),
        T_world_object=_transform_matrix(payload["T_world_object"], "T_world_object"),
    )


def object_pose_from_detection(detection: ObjectDetection) -> ObjectPose:
    return ObjectPose(
        object_id=detection.object_id,
        label=detection.class_name,
        pose=Pose3D(position=_translation(detection.T_world_object)),
        T_world_object=detection.T_world_object,
    )


def _required(payload: dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError(f"YOLO detection is missing required field: {key}")
    return payload[key]


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = _required(payload, key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"YOLO detection field {key} must be a non-empty string")
    return value


def _float_tuple(payload: dict[str, Any], key: str, length: int) -> tuple[float, ...]:
    value = _required(payload, key)
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"YOLO detection field {key} must be a list of {length} numbers")
    return tuple(float(item) for item in value)


def _transform_matrix(value: Any, key: str) -> TransformMatrix:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"YOLO detection field {key} must be a 4x4 matrix")
    rows: list[tuple[float, float, float, float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError(f"YOLO detection field {key} must be a 4x4 matrix")
        rows.append(tuple(float(item) for item in row))
    return tuple(rows)  # type: ignore[return-value]


def _translation(matrix: TransformMatrix) -> tuple[float, float, float]:
    return (matrix[0][3], matrix[1][3], matrix[2][3])
