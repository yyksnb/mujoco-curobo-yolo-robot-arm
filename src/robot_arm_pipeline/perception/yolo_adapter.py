from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot_arm_pipeline.types import ObjectDetection, ObjectPose, Pose3D, TransformMatrix


CONFIG_SCHEMA_VERSION = "stage3_yolo_model_migration_v1"
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class YoloClassMapping:
    yolo_id: int
    yolo_name: str
    stage3_class_name: str
    default_object_id: str


@dataclass(frozen=True)
class Stage3YoloProfile:
    profile_name: str
    config_path: Path
    model_format: str
    model_path: Path
    dataset_yaml_path: Path | None
    default_camera_name: str
    default_mock_confidence: float
    default_mock_bbox_xyxy: tuple[float, float, float, float]
    default_mock_T_world_object: TransformMatrix
    pose_source_when_mocked: str
    class_mappings: tuple[YoloClassMapping, ...]
    raw_payload: dict[str, Any]

    def mapping_for_detection(self, detection: dict[str, Any]) -> YoloClassMapping:
        if "class_id" in detection:
            return self.mapping_for_yolo_id(int(detection["class_id"]))
        if "category_id" in detection:
            return self.mapping_for_yolo_id(int(detection["category_id"]))
        if "cls" in detection:
            return self.mapping_for_yolo_id(int(detection["cls"]))

        raw_name = detection.get("class_name") or detection.get("name") or detection.get("label")
        if raw_name is None:
            raise ValueError("YOLO raw detection must include class_id/category_id/cls or class_name/name/label")
        return self.mapping_for_name(str(raw_name))

    def mapping_for_yolo_id(self, yolo_id: int) -> YoloClassMapping:
        for mapping in self.class_mappings:
            if mapping.yolo_id == yolo_id:
                return mapping
        raise ValueError(f"YOLO class id is not mapped in profile {self.profile_name}: {yolo_id}")

    def mapping_for_name(self, name: str) -> YoloClassMapping:
        for mapping in self.class_mappings:
            if name in {mapping.yolo_name, mapping.stage3_class_name, mapping.default_object_id}:
                return mapping
        raise ValueError(f"YOLO class name is not mapped in profile {self.profile_name}: {name}")


## Stage 3 canonical detection adapter.


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


## YOLO profile and raw-output adapter.


def load_stage3_yolo_profile(path: Path | str) -> Stage3YoloProfile:
    config_path = Path(path)
    payload = _load_yaml_payload(config_path)
    if _required_str(payload, "schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"unsupported YOLO profile schema_version: {payload.get('schema_version')}")

    artifact = _required_dict(payload, "artifact")
    inference = _required_dict(payload, "inference")
    stage3 = _required_dict(payload, "stage3")

    class_mappings = tuple(_class_mapping(item) for item in _required_list(stage3, "class_map"))
    if not class_mappings:
        raise ValueError("YOLO profile stage3.class_map must not be empty")

    dataset_yaml_path = artifact.get("dataset_yaml_path")
    return Stage3YoloProfile(
        profile_name=_required_str(payload, "profile_name"),
        config_path=config_path,
        model_format=_required_str(artifact, "model_format"),
        model_path=_resolve_project_path(_required_str(artifact, "model_path"), "artifact.model_path"),
        dataset_yaml_path=(
            _resolve_project_path(str(dataset_yaml_path), "artifact.dataset_yaml_path")
            if dataset_yaml_path is not None
            else None
        ),
        default_camera_name=_required_str(inference, "default_camera_name"),
        default_mock_confidence=float(stage3.get("default_mock_confidence", 0.0)),
        default_mock_bbox_xyxy=_float_tuple_from_value(
            stage3.get("default_mock_bbox_xyxy", [0.0, 0.0, 1.0, 1.0]),
            4,
        ),
        default_mock_T_world_object=_transform_matrix(
            stage3.get(
                "default_mock_T_world_object",
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.03],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ),
            "stage3.default_mock_T_world_object",
        ),
        pose_source_when_mocked=str(stage3.get("pose_source_when_mocked", "mock_pose")),
        class_mappings=class_mappings,
        raw_payload=payload,
    )


def convert_yolo_raw_payload_to_stage3_detection(
    payload: dict[str, Any],
    profile: Stage3YoloProfile,
    *,
    detection_index: int = 0,
    allow_mock_fields: bool = False,
) -> dict[str, Any]:
    detections = _detections_from_payload(payload)
    if detection_index < 0 or detection_index >= len(detections):
        raise ValueError(f"detection_index {detection_index} is outside raw detections length {len(detections)}")

    detection = detections[detection_index]
    mapping = profile.mapping_for_detection(detection)
    confidence = _confidence(detection, profile, allow_mock_fields=allow_mock_fields)
    bbox_xyxy = _bbox_xyxy(detection, profile, allow_mock_fields=allow_mock_fields)
    t_world_object, pose_source = _t_world_object(detection, profile, allow_mock_fields=allow_mock_fields)

    result: dict[str, Any] = {
        "object_id": str(detection.get("object_id") or mapping.default_object_id),
        "class_name": mapping.stage3_class_name,
        "confidence": confidence,
        "bbox_xyxy": list(bbox_xyxy),
        "T_world_object": [list(row) for row in t_world_object],
        "source_profile": profile.profile_name,
        "source_yolo_class_id": mapping.yolo_id,
        "source_yolo_class_name": mapping.yolo_name,
        "camera_name": str(payload.get("camera_name") or detection.get("camera_name") or profile.default_camera_name),
    }
    if "image_path" in payload or "image_path" in detection:
        result["image_path"] = str(payload.get("image_path") or detection.get("image_path"))
    if pose_source is not None:
        result["pose_source"] = pose_source
    return result


def convert_yolo_raw_file_to_stage3_detection(
    raw_path: Path | str,
    profile_path: Path | str,
    *,
    detection_index: int = 0,
    allow_mock_fields: bool = False,
) -> dict[str, Any]:
    payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("YOLO raw file must contain a JSON object")
    profile = load_stage3_yolo_profile(profile_path)
    return convert_yolo_raw_payload_to_stage3_detection(
        payload,
        profile,
        detection_index=detection_index,
        allow_mock_fields=allow_mock_fields,
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
    try:
        return _float_tuple_from_value(value, length)
    except ValueError as exc:
        raise ValueError(f"YOLO detection field {key} must be a list of {length} numbers") from exc


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


def _load_yaml_payload(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise ValueError(
                "YOLO profile YAML must be JSON-compatible unless PyYAML is installed."
            ) from exc
        loaded = yaml.safe_load(text)
        payload = loaded
    if not isinstance(payload, dict):
        raise ValueError("YOLO profile YAML must contain a mapping/object")
    return payload


def _detections_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    detections = payload.get("detections")
    if detections is None and "bbox_xyxy" in payload:
        detections = [payload]
    if not isinstance(detections, list) or not detections:
        raise ValueError("YOLO raw payload must contain a non-empty detections list")
    result: list[dict[str, Any]] = []
    for detection in detections:
        if not isinstance(detection, dict):
            raise ValueError("each YOLO raw detection must be an object")
        result.append(detection)
    return result


def _confidence(detection: dict[str, Any], profile: Stage3YoloProfile, *, allow_mock_fields: bool) -> float:
    for key in ("confidence", "score", "conf"):
        if key in detection:
            return float(detection[key])
    if allow_mock_fields:
        return profile.default_mock_confidence
    raise ValueError("YOLO raw detection is missing confidence/score/conf")


def _bbox_xyxy(
    detection: dict[str, Any],
    profile: Stage3YoloProfile,
    *,
    allow_mock_fields: bool,
) -> tuple[float, float, float, float]:
    if "bbox_xyxy" in detection:
        return _float_tuple_from_value(detection["bbox_xyxy"], 4)
    if "bbox_xywh" in detection:
        x, y, width, height = _float_tuple_from_value(detection["bbox_xywh"], 4)
        return (x, y, x + width, y + height)
    if "bbox" in detection:
        bbox = _float_tuple_from_value(detection["bbox"], 4)
        bbox_format = str(detection.get("bbox_format", "xywh"))
        if bbox_format == "xyxy":
            return bbox
        if bbox_format == "xywh":
            x, y, width, height = bbox
            return (x, y, x + width, y + height)
        raise ValueError(f"unsupported YOLO bbox_format: {bbox_format}")
    if allow_mock_fields:
        return profile.default_mock_bbox_xyxy
    raise ValueError("YOLO raw detection is missing bbox_xyxy/bbox_xywh/bbox")


def _t_world_object(
    detection: dict[str, Any],
    profile: Stage3YoloProfile,
    *,
    allow_mock_fields: bool,
) -> tuple[TransformMatrix, str | None]:
    if "T_world_object" in detection:
        return _transform_matrix(detection["T_world_object"], "T_world_object"), detection.get("pose_source")
    if allow_mock_fields:
        return profile.default_mock_T_world_object, profile.pose_source_when_mocked
    raise ValueError(
        "T_world_object is required by the Stage 3 planning contract. "
        "Use --allow-mock-fields only for recognition steps that do not consume pose."
    )


def _class_mapping(payload: Any) -> YoloClassMapping:
    if not isinstance(payload, dict):
        raise ValueError("YOLO profile class_map entries must be objects")
    return YoloClassMapping(
        yolo_id=int(_required(payload, "yolo_id")),
        yolo_name=_required_str(payload, "yolo_name"),
        stage3_class_name=_required_str(payload, "stage3_class_name"),
        default_object_id=_required_str(payload, "default_object_id"),
    )


def _resolve_project_path(value: str, key: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"YOLO profile {key} must be project-relative, got absolute path: {value}")
    resolved = (REPO_ROOT / path).resolve()
    if not resolved.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError(f"YOLO profile {key} must stay inside the project: {value}")
    return resolved


def _required_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = _required(payload, key)
    if not isinstance(value, dict):
        raise ValueError(f"YOLO profile field {key} must be an object")
    return value


def _required_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = _required(payload, key)
    if not isinstance(value, list):
        raise ValueError(f"YOLO profile field {key} must be a list")
    return value


def _float_tuple_from_value(value: Any, length: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"expected a list of {length} numbers")
    return tuple(float(item) for item in value)
