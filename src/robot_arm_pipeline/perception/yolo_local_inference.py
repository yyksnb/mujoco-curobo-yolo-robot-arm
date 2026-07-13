from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

from robot_arm_pipeline.perception.yolo_adapter import (
    REPO_ROOT,
    Stage3YoloProfile,
    load_stage3_yolo_profile,
)


RAW_SCHEMA_VERSION = "yolo_raw_detections_v1"
_MODEL_CACHE: dict[tuple[Path, Any], Any] = {}
_MODEL_CACHE_LOCK = Lock()


def run_ultralytics_yolo_inference(
    *,
    image_path: Path | str,
    profile_path: Path | str,
    camera_name: str | None = None,
    confidence_threshold: float | None = None,
    iou_threshold: float | None = None,
    class_agnostic_nms: bool = False,
    image_size: int | None = None,
    device: str | None = None,
    max_detections: int | None = None,
) -> dict[str, Any]:
    profile = load_stage3_yolo_profile(profile_path)
    image = Path(image_path)
    if not image.exists():
        raise FileNotFoundError(f"YOLO input image does not exist: {image}")
    if profile.model_format != "ultralytics_yolo_pt":
        raise ValueError(f"unsupported YOLO model_format for local inference: {profile.model_format}")
    if not profile.model_path.exists():
        raise FileNotFoundError(f"YOLO model file does not exist: {profile.model_path}")

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError("ultralytics is required for local YOLO inference. Install it in the active env.") from exc

    inference_config = _inference_config(profile)
    resolved_confidence = (
        float(confidence_threshold)
        if confidence_threshold is not None
        else float(inference_config.get("confidence_threshold", 0.25))
    )
    resolved_iou = float(iou_threshold) if iou_threshold is not None else float(inference_config.get("iou_threshold", 0.7))
    resolved_image_size = int(image_size) if image_size is not None else int(inference_config.get("image_size", 640))

    model = _cached_yolo_model(profile.model_path, YOLO)
    predict_kwargs: dict[str, Any] = {
        "source": str(image),
        "conf": resolved_confidence,
        "iou": resolved_iou,
        "agnostic_nms": class_agnostic_nms,
        "imgsz": resolved_image_size,
        "verbose": False,
    }
    if device:
        predict_kwargs["device"] = device
    if max_detections is not None:
        predict_kwargs["max_det"] = int(max_detections)

    results = model.predict(**predict_kwargs)
    result = results[0] if results else None
    detections = _detections_from_ultralytics_result(result, profile=profile)
    return build_yolo_raw_payload(
        image_path=image,
        camera_name=camera_name or profile.default_camera_name,
        profile=profile,
        detections=detections,
        inference={
            "image_size": resolved_image_size,
            "confidence_threshold": resolved_confidence,
            "iou_threshold": resolved_iou,
            "class_agnostic_nms": class_agnostic_nms,
            "device": device,
            "max_detections": max_detections,
        },
    )


def _cached_yolo_model(model_path: Path, model_factory: Any) -> Any:
    cache_key = (model_path.resolve(), model_factory)
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(cache_key)
        if model is None:
            model = model_factory(str(cache_key[0]))
            _MODEL_CACHE[cache_key] = model
        return model


def build_yolo_raw_payload(
    *,
    image_path: Path | str,
    camera_name: str,
    profile: Stage3YoloProfile,
    detections: list[dict[str, Any]],
    inference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": RAW_SCHEMA_VERSION,
        "image_path": _project_relative_or_str(Path(image_path)),
        "camera_name": camera_name,
        "source_profile": profile.profile_name,
        "model_path": _project_relative_or_str(profile.model_path),
        "inference": {key: value for key, value in (inference or {}).items() if value is not None},
        "detections": [_normalize_detection(detection, profile) for detection in detections],
    }


def _detections_from_ultralytics_result(result: Any, *, profile: Stage3YoloProfile) -> list[dict[str, Any]]:
    if result is None or getattr(result, "boxes", None) is None:
        return []

    boxes = result.boxes
    xyxy_values = _to_list(getattr(boxes, "xyxy", []))
    confidence_values = _to_list(getattr(boxes, "conf", []))
    class_values = _to_list(getattr(boxes, "cls", []))
    names = getattr(result, "names", None)

    detections: list[dict[str, Any]] = []
    for bbox_xyxy, confidence, class_id in zip(xyxy_values, confidence_values, class_values):
        class_id_int = int(class_id)
        detections.append(
            {
                "class_id": class_id_int,
                "class_name": _class_name(class_id_int, names, profile),
                "confidence": float(confidence),
                "bbox_xyxy": [float(value) for value in bbox_xyxy],
            }
        )
    return detections


def _normalize_detection(detection: dict[str, Any], profile: Stage3YoloProfile) -> dict[str, Any]:
    if "class_id" not in detection:
        raise ValueError("local YOLO raw detections must include class_id")
    if "confidence" not in detection:
        raise ValueError("local YOLO raw detections must include confidence")
    if "bbox_xyxy" not in detection:
        raise ValueError("local YOLO raw detections must include bbox_xyxy")

    class_id = int(detection["class_id"])
    return {
        "class_id": class_id,
        "class_name": str(detection.get("class_name") or _class_name(class_id, None, profile)),
        "confidence": float(detection["confidence"]),
        "bbox_xyxy": _float_list(detection["bbox_xyxy"], length=4),
    }


def _class_name(class_id: int, names: Any, profile: Stage3YoloProfile) -> str:
    if isinstance(names, dict) and class_id in names:
        return str(names[class_id])
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    try:
        return profile.mapping_for_yolo_id(class_id).yolo_name
    except ValueError:
        return str(class_id)


def _inference_config(profile: Stage3YoloProfile) -> dict[str, Any]:
    value = profile.raw_payload.get("inference", {})
    return value if isinstance(value, dict) else {}


def _to_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _float_list(value: Any, *, length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"expected bbox_xyxy to be a list of {length} numbers")
    return [float(item) for item in value]


def _project_relative_or_str(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)
