from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from task1.detection import YoloInferenceConfig
from task1.vision import CandidateLocalizationPolicy


CONFIG_SCHEMA = "task1_survey_detection_config"
DEFAULT_CONFIG_PATH = Path("configs/task1/survey/detection.yaml")


@dataclass(frozen=True)
class YoloEvaluationPolicy:
    match_iou_threshold: float
    min_visible_pixels: int


@dataclass(frozen=True)
class CandidateEvaluationPolicy:
    position_tolerance_m: float


@dataclass(frozen=True)
class SurveyEvaluationConfig:
    yolo: YoloEvaluationPolicy
    candidate: CandidateEvaluationPolicy


@dataclass(frozen=True)
class SurveyDetectionConfig:
    config_path: Path
    profile_path: Path
    inference: YoloInferenceConfig
    localization: CandidateLocalizationPolicy
    evaluation: SurveyEvaluationConfig


def load_survey_detection_config(path: Path, *, repo_root: Path) -> SurveyDetectionConfig:
    config_path = path if path.is_absolute() else repo_root / path
    if not config_path.exists():
        raise FileNotFoundError(f"survey detection config does not exist: {config_path}")
    payload = _load_mapping(config_path)
    if payload.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"unsupported survey detection schema: {payload.get('schema')}")

    profile_value = _required_str(payload, "profile_path")
    profile_path = Path(profile_value)
    profile_path = profile_path if profile_path.is_absolute() else repo_root / profile_path
    if not profile_path.exists():
        raise FileNotFoundError(f"survey detection profile does not exist: {profile_path}")

    inference = _required_mapping(payload, "inference")
    localization = _required_mapping(payload, "localization")
    fusion = _required_mapping(payload, "fusion")
    evaluation = _required_mapping(payload, "evaluation")
    yolo_evaluation = _required_mapping(evaluation, "yolo")
    candidate_evaluation = _required_mapping(evaluation, "candidate")
    result = SurveyDetectionConfig(
        config_path=config_path.resolve(),
        profile_path=profile_path.resolve(),
        inference=YoloInferenceConfig(
            image_size=_required_int(inference, "image_size", section="inference"),
            confidence_threshold=_required_float(inference, "confidence_threshold", section="inference"),
            iou_threshold=_required_float(inference, "iou_threshold", section="inference"),
            class_agnostic_nms=_required_bool(inference, "class_agnostic_nms", section="inference"),
            device=_optional_str(inference.get("device"), "inference.device"),
            max_detections=_optional_int(inference.get("max_detections"), "inference.max_detections"),
        ),
        localization=CandidateLocalizationPolicy(
            sample_stride_px=_required_int(localization, "sample_stride_px", section="localization"),
            min_valid_depth_samples=_required_int(
                localization, "min_valid_depth_samples", section="localization"
            ),
            min_depth_m=_required_float(localization, "min_depth_m", section="localization"),
            max_depth_m=_optional_float(localization.get("max_depth_m"), "localization.max_depth_m"),
            min_height_above_floor_m=_required_float(
                localization, "min_height_above_floor_m", section="localization"
            ),
            max_height_above_floor_m=_required_float(
                localization, "max_height_above_floor_m", section="localization"
            ),
            visible_surface_radius_percentile=_required_float(
                localization, "visible_surface_radius_percentile", section="localization"
            ),
            footprint_simplification_tolerance_m=_required_float(
                localization, "footprint_simplification_tolerance_m", section="localization"
            ),
            minimum_surface_variance_m2=_required_float(
                localization, "minimum_surface_variance_m2", section="localization"
            ),
            min_supporting_views=_required_int(fusion, "min_supporting_views", section="fusion"),
            assignment_min_footprint_overlap=_required_float(
                fusion, "assignment_min_footprint_overlap", section="fusion"
            ),
            assignment_max_mahalanobis=_required_float(
                fusion, "assignment_max_mahalanobis", section="fusion"
            ),
            assignment_max_cost=_required_float(
                fusion, "assignment_max_cost", section="fusion"
            ),
            geometry_cost_weight=_required_float(
                fusion, "geometry_cost_weight", section="fusion"
            ),
            footprint_cost_weight=_required_float(
                fusion, "footprint_cost_weight", section="fusion"
            ),
            semantic_cost_weight=_required_float(
                fusion, "semantic_cost_weight", section="fusion"
            ),
        ),
        evaluation=SurveyEvaluationConfig(
            yolo=YoloEvaluationPolicy(
                match_iou_threshold=_required_float(
                    yolo_evaluation, "match_iou_threshold", section="evaluation.yolo"
                ),
                min_visible_pixels=_required_int(
                    yolo_evaluation, "min_visible_pixels", section="evaluation.yolo"
                ),
            ),
            candidate=CandidateEvaluationPolicy(
                position_tolerance_m=_required_float(
                    candidate_evaluation, "position_tolerance_m", section="evaluation.candidate"
                ),
            ),
        ),
    )
    _validate(result)
    return result


def _validate(config: SurveyDetectionConfig) -> None:
    inference = config.inference
    yolo_evaluation = config.evaluation.yolo
    candidate_evaluation = config.evaluation.candidate
    if inference.image_size <= 0:
        raise ValueError("survey detection image_size must be positive")
    if not 0.0 <= inference.confidence_threshold <= 1.0:
        raise ValueError("survey detection confidence_threshold must be between 0 and 1")
    if not 0.0 <= inference.iou_threshold <= 1.0:
        raise ValueError("survey detection iou_threshold must be between 0 and 1")
    if inference.max_detections is not None and inference.max_detections <= 0:
        raise ValueError("survey detection max_detections must be positive when configured")
    if not 0.0 < yolo_evaluation.match_iou_threshold <= 1.0:
        raise ValueError("survey detection match_iou_threshold must be in (0, 1]")
    if yolo_evaluation.min_visible_pixels <= 0:
        raise ValueError("survey detection min_visible_pixels must be positive")
    if candidate_evaluation.position_tolerance_m <= 0:
        raise ValueError("survey candidate position_tolerance_m must be positive")


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required for non-JSON survey detection YAML") from exc
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError("survey detection config must contain a mapping")
    return payload


def _required_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"survey detection config field {key} must be a mapping")
    return value


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"survey detection config field {key} must be a non-empty string")
    return value


def _optional_str(value: Any, key: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"survey detection config field {key} must be null or a non-empty string")
    return value


def _optional_int(value: Any, key: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"survey detection config field {key} must be null or an integer")
    if not isinstance(value, int):
        raise ValueError(f"survey detection config field {key} must be null or an integer")
    return value


def _required_int(payload: dict[str, Any], key: str, *, section: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"survey config field {section}.{key} must be an integer")
    return value


def _required_float(payload: dict[str, Any], key: str, *, section: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"survey config field {section}.{key} must be a number")
    return float(value)


def _required_bool(payload: dict[str, Any], key: str, *, section: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"survey config field {section}.{key} must be a boolean")
    return value


def _optional_float(value: Any, key: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"survey config field {key} must be null or a number")
    return float(value)
