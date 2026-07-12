from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from task1.survey.localization import Detection2D, SurveyObservation
from task1.survey.detection_config import YoloEvaluationPolicy


@dataclass(frozen=True)
class GroundTruthBox:
    object_id: str
    class_name: str
    bbox_xyxy: tuple[float, float, float, float]
    visible_pixel_count: int
    # Prefer a bbox-local mask; full-frame masks are accepted as well.
    pixel_mask: Any | None = None


@dataclass(frozen=True)
class YoloEvaluationFrame:
    view_id: str
    ground_truth: tuple[GroundTruthBox, ...]
    detections: tuple[Detection2D, ...]


def evaluate_yolo_detections(
    frames: tuple[YoloEvaluationFrame, ...],
    policy: YoloEvaluationPolicy,
    *,
    min_supporting_views: int = 2,
) -> dict[str, Any]:
    if min_supporting_views <= 0:
        raise ValueError("min_supporting_views must be positive")

    view_reports: list[dict[str, Any]] = []
    associations: list[dict[str, Any]] = []
    object_support: dict[str, dict[str, Any]] = {}
    totals = {"ground_truth": 0, "detections": 0, "true_positive": 0, "false_positive": 0, "false_negative": 0}
    class_totals: dict[str, dict[str, int]] = {}

    for frame in frames:
        ground_truth = tuple(
            item for item in frame.ground_truth if item.visible_pixel_count >= policy.min_visible_pixels
        )
        for truth in ground_truth:
            support = object_support.setdefault(
                truth.object_id,
                {
                    "object_id": truth.object_id,
                    "class_name": truth.class_name,
                    "visible_view_ids": set(),
                    "detected_view_ids": set(),
                },
            )
            if support["class_name"] != truth.class_name:
                raise ValueError(f"ground-truth object {truth.object_id!r} changed class across views")
            support["visible_view_ids"].add(frame.view_id)

        frame_associations = _associate_detections(frame, ground_truth)
        associations.extend(frame_associations)
        for association in frame_associations:
            if association["class_match"]:
                object_support[association["object_id"]]["detected_view_ids"].add(frame.view_id)
        unmatched_ground_truth = set(range(len(ground_truth)))
        matches: list[dict[str, Any]] = []
        false_positive_ids: list[str] = []

        for detection in sorted(frame.detections, key=lambda item: (-item.confidence, item.detection_id)):
            candidates = [
                (index, _bbox_iou(detection.bbox_xyxy, ground_truth[index].bbox_xyxy))
                for index in unmatched_ground_truth
                if ground_truth[index].class_name == detection.class_name
            ]
            best = max(candidates, key=lambda item: (item[1], -item[0]), default=None)
            if best is None or best[1] < policy.match_iou_threshold:
                false_positive_ids.append(detection.detection_id)
                _class_counts(class_totals, detection.class_name)["false_positive"] += 1
                continue
            index, iou = best
            unmatched_ground_truth.remove(index)
            truth = ground_truth[index]
            matches.append(
                {
                    "detection_id": detection.detection_id,
                    "object_id": truth.object_id,
                    "class_name": truth.class_name,
                    "confidence": detection.confidence,
                    "iou": iou,
                }
            )
            _class_counts(class_totals, truth.class_name)["true_positive"] += 1

        false_negatives = [ground_truth[index] for index in sorted(unmatched_ground_truth)]
        for truth in false_negatives:
            _class_counts(class_totals, truth.class_name)["false_negative"] += 1
        for truth in ground_truth:
            _class_counts(class_totals, truth.class_name)["ground_truth"] += 1

        totals["ground_truth"] += len(ground_truth)
        totals["detections"] += len(frame.detections)
        totals["true_positive"] += len(matches)
        totals["false_positive"] += len(false_positive_ids)
        totals["false_negative"] += len(false_negatives)
        view_reports.append(
            {
                "view_id": frame.view_id,
                "ground_truth_count": len(ground_truth),
                "detection_count": len(frame.detections),
                "true_positive": len(matches),
                "false_positive_detection_ids": false_positive_ids,
                "false_negative_object_ids": [item.object_id for item in false_negatives],
                "matches": matches,
                "ground_truth": [
                    {
                        "object_id": item.object_id,
                        "class_name": item.class_name,
                        "bbox_xyxy": list(item.bbox_xyxy),
                        "visible_pixel_count": item.visible_pixel_count,
                    }
                    for item in ground_truth
                ],
            }
        )

    precision, recall, f1 = _metrics(
        totals["true_positive"], totals["false_positive"], totals["false_negative"]
    )
    per_class = {}
    for class_name, counts in sorted(class_totals.items()):
        class_precision, class_recall, class_f1 = _metrics(
            counts["true_positive"], counts["false_positive"], counts["false_negative"]
        )
        per_class[class_name] = {**counts, "precision": class_precision, "recall": class_recall, "f1": class_f1}

    object_support_report = []
    for support in sorted(object_support.values(), key=lambda item: item["object_id"]):
        visible_view_ids = sorted(support["visible_view_ids"])
        detected_view_ids = sorted(support["detected_view_ids"])
        object_support_report.append(
            {
                "object_id": support["object_id"],
                "class_name": support["class_name"],
                "visible_view_ids": visible_view_ids,
                "detected_view_ids": detected_view_ids,
                "visible_view_count": len(visible_view_ids),
                "detected_view_count": len(detected_view_ids),
                "support_sufficient": len(detected_view_ids) >= min_supporting_views,
            }
        )

    return {
        "match_iou_threshold": policy.match_iou_threshold,
        "min_visible_pixels": policy.min_visible_pixels,
        "min_supporting_views": min_supporting_views,
        **totals,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_class": per_class,
        "associations": associations,
        "object_support": object_support_report,
        "views": view_reports,
    }


def _associate_detections(
    frame: YoloEvaluationFrame,
    ground_truth: tuple[GroundTruthBox, ...],
) -> list[dict[str, Any]]:
    masked_truth = tuple(item for item in ground_truth if item.pixel_mask is not None)
    associations: list[dict[str, Any]] = []
    for detection in sorted(frame.detections, key=lambda item: (-item.confidence, item.detection_id)):
        candidates = []
        for truth in masked_truth:
            overlap = _mask_overlap_pixels(detection, truth)
            if overlap <= 0:
                continue
            visible_fraction = overlap / truth.visible_pixel_count
            candidates.append(
                (
                    overlap,
                    visible_fraction,
                    _bbox_iou(detection.bbox_xyxy, truth.bbox_xyxy),
                    truth,
                )
            )
        if not candidates:
            continue
        overlap, visible_fraction, bbox_iou, truth = min(
            candidates,
            key=lambda item: (-item[0], -item[1], -item[2], item[3].object_id),
        )
        associations.append(
            {
                "view_id": frame.view_id,
                "detection_id": detection.detection_id,
                "object_id": truth.object_id,
                "class_name": truth.class_name,
                "detection_class_name": detection.class_name,
                "class_match": detection.class_name == truth.class_name,
                "mask_overlap_pixels": overlap,
                "visible_fraction": visible_fraction,
                "bbox_iou": bbox_iou,
            }
        )
    return associations


def _mask_overlap_pixels(detection: Detection2D, truth: GroundTruthBox) -> int:
    truth_mask = np.asarray(truth.pixel_mask, dtype=bool)
    if truth_mask.ndim != 2:
        raise ValueError(f"pixel mask for {truth.object_id!r} must be two-dimensional")

    x0 = int(np.floor(truth.bbox_xyxy[0]))
    y0 = int(np.floor(truth.bbox_xyxy[1]))
    x1 = int(np.ceil(truth.bbox_xyxy[2]))
    y1 = int(np.ceil(truth.bbox_xyxy[3]))
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        raise ValueError(f"pixel mask for {truth.object_id!r} has an invalid bbox")
    crop_shape = (y1 - y0, x1 - x0)
    if truth_mask.shape == crop_shape:
        truth_crop = truth_mask
    elif truth_mask.shape[0] >= y1 and truth_mask.shape[1] >= x1:
        truth_crop = truth_mask[y0:y1, x0:x1]
    else:
        raise ValueError(
            f"pixel mask for {truth.object_id!r} must be bbox-local {crop_shape} "
            f"or cover bbox {truth.bbox_xyxy}, got {truth_mask.shape}"
        )

    if detection.pixel_mask is not None:
        detection_mask = np.asarray(detection.pixel_mask, dtype=bool)
        if detection_mask.ndim != 2 or detection_mask.shape[0] < y1 or detection_mask.shape[1] < x1:
            raise ValueError(
                f"pixel mask for detection {detection.detection_id!r} must cover "
                f"ground-truth bbox {truth.bbox_xyxy}"
            )
        detection_crop = detection_mask[y0:y1, x0:x1]
    else:
        pixel_x = np.arange(x0, x1, dtype=float) + 0.5
        pixel_y = np.arange(y0, y1, dtype=float) + 0.5
        detection_crop = (
            (pixel_x[None, :] >= detection.bbox_xyxy[0])
            & (pixel_x[None, :] < detection.bbox_xyxy[2])
            & (pixel_y[:, None] >= detection.bbox_xyxy[1])
            & (pixel_y[:, None] < detection.bbox_xyxy[3])
        )
    return int(np.count_nonzero(truth_crop & detection_crop))


def _bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def diagnose_survey_detection_pipeline(
    *,
    yolo_evaluation: dict[str, Any],
    observations: list[SurveyObservation],
    candidate_evaluation: dict[str, Any],
    oracle_fusion_evaluation: dict[str, Any],
    min_supporting_views: int,
) -> dict[str, Any]:
    object_support = yolo_evaluation.get("object_support", [])
    associations = yolo_evaluation.get("associations", [])
    if not isinstance(object_support, list) or not isinstance(associations, list):
        raise ValueError("YOLO evaluation does not contain survey support diagnostics")

    expected_ids = {
        str(item["object_id"])
        for item in object_support
        if isinstance(item, dict) and "object_id" in item
    }
    association_by_detection = {
        str(item["detection_id"]): str(item["object_id"])
        for item in associations
        if (
            isinstance(item, dict)
            and bool(item.get("class_match"))
            and "detection_id" in item
            and "object_id" in item
        )
    }
    localized_views: dict[str, set[str]] = {object_id: set() for object_id in expected_ids}
    for observation in observations:
        object_id = association_by_detection.get(observation.detection_id)
        if object_id is not None:
            localized_views.setdefault(object_id, set()).add(observation.view_id)

    insufficient_yolo = sorted(
        str(item["object_id"])
        for item in object_support
        if isinstance(item, dict) and not bool(item.get("support_sufficient"))
    )
    insufficient_localization = sorted(
        object_id
        for object_id in expected_ids
        if len(localized_views.get(object_id, set())) < min_supporting_views
    )
    unobserved_objects = sorted(
        str(object_id)
        for object_id in candidate_evaluation.get("missing_object_ids", [])
        if str(object_id) not in expected_ids
    )
    actual_success = bool(candidate_evaluation.get("success"))
    oracle_fusion_success = bool(oracle_fusion_evaluation.get("success"))

    if actual_success:
        status, primary_stage, issue_scope = "healthy", None, None
        message = "Production detections produced all required candidates within tolerance."
    elif unobserved_objects:
        status, primary_stage = "issue_found", "camera_coverage"
        issue_scope = "route_scene_or_visibility_parameters"
        message = "One or more required objects are not evaluably visible from any survey view."
    elif insufficient_yolo:
        status, primary_stage = "issue_found", "yolo_detection"
        issue_scope = "model_or_inference_parameters"
        message = "YOLO lacks enough correctly classified view support for one or more objects."
    elif insufficient_localization:
        status, primary_stage = "issue_found", "depth_localization"
        issue_scope = "detection_geometry_or_localization_parameters"
        message = "YOLO has view support, but too few associated detections survive localization."
    elif not oracle_fusion_success:
        status, primary_stage = "issue_found", "multi_view_fusion"
        issue_scope = "pipeline_or_fusion_parameters"
        message = "Even ideal object-center observations do not pass the configured fusion policy."
    else:
        status, primary_stage = "issue_found", "multi_view_fusion"
        issue_scope = "detection_geometry_or_fusion_parameters"
        message = "Production observations have support, but only ideal object centers fuse successfully."

    return {
        "status": status,
        "primary_stage": primary_stage,
        "issue_scope": issue_scope,
        "message": message,
        "production_candidate_success": actual_success,
        "ideal_fusion_success": oracle_fusion_success,
        "min_supporting_views": min_supporting_views,
        "insufficient_yolo_support_object_ids": insufficient_yolo,
        "insufficient_localization_support_object_ids": insufficient_localization,
        "unobserved_object_ids": unobserved_objects,
        "localized_support": [
            {
                "object_id": object_id,
                "localized_view_ids": sorted(localized_views.get(object_id, set())),
                "localized_view_count": len(localized_views.get(object_id, set())),
                "support_sufficient": len(localized_views.get(object_id, set())) >= min_supporting_views,
            }
            for object_id in sorted(expected_ids)
        ],
        "oracle": {
            "fusion_evaluation": oracle_fusion_evaluation,
            "evaluation_only": True,
            "used_for_production_candidates": False,
        },
    }


def _class_counts(totals: dict[str, dict[str, int]], class_name: str | None) -> dict[str, int]:
    name = class_name or "__unclassified__"
    return totals.setdefault(
        name,
        {"ground_truth": 0, "true_positive": 0, "false_positive": 0, "false_negative": 0},
    )


def _metrics(true_positive: int, false_positive: int, false_negative: int) -> tuple[float, float, float]:
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1
