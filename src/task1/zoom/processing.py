from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from task1.detection import render_detection_overlay
from task1.vision import Detection2D


ZOOM_CONFIG_SCHEMA = "task1_zoom_config"
ZOOM_REPORT_SCHEMA = "task1_zoom_report"
DEFAULT_ZOOM_CONFIG_PATH = Path("configs/task1/zoom/config.yaml")
_FINAL_REPORT_SCHEMA = "task1_final_report"
_CANDIDATE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_ORIENTATION_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*")
_RESAMPLING_NAMES = {"nearest", "bilinear", "bicubic", "lanczos"}


@dataclass(frozen=True)
class ZoomOutputOrientation:
    name: str
    width: int
    height: int
    rotation_degrees_clockwise: int


@dataclass(frozen=True)
class ZoomConfig:
    config_path: Path
    target_bbox_area_fraction: float
    bbox_padding_fraction: float
    resampling: str
    render_annotations: bool
    output_orientations: tuple[ZoomOutputOrientation, ...]


@dataclass(frozen=True)
class _CropPlan:
    orientation: ZoomOutputOrientation
    crop_box_xyxy: tuple[int, int, int, int]
    source_bbox_xyxy: tuple[float, float, float, float]
    resized_bbox_xyxy: tuple[float, float, float, float]
    output_bbox_xyxy: tuple[float, float, float, float]
    output_bbox_area_fraction: float
    target_area_fraction_error: float
    bbox_padding_px: tuple[float, float, float, float]
    padding_satisfied: bool
    source_bbox_was_clipped: bool
    source_bbox_touches_image_border: bool


def load_zoom_config(path: Path, *, repo_root: Path) -> ZoomConfig:
    config_path = _resolve_path(path, repo_root)
    payload = _load_mapping(config_path)
    if payload.get("schema") != ZOOM_CONFIG_SCHEMA:
        raise ValueError(f"unsupported Task1 Zoom config schema: {payload.get('schema')}")

    raw_orientations = payload.get("output_orientations")
    if not isinstance(raw_orientations, list) or not raw_orientations:
        raise ValueError("Zoom output_orientations must be a non-empty list")
    orientations: list[ZoomOutputOrientation] = []
    names: set[str] = set()
    for index, value in enumerate(raw_orientations):
        if not isinstance(value, dict):
            raise ValueError(f"Zoom output_orientations[{index}] must be an object")
        name = value.get("name")
        if not isinstance(name, str) or _ORIENTATION_NAME.fullmatch(name) is None:
            raise ValueError(f"Zoom output_orientations[{index}] has an invalid name")
        if name in names:
            raise ValueError(f"Zoom output orientation name is duplicated: {name}")
        names.add(name)
        orientations.append(
            ZoomOutputOrientation(
                name=name,
                width=_positive_integer(value.get("width"), f"output_orientations[{index}].width"),
                height=_positive_integer(
                    value.get("height"), f"output_orientations[{index}].height"
                ),
                rotation_degrees_clockwise=_rotation_degrees(
                    value.get("rotation_degrees_clockwise"),
                    f"output_orientations[{index}].rotation_degrees_clockwise",
                ),
            )
        )

    target = _finite_number(payload.get("target_bbox_area_fraction"), "target_bbox_area_fraction")
    if not 0.0 < target <= 1.0:
        raise ValueError("Zoom target_bbox_area_fraction must be in (0, 1]")
    padding = _finite_number(payload.get("bbox_padding_fraction"), "bbox_padding_fraction")
    if not 0.0 <= padding <= 1.0:
        raise ValueError("Zoom bbox_padding_fraction must be between 0 and 1")
    resampling = payload.get("resampling")
    if not isinstance(resampling, str) or resampling.lower() not in _RESAMPLING_NAMES:
        raise ValueError(
            "Zoom resampling must be one of nearest, bilinear, bicubic, or lanczos"
        )
    render_annotations = payload.get("render_annotations")
    if not isinstance(render_annotations, bool):
        raise ValueError("Zoom render_annotations must be a boolean")
    return ZoomConfig(
        config_path=config_path,
        target_bbox_area_fraction=target,
        bbox_padding_fraction=padding,
        resampling=resampling.lower(),
        render_annotations=render_annotations,
        output_orientations=tuple(orientations),
    )


def run_zoom_stage(
    *,
    final_report_path: Path,
    output_dir: Path,
    config: ZoomConfig,
    repo_root: Path,
) -> dict[str, Any]:
    report_path = _resolve_path(final_report_path, repo_root)
    final_report = _load_mapping(report_path)
    if (
        final_report.get("schema") != _FINAL_REPORT_SCHEMA
        or final_report.get("stage") != "final"
    ):
        raise ValueError("Zoom input must be a task1_final_report from the final stage")
    if final_report.get("status") not in {"success", "partial", "failed"}:
        raise ValueError("Zoom input Final status must be success, partial, or failed")
    raw_results = final_report.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("Zoom input Final results must be a list")
    raw_stable_objects = final_report.get("stable_objects")
    if not isinstance(raw_stable_objects, list):
        raise ValueError("Zoom input Final stable_objects must be a list")
    stable_by_id = _stable_objects_by_candidate(raw_stable_objects)

    resolved_output_dir = output_dir.resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    validated_results: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, value in enumerate(raw_results):
        if not isinstance(value, dict):
            raise ValueError(f"Final result[{index}] must be an object")
        candidate_id = value.get("candidate_id")
        if not isinstance(candidate_id, str) or _CANDIDATE_ID.fullmatch(candidate_id) is None:
            raise ValueError(f"Final result[{index}] has an invalid candidate_id")
        if candidate_id in seen_ids:
            raise ValueError(f"Final result candidate_id is duplicated: {candidate_id}")
        seen_ids.add(candidate_id)
        result_status = value.get("status")
        if result_status not in {"success", "failed"}:
            raise ValueError(
                f"Final result[{index}] status must be success or failed"
            )
        validated_results.append(value)

    successful_ids = {
        str(value["candidate_id"])
        for value in validated_results
        if value["status"] == "success"
    }
    missing_stable_ids = sorted(successful_ids - set(stable_by_id))
    extra_stable_ids = sorted(set(stable_by_id) - successful_ids)
    if missing_stable_ids or extra_stable_ids:
        raise ValueError(
            "Final successful results and stable_objects must correspond one-to-one; "
            f"missing={missing_stable_ids}, extra={extra_stable_ids}"
        )

    results: list[dict[str, Any]] = []
    for value in validated_results:
        candidate_id = str(value["candidate_id"])
        result_status = value["status"]
        if result_status != "success":
            results.append(
                {
                    "candidate_id": candidate_id,
                    "status": "skipped",
                    "failure_stage": "final_candidate_not_successful",
                    "message": "Zoom skipped a candidate without a successful Final result.",
                    "source_final_status": value.get("status"),
                    "source_final_failure_stage": value.get("failure_stage"),
                    "artifact_status": "not_attempted",
                    "artifact_failures": [],
                }
            )
            continue
        results.append(
            _zoom_successful_result(
                stable_by_id[candidate_id],
                candidate_id=candidate_id,
                final_report_path=report_path,
                output_dir=resolved_output_dir,
                config=config,
                repo_root=repo_root,
            )
        )

    eligible_count = len(successful_ids)
    successful_count = sum(result["status"] == "success" for result in results)
    failed_count = sum(result["status"] == "failed" for result in results)
    skipped_count = sum(result["status"] == "skipped" for result in results)
    if eligible_count == 0 or successful_count == 0:
        status = "failed"
    elif failed_count:
        status = "partial"
    else:
        status = "success"
    report = {
        "schema": ZOOM_REPORT_SCHEMA,
        "stage": "zoom",
        "status": status,
        "message": _zoom_report_message(
            eligible_count=eligible_count,
            successful_count=successful_count,
            failed_count=failed_count,
            skipped_count=skipped_count,
        ),
        "source_final_report": str(report_path),
        "source_final_status": final_report.get("status"),
        "zoom_config_path": str(config.config_path),
        "target_bbox_area_fraction": config.target_bbox_area_fraction,
        "bbox_padding_fraction": config.bbox_padding_fraction,
        "input_candidate_count": len(raw_results),
        "eligible_candidate_count": eligible_count,
        "successful_candidate_count": successful_count,
        "failed_candidate_count": failed_count,
        "skipped_candidate_count": skipped_count,
        "artifact_generation_failure_count": sum(
            len(result.get("artifact_failures", [])) for result in results
        ),
        "results": results,
    }
    _write_json(resolved_output_dir / "zoom_report.json", report)
    return report


def _zoom_successful_result(
    stable_object: dict[str, Any],
    *,
    candidate_id: str,
    final_report_path: Path,
    output_dir: Path,
    config: ZoomConfig,
    repo_root: Path,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "candidate_id": candidate_id,
        "status": "failed",
        "failure_stage": None,
        "artifact_status": "not_attempted",
        "artifact_failures": [],
    }
    rgb_value = stable_object.get("rgb_path")
    if not isinstance(rgb_value, str) or not rgb_value:
        return _candidate_failure(
            base,
            "final_input_contract",
            "Final stable_object is missing rgb_path.",
        )
    try:
        bbox = _bbox(stable_object.get("bbox_xyxy"))
    except ValueError as exc:
        return _candidate_failure(base, "final_input_contract", str(exc))

    source_path = _resolve_artifact_path(rgb_value, repo_root)
    base.update(
        {
            "source_rgb_path": str(source_path),
            "source_final_report": str(final_report_path),
            "source_detection_id": stable_object.get("detection_id"),
            "class_name": stable_object.get("class_name"),
            "display_name": stable_object.get("display_name"),
            "confidence": stable_object.get("confidence"),
            "input_bbox_xyxy": list(bbox),
        }
    )
    try:
        from PIL import Image

        if not source_path.is_file():
            raise FileNotFoundError(f"Final RGB image does not exist: {source_path}")
        with Image.open(source_path) as source:
            image = source.convert("RGB")
        plan = _choose_crop_plan(
            bbox,
            image_width=image.width,
            image_height=image.height,
            config=config,
        )
        output_path = output_dir / "images" / f"{candidate_id}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        resized = image.crop(plan.crop_box_xyxy).resize(
            (plan.orientation.width, plan.orientation.height),
            resample=_pillow_resampling(Image, config.resampling),
        )
        output_image = _rotate_image_clockwise(
            resized,
            Image,
            plan.orientation.rotation_degrees_clockwise,
        )
        _save_png(output_path, output_image)
    except (OSError, RuntimeError, ValueError) as exc:
        return _candidate_failure(
            base,
            "image_processing",
            f"{type(exc).__name__}: {exc}",
        )

    annotated_path: Path | None = None
    artifact_failures: list[dict[str, str]] = []
    artifact_status = "not_requested"
    if config.render_annotations:
        artifact_status = "success"
        try:
            annotated_path = _render_zoom_annotation(
                output_path,
                output_dir / "annotated" / f"{candidate_id}.png",
                stable_object,
                plan.output_bbox_xyxy,
                candidate_id=candidate_id,
            )
        except Exception as exc:
            artifact_status = "failed"
            artifact_failures.append(
                {
                    "artifact": "annotated_zoom_rgb",
                    "failure_stage": "zoom_annotation",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )

    return {
        **base,
        "status": "success",
        "failure_stage": None,
        "message": "Final RGB was cropped, resized, and rotated around its selected detection.",
        "source_image_size": [image.width, image.height],
        "source_bbox_xyxy": list(plan.source_bbox_xyxy),
        "source_bbox_area_fraction": _area_fraction(
            plan.source_bbox_xyxy, image.width, image.height
        ),
        "source_bbox_was_clipped": plan.source_bbox_was_clipped,
        "source_bbox_touches_image_border": plan.source_bbox_touches_image_border,
        "crop_orientation": plan.orientation.name,
        "crop_box_xyxy": list(plan.crop_box_xyxy),
        "bbox_padding_px": list(plan.bbox_padding_px),
        "padding_satisfied": plan.padding_satisfied,
        "resized_image_size": [plan.orientation.width, plan.orientation.height],
        "resized_bbox_xyxy": list(plan.resized_bbox_xyxy),
        "rotation_degrees_clockwise": plan.orientation.rotation_degrees_clockwise,
        "output_orientation": "landscape" if output_image.width >= output_image.height else "portrait",
        "output_image_size": [output_image.width, output_image.height],
        "output_bbox_xyxy": list(plan.output_bbox_xyxy),
        "output_bbox_area_fraction": plan.output_bbox_area_fraction,
        "target_bbox_area_fraction": config.target_bbox_area_fraction,
        "target_area_fraction_error": plan.target_area_fraction_error,
        "output_rgb_path": str(output_path),
        "annotated_rgb_path": str(annotated_path) if annotated_path is not None else None,
        "artifact_status": artifact_status,
        "artifact_failures": artifact_failures,
    }


def _choose_crop_plan(
    bbox: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
    config: ZoomConfig,
) -> _CropPlan:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Zoom source image dimensions must be positive")
    raw_bbox = bbox
    clipped_bbox = (
        min(float(image_width), max(0.0, bbox[0])),
        min(float(image_height), max(0.0, bbox[1])),
        min(float(image_width), max(0.0, bbox[2])),
        min(float(image_height), max(0.0, bbox[3])),
    )
    if clipped_bbox[2] <= clipped_bbox[0] or clipped_bbox[3] <= clipped_bbox[1]:
        raise ValueError("Final bbox has no visible area in its RGB image")
    plans = [
        plan
        for orientation in config.output_orientations
        if (
            plan := _plan_orientation(
                clipped_bbox,
                raw_bbox=raw_bbox,
                image_width=image_width,
                image_height=image_height,
                orientation=orientation,
                target_area_fraction=config.target_bbox_area_fraction,
                padding_fraction=config.bbox_padding_fraction,
            )
        )
        is not None
    ]
    if not plans:
        raise ValueError("No configured Zoom orientation can contain the Final bbox")
    padded_plans = [plan for plan in plans if plan.padding_satisfied]
    candidates = padded_plans or plans
    bbox_is_landscape = (clipped_bbox[2] - clipped_bbox[0]) >= (
        clipped_bbox[3] - clipped_bbox[1]
    )
    orientation_order = {
        orientation.name: index for index, orientation in enumerate(config.output_orientations)
    }
    return min(
        candidates,
        key=lambda plan: (
            plan.target_area_fraction_error,
            -min(plan.bbox_padding_px),
            0
            if ((plan.orientation.width >= plan.orientation.height) == bbox_is_landscape)
            else 1,
            orientation_order[plan.orientation.name],
        ),
    )


def _plan_orientation(
    bbox: tuple[float, float, float, float],
    *,
    raw_bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    orientation: ZoomOutputOrientation,
    target_area_fraction: float,
    padding_fraction: float,
) -> _CropPlan | None:
    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    aspect_ratio = orientation.width / orientation.height
    maximum_height = min(image_height, math.floor(image_width / aspect_ratio))
    minimum_bbox_height = math.ceil(
        max(bbox_height, bbox_width / aspect_ratio) - 1e-12
    )
    if maximum_height < max(1, minimum_bbox_height):
        return None

    minimum_padding_height = math.ceil(
        max(
            bbox_width * (1.0 + 2.0 * padding_fraction) / aspect_ratio,
            bbox_height * (1.0 + 2.0 * padding_fraction),
        )
        - 1e-12
    )
    lower_height = max(1, minimum_bbox_height)
    if minimum_padding_height <= maximum_height:
        lower_height = max(lower_height, minimum_padding_height)
    target_height = math.sqrt(
        bbox_width * bbox_height / (target_area_fraction * aspect_ratio)
    )
    base_height_candidates = {
        lower_height,
        maximum_height,
        max(lower_height, min(maximum_height, math.floor(target_height))),
        max(lower_height, min(maximum_height, math.ceil(target_height))),
    }
    height_candidates = {
        height + offset
        for height in base_height_candidates
        for offset in (-2, -1, 0, 1, 2)
        if lower_height <= height + offset <= maximum_height
    }
    crop_sizes = [
        (round(aspect_ratio * height), height)
        for height in height_candidates
        if round(aspect_ratio * height) <= image_width
        and round(aspect_ratio * height) + 1e-9 >= bbox_width
        and height + 1e-9 >= bbox_height
    ]
    if not crop_sizes:
        return None
    required_x_padding = bbox_width * padding_fraction
    required_y_padding = bbox_height * padding_fraction
    padded_crop_sizes = [
        size
        for size in crop_sizes
        if _padded_crop_axis_is_feasible(
            x1, x2, size[0], image_width, required_x_padding
        )
        and _padded_crop_axis_is_feasible(
            y1, y2, size[1], image_height, required_y_padding
        )
    ]
    crop_width, crop_height = min(
        padded_crop_sizes or crop_sizes,
        key=lambda size: (
            abs(
                bbox_width * bbox_height / float(size[0] * size[1])
                - target_area_fraction
            ),
            abs(size[0] / size[1] - aspect_ratio),
            size[1],
        ),
    )
    left = _place_crop_axis(
        x1, x2, crop_width, image_width, required_padding=required_x_padding
    )
    top = _place_crop_axis(
        y1, y2, crop_height, image_height, required_padding=required_y_padding
    )
    crop_box = (left, top, left + crop_width, top + crop_height)
    padding = (
        x1 - left,
        y1 - top,
        left + crop_width - x2,
        top + crop_height - y2,
    )
    padding_satisfied = (
        padding[0] + 1e-9 >= required_x_padding
        and padding[2] + 1e-9 >= required_x_padding
        and padding[1] + 1e-9 >= required_y_padding
        and padding[3] + 1e-9 >= required_y_padding
    )
    output_scale_x = orientation.width / crop_width
    output_scale_y = orientation.height / crop_height
    resized_bbox = (
        (x1 - left) * output_scale_x,
        (y1 - top) * output_scale_y,
        (x2 - left) * output_scale_x,
        (y2 - top) * output_scale_y,
    )
    output_bbox = _rotate_bbox_clockwise(
        resized_bbox,
        image_width=orientation.width,
        image_height=orientation.height,
        degrees=orientation.rotation_degrees_clockwise,
    )
    achieved = bbox_width * bbox_height / float(crop_width * crop_height)
    return _CropPlan(
        orientation=orientation,
        crop_box_xyxy=crop_box,
        source_bbox_xyxy=bbox,
        resized_bbox_xyxy=resized_bbox,
        output_bbox_xyxy=output_bbox,
        output_bbox_area_fraction=achieved,
        target_area_fraction_error=abs(achieved - target_area_fraction),
        bbox_padding_px=padding,
        padding_satisfied=padding_satisfied,
        source_bbox_was_clipped=any(
            not math.isclose(raw, clipped, abs_tol=1e-9)
            for raw, clipped in zip(raw_bbox, bbox)
        ),
        source_bbox_touches_image_border=(
            x1 <= 0.0 or y1 <= 0.0 or x2 >= image_width or y2 >= image_height
        ),
    )


def _place_crop_axis(
    start: float,
    end: float,
    crop_size: int,
    source_size: int,
    *,
    required_padding: float,
) -> int:
    padded_lower, padded_upper = _crop_axis_interval(
        start,
        end,
        crop_size,
        source_size,
        required_padding,
    )
    if padded_lower <= padded_upper:
        lower, upper = padded_lower, padded_upper
    else:
        lower, upper = _crop_axis_interval(start, end, crop_size, source_size, 0.0)
    if lower > upper:
        raise ValueError("Zoom crop cannot contain the complete Final bbox")
    preferred = round((start + end - crop_size) / 2.0)
    return max(lower, min(upper, preferred))


def _padded_crop_axis_is_feasible(
    start: float,
    end: float,
    crop_size: int,
    source_size: int,
    required_padding: float,
) -> bool:
    lower, upper = _crop_axis_interval(
        start,
        end,
        crop_size,
        source_size,
        required_padding,
    )
    return lower <= upper


def _crop_axis_interval(
    start: float,
    end: float,
    crop_size: int,
    source_size: int,
    required_padding: float,
) -> tuple[int, int]:
    return (
        max(0, math.ceil(end + required_padding - crop_size - 1e-9)),
        min(
            source_size - crop_size,
            math.floor(start - required_padding + 1e-9),
        ),
    )


def _render_zoom_annotation(
    rgb_path: Path,
    output_path: Path,
    stable_object: dict[str, Any],
    bbox_xyxy: tuple[float, float, float, float],
    *,
    candidate_id: str,
) -> Path:
    detection_id = stable_object.get("detection_id")
    confidence = stable_object.get("confidence")
    class_name = stable_object.get("class_name")
    display_name = stable_object.get("display_name")
    if not isinstance(detection_id, str) or not detection_id:
        raise ValueError(f"{candidate_id}: stable_object.detection_id is invalid")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
    ):
        raise ValueError(f"{candidate_id}: stable_object.confidence is invalid")
    if class_name is not None and not isinstance(class_name, str):
        raise ValueError(f"{candidate_id}: stable_object.class_name is invalid")
    if display_name is not None and not isinstance(display_name, str):
        raise ValueError(f"{candidate_id}: stable_object.display_name is invalid")
    return render_detection_overlay(
        rgb_path,
        (
            Detection2D(
                detection_id=detection_id,
                bbox_xyxy=bbox_xyxy,
                confidence=float(confidence),
                class_name=class_name,
                display_name=display_name,
            ),
        ),
        output_path,
    )


def _rotate_image_clockwise(image: Any, image_module: Any, degrees: int) -> Any:
    operation = {
        0: None,
        90: image_module.Transpose.ROTATE_270,
        180: image_module.Transpose.ROTATE_180,
        270: image_module.Transpose.ROTATE_90,
    }[degrees]
    return image if operation is None else image.transpose(operation)


def _rotate_bbox_clockwise(
    bbox: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
    degrees: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    if degrees == 0:
        return bbox
    if degrees == 90:
        return (image_height - y2, x1, image_height - y1, x2)
    if degrees == 180:
        return (image_width - x2, image_height - y2, image_width - x1, image_height - y1)
    return (y1, image_width - x2, y2, image_width - x1)


def _candidate_failure(
    base: dict[str, Any], failure_stage: str, message: str
) -> dict[str, Any]:
    return {
        **base,
        "status": "failed",
        "failure_stage": failure_stage,
        "message": message,
    }


def _stable_objects_by_candidate(
    raw_stable_objects: list[Any],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(raw_stable_objects):
        if not isinstance(value, dict):
            raise ValueError(f"Final stable_objects[{index}] must be an object")
        candidate_id = value.get("candidate_id")
        if not isinstance(candidate_id, str) or _CANDIDATE_ID.fullmatch(candidate_id) is None:
            raise ValueError(
                f"Final stable_objects[{index}] has an invalid candidate_id"
            )
        if candidate_id in result:
            raise ValueError(
                f"Final stable_object candidate_id is duplicated: {candidate_id}"
            )
        result[candidate_id] = value
    return result


def _zoom_report_message(
    *,
    eligible_count: int,
    successful_count: int,
    failed_count: int,
    skipped_count: int,
) -> str:
    if eligible_count == 0:
        return "Final provided no successful candidates eligible for digital zoom."
    return (
        f"Zoomed {successful_count}/{eligible_count} eligible Final candidates; "
        f"{failed_count} failed and {skipped_count} unsuccessful Final candidates were skipped."
    )


def _bbox(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("stable_object.bbox_xyxy must contain four numbers")
    result = tuple(_finite_number(item, "stable_object.bbox_xyxy") for item in value)
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("stable_object.bbox_xyxy must have positive area")
    return result  # type: ignore[return-value]


def _area_fraction(
    bbox: tuple[float, float, float, float], image_width: int, image_height: int
) -> float:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / float(
        image_width * image_height
    )


def _pillow_resampling(image_module: Any, name: str) -> Any:
    return getattr(image_module.Resampling, name.upper())


def _save_png(path: Path, image: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        image.save(temporary, format="PNG")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Task1 JSON/YAML file does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise ValueError(f"{path} is not JSON and PyYAML is unavailable") from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"Task1 JSON/YAML root must be an object: {path}")
    return value


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return (path if path.is_absolute() else repo_root / path).resolve()


def _resolve_artifact_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else repo_root / path).resolve()


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Zoom {label} must be a positive integer")
    return value


def _rotation_degrees(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in {0, 90, 180, 270}:
        raise ValueError(f"Zoom {label} must be one of 0, 90, 180, or 270")
    return value


def _finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"Zoom {label} must be a finite number")
    return float(value)
