from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robot_arm_pipeline.task1.survey import DEFAULT_OUTPUT_DIR


DEFAULT_ZOOM_TARGET_AREA_RATIO = 0.60
DEFAULT_ZOOM_RATIO_TOLERANCE = 0.02
DEFAULT_ZOOM_CANDIDATE_AREA_RATIOS = (0.42, 0.52, 0.60, 0.68, 0.78, 0.90, 1.0)
DEFAULT_ZOOM_SELECTION_BORDER_MARGIN_PX = 8.0
DEFAULT_ZOOM_PADDING_RATIO = 0.0
ZOOM_POLICY_VERSION = "zoom_area_crop_policy_v1"


@dataclass(frozen=True)
class Task1FinalReport:
    path: Path
    schema_version: str
    status: str
    task1_run_dir: Path
    capture_dir: Path
    image_size: tuple[int, int]
    stable_objects: tuple[dict[str, Any], ...]
    raw_payload: dict[str, Any]


@dataclass(frozen=True)
class ZoomConfig:
    output_dir: Path = DEFAULT_OUTPUT_DIR
    target_area_ratio: float = DEFAULT_ZOOM_TARGET_AREA_RATIO
    ratio_tolerance: float = DEFAULT_ZOOM_RATIO_TOLERANCE
    candidate_area_ratios: tuple[float, ...] = DEFAULT_ZOOM_CANDIDATE_AREA_RATIOS
    padding_ratio: float = DEFAULT_ZOOM_PADDING_RATIO
    selection_border_margin_px: float = DEFAULT_ZOOM_SELECTION_BORDER_MARGIN_PX
    output_width: int | None = None
    output_height: int | None = None
    plan_only: bool = False

    def validate(self) -> None:
        if not 0.0 < self.target_area_ratio <= 1.0:
            raise ValueError("target_area_ratio must be in the range (0, 1]")
        if self.ratio_tolerance < 0.0:
            raise ValueError("ratio_tolerance must be non-negative")
        if self.padding_ratio < 0.0:
            raise ValueError("padding_ratio must be non-negative")
        if self.selection_border_margin_px < 0.0:
            raise ValueError("selection_border_margin_px must be non-negative")
        if not self.candidate_area_ratios:
            raise ValueError("candidate_area_ratios must contain at least one ratio")
        for ratio in self.candidate_area_ratios:
            if not 0.0 < float(ratio) <= 1.0:
                raise ValueError("candidate_area_ratios values must be in the range (0, 1]")
        if self.output_width is not None and self.output_width <= 0:
            raise ValueError("output_width must be positive when provided")
        if self.output_height is not None and self.output_height <= 0:
            raise ValueError("output_height must be positive when provided")


@dataclass(frozen=True)
class ZoomCropPlan:
    crop_box_xyxy: tuple[int, int, int, int]
    source_bbox_xyxy: tuple[float, float, float, float]
    zoomed_bbox_xyxy: tuple[float, float, float, float]
    source_image_size: tuple[int, int]
    output_image_size: tuple[int, int]
    source_bbox_area_ratio: float
    achieved_area_ratio: float
    target_area_ratio: float
    target_ratio_met: bool
    zoomed_bbox_min_border_margin_px: float
    limit_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "crop_box_xyxy": list(self.crop_box_xyxy),
            "source_bbox_xyxy": [_round(value) for value in self.source_bbox_xyxy],
            "zoomed_bbox_xyxy": [_round(value) for value in self.zoomed_bbox_xyxy],
            "source_image_size": list(self.source_image_size),
            "output_image_size": list(self.output_image_size),
            "source_bbox_area_ratio": _round(self.source_bbox_area_ratio),
            "achieved_area_ratio": _round(self.achieved_area_ratio),
            "target_area_ratio": _round(self.target_area_ratio),
            "target_ratio_met": self.target_ratio_met,
            "zoomed_bbox_min_border_margin_px": _round(self.zoomed_bbox_min_border_margin_px),
            "limit_reasons": list(self.limit_reasons),
        }


def find_latest_final_report(output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    root = Path(output_dir)
    candidates = sorted(root.glob("*/final/final_report.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no task1 final reports found under {root}")
    return candidates[-1]


def load_task1_final_report(path: Path | str) -> Task1FinalReport:
    report_path = Path(path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "task1_final_report_v1":
        raise ValueError(f"unsupported task1 final report schema_version: {payload.get('schema_version')}")
    stable_objects = payload.get("stable_objects")
    if not isinstance(stable_objects, list):
        raise ValueError("task1 final report must contain stable_objects list")
    image_size = _image_size(payload.get("image_size"))
    return Task1FinalReport(
        path=report_path,
        schema_version=str(payload["schema_version"]),
        status=str(payload.get("status", "unknown")),
        task1_run_dir=Path(str(payload.get("task1_run_dir") or report_path.parent.parent)),
        capture_dir=Path(str(payload.get("capture_dir") or report_path.parent)),
        image_size=image_size,
        stable_objects=tuple(dict(obj) for obj in stable_objects if isinstance(obj, dict)),
        raw_payload=payload,
    )


def plan_zoom_crop(
    *,
    bbox_xyxy: tuple[float, float, float, float],
    source_image_size: tuple[int, int],
    output_image_size: tuple[int, int],
    target_area_ratio: float = DEFAULT_ZOOM_TARGET_AREA_RATIO,
    ratio_tolerance: float = DEFAULT_ZOOM_RATIO_TOLERANCE,
    padding_ratio: float = DEFAULT_ZOOM_PADDING_RATIO,
) -> ZoomCropPlan:
    image_width, image_height = source_image_size
    output_width, output_height = output_image_size
    if image_width <= 0 or image_height <= 0:
        raise ValueError("source image dimensions must be positive")
    if output_width <= 0 or output_height <= 0:
        raise ValueError("output image dimensions must be positive")
    if not 0.0 < target_area_ratio <= 1.0:
        raise ValueError("target_area_ratio must be in the range (0, 1]")
    if padding_ratio < 0.0:
        raise ValueError("padding_ratio must be non-negative")

    x1, y1, x2, y2 = _clipped_bbox(bbox_xyxy, image_width=image_width, image_height=image_height)
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    bbox_area = bbox_width * bbox_height
    output_aspect = output_width / output_height
    scale = max(math.sqrt(1.0 / max(target_area_ratio, 1e-6)), 1.0 + 2.0 * padding_ratio)
    crop_width = bbox_width * scale
    crop_height = bbox_height * scale
    if crop_width / max(crop_height, 1e-6) > output_aspect:
        crop_height = crop_width / output_aspect
    else:
        crop_width = crop_height * output_aspect
    ideal_crop_width = crop_width
    ideal_crop_height = crop_height

    crop_width = min(float(image_width), max(crop_width, bbox_width))
    crop_height = min(float(image_height), max(crop_height, bbox_height))
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    left = _clamp(center_x - crop_width / 2.0, 0.0, image_width - crop_width)
    top = _clamp(center_y - crop_height / 2.0, 0.0, image_height - crop_height)
    right = left + crop_width
    bottom = top + crop_height

    crop_left = max(0, min(image_width - 1, int(math.floor(left))))
    crop_top = max(0, min(image_height - 1, int(math.floor(top))))
    crop_right = max(crop_left + 1, min(image_width, int(math.ceil(right))))
    crop_bottom = max(crop_top + 1, min(image_height, int(math.ceil(bottom))))
    crop_area = float((crop_right - crop_left) * (crop_bottom - crop_top))
    achieved_area_ratio = bbox_area / crop_area
    source_bbox_area_ratio = bbox_area / float(image_width * image_height)
    target_ratio_met = abs(achieved_area_ratio - target_area_ratio) <= ratio_tolerance

    scale_x = output_width / float(crop_right - crop_left)
    scale_y = output_height / float(crop_bottom - crop_top)
    zoomed_bbox = (
        (x1 - crop_left) * scale_x,
        (y1 - crop_top) * scale_y,
        (x2 - crop_left) * scale_x,
        (y2 - crop_top) * scale_y,
    )
    margin = _bbox_min_border_margin(zoomed_bbox, output_width=output_width, output_height=output_height)

    limit_reasons = _limit_reasons(
        target_ratio_met=target_ratio_met,
        achieved_area_ratio=achieved_area_ratio,
        target_area_ratio=target_area_ratio,
        crop_box=(crop_left, crop_top, crop_right, crop_bottom),
        source_image_size=source_image_size,
        crop_width=crop_width,
        crop_height=crop_height,
        ideal_crop_width=ideal_crop_width,
        ideal_crop_height=ideal_crop_height,
    )
    return ZoomCropPlan(
        crop_box_xyxy=(crop_left, crop_top, crop_right, crop_bottom),
        source_bbox_xyxy=(x1, y1, x2, y2),
        zoomed_bbox_xyxy=zoomed_bbox,
        source_image_size=source_image_size,
        output_image_size=output_image_size,
        source_bbox_area_ratio=source_bbox_area_ratio,
        achieved_area_ratio=achieved_area_ratio,
        target_area_ratio=target_area_ratio,
        target_ratio_met=target_ratio_met,
        zoomed_bbox_min_border_margin_px=margin,
        limit_reasons=tuple(limit_reasons),
    )


def build_zoom_plan(final_report: Task1FinalReport, config: ZoomConfig = ZoomConfig()) -> tuple[dict[str, Any], ...]:
    config.validate()
    planned: list[dict[str, Any]] = []
    for index, obj in enumerate(final_report.stable_objects):
        planned.append(_planned_object_zoom(obj, index=index, final_report=final_report, config=config))
    return tuple(planned)


def run_task1_zoom(final_report_path: Path | str, config: ZoomConfig = ZoomConfig()) -> dict[str, Any]:
    config.validate()
    final_report = load_task1_final_report(final_report_path)
    created_utc = datetime.now(timezone.utc).isoformat()
    zoom_dir = final_report.task1_run_dir / "zoom"
    images_dir = zoom_dir / "images"
    selected_dir = zoom_dir / "selected"
    plan_path = zoom_dir / "zoom_plan.json"
    report_path = zoom_dir / "zoom_report.json"
    planned_zooms = build_zoom_plan(final_report, config)
    plan_payload = _plan_payload(
        created_utc=created_utc,
        final_report=final_report,
        config=config,
        zoom_dir=zoom_dir,
        images_dir=images_dir,
        selected_dir=selected_dir,
        plan_path=plan_path,
        report_path=report_path,
        planned_zooms=planned_zooms,
    )

    if config.plan_only:
        report = _report_payload(
            created_utc=created_utc,
            status="plan_only",
            message=f"Planned {len(planned_zooms)} task1 zoom crops without writing zoomed images.",
            final_report=final_report,
            config=config,
            zoom_dir=zoom_dir,
            images_dir=images_dir,
            selected_dir=selected_dir,
            plan_path=plan_path,
            report_path=report_path,
            planned_zooms=planned_zooms,
            zoomed_objects=[_plan_only_result(item) for item in planned_zooms],
        )
        _write_json(plan_path, plan_payload)
        _write_json(report_path, report)
        return report

    zoomed_objects = [
        _render_zoom(item, images_dir=images_dir, selected_dir=selected_dir, config=config)
        for item in planned_zooms
    ]
    status_counts = _status_counts(zoomed_objects)
    failed_count = status_counts.get("failed", 0)
    limited_count = status_counts.get("zoomed_limited", 0)
    if zoomed_objects and failed_count == 0 and limited_count == 0:
        status = "success"
    elif zoomed_objects and failed_count < len(zoomed_objects):
        status = "partial"
    else:
        status = "failed"
    message = (
        f"Zoomed {len(zoomed_objects) - failed_count}/{len(zoomed_objects)} stable final objects; "
        f"status_counts: {_status_counts_text(status_counts)}."
    )
    report = _report_payload(
        created_utc=created_utc,
        status=status,
        message=message,
        final_report=final_report,
        config=config,
        zoom_dir=zoom_dir,
        images_dir=images_dir,
        selected_dir=selected_dir,
        plan_path=plan_path,
        report_path=report_path,
        planned_zooms=planned_zooms,
        zoomed_objects=zoomed_objects,
    )
    _write_json(plan_path, plan_payload)
    _write_json(report_path, report)
    return report


def _planned_object_zoom(
    obj: dict[str, Any],
    *,
    index: int,
    final_report: Task1FinalReport,
    config: ZoomConfig,
) -> dict[str, Any]:
    object_id = str(obj.get("object_id") or f"stable_object_{index + 1:03d}")
    source_image_path = obj.get("final_image_path")
    output_image_size = _output_image_size(final_report.image_size, config)
    selected_image_path = final_report.task1_run_dir / "zoom" / "selected" / f"zoom_{_safe_name(object_id)}.png"
    base = {
        "object_id": object_id,
        "class_name": obj.get("class_name"),
        "source_final_image_path": str(source_image_path) if source_image_path else None,
        "output_image_path": str(selected_image_path),
        "target_area_ratio": config.target_area_ratio,
        "ratio_tolerance": config.ratio_tolerance,
        "candidate_area_ratios": list(_candidate_area_ratios(config)),
        "source_stable_object": obj,
    }
    if not source_image_path:
        return {
            **base,
            "status": "failed",
            "reason": "missing_final_image_path",
            "message": "Stable object does not contain final_image_path.",
        }
    try:
        bbox = _bbox(obj.get("bbox_xyxy"))
        candidate_zooms = _candidate_zoom_plans(
            object_id=object_id,
            bbox_xyxy=bbox,
            source_image_size=final_report.image_size,
            output_image_size=output_image_size,
            config=config,
            run_dir=final_report.task1_run_dir,
        )
    except (TypeError, ValueError) as exc:
        return {
            **base,
            "status": "failed",
            "reason": "invalid_bbox_or_image_size",
            "message": str(exc),
        }
    selected = _select_zoom_candidate(candidate_zooms, config=config)
    configured_target_ratio_met = bool(
        selected.get("selection", {}).get("configured_target_ratio_met", selected["zoom_plan"]["target_ratio_met"])
    )
    status = "planned" if configured_target_ratio_met else "planned_limited"
    return {
        **base,
        "status": status,
        "zoom_plan": selected["zoom_plan"],
        "selected_zoom": selected,
        "candidate_zooms": candidate_zooms,
    }


def _candidate_area_ratios(config: ZoomConfig) -> tuple[float, ...]:
    ratios: list[float] = []
    for ratio in (config.target_area_ratio, *config.candidate_area_ratios):
        value = float(ratio)
        if not any(abs(value - previous) < 1e-9 for previous in ratios):
            ratios.append(value)
    return tuple(ratios)


def _candidate_zoom_plans(
    *,
    object_id: str,
    bbox_xyxy: tuple[float, float, float, float],
    source_image_size: tuple[int, int],
    output_image_size: tuple[int, int],
    config: ZoomConfig,
    run_dir: Path,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, area_ratio in enumerate(_candidate_area_ratios(config)):
        crop_plan = plan_zoom_crop(
            bbox_xyxy=bbox_xyxy,
            source_image_size=source_image_size,
            output_image_size=output_image_size,
            target_area_ratio=area_ratio,
            ratio_tolerance=config.ratio_tolerance,
            padding_ratio=config.padding_ratio,
        )
        image_path = run_dir / "zoom" / "images" / f"zoom_{_safe_name(object_id)}_{index:02d}.png"
        candidates.append(
            {
                "candidate_id": f"{object_id}_zoom_{index:02d}",
                "candidate_area_ratio": _round(area_ratio),
                "output_image_path": str(image_path),
                "zoom_plan": crop_plan.to_dict(),
            }
        )
    return candidates


def _select_zoom_candidate(candidate_zooms: list[dict[str, Any]], *, config: ZoomConfig) -> dict[str, Any]:
    if not candidate_zooms:
        raise ValueError("candidate_zooms must not be empty")
    selected = max(candidate_zooms, key=lambda item: _zoom_candidate_selection_score(item, config=config))
    selected = dict(selected)
    achieved_area_ratio = float(selected["zoom_plan"].get("achieved_area_ratio", 0.0))
    configured_target_ratio_met = abs(achieved_area_ratio - config.target_area_ratio) <= config.ratio_tolerance
    selected["selection"] = {
        "policy_version": "zoom_candidate_selection_policy_v1",
        "target_area_ratio": _round(config.target_area_ratio),
        "configured_target_ratio_met": configured_target_ratio_met,
        "selection_border_margin_px": _round(config.selection_border_margin_px),
        "score": list(_zoom_candidate_selection_score(selected, config=config)),
        "rules": [
            "Prefer zoom candidates whose projected bbox keeps enough border margin.",
            "Prefer candidates whose projected bbox area is closest to the configured target area ratio.",
            "Use larger projected bbox area and margin as tie-breakers.",
        ],
    }
    return selected


def _zoom_candidate_selection_score(candidate: dict[str, Any], *, config: ZoomConfig) -> tuple[float, ...]:
    plan = candidate["zoom_plan"]
    area_ratio = float(plan.get("achieved_area_ratio", 0.0))
    margin = float(plan.get("zoomed_bbox_min_border_margin_px", 0.0))
    safe = margin >= config.selection_border_margin_px
    return (
        1.0 if safe else 0.0,
        -abs(area_ratio - config.target_area_ratio),
        area_ratio,
        margin,
    )


def _render_zoom(
    planned: dict[str, Any],
    *,
    images_dir: Path,
    selected_dir: Path,
    config: ZoomConfig,
) -> dict[str, Any]:
    if planned.get("status") == "failed":
        return dict(planned)
    source_path = Path(str(planned["source_final_image_path"]))
    if not source_path.exists():
        return {
            **planned,
            "status": "failed",
            "reason": "source_image_missing",
            "message": f"source final image does not exist: {source_path}",
        }

    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on optional environment package
        return {
            **planned,
            "status": "failed",
            "reason": "pillow_unavailable",
            "message": "Pillow is required for task1 zoom image generation.",
        }

    try:
        with Image.open(source_path).convert("RGB") as image:
            actual_size = (int(image.width), int(image.height))
            candidate_zooms = list(planned.get("candidate_zooms", []))
            if candidate_zooms and actual_size != tuple(candidate_zooms[0]["zoom_plan"]["source_image_size"]):
                output_image_size = _output_image_size(actual_size, config)
                candidate_zooms = _candidate_zoom_plans(
                    object_id=str(planned["object_id"]),
                    bbox_xyxy=_bbox(planned["source_stable_object"].get("bbox_xyxy")),
                    source_image_size=actual_size,
                    output_image_size=output_image_size,
                    config=config,
                    run_dir=Path(str(planned["output_image_path"])).parents[2],
                )
            selected_zoom = _select_zoom_candidate(candidate_zooms, config=config)
            resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
            written_candidates: list[dict[str, Any]] = []
            for candidate in candidate_zooms:
                source_plan = candidate["zoom_plan"]
                crop_box = tuple(int(value) for value in source_plan["crop_box_xyxy"])
                output_size = tuple(int(value) for value in source_plan["output_image_size"])
                zoomed = image.crop(crop_box).resize(output_size, resampling)
                candidate_path = images_dir / Path(str(candidate["output_image_path"])).name
                candidate_path.parent.mkdir(parents=True, exist_ok=True)
                zoomed.save(candidate_path)
                written_candidates.append({**candidate, "output_image_path": str(candidate_path)})
                if candidate["candidate_id"] == selected_zoom["candidate_id"]:
                    selected_zoom = {**selected_zoom, "output_image_path": str(candidate_path)}
            selected_source_path = Path(str(selected_zoom["output_image_path"]))
            output_path = selected_dir / Path(str(planned["output_image_path"])).name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(selected_source_path, output_path)
            selected_zoom = {**selected_zoom, "selected_image_path": str(output_path)}
    except Exception as exc:  # noqa: BLE001 - report the concrete image failure.
        return {
            **planned,
            "status": "failed",
            "reason": "image_processing_failed",
            "message": str(exc),
        }

    source_plan = selected_zoom["zoom_plan"]
    configured_target_ratio_met = bool(
        selected_zoom.get("selection", {}).get("configured_target_ratio_met", source_plan["target_ratio_met"])
    )
    status = "zoomed" if configured_target_ratio_met else "zoomed_limited"
    return {
        **planned,
        "status": status,
        "reason": None if configured_target_ratio_met else "target_ratio_not_reached",
        "zoom_plan": source_plan,
        "candidate_zooms": written_candidates,
        "selected_zoom": selected_zoom,
        "output_image_path": str(output_path),
        "message": (
            "Wrote zoomed image at the requested object area ratio."
            if configured_target_ratio_met
            else "Wrote zoomed image, but source bounds/aspect constraints prevented the target area ratio."
        ),
    }


def _plan_only_result(planned: dict[str, Any]) -> dict[str, Any]:
    if planned.get("status") == "failed":
        return dict(planned)
    return dict(planned)


def _plan_payload(
    *,
    created_utc: str,
    final_report: Task1FinalReport,
    config: ZoomConfig,
    zoom_dir: Path,
    images_dir: Path,
    selected_dir: Path,
    plan_path: Path,
    report_path: Path,
    planned_zooms: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    return {
        "schema_version": "task1_zoom_plan_v1",
        "stage": "zoom",
        "created_utc": created_utc,
        "source_final_report_path": str(final_report.path),
        "source_final_status": final_report.status,
        "task1_run_dir": str(final_report.task1_run_dir),
        "zoom_dir": str(zoom_dir),
        "images_dir": str(images_dir),
        "selected_dir": str(selected_dir),
        "plan_path": str(plan_path),
        "report_path": str(report_path),
        "policy_version": ZOOM_POLICY_VERSION,
        "zoom_config": _config_payload(config),
        "planned_zooms": list(planned_zooms),
        "notes": [
            "Zoom consumes final stable_objects only and uses final_image_path plus bbox_xyxy as its formal inputs.",
            "The crop size is selected so bbox area divided by output image area approaches target_area_ratio.",
        ],
    }


def _report_payload(
    *,
    created_utc: str,
    status: str,
    message: str,
    final_report: Task1FinalReport,
    config: ZoomConfig,
    zoom_dir: Path,
    images_dir: Path,
    selected_dir: Path,
    plan_path: Path,
    report_path: Path,
    planned_zooms: tuple[dict[str, Any], ...],
    zoomed_objects: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "task1_zoom_report_v1",
        "stage": "zoom",
        "status": status,
        "created_utc": created_utc,
        "message": message,
        "source_final_report_path": str(final_report.path),
        "source_final_status": final_report.status,
        "task1_run_dir": str(final_report.task1_run_dir),
        "zoom_dir": str(zoom_dir),
        "images_dir": str(images_dir),
        "selected_dir": str(selected_dir),
        "plan_path": str(plan_path),
        "report_path": str(report_path),
        "policy_version": ZOOM_POLICY_VERSION,
        "zoom_config": _config_payload(config),
        "planned_zooms": list(planned_zooms),
        "zoomed_objects": zoomed_objects,
        "status_counts": _status_counts(zoomed_objects),
        "notes": [
            "Zoom is a post-final digital crop-and-resize stage; it does not change object identity or pose.",
            "Objects without final_image_path or bbox_xyxy are reported as failures instead of receiving synthetic crops.",
            "When the source frame cannot provide the exact target area ratio, the image is still written with limit_reasons and achieved_area_ratio.",
        ],
    }


def _output_image_size(source_image_size: tuple[int, int], config: ZoomConfig) -> tuple[int, int]:
    width, height = source_image_size
    return (config.output_width or width, config.output_height or height)


def _bbox(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("bbox_xyxy must be a list of four numbers")
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def _clipped_bbox(
    bbox_xyxy: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    left = _clamp(min(x1, x2), 0.0, float(image_width))
    top = _clamp(min(y1, y2), 0.0, float(image_height))
    right = _clamp(max(x1, x2), 0.0, float(image_width))
    bottom = _clamp(max(y1, y2), 0.0, float(image_height))
    if right <= left or bottom <= top:
        raise ValueError("bbox_xyxy does not overlap the image")
    return (left, top, right, bottom)


def _limit_reasons(
    *,
    target_ratio_met: bool,
    achieved_area_ratio: float,
    target_area_ratio: float,
    crop_box: tuple[int, int, int, int],
    source_image_size: tuple[int, int],
    crop_width: float,
    crop_height: float,
    ideal_crop_width: float,
    ideal_crop_height: float,
) -> list[str]:
    if target_ratio_met:
        return []
    reasons: list[str] = []
    image_width, image_height = source_image_size
    crop_left, crop_top, crop_right, crop_bottom = crop_box
    if crop_left == 0 or crop_right == image_width or crop_top == 0 or crop_bottom == image_height:
        reasons.append("source_image_boundary")
    if crop_width + 1e-9 < ideal_crop_width or crop_height + 1e-9 < ideal_crop_height:
        reasons.append("source_frame_too_tight_for_target_ratio")
    if crop_width > ideal_crop_width + 1e-9 or crop_height > ideal_crop_height + 1e-9:
        reasons.append("bbox_aspect_requires_larger_crop")
    if achieved_area_ratio > target_area_ratio:
        reasons.append("achieved_ratio_above_target")
    else:
        reasons.append("achieved_ratio_below_target")
    return reasons


def _bbox_min_border_margin(
    bbox_xyxy: tuple[float, float, float, float],
    *,
    output_width: int,
    output_height: int,
) -> float:
    x1, y1, x2, y2 = bbox_xyxy
    return min(x1, y1, float(output_width) - x2, float(output_height) - y2)


def _image_size(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("task1 final report must contain image_size [width, height]")
    width, height = int(value[0]), int(value[1])
    if width <= 0 or height <= 0:
        raise ValueError("task1 final report image_size values must be positive")
    return (width, height)


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "object"


def _status_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        status = str(item.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _status_counts_text(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))


def _config_payload(config: ZoomConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _clamp(value: float, lower: float, upper: float) -> float:
    if upper < lower:
        return lower
    return max(lower, min(upper, value))


def _round(value: float) -> float:
    return round(float(value), 6)
