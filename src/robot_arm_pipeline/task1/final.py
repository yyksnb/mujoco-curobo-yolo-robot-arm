from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robot_arm_pipeline.types import ObjectDetection
from robot_arm_pipeline.task1.survey import (
    DEFAULT_CAMERA_NAME,
    DEFAULT_DEPTH_COMPONENT_MIN_PIXELS,
    DEFAULT_DEPTH_COMPONENT_SPLIT_DISTANCE_M,
    DEFAULT_DEPTH_SAMPLE_STRIDE_PX,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_OPENING_CLEARANCE_M,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SCENE_MODEL,
    DEFAULT_TANK_OPENING_Z_M,
    DEFAULT_YOLO_PROFILE,
    MujocoSurveyBackend,
    SurveyConfig,
    SurveyDetector,
    SurveyObservation,
    SurveyView,
    SurveyWorkspace,
    load_stage0_layout,
    survey_scene_from_stage0_layout,
)


DEFAULT_FINAL_CAMERA_Z_M = 0.30
DEFAULT_FINAL_STANDOFF_M = 0.12
DEFAULT_FINAL_MIN_OBLIQUE_DISTANCE_M = 0.06
DEFAULT_FINAL_LOOK_AT_HEIGHT_OFFSET_M = 0.025
DEFAULT_FINAL_TARGET_MATCH_RADIUS_M = 0.07
DEFAULT_FINAL_ENTRY_VALIDATION_SAMPLES = 3
DEFAULT_FINAL_ENTRY_CLEARANCE_MARGIN_M = 0.02
DEFAULT_FINAL_IK_POSITION_TOLERANCE_M = 0.005
DEFAULT_FINAL_VIEW_ANGLE_OFFSETS_DEG = (0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0, 180.0)
DEFAULT_FINAL_VIEW_STANDOFF_MULTIPLIERS = (1.0, 1.5)
DEFAULT_FINAL_CENTERLINE_VIEW_ANGLE_OFFSETS_DEG = (0.0, 45.0, -45.0)
DEFAULT_FINAL_ENTRY_SIDE = "y-max"
DEFAULT_FINAL_YOLO_CONFIDENCE = 0.20
DEFAULT_FINAL_YOLO_MAX_DETECTIONS = 12
DEFAULT_FINAL_YOLO_TILE_GRID_SIZE = 1
FINAL_STABLE_SELECTION_POLICY_VERSION = "final_stable_selection_policy_v1"
DEFAULT_FINAL_BBOX_FRAGMENT_MIN_OVERLAP_RATIO = 0.15
DEFAULT_FINAL_SELECTION_BORDER_MARGIN_PX = 8.0


@dataclass(frozen=True)
class Task1RowReport:
    path: Path
    schema_version: str
    status: str
    layout_snapshot_path: Path
    task1_run_dir: Path
    row_dir: Path
    camera_name: str
    workspace: SurveyWorkspace
    image_size: tuple[int, int]
    views: tuple[dict[str, Any], ...]
    views_by_id: dict[str, dict[str, Any]]
    stable_objects: tuple[dict[str, Any], ...]
    tentative_objects: tuple[dict[str, Any], ...]
    ambiguous_objects: tuple[dict[str, Any], ...]
    object_selection_summary: dict[str, Any]
    raw_payload: dict[str, Any]


@dataclass(frozen=True)
class FinalConfig:
    scene_model_path: Path = DEFAULT_SCENE_MODEL
    yolo_profile_path: Path = DEFAULT_YOLO_PROFILE
    output_dir: Path = DEFAULT_OUTPUT_DIR
    camera_name: str = DEFAULT_CAMERA_NAME
    image_width: int = DEFAULT_IMAGE_WIDTH
    image_height: int = DEFAULT_IMAGE_HEIGHT
    camera_z_m: float = DEFAULT_FINAL_CAMERA_Z_M
    tank_opening_z_m: float = DEFAULT_TANK_OPENING_Z_M
    opening_clearance_m: float = DEFAULT_OPENING_CLEARANCE_M
    standoff_m: float = DEFAULT_FINAL_STANDOFF_M
    min_oblique_distance_m: float = DEFAULT_FINAL_MIN_OBLIQUE_DISTANCE_M
    look_at_height_offset_m: float = DEFAULT_FINAL_LOOK_AT_HEIGHT_OFFSET_M
    entry_side: str = DEFAULT_FINAL_ENTRY_SIDE
    target_match_radius_m: float = DEFAULT_FINAL_TARGET_MATCH_RADIUS_M
    entry_validation_samples: int = DEFAULT_FINAL_ENTRY_VALIDATION_SAMPLES
    entry_clearance_margin_m: float = DEFAULT_FINAL_ENTRY_CLEARANCE_MARGIN_M
    final_view_angle_offsets_deg: tuple[float, ...] = DEFAULT_FINAL_VIEW_ANGLE_OFFSETS_DEG
    final_view_standoff_multipliers: tuple[float, ...] = DEFAULT_FINAL_VIEW_STANDOFF_MULTIPLIERS
    centerline_view_angle_offsets_deg: tuple[float, ...] = DEFAULT_FINAL_CENTERLINE_VIEW_ANGLE_OFFSETS_DEG
    yolo_confidence: float = DEFAULT_FINAL_YOLO_CONFIDENCE
    yolo_iou: float | None = None
    yolo_image_size: int | None = None
    yolo_device: str | None = None
    yolo_max_detections: int | None = DEFAULT_FINAL_YOLO_MAX_DETECTIONS
    yolo_tile_grid_size: int = DEFAULT_FINAL_YOLO_TILE_GRID_SIZE
    yolo_tile_overlap: float = 0.0
    yolo_tile_nms_iou: float = 0.45
    depth_sample_stride_px: int = DEFAULT_DEPTH_SAMPLE_STRIDE_PX
    depth_component_min_pixels: int = DEFAULT_DEPTH_COMPONENT_MIN_PIXELS
    depth_component_split_distance_m: float = DEFAULT_DEPTH_COMPONENT_SPLIT_DISTANCE_M
    bbox_fragment_min_overlap_ratio: float = DEFAULT_FINAL_BBOX_FRAGMENT_MIN_OVERLAP_RATIO
    selection_border_margin_px: float = DEFAULT_FINAL_SELECTION_BORDER_MARGIN_PX
    run_yolo: bool = True
    plan_only: bool = False
    strict_yolo: bool = False
    max_ik_iterations: int = 240
    ik_position_tolerance_m: float = DEFAULT_FINAL_IK_POSITION_TOLERANCE_M
    ik_orientation_tolerance_rad: float = 0.30
    ik_damping: float = 1e-3

    def validate(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        if self.camera_z_m >= self.tank_opening_z_m:
            raise ValueError(
                "final wrist camera z must be below the tank upper opening height: "
                f"camera_z={self.camera_z_m:.4f}, tank_opening_z={self.tank_opening_z_m:.4f}"
            )
        max_camera_z = self.tank_opening_z_m - self.opening_clearance_m
        if self.camera_z_m > max_camera_z:
            raise ValueError(
                "final wrist camera must keep clearance below the tank opening: "
                f"camera_z={self.camera_z_m:.4f}, max_camera_z={max_camera_z:.4f}"
            )
        if self.standoff_m <= 0.0:
            raise ValueError("standoff_m must be positive")
        if self.min_oblique_distance_m < 0.0:
            raise ValueError("min_oblique_distance_m must be non-negative")
        if self.standoff_m < self.min_oblique_distance_m:
            raise ValueError("standoff_m must be at least min_oblique_distance_m")
        if self.look_at_height_offset_m < 0.0:
            raise ValueError("look_at_height_offset_m must be non-negative")
        if self.entry_side not in {"y-max", "y-min", "x-min", "x-max", "center"}:
            raise ValueError("entry_side must be one of y-max, y-min, x-min, x-max, or center")
        if self.target_match_radius_m <= 0.0:
            raise ValueError("target_match_radius_m must be positive")
        if self.entry_validation_samples <= 0:
            raise ValueError("entry_validation_samples must be positive")
        if self.entry_clearance_margin_m < 0.0:
            raise ValueError("entry_clearance_margin_m must be non-negative")
        max_entry_z = self.tank_opening_z_m - self.opening_clearance_m - self.entry_clearance_margin_m
        if max_entry_z <= self.camera_z_m:
            raise ValueError(
                "entry_clearance_margin_m leaves no vertical room for final entry validation samples: "
                f"camera_z={self.camera_z_m:.4f}, max_entry_z={max_entry_z:.4f}"
            )
        if not self.final_view_angle_offsets_deg:
            raise ValueError("final_view_angle_offsets_deg must contain at least one angle")
        if not self.final_view_standoff_multipliers:
            raise ValueError("final_view_standoff_multipliers must contain at least one multiplier")
        _validate_angle_offsets(self.final_view_angle_offsets_deg, field_name="final_view_angle_offsets_deg")
        _validate_angle_offsets(self.centerline_view_angle_offsets_deg, field_name="centerline_view_angle_offsets_deg")
        _validate_positive_multipliers(
            self.final_view_standoff_multipliers,
            field_name="final_view_standoff_multipliers",
        )
        if not 0.0 <= self.yolo_confidence <= 1.0:
            raise ValueError("yolo_confidence must be between 0 and 1")
        if self.yolo_tile_grid_size <= 0:
            raise ValueError("yolo_tile_grid_size must be positive")
        if self.depth_sample_stride_px <= 0:
            raise ValueError("depth_sample_stride_px must be positive")
        if self.depth_component_min_pixels <= 0:
            raise ValueError("depth_component_min_pixels must be positive")
        if self.depth_component_split_distance_m <= 0.0:
            raise ValueError("depth_component_split_distance_m must be positive")
        if not 0.0 <= self.bbox_fragment_min_overlap_ratio <= 1.0:
            raise ValueError("bbox_fragment_min_overlap_ratio must be between 0 and 1")
        if self.selection_border_margin_px < 0.0:
            raise ValueError("selection_border_margin_px must be non-negative")

    def capture_config(self) -> SurveyConfig:
        return SurveyConfig(
            scene_model_path=self.scene_model_path,
            yolo_profile_path=self.yolo_profile_path,
            output_dir=self.output_dir,
            camera_name=self.camera_name,
            grid_size=1,
            image_width=self.image_width,
            image_height=self.image_height,
            camera_z_m=self.camera_z_m,
            tank_opening_z_m=self.tank_opening_z_m,
            opening_clearance_m=self.opening_clearance_m,
            yolo_confidence=self.yolo_confidence,
            yolo_iou=self.yolo_iou,
            yolo_image_size=self.yolo_image_size,
            yolo_device=self.yolo_device,
            yolo_max_detections=self.yolo_max_detections,
            yolo_tile_grid_size=self.yolo_tile_grid_size,
            yolo_tile_overlap=self.yolo_tile_overlap,
            yolo_tile_nms_iou=self.yolo_tile_nms_iou,
            depth_sample_stride_px=self.depth_sample_stride_px,
            depth_component_min_pixels=self.depth_component_min_pixels,
            depth_component_split_distance_m=self.depth_component_split_distance_m,
            run_yolo=self.run_yolo,
            plan_only=self.plan_only,
            strict_yolo=self.strict_yolo,
            max_ik_iterations=self.max_ik_iterations,
            ik_position_tolerance_m=self.ik_position_tolerance_m,
            ik_orientation_tolerance_rad=self.ik_orientation_tolerance_rad,
            ik_damping=self.ik_damping,
        )


@dataclass(frozen=True)
class FinalTarget:
    object_id: str
    target_role: str
    source_status: str
    position_world: tuple[float, float, float]
    class_name: str | None
    confidence: float | None
    T_world_object: tuple[tuple[float, float, float, float], ...] | None
    best_image_path: str | None
    best_bbox_xyxy: tuple[float, float, float, float] | None
    supporting_views: tuple[str, ...]
    candidate_class_names: tuple[str, ...]
    source_payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "target_role": self.target_role,
            "source_status": self.source_status,
            "position_world": [_round(value) for value in self.position_world],
            "class_name": self.class_name,
            "confidence": _round(self.confidence) if self.confidence is not None else None,
            "T_world_object": [list(row) for row in self.T_world_object] if self.T_world_object else None,
            "best_image_path": self.best_image_path,
            "best_bbox_xyxy": [_round(value) for value in self.best_bbox_xyxy] if self.best_bbox_xyxy else None,
            "supporting_views": list(self.supporting_views),
            "candidate_class_names": list(self.candidate_class_names),
        }


@dataclass(frozen=True)
class FinalViewCandidate:
    candidate_id: str
    view: SurveyView
    entry_views: tuple[SurveyView, ...]
    approach_angle_rad: float
    angle_source: str
    angle_offset_deg: float
    standoff_m: float
    standoff_multiplier: float
    direction_adjusted: bool
    camera_position_world: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "view": self.view.to_dict(),
            "entry_views": [view.to_dict() for view in self.entry_views],
            "approach_angle_rad": _round(self.approach_angle_rad),
            "angle_source": self.angle_source,
            "angle_offset_deg": _round(self.angle_offset_deg),
            "standoff_m": _round(self.standoff_m),
            "standoff_multiplier": _round(self.standoff_multiplier),
            "direction_adjusted": self.direction_adjusted,
            "camera_position_world": [_round(value) for value in self.camera_position_world],
        }


@dataclass(frozen=True)
class FinalPlannedCapture:
    target: FinalTarget
    view: SurveyView
    entry_views: tuple[SurveyView, ...]
    approach_angle_rad: float
    direction_source: str
    direction_adjusted: bool
    evaluated_camera_positions: tuple[dict[str, Any], ...]
    view_candidates: tuple[FinalViewCandidate, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.to_dict(),
            "view": self.view.to_dict(),
            "entry_views": [view.to_dict() for view in self.entry_views],
            "approach_angle_rad": _round(self.approach_angle_rad),
            "direction_source": self.direction_source,
            "direction_adjusted": self.direction_adjusted,
            "evaluated_camera_positions": list(self.evaluated_camera_positions),
            "view_candidates": [candidate.to_dict() for candidate in self.view_candidates],
        }


def find_latest_row_report(output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    root = Path(output_dir)
    candidates = sorted(root.glob("*/row/row_report.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no task1 row_report.json files found under {root}")
    return candidates[-1]


def load_task1_row_report(path: Path | str) -> Task1RowReport:
    report_path = Path(path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "task1_row_report_v1":
        raise ValueError(f"unsupported task1 row report schema_version: {payload.get('schema_version')}")
    workspace = _workspace_from_payload(_required(payload, "workspace"))
    views_payload = payload.get("views", [])
    if not isinstance(views_payload, list):
        raise ValueError("task1 row report views must be a list")
    stable_objects = payload.get("stable_objects")
    tentative_objects = payload.get("tentative_objects")
    ambiguous_objects = payload.get("ambiguous_objects")
    if not isinstance(stable_objects, list):
        raise ValueError("task1 row report must contain stable_objects list")
    if not isinstance(tentative_objects, list):
        raise ValueError("task1 row report must contain tentative_objects list")
    if not isinstance(ambiguous_objects, list):
        raise ValueError("task1 row report must contain ambiguous_objects list")
    object_selection_summary = payload.get("object_selection_summary", {})
    if not isinstance(object_selection_summary, dict):
        raise ValueError("task1 row report object_selection_summary must be a dict when present")
    return Task1RowReport(
        path=report_path,
        schema_version=str(payload["schema_version"]),
        status=str(payload.get("status", "")),
        layout_snapshot_path=_required_path(payload, "layout_snapshot_path"),
        task1_run_dir=Path(str(payload.get("task1_run_dir") or report_path.parent.parent)),
        row_dir=Path(str(payload.get("row_dir") or report_path.parent)),
        camera_name=str(payload.get("camera_name") or DEFAULT_CAMERA_NAME),
        workspace=workspace,
        image_size=_image_size_from_payload(payload.get("image_size")),
        views=tuple(dict(view) for view in views_payload if isinstance(view, dict)),
        views_by_id=_views_by_id(views_payload),
        stable_objects=tuple(dict(obj) for obj in stable_objects if isinstance(obj, dict)),
        tentative_objects=tuple(dict(obj) for obj in tentative_objects if isinstance(obj, dict)),
        ambiguous_objects=tuple(dict(obj) for obj in ambiguous_objects if isinstance(obj, dict)),
        object_selection_summary=object_selection_summary,
        raw_payload=payload,
    )


def build_final_plan(
    row_report: Task1RowReport | Path | str,
    config: FinalConfig = FinalConfig(),
) -> tuple[SurveyWorkspace, tuple[FinalPlannedCapture, ...]]:
    config.validate()
    report = load_task1_row_report(row_report) if not isinstance(row_report, Task1RowReport) else row_report
    workspace = report.workspace
    if config.camera_z_m <= workspace.bottom_z_m:
        raise ValueError(
            "final wrist camera z must be above the tank bottom/object base plane: "
            f"camera_z={config.camera_z_m:.4f}, bottom_z={workspace.bottom_z_m:.4f}"
        )
    targets = _targets_from_row_report(report)
    planned: list[FinalPlannedCapture] = []
    for target_index, target in enumerate(targets):
        _validate_target_position(target, workspace)
        look_at = _look_at_for_target(target.position_world, workspace=workspace, config=config)
        if look_at[2] >= config.camera_z_m:
            raise ValueError(
                "final look-at height must stay below camera_z_m for an oblique in-tank view: "
                f"look_at_z={look_at[2]:.4f}, camera_z={config.camera_z_m:.4f}"
            )
        preferred_angle, direction_source = _target_primary_angle(target, report, config=config)
        camera_candidates, evaluated = _select_camera_position_candidates(
            target.position_world,
            requested_angle_rad=preferred_angle,
            workspace=workspace,
            config=config,
        )
        view_id = f"final_{_safe_id(target.object_id)}"
        view_candidates = _view_candidates_for_target(
            target_index=target_index,
            view_id_prefix=view_id,
            camera_candidates=camera_candidates,
            preferred_angle=preferred_angle,
            look_at=look_at,
            workspace=workspace,
            config=config,
        )
        primary_candidate = view_candidates[0]
        planned.append(
            FinalPlannedCapture(
                target=target,
                view=primary_candidate.view,
                entry_views=primary_candidate.entry_views,
                approach_angle_rad=primary_candidate.approach_angle_rad,
                direction_source=direction_source,
                direction_adjusted=primary_candidate.direction_adjusted,
                evaluated_camera_positions=tuple(evaluated),
                view_candidates=view_candidates,
            )
        )
    return workspace, tuple(planned)


def run_task1_final(
    row_report_path: Path | str,
    config: FinalConfig = FinalConfig(),
    *,
    detector: SurveyDetector | None = None,
) -> dict[str, Any]:
    config.validate()
    row_report = load_task1_row_report(row_report_path)
    workspace, planned_captures = build_final_plan(row_report, config)
    created_utc = datetime.now(timezone.utc).isoformat()
    capture_dir = row_report.task1_run_dir / "final"
    images_dir = capture_dir / "images"
    depth_dir = capture_dir / "depth"
    yolo_dir = capture_dir / "yolo_raw"
    annotated_dir = capture_dir / "annotated"
    tiles_dir = capture_dir / "tiles"
    plan_path = capture_dir / "final_plan.json"
    report_path = capture_dir / "final_report.json"
    plan_payload = _plan_payload(
        created_utc=created_utc,
        row_report=row_report,
        workspace=workspace,
        config=config,
        capture_dir=capture_dir,
        plan_path=plan_path,
        report_path=report_path,
        planned_captures=planned_captures,
    )

    if not planned_captures:
        report = _report_payload(
            created_utc=created_utc,
            status="failed",
            message="No stable, tentative, or ambiguous row targets were available for final capture.",
            row_report=row_report,
            workspace=workspace,
            config=config,
            capture_dir=capture_dir,
            plan_path=plan_path,
            report_path=report_path,
            planned_captures=planned_captures,
            object_captures=[],
        )
        _write_json(plan_path, plan_payload)
        _write_json(report_path, report)
        return report

    if config.plan_only:
        object_captures = [_planned_capture_result(planned) for planned in planned_captures]
        report = _report_payload(
            created_utc=created_utc,
            status="plan_only",
            message=(
                f"Generated final capture plan for {len(planned_captures)} targets "
                "without MuJoCo rendering or YOLO inference."
            ),
            row_report=row_report,
            workspace=workspace,
            config=config,
            capture_dir=capture_dir,
            plan_path=plan_path,
            report_path=report_path,
            planned_captures=planned_captures,
            object_captures=object_captures,
        )
        _write_json(plan_path, plan_payload)
        _write_json(report_path, report)
        return report

    if config.run_yolo and detector is None:
        object_captures = [_planned_capture_result(planned) for planned in planned_captures]
        report = _report_payload(
            created_utc=created_utc,
            status="failed",
            message="run_yolo=True requires a SurveyDetector; pass one from the task1 entrypoint or set run_yolo=False.",
            row_report=row_report,
            workspace=workspace,
            config=config,
            capture_dir=capture_dir,
            plan_path=plan_path,
            report_path=report_path,
            planned_captures=planned_captures,
            object_captures=object_captures,
        )
        _write_json(plan_path, plan_payload)
        _write_json(report_path, report)
        return report

    layout = load_stage0_layout(row_report.layout_snapshot_path)
    scene = survey_scene_from_stage0_layout(
        layout,
        tank_opening_z_m=config.tank_opening_z_m,
        opening_clearance_m=config.opening_clearance_m,
    )
    backend = MujocoSurveyBackend(config=config.capture_config(), scene=scene, detector=detector)
    object_captures: list[dict[str, Any]] = []
    try:
        backend.load()
        backend.apply_scene_objects()
        for planned in planned_captures:
            capture_result = _capture_first_reachable_candidate(
                backend=backend,
                planned=planned,
                images_dir=images_dir,
                depth_dir=depth_dir,
                yolo_dir=yolo_dir,
                annotated_dir=annotated_dir,
                tiles_dir=tiles_dir,
                config=config,
            )
            object_captures.append(capture_result)
    except Exception as exc:
        report = _report_payload(
            created_utc=created_utc,
            status="failed",
            message=str(exc),
            row_report=row_report,
            workspace=workspace,
            config=config,
            capture_dir=capture_dir,
            plan_path=plan_path,
            report_path=report_path,
            planned_captures=planned_captures,
            object_captures=object_captures,
        )
        _write_json(plan_path, plan_payload)
        _write_json(report_path, report)
        return report
    finally:
        backend.close()

    status = _overall_status(object_captures)
    message = _summary_message(object_captures, planned_count=len(planned_captures))
    report = _report_payload(
        created_utc=created_utc,
        status=status,
        message=message,
        row_report=row_report,
        workspace=workspace,
        config=config,
        capture_dir=capture_dir,
        plan_path=plan_path,
        report_path=report_path,
        planned_captures=planned_captures,
        object_captures=object_captures,
    )
    _write_json(plan_path, plan_payload)
    _write_json(report_path, report)
    return report


def select_stable_final_objects(object_captures: list[dict[str, Any]]) -> dict[str, Any]:
    if not object_captures:
        return _stable_selection_payload([], [], status="not_run")
    if all(capture.get("status") == "planned" for capture in object_captures):
        return _stable_selection_payload([], [], status="not_run")

    stable_objects: list[dict[str, Any]] = []
    rejected_objects: list[dict[str, Any]] = []
    for capture in object_captures:
        stable_object, rejection = _stable_object_from_capture(capture)
        if stable_object is not None:
            stable_objects.append(stable_object)
        elif rejection is not None:
            rejected_objects.append(rejection)
    return _stable_selection_payload(stable_objects, rejected_objects, status="success")


def _capture_first_reachable_candidate(
    *,
    backend: MujocoSurveyBackend,
    planned: FinalPlannedCapture,
    images_dir: Path,
    depth_dir: Path,
    yolo_dir: Path,
    annotated_dir: Path,
    tiles_dir: Path,
    config: FinalConfig,
) -> dict[str, Any]:
    candidate_attempts: list[dict[str, Any]] = []
    for candidate in planned.view_candidates:
        entry_results = _validate_entry_views(backend, planned, candidate)
        failed_entry = next((entry for entry in entry_results if entry.get("status") != "success"), None)
        if failed_entry is not None:
            candidate_attempts.append(
                _candidate_attempt_payload(
                    candidate=candidate,
                    entry_results=entry_results,
                    view_result=_planned_view_result(candidate.view),
                    status="entry_validation_failed",
                    message=str(failed_entry.get("message") or "entry validation failed"),
                )
            )
            continue

        view_result, observations = backend.capture_view(
            candidate.view,
            images_dir=images_dir,
            depth_dir=depth_dir,
            yolo_dir=yolo_dir,
            annotated_dir=annotated_dir,
            tiles_dir=tiles_dir,
        )
        attempt_status = "captured" if view_result.get("status") == "success" else "final_pose_failed"
        candidate_attempts.append(
            _candidate_attempt_payload(
                candidate=candidate,
                entry_results=entry_results,
                view_result=view_result,
                status=attempt_status,
                message=str(view_result.get("message") or attempt_status),
            )
        )
        if view_result.get("status") != "success":
            continue

        return _object_capture_result(
            planned=planned,
            selected_candidate=candidate,
            candidate_attempts=candidate_attempts,
            entry_results=entry_results,
            view_result=view_result,
            observations=observations,
            config=config,
        )

    return _all_candidates_failed_capture_result(planned, candidate_attempts)


def _candidate_attempt_payload(
    *,
    candidate: FinalViewCandidate,
    entry_results: list[dict[str, Any]],
    view_result: dict[str, Any],
    status: str,
    message: str,
) -> dict[str, Any]:
    return {
        "candidate": candidate.to_dict(),
        "status": status,
        "message": message,
        "entry_validation": entry_results,
        "view": view_result,
    }


def _targets_from_row_report(report: Task1RowReport) -> tuple[FinalTarget, ...]:
    targets: list[FinalTarget] = []
    for payload in report.stable_objects:
        targets.append(_target_from_stable_payload(payload))
    for payload in report.tentative_objects:
        targets.append(_target_from_tentative_payload(payload))
    for payload in report.ambiguous_objects:
        targets.append(_target_from_ambiguous_payload(payload))
    return tuple(targets)


def _target_from_stable_payload(payload: dict[str, Any]) -> FinalTarget:
    object_id = _required_text(payload, "object_id")
    class_name = _required_text(payload, "class_name")
    transform = _transform_from_value(_required(payload, "T_world_object"))
    return FinalTarget(
        object_id=object_id,
        target_role="primary",
        source_status="stable",
        position_world=_position_from_value(_required(payload, "position_world")),
        class_name=class_name,
        confidence=float(payload.get("confidence", 0.0)),
        T_world_object=transform,
        best_image_path=_optional_text(payload.get("best_image_path")),
        best_bbox_xyxy=_optional_bbox(payload.get("best_bbox_xyxy")),
        supporting_views=_string_tuple(payload.get("supporting_row_views", [])),
        candidate_class_names=(class_name,),
        source_payload=dict(payload),
    )


def _target_from_tentative_payload(payload: dict[str, Any]) -> FinalTarget:
    object_id = _required_text(payload, "object_id")
    class_name = _optional_text(payload.get("class_name"))
    return FinalTarget(
        object_id=object_id,
        target_role="follow_up",
        source_status="tentative",
        position_world=_position_from_value(_required(payload, "position_world")),
        class_name=class_name,
        confidence=float(payload["confidence"]) if payload.get("confidence") is not None else None,
        T_world_object=_optional_transform(payload.get("T_world_object")),
        best_image_path=_optional_text(payload.get("best_image_path")),
        best_bbox_xyxy=_optional_bbox(payload.get("best_bbox_xyxy")),
        supporting_views=_string_tuple(payload.get("supporting_row_views", [])),
        candidate_class_names=(class_name,) if class_name else (),
        source_payload=dict(payload),
    )


def _target_from_ambiguous_payload(payload: dict[str, Any]) -> FinalTarget:
    object_id = _required_text(payload, "object_id")
    class_candidates = payload.get("class_candidates", [])
    if not isinstance(class_candidates, list):
        raise ValueError("ambiguous object class_candidates must be a list")
    best_candidate = _best_class_candidate(class_candidates)
    class_names = _unique_texts(candidate.get("class_name") for candidate in class_candidates if isinstance(candidate, dict))
    supporting_views = _unique_texts(
        view_id
        for candidate in class_candidates
        if isinstance(candidate, dict)
        for view_id in candidate.get("supporting_row_views", [])
    )
    return FinalTarget(
        object_id=object_id,
        target_role="follow_up",
        source_status="ambiguous",
        position_world=_position_from_value(_required(payload, "position_world")),
        class_name=None,
        confidence=float(best_candidate["confidence"]) if best_candidate and best_candidate.get("confidence") is not None else None,
        T_world_object=None,
        best_image_path=_optional_text(best_candidate.get("best_image_path")) if best_candidate else None,
        best_bbox_xyxy=_optional_bbox(best_candidate.get("best_bbox_xyxy")) if best_candidate else None,
        supporting_views=supporting_views,
        candidate_class_names=class_names,
        source_payload=dict(payload),
    )


def _best_class_candidate(class_candidates: list[Any]) -> dict[str, Any] | None:
    records = [candidate for candidate in class_candidates if isinstance(candidate, dict)]
    if not records:
        return None
    return max(
        records,
        key=lambda item: (
            _float_or_zero(item.get("evidence_score")),
            _float_or_zero(item.get("confidence")),
        ),
    )


def _validate_target_position(target: FinalTarget, workspace: SurveyWorkspace) -> None:
    x, y, z = target.position_world
    if not (workspace.x_min <= x <= workspace.x_max and workspace.y_min <= y <= workspace.y_max):
        raise ValueError(f"final target {target.object_id} position is outside tank workspace: {target.position_world}")
    if not (workspace.bottom_z_m <= z <= workspace.tank_opening_z_m):
        raise ValueError(
            f"final target {target.object_id} z is outside the tank vertical range: {target.position_world}"
        )


def _look_at_for_target(
    position_world: tuple[float, float, float],
    *,
    workspace: SurveyWorkspace,
    config: FinalConfig,
) -> tuple[float, float, float]:
    target_z = max(workspace.bottom_z_m, position_world[2])
    return (
        position_world[0],
        position_world[1],
        min(workspace.max_camera_z_m, target_z + config.look_at_height_offset_m),
    )


def _target_primary_angle(
    target: FinalTarget,
    report: Task1RowReport,
    *,
    config: FinalConfig,
) -> tuple[float, str]:
    if target.best_image_path:
        view_payload = _view_for_image_path(target.best_image_path, report.views)
        angle = _angle_from_view_to_target(view_payload, target.position_world) if view_payload is not None else None
        if angle is not None:
            return angle, "best_row_image_view"

    for view_id in target.supporting_views:
        view_payload = report.views_by_id.get(view_id)
        angle = _angle_from_view_to_target(view_payload, target.position_world) if view_payload is not None else None
        if angle is not None:
            return angle, "supporting_row_view"

    return _entry_side_angle(target.position_world, workspace=report.workspace, entry_side=config.entry_side), "entry_side_policy"


def _view_for_image_path(image_path: str, views: tuple[dict[str, Any], ...]) -> dict[str, Any] | None:
    for view in views:
        if view.get("rgb_image_path") == image_path or view.get("image_path") == image_path:
            return view
    return None


def _angle_from_view_to_target(view_payload: dict[str, Any], target: tuple[float, float, float]) -> float | None:
    camera_position = _view_camera_position(view_payload)
    if camera_position is None:
        return None
    dx = camera_position[0] - target[0]
    dy = camera_position[1] - target[1]
    if math.hypot(dx, dy) <= 1e-9:
        return None
    return math.atan2(dy, dx)


def _entry_side_angle(
    target: tuple[float, float, float],
    *,
    workspace: SurveyWorkspace,
    entry_side: str,
) -> float:
    center_x = (workspace.x_min + workspace.x_max) / 2.0
    center_y = (workspace.y_min + workspace.y_max) / 2.0
    if entry_side == "y-max":
        reference = (center_x, workspace.y_max)
    elif entry_side == "y-min":
        reference = (center_x, workspace.y_min)
    elif entry_side == "x-min":
        reference = (workspace.x_min, center_y)
    elif entry_side == "x-max":
        reference = (workspace.x_max, center_y)
    elif entry_side == "center":
        reference = (center_x, center_y)
    else:
        raise ValueError(f"unsupported entry_side: {entry_side}")
    dx = reference[0] - target[0]
    dy = reference[1] - target[1]
    if math.hypot(dx, dy) <= 1e-9:
        return 0.0
    return math.atan2(dy, dx)


def _select_camera_position_candidates(
    target: tuple[float, float, float],
    *,
    requested_angle_rad: float,
    workspace: SurveyWorkspace,
    config: FinalConfig,
) -> tuple[tuple[dict[str, Any], ...], list[dict[str, Any]]]:
    center_angle = math.atan2(
        (workspace.y_min + workspace.y_max) / 2.0 - target[1],
        (workspace.x_min + workspace.x_max) / 2.0 - target[0],
    )
    angle_records = _unique_angle_records(
        [
            {
                "angle_rad": requested_angle_rad + math.radians(offset_deg),
                "angle_source": "row_evidence_offset",
                "angle_offset_deg": float(offset_deg),
            }
            for offset_deg in config.final_view_angle_offsets_deg
        ]
        + [
            {
                "angle_rad": center_angle + math.radians(offset_deg),
                "angle_source": "workspace_centerline_offset",
                "angle_offset_deg": float(offset_deg),
            }
            for offset_deg in config.centerline_view_angle_offsets_deg
        ]
    )
    standoff_records = _unique_standoff_records(config)
    evaluated: list[dict[str, Any]] = []
    valid_candidates: list[dict[str, Any]] = []
    for standoff_record in standoff_records:
        standoff_m = float(standoff_record["standoff_m"])
        standoff_multiplier = float(standoff_record["standoff_multiplier"])
        for record in angle_records:
            angle = float(record["angle_rad"])
            position = (
                target[0] + math.cos(angle) * standoff_m,
                target[1] + math.sin(angle) * standoff_m,
                config.camera_z_m,
            )
            valid, reason = _camera_position_validity(position, target, workspace=workspace, config=config)
            payload = {
                "angle_rad": _round(_normalize_angle(angle)),
                "angle_source": str(record["angle_source"]),
                "angle_offset_deg": _round(float(record["angle_offset_deg"])),
                "standoff_m": _round(standoff_m),
                "standoff_multiplier": _round(standoff_multiplier),
                "camera_position_world": [_round(value) for value in position],
                "valid": valid,
                "reason": reason,
            }
            evaluated.append(payload)
            if valid:
                valid_candidates.append(
                    {
                        "angle_rad": _normalize_angle(angle),
                        "angle_source": str(record["angle_source"]),
                        "angle_offset_deg": float(record["angle_offset_deg"]),
                        "standoff_m": _round(standoff_m),
                        "standoff_multiplier": _round(standoff_multiplier),
                        "camera_position_world": _round_vector(position),
                    }
                )
    if valid_candidates:
        return (tuple(valid_candidates), evaluated)
    raise ValueError(f"could not place final camera inside tank for target near {target}")


def _view_candidates_for_target(
    *,
    target_index: int,
    view_id_prefix: str,
    camera_candidates: tuple[dict[str, Any], ...],
    preferred_angle: float,
    look_at: tuple[float, float, float],
    workspace: SurveyWorkspace,
    config: FinalConfig,
) -> tuple[FinalViewCandidate, ...]:
    view_candidates: list[FinalViewCandidate] = []
    for candidate_index, candidate in enumerate(camera_candidates):
        suffix = "primary" if candidate_index == 0 else f"alt_{candidate_index:02d}"
        view_id = view_id_prefix if candidate_index == 0 else f"{view_id_prefix}_{suffix}"
        camera_position = _position_from_value(candidate["camera_position_world"])
        view = SurveyView(
            view_id=view_id,
            grid_row=target_index,
            grid_col=candidate_index,
            camera_name=config.camera_name,
            desired_camera_position_world=_round_vector(camera_position),
            look_at_world=_round_vector(look_at),
            T_world_camera=_make_look_at_transform(camera_position, look_at),
        )
        entry_views = _entry_validation_views(
            view_id_prefix=view_id,
            final_camera_position=camera_position,
            look_at=look_at,
            workspace=workspace,
            config=config,
            target_index=target_index,
        )
        angle = float(candidate["angle_rad"])
        view_candidates.append(
            FinalViewCandidate(
                candidate_id=f"{view_id_prefix}_{suffix}",
                view=view,
                entry_views=entry_views,
                approach_angle_rad=angle,
                angle_source=str(candidate["angle_source"]),
                angle_offset_deg=float(candidate["angle_offset_deg"]),
                standoff_m=float(candidate["standoff_m"]),
                standoff_multiplier=float(candidate["standoff_multiplier"]),
                direction_adjusted=abs(_normalize_angle(angle - preferred_angle)) > 1e-6,
                camera_position_world=camera_position,
            )
        )
    return tuple(view_candidates)


def _camera_position_validity(
    position: tuple[float, float, float],
    target: tuple[float, float, float],
    *,
    workspace: SurveyWorkspace,
    config: FinalConfig,
) -> tuple[bool, str]:
    if math.hypot(position[0] - target[0], position[1] - target[1]) < config.min_oblique_distance_m:
        return False, "below_min_oblique_distance"
    try:
        workspace.validate_camera_position(position)
    except ValueError as exc:
        return False, str(exc)
    return True, "accepted"


def _entry_validation_views(
    *,
    view_id_prefix: str,
    final_camera_position: tuple[float, float, float],
    look_at: tuple[float, float, float],
    workspace: SurveyWorkspace,
    config: FinalConfig,
    target_index: int,
) -> tuple[SurveyView, ...]:
    top_z = workspace.max_camera_z_m - config.entry_clearance_margin_m
    if top_z <= final_camera_position[2]:
        raise ValueError(
            "final entry clearance margin leaves no vertical room above final camera pose: "
            f"entry_top_z={top_z:.4f}, final_camera_z={final_camera_position[2]:.4f}"
        )
    views: list[SurveyView] = []
    for sample_index in range(config.entry_validation_samples):
        fraction_from_final = (config.entry_validation_samples - sample_index) / config.entry_validation_samples
        z = final_camera_position[2] + (top_z - final_camera_position[2]) * fraction_from_final
        camera_position = (final_camera_position[0], final_camera_position[1], z)
        workspace.validate_camera_position(camera_position)
        views.append(
            SurveyView(
                view_id=f"{view_id_prefix}_entry_{sample_index:02d}",
                grid_row=target_index,
                grid_col=sample_index + 1,
                camera_name=config.camera_name,
                desired_camera_position_world=_round_vector(camera_position),
                look_at_world=_round_vector(look_at),
                T_world_camera=_make_look_at_transform(camera_position, look_at),
            )
        )
    return tuple(views)


def _validate_entry_views(
    backend: MujocoSurveyBackend,
    planned: FinalPlannedCapture,
    candidate: FinalViewCandidate | None = None,
) -> list[dict[str, Any]]:
    active_candidate = candidate or planned.view_candidates[0]
    results: list[dict[str, Any]] = []
    for view in active_candidate.entry_views:
        result = backend.validate_view_pose(view)
        result["validation_role"] = "top_opening_entry_sample"
        result["target_object_id"] = planned.target.object_id
        result["view_candidate_id"] = active_candidate.candidate_id
        results.append(result)
        if result.get("status") != "success":
            break
    return results


def _object_capture_result(
    *,
    planned: FinalPlannedCapture,
    selected_candidate: FinalViewCandidate,
    candidate_attempts: list[dict[str, Any]],
    entry_results: list[dict[str, Any]],
    view_result: dict[str, Any],
    observations: list[SurveyObservation],
    config: FinalConfig,
) -> dict[str, Any]:
    matched = _matched_observations(planned.target, observations, config=config)
    best = matched[0] if matched else None
    recognition = _recognition_payload(planned.target, best, config=config)
    status = _capture_status(planned.target, view_result, matched, recognition, run_yolo=config.run_yolo)
    return {
        "target": planned.target.to_dict(),
        "status": status,
        "source_status": planned.target.source_status,
        "target_role": planned.target.target_role,
        "selected_view_candidate": selected_candidate.to_dict(),
        "view_candidate_attempts": candidate_attempts,
        "view": view_result,
        "entry_validation": entry_results,
        "matched_observations": matched,
        "best_observation": best,
        "recognition": recognition,
        "final_image_path": view_result.get("rgb_image_path"),
        "depth_path": view_result.get("depth_path"),
        "annotated_image_path": view_result.get("annotated_image_path"),
        "yolo_raw_path": view_result.get("yolo_raw_path"),
        "notes": _capture_notes(planned.target, status),
    }


def _all_candidates_failed_capture_result(
    planned: FinalPlannedCapture,
    candidate_attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    any_final_pose_attempt = any(attempt.get("status") == "final_pose_failed" for attempt in candidate_attempts)
    status = "capture_failed" if any_final_pose_attempt else "entry_validation_failed"
    reason = "all_final_view_candidates_failed" if any_final_pose_attempt else "all_entry_validation_candidates_failed"
    last_attempt = candidate_attempts[-1] if candidate_attempts else {}
    return {
        "target": planned.target.to_dict(),
        "status": status,
        "source_status": planned.target.source_status,
        "target_role": planned.target.target_role,
        "selected_view_candidate": None,
        "view_candidate_attempts": candidate_attempts,
        "view": last_attempt.get("view", _planned_view_result(planned.view)),
        "entry_validation": last_attempt.get("entry_validation", []),
        "matched_observations": [],
        "best_observation": None,
        "recognition": {
            "status": "not_run",
            "reason": reason,
        },
        "final_image_path": None,
        "depth_path": None,
        "annotated_image_path": None,
        "yolo_raw_path": None,
        "notes": [
            "No planned final-view candidate passed both entry validation and final pose validation.",
            "Each candidate attempt is recorded in view_candidate_attempts.",
        ],
    }


def _entry_failed_capture_result(
    planned: FinalPlannedCapture,
    entry_results: list[dict[str, Any]],
    failed_entry: dict[str, Any],
) -> dict[str, Any]:
    return {
        "target": planned.target.to_dict(),
        "status": "entry_validation_failed",
        "source_status": planned.target.source_status,
        "target_role": planned.target.target_role,
        "selected_view_candidate": None,
        "view_candidate_attempts": [
            _candidate_attempt_payload(
                candidate=planned.view_candidates[0],
                entry_results=entry_results,
                view_result=_planned_view_result(planned.view),
                status="entry_validation_failed",
                message=str(failed_entry.get("message") or "entry validation failed"),
            )
        ],
        "view": _planned_view_result(planned.view),
        "entry_validation": entry_results,
        "matched_observations": [],
        "best_observation": None,
        "recognition": {
            "status": "not_run",
            "reason": "entry_validation_failed",
        },
        "final_image_path": None,
        "depth_path": None,
        "annotated_image_path": None,
        "yolo_raw_path": None,
        "notes": [
            "The top-opening entry pose samples did not all pass IK and collision checks, so the final photo was not rendered.",
            f"First failed entry sample: {failed_entry.get('view_id')}.",
        ],
    }


def _planned_capture_result(planned: FinalPlannedCapture) -> dict[str, Any]:
    return {
        "target": planned.target.to_dict(),
        "status": "planned",
        "source_status": planned.target.source_status,
        "target_role": planned.target.target_role,
        "selected_view_candidate": None,
        "view_candidate_attempts": [],
        "view_candidates": [candidate.to_dict() for candidate in planned.view_candidates],
        "view": _planned_view_result(planned.view),
        "entry_validation": [_planned_view_result(view) for view in planned.entry_views],
        "matched_observations": [],
        "best_observation": None,
        "recognition": {
            "status": "not_run",
            "reason": "plan_only",
        },
        "final_image_path": None,
        "depth_path": None,
        "annotated_image_path": None,
        "yolo_raw_path": None,
        "notes": ["This target has a planned final capture view but was not rendered."],
    }


def _stable_object_from_capture(capture: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    target = capture.get("target")
    if not isinstance(target, dict):
        return None, _rejected_stable_object_payload(capture, reason="missing_target")
    status = str(capture.get("status") or "")
    if status == "confirmed":
        reason = "confirmed_primary_object"
    elif status == "follow_up_observed":
        reason = _follow_up_stable_reason(target, capture)
        if reason is None:
            return None, _rejected_stable_object_payload(capture, reason="follow_up_not_resolved")
    else:
        return None, _rejected_stable_object_payload(capture, reason=status or "not_stable_status")

    recognition = capture.get("recognition")
    if not isinstance(recognition, dict):
        return None, _rejected_stable_object_payload(capture, reason="missing_recognition")
    object_id = _optional_text(target.get("object_id"))
    if object_id is None:
        return None, _rejected_stable_object_payload(capture, reason="missing_object_id")
    detected_class = _optional_text(recognition.get("detected_class_name"))
    if detected_class is None:
        return None, _rejected_stable_object_payload(capture, reason="missing_detected_class")
    confidence = recognition.get("confidence")
    if confidence is None:
        return None, _rejected_stable_object_payload(capture, reason="missing_confidence")
    bbox = _optional_bbox(recognition.get("bbox_xyxy"))
    if bbox is None:
        return None, _rejected_stable_object_payload(capture, reason="missing_bbox_xyxy")
    position = _optional_position_from_value(recognition.get("position_world"))
    if position is None:
        return None, _rejected_stable_object_payload(capture, reason="missing_position_world")

    source_transform = _optional_transform(target.get("T_world_object"))
    transform = _transform_with_position(source_transform, position)
    detection = ObjectDetection(
        object_id=object_id,
        class_name=detected_class,
        confidence=float(confidence),
        bbox_xyxy=bbox,
        T_world_object=transform,
    )
    stable_object = dict(detection.to_dict())
    stable_object.update(
        {
            "position_world": [_round(value) for value in position],
            "source_status": target.get("source_status"),
            "target_role": target.get("target_role"),
            "final_image_path": capture.get("final_image_path"),
            "depth_path": capture.get("depth_path"),
            "annotated_image_path": capture.get("annotated_image_path"),
            "yolo_raw_path": capture.get("yolo_raw_path"),
            "matched_observation": capture.get("best_observation"),
            "selection": {
                "source": FINAL_STABLE_SELECTION_POLICY_VERSION,
                "reason": reason,
                "target_xy_distance_m": recognition.get("target_xy_distance_m"),
                "target_match_radius_m": recognition.get("target_match_radius_m"),
            },
            "pose_quality": {
                "position_source": "final_close_yolo_depth_observation",
                "orientation_source": (
                    "source_row_T_world_object"
                    if source_transform is not None
                    else "identity_orientation_no_source_pose"
                ),
            },
            "notes": _stable_object_notes(source_transform),
        }
    )
    return stable_object, None


def _follow_up_stable_reason(target: dict[str, Any], capture: dict[str, Any]) -> str | None:
    recognition = capture.get("recognition")
    if not isinstance(recognition, dict):
        return None
    detected_class = _optional_text(recognition.get("detected_class_name"))
    if detected_class is None:
        return None
    source_status = str(target.get("source_status") or "")
    source_class = _optional_text(target.get("class_name"))
    candidate_class_names = _string_tuple(target.get("candidate_class_names", []))
    if source_status == "tentative":
        if source_class is not None and detected_class != source_class:
            return None
        return "resolved_tentative_object"
    if source_status == "ambiguous":
        if candidate_class_names and detected_class not in candidate_class_names:
            return None
        return "resolved_ambiguous_object"
    return None


def _stable_selection_payload(
    stable_objects: list[dict[str, Any]],
    rejected_objects: list[dict[str, Any]],
    *,
    status: str,
) -> dict[str, Any]:
    rejected_counts: dict[str, int] = {}
    for rejected in rejected_objects:
        reason = str(rejected.get("reason") or "unknown")
        rejected_counts[reason] = rejected_counts.get(reason, 0) + 1
    return {
        "stable_objects": stable_objects,
        "unstable_objects": rejected_objects,
        "stable_object_selection": {
            "status": status,
            "policy_version": FINAL_STABLE_SELECTION_POLICY_VERSION,
            "stable_object_count": len(stable_objects),
            "unstable_object_count": len(rejected_objects),
            "rejected_reason_counts": rejected_counts,
            "rules": [
                "Primary row-stable targets enter stable_objects only when close capture confirms the same class.",
                "Tentative targets enter stable_objects only when close capture observes the same class, or when no tentative class was supplied.",
                "Ambiguous targets enter stable_objects only when close capture resolves to one of the candidate classes.",
                "Unconfirmed, class-conflict, skipped, and failed captures are kept out of stable_objects.",
            ],
        },
    }


def _rejected_stable_object_payload(capture: dict[str, Any], *, reason: str) -> dict[str, Any]:
    target = capture.get("target") if isinstance(capture.get("target"), dict) else {}
    recognition = capture.get("recognition") if isinstance(capture.get("recognition"), dict) else {}
    return {
        "object_id": target.get("object_id"),
        "source_status": target.get("source_status"),
        "target_role": target.get("target_role"),
        "capture_status": capture.get("status"),
        "reason": reason,
        "detected_class_name": recognition.get("detected_class_name"),
        "source_class_name": recognition.get("source_class_name"),
        "candidate_class_names": recognition.get("candidate_class_names", []),
        "final_image_path": capture.get("final_image_path"),
        "notes": [
            "This target is intentionally excluded from stable_objects because close-capture evidence did not satisfy the stable selection policy.",
        ],
    }


def _stable_object_notes(
    source_transform: tuple[tuple[float, float, float, float], ...] | None,
) -> list[str]:
    notes = [
        "Stable object selected from final close capture evidence.",
        "The translation in T_world_object is updated from close YOLO-depth association.",
    ]
    if source_transform is None:
        notes.append(
            "No source object orientation was available, so T_world_object uses identity orientation and should not be treated as final yaw."
        )
    else:
        notes.append("The orientation in T_world_object is inherited from the row-stage source pose.")
    return notes


def _matched_observations(
    target: FinalTarget,
    observations: list[SurveyObservation],
    *,
    config: FinalConfig,
) -> list[dict[str, Any]]:
    matched: list[dict[str, Any]] = []
    for observation in observations:
        if observation.rough_position_world is None:
            continue
        distance = _xy_distance(observation.rough_position_world, target.position_world)
        if distance > config.target_match_radius_m:
            continue
        payload = observation.to_dict()
        payload["target_xy_distance_m"] = _round(distance)
        payload["observation_kind"] = "single_detection"
        matched.append(payload)
    matched = _merge_same_class_observation_fragments(matched, config=config)
    matched.sort(
        key=lambda item: _final_observation_selection_score(target, item, config=config),
        reverse=True,
    )
    return matched


def _recognition_payload(
    target: FinalTarget,
    best_observation: dict[str, Any] | None,
    *,
    config: FinalConfig,
) -> dict[str, Any]:
    if not config.run_yolo:
        return {
            "status": "not_run",
            "reason": "yolo_skipped",
            "target_match_radius_m": _round(config.target_match_radius_m),
        }
    if best_observation is None:
        return {
            "status": "unconfirmed",
            "reason": "no_close_yolo_depth_observation",
            "target_match_radius_m": _round(config.target_match_radius_m),
        }
    detected_class = _optional_text(best_observation.get("class_name"))
    class_match: bool | None
    if target.class_name is None or detected_class is None:
        class_match = None
    else:
        class_match = detected_class == target.class_name
    return {
        "status": "observed",
        "source_class_name": target.class_name,
        "detected_class_name": detected_class,
        "class_match": class_match,
        "candidate_class_names": list(target.candidate_class_names),
        "confidence": best_observation.get("confidence"),
        "position_world": best_observation.get("rough_position_world"),
        "target_xy_distance_m": best_observation.get("target_xy_distance_m"),
        "bbox_xyxy": best_observation.get("bbox_xyxy"),
        "target_match_radius_m": _round(config.target_match_radius_m),
    }


def _merge_same_class_observation_fragments(
    matched: list[dict[str, Any]],
    *,
    config: FinalConfig,
) -> list[dict[str, Any]]:
    merged = list(matched)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for item in matched:
        class_name = _optional_text(item.get("class_name"))
        image_path = _optional_text(item.get("image_path"))
        view_id = _optional_text(item.get("view_id"))
        if class_name is None or image_path is None or view_id is None:
            continue
        groups.setdefault((view_id, image_path, class_name), []).append(item)

    for (_, _, class_name), group in groups.items():
        if len(group) < 2:
            continue
        for component in _overlapping_bbox_components(
            group,
            min_overlap_ratio=config.bbox_fragment_min_overlap_ratio,
        ):
            if len(component) < 2:
                continue
            merged.append(_merged_observation_payload(component, class_name=class_name))
    return merged


def _overlapping_bbox_components(
    observations: list[dict[str, Any]],
    *,
    min_overlap_ratio: float,
) -> list[list[dict[str, Any]]]:
    components: list[list[dict[str, Any]]] = []
    remaining = list(observations)
    while remaining:
        seed = remaining.pop(0)
        component = [seed]
        changed = True
        while changed:
            changed = False
            for candidate in list(remaining):
                if any(
                    _bbox_overlap_ratio(_bbox_tuple(candidate.get("bbox_xyxy")), _bbox_tuple(item.get("bbox_xyxy")))
                    >= min_overlap_ratio
                    for item in component
                ):
                    component.append(candidate)
                    remaining.remove(candidate)
                    changed = True
        components.append(component)
    return components


def _merged_observation_payload(component: list[dict[str, Any]], *, class_name: str) -> dict[str, Any]:
    bboxes = [_bbox_tuple(item.get("bbox_xyxy")) for item in component]
    union_bbox = (
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    )
    closest = min(component, key=lambda item: float(item.get("target_xy_distance_m", float("inf"))))
    highest_confidence = max(float(item.get("confidence", 0.0)) for item in component)
    source_distances = [float(item.get("target_xy_distance_m", 0.0)) for item in component]
    return {
        "view_id": closest.get("view_id"),
        "image_path": closest.get("image_path"),
        "bbox_xyxy": [_round(value) for value in union_bbox],
        "confidence": _round(highest_confidence),
        "class_id": closest.get("class_id"),
        "class_name": class_name,
        "rough_position_world": closest.get("rough_position_world"),
        "target_xy_distance_m": _round(min(source_distances)),
        "observation_kind": "merged_same_class_bbox_fragments",
        "source_observation_count": len(component),
        "source_bboxes_xyxy": [item.get("bbox_xyxy") for item in component],
        "source_confidences": [_round(float(item.get("confidence", 0.0))) for item in component],
        "source_target_xy_distances_m": [_round(value) for value in source_distances],
        "merge_policy": {
            "policy_version": "final_same_class_bbox_fragment_merge_v1",
            "reason": "same view, same class, overlapping bbox fragments matched the same target",
        },
    }


def _final_observation_selection_score(
    target: FinalTarget,
    item: dict[str, Any],
    *,
    config: FinalConfig,
) -> tuple[float, ...]:
    class_name = _optional_text(item.get("class_name"))
    candidate_classes = set(target.candidate_class_names)
    class_match = (
        class_name is not None
        and (class_name == target.class_name or (target.class_name is None and class_name in candidate_classes))
    )
    bbox = _bbox_tuple(item.get("bbox_xyxy"))
    bbox_area = _bbox_area_px(bbox)
    margin = _bbox_border_margin_px(bbox, image_width=config.image_width, image_height=config.image_height)
    border_safe = margin >= config.selection_border_margin_px
    merged_count = int(item.get("source_observation_count", 1) or 1)
    distance = float(item.get("target_xy_distance_m", float("inf")))
    confidence = float(item.get("confidence", 0.0))
    item["selection_score"] = {
        "policy_version": "final_observation_selection_policy_v2",
        "class_match": class_match,
        "bbox_area_px": _round(bbox_area),
        "bbox_min_border_margin_px": _round(margin),
        "border_safe": border_safe,
        "source_observation_count": merged_count,
        "target_xy_distance_m": _round(distance),
        "confidence": _round(confidence),
        "rules": [
            "Prefer detections whose class matches the final target.",
            "Prefer bbox fragments merged from same-class overlapping detections when available.",
            "Prefer detections with enough image-border margin.",
            "Prefer larger bbox area before using target distance as a tie-breaker.",
        ],
    }
    return (
        1.0 if class_match else 0.0,
        float(merged_count),
        1.0 if border_safe else 0.0,
        bbox_area,
        -distance,
        confidence,
    )


def _capture_status(
    target: FinalTarget,
    view_result: dict[str, Any],
    matched: list[dict[str, Any]],
    recognition: dict[str, Any],
    *,
    run_yolo: bool,
) -> str:
    if view_result.get("status") != "success":
        return "capture_failed"
    if not run_yolo:
        return "captured_yolo_skipped"
    if not matched:
        return "unconfirmed"
    if target.source_status == "stable":
        class_match = recognition.get("class_match")
        if class_match is True:
            return "confirmed"
        if class_match is False:
            return "class_conflict"
        return "unconfirmed"
    return "follow_up_observed"


def _capture_notes(target: FinalTarget, status: str) -> list[str]:
    notes = [
        "Single-object capture used row report target positions as formal inputs and did not assume a fixed object count.",
        "The final photo was rendered only after top-opening entry samples and the final wrist-camera pose passed IK and collision checks.",
    ]
    if target.source_status != "stable":
        notes.append("This target came from tentative or ambiguous row evidence and remains a follow-up result, not a stable object promotion.")
    if status == "class_conflict":
        notes.append("The closest close-view YOLO observation did not match the source stable class name.")
    if status == "unconfirmed":
        notes.append("No close YOLO-depth observation confirmed this target within the configured association radius.")
    return notes


def _overall_status(object_captures: list[dict[str, Any]]) -> str:
    if not object_captures:
        return "failed"
    success_statuses = {"confirmed", "follow_up_observed", "captured_yolo_skipped"}
    if all(capture.get("status") in success_statuses for capture in object_captures):
        return "success"
    if any(capture.get("view", {}).get("status") == "success" for capture in object_captures):
        return "partial"
    return "failed"


def _summary_message(object_captures: list[dict[str, Any]], *, planned_count: int) -> str:
    counts: dict[str, int] = {}
    for capture in object_captures:
        status = str(capture.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    captured_count = sum(1 for capture in object_captures if capture.get("view", {}).get("status") == "success")
    count_text = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
    return f"Captured {captured_count}/{planned_count} final views; result_status_counts: {count_text}."


def _plan_payload(
    *,
    created_utc: str,
    row_report: Task1RowReport,
    workspace: SurveyWorkspace,
    config: FinalConfig,
    capture_dir: Path,
    plan_path: Path,
    report_path: Path,
    planned_captures: tuple[FinalPlannedCapture, ...],
) -> dict[str, Any]:
    return {
        "schema_version": "task1_final_plan_v1",
        "stage": "final",
        "created_utc": created_utc,
        "source_row_report_path": str(row_report.path),
        "source_row_status": row_report.status,
        "layout_snapshot_path": str(row_report.layout_snapshot_path),
        "task1_run_dir": str(row_report.task1_run_dir),
        "capture_dir": str(capture_dir),
        "plan_path": str(plan_path),
        "report_path": str(report_path),
        "scene_model_path": str(config.scene_model_path),
        "camera": {
            "name": config.camera_name,
            "source": "examples/mujoco/kinova_gen3/gen3.xml camera name='wrist'",
            "image_width": config.image_width,
            "image_height": config.image_height,
            "real_camera_note": "Real hardware is expected to use a 1080P depth camera.",
        },
        "constraints": {
            "camera_must_be_inside_tank": True,
            "camera_z_must_be_below_tank_opening": True,
            "entry_clearance_margin_m": _round(config.entry_clearance_margin_m),
            "final_view_not_top_down_only": True,
            "multiple_final_view_candidates": True,
            "multiple_final_view_standoff_distances": True,
            "entry_path": "discrete wrist-camera pose samples from the top opening down to each final photo pose",
            "pose_validation": "MuJoCo IK plus robot collision check for entry samples and final photo pose.",
            "row_report_is_layered_input": True,
        },
        "workspace": workspace.to_dict(),
        "target_policy": {
            "primary_objects": "row_report.stable_objects",
            "follow_up_targets": "row_report.tentative_objects + row_report.ambiguous_objects",
            "quality_source": "row_report.object_selection_summary",
            "fixed_object_count_assumption": False,
        },
        "planned_captures": [planned.to_dict() for planned in planned_captures],
    }


def _report_payload(
    *,
    created_utc: str,
    status: str,
    message: str,
    row_report: Task1RowReport,
    workspace: SurveyWorkspace,
    config: FinalConfig,
    capture_dir: Path,
    plan_path: Path,
    report_path: Path,
    planned_captures: tuple[FinalPlannedCapture, ...],
    object_captures: list[dict[str, Any]],
) -> dict[str, Any]:
    primary_targets = [planned.target.to_dict() for planned in planned_captures if planned.target.source_status == "stable"]
    follow_up_targets = [planned.target.to_dict() for planned in planned_captures if planned.target.source_status != "stable"]
    stable_selection = select_stable_final_objects(object_captures)
    return {
        "schema_version": "task1_final_report_v1",
        "stage": "final",
        "status": status,
        "created_utc": created_utc,
        "message": message,
        "source_row_report_path": str(row_report.path),
        "source_row_status": row_report.status,
        "layout_snapshot_path": str(row_report.layout_snapshot_path),
        "task1_run_dir": str(row_report.task1_run_dir),
        "capture_dir": str(capture_dir),
        "plan_path": str(plan_path),
        "report_path": str(report_path),
        "scene_model_path": str(config.scene_model_path),
        "camera_name": config.camera_name,
        "workspace": workspace.to_dict(),
        "image_size": [config.image_width, config.image_height],
        "run_yolo": config.run_yolo,
        "yolo_profile_path": str(config.yolo_profile_path),
        "final_config": _config_payload(config),
        "primary_objects": primary_targets,
        "follow_up_targets": follow_up_targets,
        "stable_objects": stable_selection["stable_objects"],
        "unstable_objects": stable_selection["unstable_objects"],
        "stable_object_selection": stable_selection["stable_object_selection"],
        "quality": row_report.object_selection_summary,
        "object_selection_summary": row_report.object_selection_summary,
        "planned_captures": [planned.to_dict() for planned in planned_captures],
        "object_captures": object_captures,
        "notes": [
            "Stable row objects are consumed as primary final capture targets.",
            "Tentative and ambiguous row objects are consumed as explicit follow-up targets and can enter stable_objects only after close-capture resolution.",
            "Each target keeps its own capture status; unconfirmed and class-conflict results are kept out of stable_objects.",
            "The report intentionally supports any number of row targets.",
        ],
    }


def _config_payload(config: FinalConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def _planned_view_result(view: SurveyView) -> dict[str, Any]:
    payload = view.to_dict()
    payload["status"] = "planned"
    return payload


def _workspace_from_payload(value: Any) -> SurveyWorkspace:
    if not isinstance(value, dict):
        raise ValueError("workspace must be a dict")
    return SurveyWorkspace(
        x_min=float(value["x_min"]),
        x_max=float(value["x_max"]),
        y_min=float(value["y_min"]),
        y_max=float(value["y_max"]),
        bottom_z_m=float(value["bottom_z_m"]),
        tank_opening_z_m=float(value.get("tank_opening_z_m", DEFAULT_TANK_OPENING_Z_M)),
        opening_clearance_m=float(value.get("opening_clearance_m", DEFAULT_OPENING_CLEARANCE_M)),
    )


def _views_by_id(views_payload: list[Any]) -> dict[str, dict[str, Any]]:
    views: dict[str, dict[str, Any]] = {}
    for view in views_payload:
        if not isinstance(view, dict):
            continue
        view_id = view.get("view_id")
        if isinstance(view_id, str) and view_id:
            views[view_id] = view
    return views


def _view_camera_position(view_payload: dict[str, Any]) -> tuple[float, float, float] | None:
    for key in ("actual_camera_position_world", "desired_camera_position_world"):
        value = view_payload.get(key)
        if isinstance(value, list) and len(value) == 3:
            return (float(value[0]), float(value[1]), float(value[2]))
    return None


def _image_size_from_payload(value: Any) -> tuple[int, int]:
    if isinstance(value, list) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    return (DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT)


def _required(payload: dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError(f"missing required field: {key}")
    return payload[key]


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = _required(payload, key)
    if value is None or not str(value).strip():
        raise ValueError(f"field {key} must be a non-empty string")
    return str(value)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _required_path(payload: dict[str, Any], key: str) -> Path:
    value = _required(payload, key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"field {key} must be a non-empty path string")
    return Path(value)


def _position_from_value(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("position_world must be a list of 3 numbers")
    return (float(value[0]), float(value[1]), float(value[2]))


def _optional_position_from_value(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    return (float(value[0]), float(value[1]), float(value[2]))


def _optional_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("best_bbox_xyxy must be a list of 4 numbers when present")
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def _transform_from_value(value: Any) -> tuple[tuple[float, float, float, float], ...]:
    transform = _optional_transform(value)
    if transform is None:
        raise ValueError("T_world_object must be a 4x4 numeric matrix")
    return transform


def _optional_transform(value: Any) -> tuple[tuple[float, float, float, float], ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        return None
    rows: list[tuple[float, float, float, float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            return None
        rows.append((float(row[0]), float(row[1]), float(row[2]), float(row[3])))
    return tuple(rows)


def _transform_with_position(
    source_transform: tuple[tuple[float, float, float, float], ...] | None,
    position: tuple[float, float, float],
) -> tuple[tuple[float, float, float, float], ...]:
    if source_transform is None:
        return (
            (1.0, 0.0, 0.0, _round(position[0])),
            (0.0, 1.0, 0.0, _round(position[1])),
            (0.0, 0.0, 1.0, _round(position[2])),
            (0.0, 0.0, 0.0, 1.0),
        )
    return (
        (source_transform[0][0], source_transform[0][1], source_transform[0][2], _round(position[0])),
        (source_transform[1][0], source_transform[1][1], source_transform[1][2], _round(position[1])),
        (source_transform[2][0], source_transform[2][1], source_transform[2][2], _round(position[2])),
        (0.0, 0.0, 0.0, 1.0),
    )


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value if item is not None and str(item))


def _unique_texts(values: Any) -> tuple[str, ...]:
    unique: list[str] = []
    for value in values:
        text = _optional_text(value)
        if text and text not in unique:
            unique.append(text)
    return tuple(unique)


def _validate_angle_offsets(values: tuple[float, ...], *, field_name: str) -> None:
    for value in values:
        if not math.isfinite(float(value)):
            raise ValueError(f"{field_name} must contain only finite numbers")


def _validate_positive_multipliers(values: tuple[float, ...], *, field_name: str) -> None:
    for value in values:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(f"{field_name} must contain only finite positive numbers")


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_id(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)
    return safe or "target"


def _make_look_at_transform(
    camera_position: tuple[float, float, float],
    look_at: tuple[float, float, float],
) -> tuple[tuple[float, float, float, float], ...]:
    forward = _normalize(
        (
            look_at[0] - camera_position[0],
            look_at[1] - camera_position[1],
            look_at[2] - camera_position[2],
        )
    )
    world_up = (0.0, 1.0, 0.0)
    if abs(_dot(forward, world_up)) > 0.95:
        world_up = (1.0, 0.0, 0.0)
    right = _normalize(_cross(forward, world_up))
    up = _normalize(_cross(right, forward))
    camera_z = (-forward[0], -forward[1], -forward[2])
    return (
        (_round(right[0]), _round(up[0]), _round(camera_z[0]), _round(camera_position[0])),
        (_round(right[1]), _round(up[1]), _round(camera_z[1]), _round(camera_position[1])),
        (_round(right[2]), _round(up[2]), _round(camera_z[2]), _round(camera_position[2])),
        (0.0, 0.0, 0.0, 1.0),
    )


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(component * component for component in vector))
    if norm <= 1e-12:
        raise ValueError("cannot normalize zero vector")
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def _cross(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _unique_angles(values: tuple[float, ...]) -> tuple[float, ...]:
    unique: list[float] = []
    for value in values:
        normalized = _normalize_angle(value)
        if all(abs(_normalize_angle(normalized - existing)) > 1e-6 for existing in unique):
            unique.append(normalized)
    return tuple(unique)


def _unique_angle_records(records: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    unique: list[dict[str, Any]] = []
    for record in records:
        angle = _normalize_angle(float(record["angle_rad"]))
        if all(abs(_normalize_angle(angle - float(existing["angle_rad"]))) > 1e-6 for existing in unique):
            copied = dict(record)
            copied["angle_rad"] = angle
            unique.append(copied)
    return tuple(unique)


def _unique_standoff_records(config: FinalConfig) -> tuple[dict[str, float], ...]:
    unique: list[dict[str, float]] = []
    for multiplier in config.final_view_standoff_multipliers:
        numeric_multiplier = float(multiplier)
        standoff_m = config.standoff_m * numeric_multiplier
        if any(abs(standoff_m - existing["standoff_m"]) <= 1e-6 for existing in unique):
            continue
        unique.append(
            {
                "standoff_m": standoff_m,
                "standoff_multiplier": numeric_multiplier,
            }
        )
    return tuple(unique)


def _xy_distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _round_vector(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    return (_round(vector[0]), _round(vector[1]), _round(vector[2]))


def _bbox_tuple(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("bbox_xyxy must be a list of four numbers")
    x1, y1, x2, y2 = (float(item) for item in value)
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def _bbox_area_px(bbox_xyxy: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox_xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_overlap_ratio(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    left_area = _bbox_area_px(left)
    right_area = _bbox_area_px(right)
    if left_area <= 0.0 or right_area <= 0.0:
        return 0.0
    inter_left = max(left[0], right[0])
    inter_top = max(left[1], right[1])
    inter_right = min(left[2], right[2])
    inter_bottom = min(left[3], right[3])
    inter_area = _bbox_area_px((inter_left, inter_top, inter_right, inter_bottom))
    return inter_area / min(left_area, right_area)


def _bbox_border_margin_px(
    bbox_xyxy: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
) -> float:
    x1, y1, x2, y2 = bbox_xyxy
    return min(x1, y1, float(image_width) - x2, float(image_height) - y2)


def _round(value: float | None) -> float:
    if value is None:
        return 0.0
    return round(float(value), 6)


def _normalize_angle(angle: float) -> float:
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    while angle > math.pi:
        angle -= 2.0 * math.pi
    return angle


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
