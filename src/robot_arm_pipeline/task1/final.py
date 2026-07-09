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
    DEFAULT_SAVE_RAW_YOLO_ANNOTATIONS,
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
DEFAULT_FINAL_ENTRY_CLEARANCE_MARGIN_M = 0.04
DEFAULT_FINAL_ENTRY_PORTAL_MODES = ("final-vertical", "opening-grid-nearest", "workspace-center")
DEFAULT_FINAL_ENTRY_ORIENTATION_POLICY = "vertical-descent"
DEFAULT_FINAL_ENTRY_LATERAL_ORIENTATION_POLICY = "final-look-at"
DEFAULT_FINAL_ENTRY_PATH_POLICY = "portal-descent-then-lateral"
DEFAULT_FINAL_IK_POSITION_TOLERANCE_M = 0.005
DEFAULT_FINAL_VIEW_ANGLE_OFFSETS_DEG = (
    0.0,
    22.5,
    -22.5,
    45.0,
    -45.0,
    67.5,
    -67.5,
    90.0,
    -90.0,
    135.0,
    -135.0,
    180.0,
)
DEFAULT_FINAL_VIEW_STANDOFF_MULTIPLIERS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
DEFAULT_FINAL_VIEW_CAMERA_Z_OFFSETS_M = (0.0, -0.04, -0.08, -0.12, 0.04, 0.08, 0.12)
DEFAULT_FINAL_VIEW_ROLL_OFFSETS_DEG = (0.0, 45.0, -45.0, 90.0, -90.0, 135.0, -135.0, 180.0)
DEFAULT_FINAL_CENTERLINE_VIEW_ANGLE_OFFSETS_DEG = (0.0, 45.0, -45.0)
DEFAULT_FINAL_VIEW_CAMERA_POSITION_LIMIT = None
DEFAULT_FINAL_VIEW_CANDIDATE_LIMIT = None
DEFAULT_FINAL_VIEW_PRIMARY_CANDIDATE_COUNT = 24
DEFAULT_FINAL_ENTRY_SIDE = "y-max"
DEFAULT_FINAL_YOLO_CONFIDENCE = 0.20
DEFAULT_FINAL_YOLO_MAX_DETECTIONS = 12
DEFAULT_FINAL_YOLO_TILE_GRID_SIZE = 1
FINAL_REACHABLE_CAPTURE_FIXED_POSE_SOURCE = "final_reachable_capture_plan_v1"
FINAL_STABLE_SELECTION_POLICY_VERSION = "final_stable_selection_policy_v1"
FINAL_BBOX_QUALITY_POLICY_VERSION = "final_bbox_quality_policy_v1"
FINAL_VIEW_CANDIDATE_PRUNING_POLICY_VERSION = "final_view_candidate_pruning_policy_v3"
DEFAULT_FINAL_BBOX_FRAGMENT_MIN_OVERLAP_RATIO = 0.15
DEFAULT_FINAL_SELECTION_BORDER_MARGIN_PX = 8.0
DEFAULT_FINAL_FOLLOW_UP_DUPLICATE_RADIUS_M = DEFAULT_FINAL_TARGET_MATCH_RADIUS_M
DEFAULT_FINAL_FOLLOW_UP_PROMOTION_MIN_CONFIDENCE = DEFAULT_FINAL_YOLO_CONFIDENCE
DEFAULT_FINAL_SINGLE_OBSERVATION_MIN_CONFIDENCE = 0.60
DEFAULT_FINAL_DESIRED_STABLE_OBJECT_COUNT = 5
DEFAULT_SAVE_FINAL_FILTERED_ANNOTATIONS = True
DEFAULT_SAVE_FINAL_DEBUG_TRACE = False
DEFAULT_FINAL_FAILURE_SAMPLE_LIMIT = 5


@dataclass(frozen=True)
class Task1RoughReport:
    path: Path
    schema_version: str
    status: str
    layout_snapshot_path: Path
    task1_run_dir: Path
    rough_dir: Path
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
    entry_portal_modes: tuple[str, ...] = DEFAULT_FINAL_ENTRY_PORTAL_MODES
    entry_orientation_policy: str = DEFAULT_FINAL_ENTRY_ORIENTATION_POLICY
    entry_lateral_orientation_policy: str = DEFAULT_FINAL_ENTRY_LATERAL_ORIENTATION_POLICY
    entry_path_policy: str = DEFAULT_FINAL_ENTRY_PATH_POLICY
    desired_stable_object_count: int | None = DEFAULT_FINAL_DESIRED_STABLE_OBJECT_COUNT
    final_view_angle_offsets_deg: tuple[float, ...] = DEFAULT_FINAL_VIEW_ANGLE_OFFSETS_DEG
    final_view_standoff_multipliers: tuple[float, ...] = DEFAULT_FINAL_VIEW_STANDOFF_MULTIPLIERS
    final_view_camera_z_offsets_m: tuple[float, ...] = DEFAULT_FINAL_VIEW_CAMERA_Z_OFFSETS_M
    final_view_roll_offsets_deg: tuple[float, ...] = DEFAULT_FINAL_VIEW_ROLL_OFFSETS_DEG
    centerline_view_angle_offsets_deg: tuple[float, ...] = DEFAULT_FINAL_CENTERLINE_VIEW_ANGLE_OFFSETS_DEG
    final_view_camera_position_limit: int | None = DEFAULT_FINAL_VIEW_CAMERA_POSITION_LIMIT
    final_view_candidate_limit: int | None = DEFAULT_FINAL_VIEW_CANDIDATE_LIMIT
    final_view_primary_candidate_count: int = DEFAULT_FINAL_VIEW_PRIMARY_CANDIDATE_COUNT
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
    single_observation_min_confidence: float = DEFAULT_FINAL_SINGLE_OBSERVATION_MIN_CONFIDENCE
    save_filtered_annotations: bool = DEFAULT_SAVE_FINAL_FILTERED_ANNOTATIONS
    save_raw_yolo_annotations: bool = DEFAULT_SAVE_RAW_YOLO_ANNOTATIONS
    save_debug_trace: bool = DEFAULT_SAVE_FINAL_DEBUG_TRACE
    failure_sample_limit: int = DEFAULT_FINAL_FAILURE_SAMPLE_LIMIT
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
        if not self.entry_portal_modes:
            raise ValueError("entry_portal_modes must contain at least one mode")
        unsupported_entry_portal_modes = set(self.entry_portal_modes) - {
            "final-vertical",
            "workspace-center",
            "opening-grid-nearest",
        }
        if unsupported_entry_portal_modes:
            raise ValueError("entry_portal_modes values must be final-vertical, workspace-center, or opening-grid-nearest")
        if self.entry_orientation_policy not in {"vertical-descent", "target-look-at"}:
            raise ValueError("entry_orientation_policy must be vertical-descent or target-look-at")
        if self.entry_lateral_orientation_policy not in {"entry-orientation", "final-look-at"}:
            raise ValueError("entry_lateral_orientation_policy must be entry-orientation or final-look-at")
        if self.entry_path_policy not in {"portal-descent-then-lateral", "direct-interpolate"}:
            raise ValueError("entry_path_policy must be portal-descent-then-lateral or direct-interpolate")
        if self.desired_stable_object_count is not None and self.desired_stable_object_count <= 0:
            raise ValueError("desired_stable_object_count must be positive when provided")
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
        if not self.final_view_camera_z_offsets_m:
            raise ValueError("final_view_camera_z_offsets_m must contain at least one offset")
        if not self.final_view_roll_offsets_deg:
            raise ValueError("final_view_roll_offsets_deg must contain at least one angle")
        _validate_angle_offsets(self.final_view_angle_offsets_deg, field_name="final_view_angle_offsets_deg")
        _validate_angle_offsets(self.final_view_camera_z_offsets_m, field_name="final_view_camera_z_offsets_m")
        _validate_angle_offsets(self.final_view_roll_offsets_deg, field_name="final_view_roll_offsets_deg")
        _validate_angle_offsets(self.centerline_view_angle_offsets_deg, field_name="centerline_view_angle_offsets_deg")
        _validate_positive_multipliers(
            self.final_view_standoff_multipliers,
            field_name="final_view_standoff_multipliers",
        )
        if self.final_view_camera_position_limit is not None and self.final_view_camera_position_limit <= 0:
            raise ValueError("final_view_camera_position_limit must be positive when provided")
        if self.final_view_candidate_limit is not None and self.final_view_candidate_limit <= 0:
            raise ValueError("final_view_candidate_limit must be positive when provided")
        if self.final_view_primary_candidate_count <= 0:
            raise ValueError("final_view_primary_candidate_count must be positive")
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
        if not 0.0 <= self.single_observation_min_confidence <= 1.0:
            raise ValueError("single_observation_min_confidence must be between 0 and 1")
        if self.failure_sample_limit < 0:
            raise ValueError("failure_sample_limit must be non-negative")

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
            save_raw_yolo_annotations=self.save_raw_yolo_annotations,
            save_depth_arrays=True,
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
    camera_z_offset_m: float
    roll_offset_deg: float
    entry_portal_mode: str
    candidate_source: str
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
            "camera_z_offset_m": _round(self.camera_z_offset_m),
            "roll_offset_deg": _round(self.roll_offset_deg),
            "entry_portal_mode": self.entry_portal_mode,
            "candidate_source": self.candidate_source,
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
    candidate_generation_summary: dict[str, Any]

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
            "candidate_generation_summary": self.candidate_generation_summary,
        }


def find_latest_rough_report(output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    root = Path(output_dir)
    candidates = sorted(root.glob("*/rough/rough_report.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no task1 rough_report.json files found under {root}")
    return candidates[-1]


def load_task1_rough_report(path: Path | str) -> Task1RoughReport:
    report_path = Path(path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "task1_rough_report_v1":
        raise ValueError(f"unsupported task1 rough report schema_version: {payload.get('schema_version')}")
    workspace = _workspace_from_payload(_required(payload, "workspace"))
    views_payload = payload.get("views", [])
    if not isinstance(views_payload, list):
        raise ValueError("task1 rough report views must be a list")
    stable_objects = payload.get("stable_objects")
    tentative_objects = payload.get("tentative_objects")
    ambiguous_objects = payload.get("ambiguous_objects")
    if not isinstance(stable_objects, list):
        raise ValueError("task1 rough report must contain stable_objects list")
    if not isinstance(tentative_objects, list):
        raise ValueError("task1 rough report must contain tentative_objects list")
    if not isinstance(ambiguous_objects, list):
        raise ValueError("task1 rough report must contain ambiguous_objects list")
    object_selection_summary = payload.get("object_selection_summary", {})
    if not isinstance(object_selection_summary, dict):
        raise ValueError("task1 rough report object_selection_summary must be a dict when present")
    return Task1RoughReport(
        path=report_path,
        schema_version=str(payload["schema_version"]),
        status=str(payload.get("status", "")),
        layout_snapshot_path=_required_path(payload, "layout_snapshot_path"),
        task1_run_dir=Path(str(payload.get("task1_run_dir") or report_path.parent.parent)),
        rough_dir=Path(str(payload.get("rough_dir") or report_path.parent)),
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
    rough_report: Task1RoughReport | Path | str,
    config: FinalConfig = FinalConfig(),
) -> tuple[SurveyWorkspace, tuple[FinalPlannedCapture, ...]]:
    config.validate()
    report = load_task1_rough_report(rough_report) if not isinstance(rough_report, Task1RoughReport) else rough_report
    workspace = report.workspace
    if config.camera_z_m <= workspace.bottom_z_m:
        raise ValueError(
            "final wrist camera z must be above the tank bottom/object base plane: "
            f"camera_z={config.camera_z_m:.4f}, bottom_z={workspace.bottom_z_m:.4f}"
        )
    targets = _targets_from_rough_report(report)
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
        view_candidates, view_candidate_generation_summary = _view_candidates_for_target(
            target_index=target_index,
            view_id_prefix=view_id,
            camera_candidates=camera_candidates,
            preferred_angle=preferred_angle,
            look_at=look_at,
            workspace=workspace,
            config=config,
        )
        primary_candidate = view_candidates[0]
        candidate_generation_summary = _candidate_generation_summary(
            evaluated_camera_positions=evaluated,
            retained_camera_candidates=camera_candidates,
            view_candidate_summary=view_candidate_generation_summary,
            config=config,
        )
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
                candidate_generation_summary=candidate_generation_summary,
            )
        )
    return workspace, tuple(planned)


def run_task1_final(
    rough_report_path: Path | str,
    config: FinalConfig = FinalConfig(),
    *,
    detector: SurveyDetector | None = None,
) -> dict[str, Any]:
    config.validate()
    rough_report = load_task1_rough_report(rough_report_path)
    workspace, planned_captures = build_final_plan(rough_report, config)
    created_utc = datetime.now(timezone.utc).isoformat()
    capture_dir = rough_report.task1_run_dir / "final"
    images_dir = capture_dir / "images"
    depth_dir = capture_dir / "depth"
    yolo_dir = capture_dir / "yolo_raw"
    annotated_dir = capture_dir / "annotated"
    raw_annotated_dir = capture_dir / "raw_annotated"
    tiles_dir = capture_dir / "tiles"
    plan_path = capture_dir / "final_plan.json"
    report_path = capture_dir / "final_report.json"
    debug_trace_path = capture_dir / "final_debug_trace.json"
    plan_payload = _plan_payload(
        created_utc=created_utc,
        rough_report=rough_report,
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
            message="No stable, tentative, or ambiguous rough targets were available for final capture.",
            rough_report=rough_report,
            workspace=workspace,
            config=config,
            capture_dir=capture_dir,
            plan_path=plan_path,
            report_path=report_path,
            planned_captures=planned_captures,
            object_captures=[],
        )
        _write_final_outputs(
            plan_path=plan_path,
            report_path=report_path,
            debug_trace_path=debug_trace_path,
            plan_payload=plan_payload,
            report=report,
            created_utc=created_utc,
            rough_report=rough_report,
            config=config,
            capture_dir=capture_dir,
            planned_captures=planned_captures,
            object_captures=[],
        )
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
            rough_report=rough_report,
            workspace=workspace,
            config=config,
            capture_dir=capture_dir,
            plan_path=plan_path,
            report_path=report_path,
            planned_captures=planned_captures,
            object_captures=object_captures,
        )
        _write_final_outputs(
            plan_path=plan_path,
            report_path=report_path,
            debug_trace_path=debug_trace_path,
            plan_payload=plan_payload,
            report=report,
            created_utc=created_utc,
            rough_report=rough_report,
            config=config,
            capture_dir=capture_dir,
            planned_captures=planned_captures,
            object_captures=object_captures,
        )
        return report

    if config.run_yolo and detector is None:
        object_captures = [_planned_capture_result(planned) for planned in planned_captures]
        report = _report_payload(
            created_utc=created_utc,
            status="failed",
            message="run_yolo=True requires a SurveyDetector; pass one from the task1 entrypoint or set run_yolo=False.",
            rough_report=rough_report,
            workspace=workspace,
            config=config,
            capture_dir=capture_dir,
            plan_path=plan_path,
            report_path=report_path,
            planned_captures=planned_captures,
            object_captures=object_captures,
        )
        _write_final_outputs(
            plan_path=plan_path,
            report_path=report_path,
            debug_trace_path=debug_trace_path,
            plan_payload=plan_payload,
            report=report,
            created_utc=created_utc,
            rough_report=rough_report,
            config=config,
            capture_dir=capture_dir,
            planned_captures=planned_captures,
            object_captures=object_captures,
        )
        return report

    layout = load_stage0_layout(rough_report.layout_snapshot_path)
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
        primary_plans = tuple(planned for planned in planned_captures if planned.target.source_status == "stable")
        follow_up_plans = tuple(planned for planned in planned_captures if planned.target.source_status != "stable")

        for planned in primary_plans:
            capture_result = _capture_first_reachable_candidate(
                backend=backend,
                planned=planned,
                images_dir=images_dir,
                depth_dir=depth_dir,
                yolo_dir=yolo_dir,
                annotated_dir=annotated_dir,
                raw_annotated_dir=raw_annotated_dir,
                tiles_dir=tiles_dir,
                config=config,
            )
            object_captures.append(capture_result)

        for planned in follow_up_plans:
            if _desired_stable_count_reached(
                object_captures,
                desired_stable_object_count=config.desired_stable_object_count,
                single_observation_min_confidence=config.single_observation_min_confidence,
            ):
                object_captures.append(_skipped_follow_up_capture_result(planned))
                continue
            capture_result = _capture_first_reachable_candidate(
                backend=backend,
                planned=planned,
                images_dir=images_dir,
                depth_dir=depth_dir,
                yolo_dir=yolo_dir,
                annotated_dir=annotated_dir,
                raw_annotated_dir=raw_annotated_dir,
                tiles_dir=tiles_dir,
                config=config,
            )
            object_captures.append(capture_result)
    except Exception as exc:
        report = _report_payload(
            created_utc=created_utc,
            status="failed",
            message=str(exc),
            rough_report=rough_report,
            workspace=workspace,
            config=config,
            capture_dir=capture_dir,
            plan_path=plan_path,
            report_path=report_path,
            planned_captures=planned_captures,
            object_captures=object_captures,
        )
        _write_final_outputs(
            plan_path=plan_path,
            report_path=report_path,
            debug_trace_path=debug_trace_path,
            plan_payload=plan_payload,
            report=report,
            created_utc=created_utc,
            rough_report=rough_report,
            config=config,
            capture_dir=capture_dir,
            planned_captures=planned_captures,
            object_captures=object_captures,
        )
        return report
    finally:
        backend.close()

    status = _overall_status(object_captures)
    message = _summary_message(object_captures, planned_count=len(planned_captures))
    report = _report_payload(
        created_utc=created_utc,
        status=status,
        message=message,
        rough_report=rough_report,
        workspace=workspace,
        config=config,
        capture_dir=capture_dir,
        plan_path=plan_path,
        report_path=report_path,
        planned_captures=planned_captures,
        object_captures=object_captures,
    )
    _write_final_outputs(
        plan_path=plan_path,
        report_path=report_path,
        debug_trace_path=debug_trace_path,
        plan_payload=plan_payload,
        report=report,
        created_utc=created_utc,
        rough_report=rough_report,
        config=config,
        capture_dir=capture_dir,
        planned_captures=planned_captures,
        object_captures=object_captures,
    )
    return report


def select_stable_final_objects(
    object_captures: list[dict[str, Any]],
    *,
    desired_stable_object_count: int | None = DEFAULT_FINAL_DESIRED_STABLE_OBJECT_COUNT,
    single_observation_min_confidence: float = DEFAULT_FINAL_SINGLE_OBSERVATION_MIN_CONFIDENCE,
) -> dict[str, Any]:
    if not object_captures:
        return _stable_selection_payload(
            [],
            [],
            status="not_run",
            desired_stable_object_count=desired_stable_object_count,
            single_observation_min_confidence=single_observation_min_confidence,
        )
    if all(capture.get("status") == "planned" for capture in object_captures):
        return _stable_selection_payload(
            [],
            [],
            status="not_run",
            desired_stable_object_count=desired_stable_object_count,
            single_observation_min_confidence=single_observation_min_confidence,
        )

    primary_objects: list[dict[str, Any]] = []
    follow_up_objects: list[tuple[dict[str, Any], dict[str, Any]]] = []
    rejected_objects: list[dict[str, Any]] = []
    for capture in object_captures:
        stable_object, rejection = _stable_object_from_capture(capture)
        if stable_object is not None:
            if stable_object.get("source_status") == "stable":
                primary_objects.append(stable_object)
            else:
                follow_up_objects.append((stable_object, capture))
        elif rejection is not None:
            rejected_objects.append(rejection)

    stable_objects, overcomplete_rejections = _split_overcomplete_primary_objects(
        primary_objects,
        desired_stable_object_count=desired_stable_object_count,
        single_observation_min_confidence=single_observation_min_confidence,
    )
    rejected_objects.extend(overcomplete_rejections)
    for stable_object, capture in follow_up_objects:
        if desired_stable_object_count is not None and len(stable_objects) >= desired_stable_object_count:
            rejected_objects.append(
                _follow_up_not_needed_rejection_payload(
                    capture,
                    desired_stable_object_count=desired_stable_object_count,
                    stable_object_count=len(stable_objects),
                )
            )
            continue
        duplicate = _matching_stable_object(
            stable_object,
            stable_objects,
            max_distance_m=DEFAULT_FINAL_FOLLOW_UP_DUPLICATE_RADIUS_M,
        )
        if duplicate is not None:
            rejected_objects.append(
                _duplicate_follow_up_rejection_payload(
                    capture,
                    stable_object=stable_object,
                    duplicate=duplicate,
                    max_distance_m=DEFAULT_FINAL_FOLLOW_UP_DUPLICATE_RADIUS_M,
                )
            )
            continue
        stable_objects.append(stable_object)
    return _stable_selection_payload(
        stable_objects,
        rejected_objects,
        status="success",
        desired_stable_object_count=desired_stable_object_count,
        single_observation_min_confidence=single_observation_min_confidence,
    )


def _capture_first_reachable_candidate(
    *,
    backend: MujocoSurveyBackend,
    planned: FinalPlannedCapture,
    images_dir: Path,
    depth_dir: Path,
    yolo_dir: Path,
    annotated_dir: Path,
    raw_annotated_dir: Path,
    tiles_dir: Path,
    config: FinalConfig,
) -> dict[str, Any]:
    candidate_attempts: list[dict[str, Any]] = []
    best_captured_result: dict[str, Any] | None = None
    for candidate_index, candidate in enumerate(planned.view_candidates):
        search_phase = _final_candidate_search_phase(candidate_index, config=config)
        entry_results = _validate_entry_views(backend, planned, candidate)
        failed_entry = next((entry for entry in entry_results if entry.get("status") != "success"), None)
        if failed_entry is not None:
            candidate_attempts.append(
                _candidate_attempt_payload(
                    candidate=candidate,
                    entry_results=entry_results,
                    view_result=_planned_view_result(candidate.view),
                    final_pose_validation=None,
                    status="entry_validation_failed",
                    message=str(failed_entry.get("message") or "entry validation failed"),
                    search_phase=search_phase,
                )
            )
            continue

        final_pose_validation = _validate_final_view_candidate(backend, candidate)
        if final_pose_validation.get("status") != "success":
            candidate_attempts.append(
                _candidate_attempt_payload(
                    candidate=candidate,
                    entry_results=entry_results,
                    view_result=_planned_view_result(candidate.view),
                    final_pose_validation=final_pose_validation,
                    status="final_pose_failed",
                    message=str(final_pose_validation.get("message") or "final pose validation failed"),
                    search_phase=search_phase,
                )
            )
            continue

        actual_qpos = _actual_qpos_from_validation(final_pose_validation)
        if actual_qpos is None:
            final_pose_validation = dict(final_pose_validation)
            final_pose_validation["status"] = "failed"
            final_pose_validation["message"] = "Validated final candidate did not include actual_qpos for reproducible capture."
            candidate_attempts.append(
                _candidate_attempt_payload(
                    candidate=candidate,
                    entry_results=entry_results,
                    view_result=_planned_view_result(candidate.view),
                    final_pose_validation=final_pose_validation,
                    status="final_pose_failed",
                    message=str(final_pose_validation["message"]),
                    search_phase=search_phase,
                )
            )
            continue

        selected_candidate = _copy_final_view_candidate_with_fixed_qpos(candidate, fixed_qpos=actual_qpos)
        view_result, observations = backend.capture_view(
            selected_candidate.view,
            images_dir=images_dir,
            depth_dir=depth_dir,
            yolo_dir=yolo_dir,
            annotated_dir=annotated_dir,
            raw_annotated_dir=raw_annotated_dir,
            tiles_dir=tiles_dir,
        )
        attempt = _candidate_attempt_payload(
            candidate=selected_candidate,
            entry_results=entry_results,
            view_result=view_result,
            final_pose_validation=final_pose_validation,
            status="captured" if view_result.get("status") == "success" else "final_pose_failed",
            message=str(view_result.get("message") or "captured"),
            search_phase=search_phase,
        )
        candidate_attempts.append(attempt)
        if view_result.get("status") != "success":
            continue

        capture_result = _object_capture_result(
            planned=planned,
            selected_candidate=selected_candidate,
            candidate_attempts=candidate_attempts,
            entry_results=entry_results,
            view_result=view_result,
            observations=observations,
            config=config,
        )
        if config.save_filtered_annotations:
            _write_final_filtered_annotation(
                capture_result=capture_result,
                annotated_dir=annotated_dir,
                config=config,
            )
        attempt["capture_status"] = capture_result["status"]
        attempt["recognition"] = capture_result["recognition"]
        if _capture_result_satisfies_target(capture_result):
            return capture_result
        attempt["status"] = f"captured_{capture_result['status']}"
        attempt["message"] = _continue_after_capture_message(capture_result)
        if best_captured_result is None or _capture_result_fallback_score(capture_result) > _capture_result_fallback_score(
            best_captured_result
        ):
            best_captured_result = capture_result

    if best_captured_result is not None:
        best_captured_result["view_candidate_attempts"] = candidate_attempts
        best_captured_result["notes"] = list(best_captured_result.get("notes", [])) + [
            "No later collision-safe final candidate confirmed this target; this is the best captured but unresolved result.",
        ]
        return best_captured_result
    return _all_candidates_failed_capture_result(planned, candidate_attempts)


def _candidate_attempt_payload(
    *,
    candidate: FinalViewCandidate,
    entry_results: list[dict[str, Any]],
    view_result: dict[str, Any],
    final_pose_validation: dict[str, Any] | None,
    status: str,
    message: str,
    search_phase: str,
) -> dict[str, Any]:
    return {
        "candidate": candidate.to_dict(),
        "status": status,
        "message": message,
        "search_phase": search_phase,
        "entry_validation": entry_results,
        "final_pose_validation": final_pose_validation,
        "view": view_result,
    }


def _final_candidate_search_phase(candidate_index: int, *, config: FinalConfig) -> str:
    return "primary" if candidate_index < config.final_view_primary_candidate_count else "extended"


def _targets_from_rough_report(report: Task1RoughReport) -> tuple[FinalTarget, ...]:
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
        supporting_views=_string_tuple(payload.get("supporting_rough_views", [])),
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
        supporting_views=_string_tuple(payload.get("supporting_rough_views", [])),
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
        for view_id in candidate.get("supporting_rough_views", [])
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
    report: Task1RoughReport,
    *,
    config: FinalConfig,
) -> tuple[float, str]:
    if target.best_image_path:
        view_payload = _view_for_image_path(target.best_image_path, report.views)
        angle = _angle_from_view_to_target(view_payload, target.position_world) if view_payload is not None else None
        if angle is not None:
            return angle, "best_rough_image_view"

    for view_id in target.supporting_views:
        view_payload = report.views_by_id.get(view_id)
        angle = _angle_from_view_to_target(view_payload, target.position_world) if view_payload is not None else None
        if angle is not None:
            return angle, "supporting_rough_view"

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
    return _angle_from_camera_position_to_target(camera_position, target)


def _angle_from_camera_position_to_target(
    camera_position: tuple[float, float, float],
    target: tuple[float, float, float],
) -> float:
    dx = camera_position[0] - target[0]
    dy = camera_position[1] - target[1]
    if math.hypot(dx, dy) <= 1e-9:
        return 0.0
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
                "angle_source": "rough_evidence_offset",
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
    z_offset_records = _unique_float_records(config.final_view_camera_z_offsets_m)
    evaluated: list[dict[str, Any]] = []
    valid_candidates: list[dict[str, Any]] = []
    for standoff_record in standoff_records:
        standoff_m = float(standoff_record["standoff_m"])
        standoff_multiplier = float(standoff_record["standoff_multiplier"])
        for z_offset in z_offset_records:
            camera_z = config.camera_z_m + float(z_offset)
            for record in angle_records:
                angle = float(record["angle_rad"])
                position = (
                    target[0] + math.cos(angle) * standoff_m,
                    target[1] + math.sin(angle) * standoff_m,
                    camera_z,
                )
                valid, reason = _camera_position_validity(position, target, workspace=workspace, config=config)
                payload = {
                    "candidate_source": "generated_final_policy",
                    "angle_rad": _round(_normalize_angle(angle)),
                    "angle_source": str(record["angle_source"]),
                    "angle_offset_deg": _round(float(record["angle_offset_deg"])),
                    "standoff_m": _round(standoff_m),
                    "standoff_multiplier": _round(standoff_multiplier),
                    "camera_z_offset_m": _round(float(z_offset)),
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
                            "candidate_source": "generated_final_policy",
                            "angle_offset_deg": float(record["angle_offset_deg"]),
                            "standoff_m": _round(standoff_m),
                            "standoff_multiplier": _round(standoff_multiplier),
                            "camera_z_offset_m": _round(float(z_offset)),
                            "camera_position_world": _round_vector(position),
                        }
                    )
    if valid_candidates:
        valid_candidates.sort(key=lambda candidate: _camera_position_candidate_sort_key(candidate, workspace=workspace))
        limit = config.final_view_camera_position_limit
        if limit is not None:
            valid_candidates = valid_candidates[:limit]
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
) -> tuple[tuple[FinalViewCandidate, ...], dict[str, Any]]:
    view_candidates: list[FinalViewCandidate] = []
    roll_offsets = _unique_float_records(config.final_view_roll_offsets_deg)
    entry_portal_modes = _unique_text_records(config.entry_portal_modes)
    candidate_limit = config.final_view_candidate_limit
    generated_view_candidate_count = len(camera_candidates) * len(roll_offsets) * len(entry_portal_modes)
    candidate_index = 0
    for roll_offset_deg in roll_offsets:
        for entry_portal_mode in entry_portal_modes:
            for candidate in camera_candidates:
                camera_position = _position_from_value(candidate["camera_position_world"])
                base_transform = _make_look_at_transform(camera_position, look_at)
                candidate_transform = candidate.get("T_world_camera")
                if candidate_transform is not None and abs(float(roll_offset_deg)) > 1e-9:
                    continue
                suffix = "primary" if candidate_index == 0 else f"alt_{candidate_index:02d}"
                view_id = view_id_prefix if candidate_index == 0 else f"{view_id_prefix}_{suffix}"
                transform = (
                    candidate_transform
                    if candidate_transform is not None
                    else _roll_camera_transform(base_transform, math.radians(float(roll_offset_deg)))
                )
                view = SurveyView(
                    view_id=view_id,
                    grid_row=target_index,
                    grid_col=candidate_index,
                    camera_name=config.camera_name,
                    desired_camera_position_world=_round_vector(camera_position),
                    look_at_world=_round_vector(look_at),
                    T_world_camera=transform,
                )
                entry_views = _entry_validation_views(
                    view_id_prefix=view_id,
                    final_camera_position=camera_position,
                    look_at=look_at,
                    roll_offset_deg=float(roll_offset_deg),
                    entry_portal_mode=entry_portal_mode,
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
                        camera_z_offset_m=float(candidate["camera_z_offset_m"]),
                        roll_offset_deg=float(roll_offset_deg),
                        entry_portal_mode=entry_portal_mode,
                        candidate_source=str(candidate.get("candidate_source") or "generated_final_policy"),
                        direction_adjusted=abs(_normalize_angle(angle - preferred_angle)) > 1e-6,
                        camera_position_world=camera_position,
                    )
                )
                candidate_index += 1
    view_candidates.sort(key=_final_view_candidate_sort_key)
    retained_view_candidates = (
        view_candidates[:candidate_limit] if candidate_limit is not None else view_candidates
    )
    return tuple(retained_view_candidates), {
        "roll_offset_count": len(roll_offsets),
        "entry_portal_mode_count": len(entry_portal_modes),
        "generated_view_candidate_count_before_limit": generated_view_candidate_count,
        "retained_view_candidate_count": len(retained_view_candidates),
        "view_candidate_limit": candidate_limit,
        "view_candidate_limit_applied": candidate_limit is not None and generated_view_candidate_count > candidate_limit,
        "primary_candidate_count": min(config.final_view_primary_candidate_count, len(retained_view_candidates)),
        "expansion_order": "generate_all_then_priority_sort",
        "selection_order": "profile_priority_then_angle_standoff_height",
        "view_candidate_sort_keys": [
            "entry_portal_and_wrist_roll_profile",
            "rough_evidence_angles_before_centerline_angles",
            "smaller_angle_offset",
            "moderate_standoff_before_long_standoff",
            "camera_height_near_default_or_high_clearance",
        ],
    }


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


def _final_view_candidate_sort_key(candidate: FinalViewCandidate) -> tuple[Any, ...]:
    return (
        _final_view_profile_priority(candidate.roll_offset_deg, candidate.entry_portal_mode),
        _final_view_angle_priority(candidate.angle_source, candidate.angle_offset_deg),
        _final_view_standoff_priority(candidate.standoff_multiplier),
        _final_view_camera_z_priority(candidate.camera_z_offset_m),
        candidate.candidate_id,
    )


def _final_view_profile_priority(roll_offset_deg: float, entry_portal_mode: str) -> int:
    roll = _normalize_degrees(float(roll_offset_deg))
    preferred_profiles = (
        (0.0, "opening-grid-nearest"),
        (0.0, "final-vertical"),
        (-90.0, "opening-grid-nearest"),
        (-135.0, "workspace-center"),
        (135.0, "final-vertical"),
        (45.0, "final-vertical"),
        (-45.0, "final-vertical"),
        (0.0, "workspace-center"),
        (90.0, "opening-grid-nearest"),
        (135.0, "opening-grid-nearest"),
        (-135.0, "opening-grid-nearest"),
        (-90.0, "workspace-center"),
        (90.0, "workspace-center"),
    )
    for index, (preferred_roll, preferred_mode) in enumerate(preferred_profiles):
        if abs(_normalize_degrees(roll - preferred_roll)) <= 1e-6 and entry_portal_mode == preferred_mode:
            return index
    mode_priority = {
        "opening-grid-nearest": 0,
        "workspace-center": 1,
        "final-vertical": 2,
    }.get(entry_portal_mode, 3)
    return len(preferred_profiles) + int(abs(roll) // 45.0) * 4 + mode_priority


def _final_view_angle_priority(angle_source: str, angle_offset_deg: float) -> int:
    offset = abs(float(angle_offset_deg))
    if angle_source == "rough_evidence_offset":
        if offset <= 1e-6:
            return 0
        if abs(offset - 22.5) <= 1e-6:
            return 1
        if abs(offset - 45.0) <= 1e-6:
            return 3
        if abs(offset - 67.5) <= 1e-6:
            return 5
        if abs(offset - 90.0) <= 1e-6:
            return 7
        if abs(offset - 135.0) <= 1e-6:
            return 9
        return 10
    if angle_source == "workspace_centerline_offset":
        if offset <= 1e-6:
            return 2
        if abs(offset - 45.0) <= 1e-6:
            return 6
        return 8
    return 99


def _final_view_standoff_priority(standoff_multiplier: float) -> int:
    return _nearest_priority_index(float(standoff_multiplier), (1.0, 1.5, 2.0, 2.5, 3.0, 3.5))


def _final_view_camera_z_priority(camera_z_offset_m: float) -> int:
    return _nearest_priority_index(float(camera_z_offset_m), (0.0, 0.12, 0.08, 0.04, -0.04, -0.08, -0.12))


def _nearest_priority_index(value: float, preferred_values: tuple[float, ...]) -> int:
    if not preferred_values:
        return 0
    return min(range(len(preferred_values)), key=lambda index: abs(value - preferred_values[index]))


def _normalize_degrees(value: float) -> float:
    normalized = (float(value) + 180.0) % 360.0 - 180.0
    if abs(normalized + 180.0) <= 1e-9:
        return 180.0
    return normalized


def _camera_position_candidate_sort_key(candidate: dict[str, Any], *, workspace: SurveyWorkspace) -> tuple[Any, ...]:
    position = _position_from_value(candidate["camera_position_world"])
    source_priority = 0 if candidate.get("angle_source") == "rough_evidence_offset" else 1
    border_margin = min(
        position[0] - workspace.x_min,
        workspace.x_max - position[0],
        position[1] - workspace.y_min,
        workspace.y_max - position[1],
    )
    return (
        source_priority,
        abs(float(candidate.get("angle_offset_deg") or 0.0)),
        abs(float(candidate.get("standoff_multiplier") or 1.0) - 1.0),
        abs(float(candidate.get("camera_z_offset_m") or 0.0)),
        -border_margin,
        float(candidate.get("standoff_m") or 0.0),
        float(candidate.get("angle_rad") or 0.0),
    )


def _candidate_generation_summary(
    *,
    evaluated_camera_positions: list[dict[str, Any]],
    retained_camera_candidates: tuple[dict[str, Any], ...],
    view_candidate_summary: dict[str, Any],
    config: FinalConfig,
) -> dict[str, Any]:
    valid_camera_position_count = sum(1 for item in evaluated_camera_positions if item.get("valid") is True)
    camera_position_limit = config.final_view_camera_position_limit
    summary = {
        "policy_version": FINAL_VIEW_CANDIDATE_PRUNING_POLICY_VERSION,
        "evaluated_camera_position_count": len(evaluated_camera_positions),
        "valid_camera_position_count": valid_camera_position_count,
        "rejected_camera_position_count": len(evaluated_camera_positions) - valid_camera_position_count,
        "retained_camera_position_count": len(retained_camera_candidates),
        "camera_position_limit": camera_position_limit,
        "camera_position_limit_applied": (
            camera_position_limit is not None and valid_camera_position_count > camera_position_limit
        ),
        "geometry_sort_keys": [
            "rough_evidence_angle_before_workspace_centerline",
            "smaller_angle_offset",
            "default_standoff_first",
            "default_camera_height_first",
            "larger_workspace_border_margin",
        ],
        "camera_position_selection_order": "geometry_sorted_then_optional_camera_position_limit",
    }
    summary.update(view_candidate_summary)
    summary["potential_view_candidate_count_before_camera_position_limit"] = (
        valid_camera_position_count
        * int(summary.get("roll_offset_count") or 0)
        * int(summary.get("entry_portal_mode_count") or 0)
    )
    return summary


def _entry_validation_views(
    *,
    view_id_prefix: str,
    final_camera_position: tuple[float, float, float],
    look_at: tuple[float, float, float],
    roll_offset_deg: float,
    entry_portal_mode: str,
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
    entry_start_position = _entry_portal_camera_position(
        final_camera_position,
        entry_z=top_z,
        entry_portal_mode=entry_portal_mode,
        workspace=workspace,
    )
    entry_positions = _entry_validation_positions(
        final_camera_position=final_camera_position,
        entry_start_position=entry_start_position,
        config=config,
    )
    views: list[SurveyView] = []
    for sample_index, camera_position in enumerate(entry_positions):
        workspace.validate_camera_position(camera_position)
        entry_look_at = _entry_validation_look_at(
            camera_position,
            final_look_at=look_at,
            workspace=workspace,
            config=config,
            sample_index=sample_index,
        )
        transform = _roll_camera_transform(
            _make_look_at_transform(camera_position, entry_look_at),
            math.radians(roll_offset_deg),
        )
        views.append(
            SurveyView(
                view_id=f"{view_id_prefix}_entry_{sample_index:02d}",
                grid_row=target_index,
                grid_col=sample_index + 1,
                camera_name=config.camera_name,
                desired_camera_position_world=_round_vector(camera_position),
                look_at_world=_round_vector(entry_look_at),
                T_world_camera=transform,
            )
        )
    return tuple(views)


def _entry_validation_positions(
    *,
    final_camera_position: tuple[float, float, float],
    entry_start_position: tuple[float, float, float],
    config: FinalConfig,
) -> tuple[tuple[float, float, float], ...]:
    if config.entry_path_policy == "direct-interpolate":
        return tuple(
            _interpolate_position(
                final_camera_position,
                entry_start_position,
                (config.entry_validation_samples - sample_index) / config.entry_validation_samples,
            )
            for sample_index in range(config.entry_validation_samples)
        )
    if config.entry_path_policy == "portal-descent-then-lateral":
        return _staged_entry_validation_positions(
            final_camera_position=final_camera_position,
            entry_start_position=entry_start_position,
            sample_count=config.entry_validation_samples,
        )
    raise ValueError(f"unsupported final entry_path_policy: {config.entry_path_policy}")


def _staged_entry_validation_positions(
    *,
    final_camera_position: tuple[float, float, float],
    entry_start_position: tuple[float, float, float],
    sample_count: int,
) -> tuple[tuple[float, float, float], ...]:
    if sample_count <= 1:
        return (entry_start_position,)

    top_z = entry_start_position[2]
    final_z = final_camera_position[2]
    lateral_z = final_z + (top_z - final_z) / 2.0
    descent_count = _staged_entry_descent_sample_count(sample_count)
    lateral_count = sample_count - descent_count

    positions: list[tuple[float, float, float]] = []
    for index in range(descent_count):
        fraction = index / max(descent_count - 1, 1)
        z = top_z + (lateral_z - top_z) * fraction
        positions.append((entry_start_position[0], entry_start_position[1], z))

    for index in range(lateral_count):
        fraction = (index + 1) / lateral_count
        x = entry_start_position[0] + (final_camera_position[0] - entry_start_position[0]) * fraction
        y = entry_start_position[1] + (final_camera_position[1] - entry_start_position[1]) * fraction
        positions.append((x, y, lateral_z))

    return tuple(positions)


def _staged_entry_descent_sample_count(sample_count: int) -> int:
    if sample_count <= 1:
        return 1
    return max(2, (sample_count + 1) // 2)


def _entry_validation_look_at(
    camera_position: tuple[float, float, float],
    *,
    final_look_at: tuple[float, float, float],
    workspace: SurveyWorkspace,
    config: FinalConfig,
    sample_index: int,
) -> tuple[float, float, float]:
    if _entry_sample_uses_final_look_at(sample_index, config=config):
        return final_look_at
    if config.entry_orientation_policy == "target-look-at":
        return final_look_at
    if config.entry_orientation_policy == "vertical-descent":
        return (
            camera_position[0],
            camera_position[1],
            workspace.bottom_z_m,
        )
    raise ValueError(f"unsupported final entry_orientation_policy: {config.entry_orientation_policy}")


def _entry_sample_uses_final_look_at(sample_index: int, *, config: FinalConfig) -> bool:
    if config.entry_path_policy != "portal-descent-then-lateral":
        return False
    if config.entry_lateral_orientation_policy != "final-look-at":
        return False
    return sample_index >= _staged_entry_descent_sample_count(config.entry_validation_samples)


def _entry_portal_camera_position(
    final_camera_position: tuple[float, float, float],
    *,
    entry_z: float,
    entry_portal_mode: str,
    workspace: SurveyWorkspace,
) -> tuple[float, float, float]:
    if entry_portal_mode == "final-vertical":
        return (final_camera_position[0], final_camera_position[1], entry_z)
    if entry_portal_mode == "workspace-center":
        return (
            (workspace.x_min + workspace.x_max) / 2.0,
            (workspace.y_min + workspace.y_max) / 2.0,
            entry_z,
        )
    if entry_portal_mode == "opening-grid-nearest":
        x, y = _nearest_opening_grid_entry_xy(final_camera_position, workspace=workspace)
        return (x, y, entry_z)
    raise ValueError(f"unsupported final entry_portal_mode: {entry_portal_mode}")


def _nearest_opening_grid_entry_xy(
    final_camera_position: tuple[float, float, float],
    *,
    workspace: SurveyWorkspace,
) -> tuple[float, float]:
    step_x = (workspace.x_max - workspace.x_min) / 4.0
    step_y = (workspace.y_max - workspace.y_min) / 4.0
    centers = [
        (
            workspace.x_min + (col + 0.5) * step_x,
            workspace.y_max - (row + 0.5) * step_y,
        )
        for row in (1, 2, 3)
        for col in (1, 2, 3)
    ]
    return min(
        centers,
        key=lambda center: math.hypot(center[0] - final_camera_position[0], center[1] - final_camera_position[1]),
    )


def _interpolate_position(
    final_position: tuple[float, float, float],
    entry_position: tuple[float, float, float],
    fraction_from_final: float,
) -> tuple[float, float, float]:
    fraction = min(max(float(fraction_from_final), 0.0), 1.0)
    return (
        final_position[0] + (entry_position[0] - final_position[0]) * fraction,
        final_position[1] + (entry_position[1] - final_position[1]) * fraction,
        final_position[2] + (entry_position[2] - final_position[2]) * fraction,
    )


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
        if result.get("status") == "success" and _actual_qpos_from_validation(result) is None:
            result = dict(result)
            result["status"] = "failed"
            result["message"] = "Entry validation did not include actual_qpos for whole-arm path traceability."
        results.append(result)
        if result.get("status") != "success":
            break
    return results


def _validate_final_view_candidate(
    backend: MujocoSurveyBackend,
    candidate: FinalViewCandidate,
) -> dict[str, Any]:
    result = backend.validate_view_pose(candidate.view)
    result["validation_role"] = "final_photo_pose"
    result["view_candidate_id"] = candidate.candidate_id
    return result


def _actual_qpos_from_validation(validation: dict[str, Any]) -> tuple[float, ...] | None:
    actual_qpos = validation.get("actual_qpos")
    if not isinstance(actual_qpos, list) or not actual_qpos:
        return None
    try:
        return tuple(float(value) for value in actual_qpos)
    except (TypeError, ValueError):
        return None


def _copy_final_view_candidate_with_fixed_qpos(
    candidate: FinalViewCandidate,
    *,
    fixed_qpos: tuple[float, ...],
) -> FinalViewCandidate:
    return FinalViewCandidate(
        candidate_id=candidate.candidate_id,
        view=_copy_final_survey_view(
            candidate.view,
            fixed_pose_source=FINAL_REACHABLE_CAPTURE_FIXED_POSE_SOURCE,
            fixed_qpos=fixed_qpos,
        ),
        entry_views=candidate.entry_views,
        approach_angle_rad=candidate.approach_angle_rad,
        angle_source=candidate.angle_source,
        angle_offset_deg=candidate.angle_offset_deg,
        standoff_m=candidate.standoff_m,
        standoff_multiplier=candidate.standoff_multiplier,
        camera_z_offset_m=candidate.camera_z_offset_m,
        roll_offset_deg=candidate.roll_offset_deg,
        entry_portal_mode=candidate.entry_portal_mode,
        candidate_source=candidate.candidate_source,
        direction_adjusted=candidate.direction_adjusted,
        camera_position_world=candidate.camera_position_world,
    )


def _copy_final_survey_view(
    view: SurveyView,
    *,
    fixed_pose_source: str,
    fixed_qpos: tuple[float, ...],
) -> SurveyView:
    return SurveyView(
        view_id=view.view_id,
        grid_row=view.grid_row,
        grid_col=view.grid_col,
        camera_name=view.camera_name,
        desired_camera_position_world=view.desired_camera_position_world,
        look_at_world=view.look_at_world,
        T_world_camera=view.T_world_camera,
        scan_layer=view.scan_layer,
        scan_grid_size=view.scan_grid_size,
        fixed_pose_source=fixed_pose_source,
        fixed_qpos=fixed_qpos,
    )


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
        "view_candidate_summary": _planned_view_candidate_summary(planned),
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


def _write_final_filtered_annotation(
    *,
    capture_result: dict[str, Any],
    annotated_dir: Path,
    config: FinalConfig,
) -> None:
    view_result = capture_result.get("view") if isinstance(capture_result.get("view"), dict) else {}
    view_id = str(view_result.get("view_id") or "")
    rgb_image_path = view_result.get("rgb_image_path")
    if not view_id or not rgb_image_path:
        return
    detections = _final_filtered_annotation_detections(capture_result)
    annotated_path, annotation_error = _write_final_annotated_image(
        rgb_path=Path(str(rgb_image_path)),
        output_path=annotated_dir / f"{view_id}_yolo.png",
        detections=detections,
        image_width=config.image_width,
        image_height=config.image_height,
    )
    annotated_path_text = str(annotated_path) if annotated_path is not None else None
    view_result["annotated_image_path"] = annotated_path_text
    view_result["annotation"] = {
        "source": "final_selected_observation",
        "included_statuses": ["confirmed", "follow_up_observed", "quality_limited", "class_conflict", "unconfirmed"],
        "detections": len(detections),
        "status": "written" if annotated_path is not None else "unavailable",
        "error": annotation_error,
    }
    capture_result["annotated_image_path"] = annotated_path_text


def _final_filtered_annotation_detections(capture_result: dict[str, Any]) -> list[dict[str, Any]]:
    best = capture_result.get("best_observation")
    if not isinstance(best, dict):
        return []
    bbox = _optional_bbox(best.get("bbox_xyxy"))
    if bbox is None:
        return []
    target = capture_result.get("target") if isinstance(capture_result.get("target"), dict) else {}
    recognition = capture_result.get("recognition") if isinstance(capture_result.get("recognition"), dict) else {}
    object_id = str(target.get("object_id") or "final_object")
    status = str(capture_result.get("status") or "observed")
    class_name = str(recognition.get("detected_class_name") or best.get("class_name") or target.get("class_name") or "object")
    confidence = _float_or_zero(recognition.get("confidence") if recognition.get("confidence") is not None else best.get("confidence"))
    label_status = "" if status == "confirmed" else f" {status}"
    return [
        {
            "bbox_xyxy": [_round(value) for value in bbox],
            "confidence": confidence,
            "class_id": best.get("class_id"),
            "class_name": class_name,
            "annotation_label": f"{object_id}{label_status} {class_name} {confidence:.2f}",
            "source": "final_selected_observation",
            "source_status": status,
            "target_xy_distance_m": recognition.get("target_xy_distance_m"),
            "bbox_quality": recognition.get("bbox_quality"),
        }
    ]


def _write_final_annotated_image(
    *,
    rgb_path: Path,
    output_path: Path,
    detections: list[dict[str, Any]],
    image_width: int,
    image_height: int,
) -> tuple[Path | None, str | None]:
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return (None, "pillow_not_available")

    try:
        image = Image.open(rgb_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = _final_annotation_font(ImageFont)
        for index, detection in enumerate(detections):
            bbox = _optional_bbox(detection.get("bbox_xyxy"))
            if bbox is None:
                continue
            left, top, right, bottom = _bbox_pixel_bounds(
                bbox,
                image_width=image_width,
                image_height=image_height,
            )
            color = _final_annotation_box_color(index)
            draw.rectangle((left, top, right, bottom), outline=color, width=4)
            label = str(detection.get("annotation_label") or "").strip()
            if not label:
                continue
            text_bbox = draw.textbbox((left, top), label, font=font)
            text_height = text_bbox[3] - text_bbox[1]
            label_top = max(0.0, float(top) - text_height - 6.0)
            label_bbox = (
                float(left),
                label_top,
                float(left) + text_bbox[2] - text_bbox[0] + 8.0,
                label_top + text_height + 6.0,
            )
            draw.rectangle(label_bbox, fill=color)
            draw.text((float(left) + 4.0, label_top + 3.0), label, fill=(0, 0, 0), font=font)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        return (output_path, None)
    except Exception as exc:
        return (None, f"{type(exc).__name__}: {exc}")


def _final_annotation_font(image_font_module: Any) -> Any:
    for font_path in (
        Path("/home/yoda/.local/share/fonts/noto-cjk/NotoSansCJKsc-Regular.otf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/arphic/uming.ttc"),
    ):
        if font_path.exists():
            return image_font_module.truetype(str(font_path), size=24)
    return image_font_module.load_default()


def _final_annotation_box_color(index: int) -> tuple[int, int, int]:
    colors = (
        (52, 199, 89),
        (0, 122, 255),
        (255, 204, 0),
        (175, 82, 222),
        (255, 149, 0),
    )
    return colors[index % len(colors)]


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
        "view_candidate_summary": _planned_view_candidate_summary(planned),
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
            "Formal reports keep an attempt summary and limited failure samples; enable save_debug_trace for the full candidate attempt trace.",
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
        "view_candidate_summary": _planned_view_candidate_summary(planned),
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


def _skipped_follow_up_capture_result(planned: FinalPlannedCapture) -> dict[str, Any]:
    return {
        "target": planned.target.to_dict(),
        "status": "skipped_follow_up_not_needed",
        "source_status": planned.target.source_status,
        "target_role": planned.target.target_role,
        "selected_view_candidate": None,
        "view_candidate_attempts": [],
        "view_candidate_summary": _planned_view_candidate_summary(planned),
        "view": _planned_view_result(planned.view),
        "entry_validation": [],
        "matched_observations": [],
        "best_observation": None,
        "recognition": {
            "status": "not_run",
            "reason": "desired_stable_object_count_already_reached",
        },
        "final_image_path": None,
        "depth_path": None,
        "annotated_image_path": None,
        "yolo_raw_path": None,
        "notes": [
            "This follow-up target was not rendered because confirmed primary/follow-up objects already reached the configured desired stable count.",
            "Skipped follow-up targets remain visible in object_captures and unstable_objects instead of being treated as successful detections.",
        ],
    }


def _desired_stable_count_reached(
    object_captures: list[dict[str, Any]],
    *,
    desired_stable_object_count: int | None,
    single_observation_min_confidence: float = DEFAULT_FINAL_SINGLE_OBSERVATION_MIN_CONFIDENCE,
) -> bool:
    if desired_stable_object_count is None:
        return False
    selection = select_stable_final_objects(
        object_captures,
        desired_stable_object_count=desired_stable_object_count,
        single_observation_min_confidence=single_observation_min_confidence,
    )
    return int(selection["stable_object_selection"]["stable_object_count"]) >= desired_stable_object_count


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
            return None, _rejected_stable_object_payload(capture, reason=_follow_up_rejection_reason(target, capture))
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
    confidence_float = float(confidence)
    source_observation_count = _final_source_observation_count(capture)
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
        confidence=confidence_float,
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
                "bbox_quality": recognition.get("bbox_quality"),
                "source_observation_count": source_observation_count,
            },
            "pose_quality": {
                "position_source": "final_close_yolo_depth_observation",
                "orientation_source": (
                    "source_rough_T_world_object"
                    if source_transform is not None
                    else "identity_orientation_no_source_pose"
                ),
            },
            "notes": _stable_object_notes(source_transform),
        }
    )
    return stable_object, None


def _matching_stable_object(
    candidate: dict[str, Any],
    accepted: list[dict[str, Any]],
    *,
    max_distance_m: float,
) -> dict[str, Any] | None:
    candidate_class = _optional_text(candidate.get("class_name"))
    candidate_position = _optional_position_from_value(candidate.get("position_world"))
    if candidate_class is None or candidate_position is None:
        return None
    best: tuple[float, dict[str, Any]] | None = None
    for item in accepted:
        item_class = _optional_text(item.get("class_name"))
        item_position = _optional_position_from_value(item.get("position_world"))
        if item_class != candidate_class or item_position is None:
            continue
        distance = _xy_distance(candidate_position, item_position)
        if distance > max_distance_m:
            continue
        if best is None or distance < best[0]:
            best = (distance, item)
    return best[1] if best is not None else None


def _split_overcomplete_primary_objects(
    primary_objects: list[dict[str, Any]],
    *,
    desired_stable_object_count: int | None,
    single_observation_min_confidence: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if desired_stable_object_count is None or len(primary_objects) <= desired_stable_object_count:
        return list(primary_objects), []

    stable_objects = list(primary_objects)
    weak_single_objects = [
        stable_object
        for stable_object in stable_objects
        if _is_weak_single_observation_stable_object(
            stable_object,
            single_observation_min_confidence=single_observation_min_confidence,
        )
    ]
    weak_single_objects.sort(key=_weak_single_observation_sort_key)

    rejected_objects: list[dict[str, Any]] = []
    for stable_object in weak_single_objects:
        if len(stable_objects) <= desired_stable_object_count:
            break
        stable_objects.remove(stable_object)
        rejected_objects.append(
            _overcomplete_single_observation_rejection_payload(
                stable_object,
                desired_stable_object_count=desired_stable_object_count,
                single_observation_min_confidence=single_observation_min_confidence,
            )
        )

    return stable_objects, rejected_objects


def _is_weak_single_observation_stable_object(
    stable_object: dict[str, Any],
    *,
    single_observation_min_confidence: float,
) -> bool:
    selection = stable_object.get("selection")
    if not isinstance(selection, dict):
        return False
    source_count = selection.get("source_observation_count")
    try:
        source_observation_count = int(source_count)
    except (TypeError, ValueError):
        source_observation_count = 1
    if source_observation_count > 1:
        return False
    confidence = _optional_float(stable_object.get("confidence"))
    return confidence is not None and confidence < single_observation_min_confidence


def _weak_single_observation_sort_key(stable_object: dict[str, Any]) -> tuple[float, float, str]:
    confidence = _optional_float(stable_object.get("confidence"))
    selection = stable_object.get("selection") if isinstance(stable_object.get("selection"), dict) else {}
    distance = _optional_float(selection.get("target_xy_distance_m")) if isinstance(selection, dict) else None
    return (
        confidence if confidence is not None else float("inf"),
        -(distance if distance is not None else 0.0),
        str(stable_object.get("object_id") or ""),
    )


def _overcomplete_single_observation_rejection_payload(
    stable_object: dict[str, Any],
    *,
    desired_stable_object_count: int,
    single_observation_min_confidence: float,
) -> dict[str, Any]:
    selection = stable_object.get("selection") if isinstance(stable_object.get("selection"), dict) else {}
    return {
        "object_id": stable_object.get("object_id"),
        "source_status": stable_object.get("source_status"),
        "target_role": stable_object.get("target_role"),
        "capture_status": "confirmed",
        "reason": "overcomplete_single_observation_low_confidence",
        "detected_class_name": stable_object.get("class_name"),
        "source_class_name": stable_object.get("class_name"),
        "candidate_class_names": [],
        "bbox_xyxy": stable_object.get("bbox_xyxy"),
        "bbox_quality": selection.get("bbox_quality") if isinstance(selection, dict) else None,
        "target_xy_distance_m": selection.get("target_xy_distance_m") if isinstance(selection, dict) else None,
        "final_image_path": stable_object.get("final_image_path"),
        "source_observation_count": selection.get("source_observation_count") if isinstance(selection, dict) else None,
        "single_observation_min_confidence": _round(single_observation_min_confidence),
        "desired_stable_object_count": int(desired_stable_object_count),
        "notes": [
            "This confirmed primary target was excluded because the final stable set was overcomplete and this object had only weak single-observation close-view evidence.",
        ],
    }


def _final_source_observation_count(capture: dict[str, Any]) -> int:
    observation = capture.get("best_observation")
    if not isinstance(observation, dict):
        return 1
    value = observation.get("source_observation_count")
    if value is None:
        return 1
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _duplicate_follow_up_rejection_payload(
    capture: dict[str, Any],
    *,
    stable_object: dict[str, Any],
    duplicate: dict[str, Any],
    max_distance_m: float,
) -> dict[str, Any]:
    rejection = _rejected_stable_object_payload(capture, reason="duplicate_follow_up_observation")
    candidate_position = _optional_position_from_value(stable_object.get("position_world"))
    duplicate_position = _optional_position_from_value(duplicate.get("position_world"))
    distance = (
        _xy_distance(candidate_position, duplicate_position)
        if candidate_position is not None and duplicate_position is not None
        else None
    )
    rejection["duplicate_of_object_id"] = duplicate.get("object_id")
    rejection["duplicate_xy_distance_m"] = _round(distance) if distance is not None else None
    rejection["duplicate_radius_m"] = _round(max_distance_m)
    rejection["notes"] = [
        "This follow-up target was resolved visually but excluded from stable_objects because it duplicates an already confirmed stable object.",
    ]
    return rejection


def _follow_up_not_needed_rejection_payload(
    capture: dict[str, Any],
    *,
    desired_stable_object_count: int,
    stable_object_count: int,
) -> dict[str, Any]:
    rejection = _rejected_stable_object_payload(capture, reason="follow_up_not_needed")
    rejection["desired_stable_object_count"] = int(desired_stable_object_count)
    rejection["current_stable_object_count"] = int(stable_object_count)
    rejection["notes"] = [
        "This follow-up target was observed but excluded from stable_objects because primary confirmed objects already reached the configured desired count.",
    ]
    return rejection


def _follow_up_stable_reason(target: dict[str, Any], capture: dict[str, Any]) -> str | None:
    recognition = capture.get("recognition")
    if not isinstance(recognition, dict):
        return None
    confidence = _optional_float(recognition.get("confidence"))
    if confidence is None or confidence < DEFAULT_FINAL_FOLLOW_UP_PROMOTION_MIN_CONFIDENCE:
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


def _follow_up_rejection_reason(target: dict[str, Any], capture: dict[str, Any]) -> str:
    recognition = capture.get("recognition")
    if not isinstance(recognition, dict):
        return "follow_up_not_resolved"
    confidence = _optional_float(recognition.get("confidence"))
    if confidence is None or confidence < DEFAULT_FINAL_FOLLOW_UP_PROMOTION_MIN_CONFIDENCE:
        return "follow_up_low_confidence"
    detected_class = _optional_text(recognition.get("detected_class_name"))
    if detected_class is None:
        return "follow_up_not_resolved"
    source_status = str(target.get("source_status") or "")
    source_class = _optional_text(target.get("class_name"))
    candidate_class_names = _string_tuple(target.get("candidate_class_names", []))
    if source_status == "tentative" and source_class is not None and detected_class != source_class:
        return "follow_up_class_mismatch"
    if source_status == "ambiguous" and candidate_class_names and detected_class not in candidate_class_names:
        return "follow_up_class_mismatch"
    return "follow_up_not_resolved"


def _stable_selection_payload(
    stable_objects: list[dict[str, Any]],
    rejected_objects: list[dict[str, Any]],
    *,
    status: str,
    desired_stable_object_count: int | None,
    single_observation_min_confidence: float,
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
            "desired_stable_object_count": desired_stable_object_count,
            "follow_up_duplicate_radius_m": _round(DEFAULT_FINAL_FOLLOW_UP_DUPLICATE_RADIUS_M),
            "follow_up_promotion_min_confidence": _round(DEFAULT_FINAL_FOLLOW_UP_PROMOTION_MIN_CONFIDENCE),
            "single_observation_min_confidence": _round(single_observation_min_confidence),
            "rules": [
                "Primary rough-stable targets enter stable_objects only when close capture confirms the same class.",
                "If confirmed primary targets exceed the configured desired stable count, weak single-observation confirmations are excluded first instead of hard-trimming arbitrary objects.",
                "Follow-up targets are promotion candidates only while the stable object list is below the configured desired count.",
                "Follow-up targets are not rendered once the configured desired stable count has already been reached.",
                "Tentative targets enter stable_objects only when close capture observes the same class with enough confidence, or when no tentative class was supplied.",
                "Ambiguous targets enter stable_objects only when close capture resolves to one of the candidate classes with enough confidence.",
                "Resolved follow-up targets are kept out of stable_objects when they duplicate an already confirmed same-class object within the configured XY radius.",
                "Quality-limited, unconfirmed, class-conflict, skipped, and failed captures are kept out of stable_objects.",
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
        "bbox_xyxy": recognition.get("bbox_xyxy"),
        "bbox_quality": recognition.get("bbox_quality"),
        "target_xy_distance_m": recognition.get("target_xy_distance_m"),
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
        notes.append("The orientation in T_world_object is inherited from the rough-stage source pose.")
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
    bbox_quality = _bbox_quality_payload(best_observation, config=config)
    payload = {
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
        "bbox_quality": bbox_quality,
    }
    if not bbox_quality["accepted"]:
        payload["reason"] = "bbox_quality_limited"
    return payload


def _bbox_quality_payload(
    observation: dict[str, Any],
    *,
    config: FinalConfig,
) -> dict[str, Any]:
    bbox = _bbox_tuple(observation.get("bbox_xyxy"))
    bbox_area = _bbox_area_px(bbox)
    margin = _bbox_border_margin_px(bbox, image_width=config.image_width, image_height=config.image_height)
    border_safe = margin >= config.selection_border_margin_px
    reasons: list[str] = []
    if bbox_area <= 0.0:
        reasons.append("empty_bbox")
    if not border_safe:
        reasons.append("bbox_too_close_to_image_boundary")
    accepted = bbox_area > 0.0 and border_safe
    return {
        "policy_version": FINAL_BBOX_QUALITY_POLICY_VERSION,
        "status": "accepted" if accepted else "limited",
        "accepted": accepted,
        "bbox_area_px": _round(bbox_area),
        "bbox_min_border_margin_px": _round(margin),
        "selection_border_margin_px": _round(config.selection_border_margin_px),
        "image_width": int(config.image_width),
        "image_height": int(config.image_height),
        "reasons": reasons,
        "rules": [
            "A final confirmation bbox must have positive image area.",
            "A final confirmation bbox must keep the configured minimum distance from every image border.",
            "Border-limited observations remain visible in reports but do not confirm or promote objects.",
        ],
    }


def _recognition_bbox_quality_accepted(recognition: dict[str, Any]) -> bool:
    bbox_quality = recognition.get("bbox_quality")
    return isinstance(bbox_quality, dict) and bbox_quality.get("accepted") is True


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
        "policy_version": "final_observation_selection_policy_v3",
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
            "Prefer larger bbox area before treating image-border margin as a tie-breaker.",
            "Prefer detections with enough image-border margin before using target distance as a tie-breaker.",
        ],
    }
    return (
        1.0 if class_match else 0.0,
        float(merged_count),
        bbox_area,
        1.0 if border_safe else 0.0,
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
            if not _recognition_bbox_quality_accepted(recognition):
                return "quality_limited"
            return "confirmed"
        if class_match is False:
            return "class_conflict"
        return "unconfirmed"
    if not _recognition_bbox_quality_accepted(recognition):
        return "quality_limited"
    return "follow_up_observed"


def _capture_result_satisfies_target(capture: dict[str, Any]) -> bool:
    return str(capture.get("status") or "") in {"confirmed", "follow_up_observed", "captured_yolo_skipped"}


def _capture_result_fallback_score(capture: dict[str, Any]) -> tuple[float, float, float]:
    status = str(capture.get("status") or "")
    status_score = {
        "quality_limited": 2.5,
        "class_conflict": 2.0,
        "unconfirmed": 1.0,
        "capture_failed": 0.0,
    }.get(status, 0.0)
    recognition = capture.get("recognition") if isinstance(capture.get("recognition"), dict) else {}
    confidence = _float_or_zero(recognition.get("confidence"))
    distance = _float_or_zero(recognition.get("target_xy_distance_m"))
    return (status_score, confidence, -distance)


def _continue_after_capture_message(capture: dict[str, Any]) -> str:
    recognition = capture.get("recognition") if isinstance(capture.get("recognition"), dict) else {}
    reason = recognition.get("reason") or recognition.get("status") or capture.get("status")
    return f"Captured image did not satisfy final target confirmation policy; continuing to next candidate ({reason})."


def _capture_notes(target: FinalTarget, status: str) -> list[str]:
    notes = [
        "Single-object capture used rough report target positions as formal inputs and did not assume a fixed object count.",
        "The final photo was rendered only after top-opening entry waypoints and the final wrist-camera pose passed whole-arm IK and collision checks.",
        "The rendered pose reused the validated fixed qpos recorded in selected_view_candidate.view.",
        "If a collision-safe capture does not confirm the target, final continues trying later collision-safe candidates before returning an unresolved result.",
    ]
    if target.source_status != "stable":
        notes.append("This target came from tentative or ambiguous rough evidence and remains a follow-up result, not a stable object promotion.")
    if status == "class_conflict":
        notes.append("The closest close-view YOLO observation did not match the source stable class name.")
    if status == "quality_limited":
        notes.append("The closest close-view YOLO observation was kept in the report but its bbox did not satisfy final confirmation quality.")
    if status == "unconfirmed":
        notes.append("No close YOLO-depth observation confirmed this target within the configured association radius.")
    return notes


def _overall_status(object_captures: list[dict[str, Any]]) -> str:
    if not object_captures:
        return "failed"
    success_statuses = {"confirmed", "follow_up_observed", "captured_yolo_skipped", "skipped_follow_up_not_needed"}
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


def _final_reachable_planning_summary(
    *,
    planned_captures: tuple[FinalPlannedCapture, ...],
    object_captures: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_generation_summaries = [
        planned.candidate_generation_summary
        for planned in planned_captures
        if isinstance(planned.candidate_generation_summary, dict)
    ]
    attempt_count = 0
    attempt_status_counts: dict[str, int] = {}
    attempt_search_phase_counts: dict[str, int] = {}
    rejection_counts = {
        "entry_collision": 0,
        "entry_ik_failed": 0,
        "entry_missing_qpos": 0,
        "final_collision": 0,
        "final_ik_failed": 0,
        "final_missing_qpos": 0,
        "render_revalidation_failed": 0,
    }
    selected_fixed_qpos_count = 0
    for capture in object_captures:
        selected_candidate = capture.get("selected_view_candidate")
        selected_view = selected_candidate.get("view") if isinstance(selected_candidate, dict) else None
        if isinstance(selected_view, dict) and selected_view.get("fixed_pose_source") == FINAL_REACHABLE_CAPTURE_FIXED_POSE_SOURCE:
            selected_fixed_qpos_count += 1
        attempts = capture.get("view_candidate_attempts", [])
        if not isinstance(attempts, list):
            continue
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            attempt_count += 1
            status = str(attempt.get("status") or "unknown")
            attempt_status_counts[status] = attempt_status_counts.get(status, 0) + 1
            search_phase = str(attempt.get("search_phase") or "unknown")
            attempt_search_phase_counts[search_phase] = attempt_search_phase_counts.get(search_phase, 0) + 1
            for entry in attempt.get("entry_validation", []):
                if isinstance(entry, dict):
                    _accumulate_final_validation_rejection(
                        rejection_counts,
                        entry,
                        collision_key="entry_collision",
                        ik_key="entry_ik_failed",
                        missing_qpos_key="entry_missing_qpos",
                    )
            final_validation = attempt.get("final_pose_validation")
            if isinstance(final_validation, dict):
                _accumulate_final_validation_rejection(
                    rejection_counts,
                    final_validation,
                    collision_key="final_collision",
                    ik_key="final_ik_failed",
                    missing_qpos_key="final_missing_qpos",
                )
            view = attempt.get("view")
            if isinstance(view, dict) and view.get("status") != "success" and status == "final_pose_failed":
                rejection_counts["render_revalidation_failed"] += 1
    return {
        "strategy": "whole_arm_final_view_candidate_selection_v1",
        "planned_target_count": len(planned_captures),
        "planned_view_candidate_count": sum(len(planned.view_candidates) for planned in planned_captures),
        "potential_view_candidate_count_before_camera_position_limit": sum(
            int(summary.get("potential_view_candidate_count_before_camera_position_limit") or 0)
            for summary in candidate_generation_summaries
        ),
        "generated_view_candidate_count_before_limit": sum(
            int(summary.get("generated_view_candidate_count_before_limit") or 0)
            for summary in candidate_generation_summaries
        ),
        "retained_view_candidate_count": sum(
            int(summary.get("retained_view_candidate_count") or 0)
            for summary in candidate_generation_summaries
        ),
        "view_candidate_limit": (
            planned_captures[0].candidate_generation_summary.get("view_candidate_limit")
            if planned_captures
            else None
        ),
        "candidate_pruning_policy_version": FINAL_VIEW_CANDIDATE_PRUNING_POLICY_VERSION,
        "captured_target_count": sum(1 for capture in object_captures if capture.get("view", {}).get("status") == "success"),
        "selected_fixed_qpos_count": selected_fixed_qpos_count,
        "attempt_count": attempt_count,
        "attempt_status_counts": attempt_status_counts,
        "attempt_search_phase_counts": attempt_search_phase_counts,
        "rejection_counts": rejection_counts,
        "rules": [
            "Final candidates enumerate standoff, approach angle, camera height, camera roll, and entry portal mode from FinalConfig.",
            "Camera positions are sorted by geometric quality; final_view_camera_position_limit can optionally cap this cheap geometric list before view expansion.",
            "Expanded view candidates are priority-sorted by entry portal and wrist-roll profile before geometric preferences.",
            "final_view_candidate_limit is optional; when unset, ordered candidates are validated until one succeeds or the candidate set is exhausted.",
            "When final_view_candidate_limit is set, omitted candidates remain represented by summary counts and the target fails explicitly if none pass.",
            "Each candidate is accepted only after all top-opening portal entry waypoints and the final photo pose pass MuJoCo IK and robot collision checks.",
            "The rendered final photo reuses the validated qpos through fixed_pose_source instead of solving a new pose silently.",
            "If no candidate passes, the target remains failed or partial and is not promoted to stable_objects.",
        ],
    }


def _write_final_outputs(
    *,
    plan_path: Path,
    report_path: Path,
    debug_trace_path: Path,
    plan_payload: dict[str, Any],
    report: dict[str, Any],
    created_utc: str,
    rough_report: Task1RoughReport,
    config: FinalConfig,
    capture_dir: Path,
    planned_captures: tuple[FinalPlannedCapture, ...],
    object_captures: list[dict[str, Any]],
) -> None:
    _write_json(plan_path, plan_payload)
    _write_json(report_path, report)
    if config.save_debug_trace:
        _write_json(
            debug_trace_path,
            _debug_trace_payload(
                created_utc=created_utc,
                rough_report=rough_report,
                config=config,
                capture_dir=capture_dir,
                plan_path=plan_path,
                report_path=report_path,
                debug_trace_path=debug_trace_path,
                planned_captures=planned_captures,
                object_captures=object_captures,
            ),
        )


def _debug_trace_payload(
    *,
    created_utc: str,
    rough_report: Task1RoughReport,
    config: FinalConfig,
    capture_dir: Path,
    plan_path: Path,
    report_path: Path,
    debug_trace_path: Path,
    planned_captures: tuple[FinalPlannedCapture, ...],
    object_captures: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "task1_final_debug_trace_v1",
        "stage": "final",
        "artifact_role": "debug_trace",
        "created_utc": created_utc,
        "source_rough_report_path": str(rough_report.path),
        "source_rough_status": rough_report.status,
        "task1_run_dir": str(rough_report.task1_run_dir),
        "capture_dir": str(capture_dir),
        "plan_path": str(plan_path),
        "report_path": str(report_path),
        "debug_trace_path": str(debug_trace_path),
        "save_debug_trace": config.save_debug_trace,
        "planned_captures": [planned.to_dict() for planned in planned_captures],
        "object_captures": object_captures,
        "notes": [
            "This file contains the complete final candidate and attempt trace.",
            "Formal final_plan.json and final_report.json intentionally keep only selected candidates, summaries, and limited failure samples.",
        ],
    }


def _debug_trace_policy_payload(config: FinalConfig, *, capture_dir: Path) -> dict[str, Any]:
    return {
        "save_debug_trace": config.save_debug_trace,
        "debug_trace_path": str(capture_dir / "final_debug_trace.json") if config.save_debug_trace else None,
        "formal_outputs_include_full_candidate_trace": False,
        "formal_failure_sample_limit": config.failure_sample_limit,
        "full_trace_schema_version": "task1_final_debug_trace_v1",
    }


def _planned_capture_public_payload(planned: FinalPlannedCapture) -> dict[str, Any]:
    return {
        "target": planned.target.to_dict(),
        "view": _planned_view_result(planned.view),
        "entry_views": [_planned_view_result(view) for view in planned.entry_views],
        "approach_angle_rad": _round(planned.approach_angle_rad),
        "direction_source": planned.direction_source,
        "direction_adjusted": planned.direction_adjusted,
        "view_candidate_summary": _planned_view_candidate_summary(planned),
    }


def _planned_view_candidate_summary(planned: FinalPlannedCapture) -> dict[str, Any]:
    first_candidate = planned.view_candidates[0] if planned.view_candidates else None
    return {
        "candidate_count": len(planned.view_candidates),
        "evaluated_camera_position_count": len(planned.evaluated_camera_positions),
        "entry_view_count": len(planned.entry_views),
        "first_candidate": _view_candidate_compact_payload(first_candidate) if first_candidate is not None else None,
        "candidate_generation": planned.candidate_generation_summary,
        "full_candidate_trace_in_formal_output": False,
    }


def _object_capture_public_payload(capture: dict[str, Any], *, config: FinalConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in (
        "target",
        "status",
        "source_status",
        "target_role",
        "selected_view_candidate",
        "view_candidate_summary",
        "view",
        "entry_validation",
        "matched_observations",
        "best_observation",
        "recognition",
        "final_image_path",
        "depth_path",
        "annotated_image_path",
        "yolo_raw_path",
        "notes",
    ):
        if key in capture:
            payload[key] = capture[key]
    payload["view_candidate_attempt_summary"] = _view_candidate_attempt_summary(
        capture.get("view_candidate_attempts"),
        sample_limit=config.failure_sample_limit,
    )
    return payload


def _view_candidate_attempt_summary(value: Any, *, sample_limit: int) -> dict[str, Any]:
    attempts = value if isinstance(value, list) else []
    status_counts: dict[str, int] = {}
    failure_attempts: list[dict[str, Any]] = []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        status = str(attempt.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        if status != "captured":
            failure_attempts.append(attempt)
    samples = [_attempt_failure_sample_payload(attempt) for attempt in failure_attempts[:sample_limit]]
    return {
        "attempt_count": len(attempts),
        "status_counts": status_counts,
        "failure_sample_limit": sample_limit,
        "failure_sample_count": len(samples),
        "omitted_failure_attempt_count": max(0, len(failure_attempts) - len(samples)),
        "failure_samples": samples,
        "full_attempt_trace_in_formal_output": False,
    }


def _attempt_failure_sample_payload(attempt: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": attempt.get("status"),
        "message": attempt.get("message"),
        "search_phase": attempt.get("search_phase"),
        "candidate": _view_candidate_compact_payload(attempt.get("candidate")),
        "entry_validation": [
            _validation_compact_payload(entry)
            for entry in attempt.get("entry_validation", [])
            if isinstance(entry, dict)
        ],
    }
    final_pose_validation = attempt.get("final_pose_validation")
    if isinstance(final_pose_validation, dict):
        payload["final_pose_validation"] = _validation_compact_payload(final_pose_validation)
    recognition = attempt.get("recognition")
    if isinstance(recognition, dict):
        payload["recognition"] = {
            "status": recognition.get("status"),
            "reason": recognition.get("reason"),
            "bbox_quality": recognition.get("bbox_quality"),
        }
    return payload


def _view_candidate_compact_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, FinalViewCandidate):
        return {
            "candidate_id": value.candidate_id,
            "approach_angle_rad": _round(value.approach_angle_rad),
            "angle_source": value.angle_source,
            "angle_offset_deg": _round(value.angle_offset_deg),
            "standoff_m": _round(value.standoff_m),
            "standoff_multiplier": _round(value.standoff_multiplier),
            "camera_z_offset_m": _round(value.camera_z_offset_m),
            "roll_offset_deg": _round(value.roll_offset_deg),
            "entry_portal_mode": value.entry_portal_mode,
            "candidate_source": value.candidate_source,
            "direction_adjusted": value.direction_adjusted,
            "camera_position_world": [_round(component) for component in value.camera_position_world],
            "view_id": value.view.view_id,
            "entry_view_count": len(value.entry_views),
        }
    if not isinstance(value, dict):
        return {}
    return {
        "candidate_id": value.get("candidate_id"),
        "approach_angle_rad": value.get("approach_angle_rad"),
        "angle_source": value.get("angle_source"),
        "angle_offset_deg": value.get("angle_offset_deg"),
        "standoff_m": value.get("standoff_m"),
        "standoff_multiplier": value.get("standoff_multiplier"),
        "camera_z_offset_m": value.get("camera_z_offset_m"),
        "roll_offset_deg": value.get("roll_offset_deg"),
        "entry_portal_mode": value.get("entry_portal_mode"),
        "candidate_source": value.get("candidate_source"),
        "direction_adjusted": value.get("direction_adjusted"),
        "camera_position_world": value.get("camera_position_world"),
        "view_id": (value.get("view") if isinstance(value.get("view"), dict) else {}).get("view_id"),
        "entry_view_count": len(value.get("entry_views", [])) if isinstance(value.get("entry_views"), list) else None,
    }


def _validation_compact_payload(validation: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "view_id": validation.get("view_id"),
        "status": validation.get("status"),
        "message": validation.get("message"),
        "validation_role": validation.get("validation_role"),
        "view_candidate_id": validation.get("view_candidate_id"),
    }
    ik = validation.get("ik")
    if isinstance(ik, dict):
        payload["ik"] = {
            "success": ik.get("success"),
            "position_error_m": ik.get("position_error_m"),
            "orientation_error_rad": ik.get("orientation_error_rad"),
            "iterations": ik.get("iterations"),
        }
    collision = validation.get("collision")
    if isinstance(collision, dict):
        payload["collision"] = {
            "collision_free": collision.get("collision_free"),
            "contact_count": collision.get("contact_count"),
            "robot_tank_contact_count": collision.get("robot_tank_contact_count"),
        }
    return payload


def _accumulate_final_validation_rejection(
    counts: dict[str, int],
    validation: dict[str, Any],
    *,
    collision_key: str,
    ik_key: str,
    missing_qpos_key: str,
) -> None:
    if validation.get("status") == "success":
        return
    collision = validation.get("collision") if isinstance(validation.get("collision"), dict) else {}
    ik = validation.get("ik") if isinstance(validation.get("ik"), dict) else {}
    message = str(validation.get("message") or "")
    if collision and not collision.get("collision_free", True):
        counts[collision_key] += 1
    elif ik and not ik.get("success", False):
        counts[ik_key] += 1
    elif "actual_qpos" in message:
        counts[missing_qpos_key] += 1


def _plan_payload(
    *,
    created_utc: str,
    rough_report: Task1RoughReport,
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
        "source_rough_report_path": str(rough_report.path),
        "source_rough_status": rough_report.status,
        "layout_snapshot_path": str(rough_report.layout_snapshot_path),
        "task1_run_dir": str(rough_report.task1_run_dir),
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
            "entry_portal_modes": list(config.entry_portal_modes),
            "entry_orientation_policy": config.entry_orientation_policy,
            "entry_lateral_orientation_policy": config.entry_lateral_orientation_policy,
            "entry_path_policy": config.entry_path_policy,
            "desired_stable_object_count": config.desired_stable_object_count,
            "final_view_not_top_down_only": True,
            "multiple_final_view_candidates": True,
            "multiple_final_view_standoff_distances": True,
            "multiple_final_camera_height_candidates": True,
            "multiple_final_camera_roll_candidates": True,
            "candidate_pruning_policy_version": FINAL_VIEW_CANDIDATE_PRUNING_POLICY_VERSION,
            "final_view_camera_position_limit": config.final_view_camera_position_limit,
            "final_view_candidate_limit": config.final_view_candidate_limit,
            "final_view_primary_candidate_count": config.final_view_primary_candidate_count,
            "entry_path": (
                "discrete whole-arm entry waypoints from the configured top-opening portal to each final photo pose, "
                "with portal-descent waypoints using entry_orientation_policy and lateral waypoints using "
                "entry_lateral_orientation_policy"
            ),
            "portal_descent_then_lateral_note": (
                "portal-descent-then-lateral first validates vertical descent at the opening portal, then validates "
                "low-height lateral motion toward the final photo pose before rendering. The lateral samples use "
                "entry_lateral_orientation_policy, which defaults to final-look-at so the wrist transitions toward "
                "the photo approach inside the tank instead of sliding sideways while still pointing vertically down."
            ),
            "opening_grid_nearest_note": (
                "opening-grid-nearest maps the tank opening to the inner 3x3 centers of a 4x4 workspace grid "
                "and chooses the nearest center for each final camera candidate."
            ),
            "pose_validation": (
                "MuJoCo IK plus robot body collision checks for every entry waypoint and the final photo pose."
            ),
            "render_pose_reuse": "selected final captures are rendered with fixed_qpos from the validated final pose",
            "rough_report_is_layered_input": True,
        },
        "workspace": workspace.to_dict(),
        "target_policy": {
            "primary_objects": "rough_report.stable_objects",
            "follow_up_targets": "rough_report.tentative_objects + rough_report.ambiguous_objects",
            "quality_source": "rough_report.object_selection_summary",
            "fixed_object_count_assumption": False,
        },
        "debug_trace_policy": _debug_trace_policy_payload(config, capture_dir=capture_dir),
        "planned_captures": [_planned_capture_public_payload(planned) for planned in planned_captures],
    }


def _report_payload(
    *,
    created_utc: str,
    status: str,
    message: str,
    rough_report: Task1RoughReport,
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
    stable_selection = select_stable_final_objects(
        object_captures,
        desired_stable_object_count=config.desired_stable_object_count,
        single_observation_min_confidence=config.single_observation_min_confidence,
    )
    reachable_planning_summary = _final_reachable_planning_summary(
        planned_captures=planned_captures,
        object_captures=object_captures,
    )
    return {
        "schema_version": "task1_final_report_v1",
        "stage": "final",
        "status": status,
        "created_utc": created_utc,
        "message": message,
        "source_rough_report_path": str(rough_report.path),
        "source_rough_status": rough_report.status,
        "layout_snapshot_path": str(rough_report.layout_snapshot_path),
        "task1_run_dir": str(rough_report.task1_run_dir),
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
        "annotation_policy": {
            "save_filtered_annotations": config.save_filtered_annotations,
            "filtered_annotation_dir": str(capture_dir / "annotated"),
            "filtered_annotation_source": "final_selected_observation",
            "filtered_annotation_included_statuses": [
                "confirmed",
                "follow_up_observed",
                "quality_limited",
                "class_conflict",
                "unconfirmed",
            ],
            "save_raw_yolo_annotations": config.save_raw_yolo_annotations,
            "raw_yolo_annotation_dir": str(capture_dir / "raw_annotated"),
        },
        "primary_objects": primary_targets,
        "follow_up_targets": follow_up_targets,
        "stable_objects": stable_selection["stable_objects"],
        "unstable_objects": stable_selection["unstable_objects"],
        "stable_object_selection": stable_selection["stable_object_selection"],
        "quality": rough_report.object_selection_summary,
        "object_selection_summary": rough_report.object_selection_summary,
        "final_reachable_planning_summary": reachable_planning_summary,
        "debug_trace_policy": _debug_trace_policy_payload(config, capture_dir=capture_dir),
        "planned_captures": [_planned_capture_public_payload(planned) for planned in planned_captures],
        "object_captures": [
            _object_capture_public_payload(capture, config=config)
            for capture in object_captures
        ],
        "notes": [
            "Stable rough objects are consumed as primary final capture targets.",
            "Tentative and ambiguous rough objects are consumed as explicit follow-up targets only when the configured desired stable count still needs rescue candidates.",
            "Each target keeps its own capture status; unconfirmed and class-conflict results are kept out of stable_objects.",
            "When confirmed primary targets are overcomplete, weak single-observation confirmations are reported as unstable rather than being silently kept.",
            "Final capture views are rendered only from fixed qpos values produced by successful whole-arm IK and collision validation.",
            "Formal outputs keep selected candidates, summaries, and limited failure samples; full candidate and attempt traces are written only when save_debug_trace is enabled.",
            "The report intentionally supports any number of rough targets.",
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


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


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


def _roll_camera_transform(
    transform: tuple[tuple[float, float, float, float], ...],
    roll_rad: float,
) -> tuple[tuple[float, float, float, float], ...]:
    if abs(roll_rad) <= 1e-12:
        return transform
    right = (transform[0][0], transform[1][0], transform[2][0])
    up = (transform[0][1], transform[1][1], transform[2][1])
    camera_z = (transform[0][2], transform[1][2], transform[2][2])
    position = (transform[0][3], transform[1][3], transform[2][3])
    cos_roll = math.cos(roll_rad)
    sin_roll = math.sin(roll_rad)
    rolled_right = (
        right[0] * cos_roll + up[0] * sin_roll,
        right[1] * cos_roll + up[1] * sin_roll,
        right[2] * cos_roll + up[2] * sin_roll,
    )
    rolled_up = (
        -right[0] * sin_roll + up[0] * cos_roll,
        -right[1] * sin_roll + up[1] * cos_roll,
        -right[2] * sin_roll + up[2] * cos_roll,
    )
    return (
        (_round(rolled_right[0]), _round(rolled_up[0]), _round(camera_z[0]), _round(position[0])),
        (_round(rolled_right[1]), _round(rolled_up[1]), _round(camera_z[1]), _round(position[1])),
        (_round(rolled_right[2]), _round(rolled_up[2]), _round(camera_z[2]), _round(position[2])),
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


def _unique_angle_records(records: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    unique: list[dict[str, Any]] = []
    for record in records:
        angle = _normalize_angle(float(record["angle_rad"]))
        if all(abs(_normalize_angle(angle - float(existing["angle_rad"]))) > 1e-6 for existing in unique):
            copied = dict(record)
            copied["angle_rad"] = angle
            unique.append(copied)
    return tuple(unique)


def _unique_float_records(values: tuple[float, ...]) -> tuple[float, ...]:
    unique: list[float] = []
    for value in values:
        numeric = float(value)
        if any(abs(numeric - existing) <= 1e-9 for existing in unique):
            continue
        unique.append(numeric)
    return tuple(unique)


def _unique_text_records(values: tuple[str, ...]) -> tuple[str, ...]:
    unique: list[str] = []
    for value in values:
        text = str(value)
        if text in unique:
            continue
        unique.append(text)
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


def _bbox_pixel_bounds(
    bbox_xyxy: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    left = max(0, min(image_width - 1, int(math.floor(min(x1, x2)))))
    right = max(left + 1, min(image_width, int(math.ceil(max(x1, x2)))))
    top = max(0, min(image_height - 1, int(math.floor(min(y1, y2)))))
    bottom = max(top + 1, min(image_height, int(math.ceil(max(y1, y2)))))
    return (left, top, right, bottom)


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
