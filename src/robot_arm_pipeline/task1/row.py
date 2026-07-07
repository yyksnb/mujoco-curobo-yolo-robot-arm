from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robot_arm_pipeline.task1.survey import (
    DEFAULT_CAMERA_NAME,
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
    SurveySceneInput,
    SurveyView,
    SurveyWorkspace,
    estimate_detection_ground_position,
    load_stage0_layout,
    survey_scene_from_stage0_layout,
)


DEFAULT_ROW_CAMERA_Z_M = 0.34
DEFAULT_ROW_STANDOFF_M = 0.18
DEFAULT_ROW_VIEWS_PER_CANDIDATE = 3
DEFAULT_ROW_VIEW_ANGLE_SPREAD_RAD = math.radians(45.0)
DEFAULT_ROW_MIN_OBLIQUE_DISTANCE_M = 0.08
DEFAULT_ROW_LOOK_AT_HEIGHT_OFFSET_M = 0.02
DEFAULT_ROW_ENTRY_SIDE = "survey-best"
DEFAULT_ROW_CLUSTER_RADIUS_M = 0.055
DEFAULT_ROW_VIEW_COLLISION_SEARCH = True
DEFAULT_ROW_VIEW_CANDIDATE_STANDOFF_MULTIPLIERS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5)
DEFAULT_ROW_VIEW_CANDIDATE_CAMERA_Z_OFFSETS_M = (0.0, -0.04, -0.08, -0.12, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12)
DEFAULT_ROW_VIEW_CANDIDATE_ANGLE_OFFSETS_RAD = (
    0.0,
    math.radians(22.5),
    -math.radians(22.5),
    math.radians(45.0),
    -math.radians(45.0),
    math.radians(67.5),
    -math.radians(67.5),
    math.radians(90.0),
    -math.radians(90.0),
    math.radians(135.0),
    -math.radians(135.0),
    math.pi,
)
DEFAULT_ROW_VIEW_CANDIDATE_ROLL_OFFSETS_RAD = (0.0, math.radians(90.0), -math.radians(90.0), math.pi)
DEFAULT_ROW_VIEW_CANDIDATE_MAX_ATTEMPTS: int | None = None
ROW_REACHABLE_CAPTURE_FIXED_POSE_SOURCE = "row_reachable_capture_plan_v1"
DEFAULT_ROW_YOLO_CONFIDENCE = 0.20
DEFAULT_ROW_YOLO_MAX_DETECTIONS = 20
DEFAULT_ROW_YOLO_TILE_GRID_SIZE = 1
DEFAULT_ROW_DEPTH_SAMPLE_STRIDE_PX = 6
DEFAULT_ROW_DEPTH_COMPONENT_MIN_PIXELS = 24
DEFAULT_ROW_POSE_MIN_PIXELS = 36
DEFAULT_ROW_YAW_MIN_EIGEN_RATIO = 1.6
DEFAULT_STABLE_OBJECT_MIN_CONFIDENCE = 0.50
DEFAULT_STABLE_OBJECT_MIN_SUPPORT_COUNT = 2
DEFAULT_STABLE_OBJECT_MIN_EVIDENCE_SCORE = 1.25
DEFAULT_STABLE_OBJECT_SAME_CLASS_NMS_RADIUS_M = 0.22
DEFAULT_STABLE_OBJECT_WORKSPACE_MARGIN_M = 0.02
DEFAULT_TENTATIVE_OBJECT_MIN_CONFIDENCE = 0.30
DEFAULT_TENTATIVE_OBJECT_MIN_SUPPORT_COUNT = 1
DEFAULT_TENTATIVE_OBJECT_SMALL_BBOX_AREA_PX = 45_000.0
DEFAULT_CROSS_CLASS_CONFLICT_RADIUS_M = 0.045
DEFAULT_CROSS_CLASS_AMBIGUITY_SCORE_RATIO = 0.80
DEFAULT_CLASS_VOTE_AMBIGUITY_TOP_TO_SECOND_RATIO = 1.35
DEFAULT_CLASS_VOTE_AMBIGUITY_MIN_SECONDARY_VOTE = 0.50


@dataclass(frozen=True)
class SurveyCandidate:
    candidate_id: str
    rough_position_world: tuple[float, float, float]
    supporting_views: tuple[str, ...]
    best_image_path: str | None
    best_bbox_xyxy: tuple[float, float, float, float] | None
    best_view: str | None
    class_votes: dict[str, float]
    confidence: float
    support_count: int
    raw_payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "rough_position_world": [_round(value) for value in self.rough_position_world],
            "supporting_views": list(self.supporting_views),
            "best_image_path": self.best_image_path,
            "best_bbox_xyxy": (
                [_round(value) for value in self.best_bbox_xyxy] if self.best_bbox_xyxy is not None else None
            ),
            "best_view": self.best_view,
            "class_votes": {key: _round(value) for key, value in sorted(self.class_votes.items())},
            "confidence": _round(self.confidence),
            "support_count": self.support_count,
        }


@dataclass(frozen=True)
class Task1SurveyReport:
    path: Path
    schema_version: str
    status: str
    layout_snapshot_path: Path
    task1_run_dir: Path
    survey_dir: Path
    camera_name: str
    workspace: dict[str, float]
    image_size: tuple[int, int]
    candidates: tuple[SurveyCandidate, ...]
    views_by_id: dict[str, dict[str, Any]]
    raw_payload: dict[str, Any]


@dataclass(frozen=True)
class RowConfig:
    scene_model_path: Path = DEFAULT_SCENE_MODEL
    yolo_profile_path: Path = DEFAULT_YOLO_PROFILE
    output_dir: Path = DEFAULT_OUTPUT_DIR
    camera_name: str = DEFAULT_CAMERA_NAME
    image_width: int = DEFAULT_IMAGE_WIDTH
    image_height: int = DEFAULT_IMAGE_HEIGHT
    camera_z_m: float = DEFAULT_ROW_CAMERA_Z_M
    tank_opening_z_m: float = DEFAULT_TANK_OPENING_Z_M
    opening_clearance_m: float = DEFAULT_OPENING_CLEARANCE_M
    row_standoff_m: float = DEFAULT_ROW_STANDOFF_M
    row_views_per_candidate: int = DEFAULT_ROW_VIEWS_PER_CANDIDATE
    row_view_angle_spread_rad: float = DEFAULT_ROW_VIEW_ANGLE_SPREAD_RAD
    row_min_oblique_distance_m: float = DEFAULT_ROW_MIN_OBLIQUE_DISTANCE_M
    look_at_height_offset_m: float = DEFAULT_ROW_LOOK_AT_HEIGHT_OFFSET_M
    entry_side: str = DEFAULT_ROW_ENTRY_SIDE
    row_cluster_radius_m: float = DEFAULT_ROW_CLUSTER_RADIUS_M
    row_view_collision_search: bool = DEFAULT_ROW_VIEW_COLLISION_SEARCH
    row_view_candidate_standoff_multipliers: tuple[float, ...] = DEFAULT_ROW_VIEW_CANDIDATE_STANDOFF_MULTIPLIERS
    row_view_candidate_camera_z_offsets_m: tuple[float, ...] = DEFAULT_ROW_VIEW_CANDIDATE_CAMERA_Z_OFFSETS_M
    row_view_candidate_angle_offsets_rad: tuple[float, ...] = DEFAULT_ROW_VIEW_CANDIDATE_ANGLE_OFFSETS_RAD
    row_view_candidate_roll_offsets_rad: tuple[float, ...] = DEFAULT_ROW_VIEW_CANDIDATE_ROLL_OFFSETS_RAD
    row_view_candidate_max_attempts: int | None = DEFAULT_ROW_VIEW_CANDIDATE_MAX_ATTEMPTS
    yolo_confidence: float = DEFAULT_ROW_YOLO_CONFIDENCE
    yolo_iou: float | None = None
    yolo_image_size: int | None = None
    yolo_device: str | None = None
    yolo_max_detections: int | None = DEFAULT_ROW_YOLO_MAX_DETECTIONS
    yolo_tile_grid_size: int = DEFAULT_ROW_YOLO_TILE_GRID_SIZE
    yolo_tile_overlap: float = 0.0
    yolo_tile_nms_iou: float = 0.45
    depth_sample_stride_px: int = DEFAULT_ROW_DEPTH_SAMPLE_STRIDE_PX
    depth_component_min_pixels: int = DEFAULT_ROW_DEPTH_COMPONENT_MIN_PIXELS
    pose_min_pixels: int = DEFAULT_ROW_POSE_MIN_PIXELS
    yaw_min_eigen_ratio: float = DEFAULT_ROW_YAW_MIN_EIGEN_RATIO
    stable_object_min_confidence: float = DEFAULT_STABLE_OBJECT_MIN_CONFIDENCE
    stable_object_min_support_count: int = DEFAULT_STABLE_OBJECT_MIN_SUPPORT_COUNT
    stable_object_min_evidence_score: float = DEFAULT_STABLE_OBJECT_MIN_EVIDENCE_SCORE
    stable_object_same_class_nms_radius_m: float = DEFAULT_STABLE_OBJECT_SAME_CLASS_NMS_RADIUS_M
    stable_object_workspace_margin_m: float = DEFAULT_STABLE_OBJECT_WORKSPACE_MARGIN_M
    tentative_object_min_confidence: float = DEFAULT_TENTATIVE_OBJECT_MIN_CONFIDENCE
    tentative_object_min_support_count: int = DEFAULT_TENTATIVE_OBJECT_MIN_SUPPORT_COUNT
    tentative_object_small_bbox_area_px: float = DEFAULT_TENTATIVE_OBJECT_SMALL_BBOX_AREA_PX
    cross_class_conflict_radius_m: float = DEFAULT_CROSS_CLASS_CONFLICT_RADIUS_M
    cross_class_ambiguity_score_ratio: float = DEFAULT_CROSS_CLASS_AMBIGUITY_SCORE_RATIO
    class_vote_ambiguity_top_to_second_ratio: float = DEFAULT_CLASS_VOTE_AMBIGUITY_TOP_TO_SECOND_RATIO
    class_vote_ambiguity_min_secondary_vote: float = DEFAULT_CLASS_VOTE_AMBIGUITY_MIN_SECONDARY_VOTE
    run_yolo: bool = True
    plan_only: bool = False
    strict_yolo: bool = False
    max_ik_iterations: int = 220
    ik_position_tolerance_m: float = 0.045
    ik_orientation_tolerance_rad: float = 0.35
    ik_damping: float = 1e-3

    def validate(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        if self.camera_z_m >= self.tank_opening_z_m:
            raise ValueError(
                "row wrist camera z must be below the tank upper opening height: "
                f"camera_z={self.camera_z_m:.4f}, tank_opening_z={self.tank_opening_z_m:.4f}"
            )
        if self.camera_z_m > self.tank_opening_z_m - self.opening_clearance_m:
            raise ValueError(
                "row wrist camera must keep clearance below the tank opening: "
                f"camera_z={self.camera_z_m:.4f}, max_camera_z={self.tank_opening_z_m - self.opening_clearance_m:.4f}"
            )
        if self.row_standoff_m <= 0.0:
            raise ValueError("row_standoff_m must be positive")
        if self.row_views_per_candidate <= 0:
            raise ValueError("row_views_per_candidate must be positive")
        if self.row_min_oblique_distance_m < 0.0:
            raise ValueError("row_min_oblique_distance_m must be non-negative")
        if self.look_at_height_offset_m < 0.0:
            raise ValueError("look_at_height_offset_m must be non-negative")
        if self.entry_side not in {"y-max", "y-min", "x-min", "x-max", "center", "survey-best"}:
            raise ValueError("entry_side must be one of y-max, y-min, x-min, x-max, center, or survey-best")
        if self.row_cluster_radius_m <= 0.0:
            raise ValueError("row_cluster_radius_m must be positive")
        if not self.row_view_candidate_standoff_multipliers:
            raise ValueError("row_view_candidate_standoff_multipliers must not be empty")
        if any(value <= 0.0 for value in self.row_view_candidate_standoff_multipliers):
            raise ValueError("row_view_candidate_standoff_multipliers must be positive")
        if not self.row_view_candidate_camera_z_offsets_m:
            raise ValueError("row_view_candidate_camera_z_offsets_m must not be empty")
        if not self.row_view_candidate_angle_offsets_rad:
            raise ValueError("row_view_candidate_angle_offsets_rad must not be empty")
        if not self.row_view_candidate_roll_offsets_rad:
            raise ValueError("row_view_candidate_roll_offsets_rad must not be empty")
        if self.row_view_candidate_max_attempts is not None and self.row_view_candidate_max_attempts <= 0:
            raise ValueError("row_view_candidate_max_attempts must be positive when set")
        if not 0.0 <= self.yolo_confidence <= 1.0:
            raise ValueError("yolo_confidence must be between 0 and 1")
        if self.yolo_tile_grid_size <= 0:
            raise ValueError("yolo_tile_grid_size must be positive")
        if self.depth_sample_stride_px <= 0:
            raise ValueError("depth_sample_stride_px must be positive")
        if self.depth_component_min_pixels <= 0:
            raise ValueError("depth_component_min_pixels must be positive")
        if self.pose_min_pixels <= 0:
            raise ValueError("pose_min_pixels must be positive")
        if self.yaw_min_eigen_ratio < 1.0:
            raise ValueError("yaw_min_eigen_ratio must be at least 1")
        if not 0.0 <= self.stable_object_min_confidence <= 1.0:
            raise ValueError("stable_object_min_confidence must be between 0 and 1")
        if self.stable_object_min_support_count <= 0:
            raise ValueError("stable_object_min_support_count must be positive")
        if self.stable_object_min_evidence_score <= 0.0:
            raise ValueError("stable_object_min_evidence_score must be positive")
        if self.stable_object_same_class_nms_radius_m <= 0.0:
            raise ValueError("stable_object_same_class_nms_radius_m must be positive")
        if self.stable_object_workspace_margin_m < 0.0:
            raise ValueError("stable_object_workspace_margin_m must be non-negative")
        if not 0.0 <= self.tentative_object_min_confidence <= 1.0:
            raise ValueError("tentative_object_min_confidence must be between 0 and 1")
        if self.tentative_object_min_support_count <= 0:
            raise ValueError("tentative_object_min_support_count must be positive")
        if self.tentative_object_small_bbox_area_px <= 0.0:
            raise ValueError("tentative_object_small_bbox_area_px must be positive")
        if self.cross_class_conflict_radius_m <= 0.0:
            raise ValueError("cross_class_conflict_radius_m must be positive")
        if not 0.0 <= self.cross_class_ambiguity_score_ratio <= 1.0:
            raise ValueError("cross_class_ambiguity_score_ratio must be between 0 and 1")
        if self.class_vote_ambiguity_top_to_second_ratio < 1.0:
            raise ValueError("class_vote_ambiguity_top_to_second_ratio must be at least 1")
        if self.class_vote_ambiguity_min_secondary_vote < 0.0:
            raise ValueError("class_vote_ambiguity_min_secondary_vote must be non-negative")

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
            run_yolo=self.run_yolo,
            plan_only=self.plan_only,
            strict_yolo=self.strict_yolo,
            max_ik_iterations=self.max_ik_iterations,
            ik_position_tolerance_m=self.ik_position_tolerance_m,
            ik_orientation_tolerance_rad=self.ik_orientation_tolerance_rad,
            ik_damping=self.ik_damping,
        )


@dataclass(frozen=True)
class RowPlannedView:
    candidate_id: str
    candidate_rough_position_world: tuple[float, float, float]
    approach_angle_rad: float
    view: SurveyView

    def to_dict(self) -> dict[str, Any]:
        payload = self.view.to_dict()
        payload.update(
            {
                "candidate_id": self.candidate_id,
                "candidate_rough_position_world": [
                    _round(value) for value in self.candidate_rough_position_world
                ],
                "approach_angle_rad": _round(self.approach_angle_rad),
            }
        )
        return payload


@dataclass(frozen=True)
class RowObservation:
    observation_id: str
    candidate_id: str
    row_view_id: str
    image_path: str | None
    depth_path: str | None
    yolo_raw_path: str | None
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int | None
    class_name: str | None
    position_world: tuple[float, float, float]
    yaw_rad: float | None
    yaw_confidence: float
    extent_xy_m: tuple[float, float] | None
    T_world_object: tuple[tuple[float, float, float, float], ...]
    position_source: str = "close RGB bbox ground-plane projection"

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "candidate_id": self.candidate_id,
            "row_view_id": self.row_view_id,
            "image_path": self.image_path,
            "depth_path": self.depth_path,
            "yolo_raw_path": self.yolo_raw_path,
            "bbox_xyxy": [_round(value) for value in self.bbox_xyxy],
            "confidence": _round(self.confidence),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "position_world": [_round(value) for value in self.position_world],
            "yaw_rad": _round(self.yaw_rad) if self.yaw_rad is not None else None,
            "yaw_confidence": _round(self.yaw_confidence),
            "extent_xy_m": [_round(value) for value in self.extent_xy_m] if self.extent_xy_m else None,
            "T_world_object": [list(row) for row in self.T_world_object],
            "position_source": self.position_source,
        }


def find_latest_survey_report(output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    root = Path(output_dir)
    candidates = sorted(root.glob("*/survey/survey_report.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no task1 survey_report.json files found under {root}")
    return candidates[-1]


def load_task1_survey_report(path: Path | str) -> Task1SurveyReport:
    report_path = Path(path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "task1_survey_report_v1":
        raise ValueError(f"unsupported task1 survey report schema_version: {payload.get('schema_version')}")
    layout_snapshot_path = _required_path(payload, "layout_snapshot_path")
    task1_run_dir = Path(str(payload.get("task1_run_dir") or report_path.parent.parent))
    survey_dir = Path(str(payload.get("survey_dir") or report_path.parent))
    workspace = payload.get("workspace")
    if not isinstance(workspace, dict):
        raise ValueError("task1 survey report must contain workspace")
    image_size = _image_size_from_payload(payload.get("image_size"))
    candidates_payload = payload.get("candidates")
    if not isinstance(candidates_payload, list):
        raise ValueError("task1 survey report must contain candidates list")
    views_payload = payload.get("views", [])
    if not isinstance(views_payload, list):
        raise ValueError("task1 survey report views must be a list when present")
    return Task1SurveyReport(
        path=report_path,
        schema_version=str(payload["schema_version"]),
        status=str(payload.get("status", "")),
        layout_snapshot_path=layout_snapshot_path,
        task1_run_dir=task1_run_dir,
        survey_dir=survey_dir,
        camera_name=str(payload.get("camera_name") or DEFAULT_CAMERA_NAME),
        workspace={key: float(value) for key, value in workspace.items()},
        image_size=image_size,
        candidates=tuple(_survey_candidate(candidate, index) for index, candidate in enumerate(candidates_payload)),
        views_by_id=_views_by_id(views_payload),
        raw_payload=payload,
    )


def build_row_inspection_plan(
    survey_report: Task1SurveyReport | Path | str,
    config: RowConfig = RowConfig(),
) -> tuple[SurveyWorkspace, tuple[RowPlannedView, ...]]:
    config.validate()
    report = load_task1_survey_report(survey_report) if not isinstance(survey_report, Task1SurveyReport) else survey_report
    layout = load_stage0_layout(report.layout_snapshot_path)
    scene = survey_scene_from_stage0_layout(
        layout,
        tank_opening_z_m=config.tank_opening_z_m,
        opening_clearance_m=config.opening_clearance_m,
    )
    return _build_row_inspection_plan_for_scene(report, scene=scene, config=config)


def run_task1_row(
    survey_report_path: Path | str,
    config: RowConfig = RowConfig(),
    *,
    detector: SurveyDetector | None = None,
) -> dict[str, Any]:
    config.validate()
    survey_report = load_task1_survey_report(survey_report_path)
    layout = load_stage0_layout(survey_report.layout_snapshot_path)
    scene = survey_scene_from_stage0_layout(
        layout,
        tank_opening_z_m=config.tank_opening_z_m,
        opening_clearance_m=config.opening_clearance_m,
    )
    workspace, planned_views = _build_row_inspection_plan_for_scene(survey_report, scene=scene, config=config)
    created_utc = datetime.now(timezone.utc).isoformat()
    run_dir = survey_report.task1_run_dir
    row_dir = run_dir / "row"
    images_dir = row_dir / "images"
    depth_dir = row_dir / "depth"
    yolo_dir = row_dir / "yolo_raw"
    annotated_dir = row_dir / "annotated"
    tiles_dir = row_dir / "tiles"
    plan_path = row_dir / "row_plan.json"
    report_path = row_dir / "row_report.json"
    plan_payload = _plan_payload(
        created_utc=created_utc,
        survey_report=survey_report,
        scene=scene,
        config=config,
        row_dir=row_dir,
        plan_path=plan_path,
        report_path=report_path,
        planned_views=planned_views,
    )

    if config.plan_only:
        _write_json(plan_path, plan_payload)
        report = _report_payload(
            created_utc=created_utc,
            status="plan_only",
            message="Generated the task1 row close-inspection plan without MuJoCo rendering or YOLO inference.",
            survey_report=survey_report,
            scene=scene,
            config=config,
            workspace=workspace,
            row_dir=row_dir,
            plan_path=plan_path,
            report_path=report_path,
            planned_views=planned_views,
            view_results=[_planned_row_view_result(planned) for planned in planned_views],
            row_observations=[],
            object_hypotheses=[],
            stable_objects=[],
            tentative_objects=[],
            ambiguous_objects=[],
            rejected_hypotheses=[],
            object_selection_summary=_stable_selection_empty_metadata(config),
        )
        _write_json(report_path, report)
        return report

    if config.run_yolo and detector is None:
        _write_json(plan_path, plan_payload)
        report = _report_payload(
            created_utc=created_utc,
            status="failed",
            message="run_yolo=True requires a SurveyDetector; pass one from the task1 entrypoint or set run_yolo=False.",
            survey_report=survey_report,
            scene=scene,
            config=config,
            workspace=workspace,
            row_dir=row_dir,
            plan_path=plan_path,
            report_path=report_path,
            planned_views=planned_views,
            view_results=[_planned_row_view_result(planned) for planned in planned_views],
            row_observations=[],
            object_hypotheses=[],
            stable_objects=[],
            tentative_objects=[],
            ambiguous_objects=[],
            rejected_hypotheses=[],
            object_selection_summary=_stable_selection_empty_metadata(config),
        )
        _write_json(report_path, report)
        return report

    backend = MujocoSurveyBackend(config=config.capture_config(), scene=scene, detector=detector)
    view_results: list[dict[str, Any]] = []
    row_observations: list[RowObservation] = []
    row_reachable_planning_summary: dict[str, Any] | None = None
    try:
        backend.load()
        backend.apply_scene_objects()
        planned_views, row_view_selections, row_reachable_planning_summary = _select_reachable_row_capture_plan(
            backend=backend,
            planned_views=planned_views,
            workspace=workspace,
            config=config,
        )
        plan_payload = _plan_payload(
            created_utc=created_utc,
            survey_report=survey_report,
            scene=scene,
            config=config,
            row_dir=row_dir,
            plan_path=plan_path,
            report_path=report_path,
            planned_views=planned_views,
        )
        for planned in planned_views:
            view_result, survey_observations = backend.capture_view(
                planned.view,
                images_dir=images_dir,
                depth_dir=depth_dir,
                yolo_dir=yolo_dir,
                annotated_dir=annotated_dir,
                tiles_dir=tiles_dir,
            )
            view_result["row_view_selection"] = row_view_selections.get(
                planned.view.view_id,
                {
                    "status": "missing_selection_record",
                    "strategy": "candidate_level_reachable_row_capture_plan_v1",
                    "planned_view_id": planned.view.view_id,
                    "notes": ["Internal row planner did not provide a selection record for this reachable view."],
                },
            )
            view_result["candidate_id"] = planned.candidate_id
            view_result["candidate_rough_position_world"] = [
                _round(value) for value in planned.candidate_rough_position_world
            ]
            view_results.append(view_result)
            row_observations.extend(
                _row_observations_from_capture(
                    planned=planned,
                    view_result=view_result,
                    survey_observations=survey_observations,
                    config=config,
                    workspace=workspace,
                    observation_offset=len(row_observations),
                )
            )
    except Exception as exc:
        _write_json(plan_path, plan_payload)
        report = _report_payload(
            created_utc=created_utc,
            status="failed",
            message=str(exc),
            survey_report=survey_report,
            scene=scene,
            config=config,
            workspace=workspace,
            row_dir=row_dir,
            plan_path=plan_path,
            report_path=report_path,
            planned_views=planned_views,
            view_results=view_results or [_planned_row_view_result(planned) for planned in planned_views],
            row_observations=[],
            object_hypotheses=[],
            stable_objects=[],
            tentative_objects=[],
            ambiguous_objects=[],
            rejected_hypotheses=[],
            object_selection_summary=_stable_selection_empty_metadata(config),
            row_reachable_planning_summary=row_reachable_planning_summary,
        )
        _write_json(report_path, report)
        return report
    finally:
        backend.close()

    object_hypotheses = fuse_row_observations(
        row_observations,
        cluster_radius_m=config.row_cluster_radius_m,
    )
    object_selection = select_row_objects_with_policy(
        object_hypotheses,
        workspace=workspace,
        min_confidence=config.stable_object_min_confidence,
        min_support_count=config.stable_object_min_support_count,
        min_evidence_score=config.stable_object_min_evidence_score,
        same_class_nms_radius_m=config.stable_object_same_class_nms_radius_m,
        workspace_margin_m=config.stable_object_workspace_margin_m,
        tentative_min_confidence=config.tentative_object_min_confidence,
        tentative_min_support_count=config.tentative_object_min_support_count,
        tentative_small_bbox_area_px=config.tentative_object_small_bbox_area_px,
        cross_class_conflict_radius_m=config.cross_class_conflict_radius_m,
        cross_class_ambiguity_score_ratio=config.cross_class_ambiguity_score_ratio,
        class_vote_ambiguity_top_to_second_ratio=config.class_vote_ambiguity_top_to_second_ratio,
        class_vote_ambiguity_min_secondary_vote=config.class_vote_ambiguity_min_secondary_vote,
    )
    stable_objects = object_selection["stable_objects"]
    tentative_objects = object_selection["tentative_objects"]
    ambiguous_objects = object_selection["ambiguous_objects"]
    rejected_hypotheses = object_selection["rejected_hypotheses"]
    object_selection_summary = object_selection["object_selection_summary"]
    success_count = sum(1 for view in view_results if view.get("status") == "success")
    planning_shortfall = (
        row_reachable_planning_summary is not None
        and int(row_reachable_planning_summary.get("insufficient_candidate_count", 0)) > 0
    )
    if view_results and success_count == len(view_results) and not planning_shortfall:
        status = "success"
    elif success_count > 0:
        status = "partial"
    else:
        status = "failed"
    _write_json(plan_path, plan_payload)
    report = _report_payload(
        created_utc=created_utc,
        status=status,
        message=(
            f"Captured {success_count}/{len(view_results)} task1 row close-inspection views "
            f"from {row_reachable_planning_summary.get('requested_view_count') if row_reachable_planning_summary else len(view_results)} requested reachable slots "
            f"and selected {len(stable_objects)} stable, {len(tentative_objects)} tentative, "
            f"{len(ambiguous_objects)} ambiguous objects from {len(object_hypotheses)} hypotheses."
        ),
        survey_report=survey_report,
        scene=scene,
        config=config,
        workspace=workspace,
        row_dir=row_dir,
        plan_path=plan_path,
        report_path=report_path,
        planned_views=planned_views,
        view_results=view_results,
        row_observations=[observation.to_dict() for observation in row_observations],
        object_hypotheses=object_hypotheses,
        stable_objects=stable_objects,
        tentative_objects=tentative_objects,
        ambiguous_objects=ambiguous_objects,
        rejected_hypotheses=rejected_hypotheses,
        object_selection_summary=object_selection_summary,
        row_reachable_planning_summary=row_reachable_planning_summary,
    )
    _write_json(report_path, report)
    return report


def fuse_row_observations(
    observations: list[RowObservation],
    *,
    cluster_radius_m: float = DEFAULT_ROW_CLUSTER_RADIUS_M,
) -> list[dict[str, Any]]:
    if cluster_radius_m <= 0.0:
        raise ValueError("cluster_radius_m must be positive")

    clusters: list[list[RowObservation]] = []
    for observation in observations:
        best_cluster: list[RowObservation] | None = None
        best_distance = float("inf")
        for cluster in clusters:
            distance = _xy_distance(observation.position_world, _row_cluster_center(cluster))
            if distance <= cluster_radius_m and distance < best_distance:
                best_cluster = cluster
                best_distance = distance
        if best_cluster is None:
            clusters.append([observation])
        else:
            best_cluster.append(observation)

    clusters.sort(key=lambda cluster: (_row_cluster_center(cluster)[0], _row_cluster_center(cluster)[1]))
    return [_row_hypothesis_payload(index, cluster) for index, cluster in enumerate(clusters)]


def select_row_objects(
    object_hypotheses: list[dict[str, Any]],
    *,
    workspace: SurveyWorkspace | dict[str, float],
    min_confidence: float = DEFAULT_STABLE_OBJECT_MIN_CONFIDENCE,
    min_support_count: int = DEFAULT_STABLE_OBJECT_MIN_SUPPORT_COUNT,
    min_evidence_score: float = DEFAULT_STABLE_OBJECT_MIN_EVIDENCE_SCORE,
    same_class_nms_radius_m: float = DEFAULT_STABLE_OBJECT_SAME_CLASS_NMS_RADIUS_M,
    workspace_margin_m: float = DEFAULT_STABLE_OBJECT_WORKSPACE_MARGIN_M,
) -> dict[str, Any]:
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1")
    if min_support_count <= 0:
        raise ValueError("min_support_count must be positive")
    if same_class_nms_radius_m <= 0.0:
        raise ValueError("same_class_nms_radius_m must be positive")
    return select_row_objects_with_policy(
        object_hypotheses,
        workspace=workspace,
        min_confidence=min_confidence,
        min_support_count=min_support_count,
        min_evidence_score=min_evidence_score,
        same_class_nms_radius_m=same_class_nms_radius_m,
        workspace_margin_m=workspace_margin_m,
        tentative_min_confidence=DEFAULT_TENTATIVE_OBJECT_MIN_CONFIDENCE,
        tentative_min_support_count=DEFAULT_TENTATIVE_OBJECT_MIN_SUPPORT_COUNT,
        tentative_small_bbox_area_px=DEFAULT_TENTATIVE_OBJECT_SMALL_BBOX_AREA_PX,
        cross_class_conflict_radius_m=DEFAULT_CROSS_CLASS_CONFLICT_RADIUS_M,
        cross_class_ambiguity_score_ratio=DEFAULT_CROSS_CLASS_AMBIGUITY_SCORE_RATIO,
        class_vote_ambiguity_top_to_second_ratio=DEFAULT_CLASS_VOTE_AMBIGUITY_TOP_TO_SECOND_RATIO,
        class_vote_ambiguity_min_secondary_vote=DEFAULT_CLASS_VOTE_AMBIGUITY_MIN_SECONDARY_VOTE,
    )


def select_row_objects_with_policy(
    object_hypotheses: list[dict[str, Any]],
    *,
    workspace: SurveyWorkspace | dict[str, float],
    min_confidence: float = DEFAULT_STABLE_OBJECT_MIN_CONFIDENCE,
    min_support_count: int = DEFAULT_STABLE_OBJECT_MIN_SUPPORT_COUNT,
    min_evidence_score: float = DEFAULT_STABLE_OBJECT_MIN_EVIDENCE_SCORE,
    same_class_nms_radius_m: float = DEFAULT_STABLE_OBJECT_SAME_CLASS_NMS_RADIUS_M,
    workspace_margin_m: float = DEFAULT_STABLE_OBJECT_WORKSPACE_MARGIN_M,
    tentative_min_confidence: float = DEFAULT_TENTATIVE_OBJECT_MIN_CONFIDENCE,
    tentative_min_support_count: int = DEFAULT_TENTATIVE_OBJECT_MIN_SUPPORT_COUNT,
    tentative_small_bbox_area_px: float = DEFAULT_TENTATIVE_OBJECT_SMALL_BBOX_AREA_PX,
    cross_class_conflict_radius_m: float = DEFAULT_CROSS_CLASS_CONFLICT_RADIUS_M,
    cross_class_ambiguity_score_ratio: float = DEFAULT_CROSS_CLASS_AMBIGUITY_SCORE_RATIO,
    class_vote_ambiguity_top_to_second_ratio: float = DEFAULT_CLASS_VOTE_AMBIGUITY_TOP_TO_SECOND_RATIO,
    class_vote_ambiguity_min_secondary_vote: float = DEFAULT_CLASS_VOTE_AMBIGUITY_MIN_SECONDARY_VOTE,
) -> dict[str, Any]:
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1")
    if min_support_count <= 0:
        raise ValueError("min_support_count must be positive")
    if min_evidence_score <= 0.0:
        raise ValueError("min_evidence_score must be positive")
    if same_class_nms_radius_m <= 0.0:
        raise ValueError("same_class_nms_radius_m must be positive")
    if workspace_margin_m < 0.0:
        raise ValueError("workspace_margin_m must be non-negative")
    if not 0.0 <= tentative_min_confidence <= 1.0:
        raise ValueError("tentative_min_confidence must be between 0 and 1")
    if tentative_min_support_count <= 0:
        raise ValueError("tentative_min_support_count must be positive")
    if tentative_small_bbox_area_px <= 0.0:
        raise ValueError("tentative_small_bbox_area_px must be positive")
    if cross_class_conflict_radius_m <= 0.0:
        raise ValueError("cross_class_conflict_radius_m must be positive")
    if not 0.0 <= cross_class_ambiguity_score_ratio <= 1.0:
        raise ValueError("cross_class_ambiguity_score_ratio must be between 0 and 1")
    if class_vote_ambiguity_top_to_second_ratio < 1.0:
        raise ValueError("class_vote_ambiguity_top_to_second_ratio must be at least 1")
    if class_vote_ambiguity_min_secondary_vote < 0.0:
        raise ValueError("class_vote_ambiguity_min_secondary_vote must be non-negative")

    bounds = _workspace_bounds(workspace)
    rejected_counts: dict[str, int] = {
        "missing_class": 0,
        "outside_workspace": 0,
        "low_confidence": 0,
        "low_support_count": 0,
        "same_class_duplicate": 0,
        "cross_class_conflict_weaker": 0,
        "tentative_near_selected_object": 0,
    }
    candidates: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []
    for hypothesis in object_hypotheses:
        reasons: list[str] = []
        class_name = str(hypothesis.get("class_name") or "").strip()
        if not class_name:
            reasons.append("missing_class")
        position = _position_from_hypothesis(hypothesis)
        if position is None or not _position_inside_workspace(position, bounds, margin_m=workspace_margin_m):
            reasons.append("outside_workspace")
        evidence_score = _stable_evidence_score(hypothesis)
        confidence = float(hypothesis.get("confidence", 0.0))
        if confidence < min_confidence and evidence_score < min_evidence_score:
            reasons.append("low_confidence")
        support_count = int(hypothesis.get("support_count", 0))
        if support_count < min_support_count:
            reasons.append("low_support_count")
        if reasons:
            _append_rejection_record(
                rejected_records,
                rejected_counts,
                hypothesis=hypothesis,
                reasons=reasons,
                evidence_score=evidence_score,
            )
            continue
        candidates.append(
            {
                "hypothesis": hypothesis,
                "evidence_score": evidence_score,
                "duplicate_hypothesis_ids": [],
            }
        )

    representatives: list[dict[str, Any]] = []
    suppressed_count = 0
    for record in sorted(candidates, key=lambda item: item["evidence_score"], reverse=True):
        duplicate_of = _same_class_duplicate_representative(
            record["hypothesis"],
            representatives,
            same_class_nms_radius_m=same_class_nms_radius_m,
        )
        if duplicate_of is not None:
            duplicate_of["duplicate_hypothesis_ids"].append(str(record["hypothesis"].get("hypothesis_id")))
            suppressed_count += 1
            _append_rejection_record(
                rejected_records,
                rejected_counts,
                hypothesis=record["hypothesis"],
                reasons=["same_class_duplicate"],
                evidence_score=record["evidence_score"],
            )
            continue
        representatives.append(record)

    confident_representatives: list[dict[str, Any]] = []
    ambiguous_groups: list[dict[str, Any]] = []
    for record in sorted(representatives, key=lambda item: item["evidence_score"], reverse=True):
        if _has_ambiguous_class_votes(
            record["hypothesis"],
            top_to_second_ratio=class_vote_ambiguity_top_to_second_ratio,
            min_secondary_vote=class_vote_ambiguity_min_secondary_vote,
        ):
            ambiguous_groups.append(
                {
                    "reason": "class_vote_ambiguity",
                    "representatives": [record],
                }
            )
            continue
        confident_representatives.append(record)

    selected: list[dict[str, Any]] = []
    for record in sorted(confident_representatives, key=lambda item: item["evidence_score"], reverse=True):
        ambiguous_conflict = _cross_class_ambiguous_conflict(
            record,
            ambiguous_groups,
            conflict_radius_m=cross_class_conflict_radius_m,
        )
        if ambiguous_conflict is not None:
            ratio = _score_similarity_ratio(record["evidence_score"], _ambiguous_group_max_score(ambiguous_conflict))
            if ratio >= cross_class_ambiguity_score_ratio:
                ambiguous_conflict["representatives"].append(record)
                continue
            _append_rejection_record(
                rejected_records,
                rejected_counts,
                hypothesis=record["hypothesis"],
                reasons=["cross_class_conflict_weaker"],
                evidence_score=record["evidence_score"],
            )
            continue

        selected_conflict = _cross_class_selected_conflict(
            record,
            selected,
            conflict_radius_m=cross_class_conflict_radius_m,
        )
        if selected_conflict is None:
            selected.append(record)
            continue
        ratio = _score_similarity_ratio(record["evidence_score"], selected_conflict["evidence_score"])
        if ratio >= cross_class_ambiguity_score_ratio:
            selected.remove(selected_conflict)
            ambiguous_groups.append(
                {
                    "reason": "cross_class_conflict_similar_evidence",
                    "representatives": [selected_conflict, record],
                }
            )
            continue
        _append_rejection_record(
            rejected_records,
            rejected_counts,
            hypothesis=record["hypothesis"],
            reasons=["cross_class_conflict_weaker"],
            evidence_score=record["evidence_score"],
        )

    selected.sort(key=lambda record: _representative_position(record)[:2])
    stable_objects = [
        _stable_object_payload(
            index,
            record["hypothesis"],
            evidence_score=record["evidence_score"],
            duplicate_hypothesis_ids=record["duplicate_hypothesis_ids"],
            workspace_bounds=bounds,
            workspace_margin_m=workspace_margin_m,
        )
        for index, record in enumerate(selected)
    ]

    ambiguous_groups.sort(key=lambda group: _ambiguous_group_center(group)[:2])
    ambiguous_objects = [
        _ambiguous_object_payload(
            index,
            group,
            conflict_radius_m=cross_class_conflict_radius_m,
            ambiguity_score_ratio=cross_class_ambiguity_score_ratio,
        )
        for index, group in enumerate(ambiguous_groups)
    ]

    tentative_records: list[dict[str, Any]] = []
    final_rejections: list[dict[str, Any]] = []
    for record in rejected_records:
        if _is_tentative_recovery_candidate(
            record,
            min_confidence=tentative_min_confidence,
            min_support_count=tentative_min_support_count,
            small_bbox_area_px=tentative_small_bbox_area_px,
        ):
            if _near_selected_or_ambiguous(
                record["hypothesis"],
                selected=selected,
                ambiguous_groups=ambiguous_groups,
                conflict_radius_m=cross_class_conflict_radius_m,
            ):
                record = dict(record)
                record["reasons"] = _unique_reasons(
                    [*record["reasons"], "tentative_near_selected_object"]
                )
                rejected_counts["tentative_near_selected_object"] += 1
                final_rejections.append(record)
                continue
            tentative_records.append(record)
            continue
        final_rejections.append(record)

    tentative_records.sort(key=lambda record: _representative_position(record)[:2])
    tentative_objects = [
        _tentative_object_payload(
            index,
            record["hypothesis"],
            evidence_score=record["evidence_score"],
            evidence_gaps=record["reasons"],
            min_confidence=tentative_min_confidence,
            min_support_count=tentative_min_support_count,
            small_bbox_area_px=tentative_small_bbox_area_px,
        )
        for index, record in enumerate(tentative_records)
    ]
    rejected_hypotheses = [_rejected_hypothesis_payload(record) for record in final_rejections]
    metadata = {
        "status": "success",
        "policy_version": "row_object_selection_policy_v2",
        "input_hypothesis_count": len(object_hypotheses),
        "candidate_after_filter_count": len(candidates),
        "same_class_representative_count": len(representatives),
        "confident_representative_count": len(confident_representatives),
        "stable_object_count": len(stable_objects),
        "tentative_object_count": len(tentative_objects),
        "ambiguous_object_count": len(ambiguous_objects),
        "class_vote_ambiguous_count": sum(1 for group in ambiguous_groups if group["reason"] == "class_vote_ambiguity"),
        "rejected_hypothesis_count": len(rejected_hypotheses),
        "suppressed_same_class_duplicate_count": suppressed_count,
        "rejected_counts": rejected_counts,
        "thresholds": {
            "min_confidence": _round(min_confidence),
            "min_support_count": min_support_count,
            "min_evidence_score": _round(min_evidence_score),
            "same_class_nms_radius_m": _round(same_class_nms_radius_m),
            "workspace_margin_m": _round(workspace_margin_m),
            "tentative_min_confidence": _round(tentative_min_confidence),
            "tentative_min_support_count": tentative_min_support_count,
            "tentative_small_bbox_area_px": _round(tentative_small_bbox_area_px),
            "cross_class_conflict_radius_m": _round(cross_class_conflict_radius_m),
            "cross_class_ambiguity_score_ratio": _round(cross_class_ambiguity_score_ratio),
            "class_vote_ambiguity_top_to_second_ratio": _round(class_vote_ambiguity_top_to_second_ratio),
            "class_vote_ambiguity_min_secondary_vote": _round(class_vote_ambiguity_min_secondary_vote),
            "workspace_filter": True,
        },
        "notes": [
            "Stable objects are selected from close RGB-D row hypotheses, not copied from survey candidates.",
            "Lower-scoring same-class hypotheses within the merge radius are treated as duplicate close-view fragments.",
            "Small low-evidence hypotheses are retained as tentative objects instead of being silently dropped.",
            "Nearby cross-class hypotheses with similar evidence are reported as ambiguous objects instead of forced labels.",
            "Single hypotheses with close top-two class votes are reported as ambiguous instead of forced labels.",
        ],
    }
    return {
        "stable_objects": stable_objects,
        "tentative_objects": tentative_objects,
        "ambiguous_objects": ambiguous_objects,
        "rejected_hypotheses": rejected_hypotheses,
        "object_selection_summary": metadata,
    }


def select_stable_row_objects(
    object_hypotheses: list[dict[str, Any]],
    *,
    workspace: SurveyWorkspace | dict[str, float],
    min_confidence: float = DEFAULT_STABLE_OBJECT_MIN_CONFIDENCE,
    min_support_count: int = DEFAULT_STABLE_OBJECT_MIN_SUPPORT_COUNT,
    same_class_nms_radius_m: float = DEFAULT_STABLE_OBJECT_SAME_CLASS_NMS_RADIUS_M,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selection = select_row_objects(
        object_hypotheses,
        workspace=workspace,
        min_confidence=min_confidence,
        min_support_count=min_support_count,
        same_class_nms_radius_m=same_class_nms_radius_m,
    )
    return selection["stable_objects"], selection["object_selection_summary"]


def _select_row_capture_view(
    *,
    backend: MujocoSurveyBackend,
    planned: RowPlannedView,
    workspace: SurveyWorkspace,
    config: RowConfig,
) -> tuple[SurveyView | None, dict[str, Any]]:
    if not config.row_view_collision_search:
        return planned.view, {
            "status": "disabled",
            "strategy": "nominal_row_view_without_candidate_search",
            "planned_view_id": planned.view.view_id,
            "selected_candidate_index": 0,
            "attempt_count": 0,
            "notes": [
                "Row view collision-aware candidate search was disabled by configuration.",
            ],
        }

    candidate_views, candidate_limit_reached = _row_view_candidate_views(planned, workspace=workspace, config=config)
    attempts: list[dict[str, Any]] = []
    rejected_counts = {
        "ik_failed": 0,
        "collision": 0,
        "camera_constraint": 0,
        "other_failed": 0,
    }
    for candidate_index, (candidate_view, candidate_metadata) in enumerate(candidate_views):
        validation = backend.validate_view_pose(candidate_view)
        attempt = _row_view_validation_attempt_summary(
            candidate_index=candidate_index,
            metadata=candidate_metadata,
            validation=validation,
        )
        attempts.append(attempt)
        if validation.get("status") == "success":
            return candidate_view, {
                "status": "selected",
                "strategy": "ik_collision_checked_row_view_candidates_v1",
                "planned_view_id": planned.view.view_id,
                "selected_candidate_index": candidate_index,
                "attempt_count": len(attempts),
                "candidate_limit_reached": candidate_limit_reached,
                "rejected_counts": rejected_counts,
                "selected_candidate": candidate_metadata,
                "attempts": attempts,
                "notes": [
                    "Row chose the first candidate whose full MuJoCo IK pose passed robot collision checks.",
                ],
            }
        _accumulate_row_view_rejection(rejected_counts, validation)

    return None, {
        "status": "failed_no_collision_free_candidate",
        "strategy": "ik_collision_checked_row_view_candidates_v1",
        "planned_view_id": planned.view.view_id,
        "selected_candidate_index": None,
        "attempt_count": len(attempts),
        "candidate_limit_reached": candidate_limit_reached,
        "rejected_counts": rejected_counts,
        "attempts": attempts,
        "notes": [
            "No row view candidate passed IK and full robot collision checks; this planned view was not rendered.",
        ],
    }


def _select_reachable_row_capture_plan(
    *,
    backend: MujocoSurveyBackend,
    planned_views: tuple[RowPlannedView, ...],
    workspace: SurveyWorkspace,
    config: RowConfig,
) -> tuple[tuple[RowPlannedView, ...], dict[str, dict[str, Any]], dict[str, Any]]:
    if not config.row_view_collision_search:
        selections = {
            planned.view.view_id: {
                "status": "disabled",
                "strategy": "nominal_row_view_without_candidate_search",
                "planned_view_id": planned.view.view_id,
                "selected_candidate_index": 0,
                "attempt_count": 0,
                "notes": [
                    "Debug-only row view collision-aware planning was disabled; reachability is not guaranteed.",
                ],
            }
            for planned in planned_views
        }
        return planned_views, selections, {
            "strategy": "candidate_level_reachable_row_capture_plan_v1",
            "enabled": False,
            "requested_view_count": len(planned_views),
            "selected_view_count": len(planned_views),
            "insufficient_candidate_count": 0,
            "candidate_summaries": [],
            "notes": [
                "Reachable row capture planning was disabled by configuration.",
            ],
        }

    selected_views: list[RowPlannedView] = []
    selections: dict[str, dict[str, Any]] = {}
    candidate_summaries: list[dict[str, Any]] = []
    for candidate_id, candidate_plans in _group_row_plans_by_candidate(planned_views):
        selected_for_candidate: list[RowPlannedView] = []
        rejected_counts = {
            "ik_failed": 0,
            "collision": 0,
            "camera_constraint": 0,
            "other_failed": 0,
            "missing_validated_qpos": 0,
        }
        failed_attempt_samples: list[dict[str, Any]] = []
        attempted_pose_keys: set[tuple[float, ...]] = set()
        candidate_limit_reached = False
        duplicate_success_count = 0
        attempt_count = 0
        requested_count = len(candidate_plans)
        primary_missed_view_ids: list[str] = []
        primary_selected_count = 0
        secondary_selected_count = 0

        def try_select_candidate_view(
            *,
            source_planned: RowPlannedView,
            candidate_index: int,
            candidate_view: SurveyView,
            metadata: dict[str, Any],
            planning_pass: str,
        ) -> bool:
            nonlocal attempt_count, duplicate_success_count, primary_selected_count, secondary_selected_count
            pose_key = _row_view_pose_key(candidate_view)
            if pose_key in attempted_pose_keys:
                return False
            attempted_pose_keys.add(pose_key)
            validation = backend.validate_view_pose(candidate_view)
            attempt_count += 1
            attempt = _row_view_validation_attempt_summary(
                candidate_index=attempt_count - 1,
                metadata={
                    **metadata,
                    "source_planned_view_id": source_planned.view.view_id,
                    "source_candidate_index": candidate_index,
                    "planning_pass": planning_pass,
                },
                validation=validation,
            )
            if validation.get("status") != "success":
                _accumulate_row_view_rejection(rejected_counts, validation)
                if len(failed_attempt_samples) < 8:
                    failed_attempt_samples.append(attempt)
                return False
            actual_qpos = validation.get("actual_qpos")
            if not isinstance(actual_qpos, list) or not actual_qpos:
                rejected_counts["missing_validated_qpos"] += 1
                attempt["status"] = "failed"
                attempt["message"] = "Validated row candidate did not include actual_qpos for reproducible capture."
                if len(failed_attempt_samples) < 8:
                    failed_attempt_samples.append(attempt)
                return False
            if _duplicates_selected_camera_position(candidate_view, selected_for_candidate):
                duplicate_success_count += 1
                return False

            local_index = len(selected_for_candidate)
            selected_view_id = f"row_{candidate_id}_{local_index:02d}"
            selected_view = _copy_row_survey_view(
                candidate_view,
                view_id=selected_view_id,
                grid_row=source_planned.view.grid_row,
                grid_col=local_index,
                fixed_pose_source=ROW_REACHABLE_CAPTURE_FIXED_POSE_SOURCE,
                fixed_qpos=tuple(float(value) for value in actual_qpos),
            )
            selected_planned = RowPlannedView(
                candidate_id=candidate_id,
                candidate_rough_position_world=source_planned.candidate_rough_position_world,
                approach_angle_rad=float(metadata["approach_angle_rad"]),
                view=selected_view,
            )
            selected_for_candidate.append(selected_planned)
            if planning_pass == "primary_slot":
                primary_selected_count += 1
            else:
                secondary_selected_count += 1
            selections[selected_view_id] = {
                "status": "selected",
                "strategy": "candidate_level_reachable_row_capture_plan_v1",
                "planned_view_id": selected_view_id,
                "source_planned_view_id": source_planned.view.view_id,
                "selected_candidate_index": attempt_count - 1,
                "attempt_count": attempt_count,
                "candidate_limit_reached": candidate_limit_reached,
                "planning_pass": planning_pass,
                "selected_candidate": {
                    **metadata,
                    "source_planned_view_id": source_planned.view.view_id,
                    "source_candidate_index": candidate_index,
                    "planning_pass": planning_pass,
                },
                "validation": attempt,
                "notes": [
                    "Row planned this capture only after the full MuJoCo IK pose passed robot collision checks.",
                ],
            }
            return True

        for source_planned in candidate_plans:
            candidate_views, limit_reached = _row_view_candidate_views(
                source_planned,
                workspace=workspace,
                config=config,
            )
            candidate_limit_reached = candidate_limit_reached or limit_reached
            slot_selected = False
            for candidate_index, (candidate_view, metadata) in enumerate(candidate_views):
                slot_selected = try_select_candidate_view(
                    source_planned=source_planned,
                    candidate_index=candidate_index,
                    candidate_view=candidate_view,
                    metadata=metadata,
                    planning_pass="primary_slot",
                )
                if slot_selected:
                    break
            if not slot_selected:
                primary_missed_view_ids.append(source_planned.view.view_id)
            if len(selected_for_candidate) >= requested_count:
                break

        if len(selected_for_candidate) < requested_count:
            for source_planned in candidate_plans:
                candidate_views, limit_reached = _row_view_candidate_views(
                    source_planned,
                    workspace=workspace,
                    config=config,
                )
                candidate_limit_reached = candidate_limit_reached or limit_reached
                for candidate_index, (candidate_view, metadata) in enumerate(candidate_views):
                    selected = try_select_candidate_view(
                        source_planned=source_planned,
                        candidate_index=candidate_index,
                        candidate_view=candidate_view,
                        metadata=metadata,
                        planning_pass="secondary_fill",
                    )
                    if selected and len(selected_for_candidate) >= requested_count:
                        break
                if len(selected_for_candidate) >= requested_count:
                    break

        selected_views.extend(selected_for_candidate)
        candidate_summaries.append(
            {
                "candidate_id": candidate_id,
                "status": (
                    "selected_requested_view_count"
                    if len(selected_for_candidate) >= requested_count
                    else "insufficient_reachable_views"
                ),
                "requested_view_count": requested_count,
                "selected_view_count": len(selected_for_candidate),
                "primary_selected_view_count": primary_selected_count,
                "secondary_selected_view_count": secondary_selected_count,
                "primary_missed_view_ids": primary_missed_view_ids,
                "attempt_count": attempt_count,
                "unique_pose_attempt_count": len(attempted_pose_keys),
                "duplicate_success_count": duplicate_success_count,
                "candidate_limit_reached": candidate_limit_reached,
                "rejected_counts": rejected_counts,
                "selected_view_ids": [planned.view.view_id for planned in selected_for_candidate],
                "failed_attempt_samples": failed_attempt_samples,
            }
        )

    insufficient_count = sum(
        1 for summary in candidate_summaries if summary["status"] == "insufficient_reachable_views"
    )
    return tuple(selected_views), selections, {
        "strategy": "candidate_level_reachable_row_capture_plan_v1",
        "enabled": True,
        "requested_view_count": len(planned_views),
        "selected_view_count": len(selected_views),
        "candidate_count": len(candidate_summaries),
        "insufficient_candidate_count": insufficient_count,
        "candidate_limit_reached_count": sum(1 for summary in candidate_summaries if summary["candidate_limit_reached"]),
        "candidate_summaries": candidate_summaries,
        "notes": [
            "Only IK/collision-validated row views are included in planned_views and rendered.",
            "If selected_view_count is lower than requested_view_count, the report is partial instead of rendering an unreachable row view.",
        ],
    }


def _row_view_candidate_views(
    planned: RowPlannedView,
    *,
    workspace: SurveyWorkspace,
    config: RowConfig,
) -> tuple[list[tuple[SurveyView, dict[str, Any]]], bool]:
    target = planned.candidate_rough_position_world
    look_at = (
        target[0],
        target[1],
        min(workspace.max_camera_z_m, workspace.bottom_z_m + config.look_at_height_offset_m),
    )
    candidates: list[tuple[SurveyView, dict[str, Any]]] = []
    seen: set[tuple[float, float, float, float]] = set()
    for angle_offset in config.row_view_candidate_angle_offsets_rad:
        angle = _normalize_angle(planned.approach_angle_rad + angle_offset)
        for z_offset in config.row_view_candidate_camera_z_offsets_m:
            camera_z = config.camera_z_m + z_offset
            for standoff_multiplier in config.row_view_candidate_standoff_multipliers:
                standoff = config.row_standoff_m * standoff_multiplier
                camera_position = (
                    target[0] + math.cos(angle) * standoff,
                    target[1] + math.sin(angle) * standoff,
                    camera_z,
                )
                if not _camera_position_is_valid(camera_position, target, workspace=workspace, config=config):
                    continue
                key = (
                    _round(camera_position[0]),
                    _round(camera_position[1]),
                    _round(camera_position[2]),
                    _round(angle),
                )
                if key in seen:
                    continue
                seen.add(key)
                base_transform = _make_look_at_transform(camera_position, look_at)
                for roll_offset in config.row_view_candidate_roll_offsets_rad:
                    view = SurveyView(
                        view_id=planned.view.view_id,
                        grid_row=planned.view.grid_row,
                        grid_col=planned.view.grid_col,
                        camera_name=planned.view.camera_name,
                        desired_camera_position_world=_round_vector(camera_position),
                        look_at_world=_round_vector(look_at),
                        T_world_camera=_roll_camera_transform(base_transform, roll_offset),
                        scan_layer=planned.view.scan_layer,
                        scan_grid_size=planned.view.scan_grid_size,
                    )
                    metadata = {
                        "view_id": planned.view.view_id,
                        "angle_offset_rad": _round(angle_offset),
                        "roll_offset_rad": _round(roll_offset),
                        "approach_angle_rad": _round(angle),
                        "standoff_multiplier": _round(standoff_multiplier),
                        "standoff_m": _round(standoff),
                        "camera_z_offset_m": _round(z_offset),
                        "desired_camera_position_world": [_round(value) for value in view.desired_camera_position_world],
                        "look_at_world": [_round(value) for value in view.look_at_world],
                    }
                    candidates.append((view, metadata))
                    if (
                        config.row_view_candidate_max_attempts is not None
                        and len(candidates) >= config.row_view_candidate_max_attempts
                    ):
                        return candidates, True
    return candidates, False


def _group_row_plans_by_candidate(
    planned_views: tuple[RowPlannedView, ...],
) -> list[tuple[str, list[RowPlannedView]]]:
    groups: list[tuple[str, list[RowPlannedView]]] = []
    by_id: dict[str, list[RowPlannedView]] = {}
    for planned in planned_views:
        if planned.candidate_id not in by_id:
            by_id[planned.candidate_id] = []
            groups.append((planned.candidate_id, by_id[planned.candidate_id]))
        by_id[planned.candidate_id].append(planned)
    return groups


def _row_view_pose_key(view: SurveyView) -> tuple[float, ...]:
    return tuple(
        _round(value)
        for value in (
            *view.desired_camera_position_world,
            *(component for row in view.T_world_camera[:3] for component in row[:3]),
        )
    )


def _duplicates_selected_camera_position(
    view: SurveyView,
    selected_plans: list[RowPlannedView],
    *,
    min_distance_m: float = 0.025,
) -> bool:
    return any(
        _xy_distance(view.desired_camera_position_world, selected.view.desired_camera_position_world) < min_distance_m
        for selected in selected_plans
    )


def _copy_row_survey_view(
    view: SurveyView,
    *,
    view_id: str,
    grid_row: int,
    grid_col: int,
    fixed_pose_source: str | None = None,
    fixed_qpos: tuple[float, ...] | None = None,
) -> SurveyView:
    return SurveyView(
        view_id=view_id,
        grid_row=grid_row,
        grid_col=grid_col,
        camera_name=view.camera_name,
        desired_camera_position_world=view.desired_camera_position_world,
        look_at_world=view.look_at_world,
        T_world_camera=view.T_world_camera,
        scan_layer=view.scan_layer,
        scan_grid_size=view.scan_grid_size,
        fixed_pose_source=fixed_pose_source if fixed_pose_source is not None else view.fixed_pose_source,
        fixed_qpos=fixed_qpos if fixed_qpos is not None else view.fixed_qpos,
    )


def _row_view_validation_attempt_summary(
    *,
    candidate_index: int,
    metadata: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    collision = validation.get("collision") if isinstance(validation.get("collision"), dict) else {}
    ik = validation.get("ik") if isinstance(validation.get("ik"), dict) else {}
    return {
        "candidate_index": candidate_index,
        "status": validation.get("status"),
        "message": validation.get("message"),
        "candidate": metadata,
        "actual_camera_position_world": validation.get("actual_camera_position_world"),
        "ik": {
            "success": ik.get("success"),
            "iterations": ik.get("iterations"),
            "position_error_m": ik.get("position_error_m"),
            "orientation_error_rad": ik.get("orientation_error_rad"),
        },
        "collision": {
            "collision_free": collision.get("collision_free"),
            "robot_scene_contact_count": collision.get("robot_scene_contact_count"),
            "robot_self_contact_count": collision.get("robot_self_contact_count"),
            "min_robot_scene_distance_m": collision.get("min_robot_scene_distance_m"),
            "min_robot_self_distance_m": collision.get("min_robot_self_distance_m"),
            "robot_scene_contacts": collision.get("robot_scene_contacts", [])[:3],
            "robot_self_contacts": collision.get("robot_self_contacts", [])[:3],
        },
    }


def _accumulate_row_view_rejection(rejected_counts: dict[str, int], validation: dict[str, Any]) -> None:
    ik = validation.get("ik") if isinstance(validation.get("ik"), dict) else {}
    collision = validation.get("collision") if isinstance(validation.get("collision"), dict) else {}
    message = str(validation.get("message") or "")
    if ik and not ik.get("success", False):
        rejected_counts["ik_failed"] += 1
    elif collision and not collision.get("collision_free", True):
        rejected_counts["collision"] += 1
    elif "camera" in message and ("inside" in message or "opening" in message or "workspace" in message):
        rejected_counts["camera_constraint"] += 1
    else:
        rejected_counts["other_failed"] += 1


def _row_view_selection_failed_result(planned: RowPlannedView, row_view_selection: dict[str, Any]) -> dict[str, Any]:
    result = _planned_row_view_result(planned)
    result["status"] = "failed"
    result["message"] = "No IK/collision-safe row view candidate was available; row capture was not rendered."
    result["candidate_id"] = planned.candidate_id
    result["candidate_rough_position_world"] = [
        _round(value) for value in planned.candidate_rough_position_world
    ]
    result["row_view_selection"] = row_view_selection
    return result


def _append_rejection_record(
    records: list[dict[str, Any]],
    rejected_counts: dict[str, int],
    *,
    hypothesis: dict[str, Any],
    reasons: list[str],
    evidence_score: float,
) -> None:
    unique = _unique_reasons(reasons)
    records.append(
        {
            "hypothesis": hypothesis,
            "reasons": unique,
            "evidence_score": float(evidence_score),
        }
    )
    for reason in unique:
        rejected_counts[reason] = rejected_counts.get(reason, 0) + 1


def _unique_reasons(reasons: list[str]) -> list[str]:
    unique: list[str] = []
    for reason in reasons:
        if reason not in unique:
            unique.append(reason)
    return unique


def _same_class_duplicate_representative(
    hypothesis: dict[str, Any],
    representatives: list[dict[str, Any]],
    *,
    same_class_nms_radius_m: float,
) -> dict[str, Any] | None:
    class_name = str(hypothesis.get("class_name") or "")
    position = _position_from_hypothesis(hypothesis)
    if position is None:
        return None
    for representative in representatives:
        selected_hypothesis = representative["hypothesis"]
        if str(selected_hypothesis.get("class_name") or "") != class_name:
            continue
        selected_position = _position_from_hypothesis(selected_hypothesis)
        if selected_position is None:
            continue
        if _xy_distance(position, selected_position) <= same_class_nms_radius_m:
            return representative
    return None


def _cross_class_selected_conflict(
    record: dict[str, Any],
    selected: list[dict[str, Any]],
    *,
    conflict_radius_m: float,
) -> dict[str, Any] | None:
    position = _representative_position(record)
    class_name = _representative_class_name(record)
    for selected_record in selected:
        if _representative_class_name(selected_record) == class_name:
            continue
        if _xy_distance(position, _representative_position(selected_record)) <= conflict_radius_m:
            return selected_record
    return None


def _cross_class_ambiguous_conflict(
    record: dict[str, Any],
    ambiguous_groups: list[dict[str, Any]],
    *,
    conflict_radius_m: float,
) -> dict[str, Any] | None:
    position = _representative_position(record)
    class_name = _representative_class_name(record)
    for group in ambiguous_groups:
        group_classes = {_representative_class_name(item) for item in group["representatives"]}
        if class_name in group_classes:
            continue
        if _xy_distance(position, _ambiguous_group_center(group)) <= conflict_radius_m:
            return group
    return None


def _score_similarity_ratio(left: float, right: float) -> float:
    larger = max(float(left), float(right))
    if larger <= 1e-12:
        return 1.0
    return min(float(left), float(right)) / larger


def _has_ambiguous_class_votes(
    hypothesis: dict[str, Any],
    *,
    top_to_second_ratio: float,
    min_secondary_vote: float,
) -> bool:
    ranked = _ranked_class_votes(hypothesis)
    if len(ranked) < 2:
        return False
    top_vote = ranked[0][1]
    second_vote = ranked[1][1]
    if second_vote < min_secondary_vote:
        return False
    if top_vote <= 1e-12:
        return False
    return (top_vote / second_vote) <= top_to_second_ratio


def _ranked_class_votes(hypothesis: dict[str, Any]) -> list[tuple[str, float]]:
    class_votes = hypothesis.get("class_votes")
    if not isinstance(class_votes, dict):
        return []
    ranked = [
        (str(class_name), float(vote))
        for class_name, vote in class_votes.items()
        if class_name is not None and float(vote) > 0.0
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def _ambiguous_group_max_score(group: dict[str, Any]) -> float:
    return max(float(record["evidence_score"]) for record in group["representatives"])


def _representative_class_name(record: dict[str, Any]) -> str:
    return str(record["hypothesis"].get("class_name") or "")


def _representative_position(record: dict[str, Any]) -> tuple[float, float, float]:
    position = _position_from_hypothesis(record["hypothesis"])
    if position is None:
        return (float("inf"), float("inf"), float("inf"))
    return position


def _ambiguous_group_center(group: dict[str, Any]) -> tuple[float, float, float]:
    return _weighted_record_center(group["representatives"])


def _weighted_record_center(records: list[dict[str, Any]]) -> tuple[float, float, float]:
    total_weight = sum(max(float(record["evidence_score"]), 1e-6) for record in records)
    if total_weight <= 0.0:
        total_weight = float(len(records))
    return (
        sum(_representative_position(record)[0] * max(float(record["evidence_score"]), 1e-6) for record in records)
        / total_weight,
        sum(_representative_position(record)[1] * max(float(record["evidence_score"]), 1e-6) for record in records)
        / total_weight,
        sum(_representative_position(record)[2] * max(float(record["evidence_score"]), 1e-6) for record in records)
        / total_weight,
    )


def _is_tentative_recovery_candidate(
    record: dict[str, Any],
    *,
    min_confidence: float,
    min_support_count: int,
    small_bbox_area_px: float,
) -> bool:
    reasons = set(record["reasons"])
    if not reasons.intersection({"low_confidence", "low_support_count"}):
        return False
    if reasons.difference({"low_confidence", "low_support_count"}):
        return False
    hypothesis = record["hypothesis"]
    if _position_from_hypothesis(hypothesis) is None:
        return False
    if not str(hypothesis.get("class_name") or "").strip():
        return False
    if float(hypothesis.get("confidence", 0.0)) < min_confidence:
        return False
    if int(hypothesis.get("support_count", 0)) < min_support_count:
        return False
    bbox_area = _hypothesis_bbox_area(hypothesis)
    return bbox_area is not None and bbox_area <= small_bbox_area_px


def _near_selected_or_ambiguous(
    hypothesis: dict[str, Any],
    *,
    selected: list[dict[str, Any]],
    ambiguous_groups: list[dict[str, Any]],
    conflict_radius_m: float,
) -> bool:
    position = _position_from_hypothesis(hypothesis)
    if position is None:
        return True
    for record in selected:
        if _xy_distance(position, _representative_position(record)) <= conflict_radius_m:
            return True
    for group in ambiguous_groups:
        if _xy_distance(position, _ambiguous_group_center(group)) <= conflict_radius_m:
            return True
    return False


def _ambiguous_object_payload(
    index: int,
    group: dict[str, Any],
    *,
    conflict_radius_m: float,
    ambiguity_score_ratio: float,
) -> dict[str, Any]:
    representatives = sorted(group["representatives"], key=lambda record: record["evidence_score"], reverse=True)
    center = _ambiguous_group_center(group)
    if group["reason"] == "class_vote_ambiguity":
        class_candidates = [
            candidate
            for record in representatives
            for candidate in _class_vote_candidate_payloads(record)
        ]
    else:
        class_candidates = [_class_candidate_payload(record) for record in representatives]
    return {
        "object_id": f"row_ambiguous_{index + 1:03d}",
        "status": "ambiguous",
        "position_world": [_round(value) for value in center],
        "class_candidates": class_candidates,
        "selection": {
            "source": "row_object_selection_policy_v2",
            "reason": group["reason"],
            "cross_class_conflict_radius_m": _round(conflict_radius_m),
            "ambiguity_score_ratio": _round(ambiguity_score_ratio),
        },
        "notes": [
            "Close RGB-D evidence found nearby hypotheses with different classes and similar scores.",
            "This object is intentionally excluded from stable_objects until another view or later stage resolves the class.",
        ],
    }


def _class_vote_candidate_payloads(record: dict[str, Any]) -> list[dict[str, Any]]:
    hypothesis = record["hypothesis"]
    ranked = _ranked_class_votes(hypothesis)
    total_vote = sum(vote for _, vote in ranked)
    candidates: list[dict[str, Any]] = []
    for class_name, vote in ranked:
        payload = _class_candidate_payload(record)
        payload["class_name"] = class_name
        payload["class_vote"] = _round(vote)
        payload["class_vote_fraction"] = _round(vote / total_vote) if total_vote > 0.0 else 0.0
        payload["hypothesis_class_name"] = str(hypothesis.get("class_name") or "")
        candidates.append(payload)
    return candidates


def _class_candidate_payload(record: dict[str, Any]) -> dict[str, Any]:
    hypothesis = record["hypothesis"]
    return {
        "source_hypothesis_id": hypothesis.get("hypothesis_id"),
        "suppressed_duplicate_hypothesis_ids": record.get("duplicate_hypothesis_ids", []),
        "class_name": str(hypothesis.get("class_name") or ""),
        "confidence": _round(float(hypothesis.get("confidence", 0.0))),
        "evidence_score": _round(float(record["evidence_score"])),
        "support_count": int(hypothesis.get("support_count", 0)),
        "source_candidate_ids": _string_list(hypothesis.get("source_candidate_ids", [])),
        "supporting_row_views": _string_list(hypothesis.get("supporting_row_views", [])),
        "position_world": [_round(value) for value in _representative_position(record)],
        "best_image_path": hypothesis.get("best_image_path"),
        "best_bbox_xyxy": hypothesis.get("best_bbox_xyxy"),
    }


def _tentative_object_payload(
    index: int,
    hypothesis: dict[str, Any],
    *,
    evidence_score: float,
    evidence_gaps: list[str],
    min_confidence: float,
    min_support_count: int,
    small_bbox_area_px: float,
) -> dict[str, Any]:
    position = _position_from_hypothesis(hypothesis)
    if position is None:
        raise ValueError("tentative object hypothesis must contain position_world")
    return {
        "object_id": f"row_tentative_{index + 1:03d}",
        "status": "tentative",
        "source_hypothesis_id": hypothesis.get("hypothesis_id"),
        "class_name": str(hypothesis.get("class_name") or ""),
        "confidence": _round(float(hypothesis.get("confidence", 0.0))),
        "evidence_score": _round(evidence_score),
        "support_count": int(hypothesis.get("support_count", 0)),
        "source_candidate_ids": _string_list(hypothesis.get("source_candidate_ids", [])),
        "supporting_row_views": _string_list(hypothesis.get("supporting_row_views", [])),
        "position_world": [_round(value) for value in position],
        "yaw_rad": hypothesis.get("yaw_rad"),
        "T_world_object": hypothesis.get("T_world_object"),
        "best_image_path": hypothesis.get("best_image_path"),
        "best_bbox_xyxy": hypothesis.get("best_bbox_xyxy"),
        "best_bbox_area_px": _round(_hypothesis_bbox_area(hypothesis) or 0.0),
        "selection": {
            "source": "row_object_selection_policy_v2",
            "quality": "tentative",
            "evidence_gaps": evidence_gaps,
            "recovery_gate": "small_bbox_low_evidence",
            "min_confidence": _round(min_confidence),
            "min_support_count": min_support_count,
            "small_bbox_area_px": _round(small_bbox_area_px),
        },
        "pose_quality": hypothesis.get("pose_quality", {}),
        "notes": [
            "Recovered as a tentative small-object hypothesis from close RGB-D evidence.",
            "Tentative objects are visible to downstream stages but should not be treated as final stable detections.",
        ],
    }


def _rejected_hypothesis_payload(record: dict[str, Any]) -> dict[str, Any]:
    hypothesis = record["hypothesis"]
    position = _position_from_hypothesis(hypothesis)
    return {
        "source_hypothesis_id": hypothesis.get("hypothesis_id"),
        "class_name": hypothesis.get("class_name"),
        "confidence": _round(float(hypothesis.get("confidence", 0.0))),
        "evidence_score": _round(float(record["evidence_score"])),
        "support_count": int(hypothesis.get("support_count", 0)),
        "position_world": [_round(value) for value in position] if position is not None else None,
        "best_bbox_area_px": _round(_hypothesis_bbox_area(hypothesis) or 0.0),
        "reasons": record["reasons"],
    }


def _hypothesis_bbox_area(hypothesis: dict[str, Any]) -> float | None:
    bbox = hypothesis.get("best_bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    return _bbox_area((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])))


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _build_row_inspection_plan_for_scene(
    report: Task1SurveyReport,
    *,
    scene: SurveySceneInput,
    config: RowConfig,
) -> tuple[SurveyWorkspace, tuple[RowPlannedView, ...]]:
    workspace = scene.workspace
    if config.camera_z_m <= workspace.bottom_z_m:
        raise ValueError(
            "row wrist camera z must be above the tank bottom/object base plane: "
            f"camera_z={config.camera_z_m:.4f}, bottom_z={workspace.bottom_z_m:.4f}"
        )
    planned_views: list[RowPlannedView] = []
    for candidate_index, candidate in enumerate(report.candidates):
        target = (
            _clamp(candidate.rough_position_world[0], workspace.x_min, workspace.x_max),
            _clamp(candidate.rough_position_world[1], workspace.y_min, workspace.y_max),
            workspace.bottom_z_m,
        )
        primary_angle = _candidate_primary_angle(candidate, report.views_by_id, workspace, config=config)
        for local_index, offset in enumerate(_view_angle_offsets(config.row_views_per_candidate, config.row_view_angle_spread_rad)):
            approach_angle = primary_angle + offset
            camera_position, resolved_angle = _select_camera_position(
                target,
                requested_angle_rad=approach_angle,
                workspace=workspace,
                config=config,
            )
            look_at = (
                target[0],
                target[1],
                min(workspace.max_camera_z_m, workspace.bottom_z_m + config.look_at_height_offset_m),
            )
            workspace.validate_camera_position(camera_position)
            view_id = f"row_{candidate.candidate_id}_{local_index:02d}"
            view = SurveyView(
                view_id=view_id,
                grid_row=candidate_index,
                grid_col=local_index,
                camera_name=config.camera_name,
                desired_camera_position_world=_round_vector(camera_position),
                look_at_world=_round_vector(look_at),
                T_world_camera=_make_look_at_transform(camera_position, look_at),
            )
            planned_views.append(
                RowPlannedView(
                    candidate_id=candidate.candidate_id,
                    candidate_rough_position_world=target,
                    approach_angle_rad=resolved_angle,
                    view=view,
                )
            )
    return workspace, tuple(planned_views)


def _candidate_primary_angle(
    candidate: SurveyCandidate,
    survey_views_by_id: dict[str, dict[str, Any]],
    workspace: SurveyWorkspace,
    *,
    config: RowConfig,
) -> float:
    target = candidate.rough_position_world
    if config.entry_side != "survey-best":
        return _entry_side_angle(target, workspace=workspace, entry_side=config.entry_side)

    view_ids = tuple(view_id for view_id in (candidate.best_view, *candidate.supporting_views) if view_id)
    for view_id in view_ids:
        view_payload = survey_views_by_id.get(str(view_id))
        if not view_payload:
            continue
        camera_position = _view_camera_position(view_payload)
        if camera_position is None:
            continue
        dx = camera_position[0] - target[0]
        dy = camera_position[1] - target[1]
        if math.hypot(dx, dy) >= 1e-6:
            return math.atan2(dy, dx)

    center_x = (workspace.x_min + workspace.x_max) / 2.0
    center_y = (workspace.y_min + workspace.y_max) / 2.0
    dx = center_x - target[0]
    dy = center_y - target[1]
    if math.hypot(dx, dy) < 1e-6:
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
        raise ValueError(f"unsupported row entry_side: {entry_side}")
    dx = reference[0] - target[0]
    dy = reference[1] - target[1]
    if math.hypot(dx, dy) < 1e-6:
        return 0.0
    return math.atan2(dy, dx)


def _select_camera_position(
    target: tuple[float, float, float],
    *,
    requested_angle_rad: float,
    workspace: SurveyWorkspace,
    config: RowConfig,
) -> tuple[tuple[float, float, float], float]:
    center_angle = math.atan2(
        (workspace.y_min + workspace.y_max) / 2.0 - target[1],
        (workspace.x_min + workspace.x_max) / 2.0 - target[0],
    )
    candidates = (
        requested_angle_rad,
        requested_angle_rad + config.row_view_angle_spread_rad / 2.0,
        requested_angle_rad - config.row_view_angle_spread_rad / 2.0,
        requested_angle_rad + config.row_view_angle_spread_rad,
        requested_angle_rad - config.row_view_angle_spread_rad,
        center_angle,
        center_angle + config.row_view_angle_spread_rad,
        center_angle - config.row_view_angle_spread_rad,
        requested_angle_rad + math.pi,
    )
    for angle in candidates:
        position = (
            target[0] + math.cos(angle) * config.row_standoff_m,
            target[1] + math.sin(angle) * config.row_standoff_m,
            config.camera_z_m,
        )
        if _camera_position_is_valid(position, target, workspace=workspace, config=config):
            return (_round_vector(position), _normalize_angle(angle))

    fallback = _clamped_camera_position(target, center_angle, workspace=workspace, config=config)
    if not _camera_position_is_valid(fallback, target, workspace=workspace, config=config):
        raise ValueError(f"could not place row camera inside tank for candidate near {target}")
    return (_round_vector(fallback), _normalize_angle(center_angle))


def _camera_position_is_valid(
    position: tuple[float, float, float],
    target: tuple[float, float, float],
    *,
    workspace: SurveyWorkspace,
    config: RowConfig,
) -> bool:
    if math.hypot(position[0] - target[0], position[1] - target[1]) < config.row_min_oblique_distance_m:
        return False
    try:
        workspace.validate_camera_position(position)
    except ValueError:
        return False
    return True


def _clamped_camera_position(
    target: tuple[float, float, float],
    angle: float,
    *,
    workspace: SurveyWorkspace,
    config: RowConfig,
) -> tuple[float, float, float]:
    margin = min(config.row_min_oblique_distance_m / 2.0, 0.02)
    x = target[0] + math.cos(angle) * config.row_standoff_m
    y = target[1] + math.sin(angle) * config.row_standoff_m
    return (
        _clamp(x, workspace.x_min + margin, workspace.x_max - margin),
        _clamp(y, workspace.y_min + margin, workspace.y_max - margin),
        config.camera_z_m,
    )


def _view_angle_offsets(count: int, spread_rad: float) -> tuple[float, ...]:
    if count == 1:
        return (0.0,)
    offsets = [0.0]
    step = 1
    while len(offsets) < count:
        offsets.append(-spread_rad * step)
        if len(offsets) < count:
            offsets.append(spread_rad * step)
        step += 1
    return tuple(offsets)


def _row_observations_from_capture(
    *,
    planned: RowPlannedView,
    view_result: dict[str, Any],
    survey_observations: list[SurveyObservation],
    config: RowConfig,
    workspace: SurveyWorkspace,
    observation_offset: int,
) -> list[RowObservation]:
    if view_result.get("status") != "success":
        return []
    depth_path = view_result.get("depth_path")
    yolo_raw_path = view_result.get("yolo_raw_path")
    image_path = view_result.get("rgb_image_path")
    camera_transform = _transform_or_none(view_result.get("actual_T_world_camera")) or planned.view.T_world_camera
    fovy_rad = float(view_result.get("camera_fovy_rad") or _default_fovy_rad())
    depth = _load_depth(Path(str(depth_path))) if depth_path else None
    observations: list[RowObservation] = []
    for local_index, survey_observation in enumerate(survey_observations):
        if survey_observation.rough_position_world is None:
            continue
        ground_position = estimate_detection_ground_position(
            bbox_xyxy=survey_observation.bbox_xyxy,
            depth=None,
            T_world_camera=camera_transform,
            image_width=config.image_width,
            image_height=config.image_height,
            fovy_rad=fovy_rad,
            ground_z_m=workspace.bottom_z_m,
        )
        if ground_position is None:
            position = survey_observation.rough_position_world
            position_source = "close RGB-D YOLO bbox depth projection fallback"
        else:
            position = ground_position
            position_source = "close RGB bbox ground-plane projection"
        yaw_rad, yaw_confidence, extent_xy_m = _estimate_yaw_from_depth(
            bbox_xyxy=survey_observation.bbox_xyxy,
            depth=depth,
            T_world_camera=camera_transform,
            image_width=config.image_width,
            image_height=config.image_height,
            fovy_rad=fovy_rad,
            pose_min_pixels=config.pose_min_pixels,
            sample_stride_px=config.depth_sample_stride_px,
            yaw_min_eigen_ratio=config.yaw_min_eigen_ratio,
        )
        yaw_for_transform = yaw_rad if yaw_rad is not None else 0.0
        observations.append(
            RowObservation(
                observation_id=f"row_obs_{observation_offset + local_index + 1:04d}",
                candidate_id=planned.candidate_id,
                row_view_id=planned.view.view_id,
                image_path=str(image_path) if image_path else survey_observation.image_path,
                depth_path=str(depth_path) if depth_path else None,
                yolo_raw_path=str(yolo_raw_path) if yolo_raw_path else None,
                bbox_xyxy=survey_observation.bbox_xyxy,
                confidence=survey_observation.confidence,
                class_id=survey_observation.class_id,
                class_name=survey_observation.class_name,
                position_world=position,
                yaw_rad=yaw_rad,
                yaw_confidence=yaw_confidence,
                extent_xy_m=extent_xy_m,
                T_world_object=_make_yaw_transform(position, yaw_for_transform),
                position_source=position_source,
            )
        )
    return observations


def _estimate_yaw_from_depth(
    *,
    bbox_xyxy: tuple[float, float, float, float],
    depth: Any | None,
    T_world_camera: tuple[tuple[float, float, float, float], ...],
    image_width: int,
    image_height: int,
    fovy_rad: float,
    pose_min_pixels: int,
    sample_stride_px: int,
    yaw_min_eigen_ratio: float,
) -> tuple[float | None, float, tuple[float, float] | None]:
    if depth is None:
        return (None, 0.0, None)
    np = _import_numpy()
    left, top, right, bottom = _bbox_pixel_bounds(
        bbox_xyxy,
        image_width=image_width,
        image_height=image_height,
    )
    if right <= left or bottom <= top:
        return (None, 0.0, None)
    ys = np.arange(top, bottom, sample_stride_px, dtype=int)
    xs = np.arange(left, right, sample_stride_px, dtype=int)
    if ys.size == 0 or xs.size == 0:
        return (None, 0.0, None)
    grid_x, grid_y = np.meshgrid(xs, ys)
    depth_values = np.asarray(depth)[grid_y, grid_x]
    valid = np.isfinite(depth_values) & (depth_values > 0.0)
    if int(np.count_nonzero(valid)) < pose_min_pixels:
        return (None, 0.0, None)
    valid_depths = depth_values[valid]
    foreground_limit = float(np.percentile(valid_depths, 45.0)) + 0.025
    foreground = valid_depths <= foreground_limit
    if int(np.count_nonzero(foreground)) < pose_min_pixels:
        foreground = np.ones_like(valid_depths, dtype=bool)
    points = _project_pixels_to_world(
        np=np,
        pixel_x=grid_x[valid][foreground],
        pixel_y=grid_y[valid][foreground],
        depth_values=valid_depths[foreground],
        T_world_camera=T_world_camera,
        image_width=image_width,
        image_height=image_height,
        fovy_rad=fovy_rad,
    )
    if points.shape[0] < pose_min_pixels:
        return (None, 0.0, None)
    xy = points[:, :2]
    centered = xy - np.mean(xy, axis=0)
    if centered.shape[0] < 2:
        return (None, 0.0, None)
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    major_value = float(eigenvalues[order[0]])
    minor_value = float(max(eigenvalues[order[1]], 1e-12))
    if major_value <= 1e-12:
        return (None, 0.0, _xy_extent(np, xy))
    ratio = major_value / minor_value
    major_axis = eigenvectors[:, order[0]]
    yaw = _normalize_angle(math.atan2(float(major_axis[1]), float(major_axis[0])))
    yaw_confidence = max(0.0, min(1.0, (ratio - 1.0) / max(1e-9, yaw_min_eigen_ratio - 1.0)))
    if ratio < yaw_min_eigen_ratio:
        return (None, _round(yaw_confidence), _xy_extent(np, xy))
    return (_round(yaw), _round(yaw_confidence), _xy_extent(np, xy))


def _project_pixels_to_world(
    *,
    np: Any,
    pixel_x: Any,
    pixel_y: Any,
    depth_values: Any,
    T_world_camera: tuple[tuple[float, float, float, float], ...],
    image_width: int,
    image_height: int,
    fovy_rad: float,
) -> Any:
    fy = (image_height / 2.0) / math.tan(fovy_rad / 2.0)
    fx = fy
    camera_x = (pixel_x.astype(float) - image_width / 2.0) / fx * depth_values
    camera_y = (image_height / 2.0 - pixel_y.astype(float)) / fy * depth_values
    camera_z = -depth_values
    points_camera = np.stack((camera_x, camera_y, camera_z), axis=1)
    rotation = np.asarray([row[:3] for row in T_world_camera[:3]], dtype=float)
    position = np.asarray((T_world_camera[0][3], T_world_camera[1][3], T_world_camera[2][3]), dtype=float)
    return points_camera @ rotation.T + position


def _row_hypothesis_payload(index: int, cluster: list[RowObservation]) -> dict[str, Any]:
    center = _row_cluster_center(cluster)
    class_votes = _row_class_votes(cluster)
    class_name = max(class_votes, key=class_votes.get) if class_votes else None
    yaw_rad, yaw_confidence = _average_yaw(cluster)
    yaw_for_transform = yaw_rad if yaw_rad is not None else 0.0
    best = max(cluster, key=lambda observation: (_bbox_area(observation.bbox_xyxy) * observation.confidence, observation.confidence))
    supporting_views = sorted({observation.row_view_id for observation in cluster})
    source_candidates = sorted({observation.candidate_id for observation in cluster})
    average_confidence = sum(observation.confidence for observation in cluster) / len(cluster)
    fused_confidence = min(1.0, average_confidence + 0.10 * max(0, len(supporting_views) - 1))
    return {
        "hypothesis_id": f"object_hypothesis_{index + 1:03d}",
        "source_candidate_ids": source_candidates,
        "supporting_row_views": supporting_views,
        "support_count": len(cluster),
        "class_name": class_name,
        "class_votes": {key: _round(value) for key, value in sorted(class_votes.items())},
        "confidence": _round(fused_confidence),
        "position_world": [_round(value) for value in center],
        "yaw_rad": _round(yaw_rad) if yaw_rad is not None else None,
        "T_world_object": [list(row) for row in _make_yaw_transform(center, yaw_for_transform)],
        "best_image_path": best.image_path,
        "best_bbox_xyxy": [_round(value) for value in best.bbox_xyxy],
        "pose_quality": {
            "position_source": _row_position_source(cluster),
            "yaw_source": "depth_pca" if yaw_rad is not None else "unresolved_fallback_zero_in_T_world_object",
            "yaw_confidence": _round(yaw_confidence),
            "multi_candidate_merge": len(source_candidates) > 1,
        },
        "notes": [
            "Row hypothesis re-estimated from close RGB-D views; survey class votes are not inherited as final labels.",
            "Yaw is estimated from depth PCA when the local point cloud is elongated; otherwise T_world_object uses zero yaw as a placeholder.",
        ],
    }


def _row_position_source(cluster: list[RowObservation]) -> str:
    sources = sorted({observation.position_source for observation in cluster if observation.position_source})
    if not sources:
        return "unknown"
    if len(sources) == 1:
        return sources[0]
    return "mixed: " + ", ".join(sources)


def _stable_object_payload(
    index: int,
    hypothesis: dict[str, Any],
    *,
    evidence_score: float,
    duplicate_hypothesis_ids: list[str],
    workspace_bounds: dict[str, float],
    workspace_margin_m: float,
) -> dict[str, Any]:
    position = _position_from_hypothesis(hypothesis)
    if position is None:
        raise ValueError("stable object hypothesis must contain position_world")
    class_name = str(hypothesis.get("class_name") or "")
    confidence = float(hypothesis.get("confidence", 0.0))
    support_count = int(hypothesis.get("support_count", 0))
    source_candidate_ids = [
        str(candidate_id)
        for candidate_id in hypothesis.get("source_candidate_ids", [])
        if candidate_id is not None
    ]
    supporting_row_views = [
        str(view_id)
        for view_id in hypothesis.get("supporting_row_views", [])
        if view_id is not None
    ]
    within_nominal_workspace = _position_inside_workspace(position, workspace_bounds, margin_m=0.0)
    pose_quality = dict(hypothesis.get("pose_quality", {}))
    notes = [
        "Stable object selected from close RGB-D row hypotheses after evidence filtering, same-class merging, and cross-class ambiguity checks.",
    ]
    if not within_nominal_workspace:
        pose_quality["workspace_bounds_status"] = "within_configured_selection_margin"
        pose_quality["workspace_margin_m"] = _round(workspace_margin_m)
        notes.append(
            "Position is outside nominal tank XY bounds but within the configured row selection margin; raw RGB-D estimate is retained."
        )
    return {
        "object_id": f"row_object_{index + 1:03d}",
        "source_hypothesis_id": hypothesis.get("hypothesis_id"),
        "suppressed_duplicate_hypothesis_ids": duplicate_hypothesis_ids,
        "class_name": class_name,
        "confidence": _round(confidence),
        "evidence_score": _round(evidence_score),
        "support_count": support_count,
        "source_candidate_ids": source_candidate_ids,
        "supporting_row_views": supporting_row_views,
        "position_world": [_round(value) for value in position],
        "yaw_rad": hypothesis.get("yaw_rad"),
        "T_world_object": hypothesis.get("T_world_object"),
        "best_image_path": hypothesis.get("best_image_path"),
        "best_bbox_xyxy": hypothesis.get("best_bbox_xyxy"),
        "selection": {
            "source": "row_object_selection_policy_v2",
            "quality": "stable",
            "from_close_rgbd": True,
            "same_class_duplicates_suppressed": len(duplicate_hypothesis_ids),
            "workspace_filter": {
                "within_nominal_workspace": within_nominal_workspace,
                "workspace_margin_m": _round(workspace_margin_m),
            },
        },
        "pose_quality": pose_quality,
        "notes": notes,
    }


def _stable_evidence_score(hypothesis: dict[str, Any]) -> float:
    confidence = float(hypothesis.get("confidence", 0.0))
    support_count = max(0, int(hypothesis.get("support_count", 0)))
    source_candidate_count = len(hypothesis.get("source_candidate_ids", []) or [])
    row_view_count = len(hypothesis.get("supporting_row_views", []) or [])
    yaw_bonus = 1.05 if hypothesis.get("yaw_rad") is not None else 1.0
    candidate_bonus = 1.0 + 0.04 * max(0, source_candidate_count - 1)
    view_bonus = 1.0 + 0.02 * max(0, row_view_count - 1)
    return confidence * math.log1p(support_count) * candidate_bonus * view_bonus * yaw_bonus


def _same_class_duplicate_target(
    hypothesis: dict[str, Any],
    selected: list[dict[str, Any]],
    *,
    same_class_nms_radius_m: float,
) -> dict[str, Any] | None:
    class_name = str(hypothesis.get("class_name") or "")
    position = _position_from_hypothesis(hypothesis)
    if position is None:
        return None
    for selected_hypothesis in selected:
        if str(selected_hypothesis.get("class_name") or "") != class_name:
            continue
        selected_position = _position_from_hypothesis(selected_hypothesis)
        if selected_position is None:
            continue
        if _xy_distance(position, selected_position) <= same_class_nms_radius_m:
            return selected_hypothesis
    return None


def _position_from_hypothesis(hypothesis: dict[str, Any]) -> tuple[float, float, float] | None:
    value = hypothesis.get("position_world")
    if not isinstance(value, list) or len(value) != 3:
        return None
    return (float(value[0]), float(value[1]), float(value[2]))


def _workspace_bounds(workspace: SurveyWorkspace | dict[str, float]) -> dict[str, float]:
    if isinstance(workspace, SurveyWorkspace):
        return workspace.to_dict()
    return {key: float(workspace[key]) for key in ("x_min", "x_max", "y_min", "y_max", "bottom_z_m", "tank_opening_z_m")}


def _position_inside_workspace(
    position: tuple[float, float, float],
    bounds: dict[str, float],
    *,
    margin_m: float = 0.0,
) -> bool:
    return (
        bounds["x_min"] - margin_m <= position[0] <= bounds["x_max"] + margin_m
        and bounds["y_min"] - margin_m <= position[1] <= bounds["y_max"] + margin_m
    )


def _stable_selection_empty_metadata(config: RowConfig) -> dict[str, Any]:
    return {
        "status": "not_run",
        "policy_version": "row_object_selection_policy_v2",
        "input_hypothesis_count": 0,
        "candidate_after_filter_count": 0,
        "same_class_representative_count": 0,
        "confident_representative_count": 0,
        "stable_object_count": 0,
        "tentative_object_count": 0,
        "ambiguous_object_count": 0,
        "class_vote_ambiguous_count": 0,
        "rejected_hypothesis_count": 0,
        "suppressed_same_class_duplicate_count": 0,
        "rejected_counts": {},
        "thresholds": {
            "min_confidence": _round(config.stable_object_min_confidence),
            "min_support_count": config.stable_object_min_support_count,
            "min_evidence_score": _round(config.stable_object_min_evidence_score),
            "same_class_nms_radius_m": _round(config.stable_object_same_class_nms_radius_m),
            "workspace_margin_m": _round(config.stable_object_workspace_margin_m),
            "tentative_min_confidence": _round(config.tentative_object_min_confidence),
            "tentative_min_support_count": config.tentative_object_min_support_count,
            "tentative_small_bbox_area_px": _round(config.tentative_object_small_bbox_area_px),
            "cross_class_conflict_radius_m": _round(config.cross_class_conflict_radius_m),
            "cross_class_ambiguity_score_ratio": _round(config.cross_class_ambiguity_score_ratio),
            "class_vote_ambiguity_top_to_second_ratio": _round(config.class_vote_ambiguity_top_to_second_ratio),
            "class_vote_ambiguity_min_secondary_vote": _round(config.class_vote_ambiguity_min_secondary_vote),
            "workspace_filter": True,
        },
    }


def _row_cluster_center(cluster: list[RowObservation]) -> tuple[float, float, float]:
    total_weight = sum(max(observation.confidence, 0.05) for observation in cluster)
    if total_weight <= 0.0:
        total_weight = float(len(cluster))
    return (
        sum(observation.position_world[0] * max(observation.confidence, 0.05) for observation in cluster) / total_weight,
        sum(observation.position_world[1] * max(observation.confidence, 0.05) for observation in cluster) / total_weight,
        sum(observation.position_world[2] * max(observation.confidence, 0.05) for observation in cluster) / total_weight,
    )


def _average_yaw(cluster: list[RowObservation]) -> tuple[float | None, float]:
    weighted_sin = 0.0
    weighted_cos = 0.0
    total_weight = 0.0
    for observation in cluster:
        if observation.yaw_rad is None or observation.yaw_confidence <= 0.0:
            continue
        weight = max(observation.confidence, 0.05) * observation.yaw_confidence
        weighted_sin += math.sin(observation.yaw_rad) * weight
        weighted_cos += math.cos(observation.yaw_rad) * weight
        total_weight += weight
    if total_weight <= 1e-12:
        return (None, 0.0)
    return (_round(math.atan2(weighted_sin, weighted_cos)), _round(min(1.0, total_weight / len(cluster))))


def _row_class_votes(cluster: list[RowObservation]) -> dict[str, float]:
    votes: dict[str, float] = {}
    for observation in cluster:
        if not observation.class_name:
            continue
        votes[observation.class_name] = votes.get(observation.class_name, 0.0) + observation.confidence
    return votes


def _plan_payload(
    *,
    created_utc: str,
    survey_report: Task1SurveyReport,
    scene: SurveySceneInput,
    config: RowConfig,
    row_dir: Path,
    plan_path: Path,
    report_path: Path,
    planned_views: tuple[RowPlannedView, ...],
) -> dict[str, Any]:
    return {
        "schema_version": "task1_row_plan_v1",
        "stage": "row",
        "created_utc": created_utc,
        "source_survey_report_path": str(survey_report.path),
        "layout_snapshot_path": str(survey_report.layout_snapshot_path),
        "task1_run_dir": str(survey_report.task1_run_dir),
        "row_dir": str(row_dir),
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
            "arm_entry": "from top square opening",
            "pose_validation": "MuJoCo IK plus robot collision check before rendering each row view.",
            "survey_candidates_are_recall_first": True,
        },
        "workspace": scene.workspace.to_dict(),
        "survey_candidates": [candidate.to_dict() for candidate in survey_report.candidates],
        "views": [planned.to_dict() for planned in planned_views],
    }


def _report_payload(
    *,
    created_utc: str,
    status: str,
    message: str,
    survey_report: Task1SurveyReport,
    scene: SurveySceneInput,
    config: RowConfig,
    workspace: SurveyWorkspace,
    row_dir: Path,
    plan_path: Path,
    report_path: Path,
    planned_views: tuple[RowPlannedView, ...],
    view_results: list[dict[str, Any]],
    row_observations: list[dict[str, Any]],
    object_hypotheses: list[dict[str, Any]],
    stable_objects: list[dict[str, Any]],
    tentative_objects: list[dict[str, Any]],
    ambiguous_objects: list[dict[str, Any]],
    rejected_hypotheses: list[dict[str, Any]],
    object_selection_summary: dict[str, Any],
    row_reachable_planning_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": "task1_row_report_v1",
        "stage": "row",
        "status": status,
        "created_utc": created_utc,
        "message": message,
        "source_survey_report_path": str(survey_report.path),
        "source_survey_status": survey_report.status,
        "layout_snapshot_path": str(survey_report.layout_snapshot_path),
        "task1_run_dir": str(survey_report.task1_run_dir),
        "row_dir": str(row_dir),
        "plan_path": str(plan_path),
        "report_path": str(report_path),
        "scene_model_path": str(config.scene_model_path),
        "camera_name": config.camera_name,
        "workspace": workspace.to_dict(),
        "image_size": [config.image_width, config.image_height],
        "run_yolo": config.run_yolo,
        "yolo_profile_path": str(config.yolo_profile_path),
        "row_config": _row_config_payload(config),
        "selected_objects": [obj.to_dict() for obj in scene.objects],
        "survey_candidates": [candidate.to_dict() for candidate in survey_report.candidates],
        "planned_views": [planned.to_dict() for planned in planned_views],
        "views": view_results,
        "row_reachable_planning_summary": row_reachable_planning_summary,
        "row_view_selection_summary": _row_view_selection_summary(view_results),
        "row_observations": row_observations,
        "object_hypotheses": object_hypotheses,
        "stable_objects": stable_objects,
        "tentative_objects": tentative_objects,
        "ambiguous_objects": ambiguous_objects,
        "rejected_hypotheses": rejected_hypotheses,
        "object_selection_summary": object_selection_summary,
        "stable_object_selection": object_selection_summary,
        "notes": [
            "Row consumes survey candidates as proposal points; candidate count is intentionally not assumed to be five.",
            "Close views are planned inside the tank, below the tank opening, then accepted only after IK and collision checks.",
            "Stable objects are the recommended input for later final-photo targeting.",
            "Tentative and ambiguous objects are retained for explicit follow-up instead of being silently filtered.",
            "Object hypotheses are retained as a recall-first diagnostic pool; unresolved yaw is explicitly marked.",
        ],
    }
    return payload


def _row_view_selection_summary(view_results: list[dict[str, Any]]) -> dict[str, Any]:
    selected_count = 0
    selected_alternative_count = 0
    failed_no_safe_candidate_count = 0
    total_attempt_count = 0
    collision_rejection_count = 0
    ik_rejection_count = 0
    candidate_limit_reached_count = 0
    for view_result in view_results:
        selection = view_result.get("row_view_selection")
        if not isinstance(selection, dict):
            continue
        total_attempt_count += int(selection.get("attempt_count", 0))
        rejected = selection.get("rejected_counts")
        if isinstance(rejected, dict):
            collision_rejection_count += int(rejected.get("collision", 0))
            ik_rejection_count += int(rejected.get("ik_failed", 0))
        if selection.get("candidate_limit_reached"):
            candidate_limit_reached_count += 1
        if selection.get("status") == "selected":
            selected_count += 1
            if int(selection.get("selected_candidate_index") or 0) > 0:
                selected_alternative_count += 1
        elif selection.get("status") == "failed_no_collision_free_candidate":
            failed_no_safe_candidate_count += 1
    return {
        "strategy": "ik_collision_checked_row_view_candidates_v1",
        "enabled": any(
            isinstance(view.get("row_view_selection"), dict)
            and view["row_view_selection"].get("status") != "disabled"
            for view in view_results
        ),
        "selected_count": selected_count,
        "selected_alternative_count": selected_alternative_count,
        "failed_no_safe_candidate_count": failed_no_safe_candidate_count,
        "total_attempt_count": total_attempt_count,
        "collision_rejection_count": collision_rejection_count,
        "ik_rejection_count": ik_rejection_count,
        "candidate_limit_reached_count": candidate_limit_reached_count,
    }


def _row_config_payload(config: RowConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def _planned_row_view_result(planned: RowPlannedView) -> dict[str, Any]:
    payload = planned.to_dict()
    payload["status"] = "planned"
    return payload


def _survey_candidate(payload: Any, index: int) -> SurveyCandidate:
    if not isinstance(payload, dict):
        raise ValueError("task1 survey candidates must be JSON objects")
    candidate_id = str(payload.get("candidate_id") or f"candidate_{index + 1:03d}")
    rough_position = _float_tuple_from_value(_required(payload, "rough_position_world"), 3)
    supporting_views_payload = payload.get("supporting_views", [])
    if not isinstance(supporting_views_payload, list):
        raise ValueError("survey candidate supporting_views must be a list")
    class_votes_payload = payload.get("class_votes", {})
    if not isinstance(class_votes_payload, dict):
        raise ValueError("survey candidate class_votes must be a dict")
    return SurveyCandidate(
        candidate_id=candidate_id,
        rough_position_world=rough_position,
        supporting_views=tuple(str(value) for value in supporting_views_payload),
        best_image_path=_optional_str(payload.get("best_image_path")),
        best_bbox_xyxy=(
            _float_tuple_from_value(payload["best_bbox_xyxy"], 4) if payload.get("best_bbox_xyxy") is not None else None
        ),
        best_view=_optional_str(payload.get("best_view")),
        class_votes={str(key): float(value) for key, value in class_votes_payload.items()},
        confidence=float(payload.get("confidence", 0.0)),
        support_count=int(payload.get("support_count", len(supporting_views_payload))),
        raw_payload=dict(payload),
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


def _required(payload: dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError(f"missing required field: {key}")
    return payload[key]


def _required_path(payload: dict[str, Any], key: str) -> Path:
    value = _required(payload, key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"field {key} must be a non-empty path string")
    return Path(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _image_size_from_payload(value: Any) -> tuple[int, int]:
    if isinstance(value, list) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    return (DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT)


def _float_tuple_from_value(value: Any, length: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"expected a list of {length} numbers")
    return tuple(float(item) for item in value)


def _transform_or_none(value: Any) -> tuple[tuple[float, float, float, float], ...] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    rows = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            return None
        rows.append(tuple(float(item) for item in row))
    return tuple(rows)


def _load_depth(path: Path) -> Any | None:
    if not path.exists():
        return None
    np = _import_numpy()
    return np.load(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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


def _make_yaw_transform(
    position: tuple[float, float, float],
    yaw_rad: float,
) -> tuple[tuple[float, float, float, float], ...]:
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    x, y, z = position
    return (
        (_round(cos_yaw), _round(-sin_yaw), 0.0, _round(x)),
        (_round(sin_yaw), _round(cos_yaw), 0.0, _round(y)),
        (0.0, 0.0, 1.0, _round(z)),
        (0.0, 0.0, 0.0, 1.0),
    )


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


def _xy_extent(np: Any, xy: Any) -> tuple[float, float]:
    if xy.size == 0:
        return (0.0, 0.0)
    return (_round(float(np.max(xy[:, 0]) - np.min(xy[:, 0]))), _round(float(np.max(xy[:, 1]) - np.min(xy[:, 1]))))


def _bbox_area(bbox_xyxy: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox_xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _xy_distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _round_vector(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    return (_round(vector[0]), _round(vector[1]), _round(vector[2]))


def _round(value: float | None) -> float:
    if value is None:
        raise ValueError("cannot round None")
    return round(float(value), 6)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _cross(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(_dot(vector, vector))
    if norm <= 1e-12:
        raise ValueError("cannot normalize a zero vector")
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def _default_fovy_rad() -> float:
    return math.radians(41.83792730009236)


def _import_numpy() -> Any:
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError("numpy is required for task1 row RGB-D pose estimation.") from exc
    return np
