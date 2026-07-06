from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


DEFAULT_SCENE_MODEL = Path("examples/mujoco/gen3_with_tank.xml")
DEFAULT_YOLO_PROFILE = Path("configs/yolo/stage3_default.yaml")
DEFAULT_OUTPUT_DIR = Path("outputs/task1")
DEFAULT_CAMERA_NAME = "wrist"
DEFAULT_GRID_SIZE = 4
DEFAULT_IMAGE_WIDTH = 1920
DEFAULT_IMAGE_HEIGHT = 1080
DEFAULT_TANK_OPENING_Z_M = 0.50
DEFAULT_CAMERA_Z_M = 0.38
DEFAULT_OPENING_CLEARANCE_M = 0.035
DEFAULT_OBLIQUE_OFFSET_M = 0.075
DEFAULT_CLUSTER_RADIUS_M = 0.11
DEFAULT_CROSS_CLASS_MERGE_RADIUS_M = 0.06
DEFAULT_CLUSTER_SPLIT_DISTANCE_M = 0.075
DEFAULT_CLUSTER_SPLIT_MIN_VOTE = 0.8
DEFAULT_WEAK_CANDIDATE_MERGE_RADIUS_M = 0.08
DEFAULT_WEAK_CANDIDATE_VOTE_THRESHOLD = 1.05
DEFAULT_TINY_CANDIDATE_BBOX_AREA_PX = 30_000.0
DEFAULT_WEAK_TINY_CANDIDATE_MAX_SPREAD_M = 0.045
DEFAULT_SINGLE_VIEW_FALLBACK_CONFIDENCE = 0.15
DEFAULT_SINGLE_VIEW_FALLBACK_MIN_DISTANCE_M = 0.10
DEFAULT_MIN_CANDIDATE_SUPPORT_VIEWS = 2
DEFAULT_LARGE_SAME_CLASS_MERGE_RADIUS_M = 0.17
DEFAULT_LARGE_SAME_CLASS_BBOX_AREA_PX = 120_000.0
DEFAULT_MIXED_CLASS_SPLIT_MIN_VOTE = 1.5
DEFAULT_MIXED_CLASS_SPLIT_DISTANCE_M = 0.055
DEFAULT_WEAK_MULTIVIEW_FALLBACK_VOTE = 0.65
DEFAULT_WEAK_MULTIVIEW_FALLBACK_MIN_DISTANCE_M = 0.11
DEFAULT_WEAK_MULTIVIEW_FALLBACK_MAX_BBOX_AREA_PX = 6_000.0
DEFAULT_WEAK_MULTIVIEW_FALLBACK_ELONGATED_ASPECT_RATIO = 3.0
DEFAULT_WEAK_MULTIVIEW_FALLBACK_ELONGATED_MAX_BBOX_AREA_PX = 35_000.0
DEFAULT_YOLO_TILE_GRID_SIZE = 2
DEFAULT_YOLO_TILE_OVERLAP = 0.12
DEFAULT_YOLO_TILE_NMS_IOU = 0.45
DEFAULT_DEPTH_SAMPLE_STRIDE_PX = 10
DEFAULT_DEPTH_COMPONENT_MIN_PIXELS = 40
DEFAULT_DEPTH_COMPONENT_SPLIT_DISTANCE_M = 0.07
DEFAULT_DEPTH_MODE_BAND_M = 0.025
DEFAULT_DEPTH_FOREGROUND_PERCENTILE = 45.0
DEFAULT_DEPTH_FOREGROUND_MARGIN_M = 0.025


class SurveyDetector(Protocol):
    def predict(self, *, image_path: Path | str, camera_name: str | None = None) -> dict[str, Any]:
        """Return raw bbox detections for one survey RGB image."""


@dataclass(frozen=True)
class Stage0Object:
    object_id: str
    class_name: str
    position: tuple[float, float, float]
    yaw_rad: float
    T_world_object: tuple[tuple[float, float, float, float], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "class_name": self.class_name,
            "position": list(self.position),
            "yaw_rad": self.yaw_rad,
            "T_world_object": [list(row) for row in self.T_world_object],
        }


@dataclass(frozen=True)
class Stage0Layout:
    path: Path
    schema_version: str
    seed: int | None
    base_height_m: float
    placement_bounds: dict[str, float]
    objects: tuple[Stage0Object, ...]
    raw_payload: dict[str, Any]

    @property
    def selected_object_ids(self) -> tuple[str, ...]:
        return tuple(obj.object_id for obj in self.objects)


@dataclass(frozen=True)
class SurveySceneObject:
    body_name: str
    class_name: str
    position: tuple[float, float, float]
    yaw_rad: float
    T_world_object: tuple[tuple[float, float, float, float], ...]

    @property
    def object_id(self) -> str:
        return self.body_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "body_name": self.body_name,
            "object_id": self.body_name,
            "class_name": self.class_name,
            "position": list(self.position),
            "yaw_rad": self.yaw_rad,
            "T_world_object": [list(row) for row in self.T_world_object],
        }


@dataclass(frozen=True)
class SurveyWorkspace:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    bottom_z_m: float
    tank_opening_z_m: float = DEFAULT_TANK_OPENING_Z_M
    opening_clearance_m: float = DEFAULT_OPENING_CLEARANCE_M

    @property
    def max_camera_z_m(self) -> float:
        return self.tank_opening_z_m - self.opening_clearance_m

    def validate_camera_position(self, position: tuple[float, float, float]) -> None:
        x, y, z = position
        if not (self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max):
            raise ValueError(f"survey camera xy must stay inside tank workspace, got {position}")
        if z >= self.tank_opening_z_m:
            raise ValueError(
                "survey wrist camera z must be below the tank upper opening height: "
                f"camera_z={z:.4f}, tank_opening_z={self.tank_opening_z_m:.4f}"
            )
        if z > self.max_camera_z_m:
            raise ValueError(
                "survey wrist camera must keep clearance below the tank opening: "
                f"camera_z={z:.4f}, max_camera_z={self.max_camera_z_m:.4f}"
            )
        if z <= self.bottom_z_m:
            raise ValueError(f"survey wrist camera z must be above the tank bottom/object base plane, got {position}")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class SurveySceneInput:
    source_path: Path
    source_schema_version: str
    seed: int | None
    workspace: SurveyWorkspace
    objects: tuple[SurveySceneObject, ...]

    @property
    def selected_object_ids(self) -> tuple[str, ...]:
        return tuple(obj.object_id for obj in self.objects)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "source_schema_version": self.source_schema_version,
            "seed": self.seed,
            "workspace": self.workspace.to_dict(),
            "selected_objects": [obj.to_dict() for obj in self.objects],
        }


@dataclass(frozen=True)
class SurveyFusionPolicy:
    cluster_radius_m: float = DEFAULT_CLUSTER_RADIUS_M
    cross_class_merge_radius_m: float = DEFAULT_CROSS_CLASS_MERGE_RADIUS_M
    cluster_split_distance_m: float = DEFAULT_CLUSTER_SPLIT_DISTANCE_M
    cluster_split_min_vote: float = DEFAULT_CLUSTER_SPLIT_MIN_VOTE
    weak_candidate_merge_radius_m: float = DEFAULT_WEAK_CANDIDATE_MERGE_RADIUS_M
    weak_candidate_vote_threshold: float = DEFAULT_WEAK_CANDIDATE_VOTE_THRESHOLD
    tiny_candidate_bbox_area_px: float = DEFAULT_TINY_CANDIDATE_BBOX_AREA_PX
    weak_tiny_candidate_max_spread_m: float = DEFAULT_WEAK_TINY_CANDIDATE_MAX_SPREAD_M
    single_view_fallback_confidence: float = DEFAULT_SINGLE_VIEW_FALLBACK_CONFIDENCE
    single_view_fallback_min_distance_m: float = DEFAULT_SINGLE_VIEW_FALLBACK_MIN_DISTANCE_M
    min_candidate_support_views: int = DEFAULT_MIN_CANDIDATE_SUPPORT_VIEWS
    duplicate_large_same_class_merge_radius_m: float = DEFAULT_LARGE_SAME_CLASS_MERGE_RADIUS_M
    duplicate_large_same_class_bbox_area_px: float = DEFAULT_LARGE_SAME_CLASS_BBOX_AREA_PX
    mixed_class_split_min_vote: float = DEFAULT_MIXED_CLASS_SPLIT_MIN_VOTE
    mixed_class_split_distance_m: float = DEFAULT_MIXED_CLASS_SPLIT_DISTANCE_M
    weak_multiview_fallback_vote: float = DEFAULT_WEAK_MULTIVIEW_FALLBACK_VOTE
    weak_multiview_fallback_min_distance_m: float = DEFAULT_WEAK_MULTIVIEW_FALLBACK_MIN_DISTANCE_M
    weak_multiview_fallback_max_bbox_area_px: float = DEFAULT_WEAK_MULTIVIEW_FALLBACK_MAX_BBOX_AREA_PX
    weak_multiview_fallback_elongated_aspect_ratio: float = DEFAULT_WEAK_MULTIVIEW_FALLBACK_ELONGATED_ASPECT_RATIO
    weak_multiview_fallback_elongated_max_bbox_area_px: float = (
        DEFAULT_WEAK_MULTIVIEW_FALLBACK_ELONGATED_MAX_BBOX_AREA_PX
    )

    def validate(self) -> None:
        if self.cluster_radius_m <= 0:
            raise ValueError("cluster_radius_m must be positive")
        if self.cross_class_merge_radius_m <= 0:
            raise ValueError("cross_class_merge_radius_m must be positive")
        if self.cluster_split_distance_m <= 0:
            raise ValueError("cluster_split_distance_m must be positive")
        if self.cluster_split_min_vote < 0:
            raise ValueError("cluster_split_min_vote must be non-negative")
        if self.weak_candidate_merge_radius_m <= 0:
            raise ValueError("weak_candidate_merge_radius_m must be positive")
        if self.weak_candidate_vote_threshold < 0:
            raise ValueError("weak_candidate_vote_threshold must be non-negative")
        if self.tiny_candidate_bbox_area_px < 0:
            raise ValueError("tiny_candidate_bbox_area_px must be non-negative")
        if self.weak_tiny_candidate_max_spread_m < 0:
            raise ValueError("weak_tiny_candidate_max_spread_m must be non-negative")
        if not 0.0 <= self.single_view_fallback_confidence <= 1.0:
            raise ValueError("single_view_fallback_confidence must be between 0 and 1")
        if self.single_view_fallback_min_distance_m < 0:
            raise ValueError("single_view_fallback_min_distance_m must be non-negative")
        if self.min_candidate_support_views <= 0:
            raise ValueError("min_candidate_support_views must be positive")
        if self.duplicate_large_same_class_merge_radius_m < 0:
            raise ValueError("duplicate_large_same_class_merge_radius_m must be non-negative")
        if self.duplicate_large_same_class_bbox_area_px < 0:
            raise ValueError("duplicate_large_same_class_bbox_area_px must be non-negative")
        if self.mixed_class_split_min_vote < 0:
            raise ValueError("mixed_class_split_min_vote must be non-negative")
        if self.mixed_class_split_distance_m < 0:
            raise ValueError("mixed_class_split_distance_m must be non-negative")
        if self.weak_multiview_fallback_vote < 0:
            raise ValueError("weak_multiview_fallback_vote must be non-negative")
        if self.weak_multiview_fallback_min_distance_m < 0:
            raise ValueError("weak_multiview_fallback_min_distance_m must be non-negative")
        if self.weak_multiview_fallback_max_bbox_area_px < 0:
            raise ValueError("weak_multiview_fallback_max_bbox_area_px must be non-negative")
        if self.weak_multiview_fallback_elongated_aspect_ratio < 1.0:
            raise ValueError("weak_multiview_fallback_elongated_aspect_ratio must be at least 1")
        if self.weak_multiview_fallback_elongated_max_bbox_area_px < 0:
            raise ValueError("weak_multiview_fallback_elongated_max_bbox_area_px must be non-negative")


@dataclass(frozen=True)
class SurveyConfig:
    scene_model_path: Path = DEFAULT_SCENE_MODEL
    yolo_profile_path: Path = DEFAULT_YOLO_PROFILE
    output_dir: Path = DEFAULT_OUTPUT_DIR
    run_dir: Path | None = None
    created_utc: str | None = None
    camera_name: str = DEFAULT_CAMERA_NAME
    grid_size: int = DEFAULT_GRID_SIZE
    image_width: int = DEFAULT_IMAGE_WIDTH
    image_height: int = DEFAULT_IMAGE_HEIGHT
    camera_z_m: float = DEFAULT_CAMERA_Z_M
    tank_opening_z_m: float = DEFAULT_TANK_OPENING_Z_M
    opening_clearance_m: float = DEFAULT_OPENING_CLEARANCE_M
    oblique_offset_m: float = DEFAULT_OBLIQUE_OFFSET_M
    cluster_radius_m: float = DEFAULT_CLUSTER_RADIUS_M
    cross_class_merge_radius_m: float = DEFAULT_CROSS_CLASS_MERGE_RADIUS_M
    cluster_split_distance_m: float = DEFAULT_CLUSTER_SPLIT_DISTANCE_M
    cluster_split_min_vote: float = DEFAULT_CLUSTER_SPLIT_MIN_VOTE
    weak_candidate_merge_radius_m: float = DEFAULT_WEAK_CANDIDATE_MERGE_RADIUS_M
    weak_candidate_vote_threshold: float = DEFAULT_WEAK_CANDIDATE_VOTE_THRESHOLD
    tiny_candidate_bbox_area_px: float = DEFAULT_TINY_CANDIDATE_BBOX_AREA_PX
    weak_tiny_candidate_max_spread_m: float = DEFAULT_WEAK_TINY_CANDIDATE_MAX_SPREAD_M
    single_view_fallback_confidence: float = DEFAULT_SINGLE_VIEW_FALLBACK_CONFIDENCE
    single_view_fallback_min_distance_m: float = DEFAULT_SINGLE_VIEW_FALLBACK_MIN_DISTANCE_M
    min_candidate_support_views: int = DEFAULT_MIN_CANDIDATE_SUPPORT_VIEWS
    large_same_class_merge_radius_m: float = DEFAULT_LARGE_SAME_CLASS_MERGE_RADIUS_M
    large_same_class_bbox_area_px: float = DEFAULT_LARGE_SAME_CLASS_BBOX_AREA_PX
    mixed_class_split_min_vote: float = DEFAULT_MIXED_CLASS_SPLIT_MIN_VOTE
    mixed_class_split_distance_m: float = DEFAULT_MIXED_CLASS_SPLIT_DISTANCE_M
    weak_multiview_fallback_vote: float = DEFAULT_WEAK_MULTIVIEW_FALLBACK_VOTE
    weak_multiview_fallback_min_distance_m: float = DEFAULT_WEAK_MULTIVIEW_FALLBACK_MIN_DISTANCE_M
    weak_multiview_fallback_max_bbox_area_px: float = DEFAULT_WEAK_MULTIVIEW_FALLBACK_MAX_BBOX_AREA_PX
    weak_multiview_fallback_elongated_aspect_ratio: float = DEFAULT_WEAK_MULTIVIEW_FALLBACK_ELONGATED_ASPECT_RATIO
    weak_multiview_fallback_elongated_max_bbox_area_px: float = (
        DEFAULT_WEAK_MULTIVIEW_FALLBACK_ELONGATED_MAX_BBOX_AREA_PX
    )
    yolo_confidence: float = 0.15
    yolo_iou: float | None = None
    yolo_image_size: int | None = None
    yolo_device: str | None = None
    yolo_max_detections: int | None = 30
    yolo_tile_grid_size: int = DEFAULT_YOLO_TILE_GRID_SIZE
    yolo_tile_overlap: float = DEFAULT_YOLO_TILE_OVERLAP
    yolo_tile_nms_iou: float = DEFAULT_YOLO_TILE_NMS_IOU
    depth_sample_stride_px: int = DEFAULT_DEPTH_SAMPLE_STRIDE_PX
    depth_component_min_pixels: int = DEFAULT_DEPTH_COMPONENT_MIN_PIXELS
    depth_component_split_distance_m: float = DEFAULT_DEPTH_COMPONENT_SPLIT_DISTANCE_M
    run_yolo: bool = True
    plan_only: bool = False
    strict_yolo: bool = False
    max_ik_iterations: int = 200
    ik_position_tolerance_m: float = 0.05
    ik_orientation_tolerance_rad: float = 0.35
    ik_damping: float = 1e-3

    def fusion_policy(self) -> SurveyFusionPolicy:
        return SurveyFusionPolicy(
            cluster_radius_m=self.cluster_radius_m,
            cross_class_merge_radius_m=self.cross_class_merge_radius_m,
            cluster_split_distance_m=self.cluster_split_distance_m,
            cluster_split_min_vote=self.cluster_split_min_vote,
            weak_candidate_merge_radius_m=self.weak_candidate_merge_radius_m,
            weak_candidate_vote_threshold=self.weak_candidate_vote_threshold,
            tiny_candidate_bbox_area_px=self.tiny_candidate_bbox_area_px,
            weak_tiny_candidate_max_spread_m=self.weak_tiny_candidate_max_spread_m,
            single_view_fallback_confidence=self.single_view_fallback_confidence,
            single_view_fallback_min_distance_m=self.single_view_fallback_min_distance_m,
            min_candidate_support_views=self.min_candidate_support_views,
            duplicate_large_same_class_merge_radius_m=self.large_same_class_merge_radius_m,
            duplicate_large_same_class_bbox_area_px=self.large_same_class_bbox_area_px,
            mixed_class_split_min_vote=self.mixed_class_split_min_vote,
            mixed_class_split_distance_m=self.mixed_class_split_distance_m,
            weak_multiview_fallback_vote=self.weak_multiview_fallback_vote,
            weak_multiview_fallback_min_distance_m=self.weak_multiview_fallback_min_distance_m,
            weak_multiview_fallback_max_bbox_area_px=self.weak_multiview_fallback_max_bbox_area_px,
            weak_multiview_fallback_elongated_aspect_ratio=self.weak_multiview_fallback_elongated_aspect_ratio,
            weak_multiview_fallback_elongated_max_bbox_area_px=(
                self.weak_multiview_fallback_elongated_max_bbox_area_px
            ),
        )


@dataclass(frozen=True)
class SurveyView:
    view_id: str
    grid_row: int
    grid_col: int
    camera_name: str
    desired_camera_position_world: tuple[float, float, float]
    look_at_world: tuple[float, float, float]
    T_world_camera: tuple[tuple[float, float, float, float], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "grid_row": self.grid_row,
            "grid_col": self.grid_col,
            "camera_name": self.camera_name,
            "desired_camera_position_world": list(self.desired_camera_position_world),
            "look_at_world": list(self.look_at_world),
            "T_world_camera": [list(row) for row in self.T_world_camera],
        }


@dataclass(frozen=True)
class SurveyObservation:
    view_id: str
    image_path: str | None
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int | None
    class_name: str | None
    rough_position_world: tuple[float, float, float] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "image_path": self.image_path,
            "bbox_xyxy": list(self.bbox_xyxy),
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "rough_position_world": (
                list(self.rough_position_world) if self.rough_position_world is not None else None
            ),
        }


def load_stage0_layout(path: Path | str) -> Stage0Layout:
    layout_path = Path(path)
    payload = json.loads(layout_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "target_object_pose_layout_v1":
        raise ValueError(f"unsupported target object layout schema_version: {payload.get('schema_version')}")
    objects = payload.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("target object layout must contain a non-empty objects list")
    placement_bounds = payload.get("placement_bounds")
    if not isinstance(placement_bounds, dict):
        raise ValueError("target object layout must contain placement_bounds")
    return Stage0Layout(
        path=layout_path,
        schema_version=str(payload["schema_version"]),
        seed=int(payload["seed"]) if payload.get("seed") is not None else None,
        base_height_m=float(payload.get("base_height_m", 0.03)),
        placement_bounds={key: float(placement_bounds[key]) for key in ("x_min", "x_max", "y_min", "y_max")},
        objects=tuple(_stage0_object(obj) for obj in objects),
        raw_payload=payload,
    )


def build_workspace(
    layout: Stage0Layout,
    *,
    tank_opening_z_m: float = DEFAULT_TANK_OPENING_Z_M,
    opening_clearance_m: float = DEFAULT_OPENING_CLEARANCE_M,
) -> SurveyWorkspace:
    """Adapt the upstream layout workspace fields into the survey workspace contract."""
    return SurveyWorkspace(
        x_min=layout.placement_bounds["x_min"],
        x_max=layout.placement_bounds["x_max"],
        y_min=layout.placement_bounds["y_min"],
        y_max=layout.placement_bounds["y_max"],
        bottom_z_m=layout.base_height_m,
        tank_opening_z_m=tank_opening_z_m,
        opening_clearance_m=opening_clearance_m,
    )


def survey_scene_from_stage0_layout(
    layout: Stage0Layout,
    *,
    tank_opening_z_m: float = DEFAULT_TANK_OPENING_Z_M,
    opening_clearance_m: float = DEFAULT_OPENING_CLEARANCE_M,
) -> SurveySceneInput:
    workspace = build_workspace(
        layout,
        tank_opening_z_m=tank_opening_z_m,
        opening_clearance_m=opening_clearance_m,
    )
    return SurveySceneInput(
        source_path=layout.path,
        source_schema_version=layout.schema_version,
        seed=layout.seed,
        workspace=workspace,
        objects=tuple(
            SurveySceneObject(
                body_name=obj.object_id,
                class_name=obj.class_name,
                position=obj.position,
                yaw_rad=obj.yaw_rad,
                T_world_object=obj.T_world_object,
            )
            for obj in layout.objects
        ),
    )


def _survey_scene_input(
    scene_or_layout: SurveySceneInput | Stage0Layout,
    *,
    config: SurveyConfig,
) -> SurveySceneInput:
    if isinstance(scene_or_layout, SurveySceneInput):
        return scene_or_layout
    return survey_scene_from_stage0_layout(
        scene_or_layout,
        tank_opening_z_m=config.tank_opening_z_m,
        opening_clearance_m=config.opening_clearance_m,
    )


def build_grid_survey_plan(
    scene_or_layout: SurveySceneInput | Stage0Layout,
    config: SurveyConfig = SurveyConfig(),
) -> tuple[SurveyWorkspace, tuple[SurveyView, ...]]:
    if config.grid_size <= 0:
        raise ValueError("grid_size must be positive")
    if config.image_width <= 0 or config.image_height <= 0:
        raise ValueError("image dimensions must be positive")

    scene = _survey_scene_input(scene_or_layout, config=config)
    workspace = scene.workspace
    center_x = (workspace.x_min + workspace.x_max) / 2.0
    center_y = (workspace.y_min + workspace.y_max) / 2.0
    step_x = (workspace.x_max - workspace.x_min) / config.grid_size
    step_y = (workspace.y_max - workspace.y_min) / config.grid_size

    views: list[SurveyView] = []
    for row in range(config.grid_size):
        for col in range(config.grid_size):
            target_x = workspace.x_min + (col + 0.5) * step_x
            target_y = workspace.y_max - (row + 0.5) * step_y
            offset_x, offset_y = _oblique_offset(
                target_x=target_x,
                target_y=target_y,
                center_x=center_x,
                center_y=center_y,
                offset_m=config.oblique_offset_m,
            )
            camera_position = (
                _clamp(target_x + offset_x, workspace.x_min, workspace.x_max),
                _clamp(target_y + offset_y, workspace.y_min, workspace.y_max),
                config.camera_z_m,
            )
            workspace.validate_camera_position(camera_position)
            look_at = (target_x, target_y, workspace.bottom_z_m)
            view_index = row * config.grid_size + col
            views.append(
                SurveyView(
                    view_id=f"survey_{view_index:04d}",
                    grid_row=row,
                    grid_col=col,
                    camera_name=config.camera_name,
                    desired_camera_position_world=_round_vector(camera_position),
                    look_at_world=_round_vector(look_at),
                    T_world_camera=_make_look_at_transform(camera_position, look_at),
                )
            )
    return workspace, tuple(views)


def run_task1_survey(
    layout_path: Path | str,
    config: SurveyConfig = SurveyConfig(),
    *,
    detector: SurveyDetector | None = None,
) -> dict[str, Any]:
    layout = load_stage0_layout(layout_path)
    scene = survey_scene_from_stage0_layout(
        layout,
        tank_opening_z_m=config.tank_opening_z_m,
        opening_clearance_m=config.opening_clearance_m,
    )
    workspace, views = build_grid_survey_plan(scene, config)
    created_utc = config.created_utc or datetime.now(timezone.utc).isoformat()
    run_dir = config.run_dir or config.output_dir / _run_id(created_utc, scene)
    layout_dir = run_dir / "layout"
    survey_dir = run_dir / "survey"
    layout_snapshot_path = layout_dir / "target_object_poses.json"
    images_dir = survey_dir / "images"
    depth_dir = survey_dir / "depth"
    yolo_dir = survey_dir / "yolo_raw"
    annotated_dir = survey_dir / "annotated"
    tiles_dir = survey_dir / "tiles"
    plan_path = survey_dir / "survey_plan.json"
    report_path = survey_dir / "survey_report.json"
    plan_payload = _plan_payload(
        created_utc=created_utc,
        scene=scene,
        config=config,
        run_dir=run_dir,
        survey_dir=survey_dir,
        layout_snapshot_path=layout_snapshot_path,
        views=views,
    )

    if config.plan_only:
        _write_layout_snapshot(layout_snapshot_path, layout)
        _write_json(plan_path, plan_payload)
        report = _report_payload(
            created_utc=created_utc,
            status="plan_only",
            scene=scene,
            config=config,
            run_dir=run_dir,
            survey_dir=survey_dir,
            layout_snapshot_path=layout_snapshot_path,
            plan_path=plan_path,
            report_path=report_path,
            views=[_planned_view_result(view) for view in views],
            observations=[],
            candidates=[],
            message="Generated the task1 survey plan without MuJoCo rendering or YOLO inference.",
        )
        _write_json(report_path, report)
        return report

    if config.run_yolo and detector is None:
        _write_layout_snapshot(layout_snapshot_path, layout)
        _write_json(plan_path, plan_payload)
        report = _report_payload(
            created_utc=created_utc,
            status="failed",
            scene=scene,
            config=config,
            run_dir=run_dir,
            survey_dir=survey_dir,
            layout_snapshot_path=layout_snapshot_path,
            plan_path=plan_path,
            report_path=report_path,
            views=[_planned_view_result(view) for view in views],
            observations=[],
            candidates=[],
            message="run_yolo=True requires a SurveyDetector; pass one from the task1 entrypoint or set run_yolo=False.",
        )
        _write_json(report_path, report)
        return report

    backend = MujocoSurveyBackend(config=config, scene=scene, detector=detector)
    view_results: list[dict[str, Any]] = []
    observations: list[SurveyObservation] = []
    try:
        _write_layout_snapshot(layout_snapshot_path, layout)
        backend.load()
        backend.apply_scene_objects()
        for view in views:
            view_result, view_observations = backend.capture_view(
                view,
                images_dir=images_dir,
                depth_dir=depth_dir,
                yolo_dir=yolo_dir,
                annotated_dir=annotated_dir,
                tiles_dir=tiles_dir,
            )
            view_results.append(view_result)
            observations.extend(view_observations)
    except Exception as exc:
        _write_layout_snapshot(layout_snapshot_path, layout)
        _write_json(plan_path, plan_payload)
        report = _report_payload(
            created_utc=created_utc,
            status="failed",
            scene=scene,
            config=config,
            run_dir=run_dir,
            survey_dir=survey_dir,
            layout_snapshot_path=layout_snapshot_path,
            plan_path=plan_path,
            report_path=report_path,
            views=view_results or [_planned_view_result(view) for view in views],
            observations=[],
            candidates=[],
            message=str(exc),
        )
        _write_json(report_path, report)
        return report
    finally:
        backend.close()

    fusion_policy = config.fusion_policy()
    candidates = fuse_survey_observations(
        observations,
        policy=fusion_policy,
    )
    success_count = sum(1 for view in view_results if view.get("status") == "success")
    if success_count == len(view_results):
        status = "success"
    elif success_count > 0:
        status = "partial"
    else:
        status = "failed"
    _write_layout_snapshot(layout_snapshot_path, layout)
    _write_json(plan_path, plan_payload)
    report = _report_payload(
        created_utc=created_utc,
        status=status,
        scene=scene,
        config=config,
        run_dir=run_dir,
        survey_dir=survey_dir,
        layout_snapshot_path=layout_snapshot_path,
        plan_path=plan_path,
        report_path=report_path,
        views=view_results,
        observations=[observation.to_dict() for observation in observations],
        candidates=candidates,
        message=f"Captured {success_count}/{len(view_results)} task1 survey views.",
    )
    _write_json(report_path, report)
    return report


def fuse_survey_observations(
    observations: list[SurveyObservation],
    *,
    cluster_radius_m: float = DEFAULT_CLUSTER_RADIUS_M,
    cross_class_merge_radius_m: float = DEFAULT_CROSS_CLASS_MERGE_RADIUS_M,
    cluster_split_distance_m: float = DEFAULT_CLUSTER_SPLIT_DISTANCE_M,
    cluster_split_min_vote: float = DEFAULT_CLUSTER_SPLIT_MIN_VOTE,
    weak_candidate_merge_radius_m: float = DEFAULT_WEAK_CANDIDATE_MERGE_RADIUS_M,
    weak_candidate_vote_threshold: float = DEFAULT_WEAK_CANDIDATE_VOTE_THRESHOLD,
    tiny_candidate_bbox_area_px: float = DEFAULT_TINY_CANDIDATE_BBOX_AREA_PX,
    weak_tiny_candidate_max_spread_m: float = DEFAULT_WEAK_TINY_CANDIDATE_MAX_SPREAD_M,
    single_view_fallback_confidence: float = DEFAULT_SINGLE_VIEW_FALLBACK_CONFIDENCE,
    single_view_fallback_min_distance_m: float = DEFAULT_SINGLE_VIEW_FALLBACK_MIN_DISTANCE_M,
    min_candidate_support_views: int = DEFAULT_MIN_CANDIDATE_SUPPORT_VIEWS,
    large_same_class_merge_radius_m: float = DEFAULT_LARGE_SAME_CLASS_MERGE_RADIUS_M,
    large_same_class_bbox_area_px: float = DEFAULT_LARGE_SAME_CLASS_BBOX_AREA_PX,
    mixed_class_split_min_vote: float = DEFAULT_MIXED_CLASS_SPLIT_MIN_VOTE,
    mixed_class_split_distance_m: float = DEFAULT_MIXED_CLASS_SPLIT_DISTANCE_M,
    weak_multiview_fallback_vote: float = DEFAULT_WEAK_MULTIVIEW_FALLBACK_VOTE,
    weak_multiview_fallback_min_distance_m: float = DEFAULT_WEAK_MULTIVIEW_FALLBACK_MIN_DISTANCE_M,
    weak_multiview_fallback_max_bbox_area_px: float = DEFAULT_WEAK_MULTIVIEW_FALLBACK_MAX_BBOX_AREA_PX,
    weak_multiview_fallback_elongated_aspect_ratio: float = DEFAULT_WEAK_MULTIVIEW_FALLBACK_ELONGATED_ASPECT_RATIO,
    weak_multiview_fallback_elongated_max_bbox_area_px: float = (
        DEFAULT_WEAK_MULTIVIEW_FALLBACK_ELONGATED_MAX_BBOX_AREA_PX
    ),
    policy: SurveyFusionPolicy | None = None,
) -> list[dict[str, Any]]:
    if policy is None:
        policy = SurveyFusionPolicy(
            cluster_radius_m=cluster_radius_m,
            cross_class_merge_radius_m=cross_class_merge_radius_m,
            cluster_split_distance_m=cluster_split_distance_m,
            cluster_split_min_vote=cluster_split_min_vote,
            weak_candidate_merge_radius_m=weak_candidate_merge_radius_m,
            weak_candidate_vote_threshold=weak_candidate_vote_threshold,
            tiny_candidate_bbox_area_px=tiny_candidate_bbox_area_px,
            weak_tiny_candidate_max_spread_m=weak_tiny_candidate_max_spread_m,
            single_view_fallback_confidence=single_view_fallback_confidence,
            single_view_fallback_min_distance_m=single_view_fallback_min_distance_m,
            min_candidate_support_views=min_candidate_support_views,
            duplicate_large_same_class_merge_radius_m=large_same_class_merge_radius_m,
            duplicate_large_same_class_bbox_area_px=large_same_class_bbox_area_px,
            mixed_class_split_min_vote=mixed_class_split_min_vote,
            mixed_class_split_distance_m=mixed_class_split_distance_m,
            weak_multiview_fallback_vote=weak_multiview_fallback_vote,
            weak_multiview_fallback_min_distance_m=weak_multiview_fallback_min_distance_m,
            weak_multiview_fallback_max_bbox_area_px=weak_multiview_fallback_max_bbox_area_px,
            weak_multiview_fallback_elongated_aspect_ratio=weak_multiview_fallback_elongated_aspect_ratio,
            weak_multiview_fallback_elongated_max_bbox_area_px=(
                weak_multiview_fallback_elongated_max_bbox_area_px
            ),
        )
    policy.validate()

    clusters: list[list[SurveyObservation]] = []
    for observation in observations:
        if observation.rough_position_world is None:
            continue
        best_cluster: list[SurveyObservation] | None = None
        best_distance = float("inf")
        for cluster in clusters:
            center = _cluster_center(cluster)
            distance = _xy_distance(observation.rough_position_world, center)
            dominant_class, _ = _cluster_dominant_vote(cluster)
            same_class_or_unknown = (
                not observation.class_name
                or not dominant_class
                or observation.class_name == dominant_class
            )
            distance_limit = policy.cluster_radius_m if same_class_or_unknown else policy.cross_class_merge_radius_m
            if distance <= distance_limit and distance < best_distance:
                best_distance = distance
                best_cluster = cluster
        if best_cluster is not None:
            best_cluster.append(observation)
        else:
            clusters.append([observation])

    refined_clusters: list[list[SurveyObservation]] = []
    for cluster in clusters:
        refined_clusters.extend(
            _split_cluster_spatial_modes(
                cluster,
                split_distance_m=policy.cluster_split_distance_m,
                min_vote=policy.cluster_split_min_vote,
                min_support_views=policy.min_candidate_support_views,
            )
        )
    split_clusters: list[list[SurveyObservation]] = []
    for cluster in refined_clusters:
        split_clusters.extend(
            _split_cluster_spatial_modes(
                cluster,
                split_distance_m=policy.cluster_split_distance_m,
                min_vote=policy.cluster_split_min_vote,
                min_support_views=policy.min_candidate_support_views,
            )
        )
    clusters = _merge_weak_neighbor_clusters(
        split_clusters,
        merge_radius_m=policy.weak_candidate_merge_radius_m,
        weak_vote_threshold=policy.weak_candidate_vote_threshold,
    )
    clusters = _merge_duplicate_large_same_class_clusters(
        clusters,
        merge_radius_m=policy.duplicate_large_same_class_merge_radius_m,
        large_bbox_area_px=policy.duplicate_large_same_class_bbox_area_px,
    )
    clusters = _split_mixed_class_clusters(
        clusters,
        min_vote=policy.mixed_class_split_min_vote,
        min_support_views=policy.min_candidate_support_views,
        split_distance_m=policy.mixed_class_split_distance_m,
    )
    clusters = _merge_duplicate_large_same_class_clusters(
        clusters,
        merge_radius_m=policy.duplicate_large_same_class_merge_radius_m,
        large_bbox_area_px=policy.duplicate_large_same_class_bbox_area_px,
    )

    candidates: list[dict[str, Any]] = []
    kept_clusters: list[list[SurveyObservation]] = []
    rejected_clusters: list[list[SurveyObservation]] = []
    for cluster in clusters:
        if not _keep_candidate_cluster(
            cluster,
            min_support_views=policy.min_candidate_support_views,
            weak_vote_threshold=policy.weak_candidate_vote_threshold,
            tiny_bbox_area_px=policy.tiny_candidate_bbox_area_px,
            weak_tiny_max_spread_m=policy.weak_tiny_candidate_max_spread_m,
        ):
            rejected_clusters.append(cluster)
            continue
        kept_clusters.append(cluster)
        candidates.append(_candidate_payload(len(candidates), cluster, single_view_fallback=False))

    weak_multiview_fallbacks = _select_weak_multiview_fallback_clusters(
        rejected_clusters,
        accepted_clusters=kept_clusters,
        min_support_views=policy.min_candidate_support_views,
        min_vote=policy.weak_multiview_fallback_vote,
        min_distance_m=policy.weak_multiview_fallback_min_distance_m,
        max_bbox_area_px=policy.weak_multiview_fallback_max_bbox_area_px,
        elongated_aspect_ratio=policy.weak_multiview_fallback_elongated_aspect_ratio,
        elongated_max_bbox_area_px=policy.weak_multiview_fallback_elongated_max_bbox_area_px,
    )
    weak_multiview_ids = {id(cluster) for cluster in weak_multiview_fallbacks}
    for cluster in weak_multiview_fallbacks:
        kept_clusters.append(cluster)
        candidates.append(
            _candidate_payload(
                len(candidates),
                cluster,
                single_view_fallback=False,
                note_override="Survey candidate from weak but spatially distinct multi-view class evidence.",
            )
        )

    for cluster in rejected_clusters:
        if id(cluster) in weak_multiview_ids:
            continue
        if not _keep_single_view_fallback_cluster(
            cluster,
            accepted_clusters=kept_clusters,
            confidence_threshold=policy.single_view_fallback_confidence,
            min_distance_m=policy.single_view_fallback_min_distance_m,
        ):
            continue
        kept_clusters.append(cluster)
        candidates.append(
            _candidate_payload(
                len(candidates),
                cluster,
                single_view_fallback=True,
            )
        )
    return candidates


class MujocoSurveyBackend:
    def __init__(
        self,
        *,
        config: SurveyConfig,
        scene: SurveySceneInput,
        detector: SurveyDetector | None = None,
    ) -> None:
        self.config = config
        self.scene = scene
        self.workspace = scene.workspace
        self.detector = detector
        self.mujoco: Any | None = None
        self.model: Any | None = None
        self.data: Any | None = None
        self.renderer: Any | None = None
        self.camera_id: int | None = None
        self.resolved_camera_name: str | None = None

    def load(self) -> None:
        try:
            import mujoco  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeError("MuJoCo is required for task1 survey capture. Use the mujoco-curobo conda env.") from exc

        if not self.config.scene_model_path.exists():
            raise FileNotFoundError(f"MuJoCo scene file does not exist: {self.config.scene_model_path}")

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(self.config.scene_model_path.resolve()))
        self._ensure_offscreen_framebuffer_size()
        self.data = mujoco.MjData(self.model)
        self._apply_home_keyframe()
        mujoco.mj_forward(self.model, self.data)
        self.camera_id, self.resolved_camera_name = self._resolve_camera(self.config.camera_name)
        try:
            self.renderer = mujoco.Renderer(
                self.model,
                height=self.config.image_height,
                width=self.config.image_width,
            )
        except Exception as exc:
            raise RuntimeError(
                "MuJoCo renderer could not create an OpenGL context. "
                "Run inside a display session or set MUJOCO_GL=egl/osmesa in the mujoco-curobo environment. "
                "If the error mentions framebuffer width/height, increase <visual><global offwidth/offheight> "
                "or let the task1 survey backend resize model.vis.global_ before creating Renderer."
            ) from exc

    def close(self) -> None:
        if self.renderer is not None and hasattr(self.renderer, "close"):
            self.renderer.close()

    def _ensure_offscreen_framebuffer_size(self) -> None:
        model = self._model()
        if int(model.vis.global_.offwidth) < self.config.image_width:
            model.vis.global_.offwidth = int(self.config.image_width)
        if int(model.vis.global_.offheight) < self.config.image_height:
            model.vis.global_.offheight = int(self.config.image_height)

    def apply_scene_objects(self) -> None:
        mujoco = self._mujoco()
        model = self._model()
        data = self._data()
        selected = {obj.body_name: obj for obj in self.scene.objects}
        for body_name in self._target_body_names():
            body_id = self._name_to_id(mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_name in selected:
                obj = selected[body_name]
                model.body_pos[body_id] = obj.position
                model.body_quat[body_id] = _yaw_quat_wxyz(obj.yaw_rad)
                self._set_body_geom_alpha(body_id, alpha=1.0)
            else:
                model.body_pos[body_id] = (0.0, 0.0, -10.0)
                model.body_quat[body_id] = (1.0, 0.0, 0.0, 0.0)
                self._set_body_geom_alpha(body_id, alpha=0.0)
        mujoco.mj_forward(model, data)

    def apply_stage0_layout(self) -> None:
        self.apply_scene_objects()

    def validate_view_pose(self, view: SurveyView) -> dict[str, Any]:
        result = _planned_view_result(view)
        try:
            ik = self._solve_ik_for_view(view)
            result["ik"] = ik
            result["resolved_camera_name"] = self.resolved_camera_name
            actual_position = self._camera_position()
            result["actual_camera_position_world"] = [_round(value) for value in actual_position]
            result["actual_T_world_camera"] = [list(row) for row in self._camera_transform()]
            result["camera_fovy_rad"] = _round(self._camera_fovy_rad())
            self.workspace.validate_camera_position(actual_position)
            collision = self._collision_report()
            result["collision"] = collision
            if not ik["success"]:
                result["status"] = "failed"
                result["message"] = "IK did not reach the requested wrist-camera pose."
                return result
            if not collision["collision_free"]:
                result["status"] = "failed"
                result["message"] = "IK pose was rejected by robot collision check."
                return result
            result["status"] = "success"
            result["message"] = "Validated the requested wrist-camera pose with IK and collision checks."
            return result
        except Exception as exc:
            result["status"] = "failed"
            result["message"] = str(exc)
            return result

    def capture_view(
        self,
        view: SurveyView,
        *,
        images_dir: Path,
        depth_dir: Path,
        yolo_dir: Path,
        annotated_dir: Path,
        tiles_dir: Path,
    ) -> tuple[dict[str, Any], list[SurveyObservation]]:
        result = _planned_view_result(view)
        try:
            result = self.validate_view_pose(view)
            if result.get("status") != "success":
                return result, []

            images_dir.mkdir(parents=True, exist_ok=True)
            depth_dir.mkdir(parents=True, exist_ok=True)
            rgb_path, depth_path, depth = self._render(view, images_dir=images_dir, depth_dir=depth_dir)
            result["rgb_image_path"] = str(rgb_path)
            result["depth_path"] = str(depth_path)
            raw_yolo_payload = self._run_yolo(view, rgb_path, yolo_dir, tiles_dir=tiles_dir)
            result["yolo_raw_path"] = raw_yolo_payload.get("_path")
            annotated_path = _write_annotated_image(
                rgb_path=rgb_path,
                output_path=annotated_dir / f"{view.view_id}_yolo.png",
                detections=raw_yolo_payload.get("detections", []),
                image_width=self.config.image_width,
                image_height=self.config.image_height,
            )
            result["annotated_image_path"] = str(annotated_path) if annotated_path is not None else None
            result["yolo"] = {
                "status": raw_yolo_payload.get("_status", "success"),
                "detections": len(raw_yolo_payload.get("detections", [])),
                "message": raw_yolo_payload.get("_message"),
                "tile_grid_size": raw_yolo_payload.get("_tile_grid_size", 1),
                "tile_raw_detections": raw_yolo_payload.get("_tile_raw_detections", 0),
            }
            observations = self._observations_from_yolo(view, raw_yolo_payload, rgb_path, depth)
            result["status"] = "success"
            result["message"] = "Captured survey image, depth, and YOLO raw detections."
            return result, observations
        except Exception as exc:
            result["status"] = "failed"
            result["message"] = str(exc)
            return result, []

    def _solve_ik_for_view(self, view: SurveyView) -> dict[str, Any]:
        mujoco = self._mujoco()
        model = self._model()
        data = self._data()
        np = _import_numpy()
        camera_id = self._camera_id()
        target_position = np.asarray(view.desired_camera_position_world, dtype=float)
        target_rotation = np.asarray([row[:3] for row in view.T_world_camera[:3]], dtype=float)
        camera_body_id = int(model.cam_bodyid[camera_id])
        nv = int(model.nv)
        qpos_count = min(int(model.nq), nv)
        last_position_error = float("inf")
        last_orientation_error = float("inf")

        for iteration in range(self.config.max_ik_iterations):
            mujoco.mj_forward(model, data)
            current_position = np.asarray(data.cam_xpos[camera_id], dtype=float)
            current_rotation = np.asarray(data.cam_xmat[camera_id], dtype=float).reshape(3, 3)
            position_error = target_position - current_position
            orientation_error = _orientation_error(np, current_rotation, target_rotation)
            last_position_error = float(np.linalg.norm(position_error))
            last_orientation_error = float(np.linalg.norm(orientation_error))
            if (
                last_position_error <= self.config.ik_position_tolerance_m
                and last_orientation_error <= self.config.ik_orientation_tolerance_rad
            ):
                return {
                    "success": True,
                    "iterations": iteration,
                    "position_error_m": _round(last_position_error),
                    "orientation_error_rad": _round(last_orientation_error),
                }

            jacp = np.zeros((3, nv), dtype=float)
            jacr = np.zeros((3, nv), dtype=float)
            mujoco.mj_jac(model, data, jacp, jacr, current_position, camera_body_id)
            jacobian = np.vstack((jacp, 0.35 * jacr))
            error = np.concatenate((position_error, 0.35 * orientation_error))
            lhs = jacobian @ jacobian.T + float(self.config.ik_damping) * np.eye(6)
            dq = jacobian.T @ np.linalg.solve(lhs, error)
            step_norm = float(np.linalg.norm(dq))
            if step_norm > 0.08:
                dq *= 0.08 / step_norm
            data.qpos[:qpos_count] += dq[:qpos_count]
            self._clip_joint_limits()

        mujoco.mj_forward(model, data)
        return {
            "success": False,
            "iterations": self.config.max_ik_iterations,
            "position_error_m": _round(last_position_error),
            "orientation_error_rad": _round(last_orientation_error),
        }

    def _render(self, view: SurveyView, *, images_dir: Path, depth_dir: Path) -> tuple[Path, Path, Any]:
        mujoco = self._mujoco()
        renderer = self._renderer()
        data = self._data()
        camera_name = self.resolved_camera_name or view.camera_name
        renderer.disable_depth_rendering()
        renderer.update_scene(data, camera=camera_name)
        rgb = renderer.render()
        rgb_path = _write_rgb_image(images_dir / f"{view.view_id}_rgb.png", rgb)
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=camera_name)
        depth = renderer.render()
        renderer.disable_depth_rendering()
        depth_path = depth_dir / f"{view.view_id}_depth.npy"
        _import_numpy().save(depth_path, depth)
        mujoco.mj_forward(self._model(), data)
        return rgb_path, depth_path, depth

    def _run_yolo(self, view: SurveyView, rgb_path: Path, yolo_dir: Path, *, tiles_dir: Path) -> dict[str, Any]:
        if not self.config.run_yolo:
            return {"_status": "skipped", "_message": "YOLO inference was skipped.", "detections": []}
        yolo_dir.mkdir(parents=True, exist_ok=True)
        output_path = yolo_dir / f"{view.view_id}.json"
        try:
            runner = self._yolo_inference_runner()
            payload = runner.predict(
                image_path=rgb_path,
                camera_name=self.config.camera_name,
            )
            full_detections = [_with_detection_source(detection, source="full_frame") for detection in payload.get("detections", [])]
            tile_detections = self._run_yolo_tiles(view, rgb_path, tiles_dir=tiles_dir)
            payload["detections"] = _merge_yolo_detections(
                full_detections + tile_detections,
                image_width=self.config.image_width,
                image_height=self.config.image_height,
                iou_threshold=self.config.yolo_tile_nms_iou,
            )
            inference = payload.get("inference")
            if isinstance(inference, dict):
                inference["tile_grid_size"] = self.config.yolo_tile_grid_size
                inference["tile_overlap"] = self.config.yolo_tile_overlap
                inference["tile_nms_iou"] = self.config.yolo_tile_nms_iou
            payload["_tile_grid_size"] = self.config.yolo_tile_grid_size
            payload["_tile_raw_detections"] = len(tile_detections)
            _write_json(output_path, payload)
            payload["_path"] = str(output_path)
            payload["_status"] = "success"
            return payload
        except Exception as exc:
            if self.config.strict_yolo:
                raise
            payload = {
                "schema_version": "yolo_raw_detections_v1",
                "image_path": str(rgb_path),
                "camera_name": self.config.camera_name,
                "detections": [],
                "_status": "failed",
                "_message": str(exc),
                "_path": str(output_path),
            }
            _write_json(output_path, payload)
            return payload

    def _yolo_inference_runner(self) -> Any:
        if self.detector is None:
            raise RuntimeError("task1 survey YOLO is enabled, but no SurveyDetector was supplied")
        return self.detector

    def _run_yolo_tiles(self, view: SurveyView, rgb_path: Path, *, tiles_dir: Path) -> list[dict[str, Any]]:
        if self.config.yolo_tile_grid_size <= 1:
            return []
        if not 0.0 <= self.config.yolo_tile_overlap < 0.5:
            raise ValueError("yolo_tile_overlap must be in [0.0, 0.5)")

        try:
            from PIL import Image  # type: ignore[import-not-found]
        except ModuleNotFoundError:
            return []

        runner = self._yolo_inference_runner()
        detections: list[dict[str, Any]] = []
        tile_root = tiles_dir / view.view_id
        tile_root.mkdir(parents=True, exist_ok=True)
        with Image.open(rgb_path).convert("RGB") as image:
            for tile_id, rect in enumerate(
                _image_tile_rects(
                    image_width=self.config.image_width,
                    image_height=self.config.image_height,
                    grid_size=self.config.yolo_tile_grid_size,
                    overlap=self.config.yolo_tile_overlap,
                )
            ):
                left, top, right, bottom = rect
                tile_path = tile_root / f"tile_{tile_id:02d}.png"
                image.crop((left, top, right, bottom)).save(tile_path)
                tile_payload = runner.predict(image_path=tile_path, camera_name=self.config.camera_name)
                for detection in tile_payload.get("detections", []):
                    if not isinstance(detection, dict) or "bbox_xyxy" not in detection:
                        continue
                    mapped = dict(detection)
                    mapped["bbox_xyxy"] = _offset_bbox_xyxy(
                        _float_tuple_from_value(detection["bbox_xyxy"], 4),
                        offset_x=float(left),
                        offset_y=float(top),
                    )
                    mapped["source"] = "tile"
                    mapped["tile_id"] = f"{view.view_id}_tile_{tile_id:02d}"
                    mapped["tile_bbox_xyxy"] = [float(left), float(top), float(right), float(bottom)]
                    mapped["tile_local_bbox_xyxy"] = list(_float_tuple_from_value(detection["bbox_xyxy"], 4))
                    detections.append(mapped)
        return detections

    def _observations_from_yolo(
        self,
        view: SurveyView,
        raw_payload: dict[str, Any],
        rgb_path: Path,
        depth: Any,
    ) -> list[SurveyObservation]:
        detections = raw_payload.get("detections")
        if not isinstance(detections, list):
            return []
        observations: list[SurveyObservation] = []
        camera_transform = self._camera_transform()
        fovy = self._camera_fovy_rad()
        for detection in detections:
            if not isinstance(detection, dict) or "bbox_xyxy" not in detection:
                continue
            bbox = _float_tuple_from_value(detection["bbox_xyxy"], 4)
            rough_positions = estimate_detection_ground_positions(
                bbox_xyxy=bbox,
                depth=depth,
                T_world_camera=camera_transform,
                image_width=self.config.image_width,
                image_height=self.config.image_height,
                fovy_rad=fovy,
                ground_z_m=self.scene.workspace.bottom_z_m,
                depth_sample_stride_px=self.config.depth_sample_stride_px,
                depth_component_min_pixels=self.config.depth_component_min_pixels,
                depth_component_split_distance_m=self.config.depth_component_split_distance_m,
            )
            confidence = float(detection.get("confidence", detection.get("score", detection.get("conf", 0.0))))
            split_count = max(1, len(rough_positions))
            for rough_position in rough_positions:
                observations.append(
                    SurveyObservation(
                        view_id=view.view_id,
                        image_path=str(rgb_path),
                        bbox_xyxy=bbox,
                        confidence=confidence / split_count,
                        class_id=int(detection["class_id"]) if "class_id" in detection else None,
                        class_name=str(detection["class_name"]) if "class_name" in detection else None,
                        rough_position_world=rough_position,
                    )
                )
        return observations

    def _collision_report(self) -> dict[str, Any]:
        mujoco = self._mujoco()
        model = self._model()
        data = self._data()
        mujoco.mj_forward(model, data)
        robot_scene_contacts: list[dict[str, str]] = []
        robot_self_contacts: list[dict[str, str]] = []
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            name1 = self._geom_name(int(contact.geom1))
            name2 = self._geom_name(int(contact.geom2))
            is_robot1 = name1.startswith("gen3_")
            is_robot2 = name2.startswith("gen3_")
            if is_robot1 and is_robot2:
                robot_self_contacts.append({"geom1": name1, "geom2": name2})
            elif is_robot1 or is_robot2:
                robot_scene_contacts.append({"geom1": name1, "geom2": name2})
        return {
            "collision_free": not robot_scene_contacts and not robot_self_contacts,
            "robot_scene_contact_count": len(robot_scene_contacts),
            "robot_self_contact_count": len(robot_self_contacts),
            "robot_scene_contacts": robot_scene_contacts[:10],
            "robot_self_contacts": robot_self_contacts[:10],
        }

    def _clip_joint_limits(self) -> None:
        model = self._model()
        data = self._data()
        for joint_id in range(int(model.njnt)):
            if int(model.jnt_limited[joint_id]) == 0:
                continue
            qpos_addr = int(model.jnt_qposadr[joint_id])
            low, high = float(model.jnt_range[joint_id][0]), float(model.jnt_range[joint_id][1])
            data.qpos[qpos_addr] = min(max(float(data.qpos[qpos_addr]), low), high)

    def _camera_transform(self) -> tuple[tuple[float, float, float, float], ...]:
        camera_id = self._camera_id()
        data = self._data()
        rotation = data.cam_xmat[camera_id].reshape(3, 3)
        position = data.cam_xpos[camera_id]
        return (
            (float(rotation[0][0]), float(rotation[0][1]), float(rotation[0][2]), float(position[0])),
            (float(rotation[1][0]), float(rotation[1][1]), float(rotation[1][2]), float(position[1])),
            (float(rotation[2][0]), float(rotation[2][1]), float(rotation[2][2]), float(position[2])),
            (0.0, 0.0, 0.0, 1.0),
        )

    def _camera_position(self) -> tuple[float, float, float]:
        camera_id = self._camera_id()
        position = self._data().cam_xpos[camera_id]
        return (float(position[0]), float(position[1]), float(position[2]))

    def _camera_fovy_rad(self) -> float:
        model = self._model()
        return math.radians(float(model.cam_fovy[self._camera_id()]))

    def _resolve_camera(self, camera_name: str) -> tuple[int, str]:
        mujoco = self._mujoco()
        for candidate in (camera_name, f"gen3_{camera_name}"):
            camera_id = mujoco.mj_name2id(self._model(), mujoco.mjtObj.mjOBJ_CAMERA, candidate)
            if camera_id >= 0:
                return int(camera_id), candidate
        raise ValueError(f"camera not found in MuJoCo scene: {camera_name} (also tried gen3_{camera_name})")

    def _apply_home_keyframe(self) -> None:
        mujoco = self._mujoco()
        model = self._model()
        data = self._data()
        for name in ("gen3_home", "home"):
            key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
            if key_id >= 0:
                mujoco.mj_resetDataKeyframe(model, data, key_id)
                return

    def _target_body_names(self) -> tuple[str, ...]:
        mujoco = self._mujoco()
        model = self._model()
        names: list[str] = []
        for body_id in range(int(model.nbody)):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if body_name.startswith("target_") and body_name != "target_object_include_root":
                names.append(body_name)
        return tuple(names)

    def _set_body_geom_alpha(self, body_id: int, *, alpha: float) -> None:
        model = self._model()
        for geom_id in range(int(model.ngeom)):
            if int(model.geom_bodyid[geom_id]) == body_id:
                model.geom_rgba[geom_id][3] = alpha

    def _name_to_id(self, object_type: Any, name: str) -> int:
        mujoco = self._mujoco()
        object_id = int(mujoco.mj_name2id(self._model(), object_type, name))
        if object_id < 0:
            raise ValueError(f"MuJoCo object not found: {name}")
        return object_id

    def _geom_name(self, geom_id: int) -> str:
        mujoco = self._mujoco()
        return mujoco.mj_id2name(self._model(), mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"

    def _mujoco(self) -> Any:
        if self.mujoco is None:
            raise RuntimeError("MuJoCo backend has not been loaded")
        return self.mujoco

    def _model(self) -> Any:
        if self.model is None:
            raise RuntimeError("MuJoCo model has not been loaded")
        return self.model

    def _data(self) -> Any:
        if self.data is None:
            raise RuntimeError("MuJoCo data has not been loaded")
        return self.data

    def _renderer(self) -> Any:
        if self.renderer is None:
            raise RuntimeError("MuJoCo renderer has not been loaded")
        return self.renderer

    def _camera_id(self) -> int:
        if self.camera_id is None:
            raise RuntimeError("MuJoCo camera has not been resolved")
        return self.camera_id


def estimate_detection_ground_positions(
    *,
    bbox_xyxy: tuple[float, float, float, float],
    depth: Any | None = None,
    T_world_camera: tuple[tuple[float, float, float, float], ...],
    image_width: int,
    image_height: int,
    fovy_rad: float,
    ground_z_m: float,
    depth_sample_stride_px: int = DEFAULT_DEPTH_SAMPLE_STRIDE_PX,
    depth_component_min_pixels: int = DEFAULT_DEPTH_COMPONENT_MIN_PIXELS,
    depth_component_split_distance_m: float = DEFAULT_DEPTH_COMPONENT_SPLIT_DISTANCE_M,
) -> tuple[tuple[float, float, float], ...]:
    if depth is not None:
        depth_positions = _bbox_depth_component_positions(
            depth=depth,
            bbox_xyxy=bbox_xyxy,
            T_world_camera=T_world_camera,
            image_width=image_width,
            image_height=image_height,
            fovy_rad=fovy_rad,
            ground_z_m=ground_z_m,
            sample_stride_px=depth_sample_stride_px,
            min_pixels=depth_component_min_pixels,
            split_distance_m=depth_component_split_distance_m,
        )
        if depth_positions:
            return depth_positions

    position = estimate_detection_ground_position(
        bbox_xyxy=bbox_xyxy,
        depth=depth,
        T_world_camera=T_world_camera,
        image_width=image_width,
        image_height=image_height,
        fovy_rad=fovy_rad,
        ground_z_m=ground_z_m,
    )
    return (position,) if position is not None else ()


def estimate_detection_ground_position(
    *,
    bbox_xyxy: tuple[float, float, float, float],
    depth: Any | None = None,
    T_world_camera: tuple[tuple[float, float, float, float], ...],
    image_width: int,
    image_height: int,
    fovy_rad: float,
    ground_z_m: float,
) -> tuple[float, float, float] | None:
    np = _import_numpy()
    x1, y1, x2, y2 = bbox_xyxy
    u = _clamp((x1 + x2) / 2.0, 0.0, float(image_width - 1))
    v = _clamp((y1 + y2) / 2.0, 0.0, float(image_height - 1))
    fy = (image_height / 2.0) / math.tan(fovy_rad / 2.0)
    fx = fy
    x = (u - image_width / 2.0) / fx
    y = (image_height / 2.0 - v) / fy
    ray_camera = np.asarray((x, y, -1.0), dtype=float)
    ray_camera /= np.linalg.norm(ray_camera)
    rotation = np.asarray([row[:3] for row in T_world_camera[:3]], dtype=float)
    position = np.asarray((T_world_camera[0][3], T_world_camera[1][3], T_world_camera[2][3]), dtype=float)
    if depth is not None:
        depth_m = _bbox_foreground_depth_m(np, depth, bbox_xyxy, image_width=image_width, image_height=image_height)
        if depth_m is not None and depth_m > 0.0:
            point_camera = np.asarray((x * depth_m, y * depth_m, -depth_m), dtype=float)
            point = position + rotation @ point_camera
            return (_round(float(point[0])), _round(float(point[1])), _round(float(ground_z_m)))

    ray_world = rotation @ ray_camera
    ray_world /= np.linalg.norm(ray_world)
    if abs(float(ray_world[2])) < 1e-9:
        return None
    t = (ground_z_m - float(position[2])) / float(ray_world[2])
    if t <= 0.0:
        return None
    point = position + t * ray_world
    return (_round(float(point[0])), _round(float(point[1])), _round(float(ground_z_m)))


def _bbox_foreground_depth_m(
    np: Any,
    depth: Any,
    bbox_xyxy: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
) -> float | None:
    x1, y1, x2, y2 = bbox_xyxy
    left = max(0, min(image_width - 1, int(math.floor(x1))))
    right = max(left + 1, min(image_width, int(math.ceil(x2))))
    top = max(0, min(image_height - 1, int(math.floor(y1))))
    bottom = max(top + 1, min(image_height, int(math.ceil(y2))))
    depth_array = np.asarray(depth)
    crop = depth_array[top:bottom, left:right]
    if crop.size == 0:
        return None
    values = crop[np.isfinite(crop)]
    values = values[values > 0.0]
    if values.size == 0:
        return None
    return float(np.percentile(values, 10.0))


def _bbox_depth_component_positions(
    *,
    depth: Any,
    bbox_xyxy: tuple[float, float, float, float],
    T_world_camera: tuple[tuple[float, float, float, float], ...],
    image_width: int,
    image_height: int,
    fovy_rad: float,
    ground_z_m: float,
    sample_stride_px: int,
    min_pixels: int,
    split_distance_m: float,
) -> tuple[tuple[float, float, float], ...]:
    np = _import_numpy()
    if sample_stride_px <= 0 or min_pixels <= 0 or split_distance_m <= 0:
        return ()
    left, top, right, bottom = _bbox_pixel_bounds(
        bbox_xyxy,
        image_width=image_width,
        image_height=image_height,
    )
    if right <= left or bottom <= top:
        return ()

    depth_array = np.asarray(depth)
    ys = np.arange(top, bottom, sample_stride_px, dtype=int)
    xs = np.arange(left, right, sample_stride_px, dtype=int)
    if ys.size == 0 or xs.size == 0:
        return ()
    grid_x, grid_y = np.meshgrid(xs, ys)
    depth_values = depth_array[grid_y, grid_x]
    valid = np.isfinite(depth_values) & (depth_values > 0.0)
    if int(np.count_nonzero(valid)) < min_pixels:
        return ()

    valid_depths = depth_values[valid]
    mode_positions = _depth_mode_component_positions(
        np=np,
        pixel_x=grid_x[valid],
        pixel_y=grid_y[valid],
        depth_values=valid_depths,
        T_world_camera=T_world_camera,
        image_width=image_width,
        image_height=image_height,
        fovy_rad=fovy_rad,
        ground_z_m=ground_z_m,
        min_pixels=min_pixels,
        split_distance_m=split_distance_m,
    )
    if mode_positions:
        return mode_positions

    foreground_limit = float(np.percentile(valid_depths, DEFAULT_DEPTH_FOREGROUND_PERCENTILE)) + DEFAULT_DEPTH_FOREGROUND_MARGIN_M
    foreground = valid_depths <= foreground_limit
    if int(np.count_nonzero(foreground)) < min_pixels:
        return ()
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
    return (_world_points_center(np, points, ground_z_m=ground_z_m),)


def _depth_mode_component_positions(
    *,
    np: Any,
    pixel_x: Any,
    pixel_y: Any,
    depth_values: Any,
    T_world_camera: tuple[tuple[float, float, float, float], ...],
    image_width: int,
    image_height: int,
    fovy_rad: float,
    ground_z_m: float,
    min_pixels: int,
    split_distance_m: float,
) -> tuple[tuple[float, float, float], ...]:
    if depth_values.size < min_pixels * 2:
        return ()
    depth_min = float(np.min(depth_values))
    depth_max = float(np.max(depth_values))
    if depth_max - depth_min < DEFAULT_DEPTH_MODE_BAND_M * 2.0:
        return ()
    bins = min(48, max(12, int(math.sqrt(float(depth_values.size)))))
    counts, edges = np.histogram(depth_values, bins=bins)
    candidate_bins = sorted(range(len(counts)), key=lambda index: int(counts[index]), reverse=True)
    modes: list[tuple[float, int]] = []
    for bin_index in candidate_bins:
        count = int(counts[bin_index])
        if count < min_pixels:
            continue
        center = float((edges[bin_index] + edges[bin_index + 1]) / 2.0)
        if all(abs(center - existing_center) >= DEFAULT_DEPTH_MODE_BAND_M * 2.0 for existing_center, _ in modes):
            modes.append((center, count))
        if len(modes) == 2:
            break
    if len(modes) < 2:
        return ()

    positions: list[tuple[float, float, float]] = []
    for center, _ in sorted(modes, key=lambda item: item[0]):
        in_band = np.abs(depth_values - center) <= DEFAULT_DEPTH_MODE_BAND_M
        if int(np.count_nonzero(in_band)) < min_pixels:
            return ()
        points = _project_pixels_to_world(
            np=np,
            pixel_x=pixel_x[in_band],
            pixel_y=pixel_y[in_band],
            depth_values=depth_values[in_band],
            T_world_camera=T_world_camera,
            image_width=image_width,
            image_height=image_height,
            fovy_rad=fovy_rad,
        )
        positions.append(_world_points_center(np, points, ground_z_m=ground_z_m))
    if len(positions) != 2 or _xy_distance(positions[0], positions[1]) < split_distance_m:
        return ()
    return tuple(sorted(positions, key=lambda position: (position[0], position[1])))


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


def _world_points_center(np: Any, points: Any, *, ground_z_m: float) -> tuple[float, float, float]:
    if points.size == 0:
        raise ValueError("cannot compute center of empty world points")
    return (
        _round(float(np.median(points[:, 0]))),
        _round(float(np.median(points[:, 1]))),
        _round(float(ground_z_m)),
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


def _plan_payload(
    *,
    created_utc: str,
    scene: SurveySceneInput,
    config: SurveyConfig,
    run_dir: Path,
    survey_dir: Path,
    layout_snapshot_path: Path,
    views: tuple[SurveyView, ...],
) -> dict[str, Any]:
    return {
        "schema_version": "task1_survey_plan_v1",
        "stage": "survey",
        "created_utc": created_utc,
        "layout_source_path": str(scene.source_path),
        "layout_snapshot_path": str(layout_snapshot_path),
        "task1_run_dir": str(run_dir),
        "survey_dir": str(survey_dir),
        "source_schema_version": scene.source_schema_version,
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
            "pose_validation": "MuJoCo IK plus robot collision check before rendering.",
        },
        "workspace": scene.workspace.to_dict(),
        "selected_object_ids": list(scene.selected_object_ids),
        "views": [view.to_dict() for view in views],
    }


def _report_payload(
    *,
    created_utc: str,
    status: str,
    scene: SurveySceneInput,
    config: SurveyConfig,
    run_dir: Path,
    survey_dir: Path,
    layout_snapshot_path: Path,
    plan_path: Path,
    report_path: Path,
    views: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    message: str,
) -> dict[str, Any]:
    return {
        "schema_version": "task1_survey_report_v1",
        "stage": "survey",
        "status": status,
        "created_utc": created_utc,
        "message": message,
        "layout_source_path": str(scene.source_path),
        "layout_snapshot_path": str(layout_snapshot_path),
        "task1_run_dir": str(run_dir),
        "survey_dir": str(survey_dir),
        "source_schema_version": scene.source_schema_version,
        "scene_model_path": str(config.scene_model_path),
        "plan_path": str(plan_path),
        "report_path": str(report_path),
        "camera_name": config.camera_name,
        "workspace": scene.workspace.to_dict(),
        "grid_size": config.grid_size,
        "image_size": [config.image_width, config.image_height],
        "run_yolo": config.run_yolo,
        "yolo_profile_path": str(config.yolo_profile_path),
        "selected_objects": [obj.to_dict() for obj in scene.objects],
        "views": views,
        "observations": observations,
        "candidates": candidates,
        "notes": [
            "Survey is a multi-position scan for object candidate recall, not final classification.",
            "Partial detections are allowed here; later rough/final stages should estimate precise pose and full-view images.",
            "Do not feed survey bbox-only detections directly into planning; Stage3 still requires T_world_object.",
        ],
    }


def _planned_view_result(view: SurveyView) -> dict[str, Any]:
    return {
        "view_id": view.view_id,
        "grid_row": view.grid_row,
        "grid_col": view.grid_col,
        "desired_camera_position_world": list(view.desired_camera_position_world),
        "look_at_world": list(view.look_at_world),
        "status": "planned",
    }


def _stage0_object(payload: Any) -> Stage0Object:
    if not isinstance(payload, dict):
        raise ValueError("target object layout objects must be JSON objects")
    return Stage0Object(
        object_id=_required_str(payload, "object_id"),
        class_name=_required_str(payload, "class_name"),
        position=_float_tuple_from_value(_required(payload, "position"), 3),
        yaw_rad=float(_required(payload, "yaw_rad")),
        T_world_object=_transform_matrix(_required(payload, "T_world_object")),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_layout_snapshot(path: Path, layout: Stage0Layout) -> None:
    _write_json(path, layout.raw_payload)


def _write_rgb_image(path: Path, rgb: Any) -> Path:
    try:
        from PIL import Image  # type: ignore[import-not-found]

        Image.fromarray(rgb).save(path)
        return path
    except ModuleNotFoundError:
        ppm_path = path.with_suffix(".ppm")
        array = _import_numpy().asarray(rgb)
        height, width = int(array.shape[0]), int(array.shape[1])
        ppm_path.parent.mkdir(parents=True, exist_ok=True)
        with ppm_path.open("wb") as handle:
            handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
            handle.write(array[:, :, :3].astype("uint8").tobytes())
        return ppm_path


def _write_annotated_image(
    *,
    rgb_path: Path,
    output_path: Path,
    detections: Any,
    image_width: int,
    image_height: int,
) -> Path | None:
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return None

    try:
        image = Image.open(rgb_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = _annotation_font(ImageFont)
        if isinstance(detections, list):
            for index, detection in enumerate(detections):
                if not isinstance(detection, dict) or "bbox_xyxy" not in detection:
                    continue
                bbox = _float_tuple_from_value(detection["bbox_xyxy"], 4)
                x1, y1, x2, y2 = _clamped_bbox_xyxy(
                    bbox,
                    image_width=image_width,
                    image_height=image_height,
                )
                color = _annotation_box_color(index)
                draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
                label = _annotation_detection_label(detection)
                if label:
                    text_bbox = draw.textbbox((x1, y1), label, font=font)
                    text_height = text_bbox[3] - text_bbox[1]
                    label_top = max(0.0, y1 - text_height - 6.0)
                    label_bbox = (x1, label_top, x1 + text_bbox[2] - text_bbox[0] + 8.0, label_top + text_height + 6.0)
                    draw.rectangle(label_bbox, fill=color)
                    draw.text((x1 + 4.0, label_top + 3.0), label, fill=(0, 0, 0), font=font)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        return output_path
    except Exception:
        return None


def _image_tile_rects(
    *,
    image_width: int,
    image_height: int,
    grid_size: int,
    overlap: float,
) -> tuple[tuple[int, int, int, int], ...]:
    if grid_size <= 1:
        return ()
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    rects: list[tuple[int, int, int, int]] = []
    tile_width = image_width / grid_size
    tile_height = image_height / grid_size
    expand_x = tile_width * overlap / 2.0
    expand_y = tile_height * overlap / 2.0
    for row in range(grid_size):
        for col in range(grid_size):
            left = max(0, int(math.floor(col * tile_width - expand_x)))
            right = min(image_width, int(math.ceil((col + 1) * tile_width + expand_x)))
            top = max(0, int(math.floor(row * tile_height - expand_y)))
            bottom = min(image_height, int(math.ceil((row + 1) * tile_height + expand_y)))
            rects.append((left, top, right, bottom))
    return tuple(rects)


def _offset_bbox_xyxy(
    bbox_xyxy: tuple[float, float, float, float],
    *,
    offset_x: float,
    offset_y: float,
) -> list[float]:
    x1, y1, x2, y2 = bbox_xyxy
    return [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y]


def _with_detection_source(detection: Any, *, source: str) -> dict[str, Any]:
    if not isinstance(detection, dict):
        return {}
    enriched = dict(detection)
    enriched.setdefault("source", source)
    return enriched


def _merge_yolo_detections(
    detections: list[dict[str, Any]],
    *,
    image_width: int,
    image_height: int,
    iou_threshold: float,
) -> list[dict[str, Any]]:
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")
    normalized: list[dict[str, Any]] = []
    for detection in detections:
        if not detection or "bbox_xyxy" not in detection:
            continue
        bbox = _clamped_bbox_xyxy(
            _float_tuple_from_value(detection["bbox_xyxy"], 4),
            image_width=image_width,
            image_height=image_height,
        )
        if _bbox_area(bbox) <= 1.0:
            continue
        item = dict(detection)
        item["bbox_xyxy"] = [_round(value) for value in bbox]
        normalized.append(item)

    selected: list[dict[str, Any]] = []
    for detection in sorted(normalized, key=_detection_sort_key, reverse=True):
        if any(_should_suppress_detection(detection, kept, iou_threshold=iou_threshold) for kept in selected):
            continue
        selected.append(detection)
    return selected


def _detection_sort_key(detection: dict[str, Any]) -> tuple[float, float]:
    confidence = float(detection.get("confidence", detection.get("score", detection.get("conf", 0.0))))
    return (confidence, _bbox_area(_float_tuple_from_value(detection["bbox_xyxy"], 4)))


def _should_suppress_detection(candidate: dict[str, Any], kept: dict[str, Any], *, iou_threshold: float) -> bool:
    candidate_bbox = _float_tuple_from_value(candidate["bbox_xyxy"], 4)
    kept_bbox = _float_tuple_from_value(kept["bbox_xyxy"], 4)
    iou = _bbox_iou(candidate_bbox, kept_bbox)
    if iou <= 0.0:
        return False
    if _same_detection_class(candidate, kept) and iou >= iou_threshold:
        return True
    return iou >= 0.90


def _same_detection_class(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if "class_id" in left and "class_id" in right:
        return int(left["class_id"]) == int(right["class_id"])
    left_name = str(left.get("class_name", "")).strip()
    right_name = str(right.get("class_name", "")).strip()
    return bool(left_name and right_name and left_name == right_name)


def _bbox_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    left_x1, left_y1, left_x2, left_y2 = left
    right_x1, right_y1, right_x2, right_y2 = right
    inter_left = max(min(left_x1, left_x2), min(right_x1, right_x2))
    inter_top = max(min(left_y1, left_y2), min(right_y1, right_y2))
    inter_right = min(max(left_x1, left_x2), max(right_x1, right_x2))
    inter_bottom = min(max(left_y1, left_y2), max(right_y1, right_y2))
    inter_area = max(0.0, inter_right - inter_left) * max(0.0, inter_bottom - inter_top)
    if inter_area <= 0.0:
        return 0.0
    union_area = _bbox_area(left) + _bbox_area(right) - inter_area
    return inter_area / union_area if union_area > 0.0 else 0.0


def _clamped_bbox_xyxy(
    bbox_xyxy: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    left = _clamp(min(x1, x2), 0.0, float(max(0, image_width - 1)))
    right = _clamp(max(x1, x2), 0.0, float(max(0, image_width - 1)))
    top = _clamp(min(y1, y2), 0.0, float(max(0, image_height - 1)))
    bottom = _clamp(max(y1, y2), 0.0, float(max(0, image_height - 1)))
    return (left, top, right, bottom)


def _annotation_font(image_font_module: Any) -> Any:
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


def _annotation_box_color(index: int) -> tuple[int, int, int]:
    colors = (
        (255, 59, 48),
        (52, 199, 89),
        (0, 122, 255),
        (255, 204, 0),
        (175, 82, 222),
        (255, 149, 0),
        (90, 200, 250),
        (255, 45, 85),
    )
    return colors[index % len(colors)]


def _annotation_detection_label(detection: dict[str, Any]) -> str:
    confidence = float(detection.get("confidence", detection.get("score", detection.get("conf", 0.0))))
    class_name = str(detection.get("class_name", "")).strip()
    if class_name:
        return f"{class_name} {confidence:.2f}"
    class_id = detection.get("class_id")
    if class_id is not None:
        return f"class {int(class_id)} {confidence:.2f}"
    return f"object {confidence:.2f}"


def _run_id(created_utc: str, scene: SurveySceneInput) -> str:
    timestamp = created_utc.replace("+00:00", "Z").replace("-", "").replace(":", "").replace(".", "")
    seed = f"seed{scene.seed}" if scene.seed is not None else "seedunknown"
    return f"{timestamp}_{seed}"


def _oblique_offset(*, target_x: float, target_y: float, center_x: float, center_y: float, offset_m: float) -> tuple[float, float]:
    direction_x = center_x - target_x
    direction_y = center_y - target_y
    norm = math.hypot(direction_x, direction_y)
    if norm <= 1e-9:
        return (0.0, 0.0)
    return (offset_m * direction_x / norm, offset_m * direction_y / norm)


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


def _orientation_error(np: Any, current_rotation: Any, target_rotation: Any) -> Any:
    return 0.5 * (
        np.cross(current_rotation[:, 0], target_rotation[:, 0])
        + np.cross(current_rotation[:, 1], target_rotation[:, 1])
        + np.cross(current_rotation[:, 2], target_rotation[:, 2])
    )


def _split_cluster_spatial_modes(
    cluster: list[SurveyObservation],
    *,
    split_distance_m: float,
    min_vote: float,
    min_support_views: int,
) -> list[list[SurveyObservation]]:
    if len(cluster) < 4:
        return [cluster]

    seed_pair = _farthest_observation_pair(cluster)
    if seed_pair is None:
        return [cluster]
    left_seed, right_seed, seed_distance = seed_pair
    if seed_distance < split_distance_m:
        return [cluster]

    left_center = _required_observation_position(cluster[left_seed])
    right_center = _required_observation_position(cluster[right_seed])
    groups: list[list[SurveyObservation]] = [[], []]
    for _ in range(6):
        groups = [[], []]
        for observation in cluster:
            position = _required_observation_position(observation)
            if _xy_distance(position, left_center) <= _xy_distance(position, right_center):
                groups[0].append(observation)
            else:
                groups[1].append(observation)
        if not groups[0] or not groups[1]:
            return [cluster]
        next_left_center = _cluster_center(groups[0])
        next_right_center = _cluster_center(groups[1])
        if (
            _xy_distance(left_center, next_left_center) <= 1e-9
            and _xy_distance(right_center, next_right_center) <= 1e-9
        ):
            break
        left_center = next_left_center
        right_center = next_right_center

    if _xy_distance(_cluster_center(groups[0]), _cluster_center(groups[1])) < split_distance_m:
        return [cluster]
    if _cluster_dominant_vote(groups[0])[1] < min_vote or _cluster_dominant_vote(groups[1])[1] < min_vote:
        return [cluster]
    if (
        len(_cluster_supporting_views(groups[0])) < min_support_views
        or len(_cluster_supporting_views(groups[1])) < min_support_views
    ):
        return [cluster]
    return sorted(groups, key=_cluster_sort_key)


def _merge_weak_neighbor_clusters(
    clusters: list[list[SurveyObservation]],
    *,
    merge_radius_m: float,
    weak_vote_threshold: float,
) -> list[list[SurveyObservation]]:
    merged = [list(cluster) for cluster in clusters]
    changed = True
    while changed:
        changed = False
        for index, cluster in enumerate(list(merged)):
            _, dominant_vote = _cluster_dominant_vote(cluster)
            if dominant_vote >= weak_vote_threshold:
                continue
            best_index: int | None = None
            best_distance = float("inf")
            center = _cluster_center(cluster)
            votes = _cluster_class_votes(cluster)
            for other_index, other in enumerate(merged):
                if other_index == index:
                    continue
                other_center = _cluster_center(other)
                distance = _xy_distance(center, other_center)
                if distance > merge_radius_m or distance >= best_distance:
                    continue
                other_votes = _cluster_class_votes(other)
                if any(other_votes.get(class_name, 0.0) > vote for class_name, vote in votes.items()):
                    best_index = other_index
                    best_distance = distance
            if best_index is not None:
                merged[best_index].extend(cluster)
                del merged[index]
                changed = True
                break
    return merged


def _merge_duplicate_large_same_class_clusters(
    clusters: list[list[SurveyObservation]],
    *,
    merge_radius_m: float,
    large_bbox_area_px: float,
) -> list[list[SurveyObservation]]:
    if merge_radius_m <= 0.0 or large_bbox_area_px <= 0.0:
        return clusters
    merged = [list(cluster) for cluster in clusters]
    changed = True
    while changed:
        changed = False
        for left_index in range(len(merged)):
            left_class, _ = _cluster_dominant_vote(merged[left_index])
            if not left_class or _cluster_max_bbox_area(merged[left_index]) < large_bbox_area_px:
                continue
            left_center = _cluster_center(merged[left_index])
            for right_index in range(left_index + 1, len(merged)):
                right_class, _ = _cluster_dominant_vote(merged[right_index])
                if left_class != right_class:
                    continue
                if _xy_distance(left_center, _cluster_center(merged[right_index])) > merge_radius_m:
                    continue
                merged[left_index].extend(merged[right_index])
                del merged[right_index]
                changed = True
                break
            if changed:
                break
    return merged


def _split_mixed_class_clusters(
    clusters: list[list[SurveyObservation]],
    *,
    min_vote: float,
    min_support_views: int,
    split_distance_m: float,
) -> list[list[SurveyObservation]]:
    if min_vote <= 0.0 or min_support_views <= 0 or split_distance_m <= 0.0:
        return clusters
    split_clusters: list[list[SurveyObservation]] = []
    for cluster in clusters:
        class_groups = _cluster_observations_by_class(cluster)
        if len(class_groups) < 2:
            split_clusters.append(cluster)
            continue

        dominant_class, dominant_vote = _cluster_dominant_vote(cluster)
        if not dominant_class or dominant_vote <= 0.0 or dominant_class not in class_groups:
            split_clusters.append(cluster)
            continue
        dominant_center = _cluster_center(class_groups[dominant_class])
        split_class_names: set[str] = set()
        for class_name, observations in class_groups.items():
            if class_name == dominant_class:
                continue
            if _cluster_dominant_vote(observations)[1] < min_vote:
                continue
            if len(_cluster_supporting_views(observations)) < min_support_views:
                continue
            if _xy_distance(_cluster_center(observations), dominant_center) < split_distance_m:
                continue
            split_class_names.add(class_name)

        if not split_class_names:
            split_clusters.append(cluster)
            continue

        remainder = [observation for observation in cluster if observation.class_name not in split_class_names]
        if remainder:
            split_clusters.append(remainder)
        for class_name in sorted(split_class_names):
            split_clusters.append(class_groups[class_name])
    return split_clusters


def _select_weak_multiview_fallback_clusters(
    rejected_clusters: list[list[SurveyObservation]],
    *,
    accepted_clusters: list[list[SurveyObservation]],
    min_support_views: int,
    min_vote: float,
    min_distance_m: float,
    max_bbox_area_px: float,
    elongated_aspect_ratio: float,
    elongated_max_bbox_area_px: float,
) -> list[list[SurveyObservation]]:
    accepted_classes = {
        class_name
        for class_name, _ in (_cluster_dominant_vote(cluster) for cluster in accepted_clusters)
        if class_name
    }
    best_by_class: dict[str, tuple[float, list[SurveyObservation]]] = {}
    for cluster in rejected_clusters:
        class_name, vote = _cluster_dominant_vote(cluster)
        if not class_name or class_name in accepted_classes:
            continue
        if vote < min_vote or len(_cluster_supporting_views(cluster)) < min_support_views:
            continue
        if not _weak_multiview_fallback_passes_bbox_gate(
            cluster,
            max_bbox_area_px=max_bbox_area_px,
            elongated_aspect_ratio=elongated_aspect_ratio,
            elongated_max_bbox_area_px=elongated_max_bbox_area_px,
        ):
            continue
        distance = _distance_to_nearest_cluster(cluster, accepted_clusters)
        if distance < min_distance_m:
            continue
        score = vote * distance
        previous = best_by_class.get(class_name)
        if previous is None or score > previous[0]:
            best_by_class[class_name] = (score, cluster)
    return [cluster for _, cluster in sorted(best_by_class.values(), key=lambda item: _cluster_sort_key(item[1]))]


def _weak_multiview_fallback_passes_bbox_gate(
    cluster: list[SurveyObservation],
    *,
    max_bbox_area_px: float,
    elongated_aspect_ratio: float,
    elongated_max_bbox_area_px: float,
) -> bool:
    max_area = _cluster_max_bbox_area(cluster)
    if max_bbox_area_px <= 0.0 or max_area <= max_bbox_area_px:
        return True
    if elongated_max_bbox_area_px <= 0.0 or max_area > elongated_max_bbox_area_px:
        return False
    return _cluster_max_bbox_aspect_ratio(cluster) >= elongated_aspect_ratio


def _keep_candidate_cluster(
    cluster: list[SurveyObservation],
    *,
    min_support_views: int,
    weak_vote_threshold: float,
    tiny_bbox_area_px: float,
    weak_tiny_max_spread_m: float,
) -> bool:
    if len(_cluster_supporting_views(cluster)) < min_support_views:
        return False
    _, dominant_vote = _cluster_dominant_vote(cluster)
    max_bbox_area = _cluster_max_bbox_area(cluster)
    if dominant_vote < weak_vote_threshold and max_bbox_area < tiny_bbox_area_px:
        return False
    if (
        len(_cluster_supporting_views(cluster)) <= 2
        and max_bbox_area < tiny_bbox_area_px
        and _cluster_xy_spread(cluster) > weak_tiny_max_spread_m
    ):
        return False
    return True


def _keep_single_view_fallback_cluster(
    cluster: list[SurveyObservation],
    *,
    accepted_clusters: list[list[SurveyObservation]],
    confidence_threshold: float,
    min_distance_m: float,
) -> bool:
    if len(cluster) != 1 or len(_cluster_supporting_views(cluster)) != 1:
        return False
    observation = cluster[0]
    if observation.rough_position_world is None or observation.confidence < confidence_threshold:
        return False
    if any(_xy_distance(observation.rough_position_world, _cluster_center(other)) <= min_distance_m for other in accepted_clusters):
        return False
    return True


def _candidate_payload(
    index: int,
    cluster: list[SurveyObservation],
    *,
    single_view_fallback: bool,
    note_override: str | None = None,
) -> dict[str, Any]:
    center = _cluster_center(cluster)
    class_votes = _cluster_class_votes(cluster)
    best = max(cluster, key=lambda obs: (_bbox_area(obs.bbox_xyxy) * max(obs.confidence, 0.0), obs.confidence))
    average_confidence = sum(obs.confidence for obs in cluster) / len(cluster)
    fused_confidence = min(1.0, average_confidence + 0.12 * max(0, len({obs.view_id for obs in cluster}) - 1))
    notes = [
        "Survey candidate from multi-view, class-aware detection clustering.",
        "Class votes are advisory only; final class and T_world_object are deferred to rough/depth pose estimation.",
    ]
    if single_view_fallback:
        notes[0] = "Survey candidate from isolated high-confidence single-view fallback."
    if note_override:
        notes[0] = note_override
    return {
        "candidate_id": f"candidate_{index + 1:03d}",
        "rough_position_world": [_round(value) for value in center],
        "support_count": len(cluster),
        "supporting_views": sorted({obs.view_id for obs in cluster}),
        "confidence": _round(fused_confidence),
        "class_votes": {key: _round(value) for key, value in sorted(class_votes.items())},
        "best_view": best.view_id,
        "best_image_path": best.image_path,
        "best_bbox_xyxy": [_round(value) for value in best.bbox_xyxy],
        "notes": notes,
    }


def _farthest_observation_pair(cluster: list[SurveyObservation]) -> tuple[int, int, float] | None:
    best_pair: tuple[int, int, float] | None = None
    for left_index, left in enumerate(cluster):
        left_position = left.rough_position_world
        if left_position is None:
            continue
        for right_index in range(left_index + 1, len(cluster)):
            right_position = cluster[right_index].rough_position_world
            if right_position is None:
                continue
            distance = _xy_distance(left_position, right_position)
            if best_pair is None or distance > best_pair[2]:
                best_pair = (left_index, right_index, distance)
    return best_pair


def _cluster_class_votes(cluster: list[SurveyObservation]) -> dict[str, float]:
    class_votes: dict[str, float] = {}
    for observation in cluster:
        if observation.class_name:
            class_votes[observation.class_name] = class_votes.get(observation.class_name, 0.0) + observation.confidence
    return class_votes


def _cluster_observations_by_class(cluster: list[SurveyObservation]) -> dict[str, list[SurveyObservation]]:
    groups: dict[str, list[SurveyObservation]] = {}
    for observation in cluster:
        if observation.class_name:
            groups.setdefault(observation.class_name, []).append(observation)
    return groups


def _cluster_dominant_vote(cluster: list[SurveyObservation]) -> tuple[str | None, float]:
    class_votes = _cluster_class_votes(cluster)
    if not class_votes:
        return (None, 0.0)
    class_name, vote = max(class_votes.items(), key=lambda item: item[1])
    return (class_name, vote)


def _cluster_supporting_views(cluster: list[SurveyObservation]) -> tuple[str, ...]:
    return tuple(sorted({observation.view_id for observation in cluster}))


def _cluster_xy_spread(cluster: list[SurveyObservation]) -> float:
    points = [observation.rough_position_world for observation in cluster if observation.rough_position_world is not None]
    if len(points) < 2:
        return 0.0
    return max(_xy_distance(left, right) for index, left in enumerate(points) for right in points[index + 1 :])


def _cluster_max_bbox_area(cluster: list[SurveyObservation]) -> float:
    return max((_bbox_area(obs.bbox_xyxy) for obs in cluster), default=0.0)


def _cluster_max_bbox_aspect_ratio(cluster: list[SurveyObservation]) -> float:
    ratios: list[float] = []
    for observation in cluster:
        x1, y1, x2, y2 = observation.bbox_xyxy
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        if width <= 1e-9 or height <= 1e-9:
            continue
        ratios.append(max(width / height, height / width))
    return max(ratios, default=1.0)


def _distance_to_nearest_cluster(cluster: list[SurveyObservation], other_clusters: list[list[SurveyObservation]]) -> float:
    if not other_clusters:
        return float("inf")
    center = _cluster_center(cluster)
    return min(_xy_distance(center, _cluster_center(other)) for other in other_clusters)


def _cluster_sort_key(cluster: list[SurveyObservation]) -> tuple[str, float, float]:
    center = _cluster_center(cluster)
    return (_cluster_supporting_views(cluster)[0], center[0], center[1])


def _required_observation_position(observation: SurveyObservation) -> tuple[float, float, float]:
    if observation.rough_position_world is None:
        raise ValueError("observation must have a rough world position")
    return observation.rough_position_world


def _cluster_center(cluster: list[SurveyObservation]) -> tuple[float, float, float]:
    points = [obs.rough_position_world for obs in cluster if obs.rough_position_world is not None]
    if not points:
        raise ValueError("cannot compute a cluster center without points")
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
        sum(point[2] for point in points) / len(points),
    )


def _xy_distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _bbox_area(bbox_xyxy: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox_xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _yaw_quat_wxyz(yaw_rad: float) -> tuple[float, float, float, float]:
    return (_round(math.cos(yaw_rad / 2.0)), 0.0, 0.0, _round(math.sin(yaw_rad / 2.0)))


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


def _required(payload: dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError(f"missing required field: {key}")
    return payload[key]


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = _required(payload, key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"field {key} must be a non-empty string")
    return value


def _float_tuple_from_value(value: Any, length: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"expected a list of {length} numbers")
    return tuple(float(item) for item in value)


def _transform_matrix(value: Any) -> tuple[tuple[float, float, float, float], ...]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("expected a 4x4 transform matrix")
    rows = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError("expected a 4x4 transform matrix")
        rows.append(tuple(float(item) for item in row))
    return tuple(rows)


def _round_vector(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    return (_round(vector[0]), _round(vector[1]), _round(vector[2]))


def _round(value: float) -> float:
    return round(float(value), 6)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


def _import_numpy() -> Any:
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError("numpy is required for task1 MuJoCo survey rendering and depth projection.") from exc
    return np
