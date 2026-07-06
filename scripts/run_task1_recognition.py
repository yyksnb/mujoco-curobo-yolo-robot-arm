from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from robot_arm_pipeline.task1.survey import (  # noqa: E402
    DEFAULT_CAMERA_NAME,
    DEFAULT_CLUSTER_RADIUS_M,
    DEFAULT_CLUSTER_SPLIT_DISTANCE_M,
    DEFAULT_CLUSTER_SPLIT_MIN_VOTE,
    DEFAULT_CROSS_CLASS_MERGE_RADIUS_M,
    DEFAULT_GRID_SIZE,
    DEFAULT_LARGE_SAME_CLASS_BBOX_AREA_PX,
    DEFAULT_LARGE_SAME_CLASS_MERGE_RADIUS_M,
    DEFAULT_MIN_CANDIDATE_SUPPORT_VIEWS,
    DEFAULT_MIXED_CLASS_SPLIT_DISTANCE_M,
    DEFAULT_MIXED_CLASS_SPLIT_MIN_VOTE,
    DEFAULT_OBLIQUE_OFFSET_M,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SCENE_MODEL,
    DEFAULT_SINGLE_VIEW_FALLBACK_CONFIDENCE,
    DEFAULT_SINGLE_VIEW_FALLBACK_MIN_DISTANCE_M,
    DEFAULT_TINY_CANDIDATE_BBOX_AREA_PX,
    DEFAULT_WEAK_CANDIDATE_MERGE_RADIUS_M,
    DEFAULT_WEAK_CANDIDATE_VOTE_THRESHOLD,
    DEFAULT_WEAK_MULTIVIEW_FALLBACK_MIN_DISTANCE_M,
    DEFAULT_WEAK_MULTIVIEW_FALLBACK_MAX_BBOX_AREA_PX,
    DEFAULT_WEAK_MULTIVIEW_FALLBACK_VOTE,
    DEFAULT_WEAK_MULTIVIEW_FALLBACK_ELONGATED_ASPECT_RATIO,
    DEFAULT_WEAK_MULTIVIEW_FALLBACK_ELONGATED_MAX_BBOX_AREA_PX,
    DEFAULT_WEAK_TINY_CANDIDATE_MAX_SPREAD_M,
    DEFAULT_YOLO_TILE_GRID_SIZE,
    DEFAULT_YOLO_TILE_NMS_IOU,
    DEFAULT_YOLO_TILE_OVERLAP,
    DEFAULT_YOLO_PROFILE,
    SurveyConfig,
    run_task1_survey,
)
from robot_arm_pipeline.scene.target_object_layout import (  # noqa: E402
    DEFAULT_BASE_HEIGHT_M,
    DEFAULT_COLLISION_MARGIN_M,
    DEFAULT_OBJECT_COUNT,
    PlacementBounds,
    make_random_target_object_pose_payload,
)
from robot_arm_pipeline.task1.row import (  # noqa: E402
    DEFAULT_ROW_CAMERA_Z_M,
    DEFAULT_ROW_CLUSTER_RADIUS_M,
    DEFAULT_ROW_ENTRY_SIDE,
    DEFAULT_ROW_LOOK_AT_HEIGHT_OFFSET_M,
    DEFAULT_ROW_MIN_OBLIQUE_DISTANCE_M,
    DEFAULT_ROW_STANDOFF_M,
    DEFAULT_ROW_VIEW_ANGLE_SPREAD_RAD,
    DEFAULT_ROW_VIEWS_PER_CANDIDATE,
    DEFAULT_STABLE_OBJECT_MIN_CONFIDENCE,
    DEFAULT_STABLE_OBJECT_MIN_SUPPORT_COUNT,
    DEFAULT_STABLE_OBJECT_SAME_CLASS_NMS_RADIUS_M,
    DEFAULT_TENTATIVE_OBJECT_MIN_CONFIDENCE,
    DEFAULT_TENTATIVE_OBJECT_MIN_SUPPORT_COUNT,
    DEFAULT_TENTATIVE_OBJECT_SMALL_BBOX_AREA_PX,
    DEFAULT_CROSS_CLASS_CONFLICT_RADIUS_M,
    DEFAULT_CROSS_CLASS_AMBIGUITY_SCORE_RATIO,
    DEFAULT_CLASS_VOTE_AMBIGUITY_TOP_TO_SECOND_RATIO,
    DEFAULT_CLASS_VOTE_AMBIGUITY_MIN_SECONDARY_VOTE,
    RowConfig,
    find_latest_survey_report,
    run_task1_row,
)
from robot_arm_pipeline.task1.final import (  # noqa: E402
    DEFAULT_FINAL_CAMERA_Z_M,
    DEFAULT_FINAL_ENTRY_CLEARANCE_MARGIN_M,
    DEFAULT_FINAL_ENTRY_SIDE,
    DEFAULT_FINAL_ENTRY_VALIDATION_SAMPLES,
    DEFAULT_FINAL_VIEW_ANGLE_OFFSETS_DEG,
    DEFAULT_FINAL_VIEW_STANDOFF_MULTIPLIERS,
    DEFAULT_FINAL_IK_POSITION_TOLERANCE_M,
    DEFAULT_FINAL_LOOK_AT_HEIGHT_OFFSET_M,
    DEFAULT_FINAL_MIN_OBLIQUE_DISTANCE_M,
    DEFAULT_FINAL_STANDOFF_M,
    DEFAULT_FINAL_TARGET_MATCH_RADIUS_M,
    FinalConfig,
    find_latest_row_report,
    run_task1_final,
)


def _build_yolo_detector(config):
    if not config.run_yolo or config.plan_only:
        return None
    from robot_arm_pipeline.perception.yolo_local_inference import YoloLocalInferenceRunner

    return YoloLocalInferenceRunner(
        profile_path=config.yolo_profile_path,
        confidence_threshold=config.yolo_confidence,
        iou_threshold=config.yolo_iou,
        image_size=config.yolo_image_size,
        device=config.yolo_device,
        max_detections=config.yolo_max_detections,
    )


def _value_or_default(value, default):
    return default if value is None else value


def _parse_float_tuple(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected one or more comma-separated numbers")
    return values


def _task1_run_name(created_utc: str, seed: int) -> str:
    timestamp = created_utc.replace("+00:00", "Z").replace("-", "").replace(":", "").replace(".", "")
    return f"{timestamp}_seed{seed}"


def _write_seeded_layout(args: argparse.Namespace, *, created_utc: str) -> tuple[Path, Path]:
    if args.seed is None:
        raise SystemExit("Provide either --layout or --seed for the survey stage.")
    run_dir = args.output_dir / _task1_run_name(created_utc, args.seed)
    layout_path = run_dir / "layout" / "target_object_poses.json"
    if layout_path.exists():
        raise SystemExit(f"Refusing to overwrite existing task1 layout: {layout_path}")

    payload = make_random_target_object_pose_payload(
        model_path=args.scene_model,
        seed=args.seed,
        created_utc=created_utc,
        object_count=args.object_count,
        base_height_m=args.base_height,
        bounds=PlacementBounds(
            x_min=args.x_min,
            x_max=args.x_max,
            y_min=args.y_min,
            y_max=args.y_max,
        ),
        collision_margin_m=args.collision_margin,
        max_attempts_per_object=args.max_attempts_per_object,
    )
    layout_path.parent.mkdir(parents=True, exist_ok=True)
    layout_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return layout_path, run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Task1 recognition pipeline entrypoint.")
    parser.add_argument(
        "--stage",
        choices=("survey", "row", "final"),
        default="survey",
        help="Recognition stage to run.",
    )
    parser.add_argument(
        "--layout",
        type=Path,
        default=None,
        help="Explicit target object pose layout JSON. When omitted, --seed generates one under the task1 run directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed used to generate a task1 layout when --layout is omitted.",
    )
    parser.add_argument(
        "--object-count",
        type=int,
        default=DEFAULT_OBJECT_COUNT,
        help="Number of target objects to place for generated task1 layouts.",
    )
    parser.add_argument(
        "--base-height",
        type=float,
        default=DEFAULT_BASE_HEIGHT_M,
        help="Common z coordinate for generated object body poses, in meters.",
    )
    parser.add_argument("--x-min", type=float, default=PlacementBounds.x_min)
    parser.add_argument("--x-max", type=float, default=PlacementBounds.x_max)
    parser.add_argument("--y-min", type=float, default=PlacementBounds.y_min)
    parser.add_argument("--y-max", type=float, default=PlacementBounds.y_max)
    parser.add_argument(
        "--collision-margin",
        type=float,
        default=DEFAULT_COLLISION_MARGIN_M,
        help="Extra 2D footprint separation margin for generated task1 layouts, in meters.",
    )
    parser.add_argument("--max-attempts-per-object", type=int, default=6000)
    parser.add_argument(
        "--survey-report",
        type=Path,
        default=None,
        help="Survey report JSON for --stage row. Defaults to the latest outputs/task1/*/survey/survey_report.json.",
    )
    parser.add_argument(
        "--row-report",
        type=Path,
        default=None,
        help="Row report JSON for --stage final. Defaults to the latest outputs/task1/*/row/row_report.json.",
    )
    parser.add_argument("--scene-model", type=Path, default=DEFAULT_SCENE_MODEL, help="MuJoCo scene MJCF/XML path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Task1 run output root directory.")
    parser.add_argument("--camera-name", default=DEFAULT_CAMERA_NAME, help="Logical camera name. Default is wrist.")
    parser.add_argument(
        "--grid-size",
        type=int,
        default=DEFAULT_GRID_SIZE,
        help="Survey scan grid size, e.g. 4 means 4x4 views.",
    )
    parser.add_argument("--image-width", type=int, default=1920, help="Rendered RGB/depth width. 1920x1080 matches 1080P.")
    parser.add_argument("--image-height", type=int, default=1080, help="Rendered RGB/depth height. 1920x1080 matches 1080P.")
    parser.add_argument("--camera-z", type=float, default=0.38, help="Desired wrist camera world z inside the tank.")
    parser.add_argument("--tank-opening-z", type=float, default=0.50, help="Tank upper opening world z.")
    parser.add_argument("--opening-clearance", type=float, default=0.035, help="Required clearance below tank upper opening.")
    parser.add_argument(
        "--oblique-offset",
        type=float,
        default=DEFAULT_OBLIQUE_OFFSET_M,
        help="Lateral inward offset for multi-angle survey views.",
    )
    parser.add_argument(
        "--cluster-radius",
        type=float,
        default=DEFAULT_CLUSTER_RADIUS_M,
        help="World XY radius for merging multi-view detections.",
    )
    parser.add_argument(
        "--cross-class-merge-radius",
        type=float,
        default=DEFAULT_CROSS_CLASS_MERGE_RADIUS_M,
        help="World XY radius for merging detections whose advisory YOLO classes differ.",
    )
    parser.add_argument(
        "--cluster-split-distance",
        type=float,
        default=DEFAULT_CLUSTER_SPLIT_DISTANCE_M,
        help="World XY distance for splitting one fused candidate into two spatial modes.",
    )
    parser.add_argument(
        "--cluster-split-min-vote",
        type=float,
        default=DEFAULT_CLUSTER_SPLIT_MIN_VOTE,
        help="Minimum class-vote evidence required for each side of a spatial split.",
    )
    parser.add_argument(
        "--weak-candidate-merge-radius",
        type=float,
        default=DEFAULT_WEAK_CANDIDATE_MERGE_RADIUS_M,
        help="World XY radius for merging weak noisy clusters into nearby stronger candidates.",
    )
    parser.add_argument(
        "--weak-candidate-vote",
        type=float,
        default=DEFAULT_WEAK_CANDIDATE_VOTE_THRESHOLD,
        help="Dominant class-vote threshold below which a candidate is considered weak.",
    )
    parser.add_argument(
        "--tiny-candidate-bbox-area",
        type=float,
        default=DEFAULT_TINY_CANDIDATE_BBOX_AREA_PX,
        help="Pixel area threshold for filtering weak tiny candidates.",
    )
    parser.add_argument(
        "--weak-tiny-candidate-max-spread",
        type=float,
        default=DEFAULT_WEAK_TINY_CANDIDATE_MAX_SPREAD_M,
        help="Maximum world XY spread for keeping a weak tiny candidate.",
    )
    parser.add_argument(
        "--single-view-fallback-conf",
        type=float,
        default=DEFAULT_SINGLE_VIEW_FALLBACK_CONFIDENCE,
        help="Minimum confidence for keeping an isolated single-view fallback candidate.",
    )
    parser.add_argument(
        "--single-view-fallback-min-distance",
        type=float,
        default=DEFAULT_SINGLE_VIEW_FALLBACK_MIN_DISTANCE_M,
        help="Minimum world XY distance from accepted candidates for single-view fallback.",
    )
    parser.add_argument(
        "--min-candidate-support-views",
        type=int,
        default=DEFAULT_MIN_CANDIDATE_SUPPORT_VIEWS,
        help="Minimum number of survey views required for a fused candidate.",
    )
    parser.add_argument(
        "--large-same-class-merge-radius",
        type=float,
        default=DEFAULT_LARGE_SAME_CLASS_MERGE_RADIUS_M,
        help="World XY radius for merging likely duplicate large same-class candidates.",
    )
    parser.add_argument(
        "--large-same-class-bbox-area",
        type=float,
        default=DEFAULT_LARGE_SAME_CLASS_BBOX_AREA_PX,
        help="Pixel area threshold that enables large same-class duplicate merging.",
    )
    parser.add_argument(
        "--mixed-class-split-min-vote",
        type=float,
        default=DEFAULT_MIXED_CLASS_SPLIT_MIN_VOTE,
        help="Minimum secondary class vote required to split a mixed-class fused cluster.",
    )
    parser.add_argument(
        "--mixed-class-split-distance",
        type=float,
        default=DEFAULT_MIXED_CLASS_SPLIT_DISTANCE_M,
        help="Minimum world XY separation for splitting a mixed-class fused cluster.",
    )
    parser.add_argument(
        "--weak-multiview-fallback-vote",
        type=float,
        default=DEFAULT_WEAK_MULTIVIEW_FALLBACK_VOTE,
        help="Minimum class vote for keeping a weak but distinct multi-view fallback candidate.",
    )
    parser.add_argument(
        "--weak-multiview-fallback-min-distance",
        type=float,
        default=DEFAULT_WEAK_MULTIVIEW_FALLBACK_MIN_DISTANCE_M,
        help="Minimum world XY distance from accepted candidates for weak multi-view fallback.",
    )
    parser.add_argument(
        "--weak-multiview-fallback-max-bbox-area",
        type=float,
        default=DEFAULT_WEAK_MULTIVIEW_FALLBACK_MAX_BBOX_AREA_PX,
        help="Maximum bbox pixel area for weak multi-view fallback candidates.",
    )
    parser.add_argument(
        "--weak-multiview-fallback-elongated-aspect",
        type=float,
        default=DEFAULT_WEAK_MULTIVIEW_FALLBACK_ELONGATED_ASPECT_RATIO,
        help="Aspect-ratio gate for class-agnostic elongated weak multi-view fallback candidates.",
    )
    parser.add_argument(
        "--weak-multiview-fallback-elongated-max-bbox-area",
        type=float,
        default=DEFAULT_WEAK_MULTIVIEW_FALLBACK_ELONGATED_MAX_BBOX_AREA_PX,
        help="Maximum bbox pixel area for elongated weak multi-view fallback candidates.",
    )
    parser.add_argument("--yolo-config", type=Path, default=DEFAULT_YOLO_PROFILE, help="YOLO Stage3 profile YAML/JSON.")
    parser.add_argument("--yolo-conf", type=float, default=0.15, help="YOLO confidence threshold for the active stage.")
    parser.add_argument("--yolo-iou", type=float, default=None, help="Optional YOLO IoU threshold.")
    parser.add_argument("--yolo-imgsz", type=int, default=None, help="Optional YOLO inference image size.")
    parser.add_argument("--yolo-device", default=None, help="Optional YOLO device, e.g. cpu or 0.")
    parser.add_argument("--yolo-max-det", type=int, default=30, help="Maximum detections per captured image.")
    parser.add_argument(
        "--yolo-tile-grid",
        type=int,
        default=DEFAULT_YOLO_TILE_GRID_SIZE,
        help="Tile grid size for YOLO. Use 1 to disable tile-based detection.",
    )
    parser.add_argument(
        "--yolo-tile-overlap",
        type=float,
        default=DEFAULT_YOLO_TILE_OVERLAP,
        help="Fractional overlap between adjacent YOLO tiles.",
    )
    parser.add_argument(
        "--yolo-tile-nms-iou",
        type=float,
        default=DEFAULT_YOLO_TILE_NMS_IOU,
        help="IoU threshold for merging full-frame and tile YOLO detections.",
    )
    parser.add_argument("--skip-yolo", action="store_true", help="Capture images/depth but skip YOLO inference.")
    parser.add_argument("--strict-yolo", action="store_true", help="Fail the active stage if YOLO inference fails.")
    parser.add_argument("--plan-only", action="store_true", help="Only write the active stage plan; do not load MuJoCo or YOLO.")
    parser.add_argument(
        "--max-ik-iterations",
        type=int,
        default=None,
        help="Maximum IK iterations per capture view. Defaults are stage-specific.",
    )
    parser.add_argument(
        "--ik-position-tolerance",
        type=float,
        default=None,
        help="IK camera position tolerance in meters. Defaults are stage-specific.",
    )
    parser.add_argument(
        "--ik-orientation-tolerance",
        type=float,
        default=None,
        help="IK camera orientation tolerance in radians. Defaults are stage-specific.",
    )
    parser.add_argument("--row-camera-z", type=float, default=DEFAULT_ROW_CAMERA_Z_M, help="Desired row-stage wrist camera world z inside the tank.")
    parser.add_argument("--row-standoff", type=float, default=DEFAULT_ROW_STANDOFF_M, help="Row-stage lateral standoff from candidate rough position.")
    parser.add_argument(
        "--row-views-per-candidate",
        type=int,
        default=DEFAULT_ROW_VIEWS_PER_CANDIDATE,
        help="Number of close-inspection views to plan for each survey candidate.",
    )
    parser.add_argument(
        "--row-view-angle-spread-deg",
        type=float,
        default=DEFAULT_ROW_VIEW_ANGLE_SPREAD_RAD * 180.0 / 3.141592653589793,
        help="Angular spread between row close-inspection views.",
    )
    parser.add_argument(
        "--row-min-oblique-distance",
        type=float,
        default=DEFAULT_ROW_MIN_OBLIQUE_DISTANCE_M,
        help="Minimum row-stage horizontal camera offset from the candidate, preventing pure top-down shots.",
    )
    parser.add_argument(
        "--row-look-at-height-offset",
        type=float,
        default=DEFAULT_ROW_LOOK_AT_HEIGHT_OFFSET_M,
        help="Height above the tank bottom/object base plane used as the row view look-at target.",
    )
    parser.add_argument(
        "--row-entry-side",
        choices=("y-max", "y-min", "x-min", "x-max", "center", "survey-best"),
        default=DEFAULT_ROW_ENTRY_SIDE,
        help="Preferred tank-side reference for row camera placement. survey-best uses the old survey-image direction.",
    )
    parser.add_argument(
        "--row-cluster-radius",
        type=float,
        default=DEFAULT_ROW_CLUSTER_RADIUS_M,
        help="World XY radius for fusing close RGB-D row observations into object hypotheses.",
    )
    parser.add_argument(
        "--stable-object-min-confidence",
        type=float,
        default=DEFAULT_STABLE_OBJECT_MIN_CONFIDENCE,
        help="Minimum fused confidence for a row hypothesis to enter stable_objects.",
    )
    parser.add_argument(
        "--stable-object-min-support-count",
        type=int,
        default=DEFAULT_STABLE_OBJECT_MIN_SUPPORT_COUNT,
        help="Minimum close-view observation support count for a row hypothesis to enter stable_objects.",
    )
    parser.add_argument(
        "--stable-object-same-class-nms-radius",
        type=float,
        default=DEFAULT_STABLE_OBJECT_SAME_CLASS_NMS_RADIUS_M,
        help="World XY same-class suppression radius for stable row objects.",
    )
    parser.add_argument(
        "--tentative-object-min-confidence",
        type=float,
        default=DEFAULT_TENTATIVE_OBJECT_MIN_CONFIDENCE,
        help="Minimum fused confidence for a low-evidence small row hypothesis to enter tentative_objects.",
    )
    parser.add_argument(
        "--tentative-object-min-support-count",
        type=int,
        default=DEFAULT_TENTATIVE_OBJECT_MIN_SUPPORT_COUNT,
        help="Minimum close-view support count for a low-evidence small row hypothesis to enter tentative_objects.",
    )
    parser.add_argument(
        "--tentative-object-small-bbox-area",
        type=float,
        default=DEFAULT_TENTATIVE_OBJECT_SMALL_BBOX_AREA_PX,
        help="Maximum best bbox pixel area for class-agnostic small-object tentative recovery.",
    )
    parser.add_argument(
        "--cross-class-conflict-radius",
        type=float,
        default=DEFAULT_CROSS_CLASS_CONFLICT_RADIUS_M,
        help="World XY radius for treating nearby different-class row hypotheses as a conflict.",
    )
    parser.add_argument(
        "--cross-class-ambiguity-score-ratio",
        type=float,
        default=DEFAULT_CROSS_CLASS_AMBIGUITY_SCORE_RATIO,
        help="Minimum weaker/stronger score ratio that reports a cross-class conflict as ambiguous.",
    )
    parser.add_argument(
        "--class-vote-ambiguity-top-to-second-ratio",
        type=float,
        default=DEFAULT_CLASS_VOTE_AMBIGUITY_TOP_TO_SECOND_RATIO,
        help="Maximum top/second class-vote ratio that reports one row hypothesis as ambiguous.",
    )
    parser.add_argument(
        "--class-vote-ambiguity-min-secondary-vote",
        type=float,
        default=DEFAULT_CLASS_VOTE_AMBIGUITY_MIN_SECONDARY_VOTE,
        help="Minimum secondary class-vote evidence before top-two class votes can mark a row hypothesis ambiguous.",
    )
    parser.add_argument(
        "--final-camera-z",
        type=float,
        default=DEFAULT_FINAL_CAMERA_Z_M,
        help="Desired final wrist camera world z inside the tank.",
    )
    parser.add_argument(
        "--final-standoff",
        type=float,
        default=DEFAULT_FINAL_STANDOFF_M,
        help="Horizontal standoff from each row target for final capture.",
    )
    parser.add_argument(
        "--final-min-oblique-distance",
        type=float,
        default=DEFAULT_FINAL_MIN_OBLIQUE_DISTANCE_M,
        help="Minimum horizontal camera offset from each target, preventing pure top-down final photos.",
    )
    parser.add_argument(
        "--final-look-at-height-offset",
        type=float,
        default=DEFAULT_FINAL_LOOK_AT_HEIGHT_OFFSET_M,
        help="Height above the object base plane used as the final look-at target.",
    )
    parser.add_argument(
        "--final-entry-side",
        choices=("y-max", "y-min", "x-min", "x-max", "center"),
        default=DEFAULT_FINAL_ENTRY_SIDE,
        help="Preferred tank-side reference when row report evidence does not provide a previous view direction.",
    )
    parser.add_argument(
        "--final-target-match-radius",
        type=float,
        default=DEFAULT_FINAL_TARGET_MATCH_RADIUS_M,
        help="World XY radius for associating close YOLO-depth observations with one planned target.",
    )
    parser.add_argument(
        "--final-entry-validation-samples",
        type=int,
        default=DEFAULT_FINAL_ENTRY_VALIDATION_SAMPLES,
        help="Number of top-opening entry pose samples to validate before each final photo.",
    )
    parser.add_argument(
        "--final-entry-clearance-margin",
        type=float,
        default=DEFAULT_FINAL_ENTRY_CLEARANCE_MARGIN_M,
        help="Additional vertical margin below the tank opening for final entry validation samples.",
    )
    parser.add_argument(
        "--final-view-angle-offsets-deg",
        type=_parse_float_tuple,
        default=DEFAULT_FINAL_VIEW_ANGLE_OFFSETS_DEG,
        help="Comma-separated final-view angle offsets around the row-evidence direction.",
    )
    parser.add_argument(
        "--final-view-standoff-multipliers",
        type=_parse_float_tuple,
        default=DEFAULT_FINAL_VIEW_STANDOFF_MULTIPLIERS,
        help="Comma-separated standoff multipliers for final-view candidates, tried in policy order.",
    )
    args = parser.parse_args()

    if args.stage == "survey":
        created_utc = datetime.now(timezone.utc).isoformat()
        if args.layout is None:
            layout_path, run_dir = _write_seeded_layout(args, created_utc=created_utc)
        else:
            layout_path = args.layout
            run_dir = None
        config = SurveyConfig(
            scene_model_path=args.scene_model,
            yolo_profile_path=args.yolo_config,
            output_dir=args.output_dir,
            run_dir=run_dir,
            created_utc=created_utc if run_dir is not None else None,
            camera_name=args.camera_name,
            grid_size=args.grid_size,
            image_width=args.image_width,
            image_height=args.image_height,
            camera_z_m=args.camera_z,
            tank_opening_z_m=args.tank_opening_z,
            opening_clearance_m=args.opening_clearance,
            oblique_offset_m=args.oblique_offset,
            cluster_radius_m=args.cluster_radius,
            cross_class_merge_radius_m=args.cross_class_merge_radius,
            cluster_split_distance_m=args.cluster_split_distance,
            cluster_split_min_vote=args.cluster_split_min_vote,
            weak_candidate_merge_radius_m=args.weak_candidate_merge_radius,
            weak_candidate_vote_threshold=args.weak_candidate_vote,
            tiny_candidate_bbox_area_px=args.tiny_candidate_bbox_area,
            weak_tiny_candidate_max_spread_m=args.weak_tiny_candidate_max_spread,
            single_view_fallback_confidence=args.single_view_fallback_conf,
            single_view_fallback_min_distance_m=args.single_view_fallback_min_distance,
            min_candidate_support_views=args.min_candidate_support_views,
            large_same_class_merge_radius_m=args.large_same_class_merge_radius,
            large_same_class_bbox_area_px=args.large_same_class_bbox_area,
            mixed_class_split_min_vote=args.mixed_class_split_min_vote,
            mixed_class_split_distance_m=args.mixed_class_split_distance,
            weak_multiview_fallback_vote=args.weak_multiview_fallback_vote,
            weak_multiview_fallback_min_distance_m=args.weak_multiview_fallback_min_distance,
            weak_multiview_fallback_max_bbox_area_px=args.weak_multiview_fallback_max_bbox_area,
            weak_multiview_fallback_elongated_aspect_ratio=args.weak_multiview_fallback_elongated_aspect,
            weak_multiview_fallback_elongated_max_bbox_area_px=args.weak_multiview_fallback_elongated_max_bbox_area,
            yolo_confidence=args.yolo_conf,
            yolo_iou=args.yolo_iou,
            yolo_image_size=args.yolo_imgsz,
            yolo_device=args.yolo_device,
            yolo_max_detections=args.yolo_max_det,
            yolo_tile_grid_size=args.yolo_tile_grid,
            yolo_tile_overlap=args.yolo_tile_overlap,
            yolo_tile_nms_iou=args.yolo_tile_nms_iou,
            run_yolo=not args.skip_yolo,
            plan_only=args.plan_only,
            strict_yolo=args.strict_yolo,
            max_ik_iterations=_value_or_default(args.max_ik_iterations, SurveyConfig().max_ik_iterations),
            ik_position_tolerance_m=_value_or_default(
                args.ik_position_tolerance,
                SurveyConfig().ik_position_tolerance_m,
            ),
            ik_orientation_tolerance_rad=_value_or_default(
                args.ik_orientation_tolerance,
                SurveyConfig().ik_orientation_tolerance_rad,
            ),
        )
        detector = _build_yolo_detector(config)
        report = run_task1_survey(layout_path, config, detector=detector)
    elif args.stage == "row":
        try:
            survey_report_path = args.survey_report or find_latest_survey_report(args.output_dir)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
        config = RowConfig(
            scene_model_path=args.scene_model,
            yolo_profile_path=args.yolo_config,
            output_dir=args.output_dir,
            camera_name=args.camera_name,
            image_width=args.image_width,
            image_height=args.image_height,
            camera_z_m=args.row_camera_z,
            tank_opening_z_m=args.tank_opening_z,
            opening_clearance_m=args.opening_clearance,
            row_standoff_m=args.row_standoff,
            row_views_per_candidate=args.row_views_per_candidate,
            row_view_angle_spread_rad=args.row_view_angle_spread_deg * 3.141592653589793 / 180.0,
            row_min_oblique_distance_m=args.row_min_oblique_distance,
            look_at_height_offset_m=args.row_look_at_height_offset,
            entry_side=args.row_entry_side,
            row_cluster_radius_m=args.row_cluster_radius,
            stable_object_min_confidence=args.stable_object_min_confidence,
            stable_object_min_support_count=args.stable_object_min_support_count,
            stable_object_same_class_nms_radius_m=args.stable_object_same_class_nms_radius,
            tentative_object_min_confidence=args.tentative_object_min_confidence,
            tentative_object_min_support_count=args.tentative_object_min_support_count,
            tentative_object_small_bbox_area_px=args.tentative_object_small_bbox_area,
            cross_class_conflict_radius_m=args.cross_class_conflict_radius,
            cross_class_ambiguity_score_ratio=args.cross_class_ambiguity_score_ratio,
            class_vote_ambiguity_top_to_second_ratio=args.class_vote_ambiguity_top_to_second_ratio,
            class_vote_ambiguity_min_secondary_vote=args.class_vote_ambiguity_min_secondary_vote,
            yolo_confidence=args.yolo_conf,
            yolo_iou=args.yolo_iou,
            yolo_image_size=args.yolo_imgsz,
            yolo_device=args.yolo_device,
            yolo_max_detections=args.yolo_max_det,
            yolo_tile_grid_size=args.yolo_tile_grid,
            yolo_tile_overlap=args.yolo_tile_overlap,
            yolo_tile_nms_iou=args.yolo_tile_nms_iou,
            run_yolo=not args.skip_yolo,
            plan_only=args.plan_only,
            strict_yolo=args.strict_yolo,
            max_ik_iterations=_value_or_default(args.max_ik_iterations, RowConfig().max_ik_iterations),
            ik_position_tolerance_m=_value_or_default(
                args.ik_position_tolerance,
                RowConfig().ik_position_tolerance_m,
            ),
            ik_orientation_tolerance_rad=_value_or_default(
                args.ik_orientation_tolerance,
                RowConfig().ik_orientation_tolerance_rad,
            ),
        )
        detector = _build_yolo_detector(config)
        report = run_task1_row(survey_report_path, config, detector=detector)
    else:
        try:
            row_report_path = args.row_report or find_latest_row_report(args.output_dir)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
        config = FinalConfig(
            scene_model_path=args.scene_model,
            yolo_profile_path=args.yolo_config,
            output_dir=args.output_dir,
            camera_name=args.camera_name,
            image_width=args.image_width,
            image_height=args.image_height,
            camera_z_m=args.final_camera_z,
            tank_opening_z_m=args.tank_opening_z,
            opening_clearance_m=args.opening_clearance,
            standoff_m=args.final_standoff,
            min_oblique_distance_m=args.final_min_oblique_distance,
            look_at_height_offset_m=args.final_look_at_height_offset,
            entry_side=args.final_entry_side,
            target_match_radius_m=args.final_target_match_radius,
            entry_validation_samples=args.final_entry_validation_samples,
            entry_clearance_margin_m=args.final_entry_clearance_margin,
            final_view_angle_offsets_deg=args.final_view_angle_offsets_deg,
            final_view_standoff_multipliers=args.final_view_standoff_multipliers,
            yolo_confidence=args.yolo_conf,
            yolo_iou=args.yolo_iou,
            yolo_image_size=args.yolo_imgsz,
            yolo_device=args.yolo_device,
            yolo_max_detections=args.yolo_max_det,
            yolo_tile_grid_size=args.yolo_tile_grid,
            yolo_tile_overlap=args.yolo_tile_overlap,
            yolo_tile_nms_iou=args.yolo_tile_nms_iou,
            run_yolo=not args.skip_yolo,
            plan_only=args.plan_only,
            strict_yolo=args.strict_yolo,
            max_ik_iterations=_value_or_default(
                args.max_ik_iterations,
                FinalConfig().max_ik_iterations,
            ),
            ik_position_tolerance_m=_value_or_default(
                args.ik_position_tolerance,
                DEFAULT_FINAL_IK_POSITION_TOLERANCE_M,
            ),
            ik_orientation_tolerance_rad=_value_or_default(
                args.ik_orientation_tolerance,
                FinalConfig().ik_orientation_tolerance_rad,
            ),
        )
        detector = _build_yolo_detector(config)
        report = run_task1_final(row_report_path, config, detector=detector)
    print(json.dumps({"status": report["status"], "message": report["message"]}, ensure_ascii=False))
    print(f"report={report['report_path']}")


if __name__ == "__main__":
    main()
