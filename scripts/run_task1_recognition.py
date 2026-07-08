from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from robot_arm_pipeline.task1.survey import (  # noqa: E402
    DEFAULT_CAMERA_NAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SAVE_RAW_YOLO_ANNOTATIONS,
    DEFAULT_SCENE_MODEL,
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
    DEFAULT_ROW_VIEW_CANDIDATE_ANGLE_OFFSETS_RAD,
    DEFAULT_ROW_VIEW_CANDIDATE_CAMERA_Z_OFFSETS_M,
    DEFAULT_ROW_VIEW_CANDIDATE_MAX_ATTEMPTS,
    DEFAULT_ROW_VIEW_CANDIDATE_ROLL_OFFSETS_RAD,
    DEFAULT_ROW_VIEW_CANDIDATE_STANDOFF_MULTIPLIERS,
    DEFAULT_ROW_VIEW_COLLISION_SEARCH,
    DEFAULT_ROW_VIEWS_PER_CANDIDATE,
    DEFAULT_STABLE_OBJECT_MIN_CONFIDENCE,
    DEFAULT_STABLE_OBJECT_MIN_EVIDENCE_SCORE,
    DEFAULT_STABLE_OBJECT_MIN_SUPPORT_COUNT,
    DEFAULT_STABLE_OBJECT_SAME_CLASS_NMS_RADIUS_M,
    DEFAULT_STABLE_OBJECT_WORKSPACE_MARGIN_M,
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
    DEFAULT_FINAL_DESIRED_STABLE_OBJECT_COUNT,
    DEFAULT_FINAL_ENTRY_CLEARANCE_MARGIN_M,
    DEFAULT_FINAL_ENTRY_LATERAL_ORIENTATION_POLICY,
    DEFAULT_FINAL_ENTRY_ORIENTATION_POLICY,
    DEFAULT_FINAL_ENTRY_PATH_POLICY,
    DEFAULT_FINAL_ENTRY_PORTAL_MODES,
    DEFAULT_FINAL_ENTRY_SIDE,
    DEFAULT_FINAL_ENTRY_VALIDATION_SAMPLES,
    DEFAULT_FINAL_VIEW_ANGLE_OFFSETS_DEG,
    DEFAULT_FINAL_VIEW_CAMERA_Z_OFFSETS_M,
    DEFAULT_FINAL_VIEW_ROLL_OFFSETS_DEG,
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
from robot_arm_pipeline.task1.zoom import (  # noqa: E402
    DEFAULT_ZOOM_CANDIDATE_AREA_RATIOS,
    DEFAULT_ZOOM_PADDING_RATIO,
    DEFAULT_ZOOM_RATIO_TOLERANCE,
    DEFAULT_ZOOM_SELECTION_BORDER_MARGIN_PX,
    DEFAULT_ZOOM_TARGET_AREA_RATIO,
    ZoomConfig,
    find_latest_final_report,
    run_task1_zoom,
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


def _parse_text_tuple(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected one or more comma-separated values")
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
        choices=("survey", "row", "final", "zoom"),
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
    parser.add_argument(
        "--final-report",
        type=Path,
        default=None,
        help="Final report JSON for --stage zoom. Defaults to the latest outputs/task1/*/final/final_report.json.",
    )
    parser.add_argument("--scene-model", type=Path, default=DEFAULT_SCENE_MODEL, help="MuJoCo scene MJCF/XML path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Task1 run output root directory.")
    parser.add_argument("--camera-name", default=DEFAULT_CAMERA_NAME, help="Logical camera name. Default is wrist.")
    parser.add_argument("--image-width", type=int, default=1920, help="Rendered RGB/depth width. 1920x1080 matches 1080P.")
    parser.add_argument("--image-height", type=int, default=1080, help="Rendered RGB/depth height. 1920x1080 matches 1080P.")
    parser.add_argument("--tank-opening-z", type=float, default=0.50, help="Tank upper opening world z.")
    parser.add_argument("--opening-clearance", type=float, default=0.035, help="Required clearance below tank upper opening.")
    parser.add_argument(
        "--save-raw-yolo-annotations",
        action="store_true",
        default=DEFAULT_SAVE_RAW_YOLO_ANNOTATIONS,
        help="Also save raw YOLO annotated survey images under survey/raw_annotated.",
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
        "--disable-row-view-collision-search",
        action="store_true",
        default=not DEFAULT_ROW_VIEW_COLLISION_SEARCH,
        help="Debug only: render nominal row views without row-stage IK/collision-checked candidate selection.",
    )
    parser.add_argument(
        "--row-view-candidate-standoff-multipliers",
        type=_parse_float_tuple,
        default=DEFAULT_ROW_VIEW_CANDIDATE_STANDOFF_MULTIPLIERS,
        help="Comma-separated standoff multipliers tried by row-stage collision-safe view selection.",
    )
    parser.add_argument(
        "--row-view-candidate-camera-z-offsets",
        type=_parse_float_tuple,
        default=DEFAULT_ROW_VIEW_CANDIDATE_CAMERA_Z_OFFSETS_M,
        help="Comma-separated camera-z offsets tried by row-stage collision-safe view selection.",
    )
    parser.add_argument(
        "--row-view-candidate-angle-offsets-deg",
        type=_parse_float_tuple,
        default=tuple(value * 180.0 / 3.141592653589793 for value in DEFAULT_ROW_VIEW_CANDIDATE_ANGLE_OFFSETS_RAD),
        help="Comma-separated angle offsets in degrees tried by row-stage collision-safe view selection.",
    )
    parser.add_argument(
        "--row-view-candidate-roll-offsets-deg",
        type=_parse_float_tuple,
        default=tuple(value * 180.0 / 3.141592653589793 for value in DEFAULT_ROW_VIEW_CANDIDATE_ROLL_OFFSETS_RAD),
        help="Comma-separated camera roll offsets in degrees tried by row-stage collision-safe view selection.",
    )
    parser.add_argument(
        "--row-view-candidate-max-attempts",
        type=int,
        default=DEFAULT_ROW_VIEW_CANDIDATE_MAX_ATTEMPTS,
        help="Maximum IK/collision validation attempts per planned row view.",
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
        "--stable-object-min-evidence-score",
        type=float,
        default=DEFAULT_STABLE_OBJECT_MIN_EVIDENCE_SCORE,
        help="Minimum multi-view evidence score that can compensate for slightly low row confidence.",
    )
    parser.add_argument(
        "--stable-object-same-class-nms-radius",
        type=float,
        default=DEFAULT_STABLE_OBJECT_SAME_CLASS_NMS_RADIUS_M,
        help="World XY same-class suppression radius for stable row objects.",
    )
    parser.add_argument(
        "--stable-object-workspace-margin",
        type=float,
        default=DEFAULT_STABLE_OBJECT_WORKSPACE_MARGIN_M,
        help="XY tolerance for accepting close RGB-D row object estimates near tank workspace bounds.",
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
        "--final-entry-portal-modes",
        type=_parse_text_tuple,
        default=DEFAULT_FINAL_ENTRY_PORTAL_MODES,
        help="Comma-separated top-opening portal policies tried by final entry waypoint validation.",
    )
    parser.add_argument(
        "--final-entry-orientation-policy",
        choices=("vertical-descent", "target-look-at"),
        default=DEFAULT_FINAL_ENTRY_ORIENTATION_POLICY,
        help="Wrist-camera orientation policy used for final entry validation samples before the close photo pose.",
    )
    parser.add_argument(
        "--final-entry-lateral-orientation-policy",
        choices=("entry-orientation", "final-look-at"),
        default=DEFAULT_FINAL_ENTRY_LATERAL_ORIENTATION_POLICY,
        help="Orientation policy used by lateral in-tank entry samples after the portal descent.",
    )
    parser.add_argument(
        "--final-entry-path-policy",
        choices=("portal-descent-then-lateral", "direct-interpolate"),
        default=DEFAULT_FINAL_ENTRY_PATH_POLICY,
        help="Waypoint geometry used for validating final top-opening entry before the close photo pose.",
    )
    parser.add_argument(
        "--final-desired-stable-object-count",
        type=int,
        default=DEFAULT_FINAL_DESIRED_STABLE_OBJECT_COUNT,
        help="Desired final stable object count; follow-up targets are promoted only while the stable list is below this count.",
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
    parser.add_argument(
        "--final-view-camera-z-offsets",
        type=_parse_float_tuple,
        default=DEFAULT_FINAL_VIEW_CAMERA_Z_OFFSETS_M,
        help="Comma-separated camera-z offsets tried by final-view whole-arm candidate selection.",
    )
    parser.add_argument(
        "--final-view-roll-offsets-deg",
        type=_parse_float_tuple,
        default=DEFAULT_FINAL_VIEW_ROLL_OFFSETS_DEG,
        help="Comma-separated wrist-camera roll offsets in degrees tried by final-view whole-arm candidate selection.",
    )
    parser.add_argument(
        "--zoom-target-area-ratio",
        type=float,
        default=DEFAULT_ZOOM_TARGET_AREA_RATIO,
        help="Desired object bbox area ratio in each zoomed image.",
    )
    parser.add_argument(
        "--zoom-ratio-tolerance",
        type=float,
        default=DEFAULT_ZOOM_RATIO_TOLERANCE,
        help="Allowed absolute error around --zoom-target-area-ratio before a zoom result is marked limited.",
    )
    parser.add_argument(
        "--zoom-candidate-area-ratios",
        type=_parse_float_tuple,
        default=DEFAULT_ZOOM_CANDIDATE_AREA_RATIOS,
        help="Comma-separated target area-ratio candidates generated before selecting one zoom output.",
    )
    parser.add_argument(
        "--zoom-padding-ratio",
        type=float,
        default=DEFAULT_ZOOM_PADDING_RATIO,
        help="Minimum per-side ROI padding fraction around the selected bbox before resize.",
    )
    parser.add_argument(
        "--zoom-selection-border-margin",
        type=float,
        default=DEFAULT_ZOOM_SELECTION_BORDER_MARGIN_PX,
        help="Minimum projected bbox border margin in pixels preferred by zoom candidate selection.",
    )
    parser.add_argument(
        "--zoom-output-width",
        type=int,
        default=None,
        help="Optional zoomed image width. Defaults to the final source image width.",
    )
    parser.add_argument(
        "--zoom-output-height",
        type=int,
        default=None,
        help="Optional zoomed image height. Defaults to the final source image height.",
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
            image_width=args.image_width,
            image_height=args.image_height,
            tank_opening_z_m=args.tank_opening_z,
            opening_clearance_m=args.opening_clearance,
            save_raw_yolo_annotations=args.save_raw_yolo_annotations,
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
            row_view_collision_search=not args.disable_row_view_collision_search,
            row_view_candidate_standoff_multipliers=args.row_view_candidate_standoff_multipliers,
            row_view_candidate_camera_z_offsets_m=args.row_view_candidate_camera_z_offsets,
            row_view_candidate_angle_offsets_rad=tuple(
                value * 3.141592653589793 / 180.0 for value in args.row_view_candidate_angle_offsets_deg
            ),
            row_view_candidate_roll_offsets_rad=tuple(
                value * 3.141592653589793 / 180.0 for value in args.row_view_candidate_roll_offsets_deg
            ),
            row_view_candidate_max_attempts=args.row_view_candidate_max_attempts,
            stable_object_min_confidence=args.stable_object_min_confidence,
            stable_object_min_support_count=args.stable_object_min_support_count,
            stable_object_min_evidence_score=args.stable_object_min_evidence_score,
            stable_object_same_class_nms_radius_m=args.stable_object_same_class_nms_radius,
            stable_object_workspace_margin_m=args.stable_object_workspace_margin,
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
    elif args.stage == "final":
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
            entry_portal_modes=args.final_entry_portal_modes,
            entry_orientation_policy=args.final_entry_orientation_policy,
            entry_lateral_orientation_policy=args.final_entry_lateral_orientation_policy,
            entry_path_policy=args.final_entry_path_policy,
            desired_stable_object_count=args.final_desired_stable_object_count,
            final_view_angle_offsets_deg=args.final_view_angle_offsets_deg,
            final_view_standoff_multipliers=args.final_view_standoff_multipliers,
            final_view_camera_z_offsets_m=args.final_view_camera_z_offsets,
            final_view_roll_offsets_deg=args.final_view_roll_offsets_deg,
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
    else:
        try:
            final_report_path = args.final_report or find_latest_final_report(args.output_dir)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
        config = ZoomConfig(
            output_dir=args.output_dir,
            target_area_ratio=args.zoom_target_area_ratio,
            ratio_tolerance=args.zoom_ratio_tolerance,
            candidate_area_ratios=args.zoom_candidate_area_ratios,
            padding_ratio=args.zoom_padding_ratio,
            selection_border_margin_px=args.zoom_selection_border_margin,
            output_width=args.zoom_output_width,
            output_height=args.zoom_output_height,
            plan_only=args.plan_only,
        )
        report = run_task1_zoom(final_report_path, config)
    print(json.dumps({"status": report["status"], "message": report["message"]}, ensure_ascii=False))
    print(f"report={report['report_path']}")


if __name__ == "__main__":
    main()
