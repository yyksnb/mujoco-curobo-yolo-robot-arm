from __future__ import annotations

import argparse
import json
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
    find_latest_stage0_layout,
    run_task1_survey,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task1 recognition pipeline entrypoint. Currently implements survey.")
    parser.add_argument(
        "--step",
        choices=("survey",),
        default="survey",
        help="Recognition stage to run. Later row/final/zoom stages should be added here.",
    )
    parser.add_argument(
        "--layout",
        type=Path,
        default=None,
        help="Target object pose layout JSON. Defaults to the latest outputs/object_poses/*_target_object_poses.json.",
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
    parser.add_argument("--yolo-conf", type=float, default=0.15, help="Survey YOLO confidence threshold.")
    parser.add_argument("--yolo-iou", type=float, default=None, help="Optional YOLO IoU threshold.")
    parser.add_argument("--yolo-imgsz", type=int, default=None, help="Optional YOLO inference image size.")
    parser.add_argument("--yolo-device", default=None, help="Optional YOLO device, e.g. cpu or 0.")
    parser.add_argument("--yolo-max-det", type=int, default=30, help="Maximum detections per survey image.")
    parser.add_argument(
        "--yolo-tile-grid",
        type=int,
        default=DEFAULT_YOLO_TILE_GRID_SIZE,
        help="Tile grid size for survey YOLO. Use 1 to disable tile-based detection.",
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
    parser.add_argument("--strict-yolo", action="store_true", help="Fail the survey if YOLO inference fails.")
    parser.add_argument("--plan-only", action="store_true", help="Only write the survey plan; do not load MuJoCo or YOLO.")
    parser.add_argument("--max-ik-iterations", type=int, default=200, help="Maximum IK iterations per survey view.")
    parser.add_argument("--ik-position-tolerance", type=float, default=0.05, help="IK camera position tolerance in meters.")
    parser.add_argument(
        "--ik-orientation-tolerance",
        type=float,
        default=0.35,
        help="IK camera orientation tolerance in radians.",
    )
    args = parser.parse_args()

    if args.step != "survey":
        raise SystemExit(f"unsupported task1 recognition step: {args.step}")

    try:
        layout_path = args.layout or find_latest_stage0_layout()
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    config = SurveyConfig(
        scene_model_path=args.scene_model,
        yolo_profile_path=args.yolo_config,
        output_dir=args.output_dir,
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
        max_ik_iterations=args.max_ik_iterations,
        ik_position_tolerance_m=args.ik_position_tolerance,
        ik_orientation_tolerance_rad=args.ik_orientation_tolerance,
    )
    detector = None
    if config.run_yolo and not config.plan_only:
        from robot_arm_pipeline.perception.yolo_local_inference import YoloLocalInferenceRunner

        detector = YoloLocalInferenceRunner(
            profile_path=config.yolo_profile_path,
            confidence_threshold=config.yolo_confidence,
            iou_threshold=config.yolo_iou,
            image_size=config.yolo_image_size,
            device=config.yolo_device,
            max_detections=config.yolo_max_detections,
        )
    report = run_task1_survey(layout_path, config, detector=detector)
    print(json.dumps({"status": report["status"], "message": report["message"]}, ensure_ascii=False))
    print(f"report={report['report_path']}")


if __name__ == "__main__":
    main()
