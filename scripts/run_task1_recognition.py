from __future__ import annotations

import argparse
import json
import sys
from time import perf_counter
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
    load_stage0_layout,
    run_task1_survey,
)
from robot_arm_pipeline.scene.target_object_layout import (  # noqa: E402
    DEFAULT_BASE_HEIGHT_M,
    DEFAULT_COLLISION_MARGIN_M,
    DEFAULT_OBJECT_COUNT,
    PlacementBounds,
    make_random_target_object_pose_payload,
)
from robot_arm_pipeline.task1.rough import (  # noqa: E402
    RoughConfig,
    find_latest_survey_report,
    run_task1_rough,
)
from robot_arm_pipeline.task1.final import (  # noqa: E402
    FinalConfig,
    find_latest_rough_report,
    run_task1_final,
)
from robot_arm_pipeline.task1.zoom import (  # noqa: E402
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


def _log_stage_event(event: str, stage: str, **fields) -> None:
    parts = [f"[task1] {event}", f"stage={stage}"]
    for key, value in fields.items():
        if value is not None:
            parts.append(f"{key}={value}")
    print(" ".join(parts), file=sys.stderr, flush=True)


def _run_timed_stage(stage: str, fn):
    _log_stage_event("start", stage)
    started = perf_counter()
    try:
        report = fn()
    except BaseException:
        duration_s = perf_counter() - started
        _log_stage_event("failed", stage, duration_s=f"{duration_s:.3f}")
        raise
    duration_s = perf_counter() - started
    _log_stage_event(
        "done",
        stage,
        status=report.get("status"),
        duration_s=f"{duration_s:.3f}",
        report=report.get("report_path"),
    )
    return report, duration_s


def _task1_run_name(created_utc: str, seed: int) -> str:
    timestamp = created_utc.replace("+00:00", "Z").replace("-", "").replace(":", "").replace(".", "")
    return f"{timestamp}_seed{seed}"


def _write_seeded_layout(args: argparse.Namespace, *, created_utc: str) -> tuple[Path, Path]:
    if args.seed is None:
        raise SystemExit("Provide --seed to generate a task1 layout.")
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


def _run_layout_stage(args: argparse.Namespace, *, created_utc: str | None = None) -> dict:
    if args.layout is not None:
        layout = load_stage0_layout(args.layout)
        return {
            "status": "success",
            "message": f"Using existing task1 layout with {len(layout.objects)} objects.",
            "report_path": str(args.layout),
            "layout_path": str(args.layout),
            "task1_run_dir": None,
        }
    if created_utc is None:
        created_utc = datetime.now(timezone.utc).isoformat()
    layout_path, run_dir = _write_seeded_layout(args, created_utc=created_utc)
    return {
        "status": "success",
        "message": f"Generated task1 layout for seed {args.seed}.",
        "report_path": str(layout_path),
        "layout_path": str(layout_path),
        "task1_run_dir": str(run_dir),
    }


def _run_survey_stage(
    args: argparse.Namespace,
    *,
    layout_path: Path | None = None,
    run_dir: Path | None = None,
    created_utc: str | None = None,
) -> dict:
    if created_utc is None:
        created_utc = datetime.now(timezone.utc).isoformat()
    if layout_path is None:
        if args.layout is None:
            raise SystemExit("Provide --layout for the survey stage, or omit --stage to run layout first.")
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
    return run_task1_survey(layout_path, config, detector=detector)


def _run_rough_stage(args: argparse.Namespace, *, survey_report_path: Path | None = None) -> dict:
    if survey_report_path is None:
        try:
            survey_report_path = args.survey_report or find_latest_survey_report(args.output_dir)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
    config = RoughConfig(
        scene_model_path=args.scene_model,
        yolo_profile_path=args.yolo_config,
        output_dir=args.output_dir,
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
        max_ik_iterations=_value_or_default(args.max_ik_iterations, RoughConfig().max_ik_iterations),
        ik_position_tolerance_m=_value_or_default(
            args.ik_position_tolerance,
            RoughConfig().ik_position_tolerance_m,
        ),
        ik_orientation_tolerance_rad=_value_or_default(
            args.ik_orientation_tolerance,
            RoughConfig().ik_orientation_tolerance_rad,
        ),
    )
    detector = _build_yolo_detector(config)
    return run_task1_rough(survey_report_path, config, detector=detector)


def _run_final_stage(args: argparse.Namespace, *, rough_report_path: Path | None = None) -> dict:
    if rough_report_path is None:
        try:
            rough_report_path = args.rough_report or find_latest_rough_report(args.output_dir)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
    config = FinalConfig(
        scene_model_path=args.scene_model,
        yolo_profile_path=args.yolo_config,
        output_dir=args.output_dir,
        camera_name=args.camera_name,
        image_width=args.image_width,
        image_height=args.image_height,
        tank_opening_z_m=args.tank_opening_z,
        opening_clearance_m=args.opening_clearance,
        yolo_confidence=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        yolo_image_size=args.yolo_imgsz,
        yolo_device=args.yolo_device,
        yolo_max_detections=args.yolo_max_det,
        yolo_tile_grid_size=args.yolo_tile_grid,
        yolo_tile_overlap=args.yolo_tile_overlap,
        yolo_tile_nms_iou=args.yolo_tile_nms_iou,
        save_raw_yolo_annotations=args.save_raw_yolo_annotations,
        save_debug_trace=args.save_debug_trace,
        run_yolo=not args.skip_yolo,
        plan_only=args.plan_only,
        strict_yolo=args.strict_yolo,
        max_ik_iterations=_value_or_default(
            args.max_ik_iterations,
            FinalConfig().max_ik_iterations,
        ),
        ik_position_tolerance_m=_value_or_default(
            args.ik_position_tolerance,
            FinalConfig().ik_position_tolerance_m,
        ),
        ik_orientation_tolerance_rad=_value_or_default(
            args.ik_orientation_tolerance,
            FinalConfig().ik_orientation_tolerance_rad,
        ),
    )
    detector = _build_yolo_detector(config)
    return run_task1_final(rough_report_path, config, detector=detector)


def _run_zoom_stage(args: argparse.Namespace, *, final_report_path: Path | None = None) -> dict:
    if final_report_path is None:
        try:
            final_report_path = args.final_report or find_latest_final_report(args.output_dir)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
    config = ZoomConfig(
        output_dir=args.output_dir,
        plan_only=args.plan_only,
    )
    return run_task1_zoom(final_report_path, config)


def _run_full_pipeline(args: argparse.Namespace) -> dict:
    created_utc = datetime.now(timezone.utc).isoformat()
    stage_durations: dict[str, float] = {}
    total_started = perf_counter()
    layout_report, stage_durations["layout"] = _run_timed_stage(
        "layout",
        lambda: _run_layout_stage(args, created_utc=created_utc),
    )
    layout_path = Path(str(layout_report["layout_path"]))
    run_dir_value = layout_report.get("task1_run_dir")
    run_dir = Path(str(run_dir_value)) if run_dir_value else None
    survey_report, stage_durations["survey"] = _run_timed_stage(
        "survey",
        lambda: _run_survey_stage(args, layout_path=layout_path, run_dir=run_dir, created_utc=created_utc),
    )
    rough_report, stage_durations["rough"] = _run_timed_stage(
        "rough",
        lambda: _run_rough_stage(args, survey_report_path=Path(str(survey_report["report_path"]))),
    )
    final_report, stage_durations["final"] = _run_timed_stage(
        "final",
        lambda: _run_final_stage(args, rough_report_path=Path(str(rough_report["report_path"]))),
    )
    zoom_report, stage_durations["zoom"] = _run_timed_stage(
        "zoom",
        lambda: _run_zoom_stage(args, final_report_path=Path(str(final_report["report_path"]))),
    )
    stage_durations["total"] = perf_counter() - total_started
    reports = {
        "layout": survey_report.get("layout_snapshot_path") or layout_report["layout_path"],
        "survey": survey_report["report_path"],
        "rough": rough_report["report_path"],
        "final": final_report["report_path"],
        "zoom": zoom_report["report_path"],
    }
    recognition_statuses = [
        str(survey_report.get("status")),
        str(rough_report.get("status")),
        str(final_report.get("status")),
        str(zoom_report.get("status")),
    ]
    if any(status == "failed" for status in recognition_statuses):
        status = "failed"
    elif any(status == "partial" for status in recognition_statuses):
        status = "partial"
    elif all(status == "plan_only" for status in recognition_statuses):
        status = "plan_only"
    elif any(status == "plan_only" for status in recognition_statuses):
        status = "partial"
    else:
        status = "success"
    stage_statuses = {
        "layout": str(layout_report.get("status")),
        "survey": str(survey_report.get("status")),
        "rough": str(rough_report.get("status")),
        "final": str(final_report.get("status")),
        "zoom": str(zoom_report.get("status")),
    }
    return {
        "status": status,
        "message": "Ran task1 recognition pipeline: "
        + ", ".join(f"{name}={stage_statuses[name]}" for name in reports),
        "report_path": zoom_report["report_path"],
        "stage_reports": reports,
        "stage_durations_seconds": _rounded_durations(stage_durations),
    }


def _rounded_durations(durations: dict[str, float]) -> dict[str, float]:
    return {stage: round(duration, 3) for stage, duration in durations.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Task1 recognition pipeline entrypoint.")
    parser.add_argument(
        "--stage",
        choices=("layout", "survey", "rough", "final", "zoom"),
        default=None,
        help="Recognition stage to run. Omit this option to run layout, survey, rough, final, and zoom in sequence.",
    )
    parser.add_argument(
        "--layout",
        type=Path,
        default=None,
        help="Explicit target object pose layout JSON. Required for --stage survey; full pipeline may use this instead of --seed.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed used by the layout stage when generating a task1 layout.",
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
        help="Survey report JSON for --stage rough. Defaults to the latest outputs/task1/*/survey/survey_report.json.",
    )
    parser.add_argument(
        "--rough-report",
        type=Path,
        default=None,
        help="Rough report JSON for --stage final. Defaults to the latest outputs/task1/*/rough/rough_report.json.",
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
        help="Also save raw YOLO annotated images under the active stage's raw_annotated directory.",
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
        "--save-debug-trace",
        action="store_true",
        help="Save full debug traces for stages that support separate debug artifacts.",
    )
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
    args = parser.parse_args()

    if args.stage is None:
        report = _run_full_pipeline(args)
    elif args.stage == "layout":
        report, duration_s = _run_timed_stage("layout", lambda: _run_layout_stage(args))
        report["stage_durations_seconds"] = _rounded_durations({"layout": duration_s, "total": duration_s})
    elif args.stage == "survey":
        report, duration_s = _run_timed_stage("survey", lambda: _run_survey_stage(args))
        report["stage_durations_seconds"] = _rounded_durations({"survey": duration_s, "total": duration_s})
    elif args.stage == "rough":
        report, duration_s = _run_timed_stage("rough", lambda: _run_rough_stage(args))
        report["stage_durations_seconds"] = _rounded_durations({"rough": duration_s, "total": duration_s})
    elif args.stage == "final":
        report, duration_s = _run_timed_stage("final", lambda: _run_final_stage(args))
        report["stage_durations_seconds"] = _rounded_durations({"final": duration_s, "total": duration_s})
    else:
        report, duration_s = _run_timed_stage("zoom", lambda: _run_zoom_stage(args))
        report["stage_durations_seconds"] = _rounded_durations({"zoom": duration_s, "total": duration_s})
    summary = {"status": report["status"], "message": report["message"]}
    if "stage_reports" in report:
        summary["stage_reports"] = report["stage_reports"]
    if "stage_durations_seconds" in report:
        summary["stage_durations_seconds"] = report["stage_durations_seconds"]
    print(json.dumps(summary, ensure_ascii=False))
    print(f"report={report['report_path']}")


if __name__ == "__main__":
    main()
