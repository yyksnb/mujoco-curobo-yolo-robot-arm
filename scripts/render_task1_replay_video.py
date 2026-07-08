from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path

add_src_to_path()

import replay_task1_output as replay_gui  # noqa: E402
from robot_arm_pipeline.task1.replay import (  # noqa: E402
    find_latest_task1_run,
    load_task1_replay_manifest,
    replay_manifest_path_for_run,
    write_task1_replay_manifest,
)
from robot_arm_pipeline.task1.survey import DEFAULT_OUTPUT_DIR  # noqa: E402


DEFAULT_VIDEO_WIDTH = 1280
DEFAULT_VIDEO_HEIGHT = 720
DEFAULT_VIDEO_FPS = 50.0
DEFAULT_CAPTURE_HOLD_SECONDS = 2.0
DEFAULT_FINAL_ZOOM_PHOTO_HOLD_SECONDS = 3.0
DEFAULT_VIDEO_NAME = "task1_replay_video.mp4"


@dataclass(frozen=True)
class _VideoFrameSpec:
    display_index: int
    wrist_index: int
    role: str
    label: str
    repeat_count: int


@dataclass(frozen=True)
class _FinalZoomPhotoSpec:
    object_id: str
    final_image_path: Path
    zoom_image_path: Path
    ordinal: int
    total: int


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a task1 replay as a side-by-side MP4 video.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Task1 run directory. Defaults to latest.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Task1 output root used when --run-dir is omitted.",
    )
    parser.add_argument("--manifest", type=Path, default=None, help="Existing replay_manifest.json.")
    parser.add_argument("--rebuild-manifest", action="store_true", help="Rebuild the replay manifest first.")
    parser.add_argument("--final-report", type=Path, default=None, help="Final report JSON used for the optional final/zoom photo tail.")
    parser.add_argument("--zoom-report", type=Path, default=None, help="Zoom report JSON used for the optional final/zoom photo tail.")
    parser.add_argument(
        "--phases",
        default="survey,row,final",
        help="Comma-separated phases to include when building a manifest: survey,row,final.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output MP4 path. Defaults to run_dir/replay/task1_replay_video.mp4.")
    parser.add_argument("--width", type=int, default=DEFAULT_VIDEO_WIDTH, help="Video width in pixels.")
    parser.add_argument("--height", type=int, default=DEFAULT_VIDEO_HEIGHT, help="Video height in pixels.")
    parser.add_argument("--fps", type=float, default=DEFAULT_VIDEO_FPS, help="Output video frames per second.")
    parser.add_argument(
        "--capture-hold-seconds",
        type=float,
        default=DEFAULT_CAPTURE_HOLD_SECONDS,
        help="Seconds to hold each task1 capture frame before moving to the next capture.",
    )
    parser.add_argument(
        "--final-zoom-photo-hold-seconds",
        type=float,
        default=DEFAULT_FINAL_ZOOM_PHOTO_HOLD_SECONDS,
        help="Seconds to hold each final/zoom composite photo appended after replay motion.",
    )
    parser.add_argument(
        "--append-final-zoom-photos",
        dest="append_final_zoom_photos",
        action="store_true",
        help="Append final/zoom composite photos after the replay sequence.",
    )
    parser.add_argument(
        "--no-append-final-zoom-photos",
        dest="append_final_zoom_photos",
        action="store_false",
        help="Do not append final/zoom composite photos.",
    )
    parser.add_argument(
        "--motion-frame-duration",
        type=float,
        default=replay_gui.DEFAULT_REPLAY_INTERPOLATED_FRAME_DURATION_S / replay_gui.DEFAULT_REPLAY_PLAYBACK_SPEED,
        help="Seconds assigned to each replay-only motion frame in the exported video.",
    )
    parser.add_argument(
        "--global-fovy-deg",
        type=float,
        default=replay_gui.DEFAULT_GLOBAL_FOVY_DEG,
        help="Vertical field of view for the global free camera.",
    )
    parser.add_argument(
        "--global-lookat",
        type=replay_gui._parse_float_triplet,
        default=replay_gui.DEFAULT_GLOBAL_LOOKAT,
        help="Interior global free-camera lookat point as x,y,z.",
    )
    parser.add_argument(
        "--global-distance",
        type=float,
        default=replay_gui.DEFAULT_GLOBAL_DISTANCE_M,
        help="Interior global free-camera orbit distance in meters.",
    )
    parser.add_argument(
        "--global-elevation-deg",
        type=float,
        default=replay_gui.DEFAULT_GLOBAL_ELEVATION_DEG,
        help="Interior global free-camera elevation angle in degrees.",
    )
    parser.add_argument(
        "--survey-global-fovy-deg",
        type=float,
        default=replay_gui.DEFAULT_SURVEY_GLOBAL_FOVY_DEG,
        help="Survey-phase global free-camera field of view.",
    )
    parser.add_argument(
        "--survey-global-lookat",
        type=replay_gui._parse_float_triplet,
        default=replay_gui.DEFAULT_SURVEY_GLOBAL_LOOKAT,
        help="Survey-phase global free-camera lookat point as x,y,z.",
    )
    parser.add_argument(
        "--survey-global-distance",
        type=float,
        default=replay_gui.DEFAULT_SURVEY_GLOBAL_DISTANCE_M,
        help="Survey-phase global free-camera orbit distance in meters.",
    )
    parser.add_argument(
        "--survey-global-azimuth-deg",
        type=float,
        default=replay_gui.DEFAULT_SURVEY_GLOBAL_AZIMUTH_DEG,
        help="Survey-phase global free-camera azimuth angle in degrees.",
    )
    parser.add_argument(
        "--survey-global-elevation-deg",
        type=float,
        default=replay_gui.DEFAULT_SURVEY_GLOBAL_ELEVATION_DEG,
        help="Survey-phase global free-camera elevation angle in degrees.",
    )
    parser.add_argument(
        "--replay-max-joint-step-rad",
        type=float,
        default=replay_gui.DEFAULT_REPLAY_MAX_JOINT_STEP_RAD,
        help="Maximum per-joint step, in radians, for replay-only interpolation.",
    )
    parser.add_argument(
        "--replay-interpolated-frame-duration",
        type=float,
        default=replay_gui.DEFAULT_REPLAY_INTERPOLATED_FRAME_DURATION_S,
        help="Seconds per generated interpolation frame before playback-speed scaling.",
    )
    parser.add_argument(
        "--replay-rrt-max-nodes",
        type=int,
        default=replay_gui.DEFAULT_REPLAY_RRT_MAX_NODES,
        help="Maximum sampled nodes per segment for replay-only smoothing.",
    )
    parser.add_argument(
        "--replay-rrt-step-rad",
        type=float,
        default=replay_gui.DEFAULT_REPLAY_RRT_STEP_RAD,
        help="RRT extension step in radians.",
    )
    parser.add_argument(
        "--replay-rrt-goal-sample-rate",
        type=float,
        default=replay_gui.DEFAULT_REPLAY_RRT_GOAL_SAMPLE_RATE,
        help="Goal sampling probability for replay-only RRT smoothing.",
    )
    parser.add_argument(
        "--replay-rrt-seed",
        type=int,
        default=replay_gui.DEFAULT_REPLAY_RRT_SEED,
        help="Deterministic random seed for replay-only RRT smoothing.",
    )
    parser.add_argument(
        "--replay-rrt-attempt-count",
        type=int,
        default=replay_gui.DEFAULT_REPLAY_RRT_ATTEMPT_COUNT,
        help="Deterministic RRT attempts per replay connector.",
    )
    parser.add_argument(
        "--replay-recorded-waypoint-window",
        type=int,
        default=replay_gui.DEFAULT_REPLAY_RECORDED_WAYPOINT_WINDOW,
        help="Number of neighboring recorded keyframes to try as replay-only connector waypoints.",
    )
    parser.add_argument(
        "--replay-shortcut-passes",
        type=int,
        default=replay_gui.DEFAULT_REPLAY_SHORTCUT_PASSES,
        help="Deterministic collision-checked shortcut passes for replay-only RRT paths.",
    )
    parser.add_argument(
        "--replay-heavy-connector",
        dest="replay_heavy_connector",
        action="store_true",
        help="Enable high-budget replay-only RRT as the final connector attempt.",
    )
    parser.add_argument(
        "--no-replay-heavy-connector",
        dest="replay_heavy_connector",
        action="store_false",
        help="Disable the high-budget final replay connector attempt.",
    )
    parser.add_argument(
        "--replay-heavy-rrt-max-nodes",
        type=int,
        default=replay_gui.DEFAULT_REPLAY_HEAVY_RRT_MAX_NODES,
        help="Maximum sampled nodes for the high-budget final replay connector.",
    )
    parser.add_argument(
        "--replay-heavy-rrt-step-rad",
        type=float,
        default=replay_gui.DEFAULT_REPLAY_HEAVY_RRT_STEP_RAD,
        help="RRT extension step for the high-budget final replay connector.",
    )
    parser.add_argument(
        "--replay-heavy-rrt-goal-sample-rate",
        type=float,
        default=replay_gui.DEFAULT_REPLAY_HEAVY_RRT_GOAL_SAMPLE_RATE,
        help="Goal sampling probability for the high-budget final replay connector.",
    )
    parser.add_argument(
        "--replay-motion-failure-policy",
        choices=("keyframe-cut", "error"),
        default="keyframe-cut",
        help="What to do when no collision-free connector is found.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root used to resolve relative report paths.",
    )
    parser.set_defaults(
        append_final_zoom_photos=True,
        replay_heavy_connector=replay_gui.DEFAULT_REPLAY_HEAVY_CONNECTOR,
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    _validate_args(args)

    manifest = _load_or_build_manifest(args)
    frames = manifest.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise SystemExit("Replay manifest contains no frames with robot qpos.")

    if find_spec("mujoco") is None:
        raise SystemExit("MuJoCo is not available. Run inside the mujoco-curobo conda environment.")
    if find_spec("cv2") is None:
        raise SystemExit("OpenCV cv2 is required to write MP4 video. Install opencv-python or conda-forge opencv.")
    if find_spec("PIL") is None:
        raise SystemExit("Pillow is required for video annotations. Install pillow.")

    import cv2  # type: ignore[import-not-found]
    import mujoco  # type: ignore[import-not-found]

    run_dir = replay_gui._resolve_run_dir(manifest, args.repo_root)
    output_path = args.output if args.output is not None else run_dir / "replay" / DEFAULT_VIDEO_NAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_path = replay_gui._resolve_required_path(
        manifest.get("scene_model_path"),
        repo_root=args.repo_root,
        run_dir=run_dir,
        description="scene_model_path",
    )
    layout_path = replay_gui._resolve_required_path(
        manifest.get("layout_snapshot_path"),
        repo_root=args.repo_root,
        run_dir=run_dir,
        description="layout_snapshot_path",
    )

    replay_scene = replay_gui._load_replay_scene(
        mujoco,
        model_path=model_path,
        layout_path=layout_path,
        camera_name=str(manifest.get("camera_name") or "wrist"),
    )
    frames, smoothing_summary = replay_gui._build_collision_checked_replay_motion(
        mujoco,
        replay_scene.model,
        replay_scene.data,
        frames,
        max_joint_step_rad=args.replay_max_joint_step_rad,
        interpolated_frame_duration_s=args.replay_interpolated_frame_duration,
        rrt_max_nodes=args.replay_rrt_max_nodes,
        rrt_step_rad=args.replay_rrt_step_rad,
        rrt_goal_sample_rate=args.replay_rrt_goal_sample_rate,
        rrt_seed=args.replay_rrt_seed,
        rrt_attempt_count=args.replay_rrt_attempt_count,
        recorded_waypoint_window=args.replay_recorded_waypoint_window,
        shortcut_passes=args.replay_shortcut_passes,
        heavy_connector=args.replay_heavy_connector,
        heavy_rrt_max_nodes=args.replay_heavy_rrt_max_nodes,
        heavy_rrt_step_rad=args.replay_heavy_rrt_step_rad,
        heavy_rrt_goal_sample_rate=args.replay_heavy_rrt_goal_sample_rate,
        motion_failure_policy=args.replay_motion_failure_policy,
    )
    timeline = _build_video_timeline(
        frames,
        fps=args.fps,
        capture_hold_seconds=args.capture_hold_seconds,
        motion_frame_duration_s=args.motion_frame_duration,
    )
    final_zoom_photos = (
        _load_final_zoom_photo_specs(
            run_dir=run_dir,
            repo_root=args.repo_root,
            final_report_path=args.final_report,
            zoom_report_path=args.zoom_report,
        )
        if args.append_final_zoom_photos
        else []
    )
    _render_video(
        cv2,
        mujoco,
        manifest=manifest,
        frames=frames,
        timeline=timeline,
        model=replay_scene.model,
        data=replay_scene.data,
        camera_name=str(manifest.get("camera_name") or "wrist"),
        final_zoom_photos=final_zoom_photos,
        output_path=output_path,
        width=args.width,
        height=args.height,
        fps=args.fps,
        final_zoom_photo_hold_seconds=args.final_zoom_photo_hold_seconds,
        global_fovy_deg=args.global_fovy_deg,
        global_lookat=args.global_lookat,
        global_distance=args.global_distance,
        global_elevation_deg=args.global_elevation_deg,
        survey_global_fovy_deg=args.survey_global_fovy_deg,
        survey_global_lookat=args.survey_global_lookat,
        survey_global_distance=args.survey_global_distance,
        survey_global_azimuth_deg=args.survey_global_azimuth_deg,
        survey_global_elevation_deg=args.survey_global_elevation_deg,
    )
    final_zoom_repeat_count = max(1, int(round(args.final_zoom_photo_hold_seconds * args.fps)))
    total_video_frames = sum(spec.repeat_count for spec in timeline) + final_zoom_repeat_count * len(final_zoom_photos)
    print(
        json.dumps(
            {
                "status": "success",
                "video_path": str(output_path),
                "fps": args.fps,
                "video_frame_count": total_video_frames,
                "capture_frame_count": sum(1 for frame in frames if not replay_gui._is_generated_motion_frame(frame)),
                "final_zoom_photo_count": len(final_zoom_photos),
                "display_frame_count": len(frames),
                "replay_motion": smoothing_summary.to_dict(),
            },
            ensure_ascii=False,
        )
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.width < 2 or args.height < 2:
        raise SystemExit("--width and --height must be at least 2 pixels")
    if args.fps <= 0.0:
        raise SystemExit("--fps must be positive")
    if args.capture_hold_seconds <= 0.0:
        raise SystemExit("--capture-hold-seconds must be positive")
    if args.final_zoom_photo_hold_seconds <= 0.0:
        raise SystemExit("--final-zoom-photo-hold-seconds must be positive")
    if args.motion_frame_duration <= 0.0:
        raise SystemExit("--motion-frame-duration must be positive")
    if args.replay_max_joint_step_rad <= 0.0:
        raise SystemExit("--replay-max-joint-step-rad must be positive")
    if args.replay_interpolated_frame_duration <= 0.0:
        raise SystemExit("--replay-interpolated-frame-duration must be positive")
    if args.replay_rrt_max_nodes <= 0:
        raise SystemExit("--replay-rrt-max-nodes must be positive")
    if args.replay_rrt_step_rad <= 0.0:
        raise SystemExit("--replay-rrt-step-rad must be positive")
    if args.replay_rrt_attempt_count <= 0:
        raise SystemExit("--replay-rrt-attempt-count must be positive")
    if args.replay_recorded_waypoint_window < 0:
        raise SystemExit("--replay-recorded-waypoint-window must be non-negative")
    if args.replay_shortcut_passes < 0:
        raise SystemExit("--replay-shortcut-passes must be non-negative")
    if args.replay_heavy_rrt_max_nodes <= 0:
        raise SystemExit("--replay-heavy-rrt-max-nodes must be positive")
    if args.replay_heavy_rrt_step_rad <= 0.0:
        raise SystemExit("--replay-heavy-rrt-step-rad must be positive")
    for name in ("replay_rrt_goal_sample_rate", "replay_heavy_rrt_goal_sample_rate"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be in [0, 1]")


def _load_or_build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if args.manifest is not None:
        return load_task1_replay_manifest(args.manifest)
    run_dir = args.run_dir or find_latest_task1_run(args.output_dir)
    manifest_path = replay_manifest_path_for_run(run_dir)
    phases = tuple(item.strip() for item in args.phases.split(",") if item.strip())
    if args.rebuild_manifest or not manifest_path.exists():
        return write_task1_replay_manifest(run_dir, phases=phases, manifest_path=manifest_path)
    return load_task1_replay_manifest(manifest_path)


def _build_video_timeline(
    frames: list[dict[str, Any]],
    *,
    fps: float,
    capture_hold_seconds: float,
    motion_frame_duration_s: float,
) -> list[_VideoFrameSpec]:
    capture_indices = [index for index, frame in enumerate(frames) if not replay_gui._is_generated_motion_frame(frame)]
    if not capture_indices:
        raise SystemExit("Replay contains no task1 capture frames.")

    labels = _capture_phase_labels(frames)
    hold_repeats = max(1, int(round(capture_hold_seconds * fps)))
    motion_repeats = max(1, int(round(motion_frame_duration_s * fps)))
    timeline: list[_VideoFrameSpec] = []
    for capture_order, capture_index in enumerate(capture_indices):
        timeline.append(
            _VideoFrameSpec(
                display_index=capture_index,
                wrist_index=capture_index,
                role="capture_hold",
                label=labels[capture_index],
                repeat_count=hold_repeats,
            )
        )
        if capture_order + 1 >= len(capture_indices):
            continue
        next_capture_index = capture_indices[capture_order + 1]
        for motion_index in range(capture_index + 1, next_capture_index):
            timeline.append(
                _VideoFrameSpec(
                    display_index=motion_index,
                    wrist_index=capture_index,
                    role="motion",
                    label="moving...",
                    repeat_count=motion_repeats,
                )
            )
    return timeline


def _capture_phase_labels(frames: list[dict[str, Any]]) -> dict[int, str]:
    totals: dict[str, int] = {}
    capture_indices = []
    for index, frame in enumerate(frames):
        if replay_gui._is_generated_motion_frame(frame):
            continue
        phase = _frame_phase_label(frame)
        totals[phase] = totals.get(phase, 0) + 1
        capture_indices.append(index)

    seen: dict[str, int] = {}
    labels: dict[int, str] = {}
    for index in capture_indices:
        phase = _frame_phase_label(frames[index])
        seen[phase] = seen.get(phase, 0) + 1
        labels[index] = f"{phase} {seen[phase]}/{totals[phase]}"
    return labels


def _frame_phase_label(frame: dict[str, Any]) -> str:
    phase = str(frame.get("phase") or "capture").strip()
    return phase if phase else "capture"


def _load_final_zoom_photo_specs(
    *,
    run_dir: Path,
    repo_root: Path,
    final_report_path: Path | None,
    zoom_report_path: Path | None,
) -> list[_FinalZoomPhotoSpec]:
    resolved_final_report = final_report_path or run_dir / "final" / "final_report.json"
    resolved_zoom_report = zoom_report_path or run_dir / "zoom" / "zoom_report.json"
    if not resolved_final_report.exists():
        raise SystemExit(f"Final/zoom photo tail requires final report: {resolved_final_report}")
    if not resolved_zoom_report.exists():
        raise SystemExit(f"Final/zoom photo tail requires zoom report: {resolved_zoom_report}")

    final_report = json.loads(resolved_final_report.read_text(encoding="utf-8"))
    zoom_report = json.loads(resolved_zoom_report.read_text(encoding="utf-8"))
    if final_report.get("schema_version") != "task1_final_report_v1":
        raise SystemExit(f"unsupported final report schema_version: {final_report.get('schema_version')}")
    if zoom_report.get("schema_version") != "task1_zoom_report_v1":
        raise SystemExit(f"unsupported zoom report schema_version: {zoom_report.get('schema_version')}")

    zoom_by_object_id = _zoom_objects_by_id(zoom_report)
    final_objects = final_report.get("stable_objects")
    if not isinstance(final_objects, list) or not final_objects:
        raise SystemExit("Final/zoom photo tail requires final_report.stable_objects.")

    pairs: list[tuple[str, Path, Path]] = []
    for index, final_object in enumerate(final_objects):
        if not isinstance(final_object, dict):
            raise SystemExit(f"final_report.stable_objects[{index}] must be an object")
        object_id = str(final_object.get("object_id") or "").strip()
        if not object_id:
            raise SystemExit(f"final_report.stable_objects[{index}] is missing object_id")
        zoom_object = zoom_by_object_id.get(object_id)
        if zoom_object is None:
            raise SystemExit(f"zoom_report.zoomed_objects is missing object_id={object_id}")
        final_image = _resolve_existing_path(
            final_object.get("final_image_path"),
            repo_root=repo_root,
            run_dir=run_dir,
            description=f"final_image_path for {object_id}",
        )
        zoom_image = _resolve_existing_path(
            _zoom_selected_image_path(zoom_object),
            repo_root=repo_root,
            run_dir=run_dir,
            description=f"selected zoom image for {object_id}",
        )
        source_final = zoom_object.get("source_final_image_path")
        if source_final:
            resolved_source_final = _resolve_existing_path(
                source_final,
                repo_root=repo_root,
                run_dir=run_dir,
                description=f"source_final_image_path for {object_id}",
            )
            if resolved_source_final.resolve() != final_image.resolve():
                raise SystemExit(
                    "zoom source_final_image_path does not match final stable object image: "
                    f"object_id={object_id}, final={final_image}, zoom_source={resolved_source_final}"
                )
        pairs.append((object_id, final_image, zoom_image))

    total = len(pairs)
    return [
        _FinalZoomPhotoSpec(
            object_id=object_id,
            final_image_path=final_image,
            zoom_image_path=zoom_image,
            ordinal=index + 1,
            total=total,
        )
        for index, (object_id, final_image, zoom_image) in enumerate(pairs)
    ]


def _zoom_objects_by_id(zoom_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    zoomed_objects = zoom_report.get("zoomed_objects")
    if not isinstance(zoomed_objects, list) or not zoomed_objects:
        raise SystemExit("Final/zoom photo tail requires zoom_report.zoomed_objects.")
    result: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(zoomed_objects):
        if not isinstance(item, dict):
            raise SystemExit(f"zoom_report.zoomed_objects[{index}] must be an object")
        object_id = str(item.get("object_id") or "").strip()
        if not object_id:
            raise SystemExit(f"zoom_report.zoomed_objects[{index}] is missing object_id")
        if object_id in result:
            raise SystemExit(f"zoom_report.zoomed_objects has duplicate object_id={object_id}")
        result[object_id] = item
    return result


def _zoom_selected_image_path(zoom_object: dict[str, Any]) -> Any:
    selected_zoom = zoom_object.get("selected_zoom")
    if isinstance(selected_zoom, dict):
        value = selected_zoom.get("selected_image_path") or selected_zoom.get("output_image_path")
        if value:
            return value
    return zoom_object.get("output_image_path")


def _resolve_existing_path(value: Any, *, repo_root: Path, run_dir: Path, description: str) -> Path:
    if value is None:
        raise SystemExit(f"missing {description}")
    path = Path(str(value))
    candidates = [path] if path.is_absolute() else [repo_root / path, run_dir / path, path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(f"{description} does not exist: {value}")


def _render_video(
    cv2: Any,
    mujoco: Any,
    *,
    manifest: dict[str, Any],
    frames: list[dict[str, Any]],
    timeline: list[_VideoFrameSpec],
    model: Any,
    data: Any,
    camera_name: str,
    final_zoom_photos: list[_FinalZoomPhotoSpec],
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    final_zoom_photo_hold_seconds: float,
    global_fovy_deg: float,
    global_lookat: tuple[float, float, float] | None,
    global_distance: float | None,
    global_elevation_deg: float | None,
    survey_global_fovy_deg: float | None,
    survey_global_lookat: tuple[float, float, float] | None,
    survey_global_distance: float | None,
    survey_global_azimuth_deg: float | None,
    survey_global_elevation_deg: float | None,
) -> None:
    import numpy as np

    left_width = max(1, width // 2)
    right_width = max(1, width - left_width)
    _ensure_offscreen_size(model, width=max(left_width, right_width), height=height)
    global_renderer = mujoco.Renderer(model, height=height, width=left_width)
    wrist_renderer = mujoco.Renderer(model, height=height, width=right_width)
    wrist_data = mujoco.MjData(model)
    global_camera = mujoco.MjvCamera()
    wrist_camera = mujoco.MjvCamera()
    wrist_camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    wrist_camera.fixedcamid = replay_gui._resolve_camera_id(mujoco, model, camera_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (int(width), int(height)))
    if not writer.isOpened():
        raise SystemExit(f"OpenCV could not open video writer for {output_path}")
    try:
        written = 0
        for spec in timeline:
            frame = frames[spec.display_index]
            wrist_frame = frames[spec.wrist_index]
            replay_gui._apply_frame(mujoco, model, data, frame)
            replay_gui._apply_frame(mujoco, model, wrist_data, wrist_frame)
            replay_gui._configure_mjv_global_camera(
                mujoco,
                model,
                data,
                global_camera,
                manifest,
                frame,
                global_lookat=global_lookat,
                global_distance=global_distance,
                global_elevation_deg=global_elevation_deg,
                survey_global_lookat=survey_global_lookat,
                survey_global_distance=survey_global_distance,
                survey_global_azimuth_deg=survey_global_azimuth_deg,
                survey_global_elevation_deg=survey_global_elevation_deg,
            )
            global_rgb = _render_rgb(
                global_renderer,
                model=model,
                data=data,
                camera=global_camera,
                global_fovy_deg=replay_gui._global_fovy_for_frame(
                    frame,
                    global_fovy_deg=global_fovy_deg,
                    survey_global_fovy_deg=survey_global_fovy_deg,
                ),
            )
            wrist_rgb = _render_rgb(wrist_renderer, model=model, data=wrist_data, camera=wrist_camera)
            composed = np.concatenate([global_rgb, wrist_rgb], axis=1)
            composed = _annotate_video_frame(
                composed,
                left_width=left_width,
                role=spec.role,
                label=spec.label,
            )
            bgr = cv2.cvtColor(composed, cv2.COLOR_RGB2BGR)
            for _repeat in range(spec.repeat_count):
                writer.write(bgr)
                written += 1
        photo_repeat_count = max(1, int(round(final_zoom_photo_hold_seconds * fps)))
        for photo in final_zoom_photos:
            composed = _compose_final_zoom_photo_frame(photo, width=width, height=height)
            bgr = cv2.cvtColor(composed, cv2.COLOR_RGB2BGR)
            for _repeat in range(photo_repeat_count):
                writer.write(bgr)
                written += 1
        if written <= 0:
            raise SystemExit("No video frames were written.")
    finally:
        writer.release()
        if hasattr(global_renderer, "close"):
            global_renderer.close()
        if hasattr(wrist_renderer, "close"):
            wrist_renderer.close()


def _ensure_offscreen_size(model: Any, *, width: int, height: int) -> None:
    if getattr(model, "vis", None) is None:
        return
    global_vis = getattr(model.vis, "global_", None)
    if global_vis is None:
        return
    if int(global_vis.offwidth) < width:
        global_vis.offwidth = int(width)
    if int(global_vis.offheight) < height:
        global_vis.offheight = int(height)


def _render_rgb(renderer: Any, *, model: Any, data: Any, camera: Any, global_fovy_deg: float | None = None) -> Any:
    original_fovy = replay_gui._global_free_camera_fovy(model)
    if global_fovy_deg is not None:
        replay_gui._set_global_free_camera_fovy(model, global_fovy_deg)
    try:
        renderer.update_scene(data, camera=camera)
        return renderer.render()[:, :, :3].copy()
    finally:
        if original_fovy is not None and global_fovy_deg is not None:
            replay_gui._set_global_free_camera_fovy(model, original_fovy)


def _annotate_video_frame(rgb: Any, *, left_width: int, role: str, label: str) -> Any:
    import numpy as np
    from PIL import Image, ImageDraw

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    if role == "motion":
        draw.rectangle((left_width, 0, width, height), fill=(128, 128, 128, 145))
        _draw_large_text(draw, (14, height - _large_font_size(height) - 20), "moving...", height=height)
        _draw_centered_text(draw, (left_width, 0, width, height), "moving...", height=height)
    else:
        _draw_large_text(draw, (14, height - _large_font_size(height) - 20), label, height=height)
    return np.asarray(image.convert("RGB"))


def _compose_final_zoom_photo_frame(photo: _FinalZoomPhotoSpec, *, width: int, height: int) -> Any:
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps

    canvas = Image.new("RGB", (int(width), int(height)), color=(0, 0, 0))
    tile_width = int(width) // 2
    tile_height = int(height) // 2
    if tile_width <= 0 or tile_height <= 0:
        raise SystemExit("video size is too small for final/zoom composite photo")

    with Image.open(photo.final_image_path).convert("RGB") as final_image:
        final_tile = ImageOps.fit(final_image, (tile_width, tile_height), method=Image.Resampling.LANCZOS)
    with Image.open(photo.zoom_image_path).convert("RGB") as zoom_image:
        zoom_tile = ImageOps.fit(zoom_image, (int(width) - tile_width, int(height) - tile_height), method=Image.Resampling.LANCZOS)

    canvas.paste(final_tile, (0, 0))
    canvas.paste(zoom_tile, (tile_width, tile_height))
    draw = ImageDraw.Draw(canvas, "RGBA")
    final_text = f"final {photo.ordinal}/{photo.total}"
    zoom_text = f"zoom {photo.ordinal}/{photo.total}"
    final_xy, zoom_xy, zoom_font = _final_zoom_label_layout(
        draw,
        width=width,
        height=height,
        tile_width=tile_width,
        tile_height=tile_height,
        zoom_text=zoom_text,
    )
    _draw_large_text(draw, final_xy, final_text, height=height)
    _draw_text(draw, zoom_xy, zoom_text, font=zoom_font)
    return np.asarray(canvas.convert("RGB"))


def _final_zoom_label_layout(
    draw: Any,
    *,
    width: int,
    height: int,
    tile_width: int,
    tile_height: int,
    zoom_text: str,
) -> tuple[tuple[int, int], tuple[int, int], Any]:
    font_size = _large_font_size(height)
    margin = max(12, int(round(height * 0.025)))
    final_y = min(tile_height + margin, max(height - font_size - margin, 0))
    zoom_font = _annotation_font(font_size)
    zoom_bbox = draw.textbbox((0, 0), zoom_text, font=zoom_font, stroke_width=2)
    zoom_width = zoom_bbox[2] - zoom_bbox[0]
    zoom_height = zoom_bbox[3] - zoom_bbox[1]
    zoom_x = max(width - zoom_width - 16, tile_width + 12)
    zoom_y = max(margin, tile_height - zoom_height - margin)
    return (14, final_y), (zoom_x, zoom_y), zoom_font


def _draw_large_text(draw: Any, xy: tuple[int, int], text: str, *, height: int) -> None:
    font = _annotation_font(_large_font_size(height))
    _draw_text(draw, xy, text, font=font)


def _draw_centered_text(draw: Any, rect: tuple[int, int, int, int], text: str, *, height: int) -> None:
    font = _annotation_font(max(36, int(height * 0.085)))
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=2)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    left, top, right, bottom = rect
    xy = (
        left + max((right - left - text_width) // 2, 0),
        top + max((bottom - top - text_height) // 2, 0),
    )
    _draw_text(draw, xy, text, font=font)


def _draw_text(draw: Any, xy: tuple[int, int], text: str, *, font: Any) -> None:
    draw.text(xy, text, font=font, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(82, 128, 210, 255))


def _large_font_size(height: int) -> int:
    return max(38, int(height * 0.085))


def _annotation_font(size: int) -> Any:
    from PIL import ImageFont

    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


if __name__ == "__main__":
    main()
