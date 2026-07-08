from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from threading import Lock
from typing import Any

from _bootstrap import add_src_to_path

add_src_to_path()

from robot_arm_pipeline.task1.replay import (  # noqa: E402
    apply_task1_layout_to_mujoco,
    find_latest_task1_run,
    load_task1_replay_manifest,
    replay_manifest_path_for_run,
    write_task1_replay_manifest,
)
from robot_arm_pipeline.task1.survey import DEFAULT_OUTPUT_DIR  # noqa: E402


DEFAULT_GLOBAL_LOOKAT = (0.5, 0.5, 0.15)
DEFAULT_GLOBAL_DISTANCE_M = 0.75
DEFAULT_GLOBAL_ELEVATION_DEG = -20.0
DEFAULT_GLOBAL_FOVY_DEG = 75.0
DEFAULT_SURVEY_GLOBAL_LOOKAT = (0.5, 0.5, 0.4)
DEFAULT_SURVEY_GLOBAL_DISTANCE_M = 1.0
DEFAULT_SURVEY_GLOBAL_AZIMUTH_DEG = 135.0
DEFAULT_SURVEY_GLOBAL_ELEVATION_DEG = -30.0
DEFAULT_SURVEY_GLOBAL_FOVY_DEG = 80.0
DEFAULT_REPLAY_MAX_JOINT_STEP_RAD = 0.08
DEFAULT_REPLAY_INTERPOLATED_FRAME_DURATION_S = 0.04
DEFAULT_REPLAY_PLAYBACK_SPEED = 2.0
DEFAULT_REPLAY_RRT_STEP_RAD = 0.12
DEFAULT_REPLAY_RRT_MAX_NODES = 2500
DEFAULT_REPLAY_RRT_GOAL_SAMPLE_RATE = 0.20
DEFAULT_REPLAY_RRT_SEED = 0
DEFAULT_REPLAY_RRT_ATTEMPT_COUNT = 1
DEFAULT_REPLAY_RECORDED_WAYPOINT_WINDOW = 2
DEFAULT_REPLAY_SHORTCUT_PASSES = 2
DEFAULT_REPLAY_HEAVY_CONNECTOR = True
DEFAULT_REPLAY_HEAVY_RRT_MAX_NODES = 50000
DEFAULT_REPLAY_HEAVY_RRT_STEP_RAD = 0.20
DEFAULT_REPLAY_HEAVY_RRT_GOAL_SAMPLE_RATE = 0.15


def _parse_float_triplet(text: str) -> tuple[float, float, float]:
    try:
        values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected three comma-separated numbers, e.g. 0.5,0.5,0.13") from exc
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated numbers, e.g. 0.5,0.5,0.13")
    return values


@dataclass(frozen=True)
class _GuiKeyMap:
    space: int
    next_frame: int
    reset: int


@dataclass
class _ReplayControl:
    paused: bool
    frame_requests: int = 0
    reset_requested: bool = False
    lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def handle_key(self, key: int, keymap: _GuiKeyMap) -> None:
        with self.lock:
            if key == keymap.space:
                self.paused = not self.paused
                if not self.paused:
                    self.frame_requests = 0
            elif key == keymap.next_frame:
                self.paused = True
                self.frame_requests += 1
            elif key == keymap.reset:
                self.paused = True
                self.frame_requests = 0
                self.reset_requested = True

    def snapshot(self) -> tuple[bool, int]:
        with self.lock:
            return self.paused, self.frame_requests

    def consume_frame_request(self) -> bool:
        with self.lock:
            if self.frame_requests <= 0:
                return False
            self.frame_requests -= 1
            return True

    def consume_reset_request(self) -> bool:
        with self.lock:
            if not self.reset_requested:
                return False
            self.reset_requested = False
            return True


@dataclass(frozen=True)
class _ReplayScene:
    model: Any
    data: Any
    camera_id: int


@dataclass(frozen=True)
class _ReplayMotionSummary:
    input_frame_count: int
    display_frame_count: int
    interpolated_frame_count: int
    collision_checked_qpos_count: int
    max_joint_step_rad: float
    continuous_joint_adjusted_keyframe_count: int
    continuous_joint_adjusted_joint_count: int
    continuous_joint_max_adjustment_rad: float
    planned_boundary_connector_count: int
    waypoint_candidate_count: int
    waypoint_connector_count: int
    coordinate_sweep_connector_count: int
    path_shortcut_passes: int
    path_shortcut_count: int
    path_shortcut_collision_check_count: int
    heavy_rrt_connector_count: int
    heavy_rrt_max_nodes: int
    heavy_rrt_step_rad: float
    direct_segment_count: int
    rrt_segment_count: int
    rrt_step_rad: float
    keyframe_cut_count: int
    failed_connector_cut_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_frame_count": self.input_frame_count,
            "display_frame_count": self.display_frame_count,
            "interpolated_frame_count": self.interpolated_frame_count,
            "collision_checked_qpos_count": self.collision_checked_qpos_count,
            "max_joint_step_rad": self.max_joint_step_rad,
            "continuous_joint_adjusted_keyframe_count": self.continuous_joint_adjusted_keyframe_count,
            "continuous_joint_adjusted_joint_count": self.continuous_joint_adjusted_joint_count,
            "continuous_joint_max_adjustment_rad": self.continuous_joint_max_adjustment_rad,
            "planned_boundary_connector_count": self.planned_boundary_connector_count,
            "waypoint_candidate_count": self.waypoint_candidate_count,
            "waypoint_connector_count": self.waypoint_connector_count,
            "coordinate_sweep_connector_count": self.coordinate_sweep_connector_count,
            "path_shortcut_passes": self.path_shortcut_passes,
            "path_shortcut_count": self.path_shortcut_count,
            "path_shortcut_collision_check_count": self.path_shortcut_collision_check_count,
            "heavy_rrt_connector_count": self.heavy_rrt_connector_count,
            "heavy_rrt_max_nodes": self.heavy_rrt_max_nodes,
            "heavy_rrt_step_rad": self.heavy_rrt_step_rad,
            "direct_segment_count": self.direct_segment_count,
            "rrt_segment_count": self.rrt_segment_count,
            "rrt_step_rad": self.rrt_step_rad,
            "keyframe_cut_count": self.keyframe_cut_count,
            "failed_connector_cut_count": self.failed_connector_cut_count,
        }


@dataclass(frozen=True)
class _ReplaySegmentPlan:
    qpos_samples: list[list[float]]
    strategy: str
    collision_check_count: int
    direct_edge_count: int = 0
    rrt_edge_count: int = 0
    waypoint_count: int = 0
    coordinate_sweep_count: int = 0
    shortcut_count: int = 0
    shortcut_collision_check_count: int = 0
    heavy_rrt_count: int = 0


@dataclass(frozen=True)
class _ReplayKeyframeQpos:
    qpos: list[float]
    adjusted_joint_count: int = 0
    max_adjustment_rad: float = 0.0


@dataclass(frozen=True)
class _ReplayWaypointCandidate:
    name: str
    qpos: list[float]
    source: str


@dataclass(frozen=True)
class _ReplayAdvanceResult:
    current_index: int
    finished: bool
    motion_target_index: int | None
    elapsed_remainder_s: float
    reached_indices: tuple[int, ...]
    advance_limit_reached: bool = False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a task1 output directory in MuJoCo GUI.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Task1 run directory, e.g. outputs/task1/20260707T130704759946Z_seed30. Defaults to latest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Task1 output root used when --run-dir is omitted.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Existing replay_manifest.json. When omitted the script uses or builds run_dir/replay/replay_manifest.json.",
    )
    parser.add_argument(
        "--rebuild-manifest",
        action="store_true",
        help="Rebuild run_dir/replay/replay_manifest.json from the task1 reports before replay.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Build/load the replay manifest and exit without opening MuJoCo GUI.",
    )
    parser.add_argument(
        "--phases",
        default="survey,row,final",
        help="Comma-separated phases to include when building a manifest: survey,row,final.",
    )
    parser.add_argument(
        "--view",
        choices=("global", "wrist", "both"),
        default="global",
        help="Viewer mode. both renders global and wrist-camera views side by side in one window.",
    )
    parser.add_argument("--frame-duration", type=float, default=1.2, help="Seconds per frame in continuous playback.")
    parser.add_argument("--poll-seconds", type=float, default=0.005, help="GUI polling interval.")
    parser.add_argument(
        "--global-fovy-deg",
        type=float,
        default=DEFAULT_GLOBAL_FOVY_DEG,
        help="Vertical field of view for the global free camera. The wrist fixed camera keeps its MJCF fovy.",
    )
    parser.add_argument(
        "--global-lookat",
        type=_parse_float_triplet,
        default=DEFAULT_GLOBAL_LOOKAT,
        help="Interior global free-camera lookat point as x,y,z in world coordinates for row/final replay.",
    )
    parser.add_argument(
        "--global-distance",
        type=float,
        default=DEFAULT_GLOBAL_DISTANCE_M,
        help="Interior global free-camera orbit distance in meters for row/final replay.",
    )
    parser.add_argument(
        "--global-elevation-deg",
        type=float,
        default=DEFAULT_GLOBAL_ELEVATION_DEG,
        help="Interior global free-camera elevation angle in degrees for row/final replay.",
    )
    parser.add_argument(
        "--survey-global-fovy-deg",
        type=float,
        default=DEFAULT_SURVEY_GLOBAL_FOVY_DEG,
        help="Survey-phase global free-camera field of view.",
    )
    parser.add_argument(
        "--survey-global-lookat",
        type=_parse_float_triplet,
        default=DEFAULT_SURVEY_GLOBAL_LOOKAT,
        help="Survey-phase global free-camera lookat point as x,y,z.",
    )
    parser.add_argument(
        "--survey-global-distance",
        type=float,
        default=DEFAULT_SURVEY_GLOBAL_DISTANCE_M,
        help="Survey-phase global free-camera orbit distance in meters.",
    )
    parser.add_argument(
        "--survey-global-azimuth-deg",
        type=float,
        default=DEFAULT_SURVEY_GLOBAL_AZIMUTH_DEG,
        help="Survey-phase global free-camera azimuth angle in degrees.",
    )
    parser.add_argument(
        "--survey-global-elevation-deg",
        type=float,
        default=DEFAULT_SURVEY_GLOBAL_ELEVATION_DEG,
        help="Survey-phase global free-camera elevation angle in degrees.",
    )
    parser.add_argument(
        "--smooth-replay",
        dest="smooth_replay",
        action="store_true",
        help="Build collision-checked in-memory interpolation frames for GUI display.",
    )
    parser.add_argument(
        "--no-smooth-replay",
        dest="smooth_replay",
        action="store_false",
        help="Show only recorded replay frames without GUI-only interpolation.",
    )
    parser.add_argument(
        "--replay-max-joint-step-rad",
        type=float,
        default=DEFAULT_REPLAY_MAX_JOINT_STEP_RAD,
        help="Maximum per-joint step, in radians, for GUI-only replay interpolation.",
    )
    parser.add_argument(
        "--replay-interpolated-frame-duration",
        type=float,
        default=DEFAULT_REPLAY_INTERPOLATED_FRAME_DURATION_S,
        help="Seconds per generated interpolation frame in continuous GUI playback.",
    )
    parser.add_argument(
        "--replay-playback-speed",
        type=float,
        default=DEFAULT_REPLAY_PLAYBACK_SPEED,
        help="GUI-only playback speed multiplier applied to recorded and generated replay frame durations.",
    )
    parser.add_argument(
        "--replay-rrt-max-nodes",
        type=int,
        default=DEFAULT_REPLAY_RRT_MAX_NODES,
        help="Maximum sampled nodes per segment for replay-only collision-free motion smoothing.",
    )
    parser.add_argument(
        "--replay-rrt-step-rad",
        type=float,
        default=DEFAULT_REPLAY_RRT_STEP_RAD,
        help="RRT extension step, in radians, for replay-only motion smoothing. Collision checks still use --replay-max-joint-step-rad.",
    )
    parser.add_argument(
        "--replay-rrt-goal-sample-rate",
        type=float,
        default=DEFAULT_REPLAY_RRT_GOAL_SAMPLE_RATE,
        help="Goal sampling probability for replay-only RRT smoothing.",
    )
    parser.add_argument(
        "--replay-rrt-seed",
        type=int,
        default=DEFAULT_REPLAY_RRT_SEED,
        help="Deterministic random seed for replay-only RRT smoothing.",
    )
    parser.add_argument(
        "--replay-rrt-attempt-count",
        type=int,
        default=DEFAULT_REPLAY_RRT_ATTEMPT_COUNT,
        help="Deterministic RRT attempts per replay connector before falling back to waypoint strategies.",
    )
    parser.add_argument(
        "--replay-recorded-waypoint-window",
        type=int,
        default=DEFAULT_REPLAY_RECORDED_WAYPOINT_WINDOW,
        help="Number of neighboring recorded keyframes on each side to try as replay-only connector waypoints after direct/RRT failure.",
    )
    parser.add_argument(
        "--replay-shortcut-passes",
        type=int,
        default=DEFAULT_REPLAY_SHORTCUT_PASSES,
        help="Deterministic collision-checked shortcut passes applied to replay-only RRT paths before display densification. Use 0 to disable.",
    )
    parser.add_argument(
        "--replay-heavy-connector",
        dest="replay_heavy_connector",
        action="store_true",
        help="Enable high-budget replay-only RRT as the final connector attempt before cutting a frame.",
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
        default=DEFAULT_REPLAY_HEAVY_RRT_MAX_NODES,
        help="Maximum sampled nodes for the high-budget final replay connector.",
    )
    parser.add_argument(
        "--replay-heavy-rrt-step-rad",
        type=float,
        default=DEFAULT_REPLAY_HEAVY_RRT_STEP_RAD,
        help="RRT extension step for the high-budget final replay connector.",
    )
    parser.add_argument(
        "--replay-heavy-rrt-goal-sample-rate",
        type=float,
        default=DEFAULT_REPLAY_HEAVY_RRT_GOAL_SAMPLE_RATE,
        help="Goal sampling probability for the high-budget final replay connector.",
    )
    parser.add_argument(
        "--replay-motion-failure-policy",
        choices=("keyframe-cut", "error"),
        default="keyframe-cut",
        help="What replay GUI does when no collision-free connector is found. keyframe-cut marks a visible cut; error exits.",
    )
    parser.add_argument("--loop", action="store_true", help="Loop back to the first frame after the last frame.")
    parser.add_argument(
        "--wait-for-close",
        dest="wait_for_close",
        action="store_true",
        help="Keep the viewer open after the last frame until closed manually.",
    )
    parser.add_argument(
        "--close-on-finish",
        dest="wait_for_close",
        action="store_false",
        help="Close the viewer after the last frame in continuous mode.",
    )
    parser.add_argument(
        "--step-by-step",
        dest="step_by_step",
        action="store_true",
        help="Start paused and advance frames with F9.",
    )
    parser.add_argument(
        "--continuous",
        dest="step_by_step",
        action="store_false",
        help="Start in timed continuous playback.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root used to resolve relative paths stored in reports.",
    )
    parser.set_defaults(
        step_by_step=True,
        wait_for_close=True,
        smooth_replay=True,
        replay_heavy_connector=DEFAULT_REPLAY_HEAVY_CONNECTOR,
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.frame_duration <= 0.0:
        raise SystemExit("--frame-duration must be positive")
    if args.poll_seconds <= 0.0:
        raise SystemExit("--poll-seconds must be positive")
    if not 1.0 <= args.global_fovy_deg <= 170.0:
        raise SystemExit("--global-fovy-deg must be in [1, 170]")
    if args.global_distance is not None and args.global_distance <= 0.0:
        raise SystemExit("--global-distance must be positive")
    if args.global_elevation_deg is not None and not -89.0 <= args.global_elevation_deg <= 89.0:
        raise SystemExit("--global-elevation-deg must be in [-89, 89]")
    if args.survey_global_fovy_deg is not None and not 1.0 <= args.survey_global_fovy_deg <= 170.0:
        raise SystemExit("--survey-global-fovy-deg must be in [1, 170]")
    if args.survey_global_distance is not None and args.survey_global_distance <= 0.0:
        raise SystemExit("--survey-global-distance must be positive")
    if args.survey_global_elevation_deg is not None and not -89.0 <= args.survey_global_elevation_deg <= 89.0:
        raise SystemExit("--survey-global-elevation-deg must be in [-89, 89]")
    if args.replay_max_joint_step_rad <= 0.0:
        raise SystemExit("--replay-max-joint-step-rad must be positive")
    if args.replay_interpolated_frame_duration <= 0.0:
        raise SystemExit("--replay-interpolated-frame-duration must be positive")
    if args.replay_playback_speed <= 0.0:
        raise SystemExit("--replay-playback-speed must be positive")
    if args.replay_rrt_max_nodes <= 0:
        raise SystemExit("--replay-rrt-max-nodes must be positive")
    if args.replay_rrt_step_rad <= 0.0:
        raise SystemExit("--replay-rrt-step-rad must be positive")
    if args.replay_rrt_attempt_count <= 0:
        raise SystemExit("--replay-rrt-attempt-count must be positive")
    if not 0.0 <= args.replay_rrt_goal_sample_rate <= 1.0:
        raise SystemExit("--replay-rrt-goal-sample-rate must be in [0, 1]")
    if args.replay_recorded_waypoint_window < 0:
        raise SystemExit("--replay-recorded-waypoint-window must be non-negative")
    if args.replay_shortcut_passes < 0:
        raise SystemExit("--replay-shortcut-passes must be non-negative")
    if args.replay_heavy_rrt_max_nodes <= 0:
        raise SystemExit("--replay-heavy-rrt-max-nodes must be positive")
    if args.replay_heavy_rrt_step_rad <= 0.0:
        raise SystemExit("--replay-heavy-rrt-step-rad must be positive")
    if not 0.0 <= args.replay_heavy_rrt_goal_sample_rate <= 1.0:
        raise SystemExit("--replay-heavy-rrt-goal-sample-rate must be in [0, 1]")

    manifest = _load_or_build_manifest(args)
    print(f"manifest={manifest['manifest_path']}")
    print(json.dumps({"status": manifest["status"], "frame_count": manifest["frame_count"]}, ensure_ascii=False))
    if args.manifest_only:
        return

    frames = manifest.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise SystemExit("Replay manifest contains no frames with robot qpos.")
    if find_spec("mujoco") is None or find_spec("mujoco.viewer") is None:
        raise SystemExit("MuJoCo viewer is not available. Run inside the mujoco-curobo conda environment.")

    import mujoco  # type: ignore[import-not-found]
    import mujoco.viewer  # type: ignore[import-not-found]

    run_dir = _resolve_run_dir(manifest, args.repo_root)
    model_path = _resolve_required_path(
        manifest.get("scene_model_path"),
        repo_root=args.repo_root,
        run_dir=run_dir,
        description="scene_model_path",
    )
    layout_path = _resolve_required_path(
        manifest.get("layout_snapshot_path"),
        repo_root=args.repo_root,
        run_dir=run_dir,
        description="layout_snapshot_path",
    )

    if args.view == "both":
        _run_split_viewer(
            mujoco,
            manifest=manifest,
            frames=frames,
            model_path=model_path,
            layout_path=layout_path,
            camera_name=str(manifest.get("camera_name") or "wrist"),
            step_by_step=args.step_by_step,
            frame_duration_s=args.frame_duration,
            poll_seconds=args.poll_seconds,
            global_fovy_deg=args.global_fovy_deg,
            global_lookat=args.global_lookat,
            global_distance=args.global_distance,
            global_elevation_deg=args.global_elevation_deg,
            survey_global_fovy_deg=args.survey_global_fovy_deg,
            survey_global_lookat=args.survey_global_lookat,
            survey_global_distance=args.survey_global_distance,
            survey_global_azimuth_deg=args.survey_global_azimuth_deg,
            survey_global_elevation_deg=args.survey_global_elevation_deg,
            smooth_replay=args.smooth_replay,
            replay_max_joint_step_rad=args.replay_max_joint_step_rad,
            replay_interpolated_frame_duration_s=args.replay_interpolated_frame_duration,
            replay_playback_speed=args.replay_playback_speed,
            replay_rrt_max_nodes=args.replay_rrt_max_nodes,
            replay_rrt_step_rad=args.replay_rrt_step_rad,
            replay_rrt_goal_sample_rate=args.replay_rrt_goal_sample_rate,
            replay_rrt_seed=args.replay_rrt_seed,
            replay_rrt_attempt_count=args.replay_rrt_attempt_count,
            replay_recorded_waypoint_window=args.replay_recorded_waypoint_window,
            replay_shortcut_passes=args.replay_shortcut_passes,
            replay_heavy_connector=args.replay_heavy_connector,
            replay_heavy_rrt_max_nodes=args.replay_heavy_rrt_max_nodes,
            replay_heavy_rrt_step_rad=args.replay_heavy_rrt_step_rad,
            replay_heavy_rrt_goal_sample_rate=args.replay_heavy_rrt_goal_sample_rate,
            replay_motion_failure_policy=args.replay_motion_failure_policy,
            loop=args.loop,
            wait_for_close=args.wait_for_close,
        )
        return

    scenes = {
        mode: _load_replay_scene(
            mujoco,
            model_path=model_path,
            layout_path=layout_path,
            camera_name=str(manifest.get("camera_name") or "wrist"),
        )
        for mode in _viewer_modes(args.view)
    }
    if args.smooth_replay:
        first_scene = next(iter(scenes.values()))
        frames, smoothing_summary = _build_collision_checked_replay_motion(
            mujoco,
            first_scene.model,
            first_scene.data,
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
        _print_replay_motion_summary(smoothing_summary)
    keymap = _resolve_keymap(mujoco)
    control = _ReplayControl(paused=args.step_by_step)
    launch_kwargs = {}
    if _launch_passive_supports_key_callback(mujoco.viewer.launch_passive):
        launch_kwargs["key_callback"] = lambda key: control.handle_key(int(key), keymap)

    current_index = 0
    motion_target_index: int | None = None
    _apply_frame_to_all(mujoco, scenes, frames[current_index])
    last_switch = time.monotonic()
    advance_limit_warning_printed = False
    finished = False

    with ExitStack() as stack:
        viewers = {
            mode: stack.enter_context(mujoco.viewer.launch_passive(scene.model, scene.data, **launch_kwargs))
            for mode, scene in scenes.items()
        }
        for mode, viewer in viewers.items():
            scene = scenes[mode]
            _configure_viewer_camera(
                viewer,
                mujoco,
                scene.model,
                scene.data,
                mode,
                manifest,
                frames[current_index],
                scene.camera_id,
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

        while all(viewer.is_running() for viewer in viewers.values()):
            if control.consume_reset_request():
                current_index = 0
                finished = False
                last_switch = time.monotonic()
                _apply_frame_to_all(mujoco, scenes, frames[current_index])
            else:
                paused, _ = control.snapshot()
                should_advance = False
                if not finished:
                    now = time.monotonic()
                    force_first_step = False
                    if motion_target_index is not None:
                        should_advance = True
                    elif paused:
                        if control.consume_frame_request():
                            motion_target_index = _next_replay_keyframe_index(
                                frames,
                                current_index,
                                loop=args.loop,
                            )
                            should_advance = motion_target_index is not None
                            force_first_step = should_advance
                            last_switch = now
                    else:
                        should_advance = True
                if should_advance:
                    advance = _advance_replay_index(
                        frames,
                        current_index=current_index,
                        finished=finished,
                        motion_target_index=motion_target_index,
                        elapsed_s=0.0 if force_first_step else now - last_switch,
                        default_duration_s=args.frame_duration,
                        playback_speed=args.replay_playback_speed,
                        loop=args.loop,
                        force_first_step=force_first_step,
                    )
                    current_index = advance.current_index
                    finished = advance.finished
                    motion_target_index = advance.motion_target_index
                    if advance.reached_indices and (not finished or args.loop):
                        _apply_frame_to_all(mujoco, scenes, frames[current_index])
                        for reached_index in advance.reached_indices:
                            if not _is_generated_motion_frame(frames[reached_index]):
                                _print_frame(frames[reached_index])
                    if advance.reached_indices:
                        last_switch = now - advance.elapsed_remainder_s
                    if advance.advance_limit_reached and not advance_limit_warning_printed:
                        _print_replay_timing_warning(max_advances=len(frames))
                        advance_limit_warning_printed = True

            for mode, viewer in viewers.items():
                scene = scenes[mode]
                _configure_viewer_camera(
                    viewer,
                    mujoco,
                    scene.model,
                    scene.data,
                    mode,
                    manifest,
                    frames[current_index],
                    scene.camera_id,
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
                _render_overlay(
                    viewer,
                    mujoco,
                    mode=mode,
                    frame=frames[current_index],
                    frame_index=_capture_frame_display_position(frames, current_index)[0] - 1,
                    frame_count=_capture_frame_display_position(frames, current_index)[1],
                    control=control,
                    finished=finished,
                )
                viewer.sync()

            if finished and not args.wait_for_close:
                break
            time.sleep(args.poll_seconds)


def _load_or_build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if args.manifest is not None:
        return load_task1_replay_manifest(args.manifest)

    run_dir = args.run_dir or find_latest_task1_run(args.output_dir)
    manifest_path = replay_manifest_path_for_run(run_dir)
    phases = tuple(item.strip() for item in args.phases.split(",") if item.strip())
    if args.rebuild_manifest or not manifest_path.exists():
        return write_task1_replay_manifest(run_dir, phases=phases, manifest_path=manifest_path)
    try:
        return load_task1_replay_manifest(manifest_path)
    except ValueError as exc:
        print(f"rebuilding replay manifest because existing manifest could not be loaded: {exc}", file=sys.stderr)
        return write_task1_replay_manifest(run_dir, phases=phases, manifest_path=manifest_path)


def _viewer_modes(view: str) -> tuple[str, ...]:
    if view == "both":
        return ("global", "wrist")
    return (view,)


def _run_split_viewer(
    mujoco: Any,
    *,
    manifest: dict[str, Any],
    frames: list[dict[str, Any]],
    model_path: Path,
    layout_path: Path,
    camera_name: str,
    step_by_step: bool,
    frame_duration_s: float,
    poll_seconds: float,
    global_fovy_deg: float,
    global_lookat: tuple[float, float, float] | None,
    global_distance: float | None,
    global_elevation_deg: float | None,
    survey_global_fovy_deg: float | None,
    survey_global_lookat: tuple[float, float, float] | None,
    survey_global_distance: float | None,
    survey_global_azimuth_deg: float | None,
    survey_global_elevation_deg: float | None,
    smooth_replay: bool,
    replay_max_joint_step_rad: float,
    replay_interpolated_frame_duration_s: float,
    replay_playback_speed: float,
    replay_rrt_max_nodes: int,
    replay_rrt_step_rad: float,
    replay_rrt_goal_sample_rate: float,
    replay_rrt_seed: int,
    replay_rrt_attempt_count: int,
    replay_recorded_waypoint_window: int,
    replay_shortcut_passes: int,
    replay_heavy_connector: bool,
    replay_heavy_rrt_max_nodes: int,
    replay_heavy_rrt_step_rad: float,
    replay_heavy_rrt_goal_sample_rate: float,
    replay_motion_failure_policy: str,
    loop: bool,
    wait_for_close: bool,
) -> None:
    try:
        from mujoco.glfw import glfw  # type: ignore[import-not-found]
    except Exception as exc:
        raise SystemExit("MuJoCo GLFW bindings are not available in this environment.") from exc

    if not glfw.init():
        raise SystemExit("GLFW could not initialize a window for task1 split replay.")

    window = None
    scene = None
    context = None
    try:
        window = glfw.create_window(1600, 900, "Task1 replay: global | wrist", None, None)
        if window is None:
            raise SystemExit("GLFW could not create a task1 split replay window.")
        glfw.make_context_current(window)
        glfw.swap_interval(1)

        replay_scene = _load_replay_scene(
            mujoco,
            model_path=model_path,
            layout_path=layout_path,
            camera_name=camera_name,
        )
        model = replay_scene.model
        data = replay_scene.data
        wrist_data = mujoco.MjData(model)
        _apply_home_keyframe(mujoco, model, wrist_data)
        scene = mujoco.MjvScene(model, maxgeom=10000)
        context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
        options = mujoco.MjvOption()
        global_camera = mujoco.MjvCamera()
        wrist_camera = mujoco.MjvCamera()
        wrist_camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        wrist_camera.fixedcamid = replay_scene.camera_id
        if smooth_replay:
            frames, smoothing_summary = _build_collision_checked_replay_motion(
                mujoco,
                model,
                data,
                frames,
                max_joint_step_rad=replay_max_joint_step_rad,
                interpolated_frame_duration_s=replay_interpolated_frame_duration_s,
                rrt_max_nodes=replay_rrt_max_nodes,
                rrt_step_rad=replay_rrt_step_rad,
                rrt_goal_sample_rate=replay_rrt_goal_sample_rate,
                rrt_seed=replay_rrt_seed,
                rrt_attempt_count=replay_rrt_attempt_count,
                recorded_waypoint_window=replay_recorded_waypoint_window,
                shortcut_passes=replay_shortcut_passes,
                heavy_connector=replay_heavy_connector,
                heavy_rrt_max_nodes=replay_heavy_rrt_max_nodes,
                heavy_rrt_step_rad=replay_heavy_rrt_step_rad,
                heavy_rrt_goal_sample_rate=replay_heavy_rrt_goal_sample_rate,
                motion_failure_policy=replay_motion_failure_policy,
            )
            _print_replay_motion_summary(smoothing_summary)

        keymap = _GuiKeyMap(space=int(glfw.KEY_SPACE), next_frame=int(glfw.KEY_F9), reset=int(glfw.KEY_F12))
        control = _ReplayControl(paused=step_by_step)

        def _on_key(_window: Any, key: int, _scancode: int, action: int, _mods: int) -> None:
            if action == glfw.PRESS:
                control.handle_key(int(key), keymap)

        glfw.set_key_callback(window, _on_key)
        current_index = 0
        wrist_index = _keyframe_index_at_or_before(frames, current_index)
        motion_target_index: int | None = None
        _apply_frame(mujoco, model, data, frames[current_index])
        _apply_frame(mujoco, model, wrist_data, frames[wrist_index])
        last_switch = time.monotonic()
        advance_limit_warning_printed = False
        finished = False

        while not glfw.window_should_close(window):
            glfw.poll_events()
            if control.consume_reset_request():
                current_index = 0
                wrist_index = _keyframe_index_at_or_before(frames, current_index)
                motion_target_index = None
                finished = False
                last_switch = time.monotonic()
                _apply_frame(mujoco, model, data, frames[current_index])
                _apply_frame(mujoco, model, wrist_data, frames[wrist_index])
            else:
                paused, _ = control.snapshot()
                should_advance = False
                if not finished:
                    now = time.monotonic()
                    force_first_step = False
                    if motion_target_index is not None:
                        should_advance = True
                    elif paused:
                        if control.consume_frame_request():
                            motion_target_index = _next_replay_keyframe_index(
                                frames,
                                current_index,
                                loop=loop,
                            )
                            should_advance = motion_target_index is not None
                            force_first_step = should_advance
                            last_switch = now
                    else:
                        should_advance = True
                if should_advance:
                    advance = _advance_replay_index(
                        frames,
                        current_index=current_index,
                        finished=finished,
                        motion_target_index=motion_target_index,
                        elapsed_s=0.0 if force_first_step else now - last_switch,
                        default_duration_s=frame_duration_s,
                        playback_speed=replay_playback_speed,
                        loop=loop,
                        force_first_step=force_first_step,
                    )
                    current_index = advance.current_index
                    finished = advance.finished
                    motion_target_index = advance.motion_target_index
                    if advance.reached_indices and (not finished or loop):
                        _apply_frame(mujoco, model, data, frames[current_index])
                        next_wrist_index = _keyframe_index_at_or_before(frames, current_index)
                        if next_wrist_index != wrist_index:
                            wrist_index = next_wrist_index
                            _apply_frame(mujoco, model, wrist_data, frames[wrist_index])
                        for reached_index in advance.reached_indices:
                            if not _is_generated_motion_frame(frames[reached_index]):
                                _print_frame(frames[reached_index])
                    if advance.reached_indices:
                        last_switch = now - advance.elapsed_remainder_s
                    if advance.advance_limit_reached and not advance_limit_warning_printed:
                        _print_replay_timing_warning(max_advances=len(frames))
                        advance_limit_warning_printed = True
                    if motion_target_index is None and wrist_index != _keyframe_index_at_or_before(frames, current_index):
                        wrist_index = _keyframe_index_at_or_before(frames, current_index)
                        _apply_frame(mujoco, model, wrist_data, frames[wrist_index])

            width, height = glfw.get_framebuffer_size(window)
            if width <= 0 or height <= 0:
                time.sleep(poll_seconds)
                continue
            left, divider, right = _split_viewports(mujoco, width=width, height=height)
            full = mujoco.MjrRect(0, 0, int(width), int(height))
            mujoco.mjr_rectangle(full, 0.04, 0.045, 0.05, 1.0)
            _configure_mjv_global_camera(
                mujoco,
                model,
                data,
                global_camera,
                manifest,
                frames[current_index],
                global_lookat=global_lookat,
                global_distance=global_distance,
                global_elevation_deg=global_elevation_deg,
                survey_global_lookat=survey_global_lookat,
                survey_global_distance=survey_global_distance,
                survey_global_azimuth_deg=survey_global_azimuth_deg,
                survey_global_elevation_deg=survey_global_elevation_deg,
            )
            _render_scene_view(
                mujoco,
                model=model,
                data=data,
                options=options,
                scene=scene,
                context=context,
                camera=global_camera,
                viewport=left,
                global_fovy_deg=_global_fovy_for_frame(
                    frames[current_index],
                    global_fovy_deg=global_fovy_deg,
                    survey_global_fovy_deg=survey_global_fovy_deg,
                ),
            )
            mujoco.mjr_rectangle(divider, 0.15, 0.16, 0.17, 1.0)
            _render_scene_view(
                mujoco,
                model=model,
                data=wrist_data,
                options=options,
                scene=scene,
                context=context,
                camera=wrist_camera,
                viewport=right,
            )
            _render_split_overlay(
                mujoco,
                context=context,
                left=left,
                right=right,
                frame=frames[current_index],
                frame_index=_capture_frame_display_position(frames, current_index)[0] - 1,
                frame_count=_capture_frame_display_position(frames, current_index)[1],
                wrist_frame=frames[wrist_index],
                wrist_frame_index=_capture_frame_display_position(frames, wrist_index)[0] - 1,
                control=control,
                finished=finished,
            )
            glfw.swap_buffers(window)
            if finished and not wait_for_close:
                break
            time.sleep(poll_seconds)
    finally:
        if context is not None and hasattr(mujoco, "mjr_freeContext"):
            mujoco.mjr_freeContext(context)
        if scene is not None and hasattr(mujoco, "mjv_freeScene"):
            mujoco.mjv_freeScene(scene)
        if window is not None:
            glfw.destroy_window(window)
        glfw.terminate()


def _resolve_run_dir(manifest: dict[str, Any], repo_root: Path) -> Path:
    value = manifest.get("task1_run_dir")
    if not value:
        raise SystemExit("Replay manifest is missing task1_run_dir.")
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidate = repo_root / path
    return candidate if candidate.exists() else path


def _resolve_required_path(value: Any, *, repo_root: Path, run_dir: Path, description: str) -> Path:
    if value is None:
        raise SystemExit(f"Replay manifest is missing {description}.")
    raw = Path(str(value))
    candidates = [raw] if raw.is_absolute() else [repo_root / raw, run_dir / raw, raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    candidate_text = ", ".join(str(candidate) for candidate in candidates)
    raise SystemExit(f"Could not resolve {description}: {value}. Tried: {candidate_text}")


def _apply_home_keyframe(mujoco: Any, model: Any, data: Any) -> None:
    for name in ("gen3_home", "home"):
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
            return
    mujoco.mj_forward(model, data)


def _load_replay_scene(mujoco: Any, *, model_path: Path, layout_path: Path, camera_name: str) -> _ReplayScene:
    model = mujoco.MjModel.from_xml_path(str(model_path.resolve()))
    data = mujoco.MjData(model)
    _apply_home_keyframe(mujoco, model, data)
    apply_task1_layout_to_mujoco(mujoco, model, data, layout_path)
    camera_id = _resolve_camera_id(mujoco, model, camera_name)
    return _ReplayScene(model=model, data=data, camera_id=camera_id)


def _apply_frame_to_all(mujoco: Any, scenes: dict[str, _ReplayScene], frame: dict[str, Any]) -> None:
    for scene in scenes.values():
        _apply_frame(mujoco, scene.model, scene.data, frame)


def _apply_frame(mujoco: Any, model: Any, data: Any, frame: dict[str, Any]) -> None:
    qpos = frame.get("qpos")
    if not isinstance(qpos, list) or not qpos:
        raise SystemExit(f"Replay frame is missing qpos: {frame.get('frame_id')}")
    _apply_qpos(mujoco, model, data, [float(value) for value in qpos])


def _apply_qpos(mujoco: Any, model: Any, data: Any, qpos: list[float]) -> None:
    count = min(len(qpos), int(model.nq))
    for index in range(count):
        data.qpos[index] = float(qpos[index])
    for index in range(int(model.nv)):
        data.qvel[index] = 0.0
    ctrl_count = min(len(qpos), int(model.nu))
    for index in range(ctrl_count):
        data.ctrl[index] = float(qpos[index])
    mujoco.mj_forward(model, data)


def _build_collision_checked_replay_motion(
    mujoco: Any,
    model: Any,
    data: Any,
    frames: list[dict[str, Any]],
    *,
    max_joint_step_rad: float,
    interpolated_frame_duration_s: float,
    rrt_max_nodes: int = DEFAULT_REPLAY_RRT_MAX_NODES,
    rrt_step_rad: float = DEFAULT_REPLAY_RRT_STEP_RAD,
    rrt_goal_sample_rate: float = DEFAULT_REPLAY_RRT_GOAL_SAMPLE_RATE,
    rrt_seed: int = DEFAULT_REPLAY_RRT_SEED,
    rrt_attempt_count: int = DEFAULT_REPLAY_RRT_ATTEMPT_COUNT,
    recorded_waypoint_window: int = DEFAULT_REPLAY_RECORDED_WAYPOINT_WINDOW,
    shortcut_passes: int = DEFAULT_REPLAY_SHORTCUT_PASSES,
    heavy_connector: bool = DEFAULT_REPLAY_HEAVY_CONNECTOR,
    heavy_rrt_max_nodes: int = DEFAULT_REPLAY_HEAVY_RRT_MAX_NODES,
    heavy_rrt_step_rad: float = DEFAULT_REPLAY_HEAVY_RRT_STEP_RAD,
    heavy_rrt_goal_sample_rate: float = DEFAULT_REPLAY_HEAVY_RRT_GOAL_SAMPLE_RATE,
    motion_failure_policy: str = "keyframe-cut",
) -> tuple[list[dict[str, Any]], _ReplayMotionSummary]:
    if len(frames) < 2:
        return list(frames), _ReplayMotionSummary(
            input_frame_count=len(frames),
            display_frame_count=len(frames),
            interpolated_frame_count=0,
            collision_checked_qpos_count=len(frames),
            max_joint_step_rad=max_joint_step_rad,
            continuous_joint_adjusted_keyframe_count=0,
            continuous_joint_adjusted_joint_count=0,
            continuous_joint_max_adjustment_rad=0.0,
            planned_boundary_connector_count=0,
            waypoint_candidate_count=0,
            waypoint_connector_count=0,
            coordinate_sweep_connector_count=0,
            path_shortcut_passes=shortcut_passes,
            path_shortcut_count=0,
            path_shortcut_collision_check_count=0,
            heavy_rrt_connector_count=0,
            heavy_rrt_max_nodes=heavy_rrt_max_nodes,
            heavy_rrt_step_rad=heavy_rrt_step_rad,
            direct_segment_count=0,
            rrt_segment_count=0,
            rrt_step_rad=rrt_step_rad,
            keyframe_cut_count=0,
            failed_connector_cut_count=0,
        )
    original_qpos = [float(value) for value in data.qpos]
    original_qvel = [float(value) for value in data.qvel]
    original_ctrl = [float(value) for value in data.ctrl]
    display_frames: list[dict[str, Any]] = []
    interpolated_count = 0
    collision_checked_count = 0
    direct_segment_count = 0
    rrt_segment_count = 0
    keyframe_cut_count = 0
    failed_connector_cut_count = 0
    planned_boundary_connector_count = 0
    waypoint_connector_count = 0
    coordinate_sweep_connector_count = 0
    path_shortcut_count = 0
    path_shortcut_collision_check_count = 0
    heavy_rrt_connector_count = 0
    keyframe_qpos = _continuous_replay_keyframe_qpos(model, frames)
    waypoint_candidates, waypoint_validation_checks = _replay_waypoint_candidates(
        mujoco,
        model,
        data,
        neutral_qpos=original_qpos[: int(model.nq)],
    )
    collision_checked_count += waypoint_validation_checks
    max_waypoint_candidate_count = len(waypoint_candidates)
    continuous_joint_adjusted_keyframe_count = sum(1 for item in keyframe_qpos if item.adjusted_joint_count > 0)
    continuous_joint_adjusted_joint_count = sum(item.adjusted_joint_count for item in keyframe_qpos)
    continuous_joint_max_adjustment_rad = max((item.max_adjustment_rad for item in keyframe_qpos), default=0.0)
    try:
        for frame_index, frame in enumerate(frames):
            start_info = keyframe_qpos[frame_index]
            start_qpos = start_info.qpos
            collision_checked_count += 1
            _require_collision_free_qpos(
                mujoco,
                model,
                data,
                start_qpos,
                frame_id=str(frame.get("frame_id") or frame_index),
                segment_id=f"keyframe/{frame_index:04d}",
            )
            display_frames.append(
                _key_display_frame(
                    frame,
                    keyframe_index=frame_index,
                    qpos=start_qpos,
                    adjusted_joint_count=start_info.adjusted_joint_count,
                    max_adjustment_rad=start_info.max_adjustment_rad,
                )
            )
            if frame_index + 1 >= len(frames):
                continue

            next_frame = frames[frame_index + 1]
            should_smooth, cut_reason = _should_smooth_replay_transition(frame, next_frame)
            if not should_smooth:
                display_frames[-1]["replay_cut_after"] = True
                display_frames[-1]["replay_cut_reason"] = cut_reason
                keyframe_cut_count += 1
                continue
            if cut_reason is not None:
                display_frames[-1]["replay_planned_boundary"] = cut_reason
                planned_boundary_connector_count += 1

            end_qpos = keyframe_qpos[frame_index + 1].qpos
            segment_waypoint_candidates = _segment_replay_waypoint_candidates(
                frames,
                keyframe_qpos,
                frame_index=frame_index,
                neutral_waypoints=waypoint_candidates,
                recorded_waypoint_window=recorded_waypoint_window,
            )
            max_waypoint_candidate_count = max(max_waypoint_candidate_count, len(segment_waypoint_candidates))
            segment_id = (
                f"{frame.get('frame_id') or frame_index} -> "
                f"{next_frame.get('frame_id') or frame_index + 1}"
            )
            try:
                segment_plan = _plan_replay_motion_segment(
                    mujoco,
                    model,
                    data,
                    start_qpos,
                    end_qpos,
                    max_joint_step_rad=max_joint_step_rad,
                    rrt_max_nodes=rrt_max_nodes,
                    rrt_step_rad=rrt_step_rad,
                    rrt_goal_sample_rate=rrt_goal_sample_rate,
                    rrt_attempt_count=rrt_attempt_count,
                    rng=random.Random(rrt_seed + frame_index),
                    segment_id=segment_id,
                    waypoint_candidates=segment_waypoint_candidates,
                    shortcut_passes=shortcut_passes,
                    heavy_connector=heavy_connector,
                    heavy_rrt_max_nodes=heavy_rrt_max_nodes,
                    heavy_rrt_step_rad=heavy_rrt_step_rad,
                    heavy_rrt_goal_sample_rate=heavy_rrt_goal_sample_rate,
                )
            except SystemExit as exc:
                if motion_failure_policy != "keyframe-cut":
                    raise
                display_frames[-1]["replay_cut_after"] = True
                display_frames[-1]["replay_cut_reason"] = "connector_not_found"
                display_frames[-1]["replay_connector_error"] = str(exc)
                keyframe_cut_count += 1
                failed_connector_cut_count += 1
                continue
            collision_checked_count += segment_plan.collision_check_count
            direct_segment_count += segment_plan.direct_edge_count
            rrt_segment_count += segment_plan.rrt_edge_count
            waypoint_connector_count += segment_plan.waypoint_count
            coordinate_sweep_connector_count += segment_plan.coordinate_sweep_count
            path_shortcut_count += segment_plan.shortcut_count
            path_shortcut_collision_check_count += segment_plan.shortcut_collision_check_count
            heavy_rrt_connector_count += segment_plan.heavy_rrt_count
            display_frames[-1]["replay_connector_strategy"] = segment_plan.strategy
            for interpolation_index, qpos in enumerate(
                segment_plan.qpos_samples,
                start=1,
            ):
                display_frames.append(
                    _interpolated_display_frame(
                        from_frame=frame,
                        to_frame=next_frame,
                        from_index=frame_index,
                        interpolation_index=interpolation_index,
                        qpos=qpos,
                        duration_s=interpolated_frame_duration_s,
                    )
                )
                interpolated_count += 1
    finally:
        _apply_qpos(mujoco, model, data, original_qpos)
        for index, value in enumerate(original_qvel[: int(model.nv)]):
            data.qvel[index] = value
        for index, value in enumerate(original_ctrl[: int(model.nu)]):
            data.ctrl[index] = value
        mujoco.mj_forward(model, data)

    summary = _ReplayMotionSummary(
        input_frame_count=len(frames),
        display_frame_count=len(display_frames),
        interpolated_frame_count=interpolated_count,
        collision_checked_qpos_count=collision_checked_count,
        max_joint_step_rad=max_joint_step_rad,
        continuous_joint_adjusted_keyframe_count=continuous_joint_adjusted_keyframe_count,
        continuous_joint_adjusted_joint_count=continuous_joint_adjusted_joint_count,
        continuous_joint_max_adjustment_rad=_round_replay_value(continuous_joint_max_adjustment_rad) or 0.0,
        planned_boundary_connector_count=planned_boundary_connector_count,
        waypoint_candidate_count=max_waypoint_candidate_count,
        waypoint_connector_count=waypoint_connector_count,
        coordinate_sweep_connector_count=coordinate_sweep_connector_count,
        path_shortcut_passes=shortcut_passes,
        path_shortcut_count=path_shortcut_count,
        path_shortcut_collision_check_count=path_shortcut_collision_check_count,
        heavy_rrt_connector_count=heavy_rrt_connector_count,
        heavy_rrt_max_nodes=heavy_rrt_max_nodes,
        heavy_rrt_step_rad=heavy_rrt_step_rad,
        direct_segment_count=direct_segment_count,
        rrt_segment_count=rrt_segment_count,
        rrt_step_rad=rrt_step_rad,
        keyframe_cut_count=keyframe_cut_count,
        failed_connector_cut_count=failed_connector_cut_count,
    )
    return display_frames, summary


def _qpos_for_replay_motion_frame(frame: dict[str, Any], *, nq: int) -> list[float]:
    qpos = frame.get("qpos")
    if not isinstance(qpos, list) or len(qpos) < nq:
        raise SystemExit(f"Replay smoothing requires full qpos for frame: {frame.get('frame_id')}")
    try:
        return [float(value) for value in qpos[:nq]]
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Replay smoothing found non-numeric qpos in frame: {frame.get('frame_id')}") from exc


def _continuous_replay_keyframe_qpos(model: Any, frames: list[dict[str, Any]]) -> list[_ReplayKeyframeQpos]:
    nq = int(model.nq)
    keyframes: list[_ReplayKeyframeQpos] = []
    previous_qpos: list[float] | None = None
    for frame in frames:
        raw_qpos = _qpos_for_replay_motion_frame(frame, nq=nq)
        if previous_qpos is None:
            keyframes.append(_ReplayKeyframeQpos(qpos=raw_qpos))
            previous_qpos = raw_qpos
            continue
        adjusted_qpos = _normalised_replay_goal_qpos(model, previous_qpos, raw_qpos)
        adjustments = [abs(float(adjusted_qpos[index]) - float(raw_qpos[index])) for index in range(len(adjusted_qpos))]
        adjusted_joint_count = sum(1 for value in adjustments if value > 1e-9)
        max_adjustment_rad = max(adjustments, default=0.0)
        keyframes.append(
            _ReplayKeyframeQpos(
                qpos=adjusted_qpos,
                adjusted_joint_count=adjusted_joint_count,
                max_adjustment_rad=max_adjustment_rad,
            )
        )
        previous_qpos = adjusted_qpos
    return keyframes


def _replay_waypoint_candidates(
    mujoco: Any,
    model: Any,
    data: Any,
    *,
    neutral_qpos: list[float],
) -> tuple[list[_ReplayWaypointCandidate], int]:
    qpos = [float(value) for value in neutral_qpos[: int(model.nq)]]
    if not qpos:
        return [], 0
    report = _replay_collision_report(mujoco, model, data, qpos)
    if not report["collision_free"]:
        return [], 1
    return [_ReplayWaypointCandidate(name="neutral_home", qpos=qpos, source="loaded_model_home_qpos")], 1


def _segment_replay_waypoint_candidates(
    frames: list[dict[str, Any]],
    keyframe_qpos: list[_ReplayKeyframeQpos],
    *,
    frame_index: int,
    neutral_waypoints: list[_ReplayWaypointCandidate] | tuple[_ReplayWaypointCandidate, ...],
    recorded_waypoint_window: int,
) -> list[_ReplayWaypointCandidate]:
    candidates: list[_ReplayWaypointCandidate] = list(neutral_waypoints)
    if recorded_waypoint_window <= 0:
        return _dedupe_replay_waypoints(candidates)

    phase = frames[frame_index].get("phase") if 0 <= frame_index < len(frames) else None
    next_phase = frames[frame_index + 1].get("phase") if frame_index + 1 < len(frames) else None
    for offset in _recorded_waypoint_offsets(recorded_waypoint_window):
        index = frame_index + offset
        if index < 0 or index >= len(keyframe_qpos):
            continue
        if index in {frame_index, frame_index + 1}:
            continue
        if frames[index].get("phase") not in {phase, next_phase}:
            continue
        candidates.append(
            _ReplayWaypointCandidate(
                name=f"recorded_keyframe_{index:04d}",
                qpos=keyframe_qpos[index].qpos,
                source=str(frames[index].get("frame_id") or index),
            )
        )
    return _dedupe_replay_waypoints(candidates)


def _recorded_waypoint_offsets(window: int) -> list[int]:
    offsets: list[int] = []
    for distance in range(1, window + 1):
        offsets.extend((-distance, 1 + distance))
    return offsets


def _dedupe_replay_waypoints(candidates: list[_ReplayWaypointCandidate]) -> list[_ReplayWaypointCandidate]:
    deduped: list[_ReplayWaypointCandidate] = []
    seen: set[tuple[float, ...]] = set()
    for candidate in candidates:
        key = tuple(round(float(value), 9) for value in candidate.qpos)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _should_smooth_replay_transition(from_frame: dict[str, Any], to_frame: dict[str, Any]) -> tuple[bool, str | None]:
    from_key = _replay_motion_group_key(from_frame)
    to_key = _replay_motion_group_key(to_frame)
    if from_key is not None and from_key == to_key:
        return True, None
    from_phase = str(from_frame.get("phase") or "")
    to_phase = str(to_frame.get("phase") or "")
    if from_phase == to_phase and from_phase in {"row", "final"}:
        return True, "target_group_boundary"
    if from_phase == "row" and to_phase == "final":
        return True, "row_to_final_boundary"
    if from_frame.get("phase") != to_frame.get("phase"):
        return False, "phase_boundary"
    return False, "target_group_boundary"


def _replay_motion_group_key(frame: dict[str, Any]) -> tuple[str, str] | None:
    phase = str(frame.get("phase") or "")
    if phase == "survey":
        return ("survey", "survey")
    if phase == "row":
        target_id = frame.get("target_id")
        return ("row", str(target_id)) if target_id else None
    if phase == "final":
        target_id = frame.get("target_id")
        return ("final", str(target_id)) if target_id else None
    return None


def _interpolated_replay_qpos(
    model: Any,
    start_qpos: list[float],
    end_qpos: list[float],
    *,
    max_joint_step_rad: float,
) -> list[list[float]]:
    count = min(len(start_qpos), len(end_qpos), int(model.nq))
    delta = [float(end_qpos[index]) - float(start_qpos[index]) for index in range(count)]
    max_abs_delta = max((abs(value) for value in delta), default=0.0)
    step_count = max(1, int(math.ceil(max_abs_delta / max_joint_step_rad)))
    if step_count <= 1:
        return []
    return [
        [start_qpos[index] + delta[index] * (step_index / step_count) for index in range(len(delta))]
        for step_index in range(1, step_count)
    ]


def _plan_replay_motion_segment(
    mujoco: Any,
    model: Any,
    data: Any,
    start_qpos: list[float],
    raw_end_qpos: list[float],
    *,
    max_joint_step_rad: float,
    rrt_max_nodes: int,
    rrt_step_rad: float,
    rrt_goal_sample_rate: float,
    rrt_attempt_count: int,
    rng: random.Random,
    segment_id: str,
    waypoint_candidates: list[_ReplayWaypointCandidate] | tuple[_ReplayWaypointCandidate, ...] = (),
    shortcut_passes: int = DEFAULT_REPLAY_SHORTCUT_PASSES,
    heavy_connector: bool = DEFAULT_REPLAY_HEAVY_CONNECTOR,
    heavy_rrt_max_nodes: int = DEFAULT_REPLAY_HEAVY_RRT_MAX_NODES,
    heavy_rrt_step_rad: float = DEFAULT_REPLAY_HEAVY_RRT_STEP_RAD,
    heavy_rrt_goal_sample_rate: float = DEFAULT_REPLAY_HEAVY_RRT_GOAL_SAMPLE_RATE,
) -> _ReplaySegmentPlan:
    end_qpos = _normalised_replay_goal_qpos(model, start_qpos, raw_end_qpos)
    edge_plan, edge_checks = _plan_replay_edge_with_rrt(
        mujoco,
        model,
        data,
        start_qpos,
        end_qpos,
        max_joint_step_rad=max_joint_step_rad,
        rrt_max_nodes=rrt_max_nodes,
        rrt_step_rad=rrt_step_rad,
        rrt_goal_sample_rate=rrt_goal_sample_rate,
        rrt_attempt_count=rrt_attempt_count,
        rng=rng,
        shortcut_passes=shortcut_passes,
        heavy_connector=heavy_connector,
        heavy_rrt_max_nodes=heavy_rrt_max_nodes,
        heavy_rrt_step_rad=heavy_rrt_step_rad,
        heavy_rrt_goal_sample_rate=heavy_rrt_goal_sample_rate,
    )
    if edge_plan is not None:
        return edge_plan

    waypoint_plan, waypoint_checks = _plan_replay_segment_via_waypoint(
        mujoco,
        model,
        data,
        start_qpos,
        end_qpos,
        waypoint_candidates=waypoint_candidates,
        max_joint_step_rad=max_joint_step_rad,
        rrt_max_nodes=rrt_max_nodes,
        rrt_step_rad=rrt_step_rad,
        rrt_goal_sample_rate=rrt_goal_sample_rate,
        rrt_attempt_count=rrt_attempt_count,
        rng=rng,
        shortcut_passes=shortcut_passes,
        heavy_connector=heavy_connector,
        heavy_rrt_max_nodes=heavy_rrt_max_nodes,
        heavy_rrt_step_rad=heavy_rrt_step_rad,
        heavy_rrt_goal_sample_rate=heavy_rrt_goal_sample_rate,
    )
    if waypoint_plan is not None:
        return _ReplaySegmentPlan(
            qpos_samples=waypoint_plan.qpos_samples,
            strategy=waypoint_plan.strategy,
            collision_check_count=edge_checks + waypoint_plan.collision_check_count,
            direct_edge_count=waypoint_plan.direct_edge_count,
            rrt_edge_count=waypoint_plan.rrt_edge_count,
            waypoint_count=waypoint_plan.waypoint_count,
            coordinate_sweep_count=waypoint_plan.coordinate_sweep_count,
            shortcut_count=waypoint_plan.shortcut_count,
            shortcut_collision_check_count=waypoint_plan.shortcut_collision_check_count,
            heavy_rrt_count=waypoint_plan.heavy_rrt_count,
        )
    raise SystemExit(
        "Replay smoothing could not find a collision-free connector: "
        f"segment={segment_id}, edge_checks={edge_checks}, waypoint_checks={waypoint_checks}, "
        f"rrt_max_nodes={rrt_max_nodes}"
    )


def _plan_replay_edge_with_rrt(
    mujoco: Any,
    model: Any,
    data: Any,
    start_qpos: list[float],
    end_qpos: list[float],
    *,
    max_joint_step_rad: float,
    rrt_max_nodes: int,
    rrt_step_rad: float,
    rrt_goal_sample_rate: float,
    rrt_attempt_count: int,
    rng: random.Random,
    shortcut_passes: int = DEFAULT_REPLAY_SHORTCUT_PASSES,
    heavy_connector: bool = DEFAULT_REPLAY_HEAVY_CONNECTOR,
    heavy_rrt_max_nodes: int = DEFAULT_REPLAY_HEAVY_RRT_MAX_NODES,
    heavy_rrt_step_rad: float = DEFAULT_REPLAY_HEAVY_RRT_STEP_RAD,
    heavy_rrt_goal_sample_rate: float = DEFAULT_REPLAY_HEAVY_RRT_GOAL_SAMPLE_RATE,
) -> tuple[_ReplaySegmentPlan | None, int]:
    direct_samples = _interpolated_replay_qpos(
        model,
        start_qpos,
        end_qpos,
        max_joint_step_rad=max_joint_step_rad,
    )
    direct_ok, direct_checks = _collision_free_replay_edge(
        mujoco,
        model,
        data,
        start_qpos,
        end_qpos,
        max_joint_step_rad=max_joint_step_rad,
    )
    if direct_ok:
        return (
            _ReplaySegmentPlan(
                qpos_samples=direct_samples,
                strategy="direct",
                collision_check_count=direct_checks,
                direct_edge_count=1,
            ),
            direct_checks,
        )

    coordinate_plan, coordinate_checks = _plan_replay_coordinate_sweep(
        mujoco,
        model,
        data,
        start_qpos,
        end_qpos,
        max_joint_step_rad=max_joint_step_rad,
    )
    if coordinate_plan is not None:
        coordinate_plan = _ReplaySegmentPlan(
            qpos_samples=coordinate_plan.qpos_samples,
            strategy=coordinate_plan.strategy,
            collision_check_count=direct_checks + coordinate_plan.collision_check_count,
            direct_edge_count=coordinate_plan.direct_edge_count,
            coordinate_sweep_count=coordinate_plan.coordinate_sweep_count,
        )
        return coordinate_plan, direct_checks + coordinate_checks

    rrt_samples: list[list[float]] | None = None
    rrt_checks = 0
    for _attempt_index in range(rrt_attempt_count):
        attempt_samples, attempt_checks = _rrt_connect_replay_segment(
            mujoco,
            model,
            data,
            start_qpos,
            end_qpos,
            max_joint_step_rad=max_joint_step_rad,
            rrt_step_rad=rrt_step_rad,
            max_nodes=rrt_max_nodes,
            goal_sample_rate=rrt_goal_sample_rate,
            rng=random.Random(rng.randrange(0, 2**63)),
        )
        rrt_checks += attempt_checks
        if attempt_samples is not None:
            rrt_samples = attempt_samples
            break
    total_checks = direct_checks + coordinate_checks + rrt_checks
    if rrt_samples is None and heavy_connector:
        rrt_samples, heavy_checks = _rrt_connect_replay_segment(
            mujoco,
            model,
            data,
            start_qpos,
            end_qpos,
            max_joint_step_rad=max_joint_step_rad,
            rrt_step_rad=heavy_rrt_step_rad,
            max_nodes=heavy_rrt_max_nodes,
            goal_sample_rate=heavy_rrt_goal_sample_rate,
            rng=random.Random(rng.randrange(0, 2**63)),
        )
        total_checks += heavy_checks
        if rrt_samples is not None:
            shortcut_path, shortcut_checks, shortcut_count = _shortcut_replay_qpos_path(
                mujoco,
                model,
                data,
                [start_qpos, *rrt_samples, end_qpos],
                max_joint_step_rad=max_joint_step_rad,
                shortcut_passes=shortcut_passes,
            )
            total_checks += shortcut_checks
            dense_samples = _densify_replay_qpos_path(
                model,
                shortcut_path,
                max_joint_step_rad=max_joint_step_rad,
            )
            return (
                _ReplaySegmentPlan(
                    qpos_samples=dense_samples,
                    strategy="heavy_rrt_connect+shortcut" if shortcut_count > 0 else "heavy_rrt_connect",
                    collision_check_count=total_checks,
                    rrt_edge_count=1,
                    shortcut_count=shortcut_count,
                    shortcut_collision_check_count=shortcut_checks,
                    heavy_rrt_count=1,
                ),
                total_checks,
            )
    if rrt_samples is None:
        return None, total_checks
    shortcut_path, shortcut_checks, shortcut_count = _shortcut_replay_qpos_path(
        mujoco,
        model,
        data,
        [start_qpos, *rrt_samples, end_qpos],
        max_joint_step_rad=max_joint_step_rad,
        shortcut_passes=shortcut_passes,
    )
    total_checks += shortcut_checks
    dense_samples = _densify_replay_qpos_path(
        model,
        shortcut_path,
        max_joint_step_rad=max_joint_step_rad,
    )
    return (
        _ReplaySegmentPlan(
            qpos_samples=dense_samples,
            strategy="rrt_connect+shortcut" if shortcut_count > 0 else "rrt_connect",
            collision_check_count=total_checks,
            rrt_edge_count=1,
            shortcut_count=shortcut_count,
            shortcut_collision_check_count=shortcut_checks,
        ),
        total_checks,
    )


def _plan_replay_coordinate_sweep(
    mujoco: Any,
    model: Any,
    data: Any,
    start_qpos: list[float],
    end_qpos: list[float],
    *,
    max_joint_step_rad: float,
) -> tuple[_ReplaySegmentPlan | None, int]:
    total_checks = 0
    for order in _coordinate_sweep_orders(start_qpos, end_qpos):
        current = list(start_qpos)
        path = [list(start_qpos)]
        edge_count = 0
        order_checks = 0
        failed = False
        for joint_index in order:
            if abs(float(current[joint_index]) - float(end_qpos[joint_index])) <= 1e-9:
                continue
            next_qpos = list(current)
            next_qpos[joint_index] = float(end_qpos[joint_index])
            edge_ok, checks = _collision_free_replay_edge(
                mujoco,
                model,
                data,
                current,
                next_qpos,
                max_joint_step_rad=max_joint_step_rad,
            )
            order_checks += checks
            if not edge_ok:
                failed = True
                break
            path.append(next_qpos)
            current = next_qpos
            edge_count += 1
        total_checks += order_checks
        if failed:
            continue
        samples = _densify_replay_qpos_path(model, path, max_joint_step_rad=max_joint_step_rad)
        return (
            _ReplaySegmentPlan(
                qpos_samples=samples,
                strategy="coordinate_sweep",
                collision_check_count=order_checks,
                direct_edge_count=edge_count,
                coordinate_sweep_count=1,
            ),
            total_checks,
        )
    return None, total_checks


def _coordinate_sweep_orders(start_qpos: list[float], end_qpos: list[float]) -> list[list[int]]:
    active = [
        index
        for index, (start, end) in enumerate(zip(start_qpos, end_qpos, strict=False))
        if abs(float(end) - float(start)) > 1e-9
    ]
    if not active:
        return []
    candidates = [
        active,
        list(reversed(active)),
        sorted(active, key=lambda index: abs(float(end_qpos[index]) - float(start_qpos[index])), reverse=True),
        sorted(active, key=lambda index: abs(float(end_qpos[index]) - float(start_qpos[index]))),
    ]
    orders: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for order in candidates:
        key = tuple(order)
        if key in seen:
            continue
        seen.add(key)
        orders.append(order)
    return orders


def _plan_replay_segment_via_waypoint(
    mujoco: Any,
    model: Any,
    data: Any,
    start_qpos: list[float],
    end_qpos: list[float],
    *,
    waypoint_candidates: list[_ReplayWaypointCandidate] | tuple[_ReplayWaypointCandidate, ...],
    max_joint_step_rad: float,
    rrt_max_nodes: int,
    rrt_step_rad: float,
    rrt_goal_sample_rate: float,
    rrt_attempt_count: int,
    rng: random.Random,
    shortcut_passes: int,
    heavy_connector: bool,
    heavy_rrt_max_nodes: int,
    heavy_rrt_step_rad: float,
    heavy_rrt_goal_sample_rate: float,
) -> tuple[_ReplaySegmentPlan | None, int]:
    checks = 0
    for waypoint in waypoint_candidates:
        waypoint_qpos = _normalised_replay_goal_qpos(model, start_qpos, waypoint.qpos)
        if _replay_qpos_distance(start_qpos, waypoint_qpos) <= 1e-9:
            continue
        if _replay_qpos_distance(waypoint_qpos, end_qpos) <= 1e-9:
            continue
        first_plan, first_checks = _plan_replay_edge_with_rrt(
            mujoco,
            model,
            data,
            start_qpos,
            waypoint_qpos,
            max_joint_step_rad=max_joint_step_rad,
            rrt_max_nodes=rrt_max_nodes,
            rrt_step_rad=rrt_step_rad,
            rrt_goal_sample_rate=rrt_goal_sample_rate,
            rrt_attempt_count=rrt_attempt_count,
            rng=rng,
            shortcut_passes=shortcut_passes,
            heavy_connector=heavy_connector,
            heavy_rrt_max_nodes=heavy_rrt_max_nodes,
            heavy_rrt_step_rad=heavy_rrt_step_rad,
            heavy_rrt_goal_sample_rate=heavy_rrt_goal_sample_rate,
        )
        checks += first_checks
        if first_plan is None:
            continue
        second_plan, second_checks = _plan_replay_edge_with_rrt(
            mujoco,
            model,
            data,
            waypoint_qpos,
            end_qpos,
            max_joint_step_rad=max_joint_step_rad,
            rrt_max_nodes=rrt_max_nodes,
            rrt_step_rad=rrt_step_rad,
            rrt_goal_sample_rate=rrt_goal_sample_rate,
            rrt_attempt_count=rrt_attempt_count,
            rng=rng,
            shortcut_passes=shortcut_passes,
            heavy_connector=heavy_connector,
            heavy_rrt_max_nodes=heavy_rrt_max_nodes,
            heavy_rrt_step_rad=heavy_rrt_step_rad,
            heavy_rrt_goal_sample_rate=heavy_rrt_goal_sample_rate,
        )
        checks += second_checks
        if second_plan is None:
            continue
        samples = [*first_plan.qpos_samples, waypoint_qpos, *second_plan.qpos_samples]
        return (
            _ReplaySegmentPlan(
                qpos_samples=samples,
                strategy=f"via_waypoint:{waypoint.name}:{first_plan.strategy}+{second_plan.strategy}",
                collision_check_count=checks,
                direct_edge_count=first_plan.direct_edge_count + second_plan.direct_edge_count,
                rrt_edge_count=first_plan.rrt_edge_count + second_plan.rrt_edge_count,
                waypoint_count=1,
                coordinate_sweep_count=first_plan.coordinate_sweep_count + second_plan.coordinate_sweep_count,
                shortcut_count=first_plan.shortcut_count + second_plan.shortcut_count,
                shortcut_collision_check_count=(
                    first_plan.shortcut_collision_check_count + second_plan.shortcut_collision_check_count
                ),
                heavy_rrt_count=first_plan.heavy_rrt_count + second_plan.heavy_rrt_count,
            ),
            checks,
        )
    return None, checks


def _normalised_replay_goal_qpos(model: Any, start_qpos: list[float], end_qpos: list[float]) -> list[float]:
    delta = _joint_space_delta(model, start_qpos, end_qpos)
    return [float(start_qpos[index]) + delta[index] for index in range(len(delta))]


def _collision_free_replay_edge(
    mujoco: Any,
    model: Any,
    data: Any,
    start_qpos: list[float],
    end_qpos: list[float],
    *,
    max_joint_step_rad: float,
) -> tuple[bool, int]:
    checks = 0
    samples = _interpolated_replay_qpos(
        model,
        start_qpos,
        end_qpos,
        max_joint_step_rad=max_joint_step_rad,
    )
    for qpos in [*samples, end_qpos]:
        checks += 1
        if not _replay_collision_report(mujoco, model, data, qpos)["collision_free"]:
            return False, checks
    return True, checks


def _densify_replay_qpos_path(
    model: Any,
    path: list[list[float]],
    *,
    max_joint_step_rad: float,
) -> list[list[float]]:
    if len(path) < 2:
        return []
    samples: list[list[float]] = []
    for index in range(len(path) - 1):
        segment_start = path[index]
        segment_end = path[index + 1]
        samples.extend(
            _interpolated_replay_qpos(
                model,
                segment_start,
                segment_end,
                max_joint_step_rad=max_joint_step_rad,
            )
        )
        if index + 1 < len(path) - 1:
            samples.append(list(segment_end))
    return samples


def _shortcut_replay_qpos_path(
    mujoco: Any,
    model: Any,
    data: Any,
    path: list[list[float]],
    *,
    max_joint_step_rad: float,
    shortcut_passes: int,
) -> tuple[list[list[float]], int, int]:
    shortcut_path = [list(qpos) for qpos in path]
    if shortcut_passes <= 0 or len(shortcut_path) < 3:
        return shortcut_path, 0, 0

    collision_checks = 0
    accepted_shortcuts = 0
    for _pass_index in range(shortcut_passes):
        changed = False
        start_index = 0
        while start_index + 2 < len(shortcut_path):
            accepted_end_index: int | None = None
            for end_index in range(len(shortcut_path) - 1, start_index + 1, -1):
                edge_ok, checks = _collision_free_replay_edge(
                    mujoco,
                    model,
                    data,
                    shortcut_path[start_index],
                    shortcut_path[end_index],
                    max_joint_step_rad=max_joint_step_rad,
                )
                collision_checks += checks
                if edge_ok:
                    accepted_end_index = end_index
                    break
            if accepted_end_index is None:
                start_index += 1
                continue
            shortcut_path = [
                *shortcut_path[: start_index + 1],
                *shortcut_path[accepted_end_index:],
            ]
            accepted_shortcuts += 1
            changed = True
            start_index += 1
        if not changed:
            break
    return shortcut_path, collision_checks, accepted_shortcuts


def _rrt_connect_replay_segment(
    mujoco: Any,
    model: Any,
    data: Any,
    start_qpos: list[float],
    end_qpos: list[float],
    *,
    max_joint_step_rad: float,
    rrt_step_rad: float,
    max_nodes: int,
    goal_sample_rate: float,
    rng: random.Random,
) -> tuple[list[list[float]] | None, int]:
    bounds = _replay_joint_bounds(model, start_qpos, end_qpos)
    nodes_a: list[list[float]] = [list(start_qpos)]
    parents_a: list[int] = [-1]
    nodes_b: list[list[float]] = [list(end_qpos)]
    parents_b: list[int] = [-1]
    checks = 0
    swapped = False
    for _iteration in range(max_nodes):
        if len(nodes_a) + len(nodes_b) >= max_nodes:
            break
        sample = _sample_replay_rrt_target(
            bounds,
            start_qpos,
            end_qpos,
            rng,
            goal_sample_rate=goal_sample_rate,
            tube_radius_rad=max(rrt_step_rad * 4.0, max_joint_step_rad),
        )
        new_index, extend_checks, _reached = _extend_replay_rrt_tree(
            mujoco,
            model,
            data,
            nodes_a,
            parents_a,
            sample,
            rrt_step_rad=rrt_step_rad,
            collision_check_step_rad=max_joint_step_rad,
        )
        checks += extend_checks
        if new_index is None:
            nodes_a, nodes_b = nodes_b, nodes_a
            parents_a, parents_b = parents_b, parents_a
            swapped = not swapped
            continue

        connect_index, connect_checks, connected = _connect_replay_rrt_tree(
            mujoco,
            model,
            data,
            nodes_b,
            parents_b,
            nodes_a[new_index],
            rrt_step_rad=rrt_step_rad,
            collision_check_step_rad=max_joint_step_rad,
        )
        checks += connect_checks
        if connected and connect_index is not None:
            path_a = _replay_node_path(nodes_a, parents_a, new_index)
            path_b = _replay_node_path(nodes_b, parents_b, connect_index)
            if not swapped:
                path = path_a + list(reversed(path_b[:-1]))
            else:
                path = path_b + list(reversed(path_a[:-1]))
            return path[1:-1], checks

        nodes_a, nodes_b = nodes_b, nodes_a
        parents_a, parents_b = parents_b, parents_a
        swapped = not swapped
    return None, checks


def _extend_replay_rrt_tree(
    mujoco: Any,
    model: Any,
    data: Any,
    nodes: list[list[float]],
    parents: list[int],
    target_qpos: list[float],
    *,
    rrt_step_rad: float,
    collision_check_step_rad: float,
) -> tuple[int | None, int, bool]:
    nearest_index = _nearest_replay_node_index(nodes, target_qpos)
    new_qpos = _steer_replay_qpos(nodes[nearest_index], target_qpos, max_joint_step_rad=rrt_step_rad)
    edge_ok, checks = _collision_free_replay_edge(
        mujoco,
        model,
        data,
        nodes[nearest_index],
        new_qpos,
        max_joint_step_rad=collision_check_step_rad,
    )
    if not edge_ok:
        return None, checks, False
    nodes.append(new_qpos)
    parents.append(nearest_index)
    reached = _replay_qpos_distance(new_qpos, target_qpos) <= 1e-9
    return len(nodes) - 1, checks, reached


def _connect_replay_rrt_tree(
    mujoco: Any,
    model: Any,
    data: Any,
    nodes: list[list[float]],
    parents: list[int],
    target_qpos: list[float],
    *,
    rrt_step_rad: float,
    collision_check_step_rad: float,
) -> tuple[int | None, int, bool]:
    checks = 0
    last_index: int | None = None
    while True:
        new_index, extend_checks, reached = _extend_replay_rrt_tree(
            mujoco,
            model,
            data,
            nodes,
            parents,
            target_qpos,
            rrt_step_rad=rrt_step_rad,
            collision_check_step_rad=collision_check_step_rad,
        )
        checks += extend_checks
        if new_index is None:
            return last_index, checks, False
        last_index = new_index
        if reached:
            return new_index, checks, True


def _replay_joint_bounds(model: Any, start_qpos: list[float], end_qpos: list[float]) -> list[tuple[float, float]]:
    bounds = []
    for index in range(min(len(start_qpos), len(end_qpos), int(model.nq))):
        low = min(float(start_qpos[index]), float(end_qpos[index])) - math.pi
        high = max(float(start_qpos[index]), float(end_qpos[index])) + math.pi
        bounds.append((low, high))
    for joint_id in range(int(getattr(model, "njnt", 0))):
        qpos_addr = int(model.jnt_qposadr[joint_id])
        if qpos_addr < 0 or qpos_addr >= len(bounds):
            continue
        if int(model.jnt_limited[joint_id]) == 0:
            continue
        low = float(model.jnt_range[joint_id][0])
        high = float(model.jnt_range[joint_id][1])
        bounds[qpos_addr] = (low, high)
    return bounds


def _sample_replay_qpos(bounds: list[tuple[float, float]], rng: random.Random) -> list[float]:
    return [rng.uniform(low, high) for low, high in bounds]


def _sample_replay_rrt_target(
    bounds: list[tuple[float, float]],
    start_qpos: list[float],
    end_qpos: list[float],
    rng: random.Random,
    *,
    goal_sample_rate: float,
    tube_radius_rad: float,
) -> list[float]:
    draw = rng.random()
    if draw < goal_sample_rate:
        return list(end_qpos)
    if draw < min(0.85, goal_sample_rate + 0.60):
        ratio = rng.random()
        sample = []
        for index, (low, high) in enumerate(bounds):
            center = float(start_qpos[index]) + (float(end_qpos[index]) - float(start_qpos[index])) * ratio
            value = center + rng.uniform(-tube_radius_rad, tube_radius_rad)
            sample.append(min(max(value, low), high))
        return sample
    return _sample_replay_qpos(bounds, rng)


def _nearest_replay_node_index(nodes: list[list[float]], target: list[float]) -> int:
    return min(range(len(nodes)), key=lambda index: _replay_qpos_distance(nodes[index], target))


def _replay_qpos_distance(left: list[float], right: list[float]) -> float:
    return max((abs(float(a) - float(b)) for a, b in zip(left, right, strict=False)), default=0.0)


def _steer_replay_qpos(start_qpos: list[float], target_qpos: list[float], *, max_joint_step_rad: float) -> list[float]:
    distance = _replay_qpos_distance(start_qpos, target_qpos)
    if distance <= max_joint_step_rad:
        return list(target_qpos)
    ratio = max_joint_step_rad / distance
    return [
        float(start_qpos[index]) + (float(target_qpos[index]) - float(start_qpos[index])) * ratio
        for index in range(len(start_qpos))
    ]


def _replay_node_path(nodes: list[list[float]], parents: list[int], node_index: int) -> list[list[float]]:
    path = []
    current = node_index
    while current >= 0:
        path.append(nodes[current])
        current = parents[current]
    path.reverse()
    return path


def _joint_space_delta(model: Any, start_qpos: list[float], end_qpos: list[float]) -> list[float]:
    count = min(len(start_qpos), len(end_qpos), int(model.nq))
    delta = [float(end_qpos[index]) - float(start_qpos[index]) for index in range(count)]
    for joint_id in range(int(getattr(model, "njnt", 0))):
        qpos_addr = int(model.jnt_qposadr[joint_id])
        if qpos_addr < 0 or qpos_addr >= count:
            continue
        joint_type = int(model.jnt_type[joint_id])
        is_hinge = joint_type == 3
        if not is_hinge or int(model.jnt_limited[joint_id]) != 0:
            continue
        delta[qpos_addr] = _wrap_to_pi(delta[qpos_addr])
    return delta


def _wrap_to_pi(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _key_display_frame(
    frame: dict[str, Any],
    *,
    keyframe_index: int,
    qpos: list[float] | None = None,
    adjusted_joint_count: int = 0,
    max_adjustment_rad: float = 0.0,
) -> dict[str, Any]:
    item = dict(frame)
    if qpos is not None:
        item["qpos"] = [float(value) for value in qpos]
    item["replay_generated"] = False
    item["replay_display_role"] = "keyframe"
    item["replay_keyframe_index"] = keyframe_index
    if adjusted_joint_count > 0:
        item["replay_qpos_normalized"] = True
        item["replay_qpos_normalization_reason"] = "nearest_equivalent_unlimited_hinge"
        item["replay_qpos_adjusted_joint_count"] = adjusted_joint_count
        item["replay_qpos_max_adjustment_rad"] = _round_replay_value(max_adjustment_rad)
    return item


def _interpolated_display_frame(
    *,
    from_frame: dict[str, Any],
    to_frame: dict[str, Any],
    from_index: int,
    interpolation_index: int,
    qpos: list[float],
    duration_s: float,
) -> dict[str, Any]:
    return {
        "frame_id": f"replay_motion/{from_index:04d}/{interpolation_index:03d}",
        "phase": to_frame.get("phase"),
        "view_role": "replay_motion",
        "view_id": None,
        "status": "generated_for_replay",
        "qpos": [_round_replay_value(value) for value in qpos],
        "qpos_source": "replay_collision_checked_interpolation",
        "camera_name": to_frame.get("camera_name") or from_frame.get("camera_name"),
        "target_id": to_frame.get("target_id"),
        "target_position_world": to_frame.get("target_position_world"),
        "replay_generated": True,
        "replay_display_role": "interpolated_motion",
        "replay_duration_s": float(duration_s),
        "replay_source_from_frame_id": from_frame.get("frame_id"),
        "replay_source_to_frame_id": to_frame.get("frame_id"),
        "replay_source_from_sequence_index": from_frame.get("sequence_index"),
        "replay_source_to_sequence_index": to_frame.get("sequence_index"),
    }


def _require_collision_free_qpos(
    mujoco: Any,
    model: Any,
    data: Any,
    qpos: list[float],
    *,
    frame_id: str,
    segment_id: str,
) -> None:
    report = _replay_collision_report(mujoco, model, data, qpos)
    if report["collision_free"]:
        return
    contacts = report["robot_scene_contacts"] or report["robot_self_contacts"]
    first_contact = contacts[0] if contacts else {}
    raise SystemExit(
        "Replay smoothing collision check failed: "
        f"segment={segment_id}, frame={frame_id}, "
        f"robot_scene_contacts={report['robot_scene_contact_count']}, "
        f"robot_self_contacts={report['robot_self_contact_count']}, "
        f"first_contact={first_contact}"
    )


def _replay_collision_report(mujoco: Any, model: Any, data: Any, qpos: list[float]) -> dict[str, Any]:
    _apply_qpos(mujoco, model, data, qpos)
    robot_scene_contacts: list[dict[str, Any]] = []
    robot_self_contacts: list[dict[str, Any]] = []
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        body1 = _geom_body_id(model, geom1)
        body2 = _geom_body_id(model, geom2)
        is_robot1 = _is_robot_body(mujoco, model, body1)
        is_robot2 = _is_robot_body(mujoco, model, body2)
        item = {
            "geom1": _geom_name(mujoco, model, geom1),
            "geom2": _geom_name(mujoco, model, geom2),
            "body1": _body_name(mujoco, model, body1),
            "body2": _body_name(mujoco, model, body2),
            "distance_m": _round_replay_value(float(contact.dist)),
        }
        if is_robot1 and is_robot2:
            robot_self_contacts.append(item)
        elif is_robot1 or is_robot2:
            robot_scene_contacts.append(item)
    return {
        "collision_free": not robot_scene_contacts and not robot_self_contacts,
        "robot_scene_contact_count": len(robot_scene_contacts),
        "robot_self_contact_count": len(robot_self_contacts),
        "robot_scene_contacts": robot_scene_contacts[:5],
        "robot_self_contacts": robot_self_contacts[:5],
    }


def _geom_body_id(model: Any, geom_id: int) -> int:
    return int(model.geom_bodyid[geom_id])


def _geom_name(mujoco: Any, model: Any, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"


def _body_name(mujoco: Any, model: Any, body_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"


def _is_robot_body(mujoco: Any, model: Any, body_id: int) -> bool:
    current = int(body_id)
    while current >= 0:
        if _body_name(mujoco, model, current).startswith("gen3_"):
            return True
        parent = int(model.body_parentid[current])
        if parent == current:
            return False
        current = parent
    return False


def _display_duration_s(frame: dict[str, Any], *, default_duration_s: float) -> float:
    value = frame.get("replay_duration_s")
    if value is None:
        return float(default_duration_s)
    return max(float(value), 1e-6)


def _playback_duration_s(frame: dict[str, Any], *, default_duration_s: float, playback_speed: float) -> float:
    return max(_display_duration_s(frame, default_duration_s=default_duration_s) / playback_speed, 1e-6)


def _advance_replay_index(
    frames: list[dict[str, Any]],
    *,
    current_index: int,
    finished: bool,
    motion_target_index: int | None,
    elapsed_s: float,
    default_duration_s: float,
    playback_speed: float,
    loop: bool,
    force_first_step: bool = False,
) -> _ReplayAdvanceResult:
    if not frames:
        raise SystemExit("Replay contains no frames.")
    if playback_speed <= 0.0:
        raise SystemExit("Replay playback speed must be positive.")
    if finished:
        return _ReplayAdvanceResult(
            current_index=current_index,
            finished=True,
            motion_target_index=motion_target_index,
            elapsed_remainder_s=0.0,
            reached_indices=(),
        )

    bounded_index = min(max(current_index, 0), len(frames) - 1)
    remaining_s = max(float(elapsed_s), 0.0)
    reached_indices: list[int] = []
    force_pending = force_first_step
    max_advances = max(len(frames), 1)

    for _ in range(max_advances):
        duration_s = _playback_duration_s(
            frames[bounded_index],
            default_duration_s=default_duration_s,
            playback_speed=playback_speed,
        )
        if force_pending:
            force_pending = False
        elif remaining_s + 1e-12 < duration_s:
            break
        else:
            remaining_s = max(remaining_s - duration_s, 0.0)

        if bounded_index + 1 < len(frames):
            bounded_index += 1
        elif loop:
            bounded_index = 0
        else:
            if reached_indices:
                return _ReplayAdvanceResult(
                    current_index=bounded_index,
                    finished=False,
                    motion_target_index=motion_target_index,
                    elapsed_remainder_s=0.0,
                    reached_indices=tuple(reached_indices),
                )
            return _ReplayAdvanceResult(
                current_index=bounded_index,
                finished=True,
                motion_target_index=motion_target_index,
                elapsed_remainder_s=0.0,
                reached_indices=tuple(reached_indices),
            )
        reached_indices.append(bounded_index)

        if motion_target_index is not None and bounded_index == motion_target_index:
            return _ReplayAdvanceResult(
                current_index=bounded_index,
                finished=False,
                motion_target_index=None,
                elapsed_remainder_s=0.0,
                reached_indices=tuple(reached_indices),
            )

    advance_limit_reached = len(reached_indices) >= max_advances
    if advance_limit_reached:
        remaining_s = 0.0
    return _ReplayAdvanceResult(
        current_index=bounded_index,
        finished=False,
        motion_target_index=motion_target_index,
        elapsed_remainder_s=remaining_s,
        reached_indices=tuple(reached_indices),
        advance_limit_reached=advance_limit_reached,
    )


def _next_replay_keyframe_index(frames: list[dict[str, Any]], current_index: int, *, loop: bool) -> int | None:
    for index in range(current_index + 1, len(frames)):
        if not _is_generated_motion_frame(frames[index]):
            return index
    if not loop:
        return None
    for index, frame in enumerate(frames[: max(current_index, 0) + 1]):
        if not _is_generated_motion_frame(frame):
            return index
    return None


def _keyframe_index_at_or_before(frames: list[dict[str, Any]], current_index: int) -> int:
    if not frames:
        raise SystemExit("Replay contains no frames.")
    bounded_index = min(max(current_index, 0), len(frames) - 1)
    for index in range(bounded_index, -1, -1):
        if not _is_generated_motion_frame(frames[index]):
            return index
    for index, frame in enumerate(frames):
        if not _is_generated_motion_frame(frame):
            return index
    raise SystemExit("Replay contains no task1 capture frames.")


def _capture_frame_display_position(frames: list[dict[str, Any]], current_index: int) -> tuple[int, int]:
    capture_indices = [index for index, frame in enumerate(frames) if not _is_generated_motion_frame(frame)]
    if not capture_indices:
        raise SystemExit("Replay contains no task1 capture frames.")
    keyframe_index = _keyframe_index_at_or_before(frames, current_index)
    ordinal = sum(1 for index in capture_indices if index <= keyframe_index)
    return max(ordinal, 1), len(capture_indices)


def _is_generated_motion_frame(frame: dict[str, Any]) -> bool:
    return bool(frame.get("replay_generated"))


def _print_replay_motion_summary(summary: _ReplayMotionSummary) -> None:
    print("replay_motion=" + json.dumps(summary.to_dict(), ensure_ascii=False))


def _print_replay_timing_warning(*, max_advances: int) -> None:
    payload = {
        "status": "catchup_limited",
        "reason": "one_render_tick_reached_replay_advance_limit",
        "max_advances": max_advances,
    }
    print("replay_timing=" + json.dumps(payload, ensure_ascii=False), file=sys.stderr)


def _round_replay_value(value: float | int | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _resolve_camera_id(mujoco: Any, model: Any, camera_name: str) -> int:
    for name in (camera_name, f"gen3_{camera_name}"):
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if camera_id >= 0:
            return int(camera_id)
    raise SystemExit(f"Camera not found in MuJoCo model: {camera_name} (also tried gen3_{camera_name})")


def _resolve_keymap(mujoco: Any) -> _GuiKeyMap:
    viewer_module = getattr(mujoco, "viewer", None)
    glfw = getattr(viewer_module, "glfw", None)
    if glfw is not None:
        return _GuiKeyMap(
            space=int(glfw.KEY_SPACE),
            next_frame=int(glfw.KEY_F9),
            reset=int(glfw.KEY_F12),
        )
    return _GuiKeyMap(space=32, next_frame=298, reset=301)


def _launch_passive_supports_key_callback(launch_passive: Any) -> bool:
    try:
        return "key_callback" in inspect.signature(launch_passive).parameters
    except (TypeError, ValueError):
        return False


def _configure_viewer_camera(
    viewer: Any,
    mujoco: Any,
    model: Any,
    data: Any,
    mode: str,
    manifest: dict[str, Any],
    frame: dict[str, Any],
    camera_id: int,
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
    if mode == "wrist":
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = camera_id
        return

    camera_params = _global_camera_params_for_frame(
        model,
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
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[0] = camera_params["lookat"][0]
    viewer.cam.lookat[1] = camera_params["lookat"][1]
    viewer.cam.lookat[2] = camera_params["lookat"][2]
    viewer.cam.distance = camera_params["distance"]
    viewer.cam.azimuth = camera_params["azimuth"]
    viewer.cam.elevation = camera_params["elevation"]
    _set_global_free_camera_fovy(
        model,
        _global_fovy_for_frame(
            frame,
            global_fovy_deg=global_fovy_deg,
            survey_global_fovy_deg=survey_global_fovy_deg,
        ),
    )
    mujoco.mj_forward(model, data)


def _configure_mjv_global_camera(
    mujoco: Any,
    model: Any,
    data: Any,
    camera: Any,
    manifest: dict[str, Any],
    frame: dict[str, Any],
    *,
    global_lookat: tuple[float, float, float] | None,
    global_distance: float | None,
    global_elevation_deg: float | None,
    survey_global_lookat: tuple[float, float, float] | None,
    survey_global_distance: float | None,
    survey_global_azimuth_deg: float | None,
    survey_global_elevation_deg: float | None,
) -> None:
    camera_params = _global_camera_params_for_frame(
        model,
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
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[0] = camera_params["lookat"][0]
    camera.lookat[1] = camera_params["lookat"][1]
    camera.lookat[2] = camera_params["lookat"][2]
    camera.distance = camera_params["distance"]
    camera.azimuth = camera_params["azimuth"]
    camera.elevation = camera_params["elevation"]
    mujoco.mj_forward(model, data)


def _global_camera_params_for_frame(
    model: Any,
    manifest: dict[str, Any],
    frame: dict[str, Any],
    *,
    global_lookat: tuple[float, float, float] | None = DEFAULT_GLOBAL_LOOKAT,
    global_distance: float | None = DEFAULT_GLOBAL_DISTANCE_M,
    global_elevation_deg: float | None = DEFAULT_GLOBAL_ELEVATION_DEG,
    survey_global_lookat: tuple[float, float, float] | None = DEFAULT_SURVEY_GLOBAL_LOOKAT,
    survey_global_distance: float | None = DEFAULT_SURVEY_GLOBAL_DISTANCE_M,
    survey_global_azimuth_deg: float | None = DEFAULT_SURVEY_GLOBAL_AZIMUTH_DEG,
    survey_global_elevation_deg: float | None = DEFAULT_SURVEY_GLOBAL_ELEVATION_DEG,
) -> dict[str, Any]:
    if frame.get("phase") == "survey":
        return _survey_overview_camera_params(
            model,
            manifest,
            survey_global_lookat=survey_global_lookat,
            survey_global_distance=survey_global_distance,
            survey_global_azimuth_deg=survey_global_azimuth_deg,
            survey_global_elevation_deg=survey_global_elevation_deg,
        )
    return _interior_overview_camera_params(
        model,
        manifest,
        global_lookat=global_lookat,
        global_distance=global_distance,
        global_elevation_deg=global_elevation_deg,
    )


def _survey_overview_camera_params(
    model: Any,
    manifest: dict[str, Any],
    *,
    survey_global_lookat: tuple[float, float, float] | None = DEFAULT_SURVEY_GLOBAL_LOOKAT,
    survey_global_distance: float | None = DEFAULT_SURVEY_GLOBAL_DISTANCE_M,
    survey_global_azimuth_deg: float | None = DEFAULT_SURVEY_GLOBAL_AZIMUTH_DEG,
    survey_global_elevation_deg: float | None = DEFAULT_SURVEY_GLOBAL_ELEVATION_DEG,
) -> dict[str, Any]:
    return {
        "lookat": tuple(float(value) for value in survey_global_lookat)
        if survey_global_lookat is not None
        else DEFAULT_SURVEY_GLOBAL_LOOKAT,
        "distance": float(survey_global_distance)
        if survey_global_distance is not None
        else DEFAULT_SURVEY_GLOBAL_DISTANCE_M,
        "azimuth": float(survey_global_azimuth_deg)
        if survey_global_azimuth_deg is not None
        else DEFAULT_SURVEY_GLOBAL_AZIMUTH_DEG,
        "elevation": float(survey_global_elevation_deg)
        if survey_global_elevation_deg is not None
        else DEFAULT_SURVEY_GLOBAL_ELEVATION_DEG,
    }


def _interior_overview_camera_params(
    model: Any,
    manifest: dict[str, Any],
    *,
    global_lookat: tuple[float, float, float] | None = DEFAULT_GLOBAL_LOOKAT,
    global_distance: float | None = DEFAULT_GLOBAL_DISTANCE_M,
    global_elevation_deg: float | None = DEFAULT_GLOBAL_ELEVATION_DEG,
) -> dict[str, Any]:
    workspace = manifest.get("workspace") if isinstance(manifest.get("workspace"), dict) else {}
    x_min = float(workspace.get("x_min", 0.15))
    x_max = float(workspace.get("x_max", 0.85))
    y_min = float(workspace.get("y_min", 0.15))
    y_max = float(workspace.get("y_max", 0.85))
    bottom_z = float(workspace.get("bottom_z_m", 0.03))
    span = max(abs(x_max - x_min), abs(y_max - y_min), 1e-6)
    distance = min(max(span * 0.60, 0.38), 0.46)
    if getattr(model, "stat", None) is not None:
        distance = min(distance, max(float(model.stat.extent) * 0.38, 0.34))
    lookat = (
        (x_min + x_max) / 2.0,
        (y_min + y_max) / 2.0,
        bottom_z + 0.10,
    )
    return {
        "lookat": tuple(float(value) for value in global_lookat) if global_lookat is not None else lookat,
        "distance": float(global_distance) if global_distance is not None else distance,
        "azimuth": 135.0,
        "elevation": float(global_elevation_deg) if global_elevation_deg is not None else -22.0,
    }


def _global_fovy_for_frame(
    frame: dict[str, Any],
    *,
    global_fovy_deg: float,
    survey_global_fovy_deg: float | None,
) -> float:
    if frame.get("phase") == "survey" and survey_global_fovy_deg is not None:
        return float(survey_global_fovy_deg)
    return float(global_fovy_deg)


def _split_viewports(mujoco: Any, *, width: int, height: int) -> tuple[Any, Any, Any]:
    divider_width = 2
    left_width = max(1, (int(width) - divider_width) // 2)
    right_left = left_width + divider_width
    right_width = max(1, int(width) - right_left)
    left = mujoco.MjrRect(0, 0, left_width, int(height))
    divider = mujoco.MjrRect(left_width, 0, divider_width, int(height))
    right = mujoco.MjrRect(right_left, 0, right_width, int(height))
    return left, divider, right


def _render_scene_view(
    mujoco: Any,
    *,
    model: Any,
    data: Any,
    options: Any,
    scene: Any,
    context: Any,
    camera: Any,
    viewport: Any,
    global_fovy_deg: float | None = None,
) -> None:
    original_fovy = _global_free_camera_fovy(model)
    if global_fovy_deg is not None:
        _set_global_free_camera_fovy(model, global_fovy_deg)
    mujoco.mjv_updateScene(
        model,
        data,
        options,
        None,
        camera,
        int(mujoco.mjtCatBit.mjCAT_ALL),
        scene,
    )
    mujoco.mjr_render(viewport, scene, context)
    if original_fovy is not None and global_fovy_deg is not None:
        _set_global_free_camera_fovy(model, original_fovy)


def _set_global_free_camera_fovy(model: Any, fovy_deg: float) -> None:
    if getattr(model, "vis", None) is None:
        return
    global_vis = getattr(model.vis, "global_", None)
    if global_vis is None or not hasattr(global_vis, "fovy"):
        return
    global_vis.fovy = float(fovy_deg)


def _global_free_camera_fovy(model: Any) -> float | None:
    if getattr(model, "vis", None) is None:
        return None
    global_vis = getattr(model.vis, "global_", None)
    if global_vis is None or not hasattr(global_vis, "fovy"):
        return None
    return float(global_vis.fovy)


def _render_split_overlay(
    mujoco: Any,
    *,
    context: Any,
    left: Any,
    right: Any,
    frame: dict[str, Any],
    frame_index: int,
    frame_count: int,
    wrist_frame: dict[str, Any],
    wrist_frame_index: int,
    control: _ReplayControl,
    finished: bool,
) -> None:
    font = int(mujoco.mjtFont.mjFONT_NORMAL)
    top_left = int(mujoco.mjtGridPos.mjGRID_TOPLEFT)
    top_right = int(mujoco.mjtGridPos.mjGRID_TOPRIGHT)
    state_text = _control_state_text(control, finished=finished)
    frame_text = "\n".join(
        [
            "Global view",
            f"Frame {frame_index + 1}/{frame_count}",
            f"{frame.get('phase')} / {frame.get('view_role')}",
            str(frame.get("view_id") or frame.get("frame_id")),
            str(frame.get("target_id") or ""),
        ]
    )
    keys_text = "\n".join(
        [
            "Keys:",
            "Space: pause/resume",
            "F9: next capture frame",
            "F12: reset",
            f"State: {state_text}",
        ]
    )
    wrist_text = "\n".join(
        [
            "Wrist camera",
            f"Capture frame {wrist_frame_index + 1}/{frame_count}",
            f"{wrist_frame.get('phase')} / {wrist_frame.get('view_role')}",
            str(wrist_frame.get("view_id") or wrist_frame.get("frame_id")),
            str(wrist_frame.get("rgb_image_path") or wrist_frame.get("annotated_image_path") or ""),
        ]
    )
    mujoco.mjr_overlay(font, top_left, left, keys_text, "", context)
    mujoco.mjr_overlay(font, top_right, left, frame_text, "", context)
    mujoco.mjr_overlay(font, top_left, right, wrist_text, "", context)


def _control_state_text(control: _ReplayControl, *, finished: bool) -> str:
    paused, pending = control.snapshot()
    state = "paused" if paused else "running"
    if finished:
        return f"{state}, finished"
    if pending:
        return f"{state}, queued={pending}"
    return state


def _render_overlay(
    viewer: Any,
    mujoco: Any,
    *,
    mode: str,
    frame: dict[str, Any],
    frame_index: int,
    frame_count: int,
    control: _ReplayControl,
    finished: bool,
) -> None:
    if not hasattr(viewer, "set_texts"):
        return
    font = getattr(mujoco.mjtFontScale, "mjFONTSCALE_100", mujoco.mjtFontScale.mjFONTSCALE_150)
    top_right = getattr(mujoco.mjtGridPos, "mjGRID_TOPRIGHT", None)
    top_left = getattr(mujoco.mjtGridPos, "mjGRID_TOPLEFT", None)
    if top_right is None:
        return

    paused, pending = control.snapshot()
    state = "paused" if paused else "running"
    if finished:
        state = f"{state}, finished"
    elif pending:
        state = f"{state}, queued={pending}"
    label = "\n".join(
        [
            f"{mode} view",
            f"Frame {frame_index + 1}/{frame_count}",
            f"{frame.get('phase')} / {frame.get('view_role')}",
            str(frame.get("view_id") or frame.get("frame_id")),
            str(frame.get("target_id") or ""),
        ]
    )
    keys = "\n".join(
        [
            "Keys:",
            "Space: pause/resume",
            "F9: next capture frame",
            "F12: reset",
            f"State: {state}",
        ]
    )
    try:
        texts = [(font, top_right, label, "")]
        if top_left is not None:
            texts.insert(0, (font, top_left, keys, ""))
        viewer.set_texts(texts)
    except Exception:
        return


def _print_frame(frame: dict[str, Any]) -> None:
    image_path = frame.get("rgb_image_path") or frame.get("annotated_image_path")
    print(
        "frame="
        f"{frame.get('sequence_index')} "
        f"phase={frame.get('phase')} "
        f"role={frame.get('view_role')} "
        f"view={frame.get('view_id')} "
        f"target={frame.get('target_id')} "
        f"image={image_path}"
    )


if __name__ == "__main__":
    main()
