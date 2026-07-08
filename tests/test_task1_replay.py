import json
import importlib.util
import math
import random
import subprocess
import sys
from pathlib import Path

import pytest

from robot_arm_pipeline.task1.replay import (
    TASK1_REPLAY_SCHEMA_VERSION,
    build_task1_replay_manifest,
    find_latest_task1_run,
    replay_manifest_path_for_run,
    write_task1_replay_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_task1_replay_manifest_collects_recorded_robot_poses(tmp_path: Path) -> None:
    run_dir = _write_task1_run(tmp_path)

    manifest = build_task1_replay_manifest(run_dir)

    assert manifest["schema_version"] == TASK1_REPLAY_SCHEMA_VERSION
    assert manifest["status"] == "success"
    assert manifest["frame_count"] == 3
    assert [frame["phase"] for frame in manifest["frames"]] == ["survey", "rough", "final"]
    assert [frame["view_role"] for frame in manifest["frames"]] == [
        "survey_capture",
        "rough_capture",
        "final_photo",
    ]
    assert manifest["frames"][0]["qpos_source"] == "actual_qpos"
    assert manifest["frames"][1]["qpos_source"] == "fixed_qpos"
    assert manifest["frames"][-1]["target_id"] == "rough_object_001"
    assert manifest["frames"][-1]["rgb_image_path"].endswith("final_rough_object_001_rgb.png")


def test_task1_replay_manifest_writer_and_latest_run(tmp_path: Path) -> None:
    run_dir = _write_task1_run(tmp_path)

    manifest = write_task1_replay_manifest(run_dir)

    manifest_path = replay_manifest_path_for_run(run_dir)
    assert manifest_path.exists()
    assert manifest["manifest_path"] == str(manifest_path)
    assert find_latest_task1_run(tmp_path / "task1") == run_dir


def test_replay_script_manifest_only_builds_manifest(tmp_path: Path) -> None:
    run_dir = _write_task1_run(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "replay_task1_output.py"),
            "--run-dir",
            str(run_dir),
            "--manifest-only",
            "--rebuild-manifest",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "manifest=" in result.stdout
    manifest_path = replay_manifest_path_for_run(run_dir)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["frame_count"] == 3
    assert '"status": "success"' in result.stdout


def test_replay_script_split_viewports_share_one_framebuffer() -> None:
    mujoco = pytest.importorskip("mujoco")
    module = _load_replay_script_module()

    left, divider, right = module._split_viewports(mujoco, width=1001, height=700)

    assert (left.left, left.bottom, left.width, left.height) == (0, 0, 499, 700)
    assert (divider.left, divider.bottom, divider.width, divider.height) == (499, 0, 2, 700)
    assert (right.left, right.bottom, right.width, right.height) == (501, 0, 500, 700)


def test_replay_global_camera_uses_interior_overview_pose() -> None:
    module = _load_replay_script_module()
    model = _FakeModel(extent=1.2)
    manifest = {
        "workspace": {
            "x_min": 0.15,
            "x_max": 0.85,
            "y_min": 0.15,
            "y_max": 0.85,
            "bottom_z_m": 0.03,
        }
    }

    params = module._interior_overview_camera_params(model, manifest)

    assert params["lookat"] == pytest.approx((0.5, 0.5, 0.15))
    assert params["distance"] == pytest.approx(0.75)
    assert params["azimuth"] == pytest.approx(135.0)
    assert params["elevation"] == pytest.approx(-20.0)


def test_replay_global_camera_uses_survey_external_overview_for_survey_frames() -> None:
    module = _load_replay_script_module()
    model = _FakeModel(extent=1.2)
    manifest = {
        "workspace": {
            "x_min": 0.15,
            "x_max": 0.85,
            "y_min": 0.15,
            "y_max": 0.85,
            "bottom_z_m": 0.03,
            "tank_opening_z_m": 0.50,
        }
    }

    survey_params = module._global_camera_params_for_frame(model, manifest, {"phase": "survey"})
    rough_params = module._global_camera_params_for_frame(model, manifest, {"phase": "rough"})

    assert survey_params["lookat"] == pytest.approx((0.5, 0.5, 0.4))
    assert survey_params["distance"] == pytest.approx(1.0)
    assert survey_params["azimuth"] == pytest.approx(135.0)
    assert survey_params["elevation"] == pytest.approx(-30.0)
    assert rough_params["lookat"] == pytest.approx((0.5, 0.5, 0.15))
    assert rough_params["distance"] == pytest.approx(0.75)
    assert rough_params["elevation"] == pytest.approx(-20.0)


def test_replay_survey_global_camera_pose_can_be_overridden() -> None:
    module = _load_replay_script_module()
    model = _FakeModel(extent=1.2)
    manifest = {
        "workspace": {
            "x_min": 0.15,
            "x_max": 0.85,
            "y_min": 0.15,
            "y_max": 0.85,
            "bottom_z_m": 0.03,
            "tank_opening_z_m": 0.50,
        }
    }

    params = module._global_camera_params_for_frame(
        model,
        manifest,
        {"phase": "survey"},
        survey_global_lookat=(0.44, 0.95, 0.52),
        survey_global_distance=1.35,
        survey_global_azimuth_deg=105.0,
        survey_global_elevation_deg=-18.0,
    )

    assert params["lookat"] == pytest.approx((0.44, 0.95, 0.52))
    assert params["distance"] == pytest.approx(1.35)
    assert params["azimuth"] == pytest.approx(105.0)
    assert params["elevation"] == pytest.approx(-18.0)


def test_replay_cli_global_camera_defaults_match_interior_view() -> None:
    module = _load_replay_script_module()

    args = module.build_arg_parser().parse_args([])

    assert args.global_lookat == pytest.approx((0.5, 0.5, 0.15))
    assert args.global_distance == pytest.approx(0.75)
    assert args.global_elevation_deg == pytest.approx(-20.0)
    assert args.global_fovy_deg == pytest.approx(75.0)
    assert args.survey_global_lookat == pytest.approx((0.5, 0.5, 0.4))
    assert args.survey_global_distance == pytest.approx(1.0)
    assert args.survey_global_azimuth_deg == pytest.approx(135.0)
    assert args.survey_global_elevation_deg == pytest.approx(-30.0)
    assert args.survey_global_fovy_deg == pytest.approx(80.0)
    assert args.smooth_replay is True
    assert args.poll_seconds == pytest.approx(0.005)
    assert args.replay_max_joint_step_rad == pytest.approx(0.08)
    assert args.replay_interpolated_frame_duration == pytest.approx(0.04)
    assert args.replay_playback_speed == pytest.approx(2.0)
    assert args.replay_rrt_step_rad == pytest.approx(0.12)
    assert args.replay_rrt_max_nodes == 2500
    assert args.replay_rrt_attempt_count == 1
    assert args.replay_recorded_waypoint_window == 2
    assert args.replay_shortcut_passes == 2
    assert args.replay_heavy_connector is True
    assert args.replay_heavy_rrt_max_nodes == 50000
    assert args.replay_heavy_rrt_step_rad == pytest.approx(0.20)
    assert args.replay_heavy_rrt_goal_sample_rate == pytest.approx(0.15)
    assert args.replay_motion_failure_policy == "keyframe-cut"


def test_replay_cli_accepts_survey_global_camera_overrides() -> None:
    module = _load_replay_script_module()

    args = module.build_arg_parser().parse_args(
        [
            "--survey-global-lookat",
            "0.4,0.9,0.5",
            "--survey-global-distance",
            "1.25",
            "--survey-global-azimuth-deg",
            "110",
            "--survey-global-elevation-deg",
            "-16",
            "--survey-global-fovy-deg",
            "92",
        ]
    )

    assert args.survey_global_lookat == pytest.approx((0.4, 0.9, 0.5))
    assert args.survey_global_distance == pytest.approx(1.25)
    assert args.survey_global_azimuth_deg == pytest.approx(110.0)
    assert args.survey_global_elevation_deg == pytest.approx(-16.0)
    assert args.survey_global_fovy_deg == pytest.approx(92.0)


def test_replay_cli_accepts_playback_speed() -> None:
    module = _load_replay_script_module()

    args = module.build_arg_parser().parse_args(["--replay-playback-speed", "8"])

    assert args.replay_playback_speed == pytest.approx(8.0)


def test_replay_cli_accepts_recorded_waypoint_window() -> None:
    module = _load_replay_script_module()

    args = module.build_arg_parser().parse_args(
        [
            "--replay-recorded-waypoint-window",
            "4",
            "--replay-rrt-attempt-count",
            "3",
            "--replay-shortcut-passes",
            "5",
            "--no-replay-heavy-connector",
            "--replay-heavy-rrt-max-nodes",
            "60000",
        ]
    )

    assert args.replay_recorded_waypoint_window == 4
    assert args.replay_rrt_attempt_count == 3
    assert args.replay_shortcut_passes == 5
    assert args.replay_heavy_connector is False
    assert args.replay_heavy_rrt_max_nodes == 60000


def test_replay_global_fovy_uses_survey_override_only_for_survey_frames() -> None:
    module = _load_replay_script_module()

    assert module._global_fovy_for_frame(
        {"phase": "survey"},
        global_fovy_deg=75.0,
        survey_global_fovy_deg=92.0,
    ) == pytest.approx(92.0)
    assert module._global_fovy_for_frame(
        {"phase": "rough"},
        global_fovy_deg=75.0,
        survey_global_fovy_deg=92.0,
    ) == pytest.approx(75.0)


def test_replay_global_camera_pose_can_be_overridden() -> None:
    module = _load_replay_script_module()
    model = _FakeModel(extent=1.2)
    manifest = {"workspace": {"x_min": 0.15, "x_max": 0.85, "y_min": 0.15, "y_max": 0.85, "bottom_z_m": 0.03}}

    params = module._interior_overview_camera_params(
        model,
        manifest,
        global_lookat=(0.42, 0.58, 0.16),
        global_distance=0.35,
        global_elevation_deg=-12.0,
    )

    assert params["lookat"] == pytest.approx((0.42, 0.58, 0.16))
    assert params["distance"] == pytest.approx(0.35)
    assert params["elevation"] == pytest.approx(-12.0)


def test_replay_global_camera_fovy_is_configurable() -> None:
    module = _load_replay_script_module()
    model = _FakeModel(extent=1.2, fovy=45.0)

    assert module._global_free_camera_fovy(model) == pytest.approx(45.0)

    module._set_global_free_camera_fovy(model, 80.0)

    assert module._global_free_camera_fovy(model) == pytest.approx(80.0)


def test_replay_motion_interpolates_display_frames_without_changing_manifest_frames() -> None:
    module = _load_replay_script_module()
    mujoco = _FakeMujoco()
    model = _FakeMotionModel(nq=2)
    data = _FakeMotionData(nq=2)
    frames = [
        {"frame_id": "a", "phase": "survey", "qpos": [0.0, 0.0], "sequence_index": 0},
        {"frame_id": "b", "phase": "survey", "qpos": [0.24, 0.0], "sequence_index": 1},
    ]

    display_frames, summary = module._build_collision_checked_replay_motion(
        mujoco,
        model,
        data,
        frames,
        max_joint_step_rad=0.1,
        interpolated_frame_duration_s=0.04,
    )

    assert len(frames) == 2
    assert [frame["frame_id"] for frame in frames] == ["a", "b"]
    assert summary.input_frame_count == 2
    assert summary.display_frame_count == 4
    assert summary.interpolated_frame_count == 2
    assert display_frames[0]["replay_display_role"] == "keyframe"
    assert display_frames[1]["replay_display_role"] == "interpolated_motion"
    assert display_frames[1]["replay_duration_s"] == pytest.approx(0.04)
    assert display_frames[-1]["frame_id"] == "b"


def test_replay_motion_plans_between_rough_target_groups() -> None:
    module = _load_replay_script_module()
    mujoco = _FakeMujoco()
    model = _FakeMotionModel(nq=1)
    data = _FakeMotionData(nq=1)
    frames = [
        {"frame_id": "rough/a", "phase": "rough", "target_id": "candidate_a", "qpos": [0.0]},
        {"frame_id": "rough/b", "phase": "rough", "target_id": "candidate_b", "qpos": [0.24]},
    ]

    display_frames, summary = module._build_collision_checked_replay_motion(
        mujoco,
        model,
        data,
        frames,
        max_joint_step_rad=0.1,
        interpolated_frame_duration_s=0.04,
    )

    assert summary.keyframe_cut_count == 0
    assert summary.planned_boundary_connector_count == 1
    assert summary.interpolated_frame_count == 2
    assert len(display_frames) == 4
    assert display_frames[0]["replay_planned_boundary"] == "target_group_boundary"
    assert "replay_cut_after" not in display_frames[0]


def test_replay_motion_plans_rough_to_final_photo_boundary() -> None:
    module = _load_replay_script_module()
    mujoco = _FakeMujoco()
    model = _FakeMotionModel(nq=1)
    data = _FakeMotionData(nq=1)
    frames = [
        {
            "frame_id": "rough/rough_candidate_005_02",
            "phase": "rough",
            "target_id": "rough_candidate_005",
            "qpos": [0.0],
        },
        {
            "frame_id": "final/rough_object_001/photo",
            "phase": "final",
            "target_id": "rough_object_001",
            "qpos": [0.24],
        },
    ]

    display_frames, summary = module._build_collision_checked_replay_motion(
        mujoco,
        model,
        data,
        frames,
        max_joint_step_rad=0.1,
        interpolated_frame_duration_s=0.04,
    )

    assert summary.keyframe_cut_count == 0
    assert summary.planned_boundary_connector_count == 1
    assert summary.interpolated_frame_count == 2
    assert len(display_frames) == 4
    assert display_frames[0]["replay_planned_boundary"] == "rough_to_final_boundary"
    assert "replay_cut_after" not in display_frames[0]


def test_replay_motion_keeps_survey_to_rough_phase_boundary_cut() -> None:
    module = _load_replay_script_module()
    mujoco = _FakeMujoco()
    model = _FakeMotionModel(nq=1)
    data = _FakeMotionData(nq=1)
    frames = [
        {"frame_id": "survey/survey_04", "phase": "survey", "qpos": [0.0]},
        {
            "frame_id": "rough/rough_candidate_001_00",
            "phase": "rough",
            "target_id": "rough_candidate_001",
            "qpos": [0.24],
        },
    ]

    display_frames, summary = module._build_collision_checked_replay_motion(
        mujoco,
        model,
        data,
        frames,
        max_joint_step_rad=0.1,
        interpolated_frame_duration_s=0.04,
    )

    assert summary.keyframe_cut_count == 1
    assert summary.planned_boundary_connector_count == 0
    assert summary.interpolated_frame_count == 0
    assert display_frames[0]["replay_cut_reason"] == "phase_boundary"


def test_replay_motion_marks_keyframe_cut_when_connector_is_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_replay_script_module()
    mujoco = _FakeMujoco()
    model = _FakeMotionModel(nq=1)
    data = _FakeMotionData(nq=1)
    frames = [
        {"frame_id": "survey/a", "phase": "survey", "qpos": [0.0]},
        {"frame_id": "survey/b", "phase": "survey", "qpos": [0.24]},
    ]

    def _raise_no_connector(*_args, **_kwargs):
        raise SystemExit("no connector")

    monkeypatch.setattr(module, "_plan_replay_motion_segment", _raise_no_connector)

    display_frames, summary = module._build_collision_checked_replay_motion(
        mujoco,
        model,
        data,
        frames,
        max_joint_step_rad=0.1,
        interpolated_frame_duration_s=0.04,
    )

    assert summary.keyframe_cut_count == 1
    assert summary.failed_connector_cut_count == 1
    assert display_frames[0]["replay_cut_reason"] == "connector_not_found"
    assert "no connector" in display_frames[0]["replay_connector_error"]


def test_replay_motion_can_connect_via_waypoint_after_direct_edge_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_replay_script_module()
    mujoco = _FakeMujoco()
    model = _FakeMotionModel(nq=1)
    data = _FakeMotionData(nq=1)

    def _fake_edge_plan(_mujoco, _model, _data, start_qpos, end_qpos, **_kwargs):
        if start_qpos == [0.0] and end_qpos == [1.0]:
            return None, 3
        return (
            module._ReplaySegmentPlan(
                qpos_samples=[],
                strategy="direct",
                collision_check_count=1,
                direct_edge_count=1,
            ),
            1,
        )

    monkeypatch.setattr(module, "_plan_replay_edge_with_rrt", _fake_edge_plan)

    plan = module._plan_replay_motion_segment(
        mujoco,
        model,
        data,
        [0.0],
        [1.0],
        max_joint_step_rad=0.1,
        rrt_max_nodes=100,
        rrt_step_rad=0.1,
        rrt_goal_sample_rate=0.2,
        rrt_attempt_count=1,
        rng=random.Random(0),
        segment_id="a -> b",
        waypoint_candidates=[
            module._ReplayWaypointCandidate(name="neutral_home", qpos=[0.5], source="test")
        ],
    )

    assert plan.strategy == "via_waypoint:neutral_home:direct+direct"
    assert plan.qpos_samples == [[0.5]]
    assert plan.collision_check_count == 5
    assert plan.direct_edge_count == 2
    assert plan.waypoint_count == 1


def test_replay_edge_can_use_coordinate_sweep_after_direct_edge_collision(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_replay_script_module()
    mujoco = _FakeMujoco()
    model = _FakeMotionModel(nq=2)
    data = _FakeMotionData(nq=2)

    def _fake_collision_edge(_mujoco, _model, _data, start_qpos, end_qpos, **_kwargs):
        if start_qpos == [0.0, 0.0] and end_qpos == [1.0, 1.0]:
            return False, 1
        return True, 1

    monkeypatch.setattr(module, "_collision_free_replay_edge", _fake_collision_edge)

    plan, checks = module._plan_replay_edge_with_rrt(
        mujoco,
        model,
        data,
        [0.0, 0.0],
        [1.0, 1.0],
        max_joint_step_rad=0.1,
        rrt_max_nodes=100,
        rrt_step_rad=0.1,
        rrt_goal_sample_rate=0.2,
        rrt_attempt_count=1,
        rng=random.Random(0),
    )

    assert plan is not None
    assert plan.strategy == "coordinate_sweep"
    assert plan.coordinate_sweep_count == 1
    assert plan.direct_edge_count == 2
    assert checks == 3


def test_replay_shortcut_removes_collision_free_rrt_detours(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_replay_script_module()
    mujoco = _FakeMujoco()
    model = _FakeMotionModel(nq=1)
    data = _FakeMotionData(nq=1)

    def _fake_edge(_mujoco, _model, _data, start_qpos, end_qpos, **_kwargs):
        return abs(float(end_qpos[0]) - float(start_qpos[0])) <= 2.0, 1

    monkeypatch.setattr(module, "_collision_free_replay_edge", _fake_edge)

    path, checks, shortcut_count = module._shortcut_replay_qpos_path(
        mujoco,
        model,
        data,
        [[0.0], [1.0], [2.0], [3.0]],
        max_joint_step_rad=0.1,
        shortcut_passes=1,
    )

    assert path == [[0.0], [2.0], [3.0]]
    assert checks == 2
    assert shortcut_count == 1


def test_replay_edge_shortcuts_rrt_path_before_display_densification(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_replay_script_module()
    mujoco = _FakeMujoco()
    model = _FakeMotionModel(nq=1)
    data = _FakeMotionData(nq=1)

    def _fake_edge(_mujoco, _model, _data, start_qpos, end_qpos, **_kwargs):
        return abs(float(end_qpos[0]) - float(start_qpos[0])) <= 2.0, 1

    monkeypatch.setattr(module, "_collision_free_replay_edge", _fake_edge)
    monkeypatch.setattr(module, "_plan_replay_coordinate_sweep", lambda *_args, **_kwargs: (None, 0))
    monkeypatch.setattr(module, "_rrt_connect_replay_segment", lambda *_args, **_kwargs: ([[1.0], [2.0]], 4))

    plan, checks = module._plan_replay_edge_with_rrt(
        mujoco,
        model,
        data,
        [0.0],
        [3.0],
        max_joint_step_rad=1.0,
        rrt_max_nodes=100,
        rrt_step_rad=0.1,
        rrt_goal_sample_rate=0.2,
        rrt_attempt_count=1,
        rng=random.Random(0),
        shortcut_passes=1,
        heavy_connector=False,
    )

    assert plan is not None
    assert plan.strategy == "rrt_connect+shortcut"
    assert plan.shortcut_count == 1
    assert plan.shortcut_collision_check_count == 2
    assert plan.qpos_samples == [[1.0], [2.0]]
    assert checks == 7


def test_replay_edge_can_use_heavy_rrt_after_regular_rrt_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_replay_script_module()
    mujoco = _FakeMujoco()
    model = _FakeMotionModel(nq=2)
    data = _FakeMotionData(nq=2)

    monkeypatch.setattr(module, "_collision_free_replay_edge", lambda *_args, **_kwargs: (False, 1))
    monkeypatch.setattr(module, "_plan_replay_coordinate_sweep", lambda *_args, **_kwargs: (None, 2))

    def _fake_rrt(*_args, max_nodes: int, **_kwargs):
        if max_nodes == 100:
            return None, 3
        return [[0.5, 0.5]], 4

    monkeypatch.setattr(module, "_rrt_connect_replay_segment", _fake_rrt)

    plan, checks = module._plan_replay_edge_with_rrt(
        mujoco,
        model,
        data,
        [0.0, 0.0],
        [1.0, 1.0],
        max_joint_step_rad=0.1,
        rrt_max_nodes=100,
        rrt_step_rad=0.1,
        rrt_goal_sample_rate=0.2,
        rrt_attempt_count=1,
        rng=random.Random(0),
        shortcut_passes=0,
        heavy_connector=True,
        heavy_rrt_max_nodes=500,
        heavy_rrt_step_rad=0.2,
        heavy_rrt_goal_sample_rate=0.15,
    )

    assert plan is not None
    assert plan.strategy == "heavy_rrt_connect"
    assert plan.heavy_rrt_count == 1
    assert plan.rrt_edge_count == 1
    assert checks == 10


def test_replay_step_helpers_skip_generated_motion_frames() -> None:
    module = _load_replay_script_module()
    frames = [
        {"frame_id": "capture_0"},
        {"frame_id": "motion_0", "replay_generated": True},
        {"frame_id": "motion_1", "replay_generated": True},
        {"frame_id": "capture_1"},
        {"frame_id": "motion_2", "replay_generated": True},
        {"frame_id": "capture_2"},
    ]

    assert module._next_replay_keyframe_index(frames, 0, loop=False) == 3
    assert module._next_replay_keyframe_index(frames, 1, loop=False) == 3
    assert module._next_replay_keyframe_index(frames, 3, loop=False) == 5
    assert module._next_replay_keyframe_index(frames, 5, loop=False) is None
    assert module._next_replay_keyframe_index(frames, 5, loop=True) == 0
    assert module._keyframe_index_at_or_before(frames, 2) == 0
    assert module._keyframe_index_at_or_before(frames, 4) == 3


def test_replay_overlay_counts_only_capture_frames() -> None:
    module = _load_replay_script_module()
    frames = [
        {"frame_id": "capture_0"},
        {"frame_id": "motion_0", "replay_generated": True},
        {"frame_id": "motion_1", "replay_generated": True},
        {"frame_id": "capture_1"},
        {"frame_id": "motion_2", "replay_generated": True},
        {"frame_id": "capture_2"},
    ]

    assert module._capture_frame_display_position(frames, 0) == (1, 3)
    assert module._capture_frame_display_position(frames, 1) == (1, 3)
    assert module._capture_frame_display_position(frames, 2) == (1, 3)
    assert module._capture_frame_display_position(frames, 3) == (2, 3)
    assert module._capture_frame_display_position(frames, 4) == (2, 3)
    assert module._capture_frame_display_position(frames, 5) == (3, 3)


def test_replay_video_timeline_holds_captures_and_marks_motion() -> None:
    module = _load_replay_video_script_module()
    frames = [
        {"frame_id": "survey_0000", "phase": "survey"},
        {"frame_id": "motion_0", "phase": "survey", "replay_generated": True},
        {"frame_id": "motion_1", "phase": "survey", "replay_generated": True},
        {"frame_id": "survey_0001", "phase": "survey"},
        {"frame_id": "final_0000", "phase": "final"},
    ]

    timeline = module._build_video_timeline(
        frames,
        fps=10.0,
        capture_hold_seconds=2.0,
        motion_frame_duration_s=0.04,
    )

    assert [item.role for item in timeline] == [
        "capture_hold",
        "motion",
        "motion",
        "capture_hold",
        "capture_hold",
    ]
    assert timeline[0].display_index == 0
    assert timeline[0].wrist_index == 0
    assert timeline[0].label == "survey 1/2"
    assert timeline[0].repeat_count == 20
    assert timeline[1].display_index == 1
    assert timeline[1].wrist_index == 0
    assert timeline[1].label == "moving..."
    assert timeline[1].repeat_count == 1
    assert timeline[3].label == "survey 2/2"
    assert timeline[4].label == "final 1/1"


def test_replay_video_phase_labels_ignore_generated_motion_frames() -> None:
    module = _load_replay_video_script_module()
    frames = [
        {"frame_id": "survey_0000", "phase": "survey"},
        {"frame_id": "motion_0", "phase": "survey", "replay_generated": True},
        {"frame_id": "survey_0001", "phase": "survey"},
        {"frame_id": "rough_0000", "phase": "rough"},
        {"frame_id": "final_0000", "phase": "final"},
        {"frame_id": "final_0001", "phase": "final"},
    ]

    labels = module._capture_phase_labels(frames)

    assert labels == {
        0: "survey 1/2",
        2: "survey 2/2",
        3: "rough 1/1",
        4: "final 1/2",
        5: "final 2/2",
    }


def test_replay_video_loads_final_zoom_photo_pairs(tmp_path: Path) -> None:
    image_module = pytest.importorskip("PIL.Image")
    module = _load_replay_video_script_module()
    run_dir = tmp_path / "task1" / "run_seed101"
    final_image = run_dir / "final" / "images" / "final_object.png"
    zoom_image = run_dir / "zoom" / "selected" / "zoom_object.png"
    final_image.parent.mkdir(parents=True, exist_ok=True)
    zoom_image.parent.mkdir(parents=True, exist_ok=True)
    image_module.new("RGB", (1920, 1080), color=(20, 30, 40)).save(final_image)
    image_module.new("RGB", (1920, 1080), color=(80, 90, 100)).save(zoom_image)
    _write_report(
        run_dir / "final" / "final_report.json",
        {
            "schema_version": "task1_final_report_v1",
            "stable_objects": [
                {
                    "object_id": "rough_object_001",
                    "final_image_path": str(final_image),
                }
            ],
        },
    )
    _write_report(
        run_dir / "zoom" / "zoom_report.json",
        {
            "schema_version": "task1_zoom_report_v1",
            "zoomed_objects": [
                {
                    "object_id": "rough_object_001",
                    "source_final_image_path": str(final_image),
                    "selected_zoom": {"selected_image_path": str(zoom_image)},
                }
            ],
        },
    )

    specs = module._load_final_zoom_photo_specs(
        run_dir=run_dir,
        repo_root=tmp_path,
        final_report_path=None,
        zoom_report_path=None,
    )

    assert len(specs) == 1
    assert specs[0].object_id == "rough_object_001"
    assert specs[0].final_image_path == final_image
    assert specs[0].zoom_image_path == zoom_image
    assert specs[0].ordinal == 1
    assert specs[0].total == 1


def test_replay_video_composes_final_zoom_photo_frame(tmp_path: Path) -> None:
    image_module = pytest.importorskip("PIL.Image")
    module = _load_replay_video_script_module()
    final_image = tmp_path / "final.png"
    zoom_image = tmp_path / "zoom.png"
    image_module.new("RGB", (1920, 1080), color=(20, 30, 40)).save(final_image)
    image_module.new("RGB", (1920, 1080), color=(80, 90, 100)).save(zoom_image)

    frame = module._compose_final_zoom_photo_frame(
        module._FinalZoomPhotoSpec(
            object_id="rough_object_001",
            final_image_path=final_image,
            zoom_image_path=zoom_image,
            ordinal=1,
            total=5,
        ),
        width=320,
        height=180,
    )

    assert frame.shape == (180, 320, 3)
    assert tuple(frame[10, 10]) == (20, 30, 40)
    assert tuple(frame[170, 310]) == (80, 90, 100)
    assert tuple(frame[170, 10]) == (0, 0, 0)
    assert tuple(frame[10, 310]) == (0, 0, 0)


def test_replay_video_final_zoom_labels_stay_in_black_quadrants() -> None:
    image_module = pytest.importorskip("PIL.Image")
    image_draw_module = pytest.importorskip("PIL.ImageDraw")
    module = _load_replay_video_script_module()
    image = image_module.new("RGB", (1280, 720), color=(0, 0, 0))
    draw = image_draw_module.Draw(image)

    final_xy, zoom_xy, zoom_font = module._final_zoom_label_layout(
        draw,
        width=1280,
        height=720,
        tile_width=640,
        tile_height=360,
        zoom_text="zoom 1/5",
    )
    zoom_bbox = draw.textbbox(zoom_xy, "zoom 1/5", font=zoom_font, stroke_width=2)

    assert final_xy[1] > 360
    assert zoom_xy[1] > 0
    assert zoom_bbox[3] < 360


def test_replay_advance_catches_up_generated_frames_between_renders() -> None:
    module = _load_replay_script_module()
    frames = [
        {"frame_id": "capture_0"},
        {"frame_id": "motion_0", "replay_generated": True, "replay_duration_s": 0.001},
        {"frame_id": "motion_1", "replay_generated": True, "replay_duration_s": 0.001},
        {"frame_id": "capture_1"},
        {"frame_id": "motion_2", "replay_generated": True, "replay_duration_s": 0.001},
        {"frame_id": "capture_2"},
    ]

    result = module._advance_replay_index(
        frames,
        current_index=0,
        finished=False,
        motion_target_index=3,
        elapsed_s=0.003,
        default_duration_s=1.2,
        playback_speed=1.0,
        loop=False,
        force_first_step=True,
    )

    assert result.current_index == 3
    assert result.motion_target_index is None
    assert result.reached_indices == (1, 2, 3)
    assert result.elapsed_remainder_s == pytest.approx(0.0)


def test_replay_advance_applies_playback_speed_multiplier() -> None:
    module = _load_replay_script_module()
    frames = [
        {"frame_id": "capture_0", "replay_duration_s": 0.04},
        {"frame_id": "capture_1", "replay_duration_s": 0.04},
    ]

    result = module._advance_replay_index(
        frames,
        current_index=0,
        finished=False,
        motion_target_index=None,
        elapsed_s=0.02,
        default_duration_s=1.2,
        playback_speed=2.0,
        loop=False,
    )

    assert result.current_index == 1
    assert result.reached_indices == (1,)


def test_replay_motion_uses_shortest_path_for_unlimited_hinge_joints() -> None:
    module = _load_replay_script_module()
    model = _FakeMotionModel(nq=1, limited=(0,))

    delta = module._joint_space_delta(model, [0.05], [6.25])

    assert delta == pytest.approx([-0.08318530717958605])


def test_replay_keyframes_use_nearest_equivalent_unlimited_hinge_angle() -> None:
    module = _load_replay_script_module()
    mujoco = _FakeMujoco()
    model = _FakeMotionModel(nq=1, limited=(0,))
    data = _FakeMotionData(nq=1)
    frames = [
        {"frame_id": "a", "phase": "survey", "qpos": [math.radians(20.0)]},
        {"frame_id": "b", "phase": "survey", "qpos": [math.radians(320.0)]},
    ]

    display_frames, summary = module._build_collision_checked_replay_motion(
        mujoco,
        model,
        data,
        frames,
        max_joint_step_rad=0.2,
        interpolated_frame_duration_s=0.04,
    )

    assert display_frames[-1]["frame_id"] == "b"
    assert display_frames[-1]["qpos"][0] == pytest.approx(math.radians(-40.0), abs=1e-6)
    assert display_frames[-1]["replay_qpos_normalized"] is True
    assert display_frames[-1]["replay_qpos_normalization_reason"] == "nearest_equivalent_unlimited_hinge"
    assert summary.continuous_joint_adjusted_keyframe_count == 1
    assert summary.continuous_joint_adjusted_joint_count == 1
    assert summary.continuous_joint_max_adjustment_rad == pytest.approx(2.0 * math.pi, abs=1e-6)


def test_replay_keyframes_do_not_wrap_limited_hinge_angles() -> None:
    module = _load_replay_script_module()
    mujoco = _FakeMujoco()
    model = _FakeMotionModel(nq=1, limited=(1,))
    data = _FakeMotionData(nq=1)
    frames = [
        {"frame_id": "a", "phase": "survey", "qpos": [math.radians(20.0)]},
        {"frame_id": "b", "phase": "survey", "qpos": [math.radians(320.0)]},
    ]

    display_frames, summary = module._build_collision_checked_replay_motion(
        mujoco,
        model,
        data,
        frames,
        max_joint_step_rad=0.2,
        interpolated_frame_duration_s=0.04,
    )

    assert display_frames[-1]["qpos"][0] == pytest.approx(math.radians(320.0), abs=1e-6)
    assert "replay_qpos_normalized" not in display_frames[-1]
    assert summary.continuous_joint_adjusted_keyframe_count == 0
    assert summary.continuous_joint_adjusted_joint_count == 0


def test_replay_motion_fails_when_collision_check_finds_robot_contact() -> None:
    module = _load_replay_script_module()
    mujoco = _FakeMujoco(
        body_names={1: "gen3_link", 2: "tank_root"},
        geom_names={0: "gen3_geom", 1: "tank_geom"},
    )
    model = _FakeMotionModel(nq=1, body_parentid=[0, 0, 0], geom_bodyid=[1, 2])
    data = _FakeMotionData(
        nq=1,
        contacts=[_FakeContact(geom1=0, geom2=1, dist=-0.001)],
    )
    frames = [
        {"frame_id": "a", "phase": "survey", "qpos": [0.0]},
        {"frame_id": "b", "phase": "rough", "qpos": [0.1]},
    ]

    with pytest.raises(SystemExit, match="collision check failed"):
        module._build_collision_checked_replay_motion(
            mujoco,
            model,
            data,
            frames,
            max_joint_step_rad=0.1,
            interpolated_frame_duration_s=0.04,
        )


def _write_task1_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "task1" / "20260707T000000Z_seed101"
    layout_path = run_dir / "layout" / "target_object_poses.json"
    layout_path.parent.mkdir(parents=True, exist_ok=True)
    layout_path.write_text(
        json.dumps(
            {
                "schema_version": "target_object_poses_v1",
                "seed": 101,
                "base_height_m": 0.03,
                "placement_bounds": {"x_min": 0.15, "x_max": 0.85, "y_min": 0.15, "y_max": 0.85},
                "objects": [],
            }
        ),
        encoding="utf-8",
    )
    workspace = {
        "x_min": 0.15,
        "x_max": 0.85,
        "y_min": 0.15,
        "y_max": 0.85,
        "bottom_z_m": 0.03,
        "tank_opening_z_m": 0.50,
        "opening_clearance_m": 0.035,
    }
    common = {
        "task1_run_dir": str(run_dir),
        "layout_snapshot_path": str(layout_path),
        "scene_model_path": "examples/mujoco/gen3_with_tank.xml",
        "camera_name": "wrist",
        "workspace": workspace,
        "image_size": [640, 480],
    }
    _write_report(
        run_dir / "survey" / "survey_report.json",
        {
            **common,
            "schema_version": "task1_survey_report_v1",
            "stage": "survey",
            "status": "success",
            "report_path": str(run_dir / "survey" / "survey_report.json"),
            "views": [
                {
                    "view_id": "survey_0000",
                    "status": "success",
                    "actual_qpos": [0.1, 0.2, 0.3],
                    "actual_camera_position_world": [0.3, 0.4, 0.42],
                    "rgb_image_path": str(run_dir / "survey" / "images" / "survey_0000_rgb.png"),
                }
            ],
        },
    )
    _write_report(
        run_dir / "rough" / "rough_report.json",
        {
            **common,
            "schema_version": "task1_rough_report_v1",
            "stage": "rough",
            "status": "success",
            "report_path": str(run_dir / "rough" / "rough_report.json"),
            "views": [
                {
                    "view_id": "rough_candidate_001_00",
                    "status": "success",
                    "fixed_qpos": [0.4, 0.5, 0.6],
                    "candidate_id": "candidate_001",
                    "candidate_rough_position_world": [0.31, 0.32, 0.03],
                    "rgb_image_path": str(run_dir / "rough" / "images" / "rough_candidate_001_00_rgb.png"),
                }
            ],
        },
    )
    _write_report(
        run_dir / "final" / "final_report.json",
        {
            **common,
            "schema_version": "task1_final_report_v1",
            "stage": "final",
            "status": "success",
            "report_path": str(run_dir / "final" / "final_report.json"),
            "object_captures": [
                {
                    "status": "confirmed",
                    "target": {
                        "object_id": "rough_object_001",
                        "target_role": "primary",
                        "source_status": "stable",
                        "class_name": "notebook",
                        "position_world": [0.31, 0.32, 0.03],
                    },
                    "entry_validation": [
                        {
                            "view_id": "final_rough_object_001_entry_00",
                            "status": "success",
                            "actual_qpos": [0.7, 0.8, 0.9],
                        }
                    ],
                    "view": {
                        "view_id": "final_rough_object_001",
                        "status": "success",
                        "actual_qpos": [1.0, 1.1, 1.2],
                        "rgb_image_path": str(run_dir / "final" / "images" / "final_rough_object_001_rgb.png"),
                    },
                    "final_image_path": str(run_dir / "final" / "images" / "final_rough_object_001_rgb.png"),
                }
            ],
        },
    )
    return run_dir


def _write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class _FakeModel:
    def __init__(self, *, extent: float, fovy: float = 45.0) -> None:
        self.stat = type("FakeStat", (), {"extent": extent})()
        self.vis = type("FakeVis", (), {"global_": type("FakeGlobalVis", (), {"fovy": fovy})()})()


class _FakeMotionModel:
    def __init__(
        self,
        *,
        nq: int,
        limited: tuple[int, ...] | None = None,
        body_parentid: list[int] | None = None,
        geom_bodyid: list[int] | None = None,
    ) -> None:
        self.nq = nq
        self.nv = nq
        self.nu = nq
        self.njnt = nq
        self.jnt_qposadr = list(range(nq))
        self.jnt_type = [3] * nq
        limited = limited or tuple(0 for _ in range(nq))
        self.jnt_limited = list(limited)
        self.body_parentid = body_parentid or [0]
        self.geom_bodyid = geom_bodyid or []


class _FakeMotionData:
    def __init__(self, *, nq: int, contacts: list["_FakeContact"] | None = None) -> None:
        self.qpos = [0.0] * nq
        self.qvel = [0.0] * nq
        self.ctrl = [0.0] * nq
        self.contact = contacts or []
        self.ncon = len(self.contact)


class _FakeContact:
    def __init__(self, *, geom1: int, geom2: int, dist: float) -> None:
        self.geom1 = geom1
        self.geom2 = geom2
        self.dist = dist


class _FakeMujoco:
    class mjtObj:
        mjOBJ_GEOM = "geom"
        mjOBJ_BODY = "body"

    def __init__(self, *, body_names: dict[int, str] | None = None, geom_names: dict[int, str] | None = None) -> None:
        self._body_names = body_names or {}
        self._geom_names = geom_names or {}

    def mj_forward(self, _model, _data) -> None:
        return None

    def mj_id2name(self, _model, obj_type, obj_id: int) -> str | None:
        if obj_type == self.mjtObj.mjOBJ_BODY:
            return self._body_names.get(int(obj_id))
        if obj_type == self.mjtObj.mjOBJ_GEOM:
            return self._geom_names.get(int(obj_id))
        return None


def _load_replay_script_module():
    scripts_dir = str(REPO_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "replay_task1_output_for_test",
        REPO_ROOT / "scripts" / "replay_task1_output.py",
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_replay_video_script_module():
    scripts_dir = str(REPO_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "render_task1_replay_video_for_test",
        REPO_ROOT / "scripts" / "render_task1_replay_video.py",
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
