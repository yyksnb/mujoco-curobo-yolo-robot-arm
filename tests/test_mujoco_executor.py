import json
import subprocess
import sys
from pathlib import Path

import pytest
from robot_arm_pipeline.evaluation import save_trajectory
from robot_arm_pipeline.execution.mujoco_executor import MujocoExecutor
from robot_arm_pipeline.execution.mujoco_executor_gui import MujocoGuiExecutor
from robot_arm_pipeline.execution.mujoco_executor_gui import _GuiKeyMap
from robot_arm_pipeline.execution.mujoco_executor_gui import _ReplayControl
from robot_arm_pipeline.pipeline import default_robot_state
from robot_arm_pipeline.pipeline import run_mock_pipeline
from robot_arm_pipeline.perception import fake_bodex_grasp_targets, fake_yolo_object_poses
from robot_arm_pipeline.planning import MockPlanner
from robot_arm_pipeline.scene import build_collision_scene


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_MODEL = REPO_ROOT / "examples" / "mujoco" / "minimal_six_joint_arm.xml"
GEN3_MODEL = REPO_ROOT / "examples" / "mujoco" / "gen3_with_tank.xml"


def test_mujoco_executor_missing_model_does_not_crash(tmp_path: Path) -> None:
    run_mock_pipeline(tmp_path)
    trajectory_path = tmp_path / "trajectories" / "object_001_trajectory.json"

    report = MujocoExecutor(output_dir=tmp_path).execute(trajectory_path, tmp_path / "missing.xml")

    assert report.success is False
    assert "model file does not exist" in report.message
    assert (tmp_path / "reports" / "mujoco_execution_report.json").exists()


def test_mujoco_executor_reports_availability() -> None:
    assert isinstance(MujocoExecutor.is_available(), bool)


def test_run_mujoco_executor_script_generates_execution_report(tmp_path: Path) -> None:
    run_mock_pipeline(tmp_path)
    trajectory_path = tmp_path / "trajectories" / "object_001_trajectory.json"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_mujoco_executor.py"),
            "--trajectory",
            str(trajectory_path),
            "--model",
            str(EXAMPLE_MODEL),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report_path = tmp_path / "reports" / "mujoco_execution_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert "success=" in result.stdout
    assert report_path.exists()
    assert report["model_path"] == str(EXAMPLE_MODEL)
    if MujocoExecutor.is_available():
        assert report["success"] is True
        assert report["num_steps"] > 0
        assert (tmp_path / "logs" / "mujoco_execution_log.csv").exists()
    else:
        assert report["success"] is False
        assert "pip install mujoco" in report["message"]


def test_mujoco_executor_runs_gen3_scene_with_prefixed_joint_names(tmp_path: Path) -> None:
    object_poses = fake_yolo_object_poses()
    scene = build_collision_scene(object_poses)
    grasp_target = fake_bodex_grasp_targets(object_poses)[0]
    trajectory = MockPlanner().plan(scene, default_robot_state(), grasp_target)
    trajectory_path = save_trajectory(trajectory, tmp_path)

    report = MujocoExecutor(output_dir=tmp_path).execute(trajectory_path, GEN3_MODEL)

    assert report.model_path == str(GEN3_MODEL)
    assert (tmp_path / "reports" / "mujoco_execution_report.json").exists()
    if MujocoExecutor.is_available():
        assert report.success is True
        assert report.num_steps == len(trajectory.waypoints)
        assert (tmp_path / "logs" / "mujoco_execution_log.csv").exists()
    else:
        assert report.success is False
        assert "pip install mujoco" in report.message


def test_mujoco_executor_rejects_mismatched_waypoint_joint_count(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "bad_trajectory.json"
    trajectory_path.write_text(
        json.dumps(
            {
                "joint_names": ["joint_1", "joint_2"],
                "waypoints": [{"time_s": 0.0, "joint_positions": [0.0]}],
                "planner_name": "mock_planner",
                "target_object_id": "object_001",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report = MujocoExecutor(output_dir=tmp_path).execute(trajectory_path, EXAMPLE_MODEL)

    assert report.success is False
    assert "joint_positions must match the length of joint_names" in report.message


def test_mujoco_gui_replay_control_handles_space_n_and_r() -> None:
    keymap = _GuiKeyMap(space=32, n=78, r=82)
    control = _ReplayControl(paused=True)

    assert control.snapshot() == (True, 0)
    control.handle_key(32, keymap)
    assert control.snapshot() == (False, 0)
    control.handle_key(78, keymap)
    assert control.snapshot() == (True, 1)
    assert control.consume_segment_request() is True
    control.handle_key(82, keymap)
    assert control.snapshot() == (True, 0)
    assert control.consume_reset_request() is True
    assert control.consume_reset_request() is False


def test_mujoco_gui_executor_step_by_step_advances_on_n(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    if not MujocoGuiExecutor.is_available():
        pytest.skip("MuJoCo GUI viewer not available")

    import mujoco
    import mujoco.viewer

    object_poses = fake_yolo_object_poses()
    scene = build_collision_scene(object_poses)
    grasp_target = fake_bodex_grasp_targets(object_poses)[0]
    trajectory = MockPlanner().plan(scene, default_robot_state(), grasp_target)
    trajectory_path = save_trajectory(trajectory, tmp_path)

    control_events: list[int] = []
    viewer_instances: list[object] = []

    class FakeViewer:
        def __init__(self, model, data, n_events: int):
            self.model = model
            self.data = data
            self.sync_calls = 0
            self.running = True
            self.running_checks = 0
            self.key_callback = None
            self.n_events_remaining = n_events

        def __enter__(self):
            viewer_instances.append(self)
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def is_running(self):
            self.running_checks += 1
            if self.running_checks > 5000:
                self.running = False
            return self.running

        def sync(self):
            self.sync_calls += 1
            if self.key_callback is not None and self.n_events_remaining > 0 and self.sync_calls >= 2:
                self.key_callback(int(mujoco.viewer.glfw.KEY_N))
                self.n_events_remaining -= 1

    def fake_launch_passive(model, data, *, key_callback=None, **kwargs):
        viewer = FakeViewer(model, data, n_events=len(trajectory.waypoints))
        viewer.key_callback = key_callback
        if key_callback is not None:
            control_events.append(1)
        return viewer

    monkeypatch.setattr(mujoco.viewer, "launch_passive", fake_launch_passive)
    monkeypatch.setattr("robot_arm_pipeline.execution.mujoco_executor_gui.time.sleep", lambda seconds: None)

    executor = MujocoGuiExecutor(
        output_dir=tmp_path,
        final_hold_seconds=0.0,
        gui_poll_seconds=0.01,
        step_by_step=True,
    )
    monkeypatch.setattr(executor, "is_available", lambda: True)
    report = executor.execute(trajectory_path, EXAMPLE_MODEL)

    assert report.success is True
    assert report.num_steps == len(trajectory.waypoints)
    assert control_events
    assert viewer_instances
    assert viewer_instances[0].sync_calls > 0
