import json
import subprocess
import sys
from pathlib import Path

from robot_arm_pipeline.perception import fake_bodex_grasp_targets, fake_yolo_object_poses
from robot_arm_pipeline.pipeline import default_robot_state
from robot_arm_pipeline.planning import MockPlanner
from robot_arm_pipeline.planning.curobo_conversions import (
    collision_scene_to_curobo_world_config,
    joint_trajectory_to_planned_trajectory,
    transform_to_curobo_pose,
)
from robot_arm_pipeline.planning.curobo_planner import CUROBO_UNAVAILABLE_MESSAGE, CuroboPlanner
from robot_arm_pipeline.scene import build_collision_scene
from robot_arm_pipeline.stage3_pipeline import run_stage3_pipeline
from robot_arm_pipeline.types import ObjectPose, PlanningRequest


REPO_ROOT = Path(__file__).resolve().parents[1]
BODEX_EXAMPLE = REPO_ROOT / "examples" / "bodex_grasp_target.json"
YOLO_EXAMPLE = REPO_ROOT / "examples" / "yolo_detection.json"
MUJOCO_MODEL = REPO_ROOT / "examples" / "mujoco" / "minimal_six_joint_arm.xml"


def test_curobo_planner_unavailable_does_not_crash(monkeypatch) -> None:
    object_pose = fake_yolo_object_poses()[0]
    grasp_target = fake_bodex_grasp_targets((object_pose,))[0]
    request = PlanningRequest(
        object_pose=object_pose,
        grasp_target=grasp_target,
        robot_state=default_robot_state(),
    )
    monkeypatch.setattr(CuroboPlanner, "is_available", staticmethod(lambda: False))
    monkeypatch.setattr(
        CuroboPlanner,
        "_load_optional_modules",
        lambda self: type("Modules", (), {"curobo": None, "torch": None, "message": CUROBO_UNAVAILABLE_MESSAGE})(),
    )

    result = CuroboPlanner().plan(request)

    assert result.success is False
    assert result.trajectory is None
    assert "cuRobo is not installed or not configured" in result.message


def test_run_curobo_planner_script_generates_failure_report(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_curobo_planner.py"),
            "--bodex-grasp",
            str(BODEX_EXAMPLE),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report_path = tmp_path / "reports" / "curobo_planner_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert "success=False" in result.stdout
    assert report["success"] is False
    assert report["trajectory_available"] is False
    assert report["message"]


def test_mock_planner_still_works() -> None:
    object_pose = fake_yolo_object_poses()[0]
    grasp_target = fake_bodex_grasp_targets((object_pose,))[0]
    scene = build_collision_scene((object_pose,))

    trajectory = MockPlanner().plan(scene, default_robot_state(), grasp_target)

    assert trajectory.planner_name == "mock_planner"
    assert trajectory.waypoints


def test_stage3_pipeline_still_uses_mock_planner(tmp_path: Path) -> None:
    result = run_stage3_pipeline(
        yolo_detection_path=YOLO_EXAMPLE,
        bodex_grasp_path=BODEX_EXAMPLE,
        mujoco_model_path=MUJOCO_MODEL,
        output_dir=tmp_path,
    )
    trajectory = json.loads((tmp_path / "trajectories" / "object_001_trajectory.json").read_text(encoding="utf-8"))

    assert result.success is True
    assert trajectory["planner_name"] == "mock_planner"


def test_curobo_conversion_helpers_validate_schema() -> None:
    transform = (
        (1.0, 0.0, 0.0, 0.45),
        (0.0, 1.0, 0.0, 0.05),
        (0.0, 0.0, 1.0, 0.20),
        (0.0, 0.0, 0.0, 1.0),
    )
    pose = transform_to_curobo_pose(transform)

    assert pose == [0.45, 0.05, 0.2, 1.0, 0.0, 0.0, 0.0]

    object_pose = ObjectPose(object_id="object_001", label="cube", pose=fake_yolo_object_poses()[0].pose)
    world = collision_scene_to_curobo_world_config(build_collision_scene((object_pose,)))
    assert "object_001" in world["cuboid"]

    trajectory = joint_trajectory_to_planned_trajectory(
        joint_names=("joint_1", "joint_2"),
        joint_positions=((0.0, 0.1), (0.2, 0.3)),
        target_object_id="object_001",
        interpolation_dt=0.1,
    )
    assert trajectory.planner_name == "curobo_planner"
    assert trajectory.waypoints[-1].time_s == 0.1
