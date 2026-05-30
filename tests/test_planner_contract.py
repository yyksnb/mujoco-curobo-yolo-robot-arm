from robot_arm_pipeline.perception import fake_bodex_grasp_targets, fake_yolo_object_poses
from robot_arm_pipeline.pipeline import default_robot_state
from robot_arm_pipeline.planning import MockPlanner
from robot_arm_pipeline.scene import build_collision_scene


def test_mock_planner_returns_trajectory_like_object() -> None:
    object_poses = fake_yolo_object_poses()
    scene = build_collision_scene(object_poses)
    target = fake_bodex_grasp_targets(object_poses)[0]

    trajectory = MockPlanner().plan(scene, default_robot_state(), target)

    assert trajectory.planner_name == "mock_planner"
    assert trajectory.target_object_id == target.object_id
    assert len(trajectory.waypoints) >= 2
    assert trajectory.waypoints[-1].time_s > 0

