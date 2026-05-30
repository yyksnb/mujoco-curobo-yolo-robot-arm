from robot_arm_pipeline.execution import MockExecutor
from robot_arm_pipeline.perception import fake_bodex_grasp_targets, fake_yolo_object_poses
from robot_arm_pipeline.pipeline import default_robot_state
from robot_arm_pipeline.planning import MockPlanner
from robot_arm_pipeline.scene import build_collision_scene


def test_mock_executor_returns_execution_result() -> None:
    object_poses = fake_yolo_object_poses()
    scene = build_collision_scene(object_poses)
    target = fake_bodex_grasp_targets(object_poses)[0]
    trajectory = MockPlanner().plan(scene, default_robot_state(), target)

    result = MockExecutor().execute(trajectory)

    assert result.success is True
    assert result.executor_name == "mock_executor"
    assert result.final_state.joint_positions == trajectory.waypoints[-1].joint_positions

