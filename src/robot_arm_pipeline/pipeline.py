from __future__ import annotations

from pathlib import Path

from robot_arm_pipeline.evaluation import evaluate_execution, save_evaluation_report, save_trajectory
from robot_arm_pipeline.execution import MockExecutor
from robot_arm_pipeline.perception import fake_bodex_grasp_targets, fake_yolo_object_poses
from robot_arm_pipeline.planning import MockPlanner
from robot_arm_pipeline.scene import build_collision_scene
from robot_arm_pipeline.types import EvaluationReport, RobotState


DEFAULT_OUTPUT_DIR = Path("outputs")


def default_robot_state() -> RobotState:
    return RobotState(
        joint_names=("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"),
        joint_positions=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )


def run_mock_pipeline(output_dir: Path = DEFAULT_OUTPUT_DIR) -> EvaluationReport:
    object_poses = fake_yolo_object_poses()
    grasp_targets = fake_bodex_grasp_targets(object_poses)
    scene = build_collision_scene(object_poses)

    planner = MockPlanner()
    trajectory = planner.plan(scene, default_robot_state(), grasp_targets[0])
    save_trajectory(trajectory, output_dir)

    executor = MockExecutor()
    execution_result = executor.execute(trajectory)

    report = evaluate_execution(trajectory, execution_result)
    save_evaluation_report(report, output_dir)
    return report

