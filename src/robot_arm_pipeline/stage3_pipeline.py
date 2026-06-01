from __future__ import annotations

import json
from pathlib import Path

from robot_arm_pipeline.evaluation import save_trajectory
from robot_arm_pipeline.execution.mujoco_executor import MujocoExecutionReport, MujocoExecutor
from robot_arm_pipeline.perception import (
    load_bodex_grasp_target,
    load_yolo_detection,
    object_pose_from_detection,
)
from robot_arm_pipeline.pipeline import default_robot_state
from robot_arm_pipeline.planning import MockPlanner
from robot_arm_pipeline.scene import build_collision_scene
from robot_arm_pipeline.types import EvaluationResult, PlanningRequest, PlanningResult


def run_stage3_pipeline(
    *,
    yolo_detection_path: Path | str = Path("examples/yolo_detection.json"),
    bodex_grasp_path: Path | str = Path("examples/bodex_grasp_target.json"),
    mujoco_model_path: Path | str = Path("examples/mujoco/minimal_six_joint_arm.xml"),
    output_dir: Path | str = Path("outputs"),
) -> EvaluationResult:
    output_dir = Path(output_dir)

    detection = load_yolo_detection(yolo_detection_path)
    object_pose = object_pose_from_detection(detection)
    grasp_target = load_bodex_grasp_target(bodex_grasp_path)

    if object_pose.object_id != grasp_target.object_id:
        raise ValueError(
            "YOLO detection object_id does not match BODex grasp target object_id: "
            f"{object_pose.object_id!r} != {grasp_target.object_id!r}"
        )

    request = PlanningRequest(
        object_pose=object_pose,
        grasp_target=grasp_target,
        robot_state=default_robot_state(),
    )
    planning_result = _plan(request)
    if planning_result.trajectory is None:
        return _save_final_evaluation(
            output_dir,
            EvaluationResult(
                success=False,
                object_id=object_pose.object_id,
                planning_success=False,
                execution_success=False,
                trajectory_path="",
                execution_report_path="",
                message=planning_result.message,
            ),
        )

    trajectory_path = save_trajectory(planning_result.trajectory, output_dir)
    execution_report = MujocoExecutor(output_dir=output_dir).execute(trajectory_path, mujoco_model_path)
    return _save_final_evaluation(
        output_dir,
        _evaluate_stage3(
            object_id=object_pose.object_id,
            planning_result=planning_result,
            execution_report=execution_report,
            trajectory_path=trajectory_path,
            output_dir=output_dir,
        ),
    )


def _plan(request: PlanningRequest) -> PlanningResult:
    scene = build_collision_scene((request.object_pose,))
    try:
        trajectory = MockPlanner().plan(scene, request.robot_state, request.grasp_target)
    except ValueError as exc:
        return PlanningResult(success=False, trajectory=None, message=str(exc))
    return PlanningResult(success=True, trajectory=trajectory, message="mock planning completed")


def _evaluate_stage3(
    *,
    object_id: str,
    planning_result: PlanningResult,
    execution_report: MujocoExecutionReport,
    trajectory_path: Path,
    output_dir: Path,
) -> EvaluationResult:
    success = planning_result.success and execution_report.success
    execution_report_path = output_dir / "reports" / "mujoco_execution_report.json"
    message = "Stage 3 pipeline completed" if success else execution_report.message
    return EvaluationResult(
        success=success,
        object_id=object_id,
        planning_success=planning_result.success,
        execution_success=execution_report.success,
        trajectory_path=str(trajectory_path),
        execution_report_path=str(execution_report_path),
        message=message,
    )


def _save_final_evaluation(output_dir: Path, result: EvaluationResult) -> EvaluationResult:
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "stage3_evaluation_report.json"
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result
