from __future__ import annotations

import json
from pathlib import Path

from robot_arm_pipeline.types import EvaluationReport, ExecutionResult, PlannedTrajectory


def evaluate_execution(
    trajectory: PlannedTrajectory,
    execution_result: ExecutionResult,
) -> EvaluationReport:
    return EvaluationReport(
        success=execution_result.success,
        target_object_id=trajectory.target_object_id,
        waypoint_count=len(trajectory.waypoints),
        duration_s=execution_result.duration_s,
        message=execution_result.message,
    )


def save_trajectory(trajectory: PlannedTrajectory, output_dir: Path) -> Path:
    trajectory_dir = output_dir / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    path = trajectory_dir / f"{trajectory.target_object_id}_trajectory.json"
    path.write_text(json.dumps(trajectory.to_dict(), indent=2), encoding="utf-8")
    return path


def save_evaluation_report(report: EvaluationReport, output_dir: Path) -> Path:
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{report.target_object_id}_evaluation.json"
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return path

