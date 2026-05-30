import json
from pathlib import Path

from robot_arm_pipeline.pipeline import run_mock_pipeline


def test_mock_pipeline_runs_end_to_end(tmp_path: Path) -> None:
    report = run_mock_pipeline(tmp_path)

    trajectory_path = tmp_path / "trajectories" / "object_001_trajectory.json"
    report_path = tmp_path / "reports" / "object_001_evaluation.json"

    assert report.success is True
    assert trajectory_path.exists()
    assert report_path.exists()
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved_report["success"] is True

