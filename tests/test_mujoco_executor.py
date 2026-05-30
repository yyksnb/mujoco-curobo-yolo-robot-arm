import json
import subprocess
import sys
from pathlib import Path

from robot_arm_pipeline.execution.mujoco_executor import MujocoExecutor
from robot_arm_pipeline.pipeline import run_mock_pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_MODEL = REPO_ROOT / "examples" / "mujoco" / "minimal_six_joint_arm.xml"


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
