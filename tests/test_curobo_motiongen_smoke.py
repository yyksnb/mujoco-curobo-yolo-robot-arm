import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_curobo_motiongen_smoke_generates_report(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_curobo_motiongen_smoke.py"),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report_path = tmp_path / "reports" / "curobo_motiongen_smoke_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert "motiongen_api_called=" in result.stdout
    assert report_path.exists()
    assert "success" in report
    assert "motiongen_api_called" in report
    assert "failure_category" in report
    assert "trajectory_available" in report
    assert report["is_demo_config"] is True
    if report["success"]:
        assert report["motiongen_api_called"] is True
        assert report["trajectory_available"] is True
        trajectory_path = tmp_path / "trajectories" / "curobo_motiongen_smoke_trajectory.json"
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        assert trajectory["joint_names"]
        assert trajectory["waypoints"]
