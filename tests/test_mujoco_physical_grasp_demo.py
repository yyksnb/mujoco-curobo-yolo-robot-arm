import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_mujoco_physical_grasp_demo_generates_report(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_mujoco_physical_grasp_demo.py"),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report_path = tmp_path / "reports" / "mujoco_physical_grasp_demo_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert "simulation_ran=" in result.stdout
    assert report_path.exists()
    assert (tmp_path / "logs" / "mujoco_physical_grasp_demo_log.csv").exists()
    assert "success" in report
    assert report["message"]
    assert "cube_initial_z" in report
    assert "cube_final_z" in report
    assert "lifted_distance" in report
    assert report["simulation_ran"] is True
    assert report["is_demo_model"] is True
