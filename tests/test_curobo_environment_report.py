import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_save_curobo_environment_report_does_not_crash(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "save_curobo_environment_report.py"),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report_path = tmp_path / "reports" / "curobo_environment_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert "recommended_next_step=" in result.stdout
    assert report_path.exists()
    assert "torch_installed" in report
    assert "cuda_available" in report
    assert "curobo_installed" in report
    assert "recommended_next_step" in report


def test_saved_curobo_environment_report_has_boolean_fields(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "save_curobo_environment_report.py"),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads((tmp_path / "reports" / "curobo_environment_report.json").read_text(encoding="utf-8"))

    assert isinstance(report["torch_installed"], bool)
    assert isinstance(report["cuda_available"], bool)
    assert isinstance(report["curobo_installed"], bool)
    assert report["recommended_next_step"]
