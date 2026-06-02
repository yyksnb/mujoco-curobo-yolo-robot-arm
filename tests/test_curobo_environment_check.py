import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_check_curobo_environment_script_does_not_crash() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_curobo_environment.py")],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout)

    assert "recommended_next_step" in report
    assert isinstance(report["torch_installed"], bool)
    assert isinstance(report["curobo_installed"], bool)
    assert isinstance(report["can_allocate_cuda_tensor"], bool)


def test_check_curobo_environment_recommends_next_step() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_curobo_environment.py")],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout)

    assert report["recommended_next_step"]
