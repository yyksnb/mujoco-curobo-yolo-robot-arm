from __future__ import annotations

import argparse
import json
from pathlib import Path

from check_curobo_environment import check_environment


def main() -> None:
    parser = argparse.ArgumentParser(description="Save the cuRobo environment check report.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    report = check_environment()
    report_path = _save_report(args.output_dir, report)
    print(f"recommended_next_step={report['recommended_next_step']}")
    print(f"report={report_path}")


def _save_report(output_dir: Path, report: dict[str, object]) -> Path:
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "curobo_environment_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
