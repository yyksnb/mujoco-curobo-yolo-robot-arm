from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Print saved Stage 1 evaluation reports.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    report_dir = args.output_dir / "reports"
    reports = sorted(report_dir.glob("*_evaluation.json"))
    if not reports:
        raise SystemExit(f"no evaluation reports found under {report_dir}")

    for report_path in reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print(
            f"{report_path}: success={report['success']} "
            f"target={report['target_object_id']} duration_s={report['duration_s']}"
        )


if __name__ == "__main__":
    main()

