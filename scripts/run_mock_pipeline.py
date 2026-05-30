from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from robot_arm_pipeline.pipeline import run_mock_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stage 1 mock robot arm pipeline.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    report = run_mock_pipeline(args.output_dir)
    print(f"success={report.success} target={report.target_object_id} duration_s={report.duration_s}")


if __name__ == "__main__":
    main()
