from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from robot_arm_pipeline.task1.attribution import (  # noqa: E402
    AttributionConfig,
    analyze_task1_regression,
    write_attribution_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Attribute Task1 object losses across saved pipeline reports.")
    parser.add_argument("input_dir", type=Path, help="Directory containing one or more complete Task1 run directories.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--yolo-profile", type=Path, default=Path("configs/yolo/stage3_default.yaml"))
    parser.add_argument("--footprint-margin-m", type=float, default=0.04)
    parser.add_argument("--center-match-radius-m", type=float, default=0.08)
    args = parser.parse_args()

    report = analyze_task1_regression(
        args.input_dir,
        yolo_profile_path=args.yolo_profile,
        config=AttributionConfig(
            footprint_margin_m=args.footprint_margin_m,
            center_match_radius_m=args.center_match_radius_m,
        ),
    )
    json_path, csv_path = write_attribution_outputs(report, args.output_dir)
    print(
        f"runs={report['run_count']} objects={report['object_count']} "
        f"correct_stable={report['correct_final_stable_count']} "
        f"reported_stable={report['reported_final_stable_count']} lost={report['lost_object_count']}"
    )
    print(f"loss_stage_counts={report['loss_stage_counts']}")
    print(f"responsibility_counts={report['responsibility_counts']}")
    print(f"json={json_path}")
    print(f"csv={csv_path}")


if __name__ == "__main__":
    main()
