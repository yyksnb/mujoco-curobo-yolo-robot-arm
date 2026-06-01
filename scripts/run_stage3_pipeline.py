from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from robot_arm_pipeline.stage3_pipeline import run_stage3_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stage 3 upstream-interface pipeline.")
    parser.add_argument("--yolo-detection", type=Path, default=Path("examples/yolo_detection.json"))
    parser.add_argument("--bodex-grasp", type=Path, default=Path("examples/bodex_grasp_target.json"))
    parser.add_argument("--model", type=Path, default=Path("examples/mujoco/minimal_six_joint_arm.xml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    result = run_stage3_pipeline(
        yolo_detection_path=args.yolo_detection,
        bodex_grasp_path=args.bodex_grasp,
        mujoco_model_path=args.model,
        output_dir=args.output_dir,
    )
    print(
        f"success={result.success} object_id={result.object_id} "
        f"planning={result.planning_success} execution={result.execution_success} "
        f"message={result.message}"
    )


if __name__ == "__main__":
    main()
