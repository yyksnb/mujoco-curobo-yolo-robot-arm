from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from robot_arm_pipeline.execution.mujoco_executor import MujocoExecutor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stage 2 MuJoCo trajectory executor.")
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path("outputs/trajectories/object_001_trajectory.json"),
        help="Path to a Stage 1 trajectory JSON file.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("examples/mujoco/minimal_six_joint_arm.xml"),
        help="Path to a MuJoCo MJCF/XML model.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    report = MujocoExecutor(output_dir=args.output_dir).execute(args.trajectory, args.model)
    print(
        f"success={report.success} steps={report.num_steps} "
        f"duration_s={report.duration_s} message={report.message}"
    )


if __name__ == "__main__":
    main()
