from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from robot_arm_pipeline.execution.mujoco_executor import MujocoExecutor
from robot_arm_pipeline.execution.mujoco_executor_gui import MujocoGuiExecutor


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
    parser.add_argument(
        "--keyframe-name",
        type=str,
        default=None,
        help="Optional MuJoCo keyframe to apply before execution or GUI replay.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--gui", action="store_true", help="Replay the trajectory in the MuJoCo GUI.")
    parser.add_argument("--replay-speed", type=float, default=1.0, help="Replay speed multiplier for GUI mode.")
    args = parser.parse_args()

    executor = (
        MujocoGuiExecutor(
            output_dir=args.output_dir,
            keyframe_name=args.keyframe_name,
            replay_speed=args.replay_speed,
        )
        if args.gui
        else MujocoExecutor(output_dir=args.output_dir, keyframe_name=args.keyframe_name)
    )
    report = executor.execute(args.trajectory, args.model)
    print(
        f"success={report.success} steps={report.num_steps} "
        f"duration_s={report.duration_s} message={report.message}"
    )


if __name__ == "__main__":
    main()
