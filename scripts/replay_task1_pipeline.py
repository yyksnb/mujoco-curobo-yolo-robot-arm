from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from task1.replay import (  # noqa: E402
    find_latest_task1_run,
    load_task1_replay_plan,
    replay_task1_in_mujoco,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay a completed Task1 Survey and Final run in the MuJoCo GUI."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Task1 run directory. Defaults to the latest complete run under outputs/task1.",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Play continuously instead of pausing at the start and after each capture.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_dir = (
        args.run_dir
        if args.run_dir is not None
        else find_latest_task1_run(REPO_ROOT / "outputs" / "task1")
    )
    plan = load_task1_replay_plan(run_dir, repo_root=REPO_ROOT)
    print(f"replay: {run_dir.resolve()} | captures={len(plan.segments)}", flush=True)
    print(
        "controls: Space=pause/resume, F9=play to next capture, F12=reset",
        flush=True,
    )
    replay_task1_in_mujoco(plan, continuous=args.continuous)


if __name__ == "__main__":
    main()
