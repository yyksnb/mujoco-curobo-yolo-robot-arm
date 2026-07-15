from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from task1.replay import (  # noqa: E402
    find_latest_task1_run,
    load_task1_replay_plan,
    record_task1_video,
    replay_task1_in_mujoco,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay completed Task1 artifacts in the MuJoCo GUI or an MP4 video."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Task1 run directory. Defaults to the latest complete run under outputs/task1.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--continuous",
        action="store_true",
        help="Play continuously instead of pausing at the start and after each capture.",
    )
    mode.add_argument(
        "--video",
        action="store_true",
        help="Write the complete replay to <run-dir>/replay/task1_pipeline.mp4.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_dir = (
        args.run_dir
        if args.run_dir is not None
        else find_latest_task1_run(
            REPO_ROOT / "outputs" / "task1",
            require_zoom=args.video,
        )
    )
    plan = load_task1_replay_plan(
        run_dir,
        repo_root=REPO_ROOT,
        include_result_stills=args.video,
    )
    print(f"replay: {run_dir.resolve()} | captures={len(plan.segments)}", flush=True)
    if args.video:
        output_path = run_dir.resolve() / "replay" / "task1_pipeline.mp4"
        print(
            "video: "
            f"fps=60 | capture_hold=2s | result_hold=3s | output={output_path}",
            flush=True,
        )
        record_task1_video(plan, output_path)
        print(f"video complete: {output_path}", flush=True)
        return
    print(
        "controls: Space=pause/resume, F9=play to next capture, F12=reset",
        flush=True,
    )
    replay_task1_in_mujoco(plan, continuous=args.continuous)


if __name__ == "__main__":
    main()
