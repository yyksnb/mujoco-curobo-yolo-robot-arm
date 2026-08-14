#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from task2 import Task2Pipeline
from task2.artifacts import load_request_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        choices=("full", "targeting", "planning", "execution"),
        default="full",
    )
    parser.add_argument("--request", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/task2/pipeline.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "outputs/task2/runs",
    )
    args = parser.parse_args()
    if args.request is not None and args.run_dir is not None:
        parser.error("--request and --run-dir cannot be used together")
    if args.step in {"full", "targeting"} and args.request is None:
        parser.error(f"--step {args.step} requires --request")
    if args.step in {"planning", "execution"} and args.run_dir is None:
        parser.error(f"--step {args.step} requires --run-dir")

    output_root = args.output_root
    if args.run_dir is not None:
        output_root = args.run_dir.resolve().parent
    pipeline = Task2Pipeline.from_config(args.config, output_root)
    request = load_request_file(args.request) if args.request is not None else None
    result = pipeline._run_cli_step(
        args.step,
        request=request,
        run_dir=args.run_dir,
    )
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
