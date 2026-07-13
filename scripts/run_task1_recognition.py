from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from task1.pipeline import PipelineOptions, PipelineStageEvent, Task1Pipeline  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]


def _log_stage_event(event: PipelineStageEvent) -> None:
    if event.elapsed_s is None:
        message = f"[task1] stage={event.stage} status={event.status}"
    else:
        message = (
            f"[task1] stage={event.stage} status={event.status} "
            f"elapsed_s={event.elapsed_s:.3f}"
        )
    print(message, file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Task1 recognition pipeline.")
    parser.add_argument("--step", help="Run one registered stage; omit to run all implemented stages.")
    parser.add_argument("--seed", type=int, help="Generate a deterministic MuJoCo object layout.")
    parser.add_argument("--layout", type=Path, help="Use an existing target_object_pose_layout JSON.")
    parser.add_argument(
        "--survey-report",
        type=Path,
        help="Consume an existing task1_survey_report when running Final independently.",
    )
    parser.add_argument("--capture-manifest", type=Path, help="Use recorded/real RGB-D instead of MuJoCo capture.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task1"))
    parser.add_argument(
        "--final-config",
        type=Path,
        default=Path("configs/task1/final/config.yaml"),
        help="Configure Task1 Final camera geometry, association, and evaluation.",
    )
    parser.add_argument(
        "--retain-survey-depth",
        action="store_true",
        help="Persist MuJoCo survey depth frames as float32 NPY artifacts for diagnosis or data retention.",
    )
    parser.add_argument(
        "--survey-config",
        type=Path,
        default=Path("configs/task1/survey/detection.yaml"),
        help="Configure YOLO inference, RGB-D localization, fusion, and evaluation.",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    try:
        result = Task1Pipeline(
            PipelineOptions(
                repo_root=REPO_ROOT,
                output_dir=args.output_dir,
                seed=args.seed,
                step=args.step,
                layout_path=args.layout,
                capture_manifest=args.capture_manifest,
                survey_config=args.survey_config,
                final_config=args.final_config,
                survey_report_path=args.survey_report,
                retain_survey_depth=args.retain_survey_depth,
            ),
            stage_observer=_log_stage_event,
        ).run()
    except Exception:
        print(
            f"[task1] pipeline status=failed elapsed_s={time.perf_counter() - started:.3f}",
            file=sys.stderr,
            flush=True,
        )
        raise
    print(
        f"[task1] pipeline status={result.status} elapsed_s={time.perf_counter() - started:.3f}",
        file=sys.stderr,
        flush=True,
    )
    print(json.dumps(result.to_dict()))
    raise SystemExit(0 if result.status == "success" else 1)


if __name__ == "__main__":
    main()
