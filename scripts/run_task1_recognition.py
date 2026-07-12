from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from task1.pipeline import PipelineOptions, Task1Pipeline  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Task1 recognition pipeline.")
    parser.add_argument("--step", help="Run one registered stage; omit to run all implemented stages.")
    parser.add_argument("--seed", type=int, help="Generate a deterministic MuJoCo object layout.")
    parser.add_argument("--layout", type=Path, help="Use an existing target_object_pose_layout JSON.")
    parser.add_argument("--capture-manifest", type=Path, help="Use recorded/real RGB-D instead of MuJoCo capture.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task1"))
    parser.add_argument(
        "--retain-survey-depth",
        action="store_true",
        help="Persist MuJoCo survey depth frames as float32 NPY artifacts for diagnosis or data retention.",
    )
    parser.add_argument(
        "--survey-config",
        type=Path,
        default=Path("configs/task1/survey_detection.yaml"),
        help="Configure YOLO inference, RGB-D localization, fusion, and evaluation.",
    )
    args = parser.parse_args()

    result = Task1Pipeline(
        PipelineOptions(
            repo_root=REPO_ROOT,
            output_dir=args.output_dir,
            seed=args.seed,
            step=args.step,
            layout_path=args.layout,
            capture_manifest=args.capture_manifest,
            survey_config=args.survey_config,
            retain_survey_depth=args.retain_survey_depth,
        )
    ).run()
    print(json.dumps(result.to_dict()))
    raise SystemExit(0 if result.status == "success" else 1)


if __name__ == "__main__":
    main()
