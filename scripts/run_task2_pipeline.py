from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from task2.pipeline import build_task2_pipeline  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Task2 pipeline from Task1 observations to grasp targets."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Task1 final object_observation_manifest.json.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/task2/targeting.json"),
    )
    parser.add_argument(
        "--end-effector",
        help="Target end_effector_id; defaults to the Targeting config selection.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/task2/targeting_report.json"),
    )
    args = parser.parse_args()

    pipeline = build_task2_pipeline(
        _resolve_from_repo(args.config),
        end_effector_id=args.end_effector,
    )
    report = pipeline.run(
        _resolve_from_repo(args.manifest),
        output_path=_resolve_from_repo(args.output),
    )
    print(json.dumps(report.summary_dict(), ensure_ascii=False))
    raise SystemExit(0 if report.status == "success" else 1)


def _resolve_from_repo(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


if __name__ == "__main__":
    main()
