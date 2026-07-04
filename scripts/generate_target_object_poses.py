from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from robot_arm_pipeline.scene.target_object_layout import (  # noqa: E402
    DEFAULT_BASE_HEIGHT_M,
    DEFAULT_COLLISION_MARGIN_M,
    DEFAULT_MODEL_PATH,
    DEFAULT_OBJECT_COUNT,
    DEFAULT_OUTPUT_DIR,
    PlacementBounds,
    make_random_target_object_pose_payload,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random non-overlapping target object poses in the tank.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="MuJoCo scene XML/MJCF path.")
    parser.add_argument("--output", type=Path, default=None, help="Explicit output JSON path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory used when --output is omitted.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed. A seed is generated when omitted.")
    parser.add_argument("--count", type=int, default=DEFAULT_OBJECT_COUNT, help="Number of target objects to select.")
    parser.add_argument(
        "--base-height",
        type=float,
        default=DEFAULT_BASE_HEIGHT_M,
        help="Common z coordinate for generated object body poses, in meters.",
    )
    parser.add_argument("--x-min", type=float, default=PlacementBounds.x_min)
    parser.add_argument("--x-max", type=float, default=PlacementBounds.x_max)
    parser.add_argument("--y-min", type=float, default=PlacementBounds.y_min)
    parser.add_argument("--y-max", type=float, default=PlacementBounds.y_max)
    parser.add_argument(
        "--collision-margin",
        type=float,
        default=DEFAULT_COLLISION_MARGIN_M,
        help="Extra 2D footprint separation margin, in meters.",
    )
    parser.add_argument("--max-attempts-per-object", type=int, default=6000)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output JSON.")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
    created_at = datetime.now(timezone.utc)
    created_utc = created_at.isoformat()
    timestamp = created_at.strftime("%Y%m%dT%H%M%S%fZ")
    output_path = args.output or args.output_dir / f"{timestamp}_seed{seed}_target_object_poses.json"

    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing output file: {output_path}")

    payload = make_random_target_object_pose_payload(
        model_path=args.model,
        seed=seed,
        created_utc=created_utc,
        object_count=args.count,
        base_height_m=args.base_height,
        bounds=PlacementBounds(
            x_min=args.x_min,
            x_max=args.x_max,
            y_min=args.y_min,
            y_max=args.y_max,
        ),
        collision_margin_m=args.collision_margin,
        max_attempts_per_object=args.max_attempts_per_object,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote={output_path} seed={seed} count={len(payload['objects'])}")


if __name__ == "__main__":
    main()
