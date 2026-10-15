from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from robot_arm_pipeline.perception.yolo_adapter import (  # noqa: E402
    convert_yolo_raw_file_to_stage3_detection,
)


DEFAULT_CONFIG = Path("configs/yolo/stage3_default.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw YOLO detections through a Stage3 YOLO profile into the Stage3 detection contract."
    )
    parser.add_argument("--raw", required=True, type=Path, help="Raw YOLO detection JSON.")
    parser.add_argument("--output", required=True, type=Path, help="Output Stage3 detection JSON.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Stage3 YOLO profile YAML.")
    parser.add_argument("--detection-index", type=int, default=0, help="Which raw detection to export.")
    parser.add_argument(
        "--allow-mock-fields",
        action="store_true",
        help="Fill missing confidence/bbox/T_world_object from profile defaults for early recognition steps.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing output file: {args.output}")

    detection = convert_yolo_raw_file_to_stage3_detection(
        args.raw,
        args.config,
        detection_index=args.detection_index,
        allow_mock_fields=args.allow_mock_fields,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(detection, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
