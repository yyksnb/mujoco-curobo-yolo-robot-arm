from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from robot_arm_pipeline.perception.yolo_local_inference import run_ultralytics_yolo_inference  # noqa: E402
from robot_arm_pipeline.perception.yolo_adapter import (  # noqa: E402
    convert_yolo_raw_payload_to_stage3_detection,
    load_stage3_yolo_profile,
)


DEFAULT_CONFIG = Path("configs/yolo/stage3_default.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local Ultralytics YOLO inference through a Stage3 YOLO profile and save raw detections."
    )
    parser.add_argument("--image", required=True, type=Path, help="Input image path.")
    parser.add_argument("--output", required=True, type=Path, help="Output raw YOLO detection JSON.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Stage3 YOLO profile YAML.")
    parser.add_argument("--camera-name", default=None, help="Camera name to store in the raw output.")
    parser.add_argument("--conf", type=float, default=None, help="Override YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=None, help="Override YOLO IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=None, help="Override YOLO image size.")
    parser.add_argument("--device", default=None, help="Optional Ultralytics device, e.g. cpu or 0.")
    parser.add_argument("--max-det", type=int, default=None, help="Optional maximum number of detections.")
    parser.add_argument("--stage3-output", type=Path, default=None, help="Optional Stage3 detection JSON for one detection.")
    parser.add_argument("--stage3-detection-index", type=int, default=0, help="Raw detection index used for --stage3-output.")
    parser.add_argument(
        "--allow-mock-fields",
        action="store_true",
        help="Allow mock T_world_object when writing --stage3-output.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files.")
    args = parser.parse_args()

    _ensure_writable(args.output, overwrite=args.overwrite)
    if args.stage3_output is not None:
        _ensure_writable(args.stage3_output, overwrite=args.overwrite)

    payload = run_ultralytics_yolo_inference(
        image_path=args.image,
        profile_path=args.config,
        camera_name=args.camera_name,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        image_size=args.imgsz,
        device=args.device,
        max_detections=args.max_det,
    )
    _write_json(args.output, payload)
    print(f"wrote={args.output} detections={len(payload['detections'])}")

    if args.stage3_output is not None:
        profile = load_stage3_yolo_profile(args.config)
        stage3_detection = convert_yolo_raw_payload_to_stage3_detection(
            payload,
            profile,
            detection_index=args.stage3_detection_index,
            allow_mock_fields=args.allow_mock_fields,
        )
        _write_json(args.stage3_output, stage3_detection)
        print(f"wrote={args.stage3_output} stage3_detection_index={args.stage3_detection_index}")


def _ensure_writable(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"Refusing to overwrite existing output file: {path}")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
