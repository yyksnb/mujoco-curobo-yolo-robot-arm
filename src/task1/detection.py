from __future__ import annotations

import hashlib
import math
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from robot_arm_pipeline.perception import load_stage3_yolo_profile, run_ultralytics_yolo_inference
from task1.vision import Detection2D, RgbdFrame


@dataclass(frozen=True)
class DetectionBatch:
    detections: tuple[Detection2D, ...]
    source_report: dict[str, Any]


@dataclass(frozen=True)
class YoloInferenceConfig:
    image_size: int
    confidence_threshold: float
    iou_threshold: float
    class_agnostic_nms: bool
    device: str | None
    max_detections: int | None


class DetectionConfig(Protocol):
    config_path: Path
    profile_path: Path
    inference: YoloInferenceConfig


class Detector(Protocol):
    def detect(self, frame: RgbdFrame) -> DetectionBatch: ...

    def source_metadata(self) -> dict[str, Any]: ...


YoloInference = Callable[..., dict[str, Any]]


class YoloDetector:
    def __init__(
        self,
        config: DetectionConfig,
        inference: YoloInference = run_ultralytics_yolo_inference,
    ) -> None:
        self.config = config
        self.profile = load_stage3_yolo_profile(config.profile_path)
        self.inference = inference

    def detect(self, frame: RgbdFrame) -> DetectionBatch:
        if frame.rgb_path is None:
            raise ValueError(f"{frame.view_id}: RGB image is required for YOLO inference")
        payload = self.inference(
            image_path=Path(frame.rgb_path),
            profile_path=self.config.profile_path,
            camera_name=self.profile.default_camera_name,
            confidence_threshold=self.config.inference.confidence_threshold,
            iou_threshold=self.config.inference.iou_threshold,
            class_agnostic_nms=self.config.inference.class_agnostic_nms,
            image_size=self.config.inference.image_size,
            device=self.config.inference.device,
            max_detections=self.config.inference.max_detections,
        )
        raw_detections = payload.get("detections")
        if not isinstance(raw_detections, list):
            raise ValueError(f"{frame.view_id}: YOLO output must contain a detections list")

        detections = []
        source_detections = []
        for index, raw in enumerate(raw_detections):
            if not isinstance(raw, dict):
                raise ValueError(f"{frame.view_id}: YOLO detections must be JSON objects")
            mapping = self.profile.mapping_for_detection(raw)
            bbox = raw.get("bbox_xyxy")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"{frame.view_id}: YOLO bbox_xyxy must contain four numbers")
            confidence = float(raw["confidence"])
            detection_id = f"{frame.view_id}:yolo:{index:04d}"
            detections.append(
                Detection2D(
                    detection_id=detection_id,
                    bbox_xyxy=tuple(float(value) for value in bbox),
                    confidence=confidence,
                    class_name=mapping.stage3_class_name,
                    display_name=mapping.yolo_name,
                )
            )
            source_detections.append(
                {
                    "detection_id": detection_id,
                    "source_class_id": mapping.yolo_id,
                    "source_class_name": mapping.yolo_name,
                    "class_name": mapping.stage3_class_name,
                    "confidence": confidence,
                    "bbox_xyxy": [float(value) for value in bbox],
                }
            )

        return DetectionBatch(
            detections=tuple(detections),
            source_report={
                "view_id": frame.view_id,
                "image_path": frame.rgb_path,
                "detection_count": len(detections),
                "detections": source_detections,
            },
        )

    def source_metadata(self) -> dict[str, Any]:
        return {
            "name": "repository_ultralytics_yolo",
            "production": True,
            "profile": self.profile.profile_name,
            "detection_config_path": str(self.config.config_path),
            "profile_path": str(self.config.profile_path),
            "model_path": str(self.profile.model_path),
            "inference": {
                "image_size": self.config.inference.image_size,
                "confidence_threshold": self.config.inference.confidence_threshold,
                "iou_threshold": self.config.inference.iou_threshold,
                "class_agnostic_nms": self.config.inference.class_agnostic_nms,
                "device": self.config.inference.device,
                "max_detections": self.config.inference.max_detections,
            },
        }


_BOX_COLORS = ("#e53935", "#00897b", "#1e88e5", "#f9a825", "#8e24aa", "#6d4c41")
_CJK_FONT_NAMES = (
    "NotoSansCJKsc-Regular.otf",
    "NotoSansCJK-Regular.ttc",
    "NotoSansSC-Regular.otf",
    "msyh.ttc",
    "simhei.ttf",
    "PingFang.ttc",
)


def render_detection_overlay(
    rgb_path: Path,
    detections: Sequence[Detection2D],
    output_path: Path,
) -> Path:
    """Render normalized survey detections without changing inference results."""
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required to render survey detection overlays") from exc

    if not rgb_path.is_file():
        raise FileNotFoundError(f"survey RGB image does not exist: {rgb_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(rgb_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = max(2, min(image.size) // 270)
    font = _load_annotation_font(max(14, min(image.size) // 45))
    padding = max(3, line_width)

    for detection in detections:
        box = tuple(float(value) for value in detection.bbox_xyxy)
        if not all(math.isfinite(value) for value in box) or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"{detection.detection_id}: cannot render invalid bbox_xyxy {box}")
        color = _class_color(detection.class_name)
        draw.rectangle(box, outline=color, width=line_width)

        display_name = detection.display_name or detection.class_name or "unknown"
        label = f"{display_name} {detection.confidence:.2f}"
        label_box = draw.textbbox((0, 0), label, font=font)
        label_width = label_box[2] - label_box[0]
        label_height = label_box[3] - label_box[1]
        label_x = max(0.0, min(box[0], image.width - label_width - 2 * padding))
        label_y = box[1] - label_height - 2 * padding
        if label_y < 0:
            label_y = min(float(image.height - label_height - 2 * padding), box[1])
        draw.rectangle(
            (
                label_x,
                label_y,
                label_x + label_width + 2 * padding,
                label_y + label_height + 2 * padding,
            ),
            fill=color,
        )
        draw.text(
            (label_x + padding, label_y + padding - label_box[1]),
            label,
            fill="white",
            font=font,
        )

    image.save(output_path)
    return output_path


def _class_color(class_name: str | None) -> str:
    key = (class_name or "unknown").encode("utf-8")
    index = int.from_bytes(hashlib.sha256(key).digest()[:2], "big") % len(_BOX_COLORS)
    return _BOX_COLORS[index]


@lru_cache(maxsize=None)
def _load_annotation_font(size: int):
    from PIL import ImageFont

    for font_name in _CJK_FONT_NAMES:
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue

    fontconfig = shutil.which("fc-match")
    if fontconfig is not None:
        result = subprocess.run(
            [fontconfig, "-f", "%{file}\n", ":lang=zh-cn"],
            check=True,
            capture_output=True,
            text=True,
        )
        font_path = next((Path(line) for line in result.stdout.splitlines() if line), None)
        if font_path is not None and font_path.is_file():
            return ImageFont.truetype(font_path, size=size)

    raise RuntimeError(
        "survey annotation requires a Simplified Chinese font such as Noto Sans CJK SC"
    )
