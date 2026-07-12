from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from task1.survey.localization import (
    CameraIntrinsics,
    CandidateLocalizationPolicy,
    RgbdFrame,
    SurveyObservation,
    fuse_observations_with_report,
    localize_detection,
)
from task1.survey.route import SURVEY_VIEWS
from task1.survey.detection import SurveyDetector, render_detection_overlay


CAPTURE_SCHEMA = "task1_rgbd_survey_capture"


def localize_capture_manifest(
    manifest_path: Path,
    detector: SurveyDetector,
    annotated_image_dir: Path,
    policy: CandidateLocalizationPolicy = CandidateLocalizationPolicy(),
) -> dict[str, Any]:
    frames = load_capture_manifest(manifest_path)
    observations: list[SurveyObservation] = []
    failures: list[dict[str, str]] = []
    detection_reports: list[dict[str, Any]] = []
    for frame in frames:
        batch = detector.detect(frame)
        if frame.rgb_path is None:
            raise ValueError(f"{frame.view_id}: RGB image is required to render detections")
        annotated_rgb_path = render_detection_overlay(
            Path(frame.rgb_path),
            batch.detections,
            annotated_image_dir / f"{frame.view_id}.png",
        )
        detection_reports.append(
            {**batch.source_report, "annotated_rgb_path": str(annotated_rgb_path)}
        )
        for detection in batch.detections:
            try:
                observations.append(localize_detection(frame, detection, policy))
            except ValueError as exc:
                failures.append(
                    {"view_id": frame.view_id, "detection_id": detection.detection_id, "message": str(exc)}
                )
    fusion_result = fuse_observations_with_report(observations, policy)
    return {
        "status": "success" if not failures else "partial",
        "source_manifest_path": str(manifest_path),
        "detection_source": detector.source_metadata(),
        "candidate_localization_policy": policy.to_dict(),
        "detection_reports": detection_reports,
        "observations": [observation.to_dict() for observation in observations],
        "candidates": [candidate.to_dict() for candidate in fusion_result.candidates],
        "fusion_diagnostics": fusion_result.diagnostics,
        "localization_failures": failures,
    }


def load_capture_manifest(manifest_path: Path) -> tuple[RgbdFrame, ...]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != CAPTURE_SCHEMA:
        raise ValueError(f"unsupported survey capture schema: {payload.get('schema')}")
    raw_views = payload.get("views")
    if not isinstance(raw_views, list):
        raise ValueError("survey capture manifest must contain a views list")
    required_view_ids = {view.view_id for view in SURVEY_VIEWS}
    supplied_view_ids = {str(view.get("view_id")) for view in raw_views if isinstance(view, dict)}
    if supplied_view_ids != required_view_ids:
        missing = sorted(required_view_ids - supplied_view_ids)
        extra = sorted(supplied_view_ids - required_view_ids)
        raise ValueError(f"survey capture manifest view mismatch; missing={missing}, extra={extra}")

    root = manifest_path.resolve().parent
    parsed: dict[str, RgbdFrame] = {}
    for raw_view in raw_views:
        if not isinstance(raw_view, dict):
            raise ValueError("survey capture views must be JSON objects")
        view_id = str(raw_view["view_id"])
        if "detections" in raw_view:
            raise ValueError(
                f"{view_id}: capture manifest must not contain detections; survey runs repository YOLO"
            )
        depth_path = _resolve_file(root, raw_view.get("depth_path"), "depth_path")
        depth = np.load(depth_path, allow_pickle=False)
        intrinsics = _intrinsics(raw_view.get("intrinsics"))
        frame = RgbdFrame(
            view_id=view_id,
            depth_m=depth,
            intrinsics=intrinsics,
            T_world_camera_optical=_transform(raw_view.get("T_world_camera_optical")),
            ground_z_m=float(raw_view["ground_z_m"]),
            rgb_path=str(_resolve_file(root, raw_view.get("rgb_path"), "rgb_path")),
            depth_path=str(depth_path),
        )
        parsed[view_id] = frame
    return tuple(parsed[view.view_id] for view in SURVEY_VIEWS)


def _intrinsics(payload: Any) -> CameraIntrinsics:
    if not isinstance(payload, dict):
        raise ValueError("survey capture intrinsics must be a JSON object")
    return CameraIntrinsics(
        width=int(payload["width"]),
        height=int(payload["height"]),
        fx=float(payload["fx"]),
        fy=float(payload["fy"]),
        cx=float(payload["cx"]),
        cy=float(payload["cy"]),
    )


def _transform(payload: Any) -> tuple[tuple[float, float, float, float], ...]:
    if not isinstance(payload, list) or len(payload) != 4:
        raise ValueError("T_world_camera_optical must be a 4x4 matrix")
    rows = tuple(tuple(float(value) for value in row) for row in payload if isinstance(row, list) and len(row) == 4)
    if len(rows) != 4:
        raise ValueError("T_world_camera_optical must be a 4x4 matrix")
    return rows


def _resolve_file(root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"survey capture {field} must be a non-empty path")
    path = Path(value)
    resolved = path if path.is_absolute() else root / path
    if not resolved.exists():
        raise FileNotFoundError(f"survey capture {field} does not exist: {resolved}")
    return resolved
