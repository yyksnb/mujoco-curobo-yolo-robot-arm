from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from robot_arm_pipeline.perception.object_observation import (
    COMPONENT_SELECTION_SEED_SOURCE,
    DEPTH_ENCODING,
    DISTORTION_MODEL,
    FINAL_OBJECT_OBSERVATION_REVISION,
    FINAL_OBJECT_OBSERVATION_SCHEMA,
    MASK_ENCODING,
    MASK_SOURCE,
    OPTICAL_AXIS_CONVENTION,
    POSITION_PRIOR_SEMANTICS,
    RGB_COLOR_SPACE,
    RGB_ENCODING,
    TRANSFORM_SEMANTICS,
    ensure_event_identifier,
    ensure_portable_identifier,
    final_observation_manifest_status,
    observation_interface_status,
    validate_final_object_observation_entry,
    validate_final_object_observation_manifest,
)
from task1.vision import (
    CameraIntrinsics,
    CandidateLocalizationPolicy,
    Detection2D,
    Matrix4,
    RgbdFrame,
)


FINAL_OBJECT_OBSERVATION_MANIFEST = "object_observation_manifest.json"


@dataclass(frozen=True)
class FinalObservationExportPolicy:
    enabled: bool
    world_frame_id: str
    camera_optical_frame_id: str
    support_frame_id: str
    mask_source: str
    connected_component_connectivity: int


class FinalObservationExporter:
    """Export Task1 observations without estimating object or grasp poses."""

    def __init__(
        self,
        *,
        output_dir: Path,
        policy: FinalObservationExportPolicy,
        localization_policy: CandidateLocalizationPolicy,
        maximum_seed_xy_distance_m: float,
    ) -> None:
        self.output_dir = output_dir.resolve()
        self.policy = policy
        self.localization_policy = localization_policy
        self.maximum_seed_xy_distance_m = maximum_seed_xy_distance_m
        self._objects: dict[str, dict[str, Any]] = {}
        self._failures: dict[str, dict[str, Any]] = {}

    def export(
        self,
        *,
        candidate_id: str,
        survey_bottom_position_world: tuple[float, float, float],
        final_bottom_position_world: tuple[float, float, float],
        detection: Detection2D,
        frame: RgbdFrame,
        rgb_path: Path,
    ) -> dict[str, Any]:
        candidate_id = ensure_portable_identifier(candidate_id, "candidate_id")
        if candidate_id in self._objects or candidate_id in self._failures:
            raise ValueError(f"duplicate Final object observation: {candidate_id}")
        class_name = ensure_portable_identifier(
            detection.class_name,
            f"{candidate_id}.class_name",
        )
        ensure_event_identifier(detection.detection_id, f"{candidate_id}.detection_id")
        ensure_event_identifier(frame.view_id, f"{candidate_id}.view_id")

        rgb_path = rgb_path.resolve()
        if frame.rgb_path is not None and Path(frame.rgb_path).resolve() != rgb_path:
            raise ValueError(f"{candidate_id}: RGB path does not belong to the captured RGB-D frame")
        rgb_relative = _relative_artifact_path(rgb_path, self.output_dir)
        depth = _validated_depth(frame)
        transform = _rigid_transform(frame.T_world_camera_optical, "T_world_camera_optical")
        mask, mask_diagnostics = depth_foreground_component_mask(
            frame=frame,
            detection=detection,
            seed_position_world=survey_bottom_position_world,
            localization_policy=self.localization_policy,
            connectivity=self.policy.connected_component_connectivity,
            maximum_seed_xy_distance_m=self.maximum_seed_xy_distance_m,
        )

        object_dir = self.output_dir / "observations" / candidate_id
        depth_path = object_dir / "depth_m.npy"
        mask_path = object_dir / "mask.png"
        _write_npy(depth_path, depth.astype(np.float32, copy=False))
        _write_mask_png(mask_path, mask)

        observation_id = f"final_{candidate_id}"
        intrinsics = frame.intrinsics
        entry = {
            "observation_id": observation_id,
            "candidate_id": candidate_id,
            "view_id": frame.view_id,
            "detection": {
                "detection_id": detection.detection_id,
                "class_name": class_name,
                "display_name": detection.display_name,
                "confidence": float(detection.confidence),
                "bbox_xyxy": [float(value) for value in detection.bbox_xyxy],
            },
            "position_priors": {
                "survey_bottom_position_world": [
                    float(value) for value in survey_bottom_position_world
                ],
                "final_bottom_position_world": [
                    float(value) for value in final_bottom_position_world
                ],
                "semantics": POSITION_PRIOR_SEMANTICS,
            },
            "rgb": {
                "path": rgb_relative,
                "sha256": _sha256_file(rgb_path),
                "encoding": RGB_ENCODING,
                "color_space": RGB_COLOR_SPACE,
            },
            "depth": {
                "path": _relative_artifact_path(depth_path, self.output_dir),
                "sha256": _sha256_file(depth_path),
                "encoding": DEPTH_ENCODING,
                "unit": "m",
                "registered_to_rgb": True,
                "invalid_sample_rule": "non_finite_or_non_positive",
            },
            "mask": {
                "path": _relative_artifact_path(mask_path, self.output_dir),
                "sha256": _sha256_file(mask_path),
                "encoding": MASK_ENCODING,
                "source": self.policy.mask_source,
                "foreground_value": 255,
                "background_value": 0,
                "uses_layout_or_simulation_segmentation": False,
            },
            "camera": {
                "world_frame_id": self.policy.world_frame_id,
                "camera_optical_frame_id": self.policy.camera_optical_frame_id,
                "optical_axis_convention": OPTICAL_AXIS_CONVENTION,
                "intrinsics": {
                    "width": intrinsics.width,
                    "height": intrinsics.height,
                    "fx": intrinsics.fx,
                    "fy": intrinsics.fy,
                    "cx": intrinsics.cx,
                    "cy": intrinsics.cy,
                    "distortion_model": DISTORTION_MODEL,
                    "distortion_coefficients": [],
                },
                "T_world_camera_optical": [list(row) for row in transform],
                "transform_semantics": TRANSFORM_SEMANTICS,
            },
            "support_plane": {
                "frame_id": self.policy.support_frame_id,
                "normal_world": [0.0, 0.0, 1.0],
                "z_world_m": float(frame.ground_z_m),
            },
            "quality": mask_diagnostics,
        }
        validate_final_object_observation_entry(self.output_dir, entry, 0)
        self._objects[candidate_id] = entry
        return {
            "status": "success",
            "observation_id": observation_id,
            "mask_source": self.policy.mask_source,
            "foreground_pixel_count": mask_diagnostics["selected_component_pixel_count"],
        }

    def record_failure(self, candidate_id: str, message: str) -> dict[str, Any]:
        candidate_id = ensure_portable_identifier(candidate_id, "candidate_id")
        if candidate_id in self._objects:
            raise ValueError(f"cannot reject an exported Final observation: {candidate_id}")
        if candidate_id in self._failures:
            raise ValueError(f"duplicate Final observation failure: {candidate_id}")
        failure = {
            "candidate_id": candidate_id,
            "failure_stage": "object_observation_export",
            "message": message,
        }
        self._failures[candidate_id] = failure
        return {"status": "failed", **failure}

    def finalize(
        self,
        *,
        eligible_candidate_ids: list[str],
        source_final_report: Path,
        source_final: dict[str, Any],
    ) -> dict[str, Any]:
        eligible_candidate_ids = [
            ensure_portable_identifier(value, "eligible_candidate_id")
            for value in eligible_candidate_ids
        ]
        if len(eligible_candidate_ids) != len(set(eligible_candidate_ids)):
            raise ValueError("Final observation eligible candidate ids must be unique")
        missing = [
            candidate_id
            for candidate_id in eligible_candidate_ids
            if candidate_id not in self._objects and candidate_id not in self._failures
        ]
        for candidate_id in missing:
            self.record_failure(
                candidate_id,
                "Final recognized the candidate but did not export or reject its observation.",
            )
        ordered_objects = [
            self._objects[candidate_id]
            for candidate_id in eligible_candidate_ids
            if candidate_id in self._objects
        ]
        ordered_failures = [
            self._failures[candidate_id]
            for candidate_id in eligible_candidate_ids
            if candidate_id in self._failures
        ]
        exported_count = len(ordered_objects)
        failure_count = len(ordered_failures)
        interface_status = observation_interface_status(
            exported_object_count=exported_count,
            failed_object_count=failure_count,
        )
        manifest_status = final_observation_manifest_status(
            source_final_status=str(source_final.get("status")),
            interface_status=interface_status,
            object_count=exported_count,
        )
        manifest_path = self.output_dir / FINAL_OBJECT_OBSERVATION_MANIFEST
        payload = {
            "schema": FINAL_OBJECT_OBSERVATION_SCHEMA,
            "revision": FINAL_OBJECT_OBSERVATION_REVISION,
            "producer_stage": "task1_final",
            "status": manifest_status,
            "message": (
                f"Exported {exported_count} of {len(eligible_candidate_ids)} "
                f"recognized Final observations; source Final status={source_final.get('status')}."
            ),
            "source_final_report": _relative_artifact_path(
                source_final_report.resolve(), self.output_dir
            ),
            "source_final": source_final,
            "interface": {
                "status": interface_status,
                "eligible_object_count": len(eligible_candidate_ids),
                "exported_object_count": exported_count,
                "failed_object_count": failure_count,
            },
            "object_count": exported_count,
            "export_failure_count": failure_count,
            "objects": ordered_objects,
            "failures": ordered_failures,
            "boundary": {
                "provided_by_task1": [
                    "same_frame_rgb",
                    "registered_metric_depth",
                    "production_foreground_mask",
                    "camera_intrinsics",
                    "T_world_camera_optical",
                    "detection_and_position_priors",
                    "source_final_failure_semantics",
                    "observation_quality",
                ],
                "not_provided_by_task1": [
                    "T_world_object",
                    "object_model_frame_or_scale",
                    "stable_pose_or_symmetry",
                    "object_state_estimate",
                    "end_effector_frame_or_grasp_target",
                ],
            },
        }
        try:
            validate_final_object_observation_manifest(payload, root=self.output_dir)
            _write_json(manifest_path, payload)
        except Exception as exc:
            return {
                "enabled": True,
                "status": "failed",
                "failure_stage": "object_observation_manifest",
                "manifest_status": "failed",
                "source_final_status": source_final.get("status"),
                "manifest_path": None,
                "eligible_object_count": len(eligible_candidate_ids),
                "prepared_object_count": exported_count,
                "exported_object_count": 0,
                "failed_object_count": len(eligible_candidate_ids),
                "message": (
                    "Could not validate or persist the object observation manifest: "
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        return {
            "enabled": True,
            "status": interface_status,
            "failure_stage": None,
            "manifest_status": manifest_status,
            "source_final_status": source_final.get("status"),
            "manifest_path": str(manifest_path),
            "eligible_object_count": len(eligible_candidate_ids),
            "prepared_object_count": exported_count,
            "exported_object_count": exported_count,
            "failed_object_count": failure_count,
            "message": payload["message"],
        }


def depth_foreground_component_mask(
    *,
    frame: RgbdFrame,
    detection: Detection2D,
    seed_position_world: tuple[float, float, float],
    localization_policy: CandidateLocalizationPolicy,
    connectivity: int,
    maximum_seed_xy_distance_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if connectivity not in {4, 8}:
        raise ValueError("connected component connectivity must be 4 or 8")
    if maximum_seed_xy_distance_m <= 0.0:
        raise ValueError("maximum seed XY distance must be positive")
    depth = _validated_depth(frame)
    left, top, right, bottom = _pixel_bounds(detection.bbox_xyxy, frame.intrinsics)
    crop_depth = depth[top:bottom, left:right]
    ys, xs = np.indices(crop_depth.shape)
    image_x = xs + left
    image_y = ys + top
    valid = np.isfinite(crop_depth) & (crop_depth > localization_policy.min_depth_m)
    if localization_policy.max_depth_m is not None:
        valid &= crop_depth <= localization_policy.max_depth_m
    if int(np.count_nonzero(valid)) < localization_policy.min_valid_depth_samples:
        raise ValueError(f"{detection.detection_id}: insufficient valid depth for mask export")

    points_world = _project_to_world(
        image_x[valid],
        image_y[valid],
        crop_depth[valid],
        frame.intrinsics,
        frame.T_world_camera_optical,
    )
    heights = points_world[:, 2] - frame.ground_z_m
    selected_height = (
        (heights >= localization_policy.min_height_above_floor_m)
        & (heights <= localization_policy.max_height_above_floor_m)
    )
    foreground = np.zeros(crop_depth.shape, dtype=bool)
    foreground[image_y[valid][selected_height] - top, image_x[valid][selected_height] - left] = True
    world_xy = np.full((*crop_depth.shape, 2), np.nan, dtype=float)
    world_xy[image_y[valid] - top, image_x[valid] - left] = points_world[:, :2]

    structure = ndimage.generate_binary_structure(2, 2 if connectivity == 8 else 1)
    labels, label_count = ndimage.label(foreground, structure=structure)
    seed_xy = np.asarray(seed_position_world[:2], dtype=float)
    components: list[tuple[float, int, int]] = []
    counts = np.bincount(labels.ravel(), minlength=label_count + 1)
    for label_id, component_slice in enumerate(ndimage.find_objects(labels), start=1):
        pixel_count = int(counts[label_id])
        if pixel_count < localization_policy.min_valid_depth_samples or component_slice is None:
            continue
        local_component = labels[component_slice] == label_id
        center_xy = np.median(world_xy[component_slice][local_component], axis=0)
        components.append((float(np.linalg.norm(center_xy - seed_xy)), label_id, pixel_count))
    if not components:
        raise ValueError(f"{detection.detection_id}: no connected depth foreground component")
    seed_distance, selected_label, selected_count = min(components)
    if seed_distance > maximum_seed_xy_distance_m:
        raise ValueError(
            f"{detection.detection_id}: nearest depth component is {seed_distance:.6f} m "
            "from the Survey candidate"
        )

    mask = np.zeros(depth.shape, dtype=bool)
    mask[top:bottom, left:right] = labels == selected_label
    return mask, {
        "mask_source": MASK_SOURCE,
        "bbox_clipped_xyxy": [left, top, right, bottom],
        "valid_depth_pixel_count": int(np.count_nonzero(valid)),
        "foreground_pixel_count_before_component_selection": int(np.count_nonzero(foreground)),
        "eligible_component_count": len(components),
        "selected_component_pixel_count": selected_count,
        "selected_component_seed_xy_distance_m": seed_distance,
        "component_selection_seed_source": COMPONENT_SELECTION_SEED_SOURCE,
    }


def _validated_depth(frame: RgbdFrame) -> np.ndarray:
    depth = np.asarray(frame.depth_m)
    expected = (frame.intrinsics.height, frame.intrinsics.width)
    if depth.ndim != 2 or depth.shape != expected:
        raise ValueError(f"{frame.view_id}: depth shape {depth.shape} does not match {expected}")
    if not np.issubdtype(depth.dtype, np.floating):
        raise ValueError(f"{frame.view_id}: depth must use a floating-point dtype")
    return depth


def _pixel_bounds(
    bbox: tuple[float, float, float, float], intrinsics: CameraIntrinsics
) -> tuple[int, int, int, int]:
    if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
        raise ValueError("bbox_xyxy must contain four finite values")
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox_xyxy must have positive area")
    left = max(0, min(intrinsics.width - 1, int(math.floor(x1))))
    right = max(left + 1, min(intrinsics.width, int(math.ceil(x2))))
    top = max(0, min(intrinsics.height - 1, int(math.floor(y1))))
    bottom = max(top + 1, min(intrinsics.height, int(math.ceil(y2))))
    return left, top, right, bottom


def _project_to_world(
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    transform: Matrix4,
) -> np.ndarray:
    matrix = np.asarray(_rigid_transform(transform, "T_world_camera_optical"), dtype=float)
    camera_x = (pixel_x.astype(float) - intrinsics.cx) * depth_m / intrinsics.fx
    camera_y = (pixel_y.astype(float) - intrinsics.cy) * depth_m / intrinsics.fy
    points_camera = np.stack((camera_x, camera_y, depth_m), axis=1)
    return points_camera @ matrix[:3, :3].T + matrix[:3, 3]


def _rigid_transform(value: Any, field: str) -> Matrix4:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{field} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-7):
        raise ValueError(f"{field} must have homogeneous bottom row [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise ValueError(f"{field} rotation must be orthonormal with determinant +1")
    return tuple(tuple(float(item) for item in row) for row in matrix)  # type: ignore[return-value]


def _write_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    temporary.replace(path)


def _write_mask_png(path: Path, mask: np.ndarray) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to export Final foreground masks") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(temporary, format="PNG")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _relative_artifact_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Final observation artifact must stay inside {root}: {path}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
