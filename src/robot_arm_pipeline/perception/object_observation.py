from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_arm_pipeline.types import TransformMatrix


FINAL_OBJECT_OBSERVATION_SCHEMA = "task1_final_object_observations"
FINAL_OBJECT_OBSERVATION_REVISION = 3
DEPTH_ENCODING = "npy_float32_m"
MASK_ENCODING = "png_uint8_0_255"
MASK_SOURCE = "depth_foreground_component"
POINT_CLOUD_ENCODING = "npz_numeric_arrays_v1"
POINT_CLOUD_SOURCE = "registered_depth_production_mask_back_projection"
RGB_ENCODING = "png_rgb_uint8"
RGB_COLOR_SPACE = "sRGB"
OPTICAL_AXIS_CONVENTION = "x_right_y_down_z_forward"
DISTORTION_MODEL = "none"
TRANSFORM_SEMANTICS = "T_A_B maps coordinates from frame B into frame A"
POSITION_PRIOR_SEMANTICS = (
    "approximate_object_footprint_center_on_support_plane; "
    "initialization_only_not_a_6d_pose"
)
COMPONENT_SELECTION_SEED_SOURCE = "survey_candidate_bottom_position_world"

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_EVENT_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*")
_FRAME_ID = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.-]*(/[A-Za-z0-9][A-Za-z0-9_.-]*)*"
)
_STATUS_VALUES = {"success", "partial", "failed"}
_INTERFACE_STATUS_VALUES = {"success", "partial", "failed"}
_POINT_CLOUD_ARRAYS = {"points_world_m", "colors_rgb_uint8", "pixels_uv"}


@dataclass(frozen=True)
class ObservationCameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class FinalObservationMaskQuality:
    mask_source: str
    bbox_clipped_xyxy: tuple[int, int, int, int]
    valid_depth_pixel_count: int
    foreground_pixel_count_before_component_selection: int
    eligible_component_count: int
    selected_component_pixel_count: int
    selected_component_seed_xy_distance_m: float
    component_selection_seed_source: str


@dataclass(frozen=True)
class FinalObservationPointCloud:
    path: Path
    world_frame_id: str
    points_world_m: np.ndarray
    colors_rgb_uint8: np.ndarray
    pixels_uv: np.ndarray


@dataclass(frozen=True)
class FinalObjectObservation:
    observation_id: str
    candidate_id: str
    view_id: str
    detection_id: str
    class_name: str
    display_name: str | None
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    survey_bottom_position_world: tuple[float, float, float]
    final_bottom_position_world: tuple[float, float, float]
    rgb_path: Path
    depth_path: Path
    mask_path: Path
    depth_m: np.ndarray
    mask: np.ndarray
    intrinsics: ObservationCameraIntrinsics
    T_world_camera_optical: TransformMatrix
    world_frame_id: str
    camera_optical_frame_id: str
    support_frame_id: str
    support_plane_z_world_m: float
    mask_source: str
    quality: FinalObservationMaskQuality
    point_cloud: FinalObservationPointCloud


@dataclass(frozen=True)
class FinalCandidateCountEvaluation:
    success: bool
    completed: bool
    expected_count: int
    survey_candidate_count: int
    recognized_candidate_count: int
    missing_count: int
    extra_count: int
    message: str


@dataclass(frozen=True)
class FinalSourceFailure:
    candidate_id: str
    failure_stage: str
    message: str


@dataclass(frozen=True)
class FinalSourceSummary:
    status: str
    failure_stage: str | None
    candidate_count: int
    processed_candidate_count: int
    successful_candidate_count: int
    failed_candidate_count: int
    unprocessed_candidate_count: int
    successful_candidate_ids: tuple[str, ...]
    failed_candidates: tuple[FinalSourceFailure, ...]
    candidate_count_evaluation: FinalCandidateCountEvaluation


@dataclass(frozen=True)
class ObservationInterfaceSummary:
    status: str
    eligible_object_count: int
    exported_object_count: int
    failed_object_count: int


@dataclass(frozen=True)
class ObservationExportFailure:
    candidate_id: str
    failure_stage: str
    message: str


@dataclass(frozen=True)
class FinalObjectObservationManifest:
    status: str
    message: str
    source_final_report_path: Path
    source_final: FinalSourceSummary
    interface: ObservationInterfaceSummary
    observations: tuple[FinalObjectObservation, ...]
    export_failures: tuple[ObservationExportFailure, ...]


def load_final_object_observations(
    manifest_path: Path,
    *,
    allow_failed: bool = False,
) -> FinalObjectObservationManifest:
    manifest_path = manifest_path.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = validate_final_object_observation_manifest(
        payload,
        root=manifest_path.parent,
    )
    if manifest.status == "failed" and not allow_failed:
        raise ValueError("Final object observation manifest has no processable observations")
    return manifest


def validate_final_object_observation_manifest(
    payload: Any,
    *,
    root: Path,
) -> FinalObjectObservationManifest:
    root = root.resolve()
    if not isinstance(payload, dict):
        raise ValueError("Final object observation manifest must contain an object")
    if payload.get("schema") != FINAL_OBJECT_OBSERVATION_SCHEMA:
        raise ValueError(
            f"unsupported Final object observation schema: {payload.get('schema')}"
        )
    if payload.get("revision") != FINAL_OBJECT_OBSERVATION_REVISION:
        raise ValueError(
            f"unsupported Final object observation revision: {payload.get('revision')}"
        )
    if payload.get("producer_stage") != "task1_final":
        raise ValueError("Final object observation producer_stage must be task1_final")

    status = _status(payload.get("status"), "status")
    message = _nonempty_string(payload.get("message"), "message")
    source_final_report_path = _contained_relative_path(
        root,
        payload.get("source_final_report"),
        "source_final_report",
        require_file=False,
    )
    source_final = _source_final_summary(payload.get("source_final"))
    interface = _interface_summary(payload.get("interface"))

    raw_objects = payload.get("objects")
    if not isinstance(raw_objects, list):
        raise ValueError("Final object observation manifest objects must be a list")
    observations = tuple(
        validate_final_object_observation_entry(root, value, index)
        for index, value in enumerate(raw_objects)
    )
    raw_failures = payload.get("failures")
    if not isinstance(raw_failures, list):
        raise ValueError("Final object observation manifest failures must be a list")
    export_failures = tuple(
        _export_failure(value, index) for index, value in enumerate(raw_failures)
    )

    if payload.get("object_count") != len(observations):
        raise ValueError("Final object observation object_count does not match objects")
    if payload.get("export_failure_count") != len(export_failures):
        raise ValueError(
            "Final object observation export_failure_count does not match failures"
        )
    if interface.exported_object_count != len(observations):
        raise ValueError("Final observation interface exported count does not match objects")
    if interface.failed_object_count != len(export_failures):
        raise ValueError("Final observation interface failed count does not match failures")
    if (
        interface.exported_object_count + interface.failed_object_count
        != interface.eligible_object_count
    ):
        raise ValueError("Final observation interface counts are inconsistent")
    if interface.eligible_object_count != source_final.successful_candidate_count:
        raise ValueError("Final observation interface eligibility differs from source Final")

    observation_ids = [item.observation_id for item in observations]
    if len(observation_ids) != len(set(observation_ids)):
        raise ValueError("Final object observation observation_id values must be unique")
    object_candidate_ids = [item.candidate_id for item in observations]
    failure_candidate_ids = [item.candidate_id for item in export_failures]
    delivered_ids = object_candidate_ids + failure_candidate_ids
    if len(delivered_ids) != len(set(delivered_ids)):
        raise ValueError("Final object observation candidate_id values must be unique")
    if set(delivered_ids) != set(source_final.successful_candidate_ids):
        raise ValueError(
            "Final observation objects and failures must cover source successful candidates"
        )
    source_failed_ids = {item.candidate_id for item in source_final.failed_candidates}
    if source_failed_ids.intersection(delivered_ids):
        raise ValueError("Source failed candidates cannot have Final object observations")

    expected_interface_status = observation_interface_status(
        exported_object_count=len(observations),
        failed_object_count=len(export_failures),
    )
    if interface.status != expected_interface_status:
        raise ValueError("Final observation interface status does not match its counts")
    expected_status = final_observation_manifest_status(
        source_final_status=source_final.status,
        interface_status=interface.status,
        object_count=len(observations),
    )
    if status != expected_status:
        raise ValueError("Final object observation status does not match source/interface state")

    return FinalObjectObservationManifest(
        status=status,
        message=message,
        source_final_report_path=source_final_report_path,
        source_final=source_final,
        interface=interface,
        observations=observations,
        export_failures=export_failures,
    )


def validate_final_object_observation_entry(
    root: Path,
    value: Any,
    index: int,
) -> FinalObjectObservation:
    root = root.resolve()
    if not isinstance(value, dict):
        raise ValueError(f"Final object observation[{index}] must be an object")
    observation_id = ensure_portable_identifier(
        value.get("observation_id"), f"objects[{index}].observation_id"
    )
    candidate_id = ensure_portable_identifier(
        value.get("candidate_id"), f"objects[{index}].candidate_id"
    )
    if observation_id != f"final_{candidate_id}":
        raise ValueError(f"objects[{index}].observation_id does not match candidate_id")
    view_id = ensure_event_identifier(value.get("view_id"), f"objects[{index}].view_id")

    detection = _mapping(value, "detection", index)
    detection_id = ensure_event_identifier(
        detection.get("detection_id"), f"objects[{index}].detection_id"
    )
    class_name = ensure_portable_identifier(
        detection.get("class_name"), f"objects[{index}].class_name"
    )
    display_name_value = detection.get("display_name")
    display_name = (
        None
        if display_name_value is None
        else _nonempty_string(display_name_value, f"objects[{index}].display_name")
    )
    confidence = _finite_number(detection.get("confidence"), f"objects[{index}].confidence")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"objects[{index}].confidence must be between 0 and 1")
    bbox = _float_tuple(detection.get("bbox_xyxy"), 4, f"objects[{index}].bbox_xyxy")
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError(f"objects[{index}].bbox_xyxy must have positive area")

    priors = _mapping(value, "position_priors", index)
    survey_prior = _float_tuple(
        priors.get("survey_bottom_position_world"),
        3,
        f"objects[{index}].survey_bottom_position_world",
    )
    final_prior = _float_tuple(
        priors.get("final_bottom_position_world"),
        3,
        f"objects[{index}].final_bottom_position_world",
    )
    if priors.get("semantics") != POSITION_PRIOR_SEMANTICS:
        raise ValueError(f"objects[{index}].position_priors semantics are unsupported")

    rgb = _mapping(value, "rgb", index)
    depth = _mapping(value, "depth", index)
    mask_value = _mapping(value, "mask", index)
    point_cloud_value = _mapping(value, "point_cloud", index)
    camera = _mapping(value, "camera", index)
    support = _mapping(value, "support_plane", index)
    quality_value = _mapping(value, "quality", index)

    if rgb.get("encoding") != RGB_ENCODING or rgb.get("color_space") != RGB_COLOR_SPACE:
        raise ValueError(f"objects[{index}] RGB encoding/color space is unsupported")
    if (
        depth.get("encoding") != DEPTH_ENCODING
        or depth.get("unit") != "m"
        or depth.get("registered_to_rgb") is not True
        or depth.get("invalid_sample_rule") != "non_finite_or_non_positive"
    ):
        raise ValueError(f"objects[{index}] depth geometry contract is unsupported")
    if (
        mask_value.get("encoding") != MASK_ENCODING
        or mask_value.get("source") != MASK_SOURCE
        or mask_value.get("foreground_value") != 255
        or mask_value.get("background_value") != 0
    ):
        raise ValueError(f"objects[{index}] mask encoding contract is unsupported")
    if mask_value.get("uses_layout_or_simulation_segmentation") is not False:
        raise ValueError(f"objects[{index}] mask must be independent of evaluation truth")

    intrinsics_value = camera.get("intrinsics")
    if not isinstance(intrinsics_value, dict):
        raise ValueError(f"objects[{index}].camera.intrinsics must be an object")
    if (
        camera.get("optical_axis_convention") != OPTICAL_AXIS_CONVENTION
        or camera.get("transform_semantics") != TRANSFORM_SEMANTICS
        or intrinsics_value.get("distortion_model") != DISTORTION_MODEL
        or intrinsics_value.get("distortion_coefficients") != []
    ):
        raise ValueError(f"objects[{index}] camera geometry contract is unsupported")
    intrinsics = ObservationCameraIntrinsics(
        width=_positive_integer(intrinsics_value.get("width"), f"objects[{index}].width"),
        height=_positive_integer(
            intrinsics_value.get("height"), f"objects[{index}].height"
        ),
        fx=_positive_number(intrinsics_value.get("fx"), f"objects[{index}].fx"),
        fy=_positive_number(intrinsics_value.get("fy"), f"objects[{index}].fy"),
        cx=_finite_number(intrinsics_value.get("cx"), f"objects[{index}].cx"),
        cy=_finite_number(intrinsics_value.get("cy"), f"objects[{index}].cy"),
    )
    world_frame_id = ensure_frame_id(
        camera.get("world_frame_id"), f"objects[{index}].world_frame_id"
    )
    camera_optical_frame_id = ensure_frame_id(
        camera.get("camera_optical_frame_id"),
        f"objects[{index}].camera_optical_frame_id",
    )
    transform = _rigid_transform(
        camera.get("T_world_camera_optical"),
        f"objects[{index}].T_world_camera_optical",
    )
    support_frame_id = ensure_frame_id(
        support.get("frame_id"), f"objects[{index}].support_frame_id"
    )
    support_normal = _float_tuple(
        support.get("normal_world"), 3, f"objects[{index}].support_plane.normal_world"
    )
    if not np.allclose(support_normal, (0.0, 0.0, 1.0), atol=1e-9):
        raise ValueError(f"objects[{index}] support plane must have world +Z normal")
    support_z = _finite_number(
        support.get("z_world_m"), f"objects[{index}].support_plane_z_world_m"
    )

    rgb_path = _verified_artifact(root, rgb, f"objects[{index}].rgb")
    depth_path = _verified_artifact(root, depth, f"objects[{index}].depth")
    mask_path = _verified_artifact(root, mask_value, f"objects[{index}].mask")
    rgb_array = _load_rgb_png(rgb_path, intrinsics.width, intrinsics.height)
    depth_m = np.load(depth_path, allow_pickle=False)
    if depth_m.dtype != np.float32 or depth_m.shape != (intrinsics.height, intrinsics.width):
        raise ValueError(f"objects[{index}] depth dtype/shape does not match the contract")
    mask_array = _load_mask_png(mask_path)
    if mask_array.shape != depth_m.shape or not np.any(mask_array):
        raise ValueError(f"objects[{index}] mask must be non-empty and aligned with depth")
    if not np.all(np.isfinite(depth_m[mask_array]) & (depth_m[mask_array] > 0.0)):
        raise ValueError(f"objects[{index}] mask must only select valid positive depth")

    clipped_bbox = _pixel_bounds(bbox, intrinsics.width, intrinsics.height)
    outside_bbox = mask_array.copy()
    left, top, right, bottom = clipped_bbox
    outside_bbox[top:bottom, left:right] = False
    if np.any(outside_bbox):
        raise ValueError(f"objects[{index}] mask must stay inside the clipped detection bbox")
    quality = _mask_quality(
        quality_value,
        index=index,
        clipped_bbox=clipped_bbox,
        selected_pixel_count=int(np.count_nonzero(mask_array)),
    )
    point_cloud = _load_point_cloud(
        root,
        point_cloud_value,
        index=index,
        world_frame_id=world_frame_id,
        rgb=rgb_array,
        depth_m=depth_m,
        mask=mask_array,
        intrinsics=intrinsics,
        transform=transform,
    )

    return FinalObjectObservation(
        observation_id=observation_id,
        candidate_id=candidate_id,
        view_id=view_id,
        detection_id=detection_id,
        class_name=class_name,
        display_name=display_name,
        confidence=confidence,
        bbox_xyxy=bbox,
        survey_bottom_position_world=survey_prior,
        final_bottom_position_world=final_prior,
        rgb_path=rgb_path,
        depth_path=depth_path,
        mask_path=mask_path,
        depth_m=depth_m,
        mask=mask_array,
        intrinsics=intrinsics,
        T_world_camera_optical=transform,
        world_frame_id=world_frame_id,
        camera_optical_frame_id=camera_optical_frame_id,
        support_frame_id=support_frame_id,
        support_plane_z_world_m=support_z,
        mask_source=MASK_SOURCE,
        quality=quality,
        point_cloud=point_cloud,
    )


def _load_point_cloud(
    root: Path,
    value: dict[str, Any],
    *,
    index: int,
    world_frame_id: str,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: ObservationCameraIntrinsics,
    transform: TransformMatrix,
) -> FinalObservationPointCloud:
    field = f"objects[{index}].point_cloud"
    if (
        value.get("encoding") != POINT_CLOUD_ENCODING
        or value.get("source") != POINT_CLOUD_SOURCE
    ):
        raise ValueError(f"{field} encoding/source is unsupported")
    if value.get("world_frame_id") != world_frame_id:
        raise ValueError(f"{field} frame must match the camera world frame")

    point_count = _positive_integer(value.get("point_count"), f"{field}.point_count")
    rows, columns = np.nonzero(mask)
    if point_count != len(rows):
        raise ValueError(f"{field} point count must match the production mask")
    path = _verified_artifact(root, value, field)
    try:
        with np.load(path, allow_pickle=False) as archive:
            array_names = set(archive.files)
            points = np.asarray(archive["points_world_m"]).copy()
            colors = np.asarray(archive["colors_rgb_uint8"]).copy()
            pixels = np.asarray(archive["pixels_uv"]).copy()
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"cannot load {field} artifact {path}: {exc}") from exc
    if array_names != _POINT_CLOUD_ARRAYS:
        raise ValueError(f"{field} arrays do not match the contract")

    if points.dtype != np.float32 or points.shape != (point_count, 3):
        raise ValueError(f"{field}.points_world_m dtype/shape is invalid")
    if colors.dtype != np.uint8 or colors.shape != (point_count, 3):
        raise ValueError(f"{field}.colors_rgb_uint8 dtype/shape is invalid")
    if pixels.dtype != np.int32 or pixels.shape != (point_count, 2):
        raise ValueError(f"{field}.pixels_uv dtype/shape is invalid")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{field}.points_world_m must be finite")

    expected_pixels = np.ascontiguousarray(
        np.column_stack((columns, rows)), dtype=np.int32
    )
    if not np.array_equal(pixels, expected_pixels):
        raise ValueError(f"{field}.pixels_uv must match the production mask")
    expected_points = _back_project_to_world(
        pixels, depth_m, intrinsics, transform
    )
    if not np.allclose(points, expected_points, rtol=0.0, atol=1e-6):
        raise ValueError(f"{field}.points_world_m does not match RGB-D back-projection")
    expected_colors = np.ascontiguousarray(rgb[rows, columns], dtype=np.uint8)
    if not np.array_equal(colors, expected_colors):
        raise ValueError(f"{field}.colors_rgb_uint8 does not match source RGB")

    centroid = _float_tuple(value.get("centroid_world_m"), 3, f"{field}.centroid")
    bounds = value.get("axis_aligned_bounds_world_m")
    if not isinstance(bounds, dict):
        raise ValueError(f"{field}.axis_aligned_bounds_world_m must be an object")
    minimum = _float_tuple(bounds.get("minimum"), 3, f"{field}.bounds.minimum")
    maximum = _float_tuple(bounds.get("maximum"), 3, f"{field}.bounds.maximum")
    if not np.allclose(
        np.mean(points, axis=0, dtype=np.float64), centroid, rtol=0.0, atol=1e-7
    ):
        raise ValueError(f"{field} centroid does not match its points")
    if not np.allclose(
        np.min(points, axis=0), minimum, rtol=0.0, atol=1e-7
    ) or not np.allclose(
        np.max(points, axis=0), maximum, rtol=0.0, atol=1e-7
    ):
        raise ValueError(f"{field} bounds do not match its points")

    return FinalObservationPointCloud(
        path=path,
        world_frame_id=world_frame_id,
        points_world_m=points,
        colors_rgb_uint8=colors,
        pixels_uv=pixels,
    )


def _back_project_to_world(
    pixels_uv: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: ObservationCameraIntrinsics,
    transform: TransformMatrix,
) -> np.ndarray:
    columns = pixels_uv[:, 0]
    rows = pixels_uv[:, 1]
    depth = depth_m[rows, columns].astype(np.float64, copy=False)
    points_camera = np.column_stack(
        (
            (columns.astype(np.float64) - intrinsics.cx) * depth / intrinsics.fx,
            (rows.astype(np.float64) - intrinsics.cy) * depth / intrinsics.fy,
            depth,
        )
    )
    matrix = np.asarray(transform, dtype=np.float64)
    points_world = points_camera @ matrix[:3, :3].T + matrix[:3, 3]
    return np.ascontiguousarray(points_world, dtype=np.float32)


def observation_interface_status(
    *,
    exported_object_count: int,
    failed_object_count: int,
) -> str:
    if exported_object_count < 0 or failed_object_count < 0:
        raise ValueError("Final observation interface counts must be non-negative")
    if failed_object_count == 0:
        return "success"
    return "partial" if exported_object_count > 0 else "failed"


def final_observation_manifest_status(
    *,
    source_final_status: str,
    interface_status: str,
    object_count: int,
) -> str:
    if source_final_status not in _STATUS_VALUES:
        raise ValueError(f"unsupported source Final status: {source_final_status}")
    if interface_status not in _INTERFACE_STATUS_VALUES:
        raise ValueError(f"unsupported observation interface status: {interface_status}")
    if object_count < 0:
        raise ValueError("Final observation object_count must be non-negative")
    if source_final_status == "success" and interface_status == "success":
        return "success"
    return "partial" if object_count > 0 else "failed"


def ensure_portable_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{field} must be a non-empty portable identifier")
    return value


def ensure_event_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or _EVENT_IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{field} must be a non-empty portable event identifier")
    return value


def ensure_frame_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or _FRAME_ID.fullmatch(value) is None:
        raise ValueError(f"{field} must be a portable relative frame id")
    return value


def _source_final_summary(value: Any) -> FinalSourceSummary:
    if not isinstance(value, dict):
        raise ValueError("Final object observation source_final must be an object")
    status = _status(value.get("status"), "source_final.status")
    failure_stage_value = value.get("failure_stage")
    failure_stage = (
        None
        if failure_stage_value is None
        else _nonempty_string(failure_stage_value, "source_final.failure_stage")
    )
    candidate_count = _nonnegative_integer(value.get("candidate_count"), "candidate_count")
    processed_count = _nonnegative_integer(
        value.get("processed_candidate_count"), "processed_candidate_count"
    )
    successful_count = _nonnegative_integer(
        value.get("successful_candidate_count"), "successful_candidate_count"
    )
    failed_count = _nonnegative_integer(
        value.get("failed_candidate_count"), "failed_candidate_count"
    )
    unprocessed_count = _nonnegative_integer(
        value.get("unprocessed_candidate_count"), "unprocessed_candidate_count"
    )
    if processed_count + unprocessed_count != candidate_count:
        raise ValueError("source Final processed/unprocessed counts are inconsistent")
    if successful_count + failed_count != processed_count:
        raise ValueError("source Final successful/failed counts are inconsistent")
    expected_status = (
        "success"
        if failed_count == 0 and unprocessed_count == 0
        else "failed"
        if successful_count == 0
        else "partial"
    )
    if status != expected_status:
        raise ValueError("source Final status does not match its candidate counts")

    raw_successful_ids = value.get("successful_candidate_ids")
    if not isinstance(raw_successful_ids, list):
        raise ValueError("source_final.successful_candidate_ids must be a list")
    successful_ids = tuple(
        ensure_portable_identifier(item, f"successful_candidate_ids[{index}]")
        for index, item in enumerate(raw_successful_ids)
    )
    if len(successful_ids) != successful_count or len(successful_ids) != len(
        set(successful_ids)
    ):
        raise ValueError("source Final successful candidate ids are inconsistent")

    raw_failures = value.get("failed_candidates")
    if not isinstance(raw_failures, list):
        raise ValueError("source_final.failed_candidates must be a list")
    failures = tuple(_source_failure(item, index) for index, item in enumerate(raw_failures))
    failure_ids = [item.candidate_id for item in failures]
    if len(failures) != failed_count or len(failure_ids) != len(set(failure_ids)):
        raise ValueError("source Final failed candidate details are inconsistent")
    if set(successful_ids).intersection(failure_ids):
        raise ValueError("source Final successful and failed candidate ids must be disjoint")

    count_evaluation = _candidate_count_evaluation(value.get("candidate_count_evaluation"))
    if count_evaluation.survey_candidate_count != candidate_count:
        raise ValueError("source Final candidate evaluation input count is inconsistent")
    if count_evaluation.recognized_candidate_count != successful_count:
        raise ValueError("source Final candidate evaluation recognized count is inconsistent")

    return FinalSourceSummary(
        status=status,
        failure_stage=failure_stage,
        candidate_count=candidate_count,
        processed_candidate_count=processed_count,
        successful_candidate_count=successful_count,
        failed_candidate_count=failed_count,
        unprocessed_candidate_count=unprocessed_count,
        successful_candidate_ids=successful_ids,
        failed_candidates=failures,
        candidate_count_evaluation=count_evaluation,
    )


def _candidate_count_evaluation(value: Any) -> FinalCandidateCountEvaluation:
    if not isinstance(value, dict):
        raise ValueError("source_final.candidate_count_evaluation must be an object")
    success = _boolean(value.get("success"), "candidate_count_evaluation.success")
    completed = _boolean(value.get("completed"), "candidate_count_evaluation.completed")
    expected = _nonnegative_integer(value.get("expected_count"), "expected_count")
    survey = _nonnegative_integer(value.get("survey_candidate_count"), "survey_candidate_count")
    recognized = _nonnegative_integer(
        value.get("recognized_candidate_count"), "recognized_candidate_count"
    )
    missing = _nonnegative_integer(value.get("missing_count"), "missing_count")
    extra = _nonnegative_integer(value.get("extra_count"), "extra_count")
    message = _nonempty_string(value.get("message"), "candidate_count_evaluation.message")
    if missing != max(0, expected - recognized) or extra != max(0, recognized - expected):
        raise ValueError("source Final candidate evaluation difference counts are inconsistent")
    if success != (completed and recognized == expected):
        raise ValueError("source Final candidate evaluation success is inconsistent")
    return FinalCandidateCountEvaluation(
        success=success,
        completed=completed,
        expected_count=expected,
        survey_candidate_count=survey,
        recognized_candidate_count=recognized,
        missing_count=missing,
        extra_count=extra,
        message=message,
    )


def _interface_summary(value: Any) -> ObservationInterfaceSummary:
    if not isinstance(value, dict):
        raise ValueError("Final object observation interface must be an object")
    status = _status(value.get("status"), "interface.status")
    eligible = _nonnegative_integer(value.get("eligible_object_count"), "eligible_object_count")
    exported = _nonnegative_integer(value.get("exported_object_count"), "exported_object_count")
    failed = _nonnegative_integer(value.get("failed_object_count"), "failed_object_count")
    return ObservationInterfaceSummary(
        status=status,
        eligible_object_count=eligible,
        exported_object_count=exported,
        failed_object_count=failed,
    )


def _source_failure(value: Any, index: int) -> FinalSourceFailure:
    if not isinstance(value, dict):
        raise ValueError(f"source_final.failed_candidates[{index}] must be an object")
    return FinalSourceFailure(
        candidate_id=ensure_portable_identifier(
            value.get("candidate_id"), f"source_final.failed_candidates[{index}].candidate_id"
        ),
        failure_stage=_nonempty_string(
            value.get("failure_stage"),
            f"source_final.failed_candidates[{index}].failure_stage",
        ),
        message=_nonempty_string(
            value.get("message"), f"source_final.failed_candidates[{index}].message"
        ),
    )


def _export_failure(value: Any, index: int) -> ObservationExportFailure:
    if not isinstance(value, dict):
        raise ValueError(f"Final object observation failures[{index}] must be an object")
    failure_stage = _nonempty_string(
        value.get("failure_stage"), f"failures[{index}].failure_stage"
    )
    if failure_stage != "object_observation_export":
        raise ValueError(f"failures[{index}].failure_stage is unsupported")
    return ObservationExportFailure(
        candidate_id=ensure_portable_identifier(
            value.get("candidate_id"), f"failures[{index}].candidate_id"
        ),
        failure_stage=failure_stage,
        message=_nonempty_string(value.get("message"), f"failures[{index}].message"),
    )


def _mask_quality(
    value: dict[str, Any],
    *,
    index: int,
    clipped_bbox: tuple[int, int, int, int],
    selected_pixel_count: int,
) -> FinalObservationMaskQuality:
    mask_source = value.get("mask_source")
    if mask_source != MASK_SOURCE:
        raise ValueError(f"objects[{index}].quality.mask_source is unsupported")
    quality_bbox = _integer_tuple(
        value.get("bbox_clipped_xyxy"), 4, f"objects[{index}].quality.bbox_clipped_xyxy"
    )
    if quality_bbox != clipped_bbox:
        raise ValueError(f"objects[{index}] quality bbox does not match detection bbox")
    valid_count = _positive_integer(
        value.get("valid_depth_pixel_count"), f"objects[{index}].valid_depth_pixel_count"
    )
    foreground_count = _positive_integer(
        value.get("foreground_pixel_count_before_component_selection"),
        f"objects[{index}].foreground_pixel_count_before_component_selection",
    )
    eligible_count = _positive_integer(
        value.get("eligible_component_count"), f"objects[{index}].eligible_component_count"
    )
    selected_count = _positive_integer(
        value.get("selected_component_pixel_count"),
        f"objects[{index}].selected_component_pixel_count",
    )
    if selected_count != selected_pixel_count:
        raise ValueError(f"objects[{index}] quality pixel count does not match mask")
    if selected_count > foreground_count or foreground_count > valid_count:
        raise ValueError(f"objects[{index}] quality pixel counts are inconsistent")
    seed_distance = _finite_number(
        value.get("selected_component_seed_xy_distance_m"),
        f"objects[{index}].selected_component_seed_xy_distance_m",
    )
    if seed_distance < 0.0:
        raise ValueError(f"objects[{index}] component seed distance must be non-negative")
    seed_source = value.get("component_selection_seed_source")
    if seed_source != COMPONENT_SELECTION_SEED_SOURCE:
        raise ValueError(f"objects[{index}] component selection seed source is unsupported")
    return FinalObservationMaskQuality(
        mask_source=mask_source,
        bbox_clipped_xyxy=quality_bbox,
        valid_depth_pixel_count=valid_count,
        foreground_pixel_count_before_component_selection=foreground_count,
        eligible_component_count=eligible_count,
        selected_component_pixel_count=selected_count,
        selected_component_seed_xy_distance_m=seed_distance,
        component_selection_seed_source=seed_source,
    )


def _pixel_bounds(
    bbox: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    left = max(0, min(width - 1, int(math.floor(x1))))
    right = max(left + 1, min(width, int(math.ceil(x2))))
    top = max(0, min(height - 1, int(math.floor(y1))))
    bottom = max(top + 1, min(height, int(math.ceil(y2))))
    return left, top, right, bottom


def _rigid_transform(value: Any, field: str) -> TransformMatrix:
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


def _contained_relative_path(
    root: Path,
    value: Any,
    field: str,
    *,
    require_file: bool,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{field} must stay inside the manifest directory")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} must stay inside the manifest directory") from exc
    if require_file and not path.is_file():
        raise FileNotFoundError(f"{field} artifact does not exist: {path}")
    return path


def _verified_artifact(root: Path, value: dict[str, Any], field: str) -> Path:
    path = _contained_relative_path(
        root,
        value.get("path"),
        f"{field}.path",
        require_file=True,
    )
    expected_hash = value.get("sha256")
    if not isinstance(expected_hash, str) or _sha256_file(path) != expected_hash:
        raise ValueError(f"{field} artifact sha256 does not match the manifest")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mask_png(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to load Final foreground masks") from exc
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "L":
            raise ValueError("Final foreground mask must be an 8-bit grayscale PNG")
        values = np.asarray(image, dtype=np.uint8)
    unique = set(int(value) for value in np.unique(values))
    if not unique.issubset({0, 255}):
        raise ValueError("Final foreground mask values must be 0 or 255")
    return values == 255


def _load_rgb_png(path: Path, width: int, height: int) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to validate Final RGB images") from exc
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB":
            raise ValueError("Final RGB image must be an RGB PNG")
        if image.size != (width, height):
            raise ValueError(
                f"Final RGB size {image.size} does not match camera intrinsics {(width, height)}"
            )
        return np.asarray(image, dtype=np.uint8).copy()


def _mapping(value: dict[str, Any], key: str, index: int) -> dict[str, Any]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"objects[{index}].{key} must be an object")
    return result


def _status(value: Any, field: str) -> str:
    if value not in _STATUS_VALUES:
        raise ValueError(f"unsupported {field}: {value}")
    return str(value)


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _float_tuple(value: Any, length: int, field: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field} must contain {length} numbers")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain numbers") from exc
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{field} must contain finite numbers")
    return result


def _integer_tuple(value: Any, length: int, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field} must contain {length} integers")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{field} must contain integers")
    return tuple(value)


def _finite_number(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _positive_number(value: Any, field: str) -> float:
    result = _finite_number(value, field)
    if result <= 0.0:
        raise ValueError(f"{field} must be positive")
    return result


def _positive_integer(value: Any, field: str) -> int:
    result = _nonnegative_integer(value, field)
    if result == 0:
        raise ValueError(f"{field} must be positive")
    return result


def _nonnegative_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value
