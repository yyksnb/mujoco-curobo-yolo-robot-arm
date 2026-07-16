from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from robot_arm_pipeline.planning import (
    PoseRoutePlanner,
    PoseTarget,
    IKSolution,
    offset_pose_target_along_local_z,
    plan_pose_route,
)
from robot_arm_pipeline.perception.object_observation import (
    MASK_SOURCE,
    ensure_frame_id,
)
from robot_arm_pipeline.types import RobotState
from task1.detection import Detector, YoloInferenceConfig, render_detection_overlay
from task1.final.observations import (
    FinalObservationExporter,
    FinalObservationExportPolicy,
)
from task1.vision import (
    CandidateLocalizationPolicy,
    Detection2D,
    RgbdFrame,
    SurveyObservation,
    localize_detection,
)
from task1.scene import RigidPose, compose


FINAL_CONFIG_SCHEMA = "task1_final_config"
FINAL_REPORT_SCHEMA = "task1_final_report"
DEFAULT_FINAL_CONFIG_PATH = Path("configs/task1/final/config.yaml")
_CANDIDATE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class FinalCameraPolicy:
    image_width: int
    image_height: int
    vertical_fov_deg: float
    coverage_margin_m: float
    aim_height_above_bottom_m: float
    secondary_standoff_minimum_distance_m: float
    standoff_distance_scales: tuple[float, ...]
    opening_position_world: tuple[float, float, float]
    optical_roll_degrees: tuple[float, ...]


@dataclass(frozen=True)
class FinalAssociationPolicy:
    maximum_xy_distance_m: float


@dataclass(frozen=True)
class FinalEvaluationPolicy:
    expected_object_count: int
    truth_match_maximum_xy_distance_m: float
    input_candidate_position_tolerance_m: float
    output_position_tolerance_m: float
    minimum_visible_pixels: int
    camera_position_tolerance_m: float
    camera_orientation_tolerance_rad: float


@dataclass(frozen=True)
class FinalPlanningPolicy:
    robot_config_path: Path
    world_config_path: Path
    graph_config_path: Path
    random_seed: int
    max_attempts: int
    enable_graph_attempt: int
    num_ik_seeds: int
    num_trajopt_seeds: int
    position_tolerance_m: float
    orientation_tolerance_rad: float
    ik_batch_size: int
    ik_solutions_per_target: int
    enable_portal_continuation: bool
    portal_offset_m: float
    continuation_step_m: float
    continuation_edge_sample_count: int
    continuation_ik_solution_count: int
    continuation_finetune_attempts: int
    continuation_joint_tolerance_rad: float
    continuation_stop_velocity_tolerance_rad_s: float


@dataclass(frozen=True)
class FinalSimulationPolicy:
    model_path: Path
    camera_name: str
    ground_z_m: float


@dataclass(frozen=True)
class FinalDetectionConfig:
    config_path: Path
    profile_path: Path
    inference: YoloInferenceConfig


@dataclass(frozen=True)
class FinalConfig:
    config_path: Path
    planning: FinalPlanningPolicy
    camera: FinalCameraPolicy
    simulation: FinalSimulationPolicy
    detection: FinalDetectionConfig
    localization: CandidateLocalizationPolicy
    association: FinalAssociationPolicy
    object_observation_interface: FinalObservationExportPolicy
    evaluation: FinalEvaluationPolicy


@dataclass(frozen=True)
class FinalCandidate:
    candidate_id: str
    bottom_position_world: tuple[float, float, float]
    footprint_polygon_xy: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class FinalCaptureResult:
    frame: RgbdFrame
    rgb_path: Path


@dataclass(frozen=True)
class _PreparedFinalCandidate:
    input_index: int
    candidate: FinalCandidate
    target_attempts: tuple[tuple[PoseTarget, dict[str, Any]], ...]
    ik_solutions: tuple[IKSolution, ...]


class FinalCapture(Protocol):
    def capture(
        self, candidate_id: str, joint_positions: tuple[float, ...]
    ) -> FinalCaptureResult: ...

    def close(self) -> None: ...


def load_final_config(path: Path, *, repo_root: Path) -> FinalConfig:
    config_path = _resolve_path(path, repo_root)
    payload = _load_mapping(config_path)
    if payload.get("schema") != FINAL_CONFIG_SCHEMA:
        raise ValueError(f"unsupported Task1 Final config schema: {payload.get('schema')}")
    camera = _mapping(payload, "camera")
    planning = _mapping(payload, "planning")
    portal_continuation = _mapping(planning, "portal_continuation")
    simulation = _mapping(payload, "simulation")
    detection = _mapping(payload, "detection")
    localization = _mapping(payload, "localization")
    association = _mapping(payload, "association")
    object_observation_interface = _mapping(payload, "object_observation_interface")
    evaluation = _mapping(payload, "evaluation")
    result = FinalConfig(
        config_path=config_path,
        planning=FinalPlanningPolicy(
            robot_config_path=_resolve_path(
                Path(_string(planning, "robot_config_path")), repo_root
            ),
            world_config_path=_resolve_path(
                Path(_string(planning, "world_config_path")), repo_root
            ),
            graph_config_path=_resolve_path(
                Path(_string(planning, "graph_config_path")), repo_root
            ),
            random_seed=_integer(planning, "random_seed", minimum=0),
            max_attempts=_integer(planning, "max_attempts", minimum=1),
            enable_graph_attempt=_integer(
                planning, "enable_graph_attempt", minimum=0
            ),
            num_ik_seeds=_integer(planning, "num_ik_seeds", minimum=1),
            num_trajopt_seeds=_integer(planning, "num_trajopt_seeds", minimum=1),
            position_tolerance_m=_number(
                planning, "position_tolerance_m", minimum=0.0
            ),
            orientation_tolerance_rad=_number(
                planning, "orientation_tolerance_rad", minimum=0.0
            ),
            ik_batch_size=_integer(planning, "ik_batch_size", minimum=1),
            ik_solutions_per_target=_integer(
                planning, "ik_solutions_per_target", minimum=1
            ),
            enable_portal_continuation=_boolean(
                portal_continuation, "enabled"
            ),
            portal_offset_m=_number(
                portal_continuation, "offset_m", minimum=0.0
            ),
            continuation_step_m=_number(
                portal_continuation, "step_m", minimum=0.0
            ),
            continuation_edge_sample_count=_integer(
                portal_continuation, "edge_sample_count", minimum=2
            ),
            continuation_ik_solution_count=_integer(
                portal_continuation, "ik_solution_count", minimum=1
            ),
            continuation_finetune_attempts=_integer(
                portal_continuation, "finetune_attempts", minimum=1
            ),
            continuation_joint_tolerance_rad=_number(
                portal_continuation, "joint_tolerance_rad", minimum=0.0
            ),
            continuation_stop_velocity_tolerance_rad_s=_number(
                portal_continuation,
                "stop_velocity_tolerance_rad_s",
                minimum=0.0,
            ),
        ),
        camera=FinalCameraPolicy(
            image_width=_integer(camera, "image_width", minimum=1),
            image_height=_integer(camera, "image_height", minimum=1),
            vertical_fov_deg=_number(camera, "vertical_fov_deg"),
            coverage_margin_m=_number(camera, "coverage_margin_m", minimum=0.0),
            aim_height_above_bottom_m=_number(
                camera, "aim_height_above_bottom_m", minimum=0.0
            ),
            secondary_standoff_minimum_distance_m=_number(
                camera, "secondary_standoff_minimum_distance_m", minimum=0.0
            ),
            standoff_distance_scales=_float_array(
                camera.get("standoff_distance_scales"),
                "camera.standoff_distance_scales",
            ),
            opening_position_world=_float_tuple(
                camera.get("opening_position_world"), 3, "camera.opening_position_world"
            ),
            optical_roll_degrees=_float_array(
                camera.get("optical_roll_degrees"), "camera.optical_roll_degrees"
            ),
        ),
        simulation=FinalSimulationPolicy(
            model_path=_resolve_path(
                Path(_string(simulation, "model_path")), repo_root
            ),
            camera_name=_string(simulation, "camera_name"),
            ground_z_m=_number(simulation, "ground_z_m"),
        ),
        detection=FinalDetectionConfig(
            config_path=config_path,
            profile_path=_resolve_path(
                Path(_string(detection, "profile_path")), repo_root
            ),
            inference=YoloInferenceConfig(
                image_size=_integer(detection, "image_size", minimum=1),
                confidence_threshold=_number(detection, "confidence_threshold"),
                iou_threshold=_number(detection, "iou_threshold"),
                class_agnostic_nms=_boolean(detection, "class_agnostic_nms"),
                device=_optional_string(detection.get("device"), "detection.device"),
                max_detections=_optional_integer(
                    detection.get("max_detections"),
                    "detection.max_detections",
                    minimum=1,
                ),
            ),
        ),
        localization=CandidateLocalizationPolicy(
            sample_stride_px=_integer(localization, "sample_stride_px", minimum=1),
            min_valid_depth_samples=_integer(
                localization, "min_valid_depth_samples", minimum=1
            ),
            min_depth_m=_number(localization, "min_depth_m", minimum=0.0),
            max_depth_m=_optional_number(
                localization.get("max_depth_m"), "localization.max_depth_m"
            ),
            min_height_above_floor_m=_number(
                localization, "min_height_above_floor_m", minimum=0.0
            ),
            max_height_above_floor_m=_number(
                localization, "max_height_above_floor_m", minimum=0.0
            ),
            visible_surface_radius_percentile=_number(
                localization, "visible_surface_radius_percentile", minimum=0.0
            ),
            footprint_simplification_tolerance_m=_number(
                localization, "footprint_simplification_tolerance_m", minimum=0.0
            ),
            minimum_surface_variance_m2=_number(
                localization, "minimum_surface_variance_m2", minimum=0.0
            ),
        ),
        association=FinalAssociationPolicy(
            maximum_xy_distance_m=_number(association, "maximum_xy_distance_m")
        ),
        object_observation_interface=FinalObservationExportPolicy(
            enabled=_boolean(object_observation_interface, "enabled"),
            world_frame_id=_string(object_observation_interface, "world_frame_id"),
            camera_optical_frame_id=_string(
                object_observation_interface, "camera_optical_frame_id"
            ),
            support_frame_id=_string(object_observation_interface, "support_frame_id"),
            mask_source=_string(object_observation_interface, "mask_source"),
            connected_component_connectivity=_integer(
                object_observation_interface,
                "connected_component_connectivity",
                minimum=1,
            ),
        ),
        evaluation=FinalEvaluationPolicy(
            expected_object_count=_integer(evaluation, "expected_object_count", minimum=1),
            truth_match_maximum_xy_distance_m=_number(
                evaluation, "truth_match_maximum_xy_distance_m", minimum=0.0
            ),
            input_candidate_position_tolerance_m=_number(
                evaluation, "input_candidate_position_tolerance_m", minimum=0.0
            ),
            output_position_tolerance_m=_number(
                evaluation, "output_position_tolerance_m", minimum=0.0
            ),
            minimum_visible_pixels=_integer(
                evaluation, "minimum_visible_pixels", minimum=1
            ),
            camera_position_tolerance_m=_number(
                evaluation, "camera_position_tolerance_m", minimum=0.0
            ),
            camera_orientation_tolerance_rad=_number(
                evaluation, "camera_orientation_tolerance_rad", minimum=0.0
            ),
        ),
    )
    _validate_final_config(result)
    return result


def load_survey_candidates(report_path: Path) -> tuple[dict[str, Any], tuple[FinalCandidate, ...]]:
    payload = _load_mapping(report_path)
    if payload.get("schema") != "task1_survey_report" or payload.get("stage") != "survey":
        raise ValueError("Final input must be a task1_survey_report from the survey stage")
    if payload.get("status") != "success":
        raise ValueError("Final requires a successfully completed Survey report")
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("Survey report candidates must be a list")
    candidates = tuple(_parse_candidate(item, index) for index, item in enumerate(raw_candidates))
    ids = [candidate.candidate_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Survey report candidate_id values must be unique")
    return payload, candidates


def make_final_camera_targets(
    candidate: FinalCandidate,
    *,
    world_pose_base: RigidPose,
    policy: FinalCameraPolicy,
) -> tuple[tuple[PoseTarget, dict[str, Any]], ...]:
    tan_vertical = math.tan(math.radians(policy.vertical_fov_deg) / 2.0)
    tan_horizontal = tan_vertical * policy.image_width / policy.image_height
    candidate_position = np.asarray(candidate.bottom_position_world, dtype=float)
    aim_position = candidate_position + np.asarray(
        (0.0, 0.0, policy.aim_height_above_bottom_m), dtype=float
    )
    opening_position = np.asarray(policy.opening_position_world, dtype=float)
    candidate_to_opening = opening_position - aim_position
    opening_distance = float(np.linalg.norm(candidate_to_opening))
    if opening_distance <= 1e-9:
        raise ValueError("Final opening and candidate positions must be different")
    opening_direction = candidate_to_opening / opening_distance
    view_direction = -opening_direction
    base_rotation = _look_at_camera_rotation(view_direction)
    attempts = []
    for roll_index, roll_degrees in enumerate(policy.optical_roll_degrees):
        rotation = base_rotation * Rotation.from_rotvec(
            np.asarray((0.0, 0.0, math.radians(roll_degrees)))
        )
        footprint_offsets = tuple(
            np.asarray(
                (
                    point[0] - aim_position[0],
                    point[1] - aim_position[1],
                    candidate_position[2] - aim_position[2],
                ),
                dtype=float,
            )
            for point in candidate.footprint_polygon_xy
        )
        camera_offsets = tuple(rotation.inv().apply(offset) for offset in footprint_offsets)
        required_viewing_distance = max(
            max(
                float(offset[2])
                + (abs(float(offset[0])) + policy.coverage_margin_m) / tan_horizontal,
                float(offset[2])
                + (abs(float(offset[1])) + policy.coverage_margin_m) / tan_vertical,
            )
            for offset in camera_offsets
        )
        qx, qy, qz, qw = rotation.as_quat()
        for distance_index, distance_scale in enumerate(
            policy.standoff_distance_scales
        ):
            viewing_distance = required_viewing_distance * distance_scale
            if distance_index > 0:
                viewing_distance = max(
                    viewing_distance,
                    policy.secondary_standoff_minimum_distance_m,
                )
            camera_position = aim_position + opening_direction * viewing_distance
            projected_x = [
                float(offset[0]) / (viewing_distance - float(offset[2]))
                for offset in camera_offsets
            ]
            projected_y = [
                float(offset[1]) / (viewing_distance - float(offset[2]))
                for offset in camera_offsets
            ]
            predicted_width_fraction = min(
                1.0,
                (max(projected_x) - min(projected_x)) / (2.0 * tan_horizontal),
            )
            predicted_height_fraction = min(
                1.0,
                (max(projected_y) - min(projected_y)) / (2.0 * tan_vertical),
            )
            predicted_area_fraction = (
                predicted_width_fraction * predicted_height_fraction
            )
            world_pose_camera = RigidPose(
                tuple(float(item) for item in camera_position),
                (float(qw), float(qx), float(qy), float(qz)),
            )
            base_pose_camera = compose(world_pose_base, world_pose_camera)
            target = PoseTarget(
                target_id=(
                    f"final_{candidate.candidate_id}_roll{roll_index:02d}"
                    f"_distance{distance_index:02d}"
                ),
                target_position=base_pose_camera.position,
                target_quaternion_wxyz=base_pose_camera.quaternion_wxyz,
            )
            geometry = {
                "optical_roll_degrees": roll_degrees,
                "standoff_distance_scale": distance_scale,
                "required_viewing_distance_m": required_viewing_distance,
                "actual_viewing_distance_m": viewing_distance,
                "predicted_footprint_bbox_area_fraction": predicted_area_fraction,
                "camera_pose_world": {
                    "position": list(world_pose_camera.position),
                    "quaternion_wxyz": list(world_pose_camera.quaternion_wxyz),
                },
            }
            attempts.append((target, geometry))
    return tuple(attempts)


def bbox_area_fraction(
    bbox_xyxy: tuple[float, float, float, float], *, image_width: int, image_height: int
) -> float:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Final image dimensions must be positive")
    x1, y1, x2, y2 = bbox_xyxy
    if not all(math.isfinite(value) for value in bbox_xyxy) or x2 <= x1 or y2 <= y1:
        raise ValueError("Final bbox must be finite and have positive area")
    left = min(float(image_width), max(0.0, x1))
    top = min(float(image_height), max(0.0, y1))
    right = min(float(image_width), max(0.0, x2))
    bottom = min(float(image_height), max(0.0, y2))
    area = max(0.0, right - left) * max(0.0, bottom - top)
    return area / float(image_width * image_height)


class FinalProcessor:
    def __init__(
        self,
        *,
        planner: PoseRoutePlanner,
        capture: FinalCapture,
        detector: Detector,
        config: FinalConfig,
        world_pose_base: RigidPose,
        output_dir: Path,
    ) -> None:
        self.planner = planner
        self.capture = capture
        self.detector = detector
        self.config = config
        self.world_pose_base = world_pose_base
        self.output_dir = output_dir
        self.observation_exporter = (
            FinalObservationExporter(
                output_dir=output_dir,
                policy=config.object_observation_interface,
                localization_policy=config.localization,
                maximum_seed_xy_distance_m=config.association.maximum_xy_distance_m,
            )
            if config.object_observation_interface.enabled
            else None
        )

    def run(
        self,
        candidates: tuple[FinalCandidate, ...],
        *,
        start_joint_positions: tuple[float, ...],
        source_survey_report: Path,
    ) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        prepared = self._prepare_candidates(candidates, start_joint_positions)
        remaining = list(prepared)
        current = start_joint_positions
        results_by_id: dict[str, dict[str, Any]] = {}
        stable_by_id: dict[str, dict[str, Any]] = {}
        processing_order: list[str] = []
        try:
            while remaining:
                selected = min(
                    remaining,
                    key=lambda item: self._candidate_order_key(item, current),
                )
                remaining.remove(selected)
                processing_order.append(selected.candidate.candidate_id)
                result, current, stable = self._process_candidate(selected, current)
                result["processing_index"] = len(processing_order) - 1
                results_by_id[selected.candidate.candidate_id] = result
                if stable is not None:
                    stable_by_id[selected.candidate.candidate_id] = stable
        finally:
            self.capture.close()

        results = [results_by_id[candidate.candidate_id] for candidate in candidates]
        stable_objects = [
            stable_by_id[candidate.candidate_id]
            for candidate in candidates
            if candidate.candidate_id in stable_by_id
        ]
        failed_count = sum(result["status"] != "success" for result in results)
        status = "success" if failed_count == 0 else ("failed" if failed_count == len(results) else "partial")
        camera_attempts = [
            attempt
            for result in results
            for attempt in result["camera_target_attempts"]
        ]
        report = _final_report_base(
            status=status,
            failure_stage=None,
            message=(
                f"Processed all {len(candidates)} Survey candidates; "
                f"re-detected {len(stable_objects)} candidates."
            ),
            source_survey_report=source_survey_report,
            final_config_path=self.config.config_path,
            expected_count=self.config.evaluation.expected_object_count,
            candidate_count=len(candidates),
            recognized_candidate_count=len(stable_objects),
            count_evaluation_completed=True,
        )
        report.update({
            "planner": getattr(self.planner, "planner_name", type(self.planner).__name__),
            "detection_source": self.detector.source_metadata(),
            "localization_policy": self.config.localization.to_dict()["localization"],
            "processed_candidate_count": len(results),
            "successful_candidate_count": len(stable_objects),
            "failed_candidate_count": failed_count,
            "unprocessed_candidate_count": 0,
            "candidate_processing_order": processing_order,
            "ik_feasible_camera_target_count": len(
                {
                    attempt["target_id"]
                    for attempt in camera_attempts
                    if attempt["ik_status"] == "success"
                }
            ),
            "ik_rejected_camera_target_count": len(
                {
                    attempt["target_id"]
                    for attempt in camera_attempts
                    if attempt["ik_status"] == "failed"
                }
            ),
            "motion_planning_attempt_count": sum(
                attempt["status"] in {"success", "failed"}
                and attempt["ik_status"] == "success"
                for attempt in camera_attempts
            ),
            "artifact_generation_failure_count": sum(
                len(result["artifact_failures"]) for result in results
            ),
            "results": results,
            "stable_objects": stable_objects,
        })
        report["object_observation_interface"] = self._finalize_object_observations(report)
        return report

    def _finalize_object_observations(
        self,
        report: dict[str, Any],
    ) -> dict[str, Any]:
        stable_objects = report["stable_objects"]
        if self.observation_exporter is None:
            return {
                "enabled": False,
                "status": "disabled",
                "failure_stage": None,
                "manifest_status": "not_run",
                "source_final_status": report["status"],
                "manifest_path": None,
                "eligible_object_count": len(stable_objects),
                "prepared_object_count": 0,
                "exported_object_count": 0,
                "failed_object_count": 0,
                "message": "Final object observation export is disabled by configuration.",
            }
        source_final = {
            "status": report["status"],
            "failure_stage": report["failure_stage"],
            "candidate_count": report["candidate_count"],
            "processed_candidate_count": report["processed_candidate_count"],
            "successful_candidate_count": report["successful_candidate_count"],
            "failed_candidate_count": report["failed_candidate_count"],
            "unprocessed_candidate_count": report["unprocessed_candidate_count"],
            "successful_candidate_ids": [
                str(item["candidate_id"]) for item in stable_objects
            ],
            "failed_candidates": [
                {
                    "candidate_id": str(result["candidate_id"]),
                    "failure_stage": str(result["failure_stage"]),
                    "message": str(result["message"]),
                }
                for result in report["results"]
                if result["status"] != "success"
            ],
            "candidate_count_evaluation": dict(report["candidate_count_evaluation"]),
        }
        return self.observation_exporter.finalize(
            eligible_candidate_ids=[str(item["candidate_id"]) for item in stable_objects],
            source_final_report=self.output_dir / "final_report.json",
            source_final=source_final,
        )

    def _prepare_candidates(
        self,
        candidates: tuple[FinalCandidate, ...],
        start_joint_positions: tuple[float, ...],
    ) -> tuple[_PreparedFinalCandidate, ...]:
        attempts_by_candidate = tuple(
            make_final_camera_targets(
                candidate,
                world_pose_base=self.world_pose_base,
                policy=self.config.camera,
            )
            for candidate in candidates
        )
        all_targets = tuple(
            target
            for scale in self.config.camera.standoff_distance_scales
            for attempts in attempts_by_candidate
            for target, geometry in attempts
            if float(geometry["standoff_distance_scale"]) == scale
        )
        solutions = self.planner.find_collision_free_ik(
            all_targets,
            RobotState(self.planner.joint_names, start_joint_positions),
        )
        solutions_by_target: dict[str, list[IKSolution]] = {}
        for solution in solutions:
            solutions_by_target.setdefault(solution.target_id, []).append(solution)
        return tuple(
            _PreparedFinalCandidate(
                input_index=index,
                candidate=candidate,
                target_attempts=attempts,
                ik_solutions=tuple(
                    solution
                    for target, _geometry in attempts
                    for solution in solutions_by_target.get(target.target_id, ())
                ),
            )
            for index, (candidate, attempts) in enumerate(
                zip(candidates, attempts_by_candidate, strict=True)
            )
        )

    @staticmethod
    def _candidate_order_key(
        prepared: _PreparedFinalCandidate,
        current: tuple[float, ...],
    ) -> tuple[float, int]:
        minimum_scale = min(
            float(geometry["standoff_distance_scale"])
            for _target, geometry in prepared.target_attempts
        )
        preferred_target_ids = {
            target.target_id
            for target, geometry in prepared.target_attempts
            if float(geometry["standoff_distance_scale"]) == minimum_scale
        }
        preferred_solutions = tuple(
            solution
            for solution in prepared.ik_solutions
            if solution.target_id in preferred_target_ids
        )
        ordering_solutions = preferred_solutions or prepared.ik_solutions
        distance = min(
            (
                math.dist(current, solution.joint_positions)
                for solution in ordering_solutions
            ),
            default=math.inf,
        )
        return distance, prepared.input_index

    def _process_candidate(
        self,
        prepared: _PreparedFinalCandidate,
        current: tuple[float, ...],
    ) -> tuple[dict[str, Any], tuple[float, ...], dict[str, Any] | None]:
        candidate = prepared.candidate
        target_attempts = prepared.target_attempts
        route_dir = self.output_dir / "route" / candidate.candidate_id
        route_dir.mkdir(parents=True, exist_ok=True)
        base_result: dict[str, Any] = {
            "candidate_id": candidate.candidate_id,
            "status": "failed",
            "camera_target_attempts": [],
            "artifact_status": "not_attempted",
            "artifact_failures": [],
        }
        solutions_by_target: dict[str, list[IKSolution]] = {}
        for solution in prepared.ik_solutions:
            solutions_by_target.setdefault(solution.target_id, []).append(solution)
        feasible_attempts = [
            (target, geometry, min(
                solutions_by_target[target.target_id],
                key=lambda solution: math.dist(current, solution.joint_positions),
            ))
            for target, geometry in target_attempts
            if target.target_id in solutions_by_target
        ]
        feasible_attempts.sort(
            key=lambda item: (
                -float(item[1]["predicted_footprint_bbox_area_fraction"]),
                math.dist(current, item[2].joint_positions),
                float(item[1]["actual_viewing_distance_m"]),
            )
        )
        rejected_attempts = [
            {
                **geometry,
                "target_id": target.target_id,
                "ik_status": "failed",
                "status": "rejected",
                "message": "Collision-aware IK did not find a feasible camera pose.",
            }
            for target, geometry in target_attempts
            if target.target_id not in solutions_by_target
        ]
        base_result["camera_target_attempts"].extend(rejected_attempts)
        if not feasible_attempts:
            return (
                {
                    **base_result,
                    "failure_stage": "collision_free_ik",
                    "message": "No optical-roll camera pose passed collision-aware IK.",
                },
                current,
                None,
            )
        segment = None
        attempted_target_ids: set[str] = set()

        def planning_requests():
            for target, geometry, ik_solution in feasible_attempts:
                yield target, geometry, ik_solution, "direct_pose", None
            if not self.config.planning.enable_portal_continuation:
                return
            portal_targets = tuple(
                offset_pose_target_along_local_z(
                    target, self.config.planning.portal_offset_m
                )
                for target, _, _ in feasible_attempts
            )
            portal_solutions = self.planner.find_collision_free_ik(
                portal_targets,
                RobotState(self.planner.joint_names, current),
            )
            portal_distances: dict[str, float] = {}
            for solution in portal_solutions:
                distance = math.dist(current, solution.joint_positions)
                portal_distances[solution.target_id] = min(
                    distance,
                    portal_distances.get(solution.target_id, math.inf),
                )
            ordered = sorted(
                enumerate(feasible_attempts),
                key=lambda item: (
                    portal_distances.get(item[1][0].target_id, math.inf),
                    item[0],
                ),
            )
            for _, (target, geometry, ik_solution) in ordered:
                portal_distance = portal_distances.get(target.target_id)
                yield (
                    target,
                    geometry,
                    ik_solution,
                    "cartesian_continuation",
                    portal_distance,
                )

        for attempt_index, (
            target,
            geometry,
            ik_solution,
            requested_strategy,
            portal_ik_distance,
        ) in enumerate(planning_requests()):
            attempted_target_ids.add(target.target_id)
            ik_distance = math.dist(current, ik_solution.joint_positions)
            try:
                attempted_plan = plan_pose_route(
                    self.planner,
                    (target,),
                    RobotState(self.planner.joint_names, current),
                    strategy=requested_strategy,
                )
            except Exception as exc:
                base_result["camera_target_attempts"].append(
                    {
                        **geometry,
                        "target_id": target.target_id,
                        "ik_status": "success",
                        "ik_joint_distance": ik_distance,
                        "portal_ik_joint_distance": portal_ik_distance,
                        "requested_planning_strategy": requested_strategy,
                        "status": "failed",
                        "message": _error_message(exc),
                    }
                )
                return (
                    {**base_result, "failure_stage": "curobo_planning", "message": _error_message(exc)},
                    current,
                    None,
                )
            attempt_path = route_dir / f"attempt_{attempt_index:02d}.json"
            attempt_path.write_text(json.dumps(attempted_plan.to_dict(), indent=2), encoding="utf-8")
            attempted_segment = attempted_plan.segments[0] if attempted_plan.segments else None
            executable = bool(
                attempted_plan.success
                and attempted_segment is not None
                and attempted_segment.success
                and attempted_segment.trajectory
            )
            base_result["camera_target_attempts"].append(
                {
                    **geometry,
                    "target_id": target.target_id,
                    "ik_status": "success",
                    "ik_joint_distance": ik_distance,
                    "portal_ik_joint_distance": portal_ik_distance,
                    "requested_planning_strategy": requested_strategy,
                    "status": "success" if executable else "failed",
                    "message": (
                        attempted_segment.message
                        if attempted_segment is not None
                        else attempted_plan.message
                    ),
                    "planner_artifact": str(attempt_path),
                    "planning_strategy": (
                        attempted_segment.planning_strategy
                        if attempted_segment is not None
                        else None
                    ),
                    "planning_time_s": (
                        attempted_segment.planning_time_s
                        if attempted_segment is not None
                        else None
                    ),
                    "portal_offset_m": (
                        attempted_segment.continuation_offset_m
                        if attempted_segment is not None
                        else None
                    ),
                }
            )
            if executable:
                segment = attempted_segment
                base_result["planner_artifact"] = str(attempt_path)
                base_result["camera_target"] = geometry
                base_result["planning_strategy"] = attempted_segment.planning_strategy
                break
        for target, geometry, ik_solution in feasible_attempts:
            if target.target_id in attempted_target_ids:
                continue
            base_result["camera_target_attempts"].append(
                {
                    **geometry,
                    "target_id": target.target_id,
                    "ik_status": "success",
                    "ik_joint_distance": math.dist(current, ik_solution.joint_positions),
                    "status": "not_attempted",
                    "message": "An earlier collision-free roll produced an executable trajectory.",
                }
            )
        if segment is None:
            return (
                {
                    **base_result,
                    "failure_stage": "curobo_planning",
                    "message": "cuRobo did not find an executable opening-line camera pose.",
                },
                current,
                None,
            )

        terminal = segment.trajectory[-1]
        try:
            captured = self.capture.capture(candidate.candidate_id, terminal)
        except Exception as exc:
            return (
                {**base_result, "failure_stage": "mujoco_capture", "message": _error_message(exc)},
                terminal,
                None,
            )
        base_result["rgb_path"] = str(captured.rgb_path)
        try:
            batch = self.detector.detect(captured.frame)
        except Exception as exc:
            return (
                {**base_result, "failure_stage": "yolo_inference", "message": _error_message(exc)},
                terminal,
                None,
            )
        base_result["detection_report"] = batch.source_report
        annotated_path = None
        try:
            annotated_path = render_detection_overlay(
                captured.rgb_path,
                batch.detections,
                self.output_dir / "annotated" / f"{candidate.candidate_id}.png",
            )
        except Exception as exc:
            base_result["artifact_status"] = "failed"
            base_result["artifact_failures"] = [
                {
                    "artifact": "annotated_rgb",
                    "failure_stage": "detection_annotation",
                    "message": _error_message(exc),
                }
            ]
        else:
            base_result["artifact_status"] = "success"
            base_result["annotated_rgb_path"] = str(annotated_path)
        if not batch.detections:
            base_result["localization_failures"] = []
            base_result["localized_detection_count"] = 0
            return (
                {
                    **base_result,
                    "failure_stage": "yolo_no_detection",
                    "message": "Final YOLO inference produced no detections.",
                },
                terminal,
                None,
            )

        selected, localization_failures = self._select_detection(candidate, captured.frame, batch.detections)
        base_result["localization_failures"] = localization_failures
        base_result["localized_detection_count"] = (
            len(batch.detections) - len(localization_failures)
        )
        if selected is None:
            all_localizations_failed = len(localization_failures) == len(batch.detections)
            return (
                {
                    **base_result,
                    "failure_stage": (
                        "depth_localization"
                        if all_localizations_failed
                        else "candidate_association"
                    ),
                    "message": (
                        "All Final detections failed depth localization."
                        if all_localizations_failed
                        else "No localized Final detection passed the configured candidate distance."
                    ),
                },
                terminal,
                None,
            )
        detection, observation, distance = selected
        area_fraction = bbox_area_fraction(
            detection.bbox_xyxy,
            image_width=self.config.camera.image_width,
            image_height=self.config.camera.image_height,
        )
        stable = {
            "candidate_id": candidate.candidate_id,
            "detection_id": detection.detection_id,
            "class_name": detection.class_name,
            "display_name": detection.display_name,
            "confidence": detection.confidence,
            "bbox_xyxy": list(detection.bbox_xyxy),
            "bbox_area_fraction": area_fraction,
            "bottom_position_world": list(observation.position_world),
            "survey_candidate_xy_distance_m": distance,
            "rgb_path": str(captured.rgb_path),
            "annotated_rgb_path": (
                str(annotated_path) if annotated_path is not None else None
            ),
        }
        if self.observation_exporter is not None:
            try:
                export_result = self.observation_exporter.export(
                    candidate_id=candidate.candidate_id,
                    survey_bottom_position_world=candidate.bottom_position_world,
                    final_bottom_position_world=observation.position_world,
                    detection=detection,
                    frame=captured.frame,
                    rgb_path=captured.rgb_path,
                )
            except Exception as exc:
                export_result = self.observation_exporter.record_failure(
                    candidate.candidate_id,
                    _error_message(exc),
                )
            base_result["object_observation"] = export_result
            if export_result["status"] == "success":
                stable["object_observation_id"] = export_result["observation_id"]
        return (
            {
                **base_result,
                "status": "success",
                "failure_stage": None,
                "message": "Final capture was re-detected and associated with the Survey candidate.",
                "selected_detection": stable,
            },
            terminal,
            stable,
        )

    def _select_detection(
        self,
        candidate: FinalCandidate,
        frame: RgbdFrame,
        detections: tuple[Detection2D, ...],
    ) -> tuple[
        tuple[Detection2D, SurveyObservation, float] | None,
        list[dict[str, str]],
    ]:
        localized: list[tuple[float, float, str, Detection2D, SurveyObservation]] = []
        failures: list[dict[str, str]] = []
        for detection in detections:
            try:
                observation = localize_detection(frame, detection, self.config.localization)
            except ValueError as exc:
                failures.append({"detection_id": detection.detection_id, "message": str(exc)})
                continue
            distance = math.hypot(
                observation.position_world[0] - candidate.bottom_position_world[0],
                observation.position_world[1] - candidate.bottom_position_world[1],
            )
            localized.append(
                (distance, -detection.confidence, detection.detection_id, detection, observation)
            )
        if not localized:
            return None, failures
        distance, _, _, detection, observation = min(localized, key=lambda item: item[:3])
        if distance > self.config.association.maximum_xy_distance_m:
            return None, failures
        return (detection, observation, distance), failures


def _parse_candidate(value: Any, index: int) -> FinalCandidate:
    if not isinstance(value, dict):
        raise ValueError(f"Survey candidate[{index}] must be an object")
    candidate_id = value.get("candidate_id")
    if not isinstance(candidate_id, str) or _CANDIDATE_ID.fullmatch(candidate_id) is None:
        raise ValueError(f"Survey candidate[{index}] has an invalid candidate_id")
    position = _float_tuple(value.get("bottom_position_world"), 3, f"candidate[{index}].bottom_position_world")
    raw_polygon = value.get("footprint_polygon_xy")
    if not isinstance(raw_polygon, list) or len(raw_polygon) < 3:
        raise ValueError(f"Survey candidate[{index}] footprint must contain at least three points")
    polygon = tuple(
        _float_tuple(point, 2, f"candidate[{index}].footprint_polygon_xy[{point_index}]")
        for point_index, point in enumerate(raw_polygon)
    )
    if _polygon_area(polygon) <= 1e-10:
        raise ValueError(f"Survey candidate[{index}] footprint must have positive area")
    return FinalCandidate(
        candidate_id=candidate_id,
        bottom_position_world=position,
        footprint_polygon_xy=polygon,
    )


def _candidate_count_evaluation(
    *,
    expected_count: int,
    survey_candidate_count: int,
    recognized_candidate_count: int,
    completed: bool,
) -> dict[str, Any]:
    success = completed and recognized_candidate_count == expected_count
    return {
        "success": success,
        "completed": completed,
        "expected_count": expected_count,
        "survey_candidate_count": survey_candidate_count,
        "recognized_candidate_count": recognized_candidate_count,
        "missing_count": max(0, expected_count - recognized_candidate_count),
        "extra_count": max(0, recognized_candidate_count - expected_count),
        "message": (
            "Final did not complete, so the expected object count was not reached."
            if not completed
            else (
                "Final recognized the expected number of objects."
                if success
                else "Final completed, but the recognized object count differs from the expected count."
            )
        ),
    }


def _final_report_base(
    *,
    status: str,
    failure_stage: str | None,
    message: str,
    source_survey_report: Path,
    final_config_path: Path,
    expected_count: int,
    candidate_count: int,
    recognized_candidate_count: int,
    count_evaluation_completed: bool,
) -> dict[str, Any]:
    return {
        "schema": FINAL_REPORT_SCHEMA,
        "stage": "final",
        "status": status,
        "failure_stage": failure_stage,
        "message": message,
        "candidate_count": candidate_count,
        "processed_candidate_count": 0,
        "successful_candidate_count": recognized_candidate_count,
        "failed_candidate_count": 0,
        "unprocessed_candidate_count": candidate_count,
        "stable_objects": [],
        "results": [],
        "candidate_count_evaluation": _candidate_count_evaluation(
            expected_count=expected_count,
            survey_candidate_count=candidate_count,
            recognized_candidate_count=recognized_candidate_count,
            completed=count_evaluation_completed,
        ),
        "candidate_processing_order": [],
        "source_survey_report": str(source_survey_report),
        "final_config_path": str(final_config_path),
        "planner": None,
        "detection_source": None,
        "localization_policy": None,
        "ik_feasible_camera_target_count": 0,
        "ik_rejected_camera_target_count": 0,
        "motion_planning_attempt_count": 0,
        "artifact_generation_failure_count": 0,
        "object_observation_interface": {
            "enabled": None,
            "status": "not_run",
            "failure_stage": None,
            "manifest_status": "not_run",
            "source_final_status": status,
            "manifest_path": None,
            "eligible_object_count": 0,
            "prepared_object_count": 0,
            "exported_object_count": 0,
            "failed_object_count": 0,
            "message": "Final object observation export did not run.",
        },
        "simulation": None,
        "diagnostics_path": None,
        "evaluation_status": "not_run",
        "evaluation_path": None,
    }


def make_final_failure_report(
    *,
    failure_stage: str,
    message: str,
    source_survey_report: Path,
    final_config_path: Path,
    expected_object_count: int,
    candidate_count: int,
) -> dict[str, Any]:
    return _final_report_base(
        status="failed",
        failure_stage=failure_stage,
        message=message,
        source_survey_report=source_survey_report,
        final_config_path=final_config_path,
        expected_count=expected_object_count,
        candidate_count=candidate_count,
        recognized_candidate_count=0,
        count_evaluation_completed=False,
    )


def _validate_final_config(config: FinalConfig) -> None:
    camera = config.camera
    if not 0.0 < camera.vertical_fov_deg < 180.0:
        raise ValueError("Final camera vertical_fov_deg must be between 0 and 180")
    if not camera.optical_roll_degrees:
        raise ValueError("Final camera optical_roll_degrees must not be empty")
    if not camera.standoff_distance_scales or any(
        scale < 1.0 for scale in camera.standoff_distance_scales
    ):
        raise ValueError("Final camera standoff_distance_scales must all be at least 1")
    if config.association.maximum_xy_distance_m <= 0.0:
        raise ValueError("Final association maximum_xy_distance_m must be positive")
    observation_interface = config.object_observation_interface
    if observation_interface.mask_source != MASK_SOURCE:
        raise ValueError(
            f"Final object observation mask_source must be {MASK_SOURCE!r}"
        )
    if observation_interface.connected_component_connectivity not in {4, 8}:
        raise ValueError("Final object observation connectivity must be 4 or 8")
    ensure_frame_id(observation_interface.world_frame_id, "world_frame_id")
    ensure_frame_id(
        observation_interface.camera_optical_frame_id,
        "camera_optical_frame_id",
    )
    ensure_frame_id(observation_interface.support_frame_id, "support_frame_id")
    planning = config.planning
    if planning.position_tolerance_m <= 0.0 or planning.orientation_tolerance_rad <= 0.0:
        raise ValueError("Final planning pose tolerances must be positive")
    if planning.ik_solutions_per_target > planning.num_ik_seeds:
        raise ValueError("Final IK solutions per target cannot exceed num_ik_seeds")
    if planning.continuation_ik_solution_count > planning.num_ik_seeds:
        raise ValueError("Final portal IK solution count cannot exceed num_ik_seeds")
    if min(
        planning.portal_offset_m,
        planning.continuation_step_m,
        planning.continuation_joint_tolerance_rad,
        planning.continuation_stop_velocity_tolerance_rad_s,
    ) <= 0.0:
        raise ValueError("Final portal continuation distances and tolerances must be positive")
    inference = config.detection.inference
    if not 0.0 <= inference.confidence_threshold <= 1.0:
        raise ValueError("Final detection confidence threshold must be between 0 and 1")
    if not 0.0 <= inference.iou_threshold <= 1.0:
        raise ValueError("Final detection IoU threshold must be between 0 and 1")
    evaluation = config.evaluation
    if min(
        evaluation.truth_match_maximum_xy_distance_m,
        evaluation.input_candidate_position_tolerance_m,
        evaluation.output_position_tolerance_m,
        evaluation.camera_position_tolerance_m,
        evaluation.camera_orientation_tolerance_rad,
    ) <= 0.0:
        raise ValueError("Final evaluation tolerances must be positive")


def _resolve_path(path: Path, repo_root: Path) -> Path:
    resolved = path if path.is_absolute() else repo_root / path
    if not resolved.is_file():
        raise FileNotFoundError(f"Task1 Final input does not exist: {resolved}")
    return resolved.resolve()


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required for non-JSON Task1 Final inputs") from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"Task1 Final input must contain an object: {path}")
    return value


def _mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise ValueError(f"Task1 Final config field {key} must be an object")
    return item


def _string(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"Task1 Final config field {key} must be a non-empty string")
    return item


def _number(
    value: dict[str, Any], key: str, *, minimum: float | None = None
) -> float:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        raise ValueError(f"Task1 Final config field {key} must be a number")
    result = float(item)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"Task1 Final config field {key} is outside its valid range")
    return result


def _integer(value: dict[str, Any], key: str, *, minimum: int) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int) or item < minimum:
        raise ValueError(f"Task1 Final config field {key} must be an integer >= {minimum}")
    return item


def _boolean(value: dict[str, Any], key: str) -> bool:
    item = value.get(key)
    if not isinstance(item, bool):
        raise ValueError(f"Task1 Final config field {key} must be a boolean")
    return item


def _optional_string(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"Task1 Final config field {field} must be null or a string")
    return value


def _optional_integer(value: Any, field: str, *, minimum: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"Task1 Final config field {field} must be null or an integer >= {minimum}"
        )
    return value


def _optional_number(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Task1 Final config field {field} must be null or a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Task1 Final config field {field} must be finite")
    return result


def _float_tuple(value: Any, length: int, field: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{field} must contain {length} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{field} must contain finite values")
    return result


def _float_array(value: Any, field: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field} must be a non-empty array")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{field} must contain finite values")
    return result


def _look_at_camera_rotation(view_direction: np.ndarray) -> Rotation:
    camera_z = -view_direction
    camera_x = np.cross(np.asarray((0.0, 0.0, 1.0)), camera_z)
    if float(np.linalg.norm(camera_x)) <= 1e-8:
        camera_x = np.asarray((1.0, 0.0, 0.0))
    else:
        camera_x /= np.linalg.norm(camera_x)
    camera_y = np.cross(camera_z, camera_x)
    return Rotation.from_matrix(np.column_stack((camera_x, camera_y, camera_z)))


def _polygon_area(points: tuple[tuple[float, float], ...]) -> float:
    return abs(
        sum(
            left[0] * right[1] - right[0] * left[1]
            for left, right in zip(points, points[1:] + points[:1])
        )
    ) / 2.0


def _error_message(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"
