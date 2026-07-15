from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_arm_pipeline.planning import MotionPlanResult
from task1.detection import Detector, render_detection_overlay
from task1.vision import (
    CameraIntrinsics,
    CandidateLocalizationPolicy,
    Detection2D,
    RgbdFrame,
    SurveyObservation,
    fuse_observations_with_report,
    localize_detection,
    observation_surface_geometry,
)
from task1.survey.route import SURVEY_VIEWS, SurveyView
from task1.survey.config import YoloEvaluationPolicy
from task1.survey.evaluation import (
    GroundTruthBox,
    YoloEvaluationFrame,
    diagnose_survey_detection_pipeline,
    evaluate_yolo_detections,
)


DEFAULT_MODEL_PATH = Path("examples/mujoco/gen3_with_tank.xml")
DEFAULT_CAMERA_NAME = "gen3_wrist"


@dataclass(frozen=True)
class SimulationConfig:
    repo_root: Path
    model_path: Path = DEFAULT_MODEL_PATH
    camera_name: str = DEFAULT_CAMERA_NAME
    image_width: int = 1920
    image_height: int = 1080
    candidate_position_tolerance_m: float = 0.05
    retain_depth_artifacts: bool = False

    def resolved_model_path(self) -> Path:
        path = self.model_path if self.model_path.is_absolute() else self.repo_root / self.model_path
        if not path.exists():
            raise FileNotFoundError(f"MuJoCo survey model does not exist: {path}")
        return path.resolve()


@dataclass(frozen=True)
class _SurveyEvaluationCapture:
    view_id: str
    joint_positions: tuple[float, ...]
    ground_z_m: float
    detections: tuple[Detection2D, ...]


class MujocoSurveySimulation:
    """Render survey RGB-D frames in MuJoCo and run the selected detector."""

    def __init__(
        self,
        config: SimulationConfig,
        layout_path: Path,
        output_dir: Path,
        detector: Detector,
        yolo_evaluation_policy: YoloEvaluationPolicy | None = None,
    ) -> None:
        self.config = config
        self.layout_path = layout_path
        self.output_dir = output_dir
        self.detector = detector
        self.yolo_evaluation_policy = yolo_evaluation_policy
        self._mujoco: Any | None = None
        self._model: Any | None = None
        self._data: Any | None = None
        self._renderer: Any | None = None
        self._selected_objects: dict[str, dict[str, Any]] = {}
        self._evaluation_captures: list[_SurveyEvaluationCapture] = []
        self._production_observations: tuple[SurveyObservation, ...] = ()

    def run(
        self,
        route_plan: MotionPlanResult,
        planner_artifact: str,
        policy: CandidateLocalizationPolicy = CandidateLocalizationPolicy(),
    ) -> dict[str, Any]:
        if not route_plan.success:
            return self._report(
                status="failed",
                failure_stage="survey_route",
                view_reports=[],
                observations=[],
                candidates=[],
                localization_failures=[],
                message=route_plan.message,
                planner_artifact=planner_artifact,
                localization_policy=policy,
            )

        self._evaluation_captures = []
        self._production_observations = ()
        view_reports: list[dict[str, Any]] = []
        observations: list[SurveyObservation] = []
        localization_failures: list[dict[str, str]] = []
        try:
            self._load()
            segment_by_view = {segment.target_id: segment for segment in route_plan.segments}
            for view in SURVEY_VIEWS:
                segment = segment_by_view.get(view.view_id)
                if segment is None or not segment.trajectory:
                    raise ValueError(f"route has no executable trajectory for {view.view_id}")
                terminal_joint_positions = segment.trajectory[-1]
                frame, detections, paths, detection_report = self._capture(view, terminal_joint_positions)
                self._evaluation_captures.append(
                    _SurveyEvaluationCapture(
                        view_id=view.view_id,
                        joint_positions=terminal_joint_positions,
                        ground_z_m=frame.ground_z_m,
                        detections=detections,
                    )
                )
                localized_count = 0
                for detection in detections:
                    try:
                        observations.append(localize_detection(frame, detection, policy))
                        localized_count += 1
                    except ValueError as exc:
                        localization_failures.append(
                            {"view_id": view.view_id, "detection_id": detection.detection_id, "message": str(exc)}
                        )
                view_reports.append(
                    {
                        "view_id": view.view_id,
                        "status": "success",
                        "detection_count": len(detections),
                        "localized_detection_count": localized_count,
                        "detection_report": detection_report,
                        **paths,
                    }
                )
        except Exception as exc:
            return self._report(
                status="failed",
                failure_stage="survey_execution",
                view_reports=view_reports,
                observations=observations,
                candidates=[],
                localization_failures=localization_failures,
                message=f"MuJoCo survey production failed: {type(exc).__name__}: {exc}",
                planner_artifact=planner_artifact,
                localization_policy=policy,
            )
        finally:
            self.close()

        fusion_result = fuse_observations_with_report(observations, policy)
        candidates = list(fusion_result.candidates)
        candidate_payloads = [candidate.to_dict() for candidate in candidates]
        self._production_observations = tuple(observations)
        return self._report(
            status="success",
            failure_stage=None,
            view_reports=view_reports,
            observations=observations,
            candidates=candidate_payloads,
            localization_failures=localization_failures,
            message=(
                f"Executed route and captured all {len(SURVEY_VIEWS)} views; localized {len(candidates)} candidates."
            ),
            fusion_diagnostics=fusion_result.diagnostics,
            planner_artifact=planner_artifact,
            localization_policy=policy,
        )

    def evaluate(
        self,
        production_report: dict[str, Any],
        policy: CandidateLocalizationPolicy = CandidateLocalizationPolicy(),
    ) -> dict[str, Any]:
        """Replay production viewpoints for simulation-only truth evaluation."""
        if (
            production_report.get("schema") != "task1_survey_report"
            or production_report.get("stage") != "survey"
            or production_report.get("status") != "success"
        ):
            raise ValueError("Survey evaluation requires a successful production report")
        if len(self._evaluation_captures) != len(SURVEY_VIEWS):
            raise ValueError("Survey evaluation requires all production capture viewpoints")
        raw_candidates = production_report.get("candidates")
        if not isinstance(raw_candidates, list):
            raise ValueError("Survey production report candidates must be a list")

        ideal_fusion_observations: list[SurveyObservation] = []
        yolo_evaluation_frames: list[YoloEvaluationFrame] = []
        try:
            self._load()
            for capture in self._evaluation_captures:
                ground_truth = self._capture_ground_truth(capture)
                yolo_evaluation_frames.append(
                    YoloEvaluationFrame(
                        capture.view_id,
                        ground_truth,
                        capture.detections,
                    )
                )
                for truth in ground_truth:
                    if (
                        self.yolo_evaluation_policy is not None
                        and truth.visible_pixel_count < self.yolo_evaluation_policy.min_visible_pixels
                    ):
                        continue
                    layout_object = self._selected_objects[truth.object_id]
                    reference_xy = _polygon_centroid(layout_object["footprint_polygon_xy"])
                    footprint_polygon, surface_covariance = observation_surface_geometry(
                        layout_object["footprint_polygon_xy"], policy
                    )
                    detection_id = f"{capture.view_id}:ideal:{truth.object_id}"
                    ideal_fusion_observations.append(
                        SurveyObservation(
                            view_id=capture.view_id,
                            detection_id=detection_id,
                            position_world=(
                                reference_xy[0],
                                reference_xy[1],
                                capture.ground_z_m,
                            ),
                            confidence=1.0,
                            class_name=truth.class_name,
                            foreground_sample_count=truth.visible_pixel_count,
                            radial_mad_m=0.0,
                            visible_surface_radius_m=0.0,
                            footprint_polygon_xy=footprint_polygon,
                            surface_covariance_xy=surface_covariance,
                            class_scores={truth.class_name: 1.0},
                            source_detection_ids=(detection_id,),
                        )
                    )
        finally:
            self.close()

        candidate_evaluation = self._evaluate_candidates(raw_candidates)
        oracle_fusion_evaluation = self._evaluate_candidates(
            [
                candidate.to_dict()
                for candidate in fuse_observations_with_report(ideal_fusion_observations, policy).candidates
            ]
        )
        yolo_evaluation = (
            evaluate_yolo_detections(
                tuple(yolo_evaluation_frames),
                self.yolo_evaluation_policy,
                min_supporting_views=policy.min_supporting_views,
            )
            if self.yolo_evaluation_policy is not None
            else None
        )
        detection_diagnosis = (
            diagnose_survey_detection_pipeline(
                yolo_evaluation=yolo_evaluation,
                observations=self._production_observations,
                candidate_evaluation=candidate_evaluation,
                oracle_fusion_evaluation=oracle_fusion_evaluation,
                min_supporting_views=policy.min_supporting_views,
            )
            if yolo_evaluation is not None
            else None
        )
        return {
            "schema": "task1_survey_simulation_evaluation",
            "status": "completed",
            "evaluation_only": True,
            "used_for_production_control": False,
            "candidate_position_evaluation": candidate_evaluation,
            "yolo_evaluation": yolo_evaluation,
            "detection_diagnosis": detection_diagnosis,
        }

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._mujoco = None
        self._model = None
        self._data = None

    def _load(self) -> None:
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("MuJoCo is required for the survey simulation") from exc
        payload = json.loads(self.layout_path.read_text(encoding="utf-8"))
        if payload.get("schema") != "target_object_pose_layout":
            raise ValueError("survey benchmark requires target_object_pose_layout")
        objects = payload.get("objects")
        if not isinstance(objects, list) or not objects:
            raise ValueError("survey benchmark layout must contain objects")
        self._selected_objects = {str(item["object_id"]): item for item in objects}
        self._mujoco = mujoco
        self._model = mujoco.MjModel.from_xml_path(str(self.config.resolved_model_path()))
        self._data = mujoco.MjData(self._model)
        self._renderer = mujoco.Renderer(self._model, self.config.image_height, self.config.image_width)
        self._apply_layout()

    def _apply_layout(self) -> None:
        mujoco, model, data = self._required_runtime()
        for body_id in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if not name.startswith("target_") or name == "target_object_include_root":
                continue
            item = self._selected_objects.get(name)
            if item is None:
                model.body_pos[body_id] = (0.0, 0.0, -10.0)
                continue
            model.body_pos[body_id] = tuple(float(value) for value in item["position"])
            yaw = float(item["yaw_rad"])
            model.body_quat[body_id] = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
        mujoco.mj_forward(model, data)

    def _capture(
        self,
        view: SurveyView,
        joint_positions: tuple[float, ...],
    ) -> tuple[
        RgbdFrame,
        tuple[Detection2D, ...],
        dict[str, Any],
        dict[str, Any],
    ]:
        mujoco, model, data = self._required_runtime()
        renderer = self._renderer
        if renderer is None:
            raise RuntimeError("MuJoCo renderer is not loaded")
        data.qpos[: len(joint_positions)] = joint_positions
        mujoco.mj_forward(model, data)

        image_dir = self.output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        rgb_path = image_dir / f"{view.view_id}.png"

        renderer.disable_depth_rendering()
        renderer.disable_segmentation_rendering()
        renderer.update_scene(data, camera=self.config.camera_name)
        rgb = renderer.render()
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required to save survey benchmark images") from exc
        Image.fromarray(rgb).save(rgb_path)

        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=self.config.camera_name)
        depth = np.asarray(renderer.render(), dtype=np.float32).copy()
        renderer.disable_depth_rendering()
        depth_path: Path | None = None
        if self.config.retain_depth_artifacts:
            depth_path = self.output_dir / "depth" / f"{view.view_id}.npy"
            depth_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(depth_path, depth)

        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, self.config.camera_name)
        if camera_id < 0:
            raise ValueError(f"MuJoCo camera does not exist: {self.config.camera_name}")
        fovy_rad = math.radians(float(model.cam_fovy[camera_id]))
        fy = (self.config.image_height / 2.0) / math.tan(fovy_rad / 2.0)
        intrinsics = CameraIntrinsics(
            width=self.config.image_width,
            height=self.config.image_height,
            fx=fy,
            fy=fy,
            cx=self.config.image_width / 2.0,
            cy=self.config.image_height / 2.0,
        )
        transform = _mujoco_camera_optical_transform(data, camera_id)
        frame = RgbdFrame(
            view_id=view.view_id,
            depth_m=depth,
            intrinsics=intrinsics,
            T_world_camera_optical=transform,
            ground_z_m=0.0,
            rgb_path=str(rgb_path),
            depth_path=str(depth_path) if depth_path is not None else None,
        )
        batch = self.detector.detect(frame)
        detections = batch.detections
        detection_report = batch.source_report
        annotated_rgb_path = self.output_dir / "annotated" / f"{view.view_id}.png"
        artifact_failures: list[dict[str, str]] = []
        try:
            render_detection_overlay(rgb_path, detections, annotated_rgb_path)
        except Exception as exc:
            artifact_failures.append(
                {
                    "artifact": "annotated_rgb",
                    "failure_stage": "detection_annotation",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
        artifact_paths = {
            "rgb_path": str(rgb_path),
            "annotated_rgb_path": (str(annotated_rgb_path) if not artifact_failures else None),
            "artifact_status": "failed" if artifact_failures else "success",
            "artifact_failures": artifact_failures,
        }
        if depth_path is not None:
            artifact_paths["depth_path"] = str(depth_path)
        return (
            frame,
            detections,
            artifact_paths,
            detection_report,
        )

    def _capture_ground_truth(
        self,
        capture: _SurveyEvaluationCapture,
    ) -> tuple[GroundTruthBox, ...]:
        mujoco, model, data = self._required_runtime()
        renderer = self._renderer
        if renderer is None:
            raise RuntimeError("MuJoCo renderer is not loaded")
        data.qpos[: len(capture.joint_positions)] = capture.joint_positions
        mujoco.mj_forward(model, data)
        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        try:
            renderer.update_scene(data, camera=self.config.camera_name)
            segmentation = np.asarray(renderer.render()).copy()
        finally:
            renderer.disable_segmentation_rendering()
        return self._ground_truth_boxes(segmentation)

    def _required_runtime(self) -> tuple[Any, Any, Any]:
        if self._mujoco is None or self._model is None or self._data is None:
            raise RuntimeError("MuJoCo benchmark is not loaded")
        return self._mujoco, self._model, self._data

    def _report(
        self,
        *,
        status: str,
        failure_stage: str | None,
        view_reports: list[dict[str, Any]],
        observations: list[SurveyObservation],
        candidates: list[dict[str, Any]],
        localization_failures: list[dict[str, str]],
        message: str,
        planner_artifact: str,
        localization_policy: CandidateLocalizationPolicy,
        fusion_diagnostics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        report = {
            "schema": "task1_survey_report",
            "stage": "survey",
            "status": status,
            "failure_stage": failure_stage,
            "message": message,
            "candidate_count": len(candidates),
            "candidates": candidates,
            "view_count": len(SURVEY_VIEWS),
            "processed_view_count": len(view_reports),
            "successful_view_count": sum(
                view.get("status") == "success" for view in view_reports
            ),
            "failed_view_count": sum(
                view.get("status") == "failed" for view in view_reports
            ),
            "unprocessed_view_count": len(SURVEY_VIEWS) - len(view_reports),
            "observation_count": len(observations),
            "localization_failure_count": len(localization_failures),
            "artifact_generation_failure_count": sum(
                len(view.get("artifact_failures", [])) for view in view_reports
            ),
            "planner_artifact": planner_artifact,
            "detection_source": {**self.detector.source_metadata(), "simulation_input": True},
            "candidate_localization_policy": localization_policy.to_dict(),
            "layout_path": str(self.layout_path),
            "simulation": {
                "enabled": True,
                "renderer": "mujoco",
                "image_width": self.config.image_width,
                "image_height": self.config.image_height,
                "camera_name": self.config.camera_name,
            },
            "artifact_retention": {
                "survey_depth": {
                    "enabled": self.config.retain_depth_artifacts,
                    "format": "npy_float32" if self.config.retain_depth_artifacts else None,
                }
            },
            "views": view_reports,
            "observations": [observation.to_dict() for observation in observations],
            "localization_failures": localization_failures,
            "fusion_diagnostics": fusion_diagnostics,
        }
        return report

    def _ground_truth_boxes(self, segmentation: np.ndarray) -> tuple[GroundTruthBox, ...]:
        mujoco, model, _ = self._required_runtime()
        if segmentation.shape != (self.config.image_height, self.config.image_width, 2):
            raise ValueError(f"unexpected MuJoCo segmentation shape: {segmentation.shape}")
        object_id = segmentation[:, :, 0]
        object_type = segmentation[:, :, 1]
        geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
        masks: dict[str, np.ndarray] = {}
        for geom_id in np.unique(object_id[object_type == geom_type]):
            geom_id_int = int(geom_id)
            if geom_id_int < 0 or geom_id_int >= model.ngeom:
                raise ValueError(f"MuJoCo segmentation returned invalid geom id: {geom_id_int}")
            target_id = self._target_body_for_geom(geom_id_int)
            if target_id is None or target_id not in self._selected_objects:
                continue
            geom_mask = (object_type == geom_type) & (object_id == geom_id_int)
            masks[target_id] = geom_mask if target_id not in masks else masks[target_id] | geom_mask

        boxes: list[GroundTruthBox] = []
        for target_id, mask in sorted(masks.items()):
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                continue
            layout_object = self._selected_objects[target_id]
            boxes.append(
                GroundTruthBox(
                    object_id=target_id,
                    class_name=str(layout_object["class_name"]),
                    bbox_xyxy=(
                        float(xs.min()),
                        float(ys.min()),
                        float(xs.max() + 1),
                        float(ys.max() + 1),
                    ),
                    visible_pixel_count=int(len(xs)),
                    pixel_mask=mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].copy(),
                )
            )
        return tuple(boxes)

    def _target_body_for_geom(self, geom_id: int) -> str | None:
        mujoco, model, _ = self._required_runtime()
        body_id = int(model.geom_bodyid[geom_id])
        while body_id > 0:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if name in self._selected_objects:
                return name
            body_id = int(model.body_parentid[body_id])
        return None

    def _evaluate_candidates(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        references = {
            str(item["class_name"]): (str(item["object_id"]), _polygon_centroid(item["footprint_polygon_xy"]))
            for item in self._selected_objects.values()
        }
        candidates_by_class: dict[str, list[dict[str, Any]]] = {}
        for candidate in candidates:
            votes = candidate.get("class_votes", {})
            if not votes:
                continue
            dominant_class = max(sorted(votes), key=lambda name: float(votes[name]))
            candidates_by_class.setdefault(dominant_class, []).append(candidate)

        errors: list[dict[str, Any]] = []
        missing: list[str] = []
        used_candidates: set[str] = set()
        for class_name, (object_id, reference) in sorted(references.items()):
            class_candidates = candidates_by_class.get(class_name, [])
            if len(class_candidates) != 1:
                missing.append(object_id)
                continue
            candidate = class_candidates[0]
            used_candidates.add(str(candidate["candidate_id"]))
            position = candidate["bottom_position_world"]
            error = math.hypot(float(position[0]) - reference[0], float(position[1]) - reference[1])
            errors.append(
                {
                    "object_id": object_id,
                    "class_name": class_name,
                    "candidate_id": candidate["candidate_id"],
                    "reference_footprint_center_xy": list(reference),
                    "candidate_xy": [float(position[0]), float(position[1])],
                    "xy_error_m": error,
                }
            )
        extras = sorted(str(candidate["candidate_id"]) for candidate in candidates if candidate["candidate_id"] not in used_candidates)
        max_error = max((item["xy_error_m"] for item in errors), default=None)
        mean_error = sum(item["xy_error_m"] for item in errors) / len(errors) if errors else None
        success = (
            not missing
            and not extras
            and len(errors) == len(references)
            and max_error is not None
            and max_error <= self.config.candidate_position_tolerance_m
        )
        return {
            "success": success,
            "position_tolerance_m": self.config.candidate_position_tolerance_m,
            "reference_count": len(references),
            "matched_count": len(errors),
            "missing_object_ids": missing,
            "extra_candidate_ids": extras,
            "mean_xy_error_m": mean_error,
            "max_xy_error_m": max_error,
            "matches": errors,
            "message": (
                "all candidate footprint centers are within tolerance"
                if success
                else "candidate/reference association or position tolerance failed"
            ),
        }


def _mujoco_camera_optical_transform(data: Any, camera_id: int) -> tuple[tuple[float, float, float, float], ...]:
    mujoco_rotation = np.asarray(data.cam_xmat[camera_id], dtype=float).reshape(3, 3)
    optical_rotation = np.column_stack(
        (mujoco_rotation[:, 0], -mujoco_rotation[:, 1], -mujoco_rotation[:, 2])
    )
    position = np.asarray(data.cam_xpos[camera_id], dtype=float)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = optical_rotation
    transform[:3, 3] = position
    return tuple(tuple(float(value) for value in row) for row in transform)


def _polygon_centroid(points: Any) -> tuple[float, float]:
    polygon = np.asarray(points, dtype=float)
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        raise ValueError("benchmark footprint polygon must contain at least three xy points")
    shifted = np.roll(polygon, -1, axis=0)
    cross = polygon[:, 0] * shifted[:, 1] - shifted[:, 0] * polygon[:, 1]
    signed_area_twice = float(np.sum(cross))
    if abs(signed_area_twice) < 1e-12:
        raise ValueError("benchmark footprint polygon has zero area")
    cx = float(np.sum((polygon[:, 0] + shifted[:, 0]) * cross) / (3.0 * signed_area_twice))
    cy = float(np.sum((polygon[:, 1] + shifted[:, 1]) * cross) / (3.0 * signed_area_twice))
    return cx, cy
