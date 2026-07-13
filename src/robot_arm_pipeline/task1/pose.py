from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_POSE_MODEL_REGISTRY = Path("configs/task1/pose_models.json")
DEFAULT_POSE_DEPTH_PIXEL_STRIDE = 1
DEFAULT_POSE_DEPTH_BBOX_EXPANSION_RATIO = 0.5
DEFAULT_POSE_MIN_OBSERVED_POINTS = 80
DEFAULT_POSE_MAX_OBSERVED_POINTS = 1800
DEFAULT_POSE_MODEL_SAMPLE_POINTS = 1800
DEFAULT_POSE_YAW_SEED_COUNT = 24
DEFAULT_POSE_ICP_MAX_ITERATIONS = 40
DEFAULT_POSE_ICP_TRIM_QUANTILE = 0.85
DEFAULT_POSE_EXTENT_QUANTILE = 0.01
DEFAULT_POSE_ICP_CONVERGENCE_COST = 1e-10
DEFAULT_POSE_DEPTH_LOWER_MARGIN_M = 0.003
DEFAULT_POSE_DEPTH_UPPER_MARGIN_M = 0.004
DEFAULT_POSE_XY_CROP_MARGIN_M = 0.03
DEFAULT_POSE_INLIER_DISTANCE_M = 0.01
DEFAULT_POSE_MIN_FITNESS = 0.90
DEFAULT_POSE_MAX_MODEL_RMSE_M = 0.005
DEFAULT_POSE_MAX_OBSERVATION_RMSE_M = 0.005
DEFAULT_POSE_SOLUTION_TRANSLATION_SEPARATION_M = 0.015
DEFAULT_POSE_SOLUTION_YAW_SEPARATION_DEG = 15.0
DEFAULT_POSE_MIN_AMBIGUITY_SCORE_GAP_RATIO = 0.03
DEFAULT_POSE_RANDOM_SEED = 17


@dataclass(frozen=True)
class PoseConfig:
    model_registry_path: Path = DEFAULT_POSE_MODEL_REGISTRY
    depth_pixel_stride: int = DEFAULT_POSE_DEPTH_PIXEL_STRIDE
    depth_bbox_expansion_ratio: float = DEFAULT_POSE_DEPTH_BBOX_EXPANSION_RATIO
    min_observed_points: int = DEFAULT_POSE_MIN_OBSERVED_POINTS
    max_observed_points: int = DEFAULT_POSE_MAX_OBSERVED_POINTS
    model_sample_points: int = DEFAULT_POSE_MODEL_SAMPLE_POINTS
    yaw_seed_count: int = DEFAULT_POSE_YAW_SEED_COUNT
    icp_max_iterations: int = DEFAULT_POSE_ICP_MAX_ITERATIONS
    icp_trim_quantile: float = DEFAULT_POSE_ICP_TRIM_QUANTILE
    extent_quantile: float = DEFAULT_POSE_EXTENT_QUANTILE
    icp_convergence_cost: float = DEFAULT_POSE_ICP_CONVERGENCE_COST
    depth_lower_margin_m: float = DEFAULT_POSE_DEPTH_LOWER_MARGIN_M
    depth_upper_margin_m: float = DEFAULT_POSE_DEPTH_UPPER_MARGIN_M
    xy_crop_margin_m: float = DEFAULT_POSE_XY_CROP_MARGIN_M
    inlier_distance_m: float = DEFAULT_POSE_INLIER_DISTANCE_M
    min_fitness: float = DEFAULT_POSE_MIN_FITNESS
    max_model_rmse_m: float = DEFAULT_POSE_MAX_MODEL_RMSE_M
    max_observation_rmse_m: float = DEFAULT_POSE_MAX_OBSERVATION_RMSE_M
    solution_translation_separation_m: float = DEFAULT_POSE_SOLUTION_TRANSLATION_SEPARATION_M
    solution_yaw_separation_deg: float = DEFAULT_POSE_SOLUTION_YAW_SEPARATION_DEG
    min_ambiguity_score_gap_ratio: float = DEFAULT_POSE_MIN_AMBIGUITY_SCORE_GAP_RATIO
    random_seed: int = DEFAULT_POSE_RANDOM_SEED
    plan_only: bool = False

    def validate(self) -> None:
        if self.depth_pixel_stride <= 0:
            raise ValueError("depth_pixel_stride must be positive")
        if self.min_observed_points <= 2:
            raise ValueError("min_observed_points must be greater than two")
        if self.max_observed_points < self.min_observed_points:
            raise ValueError("max_observed_points must be at least min_observed_points")
        if self.model_sample_points <= 2 or self.yaw_seed_count <= 0 or self.icp_max_iterations <= 0:
            raise ValueError("model_sample_points, yaw_seed_count, and icp_max_iterations must be positive")
        if not 0.0 < self.icp_trim_quantile <= 1.0:
            raise ValueError("icp_trim_quantile must be in (0, 1]")
        if not 0.0 <= self.extent_quantile < 0.5:
            raise ValueError("extent_quantile must be in [0, 0.5)")
        if self.icp_convergence_cost < 0.0:
            raise ValueError("icp_convergence_cost must be non-negative")
        non_negative = (
            self.depth_lower_margin_m,
            self.depth_upper_margin_m,
            self.depth_bbox_expansion_ratio,
            self.xy_crop_margin_m,
            self.solution_translation_separation_m,
            self.solution_yaw_separation_deg,
            self.min_ambiguity_score_gap_ratio,
        )
        if any(value < 0.0 for value in non_negative):
            raise ValueError("pose margins and ambiguity thresholds must be non-negative")
        if self.inlier_distance_m <= 0.0 or self.max_model_rmse_m <= 0.0 or self.max_observation_rmse_m <= 0.0:
            raise ValueError("pose distance thresholds must be positive")
        if not 0.0 <= self.min_fitness <= 1.0:
            raise ValueError("min_fitness must be in [0, 1]")


@dataclass(frozen=True)
class PoseModelSpec:
    model_id: str
    canonical_class_name: str
    detector_class_names: tuple[str, ...]
    body_name: str
    geom_name: str
    yaw_symmetry: str


@dataclass(frozen=True)
class Task1FinalPoseInput:
    path: Path
    task1_run_dir: Path
    scene_model_path: Path
    image_size: tuple[int, int]
    workspace: dict[str, float]
    stable_objects: tuple[dict[str, Any], ...]
    raw_payload: dict[str, Any]


def find_latest_pose_report(output_dir: Path | str) -> Path:
    root = Path(output_dir)
    candidates = sorted(root.glob("*/pose/pose_report.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no task1 pose reports found under {root}")
    return candidates[-1]


def load_task1_final_pose_input(path: Path | str) -> Task1FinalPoseInput:
    report_path = Path(path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "task1_final_report_v1":
        raise ValueError(f"unsupported task1 final report schema_version: {payload.get('schema_version')}")
    stable_objects = payload.get("stable_objects")
    if not isinstance(stable_objects, list):
        raise ValueError("task1 final report must contain a stable_objects list")
    scene_model_path = Path(str(payload.get("scene_model_path") or ""))
    if not scene_model_path.is_absolute():
        scene_model_path = scene_model_path.resolve()
    task1_run_dir = _task1_run_dir_from_report(report_path, payload.get("task1_run_dir"))
    return Task1FinalPoseInput(
        path=report_path,
        task1_run_dir=task1_run_dir,
        scene_model_path=scene_model_path,
        image_size=_image_size(payload.get("image_size")),
        workspace=dict(payload.get("workspace") or {}),
        stable_objects=tuple(dict(item) for item in stable_objects if isinstance(item, dict)),
        raw_payload=payload,
    )


def load_pose_model_registry(path: Path | str) -> tuple[dict[str, PoseModelSpec], str]:
    registry_path = Path(path)
    raw = registry_path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("schema_version") != "task1_pose_model_registry_v1":
        raise ValueError(f"unsupported pose model registry schema_version: {payload.get('schema_version')}")
    entries = payload.get("models")
    if not isinstance(entries, list) or not entries:
        raise ValueError("pose model registry must contain a non-empty models list")
    aliases: dict[str, PoseModelSpec] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("pose model entries must be JSON objects")
        symmetry = str(entry.get("yaw_symmetry") or "none")
        if symmetry not in {"none", "half_turn", "continuous"}:
            raise ValueError(f"unsupported yaw_symmetry: {symmetry}")
        names = _text_tuple(entry.get("detector_class_names"))
        spec = PoseModelSpec(
            model_id=_required_text(entry, "model_id"),
            canonical_class_name=_required_text(entry, "canonical_class_name"),
            detector_class_names=names,
            body_name=_required_text(entry, "body_name"),
            geom_name=_required_text(entry, "geom_name"),
            yaw_symmetry=symmetry,
        )
        for name in dict.fromkeys((*names, spec.canonical_class_name)):
            if name in aliases:
                raise ValueError(f"duplicate pose model class alias: {name}")
            aliases[name] = spec
    return aliases, hashlib.sha256(raw).hexdigest()


def run_task1_pose(final_report_path: Path | str, config: PoseConfig = PoseConfig()) -> dict[str, Any]:
    config.validate()
    final_input = load_task1_final_pose_input(final_report_path)
    models, registry_sha256 = load_pose_model_registry(config.model_registry_path)
    created_utc = datetime.now(timezone.utc).isoformat()
    pose_dir = final_input.task1_run_dir / "pose"
    report_path = pose_dir / "pose_report.json"
    planned_objects = [_planned_pose_object(item, models) for item in final_input.stable_objects]
    if config.plan_only:
        report = _pose_report(
            created_utc=created_utc,
            status="plan_only",
            message=f"Planned precise pose estimation for {len(planned_objects)} final stable objects.",
            final_input=final_input,
            config=config,
            registry_sha256=registry_sha256,
            report_path=report_path,
            planned_objects=planned_objects,
            posed_objects=[],
            unresolved_objects=[],
        )
        _write_json(report_path, report)
        return report

    pending: list[tuple[dict[str, Any], dict[str, Any], PoseModelSpec]] = []
    unresolved_objects: list[dict[str, Any]] = []
    for source_object in final_input.stable_objects:
        class_name = str(source_object.get("class_name") or "")
        spec = models.get(class_name)
        evidence = source_object.get("pose_evidence")
        if spec is None:
            unresolved_objects.append(
                _unresolved(source_object, reason="pose_model_not_registered", message=f"No pose model for class {class_name!r}.")
            )
            continue
        if not isinstance(evidence, dict) or evidence.get("schema_version") != "task1_final_pose_evidence_v1":
            unresolved_objects.append(
                _unresolved(
                    source_object,
                    reason="pose_evidence_schema_invalid",
                    message="Final stable object does not provide task1_final_pose_evidence_v1 data.",
                )
            )
            continue
        if evidence.get("status") != "ready":
            unresolved_objects.append(
                _unresolved(
                    source_object,
                    reason="pose_evidence_not_ready",
                    message="Final pose evidence is incomplete.",
                    diagnostics={
                        "evidence_status": evidence.get("status"),
                        "missing_fields": evidence.get("missing_fields", []),
                    },
                )
            )
            continue
        pending.append((source_object, evidence, spec))

    if not pending:
        report = _pose_report(
            created_utc=created_utc,
            status="failed",
            message=f"No grasp-ready poses could be estimated from {len(final_input.stable_objects)} Final stable objects.",
            final_input=final_input,
            config=config,
            registry_sha256=registry_sha256,
            report_path=report_path,
            planned_objects=planned_objects,
            posed_objects=[],
            unresolved_objects=unresolved_objects,
        )
        _write_json(report_path, report)
        return report

    try:
        backend = _PoseBackend(final_input.scene_model_path, config)
    except Exception as exc:
        unresolved_objects.extend(
            _unresolved(item, reason="pose_backend_unavailable", message=str(exc))
            for item, _, _ in pending
        )
        report = _pose_report(
            created_utc=created_utc,
            status="failed",
            message=f"Precise pose backend unavailable for {len(pending)} objects.",
            final_input=final_input,
            config=config,
            registry_sha256=registry_sha256,
            report_path=report_path,
            planned_objects=planned_objects,
            posed_objects=[],
            unresolved_objects=unresolved_objects,
        )
        _write_json(report_path, report)
        return report

    posed_objects: list[dict[str, Any]] = []
    for source_object, evidence, spec in pending:
        result = backend.estimate(source_object, evidence, spec, final_input.task1_run_dir)
        if result["status"] == "accepted":
            posed_objects.append(result["object"])
        else:
            unresolved_objects.append(result)

    if posed_objects and not unresolved_objects:
        status = "success"
    elif posed_objects:
        status = "partial"
    else:
        status = "failed"
    report = _pose_report(
        created_utc=created_utc,
        status=status,
        message=f"Resolved {len(posed_objects)}/{len(final_input.stable_objects)} precise object poses.",
        final_input=final_input,
        config=config,
        registry_sha256=registry_sha256,
        report_path=report_path,
        planned_objects=planned_objects,
        posed_objects=posed_objects,
        unresolved_objects=unresolved_objects,
    )
    _write_json(report_path, report)
    return report


class _PoseBackend:
    def __init__(self, scene_model_path: Path, config: PoseConfig) -> None:
        self.config = config
        self.np = _import_numpy()
        self.mujoco = _import_mujoco()
        self.trimesh = _import_trimesh()
        self.cKDTree = _import_ckdtree()
        if not scene_model_path.is_file():
            raise FileNotFoundError(f"pose scene model does not exist: {scene_model_path}")
        self.model = self.mujoco.MjModel.from_xml_path(str(scene_model_path.resolve()))
        self.data = self.mujoco.MjData(self.model)
        self.mujoco.mj_forward(self.model, self.data)
        self.model_clouds: dict[str, dict[str, Any]] = {}

    def estimate(
        self,
        source_object: dict[str, Any],
        evidence: dict[str, Any],
        spec: PoseModelSpec,
        run_dir: Path,
    ) -> dict[str, Any]:
        try:
            model_cloud = self._model_cloud(spec)
            observed = self._observed_points(source_object, evidence, model_cloud, run_dir)
            if len(observed) < self.config.min_observed_points:
                return _unresolved(
                    source_object,
                    reason="insufficient_depth_points",
                    message=(
                        f"Pose crop retained {len(observed)} points; "
                        f"requires at least {self.config.min_observed_points}."
                    ),
                    diagnostics={"observed_point_count": len(observed), "model_id": spec.model_id},
                )
            registration = register_planar_point_clouds(
                model_points_xy=model_cloud["sample_points"][:, :2],
                observed_points_xy=observed[:, :2],
                config=self.config,
                yaw_symmetry=spec.yaw_symmetry,
                cKDTree=self.cKDTree,
                trimesh=self.trimesh,
            )
            quality = self._quality(registration, spec, len(observed))
            if quality["status"] != "accepted":
                return _unresolved(
                    source_object,
                    reason=str(quality["reason"]),
                    message=str(quality["message"]),
                    diagnostics=quality,
                )
            transform = _planar_transform(
                registration["translation_xy"],
                registration["yaw_rad"],
                _source_body_z(source_object),
            )
            pose_hypotheses = _pose_hypotheses(
                registration,
                z=_source_body_z(source_object),
                yaw_symmetry=spec.yaw_symmetry,
            )
            refined = {
                "object_id": source_object.get("object_id"),
                "class_name": spec.canonical_class_name,
                "source_class_name": source_object.get("class_name"),
                "confidence": source_object.get("confidence"),
                "bbox_xyxy": source_object.get("bbox_xyxy"),
                "T_world_object": transform,
                "position_world": [transform[0][3], transform[1][3], transform[2][3]],
                "geometry_center_world": [
                    *registration["symmetry_center_world_xy"],
                    _round(_source_body_z(source_object)),
                ],
                "source_final_image_path": source_object.get("final_image_path"),
                "source_depth_path": source_object.get("depth_path"),
                "pose_symmetry": {
                    "yaw_symmetry": spec.yaw_symmetry,
                    "yaw_observability": _yaw_observability(spec.yaw_symmetry),
                    "representative_is_unique": spec.yaw_symmetry == "none",
                    "requires_symmetry_aware_consumer": spec.yaw_symmetry != "none",
                },
                "pose_hypotheses": pose_hypotheses,
                "pose_quality": quality,
                "notes": [
                    "CAD-aligned planar pose estimated from Final RGB-D evidence.",
                    "T_world_object is a representative that must be interpreted with pose_symmetry and pose_hypotheses.",
                    "No layout pose or MuJoCo target body pose was read during estimation.",
                ],
            }
            return {"status": "accepted", "object": refined}
        except Exception as exc:
            return _unresolved(source_object, reason="pose_estimation_failed", message=str(exc))

    def _model_cloud(self, spec: PoseModelSpec) -> dict[str, Any]:
        cached = self.model_clouds.get(spec.model_id)
        if cached is not None:
            return cached
        body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, spec.body_name)
        geom_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, spec.geom_name)
        if body_id < 0 or geom_id < 0:
            raise ValueError(f"pose model body/geom not found: {spec.body_name}/{spec.geom_name}")
        original_position = self.model.body_pos[body_id].copy()
        original_quaternion = self.model.body_quat[body_id].copy()
        try:
            self.model.body_pos[body_id] = self.np.zeros(3)
            self.model.body_quat[body_id] = self.np.asarray((1.0, 0.0, 0.0, 0.0))
            self.mujoco.mj_forward(self.model, self.data)
            if int(self.model.geom_type[geom_id]) != self.mujoco.mjtGeom.mjGEOM_MESH:
                raise ValueError(f"pose model geom is not a mesh: {spec.geom_name}")
            mesh_id = int(self.model.geom_dataid[geom_id])
            vertex_start = int(self.model.mesh_vertadr[mesh_id])
            vertex_count = int(self.model.mesh_vertnum[mesh_id])
            face_start = int(self.model.mesh_faceadr[mesh_id])
            face_count = int(self.model.mesh_facenum[mesh_id])
            vertices = self.np.asarray(
                self.model.mesh_vert[vertex_start : vertex_start + vertex_count], dtype=float
            ).copy()
            faces = self.np.asarray(self.model.mesh_face[face_start : face_start + face_count], dtype=int).copy()
            rotation = self.np.asarray(self.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
            position = self.np.asarray(self.data.geom_xpos[geom_id], dtype=float)
            vertices = vertices @ rotation.T + position
        finally:
            self.model.body_pos[body_id] = original_position
            self.model.body_quat[body_id] = original_quaternion
            self.mujoco.mj_forward(self.model, self.data)
        mesh = self.trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        sample_points, _ = self.trimesh.sample.sample_surface(
            mesh,
            self.config.model_sample_points,
            seed=self.config.random_seed,
        )
        cloud = {
            "sample_points": self.np.asarray(sample_points, dtype=float),
            "bounds": self.np.asarray(mesh.bounds, dtype=float),
            "center_xy": self.np.mean(sample_points[:, :2], axis=0),
            "radius_xy": float(
                self.np.max(self.np.linalg.norm(sample_points[:, :2] - self.np.mean(sample_points[:, :2], axis=0), axis=1))
            ),
        }
        self.model_clouds[spec.model_id] = cloud
        return cloud

    def _observed_points(
        self,
        source_object: dict[str, Any],
        evidence: dict[str, Any],
        model_cloud: dict[str, Any],
        run_dir: Path,
    ) -> Any:
        depth_path = _run_artifact_path(run_dir, evidence.get("depth_path"))
        if depth_path is None or not depth_path.is_file():
            raise FileNotFoundError(f"Final depth artifact is missing: {depth_path}")
        depth = self.np.load(depth_path)
        image_width, image_height = _image_size(evidence.get("image_size"))
        if depth.ndim != 2 or depth.shape != (image_height, image_width):
            raise ValueError(
                f"Final depth shape {depth.shape} does not match pose evidence image size "
                f"{image_width}x{image_height}."
            )
        camera_transform = _matrix4(evidence.get("T_world_camera"), "T_world_camera")
        fovy_rad = float(evidence.get("camera_fovy_rad"))
        if not 0.0 < fovy_rad < math.pi:
            raise ValueError("camera_fovy_rad must be in (0, pi)")
        bbox = _expanded_bbox(
            _bbox(evidence.get("bbox_xyxy")),
            expansion_ratio=self.config.depth_bbox_expansion_ratio,
        )
        left, top, right, bottom = _bbox_pixel_bounds(bbox, depth.shape[1], depth.shape[0])
        ys = self.np.arange(top, bottom, self.config.depth_pixel_stride, dtype=int)
        xs = self.np.arange(left, right, self.config.depth_pixel_stride, dtype=int)
        if ys.size == 0 or xs.size == 0:
            return self.np.empty((0, 3), dtype=float)
        pixel_x, pixel_y = self.np.meshgrid(xs, ys)
        depth_values = depth[pixel_y, pixel_x]
        valid = self.np.isfinite(depth_values) & (depth_values > 0.0)
        if not bool(self.np.any(valid)):
            return self.np.empty((0, 3), dtype=float)
        focal = (depth.shape[0] / 2.0) / math.tan(fovy_rad / 2.0)
        values = depth_values[valid]
        points_camera = self.np.stack(
            (
                (pixel_x[valid] - depth.shape[1] / 2.0) / focal * values,
                (depth.shape[0] / 2.0 - pixel_y[valid]) / focal * values,
                -values,
            ),
            axis=1,
        )
        points_world = points_camera @ camera_transform[:3, :3].T + camera_transform[:3, 3]
        body_z = _source_body_z(source_object)
        bounds = model_cloud["bounds"]
        lower_z = max(
            body_z - self.config.depth_lower_margin_m,
            body_z + float(bounds[0, 2]) - self.config.depth_lower_margin_m,
        )
        upper_z = body_z + float(bounds[1, 2]) + self.config.depth_upper_margin_m
        points_world = points_world[
            (points_world[:, 2] >= lower_z) & (points_world[:, 2] <= upper_z)
        ]
        source_center = self.np.asarray(_position(source_object.get("position_world"))[:2], dtype=float)
        radius = float(model_cloud["radius_xy"]) + self.config.xy_crop_margin_m
        points_world = points_world[
            self.np.linalg.norm(points_world[:, :2] - source_center, axis=1) <= radius
        ]
        if len(points_world) > self.config.max_observed_points:
            indices = self.np.linspace(
                0,
                len(points_world) - 1,
                self.config.max_observed_points,
                dtype=int,
            )
            points_world = points_world[indices]
        return points_world

    def _quality(self, registration: dict[str, Any], spec: PoseModelSpec, point_count: int) -> dict[str, Any]:
        quality = {
            "policy_version": "task1_cad_depth_planar_pose_quality_v1",
            "method": "cad_depth_trimmed_planar_icp",
            "model_id": spec.model_id,
            "canonical_class_name": spec.canonical_class_name,
            "pose_frame": "registered_mujoco_target_body_frame",
            "capability": "planar_x_y_yaw_with_declared_body_z",
            "observed_point_count": point_count,
            "model_point_count": self.config.model_sample_points,
            "yaw_symmetry": spec.yaw_symmetry,
            "yaw_observability": _yaw_observability(spec.yaw_symmetry),
            **registration,
            "thresholds": {
                "min_fitness": self.config.min_fitness,
                "max_model_rmse_m": self.config.max_model_rmse_m,
                "max_observation_rmse_m": self.config.max_observation_rmse_m,
                "min_ambiguity_score_gap_ratio": self.config.min_ambiguity_score_gap_ratio,
            },
        }
        if float(registration["fitness"]) < self.config.min_fitness:
            quality.update(
                {"status": "limited", "reason": "low_registration_fitness", "message": "CAD registration fitness is below policy."}
            )
        elif float(registration["model_rmse_m"]) > self.config.max_model_rmse_m:
            quality.update(
                {"status": "limited", "reason": "high_model_rmse", "message": "CAD-to-depth RMSE exceeds policy."}
            )
        elif float(registration["observation_rmse_m"]) > self.config.max_observation_rmse_m:
            quality.update(
                {"status": "limited", "reason": "high_observation_rmse", "message": "Depth-to-CAD RMSE exceeds policy."}
            )
        elif (
            registration.get("ambiguity_score_gap_ratio") is not None
            and float(registration["ambiguity_score_gap_ratio"]) < self.config.min_ambiguity_score_gap_ratio
        ):
            quality.update(
                {"status": "limited", "reason": "registration_ambiguous", "message": "Multiple distinct CAD poses have similar registration scores."}
            )
        else:
            quality.update(
                {
                    "status": "accepted",
                    "reason": (
                        "quality_policy_passed_with_declared_symmetry"
                        if spec.yaw_symmetry != "none"
                        else "quality_policy_passed"
                    ),
                    "message": "CAD-depth pose passed registration quality policy.",
                }
            )
        return quality


def register_planar_point_clouds(
    *,
    model_points_xy: Any,
    observed_points_xy: Any,
    config: PoseConfig,
    yaw_symmetry: str = "none",
    cKDTree: Any | None = None,
    trimesh: Any | None = None,
) -> dict[str, Any]:
    np = _import_numpy()
    tree_type = cKDTree or _import_ckdtree()
    trimesh_module = trimesh or _import_trimesh()
    model_points = np.asarray(model_points_xy, dtype=float)
    observed_points = np.asarray(observed_points_xy, dtype=float)
    if model_points.ndim != 2 or model_points.shape[1] != 2 or len(model_points) < 3:
        raise ValueError("model_points_xy must contain at least three 2D points")
    if observed_points.ndim != 2 or observed_points.shape[1] != 2 or len(observed_points) < 3:
        raise ValueError("observed_points_xy must contain at least three 2D points")
    if yaw_symmetry not in {"none", "half_turn", "continuous"}:
        raise ValueError(f"unsupported yaw_symmetry: {yaw_symmetry}")
    observed_center = np.median(observed_points, axis=0)
    model_center = np.mean(model_points, axis=0)
    model_symmetry_center = _robust_extent_center(
        model_points,
        axes=_principal_axes(model_points),
        extent_quantile=config.extent_quantile,
    )
    solutions: list[dict[str, Any]] = []
    for yaw in np.linspace(-math.pi, math.pi, config.yaw_seed_count, endpoint=False):
        matrix = _matrix3_from_yaw(float(yaw))
        matrix[:2, 2] = observed_center - matrix[:2, :2] @ model_center
        matrix = _trimmed_planar_icp(
            model_points,
            observed_points,
            matrix,
            config=config,
            cKDTree=tree_type,
            trimesh=trimesh_module,
        )
        matrix = _refine_symmetric_translation(
            model_points,
            observed_points,
            matrix,
            yaw_symmetry=yaw_symmetry,
            extent_quantile=config.extent_quantile,
        )
        solutions.append(
            _registration_solution(
                model_points,
                observed_points,
                matrix,
                model_center=model_center,
                model_symmetry_center=model_symmetry_center,
                config=config,
                cKDTree=tree_type,
            )
        )
    unique_solutions = _unique_pose_solutions(
        solutions,
        config=config,
        yaw_symmetry=yaw_symmetry,
    )
    best = unique_solutions[0]
    alternative = unique_solutions[1] if len(unique_solutions) > 1 else None
    gap = None
    if alternative is not None:
        gap = (float(alternative["score"]) - float(best["score"])) / max(float(best["score"]), 1e-12)
    return {
        "translation_xy": best["translation_xy"],
        "yaw_rad": best["yaw_rad"],
        "score": best["score"],
        "model_rmse_m": best["model_rmse_m"],
        "observation_rmse_m": best["observation_rmse_m"],
        "fitness": best["fitness"],
        "model_geometry_center_xy": [_round(value) for value in model_center],
        "geometry_center_world_xy": best["geometry_center_world_xy"],
        "model_symmetry_center_xy": [_round(value) for value in model_symmetry_center],
        "symmetry_center_world_xy": best["symmetry_center_world_xy"],
        "ambiguity_score_gap_ratio": _round(gap, 8) if gap is not None else None,
        "distinct_solution_count": len(unique_solutions),
        "alternative_solution_sample": alternative,
        "translation_refinement": (
            "robust_extent_center_alignment"
            if yaw_symmetry in {"half_turn", "continuous"}
            else "icp_only"
        ),
    }


def _trimmed_planar_icp(
    model_points: Any,
    observed_points: Any,
    initial_matrix: Any,
    *,
    config: PoseConfig,
    cKDTree: Any,
    trimesh: Any,
) -> Any:
    np = _import_numpy()
    matrix = np.asarray(initial_matrix, dtype=float).copy()
    observed_tree = cKDTree(observed_points)
    previous_cost = math.inf
    for _ in range(config.icp_max_iterations):
        transformed = _transform_xy(model_points, matrix)
        distances, indices = observed_tree.query(transformed, k=1)
        threshold = float(np.quantile(distances, config.icp_trim_quantile))
        keep = distances <= threshold
        if int(np.count_nonzero(keep)) < 3:
            break
        source = np.column_stack((transformed[keep], np.zeros(int(np.count_nonzero(keep)))))
        target = np.column_stack((observed_points[indices[keep]], np.zeros(int(np.count_nonzero(keep)))))
        delta4, _, cost = trimesh.registration.procrustes(
            source,
            target,
            reflection=False,
            scale=False,
        )
        delta = np.eye(3)
        delta[:2, :2] = delta4[:2, :2]
        delta[:2, 2] = delta4[:2, 3]
        matrix = delta @ matrix
        if previous_cost - float(cost) < config.icp_convergence_cost:
            break
        previous_cost = float(cost)
    return matrix


def _registration_solution(
    model_points: Any,
    observed_points: Any,
    matrix: Any,
    *,
    model_center: Any,
    model_symmetry_center: Any,
    config: PoseConfig,
    cKDTree: Any,
) -> dict[str, Any]:
    np = _import_numpy()
    transformed = _transform_xy(model_points, matrix)
    model_distances = cKDTree(observed_points).query(transformed, k=1)[0]
    observation_distances = cKDTree(transformed).query(observed_points, k=1)[0]
    model_rmse = _trimmed_rmse(model_distances, config.icp_trim_quantile)
    observation_rmse = _trimmed_rmse(observation_distances, config.icp_trim_quantile)
    fitness = 0.5 * (
        float(np.mean(model_distances <= config.inlier_distance_m))
        + float(np.mean(observation_distances <= config.inlier_distance_m))
    )
    score = model_rmse * model_rmse + observation_rmse * observation_rmse
    return {
        "translation_xy": [_round(matrix[0, 2]), _round(matrix[1, 2])],
        "geometry_center_world_xy": [
            _round(value)
            for value in _transform_xy(np.asarray(model_center, dtype=float).reshape(1, 2), matrix)[0]
        ],
        "symmetry_center_world_xy": [
            _round(value)
            for value in _transform_xy(
                np.asarray(model_symmetry_center, dtype=float).reshape(1, 2),
                matrix,
            )[0]
        ],
        "yaw_rad": _round(math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))),
        "score": _round(score, 12),
        "model_rmse_m": _round(model_rmse, 8),
        "observation_rmse_m": _round(observation_rmse, 8),
        "fitness": _round(fitness),
    }


def _refine_symmetric_translation(
    model_points: Any,
    observed_points: Any,
    matrix: Any,
    *,
    yaw_symmetry: str,
    extent_quantile: float,
) -> Any:
    if yaw_symmetry == "none":
        return matrix
    np = _import_numpy()
    refined = np.asarray(matrix, dtype=float).copy()
    transformed = _transform_xy(model_points, refined)
    axes = np.eye(2) if yaw_symmetry == "continuous" else refined[:2, :2] @ _principal_axes(model_points)
    model_center = _robust_extent_center(
        transformed,
        axes=axes,
        extent_quantile=extent_quantile,
    )
    observed_center = _robust_extent_center(
        observed_points,
        axes=axes,
        extent_quantile=extent_quantile,
    )
    refined[:2, 2] += observed_center - model_center
    return refined


def _principal_axes(points: Any) -> Any:
    np = _import_numpy()
    covariance = np.cov(np.asarray(points, dtype=float).T)
    _, eigenvectors = np.linalg.eigh(covariance)
    return eigenvectors


def _robust_extent_center(points: Any, *, axes: Any, extent_quantile: float) -> Any:
    np = _import_numpy()
    values = np.asarray(points, dtype=float)
    basis = np.asarray(axes, dtype=float)
    lower = float(extent_quantile)
    upper = 1.0 - lower
    center = np.zeros(2, dtype=float)
    for axis in basis.T:
        projection = values @ axis
        center += axis * (0.5 * sum(np.quantile(projection, (lower, upper))))
    return center


def _unique_pose_solutions(
    solutions: list[dict[str, Any]],
    *,
    config: PoseConfig,
    yaw_symmetry: str,
) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    yaw_threshold = math.radians(config.solution_yaw_separation_deg)
    for candidate in sorted(solutions, key=lambda item: float(item["score"])):
        if any(
            math.dist(candidate["symmetry_center_world_xy"], existing["symmetry_center_world_xy"])
            < config.solution_translation_separation_m
            and abs(
                _symmetric_angle_difference(
                    float(candidate["yaw_rad"]),
                    float(existing["yaw_rad"]),
                    yaw_symmetry=yaw_symmetry,
                )
            )
            < yaw_threshold
            for existing in unique
        ):
            continue
        unique.append(candidate)
    return unique


def _pose_report(
    *,
    created_utc: str,
    status: str,
    message: str,
    final_input: Task1FinalPoseInput,
    config: PoseConfig,
    registry_sha256: str,
    report_path: Path,
    planned_objects: list[dict[str, Any]],
    posed_objects: list[dict[str, Any]],
    unresolved_objects: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "task1_pose_report_v1",
        "stage": "pose",
        "status": status,
        "created_utc": created_utc,
        "message": message,
        "source_final_report_path": str(final_input.path),
        "source_final_status": final_input.raw_payload.get("status"),
        "task1_run_dir": str(final_input.task1_run_dir),
        "pose_dir": str(report_path.parent),
        "report_path": str(report_path),
        "scene_model_path": str(final_input.scene_model_path),
        "model_registry_path": str(config.model_registry_path),
        "model_registry_sha256": registry_sha256,
        "pose_config": _pose_config_payload(config),
        "planned_objects": planned_objects,
        "posed_objects": posed_objects,
        "grasp_ready_objects": posed_objects,
        "unresolved_objects": unresolved_objects,
        "pose_summary": {
            "input_stable_object_count": len(final_input.stable_objects),
            "grasp_ready_object_count": len(posed_objects),
            "unresolved_object_count": len(unresolved_objects),
            "unresolved_reason_counts": dict(
                sorted(Counter(str(item.get("reason") or "unknown") for item in unresolved_objects).items())
            ),
        },
        "grasp_contract": {
            "pose_field": "T_world_object",
            "symmetric_pose_fields": ["pose_symmetry", "pose_hypotheses"],
            "consumer_requirement": (
                "A grasp consumer must honor pose_symmetry and evaluate pose_hypotheses when "
                "requires_symmetry_aware_consumer is true."
            ),
        },
        "notes": [
            "Pose consumes only Final RGB-D artifacts, camera calibration, stable object fields, and the CAD registry.",
            "The current capability assumes Task1 objects are placed on the tank bottom and estimates planar x/y/yaw.",
            "Symmetric poses remain explicit representative hypotheses; failed quality checks are excluded from grasp_ready_objects.",
        ],
    }


def _planned_pose_object(source_object: dict[str, Any], models: dict[str, PoseModelSpec]) -> dict[str, Any]:
    class_name = str(source_object.get("class_name") or "")
    spec = models.get(class_name)
    return {
        "object_id": source_object.get("object_id"),
        "class_name": source_object.get("class_name"),
        "model_id": spec.model_id if spec else None,
        "status": "planned" if spec else "model_not_registered",
    }


def _unresolved(
    source_object: dict[str, Any],
    *,
    reason: str,
    message: str,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "unresolved",
        "object_id": source_object.get("object_id"),
        "class_name": source_object.get("class_name"),
        "reason": reason,
        "message": message,
        "source_final_image_path": source_object.get("final_image_path"),
        "source_depth_path": source_object.get("depth_path"),
        "diagnostics": diagnostics or {},
    }


def _run_artifact_path(run_dir: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path

    resolved_run_dir = run_dir.resolve()
    max_prefix_length = min(len(resolved_run_dir.parts), len(path.parts))
    for prefix_length in range(max_prefix_length, 0, -1):
        if resolved_run_dir.parts[-prefix_length:] == path.parts[:prefix_length]:
            return resolved_run_dir.joinpath(*path.parts[prefix_length:])
    return resolved_run_dir / path


def _task1_run_dir_from_report(report_path: Path, declared_value: Any) -> Path:
    report_run_dir = report_path.resolve().parent.parent
    if not declared_value:
        return report_run_dir

    declared_run_dir = Path(str(declared_value))
    if declared_run_dir.is_absolute():
        return declared_run_dir
    if len(declared_run_dir.parts) <= len(report_run_dir.parts):
        if report_run_dir.parts[-len(declared_run_dir.parts) :] == declared_run_dir.parts:
            return report_run_dir
    return declared_run_dir.resolve()


def _source_body_z(source_object: dict[str, Any]) -> float:
    transform = _matrix4(source_object.get("T_world_object"), "T_world_object")
    return float(transform[2, 3])


def _planar_transform(position_xy: Any, yaw_rad: float, z: float) -> list[list[float]]:
    x, y = float(position_xy[0]), float(position_xy[1])
    cosine, sine = math.cos(float(yaw_rad)), math.sin(float(yaw_rad))
    return [
        [_round(cosine), _round(-sine), 0.0, _round(x)],
        [_round(sine), _round(cosine), 0.0, _round(y)],
        [0.0, 0.0, 1.0, _round(z)],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _pose_hypotheses(
    registration: dict[str, Any],
    *,
    z: float,
    yaw_symmetry: str,
) -> list[dict[str, Any]]:
    solutions = [
        {
            "rank": 1,
            "T_world_object": _planar_transform(
                registration["translation_xy"],
                registration["yaw_rad"],
                z,
            ),
            "translation_xy": registration["translation_xy"],
            "yaw_rad": registration["yaw_rad"],
            "score": registration["score"],
        }
    ]
    if yaw_symmetry == "half_turn":
        alternative_yaw = _angle_difference(float(registration["yaw_rad"]) + math.pi, 0.0)
        alternative_translation = _translation_for_rotated_center(
            model_center=registration["model_symmetry_center_xy"],
            world_center=registration["symmetry_center_world_xy"],
            yaw_rad=alternative_yaw,
        )
        solutions.append(
            {
                "rank": 2,
                "T_world_object": _planar_transform(
                    alternative_translation,
                    alternative_yaw,
                    z,
                ),
                "translation_xy": alternative_translation,
                "yaw_rad": _round(alternative_yaw),
                "score": registration["score"],
                "source": "declared_half_turn_symmetry",
            }
        )
    return solutions


def _translation_for_rotated_center(
    *,
    model_center: Any,
    world_center: Any,
    yaw_rad: float,
) -> list[float]:
    cosine, sine = math.cos(yaw_rad), math.sin(yaw_rad)
    model_x, model_y = float(model_center[0]), float(model_center[1])
    return [
        _round(float(world_center[0]) - (cosine * model_x - sine * model_y)),
        _round(float(world_center[1]) - (sine * model_x + cosine * model_y)),
    ]


def _matrix3_from_yaw(yaw_rad: float) -> Any:
    np = _import_numpy()
    cosine, sine = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.asarray(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)), dtype=float)


def _transform_xy(points: Any, matrix: Any) -> Any:
    return points @ matrix[:2, :2].T + matrix[:2, 2]


def _trimmed_rmse(distances: Any, quantile: float) -> float:
    np = _import_numpy()
    values = np.asarray(distances, dtype=float)
    threshold = float(np.quantile(values, quantile))
    kept = values[values <= threshold]
    return float(np.sqrt(np.mean(kept * kept)))


def _angle_difference(left: float, right: float) -> float:
    return (left - right + math.pi) % (2.0 * math.pi) - math.pi


def _symmetric_angle_difference(left: float, right: float, *, yaw_symmetry: str) -> float:
    if yaw_symmetry == "continuous":
        return 0.0
    if yaw_symmetry == "half_turn":
        return (left - right + math.pi / 2.0) % math.pi - math.pi / 2.0
    return _angle_difference(left, right)


def _yaw_observability(yaw_symmetry: str) -> str:
    if yaw_symmetry == "continuous":
        return "unobservable_within_continuous_equivalence"
    if yaw_symmetry == "half_turn":
        return "observable_modulo_180_degrees"
    return "unique"


def _bbox(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("bbox_xyxy must contain four values")
    values = tuple(float(item) for item in value)
    if values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError("bbox_xyxy must have positive area")
    return values


def _bbox_pixel_bounds(
    bbox: tuple[float, float, float, float], image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    return (
        max(0, min(image_width - 1, int(math.floor(bbox[0])))),
        max(0, min(image_height - 1, int(math.floor(bbox[1])))),
        max(1, min(image_width, int(math.ceil(bbox[2])))),
        max(1, min(image_height, int(math.ceil(bbox[3])))),
    )


def _expanded_bbox(
    bbox: tuple[float, float, float, float],
    *,
    expansion_ratio: float,
) -> tuple[float, float, float, float]:
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return (
        bbox[0] - width * expansion_ratio,
        bbox[1] - height * expansion_ratio,
        bbox[2] + width * expansion_ratio,
        bbox[3] + height * expansion_ratio,
    )


def _position(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("position_world must contain three values")
    return tuple(float(item) for item in value)


def _matrix4(value: Any, name: str) -> Any:
    np = _import_numpy()
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix")
    return matrix


def _image_size(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("image_size must contain width and height")
    width, height = int(value[0]), int(value[1])
    if width <= 0 or height <= 0:
        raise ValueError("image_size values must be positive")
    return width, height


def _text_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("detector_class_names must be a non-empty list")
    values = tuple(str(item).strip() for item in value if str(item).strip())
    if not values:
        raise ValueError("detector_class_names must contain at least one non-empty name")
    return values


def _pose_config_payload(config: PoseConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["model_registry_path"] = str(config.model_registry_path)
    return payload


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"pose model entry requires {key}")
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _round(value: Any, digits: int = 6) -> float:
    return round(float(value), digits)


def _import_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Task1 pose requires numpy") from exc
    return np


def _import_mujoco() -> Any:
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("Task1 pose requires mujoco") from exc
    return mujoco


def _import_trimesh() -> Any:
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("Task1 pose requires trimesh") from exc
    return trimesh


def _import_ckdtree() -> Any:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("Task1 pose requires scipy") from exc
    return cKDTree
