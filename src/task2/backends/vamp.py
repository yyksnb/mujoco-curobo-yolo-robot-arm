from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from math import ceil, isfinite, sqrt
from time import monotonic
from typing import Iterable, Mapping

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull as SpatialConvexHull, cKDTree

from ..contracts import (
    AttachedGeometry,
    Box,
    Capsule,
    ConvexHull,
    PointCloud,
    RigidTransform,
    RobotModel,
    SceneGeometry,
    SceneSnapshot,
    Sphere,
    StaticTriangleMesh,
    Vector3,
)
from ..contracts.geometry import compose, inverse
from .errors import BackendOperationError, BackendUnavailableError


@dataclass(frozen=True)
class VampPlan:
    path: tuple[tuple[float, ...], ...]
    elapsed_s: float
    iterations: int
    selected_goal_index: int = 0


@dataclass(frozen=True)
class _PreparedPolyhedron:
    vertices: np.ndarray
    face_normals: np.ndarray
    edge_directions: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray


@dataclass(frozen=True)
class CollisionSphereGroup:
    group: str
    first_sphere: int
    sphere_count: int
    translation_per_closed_fraction_m: Vector3
    sweep_radius_inflation_m: float

    def validate(self, total_spheres: int) -> None:
        if not self.group or self.first_sphere < 0 or self.sphere_count <= 0:
            raise ValueError("VAMP collision sphere groups require a name and positive range")
        if self.first_sphere + self.sphere_count > total_spheres:
            raise ValueError(f"VAMP collision sphere group {self.group} exceeds sphere count")
        if len(self.translation_per_closed_fraction_m) != 3 or not all(
            isfinite(value) for value in self.translation_per_closed_fraction_m
        ):
            raise ValueError("VAMP collision sphere translation must be finite xyz")
        if not isfinite(self.sweep_radius_inflation_m) or self.sweep_radius_inflation_m < 0.0:
            raise ValueError("VAMP sweep radius inflation must be non-negative")


@dataclass(frozen=True)
class CollisionSphereCalibration:
    sphere_index: int
    center_at_default_m: Vector3
    radius_m: float

    def validate(self, total_spheres: int) -> None:
        if not 0 <= self.sphere_index < total_spheres:
            raise ValueError("VAMP calibrated collision sphere index is out of range")
        if len(self.center_at_default_m) != 3 or not all(
            isfinite(value) for value in self.center_at_default_m
        ):
            raise ValueError("VAMP calibrated collision sphere center must be finite xyz")
        if not isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise ValueError("VAMP calibrated collision sphere radius must be positive")


@dataclass(frozen=True)
class SourceSupportContactHull:
    group: str
    vertices_at_maximum_opening_m: tuple[Vector3, ...]
    translation_per_closed_fraction_m: Vector3

    def validate(self, contact_groups: set[str]) -> None:
        if self.group not in contact_groups:
            raise ValueError(
                "VAMP source-support contact hull must name a collision sphere group"
            )
        vertices = np.asarray(self.vertices_at_maximum_opening_m, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[0] < 4 or vertices.shape[1] != 3:
            raise ValueError("VAMP source-support contact hull requires at least four xyz vertices")
        if not np.all(np.isfinite(vertices)) or np.linalg.matrix_rank(
            vertices[1:] - vertices[0]
        ) < 3:
            raise ValueError("VAMP source-support contact hull must be finite and three-dimensional")
        if len(self.translation_per_closed_fraction_m) != 3 or not all(
            isfinite(value) for value in self.translation_per_closed_fraction_m
        ):
            raise ValueError("VAMP source-support contact hull translation must be finite xyz")


@dataclass(frozen=True)
class VampConfig:
    version: str
    robot_module: str
    robot_sphere_count: int
    tool_fraction_midpoint: float
    maximum_gripper_opening_m: float
    collision_sphere_groups: tuple[CollisionSphereGroup, ...]
    collision_sphere_calibrations: tuple[CollisionSphereCalibration, ...]
    source_support_contact_hulls: tuple[SourceSupportContactHull, ...]
    sample_offset_start: int
    sample_offset_stride: int
    range_rad: float
    dynamic_domain: bool
    max_iterations: int
    max_samples: int
    simplify_iterations: int

    def validate(self) -> None:
        if not self.version or not self.robot_module or self.robot_sphere_count <= 0:
            raise ValueError("VAMP version, robot module, and sphere count must be explicit")
        if not 0.0 <= self.tool_fraction_midpoint <= 1.0:
            raise ValueError("VAMP tool fraction midpoint must lie in [0, 1]")
        if not isfinite(self.maximum_gripper_opening_m) or self.maximum_gripper_opening_m <= 0.0:
            raise ValueError("VAMP maximum gripper opening must be positive")
        if not isfinite(self.range_rad) or self.range_rad <= 0.0:
            raise ValueError("VAMP range must be positive")
        if self.sample_offset_start < 0 or self.sample_offset_stride < self.max_samples:
            raise ValueError(
                "VAMP sample offsets must be non-negative and use non-overlapping ranges"
            )
        if (
            self.max_iterations <= 0
            or self.max_samples <= 0
            or self.simplify_iterations <= 0
        ):
            raise ValueError("VAMP search limits must be positive")
        names = [group.group for group in self.collision_sphere_groups]
        if len(names) != len(set(names)):
            raise ValueError("VAMP collision sphere group names must be unique")
        for group in self.collision_sphere_groups:
            group.validate(self.robot_sphere_count)
        calibrated_indices = [
            item.sphere_index for item in self.collision_sphere_calibrations
        ]
        if len(calibrated_indices) != len(set(calibrated_indices)):
            raise ValueError("VAMP calibrated collision sphere indices must be unique")
        for item in self.collision_sphere_calibrations:
            item.validate(self.robot_sphere_count)
        hull_groups = [item.group for item in self.source_support_contact_hulls]
        if len(hull_groups) != len(set(hull_groups)):
            raise ValueError("VAMP source-support contact hull groups must be unique")
        contact_groups = set(names)
        for item in self.source_support_contact_hulls:
            item.validate(contact_groups)


def parse_vamp_config(
    payload: Mapping[str, object],
    end_effector_profile: Mapping[str, object],
) -> VampConfig:
    groups_value = _config_value(
        end_effector_profile, "collision_sphere_groups", "VAMP end-effector profile"
    )
    if not isinstance(groups_value, list):
        raise ValueError("VAMP collision_sphere_groups must be a list")
    groups = []
    for index, value in enumerate(groups_value):
        item = _config_mapping(value, f"VAMP collision_sphere_groups[{index}]")
        groups.append(
            CollisionSphereGroup(
                _config_string(item, "group", "VAMP collision sphere group"),
                _config_integer(item, "first_sphere", "VAMP collision sphere group"),
                _config_integer(item, "sphere_count", "VAMP collision sphere group"),
                _config_vector(
                    item,
                    "translation_per_closed_fraction_m",
                    "VAMP collision sphere group",
                ),
                _config_number(
                    item,
                    "sweep_radius_inflation_m",
                    "VAMP collision sphere group",
                ),
            )
        )
    calibrations_value = _config_value(
        end_effector_profile,
        "collision_sphere_calibrations",
        "VAMP end-effector profile",
    )
    if not isinstance(calibrations_value, list):
        raise ValueError("VAMP collision_sphere_calibrations must be a list")
    calibrations = []
    for index, value in enumerate(calibrations_value):
        item = _config_mapping(
            value, f"VAMP collision_sphere_calibrations[{index}]"
        )
        calibrations.append(
            CollisionSphereCalibration(
                _config_integer(item, "sphere_index", "VAMP sphere calibration"),
                _config_vector(
                    item,
                    "center_at_default_m",
                    "VAMP sphere calibration",
                ),
                _config_number(item, "radius_m", "VAMP sphere calibration"),
            )
        )
    hulls_value = _config_value(
        end_effector_profile,
        "source_support_contact_hulls",
        "VAMP end-effector profile",
    )
    if not isinstance(hulls_value, list):
        raise ValueError("VAMP source_support_contact_hulls must be a list")
    hulls = []
    for index, value in enumerate(hulls_value):
        item = _config_mapping(value, f"VAMP source_support_contact_hulls[{index}]")
        vertices_value = _config_value(
            item,
            "vertices_at_maximum_opening_m",
            "VAMP source-support contact hull",
        )
        if not isinstance(vertices_value, list):
            raise ValueError(
                "VAMP source-support contact hull vertices must be a list"
            )
        hulls.append(
            SourceSupportContactHull(
                _config_string(item, "group", "VAMP source-support contact hull"),
                tuple(
                    _config_vector(
                        {"vertex": vertex},
                        "vertex",
                        "VAMP source-support contact hull vertex",
                    )
                    for vertex in vertices_value
                ),
                _config_vector(
                    item,
                    "translation_per_closed_fraction_m",
                    "VAMP source-support contact hull",
                ),
            )
        )
    result = VampConfig(
        _config_string(payload, "version", "VAMP"),
        _config_string(end_effector_profile, "vamp_robot_module", "VAMP profile"),
        _config_integer(
            end_effector_profile, "vamp_robot_sphere_count", "VAMP profile"
        ),
        _config_number(
            end_effector_profile, "vamp_tool_fraction_midpoint", "VAMP profile"
        ),
        _config_number(
            end_effector_profile, "maximum_gripper_opening_m", "VAMP profile"
        ),
        tuple(groups),
        tuple(calibrations),
        tuple(hulls),
        _config_integer(payload, "sample_offset_start", "VAMP"),
        _config_integer(payload, "sample_offset_stride", "VAMP"),
        _config_number(payload, "range_rad", "VAMP"),
        _config_boolean(payload, "dynamic_domain", "VAMP"),
        _config_integer(payload, "max_iterations", "VAMP"),
        _config_integer(payload, "max_samples", "VAMP"),
        _config_integer(payload, "simplify_iterations", "VAMP"),
    )
    result.validate()
    return result


class VampBackend:
    def __init__(self, robot_model: RobotModel, config: VampConfig):
        config.validate()
        try:
            import vamp
        except (ImportError, OSError) as exc:
            raise BackendUnavailableError(f"VAMP import failed: {exc}") from exc
        try:
            installed = metadata.version("vamp-planner")
        except metadata.PackageNotFoundError as exc:
            raise BackendUnavailableError("vamp-planner package metadata is unavailable") from exc
        if installed != config.version:
            raise BackendUnavailableError(
                f"VAMP version {installed} does not match required {config.version}"
            )
        if not hasattr(vamp, "ConvexMesh"):
            raise BackendUnavailableError(
                "patched VAMP lacks the required native ConvexMesh type"
            )
        try:
            capability_environment = vamp.Environment()
        except (TypeError, RuntimeError) as exc:
            raise BackendUnavailableError(
                f"patched VAMP Environment is unavailable: {exc}"
            ) from exc
        if not hasattr(capability_environment, "add_convex_mesh") or not hasattr(
            capability_environment, "convex_meshes"
        ):
            raise BackendUnavailableError(
                "patched VAMP Environment lacks native ConvexMesh registration"
            )
        if tuple(getattr(vamp, "robots", ())) != (config.robot_module,):
            raise BackendUnavailableError(
                "patched VAMP must register only the configured Gen3/SO-101 module"
            )
        robot = getattr(vamp, config.robot_module, None)
        if robot is None:
            raise BackendUnavailableError(
                f"VAMP robot module {config.robot_module} is unavailable"
            )
        if tuple(robot.joint_names()) != robot_model.joint_names:
            raise BackendUnavailableError("VAMP joint order does not match RobotModel")
        if int(robot.n_spheres()) != config.robot_sphere_count:
            raise BackendUnavailableError("VAMP collision sphere count does not match RobotModel")
        empty_environment = vamp.Environment()
        if not robot.validate(
            robot_model.default_joint_positions, empty_environment, True
        ):
            raise BackendUnavailableError(
                "VAMP rejects the configured default RobotModel state"
            )
        for index, limits in enumerate(robot_model.joint_limits):
            if not limits.continuous:
                continue
            shifted = list(robot_model.default_joint_positions)
            shifted[index] += 2.0 * np.pi
            if not robot.validate(shifted, empty_environment, True):
                raise BackendUnavailableError(
                    "VAMP does not expose RobotModel continuous joints as periodic"
                )
        collision_spheres = tuple(robot.fk(robot_model.default_joint_positions))
        for calibration in config.collision_sphere_calibrations:
            sphere = collision_spheres[calibration.sphere_index]
            actual_center = (float(sphere.x), float(sphere.y), float(sphere.z))
            if max(
                abs(actual - expected)
                for actual, expected in zip(
                    actual_center, calibration.center_at_default_m
                )
            ) > 1e-6 or abs(float(sphere.r) - calibration.radius_m) > 1e-6:
                raise BackendUnavailableError(
                    "patched VAMP collision sphere calibration does not match Planning YAML"
                )
        for group in config.collision_sphere_groups:
            first = group.first_sphere
            last = first + group.sphere_count
            if min(item.r for item in collision_spheres[first:last]) <= group.sweep_radius_inflation_m:
                raise BackendUnavailableError(
                    f"patched VAMP module lacks the full-stroke envelope for {group.group}"
                )
        self._vamp = vamp
        self._robot = robot
        self._model = robot_model
        self._config = config
        self._pointcloud_trees: dict[int, cKDTree] = {}
        self._convex_mesh_data: dict[int, tuple[np.ndarray, ...]] = {}
        self._semantic_polyhedra: dict[tuple[int, bytes], _PreparedPolyhedron] = {}
        self._environment_cache: dict[tuple[object, ...], object] = {}
        self._semantic_geometry_cache: dict[
            tuple[tuple[int, ...], bytes], tuple[SceneGeometry, ...]
        ] = {}
        self._target_contact_cache: dict[
            tuple[tuple[float, ...], tuple[int, ...], float, bytes],
            frozenset[str],
        ] = {}
        self._support_contact_cache: dict[
            tuple[tuple[float, ...], tuple[int, ...], float, bytes, bool],
            frozenset[str],
        ] = {}
        self._attachment_contact_cache: dict[
            tuple[
                tuple[float, ...],
                bytes,
                tuple[int, ...],
                tuple[int, ...],
                bytes,
            ],
            frozenset[str],
        ] = {}
        groups_by_sphere: list[CollisionSphereGroup | None] = [
            None
        ] * config.robot_sphere_count
        for group in config.collision_sphere_groups:
            for index in range(
                group.first_sphere, group.first_sphere + group.sphere_count
            ):
                groups_by_sphere[index] = group
        self._collision_group_by_sphere = tuple(groups_by_sphere)
        self.version = installed

    def begin_request(self) -> None:
        """Drop request-owned geometry caches while retaining native robot setup."""
        self._pointcloud_trees.clear()
        self._convex_mesh_data.clear()
        self._semantic_polyhedra.clear()
        self._environment_cache.clear()
        self._semantic_geometry_cache.clear()
        self._target_contact_cache.clear()
        self._support_contact_cache.clear()
        self._attachment_contact_cache.clear()

    def environment(
        self,
        scene: SceneSnapshot,
        *,
        T_world_base: RigidTransform,
        excluded_object_ids: Iterable[str] = (),
        excluded_groups: Iterable[str] = (),
        T_tcp_object: RigidTransform | None = None,
        attached_geometry: tuple[AttachedGeometry, ...] = (),
    ):
        excluded_objects = set(excluded_object_ids)
        excluded_collision_groups = set(excluded_groups)
        cache_key = (
            tuple(id(item) for item in scene.geometries),
            np.asarray(T_world_base.matrix, dtype=np.float64).tobytes(),
            tuple(sorted(excluded_objects)),
            tuple(sorted(excluded_collision_groups)),
            (
                None
                if T_tcp_object is None
                else np.asarray(T_tcp_object.matrix, dtype=np.float64).tobytes()
            ),
            tuple(id(item) for item in attached_geometry),
        )
        cached = self._environment_cache.get(cache_key)
        if cached is not None:
            return cached
        environment = self._vamp.Environment()
        for item in scene.geometries:
            if item.object_id in excluded_objects or item.collision_group in excluded_collision_groups:
                continue
            self._add_scene_geometry(
                environment,
                SceneGeometry(
                    item.geometry_id,
                    item.object_id,
                    item.collision_group,
                    compose(inverse(T_world_base), item.T_world_geometry),
                    item.geometry,
                ),
            )
        if attached_geometry:
            if T_tcp_object is None:
                raise BackendOperationError("attached geometry requires T_tcp_object")
            attachment = self._vamp.Attachment(
                np.asfortranarray(T_tcp_object.matrix, dtype=np.float32)
            )
            for item in attached_geometry:
                for center, radius in _attached_sphere_specs(item):
                    attachment.add_sphere(self._vamp.Sphere(center.tolist(), radius))
            environment.attach(attachment)
        self._environment_cache[cache_key] = environment
        return environment

    def plan(
        self,
        start: tuple[float, ...],
        goal: tuple[float, ...],
        environment,
        *,
        sample_offset: int,
    ) -> VampPlan:
        return self.plan_to_any(
            start,
            (goal,),
            environment,
            sample_offset=sample_offset,
        )

    def plan_to_any(
        self,
        start: tuple[float, ...],
        goals: tuple[tuple[float, ...], ...],
        environment,
        *,
        sample_offset: int,
    ) -> VampPlan:
        self._validate_configuration_shape(start)
        if not goals:
            raise BackendOperationError("VAMP multi-goal planning requires at least one goal")
        for goal in goals:
            self._validate_configuration_shape(goal)
        aligned_goals = tuple(
            self._model.unwrap_positions(goal, start) for goal in goals
        )
        started = monotonic()
        direct_goals = []
        for index, aligned_goal in enumerate(aligned_goals):
            try:
                direct_valid = self._robot.validate_motion(
                    start, aligned_goal, environment, True
                )
            except (TypeError, ValueError, RuntimeError) as exc:
                raise BackendOperationError(
                    f"VAMP direct motion validation failed: {exc}"
                ) from exc
            if direct_valid:
                direct_goals.append(
                    (
                        float(
                            np.linalg.norm(
                                np.asarray(aligned_goal, dtype=np.float64)
                                - np.asarray(start, dtype=np.float64)
                            )
                        ),
                        index,
                        aligned_goal,
                    )
                )
        if direct_goals:
            _, selected_index, selected_goal = min(direct_goals)
            return VampPlan(
                path=(start, selected_goal),
                elapsed_s=monotonic() - started,
                iterations=0,
                selected_goal_index=selected_index,
            )
        settings = self._vamp.RRTCSettings()
        settings.max_iterations = self._config.max_iterations
        settings.max_samples = self._config.max_samples
        settings.range = self._config.range_rad
        settings.dynamic_domain = self._config.dynamic_domain
        rng = self._robot.halton()
        rng.skip(sample_offset)
        try:
            result = self._robot.rrtc(
                start,
                aligned_goals,
                environment,
                settings,
                rng,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise BackendOperationError(f"VAMP RRTC failed: {exc}") from exc
        if not result.solved:
            raise BackendOperationError("VAMP RRTC exhausted its configured search budget")
        planning_iterations = int(result.iterations)
        simplify_settings = self._vamp.SimplifySettings()
        simplify_settings.max_iterations = self._config.simplify_iterations
        simplify_settings.operations = [self._vamp.SimplifyRoutine.SHORTCUT]
        try:
            result = self._robot.simplify(
                result.path, environment, simplify_settings, rng
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise BackendOperationError(f"VAMP path simplification failed: {exc}") from exc
        if not result.solved:
            raise BackendOperationError("VAMP path simplification failed")
        elapsed = monotonic() - started
        path_array = np.asarray(result.path.numpy(), dtype=np.float64)
        if path_array.ndim != 2 or path_array.shape[1] != len(self._model.joint_names):
            raise BackendOperationError("VAMP returned a malformed path")
        path_rows = tuple(tuple(float(value) for value in row) for row in path_array)
        path = []
        reference = start
        for row in path_rows:
            unwrapped = self._model.unwrap_positions(row, reference)
            path.append(unwrapped)
            reference = unwrapped
        path = tuple(path)
        if not path:
            raise BackendOperationError("VAMP returned an empty path")
        endpoint = path[-1]
        goal_errors = tuple(
            max(
                abs(value - reference)
                for value, reference in zip(
                    self._model.unwrap_positions(goal, endpoint), endpoint
                )
            )
            for goal in aligned_goals
        )
        selected_goal_index = min(
            range(len(goal_errors)), key=lambda index: (goal_errors[index], index)
        )
        if goal_errors[selected_goal_index] > 1e-4:
            raise BackendOperationError(
                "VAMP multi-goal path endpoint missed every requested goal"
            )
        return VampPlan(
            path=path,
            elapsed_s=elapsed,
            iterations=planning_iterations,
            selected_goal_index=selected_goal_index,
        )

    def configuration_valid(self, configuration: tuple[float, ...], environment) -> bool:
        self._validate_configuration_shape(configuration)
        return bool(self._robot.validate(configuration, environment, True))

    def motion_valid(
        self,
        start: tuple[float, ...],
        goal: tuple[float, ...],
        environment,
    ) -> bool:
        self._validate_configuration_shape(start)
        self._validate_configuration_shape(goal)
        return bool(self._robot.validate_motion(start, goal, environment, True))

    def path_valid(
        self,
        configurations: tuple[tuple[float, ...], ...],
        environment,
    ) -> bool:
        if not configurations:
            raise BackendOperationError("VAMP path validation requires configurations")
        path = self._robot.Path()
        for configuration in configurations:
            self._validate_configuration_shape(configuration)
            path.append(configuration)
        return bool(path.validate(environment))

    def robot_spheres(self, configuration: tuple[float, ...]):
        self._validate_configuration_shape(configuration)
        return tuple(self._robot.fk(configuration))

    def tcp_pose(self, configuration: tuple[float, ...]) -> RigidTransform:
        raw = self._tcp_matrix(configuration)
        result = RigidTransform(
            self._model.base_frame,
            self._model.tcp_frame,
            tuple(tuple(float(value) for value in row) for row in raw),  # type: ignore[arg-type]
        )
        result.validate(label="VAMP TCP pose")
        return result

    def _tcp_matrix(self, configuration: tuple[float, ...]) -> np.ndarray:
        self._validate_configuration_shape(configuration)
        raw = np.asarray(self._robot.eefk(configuration), dtype=np.float64)
        if raw.shape == (4, 4):
            return raw
        elif raw.shape == (7,):
            x, y, z, qx, qy, qz, qw = raw
            rotation = Rotation.from_quat((qx, qy, qz, qw)).as_matrix()
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, :3] = rotation
            matrix[:3, 3] = (x, y, z)
            return matrix
        else:
            raise BackendOperationError(f"VAMP returned unexpected TCP pose shape {raw.shape}")

    def _geometries_in_base(
        self,
        geometries: tuple[SceneGeometry, ...],
        T_world_base: RigidTransform,
    ) -> tuple[SceneGeometry, ...]:
        key = (
            tuple(id(item) for item in geometries),
            np.asarray(T_world_base.matrix, dtype=np.float64).tobytes(),
        )
        cached = self._semantic_geometry_cache.get(key)
        if cached is not None:
            return cached
        T_base_world = inverse(T_world_base)
        transformed = tuple(
            SceneGeometry(
                item.geometry_id,
                item.object_id,
                item.collision_group,
                compose(T_base_world, item.T_world_geometry),
                item.geometry,
            )
            for item in geometries
        )
        self._semantic_geometry_cache[key] = transformed
        return transformed

    def semantic_target_contacts(
        self,
        configuration: tuple[float, ...],
        target_geometries: tuple[SceneGeometry, ...],
        opening_m: float,
        T_world_base: RigidTransform,
    ) -> frozenset[str]:
        cache_key = (
            configuration,
            tuple(id(item) for item in target_geometries),
            opening_m,
            np.asarray(T_world_base.matrix, dtype=np.float64).tobytes(),
        )
        cached = self._target_contact_cache.get(cache_key)
        if cached is not None:
            return cached
        contacts = set()
        geometries_in_base = self._geometries_in_base(
            target_geometries,
            T_world_base,
        )
        adjusted_spheres = self._semantic_robot_spheres(configuration, opening_m)
        sphere_centers, sphere_radii = _sphere_arrays(adjusted_spheres)
        for geometry in geometries_in_base:
            if isinstance(geometry.geometry, PointCloud):
                for group_name in _pointcloud_contact_groups(
                    adjusted_spheres,
                    geometry,
                    self._pointcloud_tree(geometry),
                    sphere_centers,
                ):
                    contacts.add(group_name)
                continue
            if isinstance(geometry.geometry, ConvexHull):
                contacts.update(
                    _convex_contact_groups(
                        adjusted_spheres,
                        geometry,
                        self._prepared_convex_mesh(geometry.geometry),
                        sphere_centers,
                        sphere_radii,
                    )
                )
                continue
            for adjusted, group_name in adjusted_spheres:
                if _sphere_intersects_geometry(adjusted, geometry):
                    contacts.add(group_name)
        result = frozenset(contacts)
        self._target_contact_cache[cache_key] = result
        return result

    def semantic_source_support_contacts(
        self,
        configuration: tuple[float, ...],
        support_geometries: tuple[SceneGeometry, ...],
        opening_m: float,
        T_world_base: RigidTransform,
        *,
        use_physical_pad_hulls: bool,
    ) -> frozenset[str]:
        cache_key = (
            configuration,
            tuple(id(item) for item in support_geometries),
            opening_m,
            np.asarray(T_world_base.matrix, dtype=np.float64).tobytes(),
            use_physical_pad_hulls,
        )
        cached = self._support_contact_cache.get(cache_key)
        if cached is not None:
            return cached
        sphere_contacts = self.semantic_target_contacts(
            configuration,
            support_geometries,
            opening_m,
            T_world_base,
        )
        contacts = set(sphere_contacts)
        if not use_physical_pad_hulls or not self._config.source_support_contact_hulls:
            result = frozenset(contacts)
            self._support_contact_cache[cache_key] = result
            return result
        if not 0.0 <= opening_m <= self._config.maximum_gripper_opening_m:
            raise BackendOperationError("gripper opening is outside RobotModel limits")

        geometries_in_base = self._geometries_in_base(
            support_geometries,
            T_world_base,
        )
        tcp_matrix = self._tcp_matrix(configuration)
        closed_fraction = 1.0 - opening_m / self._config.maximum_gripper_opening_m
        for hull in self._config.source_support_contact_hulls:
            contacts.discard(hull.group)
            local_vertices = np.asarray(
                hull.vertices_at_maximum_opening_m, dtype=np.float64
            ) + closed_fraction * np.asarray(
                hull.translation_per_closed_fraction_m, dtype=np.float64
            )
            pad_vertices = (
                local_vertices @ tcp_matrix[:3, :3].T + tcp_matrix[:3, 3]
            )
            pad_faces = SpatialConvexHull(pad_vertices).simplices
            unsupported_geometry = False
            for geometry in geometries_in_base:
                support_polyhedron = _convex_polyhedron_spec(geometry)
                if support_polyhedron is None:
                    unsupported_geometry = True
                    continue
                support_vertices, support_faces = support_polyhedron
                if _convex_polyhedra_intersect(
                    pad_vertices,
                    pad_faces,
                    support_vertices,
                    support_faces,
                ):
                    contacts.add(hull.group)
                    break
            else:
                if unsupported_geometry and hull.group in sphere_contacts:
                    contacts.add(hull.group)
        result = frozenset(contacts)
        self._support_contact_cache[cache_key] = result
        return result

    def semantic_robot_sphere_centers(
        self,
        configuration: tuple[float, ...],
        opening_m: float,
    ) -> tuple[Vector3, ...]:
        return tuple(
            (float(sphere.x), float(sphere.y), float(sphere.z))
            for sphere, _ in self._semantic_robot_spheres(configuration, opening_m)
        )

    def semantic_attachment_sphere_centers(
        self,
        configuration: tuple[float, ...],
        T_tcp_object: RigidTransform,
        attached_geometry: tuple[AttachedGeometry, ...],
    ) -> tuple[Vector3, ...]:
        return tuple(
            (float(sphere.x), float(sphere.y), float(sphere.z))
            for sphere in self._attachment_spheres(
                configuration, T_tcp_object, attached_geometry
            )
        )

    def semantic_attachment_contacts(
        self,
        configuration: tuple[float, ...],
        T_tcp_object: RigidTransform,
        attached_geometry: tuple[AttachedGeometry, ...],
        scene_geometries: tuple[SceneGeometry, ...],
        T_world_base: RigidTransform,
    ) -> frozenset[str]:
        cache_key = (
            configuration,
            np.asarray(T_tcp_object.matrix, dtype=np.float64).tobytes(),
            tuple(id(item) for item in attached_geometry),
            tuple(id(item) for item in scene_geometries),
            np.asarray(T_world_base.matrix, dtype=np.float64).tobytes(),
        )
        cached = self._attachment_contact_cache.get(cache_key)
        if cached is not None:
            return cached
        tcp_matrix = self._tcp_matrix(configuration)
        object_matrix = tcp_matrix @ np.asarray(T_tcp_object.matrix, dtype=np.float64)
        geometries_in_base = self._geometries_in_base(
            scene_geometries,
            T_world_base,
        )
        environment_polyhedra = []
        for geometry in geometries_in_base:
            transform_key = np.asarray(
                geometry.T_world_geometry.matrix, dtype=np.float64
            ).tobytes()
            polyhedron_key = (id(geometry.geometry), transform_key)
            environment_polyhedron = self._semantic_polyhedra.get(polyhedron_key)
            if environment_polyhedron is None:
                environment_polyhedron = _prepared_convex_polyhedron(geometry)
                if environment_polyhedron is not None:
                    self._semantic_polyhedra[polyhedron_key] = environment_polyhedron
            environment_polyhedra.append((geometry, environment_polyhedron))
        contacts = set()
        for attached in attached_geometry:
            attached_matrix = object_matrix @ np.asarray(
                attached.T_object_geometry.matrix, dtype=np.float64
            )
            attached_in_base = SceneGeometry(
                attached.geometry_id,
                "attached_object",
                "attached_object",
                RigidTransform(
                    self._model.base_frame,
                    attached.geometry_id,
                    tuple(tuple(float(value) for value in row) for row in attached_matrix),  # type: ignore[arg-type]
                ),
                attached.geometry,
            )
            attached_polyhedron = _prepared_convex_polyhedron(attached_in_base)
            if attached_polyhedron is not None:
                for geometry, environment_polyhedron in environment_polyhedra:
                    if environment_polyhedron is None:
                        continue
                    if np.any(
                        attached_polyhedron.maximum
                        < environment_polyhedron.minimum - 1e-12
                    ) or np.any(
                        environment_polyhedron.maximum
                        < attached_polyhedron.minimum - 1e-12
                    ):
                        continue
                    if _prepared_polyhedra_intersect(
                        attached_polyhedron,
                        environment_polyhedron,
                    ):
                        contacts.add(geometry.collision_group)
                continue

            attached_spheres = tuple(
                self._vamp.Sphere(
                    (attached_matrix[:3, :3] @ center + attached_matrix[:3, 3]).tolist(),
                    radius,
                )
                for center, radius in _attached_sphere_specs(attached)
            )
            named_spheres = [(sphere, "attached_object") for sphere in attached_spheres]
            for geometry in geometries_in_base:
                if isinstance(geometry.geometry, PointCloud):
                    hit = bool(
                        _pointcloud_contact_groups(
                            named_spheres, geometry, self._pointcloud_tree(geometry)
                        )
                    )
                elif isinstance(geometry.geometry, ConvexHull):
                    hit = bool(
                        _convex_contact_groups(
                            named_spheres,
                            geometry,
                            self._prepared_convex_mesh(geometry.geometry),
                        )
                    )
                else:
                    hit = any(
                        _sphere_intersects_geometry(sphere, geometry)
                        for sphere in attached_spheres
                    )
                if hit:
                    contacts.add(geometry.collision_group)
        result = frozenset(contacts)
        self._attachment_contact_cache[cache_key] = result
        return result

    def _semantic_robot_spheres(
        self,
        configuration: tuple[float, ...],
        opening_m: float,
    ) -> tuple[tuple[object, str], ...]:
        if not 0.0 <= opening_m <= self._config.maximum_gripper_opening_m:
            raise BackendOperationError("gripper opening is outside RobotModel limits")
        closed_fraction = 1.0 - opening_m / self._config.maximum_gripper_opening_m
        tcp_rotation = self._tcp_matrix(configuration)[:3, :3]
        adjusted_spheres = []
        for index, robot_sphere in enumerate(self.robot_spheres(configuration)):
            adjusted = robot_sphere
            group = self._collision_group_by_sphere[index]
            if group is not None:
                local_shift = np.asarray(
                    group.translation_per_closed_fraction_m, dtype=np.float64
                ) * (closed_fraction - self._config.tool_fraction_midpoint)
                base_shift = tcp_rotation @ local_shift
                adjusted = self._vamp.Sphere(
                    [
                        robot_sphere.x + base_shift[0],
                        robot_sphere.y + base_shift[1],
                        robot_sphere.z + base_shift[2],
                    ],
                    robot_sphere.r - group.sweep_radius_inflation_m,
                )
            adjusted_spheres.append(
                (adjusted, group.group if group is not None else "robot_non_pad")
            )
        return tuple(adjusted_spheres)

    def _attachment_spheres(
        self,
        configuration: tuple[float, ...],
        T_tcp_object: RigidTransform,
        attached_geometry: tuple[AttachedGeometry, ...],
    ) -> tuple[object, ...]:
        tcp_matrix_fn = getattr(self, "_tcp_matrix", None)
        tcp_matrix = (
            tcp_matrix_fn(configuration)
            if callable(tcp_matrix_fn)
            else np.asarray(self.tcp_pose(configuration).matrix, dtype=np.float64)
        )
        object_matrix = tcp_matrix @ np.asarray(T_tcp_object.matrix, dtype=np.float64)
        attached_spheres = []
        for item in attached_geometry:
            for center, radius in _attached_sphere_specs(item):
                center_base = object_matrix[:3, :3] @ center + object_matrix[:3, 3]
                attached_spheres.append(self._vamp.Sphere(center_base.tolist(), radius))
        return tuple(attached_spheres)

    def _pointcloud_tree(self, item: SceneGeometry) -> cKDTree:
        key = id(item.geometry)
        tree = self._pointcloud_trees.get(key)
        if tree is None:
            geometry = item.geometry
            if not isinstance(geometry, PointCloud):
                raise TypeError("point-cloud index requires PointCloud geometry")
            tree = cKDTree(np.asarray(geometry.points_m, dtype=np.float64))
            self._pointcloud_trees[key] = tree
        return tree

    def _prepared_convex_mesh(
        self, geometry: ConvexHull
    ) -> tuple[np.ndarray, ...]:
        key = id(geometry)
        prepared = self._convex_mesh_data.get(key)
        if prepared is None:
            vertices = np.asarray(geometry.vertices_m, dtype=np.float64)
            faces = _outward_convex_faces(vertices, geometry.triangles)
            a = vertices[faces[:, 0]]
            b = vertices[faces[:, 1]]
            c = vertices[faces[:, 2]]
            normals = np.cross(b - a, c - a)
            normals /= np.linalg.norm(normals, axis=1)[:, None]
            prepared = (
                a,
                b,
                c,
                normals,
                np.min(vertices, axis=0),
                np.max(vertices, axis=0),
            )
            self._convex_mesh_data[key] = prepared
        return prepared

    def _add_scene_geometry(self, environment, item: SceneGeometry) -> None:
        position, euler = _position_and_euler(item.T_world_geometry)
        geometry = item.geometry
        if isinstance(geometry, Sphere):
            obstacle = self._vamp.Sphere(position, geometry.radius_m)
            obstacle.name = item.geometry_id
            environment.add_sphere(obstacle)
        elif isinstance(geometry, Capsule):
            obstacle = self._vamp.Cylinder(position, euler, geometry.radius_m, geometry.length_m)
            obstacle.name = item.geometry_id
            environment.add_capsule(obstacle)
        elif isinstance(geometry, Box):
            obstacle = self._vamp.Cuboid(position, euler, [value / 2.0 for value in geometry.size_m])
            obstacle.name = item.geometry_id
            environment.add_cuboid(obstacle)
        elif isinstance(geometry, PointCloud):
            points = _transform_points(item.T_world_geometry, geometry.points_m)
            radii = [sphere.r for sphere in self.robot_spheres(self._model.default_joint_positions)]
            environment.add_pointcloud(points, min(radii), max(radii), geometry.point_radius_m)
        elif isinstance(geometry, ConvexHull):
            vertices, faces, bounds_center, bounds_euler, bounds_half_extents = (
                _convex_mesh_spec(item)
            )
            obstacle = self._vamp.ConvexMesh(
                vertices,
                faces,
                bounds_center,
                bounds_euler,
                bounds_half_extents,
                0.0,
            )
            obstacle.name = item.geometry_id
            environment.add_convex_mesh(obstacle)
        elif isinstance(geometry, StaticTriangleMesh):
            raise BackendUnavailableError(
                "patched VAMP supports closed convex hulls, not arbitrary static triangle meshes; "
                "provide an explicit convex decomposition in SceneSnapshot"
            )
        else:
            raise BackendOperationError(f"unsupported VAMP scene geometry: {type(geometry).__name__}")

    def _validate_configuration_shape(self, configuration: tuple[float, ...]) -> None:
        if len(configuration) != len(self._model.joint_names):
            raise BackendOperationError("configuration size does not match RobotModel")


def _position_and_euler(transform: RigidTransform) -> tuple[list[float], list[float]]:
    matrix = np.asarray(transform.matrix, dtype=np.float64)
    return matrix[:3, 3].tolist(), Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz").tolist()


def _transform_points(transform: RigidTransform, points: tuple[tuple[float, float, float], ...]):
    matrix = np.asarray(transform.matrix, dtype=np.float64)
    values = np.asarray(points, dtype=np.float64)
    return (values @ matrix[:3, :3].T + matrix[:3, 3]).tolist()


def _convex_mesh_spec(
    item: SceneGeometry,
) -> tuple[list[list[float]], list[list[int]], list[float], list[float], list[float]]:
    geometry = item.geometry
    if not isinstance(geometry, ConvexHull):
        raise TypeError("native convex mesh conversion requires ConvexHull geometry")
    local_vertices = np.asarray(geometry.vertices_m, dtype=np.float64)
    faces = _outward_convex_faces(local_vertices, geometry.triangles)
    local_minimum = np.min(local_vertices, axis=0)
    local_maximum = np.max(local_vertices, axis=0)
    half_extents = 0.5 * (local_maximum - local_minimum)
    if np.any(half_extents <= 1e-9):
        raise BackendOperationError("convex hull must have positive extent on all axes")
    matrix = np.asarray(item.T_world_geometry.matrix, dtype=np.float64)
    vertices = local_vertices @ matrix[:3, :3].T + matrix[:3, 3]
    local_center = 0.5 * (local_minimum + local_maximum)
    bounds_center = matrix[:3, :3] @ local_center + matrix[:3, 3]
    bounds_euler = Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz")
    return (
        vertices.tolist(),
        faces.tolist(),
        bounds_center.tolist(),
        bounds_euler.tolist(),
        half_extents.tolist(),
    )


def _convex_polyhedron_spec(
    item: SceneGeometry,
) -> tuple[np.ndarray, np.ndarray] | None:
    transform = np.asarray(item.T_world_geometry.matrix, dtype=np.float64)
    geometry = item.geometry
    if isinstance(geometry, Box):
        half = np.asarray(geometry.size_m, dtype=np.float64) / 2.0
        local_vertices = np.asarray(
            [
                (x, y, z)
                for x in (-half[0], half[0])
                for y in (-half[1], half[1])
                for z in (-half[2], half[2])
            ],
            dtype=np.float64,
        )
        faces = SpatialConvexHull(local_vertices).simplices
    elif isinstance(geometry, ConvexHull):
        local_vertices = np.asarray(geometry.vertices_m, dtype=np.float64)
        faces = _outward_convex_faces(local_vertices, geometry.triangles)
    else:
        return None
    vertices = local_vertices @ transform[:3, :3].T + transform[:3, 3]
    return vertices, np.asarray(faces, dtype=np.int64)


def _convex_polyhedra_intersect(
    left_vertices: np.ndarray,
    left_faces: np.ndarray,
    right_vertices: np.ndarray,
    right_faces: np.ndarray,
) -> bool:
    return _prepared_polyhedra_intersect(
        _prepare_polyhedron(left_vertices, left_faces),
        _prepare_polyhedron(right_vertices, right_faces),
    )


def _prepared_convex_polyhedron(
    item: SceneGeometry,
) -> _PreparedPolyhedron | None:
    if isinstance(item.geometry, Box):
        transform = np.asarray(item.T_world_geometry.matrix, dtype=np.float64)
        half = np.asarray(item.geometry.size_m, dtype=np.float64) / 2.0
        local_vertices = np.asarray(
            [
                (x, y, z)
                for x in (-half[0], half[0])
                for y in (-half[1], half[1])
                for z in (-half[2], half[2])
            ],
            dtype=np.float64,
        )
        vertices = (
            local_vertices @ transform[:3, :3].T + transform[:3, 3]
        )
        axes = transform[:3, :3].T
        return _PreparedPolyhedron(
            vertices,
            axes,
            axes,
            np.min(vertices, axis=0),
            np.max(vertices, axis=0),
        )
    spec = _convex_polyhedron_spec(item)
    if spec is None:
        return None
    vertices, faces = spec
    return _prepare_polyhedron(vertices, faces)


def _prepare_polyhedron(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    edge_directions: np.ndarray | None = None,
) -> _PreparedPolyhedron:
    triangles = vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    normal_lengths = np.linalg.norm(normals, axis=1)
    normals = normals[normal_lengths > 1e-12] / normal_lengths[
        normal_lengths > 1e-12, None
    ]
    if edge_directions is None:
        pairs = np.concatenate(
            (faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]), axis=0
        )
        pairs = np.unique(np.sort(pairs, axis=1), axis=0)
        edge_directions = vertices[pairs[:, 1]] - vertices[pairs[:, 0]]
    edge_lengths = np.linalg.norm(edge_directions, axis=1)
    edge_directions = edge_directions[edge_lengths > 1e-12] / edge_lengths[
        edge_lengths > 1e-12, None
    ]
    return _PreparedPolyhedron(
        vertices,
        normals,
        edge_directions,
        np.min(vertices, axis=0),
        np.max(vertices, axis=0),
    )


def _prepared_polyhedra_intersect(
    left: _PreparedPolyhedron,
    right: _PreparedPolyhedron,
) -> bool:
    if np.any(left.maximum < right.minimum - 1e-12) or np.any(
        right.maximum < left.minimum - 1e-12
    ):
        return False
    edge_axes = np.cross(
        left.edge_directions[:, None, :], right.edge_directions[None, :, :]
    ).reshape(
        -1, 3
    )
    edge_lengths = np.linalg.norm(edge_axes, axis=1)
    edge_axes = edge_axes[edge_lengths > 1e-12] / edge_lengths[
        edge_lengths > 1e-12, None
    ]
    axes = np.concatenate(
        (
            left.face_normals,
            right.face_normals,
            edge_axes,
        ),
        axis=0,
    )
    left_projection = left.vertices @ axes.T
    right_projection = right.vertices @ axes.T
    return not bool(
        np.any(
            (np.max(left_projection, axis=0) < np.min(right_projection, axis=0) - 1e-12)
            | (
                np.max(right_projection, axis=0)
                < np.min(left_projection, axis=0) - 1e-12
            )
        )
    )


def _outward_convex_faces(
    vertices: np.ndarray,
    triangles: tuple[tuple[int, int, int], ...],
) -> np.ndarray:
    centroid = np.mean(vertices, axis=0)
    result = []
    for raw_face in triangles:
        face = list(raw_face)
        a, b, c = vertices[face]
        normal = np.cross(b - a, c - a)
        if float(np.linalg.norm(normal)) <= 1e-12:
            raise BackendOperationError("convex hull contains a degenerate triangle")
        if float(np.dot(normal, a - centroid)) < 0.0:
            face[1], face[2] = face[2], face[1]
        result.append(face)
    return np.asarray(result, dtype=np.int64)


def _sphere_intersects_geometry(robot_sphere, item: SceneGeometry) -> bool:
    center_world = np.array([robot_sphere.x, robot_sphere.y, robot_sphere.z], dtype=np.float64)
    radius = float(robot_sphere.r)
    transform = np.asarray(item.T_world_geometry.matrix, dtype=np.float64)
    center_local = transform[:3, :3].T @ (center_world - transform[:3, 3])
    geometry = item.geometry
    if isinstance(geometry, Sphere):
        return float(np.linalg.norm(center_local)) <= radius + geometry.radius_m
    if isinstance(geometry, Box):
        half = np.asarray(geometry.size_m, dtype=np.float64) / 2.0
        closest = np.minimum(np.maximum(center_local, -half), half)
        return float(np.linalg.norm(center_local - closest)) <= radius
    if isinstance(geometry, Capsule):
        endpoint = geometry.length_m / 2.0
        closest = np.array([0.0, 0.0, min(max(center_local[2], -endpoint), endpoint)])
        return float(np.linalg.norm(center_local - closest)) <= radius + geometry.radius_m
    if isinstance(geometry, PointCloud):
        distance, _ = cKDTree(np.asarray(geometry.points_m, dtype=np.float64)).query(
            center_local, k=1, workers=1
        )
        return float(distance) <= radius + geometry.point_radius_m
    if isinstance(geometry, ConvexHull):
        vertices = np.asarray(geometry.vertices_m, dtype=np.float64)
        faces = _outward_convex_faces(vertices, geometry.triangles)
        surface_distance_squared = float("inf")
        inside = True
        for face in faces:
            a, b, c = vertices[face]
            normal = np.cross(b - a, c - a)
            normal /= np.linalg.norm(normal)
            if float(np.dot(center_local - a, normal)) > 1e-12:
                inside = False
            surface_distance_squared = min(
                surface_distance_squared,
                _point_triangle_distance_squared(center_local, a, b, c, normal),
            )
        return inside or surface_distance_squared <= radius * radius + 1e-12
    if isinstance(geometry, StaticTriangleMesh):
        raise BackendUnavailableError(
            "patched VAMP supports closed convex hulls, not arbitrary static triangle meshes; "
            "provide an explicit convex decomposition in SceneSnapshot"
        )
    raise BackendOperationError(f"unsupported semantic collision geometry: {type(geometry).__name__}")


def _point_triangle_distance_squared(
    point: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    unit_normal: np.ndarray,
) -> float:
    signed_plane_distance = float(np.dot(point - a, unit_normal))
    projection = point - signed_plane_distance * unit_normal
    ab = b - a
    ac = c - a
    relative = projection - a
    d00 = float(np.dot(ab, ab))
    d01 = float(np.dot(ab, ac))
    d11 = float(np.dot(ac, ac))
    d20 = float(np.dot(relative, ab))
    d21 = float(np.dot(relative, ac))
    denominator = d00 * d11 - d01 * d01
    if denominator <= 1e-18:
        raise BackendOperationError("convex hull contains a degenerate triangle")
    v = (d11 * d20 - d01 * d21) / denominator
    w = (d00 * d21 - d01 * d20) / denominator
    u = 1.0 - v - w
    if min(u, v, w) >= -1e-12:
        return signed_plane_distance * signed_plane_distance
    return min(
        _point_segment_distance_squared(point, a, b),
        _point_segment_distance_squared(point, b, c),
        _point_segment_distance_squared(point, c, a),
    )


def _point_segment_distance_squared(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> float:
    segment = end - start
    fraction = float(np.dot(point - start, segment) / np.dot(segment, segment))
    closest = start + min(max(fraction, 0.0), 1.0) * segment
    offset = point - closest
    return float(np.dot(offset, offset))


def _attached_sphere_specs(item: AttachedGeometry) -> tuple[tuple[np.ndarray, float], ...]:
    geometry = item.geometry
    centers_and_radii: list[tuple[np.ndarray, float]] = []
    if isinstance(geometry, Sphere):
        centers_and_radii.append((np.zeros(3, dtype=np.float64), geometry.radius_m))
    elif isinstance(geometry, Capsule):
        intervals = max(1, int(ceil(geometry.length_m / geometry.radius_m)))
        spacing = geometry.length_m / intervals
        covering_radius = sqrt(geometry.radius_m**2 + (0.5 * spacing) ** 2)
        for index in range(intervals + 1):
            z = -0.5 * geometry.length_m + index * spacing
            centers_and_radii.append((np.asarray((0.0, 0.0, z)), covering_radius))
    elif isinstance(geometry, Box):
        size = np.asarray(geometry.size_m, dtype=np.float64)
        cell_target = float(np.min(size))
        counts = np.maximum(1, np.ceil(size / cell_target).astype(int))
        cell_size = size / counts
        covering_radius = 0.5 * float(np.linalg.norm(cell_size))
        for x_index in range(int(counts[0])):
            for y_index in range(int(counts[1])):
                for z_index in range(int(counts[2])):
                    center = -0.5 * size + cell_size * (
                        np.asarray((x_index, y_index, z_index), dtype=np.float64) + 0.5
                    )
                    centers_and_radii.append((center, covering_radius))
    else:
        raise BackendOperationError(
            f"unsupported attached geometry: {type(geometry).__name__}"
        )
    transform = np.asarray(item.T_object_geometry.matrix, dtype=np.float64)
    return tuple(
        (transform[:3, :3] @ center + transform[:3, 3], radius)
        for center, radius in centers_and_radii
    )


def _pointcloud_contact_groups(
    robot_spheres: list[tuple[object, str]],
    item: SceneGeometry,
    tree: cKDTree,
    sphere_centers: np.ndarray | None = None,
) -> frozenset[str]:
    geometry = item.geometry
    if not isinstance(geometry, PointCloud):
        raise TypeError("point-cloud collision query requires PointCloud geometry")
    transform = np.asarray(item.T_world_geometry.matrix, dtype=np.float64)
    centers = (
        sphere_centers
        if sphere_centers is not None
        else _sphere_arrays(robot_spheres)[0]
    )
    centers_local = (centers - transform[:3, 3]) @ transform[:3, :3]
    distances, _ = tree.query(centers_local, k=1, workers=1)
    return frozenset(
        group_name
        for distance, (sphere, group_name) in zip(distances, robot_spheres)
        if float(distance) <= float(sphere.r) + geometry.point_radius_m
    )


def _convex_contact_groups(
    robot_spheres: list[tuple[object, str]],
    item: SceneGeometry,
    prepared: tuple[np.ndarray, ...],
    sphere_centers: np.ndarray | None = None,
    sphere_radii: np.ndarray | None = None,
) -> frozenset[str]:
    geometry = item.geometry
    if not isinstance(geometry, ConvexHull):
        raise TypeError("convex collision query requires ConvexHull geometry")
    transform = np.asarray(item.T_world_geometry.matrix, dtype=np.float64)
    if sphere_centers is None or sphere_radii is None:
        centers, radii = _sphere_arrays(robot_spheres)
    else:
        centers, radii = sphere_centers, sphere_radii
    centers_local = (centers - transform[:3, 3]) @ transform[:3, :3]
    a, b, c, normals = prepared[:4]
    if len(prepared) == 6:
        lower, upper = prepared[4:]
    else:
        vertices = np.concatenate((a, b, c), axis=0)
        lower = np.min(vertices, axis=0)
        upper = np.max(vertices, axis=0)
    broadphase_indices = np.flatnonzero(
        np.all(centers_local + radii[:, None] >= lower[None, :], axis=1)
        & np.all(centers_local - radii[:, None] <= upper[None, :], axis=1)
    )
    if not broadphase_indices.size:
        return frozenset()
    centers_local = centers_local[broadphase_indices]
    radii = radii[broadphase_indices]
    candidate_spheres = [robot_spheres[index] for index in broadphase_indices]
    relative = centers_local[:, None, :] - a[None, :, :]
    signed_plane_distances = np.einsum(
        "sfi,fi->sf", relative, normals, optimize=True
    )
    inside = np.all(signed_plane_distances <= 1e-12, axis=1)
    separated = np.any(
        signed_plane_distances > radii[:, None] + 1e-12, axis=1
    )
    hits = inside.copy()
    candidate_indices = np.flatnonzero(~inside & ~separated)
    if candidate_indices.size:
        candidate_centers = centers_local[candidate_indices]
        candidate_relative = relative[candidate_indices]
        candidate_plane_distances = signed_plane_distances[candidate_indices]
        projections = (
            candidate_relative
            - candidate_plane_distances[:, :, None] * normals[None, :, :]
        )
        ab = b - a
        ac = c - a
        d00 = np.einsum("fi,fi->f", ab, ab, optimize=True)
        d01 = np.einsum("fi,fi->f", ab, ac, optimize=True)
        d11 = np.einsum("fi,fi->f", ac, ac, optimize=True)
        d20 = np.einsum("sfi,fi->sf", projections, ab, optimize=True)
        d21 = np.einsum("sfi,fi->sf", projections, ac, optimize=True)
        denominator = d00 * d11 - d01 * d01
        v = (d11[None, :] * d20 - d01[None, :] * d21) / denominator[None, :]
        w = (d00[None, :] * d21 - d01[None, :] * d20) / denominator[None, :]
        u = 1.0 - v - w
        projection_inside = np.minimum(np.minimum(u, v), w) >= -1e-12
        plane_distance_squared = candidate_plane_distances**2
        edge_distance_squared = np.minimum(
            np.minimum(
                _point_segment_distances_squared(candidate_centers, a, b),
                _point_segment_distances_squared(candidate_centers, b, c),
            ),
            _point_segment_distances_squared(candidate_centers, c, a),
        )
        surface_distance_squared = np.min(
            np.where(
                projection_inside,
                plane_distance_squared,
                edge_distance_squared,
            ),
            axis=1,
        )
        hits[candidate_indices] = surface_distance_squared <= (
            radii[candidate_indices] ** 2 + 1e-12
        )
    return frozenset(
        group_name
        for hit, (_, group_name) in zip(hits, candidate_spheres)
        if bool(hit)
    )


def _sphere_arrays(
    spheres: list[tuple[object, str]] | tuple[tuple[object, str], ...],
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(
            [[sphere.x, sphere.y, sphere.z] for sphere, _ in spheres],
            dtype=np.float64,
        ),
        np.asarray([sphere.r for sphere, _ in spheres], dtype=np.float64),
    )


def _point_segment_distances_squared(
    points: np.ndarray, starts: np.ndarray, ends: np.ndarray
) -> np.ndarray:
    segments = ends - starts
    relative = points[:, None, :] - starts[None, :, :]
    fractions = np.einsum(
        "sfi,fi->sf", relative, segments, optimize=True
    ) / np.einsum("fi,fi->f", segments, segments, optimize=True)[None, :]
    closest = starts[None, :, :] + np.clip(fractions, 0.0, 1.0)[:, :, None] * segments[
        None, :, :
    ]
    offsets = points[:, None, :] - closest
    return np.einsum("sfi,sfi->sf", offsets, offsets, optimize=True)


def _config_value(payload: Mapping[str, object], key: str, label: str) -> object:
    if key not in payload:
        raise ValueError(f"{label}.{key} is required")
    return payload[key]


def _config_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return value


def _config_string(payload: Mapping[str, object], key: str, label: str) -> str:
    value = _config_value(payload, key, label)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value


def _config_integer(payload: Mapping[str, object], key: str, label: str) -> int:
    value = _config_value(payload, key, label)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label}.{key} must be an integer")
    return value


def _config_boolean(payload: Mapping[str, object], key: str, label: str) -> bool:
    value = _config_value(payload, key, label)
    if not isinstance(value, bool):
        raise ValueError(f"{label}.{key} must be a boolean")
    return value


def _config_number(payload: Mapping[str, object], key: str, label: str) -> float:
    value = _config_value(payload, key, label)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}.{key} must be a number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{label}.{key} must be finite")
    return result


def _config_vector(
    payload: Mapping[str, object], key: str, label: str
) -> Vector3:
    value = _config_value(payload, key, label)
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{label}.{key} must contain three numbers")
    result = tuple(
        _config_number({"value": item}, "value", f"{label}.{key}") for item in value
    )
    return result  # type: ignore[return-value]
