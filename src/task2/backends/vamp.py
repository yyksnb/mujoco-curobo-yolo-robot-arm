from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from math import ceil, isfinite, sqrt
from time import monotonic
from typing import Iterable, Mapping

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

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
class VampConfig:
    version: str
    robot_module: str
    robot_sphere_count: int
    tool_fraction_midpoint: float
    maximum_gripper_opening_m: float
    collision_sphere_groups: tuple[CollisionSphereGroup, ...]
    collision_sphere_calibrations: tuple[CollisionSphereCalibration, ...]
    range_rad: float
    dynamic_domain: bool
    max_iterations: int
    max_samples: int

    def validate(self) -> None:
        if not self.version or not self.robot_module or self.robot_sphere_count <= 0:
            raise ValueError("VAMP version, robot module, and sphere count must be explicit")
        if not 0.0 <= self.tool_fraction_midpoint <= 1.0:
            raise ValueError("VAMP tool fraction midpoint must lie in [0, 1]")
        if not isfinite(self.maximum_gripper_opening_m) or self.maximum_gripper_opening_m <= 0.0:
            raise ValueError("VAMP maximum gripper opening must be positive")
        if not isfinite(self.range_rad) or self.range_rad <= 0.0:
            raise ValueError("VAMP range must be positive")
        if self.max_iterations <= 0 or self.max_samples <= 0:
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
        _config_number(payload, "range_rad", "VAMP"),
        _config_boolean(payload, "dynamic_domain", "VAMP"),
        _config_integer(payload, "max_iterations", "VAMP"),
        _config_integer(payload, "max_samples", "VAMP"),
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
        self.version = installed

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
        return environment

    def plan(
        self,
        start: tuple[float, ...],
        goal: tuple[float, ...],
        environment,
        *,
        seed: int,
    ) -> VampPlan:
        self._validate_configuration_shape(start)
        self._validate_configuration_shape(goal)
        settings = self._vamp.RRTCSettings()
        settings.max_iterations = self._config.max_iterations
        settings.max_samples = self._config.max_samples
        settings.range = self._config.range_rad
        settings.dynamic_domain = self._config.dynamic_domain
        rng = self._robot.halton()
        rng.skip(seed)
        started = monotonic()
        try:
            result = self._robot.rrtc(start, goal, environment, settings, rng)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise BackendOperationError(f"VAMP RRTC failed: {exc}") from exc
        if not result.solved:
            raise BackendOperationError("VAMP RRTC exhausted its configured search budget")
        planning_iterations = int(result.iterations)
        simplify_settings = self._vamp.SimplifySettings()
        simplify_settings.max_iterations = 1
        simplify_settings.operations = [
            self._vamp.SimplifyRoutine.SHORTCUT,
            self._vamp.SimplifyRoutine.BSPLINE,
        ]
        simplify_settings.bspline.max_steps = 3
        simplify_settings.bspline.min_change = 0.001
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
        return VampPlan(path=path, elapsed_s=elapsed, iterations=planning_iterations)

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
        self._validate_configuration_shape(configuration)
        raw = np.asarray(self._robot.eefk(configuration), dtype=np.float64)
        if raw.shape == (4, 4):
            matrix = tuple(tuple(float(value) for value in row) for row in raw)
        elif raw.shape == (7,):
            x, y, z, qx, qy, qz, qw = raw
            rotation = Rotation.from_quat((qx, qy, qz, qw)).as_matrix()
            matrix = tuple(
                tuple(float(rotation[row, column]) for column in range(3))
                + ((x, y, z)[row],)
                for row in range(3)
            ) + ((0.0, 0.0, 0.0, 1.0),)
        else:
            raise BackendOperationError(f"VAMP returned unexpected TCP pose shape {raw.shape}")
        result = RigidTransform(self._model.base_frame, self._model.tcp_frame, matrix)  # type: ignore[arg-type]
        result.validate(label="VAMP TCP pose")
        return result

    def semantic_target_contacts(
        self,
        configuration: tuple[float, ...],
        target_geometries: tuple[SceneGeometry, ...],
        opening_m: float,
        T_world_base: RigidTransform,
    ) -> frozenset[str]:
        contacts = set()
        geometries_in_base = tuple(
            SceneGeometry(
                item.geometry_id,
                item.object_id,
                item.collision_group,
                compose(inverse(T_world_base), item.T_world_geometry),
                item.geometry,
            )
            for item in target_geometries
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
        return frozenset(contacts)

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
        attached_spheres = self._attachment_spheres(
            configuration, T_tcp_object, attached_geometry
        )
        geometries_in_base = tuple(
            SceneGeometry(
                item.geometry_id,
                item.object_id,
                item.collision_group,
                compose(inverse(T_world_base), item.T_world_geometry),
                item.geometry,
            )
            for item in scene_geometries
        )
        contacts = set()
        named_spheres = [(sphere, "attached_object") for sphere in attached_spheres]
        for geometry in geometries_in_base:
            if isinstance(geometry.geometry, PointCloud):
                if _pointcloud_contact_groups(
                    named_spheres, geometry, self._pointcloud_tree(geometry)
                ):
                    contacts.add(geometry.collision_group)
                continue
            if isinstance(geometry.geometry, ConvexHull):
                if _convex_contact_groups(
                    named_spheres,
                    geometry,
                    self._prepared_convex_mesh(geometry.geometry),
                ):
                    contacts.add(geometry.collision_group)
                continue
            if any(
                _sphere_intersects_geometry(sphere, geometry)
                for sphere in attached_spheres
            ):
                contacts.add(geometry.collision_group)
        return frozenset(contacts)

    def _semantic_robot_spheres(
        self,
        configuration: tuple[float, ...],
        opening_m: float,
    ) -> tuple[tuple[object, str], ...]:
        if not 0.0 <= opening_m <= self._config.maximum_gripper_opening_m:
            raise BackendOperationError("gripper opening is outside RobotModel limits")
        sphere_groups = {}
        for group in self._config.collision_sphere_groups:
            for index in range(
                group.first_sphere, group.first_sphere + group.sphere_count
            ):
                sphere_groups[index] = group
        closed_fraction = 1.0 - opening_m / self._config.maximum_gripper_opening_m
        tcp_rotation = np.asarray(
            self.tcp_pose(configuration).matrix, dtype=np.float64
        )[:3, :3]
        adjusted_spheres = []
        for index, robot_sphere in enumerate(self.robot_spheres(configuration)):
            adjusted = robot_sphere
            group = sphere_groups.get(index)
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
        tcp_matrix = np.asarray(self.tcp_pose(configuration).matrix, dtype=np.float64)
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
