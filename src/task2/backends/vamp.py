from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from math import ceil, sqrt
from time import monotonic
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

from ..contracts import (
    AttachedGeometry,
    Box,
    Capsule,
    ConvexHull,
    PlanningConfig,
    PointCloud,
    RigidTransform,
    RobotModel,
    SceneGeometry,
    SceneSnapshot,
    Sphere,
    StaticTriangleMesh,
)
from ..contracts.geometry import compose, inverse
from .errors import BackendOperationError, BackendUnavailableError


@dataclass(frozen=True)
class VampPlan:
    path: tuple[tuple[float, ...], ...]
    elapsed_s: float
    iterations: int


class VampBackend:
    def __init__(self, robot_model: RobotModel, planning_config: PlanningConfig):
        try:
            import vamp
        except (ImportError, OSError) as exc:
            raise BackendUnavailableError(f"VAMP import failed: {exc}") from exc
        try:
            installed = metadata.version("vamp-planner")
        except metadata.PackageNotFoundError as exc:
            raise BackendUnavailableError("vamp-planner package metadata is unavailable") from exc
        if installed != planning_config.vamp_version:
            raise BackendUnavailableError(
                f"VAMP version {installed} does not match required {planning_config.vamp_version}"
            )
        if tuple(getattr(vamp, "robots", ())) != (robot_model.vamp_robot_module,):
            raise BackendUnavailableError(
                "patched VAMP must register only the configured Gen3/SO-101 module"
            )
        robot = getattr(vamp, robot_model.vamp_robot_module, None)
        if robot is None:
            raise BackendUnavailableError(
                f"VAMP robot module {robot_model.vamp_robot_module} is unavailable"
            )
        if tuple(robot.joint_names()) != robot_model.joint_names:
            raise BackendUnavailableError("VAMP joint order does not match RobotModel")
        if int(robot.n_spheres()) != robot_model.vamp_robot_sphere_count:
            raise BackendUnavailableError("VAMP collision sphere count does not match RobotModel")
        collision_spheres = tuple(robot.fk(robot_model.default_joint_positions))
        for group in robot_model.collision_sphere_groups:
            first = group.first_sphere
            last = first + group.sphere_count
            if min(item.r for item in collision_spheres[first:last]) <= group.sweep_radius_inflation_m:
                raise BackendUnavailableError(
                    f"patched VAMP module lacks the full-stroke envelope for {group.group}"
                )
        self._vamp = vamp
        self._robot = robot
        self._model = robot_model
        self._config = planning_config
        self._pointcloud_trees: dict[int, cKDTree] = {}
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
        settings.max_iterations = self._config.vamp_max_iterations
        settings.max_samples = self._config.vamp_max_samples
        settings.range = self._config.vamp_range_rad
        rng = self._robot.halton()
        rng.skip(seed)
        started = monotonic()
        try:
            result = self._robot.rrtc(start, goal, environment, settings, rng)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise BackendOperationError(f"VAMP RRTC failed: {exc}") from exc
        elapsed = monotonic() - started
        if not result.solved:
            raise BackendOperationError("VAMP RRTC exhausted its configured search budget")
        path_array = np.asarray(result.path.numpy(), dtype=np.float64)
        if path_array.ndim != 2 or path_array.shape[1] != len(self._model.joint_names):
            raise BackendOperationError("VAMP returned a malformed path")
        path = tuple(tuple(float(value) for value in row) for row in path_array)
        if not path:
            raise BackendOperationError("VAMP returned an empty path")
        return VampPlan(path=path, elapsed_s=elapsed, iterations=int(result.iterations))

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
        if not 0.0 <= opening_m <= self._model.maximum_gripper_opening_m:
            raise BackendOperationError("gripper opening is outside RobotModel limits")
        sphere_groups = {}
        for group in self._model.collision_sphere_groups:
            for index in range(group.first_sphere, group.first_sphere + group.sphere_count):
                sphere_groups[index] = group
        closed_fraction = 1.0 - opening_m / self._model.maximum_gripper_opening_m
        tcp_rotation = np.asarray(self.tcp_pose(configuration).matrix, dtype=np.float64)[:3, :3]
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
        adjusted_spheres = []
        for index, robot_sphere in enumerate(self.robot_spheres(configuration)):
            adjusted = robot_sphere
            group = sphere_groups.get(index)
            if group is not None:
                local_shift = np.asarray(
                    group.translation_per_closed_fraction_m, dtype=np.float64
                ) * (closed_fraction - self._model.vamp_tool_fraction_midpoint)
                world_shift = tcp_rotation @ local_shift
                adjusted = self._vamp.Sphere(
                    [
                        robot_sphere.x + world_shift[0],
                        robot_sphere.y + world_shift[1],
                        robot_sphere.z + world_shift[2],
                    ],
                    robot_sphere.r - group.sweep_radius_inflation_m,
                )
            adjusted_spheres.append(
                (adjusted, group.group if group is not None else "robot_non_pad")
            )
        for geometry in geometries_in_base:
            if isinstance(geometry.geometry, PointCloud):
                for group_name in _pointcloud_contact_groups(
                    adjusted_spheres,
                    geometry,
                    self._pointcloud_tree(geometry),
                ):
                    contacts.add(group_name)
                continue
            for adjusted, group_name in adjusted_spheres:
                if _sphere_intersects_geometry(adjusted, geometry):
                    contacts.add(group_name)
        return frozenset(contacts)

    def semantic_attachment_contacts(
        self,
        configuration: tuple[float, ...],
        T_tcp_object: RigidTransform,
        attached_geometry: tuple[AttachedGeometry, ...],
        scene_geometries: tuple[SceneGeometry, ...],
        T_world_base: RigidTransform,
    ) -> frozenset[str]:
        tcp_matrix = np.asarray(self.tcp_pose(configuration).matrix, dtype=np.float64)
        object_matrix = tcp_matrix @ np.asarray(T_tcp_object.matrix, dtype=np.float64)
        attached_spheres = []
        for item in attached_geometry:
            for center, radius in _attached_sphere_specs(item):
                center_base = object_matrix[:3, :3] @ center + object_matrix[:3, 3]
                attached_spheres.append(
                    self._vamp.Sphere(center_base.tolist(), radius)
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
            if any(
                _sphere_intersects_geometry(sphere, geometry)
                for sphere in attached_spheres
            ):
                contacts.add(geometry.collision_group)
        return frozenset(contacts)

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
        elif isinstance(geometry, (ConvexHull, StaticTriangleMesh)):
            raise BackendUnavailableError(
                f"VAMP 0.6.4 has no native {type(geometry).__name__} collision; "
                "convert this SceneSnapshot geometry to an explicit PointCloud upstream"
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
    if isinstance(geometry, (ConvexHull, StaticTriangleMesh)):
        raise BackendUnavailableError(
            f"VAMP 0.6.4 has no native {type(geometry).__name__} collision; "
            "convert this SceneSnapshot geometry to an explicit PointCloud upstream"
        )
    raise BackendOperationError(f"unsupported semantic collision geometry: {type(geometry).__name__}")


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
) -> frozenset[str]:
    geometry = item.geometry
    if not isinstance(geometry, PointCloud):
        raise TypeError("point-cloud collision query requires PointCloud geometry")
    transform = np.asarray(item.T_world_geometry.matrix, dtype=np.float64)
    centers = np.asarray(
        [[sphere.x, sphere.y, sphere.z] for sphere, _ in robot_spheres],
        dtype=np.float64,
    )
    centers_local = (centers - transform[:3, 3]) @ transform[:3, :3]
    distances, _ = tree.query(centers_local, k=1, workers=1)
    return frozenset(
        group_name
        for distance, (sphere, group_name) in zip(distances, robot_spheres)
        if float(distance) <= float(sphere.r) + geometry.point_radius_m
    )
