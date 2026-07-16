from __future__ import annotations

import math
import random
import xml.etree.ElementTree as ElementTree
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


TARGET_BODY_PREFIX = "target_"
DEFAULT_MODEL_PATH = Path("examples/mujoco/gen3_with_tank.xml")
DEFAULT_OBJECT_COUNT = 5
DEFAULT_SPAWN_HEIGHT_M = 0.025
DEFAULT_COLLISION_MARGIN_M = 0.012
_SUPPORT_GEOM_NAMES = ("tank_bottom_collision",)
_SUPPORT_GEOM_PREFIXES = ("tank_bottom_rib_",)


@dataclass(frozen=True)
class PlacementBounds:
    x_min: float = 0.15
    x_max: float = 0.85
    y_min: float = 0.15
    y_max: float = 0.85


@dataclass(frozen=True)
class SupportPlacementPolicy:
    search_lower_z_m: float = 0.0
    search_upper_z_m: float = 0.1
    search_step_m: float = 0.0005
    height_tolerance_m: float = 1e-7
    max_iterations: int = 60

    def __post_init__(self) -> None:
        if self.search_lower_z_m >= self.search_upper_z_m:
            raise ValueError("support placement z search interval is invalid")
        if (
            self.search_step_m <= 0
            or self.height_tolerance_m <= 0
            or self.max_iterations <= 0
        ):
            raise ValueError("support placement convergence policy is invalid")


@dataclass(frozen=True)
class TargetObjectSpec:
    object_id: str
    class_name: str
    geom_name: str
    footprint_points_xy: tuple[tuple[float, float], ...]
    footprint_area_m2: float
    footprint_radius_m: float
    local_min_z_m: float
    local_max_z_m: float


@dataclass(frozen=True)
class TargetObjectPose:
    object_id: str
    class_name: str
    position: tuple[float, float, float]
    yaw_rad: float
    yaw_deg: float
    quat_wxyz: tuple[float, float, float, float]
    T_world_object: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]
    footprint_polygon_xy: tuple[tuple[float, float], ...]
    footprint_area_m2: float
    footprint_radius_m: float
    placement_order: int
    selection_order: int


def make_random_target_object_pose_payload(
    *,
    model_path: Path,
    seed: int,
    object_count: int = DEFAULT_OBJECT_COUNT,
    spawn_height_m: float = DEFAULT_SPAWN_HEIGHT_M,
    bounds: PlacementBounds = PlacementBounds(),
    collision_margin_m: float = DEFAULT_COLLISION_MARGIN_M,
    max_attempts_per_object: int = 6000,
    support_policy: SupportPlacementPolicy = SupportPlacementPolicy(),
) -> dict[str, Any]:
    specs = load_target_object_specs(model_path)
    selected_specs = select_target_object_specs(specs, object_count=object_count, seed=seed)
    poses = generate_random_target_object_poses(
        selected_specs,
        seed=seed,
        spawn_height_m=spawn_height_m,
        bounds=bounds,
        collision_margin_m=collision_margin_m,
        max_attempts_per_object=max_attempts_per_object,
    )
    poses = place_target_objects_on_support(
        model_path,
        poses,
        policy=support_policy,
    )
    poses_by_name = {pose.object_id: pose for pose in poses}
    ordered_poses = tuple(poses_by_name[spec.object_id] for spec in selected_specs)
    return {
        "schema": "target_object_pose_layout",
        "model_path": str(model_path),
        "seed": seed,
        "object_count": object_count,
        "spawn_height_m": spawn_height_m,
        "support_placement_policy": asdict(support_policy),
        "placement_bounds": asdict(bounds),
        "collision_margin_m": collision_margin_m,
        "max_attempts_per_object": max_attempts_per_object,
        "target_body_prefix": TARGET_BODY_PREFIX,
        "notes": [
            "Object poses are generated from MuJoCo target_* mesh footprints.",
            "The script writes poses only; it does not modify MJCF/XML files.",
            "Collision checks use conservative 2D footprint polygons on the tank bottom plane.",
            "Spawn poses are lowered onto the first non-penetrating tank support contact.",
        ],
        "objects": [target_pose_to_dict(pose) for pose in ordered_poses],
    }


def load_target_object_specs(model_path: Path) -> tuple[TargetObjectSpec, ...]:
    mujoco = _import_mujoco()
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo model file does not exist: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path.resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    specs: list[TargetObjectSpec] = []
    for body_id in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if not _is_target_object_body(body_name):
            continue

        geom_ids = tuple(geom_id for geom_id in range(model.ngeom) if int(model.geom_bodyid[geom_id]) == body_id)
        if not geom_ids:
            continue
        geom_id = geom_ids[0]
        vertices = _target_mesh_vertices_in_body_frame(mujoco, model, data, body_id, geom_id)
        footprint = _convex_hull_xy(vertices[:, :2])
        if len(footprint) < 3:
            raise ValueError(f"target object footprint is degenerate: {body_name}")

        radius = float(np.max(np.linalg.norm(footprint, axis=1)))
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"{body_name}_geom"
        specs.append(
            TargetObjectSpec(
                object_id=body_name,
                class_name=body_name.removeprefix(TARGET_BODY_PREFIX),
                geom_name=geom_name,
                footprint_points_xy=_tuple_points(footprint),
                footprint_area_m2=round(abs(_polygon_area_xy(footprint)), 6),
                footprint_radius_m=round(radius, 6),
                local_min_z_m=round(float(np.min(vertices[:, 2])), 6),
                local_max_z_m=round(float(np.max(vertices[:, 2])), 6),
            )
        )

    if not specs:
        raise ValueError(f"no {TARGET_BODY_PREFIX} object bodies found in model: {model_path}")
    return tuple(sorted(specs, key=lambda spec: spec.object_id))


def select_target_object_specs(
    specs: tuple[TargetObjectSpec, ...],
    *,
    object_count: int,
    seed: int,
) -> tuple[TargetObjectSpec, ...]:
    if object_count <= 0:
        raise ValueError("object_count must be positive")
    if len(specs) < object_count:
        raise ValueError(f"not enough target objects in model ({len(specs)} < {object_count})")

    rng = random.Random(seed)
    selected = rng.sample(list(specs), object_count)
    return tuple(selected)


def generate_random_target_object_poses(
    specs: tuple[TargetObjectSpec, ...],
    *,
    seed: int,
    spawn_height_m: float,
    bounds: PlacementBounds = PlacementBounds(),
    collision_margin_m: float = DEFAULT_COLLISION_MARGIN_M,
    max_attempts_per_object: int = 6000,
) -> tuple[TargetObjectPose, ...]:
    _validate_bounds(bounds)
    if max_attempts_per_object <= 0:
        raise ValueError("max_attempts_per_object must be positive")

    rng = random.Random(f"{seed}:layout")
    selected_order = {spec.object_id: index for index, spec in enumerate(specs)}
    ordered_specs = sorted(
        specs,
        key=lambda spec: (spec.footprint_area_m2, spec.footprint_radius_m, spec.object_id),
        reverse=True,
    )

    placed: list[TargetObjectPose] = []
    placed_polygons: list[np.ndarray] = []
    for placement_order, spec in enumerate(ordered_specs):
        footprint = np.asarray(spec.footprint_points_xy, dtype=float)
        for _ in range(max_attempts_per_object):
            x = rng.uniform(bounds.x_min, bounds.x_max)
            y = rng.uniform(bounds.y_min, bounds.y_max)
            yaw_rad = rng.uniform(-math.pi, math.pi)
            polygon = transform_footprint_xy(footprint, x=x, y=y, yaw_rad=yaw_rad)
            if not polygon_within_bounds(polygon, bounds):
                continue
            if footprint_collides_with_any(polygon, placed_polygons, collision_margin_m):
                continue

            position = (_round(x), _round(y), _round(spawn_height_m))
            quaternion = _yaw_quaternion_wxyz(yaw_rad)
            placed.append(
                TargetObjectPose(
                    object_id=spec.object_id,
                    class_name=spec.class_name,
                    position=position,
                    yaw_rad=_round(yaw_rad),
                    yaw_deg=_round(math.degrees(yaw_rad)),
                    quat_wxyz=quaternion,
                    T_world_object=_make_transform(position, quaternion),
                    footprint_polygon_xy=_tuple_points(polygon),
                    footprint_area_m2=spec.footprint_area_m2,
                    footprint_radius_m=spec.footprint_radius_m,
                    placement_order=placement_order,
                    selection_order=selected_order[spec.object_id],
                )
            )
            placed_polygons.append(polygon)
            break
        else:
            raise RuntimeError(f"could not place target object without collision: {spec.object_id}")

    return tuple(sorted(placed, key=lambda pose: pose.selection_order))


def place_target_objects_on_support(
    model_path: Path,
    poses: tuple[TargetObjectPose, ...],
    *,
    policy: SupportPlacementPolicy = SupportPlacementPolicy(),
) -> tuple[TargetObjectPose, ...]:
    """Resolve each normalized object origin onto the tank support collision geometry."""
    mujoco = _import_mujoco()
    model = mujoco.MjModel.from_xml_path(str(Path(model_path).resolve()))
    data = mujoco.MjData(model)
    selected = {
        pose.object_id: {
            "position": pose.position,
            "quat_wxyz": pose.quat_wxyz,
        }
        for pose in poses
    }
    apply_target_object_layout(mujoco, model, data, selected)
    support_geom_ids = _support_geom_ids(mujoco, model)

    for pose in poses:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, pose.object_id)
        if body_id < 0:
            raise ValueError(f"layout target body does not exist: {pose.object_id}")
        target_geom_id = _single_target_geom_id(model, body_id, pose.object_id)
        qpos_address, dof_address = _target_free_joint_addresses(
            mujoco, model, body_id, pose.object_id
        )
        support_z = _first_support_contact_z(
            mujoco,
            model,
            data,
            target_geom_id=target_geom_id,
            qpos_address=qpos_address,
            support_geom_ids=support_geom_ids,
            policy=policy,
        )
        data.qpos[qpos_address + 2] = support_z
        data.qvel[dof_address : dof_address + 6] = 0.0
        model.qpos0[qpos_address : qpos_address + 7] = data.qpos[
            qpos_address : qpos_address + 7
        ]
    mujoco.mj_forward(model, data)

    supported: list[TargetObjectPose] = []
    for pose in poses:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, pose.object_id)
        target_geom_id = _single_target_geom_id(model, body_id, pose.object_id)
        qpos_address, _ = _target_free_joint_addresses(
            mujoco, model, body_id, pose.object_id
        )
        qpos = np.asarray(data.qpos[qpos_address : qpos_address + 7], dtype=float)
        position = (
            _round(qpos[0]),
            _round(qpos[1]),
            _round(qpos[2], 9),
        )
        quaternion = _normalized_quaternion_values(qpos[3:7])
        vertices_world = _mesh_vertices_world(model, data, target_geom_id)
        footprint = _convex_hull_xy(vertices_world[:, :2])
        if len(footprint) < 3:
            raise ValueError(f"supported target footprint is degenerate: {pose.object_id}")
        radius = float(
            np.max(
                np.linalg.norm(
                    footprint - np.asarray(position[:2], dtype=float), axis=1
                )
            )
        )
        yaw_rad = _yaw_from_quaternion(quaternion)
        supported.append(
            TargetObjectPose(
                object_id=pose.object_id,
                class_name=pose.class_name,
                position=position,
                yaw_rad=_round(yaw_rad),
                yaw_deg=_round(math.degrees(yaw_rad)),
                quat_wxyz=quaternion,
                T_world_object=_make_transform(position, quaternion),
                footprint_polygon_xy=_tuple_points(footprint),
                footprint_area_m2=round(abs(_polygon_area_xy(footprint)), 6),
                footprint_radius_m=round(radius, 6),
                placement_order=pose.placement_order,
                selection_order=pose.selection_order,
            )
        )
    return tuple(sorted(supported, key=lambda pose: pose.selection_order))


def target_pose_to_dict(pose: TargetObjectPose) -> dict[str, Any]:
    return {
        "object_id": pose.object_id,
        "class_name": pose.class_name,
        "position": list(pose.position),
        "yaw_rad": pose.yaw_rad,
        "yaw_deg": pose.yaw_deg,
        "quat_wxyz": list(pose.quat_wxyz),
        "T_world_object": [list(row) for row in pose.T_world_object],
        "footprint_polygon_xy": [list(point) for point in pose.footprint_polygon_xy],
        "footprint_area_m2": pose.footprint_area_m2,
        "footprint_radius_m": pose.footprint_radius_m,
        "placement_order": pose.placement_order,
        "selection_order": pose.selection_order,
    }


def transform_footprint_xy(points_xy: np.ndarray, *, x: float, y: float, yaw_rad: float) -> np.ndarray:
    rotation = np.asarray(
        (
            (math.cos(yaw_rad), -math.sin(yaw_rad)),
            (math.sin(yaw_rad), math.cos(yaw_rad)),
        ),
        dtype=float,
    )
    return np.asarray(points_xy, dtype=float) @ rotation.T + np.asarray((x, y), dtype=float)


def polygon_within_bounds(polygon_xy: np.ndarray, bounds: PlacementBounds, epsilon: float = 1e-9) -> bool:
    xs = polygon_xy[:, 0]
    ys = polygon_xy[:, 1]
    return (
        float(np.min(xs)) >= bounds.x_min - epsilon
        and float(np.max(xs)) <= bounds.x_max + epsilon
        and float(np.min(ys)) >= bounds.y_min - epsilon
        and float(np.max(ys)) <= bounds.y_max + epsilon
    )


def footprint_collides_with_any(
    polygon_xy: np.ndarray,
    existing_polygons_xy: list[np.ndarray],
    margin_m: float,
) -> bool:
    return any(convex_polygons_intersect(polygon_xy, existing, margin_m) for existing in existing_polygons_xy)


def convex_polygons_intersect(poly_a: np.ndarray, poly_b: np.ndarray, margin_m: float = 0.0) -> bool:
    if len(poly_a) < 3 or len(poly_b) < 3:
        return _point_sets_too_close(poly_a, poly_b, margin_m)

    for polygon in (poly_a, poly_b):
        for index in range(len(polygon)):
            edge = polygon[(index + 1) % len(polygon)] - polygon[index]
            axis = np.asarray((-edge[1], edge[0]), dtype=float)
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm <= 1e-12:
                continue
            axis /= axis_norm
            min_a, max_a = _project_polygon(poly_a, axis)
            min_b, max_b = _project_polygon(poly_b, axis)
            if max_a < min_b - margin_m or max_b < min_a - margin_m:
                return False
    return True


def _is_target_object_body(body_name: str) -> bool:
    return body_name.startswith(TARGET_BODY_PREFIX) and body_name != "target_object_include_root"


def apply_target_object_layout(
    mujoco: Any,
    model: Any,
    data: Any,
    selected_objects: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    """Initialize target free joints from a Task1 layout and hide unselected objects."""
    applied: list[str] = []
    for body_id in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if not _is_target_object_body(body_name):
            continue

        item = selected_objects.get(body_name)
        if item is None:
            position = (0.0, 0.0, -10.0)
            quaternion = (1.0, 0.0, 0.0, 0.0)
            for geom_id in range(model.ngeom):
                if int(model.geom_bodyid[geom_id]) == body_id:
                    model.geom_contype[geom_id] = 0
                    model.geom_conaffinity[geom_id] = 0
        else:
            raw_position = item.get("position")
            if not isinstance(raw_position, (list, tuple)) or len(raw_position) != 3:
                raise ValueError(f"layout position for {body_name} must contain three values")
            position = tuple(float(value) for value in raw_position)
            raw_quaternion = item.get("quat_wxyz")
            if not isinstance(raw_quaternion, (list, tuple)) or len(raw_quaternion) != 4:
                raise ValueError(f"layout quat_wxyz for {body_name} must contain four values")
            quaternion = _normalized_quaternion_values(raw_quaternion)
            if not all(math.isfinite(value) for value in position):
                raise ValueError(f"layout pose for {body_name} must be finite")
            applied.append(body_name)

        qpos_address, dof_address = _target_free_joint_addresses(
            mujoco, model, body_id, body_name
        )
        free_joint_qpos = (*position, *quaternion)
        model.qpos0[qpos_address : qpos_address + 7] = free_joint_qpos
        data.qpos[qpos_address : qpos_address + 7] = free_joint_qpos
        data.qvel[dof_address : dof_address + 6] = 0.0

    mujoco.mj_forward(model, data)
    return tuple(applied)


def _support_geom_ids(mujoco: Any, model: Any) -> tuple[int, ...]:
    support_geom_ids = tuple(
        geom_id
        for geom_id in range(model.ngeom)
        if (
            (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
            in _SUPPORT_GEOM_NAMES
            or name.startswith(_SUPPORT_GEOM_PREFIXES)
        )
    )
    if not support_geom_ids:
        raise ValueError("Task1 scene has no named tank support collision geometry")
    return support_geom_ids


def _single_target_geom_id(model: Any, body_id: int, body_name: str) -> int:
    geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == body_id
    ]
    if len(geom_ids) != 1:
        raise ValueError(f"target object body must have exactly one geom: {body_name}")
    return geom_ids[0]


def _first_support_contact_z(
    mujoco: Any,
    model: Any,
    data: Any,
    *,
    target_geom_id: int,
    qpos_address: int,
    support_geom_ids: tuple[int, ...],
    policy: SupportPlacementPolicy,
) -> float:
    support_geom_id_set = set(support_geom_ids)

    def has_support_contact(z_m: float) -> bool:
        data.qpos[qpos_address + 2] = z_m
        mujoco.mj_forward(model, data)
        return any(
            target_geom_id in (int(contact.geom1), int(contact.geom2))
            and bool(
                {int(contact.geom1), int(contact.geom2)} & support_geom_id_set
            )
            for contact in data.contact[: data.ncon]
        )

    upper = policy.search_upper_z_m
    if has_support_contact(upper):
        raise ValueError("support placement upper z intersects tank geometry")
    previous = upper
    lower = policy.search_lower_z_m
    search_steps = math.ceil((upper - lower) / policy.search_step_m)
    for step in range(1, search_steps + 1):
        candidate = max(lower, upper - step * policy.search_step_m)
        if has_support_contact(candidate):
            lower = candidate
            upper = previous
            break
        previous = candidate
    else:
        raise ValueError("target object footprint has no tank support below it")

    for _ in range(policy.max_iterations):
        middle = (lower + upper) / 2.0
        if has_support_contact(middle):
            lower = middle
        else:
            upper = middle
        if upper - lower <= policy.height_tolerance_m:
            break

    support_z = upper
    data.qpos[qpos_address + 2] = support_z
    mujoco.mj_forward(model, data)
    if has_support_contact(support_z):
        raise ValueError("resolved target support pose remains in contact penetration")
    return support_z


def _target_free_joint_addresses(
    mujoco: Any,
    model: Any,
    body_id: int,
    body_name: str,
) -> tuple[int, int]:
    if int(model.body_jntnum[body_id]) != 1:
        raise ValueError(f"target object body must have exactly one free joint: {body_name}")
    joint_id = int(model.body_jntadr[body_id])
    if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError(f"target object body joint must be free: {body_name}")
    return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def _target_mesh_vertices_in_body_frame(mujoco, model, data, body_id: int, geom_id: int) -> np.ndarray:
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or str(body_id)
    qpos_address, dof_address = _target_free_joint_addresses(
        mujoco, model, body_id, body_name
    )
    original_qpos = np.asarray(data.qpos[qpos_address : qpos_address + 7], dtype=float).copy()
    original_qvel = np.asarray(data.qvel[dof_address : dof_address + 6], dtype=float).copy()
    try:
        data.qpos[qpos_address : qpos_address + 7] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        data.qvel[dof_address : dof_address + 6] = 0.0
        mujoco.mj_forward(model, data)
        if int(model.geom_type[geom_id]) != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError(f"target object geom must be a mesh: {geom_id}")
        mesh_id = int(model.geom_dataid[geom_id])
        vertex_start = int(model.mesh_vertadr[mesh_id])
        vertex_count = int(model.mesh_vertnum[mesh_id])
        if vertex_count <= 0:
            raise ValueError(f"target object mesh has no vertices: {mesh_id}")

        vertices = np.asarray(model.mesh_vert[vertex_start : vertex_start + vertex_count], dtype=float)
        geom_rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        geom_position = np.asarray(data.geom_xpos[geom_id], dtype=float)
        return vertices @ geom_rotation.T + geom_position
    finally:
        data.qpos[qpos_address : qpos_address + 7] = original_qpos
        data.qvel[dof_address : dof_address + 6] = original_qvel
        mujoco.mj_forward(model, data)


def _mesh_vertices_world(model: Any, data: Any, geom_id: int) -> np.ndarray:
    mesh_id = int(model.geom_dataid[geom_id])
    vertex_start = int(model.mesh_vertadr[mesh_id])
    vertex_count = int(model.mesh_vertnum[mesh_id])
    if vertex_count <= 0:
        raise ValueError(f"target object mesh has no vertices: {mesh_id}")
    vertices = np.asarray(
        model.mesh_vert[vertex_start : vertex_start + vertex_count], dtype=float
    )
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    position = np.asarray(data.geom_xpos[geom_id], dtype=float)
    return vertices @ rotation.T + position


def _convex_hull_xy(points_xy: np.ndarray) -> np.ndarray:
    unique_points = sorted({(float(x), float(y)) for x, y in np.asarray(points_xy, dtype=float)[:, :2]})
    if len(unique_points) <= 2:
        return np.asarray(unique_points, dtype=float)

    def cross(origin: tuple[float, float], left: tuple[float, float], right: tuple[float, float]) -> float:
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (right[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _polygon_area_xy(points_xy: np.ndarray) -> float:
    if len(points_xy) < 3:
        return 0.0
    xs = points_xy[:, 0]
    ys = points_xy[:, 1]
    return 0.5 * float(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))


def _project_polygon(polygon_xy: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    values = polygon_xy @ axis
    return float(np.min(values)), float(np.max(values))


def _point_sets_too_close(points_a: np.ndarray, points_b: np.ndarray, margin_m: float) -> bool:
    for point_a in points_a:
        for point_b in points_b:
            if float(np.linalg.norm(point_a - point_b)) <= margin_m:
                return True
    return False


def _make_transform(
    position: tuple[float, float, float],
    quaternion_wxyz: tuple[float, float, float, float],
) -> tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]:
    w, x, y, z = quaternion_wxyz
    rotation = (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )
    x, y, z = position
    return (
        (*(_round(value) for value in rotation[0]), x),
        (*(_round(value) for value in rotation[1]), y),
        (*(_round(value) for value in rotation[2]), z),
        (0.0, 0.0, 0.0, 1.0),
    )


def _yaw_quaternion_wxyz(yaw_rad: float) -> tuple[float, float, float, float]:
    half_yaw = yaw_rad / 2.0
    return (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))


def _normalized_quaternion_values(
    values: Any,
) -> tuple[float, float, float, float]:
    quaternion = np.asarray(values, dtype=float)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("layout quaternion must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("layout quaternion must be non-zero")
    quaternion /= norm
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return tuple(float(value) for value in quaternion)


def _yaw_from_quaternion(quaternion_wxyz: tuple[float, float, float, float]) -> float:
    w, x, y, z = quaternion_wxyz
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _tuple_points(points_xy: np.ndarray) -> tuple[tuple[float, float], ...]:
    return tuple((_round(point[0]), _round(point[1])) for point in points_xy)


def _validate_bounds(bounds: PlacementBounds) -> None:
    if bounds.x_min >= bounds.x_max:
        raise ValueError("x_min must be less than x_max")
    if bounds.y_min >= bounds.y_max:
        raise ValueError("y_min must be less than y_max")


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _import_mujoco():
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("MuJoCo is required to read target object mesh footprints. Install it with: pip install mujoco") from exc
    return mujoco


@dataclass(frozen=True)
class RigidPose:
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in self.position):
            raise ValueError("pose position must be finite")
        norm = math.sqrt(sum(value * value for value in self.quaternion_wxyz))
        if not math.isclose(norm, 1.0, abs_tol=1e-6):
            raise ValueError("pose quaternion must be normalized")


def load_tank_pose_in_base(scene_path: Path) -> RigidPose:
    worldbody, angle_unit = _scene_transform_context(scene_path)
    base = _named_direct_body(worldbody, "gen3_mount")
    tank = _named_direct_body(worldbody, "tank")
    return compose(
        inverse(_body_pose(base, angle_unit=angle_unit)),
        _body_pose(tank, angle_unit=angle_unit),
    )


def load_world_pose_in_base(scene_path: Path) -> RigidPose:
    worldbody, angle_unit = _scene_transform_context(scene_path)
    base = _named_direct_body(worldbody, "gen3_mount")
    return inverse(_body_pose(base, angle_unit=angle_unit))


def compose(parent_from_middle: RigidPose, middle_from_child: RigidPose) -> RigidPose:
    rotated = _rotate(parent_from_middle.quaternion_wxyz, middle_from_child.position)
    position = tuple(
        parent_from_middle.position[index] + rotated[index] for index in range(3)
    )
    quaternion = _quaternion_multiply(
        parent_from_middle.quaternion_wxyz,
        middle_from_child.quaternion_wxyz,
    )
    return RigidPose(position, _normalized_quaternion(quaternion))


def inverse(pose: RigidPose) -> RigidPose:
    w, x, y, z = pose.quaternion_wxyz
    quaternion = (w, -x, -y, -z)
    negative_position = tuple(-value for value in pose.position)
    return RigidPose(_rotate(quaternion, negative_position), quaternion)


def _scene_transform_context(
    scene_path: Path,
) -> tuple[ElementTree.Element, str]:
    root = ElementTree.parse(scene_path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"MuJoCo scene has no worldbody: {scene_path}")
    compiler = root.find("compiler")
    angle_unit = compiler.get("angle", "degree") if compiler is not None else "degree"
    euler_sequence = compiler.get("eulerseq", "xyz") if compiler is not None else "xyz"
    if angle_unit not in {"degree", "radian"} or euler_sequence != "xyz":
        raise ValueError("Task1 scene parser requires xyz Euler angles in degrees or radians")
    return worldbody, angle_unit


def _named_direct_body(worldbody: ElementTree.Element, name: str) -> ElementTree.Element:
    matches = [body for body in worldbody.findall("body") if body.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"MuJoCo scene must contain one direct world body named {name!r}")
    return matches[0]


def _body_pose(body: ElementTree.Element, *, angle_unit: str) -> RigidPose:
    position = _pose_float_tuple(
        body.get("pos", "0 0 0"), 3, f"{body.get('name')}.pos"
    )
    unsupported = {
        name for name in ("axisangle", "xyaxes", "zaxis") if body.get(name) is not None
    }
    if unsupported:
        raise ValueError(
            f"Task1 scene body {body.get('name')!r} uses unsupported orientation "
            f"{sorted(unsupported)}"
        )
    if body.get("quat") is not None:
        quaternion = _pose_float_tuple(
            body.get("quat", ""), 4, f"{body.get('name')}.quat"
        )
    else:
        euler = _pose_float_tuple(
            body.get("euler", "0 0 0"), 3, f"{body.get('name')}.euler"
        )
        quaternion = _euler_xyz_to_quaternion(euler, degrees=angle_unit == "degree")
    return RigidPose(position, _normalized_quaternion(quaternion))


def _pose_float_tuple(value: str, length: int, field: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item) for item in value.split())
    except ValueError as exc:
        raise ValueError(f"MuJoCo {field} must contain numbers") from exc
    if len(parsed) != length or not all(math.isfinite(item) for item in parsed):
        raise ValueError(f"MuJoCo {field} must contain {length} finite numbers")
    return parsed


def _euler_xyz_to_quaternion(
    euler: tuple[float, float, float], *, degrees: bool
) -> tuple[float, float, float, float]:
    angles = tuple((math.radians(value) if degrees else value) / 2.0 for value in euler)
    qx = (math.cos(angles[0]), math.sin(angles[0]), 0.0, 0.0)
    qy = (math.cos(angles[1]), 0.0, math.sin(angles[1]), 0.0)
    qz = (math.cos(angles[2]), 0.0, 0.0, math.sin(angles[2]))
    return _normalized_quaternion(
        _quaternion_multiply(_quaternion_multiply(qx, qy), qz)
    )


def _quaternion_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _rotate(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    rotated = _quaternion_multiply(
        _quaternion_multiply(quaternion, (0.0, *vector)),
        (quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]),
    )
    return rotated[1], rotated[2], rotated[3]


def _normalized_quaternion(
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError("pose quaternion must be finite and non-zero")
    normalized = tuple(value / norm for value in quaternion)
    return normalized if normalized[0] >= 0.0 else tuple(-value for value in normalized)
