from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


TARGET_BODY_PREFIX = "target_"
DEFAULT_MODEL_PATH = Path("examples/mujoco/gen3_with_tank.xml")
DEFAULT_OUTPUT_DIR = Path("outputs/object_poses")
DEFAULT_OBJECT_COUNT = 5
DEFAULT_BASE_HEIGHT_M = 0.03
DEFAULT_COLLISION_MARGIN_M = 0.012


@dataclass(frozen=True)
class PlacementBounds:
    x_min: float = 0.15
    x_max: float = 0.85
    y_min: float = 0.15
    y_max: float = 0.85


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
    created_utc: str | None = None,
    object_count: int = DEFAULT_OBJECT_COUNT,
    base_height_m: float = DEFAULT_BASE_HEIGHT_M,
    bounds: PlacementBounds = PlacementBounds(),
    collision_margin_m: float = DEFAULT_COLLISION_MARGIN_M,
    max_attempts_per_object: int = 6000,
) -> dict[str, Any]:
    specs = load_target_object_specs(model_path)
    selected_specs = select_target_object_specs(specs, object_count=object_count, seed=seed)
    poses = generate_random_target_object_poses(
        selected_specs,
        seed=seed,
        base_height_m=base_height_m,
        bounds=bounds,
        collision_margin_m=collision_margin_m,
        max_attempts_per_object=max_attempts_per_object,
    )
    poses_by_name = {pose.object_id: pose for pose in poses}
    ordered_poses = tuple(poses_by_name[spec.object_id] for spec in selected_specs)
    return {
        "schema_version": "target_object_pose_layout_v1",
        "created_utc": created_utc or datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "seed": seed,
        "object_count": object_count,
        "base_height_m": base_height_m,
        "placement_bounds": asdict(bounds),
        "collision_margin_m": collision_margin_m,
        "max_attempts_per_object": max_attempts_per_object,
        "target_body_prefix": TARGET_BODY_PREFIX,
        "notes": [
            "Object poses are generated from MuJoCo target_* mesh footprints.",
            "The script writes poses only; it does not modify MJCF/XML files.",
            "Collision checks use conservative 2D footprint polygons on the tank bottom plane.",
            "The z coordinate is controlled by base_height_m.",
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
    base_height_m: float,
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

            position = (_round(x), _round(y), _round(base_height_m))
            placed.append(
                TargetObjectPose(
                    object_id=spec.object_id,
                    class_name=spec.class_name,
                    position=position,
                    yaw_rad=_round(yaw_rad),
                    yaw_deg=_round(math.degrees(yaw_rad)),
                    quat_wxyz=_yaw_quaternion_wxyz(yaw_rad),
                    T_world_object=_make_yaw_transform(position, yaw_rad),
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


def _target_mesh_vertices_in_body_frame(mujoco, model, data, body_id: int, geom_id: int) -> np.ndarray:
    original_pos = np.asarray(model.body_pos[body_id], dtype=float).copy()
    original_quat = np.asarray(model.body_quat[body_id], dtype=float).copy()
    try:
        model.body_pos[body_id] = np.zeros(3, dtype=float)
        model.body_quat[body_id] = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=float)
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
        model.body_pos[body_id] = original_pos
        model.body_quat[body_id] = original_quat
        mujoco.mj_forward(model, data)


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


def _make_yaw_transform(
    position: tuple[float, float, float],
    yaw_rad: float,
) -> tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]:
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    x, y, z = position
    return (
        (_round(cos_yaw), _round(-sin_yaw), 0.0, x),
        (_round(sin_yaw), _round(cos_yaw), 0.0, y),
        (0.0, 0.0, 1.0, z),
        (0.0, 0.0, 0.0, 1.0),
    )


def _yaw_quaternion_wxyz(yaw_rad: float) -> tuple[float, float, float, float]:
    half_yaw = yaw_rad / 2.0
    return (_round(math.cos(half_yaw)), 0.0, 0.0, _round(math.sin(half_yaw)))


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
