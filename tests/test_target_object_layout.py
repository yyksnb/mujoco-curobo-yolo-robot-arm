import math
from pathlib import Path

import numpy as np
import pytest

from robot_arm_pipeline.scene.target_object_layout import (
    PlacementBounds,
    TargetObjectSpec,
    convex_polygons_intersect,
    generate_random_target_object_poses,
    load_target_object_specs,
    polygon_within_bounds,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
GEN3_TANK_MODEL = REPO_ROOT / "examples" / "mujoco" / "gen3_with_tank.xml"


def test_random_target_layout_places_selected_objects_without_overlap() -> None:
    specs = (
        _box_spec("target_large", 0.16, 0.10),
        _box_spec("target_medium", 0.10, 0.08),
        _box_spec("target_small", 0.06, 0.05),
    )
    bounds = PlacementBounds(0.0, 0.5, 0.0, 0.5)

    poses = generate_random_target_object_poses(
        specs,
        seed=7,
        base_height_m=0.03,
        bounds=bounds,
        collision_margin_m=0.01,
    )

    assert len(poses) == 3
    assert {pose.object_id for pose in poses} == {"target_large", "target_medium", "target_small"}
    assert all(pose.position[2] == 0.03 for pose in poses)
    assert all(-math.pi <= pose.yaw_rad <= math.pi for pose in poses)
    polygons = [np.asarray(pose.footprint_polygon_xy, dtype=float) for pose in poses]
    assert all(polygon_within_bounds(polygon, bounds) for polygon in polygons)
    for index, polygon in enumerate(polygons):
        for other in polygons[index + 1 :]:
            assert convex_polygons_intersect(polygon, other, margin_m=0.01) is False


def test_load_target_specs_from_mujoco_model() -> None:
    pytest.importorskip("mujoco")

    specs = load_target_object_specs(GEN3_TANK_MODEL)

    assert len(specs) >= 5
    assert all(spec.object_id.startswith("target_") for spec in specs)
    assert all(spec.footprint_area_m2 > 0.0 for spec in specs)


def _box_spec(object_id: str, width: float, depth: float) -> TargetObjectSpec:
    half_width = width / 2.0
    half_depth = depth / 2.0
    return TargetObjectSpec(
        object_id=object_id,
        class_name=object_id.removeprefix("target_"),
        geom_name=f"{object_id}_geom",
        footprint_points_xy=(
            (-half_width, -half_depth),
            (half_width, -half_depth),
            (half_width, half_depth),
            (-half_width, half_depth),
        ),
        footprint_area_m2=width * depth,
        footprint_radius_m=float(math.hypot(half_width, half_depth)),
        local_min_z_m=0.0,
        local_max_z_m=0.02,
    )
