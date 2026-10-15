#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from task2.config import load_task2_config


def main() -> int:
    import vamp

    version = importlib.metadata.version("vamp-planner")
    if version != "0.6.4":
        raise SystemExit(f"expected vamp-planner 0.6.4, found {version}")
    if tuple(vamp.robots) != ("gen3so101activewrist",):
        raise SystemExit(f"unexpected patched VAMP robot modules: {vamp.robots!r}")
    robot = vamp.gen3so101activewrist
    config = load_task2_config(REPOSITORY_ROOT / "configs/task2/pipeline.yaml")
    profile = config.end_effectors[0]
    model = config.robot_model(profile)
    vamp_profile = config.planning_backends[profile.planning_backend_profile].vamp
    start = list(model.default_joint_positions)
    collision_spheres = robot.fk(start)
    if len(collision_spheres) != vamp_profile.robot_sphere_count:
        raise SystemExit(
            "patched VAMP collision sphere count does not match the formal RobotModel"
        )
    for calibration in vamp_profile.collision_sphere_calibrations:
        sphere = collision_spheres[calibration.sphere_index]
        actual_center = (float(sphere.x), float(sphere.y), float(sphere.z))
        if max(
            abs(actual - expected)
            for actual, expected in zip(
                actual_center, calibration.center_at_default_m
            )
        ) > 1e-6 or abs(float(sphere.r) - calibration.radius_m) > 1e-6:
            raise SystemExit(
                "patched VAMP collision sphere calibration does not match Planning YAML"
            )
    for group in vamp_profile.collision_sphere_groups:
        first = group.first_sphere
        last = first + group.sphere_count
        if min(item.r for item in collision_spheres[first:last]) <= group.sweep_radius_inflation_m:
            raise SystemExit(
                f"patched VAMP module lacks the full-stroke envelope for {group.group}"
            )
    environment = vamp.Environment()
    environment.add_sphere(vamp.Sphere([1.5, 1.5, 1.5], 0.1))
    environment.add_cuboid(vamp.Cuboid([1.5, -1.5, 1.5], [0.0, 0.0, 0.0], [0.1, 0.1, 0.1]))
    convex_vertices = [
        [1.4, 1.4, -1.4],
        [1.6, 1.4, -1.4],
        [1.6, 1.6, -1.4],
        [1.4, 1.6, -1.4],
        [1.4, 1.4, -1.2],
        [1.6, 1.4, -1.2],
        [1.6, 1.6, -1.2],
        [1.4, 1.6, -1.2],
    ]
    convex_faces = [
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ]
    convex = vamp.ConvexMesh(
        convex_vertices,
        convex_faces,
        [1.5, 1.5, -1.3],
        [0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1],
        0.0,
    )
    convex.name = "convex_smoke"
    environment.add_convex_mesh(convex)
    goal = [0.1, 0.15, 3.0, -2.15, 0.1, 0.85, 1.45, 0.12]
    settings = vamp.RRTCSettings()
    settings.max_iterations = 1000
    settings.max_samples = 2000
    settings.range = vamp_profile.range_rad
    settings.dynamic_domain = vamp_profile.dynamic_domain
    rng = robot.halton()
    rng.skip(7)
    result = robot.rrtc(start, goal, environment, settings, rng)
    if not result.solved or len(result.path) < 2:
        raise SystemExit("Gen3 RRTC smoke did not solve")
    sphere = collision_spheres[0]
    half_extent = max(0.01, sphere.r * 0.5)
    center = [sphere.x, sphere.y, sphere.z]
    blocking_vertices = [
        [
            center[0] + x_sign * half_extent,
            center[1] + y_sign * half_extent,
            center[2] + z_sign * half_extent,
        ]
        for x_sign, y_sign, z_sign in (
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
        )
    ]
    blocking_environment = vamp.Environment()
    blocking_environment.add_convex_mesh(
        vamp.ConvexMesh(
            blocking_vertices,
            convex_faces,
            center,
            [0.0, 0.0, 0.0],
            [half_extent, half_extent, half_extent],
            0.0,
        )
    )
    if robot.validate(start, blocking_environment, True):
        raise SystemExit("native ConvexMesh did not block an intersecting robot sphere")
    print(
        json.dumps(
            {
                "version": version,
                "robot": vamp.robots[0],
                "collision_spheres": len(collision_spheres),
                "full_stroke_collision": True,
                "rrtc_solved": True,
                "path_knots": len(result.path),
                "native_environment": type(environment).__module__.startswith("vamp._core"),
                "native_convex_mesh": len(environment.convex_meshes) == 1,
                "convex_collision_detected": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
