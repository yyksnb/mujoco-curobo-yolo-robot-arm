#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from task2.composition import load_planning_config


def main() -> int:
    import vamp

    version = importlib.metadata.version("vamp-planner")
    if version != "0.6.4":
        raise SystemExit(f"expected vamp-planner 0.6.4, found {version}")
    if tuple(vamp.robots) != ("gen3so101activewrist",):
        raise SystemExit(f"unexpected patched VAMP robot modules: {vamp.robots!r}")
    robot = vamp.gen3so101activewrist
    composition = load_planning_config(
        REPOSITORY_ROOT / "configs/task2/robot.yaml",
        REPOSITORY_ROOT / "configs/task2/planning.yaml",
        REPOSITORY_ROOT / "configs/task2/objects.yaml",
        REPOSITORY_ROOT / "configs/task2/end_effectors.yaml",
    )
    model = composition.robot_model
    start = list(model.default_joint_positions)
    collision_spheres = robot.fk(start)
    if len(collision_spheres) != model.vamp_robot_sphere_count:
        raise SystemExit(
            "patched VAMP collision sphere count does not match the formal RobotModel"
        )
    for group in model.collision_sphere_groups:
        first = group.first_sphere
        last = first + group.sphere_count
        if min(item.r for item in collision_spheres[first:last]) <= group.sweep_radius_inflation_m:
            raise SystemExit(
                f"patched VAMP module lacks the full-stroke envelope for {group.group}"
            )
    environment = vamp.Environment()
    environment.add_sphere(vamp.Sphere([1.5, 1.5, 1.5], 0.1))
    environment.add_cuboid(vamp.Cuboid([1.5, -1.5, 1.5], [0.0, 0.0, 0.0], [0.1, 0.1, 0.1]))
    goal = [0.1, 0.15, 3.0, -2.15, 0.1, 0.85, 1.45, 0.12]
    settings = vamp.RRTCSettings()
    settings.max_iterations = 1000
    settings.max_samples = 2000
    settings.range = 0.4
    rng = robot.halton()
    rng.skip(7)
    result = robot.rrtc(start, goal, environment, settings, rng)
    if not result.solved or len(result.path) < 2:
        raise SystemExit("Gen3 RRTC smoke did not solve")
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
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
