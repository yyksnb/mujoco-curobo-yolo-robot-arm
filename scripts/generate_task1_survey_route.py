from __future__ import annotations

import json
import tempfile
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from robot_arm_pipeline.planning import CuroboCameraRoutePlanner, plan_camera_route  # noqa: E402
from robot_arm_pipeline.planning.curobo_camera_route import (  # noqa: E402
    DEFAULT_GRAPH_CONFIG,
    DEFAULT_ROBOT_CONFIG,
    DEFAULT_WORLD_CONFIG,
)
from task1.scene import RigidPose, load_tank_pose_in_base  # noqa: E402
from task1.survey.route import (  # noqa: E402
    DEFAULT_MUJOCO_SCENE_PATH,
    DEFAULT_SURVEY_ROUTE_PLAN_PATH,
    make_survey_route_targets,
    write_survey_route_plan,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    scene_path = _resolve(DEFAULT_MUJOCO_SCENE_PATH)
    output_path = _resolve(DEFAULT_SURVEY_ROUTE_PLAN_PATH)
    tank_pose_base = load_tank_pose_in_base(scene_path)
    targets = make_survey_route_targets(tank_pose_base)

    with tempfile.TemporaryDirectory(prefix="task1-survey-route-") as temp_dir:
        world_path = Path(temp_dir) / "world.yml"
        _write_scene_world_config(world_path, tank_pose_base)
        planner = CuroboCameraRoutePlanner(
            repo_root=REPO_ROOT,
            robot_config_path=DEFAULT_ROBOT_CONFIG,
            world_config_path=world_path,
            graph_config_path=DEFAULT_GRAPH_CONFIG,
        )
        plan = plan_camera_route(planner, targets)
        if not plan.success:
            raise RuntimeError(
                f"offline survey route generation failed at {plan.failed_target_id}: {plan.message}"
            )

        temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
        try:
            write_survey_route_plan(
                temporary_output,
                plan,
                repo_root=REPO_ROOT,
            )
            temporary_output.replace(output_path)
        finally:
            temporary_output.unlink(missing_ok=True)

    print(
        json.dumps(
            {
                "status": "success",
                "output": str(output_path),
                "segment_count": len(plan.segments),
                "planning_time_s": sum(segment.planning_time_s for segment in plan.segments),
            }
        )
    )


def _write_scene_world_config(path: Path, tank_pose_base: RigidPose) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to generate the cuRobo survey world") from exc
    source = _resolve(DEFAULT_WORLD_CONFIG)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    try:
        tank = payload["mesh"]["tank"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"survey cuRobo world has no mesh.tank entry: {source}") from exc
    tank["pose"] = [*tank_pose_base.position, *tank_pose_base.quaternion_wxyz]
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _resolve(path: Path) -> Path:
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


if __name__ == "__main__":
    main()
