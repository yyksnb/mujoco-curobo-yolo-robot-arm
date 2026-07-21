from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path

add_src_to_path()

from robot_arm_pipeline.planning.curobo_planner import (  # noqa: E402
    DEFAULT_GRAPH_CONFIG,
    DEFAULT_ROBOT_CONFIG,
    DEFAULT_WORLD_CONFIG,
    CuroboPlanner,
)
from robot_arm_pipeline.types import (  # noqa: E402
    GraspTarget,
    ObjectPose,
    PlanningRequest,
    Pose3D,
    RobotState,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stage 4.3 cuRobo MotionGen demo adapter.")
    parser.add_argument("--config", type=Path, default=Path("configs/curobo/minimal_motiongen_demo.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    started = time.perf_counter()
    config = _load_config(args.config)
    request = _build_request(config)
    planner = CuroboPlanner(
        repo_root=REPO_ROOT,
        robot_config_path=Path(config.get("robot_config_path", DEFAULT_ROBOT_CONFIG)),
        world_config_path=Path(config.get("world_config_path", DEFAULT_WORLD_CONFIG)),
        graph_config_path=Path(config.get("graph_config_path", DEFAULT_GRAPH_CONFIG)),
    )
    result = planner.plan(request)
    planning_time_s = round(time.perf_counter() - started, 6)
    report_path = _save_report(args.output_dir, args.config, config, result, planning_time_s)
    print(
        f"success={result.success} message={result.message} "
        f"planning_time_s={planning_time_s} report={report_path}"
    )


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_request(config: dict[str, Any]) -> PlanningRequest:
    joint_names = tuple(str(name) for name in config["joint_names"])
    start_joint_state = tuple(float(value) for value in config["start_joint_state"])
    if len(joint_names) != len(start_joint_state):
        raise ValueError("joint_names and start_joint_state must have the same length")

    goal_pose = tuple(float(value) for value in config["goal_pose"])
    if len(goal_pose) != 7:
        raise ValueError("goal_pose must be [x, y, z, qw, qx, qy, qz]")

    x, y, z, qw, qx, qy, qz = goal_pose
    pregrasp = _goal_pose_to_transform(goal_pose)
    pose = Pose3D(position=(x, y, z), orientation_xyzw=(qx, qy, qz, qw))
    return PlanningRequest(
        object_pose=ObjectPose(
            object_id="motiongen_demo_goal",
            label="motiongen_goal",
            pose=pose,
            T_world_object=pregrasp,
        ),
        grasp_target=GraspTarget(
            object_id="motiongen_demo_goal",
            pose=pose,
            approach_vector=(0.0, 0.0, -1.0),
            gripper_width_m=0.0,
            T_world_pregrasp=pregrasp,
            T_world_grasp=pregrasp,
        ),
        robot_state=RobotState(joint_names=joint_names, joint_positions=start_joint_state),
    )


def _goal_pose_to_transform(goal_pose: tuple[float, ...]) -> tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]:
    x, y, z, qw, qx, qy, qz = goal_pose
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), x),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), y),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), z),
        (0.0, 0.0, 0.0, 1.0),
    )


def _save_report(
    output_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    result: Any,
    planning_time_s: float,
) -> Path:
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "curobo_motiongen_demo_report.json"
    payload = {
        "success": result.success,
        "message": result.message,
        "planning_time_s": planning_time_s,
        "trajectory_available": result.trajectory is not None,
        "trajectory_path": None,
        "config_path": str(config_path),
        "robot_config_path": config.get("robot_config_path"),
        "world_config_path": config.get("world_config_path"),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
