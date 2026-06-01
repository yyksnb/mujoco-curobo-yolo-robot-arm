from __future__ import annotations

import math
from collections.abc import Sequence

from robot_arm_pipeline.types import CollisionScene, PlannedTrajectory, TrajectoryWaypoint, TransformMatrix


def transform_to_curobo_pose(transform: TransformMatrix) -> list[float]:
    _validate_transform(transform)
    qw, qx, qy, qz = _rotation_matrix_to_quaternion_wxyz(transform)
    return [
        float(transform[0][3]),
        float(transform[1][3]),
        float(transform[2][3]),
        qw,
        qx,
        qy,
        qz,
    ]


def collision_scene_to_curobo_world_config(scene: CollisionScene) -> dict[str, object]:
    cuboids: dict[str, dict[str, object]] = {}
    for collision_object in scene.objects:
        pose = collision_object.pose
        qx, qy, qz, qw = pose.orientation_xyzw
        cuboids[collision_object.object_id] = {
            "dims": [float(value) for value in collision_object.size_m],
            "pose": [
                float(pose.position[0]),
                float(pose.position[1]),
                float(pose.position[2]),
                float(qw),
                float(qx),
                float(qy),
                float(qz),
            ],
            "label": collision_object.label,
        }
    return {"cuboid": cuboids}


def joint_trajectory_to_planned_trajectory(
    *,
    joint_names: Sequence[str],
    joint_positions: Sequence[Sequence[float]],
    target_object_id: str,
    interpolation_dt: float,
    planner_name: str = "curobo_planner",
) -> PlannedTrajectory:
    if interpolation_dt <= 0:
        raise ValueError("interpolation_dt must be positive")
    if not joint_names:
        raise ValueError("joint_names must not be empty")

    names = tuple(str(name) for name in joint_names)
    waypoints: list[TrajectoryWaypoint] = []
    for index, positions in enumerate(joint_positions):
        if len(positions) != len(names):
            raise ValueError("each joint trajectory row must match joint_names length")
        waypoints.append(
            TrajectoryWaypoint(
                time_s=round(index * interpolation_dt, 6),
                joint_positions=tuple(float(value) for value in positions),
            )
        )
    if not waypoints:
        raise ValueError("joint_positions must contain at least one waypoint")

    return PlannedTrajectory(
        joint_names=names,
        waypoints=tuple(waypoints),
        planner_name=planner_name,
        target_object_id=target_object_id,
    )


def _validate_transform(transform: TransformMatrix) -> None:
    if len(transform) != 4:
        raise ValueError("transform must be a 4x4 matrix")
    for row in transform:
        if len(row) != 4:
            raise ValueError("transform must be a 4x4 matrix")
    if tuple(float(value) for value in transform[3]) != (0.0, 0.0, 0.0, 1.0):
        raise ValueError("transform bottom row must be [0, 0, 0, 1]")


def _rotation_matrix_to_quaternion_wxyz(transform: TransformMatrix) -> tuple[float, float, float, float]:
    r00, r01, r02 = transform[0][0], transform[0][1], transform[0][2]
    r10, r11, r12 = transform[1][0], transform[1][1], transform[1][2]
    r20, r21, r22 = transform[2][0], transform[2][1], transform[2][2]
    trace = r00 + r11 + r22

    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (r21 - r12) / scale
        qy = (r02 - r20) / scale
        qz = (r10 - r01) / scale
    elif r00 > r11 and r00 > r22:
        scale = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        qw = (r21 - r12) / scale
        qx = 0.25 * scale
        qy = (r01 + r10) / scale
        qz = (r02 + r20) / scale
    elif r11 > r22:
        scale = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        qw = (r02 - r20) / scale
        qx = (r01 + r10) / scale
        qy = 0.25 * scale
        qz = (r12 + r21) / scale
    else:
        scale = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        qw = (r10 - r01) / scale
        qx = (r02 + r20) / scale
        qy = (r12 + r21) / scale
        qz = 0.25 * scale

    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    return (qw / norm, qx / norm, qy / norm, qz / norm)
