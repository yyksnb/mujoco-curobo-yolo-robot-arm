from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from robot_arm_pipeline.types import GraspTarget, ObjectPose, Pose3D, TransformMatrix


def fake_bodex_grasp_targets(object_poses: tuple[ObjectPose, ...]) -> tuple[GraspTarget, ...]:
    """Return deterministic Stage 1 grasp targets without calling BODex."""
    targets: list[GraspTarget] = []
    for object_pose in object_poses:
        x, y, z = object_pose.pose.position
        targets.append(
            GraspTarget(
                object_id=object_pose.object_id,
                pose=Pose3D(position=(x, y, z + 0.12), orientation_xyzw=object_pose.pose.orientation_xyzw),
                approach_vector=(0.0, 0.0, -1.0),
                gripper_width_m=0.06,
            )
        )
    return tuple(targets)


def load_bodex_grasp_target(path: Path | str) -> GraspTarget:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    T_world_pregrasp = _transform_matrix(_required(payload, "T_world_pregrasp"), "T_world_pregrasp")
    T_world_grasp = _transform_matrix(_required(payload, "T_world_grasp"), "T_world_grasp")
    approach_vector = _float_tuple(payload, "approach_vector_world", 3)
    hand_joint_goal = _float_tuple(payload, "hand_joint_goal", None)
    return GraspTarget(
        object_id=_required_str(payload, "object_id"),
        pose=Pose3D(position=_translation(T_world_pregrasp)),
        approach_vector=approach_vector,
        gripper_width_m=0.06,
        T_world_pregrasp=T_world_pregrasp,
        T_world_grasp=T_world_grasp,
        hand_joint_goal=hand_joint_goal,
    )


def _required(payload: dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError(f"BODex grasp target is missing required field: {key}")
    return payload[key]


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = _required(payload, key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"BODex grasp target field {key} must be a non-empty string")
    return value


def _float_tuple(payload: dict[str, Any], key: str, length: int | None) -> tuple[float, ...]:
    value = _required(payload, key)
    if not isinstance(value, list):
        raise ValueError(f"BODex grasp target field {key} must be a list of numbers")
    if length is not None and len(value) != length:
        raise ValueError(f"BODex grasp target field {key} must be a list of {length} numbers")
    return tuple(float(item) for item in value)


def _transform_matrix(value: Any, key: str) -> TransformMatrix:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"BODex grasp target field {key} must be a 4x4 matrix")
    rows: list[tuple[float, float, float, float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError(f"BODex grasp target field {key} must be a 4x4 matrix")
        rows.append(tuple(float(item) for item in row))
    return tuple(rows)  # type: ignore[return-value]


def _translation(matrix: TransformMatrix) -> tuple[float, float, float]:
    return (matrix[0][3], matrix[1][3], matrix[2][3])
