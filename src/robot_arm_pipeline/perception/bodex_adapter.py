from robot_arm_pipeline.types import GraspTarget, ObjectPose, Pose3D


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

