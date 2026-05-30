from robot_arm_pipeline.types import CollisionObject, CollisionScene, ObjectPose


DEFAULT_OBJECT_SIZE_M = (0.06, 0.06, 0.06)


def build_collision_scene(
    object_poses: tuple[ObjectPose, ...],
    *,
    frame_id: str = "world",
) -> CollisionScene:
    objects = tuple(
        CollisionObject(
            object_id=object_pose.object_id,
            label=object_pose.label,
            pose=object_pose.pose,
            size_m=DEFAULT_OBJECT_SIZE_M,
        )
        for object_pose in object_poses
    )
    return CollisionScene(frame_id=frame_id, objects=objects)

