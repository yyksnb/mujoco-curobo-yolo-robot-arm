from robot_arm_pipeline.types import ObjectPose, Pose3D


def fake_yolo_object_poses() -> tuple[ObjectPose, ...]:
    """Return deterministic Stage 1 object poses without calling YOLO."""
    return (
        ObjectPose(
            object_id="object_001",
            label="mock_cube",
            pose=Pose3D(position=(0.45, 0.05, 0.08)),
        ),
    )

