from robot_arm_pipeline.perception import fake_yolo_object_poses
from robot_arm_pipeline.scene import build_collision_scene


def test_collision_scene_can_be_constructed_from_object_poses() -> None:
    object_poses = fake_yolo_object_poses()

    scene = build_collision_scene(object_poses)

    assert scene.frame_id == "world"
    assert len(scene.objects) == len(object_poses)
    assert scene.objects[0].object_id == object_poses[0].object_id

