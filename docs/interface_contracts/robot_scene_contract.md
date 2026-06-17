# Robot Scene Contract

机械臂/仿真器准备组需要提供稳定的机器人和相机字段，供规划控制组复用。

## 必需信息

- `robot_xml` 或 `robot_mjcf`: MuJoCo 机器人模型路径。
- `camera_name`: 拍照使用的 MuJoCo camera 名称。
- `camera_frame`: camera 坐标系定义或相对 world/robot 的变换。
- `base_link`: 机器人基座 link 名。
- `ee_link`: 末端执行器 link 名。
- `joint_names`: 规划和执行使用的关节顺序。

## 约束

- `joint_names` 必须和 MuJoCo、cuRobo robot config 的关节顺序可映射。
- `camera_name` 必须和 YOLO raw output 的 `camera_name` 对齐。
- camera frame 到 world frame 的变换必须可追踪，否则无法从图像检测补齐位姿。
- robot XML/MJCF 和 cuRobo robot config 不是同一个东西；cuRobo 还需要 collision spheres、自碰撞配置等。

## 交付建议

```json
{
  "robot_xml": "examples/mujoco/robot_scene.xml",
  "camera_name": "wrist_camera",
  "camera_frame": "robot_ee",
  "base_link": "base_link",
  "ee_link": "tool0",
  "joint_names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
}
```

