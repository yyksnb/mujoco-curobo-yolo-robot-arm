# Object Asset Contract

物体/场景准备组需要记录所有进入 MuJoCo 的物体资产信息。

## 必需信息

- `object_id`: 场景中的唯一物体 ID。
- `class_name`: 和 YOLO 类别一致的物体类别名。
- `asset_path`: mesh、MJCF 或 XML 资产路径。
- `mesh_format`: `obj`、`stl`、`dae`、`xml` 等。
- `size`: 物体尺寸，单位米。
- `scale`: 导入 MuJoCo 时的缩放。
- `mass`: 质量，单位 kg。
- `initial_pose`: 初始 world 位姿，建议用 4x4 transform 或 position + quaternion。

## 约束

- `object_id` 必须能和 YOLO detection、BODex grasp target 对齐。
- `class_name` 必须能和 YOLO 类别对齐。
- 尺寸和缩放要能用于 collision scene。
- 大型 mesh、纹理、模型包不要直接提交 Git，先确定 Git LFS 或外部存储方案。

## 交付建议

```json
{
  "object_id": "object_001",
  "class_name": "mock_cube",
  "asset_path": "examples/mujoco/assets/mock_cube.obj",
  "mesh_format": "obj",
  "size": [0.06, 0.06, 0.06],
  "scale": [1.0, 1.0, 1.0],
  "mass": 0.2,
  "initial_pose": [
    [1.0, 0.0, 0.0, 0.45],
    [0.0, 1.0, 0.0, 0.05],
    [0.0, 0.0, 1.0, 0.08],
    [0.0, 0.0, 0.0, 1.0]
  ]
}
```

