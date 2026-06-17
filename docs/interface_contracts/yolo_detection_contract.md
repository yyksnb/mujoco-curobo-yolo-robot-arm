# YOLO Detection Contract

Stage 3 YOLO detection canonical contract 是单个目标检测结果：

```json
{
  "object_id": "object_001",
  "class_name": "mock_cube",
  "confidence": 0.97,
  "bbox_xyxy": [120.0, 80.0, 220.0, 180.0],
  "T_world_object": [
    [1.0, 0.0, 0.0, 0.45],
    [0.0, 1.0, 0.0, 0.05],
    [0.0, 0.0, 1.0, 0.08],
    [0.0, 0.0, 0.0, 1.0]
  ]
}
```

## 字段

- `object_id`: 本次场景中的物体唯一 ID。
- `class_name`: YOLO 类别名。
- `confidence`: 检测置信度，建议 0 到 1。
- `bbox_xyxy`: 图像坐标检测框 `[x1, y1, x2, y2]`。
- `T_world_object`: 4x4 齐次矩阵，表示 object frame 到 world frame 的位姿。

## 关键约束

`T_world_object` 是规划必需输入。`bbox_xyxy` 只能说明图像中的 2D 框，不能直接用于 cuRobo 规划。

如果 YOLO raw output 只有 bbox：

1. 先保留 raw output。
2. 用 adapter 转 Stage 3 contract。
3. 从 MuJoCo ground truth、位姿估计或显式 mock pose 补齐 `T_world_object`。

缺少 `T_world_object` 且未允许 mock pose 时，adapter 应报错：

```text
T_world_object is required by the Stage 3 planning contract. bbox_xyxy alone is not enough for cuRobo planning.
```

