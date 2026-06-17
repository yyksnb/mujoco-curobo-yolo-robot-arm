# Integration Workflow

目标流程：

```text
物体进入 MuJoCo
-> 机械臂和摄像头进入 MuJoCo
-> 相机拍照
-> YOLO raw output
-> adapter 转 Stage 3 detection contract
-> 补齐 T_world_object
-> BODex/grasp target
-> collision scene
-> cuRobo planning
-> MuJoCo execution validation
```

## 1. 物体进入 MuJoCo

物体/场景组准备：

- mesh 或 MJCF asset
- `object_id`
- `class_name`
- 尺寸、缩放、质量
- 初始位姿

这些信息用于把物体真实位姿映射到 Stage 3 的 `T_world_object`。

## 2. 机械臂和摄像头进入 MuJoCo

机械臂/仿真组准备：

- robot XML/MJCF
- `camera_name`
- camera frame
- `base_link`
- `ee_link`
- `joint_names`

相机拍照输出需要带 `camera_name`，否则后续无法确认图片来自哪个坐标系。

## 3. YOLO raw output

YOLO 组可以先输出 raw format：

```json
{
  "image_path": "outputs/images/camera_0001.png",
  "camera_name": "wrist_camera",
  "detections": [
    {
      "object_id": "object_001",
      "class_name": "mock_cube",
      "confidence": 0.97,
      "bbox_xyxy": [120.0, 80.0, 220.0, 180.0]
    }
  ]
}
```

这个 raw output 不能直接进入 planning，因为缺少 `T_world_object`。

## 4. 转成 Stage 3 detection contract

使用 adapter/converter：

```bash
python scripts/convert_yolo_raw_to_stage3_contract.py \
  --raw examples/yolo_raw_output_sample.json \
  --output outputs/tmp_yolo_detection_converted.json \
  --allow-mock-pose
```

如果没有 `--allow-mock-pose` 且 raw 中没有 `T_world_object`，脚本必须报错。

## 5. 补齐 T_world_object

`T_world_object` 是 4x4 齐次矩阵。来源可以是：

- MuJoCo ground truth
- 标定/位姿估计模块
- 临时 mock pose

如果使用 mock pose，必须在 PR 和文档里明确标注。

## 6. BODex / grasp target

BODex 输出仍按 Stage 3 grasp target contract：

- `object_id`
- `T_world_pregrasp`
- `T_world_grasp`
- `approach_vector_world`
- `hand_joint_goal`

当前 planner 使用 `T_world_pregrasp` 作为目标。

## 7. Collision scene

规划控制组将 object pose 和资产尺寸转成 `CollisionScene`。raw YOLO bbox 不进入 collision scene。

## 8. cuRobo planning

cuRobo planner 只消费 canonical contract 和 robot/world config。当前仓库有 skeleton 和环境检查脚本，真实规划应在 Linux CUDA 环境验证。

## 9. MuJoCo execution validation

规划结果保存为项目内部 trajectory JSON，然后由 MuJoCo executor 或 demo 脚本执行验证。

