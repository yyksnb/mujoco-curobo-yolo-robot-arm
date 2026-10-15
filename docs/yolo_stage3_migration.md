# YOLO Stage3 Migration

本仓库不训练 YOLO，也不把训练目录作为运行时依赖。YOLO 训练结果迁移到本工程约定位置，然后通过 profile 转成 Stage3 YOLO detection contract。

## 迁移约定

每次交付新的 YOLO 结果时，确认交付物里至少包含：

- YOLO 推理权重，例如 `best.pt`
- 类别列表或 dataset yaml，必须能确认类别 ID 到类别名的顺序
- 一份 raw detection 输出样例，说明 bbox 格式和置信度字段名

迁移到本工程：

- 权重放到 `outputs/yolo_models/current/best.pt`
- dataset yaml 或类别说明放到 `outputs/yolo_models/current/dataset.yaml`
- raw detection 样例放到 `outputs/yolo_raw/`

## Profile

默认 profile 是：

```text
configs/yolo/stage3_default.yaml
```

profile 负责声明：

- 工程内权重位置
- 推理默认参数
- YOLO 类别 ID / 类别名到 Stage3 `class_name` 和 `object_id` 的映射
- 可用的 mock 字段默认值

## 转 Stage3 Detection

raw YOLO 输出可以使用中文类别名、类别 ID、`bbox_xyxy`、`bbox_xywh` 或 COCO
风格 `bbox`：

```json
{
  "image_path": "outputs/images/scan_0001.png",
  "camera_name": "wrist",
  "detections": [
    {
      "class_id": 5,
      "confidence": 0.91,
      "bbox_xyxy": [120.0, 80.0, 220.0, 180.0]
    }
  ]
}
```

转换命令：

```bash
python scripts/convert_yolo_profile_raw_to_stage3_contract.py \
  --raw outputs/yolo_raw/scan_0001.json \
  --output outputs/yolo_stage3/scan_0001_detection.json \
  --config configs/yolo/stage3_default.yaml \
  --allow-mock-fields
```

`--allow-mock-fields` 只用于还不消费 `T_world_object` 的场景。进入规划前，`T_world_object` 必须由深度、位姿估计或 MuJoCo ground truth 补齐，并清楚标注来源。

## 本地 YOLO 推理

如果环境已安装 `ultralytics`，可以直接用 profile 里的权重位置做本地推理，并保存 raw detection JSON：

```bash
conda run -n mujoco-curobo python scripts/run_yolo_profile_inference.py \
  --image outputs/images/scan_0001.png \
  --output outputs/yolo_raw/scan_0001.json \
  --config configs/yolo/stage3_default.yaml \
  --camera-name wrist \
  --overwrite
```

也可以同时导出一个 Stage3 detection。由于 YOLO 推理本身没有 6D 位姿，识别前几步可显式允许 mock pose：

```bash
conda run -n mujoco-curobo python scripts/run_yolo_profile_inference.py \
  --image outputs/images/scan_0001.png \
  --output outputs/yolo_raw/scan_0001.json \
  --stage3-output outputs/yolo_stage3/scan_0001_detection.json \
  --allow-mock-fields \
  --overwrite
```
