# Data Format Policy

## Raw format

Raw format 是各组自己工具的原始输出。例如：

- YOLO 推理脚本输出的检测框
- 物体资产整理表
- MuJoCo 场景配置草稿
- BODex 原始抓取结果

raw format 可以存在，但不能直接进入主流程。

## Canonical contract

canonical contract 是本仓库主流程维护的标准格式。当前必须复用 Stage 3 contract：

- `examples/yolo_detection.json`
- `examples/bodex_grasp_target.json`
- `src/robot_arm_pipeline/types.py`

主流程只消费 canonical contract。

## Adapter / converter

如果 raw format 和 canonical contract 不一致，应新增 adapter/converter：

```text
raw format -> adapter/converter -> canonical contract
```

不要为了适配某个 raw format 直接改主流程字段。

## 缺字段策略

- 必需字段缺失时要清晰报错。
- YOLO 只有 `bbox_xyxy` 时不能进入 cuRobo planning。
- `T_world_object` 缺失时，应由 MuJoCo ground truth、位姿估计模块或显式 mock pose 补齐。
- mock/stub 必须明确标注，不能伪装成真实识别或真实规划。

## 字段映射

每个 adapter 应说明：

- 输入字段
- 输出字段
- 默认值
- mock/stub 字段
- 失败条件

adapter 的测试必须覆盖成功和失败路径。

