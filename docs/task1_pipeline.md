# Task1 Pipeline

本文档说明了 Task1 识别流水线的正式流程、阶段输入输出和运行命令。

## 入口

统一入口：

```bash
python scripts/run_task1_recognition.py --seed 21 --output-dir outputs/task1
```

不传 `--stage` 时会依次运行：layout -> survey -> rough -> final -> pose -> zoom。

全流程摘要分别给出 `recognition_status`、`grasp_readiness_status` 和 `artifact_status`；顶层 `status` 由识别与抓取位姿共同决定，Zoom 不影响识别成败。

输出目录：

```text
outputs/task1/<timestamp>_seed<seed>/
  layout/
  survey/
  rough/
  final/
  pose/
  zoom/
```

## Layout

作用：生成或读取目标物体初始摆放 JSON。

输入：

- `--seed`：根据 seed 生成 layout。

输出：

- `layout/target_object_poses.json`

最小命令：

```bash
python scripts/run_task1_recognition.py --stage layout --seed 21 --output-dir outputs/task1
```

## Survey

作用：使用 wrist camera 做全局粗扫，得到候选物体的大致底面位置。该阶段偏 recall-first，输出不是最终对象列表。

主要流程：

- 根据 layout 加载 MuJoCo 场景。
- 规划油箱内部 survey 视角。
- 拍摄 RGB/depth，并运行 YOLO。
- 融合多视角检测，输出候选 objects。

正式输出：

- `survey/survey_report.json`
- 每个 candidate 包含 `rough_position_world`、`supporting_views`、`best_image_path`、`best_bbox_xyxy`、`class_votes` 等字段。

最小命令：

```bash
python scripts/run_task1_recognition.py \
  --stage survey \
  --layout outputs/task1/<run>/layout/target_object_poses.json \
  --output-dir outputs/task1
```

## Rough

作用：消费 survey candidates，进行近距离粗拍和初步对象分层。

主要流程：

- 读取 `survey_report.json`。
- 对每个 survey candidate 规划近距离拍摄。
- 使用 YOLO + depth 重新确认、合并或拆分候选。
- 输出 stable / tentative / ambiguous 三层对象。

正式输出：

- `rough/rough_report.json`
- `stable_objects`：final 阶段主输入。
- `tentative_objects`、`ambiguous_objects`：final 阶段补确认目标。
- `object_selection_summary`：上游质量摘要。

最小命令：

```bash
python scripts/run_task1_recognition.py \
  --stage rough \
  --survey-report outputs/task1/<run>/survey/survey_report.json
```

## Final

作用：对 rough 输出的对象逐个进行单物体精拍摄，并产出稳定对象列表。

主要流程：

- 读取 `rough_report.json`。
- 以 `stable_objects` 作为 primary targets。
- 在稳定数量不足时，使用 `tentative_objects + ambiguous_objects` 作为 follow-up targets。
- 为每个目标规划多个精拍候选视角。
- 先按入口 portal、wrist roll、角度、距离和高度对候选排序，再依次做 IK/collision 验证；默认不硬截断候选。
- 通过 IK + collision validation 后选择最终拍照姿态。
- 使用近距离图像重新确认类别、框和稳定性。

正式输出：

- `final/final_plan.json`
- `final/final_report.json`
- `final/images/`
- `final/annotated/`：默认只画过滤后的 selected bbox。
- `stable_objects`：下游正式对象列表。
- `unstable_objects`：未满足稳定策略的对象。
- 每个稳定对象的 `pose_evidence`：选中 RGB-D、bbox、实际相机外参、FOV 和图像尺寸。

注意：

- wrist camera 来自 `examples/mujoco/kinova_gen3/gen3.xml` 的 `camera name="wrist"`。
- 最终拍照相机位姿必须在油箱内部，且 z 低于油箱上口高度。
- 正式 report 只保留 selected candidate、候选排序/可选裁剪统计摘要和少量失败样例；完整 trace 需要显式打开 debug。
- 当 primary confirmed 对象数量超过配置目标数量时，final 会优先把低置信单观测候选降为 `unstable_objects`。

最小命令：

```bash
python scripts/run_task1_recognition.py \
  --stage final \
  --rough-report outputs/task1/<run>/rough/rough_report.json
```

## Pose

作用：使用 Final 的单物体 RGB-D 证据与注册 CAD 模型，估计供抓取模块使用的规范物体位姿。

主要流程：

- 只读取 `final.stable_objects[*].pose_evidence`，不读取 layout 真值。
- 从 `configs/task1/pose_models.json` 获取类别、CAD frame 和 yaw 对称性。
- 对深度点云与 CAD 表面点云做多初值平面配准。
- 通过 fitness、双向 RMSE 和多解歧义策略筛选结果。

正式输出：

- `pose/pose_report.json`
- `grasp_ready_objects`：通过质量策略的 canonical `ObjectDetection` 兼容列表，`class_name` 使用 registry 规范类名。
- `unresolved_objects`：缺少证据、配准质量不足或姿态不可观测的对象。

注意：

- 当前 Task1 能力基于物体平放约束，精化 `x/y/yaw`，并沿用已声明的底面 body z。
- 对称物体输出代表位姿、`pose_symmetry` 和必要的竞争 `pose_hypotheses`；抓取模块不能把代表 yaw 当成唯一规范方向。

最小命令：

```bash
python scripts/run_task1_recognition.py \
  --stage pose \
  --final-report outputs/task1/<run>/final/final_report.json
```

## Zoom

作用：对 final 精拍图片做数字变焦，裁剪并放大，使物体在输出图中尽量接近目标占比。

主要流程：

- 读取 `final_report.json`。
- 只消费 `final.stable_objects`。
- 使用每个稳定对象的 `final_image_path` 和 bbox 生成 zoom 图片。
- 输出 selected zoom 和候选 zoom 评分。

正式输出：

- `zoom/zoom_report.json`
- `zoom/selected/zoom_<object_id>.png`

注意：

- Zoom 只处理图片，不改变对象身份、位姿或稳定性结论。

最小命令：

```bash
python scripts/run_task1_recognition.py \
  --stage zoom \
  --final-report outputs/task1/<run>/final/final_report.json
```

## 常用选项

```bash
--plan-only                 只写计划，不加载 MuJoCo 或 YOLO
--skip-yolo                 拍图但跳过 YOLO
--strict-yolo               YOLO 失败时直接让当前阶段失败
--save-debug-trace          保存支持 debug 的完整 trace
--save-raw-yolo-annotations 保存原始 YOLO 标注图
--pose-model-registry PATH  指定 Pose 使用的 CAD 与对称性 registry
```

最好显式指定 report；如果省略 `--survey-report`、`--rough-report` 或 `--final-report`，脚本会从 `--output-dir` 下寻找最新对应 report。

## 已知风险

- Final 阶段默认不硬截断候选会优先保 recall；类别冲突或无法确认的目标可能拖长耗时。
- Survey/Rough 对 `standard_part` 等小目标仍可能漏召回，导致后续 Final 没有目标可拍。
- 钻头和标准件等近邻小目标仍可能在 Rough 阶段进入同一个 ambiguous/fused 目标，Final follow-up 目前只能确认其中一个观测。
- Pose 当前只覆盖 Task1 平放物体的 `x/y/yaw`，不处理任意堆叠物体的完整 6D 姿态。
- 对称对象的 `T_world_object` 是代表解；下游必须消费 `pose_symmetry/pose_hypotheses`。若要求唯一 layout body frame，仍需语义关键点或非对称外观证据。
- 空候选的全流程 `--plan-only` 目前会因 Final 的既有状态语义得到顶层 `failed`，但各阶段仍会写出计划报告；正式非 plan-only 流程不受影响。
