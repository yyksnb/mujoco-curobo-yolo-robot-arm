# Task1 Pipeline

本文档说明了 Task1 识别流水线的正式流程、阶段输入输出和运行命令。

## 入口

统一入口：

```bash
python scripts/run_task1_recognition.py --seed 21 --output-dir outputs/task1
```

不传 `--step` 时会依次运行：layout -> survey -> rough -> final -> zoom。

输出目录：

```text
outputs/task1/<timestamp>_seed<seed>/
  layout/
  survey/
  rough/
  final/
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
python scripts/run_task1_recognition.py --step layout --seed 21 --output-dir outputs/task1
```

## Survey

作用：使用 wrist camera 做全局粗扫，得到候选物体的大致底面位置。

最小命令：

```bash
python scripts/run_task1_recognition.py \
  --step survey \
  --layout outputs/task1/<run>/layout/target_object_poses.json \
  --output-dir outputs/task1
```

## Rough

作用：消费 survey candidates，进行近距离粗拍和初步对象分层。

最小命令：

```bash
python scripts/run_task1_recognition.py \
  --step rough \
  --survey-report outputs/task1/<run>/survey/survey_report.json
```

## Final

作用：对 rough 输出的对象逐个进行单物体精拍摄，并产出稳定对象列表。

最小命令：

```bash
python scripts/run_task1_recognition.py \
  --step final \
  --rough-report outputs/task1/<run>/rough/rough_report.json
```

## Zoom

作用：对 final 精拍图片做数字变焦，裁剪并放大，使物体在输出图中尽量接近目标占比。

最小命令：

```bash
python scripts/run_task1_recognition.py \
  --step zoom \
  --final-report outputs/task1/<run>/final/final_report.json
```
