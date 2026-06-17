# START HERE / 团队协作入口

本仓库用于把上游识别、抓取目标、场景建图、运动规划、轨迹执行和 MuJoCo 验证串成一个可复现的机械臂仿真流程。

当前主流程已经有稳定的 Stage 3 contract：

```text
YOLO/BODex upstream raw data
-> adapter/converter
-> Stage 3 canonical contract
-> mock planner or cuRobo planner skeleton
-> MuJoCo executor / validation
```

## 每组先看什么

| 角色 | 成员 | 先看文档 | 重点 |
|---|---|---|---|
| YOLO 识别 | 王奥、刘帅彤 | `docs/interface_contracts/yolo_detection_contract.md` | raw YOLO 可以只有 bbox，但进主流程前必须补齐 `T_world_object` |
| 物体/场景 | 刘帅彤 | `docs/interface_contracts/object_asset_contract.md` | 记录物体模型路径、尺寸、缩放、质量、初始位姿 |
| 机械臂/仿真器 | 邓弘烨 | `docs/interface_contracts/robot_scene_contract.md` | 机器人 XML/MJCF、相机名、base/ee link、joint_names |
| 规划控制 | 何伟铭、邓弘烨、杨雅凯 | `docs/integration_workflow.md`、`docs/curobo_integration_notes.md` | 复用 Stage 3 contract，保持 planner/executor 接口稳定 |
| 所有人 | 全体 | `docs/team_git_flow.md`、`CONTRIBUTING.md` | 不直接 push main，开分支和 PR |

## Clone 和安装

```bash
git clone https://github.com/yyksnb/mujoco-curobo-yolo-robot-arm.git
cd mujoco-curobo-yolo-robot-arm
python -m pip install -r requirements.txt
```

## 建分支

不要直接改 `main`。每个任务从最新 `main` 新建分支：

```bash
git checkout main
git pull
git checkout -b feature/your-task-name
```

建议分支名：

- `feature/yolo-camera-capture`
- `feature/object-assets`
- `feature/robot-mujoco-scene`
- `feature/curobo-world-adapter`
- `docs/team-notes`
- `fix/stage3-adapter-validation`

## 跑测试

提交前至少运行：

```bash
pytest -q
```

如果只改 YOLO raw adapter，可额外运行：

```bash
python scripts/convert_yolo_raw_to_stage3_contract.py \
  --raw examples/yolo_raw_output_sample.json \
  --output outputs/tmp_yolo_detection_converted.json \
  --allow-mock-pose
```

`outputs/` 是本地生成产物，不要提交。

## 提 PR

```bash
git add <files>
git commit -m "clear message"
git push -u origin <your-branch>
```

然后在 GitHub 创建 Pull Request。PR 里写清：

- 改了什么
- 为什么改
- 跑了哪些命令
- 是否影响 Stage 1/2/3 pipeline
- 是否新增 raw format，是否已经写 adapter

## 数据格式原则

- 仓库维护 canonical contract，不直接把各组 raw format 接进主流程。
- raw format 不一致时，新建 adapter/converter。
- 不要改坏 `examples/yolo_detection.json` 和 `examples/bodex_grasp_target.json` 代表的 Stage 3 contract。
- YOLO 只有 `bbox_xyxy` 时不能直接做 cuRobo 规划。
- `T_world_object` 可先由 MuJoCo ground truth 或 mock pose 补齐，但 mock/stub 必须在文件或 PR 里明确标注。
- 不要提交模型权重、大数据、`outputs/`、`__pycache__/`、`.pytest_cache/`。

