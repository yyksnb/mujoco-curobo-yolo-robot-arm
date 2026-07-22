# Task2 Pipeline

本文档说明 Task2 动力学移除流水线的阶段设计、阶段输入输出和最小运行命令。

## 入口

统一入口：

```bash
python scripts/run_task2_pipeline.py \
  --manifest outputs/task1/<run>/final/object_observation_manifest.json \
  --end-effector robotiq_2f85 \
  --output outputs/task2/targeting_report.json
```

不传 `--end-effector` 时使用 `configs/task2/targeting.json` 中的默认执行器。
当前入口只执行已经实现的 Targeting；未实现的阶段不会生成占位成功结果。

完整设计流程：

```text
Task1 object observations
  -> Targeting
  -> Planning
  -> Execution
  -> Verification
```

## 准备 Targeting 服务

作用：安装本工程的轻量客户端，并启动仓库外的 GraspGenX 推理服务。

最小命令：

```bash
python -m pip install -e '.[targeting]'

cd ~/external/GraspGenX
.venv/bin/python client-server/graspgenx_server.py \
  --config ext/graspgenx_checkpoints/release \
  --assets_dir assets \
  --host 127.0.0.1 \
  --port 5556
```

## Targeting

状态：已实现。

作用：读取 Task1 的正式物体观测，根据目标末端执行器选择抓取算法，为每个合法
物体生成并排序抓取姿态候选。

输入：

- `object_observation_manifest.json`
- `configs/task2/targeting.json`
- `end_effector_id`

输出：

- `outputs/task2/targeting_report.json`

最小命令：

```bash
python scripts/run_task2_pipeline.py \
  --manifest outputs/task1/<run>/final/object_observation_manifest.json
```

## Planning

状态：待实现。

作用：将抓取候选转换为进入、闭合、保持和移出等分阶段运动计划，并检查 IK、
关节限位、连续碰撞、速度和加速度约束。

预期输入：Targeting 抓取候选、机器人与执行器配置、场景碰撞模型。

预期输出：可执行的分阶段轨迹及其负载和约束检查结果。

最小命令：暂无。实现后仍由统一 Task2 入口编排。

## Execution

状态：待实现。

作用：通过统一接口在 MuJoCo 或实机上闭环执行计划，处理夹持、接触、滑移和
控制误差，并把可观测状态和失败交给 Verification 与编排层。

预期输入：Planning 轨迹、执行器装配与控制配置、仿真或实机状态。

预期输出：逐阶段执行日志、状态观测和明确的执行失败。

最小命令：暂无。实现后仍由统一 Task2 入口编排。

## Verification

状态：待实现。

作用：验证物体能否保持 5 秒、没有不可接受的滑移或碰撞，并已成功移出目标
区域；失败时向编排层提出可追踪的恢复请求。

预期输入：Execution 日志、物体状态、接触和碰撞观测。

预期输出：动力学移除结果、失败原因及恢复请求。

最小命令：暂无。实现后仍由统一 Task2 入口编排。

## 测试

Task2 只保留一个测试入口：

```bash
python -m pytest -q tests/test_task2_pipeline.py
```

模块内部实现和边界见 [src/task2/README.md](../src/task2/README.md)。
