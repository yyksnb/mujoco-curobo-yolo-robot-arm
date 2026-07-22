# Task2 内部实现

流水线阶段设计和最小命令见
[docs/task2_targeting.md](../../docs/task2_targeting.md)。

## 模块边界

- `pipeline.py` 是 Task2 的编排层。当前只组合并调用 `TargetingStage`；Planning、
  Execution 和 Verification 在具备正式接口后再接入，不在这里堆放阶段实现。
- `targeting.py` 负责严格读取 Task1 正式观测、加载 Targeting 配置、选择末端执行器
  profile 和抓取算法、调用抓取服务、校验并汇总抓取候选。
- `configs/task2/targeting.json` 保存 backend 与末端执行器 profile 的绑定关系及算法
  参数。业务源码不根据类别、文件名、seed 或样例数据选择隐藏路径。
- `scripts/run_task2_pipeline.py` 是唯一流程入口；`tests/test_task2_pipeline.py` 是唯一
  Task2 测试入口。阶段模块不提供独立 CLI。

当前没有 `planning.py`、`execution.py` 或 `verification.py`。这些文件只会在对应
正式能力落地时创建，不使用空壳、mock 或占位成功结果表示未实现阶段。
正式阶段接入后，`pipeline.py` 负责状态机、有界重试、按恢复请求换执行器、逐物体
失败隔离和总报告；具体抓取、规划、控制及验证规则仍归各自模块所有。

## 正式输入边界

Targeting 只通过共享的 v3 loader 读取 Task1 Final 导出的
`object_observation_manifest.json`，使用每个观测的身份、类别和米制 world-frame
实例点云。它不读取 `final_report.json`、layout 真值、Task1 配置、Task1 内部规划
参数或 `backup/0722`。

Task1 不提供物体模型坐标系或精确 `T_world_object`。当前 GraspGenX 路径直接消费
world-frame 点云并返回同一 world frame 下的抓取姿态，不伪造物体 6D pose，也不
通过 ICP 或类别抓取库静默补齐结果。

## Targeting 执行顺序

1. 严格校验 Targeting 配置 schema，并解析选定的 `end_effector_id`。
2. 由末端执行器 profile 找到 backend，再通过显式注册的 factory 创建抓取生成器。
3. 按确定性空间顺序将物体点云限制到 backend 配置的最大输入点数。
4. 通过 GraspGenX msgpack/ZMQ 协议执行健康检查、读取服务 metadata 并请求推理。
5. 校验返回数量、score、branch tag 和每个 SE(3) 抓取姿态。
6. 在各生成器内部按 score 和 grasp ID 稳定排序，并为每个物体形成独立结果。
7. 原子写入 `task2_targeting_report/v3` 报告。

默认配置支持 `robotiq_2f85` 和 `robotiq_2f140`。两者使用同一个 GraspGenX
factory，但发送各自的 sweep-volume 参数。当前默认每次生成 8 个 diffusion 样本，
由 GraspMoE 合并 OBB 候选，并将输入限制为 3500 点。这些值位于配置文件，不是
类别或样例特判；末端参数来源于外部 GraspGenX checkout
`b9429097728cb1c430dd78b92edf17ba318aad03` 的 gripper description。

## 抓取算法与执行器扩展

配置关系为：

```text
end-effector profile -> backend_id -> backend type -> explicit factory
```

新增执行器时增加 profile，明确其 `grasp_frame_id`、能力要求和算法参数。新增抓取
算法时实现 `GraspGeneratorFactory`，并显式加入 `GraspGeneratorFactoryRegistry`。
不进行全局插件发现，也不让 Targeting 读取 MuJoCo 专用装配表。

不同算法的 score 只在各自声明的语义内排序，不跨算法比较绝对大小。规划/执行层
必须校验实际装配的 `end_effector_id`，并提供经过标定的
`T_grasp_frame_tcp`；Targeting 不猜测该变换。

## GraspGenX 协议与依赖

本仓库只包含 Torch-free 的客户端，GraspGenX、模型权重、Torch 和 CUDA 保留在
`~/external/GraspGenX` 的独立环境中。客户端不需要 ROS2。

- `pyzmq` 提供 localhost REQ/REP 传输。
- `msgpack` 提供请求和响应序列化。
- `msgpack-numpy` 编码点云，并解码服务端返回的 NumPy 位姿和置信度数组。

三项依赖通过 `targeting` extra 安装，并在第一次真实 wire request 时惰性导入；
导入 `task2.pipeline` 或只使用非 Targeting 模块不会加载它们。

服务没有 seed，因此候选姿态不保证逐位复现。这里的稳定性是指对象身份和顺序不
变、点云采样确定、协议响应严格校验、排序确定，以及单对象失败不影响其余对象。
服务 metadata 会进入报告，但当前协议不提供 checkpoint 哈希或服务 commit；部署
时仍需在仓库外固定 revision 和 checkpoint。

## 输出契约

- `T_world_grasp` 是 GraspGenX canonical gripper-base frame 在 Task1 world frame
  下的位姿。
- 当前 GraspGenX frame 中 `+X` 是闭合轴，`+Z` 是朝向物体的 approach 轴。
- 每个候选携带 `generator_id`、`end_effector_id`、`grasp_frame_id`、frame convention、
  approach vector、执行器能力要求、score 和 score semantics。
- Task1 Final 或 observation interface 为 `partial` 时，合法对象仍继续处理，源失败
  明细原样传播，Task2 顶层状态不能包装为完整成功。

## 失败与能力边界

服务不可用、超时、服务端错误、空候选、畸形数组或非刚体 SE(3) 都形成明确的
`grasp_generation` 失败。单对象失败与整批失败分开记录；当前没有几何抓取、缓存
结果或固定姿态 fallback。

Targeting 只输出抓取姿态候选，不执行 IK、连续碰撞、力闭合、负载检查、夹持、
搬运动力学或移除验证。这些能力分别属于后续 Planning、Execution 和 Verification
模块，不能以 Targeting 成功代替。

## 测试边界

`tests/test_task2_pipeline.py` 保留四类正式能力测试：执行器与算法路由、GraspGenX
姿态契约和重试语义、真实 msgpack/ZMQ wire protocol、Task1 partial 与逐对象失败
传播。不增加入口烟测、内部 helper 测试或依赖外部模型的慢速 E2E。
