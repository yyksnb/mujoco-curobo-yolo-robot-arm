# Task2 产物驱动流水线重构实现方案

本文档记录 Task2 在三个功能 Commit 已完成后的独立重构规格。当前基线为
`feature/task2` 分支上的 `f5a4e0e`：Targeting、Planning 和 Execution 已分别实现，
本轮不重写算法，也不改变正式数据语义。提交 `dd89f7a` 中的 Task1 只作为工程风格
基准，不作为文件数或代码量基准。

本轮目标是冻结行为，从公开入口、依赖方向、配置职责和阶段交接入手做减法：让用户
明确从哪里调用 Task2，并把 SO-101、VAMP、cuRobo 和 MuJoCo 的具体资产移出通用契约。

## 1. 冻结内容与完成定义

以下内容必须保持不变：

- `target()`、`plan()`、`execute()` 三个阶段入口；
- Targeting、Planning、Execution 的正式请求、结果、数据语义和已有稳定错误码；
- 确定性抓取候选、六阶段轨迹、按名称索引的关节及执行器通道；
- Execution 忠实执行 Planning 轨迹，不因 MuJoCo 动力学事实改道或提前停止；
- `execution_status` 与 `dynamics_diagnostics.success` 相互独立；
- 当前记号笔、纸胶带、铆枪、标准件和主动腕平行夹爪的已验证行为；
- 默认路径没有运行中 fallback，没有按点位、距离、seed 或候选索引特判。

本轮允许：

- 新增一个 Task2 总入口；
- 合并内部类型、移动文件和调整 YAML 内部布局；
- 一次性迁移仓库内配置和 import，删除旧路径且不保留兼容层；
- 删除没有正式调用方的占位实现、旧 switcher 和重复 factory。

本轮不包含：

- 吸盘、灵巧手或第二个可用执行器的实现与验收；
- 多物体调度、比赛顺序规划或自动更换另一执行器重试；
- 新的抓取姿态、路径选择、主动腕零空间或动力学诊断策略；
- 临时场景、扫描、视频、结果卡片或展示动作进入正式 pipeline；
- 运行时兼容读取、schema version、缓存回放或迁移脚本。

完成时，当前四个中心单物体场景必须继续完成 Targeting、Planning 和
`execution_status=completed`。动力学诊断成功不是本轮验收门槛。

## 2. 正式数据流

总流程固定为：

```text
Task2Request
  -> 执行器选择与激活
  -> setup.json
  -> target()
  -> targeting/result.json
  -> plan()
  -> planning/result.json
  -> execute()
  -> execution/result.json + 流式日志
  -> result.json
```

`Task2Pipeline.run()` 必须在每个阶段结束后写出正式结果，再从该结果反序列化构造下一
阶段输入。不能一边写旁路 JSON，一边通过内存对象绕过阶段边界。

所有阶段可以读取：

- 根目录只读 `request.json`；
- 顶层组合配置及其显式引用；
- 当前阶段的直接上游结果。

阶段不得读取更早阶段的私有状态、backend 对象、MuJoCo model/data 或临时脚本参数。
Targeting 不复制它不拥有的机器人状态和场景；Planning 由组合层使用根请求、setup 和
Targeting 结果构造正式 `PlanningRequest`；Execution 同理由 Planning 结果构造。

## 3. 公开接口

### 3.1 根包入口

`task2` 根包只提供明显的应用入口：

```python
from task2 import Task2Pipeline, Task2PipelineResult, Task2Request
from task2 import execute, plan, target
```

其他正式值对象从 `task2.contracts` 按需导入。`backends`、`adapters`、配置解析器、
artifact codec 和执行器 runtime provider 都是内部实现。

总入口固定为：

```python
Task2Pipeline.from_config(config_path, output_root, ...)
Task2Pipeline.run(request) -> Task2PipelineResult
```

`from_config()` 构建默认 backend、adapter 和执行器 runtime provider。构造函数允许测试
和高级调用者注入 Targeting、Planning、Execution 及 runtime provider 四个高层依赖，
但不把 VAMP、IK、Ruckig 的每个内部对象提升成总入口参数。运行中禁止 fallback。

Python API 要求显式传入组合配置和输出根目录；只有 CLI 可以提供仓库内默认路径。

### 3.2 Task2Request

总请求是类型化正式输入，至少包含：

- 稳定 `request_id`；
- 物体类型和有 frame 的世界位姿；
- 基础机械臂命名状态；
- `T_world_base`；
- `SceneSnapshot`；
- 目标物体 ID 与源支撑碰撞组；
- `PlacementGoal`；
- 单次运行 `random_seed`；
- 可选的 `end_effector_profile_id` 约束。

物体配置、机器人配置、backend 资产、adapter 绑定、规划预算和日志路径不进入请求。
候选/IK/规划预算属于部署配置；seed 属于单次运行输入。

### 3.3 Task2PipelineResult

总结果只聚合：

- 总状态；
- 已完成阶段；
- 失败阶段；
- 所选执行器 profile；
- 各阶段耗时；
- 各阶段结果文件的相对路径；
- 一个总入口错误码与消息。

总入口只新增一个稳定错误码 `PIPELINE_FAILED`，用于 setup、上游产物、组合配置或产物
IO 等流水线边界失败。阶段算法失败继续保留原阶段错误码，不包装为新的 Pipeline
错误分类。

Targeting 和 Planning 成功且 `execution_status=completed` 时，总状态为 completed。
`dynamics_diagnostics.success=false` 只进入 Execution 结果，不改变总状态。

## 4. 执行器选择与 runtime 边界

每个物体首版只对应一个 `end_effector_family`。组合配置中该 family 必须恰好解析到
一个 profile：零个或多个都返回 `PIPELINE_FAILED`。调用者可以显式指定 profile；指定
后只校验 family/capability 兼容性，不能静默换成另一个执行器。

完整运行的 setup 顺序为：

1. 根据物体模板解析唯一 family 和 profile；
2. 激活执行器；
3. 从 runtime provider 取得选定 `RobotModel` 和完整命名关节状态；
4. 写出 `setup.json`；
5. 开始 Targeting、Planning 和 Execution。

`setup.json` 只保存 profile ID、family、RobotModel ID 和激活后的完整命名关节状态。
具体 RobotModel、backend 和 adapter 配置仍从顶层配置重建，不复制到 artifact。

内部 `EndEffectorRuntime` Protocol 只负责准备/激活工具并返回内部 activated bundle。
该 bundle 包含选定 profile、RobotModel、完整初态和创建 `ExecutorAdapter` 的工厂；它不
负责 Targeting、Planning、重试或结果判定，也不能进入可序列化 artifact。

本轮只实现当前 `active_wrist_parallel` provider。吸盘等第二执行器以后通过新增 profile
和 provider 接入，并在独立功能 Commit 验收。不得因当前只有一个执行器而在正式
Pipeline 中写死 SO-101 资产名。

## 5. Artifact 契约

固定运行目录为：

```text
<output_root>/<request_id>/
  request.json
  setup.json
  targeting/result.json
  planning/result.json
  execution/result.json
  execution/control.csv
  execution/dynamics.jsonl
  result.json
```

规则：

- Targeting 结果完整保存候选、评分分项、抓取命令和位姿；
- Planning 结果完整保存六阶段轨迹、关键 knot、连续时间段、执行器通道、尝试及失败原因；
- Execution 结果完整保存状态、耗时、诊断和日志相对路径；
- `result.json` 只保存流程索引，不复制候选和轨迹；
- 失败阶段仍写正式失败结果并停止，不创建伪造的下游 `not_run` 结果；
- 重跑某阶段时覆盖该阶段结果，并删除当前运行目录中所有下游旧结果和日志；
- 不保存派生的 `targeting/request.json`、`planning/request.json` 或
  `execution/request.json`，因为根请求、setup、直接上游结果和配置足以确定性重建；
- 不加入 schema version、hash 驱动兼容、字段猜测、自动补齐或旧格式读取。

Contracts 保持类型化请求/结果，不公开通用 `from_dict()`。单一内部
`src/task2/artifacts.py` 负责 JSON 编解码和目录布局，Pipeline 与 CLI 共用。Pipeline
必须保证 artifact 能完整还原正式交接数据。

## 6. 配置与依赖方向

配置布局固定为：

```text
configs/task2/
  pipeline.yaml
  objects.yaml
  robot.yaml
  planning.yaml
  execution.yaml
  end_effectors/
    active_wrist_parallel.yaml
```

### 6.1 顶层组合配置

`pipeline.yaml` 只声明其他配置的显式路径、当前可用执行器 profile 和默认 backend/
adapter 选择。它不是第二份参数真源。

`Task2Pipeline.from_config()` 是唯一组合入口。顶层加载器解析引用和通用 profile；
VAMP、cuRobo IK、Ruckig 和 MuJoCo 分别解析自己的内部配置。backend/adapter 只接收
已经解析的路径和类型化内部配置，禁止通过 `Path(__file__)` 猜工程根目录或自行读取
其他职责 YAML。

### 6.2 RobotModel

正式 `RobotModel` 只保留：

- model/base/TCP frame；
- 唯一关节顺序；
- 关节位置、速度、加速度和 jerk 限位；
- 默认关节状态。

以下内容必须移出 RobotModel：

- cuRobo config asset；
- VAMP module、sphere count 和 tool fraction；
- VAMP 碰撞球索引组；
- 平行夹爪最大开度。

基础 `robot.yaml` 保存 Gen3 本体真源。主动腕关节、TCP、工具关节限位和 capability
属于 `active_wrist_parallel` profile，组合层将二者合成为 RobotModel。组合 URDF、
cuRobo YAML、VAMP module 和碰撞球布局分别放在对应 backend profile。

### 6.3 PlanningPolicy

正式 Planning 配置改为实现无关的 `PlanningPolicy`，只保留：

- IK 位置与朝向容差；
- 通用最大平移/旋转采样步长；
- lift、preplace 与关节裕量；
- 候选、seed、阶段和总时间预算。

VAMP/Ruckig 版本、IK backend 名称及 VAMP 搜索参数进入 `planning.yaml` 的 backend
内部 section。PlanningRequest 不知道当前使用 VAMP、cuRobo 或 Ruckig。

### 6.4 物体与 Execution 配置

`objects.yaml` 只保留与模拟器无关的：

- Targeting 抓取模板、对称性和语义命令；
- Planning 使用的球、胶囊和盒附着代理。

MuJoCo model asset、runtime body/free joint/geom、actuator、TCP site 和命令映射全部进入
`execution.yaml` 的 MuJoCo adapter section。Targeting、Planning 和 Execution 仍使用
各自职责视图，不能创建一个包含全部字段的巨型 EndEffectorModel，也不能把完整 YAML
字典传给阶段模块。

配置只校验会导致字段误解、单位/frame 歧义、危险执行或 backend 无法确定的内容。
未知顶层职责和关键控制字段必须拒绝；不再为每个普通嵌套 mapping 编写机械式全字段
防御校验。

## 7. 目标源码结构

重构后的职责结构建议为：

```text
src/task2/
  __init__.py
  pipeline.py
  artifacts.py
  config.py
  runtime.py
  targeting.py
  planning.py
  execution.py
  ik.py
  contracts/
    __init__.py
    geometry.py
    pipeline.py
    targeting.py
    planning.py
    execution.py
  backends/
    curobo_ik.py
    vamp.py
    ruckig.py
    errors.py
  adapters/
    mujoco.py
    errors.py
```

具体私有 helper 可以留在最接近的实现文件中；不要仅为缩短单个文件再拆无独立职责的
模块。`contracts/__init__.py` 只 re-export 正式请求、结果和跨模块值对象，不再全量
导出 YAML 解析中间类型。

抓取模板、yaw 对称性和 Targeting profile 类型归入 `contracts/targeting.py`；Planning
物体代理和执行器视图归入 `contracts/planning.py`；MuJoCo runtime 绑定归入 adapter
内部。Targeting、Planning、Execution 的执行器视图保持职责分离，不强行合并。

## 8. CLI

恢复唯一入口 `scripts/run_task2_pipeline.py`：

```text
--step full      --request <json> --config <yaml> --output-root <dir>
--step targeting --request <json> --config <yaml> --output-root <dir>
--step planning  --run-dir <dir>  --config <yaml>
--step execution --run-dir <dir>  --config <yaml>
```

`--request` 与 `--run-dir` 不能同时出现。开始新运行时，CLI/Pipeline 先执行 setup；后续
step 校验 `setup.json` 与当前已激活 profile 一致，不隐式选择或换装另一执行器。

CLI 只负责解析参数、调用正式 API 和输出机器可读摘要。临时场景工具负责构造正式
Task2Request；加载点、视频、可视化和结果卡片不得进入 CLI 或 Pipeline。

## 9. 测试与验收

测试按公开边界保留四类文件：

- `tests/test_task2_targeting.py`；
- `tests/test_task2_planning.py`；
- `tests/test_task2_execution.py`；
- `tests/test_task2_pipeline.py`。

当前误名的 `test_task2_pipeline.py` 先改名为 Targeting 测试，再为真实总入口新增
Pipeline 测试。保留：

- 正式 Request/Result、frame、单位和稳定错误码；
- Targeting 确定性、family/capability 匹配；
- 六阶段、命名通道、碰撞和停车语义；
- Execution 事件调度、控制/日志频率分离；
- artifact 只能经正式链路交接、失败停止和下游清理；
- 一个中心 `standard_part`、固定 seed 的真实快速 Planning 回归。

删除只验证私有 parser/helper、文件移动、一次性迁移过程和重复排列的测试。快速 Planning
回归只断言成功、六阶段完整和正式约束有效，不断言具体路径、knot 数或耗时。

最终验收重新运行四个中心单物体场景，要求：

1. 四个物体 Targeting 成功；
2. 四个物体 Planning 生成完整六阶段轨迹；
3. 四个物体 `execution_status=completed`；
4. 每个阶段只通过正式 JSON artifact 交接；
5. 候选语义、顺序、错误码和执行器通道保持一致；
6. 没有 fallback、cache replay 或点位/seed 特判。

不要求 VAMP 路径、knot 数、规划耗时或 MuJoCo 浮点日志逐字节相同。四物体完整验收
报告写入 Git 忽略的 `outputs/`，不提交视频或高频日志。

## 10. 提交范围

本轮作为三个功能 Commit 后的一个独立重构 Commit。建议提交信息：

```text
refactor(task2): add artifact-driven pipeline boundary
```

该 Commit 同时完成配置一次性迁移、总入口、artifact 链、依赖解耦、旧文件删除、测试
调整和本文档更新。不得执行 `git add` 或 commit，除非负责人明确授权。

明确禁止：

- 保留旧配置兼容读取或旧 switcher fallback；
- 因 family/profile 选择失败改试另一执行器；
- Planning 或 Execution 读取更早阶段的内部对象；
- 从 MuJoCo 反推 RobotModel、抓取模板或 Planning 代理；
- 用默认关节值冒充执行器激活后的实机状态；
- 为临时场景、点位、seed 或测试名称写正式代码特判；
- 为重构引入 provider 注册平台、schema 演进框架或新的序列化依赖。
