# Task2 产物驱动重构迁移与清理说明

本文档说明如何从当前三个功能 Commit 已完成的 Task2，迁移到
[task2_minimal_implementation_plan.md](task2_minimal_implementation_plan.md) 定义的
产物驱动总流水线。它只覆盖一次性源码/配置迁移和清理，不重新标定抓取参数，也不新增
规划或动力学策略。

## 1. 当前基线与迁移原则

当前基线：

```text
branch: feature/task2
HEAD:   f5a4e0e feat: [step3] execute planned trajectories
```

已完成：

- Commit 1：四物体 config-driven Targeting；
- Commit 2：VAMP + 可替换 IK + Ruckig Planning；
- Commit 3：忠实轨迹 Execution 与独立 MuJoCo 动力学诊断。

本轮不是第四次重写。必须冻结上述行为，从依赖图、公开入口和重复职责入手做减法。
`dd89f7a` 的 Task1 只提供“总入口明显、阶段产物清晰、脚本薄”的风格参照。

迁移规则：

- 先做 import、配置和调用方依赖审查，再移动或删除；
- 一次性改完仓库内调用方，不保留兼容层；
- 不新建分支，不操作暂存区，除非负责人明确授权；
- 不修改 Task1、Stage 1/2/3、MockPlanner 或其他正式共享行为；
- 不迁移 `outputs/`、cache、临时场景、扫描和视频；
- 不借重构修改候选、路径、碰撞、Ruckig 或 Execution 失败语义。

## 2. 迁移前冻结证据

实施前记录但不提交当前四物体中心场景基线：

- Targeting candidate ID、顺序、template/profile/family 和语义抓取命令；
- Planning 六阶段顺序、命名关节/执行器通道及稳定错误码；
- Execution 的 `execution_status`、事件通道、控制/日志采样和诊断字段；
- 四物体当前是否满足 Planning 成功及 `execution_status=completed`；
- 当前工作区、暂存区和相关配置/源码 diff。

冻结证据用于判断行为回归，不要求保存或比较精确 VAMP 路径、knot 数、耗时或 MuJoCo
浮点日志。生成报告放在被 Git 忽略的 `outputs/task2/`。

## 3. 配置一次性迁移

目标布局：

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

### 3.1 新增 pipeline.yaml

只记录：

- `objects.yaml`、`robot.yaml`、`planning.yaml`、`execution.yaml` 的相对路径；
- 可用执行器 profile 文件；
- 默认 Planning backend 与 Execution adapter 的显式选择；
- 通用候选、seed、阶段和总时间预算。

它不得复制机器人限位、物体模板、backend 参数或 MuJoCo 名称。

### 3.2 拆分 end_effectors.yaml

将当前 `configs/task2/end_effectors.yaml` 的唯一 profile 迁入
`end_effectors/active_wrist_parallel.yaml`：

- family 与 Targeting capabilities；
- 语义执行器通道和接触组；
- 主动腕、工具关节、TCP、限位和默认状态；
- Planning 使用的工具语义视图；
- 对 cuRobo/VAMP/MuJoCo 内部 profile 的引用。

MuJoCo actuator/geom 名称不进入通用 capability；它们迁往 `execution.yaml`。迁移完成后
删除旧 `end_effectors.yaml`，不保留双格式读取。

### 3.3 瘦身 robot.yaml

保留 Gen3 本体：

- model/base frame；
- 七个本体关节的唯一顺序、限位和默认状态。

移出：

- 主动腕关节和工具 TCP；
- cuRobo config asset；
- VAMP module、sphere count、tool fraction 和碰撞球索引组；
- 最大夹爪开度。

组合层使用基础机器人和选定执行器 profile 构造正式 RobotModel；cuRobo/VAMP 组合资产
仍可复用现有已验证文件，但路径只进入对应 backend 内部配置。

### 3.4 瘦身 objects.yaml

每个物体保留：

- Targeting 模板、局部变换、yaw 对称性、语义抓取命令和评分分项；
- Planning 球/胶囊/盒附着代理。

删除每个物体 `execution` 下的 MuJoCo body、free joint 和 collision geom 名称；这些绑定
迁入 `execution.yaml` 的 MuJoCo adapter section。不得修改四物体已标定数值、模板 ID、
family 或候选排序。

### 3.5 重组 planning.yaml

分为：

- 实现无关 `PlanningPolicy`；
- VAMP 内部 profile；
- cuRobo IK 内部 profile；
- Ruckig 内部 profile。

将版本、backend 名称、VAMP 搜索参数、cuRobo/VAMP 资产路径从正式 PlanningConfig/
RobotModel 移入内部 profile。版本检查和 backend 不可用语义保持不变。

### 3.6 重组 execution.yaml

保留通用控制频率、控制日志频率、诊断频率和诊断策略；在 MuJoCo adapter section 集中：

- model asset 和 TCP site；
- 机器人 runtime joint/actuator 绑定；
- 主动腕及夹爪 actuator 映射；
- 语义接触组到 geom 的映射；
- 四物体 runtime body/free joint/collision geom 绑定。

Execution 正式 contract 不得继续携带这些 MuJoCo 专用类型。

## 4. 类型与依赖迁移

### 4.1 Contracts

新增 `contracts/pipeline.py`，只定义：

- `Task2Request`；
- `Task2PipelineResult`；
- 单一 `PIPELINE_FAILED` 错误码。

移动：

- 抓取模板、yaw 对称性、Targeting diagnostics/profile 到
  `contracts/targeting.py`；
- Planning 物体代理和执行器视图到 `contracts/planning.py`；
- MuJoCo runtime 绑定到 `adapters/mujoco.py` 或紧邻的私有类型。

随后删除 `contracts/config.py`。`contracts/__init__.py` 只 re-export 正式 request、
result、error code 和跨模块值对象，不导出 YAML parser 中间类型。

### 4.2 RobotModel 与 PlanningPolicy

先让所有 backend 显式接收自己的内部 profile，再删除 RobotModel 中的后端字段，避免在
迁移中临时回读 YAML 或复制限位。确认 VAMP、cuRobo IK 和 Ruckig 仍消费同一个精简
RobotModel 的关节顺序与限位。

PlanningRequest 改为接收通用 PlanningPolicy 和必要的阶段视图。已有 Planning 错误码、
六阶段、碰撞语义、候选/seed 尝试和轨迹输出保持不变。

### 4.3 IK 边界

把 `src/task2/backends/ik.py` 的默认 backend factory 合并进 `src/task2/ik.py`，继续保留
公开 `IKSolver` Protocol、结果和错误语义。cuRobo 仍采用惰性 import；未知或不可用
backend 明确失败，不回退其他 IK。

### 4.4 配置加载边界

用较小的 `src/task2/config.py` 取代 909 行 `composition.py`：

- 顶层只解析路径、通用 profile 和职责引用；
- 各 backend/adapter 解析自己的内部 section；
- 所有路径相对 `pipeline.yaml` 解析一次；
- 阶段只接收类型化视图，不读取 YAML；
- 不通过当前工作目录或 `Path(__file__)` 猜工程根目录。

只校验会导致语义歧义、单位/frame 错误、危险命令或 backend 无法确定的字段。未知顶层
职责和关键控制字段仍拒绝；删除重复的全字段机械校验和配置 hash 扩散。

## 5. 总入口与 artifact 迁移

### 5.1 Pipeline

新增 `src/task2/pipeline.py`：

- `from_config(config_path, output_root)` 构建默认依赖；
- 构造函数可注入 Targeting、Planning、Execution 和 `EndEffectorRuntime`；
- `run(request)` 完成 setup、三个阶段和总结果；
- 每个阶段必须写结果并重新读取后再进入下一阶段；
- 阶段失败写正式失败结果并停止；
- `execution_status=completed` 决定总完成状态，动力学诊断不 gate 流程。

不要增加 `run_targeting()`、`run_planning()`、`run_execution()` 等第二套公开阶段 API。

### 5.2 Runtime provider

新增最小内部 runtime 边界，当前只接入 `active_wrist_parallel`：

- object family 在当前组合中必须唯一解析到一个 profile；
- 显式 profile 只用于约束，不允许静默替换；
- provider 激活工具后返回 RobotModel、完整状态和内部 executor factory；
- setup artifact 不序列化 adapter、MuJoCo model 或 Python 对象；
- 不引入 provider 注册平台或第二执行器的占位实现。

旧 switcher 中只迁移实际需要且可独立验证的 `MjSpec.attach`/公共状态迁移算法；不能迁移
旧 JSON schema、独立 registry 类型或兼容 API。

### 5.3 Artifact codec

新增单一 `src/task2/artifacts.py`，负责：

- 固定运行目录和相对路径；
- `request.json`、`setup.json`、三个阶段结果和 `result.json`；
- 正式结果的完整 JSON 编解码；
- 重跑阶段时删除当前 run dir 内的下游结果与日志。

Contracts 不增加公共 `from_dict()`。不要新增 Pydantic/msgspec 等序列化依赖，不加入
schema version、兼容读取、自动字段修复、hash 门禁或原子迁移框架。

### 5.4 CLI

恢复 `scripts/run_task2_pipeline.py`，保持薄适配：

```text
full/targeting: --request + --config + --output-root
planning/execution: --run-dir + --config
```

CLI 不构造临时场景、不补齐请求、不读取 MuJoCo 内部状态生成 Planning 输入，也不生成
视频。下游 step 只读取根请求、setup、直接上游结果和配置。

## 6. 明确删除清单

依赖迁移完成并用 `rg` 确认无正式调用方后删除：

```text
src/task2/composition.py
src/task2/contracts/config.py
src/task2/backends/ik.py
src/task2/adapters/hardware.py
src/robot_arm_pipeline/scene/end_effector.py
configs/task2/end_effectors.yaml
```

同步清理：

- `src/task2/adapters/__init__.py` 中的 hardware export；
- `src/robot_arm_pipeline/scene/__init__.py` 中的旧 switcher/registry export；
- 所有旧类型、旧配置路径和旧 loader import；
- 当前误名测试文件及只验证被删私有实现的测试。

不要删除仍由 Task1、Stage 1/2/3、MockPlanner、共享 scene/planner/executor 使用的其他
文件。若依赖扫描发现 `scene/end_effector.py` 仍有正式消费者，先停止删除并报告具体
调用链；不得复制旧实现或破坏消费者来满足清理清单。

## 7. 测试迁移

目标测试布局：

```text
tests/test_task2_targeting.py
tests/test_task2_planning.py
tests/test_task2_execution.py
tests/test_task2_pipeline.py
```

步骤：

1. 将当前只测试 Targeting 的 `test_task2_pipeline.py` 改名为
   `test_task2_targeting.py`；
2. 新增真正的 Pipeline/artifact 边界测试；
3. 保留稳定错误码、六阶段、命名通道、Execution 事件和日志频率已知回归；
4. 删除私有 parser/helper、文件布局和一次性迁移过程测试；
5. 固定用中心 `standard_part`、固定 seed 做一个真实快速 Planning 回归。

每个局部修改后运行相关 contract 测试和快速 Planning 回归。快速回归只守护 Planning
成功、六阶段和约束，不守护路径形状、knot 数或耗时。

## 8. 推荐实施顺序

所有改动最终组成一个独立重构 Commit；以下只是工作顺序，不要求产生中间提交：

1. 记录当前状态、四物体基线和相关 import graph；
2. 新增 pipeline 配置和 active-wrist-parallel profile，建立新 loader；
3. 迁移 RobotModel、PlanningPolicy 和阶段配置视图，保持三个阶段测试通过；
4. 合并 IK factory，更新 backend/adapter 构造路径；
5. 新增 Pipeline contracts、runtime provider、artifact codec 和总入口；
6. 恢复 CLI，验证每个 step 只能从正式 artifact 续跑；
7. 迁移 MuJoCo object/tool bindings，接通当前平行夹爪；
8. 改名并瘦身测试；
9. 删除旧配置、旧 switcher、composition、占位 hardware adapter 和重复 factory；
10. 运行局部测试、快速 Planning 回归和最终四物体完整验收；
11. 运行 `git diff --check`，检查工作区和暂存区，更新文档。

实现过程中若为了方便需要让某阶段读取其他阶段内部字段，必须停止并报告耦合风险；
正确处理是把数据加入正式上游 artifact 或根 Task2Request，而不是直接导入内部类型。

## 9. 最终验收与提交边界

最终验收必须证明：

- `task2` 根包存在唯一明显总入口和三个阶段入口；
- 四物体候选语义、顺序、抓取命令和稳定错误码未回归；
- 四物体 Planning 均生成完整六阶段轨迹；
- 四物体 Execution 均为 completed；
- 动力学诊断失败不改变总状态；
- 每个 step 只读取 `request.json`、`setup.json`、直接上游结果和组合配置；
- RobotModel/PlanningPolicy 不含 cuRobo、VAMP、Ruckig、MuJoCo 或夹爪专用字段；
- 默认路径无 fallback、cache replay、点位/距离/seed 特判；
- 删除清单无残余 import，旧配置无兼容读取；
- 全量测试、快速 Planning 回归和 `git diff --check` 通过；
- 暂存区未被操作，`outputs/` 与 cache 未进入提交。

建议提交信息：

```text
refactor(task2): add artifact-driven pipeline boundary
```

不要求 `dynamics_diagnostics.success=true`，也不要求 VAMP 路径、knot 数、耗时或 MuJoCo
日志逐字节一致。若四物体任一 Planning 或 Execution 回归失败，不得通过旧路径 fallback、
放宽碰撞或读取缓存制造通过；应报告失败阶段、错误码和最小根因。
