# Task2 最小重构实现方案

本文档记录 Task2 最小重构的已确认设计。交付工程由负责人在当前工程外创建，
以提交 `60c7bef` 为起点，使用 `feature/task2` 分支。本文档所在的实验副本只用于
提取经过验证且来源明确的资产和参数，不作为源码迁移来源。

## 1. 完成定义

正式数据流固定为：

```text
TargetingRequest
  -> target()
  -> GraspCandidate[]
  -> plan()
  -> 六阶段时间参数化轨迹
  -> execute()
  -> ExecutionStatus + DynamicsDiagnostics
```

最小方案完成时必须满足：

- Targeting 能为记号笔、纸胶带、铆枪和标准件确定性生成抓取候选。
- Planning 使用 VAMP、可替换 `IKSolver` 和 Ruckig 生成完整六阶段轨迹。
- Execution 忠实执行 Planning 给出的轨迹，不根据 MuJoCo 动力学状态改道或提前中断。
- MuJoCo 动力学事实被记录并形成独立诊断；动力学任务失败不伪装成规划失败或轨迹执行失败。
- 油箱底部中心单物体场景中，四个核心物体分别完成规划与真实动力学移除验收。
- 默认路径只有一套规划策略，没有隐式 fallback，也没有按点位、距离、seed 或候选索引进行特判。

不属于首版范围：

- 学习式抓取生成、BODex、Dexonomy 和多 provider 注册系统。
- 主动腕零空间优化策略。主动腕首版只作为普通关节参与规划。
- 实机控制策略。只保留与 MuJoCo executor 共用的 adapter 接口。
- RH56DFX 等其他执行器的默认实现。
- 多半径扫描、五物体展示、视频生成和临时场景编排。

## 2. 模块边界

### 2.1 Contracts

`src/task2/contracts/` 是独立值对象层，只包含：

- 请求、结果和稳定错误码；
- frame 明确的刚体变换；
- 抓取候选、轨迹阶段、轨迹 knot 和执行器通道；
- 场景快照、放置目标、执行状态和动力学诊断结构；
- 序列化所需的纯数据转换。

该层不得包含配置文件访问、Targeting/Planning/Execution 算法、VAMP、cuRobo、
Ruckig、MuJoCo 或 CUDA 依赖。各阶段的 Request、Result 和 ErrorCode 在 contracts
中按阶段分文件归属，不能使用一个含义模糊的通用结果类型。

单位和坐标约定：

- 长度为米，角度为弧度，时间为秒，质量为千克。
- 四元数统一使用 `wxyz`。
- 位姿必须携带 `parent_frame` 和 `child_frame`，禁止匿名 4x4 矩阵跨模块传递。
- 加载器只检查会导致语义歧义或危险执行的内容，不自动补齐、裁剪、重排或猜测输入。
- 当前代码只支持唯一配置格式，不维护 schema version，也不承诺跨 commit 的 artifact 兼容。

### 2.2 组合与配置层

组合层是唯一可以打开 YAML 的正式运行时位置。它负责：

1. 读取机器人、执行器、物体和 Planning 配置；
2. 做必要的类型、单位、frame、有限数值、合法范围和跨字段不变量检查；
3. 把同一物体 YAML 切分成互不泄漏的类型化视图；
4. 构建唯一真源 `RobotModel`；
5. 将类型化依赖注入对应阶段。

物体 YAML 建议结构：

```yaml
objects:
  marker:
    targeting:
      # 抓取模板、对称性和先验评分参数
    planning:
      # 抓取后附着碰撞代理体
    execution:
      # 质量、质心、惯量、摩擦和 MuJoCo 资产绑定
```

虽然数据位于同一文件，Targeting 只能收到 `TargetingObjectConfig`，Planning 只能
收到 `PlanningObjectConfig`，Execution 只能收到 `ExecutionObjectConfig`。不得把
完整 YAML 字典或路径传给阶段模块。

建议配置文件：

```text
configs/task2/
  objects.yaml
  end_effectors.yaml
  robot.yaml
  planning.yaml
```

场景几何、机器人初态、物体世界位姿和放置目标是每次调用的正式输入，不写入默认
YAML。配置加载器不检查后端内部状态，也不进行重复的防御性校验。

### 2.3 Targeting

唯一公共入口：

```python
TargetingResult target(TargetingRequest request)
```

Targeting 只回答“这个执行器可以怎样抓这个物体”。输入包含：

- 稳定的 `object_type` 和有 frame 的物体世界位姿；
- 所选执行器的 family 与 capabilities；
- 已切分的 Targeting 物体/执行器配置；
- 全局确定性 yaw 采样分辨率和候选上限。

它不读取机器人关节状态、场景障碍、IK、VAMP、Ruckig、MuJoCo 或 Planning 诊断。

抓取 profile 使用“物体 + 执行器 family/capabilities”匹配。首版执行器 family 为
`parallel_gripper`，不能依赖资产文件名、具体关节名或 Python 类名。

每个模板至少定义：

- 稳定 `template_id`；
- 有 frame 的 `T_object_tcp` 和 `T_tcp_pregrasp`；
- 绕物体局部轴的离散对称群或允许 yaw 区间；
- 抓取开度、最大允许夹持力、闭合速度和驻留时间等语义命令；
- `left_pad`、`right_pad` 等语义接触集合；
- 模板优先级、接触裕量和开度裕量等可解释评分项。

首版只展开局部 yaw；数据结构可表达命名倾角模板，但不实现按世界点位或机器人姿态
自适应倾角。输出候选顺序必须确定。`candidate_id` 由 object type、profile、template
和 symmetry variant 稳定派生，不能包含世界点位、运行序号或随机 seed。

候选同时保留局部抓取变换和解析后的世界变换，并携带：

- `quality_score` 与 `score_components`，不得声称是未经标定的成功概率；
- 语义化 `grasp_command`；
- `expected_contacts`；
- 由已选抓取姿态确定的抓取后 `T_tcp_object`。

Targeting 的稳定错误码至少区分未知物体、执行器不兼容、配置无效、输入 pose/frame
无效和展开后无候选。它不能在未做 IK 或碰撞规划时宣布候选可达。

### 2.4 Planning

唯一公共入口：

```python
PlanningResult plan(PlanningRequest request)
```

`PlanningRequest` 显式包含：

- 有序 `GraspCandidate[]`；
- 有序机器人初始关节状态；
- 由唯一真源构建的 `RobotModel`；
- `SceneSnapshot`；
- `PlacementGoal`；
- 类型化 Planning 物体和执行器配置；
- seed、候选预算、阶段时间预算和总预算。

`SceneSnapshot` 支持带稳定 ID 和位姿的基本体、凸体、静态三角网格、碰撞组和允许
接触对。MuJoCo、URDF 或其他来源必须在 pipeline 外转换成该契约；Planner 不读取
MuJoCo 内部 model/data。

机器人模型是关节顺序、限位、运动学链和组合碰撞模型的唯一真源。VAMP adapter、
IK adapter 和 Ruckig 都消费该模型，禁止各自维护关节顺序或限位副本。

#### 路径算法

默认且唯一的策略为：

1. `PREGRASP`：VAMP 自由空间规划至预抓姿态。
2. `APPROACH`：按最大平移/旋转步长离散笛卡尔路径，由 Planning 逐点调用 IK 并维持连续分支。
3. `GRASP`：机械臂停车，按语义执行器通道闭合并驻留。
4. `LIFT`：连续 IK 生成受约束的脱离支撑路径。
5. `EGRESS`：VAMP 进行分段退出规划。
6. `PLACE`：VAMP 到油箱外允许区域中的确定性可行放置目标，停车后产生 release 事件。
7. 对拼接后的完整关节几何路径统一使用 Ruckig 时间参数化。

阶段是有时间范围的 `TrajectoryPhase`，不是六个离散 waypoint。`GRASP` 和
`PLACE/release` 边界强制速度归零，其他阶段边界尽量保持速度连续。

PlacementGoal 支持精确位姿和区域；首版使用矩形或圆形区域及高度范围。没有显式
朝向约束时不验收物体终点朝向，目标采样保持确定性，并选择第一个满足可达性、碰撞
和限位要求的目标。不得静默把区域目标改成区域中心。

#### IK 与路径选择

首版真实 IK 后端为隔离的 cuRobo adapter，但 Planning 只依赖：

```python
IKResult solve(targets, seed_states, constraints)
```

adapter 返回关节名、有序完整关节解、残差和稳定错误码；tensor、CUDA stream 和
cuRobo 内部对象不得越过边界。后端不可用时返回 `BACKEND_UNAVAILABLE`，不得回退
MotionGen、mock IK 或另一规划器。

连续 IK 的笛卡尔离散、分支连续性和关节限位裕量由通用 Planning 层负责。多个完整
路线使用固定字典序选择：

1. 六阶段均有效且无禁止碰撞；
2. Targeting 候选顺序；
3. 同一候选内关节路径更短；
4. 再比较关节限位裕量。

同一个 VAMP 后端允许在显式预算内尝试多个候选和多个 seed，这属于正常搜索而非
fallback。每次尝试的 seed、阶段、耗时和失败原因必须记录；预算耗尽即明确失败。

#### 碰撞语义

- `PREGRASP/APPROACH/GRASP` 使用场景中的原始目标物体碰撞体。
- `APPROACH/GRASP` 只允许候选声明的指垫语义组接触目标物体。
- 手掌、指节、机械臂与目标物体或油箱的碰撞始终禁止。
- 目标物体与油箱底面的已有支撑接触允许持续到 `LIFT`。
- `GRASP` 结束时在规划语义上移除静态目标物体，并按 `T_tcp_object` 附着简化代理体。
- `LIFT/EGRESS/PLACE` 使用物体局部坐标系中的球、胶囊和盒体组合代理。
- `PLACE` 末端停车前代理体始终保持附着；release 之后的动力学不再规划。

该阶段切换是显式规划语义，不依赖 MuJoCo 是否实际抓牢物体。执行器配置声明语义
碰撞组，Targeting 候选声明预期接触，Request 声明场景已有允许接触；Planning 根据
固定阶段规则组合，YAML 不提供任意碰撞白名单。

#### 轨迹输出

PlanningResult 保存分段、连续、时间参数化的轨迹及关键 knot，不固化 MuJoCo 的控制
频率。所有可定位关节（包括可选主动腕）进入按名称索引的 position、velocity 和
acceleration 通道；夹爪进入语义化 actuator position、force limit 等通道。Planner
不能写死夹爪关节名或数量。

Ruckig 只生成时间戳、速度和加速度，不修改几何路径、不删除 waypoint，也不在失败后
静默降低限制。Planning 结果记录配置内容哈希、后端版本、seed、预算和关键参数摘要，
不复制完整 YAML 或资产。

Planning 错误码至少区分请求/配置无效、后端不可用、IK 候选耗尽、VAMP 搜索失败、
连续 IK 中断、阶段碰撞违规、Ruckig 失败、预算耗尽和后端异常。只有六阶段全部成功
才返回成功；部分轨迹不能包装为正式结果。

### 2.5 Execution

唯一公共入口：

```python
ExecutionResult execute(ExecutionRequest request)
```

Request 只包含成功的 PlanningResult、机器人/执行器运行时 adapter、场景对象绑定、
控制频率、日志频率和动力学诊断配置。Execution 不调用 Targeting 或 Planning，不读取
它们的配置和内部状态，也不从 MuJoCo 反推或替换轨迹。

MuJoCo executor 和未来实机 executor 消费完全相同的时间参数化关节及执行器通道。
adapter 只处理控制接口映射。首版实现 MuJoCo executor；实机 adapter 接口明确标记
未实现，不能使用 mock execution 伪造通过。

Execution 按自身控制频率确定性采样 Planning 轨迹，并以独立日志频率流式持久化：

- 阶段和时间戳；
- 目标/实际关节状态；
- 执行器命令与状态；
- 物体和 TCP 状态；
- 去重后的接触名称对及动力学事件。

结果必须拆分：

- `execution_status`：轨迹是否完整交给运行时并执行；
- `dynamics_diagnostics.success`：执行结束后，动力学事实是否满足验收策略。

物体滑落、没有离开支撑面或落点误差过大，不得让 executor 在中途改变路线或伪装成
Planning failure。模型加载失败、控制接口失败、数值仿真不可继续或日志 IO 失败属于
明确的 Execution 错误。日志 IO 错误不得被吞掉，也不得改变已规划控制语义。

动力学诊断至少记录：

- 目标物体是否存在；
- 抓取后是否离开原支撑面；
- 释放前后的物体 pose、线速度和角速度；
- 释放后的稳定状态；
- 目标区域误差；
- 接触名称对、最大穿透和冲击统计。

诊断可以在完整执行结束后按显式验收策略计算 success，但不能作为中途控制分支。
视频渲染、展示松手效果和结果卡片均属于 pipeline 外工具。

## 3. 三次 commit

### Commit 1: Targeting

建议提交信息：

```text
feat(task2): rebuild config-driven grasp targeting
```

范围：

- 新建最小 contracts 公共值对象和 Targeting 专属契约。
- 新建类型化 YAML 配置视图与组合层切分逻辑。
- 定义平行夹爪 family/capabilities 和语义执行器命令。
- 实现四个核心物体的确定性模板展开。
- 实现唯一 `target()` 入口及稳定失败码。
- 加入少量 Targeting 契约和模块边界测试。

不包含 VAMP、IK、Ruckig、MuJoCo、机器人场景规划、临时场景或视频。

验收：相同请求产生相同候选 ID 和顺序；frame/单位/变换闭合正确；候选能映射为平行
夹爪合法语义命令；代码不导入 Planning、MuJoCo 或机器人状态。

### Commit 2: Planning

建议提交信息：

```text
feat(task2): add VAMP planning with pluggable IK and Ruckig
```

范围：

- 增加 PlanningRequest/Result/ErrorCode、六阶段和轨迹契约。
- 增加 RobotModel、SceneSnapshot、PlacementGoal 类型及组合构建。
- 实现唯一 VAMP backend adapter。
- 实现 `IKSolver` 接口及首个 cuRobo IK adapter。
- 实现通用连续 IK、阶段拼接、碰撞切换和附着代理体。
- 实现区域放置目标的确定性采样。
- 使用 Ruckig 完成时间参数化和按名称通道输出。
- 记录候选尝试、失败阶段、耗时、seed、预算和配置哈希。

不包含 MuJoCo execution、动力学成功判据、实机控制、视频或临时场景代码。

验收：四个中心场景分别输出或明确拒绝完整六阶段轨迹；成功轨迹满足关节、速度、
加速度、碰撞和阶段停车规则。真实验收以小型 JSON 报告保存，不保存视频或高频轨迹。

### Commit 3: Execution

建议提交信息：

```text
feat(task2): execute planned trajectories with MuJoCo diagnostics
```

范围：

- 增加 ExecutionRequest/Result/ErrorCode 和动力学诊断契约。
- 实现共用 executor adapter 接口和 MuJoCo adapter。
- 忠实采样并执行六阶段 Planning 轨迹及执行器通道。
- 分离控制频率与流式日志频率。
- 记录动力学状态和接触诊断，区分 execution status 与 dynamics success。
- 保留明确未实现的实机 adapter 接口，不提供 mock 成功路径。
- 用四个中心场景完成 Planning + MuJoCo 动力学验收。

视频生成、临时场景编排和报告汇总仍放在 pipeline 外，不能成为公共模块或 pytest。

## 4. 测试策略

pytest 只保留守住正式能力边界的测试：

- Request/Result、frame、单位和必要配置不变量；
- Targeting 确定性、执行器 compatibility 和稳定失败传播；
- 六阶段顺序、按名称通道和 GRASP/PLACE 停车语义；
- `IKSolver`/VAMP adapter 隔离及后端失败传播；
- 阶段允许接触与禁止碰撞规则；
- Ruckig 输出限位；
- 一个最小、确定、快速的已知规划回归。

不保留入口烟测、重复序列化测试、内部 helper 测试、参数排列测试或慢速四物体 E2E。
真实四物体 Planning 和 MuJoCo execution 作为一次性验收命令运行，报告写入被 Git 忽略
的 `outputs/`。

## 5. 明确禁止的实现

- 按 object point、半径、世界方位、candidate ID、seed 或测试文件名选择特殊路径。
- cuRobo IK 失败后静默切换另一 IK 或 MotionGen。
- VAMP 失败后回放缓存、反向旧路径或切换另一规划器。
- 自动放宽碰撞、速度、加速度、夹持力或预算并包装为成功。
- Execution 读取 MuJoCo 接触或物体位姿后改道、重抓、跳段或提前宣告轨迹失败。
- 从 MuJoCo model/data 反推 Planning 的目标、碰撞代理或抓取结果。
- 将临时场景参数、视频动作、结果卡片或展示松手逻辑放入正式 pipeline。
- 把同一整棵 YAML 字典传给 Targeting、Planning 和 Execution。

合理的有界重试必须使用同一算法、显式配置预算、记录 seed 和每次失败原因。任何降级
路径都必须是可配置、可追踪、可测试且不改变业务语义的正式设计；首版默认不提供降级。
