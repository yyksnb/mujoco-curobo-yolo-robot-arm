# Task2 最小重构迁移与清理说明

本文档用于把实验副本中已经验证的事实迁移到新的 Task2 交付目录，同时阻止迭代代码、
临时场景和历史 workaround 回流。具体实现契约见
[task2_minimal_implementation_plan.md](task2_minimal_implementation_plan.md)。

## 1. 前提与责任

由负责人完成：

- 在当前工程外创建独立交付目录；
- 将交付目录置于提交 `60c7bef`；
- 使用现有 `feature/task2` 分支；
- 迁移四个核心物体的已验证抓取模板；
- 清理新 Task2 对旧 Stage、BODex 兼容层、实验性执行器逻辑和临时脚本的依赖。

本文档不要求修改或重置当前实验副本。实验副本包含未提交改动和大量历史实验，只能
作为只读证据源。任何迁移都必须发生在负责人创建的交付目录中。

## 2. 迁移原则

### 2.1 可以迁移

只允许迁移有来源、可独立验证的数据资产：

- 正式机器人 URDF/MJCF、网格、关节限位和 frame 标定；
- 负责人选定的一个已验证平行夹爪资产及其 TCP、驱动范围和语义接触组；
- 记号笔、纸胶带、铆枪和标准件的可视/碰撞资产；
- 四个物体的抓取模板、开度、夹持力、闭合速度和驻留参数；
- 物体质量、质心、惯量、摩擦和碰撞参数；
- 抓取后使用的球、胶囊和盒体代理；
- VAMP、cuRobo IK、Ruckig 和 MuJoCo 的已确认版本及环境事实。

迁移资产不等于复制旧实现。源码、候选选择逻辑、路径搜索策略、执行状态机和验收分支
必须按新契约从零实现。

### 2.2 禁止迁移

- `src/task2/` 中现有 Targeting、Planning、Execution 或 pipeline 源码；
- 当前 `CuroboPlanner` 中为实验成功率迭代出的 Task2 专用控制流；
- 按物体点位、半径、方位、seed、候选索引或文件名触发的参数；
- 隐式 fallback、缓存回放、反向路径兜底、自动扩大预算或自动放宽碰撞规则；
- 临时验收场景、五物体展示场景、扫描脚本和视频渲染逻辑；
- `outputs/`、视频、CSV、JSON 报告、缓存、`__pycache__` 和 pytest cache；
- Dexonomy、RH56DFX、柔性手套和其他未进入默认范围的实验逻辑；
- BODex/Stage 3 grasp target 兼容格式及其 adapter；
- 无法说明单位、frame、物理意义或来源提交的数值。

## 3. 每项资产的迁移记录

每项迁移数据至少记录以下信息；记录可以先维护在本文件的工作副本中，稳定后再转成
配置注释或独立资产清单：

| 字段 | 要求 |
|---|---|
| 目标字段 | 新 YAML 中的完整键路径 |
| 来源 | 实验副本文件路径和 Git commit；工作区数据需额外标记未提交 |
| 物理意义 | 该数值或资产实际描述什么 |
| 单位 | m、rad、s、kg、N 等 |
| 坐标系 | parent/child frame 和四元数 `wxyz` |
| 适用对象 | object type、end-effector family/capability |
| 独立验证 | 几何、质量、静置、接触或驱动范围检查方式 |
| 排除内容 | 确认未携带点位、seed 或算法内部参数 |

建议的迁移判定：

```text
能指出来源 + 能解释含义 + 能独立复核 + 不依赖实验点位
  -> 可以迁移
否则
  -> 不迁移，重新测量或显式失败
```

## 4. 四个物体抓取模板迁移

抓取模板由负责人迁移。每个物体按以下清单处理：

1. 只读取实验副本中最后经过中心场景验证的参数，不复制候选生成代码。
2. 把 TCP 和 pregrasp 变换还原为物体局部 frame 下的 `T_object_tcp`、`T_tcp_pregrasp`。
3. 核对变换方向、单位、`wxyz` 四元数顺序和执行器真实 TCP。
4. 提取物体几何允许的 yaw 对称性/区间，不迁移世界方位或半径相关采样参数。
5. 提取 pregrasp/grasp 开度、最大允许夹持力、闭合速度和驻留时间。
6. 把 MuJoCo geom 名称替换为 `left_pad`、`right_pad` 等语义接触组。
7. 用执行器行程检查开度，用物体碰撞几何检查接触截面和穿透。
8. 给模板分配稳定 `template_id` 和可解释的先验评分分项。
9. 在相同输入下重复运行 Targeting，确认 candidate ID、数量和顺序稳定。

首版对象：

| object type | Targeting 迁移重点 | 不得迁移 |
|---|---|---|
| `marker` | 细长轴、局部 yaw 对称性、预抓/抓取开度 | 100 mm 点位或特定世界 yaw |
| `masking_tape` | 圆盘抓取截面、开度裕量、对称性 | 400 mm 成功点和候选编号 |
| `rivet_gun` | 已验证夹持截面、较紧夹持命令、代理体 | 中心/200 mm 专用退出路径 |
| `standard_part` | 小物体接触裕量、最小合法开度 | 200 mm 点位或固定 seed |

若实验副本里的模板依赖世界坐标或机器人姿态才能成立，该数据不是 Targeting 模板；
应舍弃，并由新的 Planning 对局部候选做可达性选择。

## 5. 机器人与执行器迁移

只选择一个已经验证的平行夹爪作为默认执行器，不在迁移文档中假定它是 2F85、2F140
或其他具体型号。迁移前由负责人确定稳定 `end_effector_id` 和
`end_effector_family: parallel_gripper`。

必须复核：

- 机器人与执行器组合后的唯一关节顺序；
- base、flange、tool、TCP 和指垫 frame；
- 主动腕是否存在、关节限位及其是否属于可定位关节通道；
- 夹爪开口、闭合、力限制和速度命令到驱动器的映射；
- `left_pad/right_pad` 等语义组到 Planning collision geometry 和 MuJoCo geom 的映射；
- 执行器自身碰撞几何、自碰撞排除及其资产来源；
- MuJoCo asset、控制器和 mount 配置。

主动腕首版不迁移实验性的“主动腕优先”权重或零空间代价，只作为普通关节。配置可
声明可选 posture objective capability，但默认关闭。

`RobotModel` 必须由组合层一次构建并传给 VAMP、IK adapter 和 Ruckig。发现同一限位
或关节顺序在多个后端配置中重复时，应先统一真源，而不是复制并靠测试维持一致。

## 6. 物理与碰撞参数迁移

Execution 物理参数与 Planning 代理体分别验证：

- Execution 使用真实质量、质心、惯量、摩擦和 MuJoCo 碰撞资产。
- Planning 抓取前使用 SceneSnapshot 中的原始目标碰撞体。
- Planning 抓取后只使用物体局部 frame 下的球、胶囊和盒体组合代理。
- 代理体必须保守覆盖搬运时可能碰撞的实体，但不能从某次 MuJoCo 运行状态动态生成。

最低独立检查：

1. 网格尺度和轴方向；
2. 质量、质心和惯量数量级；
3. 无机械臂情况下的自由落体与静置稳定性；
4. 物体与油箱、物体与指垫的摩擦参数来源；
5. 原始碰撞体和简化代理体的可视叠加；
6. 代理体在 `T_tcp_object` 下的变换闭合。

检查失败时保留明确失败，不通过增大碰撞豁免、改变加载点或修改验收阈值制造成功。

## 7. 清理顺序

清理由负责人完成，建议在每笔功能 commit 前后分别做依赖审查，不把大规模删除与行为
修改混成无法定位的提交。

### 7.1 建立清理清单

先对交付目录执行只读依赖扫描，按以下类别标记：

- 新 Task2 必需；
- Task1 或其他正式模块仍在使用；
- 仅旧 Task2/Stage/BODex 使用；
- 仅临时验收、展示或实验使用；
- 生成物或缓存。

不得因为新 Task2 不再使用某个共享文件就直接删除它。`src/robot_arm_pipeline/` 下的
planner、executor、scene 和 types 可能仍是 Task1 或其他正式入口的依赖；只有确认没有
正式消费者后才能删除。否则新 Task2 只需停止依赖该文件。

### 7.2 优先删除或重写的旧 Task2 内容

以 `60c7bef` 为基线时，重点审查：

```text
src/task2/targeting.py
src/task2/pipeline.py
scripts/run_task2_pipeline.py
configs/task2/targeting.json
configs/task2/end_effectors.json
tests/test_task2_pipeline.py
docs/task2_pipeline.md
src/task2/README.md
```

这些文件不能作为新源码复制。需要相同路径时按新契约重写；不再需要时删除。旧测试不
应通过修改断言继续保留，应根据新公共边界重新选择最少测试。

随后审查并清理：

- 新 Task2 对 BODex adapter、example grasp target 和 Stage 3 grasp compatibility 的引用；
- 新 Task2 对 MockPlanner、MockExecutor 和旧 mock pipeline 的引用；
- 旧 MuJoCo executor skeleton 和 demo-only physical grasp；
- 旧 cuRobo MotionGen Task2 路线；
- Dexonomy、RH56DFX、手套柔性模型及对应脚本；
- 临时中心/半径扫描/多物体展示/视频脚本；
- 不再引用的配置、资产摘要和历史诊断文档。

对共享模块的处理必须以依赖扫描结果为准，不能破坏 Task1 的正式入口。
当前工程规则明确要求保留 Stage 1/2/3 和 `MockPlanner`；因此本轮只能令新 Task2 与它们
解耦，不能从仓库删除共享实现。只有项目规则被另行明确修改，且依赖审查证明没有正式
消费者后，才能在独立清理提交中讨论物理删除。

### 7.3 生成物清理

交付目录不得包含：

```text
outputs/
__pycache__/
.pytest_cache/
*.pyc
临时 MJCF
视频与帧缓存
扫描 CSV/JSON
CUDA 或抓取候选 cache
```

同时更新 `.gitignore` 覆盖这些类别，但不要使用过宽规则忽略正式配置、资产或验收工具。

## 8. 按 commit 执行迁移

### Commit 1 前：Targeting 数据

- 负责人迁移四个物体模板并完成来源记录。
- 迁移默认平行夹爪 family/capabilities 和语义接触组。
- 不迁移机器人状态、场景、IK 或动力学结果。
- 清除旧 BODex/Stage 3 grasp target 在新 Task2 默认路径中的引用。

Commit 1 后检查：Targeting 包的 import graph 不包含 Planning、VAMP、cuRobo、Ruckig
或 MuJoCo；配置视图不暴露 planning/execution namespace。

### Commit 2 前：Planning 资产

- 迁移并复核正式机器人模型、执行器组合模型和碰撞几何。
- 迁移四个物体的球/胶囊/盒附着代理。
- 确认 VAMP、cuRobo IK 和 Ruckig 版本及构建环境，不迁移旧 adapter 源码。
- 从场景资产离线产生 SceneSnapshot 输入，不让 Planner 读取 MuJoCo model/data。

Commit 2 后检查：默认路径只有 VAMP；cuRobo 只存在于 IKSolver adapter；VAMP、IK、
Ruckig 使用相同 RobotModel；没有 MuJoCo execution import。

### Commit 3 前：Execution 资产

- 迁移机器人、平行夹爪、油箱和四个物体的正式 MuJoCo assets。
- 迁移来源明确的质量、质心、惯量、摩擦和控制器参数。
- 建立语义执行器通道与 MuJoCo actuator/geom 的 adapter 映射。
- 不迁移旧执行状态机、接触失败中断、缓存轨迹或展示松手逻辑。

Commit 3 后检查：MuJoCo 状态只进入日志和 `DynamicsDiagnostics`；任何动力学任务判据
都不能改变计划轨迹；控制日志流式写入且控制频率与日志频率分离。

## 9. 清理完成判据

- `rg` 搜索不到新 Task2 对 BODex、Dexonomy、RH56DFX、MockPlanner 或旧 Stage artifact 的引用；共享旧模块本身按工程规则保留。
- 正式 Targeting/Planning/Execution 各只有一个公共入口。
- 模块只接收类型化正式输入，不打开其他模块的配置或输出内部字段。
- YAML 中不存在点位、半径、世界方位、seed 或候选索引特判。
- 默认路径不存在 fallback、cache replay 或自动放宽约束。
- pytest 只保留公共契约、关键规则、安全/失败语义和一个快速规划回归。
- 四物体真实验收报告和视频不进入 Git。
- 工作区中没有误提交的输出、cache、临时场景或大文件。

如果清理发现 Task1 仍依赖待删除的共享实现，应停止删除：先记录依赖，令新 Task2 与其
解耦，再单独决定共享模块的后续生命周期。不得为了让清理列表变绿而破坏其他模块。
