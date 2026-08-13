# Task2 资产、参数与清理台账

本文档记录 Task2 最小重构迁移的证据和边界。它不是运行时配置，也不建立
Targeting、Planning 或 Execution 的临时 schema。后续实现只能在正式组合层中读取新
schema，并把切分后的类型化视图传入各阶段。

## 1. 基线与证据快照

- 交付目录：`/home/yoda/mujoco-curobo-yolo-robot-arm`
- 交付分支：`feature/task2`
- 交付 `HEAD`：`60c7bef54eeeb695fda89322011f00e2b56f393d`
- 实验副本：`/home/yoda/mujoco-curobo-yolo-robot-arm-narrow-passage-experiment`
- 实验副本已提交证据：`53d5c149468648b8ada8629f810a28d760a2ec37`
- 参数采用的最后已提交版本：`fe2fd83b46bd5639e86d60eb6bf7d2975360b1f7`
- 实验副本工作区在迁移前已有未提交修改。未提交的
  `configs/task2/manipulation.json` 铆枪支撑接触修改未迁移。

实验副本始终只读。迁移前后用以下两个摘要检查其受版本控制的工作区状态未改变：

```text
git status --porcelain=v1 -uall | sha256sum
09dd301b725e2d85998f612ffd7d6bd1e6fa8bfb69c9c1694df751104036601a

git diff --binary | sha256sum
3384a3bf1d77e5208bcfe58ad821e8463f09b7688eea5ece864243c1e9045559
```

## 2. 模块边界与依赖审查

审查结论：旧 `src/task2/targeting.py` 直接依赖 Task1 observation loader、共享
`TransformMatrix`、GraspGenX 和旧 JSON 配置；旧 pipeline、CLI、测试和说明只服务于
该接口。Task1、Stage 1/2/3 和共享 planner/executor 没有反向导入 `task2`，因此旧
Task2 独占文件可以删除。

以下共享边界仍有正式消费者，必须保留：

- `src/robot_arm_pipeline/`：Stage 1/2/3、`MockPlanner`、`MockExecutor`、共享 scene、
  planner、executor 和 types；
- `examples/mujoco/gen3_with_tank.xml`、`examples/mujoco/things/`：Task1 scene、survey、
  replay、final simulation 和相关测试；
- `examples/bodex_grasp_target.json` 与 BODex adapter：Stage 1/3；
- Robotiq 资产和旧 MuJoCo executor：Task1、Stage 2/3 或共享测试。

新 Task2 不引用上述旧 Stage/BODex/mock 接口。本轮不实现新 Task2 源码，因此清理后
不存在为了通过测试而增加的 fallback、mock success 或静默容错。

## 3. 迁移、保留、删除与暂缓

| 类别 | 项目 | 处理 |
|---|---|---|
| 迁移 | SO-101 派生活动腕平行夹爪 MJCF、网格、许可证 | 从实验提交迁入，未接入旧 registry |
| 迁移 | Gen3 + 活动腕平行夹爪组合 URDF | 从实验提交迁入，作为待建 `RobotModel` 的资产证据 |
| 保留 | 四个核心物体的视觉网格、CoACD 碰撞网格和物理参数 | 已包含于 `60c7bef`，不复制第二真源 |
| 删除 | 旧 Task2 Targeting/pipeline/CLI/config/test/docs | 无 Task1 或 Stage 1/2/3 消费者 |
| 暂缓 | 正式 `objects.yaml`、`end_effectors.yaml`、`robot.yaml` | 组合层和类型化 schema 尚未建立 |
| 迁移 | 四物体稳定 `template_id`、`T_object_tcp`、`T_tcp_pregrasp`、局部 yaw、开度、限力、速度、驻留、接触组和评分分量 | 完整结果写入本台账的 YAML 形状输入；正式 schema 尚未建立，不形成运行时配置 |
| 仿真标定 | 四物体抓取模板的几何/动力学名义值 | 已由完整 CoACD + SO-101 pad 的离线扫描确定；明确不冒充实机测量 |
| 暂缓 | 四物体附着球/胶囊/盒代理 | 旧实现动态使用请求球集合/球数上限，不是合格的对象局部代理真源 |
| 暂缓 | MuJoCo adapter 映射 | Execution 正式 schema 尚未建立；不复用旧 registry |
| 暂缓 | 实验 Gen3/油箱 CoACD、cuRobo robot/world/graph YAML | 会修改 Task1 共享场景或复制旧 Planning 内部参数；待正式 `RobotModel`/`SceneSnapshot` schema 后离线复核 |

对象命名存在一个待正式 schema 解决的适配边界：设计名为 `masking_tape`，Task1/实验
类别为 `paper_tape`，共享 MuJoCo body 为 `target_tape`。新 Task2 不应直接依赖 Task1
或 MuJoCo 的内部名称；组合层以后必须用显式映射适配，不能改写 canonical object type。

## 4. SO-101 派生平行夹爪资产

本轮根据四个核心物体最后已提交的实验记录，选择稳定 ID
`so101_active_wrist_parallel`，family 为 `parallel_gripper`。其 capability 是双侧对向
闭合接触和最大来源规格开口 `0.084 m`；语义接触组为 `left_pad/right_pad`。活动腕是
普通可定位关节，posture objective 默认关闭。该选择只记录资产事实，尚未建立运行时
默认选择或配置 schema。

### 4.1 文件来源和完整性

| 目标文件 | 来源提交 | SHA-256 | 含义与独立验证 |
|---|---|---|---|
| `examples/mujoco/so101_active_wrist_parallel/active_wrist_parallel.xml` | 初始 `71a8466bdcce2c806beed274c4597fa4aa69122b`；碰撞/质量修订 `a5a800728e6efec7d1d3150296b3fd8b192181a0`、`fe2fd83b46bd5639e86d60eb6bf7d2975360b1f7` | `589fff37b6a0eefefb47f23010f56909981c48ed96f4abe737ef37367998e9d6` | MuJoCo 组合资产；独立加载、检查 joint/actuator/site/geom |
| `configs/curobo/gen3/gen3_active_wrist_parallel.urdf` | `71a8466bdcce2c806beed274c4597fa4aa69122b` | `5086a13de595e55a3abafefd00300f32f07d391b9187f653f585561e4bc44e35` | Gen3 与工具组合运动学；XML 解析并核对关节链、限位和 TCP |
| `assets/gripper_fixed_assembly.stl` | 初始 `71a8466...`；无相机 holder 修订 `a5a8007...` | `a3c31ec1108c69c333ada7eaaccc0ff23a49beab9a8eac2b59d1d15b1a447bc1` | 固定夹爪组件视觉网格；检查毫米到米缩放和包围盒 |
| `assets/gripper_left_clamp.stl` | `71a8466...` | `0a1c2caf47b97853e4dc0974efc5ed89f9969b89cae3c0b8f8d9e8705bff3498` | 左夹爪视觉网格；与右夹爪行程和闭合面对称核对 |
| `assets/gripper_right_clamp.stl` | `71a8466...` | `a9540533009f872423431d8405613ee52a609e2d296a287c17867f941aae50ca` | 右夹爪视觉网格；与左夹爪行程和闭合面对称核对 |
| `assets/wrist_holder.stl` | `71a8466...` | `887f92e6013cb64ea3a1ab8675e92da1e0beacfd5e001f972523540545e08011` | 腕部 holder 视觉网格；检查装配 frame |
| `assets/wrist_servo.stl` | `71a8466...` | `a37c871fb502483ab96c256baf457d36f2e97afc9205313d9c5ab275ef941cd0` | 腕部舵机视觉网格；检查装配 frame |
| `LICENSE.ROBONINE-GPL-3.0` | `71a8466...`；上游 Robonine commit `5310ee2b6e9209b7cf33c1e32cc6d0ccbe049e68` | `3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986` | Robonine 夹爪资产许可；发布前做许可证审查 |
| `LICENSE.SO_ARM100-APACHE-2.0` | `71a8466...`；上游 TheRobotStudio commit `fda892cba81032c46c40976a48c9ceadbf40a9ca` | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` | SO-ARM100 腕部资产许可；发布前做许可证审查 |

目录内全部 9 个文件的有序 SHA-256 清单摘要为
`5cebebf90e0c0f38a814b2376d5a8695ca1f46a40d3d9d426ff25b2bc4f3bf97`。

### 4.2 来源规格参数

除另行说明，MJCF 长度单位为 m、角度为 rad、质量为 kg、力为 N、力矩为 N m，四元数
顺序为 `wxyz`。URDF 使用 m/rad。`mount` 是工具资产根 frame，`wrist_output` 是活动腕
输出 frame，`tcp` 是夹持 TCP。

| 参数 | 值 | 坐标系/物理意义 | 来源 | 独立验证 |
|---|---:|---|---|---|
| 活动腕轴 | `[0, 1, 0]` | `mount` 子链中的 `wrist_output` 旋转轴 | SO-ARM100 commit `fda892...`，实验 `71a8466...` | MJCF/URDF 轴一致性与正负转动可视检查 |
| 活动腕范围 | `[-1.6580628, 1.6580628] rad` | `wrist_output` 相对 `mount` | 同上 | 读取 `jnt_range`，分别置于上下限并 `mj_forward` |
| 活动腕速度上限 | `5.236 rad/s` | 组合 URDF `task2_tool_wrist_pitch` | 同上 | URDF limit 解析；后续与实物规格复核 |
| 活动腕力矩上限 | `2.94 N m` | wrist actuator/URDF effort | STS3215 来源规格，实验 `71a8466...` | MJCF `forcerange` 与 URDF effort 相等 |
| 平行夹爪总行程 | `0.084 m` | 两指相对开口宽度 | Robonine commit `5310ee...`，实验 `71a8466...` | 两个 slide joint 各 `0.042 m`，端点差合计 `0.084 m` |
| 单指行程 | `[0, 0.042] m` | 各 jaw 相对 `wrist_output`，沿相反 X 轴 | 同上 | 在上下限调用 `mj_forward` 并测两指距离 |
| 最大来源规格夹持力 | `120 N` | 工具总夹持能力上限，不是对象模板命令 | 同上 | 对照来源规格；MJCF grip `forcerange=[-120,120]` |
| 来源规格闭合速度 | `0.014 m/s` | 平行夹爪线速度上限；尚未绑定正式语义命令 | 同上 | 对照来源规格；后续实物空载行程计时 |
| 夹爪来源规格质量 | `0.170 kg` | 固定组件加两 jaw 的来源总质量 | 同上 | MJCF `0.140 + 0.015 + 0.015 = 0.170 kg` |
| 活动腕质量 | `0.075 kg` | SO-101 CAD 派生腕部质量 | SO-ARM100 commit `fda892...` | 对照 CAD；后续称重 |
| Gen3 clearance adapter 质量 | `0.020 kg` | 仿真打印件估值，不是实测 | 实验 `71a8466...` | 打印后称重；未测前保持 uncalibrated |
| mount 组合质量 | `0.095 kg` | 活动腕加 adapter | 实验 `71a8466...` | `0.075 + 0.020 = 0.095 kg` |
| TCP | `pos=[0,0,0.155] m`, `quat=[0.70710678,0,0,-0.70710678]` | `wrist_output -> tcp` | Robonine CAD/MJCF，实验 `71a8466...` | MJCF site 与 URDF fixed joint（含 frame 表达差异）闭合检查 |
| Gen3 工具 adapter | `xyz=[0,0,-0.08149039] m`, `rpy=[-pi,0,-pi/2]` | `bracelet_link -> task2_tool_adapter` | 实验组合 URDF `71a8466...` | 与后续组合层 mount 标定独立比对；当前不接入运行时 |
| 语义接触组 | `left_pad`, `right_pad` | 分别绑定 MJCF `left_pad_collision`、`right_pad_collision` | MJCF `fe2fd83...` | 枚举 geom，确认每组唯一且左右 jaw 分属不同 body |

### 4.3 仿真派生、未标定参数

以下参数保留为资产的一部分，但不能包装为实机标定事实。来源均为
`active_wrist_parallel.xml` @ `fe2fd83...`；frame 均为参数所在 body 的局部 frame。

| 参数组 | 值和单位 | 物理意义 | 独立验证 |
|---|---|---|---|
| 默认 joint dynamics | `damping=0.60 N m s/rad`, `frictionloss=0.052 N m`, `armature=0.028 kg m^2` | wrist 默认数值稳定参数 | 无接触阶跃响应；与实机辨识结果比较 |
| jaw joint dynamics | `damping=0.5 N s/m`, `frictionloss=0.1 N`, `armature=0.002 kg` | 两个 slide joint 的仿真参数 | 左右空载阶跃响应一致性 |
| structure contact | `friction=[0.55,0.004,0.0001]`, `solref=[0.006,1]`, `solimp=[0.95,0.99,0.001,0.5,2]` | 非指垫结构接触 | 斜面/压入试验；当前 uncalibrated |
| pad contact | `friction=[0.80,0.006,0.0002]`, `solref=[0.004,1]`, `solimp=[0.95,0.99,0.001,0.5,2]` | 指垫接触 | 拉力/斜面与压入试验；当前 uncalibrated |
| mount inertia | `mass=0.095`, `com=[0,0,0.018] m`, `diag=[0.000045,0.000045,0.000024] kg m^2` | adapter + active wrist 惯性代理 | CAD/称重/摆锤复核 |
| wrist_output inertia | `mass=0.140`, `com=[0,0,0.070] m`, `diag=[0.00031,0.00024,0.00018] kg m^2` | 无 jaw 的夹爪固定组件 | CAD/称重/摆锤复核 |
| 每个 jaw inertia | `mass=0.015`, `com=[+/-0.008,0,0.050] m`, `diag=[0.000035,0.000030,0.000009] kg m^2` | 左右 jaw 惯性代理 | 左右镜像、总质量和 CAD 复核 |
| wrist position actuator | `kp=998.22`, `kv=2.731`, `force=[-2.94,2.94]`, `ctrl=[-1.6580628,1.6580628]` | MuJoCo position servo | 限幅和空载阶跃检查 |
| grip position actuator | `kp=2400`, `kv=15`, `force=[-120,120] N`, `ctrl=[0,0.042] m` | tendon position servo | 限幅、左右同步和开闭行程检查 |
| jaw coupling | tendon coefficients `0.5/0.5`; equality `polycoef=[0,1,0,0,0]`, `solref=[0.005,1]`, `solimp=[0.95,0.99,0.001]` | 左右 jaw 同步 | 全行程采样，左右 joint position 相等 |

碰撞代理尺寸也属于未标定仿真参数：adapter box `pos=[0,0,-0.010] m`,
`size=[0.022,0.022,0.010] m`；wrist mount box `pos=[0,0,0.024] m`,
`size=[0.026,0.022,0.024] m`；gripper frame box `pos=[0,0,0.062] m`,
`size=[0.064,0.030,0.012] m`；holder box
`pos=[-0.00389505,0.01414026,0.03258090] m`,
`quat=[0.70436570,0.03774885,0.70792765,0.03581096]`,
`size=[0.0085,0.024,0.0185] m`；左右 carriage box 为镜像
`pos=[-/+0.010,0,0.010] m`, `size=[0.018,0.025,0.010] m`；左右 finger box
为镜像 `pos=[-/+0.002,0,0.025] m`, `size=[0.006,0.008,0.025] m`。独立验证方式
是将碰撞体与 CAD 网格叠加并检查保守覆盖和自碰撞，而不是通过放宽碰撞规则验证。

指垫碰撞包络的 jaw-local 顶点为 X `+/-0.004 m`，在 Z `0.050 m` 处 Y
`+/-0.0140 m`，在 Z `0.079 m` 处 Y `+/-0.0055 m`。它来源于
`a5a8007...`/`fe2fd83...` 对 Robonine 锥形夹爪 CAD 的保守拟合；独立验证为 CAD
截面包围和左右镜像检查。

## 5. 四个核心物体资产与物理参数

这些文件已在交付基线中存在，不从实验副本重复复制：视觉资产来自
`09f1a760fa767036a6b35fc3489065974da46496`，CoACD 资产来自
`534ac285bc6e0ce68ba4e1761992474d7b83b47c`，动力学参数来自
`0ae70e85a2053f21471e644256f483a5a482c8f1`。共同 body frame 为“XY 包围盒中心、
mesh 底面 local z=0”；质量单位 kg，COM 单位 m，惯量单位 kg m^2，惯量四元数为
`wxyz`。`target_objects.xml` 中的世界 `pos/euler` 是 Task1 场景初值，不是 Task2
模板，明确排除。

| object type / 共享 body | 视觉与碰撞资产 | 物理参数（仿真名义值，未实测） | 接触参数 | 独立验证 |
|---|---|---|---|---|
| `marker` / `target_marker` | `Sharpie.stl`, scale `0.001`; geom offset `[0.018773122,0,0.009581937] m`, Euler `[0,90,0] deg`; 2 个 `marker_coacd_*` | mass `0.02`, range `[0.012,0.035]`; COM `[0.0113691881616,-3.8882748788e-07,0.00920583283537]`; inertia quat `[0.497665709665,0.502323443036,0.502323443036,0.497665709665]`; diag `[2.44961099878e-05,2.44623656413e-05,3.46245394055e-07]` | rigid plastic；friction `[0.45,0.004,0.0001]`, solref `[0.006,1]`, solimp `[0.95,0.99,0.001,0.5,2]` | 网格/CoACD 叠加、称重、平衡 COM、斜面摩擦、静置 |
| `masking_tape` / `target_tape` | `tape.STL`, scale `0.004`; offset `[0.038000001,0.038000001,0.020000000] m`, Euler `[180,0,90] deg`; 13 个 `tape_coacd_*` | mass `0.05`, range `[0.03,0.09]`; COM `[1.88728240674e-09,1.88728240674e-09,0.01]`; inertia quat `[-0.707106781187,7.71600526371e-17,-0.707106781187,-3.38622498254e-17]`; diag `[3.45066833096e-05,1.86949170225e-05,1.86949170225e-05]` | paper/plastic tape；friction `[0.55,0.006,0.0002]`, solref `[0.008,1]`, solimp `[0.93,0.98,0.002,0.5,2]` | 直径/厚度量测、网格叠加、称重、滚动/滑动摩擦、静置 |
| `rivet_gun` / `target_rivet_gun` | `Glue.STL`, scale `0.3`; offset `[0.032206425,0.019703770,0.013947060] m`, Euler `[0,90,0] deg`; 16 个 `rivet_gun_coacd_*` | mass `0.551498684739`, range `[0.35,0.75]`; COM `[0.00149805105377,0.012487608251,0.0140542604033]`; inertia quat `[0.69716048383,0.117980354665,0.695306269636,-0.128829682539]`; diag `[0.000379859923921,0.000310567919754,0.000102890048133]` | hard metal；friction `[0.35,0.003,0.0001]`, solref `[0.004,1]`, solimp `[0.95,0.99,0.001,0.5,2]` | 网格尺度/轴向、CoACD 覆盖、称重、平衡/摆锤、静置 |
| `standard_part` / `target_standard_part` | `Screw.stl`, scale `0.001`; offset `[0,0.007100000,0.002694798] m`, Euler `[90,0,0] deg`; 16 个 `standard_part_coacd_*` | mass `0.00222307104768`, range `[0.001,0.005]`; COM `[1.11881578138e-05,-0.000257940518394,0.00269568770419]`; inertia quat `[0.707106781187,0.707106781187,0,0]`; diag `[3.99615868774e-08,4.00351044153e-08,3.63746040556e-09]` | hard metal；friction `[0.35,0.003,0.0001]`, solref `[0.004,1]`, solimp `[0.95,0.99,0.001,0.5,2]` | 卡尺量测、CoACD 覆盖、精密称重、静置/滑动检查 |

共享文件完整性：

```text
target_object_assets.xml  2b2fdd01d75641855af0822355003313318af9bb1babee00be85f1d264f42dac
target_objects.xml        15496dadd738725e94958b06679303d4c745102a8cce877c207bfdc341fe0d31
target_physics.json       d76e10fd19c14cff2040bc0eb91212a081250fa01fbdb6fa26625b7600dd7f42
四类 CoACD 文件清单摘要  6fe51dda9a1e182f728eff99d0b3021d069fabb47c33019b3f678aa0573c27b0
```

## 6. Commit 1 前四物体抓取模板迁移记录

### 6.1 模块边界与证据判定

本节只登记迁移数据，不是运行时配置，也不定义 Targeting schema。旧
`configs/task2/manipulation.json` 把 Targeting 截面、Planning 路线/预算和 Execution
接触时序混在一个 policy 中；旧 Targeting 又以 GraspGenX 世界候选为主。因此不能复制
旧 policy 或源码，只能逐字段证明该值与世界点位、机器人状态、候选索引和路径策略无关。

采用的已提交证据为：

- 工具与公共命令：`configs/task2/manipulation.json`、
  `examples/mujoco/so101_active_wrist_parallel/active_wrist_parallel.xml`、
  `src/task2/planning.py` 和 `src/task2/execution.py` @
  `fe2fd83b46bd5639e86d60eb6bf7d2975360b1f7`；
- 纸胶带中心场景验收：提交
  `740f84534430f768021a211212bf5b6b19021711`，最后已提交参数取 `fe2fd83...`；
- marker、rivet_gun、standard_part 中心场景验收：提交
  `354a2b2c47d6678debd9782169cc85463478cf82`，最后已提交参数取 `fe2fd83...`。

实验副本中当前被 Git 忽略的 Targeting artifact 只覆盖铆枪 200 mm 和临时 side-up
场景；其 schema 明确标记为 `task2_temporary_targeting_fixture/v1`，且说明候选不是
GraspGenX 推理结果。它们没有可归属的来源 commit，并含点位、世界方位、候选编号和
退出路线，因此没有用于本节数值。

`contact_center_grasp_m` 是旧 GraspGenX canonical gripper frame 中的指垫接触截面，
不是物体位姿；`pregrasp_opening_clearance_m` 是额外直径间隙，不是绝对开度；
`contact_preload_fraction` 是旧 MuJoCo position actuator 的闭合行程 fraction，不是 N。
三者均不能填入 `T_object_tcp` 或对象安全力上限。

### 6.2 已迁移的公共模板字段

四个模板均绑定 `end_effector_family=parallel_gripper`。下表中的值在四个模板登记中
相同，但后续正式组合层仍须逐个对象显式提供，不能让 Targeting 读取旧工具 policy。

| 字段 | 迁移值 | 单位/frame/物理意义 | 来源与独立验证 |
|---|---|---|---|
| `T_tcp_pregrasp` | `[[1,0,0,0],[0,1,0,0],[0,0,1,-0.12],[0,0,0,1]]` | 平移 m；`parent=tcp_at_grasp`、`child=tcp_at_pregrasp`；旋转为单位四元数 `wxyz=[1,0,0,0]` | `approach_distance_m=0.12` 与 `_build_pose_targets()` @ `fe2fd83...`；独立计算 `(T_world_grasp T_grasp_tcp)^-1 (T_world_grasp Trans_z(-0.12) T_grasp_tcp)`，对任意世界位姿均闭合为该矩阵 |
| 旧全行程等效速度上界 | `0.014` | m/s；旧值仅是全行程算术上界，模板 `closing_speed_m_s=0.014` 另由本轮四对象闭合扫描复核 | 总行程 `0.084 m`、单指行程 `0.042 m`、`command_ramp_s=6.0 s` @ `fe2fd83...`；本轮实际开度到接触的动态扫描同时通过线/角速度边界 |
| `grasp_dwell_s` | `0.25` | s；检测到双侧接触并完成 preload 后保持抓取命令的时间 | `actions.acquire.dwell_s=0.25` 及 `_execute_contact_completed_action()` @ `fe2fd83...`；独立检查执行顺序为 contact、preload、dwell |
| 工具力上限证据 | `120` | N；SO-101 夹爪/仿真 actuator 能力上限，不是物体模板力值 | `maximum_grip_force_n=120` 与 grip actuator `forcerange=[-120,120]` @ `fe2fd83...`；本轮对象模板力由独立 1g/3g 扫描确定，实机安全仍未标定 |

`T_tcp_pregrasp` 的方向按本工程 `T_parent_child` 约定记录。旧 pregrasp 世界位姿的构造
先沿抓取 frame 的 `-Z` 平移 `0.12 m`，再乘 `T_grasp_tcp`；由于该工具标定只绕 Z
旋转并沿 Z 平移，换算到 grasp TCP frame 后仍是 `Trans_z(-0.12)`。

### 6.3 四模板登记与字段状态

`template_id` 是本次迁移分配并冻结的稳定标识，不带版本号，也不派生自世界点位、
运行序号、seed 或候选索引。后续正式 schema 应原样采用这些字面值；更换实际抓取截面
时应分配新 ID，而不是在同一 ID 下静默改变语义。

四个模板的 Commit 1 输入如下。代码块是设计字段的 YAML 形状，不是运行时 YAML，也
不建立正式 schema；矩阵为 `T_parent_child`，平移单位 m，物体 frame 是共同的
“XY bbox center + mesh bottom z=0” frame，角度单位 rad。

```yaml
templates:
  marker_parallel_body:
    template_id: marker_parallel_body
    object_type: marker
    calibration: simulation_nominal
    T_object_tcp: [[-1,0,0,0.006204], [0,1,0,0], [0,0,-1,0.00741914], [0,0,0,1]]
    T_tcp_pregrasp: [[1,0,0,0], [0,1,0,0], [0,0,1,-0.12], [0,0,0,1]]
    yaw_symmetry: {kind: discrete, group: C2, axis: approach_axis_object, axis_vector: [0, 0, -1], values_rad: [0, 3.141592654]}
    pregrasp_opening_m: 0.021639206
    target_opening_m: 0.013014206
    grip_force_limit_n: 0.75
    minimum_retention_force_n: 0.30
    closing_speed_m_s: 0.014
    dwell_s: 0.25
    expected_contact_groups: [left_pad, right_pad]
    template_priority: 1.0
    quality_score: 1.0
    score_components: {template_priority: 1.0}
    diagnostics: {contact_margin_m: 0.062260, opening_margin_m: 0.008000, force_margin_n: 0.450000}

  masking_tape_parallel_outer_wall:
    template_id: masking_tape_parallel_outer_wall
    object_type: masking_tape  # aliases paper_tape/target_tape stay in the adapter
    calibration: simulation_nominal
    T_object_tcp: [[0,1,0,0], [1,0,0,0], [0,0,-1,0.0155], [0,0,0,1]]
    T_tcp_pregrasp: [[1,0,0,0], [0,1,0,0], [0,0,1,-0.12], [0,0,0,1]]
    yaw_symmetry:
      kind: continuous
      axis: approach_axis_object
      axis_vector: [0, 0, -1]
      range_rad: {lower: -3.141592654, upper: 3.141592654, lower_inclusive: true, upper_inclusive: false}
    pregrasp_opening_m: 0.082000000
    target_opening_m: 0.074333332
    grip_force_limit_n: 2.0
    minimum_retention_force_n: 0.75
    closing_speed_m_s: 0.014
    dwell_s: 0.25
    expected_contact_groups: [left_pad, right_pad]
    template_priority: 1.0
    quality_score: 1.0
    score_components: {template_priority: 1.0}
    diagnostics: {contact_margin_m: 0.004500, opening_margin_m: 0.006000, force_margin_n: 1.250000}

  rivet_gun_parallel_handle:
    template_id: rivet_gun_parallel_handle
    object_type: rivet_gun
    calibration: simulation_nominal
    T_object_tcp: [[-0.866025404,0,0.5,-0.010494798], [0,1,0,0.017997664], [-0.5,0,-0.866025404,0.001004167], [0,0,0,1]]
    T_tcp_pregrasp: [[1,0,0,0], [0,1,0,0], [0,0,1,-0.12], [0,0,0,1]]
    yaw_symmetry: {kind: discrete, group: C2, axis: approach_axis_object, axis_vector: [0.5, 0.0, -0.866025404], values_rad: [0, 3.141592654]}
    pregrasp_opening_m: 0.084000000
    target_opening_m: 0.011946974
    grip_force_limit_n: 24.0
    minimum_retention_force_n: 12.0
    closing_speed_m_s: 0.014
    dwell_s: 0.25
    expected_contact_groups: [left_pad, right_pad]
    template_priority: 1.0
    quality_score: 1.0
    score_components: {template_priority: 1.0}
    diagnostics: {contact_margin_m: 0.017030, opening_margin_m: 0.052053, force_margin_n: 12.000000}

  standard_part_parallel_shaft:
    template_id: standard_part_parallel_shaft
    object_type: standard_part
    calibration: simulation_nominal
    T_object_tcp: [[0,1,0,0], [1,0,0,0.004], [0,0,-1,0.000694798], [0,0,0,1]]
    T_tcp_pregrasp: [[1,0,0,0], [0,1,0,0], [0,0,1,-0.12], [0,0,0,1]]
    yaw_symmetry: {kind: discrete, group: C2, axis: approach_axis_object, axis_vector: [0, 0, -1], values_rad: [0, 3.141592654]}
    pregrasp_opening_m: 0.010995694
    target_opening_m: 0.002745694
    grip_force_limit_n: 0.30
    minimum_retention_force_n: 0.10
    closing_speed_m_s: 0.014
    dwell_s: 0.25
    expected_contact_groups: [left_pad, right_pad]
    template_priority: 1.0
    quality_score: 1.0
    score_components: {template_priority: 1.0}
    diagnostics: {contact_margin_m: 0.003100, opening_margin_m: 0.008000, force_margin_n: 0.200000}
```

每个值的来源、含义和独立方法：物体 body frame/CoACD 来自
`target_objects.xml`、`target_object_assets.xml` @ `534ac285bc6e0ce68ba4e1761992474d7b83b47c`，
物理来自 `target_physics.json` @ `0ae70e85a2053f21471e644256f483a5a482c8f1`；工具
TCP、指垫局部几何、行程、`kp=2400 N/m` 和 `forcerange=120 N` 来自
`active_wrist_parallel.xml` @ `fe2fd83b46bd5639e86d60eb6bf7d2975360b1f7`。`T_object_tcp`
由上述 object-local 接触截面和 `T_grasp_tcp` 逆变换得到；rivet 同时以已提交 140 mm
成功截面记录 @ `354a2b2c47d6678debd9782169cc85463478cf82` 复核。`T_tcp_pregrasp` 由
已提交 `approach_distance_m=0.12` 构造，并对三组任意 world pose 复算，最大矩阵残差
`2.22e-16`。marker/standard 的 TCP 平移分别来自编译 CoACD 接触点
`[0.006204, +/-0.006740, 0.009419] m` 和 shaft 中心 `[0, 0.004000, 0.002695] m`；
tape 使用外壁中高 `z=0.0155 m`。

`target_opening_m` 是带预压的两指**总间距目标命令**，不是稳定后的实际 jaw pose，也
不是首次双侧接触宽度；后续 Execution adapter 才能把它映射为每个 jaw 的
`0--0.042 m` 控制量。`grip_force_limit_n` 是 3g 工况下首个同时通过保持与闭合的仿真
actuator force limit；`calibration: simulation_nominal` 明确它不是物体损伤安全上限，
实机标定前不得改称 maximum allowed force。

最小评分规则为 `quality_score = score_components.template_priority`，四个模板目前各只有
一个已验证模板，故优先级和 quality 均为无量纲 `1.0`。带单位的
`contact_margin_m/opening_margin_m/force_margin_n` 移入 `diagnostics`，只用于报告和
调试，禁止归一化前直接混合相加。它们不含世界点位、候选排名、规划预算或路径成本。
连续 yaw 的 `range_rad` 显式为半开区间 `[-pi, pi)`，因此不会重复首尾姿态。

### 6.4 排除内容

已提交历史没有保存可直接复制的四套完整 object/TCP 位姿对、对象力上限或最终 jaw
position；因此精确模板数值的来源是 6.5 的本轮离线标定，旧提交只提供资产、成功截面
和验收约束。特别是铆枪精确矩阵不是旧提交原值，而是以 140 mm 成功 handle 截面为
约束，在 object-local CoACD 与执行器 TCP 上闭合并由本轮动力学复验所得。

明确排除且未写入模板：对象世界点位、场景半径、世界 yaw、seed、候选编号/索引、
候选数量/预算、IK/路径预算、contact transition/staging 时序、支撑接触豁免、reverse
ingress、collision-planned egress 和任何退出路线。对象局部球/胶囊/盒附着代理也仍
暂缓；旧 `payload_sphere_count=216` 是工具容量/算法参数，不能迁移为对象代理。

### 6.5 离线标定方法与独立验证结果

本轮一次性脚本 `/tmp/task2_template_calibration.py` 只读加载实验副本提交的
`gen3_with_tank.xml`、四类 CoACD 和 `active_wrist_parallel.xml`，在交付目录外输出
`/tmp/task2_template_calibration2.json`；实验副本没有写入。运行环境为
`/home/yoda/miniconda3/envs/mujoco-curobo/bin/python`、Python `3.10.20`、NumPy
`2.2.6`、MuJoCo `3.10.0`，timestep `0.002 s`。脚本仅关闭与当前物体无关的场景
碰撞，保留完整目标 CoACD、左右 pad、物体质量/惯量/摩擦和工具 actuator 参数。

| 模板 | `T_object_tcp`/yaw 来源、frame、方法 | 开度、力、速度、驻留验证 | 接触与评分字段方法 |
|---|---|---|---|
| `marker_parallel_body` | `target_marker` CoACD @ `534ac285...` + TCP @ `fe2fd83...`；object body frame；compiled 截面接触点 `[0.006204,+/-0.006740,0.009419] m` 与 `T_grasp_tcp` 逆变换；`C2` 的 `0/pi` 通过双 pad 扫描，`pi/2` 无有效双侧截面 | 接触 `0.013639206 m`；pregrasp `+0.008 m`；1g 最小 `0.30 N`，3g+闭合首个通过 actuator limit `0.75 N`；`target_opening=contact-2F/2400`；`0.014 m/s`、`0.25 s` 均通过 | 仅 `left_pad/right_pad`；`quality_score=template_priority=1.0`；带单位 margin 仅作 diagnostics |
| `masking_tape_parallel_outer_wall` | `target_tape` CoACD @ `534ac285...` + TCP @ `fe2fd83...`；object body frame；外壁中高 `[0,0,0.0155] m`；连续 `0..2pi` 每 `30 deg` 扫描均保持 `0.075997--0.076000 m` 接触宽度 | 接触 `0.075999999 m`；pregrasp `0.082 m=0.076+0.006`，`0.078 m` 未采用；1g 最小 `0.75 N`，3g+闭合首个通过 actuator limit `2.0 N`；`target_opening=contact-2F/2400`；`0.014 m/s`、`0.25 s` 均通过 | 仅 `left_pad/right_pad`；yaw 区间 `[-pi,pi)`；`quality_score=template_priority=1.0`；带单位 margin 仅作 diagnostics |
| `rivet_gun_parallel_handle` | `target_rivet_gun` CoACD @ `534ac285...`、140 mm 成功截面 @ `354a2b2...`、TCP @ `fe2fd83...`；object body frame；按局部 handle 接触截面反算；`C2` 的 `0/pi` 通过，`pi/2` 无双侧截面 | 双 pad 接触扫描的窄侧为 `0.031946974 m`；pregrasp 为全行程 `0.084 m`；1g 最小 `12 N`，3g+闭合首个通过 actuator limit `24 N`；`target_opening=contact-2F/2400`；`0.014 m/s`、`0.25 s` 通过，闭合期间一侧先接触但 dwell 双侧覆盖为 `1.0` | 仅 `left_pad/right_pad`；`quality_score=template_priority=1.0`；带单位 margin 仅作 diagnostics |
| `standard_part_parallel_shaft` | `target_standard_part` CoACD @ `534ac285...` + TCP @ `fe2fd83...`；object body frame；将接触中心置于 shaft `[0,0.004000,0.002695] m`，两侧表面为 `x=+/-0.001499 m`；`C2` 的 `0/pi` 通过，`pi/2` 虽有接触但 1g 保持失败 | 接触 `0.002995694 m`；pregrasp `+0.008 m`；1g 最小 `0.10 N`，因 jaw frictionloss，3g+闭合首个通过 actuator limit `0.30 N`；`target_opening=contact-2F/2400`；`0.014 m/s`、`0.25 s` 通过 | 仅 `left_pad/right_pad`；`quality_score=template_priority=1.0`；带单位 margin 仅作 diagnostics |

动力学扫描统一使用 `1 s` 保持窗口、平移漂移 `<=3 mm`、角漂移 `<=0.15 rad`、双 pad
覆盖 `>=90%`；闭合验收另使用最终线速度 `<=0.03 m/s`、角速度 `<=0.20 rad/s`。
力候选为 `[0.1,0.2,0.3,0.5,0.75,1,1.5,2,3,4,6,8,12,16,20,24,32,48,64,96,120] N`，
是独立 actuator `forcerange` 扫描，不是旧 preload fraction 乘 `120 N`。最终记录为
`grip_force_limit_n`，含义是 3g 工况下首个同时通过保持与闭合的仿名义 actuator
force limit，不是 maximum allowed force。`target_opening_m` 则是带预压的两指总间距
目标命令，不是首次接触宽度或最终实测开度。所有四个模板的开度、力、速度、驻留和
接触结论都是**仿真名义值**，不是实机测量或对象损伤保证；正式实机标定仍需在运行时
schema 和传感器条件具备后单独完成。

铆枪另做了不含世界点位的 object-local 选择审计：固定 closing axis 为局部 Y、接触
高度 `z=0.014 m`、approach 倾角 `30 deg`，在 handle 的 `x/y` 局部网格上扫描 `18`
个截面。只有两个截面通过 `24 N/3g`，其中 `[-0.018,0.018,0.014] m` 的左右接触
不对称差 `2.367 mm` 小于另一通过点的 `3.588 mm`，且 3g 漂移为
`1.565 mm/0.0612 rad`，因此选择它并以连续碰撞求解细化到 YAML 中的矩阵。该审计只
使用 object-local 坐标，不迁移场景点位、世界 yaw、候选索引或规划策略。

## 7. 删除记录

以下文件只被旧 Task2 CLI/测试/说明消费，删除不会影响 Task1 或 Stage 1/2/3：

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

没有删除 `src/robot_arm_pipeline/`、共享 MuJoCo 资产、Robotiq 资产、Stage 3 BODex
example/adapter 或 MockPlanner。旧 Task2 的源码、规划策略、执行状态机、临时场景、扫描
脚本、视频、报告和 cache 均未从实验副本迁入。

交付目录中被 Git 忽略的旧 Task2 `__pycache__`/`.pyc`、`.pytest_cache`、`.ruff_cache`
和 `MUJOCO_LOG.TXT` 已清除；未删除 Task1 outputs、YOLO model cache、未跟踪的
`examples/mujoco/smc_lehf10/` 或其他与本任务无关的既有内容。

## 8. 验证判据

- 资产：校验上述 SHA-256；解析 XML；MuJoCo 加载工具资产与共享四物体场景；确认
  `wrist_pitch`、两 jaw、两个 actuator、TCP 和左右 pad 存在；验证 `0.084 m` 行程。
- 物理：检查质量/惯量有限且为正；无机械臂静置和自由落体；实测校准仍是残余工作。
- 边界：`rg` 确认新 Task2 没有 BODex、MockPlanner、MockExecutor、Dexonomy、RH56DFX、
  GraspGenX、旧 Stage artifact 或旧 Task2 config 引用。
- 回归：运行现有 pytest，必须保持 Task1 和 Stage 1/2/3 行为；不新增旧入口烟测。
- 工作区：不 `git add`、不提交；保留与任务无关的既有内容。

## 9. 本次验证结果

验证环境为 `/home/yoda/miniconda3/envs/mujoco-curobo/bin/python`：Python `3.10.20`、
NumPy `2.2.6`、MuJoCo `3.10.0`、pytest `9.1.1`。

- XML/URDF：两份文件解析成功；组合 URDF 的活动腕范围、速度、力矩和 TCP 字段与本
  台账一致。
- SO-101 独立 MJCF：加载成功，`nq=3`、`nv=3`、`nu=2`；腕关节、两个 jaw、两个
  actuator、TCP 和左右 pad 均存在；两个 `0.042 m` 单指行程组成 `0.084 m` 总开口。
- 直接资产装配：使用 MuJoCo `MjSpec.attach` 将资产挂到共享
  `gen3_bracelet_link` 后编译成功，组合模型为 `nq=66`、`nv=58`、`nu=9`，带 prefix 的
  腕/jaw/actuator/TCP/pad 名称均可解析。
- 旧 v1 switcher 兼容性未通过：基线 `MujocoEndEffectorSwitcher` 会额外创建
  `task2_tool_tcp`，与资产自带 `tcp` 经 prefix 后的同名 site 冲突并明确失败。实验源码
  通过旧 schema 的 `tcp.asset_site_name` 字段解决，但该字段属于禁止复制的旧 Task2
  adapter 实现。本轮不修改共享 switcher、不删除资产 TCP，也不增加特判；待正式
  Execution adapter schema 建立后再显式绑定资产 site。
- 四物体共享场景：四个 body 均存在，MJCF mass 与 `target_physics.json` 相等，惯量
  有限且为正；被动仿真 10 s 未出现 NaN/Inf。
- 四模板登记：稳定 ID 为 `marker_parallel_body`、
  `masking_tape_parallel_outer_wall`、`rivet_gun_parallel_handle` 和
  `standard_part_parallel_shaft`，四者唯一；均未包含点位、世界 yaw、seed 或候选索引。
- `T_tcp_pregrasp` 独立复算：对三组不同的任意 `T_world_grasp` 计算
  `inv(T_world_tcp_grasp) @ T_world_tcp_pregrasp`，均得到
  `Trans_z(-0.12 m)`，最大矩阵元素残差 `2.22e-16`。
- 闭合速度与驻留扫描：四模板在 `0.014 m/s` 下均完成双 pad 接触，最终线速度
  `<=0.03 m/s`、角速度 `<=0.20 rad/s`；`0.05/0.10/0.15/0.25 s` 扫描均通过，
  模板按已提交成功语义取 `0.25 s`。
- 开度/力扫描：纸胶带接触 `0.075999999 m`、pregrasp `0.082 m`，因此排除 `0.078 m`；
  铆枪窄侧接触 `0.031946974 m`。对象力来自独立 `forcerange` 扫描，未把 preload
  fraction 换算成力；`120 N` 仍只作为工具能力上限。
- 离线模板扫描：完整 CoACD + 左右 pad 的开度、进场、1g/3g 保持、闭合速度、驻留和
  yaw 探针均按 6.5 节判据完成；结果写入 6.3 的 Commit 1 输入。该结果是仿真名义值，
  未修改物体摩擦、质量或工具能力来制造成功；实机力/损伤验证仍是残余风险。
- 依赖扫描：删除后没有旧 Task2 入口/config 悬空引用；新 Task2 运行时目录没有 BODex、
  MockPlanner、MockExecutor、Dexonomy、RH56DFX、GraspGenX 或旧 Stage artifact 引用。
- 回归命令：`PYTHONDONTWRITEBYTECODE=1 .../python -m pytest -q -p no:cacheprovider`；
  结果 `64 passed in 24.96s`。
- 格式检查：`git diff --check` 通过；暂存区为空。
- 实验副本复核：迁移后 `status`/`diff` 摘要仍分别为
  `09dd301b...`/`3384a3bf...`，与迁移前一致。
