# Task1 内部实现

## 模块边界

- `pipeline.py` 只编排 `layout -> survey -> final -> zoom` 和正式 artifact 交接。
- `scene.py`、`detection.py`、`vision.py` 提供共享的坐标变换、YOLO 适配和 RGB-D 定位/融合接口。
- `survey/` 负责固定路线、Survey 配置、采集和评估。
- `final/processing.py`、`simulation.py`、`evaluation.py` 分别负责生产处理、MuJoCo 采集和评估。
- `zoom/processing.py` 只消费 Final 正式结果，负责数字裁剪、放大和独立报告。
- `replay.py` 只读消费一次运行的 layout、报告和已执行 cuRobo 轨迹，在 MuJoCo GUI 中回放。
- Task1 只依赖 `PoseRoutePlanner` 的 pose、批量 IK 和轨迹正式接口；真实 cuRobo runtime 由共享
  `robot_arm_pipeline.planning.CuroboPlanner` 提供。Task1 将开口方向策略映射为通用的 Cartesian
  continuation，不向共享规划器传入候选、类别、seed 或评估真值。

生产模块只消费正式输入。Final 只解析候选的 `candidate_id`、`bottom_position_world` 和
`footprint_polygon_xy`；编排层另读取正式的 `planner_artifact` 作为起始关节状态。seed、layout
真值、segmentation、benchmark split 和 Survey 内部诊断字段不得进入生产决策。参数按阶段集中在
`configs/task1/survey/`、`configs/task1/final/` 和 `configs/task1/zoom/`，不散落在业务源码中。

Survey 和 Final 报告使用相同的前部层次：`schema/stage/status/failure_stage/message`、数量摘要、
主结果集合、逐项结果，再放来源、策略、artifact、诊断和评估。Survey 的主结果是 `candidates`，
Final 的主结果是 `stable_objects`；保留各自单一正式字段，不增加指向同一数据的别名。

两个阶段都将 artifact 分成三层：`*_report.json` 只保存生产状态、正式结果、简要逐项状态和引用；
`*_diagnostics.json` 保存逐帧检测、定位、融合、IK 和规划尝试；`*_evaluation.json` 保存仿真真值评估。
主报告只记录 `diagnostics_path`、`evaluation_status` 和 `evaluation_path`，诊断或评估内容不重复内嵌。

## Survey

Survey 固定使用 16 个相机位，日常运行只校验并加载带输入指纹的离线路线，不在线调用 cuRobo。
scene、基座位姿、机器人配置、URDF、碰撞网格、固定起点或相机位变化后，必须显式重新生成路线。

MuJoCo 在各路线终点采集 1920x1080 RGB-D。RGB 原图用于正式检测，中文框图是展示 artifact；框图
生成失败会显式记录，但不丢弃检测、定位或融合结果。深度默认仅在内存使用，显式开启诊断时才保存。

正式链路为 `YOLO -> 同帧深度定位 -> 多视角融合`。定位使用检测框、深度、相机内参和
`T_world_camera_optical`，生成底面位置、footprint、表面协方差和软类别证据。融合先在不同 view
之间做一对一匹配，再构建全局观测图：

- 边代价由 footprint 重叠、Mahalanobis 距离和软类别证据组成。
- 同一 track 禁止包含两个相同 view 的观测。
- 至少两个不同 view 支持才输出候选。
- 不按目标数量补齐、裁剪或重排，不使用类别、seed 或样例特征做特判。

关联边、gate 统计、互斥拒绝和 tentative track 写入 `survey_diagnostics.json` 的
`fusion_diagnostics`。定位失败也在该诊断 artifact 中显式记录；layout 位置验收不改变生产
`status`。

## Final

Final 处理 Survey 的全部候选，不要求输入数量恰好为 5。首个目标从 Survey 路线末端状态开始；
后续目标串接最近一次可执行轨迹终态，并依据当前关节状态和可行 IK 选择处理顺序。单个候选失败不
阻止其余候选，报告仍按输入顺序输出，实际顺序另行记录。

相机位于候选瞄准点与油箱开口的连线上。各 optical roll 根据 footprint、FOV 和 coverage margin
计算完整入框距离，并搜索多个 standoff。所有姿态先批量执行碰撞感知 IK；可行解按预计画面占比、
关节距离和拍摄距离排序，仅必要姿态进入完整运动规划。

同一候选的直接 pose 规划全部失败后，Final 才进入可配置的 portal continuation：先沿相机局部
`+Z`（即开口方向）退到 portal pose；各 portal 先批量 IK，并按当前关节距离排序但不裁剪。cuRobo
到达 portal 后选择全局可达 IK 分支，再以固定空间步长沿同一分支向目标做碰撞连续 IK，并由独立
的 cspace trajopt 对局部段做时间参数化。两段必须满足关节端点连续、拼接处停止、逐点/边碰撞
检查和最终相机位姿容差，否则保持规划失败。

到位后重新执行 YOLO 和同帧深度定位，按候选 XY 距离关联检测；`bbox_area_fraction` 只作诊断。
顶层 `status` 由各候选处理结果决定，物体数量由 `candidate_count_evaluation` 独立表达。框图属于展示
artifact，生成失败不丢弃正式检测和位姿结果。Final 在 spawn 子进程中运行，隔离 MuJoCo/OpenGL、
YOLO CUDA 和 cuRobo native runtime。

## Zoom

Zoom 只读取 `task1_final_report` 的正式 `stable_objects`，并用简要 `results` 判断候选是否成功；
不读取 Final 的规划尝试、检测诊断、深度、layout、segmentation 或评估真值，也不重新运行 YOLO。
Final 失败候选在 Zoom 报告中显式标记为 `skipped`，不会补图或改写上游结果。

对每个 bbox，Zoom 分别计算横屏和竖屏的固定比例裁剪框。裁剪必须完整包含可见 bbox，并优先满足
配置的最小留白；在所有可行方向中选择输出 bbox 面积占比最接近 60% 的方案。原始 RGB 裁剪后以
LANCZOS 放大；横屏裁剪直接输出 1920x1080，竖屏裁剪先放大为 1080x1920，再顺时针旋转 90°，因此
最终图片统一为 1920x1080。中文框图使用旋转后的正式 bbox 重新生成。裁剪方向、旋转角度、裁剪框、
输出 bbox、实际占比、留白和源 bbox 是否触边均写入 `zoom_report.json`。

Zoom 是非门控展示阶段：Final 为 `partial/failed` 但已落盘合法报告时，完整 pipeline 仍对其中成功
候选执行 Zoom，同时顶层继续保留 `failed_stage=final`。原图损坏、缺失或裁剪失败会使 Zoom 报告为
`partial/failed`，但不会改写上游状态；输入报告 schema 错误仍作为调用错误抛出。

## Replay

Replay 不属于生产识别阶段，不重新运行 YOLO、深度定位或 cuRobo。它只读取 layout、Survey/Final
报告和 `MotionPlanResult` 正式字段，不依赖两个阶段的内部配置。Survey 完整轨迹按报告顺序回放；Final
再按 `candidate_processing_order` 读取各结果顶层 `planner_artifact`，失败规划尝试和没有可执行轨迹的
候选不会被补路。

用户 CLI 只保留 `--run-dir` 和 `--continuous`。运行
`python scripts/replay_task1_pipeline.py --run-dir <run_dir>`，单个 2:1 GUI 窗口并排显示两个 1:1 视角；
右侧将初始 MuJoCo 相机画面和 16:9 框图按中心裁剪为 1:1，不叠加文字，原始图片 artifact 不变。
框图缺失时只在左侧状态栏明确报告，不回退到未标注 RGB。

Space 暂停或继续；F9 按记录轨迹播放到下一次拍照并停下，不跳过中间运动；F12 从头重置。默认逐拍
暂停，传入 `--continuous` 后固定以 2 倍记录速度连续播放。运动画面按 `trajectory_time_s` 插值采样；
渲染赶不上时跳过过期的显示采样，而不是延长轨迹或直接跳到终点。Replay 只读现有 artifact，不向运行
目录写入 manifest 或新的业务报告。

## 评估边界

Survey 和 Final 的生产拍照都只生成 RGB-D 和正式相机变换，并先原子落盘生产报告及独立诊断；仿真
随后回放已拍摄关节状态，采集 segmentation 和评估所需真值。layout、segmentation、真值 bbox 和
benchmark 聚合只用于独立 `*_evaluation.json`，不参与生产控制。两个阶段都在 spawn 子进程中运行；
评估异常或 native 进程崩溃只更新主报告的评估状态并写评估错误，不覆盖已经持久化的生产 `status`。

## 逻辑分类

- 正式设计：多视角 Survey 融合、开口连线斜拍、roll-aware 投影定距、批量碰撞 IK、多 standoff、
  portal continuation、逐候选重检测和定位，以及基于 Final 正式 bbox 的横/竖屏数字变焦。
- 工程防御：schema/指纹校验、逐阶段失败、生产报告原子落盘、native runtime 与评估故障隔离；
  portal 路线额外校验关节端点、停止速度和碰撞；Replay 框图缺失时显式显示不可用，不回退到
  未标注 RGB；Zoom 保留完整可见 bbox、记录触边/留白并隔离框图生成失败。
- Debug/评估：layout 真值、segmentation、真值 bbox、诊断深度、benchmark 聚合和 artifact Replay。
- 临时 workaround：无；当前没有仅为样例通过而引入或计划删除的生产逻辑。

## 未解决风险

- Survey 仍依赖 YOLO 提供足够的跨视角观测；单视角支持会明确漏掉候选。
- Final footprint 不表达物体高度、遮挡和油箱口可见性，完整入框距离仍是平面近似。
- coverage margin 尚未覆盖实机内参、手眼标定和机械臂执行误差，MuJoCo 与实机存在 domain gap。
- 当前 YOLO 对小物体、旋转和斜视角敏感；模型、相机或场景分布变化后必须重新评估正式参数。
- portal continuation 只在全部直接姿态失败后运行；极端深腔下可能额外尝试多个 roll，规划时间会
  明显增加，且 portal 可达分支仍可能无法连续进入目标。
