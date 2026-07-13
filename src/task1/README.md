# Task1 内部实现

## 模块边界

- `pipeline.py` 只编排 `layout -> survey -> final` 和正式 artifact 交接。
- `scene.py`、`detection.py`、`vision.py` 提供共享的坐标变换、YOLO 适配和 RGB-D 定位/融合接口。
- `survey/` 负责固定路线、Survey 配置、采集和评估。
- `final/processing.py`、`simulation.py`、`evaluation.py` 分别负责生产处理、MuJoCo 采集和评估。
- cuRobo camera-route planning 保持在 `robot_arm_pipeline.planning`。

生产模块只消费正式输入。Final 只解析候选的 `candidate_id`、`bottom_position_world` 和
`footprint_polygon_xy`；编排层另读取正式的 `planner_artifact` 作为起始关节状态。seed、layout
真值、segmentation、benchmark split 和 Survey 内部诊断字段不得进入生产决策。参数按阶段集中在
`configs/task1/survey/` 和 `configs/task1/final/`，不散落在业务源码中。

## Survey

Survey 固定使用 16 个相机位，日常运行只校验并加载带输入指纹的离线路线，不在线调用 cuRobo。
scene、基座位姿、机器人配置、URDF、碰撞网格、固定起点或相机位变化后，必须显式重新生成路线。

MuJoCo 在各路线终点采集 1920x1080 RGB-D。RGB 和中文框图落盘；深度默认仅在内存使用，显式开启
诊断时才保存。

正式链路为 `YOLO -> 同帧深度定位 -> 多视角融合`。定位使用检测框、深度、相机内参和
`T_world_camera_optical`，生成底面位置、footprint、表面协方差和软类别证据。融合先在不同 view
之间做一对一匹配，再构建全局观测图：

- 边代价由 footprint 重叠、Mahalanobis 距离和软类别证据组成。
- 同一 track 禁止包含两个相同 view 的观测。
- 至少两个不同 view 支持才输出候选。
- 不按目标数量补齐、裁剪或重排，不使用类别、seed 或样例特征做特判。

关联边、gate 统计、互斥拒绝和 tentative track 写入 `fusion_diagnostics`。定位失败显式写入报告；
layout 位置验收不改变生产 `status`。

## Final

Final 处理 Survey 的全部候选，不要求输入数量恰好为 5。首个目标从 Survey 路线末端状态开始；
后续目标串接最近一次可执行轨迹终态，并依据当前关节状态和可行 IK 选择处理顺序。单个候选失败不
阻止其余候选，报告仍按输入顺序输出，实际顺序另行记录。

相机位于候选瞄准点与油箱开口的连线上。各 optical roll 根据 footprint、FOV 和 coverage margin
计算完整入框距离，并搜索多个 standoff。所有姿态先批量执行碰撞感知 IK；可行解按预计画面占比、
关节距离和拍摄距离排序，仅必要姿态进入完整运动规划。

到位后重新执行 YOLO 和同帧深度定位，按候选 XY 距离关联检测；`bbox_area_fraction` 只作诊断。
顶层 `status` 由各候选处理结果决定，物体数量由 `candidate_count_evaluation` 独立表达。框图属于展示
artifact，生成失败不丢弃正式检测和位姿结果。Final 在 spawn 子进程中运行，隔离 MuJoCo/OpenGL、
YOLO CUDA 和 cuRobo native runtime。

## 评估边界

Final 生产拍照只生成 RGB-D 和正式相机变换，并先原子落盘生产报告；仿真随后回放已拍摄关节状态，
采集 segmentation 和实际相机位姿。layout、segmentation、真值 bbox 和 benchmark 聚合只用于评估，
不参与生产控制。评估异常只写评估错误，不覆盖生产 `status`。

## 逻辑分类

- 正式设计：多视角 Survey 融合、开口连线斜拍、roll-aware 投影定距、批量碰撞 IK、多 standoff、
  逐候选重检测和定位。
- 工程防御：schema/指纹校验、逐阶段失败、生产报告原子落盘、native runtime 与评估故障隔离。
- Debug/评估：layout 真值、segmentation、真值 bbox、诊断深度和 benchmark 聚合。
- 临时 workaround：无；当前没有仅为样例通过而引入或计划删除的生产逻辑。

## 未解决风险

- Survey 仍依赖 YOLO 提供足够的跨视角观测；单视角支持会明确漏掉候选。
- Final footprint 不表达物体高度、遮挡和油箱口可见性，完整入框距离仍是平面近似。
- coverage margin 尚未覆盖实机内参、手眼标定和机械臂执行误差，MuJoCo 与实机存在 domain gap。
- 当前 YOLO 对小物体、旋转和斜视角敏感；模型、相机或场景分布变化后必须重新评估正式参数。
