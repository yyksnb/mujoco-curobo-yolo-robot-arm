# Task1 内部实现

## 模块边界

- `pipeline.py`：阶段编排和 artifact 交接，不实现检测、定位或规划。
- `layout.py`：从 MuJoCo 目标物体模型生成无碰撞布局。
- `survey/route.py`：固定 16 个相机位、离线路线 artifact 及输入指纹校验。
- `survey/detection.py`：正式 YOLO 适配和中文检测框图。
- `survey/localization.py`：RGB-D 底面定位和多视角融合。
- `survey/manifest.py`：实机 RGB-D manifest 输入。
- `survey/yolo_evaluation.py`：仿真评估旁路，不参与候选生成。
- `simulation/mujoco.py`：1080p RGB-D 拍摄和仿真真值生成。
- `robot_arm_pipeline.planning`：共享 cuRobo camera-route planning；Task1 不依赖其内部参数。

生产模块只消费正式输入。seed、layout 真值、segmentation 和 benchmark 归属不得进入检测、
定位或融合决策。

## Survey 实现

日常运行校验并加载 `configs/task1/survey_route_plan.json`，不在线调用 cuRobo。scene、机器人
配置、URDF、碰撞网格、固定起点或相机位变化后，必须显式重新生成路线。

MuJoCo 在 16 个路线终点拍摄 1920x1080 RGB-D。RGB 和中文框图落盘；depth 默认只在内存
使用，`--retain-survey-depth` 显式开启后才保存。

定位从检测框、同帧深度、相机内参和 `T_world_camera_optical` 生成底面位置、footprint、
表面协方差和软类别证据。融合在不同 view 对之间做一对一匹配，再构建全局观测图：

- 边代价由 footprint 重叠、Mahalanobis 距离和软类别证据组成。
- 同一轨迹禁止包含两个相同 view 的观测。
- 至少两个不同 view 支持才输出候选。
- 不按物体数量补齐、裁剪或重排，不使用类别或 seed 特判。

关联边、gate 统计、互斥拒绝和 tentative track 写入 `fusion_diagnostics`。

## 评估

`configs/task1/survey_benchmark.json` 固定互不重叠的 development、regression 和 acceptance
seed。它只记录评估集合，不作为生产 CLI。候选真值验收结果：

- development：22/22 通过。
- regression：两个融合回归通过；两个 YOLO 单视角支持样本明确失败。
- acceptance：23/25 通过；两个失败均为 YOLO 单视角支持不足。

重构后 51 个 seed 回测结果与冻结基线一致，无 cuRobo 或融合失败。

MuJoCo 顶层 `status` 只表示路线执行、拍照、感知流程和报告生成正常完成；单个检测的定位
失败继续显式写入 `localization_failures`。layout 真值验收仅写入
`candidate_position_evaluation.success`；benchmark 据此统计质量，后续生产阶段不得读取该
评估字段决定控制流。

## 风险与非主路径

- 更换 YOLO 模型、相机或场景分布后必须重新评估检测与融合参数。
- 同视角重复框缺少跨视角证据时保持不确定，不猜测合并。
- MuJoCo 与实机存在 domain gap；实机还依赖双目相机标定和手眼标定质量。
- 工程防御仅包含 schema、路线指纹、视角完整性、深度和矩阵校验；触发结果显式写入报告。
- segmentation、理想物体中心和 benchmark 统计仅用于 debug/评估。
