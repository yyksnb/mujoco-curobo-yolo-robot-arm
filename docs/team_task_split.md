# 团队任务分工

## 总体边界

YOLO 和 BODex 是上游模块。本仓库负责把它们的输出转成稳定的机械臂规划控制输入。每个小组可以维护自己的 raw format，但进入主流程前必须转换成本仓库的 canonical contract。

## 王奥 + 刘帅彤：YOLO 识别

任务：

- 在 MuJoCo 中通过机械臂摄像头拍照。
- 使用 YOLO 或后续识别模块输出物体检测结果。
- 先输出 `class_name`、`confidence`、`bbox_xyxy` 即可。

交付物：

- raw YOLO 输出样例，类似 `examples/yolo_raw_output_sample.json`。
- 如果 raw format 和样例不同，新增 adapter/converter。
- 说明图片来源、`camera_name`、坐标系假设。

关键字段：

- `image_path`
- `camera_name`
- `detections[].class_name`
- `detections[].confidence`
- `detections[].bbox_xyxy`
- 可选：`detections[].object_id`
- 可选：`detections[].T_world_object`

接口边界：

- raw YOLO bbox 不能直接进入 cuRobo planning。
- Stage 3 planning contract 必须包含 `T_world_object`。
- 没有真实 6D pose 时，可以由 MuJoCo ground truth 或 mock pose 补齐，并明确标注。

## 刘帅彤：物体/场景准备

任务：

- 寻找物体三维模型。
- 导入 MuJoCo。
- 记录物体 ID、类别、尺寸、缩放、质量和初始位姿。

交付物：

- 物体资产清单。
- MuJoCo XML/MJCF 片段或模型路径。
- object metadata 文档或 JSON。

关键字段：

- `object_id`
- `class_name`
- `asset_path`
- `mesh_format`
- `size`
- `scale`
- `mass`
- `initial_pose`

接口边界：

- 物体模型资产不是 YOLO 输出。
- 物体位姿要能映射到 `T_world_object`。
- 大型 mesh、纹理、数据集不要直接提交 Git；先和团队确认 Git LFS 或外部存储方案。

## 邓弘烨：机械臂/仿真器准备

任务：

- 寻找合适机械臂模型放入 MuJoCo。
- 配置机械臂相机。
- 明确 robot XML/MJCF、相机名和机器人关节命名。

交付物：

- MuJoCo robot XML/MJCF 或引用路径。
- 相机配置说明。
- `base_link`、`ee_link`、`joint_names` 清单。

关键字段：

- `robot_xml`
- `camera_name`
- `camera_frame`
- `base_link`
- `ee_link`
- `joint_names`

接口边界：

- MuJoCo joint order 必须能和 planner 的 `RobotState.joint_names` 对齐。
- camera frame 到 world frame 的转换必须可追踪。
- 不要在没有说明的情况下改 Stage 3 canonical contract。

## 何伟铭 + 邓弘烨 + 杨雅凯：规划控制组

任务：

- 构建 collision scene。
- 接入 cuRobo planner。
- 执行轨迹和控制接口。
- 在 MuJoCo 中验证轨迹和抓取流程。

交付物：

- planner adapter / controller adapter。
- 轨迹执行报告。
- MuJoCo 验证脚本和报告。
- cuRobo 环境检查报告。

关键字段：

- `CollisionScene`
- `RobotState`
- `GraspTarget`
- `PlannedTrajectory`
- `ExecutionResult`
- `EvaluationResult`

接口边界：

- 规划控制组消费 Stage 3 canonical contract，不消费 YOLO/BODex raw format。
- cuRobo 不可用时必须 graceful failure，不破坏 Stage 1/2/3。
- mock/stub 必须明确标注，不冒充真实规划或真实抓取。

