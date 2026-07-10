# Task1 Policy

本文档记录 Task1 pipeline 的阶段契约和收敛原则。目标是减少逐 seed 打补丁，优先让每个阶段的成功、失败和降级语义保持真实。

## 总原则

- 宁可输出 `partial` 或明确失败，也不要把低质量结果包装成稳定成功。
- 每个阶段只能依赖正式输入接口，不能读取上游内部调参、layout 真值或下游结果来修正当前判断。
- 不按 seed、类别名、路径片段或当前样例特征写隐藏特判。
- 新增策略必须进入集中 policy/config，并在 report、notes 或 summary 中可追踪。
- Debug 输出只能辅助定位，不能改变正式结果。

## 阶段契约

### Layout

职责：生成或读取目标物体初始摆放。

正式输出：目标物体真值 layout JSON。

边界：

- Layout 是仿真输入，不是识别结果。
- 下游评估可用 layout 对照质量，但正式识别流程不能用 layout 补齐或纠错。

### Survey

职责：以 recall-first 的方式生成候选物体入口。

正式输入：layout 场景、相机配置、YOLO 原始检测、深度投影。

正式输出：`candidates`，每个候选包含粗略位置、支持视角、最佳图像、最佳 bbox 和 advisory class votes。

边界：

- Survey 不输出最终类别、最终数量、yaw 或 6D pose。
- Survey candidates 可以多、少、重复或混叠。
- 如果 YOLO 完全没有给出可用观测，Survey 不应凭空生成候选。

失败语义：

- 候选不足是上游 recall 风险，应显式保留在 report 中。
- 近邻小目标被大物体融合时，只能基于真实多视角、小 bbox、空间紧聚和可追踪 policy 拆分，不能按类别名特判。

### Rough

职责：消费 Survey candidates，进行近距离拍摄并形成对象假设。

正式输入：`survey_report["candidates"]`。

正式输出：

- `stable_objects`：Final 的主输入。
- `tentative_objects`：需要 Final 补确认的目标。
- `ambiguous_objects`：类别或近邻关系不确定，需要 Final 补确认的目标。
- `object_selection_summary`：候选过滤、合并、拒绝和降级原因。

边界：

- Rough 可以重新分裂 Survey 混叠候选，但不能为了凑数量伪造对象。
- Rough 的 stable 仍是“可进入精拍确认”的对象，不是最终全流程真值。
- 对某个 Survey candidate 没有可达近拍视角时，应明确报告 `insufficient_reachable_views` 或同等原因。

失败语义：

- 可达性失败是 pipeline 规划问题，不应被 YOLO 分类问题掩盖。
- 低置信或单视角小目标应进入 tentative，而不是静默丢弃或硬升 stable。

### Final

职责：对 Rough 对象逐个做单物体精拍摄，并给出稳定对象列表。

正式输入：

- Primary targets: `rough_report["stable_objects"]`
- Follow-up targets: `rough_report["tentative_objects"] + rough_report["ambiguous_objects"]`

正式输出：

- `stable_objects`：下游正式对象列表。
- `unstable_objects`：已观察但未满足稳定策略的对象。
- capture/report summary：视角候选、IK/collision、确认和拒绝原因。

边界：

- Final 不应默认补齐到固定数量。
- `expected_object_count_hint` 只能作为 task1 的 follow-up 优化和过量弱证据审查 hint，不能作为 `stable_objects` 必须等于该数量的契约。
- Final 只能确认真实拍到、类别一致、位置合理、bbox 质量和 evidence quality 都合格的对象。
- Follow-up 目标只有在证据足够时才能升级为 stable；否则应保留 unstable/partial。
- 如果存在多个同类或近邻观测，选框策略不能只按 bbox 面积或单一距离决定，应兼顾类别一致、目标距离、边界安全、bbox 完整性和置信度。
- Follow-up 搜索可以设置候选尝试预算，但预算耗尽必须写入 report，且只能产生 unstable/partial 结果。
- 每个 Final candidate 的 entry path 必须从确定的机器人初始姿态开始验证，不能继承前一个失败或未确认 candidate 留下的关节状态。

失败语义：

- Final entry 或最终拍摄姿态不可达时，应输出明确失败原因，不应跳过后伪装成功。
- 低置信、局部框、分片合并框可以记录观察结果，但不应轻易升级为 stable。
- 单个低置信观测只有在位置严格贴近且达到配置化置信下限时才可作为确认；多个分片合并观测需要更强证据。
- 过量 stable 的处理只能基于正式质量规则降级弱证据，不能硬裁剪到目标数量。

### Zoom

职责：对 Final stable 对象的精拍图做数字裁剪和放大。

正式输入：`final_report["stable_objects"]`。

正式输出：zoom 图片和 zoom report。

边界：

- Zoom 不改变对象身份、类别、位置或稳定性。
- Zoom 不能修复 Final 的不完整 bbox；如果 source bbox 已经是局部框，Zoom 只能报告受限。
- Zoom status 不应被用于判断识别是否成功。

失败语义：

- 如果目标比例无法达到，应输出 `zoomed_limited` 或同等状态。
- 如果裁剪会破坏 bbox 边界安全，应选择更保守候选或明确受限。

## 固定回归集

逐对象回归归因只消费 layout、Survey、Rough、Final 的正式 JSON。对象身份按 layout footprint 做一对一空间匹配，类别正确性单独判断。raw YOLO 只有二维框，因此 raw 类别命中只能作为类别级证据；没有匹配到世界坐标观测时，不得宣称它是某个真值对象的正确检测。

后续不要只看随机 25 seed 的总体通过率。每次改策略前后，应至少观察以下代表类问题：

- Rough 可达性不足：`seed8475`、`seed5080`
- Final entry 不可达：`seed5337`
- Final 选框被大框带偏：`seed9690`
- 大物体局部框进入 stable 后导致 Zoom 观感差：`seed6534`、`seed6874`、`seed9426`、`seed6213`

验收重点不是所有图片都完美，而是：

- 该失败是否被归到正确阶段。
- report 是否能解释失败原因。
- 不确定对象是否保持不确定语义。
- 新策略是否没有把其他阶段的正式职责吞掉。

## 新策略准入

新增策略前先回答：

- 这是正式策略、工程防御、临时 workaround，还是 debug 辅助？
- 解决的失败模式是什么？
- 依赖哪些假设？
- 会不会掩盖真实失败？
- 如何配置、关闭和在 report 中看见？
- 哪些测试能守住正式能力边界？

默认拒绝：

- 为固定数量补齐或裁剪对象。
- 按 seed、类别名或文件路径特判。
- 捕获异常后继续输出成功。
- 把 tentative/ambiguous 直接包装成 stable。
- 让 Zoom 结果反向影响识别语义。

## 当前未解决风险

- Rough 对部分 Survey candidates 仍可能找不到可达近拍视角，导致后续 Final 没有目标可确认。
- Final 已收敛 stable evidence gate，但近邻小目标和大物体局部框仍需要继续用回归集观察。
- Final 的 primary target 候选搜索仍可能偏重，需要继续在性能与 recall 之间收敛。
- YOLO 对 notebook、standard_part 等目标的局部框和分片框会放大 pipeline 判断压力。
