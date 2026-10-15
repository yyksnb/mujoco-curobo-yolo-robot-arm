from pathlib import Path
import mujoco
import mujoco.viewer

# 加载你的 XML
model_path = Path(__file__).resolve().parent / "scene.xml"
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

# 打开可视化窗口
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
