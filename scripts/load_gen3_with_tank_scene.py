from __future__ import annotations

import argparse
from importlib.util import find_spec
from pathlib import Path
import time


DEFAULT_MODEL = Path("examples/mujoco/gen3_with_tank.xml")
DEFAULT_KEYFRAME = "gen3_home"


def main() -> None:
    parser = argparse.ArgumentParser(description="Load the Kinova Gen3 with tank MuJoCo scene.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to the scene MJCF/XML file.")
    parser.add_argument("--keyframe", default=DEFAULT_KEYFRAME, help="Initial keyframe to apply before displaying the scene.")
    parser.add_argument(
        "--kinematic",
        action="store_true",
        help="Directly map GUI control values to joint positions without physics or collision response.",
    )
    parser.add_argument("--pause", action="store_true", help="Open the GUI without updating the scene.")
    parser.add_argument("--no-gui", action="store_true", help="Load and print scene stats without opening the MuJoCo viewer.")
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"MuJoCo scene file does not exist: {args.model}")
    if find_spec("mujoco") is None:
        raise SystemExit("MuJoCo is not installed. Install it with: pip install mujoco")

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    _apply_keyframe(mujoco, model, data, args.keyframe)
    mujoco.mj_forward(model, data)

    tank_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tank")
    gen3_base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gen3_base_link")

    print(f"loaded={args.model}")
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody} ngeom={model.ngeom} nmesh={model.nmesh}")
    print(f"tank_pos={_body_position(data, tank_body_id)}")
    print(f"gen3_base_pos={_body_position(data, gen3_base_body_id)}")

    if not args.no_gui:
        if find_spec("mujoco.viewer") is None:
            raise SystemExit("MuJoCo viewer is not available in this environment.")

        import mujoco.viewer

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                if args.pause:
                    pass
                elif args.kinematic:
                    ctrl_count = min(model.nu, model.nq)
                    if ctrl_count:
                        data.qpos[:ctrl_count] = data.ctrl[:ctrl_count]
                    if model.nv:
                        data.qvel[:] = 0.0
                    mujoco.mj_forward(model, data)
                else:
                    mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(float(model.opt.timestep))


def _apply_keyframe(mujoco, model, data, keyframe_name: str) -> None:
    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
    if keyframe_id < 0:
        return
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)


def _body_position(data, body_id: int) -> list[float] | None:
    if body_id < 0:
        return None
    return [round(float(value), 6) for value in data.xpos[body_id]]


if __name__ == "__main__":
    main()
