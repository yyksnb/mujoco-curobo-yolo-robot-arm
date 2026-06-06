from __future__ import annotations

import argparse
import csv
import json
import time
from importlib.util import find_spec
from pathlib import Path
from typing import Any


DEFAULT_MODEL = Path("examples/mujoco/two_finger_grasp_cube.xml")
PHASES = (
    ("open_gripper", 0.25, (0.0, 0.0, 0.0)),
    ("approach_or_wait", 0.35, (-0.005, 0.0, 0.0)),
    ("close_gripper", 0.7, (-0.005, 0.050, 0.050)),
    ("lift_object", 0.8, (0.115, 0.050, 0.050)),
    ("hold", 0.4, (0.115, 0.050, 0.050)),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a toy MuJoCo two-finger cube grasp smoke demo.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    report = run_demo(args.model, args.output_dir)
    print(
        f"success={report['success']} simulation_ran={report['simulation_ran']} "
        f"lifted_distance={report['lifted_distance']} report={args.output_dir / 'reports' / 'mujoco_physical_grasp_demo_report.json'}"
    )


def run_demo(model_path: Path = DEFAULT_MODEL, output_dir: Path = Path("outputs")) -> dict[str, Any]:
    started = time.perf_counter()
    report_path = output_dir / "reports" / "mujoco_physical_grasp_demo_report.json"
    log_path = output_dir / "logs" / "mujoco_physical_grasp_demo_log.csv"

    if not model_path.exists():
        return _save_report(
            report_path,
            _base_report(False, False, f"MuJoCo model file does not exist: {model_path}", model_path, log_path, started),
        )
    if find_spec("mujoco") is None:
        return _save_report(
            report_path,
            _base_report(False, False, "MuJoCo is not installed. Install it with: pip install mujoco", model_path, log_path, started),
        )

    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        finger_geom_ids = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_geom"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_geom"),
        }

        mujoco.mj_forward(model, data)
        cube_initial_z = float(data.xpos[cube_body_id][2])
        rows: list[dict[str, object]] = []
        contact_detected = False
        step_index = 0

        for phase_name, duration_s, ctrl in PHASES:
            steps = max(1, int(duration_s / model.opt.timestep))
            for _ in range(steps):
                data.ctrl[:] = ctrl
                mujoco.mj_step(model, data)
                contact_detected = contact_detected or _has_cube_finger_contact(data, cube_geom_id, finger_geom_ids)
                if step_index % 10 == 0:
                    rows.append(_log_row(model, data, phase_name, step_index, cube_body_id))
                step_index += 1

        cube_final_z = float(data.xpos[cube_body_id][2])
        lifted_distance = cube_final_z - cube_initial_z
        success = bool(contact_detected and lifted_distance > 0.015)
        if success:
            message = "Toy MuJoCo physical grasp demo lifted the cube."
        else:
            message = (
                "Simulation ran, but the toy gripper did not meet the lift threshold. "
                "Next tune actuator gains, friction, contact geometry, or gripper shape."
            )

        _save_log(log_path, rows)
        return _save_report(
            report_path,
            {
                "success": success,
                "simulation_ran": True,
                "message": message,
                "cube_initial_z": round(cube_initial_z, 6),
                "cube_final_z": round(cube_final_z, 6),
                "lifted_distance": round(lifted_distance, 6),
                "contact_detected": contact_detected,
                "num_steps": step_index,
                "duration_s": round(step_index * float(model.opt.timestep), 6),
                "model_path": str(model_path),
                "log_path": str(log_path),
                "is_demo_model": True,
                "elapsed_wall_time_s": round(time.perf_counter() - started, 6),
            },
        )
    except Exception as exc:  # pragma: no cover - protects against external MuJoCo runtime differences.
        return _save_report(
            report_path,
            _base_report(False, False, f"MuJoCo physical grasp demo failed: {type(exc).__name__}: {exc}", model_path, log_path, started),
        )


def _has_cube_finger_contact(data: Any, cube_geom_id: int, finger_geom_ids: set[int]) -> bool:
    for index in range(data.ncon):
        contact = data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if cube_geom_id in pair and pair.intersection(finger_geom_ids):
            return True
    return False


def _log_row(model: Any, data: Any, phase_name: str, step_index: int, cube_body_id: int) -> dict[str, object]:
    return {
        "time": round(float(data.time), 6),
        "step": step_index,
        "phase": phase_name,
        "qpos": json.dumps([float(value) for value in data.qpos]),
        "qvel": json.dumps([float(value) for value in data.qvel]),
        "ctrl": json.dumps([float(value) for value in data.ctrl]),
        "cube_position": json.dumps([float(value) for value in data.xpos[cube_body_id]]),
        "gripper_state": json.dumps(
            {
                "lift_z": _joint_qpos(model, data, "lift_z"),
                "left_finger": _joint_qpos(model, data, "left_finger_slide"),
                "right_finger": _joint_qpos(model, data, "right_finger_slide"),
            }
        ),
    }


def _joint_qpos(model: Any, data: Any, joint_name: str) -> float:
    import mujoco

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return 0.0
    return float(data.qpos[model.jnt_qposadr[joint_id]])


def _save_log(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(
            log_file,
            fieldnames=("time", "step", "phase", "qpos", "qvel", "ctrl", "cube_position", "gripper_state"),
        )
        writer.writeheader()
        writer.writerows(rows)


def _save_report(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _base_report(
    success: bool,
    simulation_ran: bool,
    message: str,
    model_path: Path,
    log_path: Path,
    started: float,
) -> dict[str, Any]:
    return {
        "success": success,
        "simulation_ran": simulation_ran,
        "message": message,
        "cube_initial_z": 0.0,
        "cube_final_z": 0.0,
        "lifted_distance": 0.0,
        "contact_detected": False,
        "num_steps": 0,
        "duration_s": 0.0,
        "model_path": str(model_path),
        "log_path": str(log_path),
        "is_demo_model": True,
        "elapsed_wall_time_s": round(time.perf_counter() - started, 6),
    }


if __name__ == "__main__":
    main()
