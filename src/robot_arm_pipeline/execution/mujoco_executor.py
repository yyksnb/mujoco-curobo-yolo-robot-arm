from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MujocoExecutionReport:
    success: bool
    trajectory_path: str
    model_path: str
    duration_s: float
    num_steps: int
    message: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _ResolvedJointControl:
    trajectory_joint_name: str
    model_joint_name: str
    joint_id: int
    qpos_index: int
    qvel_index: int
    actuator_id: int | None
    actuator_name: str | None


class MujocoExecutor:
    name = "mujoco_executor"

    def __init__(
        self,
        output_dir: Path | str = Path("outputs"),
        *,
        keyframe_name: str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.keyframe_name = keyframe_name

    @staticmethod
    def is_available() -> bool:
        return find_spec("mujoco") is not None

    def execute(
        self,
        trajectory_path: Path | str,
        model_path: Path | str,
    ) -> MujocoExecutionReport:
        trajectory_path = Path(trajectory_path)
        model_path = Path(model_path)

        if not trajectory_path.exists():
            return self._save_report(
                MujocoExecutionReport(
                    success=False,
                    trajectory_path=str(trajectory_path),
                    model_path=str(model_path),
                    duration_s=0.0,
                    num_steps=0,
                    message=f"trajectory file does not exist: {trajectory_path}",
                )
            )

        if not model_path.exists():
            return self._save_report(
                MujocoExecutionReport(
                    success=False,
                    trajectory_path=str(trajectory_path),
                    model_path=str(model_path),
                    duration_s=0.0,
                    num_steps=0,
                    message=f"MuJoCo model file does not exist: {model_path}",
                )
            )

        if not self.is_available():
            return self._save_report(
                MujocoExecutionReport(
                    success=False,
                    trajectory_path=str(trajectory_path),
                    model_path=str(model_path),
                    duration_s=0.0,
                    num_steps=0,
                    message="MuJoCo is not installed. Install it with: pip install mujoco",
                )
            )

        try:
            return self._execute_with_mujoco(trajectory_path, model_path)
        except Exception as exc:  # pragma: no cover - depends on external MuJoCo runtime details.
            return self._save_report(
                MujocoExecutionReport(
                    success=False,
                    trajectory_path=str(trajectory_path),
                    model_path=str(model_path),
                    duration_s=0.0,
                    num_steps=0,
                    message=f"MuJoCo execution failed: {exc}",
                )
            )

    def _execute_with_mujoco(self, trajectory_path: Path, model_path: Path) -> MujocoExecutionReport:
        import mujoco

        trajectory = self._load_trajectory(trajectory_path)
        joint_names = self._load_joint_names(trajectory)
        waypoints = trajectory["waypoints"]

        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        applied_keyframe = self._apply_initial_keyframe(mujoco, model, data)
        resolved_controls = self._resolve_joint_controls(mujoco, model, joint_names)
        mujoco.mj_forward(model, data)

        log_rows: list[dict[str, object]] = []
        previous_targets: dict[str, float] | None = None
        previous_time_s = 0.0

        for step, waypoint in enumerate(waypoints):
            time_s = float(waypoint["time_s"])
            targets = self._load_waypoint_targets(waypoint, joint_names)

            for control in resolved_controls:
                target_value = targets[control.trajectory_joint_name]
                data.qpos[control.qpos_index] = target_value
                if control.actuator_id is not None:
                    data.ctrl[control.actuator_id] = target_value

                if previous_targets is not None:
                    dt = max(time_s - previous_time_s, model.opt.timestep)
                    data.qvel[control.qvel_index] = (
                        target_value - previous_targets[control.trajectory_joint_name]
                    ) / dt

            mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)

            log_rows.append(
                {
                    "time": round(time_s, 6),
                    "step": step,
                    "qpos": self._json_array(data.qpos),
                    "qvel": self._json_array(data.qvel),
                    "target": json.dumps(targets),
                    "ctrl": self._json_array(data.ctrl),
                }
            )
            previous_targets = targets
            previous_time_s = time_s

        self._save_log(log_rows)
        duration_s = float(waypoints[-1]["time_s"]) if waypoints else 0.0
        message = "MuJoCo execution completed"
        if applied_keyframe:
            message = f"{message} using keyframe {applied_keyframe}"
        return self._save_report(
            MujocoExecutionReport(
                success=True,
                trajectory_path=str(trajectory_path),
                model_path=str(model_path),
                duration_s=duration_s,
                num_steps=len(waypoints),
                message=message,
            )
        )

    def _load_trajectory(self, trajectory_path: Path) -> dict[str, Any]:
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        joint_names = trajectory.get("joint_names")
        waypoints = trajectory.get("waypoints")
        if not isinstance(joint_names, list) or not joint_names:
            raise ValueError("trajectory JSON must contain a non-empty joint_names list")
        if not isinstance(waypoints, list):
            raise ValueError("trajectory JSON must contain a waypoints list")
        if not waypoints:
            raise ValueError("trajectory JSON must contain at least one waypoint")
        return trajectory

    def _load_joint_names(self, trajectory: dict[str, Any]) -> tuple[str, ...]:
        joint_names = trajectory["joint_names"]
        if not isinstance(joint_names, list):
            raise ValueError("trajectory JSON must contain a non-empty joint_names list")

        resolved_names: list[str] = []
        for joint_name in joint_names:
            if not isinstance(joint_name, str) or not joint_name:
                raise ValueError("trajectory joint_names must contain non-empty strings")
            resolved_names.append(joint_name)
        return tuple(resolved_names)

    def _load_waypoint_targets(self, waypoint: Any, joint_names: tuple[str, ...]) -> dict[str, float]:
        if not isinstance(waypoint, dict):
            raise ValueError("trajectory waypoints must be JSON objects")

        joint_positions = waypoint.get("joint_positions")
        if not isinstance(joint_positions, list):
            raise ValueError("trajectory waypoints must contain joint_positions lists")
        if len(joint_positions) != len(joint_names):
            raise ValueError(
                "trajectory waypoint joint_positions must match the length of joint_names "
                f"({len(joint_positions)} != {len(joint_names)})"
            )

        return {joint_name: float(value) for joint_name, value in zip(joint_names, joint_positions)}

    def _apply_initial_keyframe(self, mujoco, model, data) -> str | None:
        if self.keyframe_name is not None:
            keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, self.keyframe_name)
            if keyframe_id < 0:
                raise ValueError(f"MuJoCo keyframe does not exist: {self.keyframe_name}")
            mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
            return self.keyframe_name
        return None

    def _resolve_joint_controls(
        self,
        mujoco,
        model,
        trajectory_joint_names: tuple[str, ...],
    ) -> tuple[_ResolvedJointControl, ...]:
        joint_name_to_id = self._named_object_ids(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT)
        actuator_name_to_id = self._named_object_ids(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR)

        resolved_controls: list[_ResolvedJointControl] = []
        missing_joint_names: list[str] = []

        for trajectory_joint_name in trajectory_joint_names:
            joint_id, model_joint_name = self._resolve_model_name(
                trajectory_joint_name,
                joint_name_to_id,
            )
            if joint_id is None or model_joint_name is None:
                missing_joint_names.append(trajectory_joint_name)
                continue

            actuator_id, actuator_name = self._resolve_actuator_name(
                model_joint_name,
                actuator_name_to_id,
            )
            resolved_controls.append(
                _ResolvedJointControl(
                    trajectory_joint_name=trajectory_joint_name,
                    model_joint_name=model_joint_name,
                    joint_id=joint_id,
                    qpos_index=int(model.jnt_qposadr[joint_id]),
                    qvel_index=int(model.jnt_dofadr[joint_id]),
                    actuator_id=actuator_id,
                    actuator_name=actuator_name,
                )
            )

        if missing_joint_names:
            raise ValueError(
                "trajectory joint names do not match the MuJoCo model joints: "
                + ", ".join(repr(name) for name in missing_joint_names)
            )

        return tuple(resolved_controls)

    def _named_object_ids(self, mujoco, model, obj_type) -> dict[str, int]:
        named_object_ids: dict[str, int] = {}
        obj_count = {
            mujoco.mjtObj.mjOBJ_JOINT: model.njnt,
            mujoco.mjtObj.mjOBJ_ACTUATOR: model.nu,
        }[obj_type]

        for object_id in range(obj_count):
            object_name = mujoco.mj_id2name(model, obj_type, object_id)
            if object_name:
                named_object_ids[object_name] = object_id
        return named_object_ids

    def _resolve_model_name(
        self,
        requested_name: str,
        name_to_id: dict[str, int],
    ) -> tuple[int | None, str | None]:
        if requested_name in name_to_id:
            return name_to_id[requested_name], requested_name

        for model_name, object_id in name_to_id.items():
            if self._name_matches(requested_name, model_name):
                return object_id, model_name
        return None, None

    def _resolve_actuator_name(
        self,
        model_joint_name: str,
        name_to_id: dict[str, int],
    ) -> tuple[int | None, str | None]:
        preferred_names = (
            model_joint_name,
            f"{model_joint_name}_position",
            f"{model_joint_name}_ctrl",
        )
        for preferred_name in preferred_names:
            if preferred_name in name_to_id:
                return name_to_id[preferred_name], preferred_name

        for actuator_name, object_id in name_to_id.items():
            if self._name_matches(model_joint_name, actuator_name):
                return object_id, actuator_name

        return None, None

    def _name_matches(self, requested_name: str, model_name: str) -> bool:
        if requested_name == model_name:
            return True
        if model_name.endswith(requested_name):
            prefix = model_name[: -len(requested_name)]
            return not prefix or prefix[-1] in {"_", "-", "/"}
        if requested_name.endswith(model_name):
            prefix = requested_name[: -len(model_name)]
            return not prefix or prefix[-1] in {"_", "-", "/"}
        return False

    def _save_log(self, rows: list[dict[str, object]]) -> Path:
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / "mujoco_execution_log.csv"
        with path.open("w", newline="", encoding="utf-8") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=("time", "step", "qpos", "qvel", "target", "ctrl"))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _save_report(self, report: MujocoExecutionReport) -> MujocoExecutionReport:
        report_dir = self.output_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / "mujoco_execution_report.json"
        path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        return report

    def _json_array(self, values: Any) -> str:
        return json.dumps([float(value) for value in values])
