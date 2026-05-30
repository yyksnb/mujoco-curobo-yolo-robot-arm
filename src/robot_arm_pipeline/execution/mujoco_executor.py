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


class MujocoExecutor:
    name = "mujoco_executor"

    def __init__(self, output_dir: Path | str = Path("outputs")) -> None:
        self.output_dir = Path(output_dir)

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
        waypoints = trajectory["waypoints"]

        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)

        log_rows: list[dict[str, object]] = []
        previous_positions: list[float] | None = None
        previous_time_s = 0.0

        for step, waypoint in enumerate(waypoints):
            time_s = float(waypoint["time_s"])
            target = [float(value) for value in waypoint["joint_positions"]]

            qpos_count = min(model.nq, len(target))
            data.qpos[:qpos_count] = target[:qpos_count]

            if previous_positions is not None:
                dt = max(time_s - previous_time_s, model.opt.timestep)
                qvel_target = [
                    (value - previous_positions[index]) / dt
                    for index, value in enumerate(target[: min(len(target), len(previous_positions))])
                ]
                qvel_count = min(model.nv, len(qvel_target))
                data.qvel[:qvel_count] = qvel_target[:qvel_count]

            if model.nu:
                ctrl_count = min(model.nu, len(target))
                data.ctrl[:ctrl_count] = target[:ctrl_count]

            mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)

            log_rows.append(
                {
                    "time": round(time_s, 6),
                    "step": step,
                    "qpos": self._json_array(data.qpos),
                    "qvel": self._json_array(data.qvel),
                    "target": json.dumps(target),
                    "ctrl": self._json_array(data.ctrl),
                }
            )
            previous_positions = target
            previous_time_s = time_s

        self._save_log(log_rows)
        duration_s = float(waypoints[-1]["time_s"]) if waypoints else 0.0
        return self._save_report(
            MujocoExecutionReport(
                success=True,
                trajectory_path=str(trajectory_path),
                model_path=str(model_path),
                duration_s=duration_s,
                num_steps=len(waypoints),
                message="MuJoCo execution completed",
            )
        )

    def _load_trajectory(self, trajectory_path: Path) -> dict[str, Any]:
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        waypoints = trajectory.get("waypoints")
        if not isinstance(waypoints, list):
            raise ValueError("trajectory JSON must contain a waypoints list")
        return trajectory

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
