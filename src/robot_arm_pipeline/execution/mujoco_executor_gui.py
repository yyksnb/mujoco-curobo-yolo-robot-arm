from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from threading import Lock

from robot_arm_pipeline.execution.mujoco_executor import MujocoExecutionReport, MujocoExecutor


@dataclass(frozen=True)
class _GuiKeyMap:
    space: int
    n: int
    r: int


@dataclass(frozen=True)
class _InitialReplayState:
    qpos: tuple[float, ...]
    qvel: tuple[float, ...]
    ctrl: tuple[float, ...]
    previous_targets: dict[str, float]
    time_s: float


@dataclass
class _ReplayControl:
    paused: bool
    segment_requests: int = 0
    reset_requested: bool = False
    lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def handle_key(self, key: int, keymap: _GuiKeyMap) -> None:
        with self.lock:
            if key == keymap.space:
                self.paused = not self.paused
                if not self.paused:
                    self.segment_requests = 0
            elif key == keymap.n:
                self.paused = True
                self.segment_requests += 1
            elif key == keymap.r:
                self.paused = True
                self.segment_requests = 0
                self.reset_requested = True

    def snapshot(self) -> tuple[bool, int]:
        with self.lock:
            return self.paused, self.segment_requests

    def consume_segment_request(self) -> bool:
        with self.lock:
            if self.segment_requests <= 0:
                return False
            self.segment_requests -= 1
            return True

    def consume_reset_request(self) -> bool:
        with self.lock:
            if not self.reset_requested:
                return False
            self.reset_requested = False
            return True


class MujocoGuiExecutor(MujocoExecutor):
    name = "mujoco_gui_executor"

    def __init__(
        self,
        output_dir: Path | str = Path("outputs"),
        *,
        keyframe_name: str | None = None,
        replay_speed: float = 1.0,
        final_hold_seconds: float = 0.0,
        wait_for_close: bool = True,
        gui_poll_seconds: float = 0.05,
        step_by_step: bool = True,
    ) -> None:
        super().__init__(output_dir=output_dir, keyframe_name=keyframe_name)
        if replay_speed <= 0:
            raise ValueError("replay_speed must be positive")
        if final_hold_seconds < 0:
            raise ValueError("final_hold_seconds must be non-negative")
        if gui_poll_seconds <= 0:
            raise ValueError("gui_poll_seconds must be positive")
        self.replay_speed = replay_speed
        self.final_hold_seconds = final_hold_seconds
        self.wait_for_close = wait_for_close
        self.gui_poll_seconds = gui_poll_seconds
        self.step_by_step = step_by_step

    @staticmethod
    def is_available() -> bool:
        try:
            return find_spec("mujoco") is not None and find_spec("mujoco.viewer") is not None
        except (ModuleNotFoundError, ValueError):
            return False

    def _availability_message(self) -> str:
        if find_spec("mujoco") is None:
            return "MuJoCo is not installed. Install it with: pip install mujoco"
        return "MuJoCo GUI viewer is not available in this environment."

    def _execute_with_mujoco(self, trajectory_path: Path, model_path: Path) -> MujocoExecutionReport:
        import mujoco
        import mujoco.viewer

        trajectory = self._load_trajectory(trajectory_path)
        joint_names = self._load_joint_names(trajectory)
        waypoints = trajectory["waypoints"]

        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        applied_keyframe = self._apply_initial_keyframe(mujoco, model, data)
        resolved_controls = self._resolve_joint_controls(mujoco, model, joint_names)
        mujoco.mj_forward(model, data)

        initial_state = self._capture_initial_state(data, resolved_controls)
        control = _ReplayControl(paused=self.step_by_step)
        keymap = self._resolve_keymap(mujoco)

        log_rows: list[dict[str, object]] = []
        previous_targets = initial_state.previous_targets.copy()
        previous_time_s = initial_state.time_s
        replayed_steps = 0
        stopped_early = False
        reset_occurred = False
        current_step = 0

        launch_kwargs = {}
        if self._launch_passive_supports_key_callback(mujoco.viewer.launch_passive):
            launch_kwargs["key_callback"] = lambda key: control.handle_key(int(key), keymap)

        with mujoco.viewer.launch_passive(model, data, **launch_kwargs) as viewer:
            self._configure_viewer_camera(viewer, mujoco, model, data)
            self._render_overlay(viewer, mujoco, step_index=0, total_steps=len(waypoints), control=control)
            viewer.sync()

            while True:
                while current_step < len(waypoints):
                    if not viewer.is_running():
                        stopped_early = True
                        break

                    if control.consume_reset_request():
                        reset_occurred = True
                        self._restore_initial_state(mujoco, model, data, initial_state)
                        previous_targets = initial_state.previous_targets.copy()
                        previous_time_s = initial_state.time_s
                        log_rows.clear()
                        replayed_steps = 0
                        current_step = 0
                        self._render_overlay(viewer, mujoco, step_index=0, total_steps=len(waypoints), control=control)
                        viewer.sync()
                        continue

                    waypoint = waypoints[current_step]
                    current_targets = self._load_waypoint_targets(waypoint, joint_names)
                    current_time_s = float(waypoint["time_s"])
                    segment_duration_s = max(current_time_s - previous_time_s, 0.0)
                    minimum_frames = 10 if current_step == 0 else 1
                    segment_frame_count = self._segment_frame_count(
                        segment_duration_s,
                        minimum_frames=minimum_frames,
                    )

                    wait_result = self._wait_for_segment_permission(
                        viewer,
                        mujoco,
                        control,
                        step_index=current_step,
                        total_steps=len(waypoints),
                    )
                    if wait_result == "stop":
                        stopped_early = True
                        break
                    if wait_result == "reset":
                        reset_occurred = True
                        self._restore_initial_state(mujoco, model, data, initial_state)
                        previous_targets = initial_state.previous_targets.copy()
                        previous_time_s = initial_state.time_s
                        log_rows.clear()
                        replayed_steps = 0
                        current_step = 0
                        self._render_overlay(viewer, mujoco, step_index=0, total_steps=len(waypoints), control=control)
                        viewer.sync()
                        continue

                    play_result = self._replay_segment(
                        mujoco,
                        model,
                        data,
                        viewer,
                        resolved_controls,
                        start_targets=previous_targets,
                        end_targets=current_targets,
                        current_time_s=current_time_s,
                        segment_duration_s=segment_duration_s,
                        segment_frame_count=segment_frame_count,
                        control=control,
                        joint_names=joint_names,
                        step_index=current_step,
                        total_steps=len(waypoints),
                    )
                    if play_result == "stop":
                        stopped_early = True
                        break
                    if play_result == "reset":
                        reset_occurred = True
                        self._restore_initial_state(mujoco, model, data, initial_state)
                        previous_targets = initial_state.previous_targets.copy()
                        previous_time_s = initial_state.time_s
                        log_rows.clear()
                        replayed_steps = 0
                        current_step = 0
                        self._render_overlay(viewer, mujoco, step_index=0, total_steps=len(waypoints), control=control)
                        viewer.sync()
                        continue

                    log_rows.append(
                        {
                            "time": round(current_time_s, 6),
                            "step": current_step,
                            "qpos": self._json_array(data.qpos),
                            "qvel": self._json_array(data.qvel),
                            "target": json.dumps(current_targets),
                            "ctrl": self._json_array(data.ctrl),
                        }
                    )
                    replayed_steps += 1
                    previous_targets = current_targets
                    previous_time_s = current_time_s
                    current_step += 1

                if stopped_early:
                    break

                completion_result = self._wait_for_completion(
                    viewer,
                    mujoco,
                    control,
                    total_steps=len(waypoints),
                    final_step_index=len(waypoints),
                )
                if completion_result == "reset":
                    reset_occurred = True
                    self._restore_initial_state(mujoco, model, data, initial_state)
                    previous_targets = initial_state.previous_targets.copy()
                    previous_time_s = initial_state.time_s
                    log_rows.clear()
                    replayed_steps = 0
                    current_step = 0
                    self._render_overlay(viewer, mujoco, step_index=0, total_steps=len(waypoints), control=control)
                    viewer.sync()
                    continue
                break

        self._save_log(log_rows)
        duration_s = previous_time_s if replayed_steps else 0.0
        message = "MuJoCo GUI replay completed"
        if applied_keyframe:
            message = f"{message} using keyframe {applied_keyframe}"
        if reset_occurred:
            message = f"{message} after reset"
        if stopped_early:
            message = "MuJoCo GUI replay stopped because the viewer closed early"
        elif self.wait_for_close:
            message = "MuJoCo GUI replay completed; viewer closed"

        return self._save_report(
            MujocoExecutionReport(
                success=not stopped_early,
                trajectory_path=str(trajectory_path),
                model_path=str(model_path),
                duration_s=float(duration_s),
                num_steps=replayed_steps,
                message=message,
            )
        )

    def _wait_for_segment_permission(
        self,
        viewer,
        mujoco,
        control: _ReplayControl,
        *,
        step_index: int,
        total_steps: int,
    ) -> str:
        while viewer.is_running():
            if control.consume_reset_request():
                return "reset"
            paused, step_requests = control.snapshot()
            if not paused:
                return "continue"
            if step_requests > 0 and control.consume_segment_request():
                return "continue"

            self._render_overlay(
                viewer,
                mujoco=mujoco,
                step_index=step_index,
                total_steps=total_steps,
                control=control,
            )
            viewer.sync()
            time.sleep(self.gui_poll_seconds)
        return "stop"

    def _replay_segment(
        self,
        mujoco,
        model,
        data,
        viewer,
        resolved_controls: tuple,
        *,
        start_targets: dict[str, float],
        end_targets: dict[str, float],
        current_time_s: float,
        segment_duration_s: float,
        segment_frame_count: int,
        control: _ReplayControl,
        joint_names: tuple[str, ...],
        step_index: int,
        total_steps: int,
    ) -> str:
        frame_dt = max(segment_duration_s / max(segment_frame_count, 1), model.opt.timestep)

        for frame_index in range(1, segment_frame_count + 1):
            if not viewer.is_running():
                return "stop"
            if control.consume_reset_request():
                return "reset"

            alpha = frame_index / segment_frame_count
            interpolated_targets = {
                joint_name: start_targets[joint_name] + (end_targets[joint_name] - start_targets[joint_name]) * alpha
                for joint_name in end_targets
            }
            self._apply_targets(
                model,
                data,
                resolved_controls,
                interpolated_targets,
                start_targets,
                frame_dt,
            )
            mujoco.mj_forward(model, data)
            self._render_overlay(
                viewer,
                mujoco,
                step_index=step_index,
                total_steps=total_steps,
                control=control,
            )
            viewer.sync()
            time.sleep(self.gui_poll_seconds)
        return "continue"

    def _wait_for_completion(
        self,
        viewer,
        mujoco,
        control: _ReplayControl,
        *,
        total_steps: int,
        final_step_index: int,
    ) -> str:
        hold_deadline = None
        if self.final_hold_seconds > 0:
            hold_deadline = time.monotonic() + self.final_hold_seconds

        while viewer.is_running():
            if control.consume_reset_request():
                return "reset"

            self._render_overlay(
                viewer,
                mujoco,
                step_index=final_step_index,
                total_steps=total_steps,
                control=control,
            )
            viewer.sync()

            if hold_deadline is not None:
                remaining_s = hold_deadline - time.monotonic()
                if remaining_s > 0:
                    time.sleep(min(self.gui_poll_seconds, remaining_s))
                    continue
                hold_deadline = None

            if not self.wait_for_close:
                return "continue"

            time.sleep(self.gui_poll_seconds)
        return "stop"

    def _segment_frame_count(self, segment_duration_s: float, *, minimum_frames: int = 1) -> int:
        wall_duration_s = max(segment_duration_s, 0.0) / self.replay_speed
        return max(minimum_frames, int(math.ceil(wall_duration_s / self.gui_poll_seconds)))

    def _apply_targets(
        self,
        model,
        data,
        resolved_controls: tuple,
        targets: dict[str, float],
        previous_targets: dict[str, float] | None,
        dt_s: float,
    ) -> None:
        for control in resolved_controls:
            target_value = targets[control.trajectory_joint_name]
            data.qpos[control.qpos_index] = target_value
            if control.actuator_id is not None:
                data.ctrl[control.actuator_id] = target_value
            if previous_targets is not None:
                dt = max(dt_s, model.opt.timestep)
                data.qvel[control.qvel_index] = (
                    target_value - previous_targets[control.trajectory_joint_name]
                ) / dt
            else:
                data.qvel[control.qvel_index] = 0.0

    def _current_targets_from_data(
        self,
        data,
        resolved_controls: tuple,
    ) -> dict[str, float]:
        return {
            control.trajectory_joint_name: float(data.qpos[control.qpos_index])
            for control in resolved_controls
        }

    def _capture_initial_state(self, data, resolved_controls: tuple) -> _InitialReplayState:
        return _InitialReplayState(
            qpos=tuple(float(value) for value in data.qpos),
            qvel=tuple(float(value) for value in data.qvel),
            ctrl=tuple(float(value) for value in data.ctrl),
            previous_targets=self._current_targets_from_data(data, resolved_controls),
            time_s=float(data.time),
        )

    def _restore_initial_state(self, mujoco, model, data, initial_state: _InitialReplayState) -> None:
        self._assign_sequence(data.qpos, initial_state.qpos)
        self._assign_sequence(data.qvel, initial_state.qvel)
        self._assign_sequence(data.ctrl, initial_state.ctrl)
        data.time = initial_state.time_s
        mujoco.mj_forward(model, data)

    def _configure_viewer_camera(self, viewer, mujoco, model, data) -> None:
        try:
            mount_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gen3_mount")
            if mount_body_id < 0:
                return
            mount_pos = data.xpos[mount_body_id]
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.trackbodyid = mount_body_id
            viewer.cam.lookat[0] = float(mount_pos[0])
            viewer.cam.lookat[1] = float(mount_pos[1])
            viewer.cam.lookat[2] = float(mount_pos[2] + 0.2)
            viewer.cam.distance = max(float(model.stat.extent) * 0.8, 0.8)
            viewer.cam.azimuth = 120.0
            viewer.cam.elevation = -25.0
        except Exception:
            return

    def _hold_final_pose(self, viewer, duration_s: float) -> None:
        deadline = time.monotonic() + duration_s
        while viewer.is_running():
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                break
            viewer.sync()
            time.sleep(min(self.gui_poll_seconds, remaining_s))

    def _wait_for_close(self, viewer) -> None:
        while viewer.is_running():
            viewer.sync()
            time.sleep(self.gui_poll_seconds)

    def _resolve_keymap(self, mujoco) -> _GuiKeyMap:
        viewer_module = getattr(mujoco, "viewer", None)
        glfw = getattr(viewer_module, "glfw", None)
        if glfw is not None:
            return _GuiKeyMap(space=int(glfw.KEY_SPACE), n=int(glfw.KEY_N), r=int(glfw.KEY_R))
        return _GuiKeyMap(space=32, n=78, r=82)

    def _render_overlay(
        self,
        viewer,
        mujoco,
        *,
        step_index: int,
        total_steps: int,
        control: _ReplayControl | None = None,
    ) -> None:
        if not hasattr(viewer, "set_texts"):
            return

        font = getattr(mujoco.mjtFontScale, "mjFONTSCALE_100", mujoco.mjtFontScale.mjFONTSCALE_150)
        gridpos = getattr(mujoco.mjtGridPos, "mjGRID_TOPRIGHT", None)
        left_gridpos = getattr(mujoco.mjtGridPos, "mjGRID_TOPLEFT", None)
        if gridpos is None:
            return

        label = f"Waypoint {step_index}/{max(total_steps, 1)}"
        paused = False
        pending_requests = 0
        if control is not None:
            paused, pending_requests = control.snapshot()

        keys_text = "\n".join(
            [
                "Keys:",
                "Space: pause/resume",
                "N: next waypoint",
                "R: reset to initial state",
            ]
        )
        state_text = "State: paused" if paused else "State: running"
        if pending_requests > 0:
            state_text = f"{state_text} (queued={pending_requests})"
        if control is None:
            state_text = "State: unknown"
        left_text = "\n".join([keys_text, state_text])

        try:
            texts = [(font, gridpos, label, "")]
            if left_gridpos is not None:
                texts.insert(0, (font, left_gridpos, left_text, ""))
            viewer.set_texts(texts)
        except Exception:
            return

    def _assign_sequence(self, target, values: tuple[float, ...]) -> None:
        for index, value in enumerate(values):
            target[index] = value

    def _mode_label(self) -> str:
        return "step-by-step" if self.step_by_step else "continuous"

    def _launch_passive_supports_key_callback(self, launch_passive) -> bool:
        try:
            import inspect

            return "key_callback" in inspect.signature(launch_passive).parameters
        except (TypeError, ValueError):
            return False
