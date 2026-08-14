from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from math import floor, sqrt
from pathlib import Path
from time import monotonic

from .adapters.errors import (
    AdapterUnavailableError,
    ControlInterfaceError,
    ModelLoadError,
    NumericalSimulationError,
)
from .contracts import (
    BodyKinematicState,
    ContactRecord,
    ControlCommand,
    DynamicsDiagnostics,
    ExecutionErrorCode,
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    RuntimeObservation,
)


@dataclass
class _ExecutionFacts:
    initial_state: BodyKinematicState | None = None
    pre_release_state: BodyKinematicState | None = None
    post_release_state: BodyKinematicState | None = None
    target_object_present: bool = True
    departed_source_support: bool = False

    def __post_init__(self) -> None:
        self.contacts: dict[tuple[str, str], ContactRecord] = {}

    def add_observation(self, observation: RuntimeObservation, minimum_height_m: float) -> None:
        state = observation.target_object_state
        self.target_object_present = self.target_object_present and state is not None
        if self.initial_state is None and state is not None:
            self.initial_state = state
        if self.initial_state is not None and state is not None:
            initial_z = self.initial_state.T_world_body.matrix[2][3]
            if state.T_world_body.matrix[2][3] - initial_z >= minimum_height_m:
                self.departed_source_support = True
        for item in observation.contacts:
            pair = item.normalized_pair()
            current = self.contacts.get(pair)
            if current is None:
                self.contacts[pair] = ContactRecord(
                    pair[0],
                    pair[1],
                    item.maximum_penetration_m,
                    item.maximum_normal_force_n,
                    item.maximum_impact_impulse_ns,
                )
            else:
                self.contacts[pair] = ContactRecord(
                    pair[0],
                    pair[1],
                    max(current.maximum_penetration_m, item.maximum_penetration_m),
                    max(current.maximum_normal_force_n, item.maximum_normal_force_n),
                    max(current.maximum_impact_impulse_ns, item.maximum_impact_impulse_ns),
                )


class _ControlLogWriter:
    _FIELDS = (
        "planned_time_s",
        "runtime_time_s",
        "stage",
        "events",
        "joint_names",
        "target_positions_rad",
        "target_velocities_rad_s",
        "target_accelerations_rad_s2",
        "actual_positions_rad",
        "actuator_commands",
        "actuator_states",
    )

    def __init__(self, path: str):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._stream = target.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._stream, fieldnames=self._FIELDS)
        self._writer.writeheader()
        self._stream.flush()

    def write(self, command: ControlCommand, observation: RuntimeObservation) -> None:
        self._writer.writerow(
            {
                "planned_time_s": command.planned_time_s,
                "runtime_time_s": observation.runtime_time_s,
                "stage": command.stage.value,
                "events": json.dumps(command.events, separators=(",", ":")),
                "joint_names": json.dumps(command.joint_names, separators=(",", ":")),
                "target_positions_rad": json.dumps(command.positions_rad, separators=(",", ":")),
                "target_velocities_rad_s": json.dumps(
                    command.velocities_rad_s, separators=(",", ":")
                ),
                "target_accelerations_rad_s2": json.dumps(
                    command.accelerations_rad_s2, separators=(",", ":")
                ),
                "actual_positions_rad": json.dumps(
                    [observation.joint_positions_rad[name] for name in command.joint_names],
                    separators=(",", ":"),
                ),
                "actuator_commands": json.dumps(
                    dict(command.actuator_channels), sort_keys=True, separators=(",", ":")
                ),
                "actuator_states": json.dumps(
                    dict(observation.actuator_states), sort_keys=True, separators=(",", ":")
                ),
            }
        )
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()


class _DiagnosticLogWriter:
    def __init__(self, path: str):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._stream = target.open("w", encoding="utf-8")

    def write(
        self,
        planned_time_s: float,
        stage: str,
        observation: RuntimeObservation,
        period: str,
    ) -> None:
        state = observation.target_object_state
        payload = {
            "planned_time_s": planned_time_s,
            "runtime_time_s": observation.runtime_time_s,
            "stage": stage,
            "period": period,
            "tcp_pose": _transform_payload(observation.T_world_tcp),
            "target_object_state": state.to_dict() if state is not None else None,
            "contacts": [item.to_dict() for item in observation.contacts],
        }
        self._stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()


def execute(request: ExecutionRequest) -> ExecutionResult:
    started = monotonic()
    try:
        request.validate()
    except (TypeError, ValueError) as exc:
        return _failed_result(
            ExecutionErrorCode.INVALID_REQUEST,
            str(exc),
            started,
        )

    trajectory = request.planning_result.trajectory
    assert trajectory is not None
    duration_s = trajectory.segments[-1].start_time_s + trajectory.segments[-1].duration_s
    control_writer: _ControlLogWriter | None = None
    diagnostic_writer: _DiagnosticLogWriter | None = None
    adapter_initialized = False
    control_samples = 0
    control_log_samples = 0
    diagnostic_samples = 0
    executed_duration_s = 0.0
    facts = _ExecutionFacts()
    log_error: OSError | None = None
    adapter_close_error: Exception | None = None

    try:
        control_writer = _ControlLogWriter(request.control_log_path)
        diagnostic_writer = _DiagnosticLogWriter(request.diagnostic_log_path)
    except OSError as exc:
        _close_writer(control_writer)
        _close_writer(diagnostic_writer)
        return _failed_result(
            ExecutionErrorCode.LOG_IO_FAILED,
            f"Execution log initialization failed: {exc}",
            started,
            planned_duration_s=duration_s,
        )

    try:
        control_times = _sample_times(duration_s, request.control_frequency_hz)
        control_log_times = _sample_times(
            duration_s, request.control_log_frequency_hz
        )
        diagnostic_times = _sample_times(duration_s, request.diagnostic_log_frequency_hz)
        control_keys = {_time_key(value) for value in control_times}
        control_log_keys = {_time_key(value) for value in control_log_times}
        diagnostic_keys = {_time_key(value) for value in diagnostic_times}
        timeline = sorted(control_keys | control_log_keys | diagnostic_keys)
        first_time = timeline[0]
        initial_command = _sample_command(request, first_time, -1.0)
        observation = request.adapter.initialize(
            request.scene_objects,
            request.target_object_id,
            initial_command,
        )
        adapter_initialized = True
        control_samples = 1
        facts.add_observation(observation, request.diagnostics_config.minimum_departure_height_m)
        current_command = initial_command
        if first_time in control_log_keys:
            try:
                control_writer.write(current_command, observation)
                control_log_samples += 1
            except OSError as exc:
                log_error = exc
        if first_time in diagnostic_keys and log_error is None:
            try:
                diagnostic_writer.write(
                    first_time, current_command.stage.value, observation, "planned_trajectory"
                )
                diagnostic_samples += 1
            except OSError as exc:
                log_error = exc

        previous_time = first_time
        previous_control_time = first_time
        for key in timeline[1:]:
            planned_time = float(key)
            observation = request.adapter.advance(planned_time - previous_time)
            facts.add_observation(
                observation, request.diagnostics_config.minimum_departure_height_m
            )
            if key in control_keys:
                next_command = _sample_command(
                    request, planned_time, previous_control_time
                )
                if "release" in next_command.events:
                    facts.pre_release_state = observation.target_object_state
                request.adapter.apply_control(next_command)
                current_command = next_command
                control_samples += 1
                previous_control_time = planned_time
            if key in control_log_keys and log_error is None:
                try:
                    control_writer.write(current_command, observation)
                    control_log_samples += 1
                except OSError as exc:
                    log_error = exc
            if key in diagnostic_keys and log_error is None:
                diagnostic_command = _sample_command(
                    request, planned_time, planned_time
                )
                try:
                    diagnostic_writer.write(
                        planned_time,
                        diagnostic_command.stage.value,
                        observation,
                        "planned_trajectory",
                    )
                    diagnostic_samples += 1
                except OSError as exc:
                    log_error = exc
            previous_time = planned_time
            executed_duration_s = planned_time

        post_release_times = _sample_times(
            request.diagnostics_config.post_release_observation_s,
            request.diagnostic_log_frequency_hz,
        )
        previous_post_time = 0.0
        for post_time in post_release_times[1:]:
            observation = request.adapter.advance(post_time - previous_post_time)
            facts.add_observation(
                observation, request.diagnostics_config.minimum_departure_height_m
            )
            if log_error is None:
                try:
                    diagnostic_writer.write(
                        duration_s,
                        current_command.stage.value,
                        observation,
                        "post_release_diagnostics",
                    )
                    diagnostic_samples += 1
                except OSError as exc:
                    log_error = exc
            previous_post_time = post_time
        if request.diagnostics_config.post_release_observation_s == 0.0:
            observation = request.adapter.observe()
            facts.add_observation(
                observation, request.diagnostics_config.minimum_departure_height_m
            )
        facts.post_release_state = observation.target_object_state
        diagnostics = _evaluate_diagnostics(request, facts)
    except AdapterUnavailableError as exc:
        return _failed_result(
            ExecutionErrorCode.ADAPTER_UNAVAILABLE,
            str(exc),
            started,
            planned_duration_s=duration_s,
            executed_duration_s=executed_duration_s,
            control_samples=control_samples,
            control_log_samples=control_log_samples,
            diagnostic_samples=diagnostic_samples,
        )
    except ModelLoadError as exc:
        return _failed_result(
            ExecutionErrorCode.MODEL_LOAD_FAILED,
            str(exc),
            started,
            planned_duration_s=duration_s,
            executed_duration_s=executed_duration_s,
            control_samples=control_samples,
            control_log_samples=control_log_samples,
            diagnostic_samples=diagnostic_samples,
        )
    except ControlInterfaceError as exc:
        return _failed_result(
            ExecutionErrorCode.CONTROL_INTERFACE_FAILED,
            str(exc),
            started,
            planned_duration_s=duration_s,
            executed_duration_s=executed_duration_s,
            control_samples=control_samples,
            control_log_samples=control_log_samples,
            diagnostic_samples=diagnostic_samples,
        )
    except NumericalSimulationError as exc:
        return _failed_result(
            ExecutionErrorCode.NUMERICAL_SIMULATION_FAILED,
            str(exc),
            started,
            planned_duration_s=duration_s,
            executed_duration_s=executed_duration_s,
            control_samples=control_samples,
            control_log_samples=control_log_samples,
            diagnostic_samples=diagnostic_samples,
        )
    except Exception as exc:
        return _failed_result(
            ExecutionErrorCode.ADAPTER_ERROR,
            f"Execution adapter error: {type(exc).__name__}: {exc}",
            started,
            planned_duration_s=duration_s,
            executed_duration_s=executed_duration_s,
            control_samples=control_samples,
            control_log_samples=control_log_samples,
            diagnostic_samples=diagnostic_samples,
        )
    finally:
        for writer in (control_writer, diagnostic_writer):
            close_error = _close_writer(writer)
            if log_error is None and close_error is not None:
                log_error = close_error
        if adapter_initialized:
            try:
                request.adapter.close()
            except Exception as exc:
                adapter_close_error = exc

    if log_error is not None:
        return ExecutionResult(
            ExecutionStatus.FAILED,
            ExecutionErrorCode.LOG_IO_FAILED,
            f"Execution log write failed after control began: {log_error}",
            diagnostics,
            duration_s,
            duration_s,
            control_samples,
            control_log_samples,
            diagnostic_samples,
            monotonic() - started,
        )
    if adapter_close_error is not None:
        return ExecutionResult(
            ExecutionStatus.FAILED,
            ExecutionErrorCode.ADAPTER_ERROR,
            "Execution adapter close failed after control completed: "
            f"{type(adapter_close_error).__name__}: {adapter_close_error}",
            diagnostics,
            duration_s,
            duration_s,
            control_samples,
            control_log_samples,
            diagnostic_samples,
            monotonic() - started,
        )
    return ExecutionResult(
        ExecutionStatus.COMPLETED,
        None,
        "planned trajectory execution completed",
        diagnostics,
        duration_s,
        duration_s,
        control_samples,
        control_log_samples,
        diagnostic_samples,
        monotonic() - started,
    )


def _sample_command(
    request: ExecutionRequest,
    planned_time_s: float,
    previous_control_time_s: float,
) -> ControlCommand:
    trajectory = request.planning_result.trajectory
    assert trajectory is not None
    segment = trajectory.segments[-1]
    for index, candidate in enumerate(trajectory.segments):
        end_time = candidate.start_time_s + candidate.duration_s
        if planned_time_s < end_time - 1e-10 or index == len(trajectory.segments) - 1:
            segment = candidate
            break
    local_time = min(segment.duration_s, max(0.0, planned_time_s - segment.start_time_s))
    knot = segment.sample(local_time)
    channels = dict(knot.actuator_channels)
    due_events = tuple(
        event
        for event in trajectory.events
        if previous_control_time_s < event.time_s <= planned_time_s + 1e-10
    )
    for event in due_events:
        channels.update(event.actuator_channels)
    return ControlCommand(
        knot.stage,
        planned_time_s,
        trajectory.joint_names,
        knot.positions_rad,
        knot.velocities_rad_s,
        knot.accelerations_rad_s2,
        channels,
        tuple(event.event for event in due_events),
    )


def _sample_times(duration_s: float, frequency_hz: float) -> tuple[float, ...]:
    count = floor(duration_s * frequency_hz + 1e-12)
    values = [index / frequency_hz for index in range(count + 1)]
    if not values or abs(values[-1] - duration_s) > 1e-10:
        values.append(duration_s)
    else:
        values[-1] = duration_s
    return tuple(values)


def _time_key(value: float) -> float:
    return round(value, 12)


def _evaluate_diagnostics(
    request: ExecutionRequest,
    facts: _ExecutionFacts,
) -> DynamicsDiagnostics:
    config = request.diagnostics_config
    pre_release = facts.pre_release_state
    post_release = facts.post_release_state
    target_error: float | None = None
    if pre_release is not None:
        position = tuple(pre_release.T_world_body.matrix[index][3] for index in range(3))
        target_error = sqrt(
            sum(
                (actual - expected) ** 2
                for actual, expected in zip(position, config.target_position_world_m)
            )
        )
    stable = False
    if post_release is not None:
        linear_speed = sqrt(sum(value * value for value in post_release.linear_velocity_m_s))
        angular_speed = sqrt(sum(value * value for value in post_release.angular_velocity_rad_s))
        stable = (
            linear_speed <= config.settled_linear_speed_m_s
            and angular_speed <= config.settled_angular_speed_rad_s
        )
    reasons = []
    if not facts.target_object_present or facts.initial_state is None:
        reasons.append("target object was absent from runtime observations")
    if not facts.departed_source_support:
        reasons.append("target object did not reach the required departure height")
    if pre_release is None:
        reasons.append("no target object state was available before release")
    elif target_error is None or target_error > config.target_position_tolerance_m:
        reasons.append("pre-release target position error exceeded tolerance")
    if not stable:
        reasons.append("target object was not stable after release")
    contacts = tuple(facts.contacts[key] for key in sorted(facts.contacts))
    return DynamicsDiagnostics(
        evaluated=True,
        success=not reasons,
        target_object_present=facts.target_object_present and facts.initial_state is not None,
        departed_source_support=facts.departed_source_support,
        pre_release_state=pre_release,
        post_release_state=post_release,
        stable_after_release=stable,
        target_position_error_m=target_error,
        contact_pairs=contacts,
        maximum_penetration_m=max(
            (item.maximum_penetration_m for item in contacts), default=0.0
        ),
        maximum_normal_force_n=max(
            (item.maximum_normal_force_n for item in contacts), default=0.0
        ),
        maximum_impact_impulse_ns=max(
            (item.maximum_impact_impulse_ns for item in contacts), default=0.0
        ),
        failure_reasons=tuple(reasons),
    )


def _failed_result(
    code: ExecutionErrorCode,
    message: str,
    started: float,
    *,
    planned_duration_s: float = 0.0,
    executed_duration_s: float = 0.0,
    control_samples: int = 0,
    control_log_samples: int = 0,
    diagnostic_samples: int = 0,
) -> ExecutionResult:
    return ExecutionResult(
        ExecutionStatus.FAILED,
        code,
        message,
        DynamicsDiagnostics.not_evaluated("complete execution was not available for diagnostics"),
        planned_duration_s,
        executed_duration_s,
        control_samples,
        control_log_samples,
        diagnostic_samples,
        monotonic() - started,
    )


def _close_writer(
    writer: _ControlLogWriter | _DiagnosticLogWriter | None,
) -> OSError | None:
    if writer is None:
        return None
    try:
        writer.close()
    except OSError as exc:
        return exc
    return None


def _transform_payload(transform) -> dict[str, object]:
    return {
        "parent_frame": transform.parent_frame,
        "child_frame": transform.child_frame,
        "matrix": [list(row) for row in transform.matrix],
    }
