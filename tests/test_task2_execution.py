from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from struct import pack, unpack

import pytest

from task2.adapters.mujoco import MuJoCoExecutorAdapter
from task2.adapters.errors import ControlInterfaceError
from task2.config import load_task2_config
from task2.contracts import (
    BodyKinematicState,
    ContactRecord,
    ControlCommand,
    DynamicsDiagnosticsConfig,
    ExecutionErrorCode,
    ExecutionRequest,
    ExecutionStatus,
    PlannedTrajectory,
    PlanningResult,
    RigidTransform,
    RuntimeObservation,
    SceneObjectBinding,
    TrajectoryEvent,
    TrajectoryKnot,
    TrajectoryPhase,
    TrajectoryPolynomialSection,
    TrajectorySegment,
    TrajectoryStage,
)
from task2.execution import execute


ROOT = Path(__file__).resolve().parents[1]
POSITION_CHANNEL = "parallel_gripper.opening_m"
FORCE_CHANNEL = "parallel_gripper.force_limit_n"


def _pose(parent: str, child: str, xyz=(0.0, 0.0, 0.0)) -> RigidTransform:
    return RigidTransform(
        parent,
        child,
        (
            (1.0, 0.0, 0.0, xyz[0]),
            (0.0, 1.0, 0.0, xyz[1]),
            (0.0, 0.0, 1.0, xyz[2]),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )


def _planning_result() -> PlanningResult:
    stages = tuple(TrajectoryStage)
    channels = {POSITION_CHANNEL: 0.02, FORCE_CHANNEL: 1.0}
    knots = []
    segments = []
    phases = []
    for index in range(len(stages) + 1):
        stage = stages[min(index, len(stages) - 1)]
        knots.append(
            TrajectoryKnot(stage, index * 0.1, (index * 0.1,), (0.0,), (0.0,), channels)
        )
    for index, stage in enumerate(stages):
        segments.append(
            TrajectorySegment(
                stage,
                index * 0.1,
                0.1,
                index,
                index + 1,
                (
                    TrajectoryPolynomialSection(
                        0.0,
                        0.1,
                        ((index * 0.1, 1.0, 0.0, 0.0),),
                    ),
                ),
                channels,
                channels,
            )
        )
        phases.append(
            TrajectoryPhase(stage, index, index + 1, index, index, index * 0.1, (index + 1) * 0.1)
        )
    trajectory = PlannedTrajectory(
        ("joint",),
        tuple(knots),
        tuple(segments),
        tuple(phases),
        (
            TrajectoryEvent("grip", TrajectoryStage.GRASP, 0.3, channels),
            TrajectoryEvent(
                "release",
                TrajectoryStage.PLACE,
                0.6,
                {POSITION_CHANNEL: 0.08, FORCE_CHANNEL: 1.0},
            ),
        ),
    )
    return PlanningResult.succeeded(
        trajectory,
        _pose("world", "object", (9.0, 9.0, 9.0)),
        (),
        {},
    )


def _object_binding() -> SceneObjectBinding:
    return SceneObjectBinding("object", "marker", _pose("world", "object"))


class _RecordingAdapter:
    adapter_id = "recording"

    def __init__(self):
        self.time_s = 0.0
        self.command: ControlCommand | None = None
        self.commands: list[ControlCommand] = []

    def initialize(self, scene_objects, target_object_id, initial_command):
        self.command = initial_command
        self.commands.append(initial_command)
        return self.observe()

    def apply_control(self, command):
        self.command = command
        self.commands.append(command)

    def advance(self, duration_s):
        self.time_s += duration_s
        return self.observe()

    def observe(self):
        assert self.command is not None
        departed = self.command.stage in (
            TrajectoryStage.LIFT,
            TrajectoryStage.EGRESS,
            TrajectoryStage.PLACE,
        )
        position = (1.0, 2.0, 3.0) if self.command.stage is TrajectoryStage.PLACE else (0.0, 0.0, 0.2 if departed else 0.0)
        state = BodyKinematicState(
            _pose("world", "object", position),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        )
        return RuntimeObservation(
            self.time_s,
            {"joint": self.command.positions_rad[0]},
            dict(self.command.actuator_channels),
            _pose("world", "tcp"),
            state,
            (ContactRecord("marker_geom", "support", 0.001, 2.0, 0.02),),
        )

    def close(self):
        return None


def _request(tmp_path: Path, adapter) -> ExecutionRequest:
    return ExecutionRequest(
        request_id="execution",
        planning_result=_planning_result(),
        adapter=adapter,
        scene_objects=(_object_binding(),),
        target_object_id="object",
        control_frequency_hz=20.0,
        control_log_frequency_hz=4.0,
        diagnostic_log_frequency_hz=5.0,
        control_log_path=str(tmp_path / "control.csv"),
        diagnostic_log_path=str(tmp_path / "diagnostics.jsonl"),
        diagnostics_config=DynamicsDiagnosticsConfig(
            world_frame="world",
            source_support_geom_names=("support",),
            minimum_departure_height_m=0.1,
            target_position_world_m=(9.0, 9.0, 9.0),
            target_position_tolerance_m=0.01,
            post_release_observation_s=0.2,
            settled_linear_speed_m_s=0.01,
            settled_angular_speed_rad_s=0.01,
        ),
    )


def test_execute_delivers_all_phases_and_keeps_dynamics_failure_separate(tmp_path):
    adapter = _RecordingAdapter()
    result = execute(_request(tmp_path, adapter))

    assert result.execution_status is ExecutionStatus.COMPLETED
    assert result.error_code is None
    assert not result.dynamics_diagnostics.success
    assert result.dynamics_diagnostics.departed_source_support
    assert result.executed_planned_duration_s == pytest.approx(0.6)
    assert result.control_sample_count == 13
    assert result.control_log_sample_count == 4
    assert len(adapter.commands) == result.control_sample_count
    assert tuple(dict.fromkeys(command.stage for command in adapter.commands)) == tuple(
        TrajectoryStage
    )
    assert adapter.commands[-1].events == ("release",)
    assert adapter.commands[-1].actuator_channels[POSITION_CHANNEL] == pytest.approx(0.08)

    with (tmp_path / "control.csv").open(encoding="utf-8", newline="") as stream:
        rows = tuple(csv.DictReader(stream))
    assert len(rows) == result.control_log_sample_count
    assert rows[-1]["stage"] == TrajectoryStage.PLACE.value
    diagnostics = [json.loads(line) for line in (tmp_path / "diagnostics.jsonl").read_text().splitlines()]
    assert {item["period"] for item in diagnostics} == {
        "planned_trajectory",
        "post_release_diagnostics",
    }


def test_diagnostic_samples_cannot_consume_events_between_control_samples(tmp_path):
    adapter = _RecordingAdapter()
    request = replace(
        _request(tmp_path, adapter),
        control_frequency_hz=7.0,
        control_log_frequency_hz=3.0,
        diagnostic_log_frequency_hz=11.0,
    )

    result = execute(request)

    assert result.execution_status is ExecutionStatus.COMPLETED
    event_commands = [command for command in adapter.commands if command.events]
    assert [command.events for command in event_commands] == [("grip",), ("release",)]
    assert event_commands[0].planned_time_s == pytest.approx(3.0 / 7.0)
    assert result.control_sample_count == len(adapter.commands) == 6


def test_control_log_frequency_cannot_exceed_control_frequency(tmp_path):
    adapter = _RecordingAdapter()
    request = replace(
        _request(tmp_path, adapter),
        control_log_frequency_hz=21.0,
    )

    result = execute(request)

    assert result.execution_status is ExecutionStatus.FAILED
    assert result.error_code is ExecutionErrorCode.INVALID_REQUEST
    assert not adapter.commands


def test_log_write_failure_does_not_interrupt_planned_controls(tmp_path, monkeypatch):
    from task2 import execution as execution_module

    adapter = _RecordingAdapter()
    original_write = execution_module._ControlLogWriter.write
    calls = 0

    def fail_second_write(self, command, observation):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk write failed")
        original_write(self, command, observation)

    monkeypatch.setattr(execution_module._ControlLogWriter, "write", fail_second_write)

    result = execute(_request(tmp_path, adapter))

    assert result.execution_status is ExecutionStatus.FAILED
    assert result.error_code is ExecutionErrorCode.LOG_IO_FAILED
    assert result.executed_planned_duration_s == pytest.approx(0.6)
    assert result.control_sample_count == len(adapter.commands) == 13
    assert result.control_log_sample_count == 1
    assert tuple(dict.fromkeys(command.stage for command in adapter.commands)) == tuple(
        TrajectoryStage
    )
    assert adapter.commands[-1].events == ("release",)


def test_mujoco_adapter_loads_formal_named_bindings():
    config = load_task2_config(ROOT / "configs/task2/pipeline.yaml")
    profile = config.end_effectors[0]
    robot_model = config.robot_model(profile)
    robot_config = config.mujoco_robots[profile.profile_id]
    adapter = MuJoCoExecutorAdapter(
        robot_config,
        config.mujoco_end_effectors[profile.profile_id],
        config.mujoco_objects,
    )
    command = ControlCommand(
        TrajectoryStage.PREGRASP,
        0.0,
        robot_model.joint_names,
        robot_model.default_joint_positions,
        (0.0,) * len(robot_model.joint_names),
        (0.0,) * len(robot_model.joint_names),
        {POSITION_CHANNEL: 0.02, FORCE_CHANNEL: 0.75},
    )
    binding = SceneObjectBinding(
        "center_marker",
        "marker",
        _pose("world", "center_marker", (0.5, 0.5, 0.08)),
    )

    try:
        observation = adapter.initialize((binding,), "center_marker", command)
        assert tuple(observation.joint_positions_rad) == robot_model.joint_names
        assert observation.target_object_state is not None
        assert observation.target_object_state.T_world_body.matrix[2][3] == pytest.approx(0.08)
        assert observation.T_world_tcp.child_frame == robot_model.tcp_frame

        core_bodies = (
            "target_marker",
            "target_rivet_gun",
            "target_standard_part",
            "target_tape",
        )
        assert set(robot_config.scene_object_body_names) == set(core_bodies)
        for name in core_bodies:
            assert adapter._mujoco.mj_name2id(
                adapter._model, adapter._mujoco.mjtObj.mjOBJ_BODY, name
            ) >= 0
        for object_config in config.mujoco_objects:
            assert adapter._mujoco.mj_name2id(
                adapter._model,
                adapter._mujoco.mjtObj.mjOBJ_JOINT,
                object_config.runtime_free_joint_name,
            ) >= 0
            for geom_name in object_config.runtime_collision_geom_names:
                assert adapter._mujoco.mj_name2id(
                    adapter._model,
                    adapter._mujoco.mjtObj.mjOBJ_GEOM,
                    geom_name,
                ) >= 0
        for name in (
            "target_hex_key",
            "target_drill_bit",
            "target_glove",
            "target_notebook",
        ):
            assert adapter._mujoco.mj_name2id(
                adapter._model, adapter._mujoco.mjtObj.mjOBJ_BODY, name
            ) == -1
        for name in (
            "target_hex_key_mesh",
            "target_drill_bit_mesh",
            "target_glove_mesh",
            "target_notebook_mesh",
        ):
            assert adapter._mujoco.mj_name2id(
                adapter._model, adapter._mujoco.mjtObj.mjOBJ_MESH, name
            ) == -1

        joint_index = robot_model.joint_names.index("joint_3")
        model_limit = float(
            adapter._model.actuator_ctrlrange[
                adapter._joint_actuator_ids["joint_3"], 1
            ]
        )
        float32_limit = unpack("!f", pack("!f", model_limit))[0]
        boundary_positions = list(command.positions_rad)
        boundary_positions[joint_index] = float32_limit
        adapter.apply_control(
            replace(command, positions_rad=tuple(boundary_positions))
        )
        assert adapter._data.ctrl[adapter._joint_actuator_ids["joint_3"]] == model_limit

        boundary_positions[joint_index] = float32_limit + 1e-12
        with pytest.raises(ControlInterfaceError, match="lies outside"):
            adapter.apply_control(
                replace(command, positions_rad=tuple(boundary_positions))
            )
    finally:
        adapter.close()
