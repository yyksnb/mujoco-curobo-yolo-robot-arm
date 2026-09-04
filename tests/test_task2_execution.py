from __future__ import annotations

import csv
import json
from dataclasses import replace
from math import pi
from pathlib import Path

import pytest

from task2.adapters.mujoco import MuJoCoExecutorAdapter, read_activated_joint_state
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
    RobotModel,
    RobotState,
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


def test_reusable_mujoco_assets_do_not_embed_gravity_compensation():
    import mujoco

    tool_asset = (
        ROOT
        / "examples/mujoco/so101_active_wrist_parallel/active_wrist_parallel.xml"
    )
    for asset in (
        tool_asset,
        ROOT / "examples/mujoco/kinova_gen3/gen3_with_active_wrist_parallel.xml",
    ):
        model = mujoco.MjModel.from_xml_path(str(asset))
        assert not any(float(value) for value in model.body_gravcomp)


def test_mujoco_runtime_joint_limits_match_formal_robot_model():
    config = load_task2_config(ROOT / "configs/task2/pipeline.yaml")
    profile = config.end_effectors[0]
    robot_model = config.robot_model(profile)
    robot_config = config.mujoco_robots[profile.profile_id]

    state = read_activated_joint_state(
        robot_config,
        robot_model,
        RobotState(robot_model.joint_names, robot_model.default_joint_positions),
    )

    assert state.joint_names == robot_model.joint_names


def test_mujoco_runtime_rejects_joint_limits_that_differ_from_robot_model():
    config = load_task2_config(ROOT / "configs/task2/pipeline.yaml")
    profile = config.end_effectors[0]
    robot_model = config.robot_model(profile)
    robot_config = config.mujoco_robots[profile.profile_id]
    joint_limits = list(robot_model.joint_limits)
    joint_limits[1] = replace(
        joint_limits[1],
        minimum_position_rad=-2.24,
        maximum_position_rad=2.24,
    )
    mismatched_model = RobotModel(
        robot_model.model_id,
        robot_model.base_frame,
        robot_model.tcp_frame,
        robot_model.joint_names,
        tuple(joint_limits),
        robot_model.default_joint_positions,
    )

    with pytest.raises(ControlInterfaceError, match="do not match RobotModel"):
        read_activated_joint_state(
            robot_config,
            mismatched_model,
            RobotState(
                mismatched_model.joint_names,
                mismatched_model.default_joint_positions,
            ),
        )


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


class _PostReleaseTransientAdapter(_RecordingAdapter):
    def observe(self):
        observation = super().observe()
        state = observation.target_object_state
        assert state is not None
        transient_z = 0.2 if 0.619 < self.time_s < 0.621 else 0.0
        return replace(
            observation,
            target_object_state=replace(
                state,
                T_world_body=_pose("world", "object", (0.0, 0.0, transient_z)),
            ),
        )


def test_100_hz_control_with_50_hz_logs_only_changes_persistence(tmp_path):
    adapter_50_hz = _PostReleaseTransientAdapter()
    diagnostics = replace(
        _request(tmp_path, adapter_50_hz).diagnostics_config,
        target_position_world_m=(0.0, 0.0, 0.0),
    )
    request_50_hz = replace(
        _request(tmp_path / "50_hz", adapter_50_hz),
        control_frequency_hz=100.0,
        control_log_frequency_hz=50.0,
        diagnostic_log_frequency_hz=50.0,
        diagnostics_config=diagnostics,
    )
    adapter_25_hz = _PostReleaseTransientAdapter()
    request_25_hz = replace(
        request_50_hz,
        adapter=adapter_25_hz,
        control_log_frequency_hz=25.0,
        diagnostic_log_frequency_hz=25.0,
        control_log_path=str(tmp_path / "25_hz/control.csv"),
        diagnostic_log_path=str(tmp_path / "25_hz/diagnostics.jsonl"),
    )

    result_50_hz = execute(request_50_hz)
    result_25_hz = execute(request_25_hz)

    assert result_50_hz.execution_status is ExecutionStatus.COMPLETED
    assert result_50_hz.dynamics_diagnostics.success
    assert result_50_hz.execution_status is result_25_hz.execution_status
    assert result_50_hz.dynamics_diagnostics == result_25_hz.dynamics_diagnostics
    assert result_50_hz.control_sample_count == result_25_hz.control_sample_count == 61
    assert adapter_50_hz.commands == adapter_25_hz.commands
    assert [
        event for command in adapter_50_hz.commands for event in command.events
    ] == ["grip", "release"]

    with Path(request_50_hz.control_log_path).open(
        encoding="utf-8", newline=""
    ) as stream:
        control_rows = tuple(csv.DictReader(stream))
    dynamics_rows = tuple(
        json.loads(line)
        for line in Path(request_50_hz.diagnostic_log_path).read_text().splitlines()
    )
    assert len(control_rows) == result_50_hz.control_log_sample_count == 31
    assert len(dynamics_rows) == result_50_hz.diagnostic_log_sample_count == 41
    control_times = [float(row["runtime_time_s"]) for row in control_rows]
    dynamics_times = [float(row["runtime_time_s"]) for row in dynamics_rows]
    assert all(right > left for left, right in zip(control_times, control_times[1:]))
    assert all(right > left for left, right in zip(dynamics_times, dynamics_times[1:]))
    assert [
        right - left for left, right in zip(control_times, control_times[1:])
    ] == pytest.approx([0.02] * (len(control_times) - 1))
    assert [
        right - left for left, right in zip(dynamics_times, dynamics_times[1:])
    ] == pytest.approx([0.02] * (len(dynamics_times) - 1))
    assert [
        event
        for row in control_rows
        for event in json.loads(row["events"])
    ] == ["grip", "release"]
    assert all(row["actual_positions_rad"] for row in control_rows)
    assert all(row["actuator_states"] for row in control_rows)
    assert all(row["tcp_pose"] for row in dynamics_rows)
    assert all(row["target_object_state"] for row in dynamics_rows)
    assert all("contacts" in row for row in dynamics_rows)


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


def test_mujoco_adapter_preserves_formal_control_and_gravity_semantics():
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
        {POSITION_CHANNEL: 0.084, FORCE_CHANNEL: 0.75},
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

        grip_id = adapter._position_actuator_id
        assert grip_id is not None
        assert robot_config.joint_velocity_feedforward_scale == pytest.approx(1.0)
        assert config.mujoco_end_effectors[
            profile.profile_id
        ].force_limit.command_scale == pytest.approx(1.0)
        assert tuple(adapter._model.actuator_ctrlrange[grip_id]) == pytest.approx(
            (0.0, 0.042)
        )
        assert adapter._data.ctrl[grip_id] == pytest.approx(0.042)
        assert tuple(adapter._model.actuator_forcerange[grip_id]) == pytest.approx(
            (-0.75, 0.75)
        )
        assert observation.actuator_states[POSITION_CHANNEL] == pytest.approx(0.084)
        moment_start = int(adapter._data.moment_rowadr[grip_id])
        moment_count = int(adapter._data.moment_rownnz[grip_id])
        moment_slice = slice(moment_start, moment_start + moment_count)
        state_joint_ids = tuple(
            adapter._mujoco.mj_name2id(
                adapter._model,
                adapter._mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )
            for name in config.mujoco_end_effectors[
                profile.profile_id
            ].position.state_joint_names
        )
        state_joint_dofs = tuple(
            int(adapter._model.jnt_dofadr[joint_id])
            for joint_id in state_joint_ids
        )
        state_joint_qpos = tuple(
            int(adapter._model.jnt_qposadr[joint_id])
            for joint_id in state_joint_ids
        )
        assert tuple(
            float(adapter._data.qpos[address]) for address in state_joint_qpos
        ) == pytest.approx((0.042, 0.042))
        assert set(adapter._data.moment_colind[moment_slice]) == set(state_joint_dofs)
        assert tuple(adapter._data.actuator_moment[moment_slice]) == pytest.approx(
            (0.5, 0.5)
        )
        assert adapter._model.actuator_gainprm[grip_id, 0] == pytest.approx(2400.0)
        assert adapter._model.actuator_biasprm[grip_id, 1] == pytest.approx(-2400.0)
        assert adapter._model.actuator_biasprm[grip_id, 2] == pytest.approx(-15.0)

        adapter._data.ctrl[grip_id] = 0.0
        adapter._mujoco.mj_forward(adapter._model, adapter._data)
        actuator_force = float(adapter._data.actuator_force[grip_id])
        assert actuator_force == pytest.approx(-0.75)
        assert tuple(
            float(adapter._data.qfrc_actuator[dof]) for dof in state_joint_dofs
        ) == pytest.approx((0.5 * actuator_force, 0.5 * actuator_force))
        adapter.apply_control(command)

        moving_velocities = [0.0] * len(command.joint_names)
        moving_velocities[0] = 0.2
        adapter.apply_control(
            replace(command, velocities_rad_s=tuple(moving_velocities))
        )
        joint_1_actuator = adapter._joint_actuator_ids["joint_1"]
        gain = float(adapter._model.actuator_gainprm[joint_1_actuator, 0])
        velocity_bias = float(adapter._model.actuator_biasprm[joint_1_actuator, 2])
        expected_control = command.positions_rad[0] - (
            robot_config.joint_velocity_feedforward_scale
            * velocity_bias
            * 0.2
            / gain
        )
        assert adapter._data.ctrl[joint_1_actuator] == pytest.approx(expected_control)
        assert adapter._data.ctrl[joint_1_actuator] != command.positions_rad[0]

        with pytest.raises(ControlInterfaceError, match="vector length"):
            adapter.apply_control(replace(command, velocities_rad_s=(0.0,)))

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
        compensated_bodies = {
            adapter._mujoco.mj_id2name(
                adapter._model,
                adapter._mujoco.mjtObj.mjOBJ_BODY,
                body_id,
            )
            for body_id, value in enumerate(adapter._model.body_gravcomp)
            if float(value) == 1.0
        }
        assert compensated_bodies == set(
            robot_config.ideal_unbounded_gravity_compensation_body_names
        ) == {
            "robot_base_link",
            "robot_shoulder_link",
            "robot_half_arm_1_link",
            "robot_half_arm_2_link",
            "robot_forearm_link",
            "robot_spherical_wrist_1_link",
            "robot_spherical_wrist_2_link",
            "robot_bracelet_link",
            "robot_task2_tool_mount",
            "robot_task2_tool_wrist_output",
            "robot_task2_tool_left_jaw",
            "robot_task2_tool_right_jaw",
        }
        static_positions = (0.0, 0.5, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0)
        static_command = replace(
            command,
            positions_rad=static_positions,
        )
        for name, value in zip(static_command.joint_names, static_positions):
            adapter._data.qpos[adapter._joint_qpos_addresses[name]] = value
        adapter._data.qvel[:] = 0.0
        adapter.apply_control(static_command)
        adapter._mujoco.mj_forward(adapter._model, adapter._data)
        compensated_body_ids = tuple(
            adapter._mujoco.mj_name2id(
                adapter._model,
                adapter._mujoco.mjtObj.mjOBJ_BODY,
                name,
            )
            for name in robot_config.ideal_unbounded_gravity_compensation_body_names
        )
        compensation_joint_ids = set()
        for body_id in compensated_body_ids:
            current_body_id = body_id
            while current_body_id != 0:
                first_joint_id = int(adapter._model.body_jntadr[current_body_id])
                joint_count = int(adapter._model.body_jntnum[current_body_id])
                compensation_joint_ids.update(
                    range(first_joint_id, first_joint_id + joint_count)
                )
                current_body_id = int(adapter._model.body_parentid[current_body_id])
        compensation_joint_names = {
            adapter._mujoco.mj_id2name(
                adapter._model,
                adapter._mujoco.mjtObj.mjOBJ_JOINT,
                joint_id,
            )
            for joint_id in compensation_joint_ids
        }
        assert {
            "robot_task2_tool_left_jaw_joint",
            "robot_task2_tool_right_jaw_joint",
        }.issubset(compensation_joint_names)
        compensation_dofs = tuple(
            int(adapter._model.jnt_dofadr[joint_id])
            for joint_id in sorted(compensation_joint_ids)
        )
        assert not any(
            bool(adapter._model.jnt_actgravcomp[joint_id])
            for joint_id in compensation_joint_ids
        )
        assert tuple(
            float(adapter._data.qfrc_actuator[dof]) for dof in compensation_dofs
        ) == pytest.approx((0.0,) * len(compensation_dofs), abs=1e-12)
        assert any(
            abs(float(adapter._data.qfrc_passive[dof])) > 1e-6
            for dof in compensation_dofs
        )
        for body_id in compensated_body_ids:
            adapter._model.body_gravcomp[body_id] = 0.0
        unrelated_body_id = adapter._mujoco.mj_name2id(
            adapter._model,
            adapter._mujoco.mjtObj.mjOBJ_BODY,
            "target_marker",
        )
        adapter._model.body_gravcomp[unrelated_body_id] = 0.25
        left_jaw_joint_id = adapter._mujoco.mj_name2id(
            adapter._model,
            adapter._mujoco.mjtObj.mjOBJ_JOINT,
            "robot_task2_tool_left_jaw_joint",
        )
        adapter._model.jnt_actgravcomp[left_jaw_joint_id] = True
        with pytest.raises(ControlInterfaceError, match="left_jaw_joint"):
            adapter._apply_ideal_unbounded_gravity_compensation()
        adapter._model.jnt_actgravcomp[left_jaw_joint_id] = False
        adapter._apply_ideal_unbounded_gravity_compensation()
        assert adapter._model.body_gravcomp[unrelated_body_id] == pytest.approx(0.25)
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
        joint_3_actuator = adapter._joint_actuator_ids["joint_3"]
        assert robot_model.joint_limits[joint_index].continuous
        assert not adapter._model.actuator_ctrllimited[joint_3_actuator]
        boundary_positions = list(command.positions_rad)
        boundary_positions[joint_index] = pi + 0.25
        adapter.apply_control(
            replace(command, positions_rad=tuple(boundary_positions))
        )
        assert adapter._data.ctrl[joint_3_actuator] == pytest.approx(pi + 0.25)
    finally:
        adapter.close()
