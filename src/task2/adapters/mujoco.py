from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from pathlib import Path
from struct import pack, unpack
from typing import Any, Mapping

from ..contracts.execution import (
    BodyKinematicState,
    ContactRecord,
    ControlCommand,
    RuntimeObservation,
    SceneObjectBinding,
)
from ..contracts.geometry import RigidTransform
from ..contracts.planning import RobotModel, RobotState
from .errors import (
    AdapterUnavailableError,
    ControlInterfaceError,
    ModelLoadError,
    NumericalSimulationError,
)



@dataclass(frozen=True)
class RuntimeJointBinding:
    joint_name: str
    runtime_joint_name: str
    runtime_actuator_name: str

    def validate(self) -> None:
        if not self.joint_name or not self.runtime_joint_name or not self.runtime_actuator_name:
            raise ValueError("runtime joint binding names must be non-empty")


@dataclass(frozen=True)
class MuJoCoRobotConfig:
    model_asset: Path
    tcp_frame: str
    tcp_site_name: str
    joint_velocity_feedforward_scale: float
    ideal_unbounded_gravity_compensation_body_names: tuple[str, ...]
    joint_bindings: tuple[RuntimeJointBinding, ...]
    scene_object_body_names: tuple[str, ...]

    def validate(self) -> None:
        if not self.model_asset.is_file() or not self.tcp_frame or not self.tcp_site_name:
            raise ValueError("MuJoCo model asset and TCP site must be explicit")
        if (
            not isfinite(self.joint_velocity_feedforward_scale)
            or self.joint_velocity_feedforward_scale <= 0.0
        ):
            raise ValueError("MuJoCo joint velocity feedforward scale must be positive")
        gravity_bodies = self.ideal_unbounded_gravity_compensation_body_names
        if len(gravity_bodies) != len(set(gravity_bodies)) or any(
            not name.strip() for name in gravity_bodies
        ):
            raise ValueError(
                "MuJoCo ideal unbounded gravity-compensation body names must be unique"
            )
        if not self.joint_bindings:
            raise ValueError("MuJoCo robot config requires joint bindings")
        names = [item.joint_name for item in self.joint_bindings]
        runtime_joints = [item.runtime_joint_name for item in self.joint_bindings]
        actuators = [item.runtime_actuator_name for item in self.joint_bindings]
        if any(len(values) != len(set(values)) for values in (names, runtime_joints, actuators)):
            raise ValueError("MuJoCo robot joint and actuator bindings must be unique")
        for item in self.joint_bindings:
            item.validate()
        if not self.scene_object_body_names or len(self.scene_object_body_names) != len(
            set(self.scene_object_body_names)
        ):
            raise ValueError("MuJoCo scene object body names must be unique and non-empty")


@dataclass(frozen=True)
class ActuatorPositionBinding:
    channel_name: str
    runtime_actuator_name: str
    command_scale: float
    command_offset: float
    state_joint_names: tuple[str, ...]

    def validate(self) -> None:
        if not self.channel_name or not self.runtime_actuator_name or not self.state_joint_names:
            raise ValueError("MuJoCo actuator position binding must be explicit")
        if len(self.state_joint_names) != len(set(self.state_joint_names)):
            raise ValueError("MuJoCo actuator state joints must be unique")
        if not all(isfinite(value) for value in (self.command_scale, self.command_offset)):
            raise ValueError("MuJoCo actuator position mapping must be finite")
        if self.command_scale == 0.0:
            raise ValueError("MuJoCo actuator position scale must be non-zero")


@dataclass(frozen=True)
class ActuatorForceLimitBinding:
    channel_name: str
    runtime_actuator_name: str
    command_scale: float

    def validate(self) -> None:
        if not self.channel_name or not self.runtime_actuator_name:
            raise ValueError("MuJoCo force-limit binding must be explicit")
        if not isfinite(self.command_scale) or self.command_scale <= 0.0:
            raise ValueError("MuJoCo force-limit scale must be positive")


@dataclass(frozen=True)
class MuJoCoEndEffectorConfig:
    profile_id: str
    position: ActuatorPositionBinding
    force_limit: ActuatorForceLimitBinding
    contact_group_geoms: Mapping[str, tuple[str, ...]]

    def validate(self) -> None:
        if not self.profile_id:
            raise ValueError("MuJoCo end-effector profile ID must be non-empty")
        self.position.validate()
        self.force_limit.validate()
        if self.position.channel_name == self.force_limit.channel_name:
            raise ValueError("MuJoCo actuator position and force-limit channels must differ")
        if self.position.runtime_actuator_name != self.force_limit.runtime_actuator_name:
            raise ValueError("MuJoCo position and force limit must bind the same actuator")
        geoms = [name for values in self.contact_group_geoms.values() for name in values]
        if not self.contact_group_geoms or len(geoms) != len(set(geoms)):
            raise ValueError("MuJoCo contact groups must be explicit with unique geoms")


@dataclass(frozen=True)
class MuJoCoObjectConfig:
    object_type: str
    runtime_body_name: str
    runtime_free_joint_name: str
    runtime_collision_geom_names: tuple[str, ...]

    def validate(self) -> None:
        if not self.object_type or not self.runtime_body_name or not self.runtime_free_joint_name:
            raise ValueError("MuJoCo object binding must be explicit")
        if not self.runtime_collision_geom_names or len(self.runtime_collision_geom_names) != len(
            set(self.runtime_collision_geom_names)
        ):
            raise ValueError("MuJoCo object collision geoms must be unique and non-empty")


@dataclass(frozen=True)
class MuJoCoAdapterConfig:
    robots: Mapping[str, MuJoCoRobotConfig]
    end_effectors: Mapping[str, MuJoCoEndEffectorConfig]
    objects: tuple[MuJoCoObjectConfig, ...]


def parse_mujoco_adapter_config(
    payload: Mapping[str, object],
    base_dir: Path,
    profile_bindings: tuple[tuple[str, str], ...],
) -> MuJoCoAdapterConfig:
    raw_objects = _adapter_mapping(
        _adapter_value(payload, "objects", "MuJoCo adapter"),
        "MuJoCo adapter.objects",
    )
    objects = tuple(
        MuJoCoObjectConfig(
            object_type,
            _adapter_string(item, "runtime_body_name", f"MuJoCo object {object_type}"),
            _adapter_string(
                item, "runtime_free_joint_name", f"MuJoCo object {object_type}"
            ),
            _adapter_strings(
                item,
                "runtime_collision_geom_names",
                f"MuJoCo object {object_type}",
            ),
        )
        for object_type, item in (
            (name, _adapter_mapping(value, f"MuJoCo object {name}"))
            for name, value in raw_objects.items()
        )
    )
    for item in objects:
        item.validate()
    body_names = tuple(item.runtime_body_name for item in objects)
    raw_profiles = _adapter_mapping(
        _adapter_value(payload, "profiles", "MuJoCo adapter"),
        "MuJoCo adapter.profiles",
    )
    robots = {}
    end_effectors = {}
    for profile_id, adapter_profile_id in profile_bindings:
        raw = _adapter_mapping(
            _adapter_value(
                raw_profiles, adapter_profile_id, "MuJoCo adapter.profiles"
            ),
            f"MuJoCo adapter profile {adapter_profile_id}",
        )
        position = _adapter_mapping(
            _adapter_value(raw, "position", "MuJoCo adapter profile"),
            "MuJoCo adapter profile.position",
        )
        force = _adapter_mapping(
            _adapter_value(raw, "force_limit", "MuJoCo adapter profile"),
            "MuJoCo adapter profile.force_limit",
        )
        raw_contacts = _adapter_mapping(
            _adapter_value(raw, "contact_group_geoms", "MuJoCo adapter profile"),
            "MuJoCo adapter profile.contact_group_geoms",
        )
        contacts = {
            group: _adapter_string_values(
                value, f"MuJoCo contact_group_geoms.{group}"
            )
            for group, value in raw_contacts.items()
        }
        raw_joints = _adapter_list(
            _adapter_value(raw, "joint_bindings", "MuJoCo adapter profile"),
            "MuJoCo adapter profile.joint_bindings",
        )
        robot = MuJoCoRobotConfig(
            _adapter_path(raw, "model_asset", "MuJoCo adapter profile", base_dir),
            _adapter_string(raw, "tcp_frame", "MuJoCo adapter profile"),
            _adapter_string(raw, "tcp_site_name", "MuJoCo adapter profile"),
            _adapter_number(
                raw,
                "joint_velocity_feedforward_scale",
                "MuJoCo adapter profile",
            ),
            _adapter_strings(
                raw,
                "ideal_unbounded_gravity_compensation_body_names",
                "MuJoCo adapter profile",
            ),
            tuple(
                RuntimeJointBinding(
                    _adapter_string(item, "joint_name", "MuJoCo joint binding"),
                    _adapter_string(
                        item, "runtime_joint_name", "MuJoCo joint binding"
                    ),
                    _adapter_string(
                        item, "runtime_actuator_name", "MuJoCo joint binding"
                    ),
                )
                for item in (
                    _adapter_mapping(value, "MuJoCo joint binding")
                    for value in raw_joints
                )
            ),
            body_names,
        )
        end_effector = MuJoCoEndEffectorConfig(
            profile_id,
            ActuatorPositionBinding(
                _adapter_string(position, "channel_name", "MuJoCo position"),
                _adapter_string(
                    position, "runtime_actuator_name", "MuJoCo position"
                ),
                _adapter_number(position, "command_scale", "MuJoCo position"),
                _adapter_number(position, "command_offset", "MuJoCo position"),
                _adapter_strings(position, "state_joint_names", "MuJoCo position"),
            ),
            ActuatorForceLimitBinding(
                _adapter_string(force, "channel_name", "MuJoCo force limit"),
                _adapter_string(
                    force, "runtime_actuator_name", "MuJoCo force limit"
                ),
                _adapter_number(force, "command_scale", "MuJoCo force limit"),
            ),
            contacts,
        )
        robot.validate()
        end_effector.validate()
        robots[profile_id] = robot
        end_effectors[profile_id] = end_effector
    return MuJoCoAdapterConfig(robots, end_effectors, objects)


def read_activated_joint_state(
    robot_config: MuJoCoRobotConfig,
    robot_model: RobotModel,
    base_state: RobotState,
) -> RobotState:
    robot_config.validate()
    robot_model.validate()
    configured_names = tuple(item.joint_name for item in robot_config.joint_bindings)
    if configured_names != robot_model.joint_names:
        raise ValueError("MuJoCo runtime joint order does not match RobotModel")
    if len(base_state.joint_names) != len(set(base_state.joint_names)) or len(
        base_state.joint_names
    ) != len(base_state.positions_rad):
        raise ValueError("base robot state must contain unique named positions")
    provided = dict(zip(base_state.joint_names, base_state.positions_rad))
    if not set(provided).issubset(robot_model.joint_names):
        raise ValueError("base robot state contains joints outside the activated RobotModel")
    try:
        import mujoco
    except (ImportError, OSError) as exc:
        raise AdapterUnavailableError(f"MuJoCo import failed: {exc}") from exc
    try:
        model = mujoco.MjModel.from_xml_path(str(robot_config.model_asset.resolve()))
        data = mujoco.MjData(model)
    except Exception as exc:
        raise ModelLoadError(f"MuJoCo model load failed during tool activation: {exc}") from exc
    positions = []
    for binding in robot_config.joint_bindings:
        if binding.joint_name in provided:
            positions.append(provided[binding.joint_name])
            continue
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, binding.runtime_joint_name
        )
        if joint_id < 0:
            raise ControlInterfaceError(
                f"activated MuJoCo model has no joint {binding.runtime_joint_name!r}"
            )
        positions.append(float(data.qpos[int(model.jnt_qposadr[joint_id])]))
    result = RobotState(robot_model.joint_names, tuple(positions))
    result.validate(robot_model)
    return result


class MuJoCoExecutorAdapter:
    def __init__(
        self,
        robot_config: MuJoCoRobotConfig,
        end_effector_config: MuJoCoEndEffectorConfig,
        object_configs: tuple[MuJoCoObjectConfig, ...],
    ):
        robot_config.validate()
        end_effector_config.validate()
        for item in object_configs:
            item.validate()
        if not object_configs:
            raise ValueError("MuJoCoExecutorAdapter requires object bindings")
        object_types = [item.object_type for item in object_configs]
        body_names = [item.runtime_body_name for item in object_configs]
        if len(object_types) != len(set(object_types)) or len(body_names) != len(set(body_names)):
            raise ValueError("MuJoCo object types and runtime body names must be unique")
        if not set(body_names).issubset(robot_config.scene_object_body_names):
            raise ValueError("MuJoCo object bodies must belong to the configured scene model")
        self.adapter_id = "mujoco"
        self._robot_config = robot_config
        self._end_effector_config = end_effector_config
        self._object_configs = {item.object_type: item for item in object_configs}
        self._mujoco: Any | None = None
        self._np: Any | None = None
        self._model: Any | None = None
        self._data: Any | None = None
        self._joint_qpos_addresses: dict[str, int] = {}
        self._joint_actuator_ids: dict[str, int] = {}
        self._state_joint_qpos_addresses: tuple[int, ...] = ()
        self._position_actuator_id: int | None = None
        self._position_force_limit_n: float | None = None
        self._tcp_site_id: int | None = None
        self._target_body_id: int | None = None
        self._target_object_id: str | None = None
        self._world_frame: str | None = None
        self._base_timestep_s: float | None = None

    def initialize(
        self,
        scene_objects: tuple[SceneObjectBinding, ...],
        target_object_id: str,
        initial_command: ControlCommand,
    ) -> RuntimeObservation:
        if self._model is not None:
            raise ControlInterfaceError("MuJoCo adapter is already initialized")
        try:
            import mujoco
            import numpy as np
        except (ImportError, OSError) as exc:
            raise AdapterUnavailableError(f"MuJoCo import failed: {exc}") from exc
        model_path = self._robot_config.model_asset
        if not model_path.is_file():
            raise ModelLoadError(f"MuJoCo model asset does not exist: {model_path}")
        try:
            model = mujoco.MjModel.from_xml_path(str(model_path.resolve()))
            data = mujoco.MjData(model)
        except Exception as exc:
            raise ModelLoadError(f"MuJoCo model load failed: {exc}") from exc
        self._mujoco = mujoco
        self._np = np
        self._model = model
        self._data = data
        self._base_timestep_s = float(model.opt.timestep)
        try:
            self._apply_ideal_unbounded_gravity_compensation()
            self._resolve_control_interface(initial_command)
            self._configure_scene(scene_objects, target_object_id)
            self._set_initial_joint_state(initial_command)
            self.apply_control(initial_command)
            mujoco.mj_forward(model, data)
            self._check_numerical_state()
        except (ControlInterfaceError, NumericalSimulationError):
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise ControlInterfaceError(
                f"MuJoCo control interface initialization failed: {exc}"
            ) from exc
        return self.observe()

    def _apply_ideal_unbounded_gravity_compensation(self) -> None:
        model, _ = self._runtime()
        body_ids = tuple(
            self._named_id(
                self._mujoco.mjtObj.mjOBJ_BODY,
                name,
                "ideal unbounded gravity-compensation body",
            )
            for name in self._robot_config.ideal_unbounded_gravity_compensation_body_names
        )
        embedded = tuple(
            self._mujoco.mj_id2name(model, self._mujoco.mjtObj.mjOBJ_BODY, body_id)
            or f"body[{body_id}]"
            for body_id in body_ids
            if float(model.body_gravcomp[body_id]) != 0.0
        )
        if embedded:
            raise ControlInterfaceError(
                "configured MuJoCo bodies must not embed gravity compensation: "
                + ", ".join(embedded)
            )
        relevant_joint_ids = set()
        for body_id in body_ids:
            current_body_id = body_id
            while current_body_id != 0:
                first_joint_id = int(model.body_jntadr[current_body_id])
                joint_count = int(model.body_jntnum[current_body_id])
                relevant_joint_ids.update(
                    range(first_joint_id, first_joint_id + joint_count)
                )
                current_body_id = int(model.body_parentid[current_body_id])
        for joint_id in sorted(relevant_joint_ids):
            if bool(model.jnt_actgravcomp[joint_id]):
                joint_name = self._mujoco.mj_id2name(
                    model,
                    self._mujoco.mjtObj.mjOBJ_JOINT,
                    joint_id,
                )
                raise ControlInterfaceError(
                    "ideal unbounded gravity compensation cannot use actuator gravity "
                    f"compensation on {joint_name or f'joint[{joint_id}]'}"
                )
        for body_id in body_ids:
            model.body_gravcomp[body_id] = 1.0
        self._mujoco.mj_setConst(model, self._data)

    def apply_control(self, command: ControlCommand) -> None:
        model, data = self._runtime()
        configured_names = tuple(item.joint_name for item in self._robot_config.joint_bindings)
        if command.joint_names != configured_names:
            raise ControlInterfaceError(
                "trajectory joint order does not match the MuJoCo runtime binding"
            )
        command_vectors = (
            command.positions_rad,
            command.velocities_rad_s,
            command.accelerations_rad_s2,
        )
        if any(len(values) != len(command.joint_names) for values in command_vectors):
            raise ControlInterfaceError("trajectory joint command vector length is invalid")
        if any(not isfinite(value) for values in command_vectors for value in values):
            raise ControlInterfaceError("trajectory joint command vectors must be finite")
        for name, position, velocity in zip(
            command.joint_names,
            command.positions_rad,
            command.velocities_rad_s,
        ):
            actuator_id = self._joint_actuator_ids[name]
            self._set_actuator_control(
                actuator_id,
                self._position_servo_control(actuator_id, position, velocity),
            )

        position_binding = self._end_effector_config.position
        force_binding = self._end_effector_config.force_limit
        expected_channels = {position_binding.channel_name, force_binding.channel_name}
        if set(command.actuator_channels) != expected_channels:
            raise ControlInterfaceError(
                "trajectory actuator channels do not match the MuJoCo end-effector binding"
            )
        opening = float(command.actuator_channels[position_binding.channel_name])
        force_limit = float(command.actuator_channels[force_binding.channel_name])
        if not isfinite(opening) or not isfinite(force_limit) or force_limit <= 0.0:
            raise ControlInterfaceError("end-effector commands must be finite with positive force")
        position_control = (
            opening * position_binding.command_scale + position_binding.command_offset
        )
        assert self._position_actuator_id is not None
        self._set_actuator_control(self._position_actuator_id, position_control)
        runtime_force_limit = force_limit * force_binding.command_scale
        if (
            self._position_force_limit_n is None
            or runtime_force_limit > self._position_force_limit_n + 1e-10
        ):
            raise ControlInterfaceError(
                "trajectory force-limit command exceeds the configured MuJoCo actuator range"
            )
        model.actuator_forcelimited[self._position_actuator_id] = 1
        model.actuator_forcerange[self._position_actuator_id, 0] = -runtime_force_limit
        model.actuator_forcerange[self._position_actuator_id, 1] = runtime_force_limit
        self._check_numerical_state()

    def advance(self, duration_s: float) -> RuntimeObservation:
        if not isfinite(duration_s) or duration_s < 0.0:
            raise ControlInterfaceError("MuJoCo advance duration must be finite and non-negative")
        model, data = self._runtime()
        assert self._base_timestep_s is not None
        contacts: dict[tuple[str, str], ContactRecord] = {}
        remaining = duration_s
        try:
            while remaining > 1e-12:
                step_s = min(self._base_timestep_s, remaining)
                model.opt.timestep = step_s
                self._mujoco.mj_step(model, data)
                self._check_numerical_state()
                self._merge_contacts(contacts, self._current_contacts(step_s))
                remaining -= step_s
        finally:
            model.opt.timestep = self._base_timestep_s
        return self._observation(tuple(contacts[key] for key in sorted(contacts)))

    def observe(self) -> RuntimeObservation:
        return self._observation(self._current_contacts(float(self._model.opt.timestep)))

    def close(self) -> None:
        self._data = None
        self._model = None
        self._mujoco = None
        self._np = None

    def _resolve_control_interface(self, initial_command: ControlCommand) -> None:
        model, _ = self._runtime()
        mujoco = self._mujoco
        configured_names = tuple(item.joint_name for item in self._robot_config.joint_bindings)
        if initial_command.joint_names != configured_names:
            raise ControlInterfaceError(
                "trajectory joint order does not match Execution robot config"
            )
        for binding in self._robot_config.joint_bindings:
            joint_id = self._named_id(
                mujoco.mjtObj.mjOBJ_JOINT,
                binding.runtime_joint_name,
                "joint",
            )
            actuator_id = self._named_id(
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                binding.runtime_actuator_name,
                "actuator",
            )
            if int(model.jnt_type[joint_id]) not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                raise ControlInterfaceError(
                    f"runtime joint {binding.runtime_joint_name} is not one-dimensional"
                )
            self._joint_qpos_addresses[binding.joint_name] = int(model.jnt_qposadr[joint_id])
            self._joint_actuator_ids[binding.joint_name] = actuator_id
        self._tcp_site_id = self._named_id(
            mujoco.mjtObj.mjOBJ_SITE,
            self._robot_config.tcp_site_name,
            "TCP site",
        )
        position_binding = self._end_effector_config.position
        self._position_actuator_id = self._named_id(
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            position_binding.runtime_actuator_name,
            "end-effector actuator",
        )
        state_addresses = []
        for name in position_binding.state_joint_names:
            joint_id = self._named_id(mujoco.mjtObj.mjOBJ_JOINT, name, "end-effector joint")
            if int(model.jnt_type[joint_id]) not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                raise ControlInterfaceError(f"end-effector state joint {name} is not one-dimensional")
            state_addresses.append(int(model.jnt_qposadr[joint_id]))
        self._state_joint_qpos_addresses = tuple(state_addresses)
        force_range = model.actuator_forcerange[self._position_actuator_id]
        self._position_force_limit_n = min(abs(float(force_range[0])), abs(float(force_range[1])))
        for geoms in self._end_effector_config.contact_group_geoms.values():
            for name in geoms:
                self._named_id(mujoco.mjtObj.mjOBJ_GEOM, name, "contact geom")

    def _configure_scene(
        self,
        scene_objects: tuple[SceneObjectBinding, ...],
        target_object_id: str,
    ) -> None:
        model, data = self._runtime()
        mujoco = self._mujoco
        selected = {
            self._object_configs[item.object_type].runtime_body_name: item
            for item in scene_objects
            if item.object_type in self._object_configs
        }
        target = next((item for item in scene_objects if item.object_id == target_object_id), None)
        if target is None:
            raise ControlInterfaceError("target object binding is absent from the MuJoCo scene")
        for item in scene_objects:
            configured = self._object_configs.get(item.object_type)
            if configured is None:
                raise ControlInterfaceError(
                    f"scene object {item.object_id} does not match the adapter object config"
                )
        for body_name in self._robot_config.scene_object_body_names:
            body_id = self._named_id(mujoco.mjtObj.mjOBJ_BODY, body_name, "scene object body")
            first_geom = int(model.body_geomadr[body_id])
            geom_count = int(model.body_geomnum[body_id])
            for geom_id in range(first_geom, first_geom + geom_count):
                model.geom_contype[geom_id] = 0
                model.geom_conaffinity[geom_id] = 0
            if body_name not in selected:
                joint_id = int(model.body_jntadr[body_id])
                if joint_id >= 0 and int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
                    qpos = int(model.jnt_qposadr[joint_id])
                    data.qpos[qpos : qpos + 7] = (0.0, 0.0, -5.0, 1.0, 0.0, 0.0, 0.0)

        for item in scene_objects:
            config = self._object_configs[item.object_type]
            body_id = self._named_id(mujoco.mjtObj.mjOBJ_BODY, config.runtime_body_name, "object body")
            joint_id = self._named_id(
                mujoco.mjtObj.mjOBJ_JOINT,
                config.runtime_free_joint_name,
                "object free joint",
            )
            if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
                raise ControlInterfaceError(
                    f"runtime object joint {config.runtime_free_joint_name} is not free"
                )
            if int(model.jnt_bodyid[joint_id]) != body_id:
                raise ControlInterfaceError("runtime object body and free joint are disconnected")
            for geom_name in config.runtime_collision_geom_names:
                geom_id = self._named_id(mujoco.mjtObj.mjOBJ_GEOM, geom_name, "object collision geom")
                if int(model.geom_bodyid[geom_id]) != body_id:
                    raise ControlInterfaceError(
                        f"runtime object collision geom {geom_name} belongs to another body"
                    )
                model.geom_contype[geom_id] = 1
                model.geom_conaffinity[geom_id] = 1
            matrix = item.T_world_object.matrix
            quaternion = _rotation_matrix_to_quaternion_wxyz(matrix)
            qpos = int(model.jnt_qposadr[joint_id])
            data.qpos[qpos : qpos + 7] = (
                matrix[0][3],
                matrix[1][3],
                matrix[2][3],
                *quaternion,
            )
        target_body = self._named_id(
            mujoco.mjtObj.mjOBJ_BODY,
            self._object_configs[target.object_type].runtime_body_name,
            "target object body",
        )
        self._target_body_id = target_body
        self._target_object_id = target_object_id
        self._world_frame = target.T_world_object.parent_frame

    def _set_initial_joint_state(self, command: ControlCommand) -> None:
        _, data = self._runtime()
        for name, value in zip(command.joint_names, command.positions_rad):
            data.qpos[self._joint_qpos_addresses[name]] = value
        position = self._end_effector_config.position
        opening = float(command.actuator_channels[position.channel_name])
        state_position = opening * position.command_scale + position.command_offset
        for address in self._state_joint_qpos_addresses:
            data.qpos[address] = state_position
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        data.time = 0.0

    def _set_actuator_control(self, actuator_id: int, value: float) -> None:
        model, data = self._runtime()
        if not isfinite(value):
            raise ControlInterfaceError("MuJoCo actuator control must be finite")
        if int(model.actuator_ctrllimited[actuator_id]):
            low, high = (float(item) for item in model.actuator_ctrlrange[actuator_id])
            canonical_value = _canonical_control_value(value, low, high)
            if canonical_value is None:
                name = self._mujoco.mj_id2name(
                    model, self._mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
                )
                raise ControlInterfaceError(
                    f"MuJoCo actuator command {value!r} for {name} lies outside "
                    f"[{low}, {high}]"
                )
            value = canonical_value
        data.ctrl[actuator_id] = value

    def _position_servo_control(
        self,
        actuator_id: int,
        position: float,
        velocity: float,
    ) -> float:
        model, _ = self._runtime()
        gain = float(model.actuator_gainprm[actuator_id, 0])
        position_bias = float(model.actuator_biasprm[actuator_id, 1])
        velocity_bias = float(model.actuator_biasprm[actuator_id, 2])
        if gain <= 0.0 or position_bias != -gain:
            raise ControlInterfaceError(
                "MuJoCo joint binding requires an affine position servo"
            )
        return position - (
            self._robot_config.joint_velocity_feedforward_scale
            * velocity_bias
            * velocity
            / gain
        )

    def _observation(self, contacts: tuple[ContactRecord, ...]) -> RuntimeObservation:
        model, data = self._runtime()
        assert self._tcp_site_id is not None
        assert self._target_body_id is not None
        assert self._target_object_id is not None
        assert self._world_frame is not None
        tcp_rotation = data.site_xmat[self._tcp_site_id].reshape(3, 3)
        tcp_position = data.site_xpos[self._tcp_site_id]
        tcp_pose = _rigid_transform(
            self._world_frame,
            self._robot_config.tcp_frame,
            tcp_rotation,
            tcp_position,
        )
        body_rotation = data.xmat[self._target_body_id].reshape(3, 3)
        body_position = data.xpos[self._target_body_id]
        body_pose = _rigid_transform(
            self._world_frame,
            self._target_object_id,
            body_rotation,
            body_position,
        )
        velocity = self._np.zeros(6, dtype=float)
        self._mujoco.mj_objectVelocity(
            model,
            data,
            self._mujoco.mjtObj.mjOBJ_BODY,
            self._target_body_id,
            velocity,
            0,
        )
        state = BodyKinematicState(
            body_pose,
            tuple(float(value) for value in velocity[3:6]),
            tuple(float(value) for value in velocity[0:3]),
        )
        position_binding = self._end_effector_config.position
        force_binding = self._end_effector_config.force_limit
        opening = sum(float(data.qpos[address]) for address in self._state_joint_qpos_addresses)
        assert self._position_actuator_id is not None
        return RuntimeObservation(
            runtime_time_s=float(data.time),
            joint_positions_rad={
                name: float(data.qpos[address])
                for name, address in self._joint_qpos_addresses.items()
            },
            actuator_states={
                position_binding.channel_name: opening,
                force_binding.channel_name: abs(
                    float(data.actuator_force[self._position_actuator_id])
                ),
            },
            T_world_tcp=tcp_pose,
            target_object_state=state,
            contacts=contacts,
        )

    def _current_contacts(self, step_s: float) -> tuple[ContactRecord, ...]:
        model, data = self._runtime()
        contacts: dict[tuple[str, str], ContactRecord] = {}
        for index in range(data.ncon):
            contact = data.contact[index]
            first = self._mujoco.mj_id2name(
                model, self._mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)
            ) or f"geom_{int(contact.geom1)}"
            second = self._mujoco.mj_id2name(
                model, self._mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)
            ) or f"geom_{int(contact.geom2)}"
            pair = tuple(sorted((first, second)))
            force = self._np.zeros(6, dtype=float)
            self._mujoco.mj_contactForce(model, data, index, force)
            normal_force = abs(float(force[0]))
            item = ContactRecord(
                pair[0],
                pair[1],
                max(0.0, -float(contact.dist)),
                normal_force,
                normal_force * step_s,
            )
            self._merge_contacts(contacts, (item,))
        return tuple(contacts[key] for key in sorted(contacts))

    @staticmethod
    def _merge_contacts(
        destination: dict[tuple[str, str], ContactRecord],
        source: tuple[ContactRecord, ...],
    ) -> None:
        for item in source:
            key = item.normalized_pair()
            current = destination.get(key)
            if current is None:
                destination[key] = item
            else:
                destination[key] = ContactRecord(
                    key[0],
                    key[1],
                    max(current.maximum_penetration_m, item.maximum_penetration_m),
                    max(current.maximum_normal_force_n, item.maximum_normal_force_n),
                    max(current.maximum_impact_impulse_ns, item.maximum_impact_impulse_ns),
                )

    def _check_numerical_state(self) -> None:
        _, data = self._runtime()
        if not all(
            self._np.isfinite(values).all()
            for values in (data.qpos, data.qvel, data.qacc, data.ctrl)
        ) or not isfinite(float(data.time)):
            raise NumericalSimulationError("MuJoCo state became non-finite")

    def _named_id(self, object_type: Any, name: str, label: str) -> int:
        model, _ = self._runtime()
        value = int(self._mujoco.mj_name2id(model, object_type, name))
        if value < 0:
            raise ControlInterfaceError(f"MuJoCo {label} does not exist: {name}")
        return value

    def _runtime(self) -> tuple[Any, Any]:
        if self._model is None or self._data is None or self._mujoco is None or self._np is None:
            raise ControlInterfaceError("MuJoCo adapter is not initialized")
        return self._model, self._data


def _canonical_control_value(value: float, low: float, high: float) -> float | None:
    if low <= value <= high:
        return value
    for boundary in (low, high):
        if value == unpack("!f", pack("!f", boundary))[0]:
            return boundary
    return None


def _rigid_transform(
    parent_frame: str,
    child_frame: str,
    rotation: Any,
    position: Any,
) -> RigidTransform:
    matrix = tuple(
        tuple(float(rotation[row, column]) for column in range(3))
        + (float(position[row]),)
        for row in range(3)
    ) + ((0.0, 0.0, 0.0, 1.0),)
    return RigidTransform(parent_frame, child_frame, matrix)  # type: ignore[arg-type]


def _rotation_matrix_to_quaternion_wxyz(matrix) -> tuple[float, float, float, float]:
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = 2.0 * sqrt(trace + 1.0)
        quaternion = (
            0.25 * scale,
            (matrix[2][1] - matrix[1][2]) / scale,
            (matrix[0][2] - matrix[2][0]) / scale,
            (matrix[1][0] - matrix[0][1]) / scale,
        )
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        scale = 2.0 * sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2])
        quaternion = (
            (matrix[2][1] - matrix[1][2]) / scale,
            0.25 * scale,
            (matrix[0][1] + matrix[1][0]) / scale,
            (matrix[0][2] + matrix[2][0]) / scale,
        )
    elif matrix[1][1] > matrix[2][2]:
        scale = 2.0 * sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2])
        quaternion = (
            (matrix[0][2] - matrix[2][0]) / scale,
            (matrix[0][1] + matrix[1][0]) / scale,
            0.25 * scale,
            (matrix[1][2] + matrix[2][1]) / scale,
        )
    else:
        scale = 2.0 * sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1])
        quaternion = (
            (matrix[1][0] - matrix[0][1]) / scale,
            (matrix[0][2] + matrix[2][0]) / scale,
            (matrix[1][2] + matrix[2][1]) / scale,
            0.25 * scale,
        )
    norm = sqrt(sum(value * value for value in quaternion))
    if norm <= 0.0:
        raise ControlInterfaceError("object pose rotation produced a zero quaternion")
    return tuple(value / norm for value in quaternion)


def _adapter_value(payload: Mapping[str, object], key: str, label: str) -> object:
    if key not in payload:
        raise ValueError(f"{label}.{key} is required")
    return payload[key]


def _adapter_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return value


def _adapter_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _adapter_string(payload: Mapping[str, object], key: str, label: str) -> str:
    value = _adapter_value(payload, key, label)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value


def _adapter_string_values(value: object, label: str) -> tuple[str, ...]:
    values = _adapter_list(value, label)
    result = tuple(item for item in values if isinstance(item, str) and item.strip())
    if len(result) != len(values):
        raise ValueError(f"{label} must contain non-empty strings")
    return result


def _adapter_strings(
    payload: Mapping[str, object], key: str, label: str
) -> tuple[str, ...]:
    return _adapter_string_values(_adapter_value(payload, key, label), f"{label}.{key}")


def _adapter_number(payload: Mapping[str, object], key: str, label: str) -> float:
    value = _adapter_value(payload, key, label)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}.{key} must be a number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{label}.{key} must be finite")
    return result


def _adapter_path(
    payload: Mapping[str, object], key: str, label: str, base_dir: Path
) -> Path:
    value = Path(_adapter_string(payload, key, label))
    return value.resolve() if value.is_absolute() else (base_dir / value).resolve()
