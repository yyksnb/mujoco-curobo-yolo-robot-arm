from __future__ import annotations

import json
import math
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any


END_EFFECTOR_CONFIG_SCHEMA = "task2_mujoco_end_effectors/v1"
NO_END_EFFECTOR = "none"


class EndEffectorConfigError(ValueError):
    """Raised when an end-effector registry violates its public contract."""


class EndEffectorAssemblyError(RuntimeError):
    """Raised when a configured end effector cannot be assembled safely."""


@dataclass(frozen=True)
class EndEffectorCommandDefinition:
    channel: str
    actuator_name: str
    open_ctrl: float
    closed_ctrl: float

    def control_value(self, closed_fraction: float) -> float:
        if isinstance(closed_fraction, bool) or not isinstance(
            closed_fraction, (int, float)
        ):
            raise TypeError("end-effector command fraction must be a number")
        value = float(closed_fraction)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                "end-effector command fraction must be finite and within [0, 1]"
            )
        return self.open_ctrl + value * (self.closed_ctrl - self.open_ctrl)


@dataclass(frozen=True)
class EndEffectorDefinition:
    end_effector_id: str
    model_path: Path
    mount_body_name: str
    mount_position_m: tuple[float, float, float]
    mount_quat_wxyz: tuple[float, float, float, float]
    tcp_position_m: tuple[float, float, float]
    tcp_quat_wxyz: tuple[float, float, float, float]
    commands: tuple[EndEffectorCommandDefinition, ...]


@dataclass(frozen=True)
class EndEffectorRegistry:
    source_path: Path
    prefix: str
    detached_tcp_site_name: str
    attached_tcp_site_name: str
    definitions: tuple[EndEffectorDefinition, ...]

    @property
    def available_ids(self) -> tuple[str, ...]:
        return (NO_END_EFFECTOR, *(item.end_effector_id for item in self.definitions))

    def resolve(self, end_effector_id: str) -> EndEffectorDefinition | None:
        if end_effector_id == NO_END_EFFECTOR:
            return None
        for definition in self.definitions:
            if definition.end_effector_id == end_effector_id:
                return definition
        available = ", ".join(self.available_ids)
        raise EndEffectorConfigError(
            f"unknown end_effector_id {end_effector_id!r}; available: {available}"
        )


@dataclass(frozen=True)
class ResolvedEndEffectorCommand:
    definition: EndEffectorCommandDefinition
    actuator_name: str
    actuator_id: int


@dataclass(frozen=True)
class EndEffectorHandle:
    end_effector_id: str
    tcp_site_name: str
    tcp_site_id: int
    commands: tuple[ResolvedEndEffectorCommand, ...]

    @property
    def command_channels(self) -> tuple[str, ...]:
        return tuple(item.definition.channel for item in self.commands)

    def command(self, channel: str) -> ResolvedEndEffectorCommand:
        for command in self.commands:
            if command.definition.channel == channel:
                return command
        available = ", ".join(self.command_channels) or "none"
        raise ValueError(
            f"end effector {self.end_effector_id!r} has no command channel "
            f"{channel!r}; available: {available}"
        )


@dataclass
class MujocoSimulation:
    scene_path: Path
    registry_source_path: Path
    tool_prefix: str
    model: Any
    data: Any
    end_effector: EndEffectorHandle

    def set_end_effector_command(
        self,
        channel: str,
        closed_fraction: float,
    ) -> float:
        command = self.end_effector.command(channel)
        control_value = command.definition.control_value(closed_fraction)
        self.data.ctrl[command.actuator_id] = control_value
        return control_value


class MujocoEndEffectorSwitcher:
    """Build MuJoCo scene variants and replace tools between simulation steps."""

    def __init__(
        self,
        scene_path: Path | str,
        registry: EndEffectorRegistry,
    ) -> None:
        resolved_scene_path = Path(scene_path).resolve()
        if not resolved_scene_path.is_file():
            raise FileNotFoundError(
                f"MuJoCo scene file does not exist: {resolved_scene_path}"
            )
        self.scene_path = resolved_scene_path
        self.registry = registry

    @staticmethod
    def is_available() -> bool:
        if find_spec("mujoco") is None:
            return False
        try:
            import mujoco
        except ImportError:
            return False
        return hasattr(mujoco, "MjSpec") and hasattr(mujoco.MjSpec, "attach")

    def load(
        self,
        end_effector_id: str = NO_END_EFFECTOR,
        *,
        keyframe_name: str | None = None,
    ) -> MujocoSimulation:
        mujoco = _import_mujoco_model_editing()
        definition = self.registry.resolve(end_effector_id)
        model, handle = self._compile(mujoco, definition)
        data = mujoco.MjData(model)
        if keyframe_name is not None:
            keyframe_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name
            )
            if keyframe_id < 0:
                raise EndEffectorAssemblyError(
                    f"MuJoCo keyframe does not exist after assembly: {keyframe_name}"
                )
            mujoco.mj_resetDataKeyframe(model, data, keyframe_id)

        simulation = MujocoSimulation(
            scene_path=self.scene_path,
            registry_source_path=self.registry.source_path,
            tool_prefix=self.registry.prefix,
            model=model,
            data=data,
            end_effector=handle,
        )
        for channel in handle.command_channels:
            simulation.set_end_effector_command(channel, 0.0)
        mujoco.mj_forward(model, data)
        return simulation

    def replace(
        self,
        simulation: MujocoSimulation,
        end_effector_id: str,
        *,
        preserve_state: bool = True,
        keyframe_name: str | None = None,
    ) -> MujocoSimulation:
        if simulation.scene_path.resolve() != self.scene_path:
            raise ValueError(
                "simulation was not built from this switcher's MuJoCo scene"
            )
        if (
            simulation.registry_source_path.resolve() != self.registry.source_path
            or simulation.tool_prefix != self.registry.prefix
        ):
            raise ValueError(
                "simulation was not built from this switcher's end-effector registry"
            )
        replacement = self.load(
            end_effector_id,
            keyframe_name=keyframe_name,
        )
        if preserve_state:
            mujoco = _import_mujoco_model_editing()
            _migrate_common_state(
                mujoco,
                source=simulation,
                target=replacement,
                tool_prefix=self.registry.prefix,
            )
        return replacement

    def _compile(
        self,
        mujoco: Any,
        definition: EndEffectorDefinition | None,
    ) -> tuple[Any, EndEffectorHandle]:
        try:
            scene_spec = mujoco.MjSpec.from_file(str(self.scene_path))
            if definition is not None:
                parent_body = scene_spec.body(definition.mount_body_name)
                if parent_body is None:
                    raise EndEffectorAssemblyError(
                        "configured mount body does not exist in the base scene: "
                        f"{definition.mount_body_name}"
                    )
                if scene_spec.site(self.registry.attached_tcp_site_name) is not None:
                    raise EndEffectorAssemblyError(
                        "base scene already contains the configured attached TCP site: "
                        f"{self.registry.attached_tcp_site_name}"
                    )

                mount_frame = parent_body.add_frame(
                    name=f"{self.registry.prefix}mount_frame",
                    pos=definition.mount_position_m,
                    quat=definition.mount_quat_wxyz,
                )
                tool_spec = mujoco.MjSpec.from_file(str(definition.model_path))
                scene_spec.copy_during_attach = True
                scene_spec.attach(
                    tool_spec,
                    prefix=self.registry.prefix,
                    frame=mount_frame,
                )
                parent_body.add_site(
                    name=self.registry.attached_tcp_site_name,
                    pos=definition.tcp_position_m,
                    quat=definition.tcp_quat_wxyz,
                    size=(0.001,),
                    group=4,
                )
            model = scene_spec.compile()
        except EndEffectorAssemblyError:
            raise
        except Exception as exc:
            tool_id = definition.end_effector_id if definition else NO_END_EFFECTOR
            raise EndEffectorAssemblyError(
                f"failed to assemble end effector {tool_id!r}: {exc}"
            ) from exc

        return model, self._resolve_handle(mujoco, model, definition)

    def _resolve_handle(
        self,
        mujoco: Any,
        model: Any,
        definition: EndEffectorDefinition | None,
    ) -> EndEffectorHandle:
        if definition is None:
            site_name = self.registry.detached_tcp_site_name
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id < 0:
                raise EndEffectorAssemblyError(
                    f"detached TCP site does not exist in base scene: {site_name}"
                )
            return EndEffectorHandle(
                end_effector_id=NO_END_EFFECTOR,
                tcp_site_name=site_name,
                tcp_site_id=site_id,
                commands=(),
            )

        site_name = self.registry.attached_tcp_site_name
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise EndEffectorAssemblyError(
                f"attached TCP site was not compiled: {site_name}"
            )

        commands: list[ResolvedEndEffectorCommand] = []
        for command in definition.commands:
            actuator_name = f"{self.registry.prefix}{command.actuator_name}"
            actuator_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                actuator_name,
            )
            if actuator_id < 0:
                raise EndEffectorAssemblyError(
                    f"configured actuator was not compiled: {actuator_name}"
                )
            if bool(model.actuator_ctrllimited[actuator_id]):
                lower, upper = (
                    float(value) for value in model.actuator_ctrlrange[actuator_id]
                )
                for label, value in (
                    ("open_ctrl", command.open_ctrl),
                    ("closed_ctrl", command.closed_ctrl),
                ):
                    if not lower <= value <= upper:
                        raise EndEffectorAssemblyError(
                            f"{definition.end_effector_id}.{command.channel}.{label} "
                            f"is outside actuator range [{lower}, {upper}]"
                        )
            commands.append(
                ResolvedEndEffectorCommand(
                    definition=command,
                    actuator_name=actuator_name,
                    actuator_id=actuator_id,
                )
            )
        return EndEffectorHandle(
            end_effector_id=definition.end_effector_id,
            tcp_site_name=site_name,
            tcp_site_id=site_id,
            commands=tuple(commands),
        )


def load_end_effector_registry(path: Path | str) -> EndEffectorRegistry:
    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(
            f"end-effector config file does not exist: {source_path}"
        )
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EndEffectorConfigError(
            f"end-effector config is not valid JSON: {source_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise EndEffectorConfigError("end-effector config root must be an object")
    if payload.get("schema") != END_EFFECTOR_CONFIG_SCHEMA:
        raise EndEffectorConfigError(
            f"unsupported end-effector config schema: {payload.get('schema')!r}"
        )

    prefix = _required_str(payload, "prefix", section="root")
    detached = _required_mapping(payload, "detached", section="root")
    attached = _required_mapping(payload, "attached", section="root")
    detached_tcp_site_name = _required_str(
        detached, "tcp_site_name", section="detached"
    )
    attached_tcp_site_name = _required_str(
        attached, "tcp_site_name", section="attached"
    )
    if detached_tcp_site_name == attached_tcp_site_name:
        raise EndEffectorConfigError(
            "detached and attached TCP site names must be different"
        )

    definitions_payload = _required_mapping(
        payload, "end_effectors", section="root"
    )
    definitions: list[EndEffectorDefinition] = []
    for end_effector_id, raw_definition in definitions_payload.items():
        if not isinstance(end_effector_id, str) or not end_effector_id:
            raise EndEffectorConfigError(
                "end_effectors keys must be non-empty strings"
            )
        if end_effector_id == NO_END_EFFECTOR:
            raise EndEffectorConfigError(
                f"{NO_END_EFFECTOR!r} is reserved for the detached state"
            )
        if not isinstance(raw_definition, dict):
            raise EndEffectorConfigError(
                f"end_effectors.{end_effector_id} must be an object"
            )
        section = f"end_effectors.{end_effector_id}"
        model_value = _required_str(raw_definition, "model_path", section=section)
        model_path = (source_path.parent / model_value).resolve()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"configured end-effector model does not exist: {model_path}"
            )
        mount = _required_mapping(raw_definition, "mount", section=section)
        tcp = _required_mapping(raw_definition, "tcp", section=section)
        commands_payload = _required_mapping(
            raw_definition, "commands", section=section
        )
        commands: list[EndEffectorCommandDefinition] = []
        actuator_names: set[str] = set()
        for channel, raw_command in commands_payload.items():
            if not isinstance(channel, str) or not channel:
                raise EndEffectorConfigError(
                    f"{section}.commands keys must be non-empty strings"
                )
            if not isinstance(raw_command, dict):
                raise EndEffectorConfigError(
                    f"{section}.commands.{channel} must be an object"
                )
            command_section = f"{section}.commands.{channel}"
            actuator_name = _required_str(
                raw_command, "actuator_name", section=command_section
            )
            if actuator_name.startswith(prefix):
                raise EndEffectorConfigError(
                    f"{command_section}.actuator_name must not include prefix {prefix!r}"
                )
            if actuator_name in actuator_names:
                raise EndEffectorConfigError(
                    f"{section} maps actuator {actuator_name!r} more than once"
                )
            actuator_names.add(actuator_name)
            open_ctrl = _required_number(
                raw_command, "open_ctrl", section=command_section
            )
            closed_ctrl = _required_number(
                raw_command, "closed_ctrl", section=command_section
            )
            if open_ctrl == closed_ctrl:
                raise EndEffectorConfigError(
                    f"{command_section} open_ctrl and closed_ctrl must differ"
                )
            commands.append(
                EndEffectorCommandDefinition(
                    channel=channel,
                    actuator_name=actuator_name,
                    open_ctrl=open_ctrl,
                    closed_ctrl=closed_ctrl,
                )
            )
        definitions.append(
            EndEffectorDefinition(
                end_effector_id=end_effector_id,
                model_path=model_path,
                mount_body_name=_required_str(
                    mount, "body_name", section=f"{section}.mount"
                ),
                mount_position_m=_required_vector(
                    mount,
                    "position_m",
                    length=3,
                    section=f"{section}.mount",
                ),
                mount_quat_wxyz=_required_quaternion(
                    mount, "quat_wxyz", section=f"{section}.mount"
                ),
                tcp_position_m=_required_vector(
                    tcp,
                    "position_m",
                    length=3,
                    section=f"{section}.tcp",
                ),
                tcp_quat_wxyz=_required_quaternion(
                    tcp, "quat_wxyz", section=f"{section}.tcp"
                ),
                commands=tuple(commands),
            )
        )
    if not definitions:
        raise EndEffectorConfigError(
            "end-effector config must define at least one attachable tool"
        )
    return EndEffectorRegistry(
        source_path=source_path,
        prefix=prefix,
        detached_tcp_site_name=detached_tcp_site_name,
        attached_tcp_site_name=attached_tcp_site_name,
        definitions=tuple(definitions),
    )


def _migrate_common_state(
    mujoco: Any,
    *,
    source: MujocoSimulation,
    target: MujocoSimulation,
    tool_prefix: str,
) -> None:
    target.data.time = source.data.time
    for old_joint_id in range(source.model.njnt):
        joint_name = mujoco.mj_id2name(
            source.model, mujoco.mjtObj.mjOBJ_JOINT, old_joint_id
        )
        if not joint_name or joint_name.startswith(tool_prefix):
            continue
        new_joint_id = mujoco.mj_name2id(
            target.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        if new_joint_id < 0:
            continue
        old_type = int(source.model.jnt_type[old_joint_id])
        new_type = int(target.model.jnt_type[new_joint_id])
        if old_type != new_type:
            raise EndEffectorAssemblyError(
                f"cannot migrate joint {joint_name!r}: type changed "
                f"from {old_type} to {new_type}"
            )
        qpos_width, qvel_width = _joint_state_widths(mujoco, old_type)
        old_qpos = int(source.model.jnt_qposadr[old_joint_id])
        new_qpos = int(target.model.jnt_qposadr[new_joint_id])
        old_qvel = int(source.model.jnt_dofadr[old_joint_id])
        new_qvel = int(target.model.jnt_dofadr[new_joint_id])
        target.data.qpos[new_qpos : new_qpos + qpos_width] = source.data.qpos[
            old_qpos : old_qpos + qpos_width
        ]
        target.model.qpos0[new_qpos : new_qpos + qpos_width] = source.model.qpos0[
            old_qpos : old_qpos + qpos_width
        ]
        target.data.qvel[new_qvel : new_qvel + qvel_width] = source.data.qvel[
            old_qvel : old_qvel + qvel_width
        ]

    for old_actuator_id in range(source.model.nu):
        actuator_name = mujoco.mj_id2name(
            source.model, mujoco.mjtObj.mjOBJ_ACTUATOR, old_actuator_id
        )
        if not actuator_name or actuator_name.startswith(tool_prefix):
            continue
        new_actuator_id = mujoco.mj_name2id(
            target.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_name,
        )
        if new_actuator_id >= 0:
            target.data.ctrl[new_actuator_id] = source.data.ctrl[old_actuator_id]

    # Task 1 selection changes these filters at runtime. Preserve them by geom
    # name so swapping a tool cannot silently re-enable unselected objects.
    for old_geom_id in range(source.model.ngeom):
        geom_name = mujoco.mj_id2name(
            source.model, mujoco.mjtObj.mjOBJ_GEOM, old_geom_id
        )
        if not geom_name or geom_name.startswith(tool_prefix):
            continue
        new_geom_id = mujoco.mj_name2id(
            target.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
        )
        if new_geom_id >= 0:
            target.model.geom_contype[new_geom_id] = source.model.geom_contype[
                old_geom_id
            ]
            target.model.geom_conaffinity[new_geom_id] = (
                source.model.geom_conaffinity[old_geom_id]
            )

    for old_body_id in range(source.model.nbody):
        old_mocap_id = int(source.model.body_mocapid[old_body_id])
        if old_mocap_id < 0:
            continue
        body_name = mujoco.mj_id2name(
            source.model, mujoco.mjtObj.mjOBJ_BODY, old_body_id
        )
        if not body_name or body_name.startswith(tool_prefix):
            continue
        new_body_id = mujoco.mj_name2id(
            target.model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        if new_body_id < 0:
            continue
        new_mocap_id = int(target.model.body_mocapid[new_body_id])
        if new_mocap_id >= 0:
            target.data.mocap_pos[new_mocap_id] = source.data.mocap_pos[
                old_mocap_id
            ]
            target.data.mocap_quat[new_mocap_id] = source.data.mocap_quat[
                old_mocap_id
            ]
    mujoco.mj_forward(target.model, target.data)


def _joint_state_widths(mujoco: Any, joint_type: int) -> tuple[int, int]:
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7, 6
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4, 3
    if joint_type in (
        int(mujoco.mjtJoint.mjJNT_SLIDE),
        int(mujoco.mjtJoint.mjJNT_HINGE),
    ):
        return 1, 1
    raise EndEffectorAssemblyError(f"unsupported MuJoCo joint type: {joint_type}")


def _import_mujoco_model_editing() -> Any:
    if find_spec("mujoco") is None:
        raise EndEffectorAssemblyError(
            "MuJoCo is not installed; install the project mujoco extra"
        )
    import mujoco

    if not hasattr(mujoco, "MjSpec") or not hasattr(mujoco.MjSpec, "attach"):
        raise EndEffectorAssemblyError(
            "end-effector replacement requires MuJoCo 3.2 or newer"
        )
    return mujoco


def _required_mapping(
    payload: dict[str, Any],
    key: str,
    *,
    section: str,
) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise EndEffectorConfigError(f"{section}.{key} must be an object")
    return value


def _required_str(
    payload: dict[str, Any],
    key: str,
    *,
    section: str,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value or value.strip() != value:
        raise EndEffectorConfigError(
            f"{section}.{key} must be a non-empty string without outer whitespace"
        )
    return value


def _required_number(
    payload: dict[str, Any],
    key: str,
    *,
    section: str,
) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EndEffectorConfigError(f"{section}.{key} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise EndEffectorConfigError(f"{section}.{key} must be finite")
    return result


def _required_vector(
    payload: dict[str, Any],
    key: str,
    *,
    length: int,
    section: str,
) -> tuple[float, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or len(value) != length:
        raise EndEffectorConfigError(
            f"{section}.{key} must be an array of {length} numbers"
        )
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise EndEffectorConfigError(
                f"{section}.{key}[{index}] must be a number"
            )
        number = float(item)
        if not math.isfinite(number):
            raise EndEffectorConfigError(
                f"{section}.{key}[{index}] must be finite"
            )
        result.append(number)
    return tuple(result)


def _required_quaternion(
    payload: dict[str, Any],
    key: str,
    *,
    section: str,
) -> tuple[float, float, float, float]:
    value = _required_vector(
        payload,
        key,
        length=4,
        section=section,
    )
    norm = math.sqrt(sum(component * component for component in value))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise EndEffectorConfigError(
            f"{section}.{key} must be a normalized wxyz quaternion"
        )
    return value[0], value[1], value[2], value[3]
