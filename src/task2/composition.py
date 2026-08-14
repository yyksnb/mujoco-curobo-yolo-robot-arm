from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from .contracts import (
    ContinuousYawSymmetry,
    DiscreteYawSymmetry,
    EndEffectorTargetingConfig,
    GraspTemplateConfig,
    ParallelGripperCapabilities,
    RigidTransform,
    TargetingConfig,
    TargetingDiagnosticsConfig,
    TargetingObjectConfig,
    Vector3,
    AttachedGeometry,
    Box,
    Capsule,
    CollisionSphereGroup,
    JointLimits,
    PlanningConfig,
    PlanningEndEffectorConfig,
    PlanningObjectConfig,
    RobotModel,
    Sphere,
    ActuatorForceLimitBinding,
    ActuatorPositionBinding,
    ExecutionEndEffectorConfig,
    ExecutionObjectConfig,
    ExecutionRobotConfig,
    RuntimeJointBinding,
)


@dataclass(frozen=True)
class PlanningComposition:
    robot_model: RobotModel
    planning_config: PlanningConfig
    objects: tuple[PlanningObjectConfig, ...]
    end_effectors: tuple[PlanningEndEffectorConfig, ...]

    def get_object(self, object_type: str) -> PlanningObjectConfig:
        for item in self.objects:
            if item.object_type == object_type:
                return item
        raise KeyError(f"unknown Planning object_type: {object_type}")

    def get_end_effector(self, profile_id: str) -> PlanningEndEffectorConfig:
        for item in self.end_effectors:
            if item.profile_id == profile_id:
                return item
        raise KeyError(f"unknown Planning end-effector profile_id: {profile_id}")


@dataclass(frozen=True)
class ExecutionComposition:
    robot_config: ExecutionRobotConfig
    objects: tuple[ExecutionObjectConfig, ...]
    end_effectors: tuple[ExecutionEndEffectorConfig, ...]

    def get_object(self, object_type: str) -> ExecutionObjectConfig:
        for item in self.objects:
            if item.object_type == object_type:
                return item
        raise KeyError(f"unknown Execution object_type: {object_type}")

    def get_end_effector(self, profile_id: str) -> ExecutionEndEffectorConfig:
        for item in self.end_effectors:
            if item.profile_id == profile_id:
                return item
        raise KeyError(f"unknown Execution end-effector profile_id: {profile_id}")


def load_planning_config(
    robot_path: str | Path,
    planning_path: str | Path,
    objects_path: str | Path,
    end_effectors_path: str | Path,
) -> PlanningComposition:
    paths = tuple(Path(path) for path in (robot_path, planning_path, objects_path, end_effectors_path))
    robot_payload, planning_payload, objects_payload, end_effectors_payload = (
        _load_yaml_mapping(path) for path in paths
    )
    robot = _parse_robot(_required_mapping(robot_payload, "robot", "robot"))
    config_hash = _planning_config_hash(
        robot_payload,
        planning_payload,
        objects_payload,
        end_effectors_payload,
    )
    planning = _parse_planning(
        _required_mapping(planning_payload, "planning", "planning"), config_hash
    )
    objects = _parse_planning_objects(
        _required_mapping(objects_payload, "objects", "objects")
    )
    end_effectors = _parse_planning_end_effectors(
        _required_mapping(end_effectors_payload, "end_effectors", "end_effectors")
    )
    robot.validate()
    planning.validate()
    for item in (*objects, *end_effectors):
        item.validate()
    return PlanningComposition(robot, planning, objects, end_effectors)


def load_execution_config(
    execution_path: str | Path,
    objects_path: str | Path,
    end_effectors_path: str | Path,
) -> ExecutionComposition:
    execution_payload = _load_yaml_mapping(Path(execution_path))
    objects_payload = _load_yaml_mapping(Path(objects_path))
    end_effectors_payload = _load_yaml_mapping(Path(end_effectors_path))
    robot = _parse_execution_robot(
        _required_mapping(execution_payload, "execution", "execution")
    )
    objects = _parse_execution_objects(
        _required_mapping(objects_payload, "objects", "objects")
    )
    end_effectors = _parse_execution_end_effectors(
        _required_mapping(end_effectors_payload, "end_effectors", "end_effectors")
    )
    robot.validate()
    for item in (*objects, *end_effectors):
        item.validate()
    return ExecutionComposition(robot, objects, end_effectors)


def load_targeting_config(
    objects_path: str | Path,
    end_effectors_path: str | Path,
) -> TargetingConfig:
    objects_payload = _load_yaml_mapping(Path(objects_path))
    end_effectors_payload = _load_yaml_mapping(Path(end_effectors_path))
    objects = _parse_objects(_required_mapping(objects_payload, "objects", "objects"))
    end_effectors = _parse_end_effectors(
        _required_mapping(end_effectors_payload, "end_effectors", "end_effectors")
    )
    result = TargetingConfig(objects=objects, end_effectors=end_effectors)
    result.validate()
    return result


def get_targeting_object_config(config: TargetingConfig, object_type: str) -> TargetingObjectConfig:
    for item in config.objects:
        if item.object_type == object_type:
            return item
    raise KeyError(f"unknown Targeting object_type: {object_type}")


def get_end_effector_targeting_config(
    config: TargetingConfig,
    profile_id: str,
) -> EndEffectorTargetingConfig:
    for item in config.end_effectors:
        if item.profile_id == profile_id:
            return item
    raise KeyError(f"unknown Targeting end-effector profile_id: {profile_id}")


def _parse_robot(payload: dict[str, Any]) -> RobotModel:
    path = "robot"
    _require_exact_keys(
        payload,
        {
            "model_id",
            "base_frame",
            "tcp_frame",
            "kinematics_asset",
            "curobo_config_asset",
            "vamp_robot_module",
            "vamp_robot_sphere_count",
            "vamp_tool_fraction_midpoint",
            "maximum_gripper_opening_m",
            "joints",
            "collision_sphere_groups",
        },
        path,
    )
    joints = _required_list(payload, "joints", path)
    names: list[str] = []
    limits: list[JointLimits] = []
    defaults: list[float] = []
    for index, raw in enumerate(joints):
        joint_path = f"{path}.joints[{index}]"
        joint = _as_mapping(raw, joint_path)
        _require_exact_keys(
            joint,
            {
                "name",
                "position_rad",
                "velocity_rad_s",
                "acceleration_rad_s2",
                "jerk_rad_s3",
                "default_rad",
            },
            joint_path,
        )
        position = _required_pair(joint, "position_rad", joint_path)
        names.append(_required_str(joint, "name", joint_path))
        limits.append(
            JointLimits(
                minimum_position_rad=position[0],
                maximum_position_rad=position[1],
                maximum_velocity_rad_s=_required_float(joint, "velocity_rad_s", joint_path),
                maximum_acceleration_rad_s2=_required_float(joint, "acceleration_rad_s2", joint_path),
                maximum_jerk_rad_s3=_required_float(joint, "jerk_rad_s3", joint_path),
            )
        )
        defaults.append(_required_float(joint, "default_rad", joint_path))
    groups: list[CollisionSphereGroup] = []
    for index, raw in enumerate(_required_list(payload, "collision_sphere_groups", path)):
        group_path = f"{path}.collision_sphere_groups[{index}]"
        group = _as_mapping(raw, group_path)
        _require_exact_keys(
            group,
            {
                "group",
                "first_sphere",
                "sphere_count",
                "translation_per_closed_fraction_m",
                "sweep_radius_inflation_m",
            },
            group_path,
        )
        groups.append(
            CollisionSphereGroup(
                group=_required_str(group, "group", group_path),
                first_sphere=_required_int(group, "first_sphere", group_path),
                sphere_count=_required_int(group, "sphere_count", group_path),
                translation_per_closed_fraction_m=_parse_vector3(
                    group, "translation_per_closed_fraction_m", group_path
                ),
                sweep_radius_inflation_m=_required_float(
                    group, "sweep_radius_inflation_m", group_path
                ),
            )
        )
    return RobotModel(
        model_id=_required_str(payload, "model_id", path),
        base_frame=_required_str(payload, "base_frame", path),
        tcp_frame=_required_str(payload, "tcp_frame", path),
        joint_names=tuple(names),
        joint_limits=tuple(limits),
        default_joint_positions=tuple(defaults),
        kinematics_asset=_required_str(payload, "kinematics_asset", path),
        curobo_config_asset=_required_str(payload, "curobo_config_asset", path),
        vamp_robot_module=_required_str(payload, "vamp_robot_module", path),
        vamp_robot_sphere_count=_required_int(payload, "vamp_robot_sphere_count", path),
        vamp_tool_fraction_midpoint=_required_float(
            payload, "vamp_tool_fraction_midpoint", path
        ),
        maximum_gripper_opening_m=_required_float(
            payload, "maximum_gripper_opening_m", path
        ),
        collision_sphere_groups=tuple(groups),
    )


def _parse_planning(payload: dict[str, Any], config_hash: str) -> PlanningConfig:
    expected = {
        "vamp_version",
        "ruckig_version",
        "ik_backend",
        "ik_position_tolerance_m",
        "ik_orientation_tolerance_rad",
        "ik_solution_count",
        "ik_seed_count",
        "cartesian_translation_step_m",
        "cartesian_rotation_step_rad",
        "lift_distance_m",
        "preplace_offset_m",
        "placement_sample_count",
        "joint_limit_margin_rad",
        "vamp_range_rad",
        "vamp_max_iterations",
        "vamp_max_samples",
    }
    _require_exact_keys(payload, expected, "planning")
    return PlanningConfig(
        config_hash=config_hash,
        vamp_version=_required_str(payload, "vamp_version", "planning"),
        ruckig_version=_required_str(payload, "ruckig_version", "planning"),
        ik_backend=_required_str(payload, "ik_backend", "planning"),
        ik_position_tolerance_m=_required_float(payload, "ik_position_tolerance_m", "planning"),
        ik_orientation_tolerance_rad=_required_float(payload, "ik_orientation_tolerance_rad", "planning"),
        ik_solution_count=_required_int(payload, "ik_solution_count", "planning"),
        ik_seed_count=_required_int(payload, "ik_seed_count", "planning"),
        cartesian_translation_step_m=_required_float(payload, "cartesian_translation_step_m", "planning"),
        cartesian_rotation_step_rad=_required_float(payload, "cartesian_rotation_step_rad", "planning"),
        lift_distance_m=_required_float(payload, "lift_distance_m", "planning"),
        preplace_offset_m=_required_float(payload, "preplace_offset_m", "planning"),
        placement_sample_count=_required_int(payload, "placement_sample_count", "planning"),
        joint_limit_margin_rad=_required_float(payload, "joint_limit_margin_rad", "planning"),
        vamp_range_rad=_required_float(payload, "vamp_range_rad", "planning"),
        vamp_max_iterations=_required_int(payload, "vamp_max_iterations", "planning"),
        vamp_max_samples=_required_int(payload, "vamp_max_samples", "planning"),
    )


def _parse_planning_objects(payload: dict[str, Any]) -> tuple[PlanningObjectConfig, ...]:
    result = []
    for object_type, raw in payload.items():
        object_path = f"objects.{object_type}"
        item = _as_mapping(raw, object_path)
        planning = _required_mapping(item, "planning", object_path)
        _require_exact_keys(planning, {"attached_geometry"}, f"{object_path}.planning")
        geometries = []
        for index, geometry_raw in enumerate(
            _required_list(planning, "attached_geometry", f"{object_path}.planning")
        ):
            geometry_path = f"{object_path}.planning.attached_geometry[{index}]"
            geometry_item = _as_mapping(geometry_raw, geometry_path)
            shape_keys = set(geometry_item) & {"sphere", "capsule", "box"}
            if len(shape_keys) != 1:
                raise ValueError(f"{geometry_path} must contain exactly one supported shape")
            _require_exact_keys(
                geometry_item,
                {"geometry_id", "T_object_geometry", next(iter(shape_keys))},
                geometry_path,
            )
            shape_key = next(iter(shape_keys))
            shape_payload = _required_mapping(geometry_item, shape_key, geometry_path)
            if shape_key == "sphere":
                _require_exact_keys(shape_payload, {"radius_m"}, f"{geometry_path}.sphere")
                shape = Sphere(_required_float(shape_payload, "radius_m", f"{geometry_path}.sphere"))
            elif shape_key == "capsule":
                _require_exact_keys(shape_payload, {"radius_m", "length_m"}, f"{geometry_path}.capsule")
                shape = Capsule(
                    _required_float(shape_payload, "radius_m", f"{geometry_path}.capsule"),
                    _required_float(shape_payload, "length_m", f"{geometry_path}.capsule"),
                )
            else:
                _require_exact_keys(shape_payload, {"size_m"}, f"{geometry_path}.box")
                shape = Box(_parse_vector3(shape_payload, "size_m", f"{geometry_path}.box"))
            geometries.append(
                AttachedGeometry(
                    geometry_id=_required_str(geometry_item, "geometry_id", geometry_path),
                    T_object_geometry=_parse_compact_transform(
                        _required_mapping(geometry_item, "T_object_geometry", geometry_path),
                        f"{geometry_path}.T_object_geometry",
                    ),
                    geometry=shape,
                )
            )
        result.append(PlanningObjectConfig(object_type, tuple(geometries)))
    return tuple(result)


def _parse_planning_end_effectors(
    payload: dict[str, Any],
) -> tuple[PlanningEndEffectorConfig, ...]:
    result = []
    for profile_id, raw in payload.items():
        path = f"end_effectors.{profile_id}.planning"
        item = _as_mapping(raw, f"end_effectors.{profile_id}")
        planning = _required_mapping(item, "planning", f"end_effectors.{profile_id}")
        _require_exact_keys(
            planning,
            {
                "actuator_position_channel",
                "actuator_force_limit_channel",
                "open_position_m",
                "contact_groups",
            },
            path,
        )
        contacts = _required_list(planning, "contact_groups", path)
        result.append(
            PlanningEndEffectorConfig(
                profile_id=profile_id,
                actuator_position_channel=_required_str(planning, "actuator_position_channel", path),
                actuator_force_limit_channel=_required_str(planning, "actuator_force_limit_channel", path),
                open_position_m=_required_float(planning, "open_position_m", path),
                contact_groups=tuple(_as_str(value, f"{path}.contact_groups") for value in contacts),
            )
        )
    return tuple(result)


def _parse_execution_robot(payload: dict[str, Any]) -> ExecutionRobotConfig:
    path = "execution"
    _require_exact_keys(
        payload,
        {
            "adapter_id",
            "model_asset",
            "tcp_frame",
            "tcp_site_name",
            "joint_bindings",
            "scene_object_body_names",
        },
        path,
    )
    bindings = []
    for index, raw in enumerate(_required_list(payload, "joint_bindings", path)):
        item_path = f"{path}.joint_bindings[{index}]"
        item = _as_mapping(raw, item_path)
        _require_exact_keys(
            item,
            {"joint_name", "runtime_joint_name", "runtime_actuator_name"},
            item_path,
        )
        bindings.append(
            RuntimeJointBinding(
                _required_str(item, "joint_name", item_path),
                _required_str(item, "runtime_joint_name", item_path),
                _required_str(item, "runtime_actuator_name", item_path),
            )
        )
    body_names = _required_list(payload, "scene_object_body_names", path)
    return ExecutionRobotConfig(
        adapter_id=_required_str(payload, "adapter_id", path),
        model_asset=_required_str(payload, "model_asset", path),
        tcp_frame=_required_str(payload, "tcp_frame", path),
        tcp_site_name=_required_str(payload, "tcp_site_name", path),
        joint_bindings=tuple(bindings),
        scene_object_body_names=tuple(
            _as_str(value, f"{path}.scene_object_body_names[{index}]")
            for index, value in enumerate(body_names)
        ),
    )


def _parse_execution_objects(payload: dict[str, Any]) -> tuple[ExecutionObjectConfig, ...]:
    result = []
    for object_type, raw in payload.items():
        object_path = f"objects.{object_type}"
        item = _as_mapping(raw, object_path)
        execution = _required_mapping(item, "execution", object_path)
        path = f"{object_path}.execution"
        _require_exact_keys(
            execution,
            {
                "runtime_body_name",
                "runtime_free_joint_name",
                "runtime_collision_geom_names",
            },
            path,
        )
        geom_names = _required_list(execution, "runtime_collision_geom_names", path)
        result.append(
            ExecutionObjectConfig(
                object_type=object_type,
                runtime_body_name=_required_str(execution, "runtime_body_name", path),
                runtime_free_joint_name=_required_str(
                    execution, "runtime_free_joint_name", path
                ),
                runtime_collision_geom_names=tuple(
                    _as_str(value, f"{path}.runtime_collision_geom_names[{index}]")
                    for index, value in enumerate(geom_names)
                ),
            )
        )
    return tuple(result)


def _parse_execution_end_effectors(
    payload: dict[str, Any],
) -> tuple[ExecutionEndEffectorConfig, ...]:
    result = []
    for profile_id, raw in payload.items():
        item_path = f"end_effectors.{profile_id}"
        item = _as_mapping(raw, item_path)
        execution = _required_mapping(item, "execution", item_path)
        path = f"{item_path}.execution"
        _require_exact_keys(
            execution,
            {"position", "force_limit", "contact_group_geoms"},
            path,
        )
        position = _required_mapping(execution, "position", path)
        position_path = f"{path}.position"
        _require_exact_keys(
            position,
            {
                "channel_name",
                "runtime_actuator_name",
                "command_scale",
                "command_offset",
                "state_joint_names",
            },
            position_path,
        )
        state_joint_names = _required_list(position, "state_joint_names", position_path)
        force = _required_mapping(execution, "force_limit", path)
        force_path = f"{path}.force_limit"
        _require_exact_keys(
            force,
            {"channel_name", "runtime_actuator_name", "command_scale"},
            force_path,
        )
        contact_payload = _required_mapping(execution, "contact_group_geoms", path)
        contacts: dict[str, tuple[str, ...]] = {}
        for group, raw_geoms in contact_payload.items():
            if not isinstance(raw_geoms, list):
                raise ValueError(f"{path}.contact_group_geoms.{group} must be a list")
            contacts[group] = tuple(
                _as_str(value, f"{path}.contact_group_geoms.{group}[{index}]")
                for index, value in enumerate(raw_geoms)
            )
        result.append(
            ExecutionEndEffectorConfig(
                profile_id=profile_id,
                position=ActuatorPositionBinding(
                    channel_name=_required_str(position, "channel_name", position_path),
                    runtime_actuator_name=_required_str(
                        position, "runtime_actuator_name", position_path
                    ),
                    command_scale=_required_float(position, "command_scale", position_path),
                    command_offset=_required_float(position, "command_offset", position_path),
                    state_joint_names=tuple(
                        _as_str(value, f"{position_path}.state_joint_names[{index}]")
                        for index, value in enumerate(state_joint_names)
                    ),
                ),
                force_limit=ActuatorForceLimitBinding(
                    channel_name=_required_str(force, "channel_name", force_path),
                    runtime_actuator_name=_required_str(
                        force, "runtime_actuator_name", force_path
                    ),
                    command_scale=_required_float(force, "command_scale", force_path),
                ),
                contact_group_geoms=contacts,
            )
        )
    return tuple(result)


def _planning_config_hash(
    robot_payload: dict[str, Any],
    planning_payload: dict[str, Any],
    objects_payload: dict[str, Any],
    end_effectors_payload: dict[str, Any],
) -> str:
    objects = _required_mapping(objects_payload, "objects", "objects")
    end_effectors = _required_mapping(
        end_effectors_payload, "end_effectors", "end_effectors"
    )
    payload = {
        "robot": _required_mapping(robot_payload, "robot", "robot"),
        "planning": _required_mapping(planning_payload, "planning", "planning"),
        "objects": {
            name: {
                "planning": _required_mapping(
                    _as_mapping(value, f"objects.{name}"),
                    "planning",
                    f"objects.{name}",
                )
            }
            for name, value in objects.items()
        },
        "end_effectors": {
            name: {
                "planning": _required_mapping(
                    _as_mapping(value, f"end_effectors.{name}"),
                    "planning",
                    f"end_effectors.{name}",
                )
            }
            for name, value in end_effectors.items()
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _parse_compact_transform(payload: dict[str, Any], path: str) -> RigidTransform:
    from scipy.spatial.transform import Rotation

    _require_exact_keys(
        payload,
        {"parent_frame", "child_frame", "translation_m", "quaternion_wxyz"},
        path,
    )
    translation = _parse_vector3(payload, "translation_m", path)
    raw_quaternion = _required_list(payload, "quaternion_wxyz", path)
    if len(raw_quaternion) != 4:
        raise ValueError(f"{path}.quaternion_wxyz must contain four values")
    w, x, y, z = tuple(
        _as_float(value, f"{path}.quaternion_wxyz[{index}]")
        for index, value in enumerate(raw_quaternion)
    )
    rotation = Rotation.from_quat((x, y, z, w)).as_matrix()
    matrix = tuple(
        tuple(float(rotation[row, column]) for column in range(3)) + (translation[row],)
        for row in range(3)
    ) + ((0.0, 0.0, 0.0, 1.0),)
    result = RigidTransform(
        _required_str(payload, "parent_frame", path),
        _required_str(payload, "child_frame", path),
        matrix,  # type: ignore[arg-type]
    )
    result.validate(label=path)
    return result


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Targeting config does not exist: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid Targeting YAML in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Targeting config root must be a mapping: {path}")
    return value


def _parse_objects(payload: dict[str, Any]) -> tuple[TargetingObjectConfig, ...]:
    result: list[TargetingObjectConfig] = []
    for object_type, value in payload.items():
        _require_identifier(object_type, "object type")
        object_mapping = _as_mapping(value, f"objects.{object_type}")
        targeting = _required_mapping(object_mapping, "targeting", f"objects.{object_type}")
        _require_exact_keys(targeting, {"templates"}, f"objects.{object_type}.targeting")
        templates = _required_list(targeting, "templates", f"objects.{object_type}.targeting")
        result.append(
            TargetingObjectConfig(
                object_type=object_type,
                templates=tuple(
                    _parse_template(item, object_type, index)
                    for index, item in enumerate(templates)
                ),
            )
        )
    return tuple(result)


def _parse_template(value: object, object_type: str, index: int) -> GraspTemplateConfig:
    path = f"objects.{object_type}.targeting.templates[{index}]"
    item = _as_mapping(value, path)
    _require_exact_keys(
        item,
        {
            "template_id",
            "end_effector_family",
            "calibration",
            "T_object_tcp",
            "T_tcp_pregrasp",
            "yaw_symmetry",
            "pregrasp_opening_m",
            "target_opening_m",
            "grip_force_limit_n",
            "minimum_retention_force_n",
            "closing_speed_m_s",
            "dwell_s",
            "expected_contact_groups",
            "template_priority",
            "diagnostics",
        },
        path,
    )
    template_id = _required_str(item, "template_id", path)
    yaw = _required_mapping(item, "yaw_symmetry", path)
    expected_contacts = _required_list(item, "expected_contact_groups", path)
    diagnostics = _required_mapping(item, "diagnostics", path)
    _require_exact_keys(
        diagnostics,
        {"contact_margin_m", "opening_margin_m", "force_margin_n"},
        f"{path}.diagnostics",
    )
    return GraspTemplateConfig(
        template_id=template_id,
        object_type=object_type,
        end_effector_family=_required_str(item, "end_effector_family", path),
        calibration=_required_str(item, "calibration", path),
        T_object_tcp=_parse_transform(
            _required_mapping(item, "T_object_tcp", path),
            f"{path}.T_object_tcp",
        ),
        T_tcp_pregrasp=_parse_transform(
            _required_mapping(item, "T_tcp_pregrasp", path),
            f"{path}.T_tcp_pregrasp",
        ),
        yaw_symmetry=_parse_yaw(yaw, f"{path}.yaw_symmetry"),
        pregrasp_opening_m=_required_float(item, "pregrasp_opening_m", path),
        target_opening_m=_required_float(item, "target_opening_m", path),
        grip_force_limit_n=_required_float(item, "grip_force_limit_n", path),
        minimum_retention_force_n=_required_float(item, "minimum_retention_force_n", path),
        closing_speed_m_s=_required_float(item, "closing_speed_m_s", path),
        dwell_s=_required_float(item, "dwell_s", path),
        expected_contact_groups=tuple(
            _as_str(value, f"{path}.expected_contact_groups[{contact_index}]")
            for contact_index, value in enumerate(expected_contacts)
        ),
        template_priority=_required_float(item, "template_priority", path),
        diagnostics=TargetingDiagnosticsConfig(
            contact_margin_m=_required_float(diagnostics, "contact_margin_m", f"{path}.diagnostics"),
            opening_margin_m=_required_float(diagnostics, "opening_margin_m", f"{path}.diagnostics"),
            force_margin_n=_required_float(diagnostics, "force_margin_n", f"{path}.diagnostics"),
        ),
    )


def _parse_end_effectors(payload: dict[str, Any]) -> tuple[EndEffectorTargetingConfig, ...]:
    result: list[EndEffectorTargetingConfig] = []
    for profile_id, value in payload.items():
        path = f"end_effectors.{profile_id}"
        item = _as_mapping(value, path)
        targeting = _required_mapping(item, "targeting", path)
        _require_exact_keys(targeting, {"family", "capabilities"}, f"{path}.targeting")
        capabilities = _required_mapping(targeting, "capabilities", f"{path}.targeting")
        _require_exact_keys(
            capabilities,
            {
                "opening_command_range_m",
                "grip_force_limit_range_n",
                "maximum_closing_speed_m_s",
                "contact_groups",
            },
            f"{path}.targeting.capabilities",
        )
        opening = _required_pair(
            capabilities,
            "opening_command_range_m",
            f"{path}.targeting.capabilities",
        )
        force = _required_pair(
            capabilities,
            "grip_force_limit_range_n",
            f"{path}.targeting.capabilities",
        )
        contacts = _required_list(
            capabilities,
            "contact_groups",
            f"{path}.targeting.capabilities",
        )
        result.append(
            EndEffectorTargetingConfig(
                profile_id=profile_id,
                family=_required_str(targeting, "family", f"{path}.targeting"),
                capabilities=ParallelGripperCapabilities(
                    opening_command_range_m=opening,
                    grip_force_limit_range_n=force,
                    maximum_closing_speed_m_s=_required_float(
                        capabilities,
                        "maximum_closing_speed_m_s",
                        f"{path}.targeting.capabilities",
                    ),
                    contact_groups=tuple(
                        _as_str(contact, f"{path}.targeting.capabilities.contact_groups[{index}]")
                        for index, contact in enumerate(contacts)
                    ),
                ),
            )
        )
    return tuple(result)


def _parse_transform(payload: dict[str, Any], path: str) -> RigidTransform:
    _require_exact_keys(payload, {"parent_frame", "child_frame", "matrix"}, path)
    rows = _required_list(payload, "matrix", path)
    if len(rows) != 4:
        raise ValueError(f"{path}.matrix must have four rows")
    matrix = tuple(
        tuple(
            _as_float(value, f"{path}.matrix[{row_index}][{column_index}]")
            for column_index, value in enumerate(_as_row(row, f"{path}.matrix[{row_index}]"))
        )
        for row_index, row in enumerate(rows)
    )
    transform = RigidTransform(
        parent_frame=_required_str(payload, "parent_frame", path),
        child_frame=_required_str(payload, "child_frame", path),
        matrix=matrix,  # type: ignore[arg-type]
    )
    transform.validate(label=path)
    return transform


def _parse_yaw(payload: dict[str, Any], path: str) -> DiscreteYawSymmetry | ContinuousYawSymmetry:
    kind = _required_str(payload, "kind", path)
    axis = _required_str(payload, "axis", path)
    axis_vector = _parse_vector3(payload, "axis_vector", path)
    if kind == "discrete":
        _require_exact_keys(payload, {"kind", "group", "axis", "axis_vector", "values_rad"}, path)
        values = _required_list(payload, "values_rad", path)
        return DiscreteYawSymmetry(
            group=_required_str(payload, "group", path),
            axis=axis,
            axis_vector=axis_vector,
            values_rad=tuple(
                _as_float(value, f"{path}.values_rad[{index}]")
                for index, value in enumerate(values)
            ),
        )
    if kind == "continuous":
        _require_exact_keys(payload, {"kind", "axis", "axis_vector", "range_rad"}, path)
        range_payload = _required_mapping(payload, "range_rad", path)
        range_path = f"{path}.range_rad"
        _require_exact_keys(
            range_payload,
            {"lower", "upper", "lower_inclusive", "upper_inclusive"},
            range_path,
        )
        return ContinuousYawSymmetry(
            axis=axis,
            axis_vector=axis_vector,
            lower_rad=_required_float(range_payload, "lower", range_path),
            upper_rad=_required_float(range_payload, "upper", range_path),
            lower_inclusive=_required_bool(range_payload, "lower_inclusive", range_path),
            upper_inclusive=_required_bool(range_payload, "upper_inclusive", range_path),
        )
    raise ValueError(f"{path}.kind must be 'discrete' or 'continuous'")


def _parse_vector3(payload: dict[str, Any], key: str, path: str) -> Vector3:
    values = _required_list(payload, key, path)
    if len(values) != 3:
        raise ValueError(f"{path}.{key} must contain three values")
    return tuple(
        _as_float(value, f"{path}.{key}[{index}]") for index, value in enumerate(values)
    )  # type: ignore[return-value]


def _required_mapping(payload: dict[str, Any], key: str, path: str) -> dict[str, Any]:
    if key not in payload:
        raise ValueError(f"{path}.{key} is required")
    return _as_mapping(payload[key], f"{path}.{key}")


def _required_list(payload: dict[str, Any], key: str, path: str) -> list[object]:
    if key not in payload or not isinstance(payload[key], list):
        raise ValueError(f"{path}.{key} must be a list")
    return payload[key]


def _required_pair(payload: dict[str, Any], key: str, path: str) -> tuple[float, float]:
    values = _required_list(payload, key, path)
    if len(values) != 2:
        raise ValueError(f"{path}.{key} must contain two values")
    return (
        _as_float(values[0], f"{path}.{key}[0]"),
        _as_float(values[1], f"{path}.{key}[1]"),
    )


def _required_str(payload: dict[str, Any], key: str, path: str) -> str:
    if key not in payload:
        raise ValueError(f"{path}.{key} is required")
    return _as_str(payload[key], f"{path}.{key}")


def _required_float(payload: dict[str, Any], key: str, path: str) -> float:
    if key not in payload:
        raise ValueError(f"{path}.{key} is required")
    return _as_float(payload[key], f"{path}.{key}")


def _required_bool(payload: dict[str, Any], key: str, path: str) -> bool:
    if key not in payload or not isinstance(payload[key], bool):
        raise ValueError(f"{path}.{key} must be a boolean")
    return payload[key]


def _required_int(payload: dict[str, Any], key: str, path: str) -> int:
    if key not in payload or isinstance(payload[key], bool) or not isinstance(payload[key], int):
        raise ValueError(f"{path}.{key} must be an integer")
    return payload[key]


def _as_mapping(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path} must be a string-keyed mapping")
    return value


def _as_row(value: object, path: str) -> list[object]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{path} must contain four values")
    return value


def _as_str(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _as_float(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _require_identifier(value: object, path: str) -> None:
    _as_str(value, path)


def _require_exact_keys(payload: dict[str, Any], expected: set[str], path: str) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise ValueError(f"{path} is missing required fields: {missing!r}")
    if unknown:
        raise ValueError(f"{path} contains unknown fields: {unknown!r}")
