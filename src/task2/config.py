from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from .adapters.mujoco import (
    MuJoCoEndEffectorConfig,
    MuJoCoObjectConfig,
    MuJoCoRobotConfig,
    parse_mujoco_adapter_config,
)
from .backends.curobo_ik import CuroboIKConfig, parse_curobo_ik_config
from .backends.ruckig import RuckigConfig, RuckigTimeParameterizer, parse_ruckig_config
from .backends.vamp import VampBackend, VampConfig, parse_vamp_config
from .contracts.geometry import RigidTransform
from .contracts.planning import (
    AttachedGeometry,
    Box,
    Capsule,
    JointLimits,
    PlanningBudgets,
    PlanningEndEffectorConfig,
    PlanningObjectConfig,
    PlanningPolicy,
    RobotModel,
    Sphere,
)
from .contracts.targeting import (
    ContinuousYawSymmetry,
    DiscreteYawSymmetry,
    EndEffectorTargetingConfig,
    GraspTemplateConfig,
    ParallelGripperCapabilities,
    TargetingDiagnosticsConfig,
    TargetingObjectConfig,
)
from .ik import create_ik_solver
from .planning import PlanningDependencies


@dataclass(frozen=True)
class _RobotDefinition:
    model_id: str
    base_frame: str
    joint_names: tuple[str, ...]
    joint_limits: tuple[JointLimits, ...]
    default_positions: tuple[float, ...]


@dataclass(frozen=True)
class EndEffectorProfile:
    profile_id: str
    family: str
    combined_model_id: str
    tcp_frame: str
    targeting: EndEffectorTargetingConfig
    planning: PlanningEndEffectorConfig
    joint_names: tuple[str, ...]
    joint_limits: tuple[JointLimits, ...]
    default_positions: tuple[float, ...]
    planning_backend_profile: str
    execution_adapter_profile: str


@dataclass(frozen=True)
class _PlanningBackendProfile:
    vamp: VampConfig
    ik_backend: str
    ik: CuroboIKConfig
    ruckig: RuckigConfig
    ik_solution_count: int
    pregrasp_ik_solution_count: int
    placement_sample_count: int


@dataclass(frozen=True)
class ExecutionSettings:
    control_frequency_hz: float
    control_log_frequency_hz: float
    diagnostic_log_frequency_hz: float
    world_frame: str
    source_support_geom_names: tuple[str, ...]
    minimum_departure_height_m: float
    target_position_tolerance_m: float
    post_release_observation_s: float
    settled_linear_speed_m_s: float
    settled_angular_speed_rad_s: float


@dataclass(frozen=True)
class Task2Config:
    base_robot: _RobotDefinition
    end_effectors: tuple[EndEffectorProfile, ...]
    targeting_objects: tuple[TargetingObjectConfig, ...]
    planning_objects: tuple[PlanningObjectConfig, ...]
    planning_policy: PlanningPolicy
    planning_backends: dict[str, _PlanningBackendProfile]
    planning_budgets: PlanningBudgets
    continuous_yaw_resolution_rad: float
    maximum_targeting_candidates: int
    execution_settings: ExecutionSettings
    mujoco_robots: dict[str, MuJoCoRobotConfig]
    mujoco_end_effectors: dict[str, MuJoCoEndEffectorConfig]
    mujoco_objects: tuple[MuJoCoObjectConfig, ...]

    def targeting_object(self, object_type: str) -> TargetingObjectConfig:
        for item in self.targeting_objects:
            if item.object_type == object_type:
                return item
        raise KeyError(f"unknown Targeting object_type: {object_type}")

    def planning_object(self, object_type: str) -> PlanningObjectConfig:
        for item in self.planning_objects:
            if item.object_type == object_type:
                return item
        raise KeyError(f"unknown Planning object_type: {object_type}")

    def select_end_effector(
        self,
        object_type: str,
        requested_profile_id: str | None,
    ) -> EndEffectorProfile:
        object_config = self.targeting_object(object_type)
        families = {template.end_effector_family for template in object_config.templates}
        if len(families) != 1:
            raise ValueError(f"object {object_type!r} must resolve to exactly one end-effector family")
        family = next(iter(families))
        if requested_profile_id is not None:
            matches = [item for item in self.end_effectors if item.profile_id == requested_profile_id]
            if len(matches) != 1 or matches[0].family != family:
                raise ValueError(
                    f"requested end-effector profile {requested_profile_id!r} is incompatible with {family!r}"
                )
            return matches[0]
        matches = [item for item in self.end_effectors if item.family == family]
        if len(matches) != 1:
            raise ValueError(
                f"end-effector family {family!r} resolved to {len(matches)} profiles; exactly one is required"
            )
        return matches[0]

    def robot_model(self, profile: EndEffectorProfile) -> RobotModel:
        result = RobotModel(
            model_id=profile.combined_model_id,
            base_frame=self.base_robot.base_frame,
            tcp_frame=profile.tcp_frame,
            joint_names=self.base_robot.joint_names + profile.joint_names,
            joint_limits=self.base_robot.joint_limits + profile.joint_limits,
            default_joint_positions=self.base_robot.default_positions + profile.default_positions,
        )
        result.validate()
        return result

    def planning_dependencies(self, profile: EndEffectorProfile) -> PlanningDependencies:
        backend = self.planning_backends.get(profile.planning_backend_profile)
        if backend is None:
            raise ValueError(
                f"Planning backend has no profile {profile.planning_backend_profile!r}"
            )
        return PlanningDependencies(
            create_vamp=partial(VampBackend, config=backend.vamp),
            create_ik=lambda robot_model, random_seed: create_ik_solver(
                backend.ik_backend,
                robot_model,
                self.planning_policy,
                backend.ik,
                random_seed=random_seed,
            ),
            create_ruckig=partial(RuckigTimeParameterizer, config=backend.ruckig),
            ik_solution_count=backend.ik_solution_count,
            pregrasp_ik_solution_count=backend.pregrasp_ik_solution_count,
            placement_sample_count=backend.placement_sample_count,
        )


def load_task2_config(path: str | Path) -> Task2Config:
    pipeline_path = Path(path).resolve()
    base_dir = pipeline_path.parent
    root = _root(_load_yaml(pipeline_path), "pipeline", pipeline_path)
    _exact(
        root,
        {
            "objects_config",
            "robot_config",
            "planning_config",
            "execution_config",
            "end_effector_profiles",
            "planning_backend",
            "execution_adapter",
            "targeting",
            "planning_budgets",
        },
        "pipeline",
    )
    objects_path = _resolve(base_dir, _string(root, "objects_config", "pipeline"))
    robot_path = _resolve(base_dir, _string(root, "robot_config", "pipeline"))
    planning_path = _resolve(base_dir, _string(root, "planning_config", "pipeline"))
    execution_path = _resolve(base_dir, _string(root, "execution_config", "pipeline"))
    profile_paths = tuple(
        _resolve(base_dir, _as_string(value, f"pipeline.end_effector_profiles[{index}]"))
        for index, value in enumerate(_list(root, "end_effector_profiles", "pipeline"))
    )
    if not profile_paths:
        raise ValueError("pipeline.end_effector_profiles must not be empty")

    profiles = tuple(_parse_end_effector(path) for path in profile_paths)
    profile_ids = [item.profile_id for item in profiles]
    if len(profile_ids) != len(set(profile_ids)):
        raise ValueError("end-effector profile IDs must be unique")
    base_robot = _parse_robot(robot_path)
    targeting_objects, planning_objects = _parse_objects(objects_path)
    planning_backend_id = _string(root, "planning_backend", "pipeline")
    execution_adapter_id = _string(root, "execution_adapter", "pipeline")
    policy, backends = _parse_planning(
        planning_path, base_dir, planning_backend_id
    )
    settings, robots, end_effectors, execution_objects = _parse_execution(
        execution_path, base_dir, profiles, execution_adapter_id
    )
    targeting = _mapping(root, "targeting", "pipeline")
    budgets = _mapping(root, "planning_budgets", "pipeline")
    result = Task2Config(
        base_robot=base_robot,
        end_effectors=profiles,
        targeting_objects=targeting_objects,
        planning_objects=planning_objects,
        planning_policy=policy,
        planning_backends=backends,
        planning_budgets=PlanningBudgets(
            _integer(budgets, "maximum_candidates", "pipeline.planning_budgets"),
            _integer(budgets, "seeds_per_candidate", "pipeline.planning_budgets"),
            _number(budgets, "stage_timeout_s", "pipeline.planning_budgets"),
            _number(budgets, "total_timeout_s", "pipeline.planning_budgets"),
        ),
        continuous_yaw_resolution_rad=_number(
            targeting, "continuous_yaw_resolution_rad", "pipeline.targeting"
        ),
        maximum_targeting_candidates=_integer(
            targeting, "maximum_candidates", "pipeline.targeting"
        ),
        execution_settings=settings,
        mujoco_robots=robots,
        mujoco_end_effectors=end_effectors,
        mujoco_objects=execution_objects,
    )
    result.planning_policy.validate()
    result.planning_budgets.validate()
    if not backends:
        raise ValueError("pipeline planning backend has no end-effector profiles")
    return result


def _parse_robot(path: Path) -> _RobotDefinition:
    payload = _root(_load_yaml(path), "robot", path)
    _exact(payload, {"model_id", "base_frame", "joints"}, "robot")
    names, limits, defaults = _parse_joints(_list(payload, "joints", "robot"), "robot.joints")
    return _RobotDefinition(
        _string(payload, "model_id", "robot"),
        _string(payload, "base_frame", "robot"),
        names,
        limits,
        defaults,
    )


def _parse_end_effector(path: Path) -> EndEffectorProfile:
    payload = _root(_load_yaml(path), "end_effector", path)
    _exact(
        payload,
        {
            "profile_id",
            "family",
            "combined_model_id",
            "tcp_frame",
            "capabilities",
            "joints",
            "planning",
            "execution",
        },
        "end_effector",
    )
    profile_id = _string(payload, "profile_id", "end_effector")
    family = _string(payload, "family", "end_effector")
    capabilities = _mapping(payload, "capabilities", "end_effector")
    opening = _pair(capabilities, "opening_command_range_m", "end_effector.capabilities")
    force = _pair(capabilities, "grip_force_limit_range_n", "end_effector.capabilities")
    targeting = EndEffectorTargetingConfig(
        profile_id,
        family,
        ParallelGripperCapabilities(
            opening,
            force,
            _number(capabilities, "maximum_closing_speed_m_s", "end_effector.capabilities"),
            _strings(capabilities, "contact_groups", "end_effector.capabilities"),
        ),
    )
    planning_payload = _mapping(payload, "planning", "end_effector")
    planning = PlanningEndEffectorConfig(
        profile_id,
        _string(planning_payload, "actuator_position_channel", "end_effector.planning"),
        _string(planning_payload, "actuator_force_limit_channel", "end_effector.planning"),
        _number(planning_payload, "open_position_m", "end_effector.planning"),
        _strings(planning_payload, "contact_groups", "end_effector.planning"),
    )
    execution_payload = _mapping(payload, "execution", "end_effector")
    names, limits, defaults = _parse_joints(
        _list(payload, "joints", "end_effector"), "end_effector.joints"
    )
    targeting.validate()
    planning.validate()
    return EndEffectorProfile(
        profile_id,
        family,
        _string(payload, "combined_model_id", "end_effector"),
        _string(payload, "tcp_frame", "end_effector"),
        targeting,
        planning,
        names,
        limits,
        defaults,
        _string(planning_payload, "backend_profile", "end_effector.planning"),
        _string(execution_payload, "adapter_profile", "end_effector.execution"),
    )


def _parse_joints(
    values: list[object], path: str
) -> tuple[tuple[str, ...], tuple[JointLimits, ...], tuple[float, ...]]:
    names = []
    limits = []
    defaults = []
    for index, value in enumerate(values):
        item = _as_mapping(value, f"{path}[{index}]")
        position = _pair(item, "position_rad", f"{path}[{index}]")
        continuous = item.get("continuous", False)
        if not isinstance(continuous, bool):
            raise ValueError(f"{path}[{index}].continuous must be a boolean")
        names.append(_string(item, "name", f"{path}[{index}]"))
        limits.append(
            JointLimits(
                position[0],
                position[1],
                _number(item, "velocity_rad_s", f"{path}[{index}]"),
                _number(item, "acceleration_rad_s2", f"{path}[{index}]"),
                _number(item, "jerk_rad_s3", f"{path}[{index}]"),
                continuous,
            )
        )
        defaults.append(_number(item, "default_rad", f"{path}[{index}]"))
    if not names or len(names) != len(set(names)):
        raise ValueError(f"{path} joint names must be unique and non-empty")
    return tuple(names), tuple(limits), tuple(defaults)


def _parse_objects(
    path: Path,
) -> tuple[tuple[TargetingObjectConfig, ...], tuple[PlanningObjectConfig, ...]]:
    objects = _root(_load_yaml(path), "objects", path)
    targeting = []
    planning = []
    for object_type, raw in objects.items():
        item = _as_mapping(raw, f"objects.{object_type}")
        _exact(item, {"targeting", "planning"}, f"objects.{object_type}")
        target_payload = _mapping(item, "targeting", f"objects.{object_type}")
        templates = tuple(
            _parse_template(value, object_type, index)
            for index, value in enumerate(
                _list(target_payload, "templates", f"objects.{object_type}.targeting")
            )
        )
        target_config = TargetingObjectConfig(object_type, templates)
        target_config.validate()
        targeting.append(target_config)
        plan_payload = _mapping(item, "planning", f"objects.{object_type}")
        geometries = tuple(
            _parse_attached(value, object_type, index)
            for index, value in enumerate(
                _list(plan_payload, "attached_geometry", f"objects.{object_type}.planning")
            )
        )
        plan_config = PlanningObjectConfig(object_type, geometries)
        plan_config.validate()
        planning.append(plan_config)
    return tuple(targeting), tuple(planning)


def _parse_template(value: object, object_type: str, index: int) -> GraspTemplateConfig:
    path = f"objects.{object_type}.targeting.templates[{index}]"
    item = _as_mapping(value, path)
    yaw = _mapping(item, "yaw_symmetry", path)
    axis = _string(yaw, "axis", f"{path}.yaw_symmetry")
    axis_vector = _vector(yaw, "axis_vector", f"{path}.yaw_symmetry")
    kind = _string(yaw, "kind", f"{path}.yaw_symmetry")
    if kind == "discrete":
        symmetry = DiscreteYawSymmetry(
            _string(yaw, "group", f"{path}.yaw_symmetry"),
            axis,
            axis_vector,
            tuple(_as_number(v, f"{path}.yaw_symmetry.values_rad") for v in _list(yaw, "values_rad", path)),
        )
    elif kind == "continuous":
        bounds = _mapping(yaw, "range_rad", f"{path}.yaw_symmetry")
        symmetry = ContinuousYawSymmetry(
            axis,
            axis_vector,
            _number(bounds, "lower", f"{path}.yaw_symmetry.range_rad"),
            _number(bounds, "upper", f"{path}.yaw_symmetry.range_rad"),
            _boolean(bounds, "lower_inclusive", f"{path}.yaw_symmetry.range_rad"),
            _boolean(bounds, "upper_inclusive", f"{path}.yaw_symmetry.range_rad"),
        )
    else:
        raise ValueError(f"{path}.yaw_symmetry.kind is unsupported")
    diagnostics = _mapping(item, "diagnostics", path)
    return GraspTemplateConfig(
        _string(item, "template_id", path),
        object_type,
        _string(item, "end_effector_family", path),
        _string(item, "calibration", path),
        _transform(_mapping(item, "T_object_tcp", path), f"{path}.T_object_tcp"),
        _transform(_mapping(item, "T_tcp_pregrasp", path), f"{path}.T_tcp_pregrasp"),
        symmetry,
        _number(item, "pregrasp_opening_m", path),
        _number(item, "target_opening_m", path),
        _number(item, "grip_force_limit_n", path),
        _number(item, "minimum_retention_force_n", path),
        _number(item, "closing_speed_m_s", path),
        _number(item, "dwell_s", path),
        _strings(item, "expected_contact_groups", path),
        _number(item, "template_priority", path),
        TargetingDiagnosticsConfig(
            _number(diagnostics, "contact_margin_m", f"{path}.diagnostics"),
            _number(diagnostics, "opening_margin_m", f"{path}.diagnostics"),
            _number(diagnostics, "force_margin_n", f"{path}.diagnostics"),
        ),
    )


def _parse_attached(value: object, object_type: str, index: int) -> AttachedGeometry:
    path = f"objects.{object_type}.planning.attached_geometry[{index}]"
    item = _as_mapping(value, path)
    geometry_id = _string(item, "geometry_id", path)
    pose = _compact_transform(_mapping(item, "T_object_geometry", path), f"{path}.T_object_geometry")
    kinds = [key for key in ("sphere", "capsule", "box") if key in item]
    if len(kinds) != 1:
        raise ValueError(f"{path} must define exactly one attached geometry kind")
    raw = _as_mapping(item[kinds[0]], f"{path}.{kinds[0]}")
    if kinds[0] == "sphere":
        geometry = Sphere(_number(raw, "radius_m", f"{path}.sphere"))
    elif kinds[0] == "capsule":
        geometry = Capsule(
            _number(raw, "radius_m", f"{path}.capsule"),
            _number(raw, "length_m", f"{path}.capsule"),
        )
    else:
        geometry = Box(_vector(raw, "size_m", f"{path}.box"))
    return AttachedGeometry(geometry_id, pose, geometry)


def _parse_planning(
    path: Path,
    base_dir: Path,
    backend_id: str,
) -> tuple[PlanningPolicy, dict[str, _PlanningBackendProfile]]:
    payload = _load_yaml(path)
    _exact(payload, {"policy", "backends"}, "planning config")
    policy_payload = _mapping(payload, "policy", "planning config")
    policy = PlanningPolicy(
        _number(policy_payload, "ik_position_tolerance_m", "policy"),
        _number(policy_payload, "ik_orientation_tolerance_rad", "policy"),
        _number(policy_payload, "cartesian_translation_step_m", "policy"),
        _number(policy_payload, "cartesian_rotation_step_rad", "policy"),
        _number(policy_payload, "approach_contact_transition_m", "policy"),
        _number(policy_payload, "lift_distance_m", "policy"),
        _number(policy_payload, "preplace_offset_m", "policy"),
        _number(policy_payload, "joint_limit_margin_rad", "policy"),
        _number(
            policy_payload,
            "pregrasp_joint_path_detour_ratio_limit",
            "policy",
        ),
        _number(
            policy_payload,
            "payload_tcp_max_angular_velocity_rad_s",
            "policy",
        ),
        _number(
            policy_payload,
            "payload_tcp_max_translation_velocity_m_s",
            "policy",
        ),
        _number(
            policy_payload,
            "payload_tcp_max_translation_acceleration_m_s2",
            "policy",
        ),
        _number(
            policy_payload,
            "payload_motion_validation_time_step_s",
            "policy",
        ),
        _number(policy_payload, "trajectory_validation_joint_step_rad", "policy"),
        _number(
            policy_payload,
            "trajectory_validation_tcp_translation_step_m",
            "policy",
        ),
        _number(
            policy_payload,
            "trajectory_validation_tcp_rotation_step_rad",
            "policy",
        ),
        _number(
            policy_payload,
            "trajectory_validation_actuator_position_step_m",
            "policy",
        ),
        _number(
            policy_payload,
            "trajectory_validation_swept_point_step_m",
            "policy",
        ),
        _integer(policy_payload, "trajectory_validation_max_samples", "policy"),
    )
    backends = _mapping(payload, "backends", "planning config")
    if backend_id not in backends:
        raise ValueError(f"planning backend {backend_id!r} is unavailable")
    backend = _as_mapping(backends[backend_id], f"backends.{backend_id}")
    vamp_payload = _mapping(backend, "vamp", f"backends.{backend_id}")
    ik_payload = _mapping(backend, "ik", f"backends.{backend_id}")
    ruckig_payload = _mapping(backend, "ruckig", f"backends.{backend_id}")
    end_profiles = _mapping(
        backend, "end_effector_profiles", f"backends.{backend_id}"
    )
    profiles = {}
    for profile_id, raw_profile in end_profiles.items():
        profile = _as_mapping(
            raw_profile, f"backends.{backend_id}.end_effector_profiles.{profile_id}"
        )
        ik_backend, ik_config, solution_count, pregrasp_solution_count = parse_curobo_ik_config(
            ik_payload, profile, base_dir
        )
        profiles[profile_id] = _PlanningBackendProfile(
            parse_vamp_config(vamp_payload, profile),
            ik_backend,
            ik_config,
            parse_ruckig_config(ruckig_payload),
            solution_count,
            pregrasp_solution_count,
            _integer(backend, "placement_sample_count", f"backends.{backend_id}"),
        )
    return policy, profiles


def _parse_execution(
    path: Path,
    base_dir: Path,
    profiles: tuple[EndEffectorProfile, ...],
    adapter_id: str,
) -> tuple[
    ExecutionSettings,
    dict[str, MuJoCoRobotConfig],
    dict[str, MuJoCoEndEffectorConfig],
    tuple[MuJoCoObjectConfig, ...],
]:
    payload = _load_yaml(path)
    _exact(payload, {"execution", "adapters"}, "execution config")
    execution = _mapping(payload, "execution", "execution config")
    diagnostics = _mapping(execution, "diagnostics", "execution")
    settings = ExecutionSettings(
        _number(execution, "control_frequency_hz", "execution"),
        _number(execution, "control_log_frequency_hz", "execution"),
        _number(execution, "diagnostic_log_frequency_hz", "execution"),
        _string(diagnostics, "world_frame", "execution.diagnostics"),
        _strings(diagnostics, "source_support_geom_names", "execution.diagnostics"),
        _number(diagnostics, "minimum_departure_height_m", "execution.diagnostics"),
        _number(diagnostics, "target_position_tolerance_m", "execution.diagnostics"),
        _number(diagnostics, "post_release_observation_s", "execution.diagnostics"),
        _number(diagnostics, "settled_linear_speed_m_s", "execution.diagnostics"),
        _number(diagnostics, "settled_angular_speed_rad_s", "execution.diagnostics"),
    )
    adapters = _mapping(payload, "adapters", "execution config")
    if adapter_id != "mujoco" or adapter_id not in adapters:
        raise ValueError(f"execution adapter {adapter_id!r} is unavailable")
    adapter = parse_mujoco_adapter_config(
        _as_mapping(adapters[adapter_id], f"adapters.{adapter_id}"),
        base_dir,
        tuple(
            (profile.profile_id, profile.execution_adapter_profile)
            for profile in profiles
        ),
    )
    return (
        settings,
        dict(adapter.robots),
        dict(adapter.end_effectors),
        adapter.objects,
    )


def _transform(payload: dict[str, Any], path: str) -> RigidTransform:
    rows = _list(payload, "matrix", path)
    if len(rows) != 4:
        raise ValueError(f"{path}.matrix must have four rows")
    matrix = tuple(
        tuple(_as_number(value, f"{path}.matrix[{row}]") for value in _as_list(values, path))
        for row, values in enumerate(rows)
    )
    if any(len(row) != 4 for row in matrix):
        raise ValueError(f"{path}.matrix must be 4x4")
    result = RigidTransform(
        _string(payload, "parent_frame", path),
        _string(payload, "child_frame", path),
        matrix,  # type: ignore[arg-type]
    )
    result.validate(label=path)
    return result


def _compact_transform(payload: dict[str, Any], path: str) -> RigidTransform:
    from scipy.spatial.transform import Rotation

    translation = _vector(payload, "translation_m", path)
    quaternion = _list(payload, "quaternion_wxyz", path)
    if len(quaternion) != 4:
        raise ValueError(f"{path}.quaternion_wxyz must have four values")
    w, x, y, z = (_as_number(value, path) for value in quaternion)
    rotation = Rotation.from_quat((x, y, z, w)).as_matrix()
    matrix = tuple(
        tuple(float(rotation[row, column]) for column in range(3)) + (translation[row],)
        for row in range(3)
    ) + ((0.0, 0.0, 0.0, 1.0),)
    result = RigidTransform(
        _string(payload, "parent_frame", path),
        _string(payload, "child_frame", path),
        matrix,  # type: ignore[arg-type]
    )
    result.validate(label=path)
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Task2 config does not exist: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid Task2 YAML in {path}: {exc}") from exc
    return _as_mapping(payload, str(path))


def _resolve(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _root(payload: dict[str, Any], key: str, path: object) -> dict[str, Any]:
    _exact(payload, {key}, str(path))
    return _mapping(payload, key, str(path))


def _exact(payload: dict[str, Any], expected: set[str], path: str) -> None:
    actual = set(payload)
    if missing := sorted(expected - actual):
        raise ValueError(f"{path} is missing required fields: {missing!r}")
    if unknown := sorted(actual - expected):
        raise ValueError(f"{path} contains unknown fields: {unknown!r}")


def _mapping(payload: dict[str, Any], key: str, path: str) -> dict[str, Any]:
    if key not in payload:
        raise ValueError(f"{path}.{key} is required")
    return _as_mapping(payload[key], f"{path}.{key}")


def _as_mapping(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path} must be a string-keyed mapping")
    return value


def _list(payload: dict[str, Any], key: str, path: str) -> list[object]:
    if key not in payload:
        raise ValueError(f"{path}.{key} is required")
    return _as_list(payload[key], f"{path}.{key}")


def _as_list(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list")
    return value


def _string(payload: dict[str, Any], key: str, path: str) -> str:
    if key not in payload:
        raise ValueError(f"{path}.{key} is required")
    return _as_string(payload[key], f"{path}.{key}")


def _as_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _number(payload: dict[str, Any], key: str, path: str) -> float:
    if key not in payload:
        raise ValueError(f"{path}.{key} is required")
    return _as_number(payload[key], f"{path}.{key}")


def _as_number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _integer(payload: dict[str, Any], key: str, path: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path}.{key} must be an integer")
    return value


def _boolean(payload: dict[str, Any], key: str, path: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{path}.{key} must be a boolean")
    return value


def _pair(payload: dict[str, Any], key: str, path: str) -> tuple[float, float]:
    values = _list(payload, key, path)
    if len(values) != 2:
        raise ValueError(f"{path}.{key} must contain two values")
    return _as_number(values[0], path), _as_number(values[1], path)


def _vector(payload: dict[str, Any], key: str, path: str) -> tuple[float, float, float]:
    values = _list(payload, key, path)
    if len(values) != 3:
        raise ValueError(f"{path}.{key} must contain three values")
    return tuple(_as_number(value, path) for value in values)  # type: ignore[return-value]


def _strings(payload: dict[str, Any], key: str, path: str) -> tuple[str, ...]:
    return tuple(
        _as_string(value, f"{path}.{key}[{index}]")
        for index, value in enumerate(_list(payload, key, path))
    )
