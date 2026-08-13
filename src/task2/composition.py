from __future__ import annotations

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
)


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
