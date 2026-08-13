from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi, sqrt

from .geometry import RigidTransform, Vector3


@dataclass(frozen=True)
class DiscreteYawSymmetry:
    group: str
    axis: str
    axis_vector: Vector3
    values_rad: tuple[float, ...]

    def validate(self) -> None:
        _validate_axis(self.axis, self.axis_vector)
        if not self.group:
            raise ValueError("discrete yaw group must be non-empty")
        if not self.values_rad or not all(isfinite(value) for value in self.values_rad):
            raise ValueError("discrete yaw values_rad must contain finite values")
        if len(set(self.values_rad)) != len(self.values_rad):
            raise ValueError("discrete yaw values_rad must not contain duplicates")


@dataclass(frozen=True)
class ContinuousYawSymmetry:
    axis: str
    axis_vector: Vector3
    lower_rad: float
    upper_rad: float
    lower_inclusive: bool
    upper_inclusive: bool

    def validate(self) -> None:
        _validate_axis(self.axis, self.axis_vector)
        if not isfinite(self.lower_rad) or not isfinite(self.upper_rad):
            raise ValueError("continuous yaw range bounds must be finite")
        if abs(self.lower_rad + pi) > 1e-9 or abs(self.upper_rad - pi) > 1e-9:
            raise ValueError("continuous yaw range must be [-pi, pi)")
        if not self.lower_inclusive or self.upper_inclusive:
            raise ValueError("continuous yaw range must be lower-inclusive and upper-exclusive")


YawSymmetry = DiscreteYawSymmetry | ContinuousYawSymmetry


@dataclass(frozen=True)
class TargetingDiagnosticsConfig:
    contact_margin_m: float
    opening_margin_m: float
    force_margin_n: float

    def validate(self) -> None:
        values = (self.contact_margin_m, self.opening_margin_m, self.force_margin_n)
        if not all(isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("targeting diagnostic margins must be finite and non-negative")


@dataclass(frozen=True)
class GraspTemplateConfig:
    template_id: str
    object_type: str
    end_effector_family: str
    calibration: str
    T_object_tcp: RigidTransform
    T_tcp_pregrasp: RigidTransform
    yaw_symmetry: YawSymmetry
    pregrasp_opening_m: float
    target_opening_m: float
    grip_force_limit_n: float
    minimum_retention_force_n: float
    closing_speed_m_s: float
    dwell_s: float
    expected_contact_groups: tuple[str, ...]
    template_priority: float
    diagnostics: TargetingDiagnosticsConfig

    def validate(self) -> None:
        if not self.template_id or not self.object_type or not self.end_effector_family:
            raise ValueError("template_id, object_type, and end_effector_family must be non-empty")
        if self.calibration != "simulation_nominal":
            raise ValueError("Targeting templates currently require simulation_nominal calibration")
        self.T_object_tcp.validate(label=f"{self.template_id} T_object_tcp")
        self.T_tcp_pregrasp.validate(label=f"{self.template_id} T_tcp_pregrasp")
        if self.T_object_tcp.parent_frame != "object":
            raise ValueError(f"{self.template_id} T_object_tcp parent_frame must be 'object'")
        if self.T_object_tcp.child_frame != self.T_tcp_pregrasp.parent_frame:
            raise ValueError(f"{self.template_id} TCP transforms have disconnected frames")
        self.yaw_symmetry.validate()
        command_values = (
            self.pregrasp_opening_m,
            self.target_opening_m,
            self.grip_force_limit_n,
            self.minimum_retention_force_n,
            self.closing_speed_m_s,
            self.dwell_s,
        )
        if not all(isfinite(value) for value in command_values):
            raise ValueError(f"{self.template_id} command values must be finite")
        if self.target_opening_m < 0.0 or self.pregrasp_opening_m < self.target_opening_m:
            raise ValueError(
                f"{self.template_id} openings must satisfy 0 <= target <= pregrasp"
            )
        if self.minimum_retention_force_n <= 0.0:
            raise ValueError(f"{self.template_id} minimum_retention_force_n must be positive")
        if self.grip_force_limit_n < self.minimum_retention_force_n:
            raise ValueError(
                f"{self.template_id} grip_force_limit_n must cover minimum retention force"
            )
        if self.closing_speed_m_s <= 0.0 or self.dwell_s < 0.0:
            raise ValueError(f"{self.template_id} closing speed must be positive and dwell non-negative")
        if not self.expected_contact_groups or len(set(self.expected_contact_groups)) != len(
            self.expected_contact_groups
        ):
            raise ValueError(f"{self.template_id} expected contact groups must be unique and non-empty")
        if not isfinite(self.template_priority):
            raise ValueError(f"{self.template_id} template_priority must be finite")
        self.diagnostics.validate()


@dataclass(frozen=True)
class TargetingObjectConfig:
    object_type: str
    templates: tuple[GraspTemplateConfig, ...]

    def validate(self) -> None:
        if not self.object_type or not self.templates:
            raise ValueError("Targeting object type and templates must be non-empty")
        template_ids = [template.template_id for template in self.templates]
        if len(template_ids) != len(set(template_ids)):
            raise ValueError(f"{self.object_type} template_id values must be unique")
        for template in self.templates:
            template.validate()
            if template.object_type != self.object_type:
                raise ValueError(
                    f"template {template.template_id} object_type does not match {self.object_type}"
                )


@dataclass(frozen=True)
class ParallelGripperCapabilities:
    opening_command_range_m: tuple[float, float]
    grip_force_limit_range_n: tuple[float, float]
    maximum_closing_speed_m_s: float
    contact_groups: tuple[str, ...]

    def validate(self) -> None:
        opening_min, opening_max = self.opening_command_range_m
        force_min, force_max = self.grip_force_limit_range_n
        values = (opening_min, opening_max, force_min, force_max, self.maximum_closing_speed_m_s)
        if not all(isfinite(value) for value in values):
            raise ValueError("parallel gripper capabilities must be finite")
        if opening_min < 0.0 or opening_max <= opening_min:
            raise ValueError("opening command range must be increasing and non-negative")
        if force_min < 0.0 or force_max <= force_min:
            raise ValueError("grip force limit range must be increasing and non-negative")
        if self.maximum_closing_speed_m_s <= 0.0:
            raise ValueError("maximum closing speed must be positive")
        if not self.contact_groups or len(set(self.contact_groups)) != len(self.contact_groups):
            raise ValueError("parallel gripper contact groups must be unique and non-empty")


@dataclass(frozen=True)
class EndEffectorTargetingConfig:
    profile_id: str
    family: str
    capabilities: ParallelGripperCapabilities

    def validate(self) -> None:
        if not self.profile_id:
            raise ValueError("end-effector profile_id must be non-empty")
        if not self.family:
            raise ValueError("end-effector family must be non-empty")
        self.capabilities.validate()


@dataclass(frozen=True)
class TargetingConfig:
    objects: tuple[TargetingObjectConfig, ...]
    end_effectors: tuple[EndEffectorTargetingConfig, ...]

    def validate(self) -> None:
        if not self.objects or not self.end_effectors:
            raise ValueError("Targeting config requires objects and end effectors")
        object_types = [item.object_type for item in self.objects]
        profiles = [item.profile_id for item in self.end_effectors]
        if len(object_types) != len(set(object_types)):
            raise ValueError("Targeting object_type values must be unique")
        if len(profiles) != len(set(profiles)):
            raise ValueError("Targeting end-effector profile_id values must be unique")
        for item in self.objects:
            item.validate()
        for item in self.end_effectors:
            item.validate()


def _validate_axis(name: str, vector: Vector3) -> None:
    if name != "approach_axis_object":
        raise ValueError("yaw axis must be approach_axis_object")
    if len(vector) != 3 or not all(isfinite(value) for value in vector):
        raise ValueError("yaw axis_vector must contain three finite values")
    if abs(sqrt(sum(value * value for value in vector)) - 1.0) > 1e-6:
        raise ValueError("yaw axis_vector must be normalized")
