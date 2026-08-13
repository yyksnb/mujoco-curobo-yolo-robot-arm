from __future__ import annotations

from collections.abc import Iterable
from itertools import islice
from math import ceil, isfinite, pi
from struct import pack

from .contracts.config import ContinuousYawSymmetry, DiscreteYawSymmetry, GraspTemplateConfig
from .contracts.geometry import compose, inverse, reframe, rotate_transform_about_parent_axis
from .contracts.targeting import (
    GraspCandidate,
    ParallelGripperCommand,
    ScoreComponents,
    TargetingErrorCode,
    TargetingRequest,
    TargetingResult,
)


def target(request: TargetingRequest) -> TargetingResult:
    pose_error = _validate_request_pose(request)
    if pose_error is not None:
        return TargetingResult.failed(TargetingErrorCode.INVALID_INPUT_POSE, pose_error)

    if request.object_type != request.object_config.object_type:
        return TargetingResult.failed(
            TargetingErrorCode.UNKNOWN_OBJECT,
            f"no Targeting configuration for object_type {request.object_type!r}",
        )

    try:
        request.object_config.validate()
        request.end_effector_config.validate()
        _validate_expansion_limits(request)
    except (TypeError, ValueError) as exc:
        return TargetingResult.failed(TargetingErrorCode.INVALID_CONFIGURATION, str(exc))

    compatible_templates = tuple(
        template
        for template in request.object_config.templates
        if template.end_effector_family == request.end_effector_config.family
    )
    if not compatible_templates:
        return TargetingResult.failed(
            TargetingErrorCode.INCOMPATIBLE_END_EFFECTOR,
            f"object_type {request.object_type!r} has no template for end-effector family "
            f"{request.end_effector_config.family!r}",
        )

    for template in compatible_templates:
        incompatibility = _capability_incompatibility(request, template)
        if incompatibility is not None:
            return TargetingResult.failed(
                TargetingErrorCode.INCOMPATIBLE_END_EFFECTOR,
                incompatibility,
            )

    expanded = (
        _build_candidate(request, template, yaw_rad)
        for template in sorted(
            compatible_templates,
            key=lambda item: (-item.template_priority, item.template_id),
        )
        for yaw_rad in _expand_yaw(template, request.continuous_yaw_resolution_rad)
    )
    candidates = tuple(islice(expanded, request.maximum_candidates))
    if not candidates:
        return TargetingResult.failed(
            TargetingErrorCode.NO_CANDIDATES,
            f"Targeting expansion produced no candidates for {request.object_type!r}",
        )
    return TargetingResult.succeeded(candidates)


def _validate_request_pose(request: TargetingRequest) -> str | None:
    if not isinstance(request.object_type, str) or not request.object_type.strip():
        return "object_type must be a non-empty string"
    try:
        request.T_world_object.validate(label="T_world_object")
    except (TypeError, ValueError) as exc:
        return str(exc)
    if request.T_world_object.parent_frame != "world":
        return "T_world_object parent_frame must be 'world'"
    if request.T_world_object.child_frame != "object":
        return "T_world_object child_frame must be 'object'"
    return None


def _validate_expansion_limits(request: TargetingRequest) -> None:
    if not isfinite(request.continuous_yaw_resolution_rad):
        raise ValueError("continuous_yaw_resolution_rad must be finite")
    if request.continuous_yaw_resolution_rad <= 0.0:
        raise ValueError("continuous_yaw_resolution_rad must be positive")
    if -pi + request.continuous_yaw_resolution_rad == -pi:
        raise ValueError("continuous_yaw_resolution_rad is too small for deterministic expansion")
    if isinstance(request.maximum_candidates, bool) or not isinstance(request.maximum_candidates, int):
        raise ValueError("maximum_candidates must be an integer")
    if request.maximum_candidates <= 0:
        raise ValueError("maximum_candidates must be positive")


def _capability_incompatibility(
    request: TargetingRequest,
    template: GraspTemplateConfig,
) -> str | None:
    capabilities = request.end_effector_config.capabilities
    opening_min, opening_max = capabilities.opening_command_range_m
    force_min, force_max = capabilities.grip_force_limit_range_n
    if not opening_min <= template.target_opening_m <= opening_max:
        return f"template {template.template_id!r} target_opening_m is outside capability range"
    if not opening_min <= template.pregrasp_opening_m <= opening_max:
        return f"template {template.template_id!r} pregrasp_opening_m is outside capability range"
    if not force_min <= template.grip_force_limit_n <= force_max:
        return f"template {template.template_id!r} grip_force_limit_n is outside capability range"
    if template.closing_speed_m_s > capabilities.maximum_closing_speed_m_s:
        return f"template {template.template_id!r} closing_speed_m_s exceeds capability"
    unsupported_contacts = set(template.expected_contact_groups) - set(capabilities.contact_groups)
    if unsupported_contacts:
        return (
            f"template {template.template_id!r} requires unsupported contact groups: "
            f"{sorted(unsupported_contacts)!r}"
        )
    return None


def _expand_yaw(template: GraspTemplateConfig, resolution_rad: float) -> Iterable[float]:
    symmetry = template.yaw_symmetry
    if isinstance(symmetry, DiscreteYawSymmetry):
        return tuple(sorted(symmetry.values_rad))
    if isinstance(symmetry, ContinuousYawSymmetry):
        count = int(ceil((symmetry.upper_rad - symmetry.lower_rad) / resolution_rad))
        return (
            symmetry.lower_rad + index * resolution_rad
            for index in range(count)
            if symmetry.lower_rad + index * resolution_rad < symmetry.upper_rad
        )
    raise TypeError(f"unsupported yaw symmetry: {type(symmetry).__name__}")


def _build_candidate(
    request: TargetingRequest,
    template: GraspTemplateConfig,
    yaw_rad: float,
) -> GraspCandidate:
    T_object_tcp = rotate_transform_about_parent_axis(
        template.T_object_tcp,
        template.yaw_symmetry.axis_vector,
        yaw_rad,
    )
    T_world_tcp = compose(request.T_world_object, T_object_tcp)
    T_tcp_pregrasp = reframe(
        template.T_tcp_pregrasp,
        parent_frame=T_world_tcp.child_frame,
    )
    T_world_pregrasp = compose(T_world_tcp, T_tcp_pregrasp)
    score_components = ScoreComponents(template_priority=template.template_priority)
    command = ParallelGripperCommand(
        pregrasp_opening_m=template.pregrasp_opening_m,
        target_opening_m=template.target_opening_m,
        grip_force_limit_n=template.grip_force_limit_n,
        closing_speed_m_s=template.closing_speed_m_s,
        dwell_s=template.dwell_s,
        calibration=template.calibration,
    )
    command.validate()
    return GraspCandidate(
        candidate_id=(
            f"{request.object_type}.{request.end_effector_config.profile_id}."
            f"{template.template_id}.yaw_f64_{pack('>d', yaw_rad).hex()}"
        ),
        object_type=request.object_type,
        end_effector_profile_id=request.end_effector_config.profile_id,
        template_id=template.template_id,
        symmetry_yaw_rad=yaw_rad,
        T_object_tcp=T_object_tcp,
        T_world_tcp=T_world_tcp,
        T_world_pregrasp=T_world_pregrasp,
        T_tcp_object=inverse(T_object_tcp),
        quality_score=score_components.template_priority,
        score_components=score_components,
        diagnostics=template.diagnostics,
        grasp_command=command,
        expected_contacts=template.expected_contact_groups,
    )
