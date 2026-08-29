from __future__ import annotations

from collections.abc import Iterable
from itertools import islice
from math import atan2, ceil, cos, isfinite, pi, sin, sqrt
from struct import pack

from .contracts.geometry import compose, inverse, reframe, rotate_transform_about_parent_axis
from .contracts.targeting import (
    ContinuousYawSymmetry,
    DiscreteYawSymmetry,
    GraspCandidate,
    GraspTemplateConfig,
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
        candidate
        for template in sorted(
            compatible_templates,
            key=lambda item: (-item.template_priority, item.template_id),
        )
        for candidate in _expand_template(request, template)
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
    if request.access_context is not None:
        request.access_context.validate()


def _expand_template(
    request: TargetingRequest,
    template: GraspTemplateConfig,
) -> Iterable[GraspCandidate]:
    if template.portal_adaptive_approach is not None and request.access_context is not None:
        return _build_portal_adaptive_candidates(request, template)
    return (
        _build_candidate(request, template, yaw_rad)
        for yaw_rad in _expand_yaw(template, request.continuous_yaw_resolution_rad)
    )


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
    return _candidate_from_transform(request, template, yaw_rad, T_object_tcp)


def _build_portal_adaptive_candidates(
    request: TargetingRequest,
    template: GraspTemplateConfig,
) -> tuple[GraspCandidate, ...]:
    adaptive = template.portal_adaptive_approach
    context = request.access_context
    assert adaptive is not None and context is not None
    world_object = request.T_world_object.matrix
    world_object_rotation = tuple(tuple(row[column] for column in range(3)) for row in world_object[:3])
    object_world_rotation = tuple(
        tuple(world_object_rotation[column][row] for column in range(3))
        for row in range(3)
    )
    radial = (
        world_object[0][3] - context.portal_world_xy_m[0],
        world_object[1][3] - context.portal_world_xy_m[1],
        0.0,
    )
    radial_norm = _norm(radial)
    nominal_tilt = min(
        adaptive.maximum_tilt_rad,
        atan2(radial_norm, context.vertical_span_m),
    )
    fallback_tilt = min(
        adaptive.maximum_tilt_rad,
        nominal_tilt + adaptive.tilt_margin_rad,
    )
    tilt_values = tuple(
        dict.fromkeys((nominal_tilt, 0.5 * (nominal_tilt + fallback_tilt), fallback_tilt))
    )
    axis_count = round(pi / adaptive.contact_axis_step_rad)
    ranked = []
    for axis_index in range(axis_count):
        axis_rad = axis_index * adaptive.contact_axis_step_rad
        closing_object = (cos(axis_rad), sin(axis_rad), 0.0)
        closing_world = _normalized(_matvec(world_object_rotation, closing_object))
        vertical = _normalized(_subtract((0.0, 0.0, -1.0), _scale(closing_world, -closing_world[2])))
        horizontal = _normalized(_cross(closing_world, vertical))
        if radial_norm > 1e-12 and _dot(horizontal, radial) < 0.0:
            horizontal = _scale(horizontal, -1.0)
        alignment = 0.0 if radial_norm <= 1e-12 else _dot(horizontal, radial) / radial_norm
        for tilt_index, tilt_rad in enumerate(tilt_values):
            approach_world = _normalized(
                _add(
                    _scale(vertical, cos(tilt_rad)),
                    _scale(horizontal, sin(tilt_rad)),
                )
            )
            for sign_index, closing_sign in enumerate((1.0, -1.0)):
                signed_closing = _scale(closing_world, closing_sign)
                tcp_x_world = _normalized(_cross(signed_closing, approach_world))
                tcp_rotation_world = tuple(
                    (tcp_x_world[row], signed_closing[row], approach_world[row])
                    for row in range(3)
                )
                tcp_rotation_object = _matmul(
                    object_world_rotation, tcp_rotation_world
                )
                matrix = [list(row) for row in template.T_object_tcp.matrix]
                for row in range(3):
                    for column in range(3):
                        matrix[row][column] = tcp_rotation_object[row][column]
                T_object_tcp = type(template.T_object_tcp)(
                    template.T_object_tcp.parent_frame,
                    template.T_object_tcp.child_frame,
                    tuple(tuple(float(value) for value in row) for row in matrix),
                )
                yaw_rad = _wrap_yaw(
                    axis_rad if closing_sign > 0.0 else axis_rad + pi
                )
                candidate = _candidate_from_transform(
                    request,
                    template,
                    yaw_rad,
                    T_object_tcp,
                    pregrasp_distance_m=adaptive.pregrasp_distance_m,
                    id_suffix=(
                        f"axis_f64_{pack('>d', axis_rad).hex()}."
                        f"tilt_f64_{pack('>d', tilt_rad).hex()}."
                        f"sign_{'p' if closing_sign > 0.0 else 'm'}"
                    ),
                )
                verticality = max(0.0, min(1.0, -approach_world[2]))
                access_score = (
                    0.35
                    + 0.15 * (alignment + 1.0)
                    + 0.05
                    + 0.20 * verticality
                )
                ranked.append(
                    (
                        (
                            -access_score,
                            axis_index,
                            tilt_index,
                            sign_index,
                        ),
                        candidate,
                    )
                )
    ranked.sort(key=lambda item: item[0])
    return tuple(candidate for _, candidate in ranked)


def _candidate_from_transform(
    request: TargetingRequest,
    template: GraspTemplateConfig,
    yaw_rad: float,
    T_object_tcp,
    *,
    pregrasp_distance_m: float | None = None,
    id_suffix: str | None = None,
) -> GraspCandidate:
    T_world_tcp = compose(request.T_world_object, T_object_tcp)
    if pregrasp_distance_m is None:
        T_tcp_pregrasp = reframe(
            template.T_tcp_pregrasp,
            parent_frame=T_world_tcp.child_frame,
        )
    else:
        matrix = (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, -pregrasp_distance_m),
            (0.0, 0.0, 0.0, 1.0),
        )
        T_tcp_pregrasp = type(T_object_tcp)(
            T_world_tcp.child_frame,
            template.T_tcp_pregrasp.child_frame,
            matrix,
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
            f"{template.template_id}."
            + (
                id_suffix
                if id_suffix is not None
                else f"yaw_f64_{pack('>d', yaw_rad).hex()}"
            )
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


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _norm(vector: tuple[float, float, float]) -> float:
    return sqrt(_dot(vector, vector))


def _normalized(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = _norm(vector)
    if length <= 1e-12:
        raise ValueError("portal-adaptive approach produced a degenerate axis")
    return tuple(value / length for value in vector)  # type: ignore[return-value]


def _scale(vector: tuple[float, float, float], value: float) -> tuple[float, float, float]:
    return tuple(value * item for item in vector)  # type: ignore[return-value]


def _add(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(a + b for a, b in zip(left, right))  # type: ignore[return-value]


def _subtract(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(a - b for a, b in zip(left, right))  # type: ignore[return-value]


def _cross(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _matvec(matrix, vector: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(sum(matrix[row][column] * vector[column] for column in range(3)) for row in range(3))  # type: ignore[return-value]


def _matmul(left, right) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(sum(left[row][inner] * right[inner][column] for inner in range(3)) for column in range(3))
        for row in range(3)
    )


def _wrap_yaw(value: float) -> float:
    return (value + pi) % (2.0 * pi) - pi
