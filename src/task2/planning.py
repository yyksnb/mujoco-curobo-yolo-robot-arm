from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from math import ceil, cos, inf, nextafter, pi, sin, sqrt
from time import monotonic
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .backends.errors import BackendOperationError, BackendUnavailableError
from .backends.ruckig import (
    GeometricWaypoint,
    RuckigRouteError,
    RuckigTimeParameterizer,
)
from .backends.vamp import VampBackend
from .contracts import (
    GraspCandidate,
    LocalMotionStrategy,
    PlacementGoal,
    PlacementGoalKind,
    PlacementOrientationConstraint,
    PlannedTrajectory,
    PlanningAttempt,
    PlanningErrorCode,
    PlanningPolicy,
    PlanningRequest,
    PlanningResult,
    RigidTransform,
    RobotModel,
    RobotState,
    SceneGeometry,
    SceneSnapshot,
    TrajectoryKnot,
    TrajectoryPolynomialSection,
    TrajectorySegment,
    TrajectoryStage,
    VerticalTransitCorridor,
)
from .contracts.geometry import compose, inverse, reframe
from .ik import IKConstraints, IKErrorCode, IKSolution, IKSolver, validate_ik_result


@dataclass(frozen=True)
class _RouteFailure(Exception):
    code: PlanningErrorCode
    stage: TrajectoryStage
    message: str
    attempt_recorded: bool = False


@dataclass(frozen=True)
class _Route:
    waypoints: tuple[GeometricWaypoint, ...]
    events: tuple[tuple[str, TrajectoryStage, int, dict[str, float]], ...]
    T_world_object_goal: RigidTransform
    orientation_preference_cost_rad: float
    payload_cumulative_translation_m: float = 0.0
    payload_cumulative_rotation_rad: float = 0.0
    payload_nominal_duration_s: float = 0.0


@dataclass(frozen=True)
class _TargetContactAllowance:
    endpoint: frozenset[str]
    motion: frozenset[str]
    terminal_endpoint: bool
    terminal_motion: bool


@dataclass(frozen=True)
class _PayloadMotionMetrics:
    maximum_angular_velocity_rad_s: float
    maximum_translation_velocity_m_s: float
    maximum_translation_acceleration_m_s2: float
    angular_velocity_stage: TrajectoryStage
    translation_velocity_stage: TrajectoryStage
    translation_acceleration_stage: TrajectoryStage


@dataclass(frozen=True)
class _OrientationPathMetrics:
    maximum_deviation_rad: float
    cumulative_translation_m: float
    cumulative_rotation_rad: float
    sample_count: int


@dataclass(frozen=True)
class _CartesianRouteNode:
    state: RobotState
    path: tuple[tuple[float, ...], ...]
    maximum_joint_step_rad: float
    cumulative_squared_joint_step_rad2: float
    cumulative_joint_step_rad: float
    generation: int


_PAYLOAD_STAGES = frozenset(
    (TrajectoryStage.LIFT, TrajectoryStage.EGRESS, TrajectoryStage.PLACE)
)


@dataclass(frozen=True)
class PlanningDependencies:
    create_vamp: Callable[[object], VampBackend]
    create_ik: Callable[[object], IKSolver]
    create_ruckig: Callable[[object], RuckigTimeParameterizer]
    ik_solution_count: int
    pregrasp_ik_solution_count: int
    placement_sample_count: int
    vamp_sample_offset_start: int = 0
    vamp_sample_offset_stride: int = 1

    def validate(self) -> None:
        if (
            self.ik_solution_count <= 0
            or self.pregrasp_ik_solution_count <= 0
            or self.placement_sample_count <= 0
            or self.vamp_sample_offset_start < 0
            or self.vamp_sample_offset_stride <= 0
        ):
            raise ValueError("Planning backend solution and placement counts must be positive")


def plan(
    request: PlanningRequest,
    *,
    dependencies: PlanningDependencies | None = None,
) -> PlanningResult:
    try:
        request.robot_model.validate()
        request.object_config.validate()
        request.end_effector_config.validate()
        request.planning_policy.validate()
        if dependencies is None:
            raise ValueError("Planning backend dependencies must be configured explicitly")
        dependencies.validate()
    except (TypeError, ValueError) as exc:
        return PlanningResult.failed(PlanningErrorCode.INVALID_CONFIG, str(exc))
    try:
        request.validate()
    except (TypeError, ValueError) as exc:
        return PlanningResult.failed(PlanningErrorCode.INVALID_REQUEST, str(exc))
    try:
        vamp = dependencies.create_vamp(request.robot_model)
        ik_solver = dependencies.create_ik(request.robot_model)
        ruckig = dependencies.create_ruckig(request.robot_model)
    except BackendUnavailableError as exc:
        return PlanningResult.failed(PlanningErrorCode.BACKEND_UNAVAILABLE, str(exc))
    except (RuntimeError, ValueError, OSError) as exc:
        return PlanningResult.failed(PlanningErrorCode.BACKEND_ERROR, str(exc))

    started = monotonic()
    attempts: list[PlanningAttempt] = []
    if (
        request.object_config.approach_motion_strategy
        is LocalMotionStrategy.STAGED_PORTAL
    ):
        try:
            ordered_candidates = _order_candidates_by_pregrasp_ik_distance(
                request,
                request.candidates,
                ik_solver,
                vamp,
                dependencies.pregrasp_ik_solution_count,
            )
        except BackendUnavailableError as exc:
            return PlanningResult.failed(
                PlanningErrorCode.BACKEND_UNAVAILABLE,
                str(exc),
                metadata=_metadata(
                    request, vamp.version, ik_solver.version, ruckig.version
                ),
            )
        except _RouteFailure as failure:
            return PlanningResult.failed(
                failure.code,
                failure.message,
                metadata=_metadata(
                    request, vamp.version, ik_solver.version, ruckig.version
                ),
            )
        except BackendOperationError as exc:
            return PlanningResult.failed(
                PlanningErrorCode.BACKEND_ERROR,
                f"Planning backend error: {exc}",
                metadata=_metadata(
                    request, vamp.version, ik_solver.version, ruckig.version
                ),
            )
        except (RuntimeError, TypeError, ValueError, OSError) as exc:
            return PlanningResult.failed(
                PlanningErrorCode.BACKEND_ERROR,
                f"Planning backend error: {exc}",
                metadata=_metadata(
                    request, vamp.version, ik_solver.version, ruckig.version
                ),
            )
    else:
        ordered_candidates = request.candidates
    candidates = ordered_candidates[: request.budgets.maximum_candidates]
    last_failure: _RouteFailure | None = None
    for candidate in candidates:
        candidate_routes: list[tuple[int, int, _Route]] = []
        for search_attempt in range(request.budgets.search_attempts_per_candidate):
            sample_offset = (
                dependencies.vamp_sample_offset_start
                + search_attempt * dependencies.vamp_sample_offset_stride
            )
            if monotonic() - started >= request.budgets.total_timeout_s:
                if candidate_routes:
                    break
                return PlanningResult.failed(
                    PlanningErrorCode.BUDGET_EXHAUSTED,
                    "Planning total time budget was exhausted",
                    tuple(attempts),
                    _metadata(request, vamp.version, ik_solver.version, ruckig.version),
                )
            attempt_started = monotonic()
            try:
                routes = _build_route(
                    request,
                    candidate,
                    sample_offset,
                    ik_solver,
                    vamp,
                    ruckig,
                    attempts,
                    started,
                    dependencies.ik_solution_count,
                    dependencies.pregrasp_ik_solution_count,
                    dependencies.placement_sample_count,
                )
            except _RouteFailure as failure:
                last_failure = failure
                if not failure.attempt_recorded:
                    attempts.append(
                        PlanningAttempt(
                            candidate.candidate_id,
                            sample_offset,
                            failure.stage,
                            monotonic() - attempt_started,
                            failure.code,
                            failure.message,
                        )
                    )
                continue
            except BackendUnavailableError as exc:
                return PlanningResult.failed(
                    PlanningErrorCode.BACKEND_UNAVAILABLE,
                    str(exc),
                    tuple(attempts),
                    _metadata(request, vamp.version, ik_solver.version, ruckig.version),
                )
            except BackendOperationError as exc:
                return PlanningResult.failed(
                    PlanningErrorCode.BACKEND_ERROR,
                    f"Planning backend error: {exc}",
                    tuple(attempts),
                    _metadata(request, vamp.version, ik_solver.version, ruckig.version),
                )
            except (RuntimeError, TypeError, ValueError, OSError) as exc:
                return PlanningResult.failed(
                    PlanningErrorCode.BACKEND_ERROR,
                    f"Planning backend error: {exc}",
                    tuple(attempts),
                    _metadata(request, vamp.version, ik_solver.version, ruckig.version),
                )
            candidate_routes.extend(
                (search_attempt, sample_offset, route) for route in routes
            )
        if not candidate_routes:
            continue
        for selected_search_attempt, selected_sample_offset, route in sorted(
            candidate_routes,
            key=lambda item: _route_selection_key(item[2], request, item[0]),
        ):
            parameterization_started = monotonic()
            try:
                trajectory = ruckig.parameterize(route.waypoints, route.events)
            except BackendUnavailableError as exc:
                return PlanningResult.failed(
                    PlanningErrorCode.BACKEND_UNAVAILABLE,
                    str(exc),
                    tuple(attempts),
                    _metadata(request, vamp.version, ik_solver.version, ruckig.version),
                )
            except RuckigRouteError as exc:
                last_failure = _RouteFailure(
                    PlanningErrorCode.RUCKIG_FAILED,
                    exc.stage,
                    str(exc),
                )
                attempts.append(
                    PlanningAttempt(
                        candidate.candidate_id,
                        selected_sample_offset,
                        exc.stage,
                        monotonic() - parameterization_started,
                        PlanningErrorCode.RUCKIG_FAILED,
                        str(exc),
                    )
                )
                continue
            except BackendOperationError as exc:
                return PlanningResult.failed(
                    PlanningErrorCode.BACKEND_ERROR,
                    f"Ruckig backend error: {exc}",
                    tuple(attempts),
                    _metadata(request, vamp.version, ik_solver.version, ruckig.version),
                )
            except (RuntimeError, TypeError, ValueError, OSError) as exc:
                return PlanningResult.failed(
                    PlanningErrorCode.BACKEND_ERROR,
                    f"Ruckig backend error: {exc}",
                    tuple(attempts),
                    _metadata(request, vamp.version, ik_solver.version, ruckig.version),
                )

            try:
                trajectory = _retime_payload_motion(
                    request,
                    route,
                    trajectory,
                    vamp,
                    ruckig,
                    started,
                )
            except _PayloadMotionSamplingError as exc:
                failure = _RouteFailure(
                    PlanningErrorCode.BUDGET_EXHAUSTED,
                    exc.stage,
                    exc.message,
                )
                attempts.append(
                    PlanningAttempt(
                        candidate.candidate_id,
                        selected_sample_offset,
                        exc.stage,
                        monotonic() - parameterization_started,
                        failure.code,
                        failure.message,
                    )
                )
                return PlanningResult.failed(
                    failure.code,
                    failure.message,
                    tuple(attempts),
                    _metadata(request, vamp.version, ik_solver.version, ruckig.version),
                )
            except _RouteFailure as failure:
                last_failure = failure
                attempts.append(
                    PlanningAttempt(
                        candidate.candidate_id,
                        selected_sample_offset,
                        failure.stage,
                        monotonic() - parameterization_started,
                        failure.code,
                        failure.message,
                    )
                )
                if failure.code is PlanningErrorCode.BUDGET_EXHAUSTED:
                    return PlanningResult.failed(
                        failure.code,
                        failure.message,
                        tuple(attempts),
                        _metadata(request, vamp.version, ik_solver.version, ruckig.version),
                    )
                continue
            except BackendUnavailableError as exc:
                return PlanningResult.failed(
                    PlanningErrorCode.BACKEND_UNAVAILABLE,
                    str(exc),
                    tuple(attempts),
                    _metadata(request, vamp.version, ik_solver.version, ruckig.version),
                )
            except _PayloadMotionLimitError as exc:
                last_failure = _RouteFailure(
                    PlanningErrorCode.RUCKIG_FAILED,
                    exc.stage,
                    exc.message,
                )
                attempts.append(
                    PlanningAttempt(
                        candidate.candidate_id,
                        selected_sample_offset,
                        exc.stage,
                        monotonic() - parameterization_started,
                        PlanningErrorCode.RUCKIG_FAILED,
                        exc.message,
                    )
                )
                continue
            except RuckigRouteError as exc:
                last_failure = _RouteFailure(
                    PlanningErrorCode.RUCKIG_FAILED,
                    exc.stage,
                    str(exc),
                )
                attempts.append(
                    PlanningAttempt(
                        candidate.candidate_id,
                        selected_sample_offset,
                        exc.stage,
                        monotonic() - parameterization_started,
                        PlanningErrorCode.RUCKIG_FAILED,
                        str(exc),
                    )
                )
                continue
            except BackendOperationError as exc:
                return PlanningResult.failed(
                    PlanningErrorCode.BACKEND_ERROR,
                    f"payload motion backend error: {exc}",
                    tuple(attempts),
                    _metadata(request, vamp.version, ik_solver.version, ruckig.version),
                )
            except (RuntimeError, TypeError, ValueError, OSError) as exc:
                return PlanningResult.failed(
                    PlanningErrorCode.BACKEND_ERROR,
                    f"payload motion backend error: {exc}",
                    tuple(attempts),
                    _metadata(request, vamp.version, ik_solver.version, ruckig.version),
                )
            validation_started = monotonic()
            try:
                _validate_time_parameterized_trajectory(
                    request,
                    candidate,
                    trajectory,
                    vamp,
                    started,
                )
            except _RouteFailure as failure:
                last_failure = failure
                attempts.append(
                    PlanningAttempt(
                        candidate.candidate_id,
                        selected_sample_offset,
                        failure.stage,
                        monotonic() - validation_started,
                        failure.code,
                        failure.message,
                    )
                )
                if failure.code is PlanningErrorCode.BUDGET_EXHAUSTED:
                    return PlanningResult.failed(
                        failure.code,
                        failure.message,
                        tuple(attempts),
                        _metadata(request, vamp.version, ik_solver.version, ruckig.version),
                    )
                continue
            except BackendUnavailableError as exc:
                return PlanningResult.failed(
                    PlanningErrorCode.BACKEND_UNAVAILABLE,
                    str(exc),
                    tuple(attempts),
                    _metadata(request, vamp.version, ik_solver.version, ruckig.version),
                )
            except BackendOperationError as exc:
                return PlanningResult.failed(
                    PlanningErrorCode.BACKEND_ERROR,
                    f"final trajectory validation failed: {exc}",
                    tuple(attempts),
                    _metadata(request, vamp.version, ik_solver.version, ruckig.version),
                )
            metadata = _metadata(
                request, vamp.version, ik_solver.version, ruckig.version
            )
            metadata["selected_candidate_id"] = candidate.candidate_id
            metadata["selected_search_attempt"] = selected_search_attempt
            metadata["selected_sample_offset"] = selected_sample_offset
            return PlanningResult.succeeded(
                trajectory,
                route.T_world_object_goal,
                tuple(attempts),
                metadata,
            )
    if len(request.candidates) > len(candidates):
        code = PlanningErrorCode.BUDGET_EXHAUSTED
        message = "Planning candidate budget was exhausted before a complete route was found"
    elif last_failure is None:
        code = PlanningErrorCode.IK_CANDIDATES_EXHAUSTED
        message = "Planning had no candidate attempts"
    else:
        code = last_failure.code
        message = last_failure.message
    return PlanningResult.failed(
        code,
        message,
        tuple(attempts),
        _metadata(request, vamp.version, ik_solver.version, ruckig.version),
    )


def _approach_contact_transition_m(request: PlanningRequest) -> float:
    object_value = request.object_config.approach_contact_transition_m
    return (
        object_value
        if object_value is not None
        else request.planning_policy.approach_contact_transition_m
    )


def _order_candidates_by_pregrasp_ik_distance(
    request: PlanningRequest,
    candidates: tuple[GraspCandidate, ...],
    ik_solver: IKSolver,
    vamp: VampBackend,
    maximum_solutions: int,
) -> tuple[GraspCandidate, ...]:
    """Prefer a nearby valid symmetry candidate without reordering templates."""
    if len(candidates) < 2:
        return candidates
    template_order: dict[str, int] = {}
    for candidate in candidates:
        template_order.setdefault(candidate.template_id, len(template_order))
    target_geometries = tuple(
        item
        for item in request.scene.geometries
        if item.object_id == request.target_object_id
    )
    environment = vamp.environment(
        request.scene,
        T_world_base=request.T_world_base,
        excluded_object_ids=(request.target_object_id,),
    )
    initial = np.asarray(request.initial_state.positions_rad, dtype=np.float64)
    scored = []
    for index, candidate in enumerate(candidates):
        pregrasp_pose = compose(
            inverse(request.T_world_base),
            reframe(
                candidate.T_world_pregrasp,
                child_frame=request.robot_model.tcp_frame,
            ),
        )
        distance = float("inf")
        try:
            solutions = _solve_ik(
                ik_solver,
                request,
                pregrasp_pose,
                request.initial_state,
                TrajectoryStage.PREGRASP,
                maximum_solutions,
                solution_validator=lambda state, candidate=candidate: _pregrasp_solution_valid(
                    state,
                    request,
                    vamp,
                    environment,
                    target_geometries,
                    candidate.grasp_command.pregrasp_opening_m,
                ),
            )
        except _RouteFailure as failure:
            if failure.code not in (
                PlanningErrorCode.IK_CANDIDATES_EXHAUSTED,
                PlanningErrorCode.CONTINUOUS_IK_FAILED,
                PlanningErrorCode.STAGE_COLLISION_VIOLATION,
            ):
                raise
        else:
            distance = min(
                float(
                    np.linalg.norm(
                        np.asarray(
                            request.robot_model.unwrap_positions(
                                state.positions_rad,
                                request.initial_state.positions_rad,
                            ),
                            dtype=np.float64,
                        )
                        - initial
                    )
                )
                for state in solutions
            )
        scored.append(
            (
                -candidate.quality_score,
                template_order[candidate.template_id],
                distance,
                index,
                candidate,
            )
        )
    return tuple(item[-1] for item in sorted(scored, key=lambda item: item[:-1]))


def _build_route(
    request: PlanningRequest,
    candidate: GraspCandidate,
    sample_offset: int,
    ik_solver: IKSolver,
    vamp: VampBackend,
    ruckig: RuckigTimeParameterizer,
    attempts: list[PlanningAttempt],
    total_started: float,
    ik_solution_count: int,
    pregrasp_ik_solution_count: int,
    placement_sample_count: int,
) -> tuple[_Route, ...]:
    target_geometries = tuple(
        item for item in request.scene.geometries if item.object_id == request.target_object_id
    )
    environment_geometries = tuple(
        item
        for item in request.scene.geometries
        if item.object_id != request.target_object_id
    )
    support_geometries = tuple(
        item
        for item in environment_geometries
        if item.collision_group == request.source_support_group
    )
    non_support_environment_geometries = tuple(
        item
        for item in environment_geometries
        if item.collision_group != request.source_support_group
    )
    channels = {
        request.end_effector_config.actuator_position_channel: candidate.grasp_command.pregrasp_opening_m,
        request.end_effector_config.actuator_force_limit_channel: candidate.grasp_command.grip_force_limit_n,
    }
    waypoints: list[GeometricWaypoint] = []
    initial = request.initial_state.positions_rad
    free_environment = vamp.environment(
        request.scene,
        T_world_base=request.T_world_base,
        excluded_object_ids=(request.target_object_id,),
    )
    pregrasp_pose = compose(
        inverse(request.T_world_base),
        reframe(candidate.T_world_pregrasp, child_frame=request.robot_model.tcp_frame),
    )
    grasp_pose = compose(
        inverse(request.T_world_base),
        reframe(candidate.T_world_tcp, child_frame=request.robot_model.tcp_frame),
    )
    corridor = request.vertical_transit_corridor
    vertical_ingress_poses: tuple[RigidTransform, ...] = ()
    staging_poses: tuple[RigidTransform, ...] = ()
    corridor_upper: RigidTransform | None = None
    corridor_lower: RigidTransform | None = None
    forward_vertical_ingress = False
    if corridor is not None:
        corridor_upper_world = _vertical_corridor_pose(
            candidate.T_world_tcp,
            corridor,
            corridor.ingress_top_offset_m,
            request.robot_model.tcp_frame,
            request.object_config.approach_motion_strategy,
        )
        corridor_lower_world = _vertical_corridor_pose(
            candidate.T_world_tcp,
            corridor,
            corridor.ingress_bottom_offset_m,
            request.robot_model.tcp_frame,
            request.object_config.approach_motion_strategy,
        )
        corridor_upper = compose(inverse(request.T_world_base), corridor_upper_world)
        corridor_lower = compose(inverse(request.T_world_base), corridor_lower_world)
        vertical_ingress_poses = _interpolate_poses(
            corridor_upper,
            corridor_lower,
            corridor.ingress_waypoint_step_m,
            request.planning_policy.cartesian_rotation_step_rad,
        )[1:]
        staging_poses = _interpolate_poses(
            corridor_lower,
            pregrasp_pose,
            corridor.ingress_waypoint_step_m,
            request.planning_policy.cartesian_rotation_step_rad,
        )[1:]
        forward_vertical_ingress = (
            request.object_config.payload_motion_strategy
            is LocalMotionStrategy.ADAPTIVE_PORTAL
        )
    pregrasp_solutions: tuple[RobotState, ...] = ()
    if not forward_vertical_ingress:
        pregrasp_solutions = _solve_ik(
            ik_solver,
            request,
            pregrasp_pose,
            request.initial_state,
            TrajectoryStage.PREGRASP,
            pregrasp_ik_solution_count,
            solution_validator=lambda state, environment=free_environment: _pregrasp_solution_valid(
                state,
                request,
                vamp,
                environment,
                target_geometries,
                candidate.grasp_command.pregrasp_opening_m,
            ),
        )
    interpolated_approach = _interpolate_poses(
        pregrasp_pose,
        grasp_pose,
        request.planning_policy.cartesian_translation_step_m,
        request.planning_policy.cartesian_rotation_step_rad,
        required_fractions=_approach_transition_fractions(
            pregrasp_pose,
            grasp_pose,
            _approach_contact_transition_m(request),
        ),
    )
    approach_poses = interpolated_approach[1:]
    approach_contact_allowances = _approach_contact_allowances(
        interpolated_approach,
        _approach_contact_transition_m(request),
        frozenset(candidate.expected_contacts),
    )

    pregrasp_path = None
    pregrasp_elapsed = 0.0
    approach_states = None
    approach_elapsed = 0.0
    approach_failure: _RouteFailure | None = None
    detour_rejected = False
    solution_queue = list(pregrasp_solutions)
    if forward_vertical_ingress:
        assert corridor_upper is not None
        stage_started = monotonic()
        try:
            portal_path = _joint_space_portal_route(
                request,
                ik_solver,
                vamp,
                initial,
                corridor_upper,
                free_environment,
                sample_offset,
                TrajectoryStage.PREGRASP,
                total_started,
                ik_solution_count,
                cartesian_continuation_poses=(
                    *vertical_ingress_poses,
                    *staging_poses,
                ),
            )
            candidate_pregrasp_path = portal_path
            if _path_has_forbidden_target_contact(
                candidate_pregrasp_path,
                target_geometries,
                candidate.grasp_command.pregrasp_opening_m,
                vamp,
                request.T_world_base,
            ):
                raise _RouteFailure(
                    PlanningErrorCode.STAGE_COLLISION_VIOLATION,
                    TrajectoryStage.PREGRASP,
                    "forward vertical PREGRASP has forbidden target contact",
                )
            approach_seed = RobotState(
                request.robot_model.joint_names,
                candidate_pregrasp_path[-1],
            )
            if (
                request.object_config.approach_motion_strategy
                is LocalMotionStrategy.STAGED_PORTAL
            ):
                selected_approach = _staged_approach_route(
                    request,
                    candidate,
                    ik_solver,
                    vamp,
                    tuple(approach_poses),
                    approach_seed,
                    target_geometries,
                    environment_geometries,
                    approach_contact_allowances,
                    sample_offset,
                    total_started,
                    ik_solution_count,
                )
            else:
                selected_approach = _continuous_ik(
                    request,
                    candidate,
                    ik_solver,
                    vamp,
                    tuple(approach_poses),
                    approach_seed,
                    target_geometries,
                    environment_geometries,
                    approach_contact_allowances,
                    TrajectoryStage.APPROACH,
                    total_started,
                    ik_solution_count,
                )
            pregrasp_path = candidate_pregrasp_path
            approach_states = selected_approach
            pregrasp_transit_strategy = "forward_vertical"
        except _RouteFailure as failure:
            approach_failure = failure
        pregrasp_elapsed += monotonic() - stage_started
        _check_stage_budget(
            request,
            pregrasp_elapsed,
            total_started,
            TrajectoryStage.PREGRASP,
        )
        solution_queue = []
    for solution_index, solution in enumerate(solution_queue):
        if not _has_joint_margin(solution, request):
            continue
        stage_started = monotonic()
        backward_transit_path: tuple[tuple[float, ...], ...] = ()
        route_goal = solution
        if vertical_ingress_poses:
            assert corridor_upper is not None
            try:
                backward_staging_poses = tuple(reversed(staging_poses[:-1]))
                backward_vertical_poses = (
                    *reversed(vertical_ingress_poses[:-1]),
                    corridor_upper,
                )
                backward_transit_poses = (
                    *backward_staging_poses,
                    *backward_vertical_poses,
                )
                backward_transit_path = _cartesian_continuation_route(
                    request,
                    ik_solver,
                    vamp,
                    backward_transit_poses,
                    solution.positions_rad,
                    free_environment,
                    TrajectoryStage.PREGRASP,
                    total_started,
                    ik_solution_count,
                )
            except _RouteFailure as failure:
                approach_failure = failure
                continue
            route_goal = RobotState(
                request.robot_model.joint_names,
                backward_transit_path[-1],
            )
        try:
            if corridor is None:
                candidate_base_path = vamp.plan(
                    initial,
                    route_goal.positions_rad,
                    free_environment,
                    sample_offset=sample_offset,
                ).path
            else:
                assert corridor_upper is not None
                candidate_base_path, pregrasp_transit_strategy = (
                    _vertical_transit_global_route(
                        request,
                        ik_solver,
                        vamp,
                        initial,
                        corridor_upper,
                        route_goal.positions_rad,
                        free_environment,
                        sample_offset,
                        total_started,
                        ik_solution_count,
                    )
                )
        except BackendOperationError:
            pregrasp_elapsed += monotonic() - stage_started
            _check_stage_budget(
                request,
                pregrasp_elapsed,
                total_started,
                TrajectoryStage.PREGRASP,
            )
            continue
        pregrasp_elapsed += monotonic() - stage_started
        _check_stage_budget(request, pregrasp_elapsed, total_started, TrajectoryStage.PREGRASP)
        candidate_base_path = _farthest_collision_valid_shortcut(
            _compact_joint_path(candidate_base_path),
            request.robot_model,
            vamp,
            free_environment,
            request.planning_policy.trajectory_validation_joint_step_rad,
        )
        candidate_pregrasp_path = candidate_base_path
        if backward_transit_path:
            candidate_pregrasp_path = _append_unwrapped_positions(
                request.robot_model,
                candidate_base_path,
                tuple(reversed(backward_transit_path[:-1])),
            )
        candidate_pregrasp_path = _compact_joint_path(candidate_pregrasp_path)
        if _path_has_forbidden_target_contact(
            candidate_pregrasp_path,
            target_geometries,
            candidate.grasp_command.pregrasp_opening_m,
            vamp,
            request.T_world_base,
        ):
            continue
        approach_started = monotonic()
        try:
            approach_seed = RobotState(
                request.robot_model.joint_names,
                candidate_pregrasp_path[-1],
            )
            if (
                request.object_config.approach_motion_strategy
                is LocalMotionStrategy.STAGED_PORTAL
            ):
                selected_approach = _staged_approach_route(
                    request,
                    candidate,
                    ik_solver,
                    vamp,
                    tuple(approach_poses),
                    approach_seed,
                    target_geometries,
                    environment_geometries,
                    approach_contact_allowances,
                    sample_offset,
                    total_started,
                    ik_solution_count,
                )
            else:
                selected_approach = _continuous_ik(
                    request,
                    candidate,
                    ik_solver,
                    vamp,
                    tuple(approach_poses),
                    approach_seed,
                    target_geometries,
                    environment_geometries,
                    approach_contact_allowances,
                    TrajectoryStage.APPROACH,
                    total_started,
                    ik_solution_count,
                )
        except _RouteFailure as failure:
            approach_elapsed += monotonic() - approach_started
            _check_stage_budget(
                request,
                approach_elapsed,
                total_started,
                TrajectoryStage.APPROACH,
            )
            if failure.code in (
                PlanningErrorCode.IK_CANDIDATES_EXHAUSTED,
                PlanningErrorCode.CONTINUOUS_IK_FAILED,
                PlanningErrorCode.STAGE_COLLISION_VIOLATION,
            ):
                approach_failure = failure
                continue
            raise
        approach_elapsed += monotonic() - approach_started
        _check_stage_budget(
            request,
            approach_elapsed,
            total_started,
            TrajectoryStage.APPROACH,
        )
        detour_ratio = _joint_path_detour_ratio(candidate_pregrasp_path)
        if (
            detour_ratio
            <= request.planning_policy.pregrasp_joint_path_detour_ratio_limit
        ):
            pregrasp_path = candidate_pregrasp_path
            approach_states = selected_approach
            break
        detour_rejected = True
    if pregrasp_path is None:
        if detour_rejected:
            raise _RouteFailure(
                PlanningErrorCode.VAMP_SEARCH_FAILED,
                TrajectoryStage.PREGRASP,
                "collision-valid PREGRASP routes exceeded the configured "
                "joint-path detour ratio limit",
            )
        if approach_failure is not None:
            raise approach_failure
        raise _RouteFailure(
            PlanningErrorCode.VAMP_SEARCH_FAILED,
            TrajectoryStage.PREGRASP,
            "VAMP found no collision-valid PREGRASP route for this candidate and sample offset",
        )
    attempts.append(
        PlanningAttempt(
            candidate.candidate_id,
            sample_offset,
            TrajectoryStage.PREGRASP,
            pregrasp_elapsed,
            None,
            (
                f"vertical-transit {pregrasp_transit_strategy} PREGRASP succeeded"
                if corridor is not None
                else "VAMP PREGRASP succeeded"
            ),
        )
    )
    waypoints.extend(
        GeometricWaypoint(
            TrajectoryStage.PREGRASP,
            point,
            dict(channels),
        )
        for point in pregrasp_path
    )
    del free_environment
    assert approach_states is not None
    approach_states = _compact_robot_state_path(
        pregrasp_path[-1],
        approach_states,
    )
    if not approach_states:
        raise _RouteFailure(
            PlanningErrorCode.CONTINUOUS_IK_FAILED,
            TrajectoryStage.APPROACH,
            "APPROACH contains no motion after removing repeated boundary states",
        )
    for index, state in enumerate(approach_states):
        waypoints.append(
            GeometricWaypoint(
                TrajectoryStage.APPROACH,
                state.positions_rad,
                dict(channels),
                stop=index == len(approach_states) - 1,
            )
        )
    grasp_state = approach_states[-1]
    contacts = vamp.semantic_target_contacts(
        grasp_state.positions_rad,
        target_geometries,
        candidate.grasp_command.target_opening_m,
        request.T_world_base,
    )
    if contacts - set(candidate.expected_contacts):
        raise _RouteFailure(
            PlanningErrorCode.STAGE_COLLISION_VIOLATION,
            TrajectoryStage.GRASP,
            f"GRASP has forbidden target contacts: {sorted(contacts)!r}",
        )
    if not set(candidate.expected_contacts).issubset(contacts):
        raise _RouteFailure(
            PlanningErrorCode.STAGE_COLLISION_VIOLATION,
            TrajectoryStage.GRASP,
            f"GRASP contacts {sorted(contacts)!r} do not include every "
            f"candidate-declared contact {sorted(candidate.expected_contacts)!r}",
        )
    non_support_contacts = vamp.semantic_target_contacts(
        grasp_state.positions_rad,
        non_support_environment_geometries,
        candidate.grasp_command.target_opening_m,
        request.T_world_base,
    )
    support_contacts = vamp.semantic_target_contacts(
        grasp_state.positions_rad,
        support_geometries,
        candidate.grasp_command.target_opening_m,
        request.T_world_base,
    )
    forbidden_support_contacts = support_contacts - set(
        request.end_effector_config.contact_groups
    )
    if non_support_contacts or forbidden_support_contacts:
        raise _RouteFailure(
            PlanningErrorCode.STAGE_COLLISION_VIOLATION,
            TrajectoryStage.GRASP,
            "GRASP has forbidden robot/environment contacts: "
            f"non_support={sorted(non_support_contacts)!r}, "
            f"source_support={sorted(forbidden_support_contacts)!r}",
        )
    channels = {
        request.end_effector_config.actuator_position_channel: candidate.grasp_command.target_opening_m,
        request.end_effector_config.actuator_force_limit_channel: candidate.grasp_command.grip_force_limit_n,
    }
    closing_distance = (
        candidate.grasp_command.pregrasp_opening_m
        - candidate.grasp_command.target_opening_m
    )
    if closing_distance <= 0.0:
        raise _RouteFailure(
            PlanningErrorCode.INVALID_REQUEST,
            TrajectoryStage.GRASP,
            "GRASP requires pregrasp opening to exceed target opening",
        )
    closing_duration = closing_distance / candidate.grasp_command.closing_speed_m_s
    waypoints.append(
        GeometricWaypoint(
            TrajectoryStage.GRASP,
            grasp_state.positions_rad,
            dict(channels),
            stop=True,
            minimum_duration_s=closing_duration,
        )
    )
    if candidate.grasp_command.dwell_s > 0.0:
        waypoints.append(
            GeometricWaypoint(
                TrajectoryStage.GRASP,
                grasp_state.positions_rad,
                dict(channels),
                stop=True,
                minimum_duration_s=candidate.grasp_command.dwell_s,
            )
        )
    grip_index = len(waypoints) - 1

    world_lift_pose = _translate_world_z(
        reframe(candidate.T_world_tcp, child_frame=request.robot_model.tcp_frame),
        corridor.minimum_vertical_lift_m
        if corridor is not None
        else request.planning_policy.lift_distance_m,
    )
    lift_pose = compose(inverse(request.T_world_base), world_lift_pose)
    lift_poses = _interpolate_poses(
        grasp_pose,
        lift_pose,
        request.planning_policy.cartesian_translation_step_m,
        request.planning_policy.cartesian_rotation_step_rad,
    )[1:]
    if (
        request.object_config.payload_motion_strategy
        is LocalMotionStrategy.STAGED_PORTAL
    ):
        lift_states = _staged_lift_route(
            request,
            candidate,
            ik_solver,
            vamp,
            tuple(lift_poses),
            grasp_state,
            (),
            environment_geometries,
            (),
            TrajectoryStage.LIFT,
            sample_offset,
            total_started,
            ik_solution_count,
        )
    elif (
        request.object_config.payload_motion_strategy
        is LocalMotionStrategy.ADAPTIVE_PORTAL
    ):
        departure_waypoint_step_m = (
            corridor.egress_waypoint_step_m
            if corridor is not None
            else request.planning_policy.cartesian_translation_step_m
        )
        world_reverse_endpoint = reframe(
            candidate.T_world_pregrasp,
            child_frame=request.robot_model.tcp_frame,
        )
        world_departure_endpoint = _with_translation(
            world_reverse_endpoint,
            (
                float(world_reverse_endpoint.matrix[0][3]),
                float(world_reverse_endpoint.matrix[1][3]),
                float(candidate.T_world_tcp.matrix[2][3])
                + (
                    corridor.ingress_bottom_offset_m
                    if corridor is not None
                    else request.planning_policy.lift_distance_m
                ),
            ),
        )
        departure_endpoint = compose(
            inverse(request.T_world_base),
            world_departure_endpoint,
        )
        clearance_pose = compose(
            inverse(request.T_world_base),
            _translate_world_z(
                reframe(
                    candidate.T_world_tcp,
                    child_frame=request.robot_model.tcp_frame,
                ),
                request.object_config.payload_initial_vertical_clearance_m,
            ),
        )
        initial_vertical_poses = _interpolate_poses(
            grasp_pose,
            clearance_pose,
            departure_waypoint_step_m,
            request.planning_policy.cartesian_rotation_step_rad,
        )
        reverse_approach_poses = _interpolate_poses(
            clearance_pose,
            pregrasp_pose,
            departure_waypoint_step_m,
            request.planning_policy.cartesian_rotation_step_rad,
        )
        vertical_departure_poses = _interpolate_poses(
            pregrasp_pose,
            departure_endpoint,
            departure_waypoint_step_m,
            request.planning_policy.cartesian_rotation_step_rad,
        )
        lift_transition_poses = (
            *initial_vertical_poses[1:],
            *reverse_approach_poses[1:],
            *vertical_departure_poses[1:],
        )
        lift_states = _staged_lift_route(
            request,
            candidate,
            ik_solver,
            vamp,
            lift_transition_poses,
            grasp_state,
            (),
            environment_geometries,
            (),
            TrajectoryStage.LIFT,
            sample_offset,
            total_started,
            ik_solution_count,
        )
        world_lift_pose = compose(
            request.T_world_base,
            vamp.tcp_pose(lift_states[-1].positions_rad),
        )
    else:
        lift_states = _continuous_ik(
            request,
            candidate,
            ik_solver,
            vamp,
            tuple(lift_poses),
            grasp_state,
            (),
            environment_geometries,
            (),
            TrajectoryStage.LIFT,
            total_started,
            ik_solution_count,
        )
    waypoints.extend(
        GeometricWaypoint(
            TrajectoryStage.LIFT,
            state.positions_rad,
            dict(channels),
        )
        for state in lift_states
    )

    strict_loaded_environment = vamp.environment(
        request.scene,
        T_world_base=request.T_world_base,
        excluded_object_ids=(request.target_object_id,),
        T_tcp_object=candidate.T_tcp_object,
        attached_geometry=request.object_config.attached_geometry,
    )
    regional_preplace_pose: RigidTransform | None = None
    if request.placement_goal.kind is not PlacementGoalKind.EXACT:
        regional_preplace_pose = _translate_world_z(
            _resolve_placement_orientation(
                request.placement_goal,
                request.placement_goal.T_world_tcp,
                candidate.T_world_tcp,
            ),
            request.planning_policy.preplace_offset_m,
        )
    placement_failure: _RouteFailure | None = None
    for sampled_placement_pose in _sample_placements(
        request.placement_goal,
        placement_sample_count,
        reference_pose=regional_preplace_pose,
    ):
        world_placement_pose = _resolve_placement_orientation(
            request.placement_goal,
            sampled_placement_pose,
            candidate.T_world_tcp,
        )
        world_preplace_pose = (
            regional_preplace_pose
            if regional_preplace_pose is not None
            else _translate_world_z(
                world_placement_pose,
                request.planning_policy.preplace_offset_m,
            )
        )
        if corridor is not None:
            transit_orientation = _vertical_corridor_pose(
                candidate.T_world_tcp,
                corridor,
                corridor.egress_top_offset_m,
                request.robot_model.tcp_frame,
                request.object_config.payload_motion_strategy,
            )
            world_placement_pose = _with_orientation(
                world_placement_pose,
                transit_orientation,
            )
            world_preplace_pose = _with_orientation(
                world_preplace_pose,
                transit_orientation,
            )
        preplace_pose = compose(inverse(request.T_world_base), world_preplace_pose)
        placement_pose = compose(inverse(request.T_world_base), world_placement_pose)
        try:
            if (
                request.placement_goal.orientation_constraint
                is PlacementOrientationConstraint.FREE
                and corridor is not None
            ):
                route_choices = _vertical_transit_free_route(
                    request,
                    candidate,
                    ik_solver,
                    vamp,
                    ruckig,
                    grasp_state,
                    grasp_pose,
                    lift_states,
                    world_lift_pose,
                    preplace_pose,
                    placement_pose,
                    world_preplace_pose,
                    world_placement_pose,
                    strict_loaded_environment,
                    tuple(waypoints),
                    channels,
                    sample_offset,
                    attempts,
                    total_started,
                    ik_solution_count,
                )
            elif (
                request.placement_goal.orientation_constraint
                is PlacementOrientationConstraint.FREE
            ):
                lift_endpoints = _orientation_free_lift_endpoints(
                    request,
                    candidate,
                    ik_solver,
                    vamp,
                    world_lift_pose,
                    lift_states[-1],
                    world_preplace_pose,
                    environment_geometries,
                    total_started,
                    ik_solution_count,
                )
                free_routes = []
                free_search_started = monotonic()
                for endpoint_index, extra_lift_states in enumerate(lift_endpoints):
                    egress_start = (
                        extra_lift_states[-1].positions_rad
                        if extra_lift_states
                        else lift_states[-1].positions_rad
                    )
                    try:
                        candidate_egress, _ = _orientation_preferred_vamp_stage(
                            request,
                            ik_solver,
                            vamp,
                            egress_start,
                            preplace_pose,
                            grasp_pose,
                            strict_loaded_environment,
                            candidate.candidate_id,
                            sample_offset,
                            TrajectoryStage.EGRESS,
                            attempts,
                            total_started,
                            ik_solution_count,
                        )
                        if _same_position(
                            world_preplace_pose,
                            world_placement_pose,
                        ):
                            split_path = _split_final_path_motion(
                                candidate_egress,
                                request.planning_policy.trajectory_validation_joint_step_rad,
                            )
                            if split_path is None:
                                failure = _RouteFailure(
                                    PlanningErrorCode.RUCKIG_FAILED,
                                    TrajectoryStage.PLACE,
                                    "zero-motion PLACE sample cannot form the required "
                                    "positive-duration trajectory segment",
                                    attempt_recorded=True,
                                )
                                attempts.append(
                                    PlanningAttempt(
                                        candidate.candidate_id,
                                        sample_offset,
                                        failure.stage,
                                        monotonic() - free_search_started,
                                        failure.code,
                                        failure.message,
                                    )
                                )
                                _check_stage_budget(
                                    request,
                                    monotonic() - free_search_started,
                                    total_started,
                                    TrajectoryStage.PLACE,
                                )
                                raise failure
                            candidate_egress, candidate_place = split_path
                        else:
                            candidate_place, _ = _orientation_preferred_vamp_stage(
                                request,
                                ik_solver,
                                vamp,
                                candidate_egress[-1],
                                placement_pose,
                                grasp_pose,
                                strict_loaded_environment,
                                candidate.candidate_id,
                                sample_offset,
                                TrajectoryStage.PLACE,
                                attempts,
                                total_started,
                                ik_solution_count,
                            )
                        if len(_compact_joint_path(candidate_egress)) == 1:
                            split_path = _split_final_path_motion(
                                candidate_place,
                                request.planning_policy.trajectory_validation_joint_step_rad,
                            )
                            if split_path is None:
                                raise _RouteFailure(
                                    PlanningErrorCode.RUCKIG_FAILED,
                                    TrajectoryStage.EGRESS,
                                    "zero-motion EGRESS cannot borrow motion from PLACE",
                                )
                            candidate_egress, candidate_place = split_path
                        if _path_has_forbidden_attachment_contact(
                            candidate_egress,
                            request,
                            candidate,
                            vamp,
                            environment_geometries,
                        ) or _path_has_forbidden_attachment_contact(
                            candidate_place,
                            request,
                            candidate,
                            vamp,
                            environment_geometries,
                        ):
                            continue
                    except _RouteFailure as failure:
                        placement_failure = failure
                        if failure.code is PlanningErrorCode.BUDGET_EXHAUSTED:
                            raise
                        continue
                    payload_path = (
                        grasp_state.positions_rad,
                        *(state.positions_rad for state in lift_states),
                        *(state.positions_rad for state in extra_lift_states),
                        *candidate_egress[1:],
                        *candidate_place[1:],
                    )
                    payload_metrics = _sampled_orientation_deviation(
                        payload_path,
                        grasp_pose,
                        vamp,
                        request,
                        TrajectoryStage.EGRESS,
                        free_search_started,
                        total_started,
                    )
                    try:
                        payload_duration_s = _payload_route_duration_s(
                            tuple(waypoints),
                            extra_lift_states,
                            candidate_egress,
                            candidate_place,
                            channels,
                            ruckig,
                        )
                    except RuckigRouteError as exc:
                        placement_failure = _RouteFailure(
                            PlanningErrorCode.RUCKIG_FAILED,
                            exc.stage,
                            str(exc),
                        )
                        attempts.append(
                            PlanningAttempt(
                                candidate.candidate_id,
                                sample_offset,
                                exc.stage,
                                monotonic() - free_search_started,
                                PlanningErrorCode.RUCKIG_FAILED,
                                str(exc),
                            )
                        )
                        continue
                    free_routes.append(
                        (
                            _payload_route_selection_key(
                                payload_duration_s,
                                payload_metrics.cumulative_rotation_rad,
                                payload_metrics.cumulative_translation_m,
                                payload_metrics.maximum_deviation_rad,
                                _joint_path_length(payload_path),
                                endpoint_index,
                            ),
                            payload_metrics,
                            payload_duration_s,
                            extra_lift_states,
                            candidate_egress,
                            candidate_place,
                        )
                    )
                    _check_stage_budget(
                        request,
                        monotonic() - free_search_started,
                        total_started,
                        TrajectoryStage.EGRESS,
                    )
                if not free_routes:
                    if placement_failure is not None:
                        raise placement_failure
                    raise _RouteFailure(
                        PlanningErrorCode.IK_CANDIDATES_EXHAUSTED,
                        TrajectoryStage.EGRESS,
                        "no orientation-free LIFT endpoint produced a valid route",
                    )
                route_choices = tuple(
                    (
                        payload_metrics,
                        payload_duration_s,
                        extra_lift_states,
                        candidate_egress,
                        candidate_place,
                    )
                    for (
                        _,
                        payload_metrics,
                        payload_duration_s,
                        extra_lift_states,
                        candidate_egress,
                        candidate_place,
                    ) in sorted(free_routes, key=lambda item: item[0])
                )
            else:
                egress_start = lift_states[-1].positions_rad
                preplace = _first_valid_ik(
                    request,
                    ik_solver,
                    vamp,
                    preplace_pose,
                    RobotState(request.robot_model.joint_names, egress_start),
                    strict_loaded_environment,
                    TrajectoryStage.EGRESS,
                    ik_solution_count,
                )
                place_state = _first_valid_ik(
                    request,
                    ik_solver,
                    vamp,
                    placement_pose,
                    preplace,
                    strict_loaded_environment,
                    TrajectoryStage.PLACE,
                    ik_solution_count,
                )
                egress_plan = _vamp_stage(
                    request,
                    vamp,
                    egress_start,
                    preplace.positions_rad,
                    strict_loaded_environment,
                    candidate.candidate_id,
                    sample_offset,
                    TrajectoryStage.EGRESS,
                    attempts,
                    total_started,
                )
                place_plan = _vamp_stage(
                    request,
                    vamp,
                    preplace.positions_rad,
                    place_state.positions_rad,
                    strict_loaded_environment,
                    candidate.candidate_id,
                    sample_offset,
                    TrajectoryStage.PLACE,
                    attempts,
                    total_started,
                )
                route_choices = (
                    (
                        _OrientationPathMetrics(0.0, 0.0, 0.0, 0),
                        0.0,
                        (),
                        egress_plan,
                        place_plan,
                    ),
                )
        except _RouteFailure as failure:
            placement_failure = failure
            continue
        released_channels = dict(channels)
        released_channels[request.end_effector_config.actuator_position_channel] = (
            request.end_effector_config.open_position_m
        )
        T_world_object = compose(
            world_placement_pose,
            reframe(
                candidate.T_tcp_object,
                parent_frame=world_placement_pose.child_frame,
                child_frame="object",
            ),
        )
        completed_routes = []
        for (
            payload_metrics,
            payload_duration_s,
            extra_lift_states,
            egress_plan,
            place_plan,
        ) in route_choices:
            if _path_has_forbidden_attachment_contact(
                egress_plan,
                request,
                candidate,
                vamp,
                environment_geometries,
            ):
                placement_failure = _RouteFailure(
                    PlanningErrorCode.STAGE_COLLISION_VIOLATION,
                    TrajectoryStage.EGRESS,
                    "attached object has a forbidden contact during EGRESS",
                )
                continue
            if _path_has_forbidden_attachment_contact(
                place_plan,
                request,
                candidate,
                vamp,
                environment_geometries,
            ):
                placement_failure = _RouteFailure(
                    PlanningErrorCode.STAGE_COLLISION_VIOLATION,
                    TrajectoryStage.PLACE,
                    "attached object has a forbidden contact during PLACE",
                )
                continue
            route_waypoints = list(waypoints)
            route_waypoints.extend(
                GeometricWaypoint(
                    TrajectoryStage.LIFT,
                    state.positions_rad,
                    dict(channels),
                )
                for state in extra_lift_states
            )
            route_waypoints.extend(
                GeometricWaypoint(
                    TrajectoryStage.EGRESS,
                    point,
                    dict(channels),
                )
                for point in egress_plan[1:]
            )
            route_waypoints.extend(
                GeometricWaypoint(
                    TrajectoryStage.PLACE,
                    point,
                    dict(channels),
                    stop=index == len(place_plan[1:]) - 1,
                )
                for index, point in enumerate(place_plan[1:])
            )
            release_index = len(route_waypoints) - 1
            completed_routes.append(
                _Route(
                    tuple(route_waypoints),
                    (
                        ("grip", TrajectoryStage.GRASP, grip_index, dict(channels)),
                        (
                            "release",
                            TrajectoryStage.PLACE,
                            release_index,
                            released_channels,
                        ),
                    ),
                    T_world_object,
                    payload_metrics.maximum_deviation_rad,
                    payload_metrics.cumulative_translation_m,
                    payload_metrics.cumulative_rotation_rad,
                    payload_duration_s,
                )
            )
        if completed_routes:
            return tuple(completed_routes)
    if placement_failure is not None:
        raise placement_failure
    raise _RouteFailure(
        PlanningErrorCode.IK_CANDIDATES_EXHAUSTED,
        TrajectoryStage.PLACE,
        "no deterministic placement sample was feasible",
    )


def _append_unwrapped_positions(
    robot_model: RobotModel,
    prefix: tuple[tuple[float, ...], ...],
    suffix: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    if not prefix:
        raise ValueError("continuous joint path prefix must not be empty")
    result = list(prefix)
    for positions in suffix:
        result.append(robot_model.unwrap_positions(positions, result[-1]))
    return tuple(result)


def _compact_joint_path(
    path: tuple[tuple[float, ...], ...],
    *,
    tolerance_rad: float = 1e-10,
) -> tuple[tuple[float, ...], ...]:
    compact: list[tuple[float, ...]] = []
    for point in path:
        if compact and max(
            abs(left - right) for left, right in zip(compact[-1], point)
        ) <= tolerance_rad:
            continue
        compact.append(point)
    return tuple(compact)


def _compact_robot_state_path(
    initial_positions_rad: tuple[float, ...],
    states: tuple[RobotState, ...],
    *,
    tolerance_rad: float = 1e-10,
) -> tuple[RobotState, ...]:
    """Remove repeated IK states, including a repeated preceding boundary."""
    compact: list[RobotState] = []
    previous = initial_positions_rad
    for state in states:
        if max(
            abs(left - right)
            for left, right in zip(previous, state.positions_rad, strict=True)
        ) <= tolerance_rad:
            continue
        compact.append(state)
        previous = state.positions_rad
    return tuple(compact)


def _farthest_collision_valid_shortcut(
    path: tuple[tuple[float, ...], ...],
    robot_model: RobotModel,
    vamp: VampBackend,
    environment,
    maximum_joint_step_rad: float,
) -> tuple[tuple[float, ...], ...]:
    """Keep the fewest deterministic edges valid at final-validation resolution."""
    if len(path) <= 1:
        return path
    result = [path[0]]
    source_index = 0
    while source_index < len(path) - 1:
        selected: tuple[int, tuple[float, ...]] | None = None
        for target_index in range(len(path) - 1, source_index, -1):
            aligned_target = robot_model.unwrap_positions(
                path[target_index],
                result[-1],
            )
            if _joint_segment_valid_at_resolution(
                result[-1],
                aligned_target,
                vamp,
                environment,
                maximum_joint_step_rad,
            ):
                selected = (target_index, aligned_target)
                break
        if selected is None:
            selected_index = source_index + 1
            selected = (
                selected_index,
                robot_model.unwrap_positions(path[selected_index], result[-1]),
            )
        selected_index, aligned_target = selected
        result.append(aligned_target)
        source_index = selected_index
    return tuple(result)


def _joint_segment_valid_at_resolution(
    start: tuple[float, ...],
    goal: tuple[float, ...],
    vamp: VampBackend,
    environment,
    maximum_joint_step_rad: float,
) -> bool:
    start_values = np.asarray(start, dtype=np.float64)
    goal_values = np.asarray(goal, dtype=np.float64)
    maximum_delta = float(np.max(np.abs(goal_values - start_values)))
    sample_count = max(1, int(ceil(maximum_delta / maximum_joint_step_rad)))
    previous = start
    for sample_index in range(1, sample_count + 1):
        fraction = sample_index / sample_count
        current = tuple(
            float(value)
            for value in start_values + fraction * (goal_values - start_values)
        )
        if not vamp.motion_valid(previous, current, environment):
            return False
        previous = current
    return True


def _cartesian_continuation_route(
    request: PlanningRequest,
    ik_solver: IKSolver,
    vamp: VampBackend,
    poses: tuple[RigidTransform, ...],
    start: tuple[float, ...],
    environment,
    stage: TrajectoryStage,
    total_started: float,
    ik_solution_count: int,
) -> tuple[tuple[float, ...], ...]:
    beam = (
        _CartesianRouteNode(
            RobotState(request.robot_model.joint_names, start),
            (start,),
            0.0,
            0.0,
            0.0,
            0,
        ),
    )
    started = monotonic()
    for index, pose in enumerate(poses):
        solutions = _request_ik_solutions(
            ik_solver,
            request,
            pose,
            tuple(node.state for node in beam),
            stage,
            ik_solution_count,
        )
        candidates = []
        generation = 0
        for node in beam:
            for solution in solutions:
                positions = request.robot_model.unwrap_positions(
                    solution.state.positions_rad,
                    node.state.positions_rad,
                )
                state = RobotState(request.robot_model.joint_names, positions)
                if not _has_joint_margin(
                    state, request
                ) or not vamp.configuration_valid(positions, environment):
                    continue
                if not vamp.motion_valid(
                    node.state.positions_rad, positions, environment
                ):
                    continue
                delta = np.asarray(positions, dtype=np.float64) - np.asarray(
                    node.state.positions_rad, dtype=np.float64
                )
                edge_maximum = float(np.max(np.abs(delta)))
                edge_squared = float(np.dot(delta, delta))
                edge_length = sqrt(edge_squared)
                candidates.append(
                    _CartesianRouteNode(
                        RobotState(request.robot_model.joint_names, positions),
                        (*node.path, positions),
                        max(node.maximum_joint_step_rad, edge_maximum),
                        node.cumulative_squared_joint_step_rad2 + edge_squared,
                        node.cumulative_joint_step_rad + edge_length,
                        generation,
                    )
                )
                generation += 1
        if not candidates:
            raise _RouteFailure(
                PlanningErrorCode.CONTINUOUS_IK_FAILED,
                stage,
                "no collision-valid continuous IK edge to Cartesian waypoint "
                f"{index + 1}/{len(poses)}",
            )
        candidates.sort(key=_cartesian_route_node_key)
        distinct = []
        seen = set()
        for node in candidates:
            if node.state.positions_rad in seen:
                continue
            seen.add(node.state.positions_rad)
            distinct.append(node)
            if len(distinct) >= ik_solution_count:
                break
        beam = tuple(distinct)
        _check_stage_budget(request, monotonic() - started, total_started, stage)
    return min(beam, key=_cartesian_route_node_key).path


def _cartesian_route_node_key(
    node: _CartesianRouteNode,
) -> tuple[float, float, float, int]:
    return (
        node.maximum_joint_step_rad,
        node.cumulative_squared_joint_step_rad2,
        node.cumulative_joint_step_rad,
        node.generation,
    )


def _vertical_transit_global_route(
    request: PlanningRequest,
    ik_solver: IKSolver,
    vamp: VampBackend,
    initial: tuple[float, ...],
    corridor_upper: RigidTransform,
    corridor_goal: tuple[float, ...],
    environment,
    sample_offset: int,
    total_started: float,
    ik_solution_count: int,
) -> tuple[tuple[tuple[float, ...], ...], str]:
    initial_pose = vamp.tcp_pose(initial)
    initial_matrix = np.asarray(initial_pose.matrix, dtype=np.float64)
    upper_matrix = np.asarray(corridor_upper.matrix, dtype=np.float64)
    clearance_z = max(initial_matrix[2, 3], upper_matrix[2, 3])

    clearance_matrix = initial_matrix.copy()
    clearance_matrix[2, 3] = clearance_z
    clearance_pose = RigidTransform(
        initial_pose.parent_frame,
        initial_pose.child_frame,
        tuple(tuple(float(value) for value in row) for row in clearance_matrix),
    )  # type: ignore[arg-type]

    portal_matrix = clearance_matrix.copy()
    portal_matrix[0, 3] = upper_matrix[0, 3]
    portal_matrix[1, 3] = upper_matrix[1, 3]
    portal_pose = RigidTransform(
        initial_pose.parent_frame,
        initial_pose.child_frame,
        tuple(tuple(float(value) for value in row) for row in portal_matrix),
    )  # type: ignore[arg-type]
    oriented_portal_pose = _with_orientation(portal_pose, corridor_upper)

    poses = (
        *_interpolate_poses(
            initial_pose,
            clearance_pose,
            request.vertical_transit_corridor.ingress_waypoint_step_m,  # type: ignore[union-attr]
            request.planning_policy.cartesian_rotation_step_rad,
        )[1:],
        *_interpolate_poses(
            clearance_pose,
            portal_pose,
            request.vertical_transit_corridor.ingress_waypoint_step_m,  # type: ignore[union-attr]
            request.planning_policy.cartesian_rotation_step_rad,
        )[1:],
        *_interpolate_poses(
            portal_pose,
            oriented_portal_pose,
            request.vertical_transit_corridor.ingress_waypoint_step_m,  # type: ignore[union-attr]
            request.planning_policy.cartesian_rotation_step_rad,
        )[1:],
        *_interpolate_poses(
            oriented_portal_pose,
            corridor_upper,
            request.vertical_transit_corridor.ingress_waypoint_step_m,  # type: ignore[union-attr]
            request.planning_policy.cartesian_rotation_step_rad,
        )[1:],
    )
    route_started = monotonic()
    route_candidates: list[
        tuple[
            tuple[float, float, float, float, int],
            str,
            tuple[tuple[float, ...], ...],
        ]
    ] = []
    failures: list[Exception] = []

    try:
        branch_path = _compact_joint_path(
            vamp.plan(
                initial,
                corridor_goal,
                environment,
                sample_offset=sample_offset,
            ).path
        )
        branch_metrics = _sampled_orientation_deviation(
            branch_path,
            corridor_upper,
            vamp,
            request,
            TrajectoryStage.PREGRASP,
            route_started,
            total_started,
        )
        route_candidates.append(
            (
                _pregrasp_path_quality_key(
                    branch_metrics,
                    _joint_path_length(branch_path),
                    0,
                ),
                "branch_preserving",
                branch_path,
            )
        )
    except BackendOperationError as failure:
        failures.append(failure)

    if poses:
        try:
            connector_prefix = _cartesian_continuation_route(
                request,
                ik_solver,
                vamp,
                poses,
                initial,
                environment,
                TrajectoryStage.PREGRASP,
                total_started,
                ik_solution_count,
            )
            connector_path = _compact_joint_path(
                _append_unwrapped_positions(
                    request.robot_model,
                    connector_prefix,
                    vamp.plan(
                        connector_prefix[-1],
                        corridor_goal,
                        environment,
                        sample_offset=sample_offset,
                    ).path[1:],
                )
            )
            connector_metrics = _sampled_orientation_deviation(
                connector_path,
                corridor_upper,
                vamp,
                request,
                TrajectoryStage.PREGRASP,
                route_started,
                total_started,
            )
            route_candidates.append(
                (
                    _pregrasp_path_quality_key(
                        connector_metrics,
                        _joint_path_length(connector_path),
                        1,
                    ),
                    "forward_connector",
                    connector_path,
                )
            )
        except (BackendOperationError, _RouteFailure) as failure:
            failures.append(failure)

    if not route_candidates:
        raise failures[-1]
    _, strategy, selected_path = min(route_candidates, key=lambda item: item[0])
    return selected_path, strategy


def _vertical_transit_free_route(
    request: PlanningRequest,
    candidate: GraspCandidate,
    ik_solver: IKSolver,
    vamp: VampBackend,
    ruckig: RuckigTimeParameterizer,
    grasp_state: RobotState,
    grasp_pose: RigidTransform,
    lift_states: tuple[RobotState, ...],
    world_lift_pose: RigidTransform,
    preplace_pose: RigidTransform,
    placement_pose: RigidTransform,
    world_preplace_pose: RigidTransform,
    world_placement_pose: RigidTransform,
    strict_loaded_environment,
    waypoints: tuple[GeometricWaypoint, ...],
    channels: dict[str, float],
    sample_offset: int,
    attempts: list[PlanningAttempt],
    total_started: float,
    ik_solution_count: int,
) -> tuple[
    tuple[
        _OrientationPathMetrics,
        float,
        tuple[RobotState, ...],
        tuple[tuple[float, ...], ...],
        tuple[tuple[float, ...], ...],
    ],
    ...,
]:
    corridor = request.vertical_transit_corridor
    assert corridor is not None
    route_started = monotonic()
    transition_world = _vertical_corridor_pose(
        candidate.T_world_tcp,
        corridor,
        corridor.egress_transition_offset_m,
        request.robot_model.tcp_frame,
        request.object_config.payload_motion_strategy,
    )
    upper_world = _vertical_corridor_pose(
        candidate.T_world_tcp,
        corridor,
        corridor.egress_top_offset_m,
        request.robot_model.tcp_frame,
        request.object_config.payload_motion_strategy,
    )
    retreat_poses: tuple[RigidTransform, ...] = ()
    diagonal_start = world_lift_pose
    if (
        request.object_config.payload_motion_strategy
        is LocalMotionStrategy.STAGED_PORTAL
    ):
        retreat_world = reframe(
            candidate.T_world_pregrasp,
            child_frame=request.robot_model.tcp_frame,
        )
        if (
            float(retreat_world.matrix[2][3])
            > float(world_lift_pose.matrix[2][3]) + 1e-9
        ):
            retreat_poses = _interpolate_poses(
                world_lift_pose,
                retreat_world,
                corridor.egress_waypoint_step_m,
                request.planning_policy.cartesian_rotation_step_rad,
            )[1:]
            diagonal_start = retreat_world
    corridor_world_poses = (
        *retreat_poses,
        *_interpolate_poses(
            diagonal_start,
            transition_world,
            corridor.egress_waypoint_step_m,
            request.planning_policy.cartesian_rotation_step_rad,
        )[1:],
        *_interpolate_poses(
            transition_world,
            upper_world,
            corridor.egress_waypoint_step_m,
            request.planning_policy.cartesian_rotation_step_rad,
        )[1:],
    )
    candidate_poses = tuple(
        compose(inverse(request.T_world_base), pose) for pose in corridor_world_poses
    )
    required_lift_world_z_m = (
        float(candidate.T_world_tcp.matrix[2][3])
        + corridor.minimum_vertical_lift_m
    )
    joint_space_portal = (
        request.object_config.payload_motion_strategy
        is LocalMotionStrategy.ADAPTIVE_PORTAL
    )
    extra_lift_states: tuple[RobotState, ...]

    def build_joint_space_portal() -> tuple[
        tuple[RobotState, ...],
        tuple[tuple[float, ...], ...],
        tuple[tuple[float, ...], ...],
    ]:
        transition_pose = compose(inverse(request.T_world_base), transition_world)
        upper_pose = compose(inverse(request.T_world_base), upper_world)
        candidate_path = _joint_space_portal_then_vertical_route(
            request,
            ik_solver,
            vamp,
            lift_states[-1].positions_rad,
            transition_pose,
            upper_pose,
            strict_loaded_environment,
            sample_offset,
            TrajectoryStage.EGRESS,
            total_started,
            ik_solution_count,
            path_validator=lambda path: _lift_departure_path_is_valid(
                path,
                request,
                candidate,
                vamp,
                request.scene.geometries,
            ),
        )
        extra_lift_states: tuple[RobotState, ...] = ()
        if float(world_lift_pose.matrix[2][3]) < required_lift_world_z_m - 1e-9:
            lift_boundary_world_z_m = required_lift_world_z_m
            lift_boundary_index = next(
                (
                    index
                    for index, positions in enumerate(candidate_path[1:], start=1)
                    if float(
                        compose(
                            request.T_world_base,
                            vamp.tcp_pose(positions),
                        ).matrix[2][3]
                    )
                    >= lift_boundary_world_z_m - 1e-9
                ),
                None,
            )
            if lift_boundary_index is None:
                raise _RouteFailure(
                    PlanningErrorCode.CONTINUOUS_IK_FAILED,
                    TrajectoryStage.LIFT,
                    "joint-space portal did not reach the required LIFT height",
                )
            extra_lift_states = tuple(
                RobotState(request.robot_model.joint_names, positions)
                for positions in candidate_path[1 : lift_boundary_index + 1]
            )
            candidate_path = candidate_path[lift_boundary_index:]
        tail_poses = _interpolate_poses(
            upper_pose,
            preplace_pose,
            corridor.egress_waypoint_step_m,
            request.planning_policy.cartesian_rotation_step_rad,
        )[1:]
        tail = _cartesian_continuation_route(
            request,
            ik_solver,
            vamp,
            tail_poses,
            candidate_path[-1],
            strict_loaded_environment,
            TrajectoryStage.EGRESS,
            total_started,
            ik_solution_count,
        )
        return extra_lift_states, candidate_path, tail

    if joint_space_portal:
        extra_lift_states, candidate_path, tail = build_joint_space_portal()
    else:
        candidate_path = _cartesian_continuation_route(
            request,
            ik_solver,
            vamp,
            candidate_poses,
            lift_states[-1].positions_rad,
            strict_loaded_environment,
            TrajectoryStage.EGRESS,
            total_started,
            ik_solution_count,
        )
        tail_poses = _interpolate_poses(
            candidate_poses[-1],
            preplace_pose,
            corridor.egress_waypoint_step_m,
            request.planning_policy.cartesian_rotation_step_rad,
        )[1:]
        tail = _cartesian_continuation_route(
            request,
            ik_solver,
            vamp,
            tail_poses,
            candidate_path[-1],
            strict_loaded_environment,
            TrajectoryStage.EGRESS,
            total_started,
            ik_solution_count,
        )
        extra_lift_states = ()
    candidate_egress = (*candidate_path, *tail[1:])
    attempts.append(
        PlanningAttempt(
            candidate.candidate_id,
            sample_offset,
            TrajectoryStage.EGRESS,
            monotonic() - route_started,
            None,
            (
                "EGRESS followed a collision-valid joint-space portal to the "
                "opening transition and Cartesian vertical exit"
                if joint_space_portal
                else "EGRESS followed the explicit diagonal vertical transit and "
                "Cartesian transfer"
            ),
        )
    )
    if _same_position(world_preplace_pose, world_placement_pose):
        split_path = _split_final_path_motion(
            candidate_egress,
            request.planning_policy.trajectory_validation_joint_step_rad,
        )
        if split_path is None:
            raise _RouteFailure(
                PlanningErrorCode.RUCKIG_FAILED,
                TrajectoryStage.PLACE,
                "vertical transit route cannot form a positive-duration PLACE segment",
            )
        candidate_egress, candidate_place = split_path
    else:
        candidate_place, _ = _orientation_preferred_vamp_stage(
            request,
            ik_solver,
            vamp,
            candidate_egress[-1],
            placement_pose,
            grasp_pose,
            strict_loaded_environment,
            candidate.candidate_id,
            sample_offset,
            TrajectoryStage.PLACE,
            attempts,
            total_started,
            ik_solution_count,
        )
    payload_path = (
        grasp_state.positions_rad,
        *(state.positions_rad for state in lift_states),
        *(state.positions_rad for state in extra_lift_states),
        *candidate_egress[1:],
        *candidate_place[1:],
    )
    started = monotonic()
    payload_metrics = _sampled_orientation_deviation(
        payload_path,
        grasp_pose,
        vamp,
        request,
        TrajectoryStage.EGRESS,
        started,
        total_started,
    )
    try:
        payload_duration_s = _payload_route_duration_s(
            waypoints,
            extra_lift_states,
            candidate_egress,
            candidate_place,
            channels,
            ruckig,
            stop_at_egress_waypoints=joint_space_portal,
        )
    except RuckigRouteError as exc:
        raise _RouteFailure(
            PlanningErrorCode.RUCKIG_FAILED,
            exc.stage,
            str(exc),
        ) from exc
    return (
        (
            payload_metrics,
            payload_duration_s,
            extra_lift_states,
            candidate_egress,
            candidate_place,
        ),
    )


def _joint_space_portal_route(
    request: PlanningRequest,
    ik_solver: IKSolver,
    vamp: VampBackend,
    start: tuple[float, ...],
    goal_pose: RigidTransform,
    environment,
    sample_offset: int,
    stage: TrajectoryStage,
    total_started: float,
    ik_solution_count: int,
    cartesian_continuation_poses: tuple[RigidTransform, ...] = (),
    path_validator: Callable[[tuple[tuple[float, ...], ...]], bool] | None = None,
) -> tuple[tuple[float, ...], ...]:
    started = monotonic()
    solutions = _solve_ik(
        ik_solver,
        request,
        goal_pose,
        RobotState(request.robot_model.joint_names, start),
        stage,
        ik_solution_count,
    )
    goals = tuple(
        (
            solution_index,
            request.robot_model.unwrap_positions(solution.positions_rad, start),
        )
        for solution_index, solution in enumerate(solutions)
        if _has_joint_margin(solution, request)
        and vamp.configuration_valid(solution.positions_rad, environment)
    )
    if not goals:
        raise _RouteFailure(
            PlanningErrorCode.IK_CANDIDATES_EXHAUSTED,
            stage,
            "joint-space portal route has no collision-valid goal IK",
        )
    route_choices = []
    continuation_failure: _RouteFailure | None = None
    remaining_goals = list(goals)
    while remaining_goals and len(route_choices) < 3:
        try:
            path, selected_index = _vamp_plan_to_any(
                vamp,
                start,
                tuple(goal for _, goal in remaining_goals),
                environment,
                sample_offset=sample_offset,
            )
        except BackendOperationError:
            break
        solution_index, _ = remaining_goals.pop(selected_index)
        unwrapped = _append_unwrapped_positions(
            request.robot_model,
            (start,),
            path[1:],
        )
        candidate_path = _compact_joint_path(unwrapped)
        if cartesian_continuation_poses:
            try:
                continuation = _cartesian_continuation_route(
                    request,
                    ik_solver,
                    vamp,
                    cartesian_continuation_poses,
                    candidate_path[-1],
                    environment,
                    stage,
                    total_started,
                    ik_solution_count,
                )
            except _RouteFailure as failure:
                continuation_failure = failure
                _check_stage_budget(
                    request,
                    monotonic() - started,
                    total_started,
                    stage,
                )
                continue
            candidate_path = _compact_joint_path(
                _append_unwrapped_positions(
                    request.robot_model,
                    candidate_path,
                    continuation[1:],
                )
            )
        if path_validator is not None and not path_validator(candidate_path):
            _check_stage_budget(
                request,
                monotonic() - started,
                total_started,
                stage,
            )
            continue
        metrics = _sampled_orientation_deviation(
            candidate_path,
            (
                cartesian_continuation_poses[-1]
                if cartesian_continuation_poses
                else goal_pose
            ),
            vamp,
            request,
            stage,
            started,
            total_started,
        )
        route_choices.append(
            (
                (
                    _maximum_joint_path_edge(candidate_path),
                    metrics.cumulative_rotation_rad,
                    metrics.cumulative_translation_m,
                    _joint_path_length(candidate_path),
                    solution_index,
                ),
                candidate_path,
            )
        )
        _check_stage_budget(request, monotonic() - started, total_started, stage)
    if not route_choices:
        if continuation_failure is not None:
            raise continuation_failure
        raise _RouteFailure(
            PlanningErrorCode.VAMP_SEARCH_FAILED,
            stage,
            "VAMP found no collision-valid joint-space portal route",
        )
    _check_stage_budget(request, monotonic() - started, total_started, stage)
    return min(route_choices, key=lambda item: item[0])[1]


def _joint_space_portal_then_vertical_route(
    request: PlanningRequest,
    ik_solver: IKSolver,
    vamp: VampBackend,
    start: tuple[float, ...],
    transition_pose: RigidTransform,
    upper_pose: RigidTransform,
    environment,
    sample_offset: int,
    stage: TrajectoryStage,
    total_started: float,
    ik_solution_count: int,
    path_validator: Callable[[tuple[tuple[float, ...], ...]], bool] | None = None,
) -> tuple[tuple[float, ...], ...]:
    """Reach the pose below the opening before following its vertical corridor."""
    corridor = request.vertical_transit_corridor
    assert corridor is not None
    vertical_poses = _interpolate_poses(
        transition_pose,
        upper_pose,
        corridor.egress_waypoint_step_m,
        request.planning_policy.cartesian_rotation_step_rad,
    )[1:]
    portal_path = _joint_space_portal_route(
        request,
        ik_solver,
        vamp,
        start,
        transition_pose,
        environment,
        sample_offset,
        stage,
        total_started,
        ik_solution_count,
        cartesian_continuation_poses=vertical_poses,
        path_validator=path_validator,
    )
    return portal_path


def _staged_approach_route(
    request: PlanningRequest,
    candidate: GraspCandidate,
    ik_solver: IKSolver,
    vamp: VampBackend,
    poses: tuple[RigidTransform, ...],
    seed_state: RobotState,
    target_geometries: tuple[SceneGeometry, ...],
    environment_geometries: tuple[SceneGeometry, ...],
    allowed_target_contacts: tuple[_TargetContactAllowance, ...],
    sample_offset: int,
    total_started: float,
    ik_solution_count: int,
) -> tuple[RobotState, ...]:
    transition_distance_m = _approach_contact_transition_m(request)
    if not poses:
        return _continuous_ik(
            request,
            candidate,
            ik_solver,
            vamp,
            poses,
            seed_state,
            target_geometries,
            environment_geometries,
            allowed_target_contacts,
            TrajectoryStage.APPROACH,
            total_started,
            ik_solution_count,
        )

    grasp_matrix = np.asarray(poses[-1].matrix, dtype=np.float64)
    staging_matrix = grasp_matrix.copy()
    staging_matrix[:3, 3] -= (
        transition_distance_m * grasp_matrix[:3, 2]
    )
    staging_pose = RigidTransform(
        poses[-1].parent_frame,
        poses[-1].child_frame,
        tuple(tuple(float(value) for value in row) for row in staging_matrix),
    )  # type: ignore[arg-type]
    started = monotonic()
    staging_environment = vamp.environment(
        request.scene,
        T_world_base=request.T_world_base,
        excluded_object_ids=(request.target_object_id,),
    )
    solutions = _solve_ik(
        ik_solver,
        request,
        staging_pose,
        seed_state,
        TrajectoryStage.APPROACH,
        ik_solution_count,
        solution_validator=lambda state: _pregrasp_solution_valid(
            state,
            request,
            vamp,
            staging_environment,
            target_geometries,
            candidate.grasp_command.pregrasp_opening_m,
        ),
    )
    target_contact_rejected = False
    remaining = list(enumerate(solutions))
    staging_path = None
    while remaining:
        try:
            path, selected_index = _vamp_plan_to_any(
                vamp,
                seed_state.positions_rad,
                tuple(solution.positions_rad for _, solution in remaining),
                staging_environment,
                sample_offset=sample_offset,
            )
        except BackendOperationError:
            break
        if _path_has_forbidden_target_contact(
            path,
            target_geometries,
            candidate.grasp_command.pregrasp_opening_m,
            vamp,
            request.T_world_base,
        ):
            target_contact_rejected = True
            del remaining[selected_index]
            continue
        staging_path = _compact_joint_path(path)
        _check_stage_budget(
            request,
            monotonic() - started,
            total_started,
            TrajectoryStage.APPROACH,
        )
        break
    if staging_path is None:
        raise _RouteFailure(
            (
                PlanningErrorCode.STAGE_COLLISION_VIOLATION
                if target_contact_rejected
                else PlanningErrorCode.VAMP_SEARCH_FAILED
            ),
            TrajectoryStage.APPROACH,
            (
                "route to contact staging has forbidden early target contact"
                if target_contact_rejected
                else "VAMP found no collision-valid route to contact staging"
            ),
        )
    local_interpolation = _interpolate_poses(
        staging_pose,
        poses[-1],
        request.planning_policy.cartesian_translation_step_m,
        request.planning_policy.cartesian_rotation_step_rad,
    )
    local_poses = local_interpolation[1:]
    if not local_poses:
        return tuple(
            RobotState(request.robot_model.joint_names, positions)
            for positions in staging_path[1:]
        )
    local_states = _continuous_ik(
        request,
        candidate,
        ik_solver,
        vamp,
        local_poses,
        RobotState(request.robot_model.joint_names, staging_path[-1]),
        target_geometries,
        environment_geometries,
        _approach_contact_allowances(
            local_interpolation,
            transition_distance_m,
            frozenset(candidate.expected_contacts),
        ),
        TrajectoryStage.APPROACH,
        total_started,
        ik_solution_count,
    )
    combined = _compact_joint_path(
        (
            *staging_path[1:],
            *(state.positions_rad for state in local_states),
        )
    )
    return tuple(
        RobotState(request.robot_model.joint_names, positions)
        for positions in combined
    )


def _vamp_plan_to_any(
    vamp: VampBackend,
    start: tuple[float, ...],
    goals: tuple[tuple[float, ...], ...],
    environment,
    *,
    sample_offset: int,
) -> tuple[tuple[tuple[float, ...], ...], int]:
    """Plan one native VAMP search to any collision-valid goal."""
    result = vamp.plan_to_any(
        start,
        goals,
        environment,
        sample_offset=sample_offset,
    )
    return result.path, result.selected_goal_index


def _staged_lift_route(
    request: PlanningRequest,
    candidate: GraspCandidate,
    ik_solver: IKSolver,
    vamp: VampBackend,
    poses: tuple[RigidTransform, ...],
    seed_state: RobotState,
    target_geometries: tuple[SceneGeometry, ...],
    environment_geometries: tuple[SceneGeometry, ...],
    allowed_target_contacts: tuple[_TargetContactAllowance, ...],
    stage: TrajectoryStage,
    sample_offset: int,
    total_started: float,
    ik_solution_count: int,
) -> tuple[RobotState, ...]:
    if stage is not TrajectoryStage.LIFT:
        raise ValueError("staged LIFT route requires the LIFT stage")
    if target_geometries or allowed_target_contacts:
        raise ValueError("staged LIFT route does not accept target-contact inputs")
    support_geometries = tuple(
        item
        for item in environment_geometries
        if item.collision_group == request.source_support_group
    )
    non_support_geometries = tuple(
        item
        for item in environment_geometries
        if item.collision_group != request.source_support_group
    )
    departure_scene = SceneSnapshot(
        f"{request.scene.snapshot_id}.lift_without_source_support",
        request.scene.world_frame,
        non_support_geometries,
    )
    departure_environment = vamp.environment(
        departure_scene,
        T_world_base=request.T_world_base,
        excluded_object_ids=(),
        T_tcp_object=candidate.T_tcp_object,
        attached_geometry=request.object_config.attached_geometry,
    )
    states: list[RobotState] = []
    current = seed_state
    support_departed = False
    started = monotonic()
    for pose_index, pose in enumerate(poses):
        solutions = _solve_ik(
            ik_solver,
            request,
            pose,
            current,
            stage,
            ik_solution_count,
        )
        routes = []
        for solution_index, solution in enumerate(solutions):
            if not _has_joint_margin(solution, request):
                continue
            environment = departure_environment
            if not vamp.configuration_valid(solution.positions_rad, environment):
                continue
            try:
                path = _compact_joint_path(
                    vamp.plan(
                        current.positions_rad,
                        solution.positions_rad,
                        environment,
                        sample_offset=sample_offset,
                    ).path
                )
            except BackendOperationError:
                continue
            route_departed = support_departed
            route_valid = True
            for left, right in zip(path, path[1:]):
                next_departed = _lift_departure_motion_is_valid(
                    left,
                    right,
                    route_departed,
                    request,
                    candidate,
                    vamp,
                    environment_geometries,
                    support_geometries,
                    non_support_geometries,
                )
                if next_departed is None:
                    route_valid = False
                    break
                route_departed = next_departed
            if route_valid:
                routes.append(
                    (
                        _joint_path_length(path),
                        solution_index,
                        path,
                        route_departed,
                    )
                )
            _check_stage_budget(
                request,
                monotonic() - started,
                total_started,
                stage,
            )
        if not routes:
            raise _RouteFailure(
                PlanningErrorCode.CONTINUOUS_IK_FAILED,
                stage,
                "no collision-valid joint route reached LIFT Cartesian sample "
                f"{pose_index + 1}/{len(poses)}",
            )
        _, _, path, support_departed = min(
            routes,
            key=lambda item: (item[0], item[1]),
        )
        states.extend(
            RobotState(request.robot_model.joint_names, positions)
            for positions in path[1:]
        )
        current = RobotState(request.robot_model.joint_names, path[-1])
    if not support_departed:
        raise _RouteFailure(
            PlanningErrorCode.STAGE_COLLISION_VIOLATION,
            stage,
            "attached object did not depart the declared source support during LIFT",
        )
    return tuple(states)


def _continuous_ik(
    request: PlanningRequest,
    candidate: GraspCandidate,
    ik_solver: IKSolver,
    vamp: VampBackend,
    poses: tuple[RigidTransform, ...],
    seed_state: RobotState,
    target_geometries: tuple[SceneGeometry, ...],
    environment_geometries: tuple[SceneGeometry, ...],
    allowed_target_contacts: tuple[_TargetContactAllowance, ...],
    stage: TrajectoryStage,
    total_started: float,
    ik_solution_count: int,
) -> tuple[RobotState, ...]:
    if target_geometries and len(allowed_target_contacts) != len(poses):
        raise ValueError("target contact allowances must match Cartesian samples")
    states = []
    current = seed_state
    support_departed = stage is not TrajectoryStage.LIFT
    excluded_object_ids = (request.target_object_id,)
    if stage in (TrajectoryStage.APPROACH, TrajectoryStage.LIFT):
        excluded_object_ids = tuple(
            sorted({item.object_id for item in request.scene.geometries})
        )
    strict_environment = vamp.environment(
        request.scene,
        T_world_base=request.T_world_base,
        excluded_object_ids=excluded_object_ids,
        T_tcp_object=candidate.T_tcp_object if stage is TrajectoryStage.LIFT else None,
        attached_geometry=request.object_config.attached_geometry if stage is TrajectoryStage.LIFT else (),
    )
    support_geometries = tuple(
        item
        for item in environment_geometries
        if item.collision_group == request.source_support_group
    )
    non_support_environment_geometries = tuple(
        item
        for item in environment_geometries
        if item.collision_group != request.source_support_group
    )
    started = monotonic()
    for index, pose in enumerate(poses):
        solutions = _solve_ik(
            ik_solver, request, pose, current, stage, ik_solution_count
        )
        selected = None
        for state in solutions:
            if not _has_joint_margin(state, request):
                continue
            exact_opening = (
                candidate.grasp_command.pregrasp_opening_m
                if stage is TrajectoryStage.APPROACH
                else candidate.grasp_command.target_opening_m
            )
            if (
                stage is not TrajectoryStage.LIFT
                and non_support_environment_geometries
                and vamp.semantic_target_contacts(
                    state.positions_rad,
                    non_support_environment_geometries,
                    exact_opening,
                    request.T_world_base,
                )
            ):
                continue
            if (
                stage is not TrajectoryStage.LIFT
                and support_geometries
                and vamp.semantic_source_support_contacts(
                    state.positions_rad,
                    support_geometries,
                    exact_opening,
                    request.T_world_base,
                    use_physical_pad_hulls=allowed_target_contacts[
                        index
                    ].terminal_endpoint,
                )
            ):
                continue
            if (
                stage is not TrajectoryStage.LIFT
                and non_support_environment_geometries
                and _semantic_motion_has_forbidden_contact(
                    current.positions_rad,
                    state.positions_rad,
                    non_support_environment_geometries,
                    exact_opening,
                    frozenset(),
                    vamp,
                    request.T_world_base,
                )
            ):
                continue
            if (
                stage is not TrajectoryStage.LIFT
                and support_geometries
                and _semantic_support_motion_has_forbidden_contact(
                    current.positions_rad,
                    state.positions_rad,
                    support_geometries,
                    exact_opening,
                    allowed_target_contacts[index].terminal_motion,
                    vamp,
                    request.T_world_base,
                )
            ):
                continue
            environment = strict_environment
            next_support_departed = support_departed
            if stage is TrajectoryStage.LIFT:
                next_support_departed = _lift_departure_motion_is_valid(
                    current.positions_rad,
                    state.positions_rad,
                    support_departed,
                    request,
                    candidate,
                    vamp,
                    environment_geometries,
                    support_geometries,
                    non_support_environment_geometries,
                )
                if next_support_departed is None:
                    continue
            if not vamp.configuration_valid(state.positions_rad, environment):
                continue
            if not vamp.motion_valid(current.positions_rad, state.positions_rad, environment):
                continue
            if target_geometries:
                opening = candidate.grasp_command.pregrasp_opening_m
                allowed_contacts = allowed_target_contacts[index]
                contacts = vamp.semantic_target_contacts(
                    state.positions_rad,
                    target_geometries,
                    opening,
                    request.T_world_base,
                )
                if contacts - allowed_contacts.endpoint:
                    continue
                if _semantic_motion_has_forbidden_contact(
                    current.positions_rad,
                    state.positions_rad,
                    target_geometries,
                    opening,
                    allowed_contacts.motion,
                    vamp,
                    request.T_world_base,
                    goal_allowed_groups=allowed_contacts.endpoint,
                ):
                    continue
            selected = state
            support_departed = next_support_departed
            break
        if selected is None:
            raise _RouteFailure(
                PlanningErrorCode.CONTINUOUS_IK_FAILED,
                stage,
                f"continuous IK failed at Cartesian sample {index + 1}/{len(poses)}",
            )
        states.append(selected)
        current = selected
        _check_stage_budget(request, monotonic() - started, total_started, stage)
    if stage is TrajectoryStage.LIFT and not support_departed:
        raise _RouteFailure(
            PlanningErrorCode.STAGE_COLLISION_VIOLATION,
            stage,
            "attached object did not depart the declared source support during LIFT",
        )
    return tuple(states)


def _solve_ik(
    solver: IKSolver,
    request: PlanningRequest,
    pose: RigidTransform,
    seed_state: RobotState,
    stage: TrajectoryStage,
    maximum_solutions: int,
    solution_validator: Callable[[RobotState], bool] | None = None,
) -> tuple[RobotState, ...]:
    solutions = _request_ik_solutions(
        solver,
        request,
        pose,
        (seed_state,),
        stage,
        maximum_solutions,
        solution_validator,
    )
    aligned = tuple(
        RobotState(
            request.robot_model.joint_names,
            request.robot_model.unwrap_positions(
                item.state.positions_rad, seed_state.positions_rad
            ),
        )
        for item in solutions
    )
    return tuple(
        state
        for _, state in sorted(
            enumerate(aligned),
            key=lambda item: (
                float(
                    np.linalg.norm(
                        np.asarray(item[1].positions_rad)
                        - np.asarray(seed_state.positions_rad)
                    )
                ),
                item[0],
            ),
        )
    )


def _request_ik_solutions(
    solver: IKSolver,
    request: PlanningRequest,
    pose: RigidTransform,
    seed_states: tuple[RobotState, ...],
    stage: TrajectoryStage,
    maximum_solutions: int,
    solution_validator: Callable[[RobotState], bool] | None = None,
) -> tuple[IKSolution, ...]:
    result = solver.solve(
        (pose,),
        seed_states,
        IKConstraints(
            request.planning_policy.ik_position_tolerance_m,
            request.planning_policy.ik_orientation_tolerance_rad,
            maximum_solutions,
            solution_validator,
        ),
    )
    if result.error_code == IKErrorCode.BACKEND_UNAVAILABLE:
        raise BackendUnavailableError(result.message)
    if result.error_code == IKErrorCode.BACKEND_ERROR:
        raise _RouteFailure(PlanningErrorCode.BACKEND_ERROR, stage, result.message)
    if not result.success:
        raise _RouteFailure(PlanningErrorCode.IK_CANDIDATES_EXHAUSTED, stage, result.message)
    try:
        validate_ik_result(result, request.robot_model)
    except ValueError as exc:
        raise _RouteFailure(PlanningErrorCode.BACKEND_ERROR, stage, str(exc)) from exc
    return result.solutions


def _pregrasp_solution_valid(
    state: RobotState,
    request: PlanningRequest,
    vamp: VampBackend,
    environment,
    target_geometries: tuple[SceneGeometry, ...],
    opening_m: float,
) -> bool:
    aligned = RobotState(
        request.robot_model.joint_names,
        request.robot_model.unwrap_positions(
            state.positions_rad,
            request.initial_state.positions_rad,
        ),
    )
    if not _has_joint_margin(aligned, request):
        return False
    if not vamp.configuration_valid(aligned.positions_rad, environment):
        return False
    contacts = vamp.semantic_target_contacts(
        aligned.positions_rad,
        target_geometries,
        opening_m,
        request.T_world_base,
    )
    return not contacts


def _first_valid_ik(
    request: PlanningRequest,
    solver: IKSolver,
    vamp: VampBackend,
    pose: RigidTransform,
    seed_state: RobotState,
    environment,
    stage: TrajectoryStage,
    maximum_solutions: int,
) -> RobotState:
    for state in _solve_ik(
        solver, request, pose, seed_state, stage, maximum_solutions
    ):
        if _has_joint_margin(state, request) and vamp.configuration_valid(
            state.positions_rad, environment
        ):
            return state
    raise _RouteFailure(
        PlanningErrorCode.IK_CANDIDATES_EXHAUSTED,
        stage,
        f"no collision-valid IK solution for {stage.value}",
    )


def _vamp_stage(
    request: PlanningRequest,
    vamp: VampBackend,
    start: tuple[float, ...],
    goal: tuple[float, ...],
    environment,
    candidate_id: str,
    sample_offset: int,
    stage: TrajectoryStage,
    attempts: list[PlanningAttempt],
    total_started: float,
) -> tuple[tuple[float, ...], ...]:
    started = monotonic()
    try:
        result = vamp.plan(start, goal, environment, sample_offset=sample_offset)
    except BackendOperationError as exc:
        raise _RouteFailure(PlanningErrorCode.VAMP_SEARCH_FAILED, stage, str(exc)) from exc
    elapsed = monotonic() - started
    _check_stage_budget(request, elapsed, total_started, stage)
    attempts.append(
        PlanningAttempt(
            candidate_id,
            sample_offset,
            stage,
            elapsed,
            None,
            f"VAMP {stage.value} succeeded",
        )
    )
    return result.path


def _orientation_free_lift_endpoints(
    request: PlanningRequest,
    candidate: GraspCandidate,
    ik_solver: IKSolver,
    vamp: VampBackend,
    minimum_world_lift_pose: RigidTransform,
    minimum_lift_state: RobotState,
    world_preplace_pose: RigidTransform,
    environment_geometries: tuple[SceneGeometry, ...],
    total_started: float,
    ik_solution_count: int,
) -> Iterator[tuple[RobotState, ...]]:
    minimum_z = float(minimum_world_lift_pose.matrix[2][3])
    maximum_z = float(world_preplace_pose.matrix[2][3])
    extra_distance = max(0.0, maximum_z - minimum_z)
    step = request.planning_policy.cartesian_translation_step_m
    step_count = int(ceil(extra_distance / step))
    yield ()

    elapsed = 0.0
    extension: list[RobotState] = []
    current = minimum_lift_state
    for index in range(1, step_count + 1):
        if index + 1 > request.planning_policy.trajectory_validation_max_samples:
            raise _RouteFailure(
                PlanningErrorCode.BUDGET_EXHAUSTED,
                TrajectoryStage.LIFT,
                "orientation-free LIFT endpoint search exceeded its configured "
                "sample budget",
            )
        extra_lift = min(extra_distance, index * step)
        target = compose(
            inverse(request.T_world_base),
            _translate_world_z(minimum_world_lift_pose, extra_lift),
        )
        started = monotonic()
        try:
            state = _continuous_ik(
                request,
                candidate,
                ik_solver,
                vamp,
                (target,),
                current,
                (),
                environment_geometries,
                (),
                TrajectoryStage.LIFT,
                total_started,
                ik_solution_count,
            )[0]
        except _RouteFailure as failure:
            elapsed += monotonic() - started
            _check_stage_budget(
                request,
                elapsed,
                total_started,
                TrajectoryStage.LIFT,
            )
            if failure.code in (
                PlanningErrorCode.CONTINUOUS_IK_FAILED,
                PlanningErrorCode.IK_CANDIDATES_EXHAUSTED,
                PlanningErrorCode.STAGE_COLLISION_VIOLATION,
            ):
                return
            raise
        elapsed += monotonic() - started
        extension.append(state)
        current = state
        _check_stage_budget(
            request,
            elapsed,
            total_started,
            TrajectoryStage.LIFT,
        )
        yield tuple(extension)


def _orientation_preferred_vamp_stage(
    request: PlanningRequest,
    ik_solver: IKSolver,
    vamp: VampBackend,
    start: tuple[float, ...],
    goal_pose: RigidTransform,
    preferred_orientation: RigidTransform,
    environment,
    candidate_id: str,
    sample_offset: int,
    stage: TrajectoryStage,
    attempts: list[PlanningAttempt],
    total_started: float,
    ik_solution_count: int,
) -> tuple[tuple[tuple[float, ...], ...], float]:
    started = monotonic()
    solutions = _solve_ik(
        ik_solver,
        request,
        goal_pose,
        RobotState(request.robot_model.joint_names, start),
        stage,
        ik_solution_count,
    )
    routes = []
    planner_failures = []
    for solution_index, solution in enumerate(solutions):
        if not _has_joint_margin(solution, request):
            continue
        if not vamp.configuration_valid(solution.positions_rad, environment):
            continue
        try:
            path = vamp.plan(
                start,
                solution.positions_rad,
                environment,
                sample_offset=sample_offset,
            ).path
        except BackendOperationError as exc:
            planner_failures.append(str(exc))
            _check_stage_budget(
                request,
                monotonic() - started,
                total_started,
                stage,
            )
            continue
        metrics = _sampled_orientation_deviation(
            path,
            preferred_orientation,
            vamp,
            request,
            stage,
            started,
            total_started,
        )
        routes.append(
            (
                _path_quality_key(
                    metrics,
                    _joint_path_length(path),
                    solution_index,
                ),
                metrics,
                solution_index,
                path,
            )
        )
        _check_stage_budget(
            request,
            monotonic() - started,
            total_started,
            stage,
        )
    if not routes:
        if planner_failures:
            raise _RouteFailure(
                PlanningErrorCode.VAMP_SEARCH_FAILED,
                stage,
                planner_failures[-1],
            )
        raise _RouteFailure(
            PlanningErrorCode.IK_CANDIDATES_EXHAUSTED,
            stage,
            f"no collision-valid IK solution for {stage.value}",
        )
    (
        _,
        metrics,
        solution_index,
        path,
    ) = min(
        routes,
        key=lambda item: item[0],
    )
    elapsed = monotonic() - started
    attempts.append(
        PlanningAttempt(
            candidate_id,
            sample_offset,
            stage,
            elapsed,
            None,
            f"VAMP {stage.value} selected IK solution {solution_index} "
            "returned path with "
            "sampled TCP path quality: maximum orientation deviation "
            f"{metrics.maximum_deviation_rad:.9g}rad, cumulative rotation "
            f"{metrics.cumulative_rotation_rad:.9g}rad, cumulative translation "
            f"{metrics.cumulative_translation_m:.9g}m over "
            f"{metrics.sample_count} FK samples",
        )
    )
    return path, _orientation_path_cost(metrics)


def _sampled_orientation_deviation(
    path: tuple[tuple[float, ...], ...],
    preferred_orientation: RigidTransform,
    vamp: VampBackend,
    request: PlanningRequest,
    stage: TrajectoryStage,
    stage_started: float,
    total_started: float,
) -> _OrientationPathMetrics:
    if not path:
        raise BackendOperationError("VAMP orientation scoring requires a path")
    reference = Rotation.from_matrix(
        np.asarray(preferred_orientation.matrix, dtype=np.float64)[:3, :3]
    )
    policy = request.planning_policy
    poses: dict[tuple[float, ...], tuple[np.ndarray, Rotation]] = {}
    maximum_deviation = 0.0

    def pose_at(
        configuration: tuple[float, ...],
    ) -> tuple[np.ndarray, Rotation]:
        nonlocal maximum_deviation
        if configuration not in poses:
            if len(poses) >= policy.trajectory_validation_max_samples:
                raise _RouteFailure(
                    PlanningErrorCode.BUDGET_EXHAUSTED,
                    stage,
                    f"{stage.value} orientation scoring exceeded its configured "
                    "sample budget",
                )
            matrix = np.asarray(
                vamp.tcp_pose(configuration).matrix,
                dtype=np.float64,
            )
            rotation = Rotation.from_matrix(matrix[:3, :3])
            poses[configuration] = (matrix[:3, 3], rotation)
            maximum_deviation = max(
                maximum_deviation,
                float((reference.inv() * rotation).magnitude()),
            )
            if len(poses) % 16 == 0:
                _check_stage_budget(
                    request,
                    monotonic() - stage_started,
                    total_started,
                    stage,
                )
        return poses[configuration]

    def refine_interval(
        left: tuple[float, ...],
        right: tuple[float, ...],
    ) -> tuple[tuple[float, ...], ...]:
        midpoint = tuple(
            0.5 * (left_value + right_value)
            for left_value, right_value in zip(left, right)
        )
        if midpoint == left or midpoint == right:
            return (left, right)
        left_rotation = pose_at(left)[1]
        midpoint_rotation = pose_at(midpoint)[1]
        right_rotation = pose_at(right)[1]
        rotation_sweep = float(
            (left_rotation.inv() * midpoint_rotation).magnitude()
            + (midpoint_rotation.inv() * right_rotation).magnitude()
        )
        if rotation_sweep > policy.trajectory_validation_tcp_rotation_step_rad:
            left_samples = refine_interval(left, midpoint)
            right_samples = refine_interval(midpoint, right)
            return (*left_samples[:-1], *right_samples)
        return (left, midpoint, right)

    pose_at(path[0])
    ordered_configurations = [path[0]]
    for left, right in zip(path, path[1:]):
        left_values = np.asarray(left, dtype=np.float64)
        right_values = np.asarray(right, dtype=np.float64)
        joint_distance = float(np.max(np.abs(right_values - left_values)))
        count = max(
            1,
            int(ceil(joint_distance / policy.trajectory_validation_joint_step_rad)),
        )
        previous = left
        for index in range(1, count + 1):
            if index == count:
                current = right
            else:
                fraction = index / count
                current = tuple(
                    float(value)
                    for value in left_values
                    + fraction * (right_values - left_values)
                )
            pose_at(current)
            refined = refine_interval(previous, current)
            ordered_configurations.extend(refined[1:])
            previous = current

    _check_stage_budget(
        request,
        monotonic() - stage_started,
        total_started,
        stage,
    )
    ordered_poses = tuple(
        pose_at(configuration) for configuration in ordered_configurations
    )
    cumulative_translation = sum(
        float(np.linalg.norm(right[0] - left[0]))
        for left, right in zip(ordered_poses, ordered_poses[1:])
    )
    cumulative_rotation = sum(
        float((left[1].inv() * right[1]).magnitude())
        for left, right in zip(ordered_poses, ordered_poses[1:])
    )
    return _OrientationPathMetrics(
        maximum_deviation,
        cumulative_translation,
        cumulative_rotation,
        len(ordered_poses),
    )


def _orientation_path_cost(metrics: _OrientationPathMetrics) -> float:
    return metrics.maximum_deviation_rad + metrics.cumulative_rotation_rad


def _path_quality_key(
    metrics: _OrientationPathMetrics,
    joint_path_length: float,
    tie_breaker: int,
) -> tuple[float, float, float, float, int]:
    return (
        _orientation_path_cost(metrics),
        metrics.cumulative_translation_m,
        metrics.maximum_deviation_rad,
        joint_path_length,
        tie_breaker,
    )


def _pregrasp_path_quality_key(
    metrics: _OrientationPathMetrics,
    joint_path_length: float,
    tie_breaker: int,
) -> tuple[float, float, float, float, int]:
    return (
        metrics.cumulative_rotation_rad,
        metrics.cumulative_translation_m,
        metrics.maximum_deviation_rad,
        joint_path_length,
        tie_breaker,
    )


def _joint_path_length(path: tuple[tuple[float, ...], ...]) -> float:
    return sum(
        float(np.linalg.norm(np.asarray(right) - np.asarray(left)))
        for left, right in zip(path, path[1:])
    )


def _maximum_joint_path_edge(path: tuple[tuple[float, ...], ...]) -> float:
    return max(
        (
            float(np.linalg.norm(np.asarray(right) - np.asarray(left)))
            for left, right in zip(path, path[1:])
        ),
        default=0.0,
    )


def _joint_path_detour_ratio(path: tuple[tuple[float, ...], ...]) -> float:
    if len(path) < 2:
        return 1.0
    direct = float(np.linalg.norm(np.asarray(path[-1]) - np.asarray(path[0])))
    if direct <= 1e-12:
        return 1.0
    return _joint_path_length(path) / direct


def _payload_route_duration_s(
    prefix_waypoints: tuple[GeometricWaypoint, ...],
    extra_lift_states: tuple[RobotState, ...],
    egress_path: tuple[tuple[float, ...], ...],
    place_path: tuple[tuple[float, ...], ...],
    channels: dict[str, float],
    ruckig: RuckigTimeParameterizer,
    *,
    stop_at_egress_waypoints: bool = False,
) -> float:
    waypoints = list(prefix_waypoints)
    waypoints.extend(
        GeometricWaypoint(TrajectoryStage.LIFT, state.positions_rad, dict(channels))
        for state in extra_lift_states
    )
    waypoints.extend(
        GeometricWaypoint(
            TrajectoryStage.EGRESS,
            point,
            dict(channels),
            stop=stop_at_egress_waypoints,
        )
        for point in egress_path[1:]
    )
    waypoints.extend(
        GeometricWaypoint(
            TrajectoryStage.PLACE,
            point,
            dict(channels),
            stop=index == len(place_path[1:]) - 1 or stop_at_egress_waypoints,
        )
        for index, point in enumerate(place_path[1:])
    )
    trajectory = ruckig.parameterize(tuple(waypoints), ())
    lift_phase = next(
        phase for phase in trajectory.phases
        if phase.stage is TrajectoryStage.LIFT
    )
    return trajectory.knots[-1].time_s - lift_phase.start_time_s


def _payload_route_selection_key(
    duration_s: float,
    cumulative_rotation_rad: float,
    cumulative_translation_m: float,
    maximum_orientation_deviation_rad: float,
    joint_path_length: float,
    tie_breaker: int,
) -> tuple[float, ...]:
    return (
        duration_s,
        cumulative_rotation_rad,
        cumulative_translation_m,
        maximum_orientation_deviation_rad,
        joint_path_length,
        tie_breaker,
    )


def _path_has_forbidden_target_contact(
    path: tuple[tuple[float, ...], ...],
    target_geometries: tuple[SceneGeometry, ...],
    opening_m: float,
    vamp: VampBackend,
    T_world_base: RigidTransform,
) -> bool:
    for left, right in zip(path, path[1:]):
        sample_count = max(2, int(ceil(max(abs(b - a) for a, b in zip(left, right)) / 0.02)))
        for index in range(sample_count + 1):
            fraction = index / sample_count
            state = tuple(a + fraction * (b - a) for a, b in zip(left, right))
            if vamp.semantic_target_contacts(
                state, target_geometries, opening_m, T_world_base
            ):
                return True
    return False


def _semantic_motion_has_forbidden_contact(
    start: tuple[float, ...],
    goal: tuple[float, ...],
    geometries: tuple[SceneGeometry, ...],
    opening_m: float,
    allowed_groups: frozenset[str],
    vamp: VampBackend,
    T_world_base: RigidTransform,
    *,
    goal_allowed_groups: frozenset[str] | None = None,
) -> bool:
    sample_count = max(
        2,
        int(ceil(max(abs(right - left) for left, right in zip(start, goal)) / 0.02)),
    )
    for index in range(1, sample_count + 1):
        fraction = index / sample_count
        state = tuple(
            left + fraction * (right - left) for left, right in zip(start, goal)
        )
        contacts = vamp.semantic_target_contacts(
            state, geometries, opening_m, T_world_base
        )
        permitted = (
            goal_allowed_groups
            if goal_allowed_groups is not None and index == sample_count
            else allowed_groups
        )
        if contacts - permitted:
            return True
    return False


def _semantic_support_motion_has_forbidden_contact(
    start: tuple[float, ...],
    goal: tuple[float, ...],
    support_geometries: tuple[SceneGeometry, ...],
    opening_m: float,
    use_physical_pad_hulls: bool,
    vamp: VampBackend,
    T_world_base: RigidTransform,
) -> bool:
    sample_count = max(
        2,
        int(ceil(max(abs(right - left) for left, right in zip(start, goal)) / 0.02)),
    )
    for index in range(1, sample_count + 1):
        fraction = index / sample_count
        state = tuple(
            left + fraction * (right - left) for left, right in zip(start, goal)
        )
        if vamp.semantic_source_support_contacts(
            state,
            support_geometries,
            opening_m,
            T_world_base,
            use_physical_pad_hulls=use_physical_pad_hulls,
        ):
            return True
    return False


def _lift_departure_motion_is_valid(
    start: tuple[float, ...],
    goal: tuple[float, ...],
    support_departed: bool,
    request: PlanningRequest,
    candidate: GraspCandidate,
    vamp: VampBackend,
    environment_geometries: tuple[SceneGeometry, ...],
    support_geometries: tuple[SceneGeometry, ...],
    non_support_environment_geometries: tuple[SceneGeometry, ...],
) -> bool | None:
    sample_count = max(
        2,
        int(ceil(max(abs(right - left) for left, right in zip(start, goal)) / 0.02)),
    )
    departed = support_departed
    for index in range(1, sample_count + 1):
        fraction = index / sample_count
        state = tuple(
            left + fraction * (right - left) for left, right in zip(start, goal)
        )
        contacts = vamp.semantic_attachment_contacts(
            state,
            candidate.T_tcp_object,
            request.object_config.attached_geometry,
            environment_geometries,
            request.T_world_base,
        )
        if contacts - {request.source_support_group}:
            return None
        on_support = request.source_support_group in contacts
        if departed and on_support:
            return None
        next_departed = departed or not on_support
        if vamp.semantic_target_contacts(
            state,
            non_support_environment_geometries,
            candidate.grasp_command.target_opening_m,
            request.T_world_base,
        ):
            return None
        support_contacts = vamp.semantic_target_contacts(
            state,
            support_geometries,
            candidate.grasp_command.target_opening_m,
            request.T_world_base,
        )
        permitted_support_contacts = (
            frozenset(request.end_effector_config.contact_groups)
            if not next_departed
            else frozenset()
        )
        if support_contacts - permitted_support_contacts:
            return None
        departed = next_departed
    return departed


def _lift_departure_path_is_valid(
    path: tuple[tuple[float, ...], ...],
    request: PlanningRequest,
    candidate: GraspCandidate,
    vamp: VampBackend,
    scene_geometries: tuple[SceneGeometry, ...],
) -> bool:
    environment_geometries = tuple(
        item
        for item in scene_geometries
        if item.object_id != request.target_object_id
    )
    support_geometries = tuple(
        item
        for item in environment_geometries
        if item.collision_group == request.source_support_group
    )
    non_support_geometries = tuple(
        item
        for item in environment_geometries
        if item.collision_group != request.source_support_group
    )
    for left, right in zip(path, path[1:]):
        if (
            _lift_departure_motion_is_valid(
                left,
                right,
                True,
                request,
                candidate,
                vamp,
                environment_geometries,
                support_geometries,
                non_support_geometries,
            )
            is None
        ):
            return False
    return True


def _path_has_forbidden_attachment_contact(
    path: tuple[tuple[float, ...], ...],
    request: PlanningRequest,
    candidate: GraspCandidate,
    vamp: VampBackend,
    environment_geometries: tuple[SceneGeometry, ...],
) -> bool:
    for left, right in zip(path, path[1:]):
        sample_count = max(
            2,
            int(ceil(max(abs(b - a) for a, b in zip(left, right)) / 0.02)),
        )
        for index in range(sample_count + 1):
            fraction = index / sample_count
            state = tuple(a + fraction * (b - a) for a, b in zip(left, right))
            if vamp.semantic_attachment_contacts(
                state,
                candidate.T_tcp_object,
                request.object_config.attached_geometry,
                environment_geometries,
                request.T_world_base,
            ):
                return True
    return False


def _retime_payload_motion(
    request: PlanningRequest,
    route: _Route,
    trajectory: PlannedTrajectory,
    vamp: VampBackend,
    ruckig: RuckigTimeParameterizer,
    total_started: float,
) -> PlannedTrajectory:
    policy = request.planning_policy
    waypoint_scales = [waypoint.motion_scale for waypoint in route.waypoints]
    result = trajectory
    for _ in range(4):
        corrections: dict[int, tuple[TrajectoryStage, float]] = {}
        for segment_index, segment in enumerate(result.segments):
            if segment.stage not in _PAYLOAD_STAGES:
                continue
            _check_total_budget(request, total_started, segment.stage)
            metrics = _payload_motion_metrics(
                request,
                result,
                vamp,
                total_started,
                included_stages=frozenset((segment.stage,)),
                included_segment_indices=frozenset((segment_index,)),
            )
            correction = _payload_motion_scale(policy, metrics)
            if correction < 1.0:
                previous = corrections.get(segment.end_knot)
                if previous is None or correction < previous[1]:
                    corrections[segment.end_knot] = (segment.stage, correction)
        if not corrections:
            break
        limiting_stage, _ = min(corrections.values(), key=lambda item: item[1])
        _check_total_budget(request, total_started, limiting_stage)
        for waypoint_index, (_, correction) in corrections.items():
            waypoint_scales[waypoint_index] *= correction
        scaled_waypoints = tuple(
            replace(
                waypoint,
                motion_scale=min(
                    waypoint.motion_scale,
                    waypoint_scales[index],
                ),
            )
            if index in corrections or waypoint_scales[index] < waypoint.motion_scale
            else waypoint
            for index, waypoint in enumerate(route.waypoints)
        )
        result = ruckig.parameterize(scaled_waypoints, route.events)
        _check_total_budget(request, total_started, limiting_stage)

    scaled_metrics = _payload_motion_metrics(
        request,
        result,
        vamp,
        total_started,
    )
    violation = _payload_motion_violation(request, scaled_metrics)
    if violation is not None:
        stage, message = violation
        raise _PayloadMotionLimitError(stage, message)
    return result


def _payload_motion_scale(
    policy: PlanningPolicy,
    metrics: _PayloadMotionMetrics,
) -> float:
    return min(
        1.0,
        policy.payload_tcp_max_angular_velocity_rad_s
        / max(metrics.maximum_angular_velocity_rad_s, 1e-12),
        policy.payload_tcp_max_translation_velocity_m_s
        / max(metrics.maximum_translation_velocity_m_s, 1e-12),
        sqrt(
            policy.payload_tcp_max_translation_acceleration_m_s2
            / max(metrics.maximum_translation_acceleration_m_s2, 1e-12)
        ),
    )


@dataclass(frozen=True)
class _PayloadMotionSamplingError(Exception):
    stage: TrajectoryStage
    message: str


@dataclass(frozen=True)
class _PayloadMotionLimitError(Exception):
    stage: TrajectoryStage
    message: str


def _payload_motion_metrics(
    request: PlanningRequest,
    trajectory: PlannedTrajectory,
    vamp: VampBackend,
    total_started: float,
    *,
    included_stages: frozenset[TrajectoryStage] = _PAYLOAD_STAGES,
    included_segment_indices: frozenset[int] | None = None,
) -> _PayloadMotionMetrics:
    time_step_s = request.planning_policy.payload_motion_validation_time_step_s
    maximum_samples = request.planning_policy.trajectory_validation_max_samples
    times: list[float] = []
    sample_stages: list[TrajectoryStage] = []
    positions: list[np.ndarray] = []
    rotations: list[Rotation] = []
    for segment_index, segment in enumerate(trajectory.segments):
        if segment.stage not in included_stages:
            continue
        if (
            included_segment_indices is not None
            and segment_index not in included_segment_indices
        ):
            continue
        _check_total_budget(request, total_started, segment.stage)
        sample_count = max(2, int(ceil(segment.duration_s / time_step_s)))
        for index in range(sample_count + 1):
            local_time_s = (
                segment.duration_s
                if index == sample_count
                else segment.duration_s * index / sample_count
            )
            time_s = segment.start_time_s + local_time_s
            if times and abs(time_s - times[-1]) <= 1e-12:
                continue
            if len(times) >= maximum_samples:
                raise _PayloadMotionSamplingError(
                    segment.stage,
                    "payload motion validation exceeded its configured sample budget",
                )
            if len(times) % 16 == 0:
                _check_total_budget(request, total_started, segment.stage)
            sample = segment.sample(local_time_s)
            tcp = np.asarray(vamp.tcp_pose(sample.positions_rad).matrix, dtype=np.float64)
            times.append(time_s)
            sample_stages.append(segment.stage)
            positions.append(tcp[:3, 3])
            rotations.append(Rotation.from_matrix(tcp[:3, :3]))
        _check_total_budget(request, total_started, segment.stage)
    if len(times) < 3:
        stage = sample_stages[-1] if sample_stages else TrajectoryStage.LIFT
        raise _PayloadMotionSamplingError(
            stage,
            "payload motion validation requires at least three loaded trajectory samples",
        )

    sample_times = np.asarray(times, dtype=np.float64)
    time_deltas = np.diff(sample_times)
    if np.any(time_deltas <= 0.0):
        raise BackendOperationError("payload motion validation times are not increasing")
    tcp_positions = np.asarray(positions, dtype=np.float64)
    linear_velocities = np.diff(tcp_positions, axis=0) / time_deltas[:, None]
    translation_speeds = np.linalg.norm(linear_velocities, axis=1)
    angular_speeds = np.asarray(
        [
            (left.inv() * right).magnitude() / time_delta
            for left, right, time_delta in zip(
                rotations,
                rotations[1:],
                time_deltas,
            )
        ],
        dtype=np.float64,
    )
    acceleration_time_deltas = 0.5 * (time_deltas[:-1] + time_deltas[1:])
    translation_accelerations = np.linalg.norm(
        np.diff(linear_velocities, axis=0) / acceleration_time_deltas[:, None],
        axis=1,
    )
    angular_index = int(np.argmax(angular_speeds))
    translation_index = int(np.argmax(translation_speeds))
    acceleration_index = int(np.argmax(translation_accelerations))
    return _PayloadMotionMetrics(
        float(angular_speeds[angular_index]),
        float(translation_speeds[translation_index]),
        float(translation_accelerations[acceleration_index]),
        sample_stages[angular_index + 1],
        sample_stages[translation_index + 1],
        sample_stages[acceleration_index + 1],
    )


def _payload_motion_violation(
    request: PlanningRequest,
    metrics: _PayloadMotionMetrics,
) -> tuple[TrajectoryStage, str] | None:
    policy = request.planning_policy
    checks = (
        (
            metrics.maximum_angular_velocity_rad_s,
            policy.payload_tcp_max_angular_velocity_rad_s,
            metrics.angular_velocity_stage,
            "TCP angular velocity",
            "rad/s",
        ),
        (
            metrics.maximum_translation_velocity_m_s,
            policy.payload_tcp_max_translation_velocity_m_s,
            metrics.translation_velocity_stage,
            "TCP translation velocity",
            "m/s",
        ),
        (
            metrics.maximum_translation_acceleration_m_s2,
            policy.payload_tcp_max_translation_acceleration_m_s2,
            metrics.translation_acceleration_stage,
            "TCP translation acceleration",
            "m/s^2",
        ),
    )
    for actual, limit, stage, label, unit in checks:
        if actual > limit * (1.0 + 1e-4) + 1e-9:
            return (
                stage,
                f"time-parameterized payload {label} {actual:.9g}{unit} "
                f"exceeds configured limit {limit:.9g}{unit}",
            )
    return None


def _validate_time_parameterized_trajectory(
    request: PlanningRequest,
    candidate: GraspCandidate,
    trajectory: PlannedTrajectory,
    vamp: VampBackend,
    total_started: float,
) -> None:
    target_geometries = tuple(
        item for item in request.scene.geometries if item.object_id == request.target_object_id
    )
    environment_geometries = tuple(
        item
        for item in request.scene.geometries
        if item.object_id != request.target_object_id
    )
    support_geometries = tuple(
        item
        for item in environment_geometries
        if item.collision_group == request.source_support_group
    )
    non_support_environment_geometries = tuple(
        item
        for item in environment_geometries
        if item.collision_group != request.source_support_group
    )
    all_object_ids = tuple(
        sorted({item.object_id for item in request.scene.geometries})
    )
    strict_unloaded_environment = vamp.environment(
        request.scene,
        T_world_base=request.T_world_base,
        excluded_object_ids=(request.target_object_id,),
    )
    exact_unloaded_environment = vamp.environment(
        request.scene,
        T_world_base=request.T_world_base,
        excluded_object_ids=all_object_ids,
    )
    exact_loaded_environment = vamp.environment(
        request.scene,
        T_world_base=request.T_world_base,
        excluded_object_ids=all_object_ids,
        T_tcp_object=candidate.T_tcp_object,
        attached_geometry=request.object_config.attached_geometry,
    )
    strict_loaded_environment = vamp.environment(
        request.scene,
        T_world_base=request.T_world_base,
        excluded_object_ids=(request.target_object_id,),
        T_tcp_object=candidate.T_tcp_object,
        attached_geometry=request.object_config.attached_geometry,
    )
    stage_environments = {
        TrajectoryStage.PREGRASP: strict_unloaded_environment,
        TrajectoryStage.APPROACH: exact_unloaded_environment,
        TrajectoryStage.GRASP: exact_unloaded_environment,
        TrajectoryStage.LIFT: exact_loaded_environment,
        TrajectoryStage.EGRESS: strict_loaded_environment,
        TrajectoryStage.PLACE: strict_loaded_environment,
    }
    opening_channel = request.end_effector_config.actuator_position_channel
    segment_samples: dict[int, tuple[TrajectoryKnot, ...]] = {}
    native_stage_elapsed: dict[TrajectoryStage, float] = {}
    validation_sample_count = 0
    for stage in TrajectoryStage:
        native_started = monotonic()
        native_samples: list[tuple[int, TrajectoryKnot]] = []
        previous_time_s: float | None = None
        for segment_index, segment in enumerate(trajectory.segments):
            if segment.stage is not stage:
                continue
            samples = _trajectory_segment_validation_samples(
                segment,
                opening_channel,
                vamp,
                request,
                candidate,
                request.planning_policy.trajectory_validation_max_samples
                - validation_sample_count,
            )
            validation_sample_count += len(samples)
            segment_samples[segment_index] = samples
            for sample in samples:
                if previous_time_s is not None and sample.time_s == previous_time_s:
                    continue
                native_samples.append((segment_index, sample))
                previous_time_s = sample.time_s
        first_configuration = native_samples[0][1].positions_rad
        environment = stage_environments[stage]
        if not vamp.configuration_valid(first_configuration, environment):
            raise _RouteFailure(
                PlanningErrorCode.STAGE_COLLISION_VIOLATION,
                stage,
                f"time-parameterized {stage.value} has an invalid VAMP "
                f"configuration at t={native_samples[0][1].time_s:.6f}s",
            )
        configurations = tuple(
            sample.positions_rad for _, sample in native_samples
        )
        if not vamp.path_valid(configurations, environment):
            _check_stage_budget(
                request, monotonic() - native_started, total_started, stage
            )
            _raise_native_trajectory_failure(
                request,
                stage,
                native_samples,
                environment,
                vamp,
                native_started,
                total_started,
            )
        elapsed = monotonic() - native_started
        _check_stage_budget(request, elapsed, total_started, stage)
        native_stage_elapsed[stage] = elapsed

    expected_contacts = frozenset(candidate.expected_contacts)
    grasp_pose = compose(
        inverse(request.T_world_base),
        reframe(candidate.T_world_tcp, child_frame=request.robot_model.tcp_frame),
    )
    grasp_position = np.asarray(grasp_pose.matrix, dtype=np.float64)[:3, 3]
    support_departed = False
    final_grasp_contacts = frozenset()
    stage_started = monotonic()
    previous_stage: TrajectoryStage | None = None
    previous_time_s: float | None = None
    stage_sample_count = 0

    for segment_index, segment in enumerate(trajectory.segments):
        stage = segment.stage
        if stage is not previous_stage:
            if previous_stage is not None:
                _check_stage_budget(
                    request,
                    native_stage_elapsed[previous_stage]
                    + monotonic()
                    - stage_started,
                    total_started,
                    previous_stage,
                )
            stage_started = monotonic()
            previous_stage = stage
            previous_time_s = None
            stage_sample_count = 0
        samples = segment_samples[segment_index]
        for sample in samples:
            if previous_time_s is not None and sample.time_s == previous_time_s:
                continue
            configuration = sample.positions_rad
            opening_m = float(sample.actuator_channels[opening_channel])
            time_s = sample.time_s
            if stage is TrajectoryStage.APPROACH:
                non_support_contacts = vamp.semantic_target_contacts(
                    configuration,
                    non_support_environment_geometries,
                    opening_m,
                    request.T_world_base,
                )
                tcp_position = np.asarray(
                    vamp.tcp_pose(configuration).matrix, dtype=np.float64
                )[:3, 3]
                grasp_distance_m = float(
                    np.linalg.norm(tcp_position - grasp_position)
                )
                support_contacts = vamp.semantic_source_support_contacts(
                    configuration,
                    support_geometries,
                    opening_m,
                    request.T_world_base,
                    use_physical_pad_hulls=grasp_distance_m
                    <= _approach_contact_transition_m(request) + 1e-12,
                )
                if non_support_contacts or support_contacts:
                    raise _RouteFailure(
                        PlanningErrorCode.STAGE_COLLISION_VIOLATION,
                        stage,
                        f"time-parameterized {stage.value} robot contacts the environment "
                        f"at current opening {opening_m:.6f}m: "
                        f"non_support={sorted(non_support_contacts)!r}, "
                        f"source_support={sorted(support_contacts)!r}",
                    )
            elif stage in (TrajectoryStage.GRASP, TrajectoryStage.LIFT):
                non_support_contacts = vamp.semantic_target_contacts(
                    configuration,
                    non_support_environment_geometries,
                    opening_m,
                    request.T_world_base,
                )
                if non_support_contacts:
                    raise _RouteFailure(
                        PlanningErrorCode.STAGE_COLLISION_VIOLATION,
                        stage,
                        f"time-parameterized {stage.value} robot contacts a "
                        "non-support environment geometry at current opening "
                        f"{opening_m:.6f}m: {sorted(non_support_contacts)!r}",
                    )

            if stage in (
                TrajectoryStage.PREGRASP,
                TrajectoryStage.APPROACH,
                TrajectoryStage.GRASP,
            ):
                target_contacts = vamp.semantic_target_contacts(
                    configuration,
                    target_geometries,
                    opening_m,
                    request.T_world_base,
                )
                allowed_target_contacts = frozenset()
                if stage is TrajectoryStage.APPROACH:
                    tcp_position = np.asarray(
                        vamp.tcp_pose(configuration).matrix, dtype=np.float64
                    )[:3, 3]
                    grasp_distance_m = float(
                        np.linalg.norm(tcp_position - grasp_position)
                    )
                    if grasp_distance_m <= (
                        _approach_contact_transition_m(request) + 1e-12
                    ):
                        allowed_target_contacts = expected_contacts
                elif stage is TrajectoryStage.GRASP:
                    allowed_target_contacts = expected_contacts
                    final_grasp_contacts = target_contacts
                forbidden = target_contacts - allowed_target_contacts
                if forbidden:
                    raise _RouteFailure(
                        PlanningErrorCode.STAGE_COLLISION_VIOLATION,
                        stage,
                        f"time-parameterized {stage.value} has forbidden target contacts "
                        f"at current opening {opening_m:.6f}m"
                        + (
                            f" and grasp distance {grasp_distance_m:.6f}m"
                            if stage is TrajectoryStage.APPROACH
                            else ""
                        )
                        + f": {sorted(forbidden)!r}",
                    )

            if stage is TrajectoryStage.LIFT:
                attachment_contacts = vamp.semantic_attachment_contacts(
                    configuration,
                    candidate.T_tcp_object,
                    request.object_config.attached_geometry,
                    environment_geometries,
                    request.T_world_base,
                )
                forbidden_attachment = attachment_contacts - {
                    request.source_support_group
                }
                if forbidden_attachment:
                    raise _RouteFailure(
                        PlanningErrorCode.STAGE_COLLISION_VIOLATION,
                        stage,
                        "time-parameterized LIFT attached object has forbidden "
                        f"contacts: {sorted(forbidden_attachment)!r}",
                    )
                on_support = request.source_support_group in attachment_contacts
                if support_departed and on_support:
                    raise _RouteFailure(
                        PlanningErrorCode.STAGE_COLLISION_VIOLATION,
                        stage,
                        "time-parameterized LIFT attached object recontacts the "
                        "declared source support at segment "
                        f"{segment_index}, t={time_s:.6f}s",
                    )
                next_support_departed = support_departed or not on_support
                support_contacts = vamp.semantic_target_contacts(
                    configuration,
                    support_geometries,
                    opening_m,
                    request.T_world_base,
                )
                permitted_support_contacts = (
                    frozenset(request.end_effector_config.contact_groups)
                    if not next_support_departed
                    else frozenset()
                )
                forbidden_support_contacts = (
                    support_contacts - permitted_support_contacts
                )
                if forbidden_support_contacts:
                    raise _RouteFailure(
                        PlanningErrorCode.STAGE_COLLISION_VIOLATION,
                        stage,
                        "time-parameterized LIFT has forbidden robot/source-support "
                        f"contacts: {sorted(forbidden_support_contacts)!r}",
                    )
                support_departed = next_support_departed
            elif stage is TrajectoryStage.GRASP:
                support_contacts = vamp.semantic_target_contacts(
                    configuration,
                    support_geometries,
                    opening_m,
                    request.T_world_base,
                )
                forbidden_support_contacts = support_contacts - frozenset(
                    request.end_effector_config.contact_groups
                )
                if forbidden_support_contacts:
                    raise _RouteFailure(
                        PlanningErrorCode.STAGE_COLLISION_VIOLATION,
                        stage,
                        "time-parameterized GRASP has forbidden robot/source-support "
                        f"contacts: {sorted(forbidden_support_contacts)!r}",
                    )
            previous_time_s = time_s
            if stage_sample_count % 16 == 0:
                _check_stage_budget(
                    request,
                    native_stage_elapsed[stage] + monotonic() - stage_started,
                    total_started,
                    stage,
                )
            stage_sample_count += 1

    if previous_stage is not None:
        _check_stage_budget(
            request,
            native_stage_elapsed[previous_stage] + monotonic() - stage_started,
            total_started,
            previous_stage,
        )

    if not expected_contacts.issubset(final_grasp_contacts):
        raise _RouteFailure(
            PlanningErrorCode.STAGE_COLLISION_VIOLATION,
            TrajectoryStage.GRASP,
            f"time-parameterized GRASP final contacts {sorted(final_grasp_contacts)!r} "
            f"do not include {sorted(expected_contacts)!r}",
        )
    if not support_departed:
        raise _RouteFailure(
            PlanningErrorCode.STAGE_COLLISION_VIOLATION,
            TrajectoryStage.LIFT,
            "time-parameterized LIFT attached object did not depart the declared source support",
        )
    try:
        payload_metrics = _payload_motion_metrics(
            request,
            trajectory,
            vamp,
            total_started,
        )
    except _PayloadMotionSamplingError as exc:
        raise _RouteFailure(
            PlanningErrorCode.BUDGET_EXHAUSTED,
            exc.stage,
            exc.message,
        ) from exc
    payload_violation = _payload_motion_violation(request, payload_metrics)
    if payload_violation is not None:
        stage, message = payload_violation
        raise _RouteFailure(PlanningErrorCode.RUCKIG_FAILED, stage, message)


def _raise_native_trajectory_failure(
    request: PlanningRequest,
    stage: TrajectoryStage,
    samples: list[tuple[int, TrajectoryKnot]],
    environment,
    vamp: VampBackend,
    stage_started: float,
    total_started: float,
) -> None:
    previous_configuration: tuple[float, ...] | None = None
    for sample_index, (segment_index, sample) in enumerate(samples):
        configuration = sample.positions_rad
        if previous_configuration is None:
            if not vamp.configuration_valid(configuration, environment):
                raise _RouteFailure(
                    PlanningErrorCode.STAGE_COLLISION_VIOLATION,
                    stage,
                    f"time-parameterized {stage.value} has an invalid VAMP "
                    f"configuration at t={sample.time_s:.6f}s",
                )
        elif not vamp.motion_valid(
            previous_configuration, configuration, environment
        ):
            raise _RouteFailure(
                PlanningErrorCode.STAGE_COLLISION_VIOLATION,
                stage,
                f"time-parameterized {stage.value} has invalid native robot "
                f"motion in segment {segment_index} before "
                f"t={sample.time_s:.6f}s",
            )
        previous_configuration = configuration
        if sample_index % 16 == 0:
            _check_stage_budget(
                request, monotonic() - stage_started, total_started, stage
            )
    raise _RouteFailure(
        PlanningErrorCode.STAGE_COLLISION_VIOLATION,
        stage,
        f"time-parameterized {stage.value} failed native path validation over "
        "configured-resolution samples",
    )


def _trajectory_segment_validation_samples(
    segment: TrajectorySegment,
    opening_channel: str,
    vamp: VampBackend,
    request: PlanningRequest,
    candidate: GraspCandidate,
    maximum_sample_count: int,
) -> tuple[TrajectoryKnot, ...]:
    policy = request.planning_policy
    if maximum_sample_count < 2:
        raise _RouteFailure(
            PlanningErrorCode.BUDGET_EXHAUSTED,
            segment.stage,
            "time-parameterized trajectory validation exceeded its configured "
            "sample budget",
        )
    opening_rate = abs(
        float(segment.end_actuator_channels[opening_channel])
        - float(segment.start_actuator_channels[opening_channel])
    ) / segment.duration_s
    sample_times: set[float] = set()
    for section in segment.sections:
        maximum_joint_speed = _maximum_section_joint_speed(section)
        maximum_normalized_rate = max(
            maximum_joint_speed / policy.trajectory_validation_joint_step_rad,
            opening_rate
            / policy.trajectory_validation_actuator_position_step_m,
        )
        section_sample_count = max(
            1, int(ceil(maximum_normalized_rate * section.duration_s))
        )
        for index in range(section_sample_count + 1):
            sample_times.add(
                _normalize_segment_sample_time(
                    segment,
                    section.start_time_s
                    + section.duration_s * index / section_sample_count,
                )
            )
        if len(sample_times) > maximum_sample_count:
            raise _RouteFailure(
                PlanningErrorCode.BUDGET_EXHAUSTED,
                segment.stage,
                "time-parameterized trajectory validation exceeded its configured "
                "sample budget",
            )
    if segment.stage in (
        TrajectoryStage.APPROACH,
        TrajectoryStage.GRASP,
        TrajectoryStage.LIFT,
        TrajectoryStage.EGRESS,
        TrajectoryStage.PLACE,
    ):
        sample_times = _refine_trajectory_times_by_swept_motion(
            segment,
            sample_times,
            opening_channel,
            vamp,
            request,
            candidate,
            maximum_sample_count,
        )
    return tuple(segment.sample(time_s) for time_s in sorted(sample_times))


def _normalize_segment_sample_time(
    segment: TrajectorySegment,
    local_time_s: float,
) -> float:
    if local_time_s == nextafter(segment.duration_s, inf):
        return segment.duration_s
    return local_time_s


def _refine_trajectory_times_by_swept_motion(
    segment: TrajectorySegment,
    sample_times: set[float],
    opening_channel: str,
    vamp: VampBackend,
    request: PlanningRequest,
    candidate: GraspCandidate,
    maximum_sample_count: int,
) -> set[float]:
    policy = request.planning_policy
    motion_samples: dict[
        float, tuple[np.ndarray, Rotation, np.ndarray, np.ndarray]
    ] = {}

    def motion_sample(
        time_s: float,
    ) -> tuple[np.ndarray, Rotation, np.ndarray, np.ndarray]:
        if time_s not in motion_samples:
            knot = segment.sample(time_s)
            configuration = knot.positions_rad
            opening_m = float(knot.actuator_channels[opening_channel])
            tcp_matrix = np.asarray(
                vamp.tcp_pose(configuration).matrix, dtype=np.float64
            )
            robot_centers = np.asarray(
                vamp.semantic_robot_sphere_centers(configuration, opening_m),
                dtype=np.float64,
            ).reshape((-1, 3))
            attached_centers = np.empty((0, 3), dtype=np.float64)
            if segment.stage in _PAYLOAD_STAGES:
                attached_centers = np.asarray(
                    vamp.semantic_attachment_sphere_centers(
                        configuration,
                        candidate.T_tcp_object,
                        request.object_config.attached_geometry,
                    ),
                    dtype=np.float64,
                ).reshape((-1, 3))
            motion_samples[time_s] = (
                tcp_matrix[:3, 3],
                Rotation.from_matrix(tcp_matrix[:3, :3]),
                robot_centers,
                attached_centers,
            )
        return motion_samples[time_s]

    while True:
        additions = set()
        ordered = sorted(sample_times)
        for left_time, right_time in zip(ordered, ordered[1:]):
            midpoint_time = 0.5 * (left_time + right_time)
            if midpoint_time in (left_time, right_time):
                left = motion_sample(left_time)
                right = motion_sample(right_time)
                endpoint_translation = float(np.linalg.norm(right[0] - left[0]))
                endpoint_rotation = float(
                    (left[1].inv() * right[1]).magnitude()
                )
                endpoint_robot_sweep = _maximum_point_sweep(
                    left[2], left[2], right[2]
                )
                endpoint_attachment_sweep = _maximum_point_sweep(
                    left[3], left[3], right[3]
                )
                if (
                    endpoint_translation
                    > policy.trajectory_validation_tcp_translation_step_m
                    or endpoint_rotation
                    > policy.trajectory_validation_tcp_rotation_step_rad
                    or endpoint_robot_sweep
                    > policy.trajectory_validation_swept_point_step_m
                    or endpoint_attachment_sweep
                    > policy.trajectory_validation_swept_point_step_m
                ):
                    raise BackendOperationError(
                        f"could not refine time-parameterized {segment.stage.value} "
                        "validation interval: endpoint TCP translation "
                        f"{endpoint_translation:.9g}m, rotation "
                        f"{endpoint_rotation:.9g}rad, robot sphere sweep "
                        f"{endpoint_robot_sweep:.9g}m, attachment sweep "
                        f"{endpoint_attachment_sweep:.9g}m"
                    )
                continue
            left = motion_sample(left_time)
            midpoint = motion_sample(midpoint_time)
            right = motion_sample(right_time)
            tcp_translation_sweep = float(
                np.linalg.norm(midpoint[0] - left[0])
                + np.linalg.norm(right[0] - midpoint[0])
            )
            tcp_rotation_sweep = float(
                (left[1].inv() * midpoint[1]).magnitude()
                + (midpoint[1].inv() * right[1]).magnitude()
            )
            robot_sweep = _maximum_point_sweep(left[2], midpoint[2], right[2])
            attachment_sweep = _maximum_point_sweep(
                left[3], midpoint[3], right[3]
            )
            if (
                tcp_translation_sweep
                > policy.trajectory_validation_tcp_translation_step_m
                or tcp_rotation_sweep
                > policy.trajectory_validation_tcp_rotation_step_rad
                or robot_sweep
                > policy.trajectory_validation_swept_point_step_m
                or attachment_sweep
                > policy.trajectory_validation_swept_point_step_m
            ):
                additions.add(midpoint_time)
        if not additions:
            return sample_times
        if len(sample_times) + len(additions) > maximum_sample_count:
            raise _RouteFailure(
                PlanningErrorCode.BUDGET_EXHAUSTED,
                segment.stage,
                "time-parameterized trajectory validation exceeded its configured "
                "sample budget",
            )
        sample_times.update(additions)


def _maximum_point_sweep(
    left: np.ndarray,
    midpoint: np.ndarray,
    right: np.ndarray,
) -> float:
    if left.shape != midpoint.shape or midpoint.shape != right.shape:
        raise BackendOperationError(
            "VAMP semantic collision sphere count changed during trajectory validation"
        )
    if left.size == 0:
        return 0.0
    return float(
        np.max(
            np.linalg.norm(midpoint - left, axis=1)
            + np.linalg.norm(right - midpoint, axis=1)
        )
    )


def _maximum_section_joint_speed(section: TrajectoryPolynomialSection) -> float:
    maximum = 0.0
    for _, c1, c2, c3 in section.position_coefficients:
        times = [0.0, section.duration_s]
        if abs(c3) > 1e-15:
            extremum = -c2 / (3.0 * c3)
            if 0.0 < extremum < section.duration_s:
                times.append(extremum)
        maximum = max(
            maximum,
            *(abs(c1 + time_s * (2.0 * c2 + time_s * 3.0 * c3)) for time_s in times),
        )
    return maximum


def _has_joint_margin(state: RobotState, request: PlanningRequest) -> bool:
    margin = request.planning_policy.joint_limit_margin_rad
    return all(
        not limits.position_bounded
        or limits.minimum_position_rad + margin
        <= value
        <= limits.maximum_position_rad - margin
        for value, limits in zip(state.positions_rad, request.robot_model.joint_limits)
    )


def _interpolate_poses(
    start: RigidTransform,
    goal: RigidTransform,
    translation_step_m: float,
    rotation_step_rad: float,
    *,
    required_fractions: tuple[float, ...] = (),
) -> tuple[RigidTransform, ...]:
    if start.parent_frame != goal.parent_frame or start.child_frame != goal.child_frame:
        raise ValueError("Cartesian interpolation requires matching frames")
    start_matrix = np.asarray(start.matrix, dtype=np.float64)
    goal_matrix = np.asarray(goal.matrix, dtype=np.float64)
    distance = float(np.linalg.norm(goal_matrix[:3, 3] - start_matrix[:3, 3]))
    rotations = Rotation.from_matrix(np.stack((start_matrix[:3, :3], goal_matrix[:3, :3])))
    angle = float((rotations[0].inv() * rotations[1]).magnitude())
    count = max(1, int(ceil(max(distance / translation_step_m, angle / rotation_step_rad))))
    if any(not 0.0 < fraction < 1.0 for fraction in required_fractions):
        raise ValueError("required Cartesian interpolation fractions must lie in (0, 1)")
    fractions = np.unique(
        np.concatenate(
            (np.linspace(0.0, 1.0, count + 1), np.asarray(required_fractions))
        )
    )
    slerp = Slerp((0.0, 1.0), rotations)
    result = []
    for fraction, rotation in zip(fractions, slerp(fractions).as_matrix()):
        translation = start_matrix[:3, 3] + fraction * (
            goal_matrix[:3, 3] - start_matrix[:3, 3]
        )
        matrix = tuple(
            tuple(float(rotation[row, column]) for column in range(3))
            + (float(translation[row]),)
            for row in range(3)
        ) + ((0.0, 0.0, 0.0, 1.0),)
        result.append(RigidTransform(start.parent_frame, start.child_frame, matrix))  # type: ignore[arg-type]
    return tuple(result)


def _approach_transition_fractions(
    start: RigidTransform,
    goal: RigidTransform,
    transition_distance_m: float,
) -> tuple[float, ...]:
    start_position = np.asarray(start.matrix, dtype=np.float64)[:3, 3]
    goal_position = np.asarray(goal.matrix, dtype=np.float64)[:3, 3]
    distance = float(np.linalg.norm(goal_position - start_position))
    if distance <= transition_distance_m + 1e-12:
        return ()
    return (1.0 - transition_distance_m / distance,)


def _approach_contact_allowances(
    poses: tuple[RigidTransform, ...],
    transition_distance_m: float,
    expected_contacts: frozenset[str],
) -> tuple[_TargetContactAllowance, ...]:
    goal_position = np.asarray(poses[-1].matrix, dtype=np.float64)[:3, 3]
    allowances = []
    for segment_start, segment_end in zip(poses, poses[1:]):
        start_position = np.asarray(segment_start.matrix, dtype=np.float64)[:3, 3]
        end_position = np.asarray(segment_end.matrix, dtype=np.float64)[:3, 3]
        start_remaining = float(np.linalg.norm(goal_position - start_position))
        end_remaining = float(np.linalg.norm(goal_position - end_position))
        allowances.append(
            _TargetContactAllowance(
                expected_contacts
                if end_remaining <= transition_distance_m + 1e-12
                else frozenset(),
                expected_contacts
                if start_remaining <= transition_distance_m + 1e-12
                else frozenset(),
                end_remaining <= transition_distance_m + 1e-12,
                start_remaining <= transition_distance_m + 1e-12,
            )
        )
    return tuple(allowances)


def _translate_world_z(pose: RigidTransform, distance_m: float) -> RigidTransform:
    matrix = [list(row) for row in pose.matrix]
    matrix[2][3] += distance_m
    return RigidTransform(pose.parent_frame, pose.child_frame, tuple(tuple(row) for row in matrix))  # type: ignore[arg-type]


def _vertical_corridor_pose(
    grasp_pose: RigidTransform,
    corridor: VerticalTransitCorridor,
    z_offset_m: float,
    child_frame: str,
    strategy: LocalMotionStrategy,
) -> RigidTransform:
    rotation = np.asarray(grasp_pose.matrix, dtype=np.float64)[:3, :3]
    approach = np.asarray((0.0, 0.0, -1.0), dtype=np.float64)
    closing_column = (
        1 if strategy is LocalMotionStrategy.STAGED_PORTAL else 0
    )
    closing = rotation[:, closing_column] - float(
        np.dot(rotation[:, closing_column], approach)
    ) * approach
    closing_norm = float(np.linalg.norm(closing))
    if closing_norm <= 1e-9:
        raise _RouteFailure(
            PlanningErrorCode.INVALID_REQUEST,
            TrajectoryStage.PREGRASP,
            "vertical transit approach is parallel to the candidate closing axis",
        )
    closing /= closing_norm
    matrix = np.eye(4, dtype=np.float64)
    if strategy is LocalMotionStrategy.STAGED_PORTAL:
        tcp_x = np.cross(closing, approach)
        matrix[:3, :3] = np.column_stack((tcp_x, closing, approach))
    else:
        tcp_y = np.cross(approach, closing)
        matrix[:3, :3] = np.column_stack((closing, tcp_y, approach))
    matrix[:3, 3] = (
        corridor.portal_world_xy_m[0],
        corridor.portal_world_xy_m[1],
        float(grasp_pose.matrix[2][3]) + z_offset_m,
    )
    return RigidTransform(
        grasp_pose.parent_frame,
        child_frame,
        tuple(tuple(float(value) for value in row) for row in matrix),
    )  # type: ignore[arg-type]


def _with_orientation(
    pose: RigidTransform,
    orientation: RigidTransform,
) -> RigidTransform:
    matrix = np.asarray(pose.matrix, dtype=np.float64).copy()
    matrix[:3, :3] = np.asarray(orientation.matrix, dtype=np.float64)[:3, :3]
    return RigidTransform(
        pose.parent_frame,
        pose.child_frame,
        tuple(tuple(float(value) for value in row) for row in matrix),
    )  # type: ignore[arg-type]


def _with_translation(
    pose: RigidTransform,
    translation_m: tuple[float, float, float],
) -> RigidTransform:
    matrix = np.asarray(pose.matrix, dtype=np.float64).copy()
    matrix[:3, 3] = translation_m
    return RigidTransform(
        pose.parent_frame,
        pose.child_frame,
        tuple(tuple(float(value) for value in row) for row in matrix),
    )  # type: ignore[arg-type]


def _sample_placements(
    goal: PlacementGoal,
    count: int,
    *,
    reference_pose: RigidTransform | None = None,
) -> tuple[RigidTransform, ...]:
    if goal.kind is PlacementGoalKind.EXACT:
        return (goal.T_world_tcp,)
    offsets = [(0.0, 0.0, 0.5 * sum(goal.height_range_m))]
    for index in range(1, count):
        fraction = index / max(1, count - 1)
        angle = index * pi * (3.0 - 5.0 ** 0.5)
        if goal.kind is PlacementGoalKind.CIRCLE:
            radius = goal.radius_m * fraction ** 0.5  # type: ignore[operator]
            x, y = radius * cos(angle), radius * sin(angle)
        else:
            width, depth = goal.size_m  # type: ignore[misc]
            x = width * (_radical_inverse(index, 2) - 0.5)
            y = depth * (_radical_inverse(index, 3) - 0.5)
        low, high = goal.height_range_m
        z = low + ((index * 0.6180339887498949) % 1.0) * (high - low)
        offsets.append((x, y, z))
    results = []
    for x, y, z in offsets:
        matrix = [list(row) for row in goal.T_world_tcp.matrix]
        matrix[0][3] += x
        matrix[1][3] += y
        matrix[2][3] += z
        results.append(
            RigidTransform(
                goal.T_world_tcp.parent_frame,
                goal.T_world_tcp.child_frame,
                tuple(tuple(row) for row in matrix),  # type: ignore[arg-type]
            )
        )
    if reference_pose is None:
        return tuple(results)

    projected = _project_pose_to_placement_region(goal, reference_pose)
    ordered = [projected]
    ordered.extend(
        sorted(
            results,
            key=lambda pose: _translation_distance(pose, reference_pose),
        )
    )
    unique = []
    for pose in ordered:
        if any(_same_position(pose, existing) for existing in unique):
            continue
        unique.append(pose)
        if len(unique) == count:
            break
    return tuple(unique)


def _project_pose_to_placement_region(
    goal: PlacementGoal,
    reference_pose: RigidTransform,
) -> RigidTransform:
    goal_matrix = np.asarray(goal.T_world_tcp.matrix, dtype=np.float64)
    reference_position = np.asarray(reference_pose.matrix, dtype=np.float64)[:3, 3]
    anchor = goal_matrix[:3, 3]
    offset = reference_position - anchor
    if goal.kind is PlacementGoalKind.RECTANGLE:
        width, depth = goal.size_m  # type: ignore[misc]
        offset[0] = np.clip(offset[0], -0.5 * width, 0.5 * width)
        offset[1] = np.clip(offset[1], -0.5 * depth, 0.5 * depth)
    elif goal.kind is PlacementGoalKind.CIRCLE:
        radius = goal.radius_m  # type: ignore[assignment]
        radial_distance = float(np.linalg.norm(offset[:2]))
        if radial_distance > radius:
            offset[:2] *= radius / radial_distance
    else:
        raise ValueError("placement projection requires a regional goal")
    offset[2] = np.clip(offset[2], *goal.height_range_m)
    matrix = [list(row) for row in goal.T_world_tcp.matrix]
    for index, value in enumerate(anchor + offset):
        matrix[index][3] = float(value)
    return RigidTransform(
        goal.T_world_tcp.parent_frame,
        goal.T_world_tcp.child_frame,
        tuple(tuple(row) for row in matrix),  # type: ignore[arg-type]
    )


def _translation_distance(left: RigidTransform, right: RigidTransform) -> float:
    left_position = np.asarray(left.matrix, dtype=np.float64)[:3, 3]
    right_position = np.asarray(right.matrix, dtype=np.float64)[:3, 3]
    return float(np.linalg.norm(right_position - left_position))


def _same_position(left: RigidTransform, right: RigidTransform) -> bool:
    return _translation_distance(left, right) <= 1e-12


def _split_final_path_motion(
    path: tuple[tuple[float, ...], ...],
    maximum_joint_step_rad: float,
) -> (
    tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]
    | None
):
    compact = [path[0]] if path else []
    for point in path[1:]:
        if point != compact[-1]:
            compact.append(point)
    if not compact:
        raise BackendOperationError("VAMP regional placement route is empty")
    if len(compact) == 1:
        return None
    left = np.asarray(compact[-2], dtype=np.float64)
    right = np.asarray(compact[-1], dtype=np.float64)
    maximum_delta = float(np.max(np.abs(right - left)))
    fraction = min(0.5, maximum_joint_step_rad / maximum_delta)
    split = tuple(float(value) for value in right - fraction * (right - left))
    egress = tuple(compact[:-1]) + (split,)
    place = (split, tuple(compact[-1]))
    return egress, place


def _resolve_placement_orientation(
    goal: PlacementGoal,
    sampled_pose: RigidTransform,
    post_grasp_tcp_pose: RigidTransform,
) -> RigidTransform:
    if goal.orientation_constraint is PlacementOrientationConstraint.EXACT:
        return sampled_pose
    if sampled_pose.parent_frame != post_grasp_tcp_pose.parent_frame:
        raise ValueError("placement and post-grasp TCP poses must share a parent frame")
    matrix = [list(row) for row in sampled_pose.matrix]
    for row in range(3):
        for column in range(3):
            matrix[row][column] = post_grasp_tcp_pose.matrix[row][column]
    result = RigidTransform(
        sampled_pose.parent_frame,
        sampled_pose.child_frame,
        tuple(tuple(row) for row in matrix),  # type: ignore[arg-type]
    )
    result.validate(label="orientation-free placement pose")
    return result


def _radical_inverse(index: int, base: int) -> float:
    value = 0.0
    factor = 1.0 / base
    while index:
        value += factor * (index % base)
        index //= base
        factor /= base
    return value


def _route_selection_key(
    route: _Route,
    request: PlanningRequest,
    search_attempt: int,
) -> tuple[float, ...]:
    positions = tuple(item.positions_rad for item in route.waypoints)
    length = sum(
        float(np.linalg.norm(np.asarray(right) - np.asarray(left)))
        for left, right in zip(positions, positions[1:])
    )
    minimum_margin = min(
        (
            min(
                value - limits.minimum_position_rad,
                limits.maximum_position_rad - value,
            )
            for position in positions
            for value, limits in zip(position, request.robot_model.joint_limits)
            if limits.position_bounded
        ),
        default=float("inf"),
    )
    if (
        request.placement_goal.orientation_constraint
        is PlacementOrientationConstraint.FREE
    ):
        key = _payload_route_selection_key(
            route.payload_nominal_duration_s,
            route.payload_cumulative_rotation_rad,
            route.payload_cumulative_translation_m,
            route.orientation_preference_cost_rad,
            length,
            search_attempt,
        )
        return (*key[:-1], -minimum_margin, key[-1])
    return length, -minimum_margin, search_attempt


def _check_stage_budget(
    request: PlanningRequest,
    stage_elapsed_s: float,
    total_started: float,
    stage: TrajectoryStage,
) -> None:
    if stage_elapsed_s > request.budgets.stage_timeout_s:
        raise _RouteFailure(
            PlanningErrorCode.BUDGET_EXHAUSTED,
            stage,
            f"{stage.value} exceeded its explicit time budget",
        )
    _check_total_budget(request, total_started, stage)


def _check_total_budget(
    request: PlanningRequest,
    total_started: float,
    stage: TrajectoryStage,
) -> None:
    if monotonic() - total_started > request.budgets.total_timeout_s:
        raise _RouteFailure(
            PlanningErrorCode.BUDGET_EXHAUSTED,
            stage,
            "Planning total time budget was exhausted",
        )


def _metadata(
    request: PlanningRequest,
    vamp_version: str,
    ik_version: str,
    ruckig_version: str,
) -> dict[str, object]:
    return {
        "backend_versions": {
            "vamp": vamp_version,
            "ik": f"curobo-{ik_version}",
            "ruckig": ruckig_version,
        },
        "budgets": {
            "maximum_candidates": request.budgets.maximum_candidates,
            "search_attempts_per_candidate": request.budgets.search_attempts_per_candidate,
            "stage_timeout_s": request.budgets.stage_timeout_s,
            "total_timeout_s": request.budgets.total_timeout_s,
        },
        "parameters": {
            "trajectory_collision_validation": "configured_resolution_sampling",
            "cartesian_translation_step_m": request.planning_policy.cartesian_translation_step_m,
            "cartesian_rotation_step_rad": request.planning_policy.cartesian_rotation_step_rad,
            "pregrasp_joint_path_detour_ratio_limit": request.planning_policy.pregrasp_joint_path_detour_ratio_limit,
            "payload_tcp_max_angular_velocity_rad_s": request.planning_policy.payload_tcp_max_angular_velocity_rad_s,
            "payload_tcp_max_translation_velocity_m_s": request.planning_policy.payload_tcp_max_translation_velocity_m_s,
            "payload_tcp_max_translation_acceleration_m_s2": request.planning_policy.payload_tcp_max_translation_acceleration_m_s2,
            "payload_motion_validation_time_step_s": request.planning_policy.payload_motion_validation_time_step_s,
            "trajectory_validation_joint_step_rad": request.planning_policy.trajectory_validation_joint_step_rad,
            "trajectory_validation_tcp_translation_step_m": request.planning_policy.trajectory_validation_tcp_translation_step_m,
            "trajectory_validation_tcp_rotation_step_rad": request.planning_policy.trajectory_validation_tcp_rotation_step_rad,
            "trajectory_validation_actuator_position_step_m": request.planning_policy.trajectory_validation_actuator_position_step_m,
            "trajectory_validation_swept_point_step_m": request.planning_policy.trajectory_validation_swept_point_step_m,
            "trajectory_validation_max_samples": request.planning_policy.trajectory_validation_max_samples,
        },
    }
