from __future__ import annotations

from dataclasses import dataclass
from math import ceil, cos, pi, sin
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
    PlacementGoal,
    PlacementGoalKind,
    PlannedTrajectory,
    PlanningAttempt,
    PlanningErrorCode,
    PlanningRequest,
    PlanningResult,
    RigidTransform,
    RobotState,
    SceneGeometry,
    TrajectoryKnot,
    TrajectoryPolynomialSection,
    TrajectorySegment,
    TrajectoryStage,
)
from .contracts.geometry import compose, inverse, reframe
from .ik import IKConstraints, IKErrorCode, IKSolver, validate_ik_result


@dataclass(frozen=True)
class _RouteFailure(Exception):
    code: PlanningErrorCode
    stage: TrajectoryStage
    message: str


@dataclass(frozen=True)
class _Route:
    waypoints: tuple[GeometricWaypoint, ...]
    events: tuple[tuple[str, TrajectoryStage, int, dict[str, float]], ...]
    T_world_object_goal: RigidTransform


@dataclass(frozen=True)
class _TargetContactAllowance:
    endpoint: frozenset[str]
    motion: frozenset[str]


@dataclass(frozen=True)
class PlanningDependencies:
    create_vamp: Callable[[object], VampBackend]
    create_ik: Callable[[object, int], IKSolver]
    create_ruckig: Callable[[object], RuckigTimeParameterizer]
    ik_solution_count: int
    placement_sample_count: int

    def validate(self) -> None:
        if self.ik_solution_count <= 0 or self.placement_sample_count <= 0:
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
        ik_solver = dependencies.create_ik(request.robot_model, request.random_seed)
        ruckig = dependencies.create_ruckig(request.robot_model)
    except BackendUnavailableError as exc:
        return PlanningResult.failed(PlanningErrorCode.BACKEND_UNAVAILABLE, str(exc))
    except (RuntimeError, ValueError, OSError) as exc:
        return PlanningResult.failed(PlanningErrorCode.BACKEND_ERROR, str(exc))

    started = monotonic()
    attempts: list[PlanningAttempt] = []
    candidates = request.candidates[: request.budgets.maximum_candidates]
    last_failure: _RouteFailure | None = None
    for candidate in candidates:
        candidate_routes: list[tuple[int, _Route]] = []
        for seed_offset in range(request.budgets.seeds_per_candidate):
            seed = request.random_seed + seed_offset
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
                route = _build_route(
                    request,
                    candidate,
                    seed,
                    ik_solver,
                    vamp,
                    attempts,
                    started,
                    dependencies.ik_solution_count,
                    dependencies.placement_sample_count,
                )
            except _RouteFailure as failure:
                last_failure = failure
                attempts.append(
                    PlanningAttempt(
                        candidate.candidate_id,
                        seed,
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
            candidate_routes.append((seed, route))
        if not candidate_routes:
            continue
        for selected_seed, route in sorted(
            candidate_routes,
            key=lambda item: _route_selection_key(item[1], request, item[0]),
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
                        selected_seed,
                        exc.stage,
                        monotonic() - parameterization_started,
                        PlanningErrorCode.RUCKIG_FAILED,
                        str(exc),
                    )
                )
                continue
            except BackendOperationError as exc:
                return PlanningResult.failed(
                    PlanningErrorCode.RUCKIG_FAILED,
                    str(exc),
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
                        selected_seed,
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
            metadata["selected_seed"] = selected_seed
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


def _build_route(
    request: PlanningRequest,
    candidate: GraspCandidate,
    seed: int,
    ik_solver: IKSolver,
    vamp: VampBackend,
    attempts: list[PlanningAttempt],
    total_started: float,
    ik_solution_count: int,
    placement_sample_count: int,
) -> _Route:
    target_geometries = tuple(
        item for item in request.scene.geometries if item.object_id == request.target_object_id
    )
    environment_geometries = tuple(
        item
        for item in request.scene.geometries
        if item.object_id != request.target_object_id
    )
    channels = {
        request.end_effector_config.actuator_position_channel: candidate.grasp_command.pregrasp_opening_m,
        request.end_effector_config.actuator_force_limit_channel: candidate.grasp_command.grip_force_limit_n,
    }
    waypoints: list[GeometricWaypoint] = []
    initial = request.initial_state.positions_rad
    pregrasp_pose = compose(
        inverse(request.T_world_base),
        reframe(candidate.T_world_pregrasp, child_frame=request.robot_model.tcp_frame),
    )
    pregrasp_solutions = _solve_ik(
        ik_solver,
        request,
        pregrasp_pose,
        request.initial_state,
        TrajectoryStage.PREGRASP,
        ik_solution_count,
    )
    free_environment = vamp.environment(
        request.scene,
        T_world_base=request.T_world_base,
        excluded_object_ids=(request.target_object_id,),
    )
    pregrasp_path = None
    pregrasp_elapsed = 0.0
    for solution in pregrasp_solutions:
        if not _has_joint_margin(solution, request):
            continue
        stage_started = monotonic()
        try:
            planned = vamp.plan(initial, solution.positions_rad, free_environment, seed=seed)
        except BackendOperationError:
            continue
        pregrasp_elapsed = monotonic() - stage_started
        _check_stage_budget(request, pregrasp_elapsed, total_started, TrajectoryStage.PREGRASP)
        if _path_has_forbidden_target_contact(
            planned.path,
            target_geometries,
            candidate.grasp_command.pregrasp_opening_m,
            vamp,
            request.T_world_base,
        ):
            continue
        pregrasp_path = planned.path
        break
    if pregrasp_path is None:
        raise _RouteFailure(
            PlanningErrorCode.VAMP_SEARCH_FAILED,
            TrajectoryStage.PREGRASP,
            "VAMP found no collision-valid PREGRASP route for this candidate and seed",
        )
    attempts.append(
        PlanningAttempt(
            candidate.candidate_id,
            seed,
            TrajectoryStage.PREGRASP,
            pregrasp_elapsed,
            None,
            "VAMP PREGRASP succeeded",
        )
    )
    waypoints.extend(
        GeometricWaypoint(TrajectoryStage.PREGRASP, point, dict(channels))
        for point in pregrasp_path
    )
    del free_environment

    grasp_pose = compose(
        inverse(request.T_world_base),
        reframe(candidate.T_world_tcp, child_frame=request.robot_model.tcp_frame),
    )
    interpolated_approach = _interpolate_poses(
        pregrasp_pose,
        grasp_pose,
        request.planning_policy.cartesian_translation_step_m,
        request.planning_policy.cartesian_rotation_step_rad,
        required_fractions=_approach_transition_fractions(
            pregrasp_pose,
            grasp_pose,
            request.planning_policy.approach_contact_transition_m,
        ),
    )
    approach_poses = interpolated_approach[1:]
    approach_contact_allowances = _approach_contact_allowances(
        interpolated_approach,
        request.planning_policy.approach_contact_transition_m,
        frozenset(candidate.expected_contacts),
    )
    approach_states = _continuous_ik(
        request,
        candidate,
        ik_solver,
        vamp,
        tuple(approach_poses),
        RobotState(request.robot_model.joint_names, pregrasp_path[-1]),
        target_geometries,
        environment_geometries,
        approach_contact_allowances,
        TrajectoryStage.APPROACH,
        total_started,
        ik_solution_count,
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

    lift_pose = compose(
        inverse(request.T_world_base),
        _translate_world_z(
            reframe(candidate.T_world_tcp, child_frame=request.robot_model.tcp_frame),
            request.planning_policy.lift_distance_m,
        ),
    )
    lift_poses = _interpolate_poses(
        grasp_pose,
        lift_pose,
        request.planning_policy.cartesian_translation_step_m,
        request.planning_policy.cartesian_rotation_step_rad,
    )[1:]
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
        GeometricWaypoint(TrajectoryStage.LIFT, state.positions_rad, dict(channels))
        for state in lift_states
    )

    strict_loaded_environment = vamp.environment(
        request.scene,
        T_world_base=request.T_world_base,
        excluded_object_ids=(request.target_object_id,),
        T_tcp_object=candidate.T_tcp_object,
        attached_geometry=request.object_config.attached_geometry,
    )
    egress_start = lift_states[-1].positions_rad
    placement_failure: _RouteFailure | None = None
    for placement_pose in _sample_placements(request.placement_goal, placement_sample_count):
        world_placement_pose = placement_pose
        preplace_pose = compose(
            inverse(request.T_world_base),
            _translate_world_z(placement_pose, request.planning_policy.preplace_offset_m),
        )
        placement_pose = compose(inverse(request.T_world_base), placement_pose)
        try:
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
                seed,
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
                seed,
                TrajectoryStage.PLACE,
                attempts,
                total_started,
            )
            if _path_has_forbidden_attachment_contact(
                egress_plan,
                request,
                candidate,
                vamp,
                environment_geometries,
            ):
                raise _RouteFailure(
                    PlanningErrorCode.STAGE_COLLISION_VIOLATION,
                    TrajectoryStage.EGRESS,
                    "attached object has a forbidden contact during EGRESS",
                )
            if _path_has_forbidden_attachment_contact(
                place_plan,
                request,
                candidate,
                vamp,
                environment_geometries,
            ):
                raise _RouteFailure(
                    PlanningErrorCode.STAGE_COLLISION_VIOLATION,
                    TrajectoryStage.PLACE,
                    "attached object has a forbidden contact during PLACE",
                )
        except _RouteFailure as failure:
            placement_failure = failure
            continue
        waypoints.extend(
            GeometricWaypoint(TrajectoryStage.EGRESS, point, dict(channels))
            for point in egress_plan[1:]
        )
        waypoints.extend(
            GeometricWaypoint(
                TrajectoryStage.PLACE,
                point,
                dict(channels),
                stop=index == len(place_plan[1:]) - 1,
            )
            for index, point in enumerate(place_plan[1:])
        )
        released_channels = dict(channels)
        released_channels[request.end_effector_config.actuator_position_channel] = (
            request.end_effector_config.open_position_m
        )
        release_index = len(waypoints) - 1
        T_world_object = compose(
            world_placement_pose,
            reframe(
                candidate.T_tcp_object,
                parent_frame=world_placement_pose.child_frame,
                child_frame="object",
            ),
        )
        return _Route(
            tuple(waypoints),
            (
                ("grip", TrajectoryStage.GRASP, grip_index, dict(channels)),
                ("release", TrajectoryStage.PLACE, release_index, released_channels),
            ),
            T_world_object,
        )
    if placement_failure is not None:
        raise placement_failure
    raise _RouteFailure(
        PlanningErrorCode.IK_CANDIDATES_EXHAUSTED,
        TrajectoryStage.PLACE,
        "no deterministic placement sample was feasible",
    )


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
    started = monotonic()
    for index, pose in enumerate(poses):
        solutions = _solve_ik(
            ik_solver, request, pose, current, stage, ik_solution_count
        )
        selected = None
        for state in solutions:
            if not _has_joint_margin(state, request):
                continue
            exact_environment_geometries = environment_geometries
            exact_opening = (
                candidate.grasp_command.pregrasp_opening_m
                if stage is TrajectoryStage.APPROACH
                else candidate.grasp_command.target_opening_m
            )
            if exact_environment_geometries and vamp.semantic_target_contacts(
                state.positions_rad,
                exact_environment_geometries,
                exact_opening,
                request.T_world_base,
            ):
                continue
            if exact_environment_geometries and _semantic_motion_has_forbidden_contact(
                current.positions_rad,
                state.positions_rad,
                exact_environment_geometries,
                exact_opening,
                frozenset(),
                vamp,
                request.T_world_base,
            ):
                continue
            environment = strict_environment
            next_support_departed = support_departed
            if stage is TrajectoryStage.LIFT:
                attachment_contacts = vamp.semantic_attachment_contacts(
                    state.positions_rad,
                    candidate.T_tcp_object,
                    request.object_config.attached_geometry,
                    environment_geometries,
                    request.T_world_base,
                )
                if attachment_contacts - {request.source_support_group}:
                    continue
                if support_departed and request.source_support_group in attachment_contacts:
                    continue
                next_support_departed = _attachment_motion_is_valid(
                    current.positions_rad,
                    state.positions_rad,
                    support_departed,
                    request,
                    candidate,
                    vamp,
                    environment_geometries,
                )
                if next_support_departed is None:
                    continue
                if request.source_support_group not in attachment_contacts:
                    next_support_departed = True
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
) -> tuple[RobotState, ...]:
    result = solver.solve(
        (pose,),
        (seed_state,),
        IKConstraints(
            request.planning_policy.ik_position_tolerance_m,
            request.planning_policy.ik_orientation_tolerance_rad,
            maximum_solutions,
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
    return tuple(item.state for item in result.solutions)


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
    seed: int,
    stage: TrajectoryStage,
    attempts: list[PlanningAttempt],
    total_started: float,
) -> tuple[tuple[float, ...], ...]:
    started = monotonic()
    try:
        result = vamp.plan(start, goal, environment, seed=seed)
    except BackendOperationError as exc:
        raise _RouteFailure(PlanningErrorCode.VAMP_SEARCH_FAILED, stage, str(exc)) from exc
    elapsed = monotonic() - started
    _check_stage_budget(request, elapsed, total_started, stage)
    attempts.append(
        PlanningAttempt(candidate_id, seed, stage, elapsed, None, f"VAMP {stage.value} succeeded")
    )
    return result.path


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


def _attachment_motion_is_valid(
    start: tuple[float, ...],
    goal: tuple[float, ...],
    support_departed: bool,
    request: PlanningRequest,
    candidate: GraspCandidate,
    vamp: VampBackend,
    environment_geometries: tuple[SceneGeometry, ...],
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
        if not on_support:
            departed = True
    return departed


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
            if stage in (
                TrajectoryStage.APPROACH,
                TrajectoryStage.GRASP,
                TrajectoryStage.LIFT,
            ):
                environment_contacts = vamp.semantic_target_contacts(
                    configuration,
                    environment_geometries,
                    opening_m,
                    request.T_world_base,
                )
                if environment_contacts:
                    raise _RouteFailure(
                        PlanningErrorCode.STAGE_COLLISION_VIOLATION,
                        stage,
                        f"time-parameterized {stage.value} robot contacts the environment "
                        f"at current opening {opening_m:.6f}m: "
                        f"{sorted(environment_contacts)!r}",
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
                        request.planning_policy.approach_contact_transition_m + 1e-12
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
                        "declared source support",
                    )
                if not on_support:
                    support_departed = True
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
                section.start_time_s
                + section.duration_s * index / section_sample_count
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
            if segment.stage is TrajectoryStage.LIFT:
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
        limits.minimum_position_rad + margin <= value <= limits.maximum_position_rad - margin
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
            )
        )
    return tuple(allowances)


def _translate_world_z(pose: RigidTransform, distance_m: float) -> RigidTransform:
    matrix = [list(row) for row in pose.matrix]
    matrix[2][3] += distance_m
    return RigidTransform(pose.parent_frame, pose.child_frame, tuple(tuple(row) for row in matrix))  # type: ignore[arg-type]


def _sample_placements(goal: PlacementGoal, count: int) -> tuple[RigidTransform, ...]:
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
    return tuple(results)


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
    seed: int,
) -> tuple[float, float, int]:
    positions = tuple(item.positions_rad for item in route.waypoints)
    length = sum(
        float(np.linalg.norm(np.asarray(right) - np.asarray(left)))
        for left, right in zip(positions, positions[1:])
    )
    minimum_margin = min(
        min(
            value - limits.minimum_position_rad,
            limits.maximum_position_rad - value,
        )
        for position in positions
        for value, limits in zip(position, request.robot_model.joint_limits)
    )
    return length, -minimum_margin, seed


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
        "random_seed": request.random_seed,
        "budgets": {
            "maximum_candidates": request.budgets.maximum_candidates,
            "seeds_per_candidate": request.budgets.seeds_per_candidate,
            "stage_timeout_s": request.budgets.stage_timeout_s,
            "total_timeout_s": request.budgets.total_timeout_s,
        },
        "parameters": {
            "trajectory_collision_validation": "configured_resolution_sampling",
            "cartesian_translation_step_m": request.planning_policy.cartesian_translation_step_m,
            "cartesian_rotation_step_rad": request.planning_policy.cartesian_rotation_step_rad,
            "trajectory_validation_joint_step_rad": request.planning_policy.trajectory_validation_joint_step_rad,
            "trajectory_validation_tcp_translation_step_m": request.planning_policy.trajectory_validation_tcp_translation_step_m,
            "trajectory_validation_tcp_rotation_step_rad": request.planning_policy.trajectory_validation_tcp_rotation_step_rad,
            "trajectory_validation_actuator_position_step_m": request.planning_policy.trajectory_validation_actuator_position_step_m,
            "trajectory_validation_swept_point_step_m": request.planning_policy.trajectory_validation_swept_point_step_m,
            "trajectory_validation_max_samples": request.planning_policy.trajectory_validation_max_samples,
        },
    }
