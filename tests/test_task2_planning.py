from __future__ import annotations

from dataclasses import replace
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from task2 import plan, target
from task2.backends.errors import BackendUnavailableError
from task2.backends.ruckig import RuckigRouteError, RuckigTimeParameterizer
from task2.backends.vamp import (
    VampPlan,
    _convex_contact_groups,
    _outward_convex_faces,
    _sphere_intersects_geometry,
)
from task2.config import load_task2_config
from task2.contracts import (
    AllowedContact,
    Box,
    ConvexHull,
    PlacementGoal,
    PlacementGoalKind,
    PlanningBudgets,
    PlanningErrorCode,
    PlanningRequest,
    RigidTransform,
    RobotState,
    SceneGeometry,
    SceneSnapshot,
    Sphere,
    TargetingRequest,
    TrajectoryStage,
)
from task2.ik import IKResult, IKSolution
from task2.planning import PlanningDependencies


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_task2_config(ROOT / "configs/task2/pipeline.yaml")
PROFILE = CONFIG.end_effectors[0]


class _FakeVamp:
    version = "0.6.4"

    def __init__(self, robot_model):
        self.model = robot_model

    def environment(self, scene, **kwargs):
        return kwargs

    def plan(self, start, goal, environment, *, seed):
        return VampPlan((tuple(start), tuple(goal)), 0.001, 1)

    def configuration_valid(self, configuration, environment):
        return True

    def motion_valid(self, start, goal, environment):
        return True

    def path_valid(self, configurations, environment):
        if not configurations:
            return False
        if not self.configuration_valid(configurations[0], environment):
            return False
        return all(
            self.motion_valid(left, right, environment)
            for left, right in zip(configurations, configurations[1:])
        )

    def tcp_pose(self, configuration):
        return _pose(
            self.model.base_frame,
            self.model.tcp_frame,
            tuple(value * 10.0 for value in configuration[:3]),
        )

    def semantic_robot_sphere_centers(self, configuration, opening_m):
        return (tuple(self.tcp_pose(configuration).matrix[index][3] for index in range(3)),)

    def semantic_attachment_sphere_centers(
        self, configuration, T_tcp_object, attached_geometry
    ):
        return self.semantic_robot_sphere_centers(configuration, 0.0)

    def semantic_target_contacts(self, configuration, geometries, opening_m, T_world_base):
        if any(item.collision_group == "target" for item in geometries) and opening_m < 0.02:
            return frozenset(("left_pad", "right_pad"))
        return frozenset()

    def semantic_attachment_contacts(
        self,
        configuration,
        T_tcp_object,
        attached_geometry,
        scene_geometries,
        T_world_base,
    ):
        return frozenset()


class _FakeIK:
    version = "0.8.0"

    def __init__(self, robot_model):
        self.model = robot_model

    def solve(self, targets, seed_states, constraints):
        matrix = targets[0].matrix
        values = list(seed_states[0].positions_rad)
        values[0] = 0.1 * matrix[0][3]
        values[1] = 0.1 * matrix[1][3]
        values[2] = 0.1 * matrix[2][3]
        state = RobotState(self.model.joint_names, tuple(values))
        return IKResult((IKSolution(state, 0.0, 0.0),), None, "fake IK succeeded")


class _ApproachContactVamp(_FakeVamp):
    def __init__(self, robot_model, grasp_position_m, contact_distance_m):
        super().__init__(robot_model)
        self.grasp_position_m = np.asarray(grasp_position_m, dtype=np.float64)
        self.contact_distance_m = contact_distance_m

    def semantic_target_contacts(self, configuration, geometries, opening_m, T_world_base):
        tcp_position = np.asarray(
            self.tcp_pose(configuration).matrix, dtype=np.float64
        )[:3, 3]
        if (
            any(item.collision_group == "target" for item in geometries)
            and opening_m >= 0.02
            and np.linalg.norm(tcp_position - self.grasp_position_m)
            <= self.contact_distance_m
        ):
            return frozenset(("left_pad", "right_pad"))
        return super().semantic_target_contacts(
            configuration, geometries, opening_m, T_world_base
        )


class _FinalTrajectoryVamp(_FakeVamp):
    def __init__(self, robot_model, *, reject_motion=False, reject_attachment=False):
        super().__init__(robot_model)
        self.final_validation = False
        self.reject_motion = reject_motion
        self.reject_attachment = reject_attachment
        self.final_target_openings = []

    def motion_valid(self, start, goal, environment):
        if self.final_validation and self.reject_motion and start != goal:
            return False
        return super().motion_valid(start, goal, environment)

    def semantic_target_contacts(self, configuration, geometries, opening_m, T_world_base):
        if self.final_validation and any(
            item.collision_group == "target" for item in geometries
        ):
            self.final_target_openings.append(opening_m)
        return super().semantic_target_contacts(
            configuration, geometries, opening_m, T_world_base
        )

    def semantic_attachment_contacts(
        self,
        configuration,
        T_tcp_object,
        attached_geometry,
        scene_geometries,
        T_world_base,
    ):
        if self.final_validation and self.reject_attachment:
            return frozenset(("tank_environment",))
        return super().semantic_attachment_contacts(
            configuration,
            T_tcp_object,
            attached_geometry,
            scene_geometries,
            T_world_base,
        )


class _FinalValidationRuckig:
    def __init__(self, delegate, vamp):
        self._delegate = delegate
        self._vamp = vamp
        self.version = delegate.version

    def parameterize(self, waypoints, events):
        trajectory = self._delegate.parameterize(waypoints, events)
        self._vamp.final_validation = True
        return trajectory


class _FirstRouteRejectingRuckig:
    def __init__(self, delegate):
        self._delegate = delegate
        self.version = delegate.version
        self.calls = 0

    def parameterize(self, waypoints, events):
        self.calls += 1
        if self.calls == 1:
            raise RuckigRouteError(
                TrajectoryStage.PREGRASP,
                "Ruckig rejected the first geometric route",
            )
        return self._delegate.parameterize(waypoints, events)


def _pose(parent: str, child: str, xyz=(0.0, 0.0, 0.0)) -> RigidTransform:
    return RigidTransform(
        parent,
        child,
        (
            (1.0, 0.0, 0.0, xyz[0]),
            (0.0, 1.0, 0.0, xyz[1]),
            (0.0, 0.0, 1.0, xyz[2]),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )


def _request(object_type: str = "marker") -> PlanningRequest:
    robot_model = CONFIG.robot_model(PROFILE)
    object_xyz = (0.5, 0.0, 0.1)
    T_world_base = _pose("world", robot_model.base_frame)
    placement = _pose("world", robot_model.tcp_frame, (0.6, 0.0, 0.6))
    if object_type == "standard_part":
        object_xyz = (0.39, 0.66, 0.0035)
        T_world_base = RigidTransform(
            "world",
            robot_model.base_frame,
            (
                (0.0, 1.0, 0.0, 0.43),
                (-1.0, 0.0, 0.0, 0.95),
                (0.0, 0.0, 1.0, 0.5),
                (0.0, 0.0, 0.0, 1.0),
            ),
        )
        placement = RigidTransform(
            "world",
            robot_model.tcp_frame,
            (
                (0.0, 1.0, 0.0, 0.55),
                (1.0, 0.0, 0.0, 0.75),
                (0.0, 0.0, -1.0, 0.62),
                (0.0, 0.0, 0.0, 1.0),
            ),
        )
    targeting_result = target(
        TargetingRequest(
            object_type,
            _pose("world", "object", object_xyz),
            CONFIG.targeting_object(object_type),
            PROFILE.targeting,
            0.5,
            8,
        )
    )
    assert targeting_result.success
    scene = SceneSnapshot(
        "planning_regression",
        "world",
        (
            SceneGeometry(
                "target_sphere",
                "target_object",
                "target",
                _pose("world", "target_sphere", object_xyz),
                Sphere(0.01),
            ),
            SceneGeometry(
                "floor",
                "tank",
                "source_support",
                _pose("world", "floor", (0.0, 0.0, -1.0)),
                Box((2.0, 2.0, 0.1)),
            ),
        ),
        (AllowedContact("target", "source_support"),),
    )
    return PlanningRequest(
        f"planning_{object_type}",
        targeting_result.candidates,
        RobotState(robot_model.joint_names, robot_model.default_joint_positions),
        robot_model,
        T_world_base,
        scene,
        "target_object",
        "source_support",
        PlacementGoal(
            "outside",
            PlacementGoalKind.EXACT,
            placement,
        ),
        CONFIG.planning_object(object_type),
        PROFILE.planning,
        CONFIG.planning_policy,
        7,
        PlanningBudgets(1, 1, 2.0, 10.0),
    )


def _fake_dependencies() -> PlanningDependencies:
    backend = CONFIG.planning_backends[PROFILE.planning_backend_profile]
    return PlanningDependencies(
        lambda robot_model: _FakeVamp(robot_model),
        lambda robot_model, random_seed: _FakeIK(robot_model),
        lambda robot_model: RuckigTimeParameterizer(robot_model, backend.ruckig),
        backend.ik_solution_count,
        backend.placement_sample_count,
    )


def _approach_contact_dependencies(contact_distance_m: float) -> PlanningDependencies:
    request = _request()
    grasp_position = np.asarray(
        request.candidates[0].T_world_tcp.matrix, dtype=np.float64
    )[:3, 3]
    return replace(
        _fake_dependencies(),
        create_vamp=lambda robot_model: _ApproachContactVamp(
            robot_model, grasp_position, contact_distance_m
        ),
    )


def _final_validation_dependencies(
    *, reject_motion=False, reject_attachment=False
) -> tuple[PlanningDependencies, _FinalTrajectoryVamp]:
    backend = CONFIG.planning_backends[PROFILE.planning_backend_profile]
    request = _request()
    vamp = _FinalTrajectoryVamp(
        request.robot_model,
        reject_motion=reject_motion,
        reject_attachment=reject_attachment,
    )
    dependencies = PlanningDependencies(
        lambda robot_model: vamp,
        lambda robot_model, random_seed: _FakeIK(robot_model),
        lambda robot_model: _FinalValidationRuckig(
            RuckigTimeParameterizer(robot_model, backend.ruckig), vamp
        ),
        backend.ik_solution_count,
        backend.placement_sample_count,
    )
    return dependencies, vamp


def test_plan_preserves_six_phases_named_channels_and_stops() -> None:
    request = _request()
    result = plan(request, dependencies=_fake_dependencies())
    assert result.success, result.message
    assert result.trajectory is not None
    trajectory = result.trajectory
    assert tuple(phase.stage for phase in trajectory.phases) == tuple(TrajectoryStage)
    assert tuple(event.event for event in trajectory.events) == ("grip", "release")
    expected_channels = {
        request.end_effector_config.actuator_position_channel,
        request.end_effector_config.actuator_force_limit_channel,
    }
    assert all(set(knot.actuator_channels) == expected_channels for knot in trajectory.knots)
    grasp_phase = next(
        item for item in trajectory.phases if item.stage is TrajectoryStage.GRASP
    )
    place_phase = next(
        item for item in trajectory.phases if item.stage is TrajectoryStage.PLACE
    )
    stopped_knots = set(range(grasp_phase.first_knot, grasp_phase.last_knot + 1))
    stopped_knots.update((0, place_phase.last_knot))
    for index in stopped_knots:
        assert trajectory.knots[index].velocities_rad_s == pytest.approx(
            (0.0,) * len(request.robot_model.joint_names)
        )

    for index, knot in enumerate(trajectory.knots):
        if index in stopped_knots:
            continue
        relative_speed = max(
            abs(value) / limit.maximum_velocity_rad_s
            for value, limit in zip(knot.velocities_rad_s, request.robot_model.joint_limits)
        )
        assert 0.0 < relative_speed <= 1.0

    for left, right in zip(trajectory.segments, trajectory.segments[1:]):
        assert left.end_knot == right.start_knot
        boundary = trajectory.knots[left.end_knot]
        for sample in (left.sample(left.duration_s), right.sample(0.0)):
            assert sample.positions_rad == pytest.approx(boundary.positions_rad)
            assert sample.velocities_rad_s == pytest.approx(boundary.velocities_rad_s)
            assert sample.accelerations_rad_s2 == pytest.approx(
                boundary.accelerations_rad_s2
            )


def test_plan_rejects_invalid_native_motion_after_time_parameterization() -> None:
    dependencies, _ = _final_validation_dependencies(reject_motion=True)
    request = _request()
    request = replace(request, candidates=(request.candidates[0],))
    result = plan(request, dependencies=dependencies)

    assert not result.success
    assert result.trajectory is None
    assert result.error_code is PlanningErrorCode.STAGE_COLLISION_VIOLATION
    assert result.attempts[-1].stage is TrajectoryStage.PREGRASP
    assert "time-parameterized" in result.attempts[-1].reason
    assert "native robot motion" in result.attempts[-1].reason


def test_final_validation_uses_current_opening_and_attached_geometry() -> None:
    request = _request()
    request = replace(request, candidates=(request.candidates[0],))
    dependencies, vamp = _final_validation_dependencies(reject_attachment=True)
    result = plan(request, dependencies=dependencies)

    assert not result.success
    assert result.trajectory is None
    assert result.error_code is PlanningErrorCode.STAGE_COLLISION_VIOLATION
    assert result.attempts[-1].stage is TrajectoryStage.LIFT
    assert "attached object" in result.attempts[-1].reason
    grasp = request.candidates[0].grasp_command
    assert any(
        grasp.target_opening_m < opening < grasp.pregrasp_opening_m
        for opening in vamp.final_target_openings
    )


def test_final_validation_enforces_configured_sample_budget() -> None:
    request = _request()
    request = replace(
        request,
        candidates=(request.candidates[0],),
        planning_policy=replace(
            request.planning_policy,
            trajectory_validation_max_samples=1,
        ),
    )

    result = plan(request, dependencies=_fake_dependencies())

    assert not result.success
    assert result.error_code is PlanningErrorCode.BUDGET_EXHAUSTED
    assert result.attempts[-1].error_code is PlanningErrorCode.BUDGET_EXHAUSTED
    assert "sample budget" in result.attempts[-1].reason


def test_approach_rejects_early_target_contact_but_allows_terminal_transition() -> None:
    request = _request()
    early = plan(
        request,
        dependencies=_approach_contact_dependencies(contact_distance_m=0.012),
    )
    assert not early.success
    assert early.attempts[-1].stage is TrajectoryStage.APPROACH
    assert (
        early.attempts[-1].error_code
        is PlanningErrorCode.STAGE_COLLISION_VIOLATION
    )

    terminal = plan(
        request,
        dependencies=_approach_contact_dependencies(contact_distance_m=0.008),
    )
    assert terminal.success, terminal.message

    coarse_request = replace(
        request,
        planning_policy=replace(
            request.planning_policy,
            cartesian_translation_step_m=0.02,
        ),
    )
    coarse_terminal = plan(
        coarse_request,
        dependencies=_approach_contact_dependencies(contact_distance_m=0.008),
    )
    assert coarse_terminal.success, coarse_terminal.message


def test_planning_uses_calibrated_vamp_range_and_contact_transition() -> None:
    vamp = CONFIG.planning_backends[PROFILE.planning_backend_profile].vamp
    assert vamp.range_rad == 0.5
    assert vamp.dynamic_domain is False
    assert vamp.robot_sphere_count == 432
    assert tuple(
        (group.group, group.first_sphere, group.sphere_count)
        for group in vamp.collision_sphere_groups
    ) == (("left_pad", 176, 128), ("right_pad", 304, 128))
    assert tuple(
        (item.sphere_index, item.center_at_default_m, item.radius_m)
        for item in vamp.collision_sphere_calibrations
    ) == (
        (41, (0.27569, -0.019505, 0.428311), 0.035516),
        (42, (0.287652, -0.005146, 0.399621), 0.010515),
    )
    assert CONFIG.planning_policy.approach_contact_transition_m == 0.01
    assert CONFIG.planning_policy.trajectory_validation_joint_step_rad == 0.01
    assert (
        CONFIG.planning_policy.trajectory_validation_tcp_translation_step_m
        == 0.002
    )
    assert CONFIG.planning_policy.trajectory_validation_tcp_rotation_step_rad == 0.02
    assert CONFIG.planning_policy.trajectory_validation_actuator_position_step_m == 0.001
    assert CONFIG.planning_policy.trajectory_validation_swept_point_step_m == 0.002
    assert CONFIG.planning_policy.trajectory_validation_max_samples == 20000


def test_plan_continues_after_a_geometric_route_ruckig_rejection() -> None:
    backend = CONFIG.planning_backends[PROFILE.planning_backend_profile]
    rejecting = _FirstRouteRejectingRuckig(
        RuckigTimeParameterizer(_request().robot_model, backend.ruckig)
    )
    dependencies = replace(
        _fake_dependencies(),
        create_ruckig=lambda robot_model: rejecting,
    )
    request = _request()
    request = replace(
        request,
        budgets=replace(request.budgets, seeds_per_candidate=2),
    )

    result = plan(request, dependencies=dependencies)

    assert result.success, result.message
    assert rejecting.calls == 2
    assert any(
        attempt.error_code is PlanningErrorCode.RUCKIG_FAILED
        and attempt.stage is TrajectoryStage.PREGRASP
        for attempt in result.attempts
    )


def test_convex_hull_semantic_collision_uses_exact_surface() -> None:
    vertices = tuple(
        (x, y, z)
        for x in (-0.5, 0.5)
        for y in (-0.5, 0.5)
        for z in (-0.5, 0.5)
    )
    hull = ConvexHull(
        vertices,
        (
            (0, 1, 3), (0, 3, 2),
            (4, 6, 7), (4, 7, 5),
            (0, 4, 5), (0, 5, 1),
            (2, 3, 7), (2, 7, 6),
            (0, 2, 6), (0, 6, 4),
            (1, 5, 7), (1, 7, 3),
        ),
    )
    geometry = SceneGeometry(
        "cube",
        "tank",
        "tank_environment",
        _pose("base", "cube"),
        hull,
    )

    class NativeSphere:
        def __init__(self, center, radius):
            self.x, self.y, self.z = center
            self.r = radius

    assert _sphere_intersects_geometry(NativeSphere((0.0, 0.0, 0.0), 0.01), geometry)
    assert _sphere_intersects_geometry(NativeSphere((0.55, 0.0, 0.0), 0.05), geometry)
    assert not _sphere_intersects_geometry(NativeSphere((0.56, 0.0, 0.0), 0.05), geometry)
    assert not _sphere_intersects_geometry(NativeSphere((0.55, 0.55, 0.0), 0.05), geometry)
    vertex_array = np.asarray(vertices, dtype=np.float64)
    faces = _outward_convex_faces(vertex_array, hull.triangles)
    a = vertex_array[faces[:, 0]]
    b = vertex_array[faces[:, 1]]
    c = vertex_array[faces[:, 2]]
    normals = np.cross(b - a, c - a)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    contacts = _convex_contact_groups(
        [
            (NativeSphere((0.0, 0.0, 0.0), 0.01), "inside"),
            (NativeSphere((0.55, 0.0, 0.0), 0.05), "face"),
            (NativeSphere((0.56, 0.0, 0.0), 0.05), "outside"),
            (NativeSphere((0.55, 0.55, 0.0), 0.05), "corner"),
        ],
        geometry,
        (a, b, c, normals),
    )
    assert contacts == frozenset(("inside", "face"))


def test_plan_preserves_request_config_and_backend_failure_codes() -> None:
    request = _request()
    invalid_request = replace(
        request,
        initial_state=RobotState(
            tuple(reversed(request.initial_state.joint_names)),
            request.initial_state.positions_rad,
        ),
    )
    assert plan(invalid_request, dependencies=_fake_dependencies()).error_code is PlanningErrorCode.INVALID_REQUEST
    invalid_policy = replace(
        request,
        planning_policy=replace(request.planning_policy, lift_distance_m=0.0),
    )
    assert plan(invalid_policy, dependencies=_fake_dependencies()).error_code is PlanningErrorCode.INVALID_CONFIG

    def unavailable(robot_model):
        raise BackendUnavailableError("patched VAMP is unavailable")

    dependencies = replace(_fake_dependencies(), create_vamp=unavailable)
    unavailable_result = plan(request, dependencies=dependencies)
    assert unavailable_result.error_code is PlanningErrorCode.BACKEND_UNAVAILABLE
    assert unavailable_result.trajectory is None


def test_plan_is_deterministic_for_a_fixed_request() -> None:
    request = _request()
    first = plan(request, dependencies=_fake_dependencies())
    second = plan(request, dependencies=_fake_dependencies())
    assert first.success and second.success
    assert first.trajectory == second.trajectory
    assert first.metadata == second.metadata


def test_formal_planning_types_do_not_contain_backend_or_gripper_assets() -> None:
    request = _request()
    robot_fields = set(type(request.robot_model).__dataclass_fields__)
    policy_fields = set(type(request.planning_policy).__dataclass_fields__)
    forbidden = ("vamp", "curobo", "ruckig", "mujoco", "gripper", "asset", "backend")
    assert not any(token in field for field in robot_fields | policy_fields for token in forbidden)


def test_importing_public_task2_does_not_load_native_backends() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.path.insert(0, 'src'); import task2; "
                "assert not ({'curobo', 'vamp', 'ruckig', 'mujoco'} & set(sys.modules))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert completed.returncode == 0, completed.stderr


def test_standard_part_real_quick_planning_regression() -> None:
    request = replace(
        _request("standard_part"),
        budgets=PlanningBudgets(1, 1, 120.0, 300.0),
        random_seed=19,
    )
    result = plan(request, dependencies=CONFIG.planning_dependencies(PROFILE))
    assert result.success, f"{result.error_code}: {result.message}"
    assert result.trajectory is not None
    assert tuple(phase.stage for phase in result.trajectory.phases) == tuple(TrajectoryStage)
    request.validate()
