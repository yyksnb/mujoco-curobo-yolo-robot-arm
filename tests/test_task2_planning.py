from __future__ import annotations

from dataclasses import replace
import subprocess
import sys
from pathlib import Path
from time import monotonic
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml

from task2 import plan, target
from task2.backends.errors import BackendOperationError, BackendUnavailableError
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
    PlacementOrientationConstraint,
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
from task2.planning import (
    PlanningDependencies,
    _sampled_orientation_deviation,
)


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
        angle = configuration[0]
        cosine = float(np.cos(angle))
        sine = float(np.sin(angle))
        xyz = tuple(value * 10.0 for value in configuration[:3])
        return RigidTransform(
            self.model.base_frame,
            self.model.tcp_frame,
            (
                (cosine, -sine, 0.0, xyz[0]),
                (sine, cosine, 0.0, xyz[1]),
                (0.0, 0.0, 1.0, xyz[2]),
                (0.0, 0.0, 0.0, 1.0),
            ),
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


class _FixedOrientationVamp(_FakeVamp):
    def __init__(self, robot_model, rotation):
        super().__init__(robot_model)
        self.rotation = np.asarray(rotation, dtype=np.float64)

    def tcp_pose(self, configuration):
        pose = super().tcp_pose(configuration)
        matrix = np.asarray(pose.matrix, dtype=np.float64)
        matrix[:3, :3] = self.rotation
        return RigidTransform(
            pose.parent_frame,
            pose.child_frame,
            tuple(tuple(float(value) for value in row) for row in matrix),
        )


class _ReturnedKnotVamp(_FixedOrientationVamp):
    def __init__(self, robot_model, rotation):
        super().__init__(robot_model, rotation)
        self.loaded_midpoints = []

    def plan(self, start, goal, environment, *, seed):
        if environment.get("attached_geometry") and start != goal:
            midpoint = tuple(
                0.5 * (left + right) + (0.02 if index == 3 else 0.0)
                for index, (left, right) in enumerate(zip(start, goal))
            )
            self.loaded_midpoints.append(midpoint)
            return VampPlan((tuple(start), midpoint, tuple(goal)), 0.001, 1)
        return super().plan(start, goal, environment, seed=seed)


class _SeedRouteVamp(_FixedOrientationVamp):
    def __init__(self, robot_model, rotation, detour_seed):
        super().__init__(robot_model, rotation)
        self.detour_seed = detour_seed
        self.detour_midpoints = []
        self.direct_midpoints = []

    def plan(self, start, goal, environment, *, seed):
        if environment.get("attached_geometry") and start != goal:
            midpoint = tuple(
                0.5 * (left + right) for left, right in zip(start, goal)
            )
            if seed == self.detour_seed:
                midpoint = tuple(
                    value + (0.3 if index == 0 else 0.0)
                    for index, value in enumerate(midpoint)
                )
                self.detour_midpoints.append(midpoint)
            else:
                midpoint = tuple(
                    value + (0.6 if index == 3 else 0.0)
                    for index, value in enumerate(midpoint)
                )
                self.direct_midpoints.append(midpoint)
            return VampPlan((tuple(start), midpoint, tuple(goal)), 0.001, 1)
        return super().plan(start, goal, environment, seed=seed)


class _InteriorRotationVamp(_FakeVamp):
    def tcp_pose(self, configuration):
        angle = 1.5 * float(np.sin(np.pi * configuration[0]))
        cosine = float(np.cos(angle))
        sine = float(np.sin(angle))
        return RigidTransform(
            self.model.base_frame,
            self.model.tcp_frame,
            (
                (cosine, -sine, 0.0, 0.0),
                (sine, cosine, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
        )


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


class _WrappedContinuousIK(_FakeIK):
    def solve(self, targets, seed_states, constraints):
        result = super().solve(targets, seed_states, constraints)
        solutions = []
        for solution in result.solutions:
            values = list(solution.state.positions_rad)
            values[0] = -np.pi + 0.01
            solutions.append(
                IKSolution(
                    RobotState(self.model.joint_names, tuple(values)),
                    solution.position_error_m,
                    solution.orientation_error_rad,
                )
            )
        return IKResult(tuple(solutions), result.error_code, result.message)


class _SourceSupportVamp(_FakeVamp):
    def __init__(self, robot_model, *, support_groups=("left_pad", "right_pad")):
        super().__init__(robot_model)
        self.support_groups = frozenset(support_groups)
        self.grasp_joint_z = None
        self.inject_post_departure_contact = False

    def semantic_target_contacts(
        self, configuration, geometries, opening_m, T_world_base
    ):
        groups = {item.collision_group for item in geometries}
        if "target" in groups:
            return super().semantic_target_contacts(
                configuration, geometries, opening_m, T_world_base
            )
        if "source_support" not in groups or opening_m >= 0.02:
            return frozenset()
        if self.grasp_joint_z is None:
            self.grasp_joint_z = configuration[2]
        if (
            self.inject_post_departure_contact
            or configuration[2] <= self.grasp_joint_z + 0.004
        ):
            return self.support_groups
        return frozenset()

    def semantic_attachment_contacts(
        self,
        configuration,
        T_tcp_object,
        attached_geometry,
        scene_geometries,
        T_world_base,
    ):
        if self.grasp_joint_z is None:
            return frozenset()
        if configuration[2] <= self.grasp_joint_z + 0.004:
            return frozenset(("source_support",))
        return frozenset()


class _PostRuckigContactInjector:
    def __init__(self, delegate, vamp):
        self._delegate = delegate
        self._vamp = vamp
        self.version = delegate.version

    def parameterize(self, *args, **kwargs):
        trajectory = self._delegate.parameterize(*args, **kwargs)
        self._vamp.inject_post_departure_contact = True
        return trajectory


class _RecordingIK(_FakeIK):
    def __init__(self, robot_model):
        super().__init__(robot_model)
        self.targets = []

    def solve(self, targets, seed_states, constraints):
        self.targets.extend(targets)
        return super().solve(targets, seed_states, constraints)


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


class _ScaleIgnoringRuckig:
    def __init__(self, delegate):
        self._delegate = delegate
        self.version = delegate.version

    def parameterize(self, waypoints, events):
        return self._delegate.parameterize(
            tuple(replace(item, motion_scale=1.0) for item in waypoints),
            events,
        )


class _PayloadSamplingVamp(_FakeVamp):
    def __init__(self, robot_model, *, clock=None, fail_fk=False):
        super().__init__(robot_model)
        self.payload_sampling = False
        self.clock = clock
        self.fail_fk = fail_fk

    def tcp_pose(self, configuration):
        if self.payload_sampling:
            if self.clock is not None:
                self.clock[0] += 0.01
            if self.fail_fk:
                raise BackendOperationError("VAMP FK failed during payload sampling")
        return super().tcp_pose(configuration)


class _PayloadSamplingRuckig:
    def __init__(self, delegate, vamp):
        self._delegate = delegate
        self._vamp = vamp
        self.version = delegate.version
        self.trajectories = []

    def parameterize(self, waypoints, events):
        trajectory = self._delegate.parameterize(waypoints, events)
        self.trajectories.append(trajectory)
        self._vamp.payload_sampling = True
        return trajectory


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
            PlacementOrientationConstraint.EXACT,
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


def _maximum_payload_q0_speed(trajectory, time_step_s: float) -> float:
    times = []
    positions = []
    payload_stages = {
        TrajectoryStage.LIFT,
        TrajectoryStage.EGRESS,
        TrajectoryStage.PLACE,
    }
    for segment in trajectory.segments:
        if segment.stage not in payload_stages:
            continue
        sample_count = max(2, int(np.ceil(segment.duration_s / time_step_s)))
        for index in range(sample_count + 1):
            local_time_s = segment.duration_s * index / sample_count
            time_s = segment.start_time_s + local_time_s
            if times and abs(time_s - times[-1]) <= 1e-12:
                continue
            times.append(time_s)
            positions.append(segment.sample(local_time_s).positions_rad[0])
    return max(
        abs(right - left) / (right_time - left_time)
        for left, right, left_time, right_time in zip(
            positions,
            positions[1:],
            times,
            times[1:],
        )
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


def test_exact_placement_orientation_is_strictly_used_for_place_ik() -> None:
    request = _request()
    recorder = _RecordingIK(request.robot_model)
    dependencies = replace(
        _fake_dependencies(),
        create_ik=lambda robot_model, random_seed: recorder,
    )

    result = plan(request, dependencies=dependencies)

    assert result.success, result.message
    place_target = np.asarray(recorder.targets[-1].matrix, dtype=np.float64)
    exact_target = np.asarray(
        request.placement_goal.T_world_tcp.matrix, dtype=np.float64
    )
    grasp_target = np.asarray(
        request.candidates[0].T_world_tcp.matrix, dtype=np.float64
    )
    assert place_target[:3, :3] == pytest.approx(exact_target[:3, :3])
    assert not np.allclose(place_target[:3, :3], grasp_target[:3, :3])


def test_orientation_free_placement_preserves_grasp_orientation_and_continuity() -> None:
    baseline = _request()
    request = replace(
        baseline,
        placement_goal=replace(
            baseline.placement_goal,
            orientation_constraint=PlacementOrientationConstraint.FREE,
        ),
    )
    recorder = _RecordingIK(request.robot_model)
    fixed_vamp = _FixedOrientationVamp(
        request.robot_model,
        np.asarray(request.candidates[0].T_world_tcp.matrix, dtype=np.float64)[
            :3, :3
        ],
    )
    dependencies = replace(
        _fake_dependencies(),
        create_vamp=lambda robot_model: fixed_vamp,
        create_ik=lambda robot_model, random_seed: recorder,
    )

    result = plan(request, dependencies=dependencies)

    assert result.success, result.message
    assert result.trajectory is not None
    place_target = np.asarray(recorder.targets[-1].matrix, dtype=np.float64)
    grasp_target = np.asarray(
        request.candidates[0].T_world_tcp.matrix, dtype=np.float64
    )
    assert place_target[:3, :3] == pytest.approx(grasp_target[:3, :3])
    assert place_target[:3, 3] == pytest.approx(
        np.asarray(request.placement_goal.T_world_tcp.matrix, dtype=np.float64)[
            :3, 3
        ]
    )
    assert tuple(phase.stage for phase in result.trajectory.phases) == tuple(
        TrajectoryStage
    )
    for left, right in zip(
        result.trajectory.segments,
        result.trajectory.segments[1:],
    ):
        boundary = result.trajectory.knots[left.end_knot]
        assert left.end_knot == right.start_knot
        assert left.sample(left.duration_s).velocities_rad_s == pytest.approx(
            boundary.velocities_rad_s
        )
        assert right.sample(0.0).velocities_rad_s == pytest.approx(
            boundary.velocities_rad_s
        )


def test_orientation_preference_cost_samples_between_vamp_knots() -> None:
    baseline = _request()
    request = replace(
        baseline,
        planning_policy=replace(
            baseline.planning_policy,
            trajectory_validation_joint_step_rad=1.0,
            trajectory_validation_tcp_rotation_step_rad=0.1,
        ),
    )
    vamp = _InteriorRotationVamp(request.robot_model)
    start = (0.0,) * len(request.robot_model.joint_names)
    goal = (1.0,) + start[1:]
    preferred = vamp.tcp_pose(start)
    started = monotonic()

    metrics = _sampled_orientation_deviation(
        (start, goal),
        preferred,
        vamp,
        request,
        TrajectoryStage.EGRESS,
        started,
        started,
    )

    assert metrics.maximum_deviation_rad == pytest.approx(1.5)
    assert metrics.cumulative_rotation_rad > 2.9
    assert metrics.sample_count > 3


def test_plan_selects_the_collision_valid_seed_route_without_tcp_detours() -> None:
    baseline = _request()
    candidate = baseline.candidates[0]
    request = replace(
        baseline,
        candidates=(candidate,),
        placement_goal=replace(
            baseline.placement_goal,
            orientation_constraint=PlacementOrientationConstraint.FREE,
        ),
        planning_policy=replace(
            baseline.planning_policy,
            cartesian_translation_step_m=1.0,
            payload_tcp_max_angular_velocity_rad_s=1000.0,
            payload_tcp_max_translation_velocity_m_s=1000.0,
            payload_tcp_max_translation_acceleration_m_s2=1000.0,
        ),
        budgets=PlanningBudgets(1, 2, 5.0, 20.0),
    )
    vamp = _SeedRouteVamp(
        request.robot_model,
        np.asarray(candidate.T_world_tcp.matrix, dtype=np.float64)[:3, :3],
        request.random_seed,
    )
    dependencies = replace(
        _fake_dependencies(),
        create_vamp=lambda robot_model: vamp,
    )

    result = plan(request, dependencies=dependencies)

    assert result.success, result.message
    assert result.trajectory is not None
    assert result.metadata["selected_seed"] == request.random_seed + 1
    selected_loaded_knots = tuple(
        knot.positions_rad
        for knot in result.trajectory.knots
        if knot.stage in (TrajectoryStage.EGRESS, TrajectoryStage.PLACE)
    )
    assert vamp.detour_midpoints
    assert vamp.direct_midpoints
    assert any(point in selected_loaded_knots for point in vamp.direct_midpoints)
    assert not any(point in selected_loaded_knots for point in vamp.detour_midpoints)


def test_orientation_free_route_preserves_vamp_returned_knots_for_ruckig() -> None:
    baseline = _request()
    candidate = baseline.candidates[0]
    request = replace(
        baseline,
        candidates=(candidate,),
        placement_goal=replace(
            baseline.placement_goal,
            orientation_constraint=PlacementOrientationConstraint.FREE,
        ),
        planning_policy=replace(
            baseline.planning_policy,
            payload_tcp_max_angular_velocity_rad_s=1000.0,
            payload_tcp_max_translation_velocity_m_s=1000.0,
            payload_tcp_max_translation_acceleration_m_s2=1000.0,
        ),
    )
    vamp = _ReturnedKnotVamp(
        request.robot_model,
        np.asarray(candidate.T_world_tcp.matrix, dtype=np.float64)[:3, :3],
    )
    dependencies = replace(
        _fake_dependencies(),
        create_vamp=lambda robot_model: vamp,
    )

    result = plan(request, dependencies=dependencies)

    assert result.success, result.message
    assert result.trajectory is not None
    loaded_knots = tuple(
        knot.positions_rad
        for knot in result.trajectory.knots
        if knot.stage in (TrajectoryStage.EGRESS, TrajectoryStage.PLACE)
    )
    assert vamp.loaded_midpoints
    assert any(
        midpoint == knot
        for midpoint in vamp.loaded_midpoints
        for knot in loaded_knots
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
        (
            group.group,
            group.first_sphere,
            group.sphere_count,
            group.translation_per_closed_fraction_m,
            group.sweep_radius_inflation_m,
        )
        for group in vamp.collision_sphere_groups
    ) == (
        ("left_pad", 176, 128, (0.0, 0.042, 0.0), 0.021),
        ("right_pad", 304, 128, (0.0, -0.042, 0.0), 0.021),
    )
    assert tuple(
        (item.sphere_index, item.center_at_default_m, item.radius_m)
        for item in vamp.collision_sphere_calibrations
    ) == (
        (27, (0.21516, 0.001, 0.539479), 0.036648),
        (41, (0.27569, -0.019505, 0.428311), 0.035516),
        (42, (0.287652, -0.005146, 0.399621), 0.010515),
        (176, (0.643443, -0.008751, 0.405724), 0.026419),
        (303, (0.668818, 0.005874, 0.411724), 0.024911),
        (304, (0.643443, -0.008751, 0.461724), 0.026419),
        (431, (0.668818, 0.005874, 0.455724), 0.024911),
    )
    assert CONFIG.planning_policy.approach_contact_transition_m == 0.01
    assert CONFIG.planning_policy.payload_tcp_max_angular_velocity_rad_s == 0.44
    assert CONFIG.planning_policy.payload_tcp_max_translation_velocity_m_s == 0.20
    assert (
        CONFIG.planning_policy.payload_tcp_max_translation_acceleration_m_s2
        == 2.00
    )
    assert CONFIG.planning_policy.payload_motion_validation_time_step_s == 0.01
    assert CONFIG.planning_policy.trajectory_validation_joint_step_rad == 0.01
    assert (
        CONFIG.planning_policy.trajectory_validation_tcp_translation_step_m
        == 0.002
    )
    assert CONFIG.planning_policy.trajectory_validation_tcp_rotation_step_rad == 0.02
    assert CONFIG.planning_policy.trajectory_validation_actuator_position_step_m == 0.001
    assert CONFIG.planning_policy.trajectory_validation_swept_point_step_m == 0.002
    assert CONFIG.planning_policy.trajectory_validation_max_samples == 20000


def test_forearm_proxy_sphere_is_not_shrunk_and_tracks_the_formal_distal_surface() -> None:
    spheres = yaml.safe_load(
        (ROOT / "configs/curobo/gen3/task2_active_wrist_parallel_spheres.yml").read_text()
    )["collision_spheres"]["forearm_link"]
    calibrated = spheres[1]
    center = np.asarray(calibrated["center"], dtype=np.float64)
    radius = float(calibrated["radius"])

    mesh_path = ROOT / "examples/mujoco/kinova_gen3/assets/forearm_link.stl"
    payload = mesh_path.read_bytes()
    triangle_count = int.from_bytes(payload[80:84], "little")
    records = np.frombuffer(
        payload,
        dtype=np.dtype(
            [
                ("normal", "<f4", (3,)),
                ("vertices", "<f4", (3, 3)),
                ("attribute", "<u2"),
            ]
        ),
        count=triangle_count,
        offset=84,
    )
    vertices = np.asarray(records["vertices"], dtype=np.float64).reshape((-1, 3))
    distal = vertices[vertices[:, 1] < -0.21]

    assert radius == pytest.approx(0.035643)
    assert center[1] - radius == pytest.approx(vertices[:, 1].min(), abs=1e-7)
    assert center[[0, 2]] == pytest.approx(distal[:, [0, 2]].mean(axis=0), abs=1e-7)


def test_pad_collision_spheres_cover_tapered_mujoco_envelope_without_old_box_air() -> None:
    model = ET.parse(
        ROOT / "examples/mujoco/so101_active_wrist_parallel/active_wrist_parallel.xml"
    ).getroot()
    mesh = model.find("./asset/mesh[@name='pad_collision_envelope']")
    tcp = model.find(".//site[@name='tcp']")
    assert mesh is not None and tcp is not None
    mesh_vertices = np.asarray(
        [float(value) for value in mesh.attrib["vertex"].split()], dtype=np.float64
    ).reshape((-1, 3))
    mesh_faces = np.asarray(
        [int(value) for value in mesh.attrib["face"].split()], dtype=np.int64
    ).reshape((-1, 3))
    tcp_position = np.asarray(
        [float(value) for value in tcp.attrib["pos"].split()], dtype=np.float64
    )
    quaternion = np.asarray(
        [float(value) for value in tcp.attrib["quat"].split()], dtype=np.float64
    )
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    tcp_rotation = np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    spheres = yaml.safe_load(
        (ROOT / "configs/curobo/gen3/task2_active_wrist_parallel_spheres.yml").read_text()
    )["collision_spheres"]
    groups = (
        ("left_jaw", "task2_tool_left_pad_collision", 1.0),
        ("right_jaw", "task2_tool_right_pad_collision", -1.0),
    )
    for body_name, sphere_link, closed_direction in groups:
        body = model.find(f".//body[@name='{body_name}']")
        assert body is not None
        joint = body.find("joint")
        assert joint is not None
        body_position = np.asarray(
            [float(value) for value in body.attrib["pos"].split()], dtype=np.float64
        )
        joint_axis = np.asarray(
            [float(value) for value in joint.attrib["axis"].split()], dtype=np.float64
        )
        joint_range = [float(value) for value in joint.attrib["range"].split()]
        open_vertices = (
            body_position + joint_axis * joint_range[1] + mesh_vertices - tcp_position
        ) @ tcp_rotation
        surface_points = []
        for face in mesh_faces:
            triangle = open_vertices[face]
            for first_weight in range(6):
                for second_weight in range(6 - first_weight):
                    weights = np.asarray(
                        (
                            first_weight,
                            second_weight,
                            5 - first_weight - second_weight,
                        ),
                        dtype=np.float64,
                    ) / 5.0
                    surface_points.append(weights @ triangle)
        surface_points = np.asarray(surface_points)
        pad_spheres = spheres[sphere_link]
        assert len(pad_spheres) == 128
        centers = np.asarray([item["center"] for item in pad_spheres], dtype=np.float64)
        radii = np.asarray([item["radius"] for item in pad_spheres], dtype=np.float64)

        z_edges = np.linspace(open_vertices[:, 2].min(), open_vertices[:, 2].max(), 9)
        y_edges = np.linspace(open_vertices[:, 1].min(), open_vertices[:, 1].max(), 5)
        if closed_direction < 0.0:
            y_edges = y_edges[::-1]
        u_edges = np.linspace(-1.0, 1.0, 5)
        z_min, z_max = z_edges[0], z_edges[-1]
        width_min = np.max(
            np.abs(open_vertices[np.isclose(open_vertices[:, 2], z_min), 0])
        )
        width_max = np.max(
            np.abs(open_vertices[np.isclose(open_vertices[:, 2], z_max), 0])
        )
        expected = []
        for z_low, z_high in zip(z_edges, z_edges[1:]):
            for y_low, y_high in zip(y_edges, y_edges[1:]):
                for u_low, u_high in zip(u_edges, u_edges[1:]):
                    cell = np.asarray(
                        [
                            (
                                u_value
                                * (
                                    width_min
                                    + (width_max - width_min)
                                    * (z_value - z_min)
                                    / (z_max - z_min)
                                ),
                                y_value,
                                z_value,
                            )
                            for z_value in (z_low, z_high)
                            for y_value in (y_low, y_high)
                            for u_value in (u_low, u_high)
                        ]
                    )
                    center = cell.mean(axis=0)
                    expected.append((center, np.linalg.norm(cell - center, axis=1).max()))
        assert np.allclose(centers, [item[0] for item in expected], atol=1e-12)
        assert np.all(radii >= np.asarray([item[1] for item in expected]) - 1e-12)

        for closed_fraction in (0.0, 0.5, 1.0):
            shift = np.asarray((0.0, closed_direction * 0.042 * closed_fraction, 0.0))
            shifted_surface = surface_points + shift
            shifted_centers = centers + shift
            distances = np.linalg.norm(
                shifted_surface[:, None, :] - shifted_centers[None, :, :], axis=2
            )
            assert np.all(np.min(distances - radii[None, :], axis=1) <= 1e-12)

        tip_z = open_vertices[:, 2].max()
        pad_center_y = 0.5 * (open_vertices[:, 1].min() + open_vertices[:, 1].max())
        for old_box_x in (-0.008, 0.008):
            old_box_probe = np.asarray((old_box_x, pad_center_y, tip_z))
            clearance = np.min(
                np.linalg.norm(centers - old_box_probe, axis=1) - radii - 0.001005
            )
            assert clearance > 0.0001


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
    assert rejecting.calls >= 2
    assert any(
        attempt.error_code is PlanningErrorCode.RUCKIG_FAILED
        and attempt.stage is TrajectoryStage.PREGRASP
        for attempt in result.attempts
    )


def test_plan_rejects_payload_motion_when_retiming_does_not_enforce_limit() -> None:
    backend = CONFIG.planning_backends[PROFILE.planning_backend_profile]
    request = _request()
    request = replace(
        request,
        candidates=(request.candidates[0],),
        planning_policy=replace(
            request.planning_policy,
            payload_tcp_max_angular_velocity_rad_s=0.0001,
            payload_tcp_max_translation_velocity_m_s=100.0,
            payload_tcp_max_translation_acceleration_m_s2=100.0,
        ),
    )
    dependencies = replace(
        _fake_dependencies(),
        create_ruckig=lambda robot_model: _ScaleIgnoringRuckig(
            RuckigTimeParameterizer(robot_model, backend.ruckig)
        ),
    )

    result = plan(request, dependencies=dependencies)

    assert not result.success
    assert result.error_code is PlanningErrorCode.RUCKIG_FAILED
    assert result.attempts[-1].stage in (
        TrajectoryStage.LIFT,
        TrajectoryStage.EGRESS,
        TrajectoryStage.PLACE,
    )
    assert "payload TCP angular velocity" in result.attempts[-1].reason


def test_real_ruckig_retimes_payload_motion_and_preserves_trajectory_contract() -> None:
    baseline_request = _request()
    baseline_request = replace(
        baseline_request,
        candidates=(baseline_request.candidates[0],),
        planning_policy=replace(
            baseline_request.planning_policy,
            payload_tcp_max_angular_velocity_rad_s=1000.0,
            payload_tcp_max_translation_velocity_m_s=1000.0,
            payload_tcp_max_translation_acceleration_m_s2=1000.0,
        ),
    )
    baseline = plan(baseline_request, dependencies=_fake_dependencies())
    assert baseline.success, baseline.message
    assert baseline.trajectory is not None
    time_step_s = baseline_request.planning_policy.payload_motion_validation_time_step_s
    baseline_speed = _maximum_payload_q0_speed(baseline.trajectory, time_step_s)
    angular_limit = 0.5 * baseline_speed

    request = replace(
        baseline_request,
        planning_policy=replace(
            baseline_request.planning_policy,
            payload_tcp_max_angular_velocity_rad_s=angular_limit,
        ),
    )
    backend = CONFIG.planning_backends[PROFILE.planning_backend_profile]
    vamp = _PayloadSamplingVamp(request.robot_model)
    recording_ruckig = _PayloadSamplingRuckig(
        RuckigTimeParameterizer(request.robot_model, backend.ruckig),
        vamp,
    )
    dependencies = replace(
        _fake_dependencies(),
        create_vamp=lambda robot_model: vamp,
        create_ruckig=lambda robot_model: recording_ruckig,
    )

    result = plan(request, dependencies=dependencies)

    assert result.success, result.message
    assert result.trajectory is not None
    assert len(recording_ruckig.trajectories) >= 2
    assert (
        _maximum_payload_q0_speed(recording_ruckig.trajectories[0], time_step_s)
        > angular_limit
    )
    assert _maximum_payload_q0_speed(result.trajectory, time_step_s) <= (
        angular_limit * (1.0 + 1e-4) + 1e-9
    )
    assert tuple(phase.stage for phase in result.trajectory.phases) == tuple(
        TrajectoryStage
    )
    assert tuple(event.event for event in result.trajectory.events) == (
        "grip",
        "release",
    )
    grasp_phase = next(
        phase
        for phase in result.trajectory.phases
        if phase.stage is TrajectoryStage.GRASP
    )
    place_phase = next(
        phase
        for phase in result.trajectory.phases
        if phase.stage is TrajectoryStage.PLACE
    )
    stopped_knots = set(range(grasp_phase.first_knot, grasp_phase.last_knot + 1))
    stopped_knots.update((0, place_phase.last_knot))
    for index, knot in enumerate(result.trajectory.knots):
        if index in stopped_knots:
            assert knot.velocities_rad_s == pytest.approx(
                (0.0,) * len(request.robot_model.joint_names)
            )
        else:
            assert any(abs(value) > 0.0 for value in knot.velocities_rad_s)
    for left, right in zip(
        result.trajectory.segments,
        result.trajectory.segments[1:],
    ):
        boundary = result.trajectory.knots[left.end_knot]
        assert left.end_knot == right.start_knot
        for sample in (left.sample(left.duration_s), right.sample(0.0)):
            assert sample.positions_rad == pytest.approx(boundary.positions_rad)
            assert sample.velocities_rad_s == pytest.approx(boundary.velocities_rad_s)
            assert sample.accelerations_rad_s2 == pytest.approx(
                boundary.accelerations_rad_s2
            )


def test_orientation_free_region_uses_nearest_goal_with_minimal_place_motion() -> None:
    baseline = _request()
    request = replace(
        baseline,
        placement_goal=PlacementGoal(
            "outside_region",
            PlacementGoalKind.RECTANGLE,
            baseline.placement_goal.T_world_tcp,
            PlacementOrientationConstraint.FREE,
            size_m=(0.1, 0.1),
            height_range_m=(0.0, baseline.planning_policy.preplace_offset_m),
        ),
    )
    fixed_vamp = _FixedOrientationVamp(
        request.robot_model,
        np.asarray(request.candidates[0].T_world_tcp.matrix, dtype=np.float64)[
            :3, :3
        ],
    )
    dependencies = replace(
        _fake_dependencies(),
        create_vamp=lambda robot_model: fixed_vamp,
    )

    result = plan(request, dependencies=dependencies)

    assert result.success, result.message
    assert result.trajectory is not None
    trajectory = result.trajectory
    assert tuple(phase.stage for phase in trajectory.phases) == tuple(TrajectoryStage)
    place = next(
        phase for phase in trajectory.phases if phase.stage is TrajectoryStage.PLACE
    )
    start = np.asarray(trajectory.knots[place.first_knot].positions_rad)
    end = np.asarray(trajectory.knots[place.last_knot].positions_rad)
    assert np.max(np.abs(end - start)) <= (
        request.planning_policy.trajectory_validation_joint_step_rad + 1e-12
    )
    world_tcp = np.asarray(
        fixed_vamp.tcp_pose(tuple(end)).matrix,
        dtype=np.float64,
    )
    goal_position = np.asarray(
        request.placement_goal.T_world_tcp.matrix,
        dtype=np.float64,
    )[:3, 3]
    assert world_tcp[:3, 3] == pytest.approx(
        goal_position
        + np.asarray((0.0, 0.0, request.planning_policy.preplace_offset_m))
    )
    assert trajectory.knots[place.last_knot].velocities_rad_s == pytest.approx(
        (0.0,) * len(request.robot_model.joint_names)
    )


def test_orientation_free_region_skips_zero_motion_sample_without_timing_leak() -> None:
    baseline = _request()
    candidate = baseline.candidates[0]
    placement_anchor = replace(
        candidate.T_world_tcp,
        child_frame=baseline.robot_model.tcp_frame,
    )
    request = replace(
        baseline,
        placement_goal=PlacementGoal(
            "already_reached_region",
            PlacementGoalKind.RECTANGLE,
            placement_anchor,
            PlacementOrientationConstraint.FREE,
            size_m=(0.1, 0.1),
            height_range_m=(0.0, baseline.planning_policy.preplace_offset_m),
        ),
        planning_policy=replace(
            baseline.planning_policy,
            payload_tcp_max_angular_velocity_rad_s=1000.0,
            payload_tcp_max_translation_velocity_m_s=1000.0,
            payload_tcp_max_translation_acceleration_m_s2=1000.0,
        ),
    )
    fixed_vamp = _FixedOrientationVamp(
        request.robot_model,
        np.asarray(candidate.T_world_tcp.matrix, dtype=np.float64)[:3, :3],
    )
    dependencies = replace(
        _fake_dependencies(),
        create_vamp=lambda robot_model: fixed_vamp,
    )

    slower_validation_request = replace(
        request,
        planning_policy=replace(
            request.planning_policy,
            payload_motion_validation_time_step_s=2.0
            * request.planning_policy.payload_motion_validation_time_step_s,
        ),
    )

    results = tuple(
        plan(item, dependencies=dependencies)
        for item in (request, slower_validation_request)
    )

    assert all(result.success for result in results), tuple(
        result.message for result in results
    )
    trajectories = tuple(result.trajectory for result in results)
    assert all(trajectory is not None for trajectory in trajectories)
    first, second = trajectories
    assert first is not None and second is not None
    assert first.knots[-1].time_s == pytest.approx(second.knots[-1].time_s)
    for result, trajectory in zip(results, (first, second)):
        assert tuple(phase.stage for phase in trajectory.phases) == tuple(
            TrajectoryStage
        )
        lift = next(
            phase for phase in trajectory.phases if phase.stage is TrajectoryStage.LIFT
        )
        egress = next(
            phase for phase in trajectory.phases if phase.stage is TrajectoryStage.EGRESS
        )
        place = next(
            phase for phase in trajectory.phases if phase.stage is TrajectoryStage.PLACE
        )
        assert any(
            abs(value) > 0.0
            for value in trajectory.knots[lift.last_knot].velocities_rad_s
        )
        assert any(
            abs(value) > 0.0
            for value in trajectory.knots[egress.last_knot].velocities_rad_s
        )
        assert trajectory.knots[place.first_knot].positions_rad != pytest.approx(
            trajectory.knots[place.last_knot].positions_rad
        )
        assert trajectory.knots[place.last_knot].velocities_rad_s == pytest.approx(
            (0.0,) * len(request.robot_model.joint_names)
        )
        release = next(
            event for event in trajectory.events if event.event == "release"
        )
        assert release.time_s == pytest.approx(place.end_time_s)
        assert any(
            attempt.error_code is PlanningErrorCode.RUCKIG_FAILED
            and attempt.stage is TrajectoryStage.PLACE
            and "zero-motion PLACE sample" in attempt.reason
            for attempt in result.attempts
        )

    terminal = plan(
        replace(request, candidates=(candidate,)),
        dependencies=replace(dependencies, placement_sample_count=1),
    )
    assert not terminal.success
    assert terminal.error_code is PlanningErrorCode.RUCKIG_FAILED
    zero_motion_attempts = tuple(
        attempt
        for attempt in terminal.attempts
        if attempt.error_code is PlanningErrorCode.RUCKIG_FAILED
        and attempt.stage is TrajectoryStage.PLACE
        and "zero-motion PLACE sample" in attempt.reason
    )
    assert len(zero_motion_attempts) == 1


def test_payload_fk_failure_is_backend_error_not_ruckig_failure() -> None:
    request = _request()
    request = replace(request, candidates=(request.candidates[0],))
    backend = CONFIG.planning_backends[PROFILE.planning_backend_profile]
    vamp = _PayloadSamplingVamp(request.robot_model, fail_fk=True)
    gated_ruckig = _PayloadSamplingRuckig(
        RuckigTimeParameterizer(request.robot_model, backend.ruckig),
        vamp,
    )
    dependencies = replace(
        _fake_dependencies(),
        create_vamp=lambda robot_model: vamp,
        create_ruckig=lambda robot_model: gated_ruckig,
    )

    result = plan(request, dependencies=dependencies)

    assert not result.success
    assert result.error_code is PlanningErrorCode.BACKEND_ERROR
    assert "payload motion backend error" in result.message
    assert "VAMP FK failed" in result.message


def test_payload_retiming_checks_total_budget_during_fk_sampling(monkeypatch) -> None:
    clock = [0.0]
    monkeypatch.setattr("task2.planning.monotonic", lambda: clock[0])
    request = _request()
    request = replace(
        request,
        candidates=(request.candidates[0],),
        budgets=PlanningBudgets(1, 1, 0.05, 0.05),
    )
    backend = CONFIG.planning_backends[PROFILE.planning_backend_profile]
    vamp = _PayloadSamplingVamp(request.robot_model, clock=clock)
    gated_ruckig = _PayloadSamplingRuckig(
        RuckigTimeParameterizer(request.robot_model, backend.ruckig),
        vamp,
    )
    dependencies = replace(
        _fake_dependencies(),
        create_vamp=lambda robot_model: vamp,
        create_ruckig=lambda robot_model: gated_ruckig,
    )

    result = plan(request, dependencies=dependencies)

    assert not result.success
    assert result.error_code is PlanningErrorCode.BUDGET_EXHAUSTED
    assert result.attempts[-1].stage in (
        TrajectoryStage.LIFT,
        TrajectoryStage.EGRESS,
        TrajectoryStage.PLACE,
    )
    assert "total time budget" in result.message


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


def test_public_plan_unwraps_continuous_joint_across_pi() -> None:
    request = _request()
    initial_positions = list(request.initial_state.positions_rad)
    initial_positions[0] = np.pi - 0.01
    request = replace(
        request,
        candidates=(request.candidates[0],),
        initial_state=RobotState(
            request.robot_model.joint_names, tuple(initial_positions)
        ),
    )
    dependencies = replace(
        _fake_dependencies(),
        create_ik=lambda robot_model, random_seed: _WrappedContinuousIK(
            robot_model
        ),
    )

    result = plan(request, dependencies=dependencies)

    assert result.success, result.message
    assert result.trajectory is not None
    joint_positions = tuple(
        knot.positions_rad[0] for knot in result.trajectory.knots
    )
    assert any(value > np.pi for value in joint_positions)
    assert max(
        abs(right - left) for left, right in zip(joint_positions, joint_positions[1:])
    ) < 0.1


@pytest.mark.parametrize(
    ("support_groups", "post_ruckig_contact", "expected_success"),
    (
        (("left_pad", "right_pad"), False, True),
        (("wrist",), False, False),
        (("left_pad", "right_pad"), True, False),
    ),
)
def test_source_support_contact_is_pad_only_and_ends_on_departure(
    support_groups, post_ruckig_contact, expected_success
) -> None:
    baseline = _request()
    request = replace(baseline, candidates=(baseline.candidates[0],))
    backend = CONFIG.planning_backends[PROFILE.planning_backend_profile]
    vamp = _SourceSupportVamp(
        request.robot_model, support_groups=support_groups
    )
    base_ruckig = RuckigTimeParameterizer(request.robot_model, backend.ruckig)
    ruckig = (
        _PostRuckigContactInjector(base_ruckig, vamp)
        if post_ruckig_contact
        else base_ruckig
    )
    dependencies = replace(
        _fake_dependencies(),
        create_vamp=lambda robot_model: vamp,
        create_ruckig=lambda robot_model: ruckig,
    )

    result = plan(request, dependencies=dependencies)

    assert result.success is expected_success
    if expected_success:
        assert result.trajectory is not None
        assert tuple(phase.stage for phase in result.trajectory.phases) == tuple(
            TrajectoryStage
        )
    else:
        assert result.error_code is PlanningErrorCode.STAGE_COLLISION_VIOLATION
        assert result.attempts[-1].stage in (
            TrajectoryStage.GRASP,
            TrajectoryStage.LIFT,
        )
        if post_ruckig_contact:
            assert "time-parameterized LIFT" in result.attempts[-1].reason


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
