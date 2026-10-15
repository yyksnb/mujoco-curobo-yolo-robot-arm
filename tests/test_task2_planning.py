from __future__ import annotations

from dataclasses import replace
from math import inf, nextafter, pi
import subprocess
import sys
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml

from task2 import plan, target
from task2.backends.errors import BackendOperationError, BackendUnavailableError
from task2.backends.ruckig import (
    GeometricWaypoint,
    RuckigRouteError,
    RuckigTimeParameterizer,
)
from task2.backends.vamp import (
    VampBackend,
    VampPlan,
    _convex_contact_groups,
    _outward_convex_faces,
    _prepared_convex_polyhedron,
    _prepared_polyhedra_intersect,
    _prepared_polyhedra_within_clearance,
    _sphere_intersects_geometry,
)
from task2.config import load_task2_config
from task2.contracts import (
    AllowedContact,
    AttachedGeometry,
    Box,
    Capsule,
    ConvexHull,
    LocalMotionStrategy,
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
    TargetingAccessContext,
    TargetingRequest,
    TrajectoryStage,
    VerticalTransitCorridor,
)
from task2.contracts.geometry import compose, inverse
from task2.ik import IKResult, IKSolution
from task2.planning import (
    _Route,
    _RouteFailure,
    PlanningDependencies,
    _cartesian_continuation_route,
    _compact_joint_path,
    _compact_robot_state_path,
    _continuous_ik,
    _farthest_collision_valid_shortcut,
    _joint_path_detour_ratio,
    _joint_path_length,
    _maximum_bounded_joint_path_edge,
    _maximum_joint_path_edge,
    _maximum_physical_joint_path_edge,
    _joint_space_portal_route,
    _joint_space_portal_then_vertical_route,
    _normalize_segment_sample_time,
    _retime_payload_motion,
    _sampled_orientation_deviation,
    _staged_lift_route,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_task2_config(ROOT / "configs/task2/pipeline.yaml")
PROFILE = CONFIG.end_effectors[0]


def test_segment_validation_normalizes_only_one_ulp_endpoint_roundoff():
    segment = SimpleNamespace(duration_s=0.40476199999999996)
    one_ulp_high = nextafter(segment.duration_s, inf)
    two_ulps_high = nextafter(one_ulp_high, inf)

    assert _normalize_segment_sample_time(segment, one_ulp_high) == segment.duration_s
    assert _normalize_segment_sample_time(segment, two_ulps_high) == two_ulps_high


class _FakeVamp:
    version = "0.6.4"

    def __init__(self, robot_model):
        self.model = robot_model
        self.plan_environments = []
        self.plan_calls = []

    def environment(self, scene, **kwargs):
        return {"scene": scene, **kwargs}

    def plan(self, start, goal, environment, *, sample_offset):
        self.plan_environments.append(environment)
        self.plan_calls.append((tuple(start), tuple(goal), environment, sample_offset))
        return VampPlan((tuple(start), tuple(goal)), 0.001, 1)

    def plan_to_any(self, start, goals, environment, *, sample_offset):
        plans = tuple(
            (self.plan(start, goal, environment, sample_offset=sample_offset), index)
            for index, goal in enumerate(goals)
        )
        result, selected_index = min(
            plans,
            key=lambda item: (
                _joint_path_length(item[0].path),
                item[1],
            ),
        )
        return replace(result, selected_goal_index=selected_index)

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

    def semantic_source_support_contacts(
        self,
        configuration,
        support_geometries,
        opening_m,
        T_world_base,
        *,
        use_physical_pad_hulls,
    ):
        return self.semantic_target_contacts(
            configuration, support_geometries, opening_m, T_world_base
        )

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

    def plan(self, start, goal, environment, *, sample_offset):
        if environment.get("attached_geometry") and start != goal:
            midpoint = tuple(
                0.5 * (left + right) + (0.02 if index == 3 else 0.0)
                for index, (left, right) in enumerate(zip(start, goal))
            )
            self.loaded_midpoints.append(midpoint)
            return VampPlan((tuple(start), midpoint, tuple(goal)), 0.001, 1)
        return super().plan(start, goal, environment, sample_offset=sample_offset)


class _LiftEndpointVamp(_FixedOrientationVamp):
    def __init__(self, robot_model, rotation, *, reject_minimum):
        super().__init__(robot_model, rotation)
        self.reject_minimum = reject_minimum
        self.loaded_plan_starts = []

    def plan(self, start, goal, environment, *, sample_offset):
        if environment.get("attached_geometry"):
            if environment["scene"].snapshot_id.endswith(
                ".lift_without_source_support"
            ):
                return super().plan(
                    start, goal, environment, sample_offset=sample_offset
                )
            self.loaded_plan_starts.append(tuple(start))
            if self.reject_minimum and len(self.loaded_plan_starts) == 1:
                raise BackendOperationError("minimum LIFT endpoint is blocked")
        return super().plan(start, goal, environment, sample_offset=sample_offset)


class _LiftEndpointCostVamp(_FixedOrientationVamp):
    def __init__(self, robot_model, rotation, *, minimum_is_better):
        super().__init__(robot_model, rotation)
        self.minimum_is_better = minimum_is_better
        self.loaded_plan_count = 0
        self.egress_starts = []

    def plan(self, start, goal, environment, *, sample_offset):
        if not environment.get("attached_geometry"):
            return super().plan(start, goal, environment, sample_offset=sample_offset)
        if environment["scene"].snapshot_id.endswith(
            ".lift_without_source_support"
        ):
            return super().plan(start, goal, environment, sample_offset=sample_offset)
        is_egress = self.loaded_plan_count % 2 == 0
        self.loaded_plan_count += 1
        if not is_egress:
            return super().plan(start, goal, environment, sample_offset=sample_offset)
        endpoint_index = len(self.egress_starts)
        self.egress_starts.append(tuple(start))
        detour = (endpoint_index == 0) != self.minimum_is_better
        if not detour:
            return super().plan(start, goal, environment, sample_offset=sample_offset)
        midpoint = list(
            0.5 * (left + right) for left, right in zip(start, goal)
        )
        midpoint[1] += 0.4
        return VampPlan((tuple(start), tuple(midpoint), tuple(goal)), 0.001, 1)


class _OffsetRouteVamp(_FixedOrientationVamp):
    def __init__(self, robot_model, rotation, detour_sample_offset):
        super().__init__(robot_model, rotation)
        self.detour_sample_offset = detour_sample_offset
        self.detour_midpoints = []
        self.direct_midpoints = []

    def plan(self, start, goal, environment, *, sample_offset):
        if environment.get("attached_geometry") and start != goal:
            self.plan_environments.append(environment)
            self.plan_calls.append((tuple(start), tuple(goal), environment, sample_offset))
            midpoint = tuple(
                0.5 * (left + right) for left, right in zip(start, goal)
            )
            if sample_offset == self.detour_sample_offset:
                midpoint = tuple(
                    value + (0.3 if index == 0 else 0.0)
                    for index, value in enumerate(midpoint)
                )
                self.detour_midpoints.append(midpoint)
            else:
                midpoint = tuple(
                    value + (0.1 if index == 3 else 0.0)
                    for index, value in enumerate(midpoint)
                )
                self.direct_midpoints.append(midpoint)
            return VampPlan((tuple(start), midpoint, tuple(goal)), 0.001, 1)
        return super().plan(start, goal, environment, sample_offset=sample_offset)


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


class _BranchingRouteIK(_FakeIK):
    def solve(self, targets, seed_states, constraints):
        target_x = round(targets[0].matrix[0][3])
        branches = {
            1: (0.4, 0.8),
            2: (0.3, 0.9),
            3: (1.0,),
        }[target_x]
        solutions = []
        for value in branches:
            positions = list(seed_states[0].positions_rad)
            positions[0] = value
            solutions.append(
                IKSolution(
                    RobotState(self.model.joint_names, tuple(positions)),
                    0.0,
                    0.0,
                )
            )
        return IKResult(tuple(solutions), None, "branching IK succeeded")


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


class _ApproachSupportProxyVamp(_FakeVamp):
    def __init__(
        self,
        robot_model,
        grasp_position_m,
        proxy_contact_distance_m,
        *,
        physical_contact,
    ):
        super().__init__(robot_model)
        self.grasp_position_m = np.asarray(grasp_position_m, dtype=np.float64)
        self.proxy_contact_distance_m = proxy_contact_distance_m
        self.physical_contact = physical_contact

    def _in_proxy_contact(self, configuration):
        tcp_position = np.asarray(
            self.tcp_pose(configuration).matrix, dtype=np.float64
        )[:3, 3]
        return (
            np.linalg.norm(tcp_position - self.grasp_position_m)
            <= self.proxy_contact_distance_m
        )

    def semantic_target_contacts(self, configuration, geometries, opening_m, T_world_base):
        if (
            any(item.collision_group == "source_support" for item in geometries)
            and opening_m >= 0.02
            and self._in_proxy_contact(configuration)
        ):
            return frozenset(("left_pad", "right_pad"))
        return super().semantic_target_contacts(
            configuration, geometries, opening_m, T_world_base
        )

    def semantic_source_support_contacts(
        self,
        configuration,
        support_geometries,
        opening_m,
        T_world_base,
        *,
        use_physical_pad_hulls,
    ):
        if use_physical_pad_hulls:
            if self.physical_contact and self._in_proxy_contact(configuration):
                return frozenset(("left_pad", "right_pad"))
            return frozenset()
        return self.semantic_target_contacts(
            configuration, support_geometries, opening_m, T_world_base
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


class _EdgeSpeedVamp(_FakeVamp):
    def tcp_pose(self, configuration):
        return _pose(
            self.model.base_frame,
            self.model.tcp_frame,
            (0.1 * configuration[0] + configuration[1], 0.0, 0.0),
        )


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
        PlanningBudgets(1, 1, 2.0, 10.0),
    )


def _fake_dependencies() -> PlanningDependencies:
    backend = CONFIG.planning_backends[PROFILE.planning_backend_profile]
    return PlanningDependencies(
        lambda robot_model: _FakeVamp(robot_model),
        lambda robot_model: _FakeIK(robot_model),
        lambda robot_model: RuckigTimeParameterizer(robot_model, backend.ruckig),
        backend.ik_solution_count,
        backend.pregrasp_ik_solution_count,
        backend.placement_sample_count,
    )


def _two_endpoint_free_placement_request() -> PlanningRequest:
    baseline = _request()
    candidate = baseline.candidates[0]
    step_m = 0.05
    minimum_lift_z = (
        candidate.T_world_tcp.matrix[2][3]
        + baseline.planning_policy.lift_distance_m
    )
    placement_matrix = [
        list(row) for row in baseline.placement_goal.T_world_tcp.matrix
    ]
    placement_matrix[2][3] = (
        minimum_lift_z
        + step_m
        - baseline.planning_policy.preplace_offset_m
    )
    placement_pose = RigidTransform(
        baseline.placement_goal.T_world_tcp.parent_frame,
        baseline.placement_goal.T_world_tcp.child_frame,
        tuple(tuple(row) for row in placement_matrix),
    )
    return replace(
        baseline,
        candidates=(candidate,),
        object_config=replace(
            baseline.object_config,
            payload_motion_strategy=LocalMotionStrategy.CONTINUOUS_CARTESIAN,
        ),
        placement_goal=replace(
            baseline.placement_goal,
            T_world_tcp=placement_pose,
            orientation_constraint=PlacementOrientationConstraint.FREE,
        ),
        planning_policy=replace(
            baseline.planning_policy,
            cartesian_translation_step_m=step_m,
            payload_tcp_max_angular_velocity_rad_s=1000.0,
            payload_tcp_max_translation_velocity_m_s=1000.0,
            payload_tcp_max_translation_acceleration_m_s2=1000.0,
        ),
        budgets=PlanningBudgets(1, 1, 5.0, 20.0),
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


def _approach_support_proxy_dependencies(
    proxy_contact_distance_m: float, *, physical_contact: bool
) -> PlanningDependencies:
    request = _request()
    grasp_position = np.asarray(
        request.candidates[0].T_world_tcp.matrix, dtype=np.float64
    )[:3, 3]
    return replace(
        _fake_dependencies(),
        create_vamp=lambda robot_model: _ApproachSupportProxyVamp(
            robot_model,
            grasp_position,
            proxy_contact_distance_m,
            physical_contact=physical_contact,
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
        lambda robot_model: _FakeIK(robot_model),
        lambda robot_model: _FinalValidationRuckig(
            RuckigTimeParameterizer(robot_model, backend.ruckig), vamp
        ),
        backend.ik_solution_count,
        backend.pregrasp_ik_solution_count,
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
    assert grasp_phase.first_knot < grasp_phase.last_knot
    assert place_phase.first_knot <= place_phase.last_knot
    assert all(
        knot.velocities_rad_s
        == pytest.approx((0.0,) * len(request.robot_model.joint_names))
        for knot in trajectory.knots
    )

    for left, right in zip(trajectory.segments, trajectory.segments[1:]):
        assert left.end_knot == right.start_knot
        boundary = trajectory.knots[left.end_knot]
        for sample in (left.sample(left.duration_s), right.sample(0.0)):
            assert sample.positions_rad == pytest.approx(boundary.positions_rad)
            assert sample.velocities_rad_s == pytest.approx(boundary.velocities_rad_s)
            assert sample.accelerations_rad_s2 == pytest.approx(
                boundary.accelerations_rad_s2
            )


def test_candidate_budget_does_not_probe_excluded_candidates() -> None:
    baseline = _request()
    first = baseline.candidates[0]
    excluded = replace(
        first,
        candidate_id=f"{first.candidate_id}.excluded",
        T_world_pregrasp=_pose(
            "world",
            first.T_world_pregrasp.child_frame,
            (9.0, 0.0, 0.2),
        ),
    )
    request = replace(
        baseline,
        candidates=(first, excluded),
        budgets=replace(baseline.budgets, maximum_candidates=1),
    )
    recorder = _RecordingIK(request.robot_model)
    dependencies = replace(
        _fake_dependencies(),
        create_ik=lambda robot_model: recorder,
    )

    result = plan(request, dependencies=dependencies)

    assert result.success, result.message
    assert all(target.matrix[0][3] != pytest.approx(9.0) for target in recorder.targets)


def test_exact_placement_orientation_is_strictly_used_for_place_ik() -> None:
    request = _request()
    recorder = _RecordingIK(request.robot_model)
    dependencies = replace(
        _fake_dependencies(),
        create_ik=lambda robot_model: recorder,
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
        create_ik=lambda robot_model: recorder,
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


def test_ruckig_parameterizes_each_waypoint_edge_as_scalar_path_progress() -> None:
    model = _request().robot_model
    zero = (0.0,) * len(model.joint_names)
    final = (0.4, 0.1, *zero[2:])
    channels = {"opening": 0.08, "force": 1.0}
    backend = CONFIG.planning_backends[PROFILE.planning_backend_profile]
    positions = tuple(
        tuple(fraction * value for value in final)
        for fraction in (0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0)
    )
    stages = (
        TrajectoryStage.PREGRASP,
        TrajectoryStage.PREGRASP,
        TrajectoryStage.APPROACH,
        TrajectoryStage.GRASP,
        TrajectoryStage.LIFT,
        TrajectoryStage.EGRESS,
        TrajectoryStage.PLACE,
    )
    trajectory = RuckigTimeParameterizer(model, backend.ruckig).parameterize(
        tuple(
            GeometricWaypoint(stage, point, channels)
            for stage, point in zip(stages, positions)
        ),
        (),
    )

    assert all(
        knot.velocities_rad_s == pytest.approx(zero)
        for knot in trajectory.knots
    )
    for segment, source, target_position in zip(
        trajectory.segments, positions, positions[1:]
    ):
        delta = np.asarray(target_position) - np.asarray(source)
        active = int(np.flatnonzero(np.abs(delta) > 1e-12)[0])
        for fraction in (0.1, 0.5, 0.9):
            sample = segment.sample(segment.duration_s * fraction)
            progress = (
                sample.positions_rad[active] - source[active]
            ) / delta[active]
            assert sample.positions_rad == pytest.approx(
                np.asarray(source) + progress * delta
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


def test_maximum_joint_path_edge_detects_a_direct_branch_flip() -> None:
    path = ((0.0, 0.0), (2.5, -2.5), (2.6, -2.4))

    assert _joint_path_detour_ratio(path) < 1.1
    assert _maximum_joint_path_edge(path) == pytest.approx(np.sqrt(12.5))


def test_bounded_joint_path_edge_ignores_unwrapped_continuous_joints() -> None:
    model = _request().robot_model
    start = list(model.default_joint_positions)
    goal = start.copy()
    goal[0] += 4.0
    goal[3] = -goal[3]

    assert model.joint_limits[0].continuous
    assert not model.joint_limits[3].continuous
    assert _maximum_bounded_joint_path_edge(
        (tuple(start), tuple(goal)), model
    ) == pytest.approx(2.0 * abs(start[3]))


def test_physical_joint_path_edge_uses_periodic_continuous_joint_distance() -> None:
    model = _request().robot_model
    start = list(model.default_joint_positions)
    goal = start.copy()
    goal[0] += 2.0 * np.pi - 0.1
    goal[1] += 0.2

    assert _maximum_joint_path_edge((tuple(start), tuple(goal))) > 6.0
    assert _maximum_physical_joint_path_edge(
        (tuple(start), tuple(goal)), model
    ) == pytest.approx(0.2)


def test_cartesian_route_preserves_the_globally_smoother_ik_branch() -> None:
    request = _request()
    start = list(request.initial_state.positions_rad)
    start[0] = 0.5
    poses = tuple(
        RigidTransform(
            request.robot_model.base_frame,
            request.robot_model.tcp_frame,
            (
                (1.0, 0.0, 0.0, float(index)),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
        )
        for index in (1, 2, 3)
    )

    route = _cartesian_continuation_route(
        request,
        _BranchingRouteIK(request.robot_model),
        _FakeVamp(request.robot_model),
        poses,
        tuple(start),
        {},
        TrajectoryStage.EGRESS,
        monotonic(),
        4,
    )

    assert tuple(point[0] for point in route) == pytest.approx((0.5, 0.8, 0.9, 1.0))
    assert max(
        abs(right[0] - left[0]) for left, right in zip(route, route[1:])
    ) == pytest.approx(0.3)


def test_joint_space_portal_compares_a_bounded_set_of_reachable_goals() -> None:
    baseline = _request()
    request = replace(
        baseline,
        planning_policy=replace(
            baseline.planning_policy,
            trajectory_validation_joint_step_rad=10.0,
        ),
    )
    start = request.initial_state.positions_rad

    class _PortalIK(_FakeIK):
        def solve(self, targets, seed_states, constraints):
            del targets, constraints
            solutions = []
            for value in (0.3, 0.6, 0.9, 1.2):
                positions = list(seed_states[0].positions_rad)
                positions[0] = value
                solutions.append(
                    IKSolution(
                        RobotState(self.model.joint_names, tuple(positions)),
                        0.0,
                        0.0,
                    )
                )
            return IKResult(tuple(solutions), None, "portal IK succeeded")

    class _PortalVamp(_FakeVamp):
        def __init__(self, robot_model):
            super().__init__(robot_model)
            self.multi_goal_sizes = []

        def plan_to_any(self, start, goals, environment, *, sample_offset):
            self.multi_goal_sizes.append(len(goals))
            goal = goals[0]
            midpoint = list(start)
            midpoint[0] = {0.3: 2.0, 0.6: 0.3, 0.9: 1.5, 1.2: 0.6}[
                round(goal[0], 1)
            ]
            midpoint[1] += 0.1
            return VampPlan(
                (tuple(start), tuple(midpoint), tuple(goal)),
                0.001,
                1,
                0,
            )

        def motion_valid(self, start, goal, environment):
            del environment
            return not (
                abs(goal[0] - start[0]) > 0.2
                and goal[1] == pytest.approx(start[1])
            )

    vamp = _PortalVamp(request.robot_model)
    validated_goal_values = []

    def reject_first_semantically_invalid_route(path):
        goal_value = path[-1][0]
        validated_goal_values.append(goal_value)
        return goal_value != pytest.approx(0.3)

    path = _joint_space_portal_route(
        request,
        _PortalIK(request.robot_model),
        vamp,
        start,
        _pose(request.robot_model.base_frame, request.robot_model.tcp_frame),
        {},
        19,
        TrajectoryStage.EGRESS,
        monotonic(),
        8,
        path_validator=reject_first_semantically_invalid_route,
    )

    assert vamp.multi_goal_sizes == [4, 3, 2, 1]
    assert validated_goal_values == pytest.approx((0.3, 0.6, 0.9, 1.2))
    assert path[-1][0] == pytest.approx(0.6)


def test_joint_space_portal_rejects_a_large_joint_edge_before_ranking() -> None:
    baseline = _request()
    request = replace(
        baseline,
        planning_policy=replace(
            baseline.planning_policy,
            egress_portal_maximum_joint_edge_rad=1.0,
        ),
    )
    start = request.initial_state.positions_rad

    class _PortalIK(_FakeIK):
        def solve(self, targets, seed_states, constraints):
            del targets, constraints
            solutions = []
            for value in (0.3, 0.6):
                positions = list(seed_states[0].positions_rad)
                positions[0] = value
                solutions.append(
                    IKSolution(
                        RobotState(self.model.joint_names, tuple(positions)),
                        0.0,
                        0.0,
                    )
                )
            return IKResult(tuple(solutions), None, "portal IK succeeded")

    class _PortalVamp(_FakeVamp):
        def __init__(self, robot_model):
            super().__init__(robot_model)
            self.multi_goal_sizes = []

        def plan_to_any(self, start, goals, environment, *, sample_offset):
            del environment, sample_offset
            self.multi_goal_sizes.append(len(goals))
            goal = goals[0]
            midpoint = list(start)
            midpoint[0] = 2.0 if goal[0] == pytest.approx(0.3) else 0.3
            return VampPlan(
                (tuple(start), tuple(midpoint), tuple(goal)),
                0.001,
                1,
                0,
            )

    vamp = _PortalVamp(request.robot_model)
    path = _joint_space_portal_route(
        request,
        _PortalIK(request.robot_model),
        vamp,
        start,
        _pose(request.robot_model.base_frame, request.robot_model.tcp_frame),
        {},
        19,
        TrajectoryStage.EGRESS,
        monotonic(),
        8,
    )

    assert vamp.multi_goal_sizes == [2, 1]
    assert path[-1][0] == pytest.approx(0.6)
    assert _maximum_joint_path_edge(path) <= 1.0


def test_joint_space_portal_rejects_tcp_detours() -> None:
    baseline = _request()
    request = replace(
        baseline,
        planning_policy=replace(
            baseline.planning_policy,
            egress_tcp_path_detour_ratio_limit=1.1,
            egress_portal_maximum_joint_edge_rad=10.0,
            trajectory_validation_joint_step_rad=10.0,
        ),
    )
    start = request.initial_state.positions_rad

    class _PortalIK(_FakeIK):
        def solve(self, targets, seed_states, constraints):
            del targets, constraints
            positions = list(seed_states[0].positions_rad)
            positions[0] = 0.3
            return IKResult(
                (
                    IKSolution(
                        RobotState(self.model.joint_names, tuple(positions)),
                        0.0,
                        0.0,
                    ),
                ),
                None,
                "portal IK succeeded",
            )

    class _DetouringPortalVamp(_FakeVamp):
        def plan_to_any(self, start, goals, environment, *, sample_offset):
            del environment, sample_offset
            midpoint = list(start)
            midpoint[0] = 2.0
            return VampPlan(
                (tuple(start), tuple(midpoint), tuple(goals[0])),
                0.001,
                1,
                0,
            )

    with pytest.raises(_RouteFailure, match="TCP path detour ratio limit"):
        _joint_space_portal_route(
            request,
            _PortalIK(request.robot_model),
            _DetouringPortalVamp(request.robot_model),
            start,
            _pose(request.robot_model.base_frame, request.robot_model.tcp_frame),
            {},
            19,
            TrajectoryStage.EGRESS,
            monotonic(),
            8,
        )


def test_joint_space_portal_reaches_transition_before_vertical_exit() -> None:
    baseline = _request()
    request = replace(
        baseline,
        planning_policy=replace(
            baseline.planning_policy,
            egress_portal_maximum_joint_edge_rad=4.0,
        ),
        vertical_transit_corridor=VerticalTransitCorridor(
            "test_portal",
            (0.4, 0.2),
            0.08,
            0.4,
            0.12,
            0.25,
            0.45,
            0.09,
            0.05,
        ),
        budgets=PlanningBudgets(1, 1, 5.0, 20.0),
    )
    transition_pose = _pose(
        request.robot_model.base_frame,
        request.robot_model.tcp_frame,
        (0.4, 0.2, 0.25),
    )
    upper_pose = _pose(
        request.robot_model.base_frame,
        request.robot_model.tcp_frame,
        (0.4, 0.2, 0.45),
    )
    ik_solver = _RecordingIK(request.robot_model)
    vamp = _FakeVamp(request.robot_model)

    path = _joint_space_portal_then_vertical_route(
        request,
        ik_solver,
        vamp,
        request.initial_state.positions_rad,
        transition_pose,
        upper_pose,
        {},
        19,
        TrajectoryStage.EGRESS,
        monotonic(),
        8,
    )

    tcp_positions = tuple(
        np.asarray(vamp.tcp_pose(positions).matrix, dtype=np.float64)[:3, 3]
        for positions in path
    )
    transition_index = next(
        index
        for index, position in enumerate(tcp_positions)
        if position == pytest.approx((0.4, 0.2, 0.25))
    )
    vertical_positions = tcp_positions[transition_index:]
    assert len(vertical_positions) >= 5
    assert all(position[:2] == pytest.approx((0.4, 0.2)) for position in vertical_positions)
    assert tuple(position[2] for position in vertical_positions) == pytest.approx(
        tuple(sorted(position[2] for position in vertical_positions))
    )
    assert vertical_positions[-1] == pytest.approx((0.4, 0.2, 0.45))
    assert np.allclose(
        tuple(
            (target.matrix[0][3], target.matrix[1][3])
            for target in ik_solver.targets
        ),
        tuple((0.4, 0.2) for _ in ik_solver.targets),
    )


def test_plan_selects_collision_valid_search_attempt_without_tcp_detours() -> None:
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
        ),
        object_config=replace(
            baseline.object_config,
            approach_motion_strategy=LocalMotionStrategy.STAGED_PORTAL,
            payload_motion_strategy=LocalMotionStrategy.STAGED_PORTAL,
        ),
        budgets=PlanningBudgets(1, 2, 5.0, 20.0),
    )
    vamp = _OffsetRouteVamp(
        request.robot_model,
        np.asarray(candidate.T_world_tcp.matrix, dtype=np.float64)[:3, :3],
        0,
    )
    dependencies = replace(
        _fake_dependencies(),
        create_vamp=lambda robot_model: vamp,
    )

    result = plan(request, dependencies=dependencies)

    assert result.success, result.message
    assert result.trajectory is not None
    assert result.metadata["selected_search_attempt"] == 1
    assert result.metadata["selected_sample_offset"] == 1
    selected_loaded_knots = tuple(
        knot.positions_rad
        for knot in result.trajectory.knots
        if knot.stage in (TrajectoryStage.EGRESS, TrajectoryStage.PLACE)
    )
    assert vamp.detour_midpoints
    assert vamp.direct_midpoints
    assert any(point in selected_loaded_knots for point in vamp.direct_midpoints)
    assert not any(point in selected_loaded_knots for point in vamp.detour_midpoints)
    lift_offsets = {
        call[3]
        for call in vamp.plan_calls
        if call[2]["scene"].snapshot_id.endswith(".lift_without_source_support")
    }
    assert 1 in lift_offsets


def test_formal_backend_uses_non_overlapping_deterministic_search_ranges() -> None:
    backend = CONFIG.planning_backends[PROFILE.planning_backend_profile]
    dependencies = CONFIG.planning_dependencies(PROFILE)

    assert backend.ik.random_seed == 19
    assert dependencies.vamp_sample_offset_start == 19
    assert dependencies.vamp_sample_offset_stride >= backend.vamp.max_samples
    offsets = tuple(
        dependencies.vamp_sample_offset_start
        + attempt * dependencies.vamp_sample_offset_stride
        for attempt in range(CONFIG.planning_budgets.search_attempts_per_candidate)
    )
    assert offsets == (19, 240019)


def test_plan_respects_upstream_candidate_order_within_budget() -> None:
    baseline = _request()
    candidate = baseline.candidates[0]
    far = replace(
        candidate,
        candidate_id=f"{candidate.candidate_id}.far",
        T_world_pregrasp=_pose("world", candidate.T_world_pregrasp.child_frame, (4.0, 0.0, 0.2)),
    )
    near = replace(
        candidate,
        candidate_id=f"{candidate.candidate_id}.near",
        T_world_pregrasp=_pose("world", candidate.T_world_pregrasp.child_frame, (0.1, 0.0, 0.2)),
    )
    request = replace(
        baseline,
        candidates=(far, near),
        budgets=PlanningBudgets(1, 1, 5.0, 20.0),
    )

    result = plan(request, dependencies=_fake_dependencies())

    assert result.success, result.message
    assert result.metadata["selected_candidate_id"] == far.candidate_id


def test_staged_portal_screens_candidates_by_nearest_valid_pregrasp_ik() -> None:
    baseline = _request()
    candidate = baseline.candidates[0]
    far = replace(
        candidate,
        candidate_id=f"{candidate.candidate_id}.far",
        T_world_pregrasp=_pose(
            "world", candidate.T_world_pregrasp.child_frame, (4.0, 0.0, 0.2)
        ),
    )
    near = replace(
        candidate,
        candidate_id=f"{candidate.candidate_id}.near",
        T_world_pregrasp=_pose(
            "world", candidate.T_world_pregrasp.child_frame, (0.1, 0.0, 0.2)
        ),
    )
    request = replace(
        baseline,
        candidates=(far, near),
        object_config=replace(
            baseline.object_config,
            approach_motion_strategy=LocalMotionStrategy.STAGED_PORTAL,
        ),
        budgets=PlanningBudgets(1, 1, 5.0, 20.0),
    )

    result = plan(request, dependencies=_fake_dependencies())

    assert result.success, result.message
    assert result.metadata["selected_candidate_id"] == near.candidate_id


def test_compact_joint_path_removes_repeated_branch_boundary_state() -> None:
    path = ((0.0, 0.0), (0.1, 0.2), (0.1, 0.2), (0.2, 0.3))

    assert _compact_joint_path(path) == (path[0], path[1], path[3])


def test_compact_robot_state_path_removes_repeated_approach_boundary() -> None:
    names = ("joint_1", "joint_2")
    states = (
        RobotState(names, (0.1, 0.2)),
        RobotState(names, (0.1, 0.2)),
        RobotState(names, (0.2, 0.3)),
    )

    assert _compact_robot_state_path((0.1, 0.2), states) == (states[2],)


def test_vertical_transit_corridor_materializes_vertical_ingress_and_egress() -> None:
    baseline = _request()
    request = replace(
        baseline,
        candidates=(baseline.candidates[0],),
        placement_goal=replace(
            baseline.placement_goal,
            orientation_constraint=PlacementOrientationConstraint.FREE,
        ),
        planning_policy=replace(
            baseline.planning_policy,
            payload_tcp_max_angular_velocity_rad_s=1000.0,
            payload_tcp_max_translation_velocity_m_s=1000.0,
            payload_tcp_max_translation_acceleration_m_s2=1000.0,
            trajectory_validation_max_samples=100000,
        ),
        object_config=replace(
            baseline.object_config,
            approach_motion_strategy=LocalMotionStrategy.STAGED_PORTAL,
            payload_motion_strategy=LocalMotionStrategy.STAGED_PORTAL,
        ),
        vertical_transit_corridor=VerticalTransitCorridor(
            "test_portal",
            (0.4, 0.2),
            0.08,
            0.4,
            0.12,
            0.25,
            0.45,
            0.09,
            0.03,
        ),
        budgets=PlanningBudgets(1, 1, 5.0, 20.0),
    )

    backend = CONFIG.planning_backends[PROFILE.planning_backend_profile]
    parameterizer = RuckigTimeParameterizer(request.robot_model, backend.ruckig)
    vamp = _FakeVamp(request.robot_model)
    result = plan(
        request,
        dependencies=replace(
            _fake_dependencies(),
            create_vamp=lambda robot_model: vamp,
            create_ruckig=lambda robot_model: parameterizer,
        ),
    )

    assert result.success, result.message
    assert result.trajectory is not None
    goal_tcp = compose(
        result.T_world_object_goal,
        inverse(request.candidates[0].T_tcp_object),
    )
    assert tuple(row[2] for row in goal_tcp.matrix[:3]) == pytest.approx(
        (0.0, 0.0, -1.0)
    )
    grasp_z = request.candidates[0].T_world_tcp.matrix[2][3]
    pregrasp_world = request.candidates[0].T_world_pregrasp
    pregrasp_xy = np.asarray(
        (pregrasp_world.matrix[0][3], pregrasp_world.matrix[1][3])
    )
    pregrasp_stage_tcp = [
        np.asarray(
            (
                10.0 * knot.positions_rad[0],
                10.0 * knot.positions_rad[1],
                10.0 * knot.positions_rad[2],
            )
        )
        for knot in result.trajectory.knots
        if knot.stage is TrajectoryStage.PREGRASP
    ]
    assert len(pregrasp_stage_tcp) >= 3
    portal_index = next(
        index
        for index, xyz in enumerate(pregrasp_stage_tcp)
        if xyz[:2] == pytest.approx((0.4, 0.2))
        and xyz[2] == pytest.approx(grasp_z + 0.4)
    )
    assert pregrasp_stage_tcp[-1] == pytest.approx(
        (*pregrasp_xy, pregrasp_world.matrix[2][3])
    )
    assert portal_index < len(pregrasp_stage_tcp) - 1
    pregrasp_portal_z = [
        10.0 * knot.positions_rad[2]
        for knot in result.trajectory.knots
        if knot.stage is TrajectoryStage.PREGRASP
        and 10.0 * knot.positions_rad[0] == pytest.approx(0.4)
        and 10.0 * knot.positions_rad[1] == pytest.approx(0.2)
    ]
    assert len(pregrasp_portal_z) >= 2
    assert pregrasp_portal_z[0] > pregrasp_portal_z[-1]
    assert max(pregrasp_portal_z) == pytest.approx(grasp_z + 0.4)
    assert min(pregrasp_portal_z) <= (
        grasp_z
        + request.vertical_transit_corridor.minimum_vertical_lift_m
        + request.vertical_transit_corridor.ingress_waypoint_step_m
        + 1e-9
    )
    egress_portal_z = [
        10.0 * knot.positions_rad[2]
        for knot in result.trajectory.knots
        if knot.stage is TrajectoryStage.EGRESS
        and 10.0 * knot.positions_rad[0] == pytest.approx(0.4)
        and 10.0 * knot.positions_rad[1] == pytest.approx(0.2)
    ]
    assert len(egress_portal_z) >= 2
    assert max(egress_portal_z) - min(egress_portal_z) >= 0.15
    assert any(
        value == pytest.approx(grasp_z + 0.25)
        for value in egress_portal_z
    )
    assert any(
        value == pytest.approx(grasp_z + 0.45)
        for value in egress_portal_z
    )
    assert egress_portal_z[0] < egress_portal_z[-1]
    pregrasp_positions = tuple(
        knot.positions_rad
        for knot in result.trajectory.knots
        if knot.stage is TrajectoryStage.PREGRASP
    )
    assert all(
        max(abs(a - b) for a, b in zip(left, right)) > 1e-10
        for left, right in zip(pregrasp_positions, pregrasp_positions[1:])
    )
    assert (
        _maximum_bounded_joint_path_edge(
            pregrasp_positions,
            request.robot_model,
        )
        <= pi
    )
    assert all(
        max(abs(value) for value in knot.velocities_rad_s) <= 1e-12
        for knot in result.trajectory.knots
        if knot.stage is TrajectoryStage.PREGRASP
    )
    loaded_plan_environments = [
        environment
        for environment in vamp.plan_environments
        if environment.get("attached_geometry")
    ]
    # Before source-support departure, LIFT planning excludes only the declared
    # support; non-support scene geometry remains present.
    lift_environments = [
        environment
        for environment in loaded_plan_environments
        if environment["scene"].snapshot_id.endswith(
            ".lift_without_source_support"
        )
    ]
    assert lift_environments
    assert all(
        geometry.collision_group != request.source_support_group
        for environment in lift_environments
        for geometry in environment["scene"].geometries
    )
    maximum_polynomial_xy_deviation_m = 0.0
    portal_xy = np.asarray(request.vertical_transit_corridor.portal_world_xy_m)
    for segment in result.trajectory.segments:
        if segment.stage not in (TrajectoryStage.PREGRASP, TrajectoryStage.EGRESS):
            continue
        endpoints = (
            segment.sample(0.0),
            segment.sample(segment.duration_s),
        )
        endpoint_xy = tuple(
            np.asarray(vamp.tcp_pose(item.positions_rad).matrix, dtype=np.float64)[
                :2, 3
            ]
            for item in endpoints
        )
        if any(np.linalg.norm(item - portal_xy) > 1e-6 for item in endpoint_xy):
            continue
        for index in range(51):
            local_time_s = (
                segment.duration_s
                if index == 50
                else segment.duration_s * index / 50
            )
            sample = segment.sample(local_time_s)
            sample_xy = np.asarray(
                vamp.tcp_pose(sample.positions_rad).matrix, dtype=np.float64
            )[:2, 3]
            maximum_polynomial_xy_deviation_m = max(
                maximum_polynomial_xy_deviation_m,
                float(np.linalg.norm(sample_xy - portal_xy)),
            )
    assert (
        maximum_polynomial_xy_deviation_m
        <= request.vertical_transit_corridor.egress_waypoint_step_m
    )
    transit_egress_attempts = tuple(
        attempt
        for attempt in result.attempts
        if attempt.stage is TrajectoryStage.EGRESS
        and attempt.reason.startswith("EGRESS followed")
    )
    assert len(transit_egress_attempts) == 1
    assert transit_egress_attempts[0].error_code is None
    assert transit_egress_attempts[0].reason == (
        "EGRESS followed the fixed portal-to-transition, vertical exit, and "
        "Cartesian transfer"
    )


def test_continuous_local_motion_uses_the_fixed_semantic_pregrasp(monkeypatch) -> None:
    baseline = _request("rivet_gun")
    request = replace(
        baseline,
        candidates=(baseline.candidates[0],),
        placement_goal=replace(
            baseline.placement_goal,
            orientation_constraint=PlacementOrientationConstraint.FREE,
        ),
        planning_policy=replace(
            baseline.planning_policy,
            payload_tcp_max_angular_velocity_rad_s=1000.0,
            payload_tcp_max_translation_velocity_m_s=1000.0,
            payload_tcp_max_translation_acceleration_m_s2=1000.0,
            trajectory_validation_max_samples=100000,
        ),
        object_config=replace(
            baseline.object_config,
            approach_motion_strategy=LocalMotionStrategy.CONTINUOUS_CARTESIAN,
            payload_motion_strategy=LocalMotionStrategy.CONTINUOUS_CARTESIAN,
        ),
        vertical_transit_corridor=VerticalTransitCorridor(
            "test_portal",
            (0.4, 0.2),
            0.08,
            0.4,
            0.12,
            0.25,
            0.45,
            0.09,
            0.03,
        ),
        budgets=PlanningBudgets(1, 1, 5.0, 20.0),
    )

    portal_calls = 0
    original_portal_route = _joint_space_portal_route

    def track_pregrasp_portal(*args, **kwargs):
        nonlocal portal_calls
        stage = kwargs["stage"] if "stage" in kwargs else args[7]
        if stage is TrajectoryStage.PREGRASP:
            portal_calls += 1
        return original_portal_route(*args, **kwargs)

    monkeypatch.setattr(
        "task2.planning._joint_space_portal_route", track_pregrasp_portal
    )
    result = plan(request, dependencies=_fake_dependencies())

    assert result.success, result.message
    assert result.trajectory is not None
    assert tuple(phase.stage for phase in result.trajectory.phases) == tuple(
        TrajectoryStage
    )
    assert portal_calls == 0
    assert any(
        attempt.stage is TrajectoryStage.PREGRASP
        and attempt.error_code is None
        and "semantic_connector" in attempt.reason
        for attempt in result.attempts
    )


def test_staged_portal_keeps_a_single_semantic_pregrasp_topology(monkeypatch) -> None:
    baseline = _request()
    request = replace(
        baseline,
        candidates=(baseline.candidates[0],),
        placement_goal=replace(
            baseline.placement_goal,
            orientation_constraint=PlacementOrientationConstraint.FREE,
        ),
        planning_policy=replace(
            baseline.planning_policy,
            payload_tcp_max_angular_velocity_rad_s=1000.0,
            payload_tcp_max_translation_velocity_m_s=1000.0,
            payload_tcp_max_translation_acceleration_m_s2=1000.0,
            trajectory_validation_max_samples=100000,
        ),
        object_config=replace(
            baseline.object_config,
            approach_motion_strategy=LocalMotionStrategy.STAGED_PORTAL,
            payload_motion_strategy=LocalMotionStrategy.STAGED_PORTAL,
        ),
        vertical_transit_corridor=VerticalTransitCorridor(
            "test_portal",
            (0.4, 0.2),
            0.08,
            0.4,
            0.12,
            0.25,
            0.45,
            0.09,
            0.03,
        ),
        budgets=PlanningBudgets(1, 1, 5.0, 20.0),
    )

    result = plan(request, dependencies=_fake_dependencies())

    assert result.success, result.message
    assert result.trajectory is not None
    assert tuple(phase.stage for phase in result.trajectory.phases) == tuple(
        TrajectoryStage
    )
    pregrasp_reasons = tuple(
        attempt.reason
        for attempt in result.attempts
        if attempt.stage is TrajectoryStage.PREGRASP
        and attempt.error_code is None
    )
    assert pregrasp_reasons
    assert all("semantic_connector" in reason for reason in pregrasp_reasons)


def test_continuous_local_motion_rejects_failed_cartesian_pregrasp_transit(
    monkeypatch,
) -> None:
    baseline = _request("rivet_gun")
    request = replace(
        baseline,
        candidates=(baseline.candidates[0],),
        placement_goal=replace(
            baseline.placement_goal,
            orientation_constraint=PlacementOrientationConstraint.FREE,
        ),
        object_config=replace(
            baseline.object_config,
            approach_motion_strategy=LocalMotionStrategy.CONTINUOUS_CARTESIAN,
            payload_motion_strategy=LocalMotionStrategy.CONTINUOUS_CARTESIAN,
        ),
        vertical_transit_corridor=VerticalTransitCorridor(
            "test_portal",
            (0.4, 0.2),
            0.08,
            0.4,
            0.12,
            0.25,
            0.45,
            0.09,
            0.03,
        ),
        budgets=PlanningBudgets(1, 1, 5.0, 20.0),
    )

    def reject_cartesian_transit(*args, **kwargs):
        raise _RouteFailure(
            PlanningErrorCode.CONTINUOUS_IK_FAILED,
            TrajectoryStage.PREGRASP,
            "controlled Cartesian transit failure",
        )

    monkeypatch.setattr(
        "task2.planning._cartesian_continuation_route",
        reject_cartesian_transit,
    )

    result = plan(request, dependencies=_fake_dependencies())

    assert not result.success
    assert result.error_code is PlanningErrorCode.CONTINUOUS_IK_FAILED
    assert "controlled Cartesian transit failure" in result.message


def test_adaptive_portal_uses_one_staged_payload_exit_topology(
    monkeypatch,
) -> None:
    baseline = _request()
    adaptive_targeting = target(
        TargetingRequest(
            "marker",
            _pose("world", "object", (0.5, 0.0, 0.1)),
            CONFIG.targeting_object("marker"),
            PROFILE.targeting,
            0.5,
            72,
            TargetingAccessContext((0.4, 0.2), 0.4),
        )
    )
    assert adaptive_targeting.success
    adaptive_candidate = next(
        candidate
        for candidate in adaptive_targeting.candidates
        if candidate.template_id == "marker_parallel_body_portal_tilt"
    )
    request = replace(
        baseline,
        candidates=(adaptive_candidate,),
        placement_goal=replace(
            baseline.placement_goal,
            orientation_constraint=PlacementOrientationConstraint.FREE,
        ),
        planning_policy=replace(
            baseline.planning_policy,
            payload_tcp_max_angular_velocity_rad_s=1000.0,
            payload_tcp_max_translation_velocity_m_s=1000.0,
            payload_tcp_max_translation_acceleration_m_s2=1000.0,
            trajectory_validation_max_samples=100000,
        ),
        object_config=replace(
            baseline.object_config,
            payload_motion_strategy=LocalMotionStrategy.ADAPTIVE_PORTAL,
        ),
        vertical_transit_corridor=VerticalTransitCorridor(
            "test_portal",
            (0.4, 0.2),
            0.08,
            0.4,
            0.12,
            0.25,
            0.45,
            0.09,
            0.03,
        ),
        budgets=PlanningBudgets(1, 1, 5.0, 20.0),
    )
    backend = CONFIG.planning_backends[PROFILE.planning_backend_profile]
    lift_continuation_calls = 0
    lift_pose_step_lengths_m: tuple[float, ...] = ()
    lift_pose_translations: tuple[np.ndarray, ...] = ()
    egress_portal_calls = 0
    original_staged_lift = _staged_lift_route
    original_portal_then_vertical = _joint_space_portal_then_vertical_route

    def track_continuous_ik(*args, **kwargs):
        nonlocal lift_continuation_calls
        stage = kwargs["stage"] if "stage" in kwargs else args[9]
        if stage is TrajectoryStage.LIFT:
            lift_continuation_calls += 1
        return _continuous_ik(*args, **kwargs)

    def track_staged_lift(*args, **kwargs):
        nonlocal lift_pose_step_lengths_m, lift_pose_translations
        poses = kwargs["poses"] if "poses" in kwargs else args[4]
        lift_pose_translations = tuple(
            np.asarray(pose.matrix, dtype=np.float64)[:3, 3] for pose in poses
        )
        lift_pose_step_lengths_m = tuple(
            float(np.linalg.norm(right - left))
            for left, right in zip(
                lift_pose_translations,
                lift_pose_translations[1:],
            )
        )
        return original_staged_lift(*args, **kwargs)

    def track_portal_then_vertical(*args, **kwargs):
        nonlocal egress_portal_calls
        egress_portal_calls += 1
        return original_portal_then_vertical(*args, **kwargs)

    monkeypatch.setattr("task2.planning._continuous_ik", track_continuous_ik)
    monkeypatch.setattr("task2.planning._staged_lift_route", track_staged_lift)
    monkeypatch.setattr(
        "task2.planning._joint_space_portal_then_vertical_route",
        track_portal_then_vertical,
    )
    vamp = _FakeVamp(request.robot_model)
    result = plan(
        request,
        dependencies=replace(
            _fake_dependencies(),
            create_vamp=lambda robot_model: vamp,
            create_ruckig=lambda robot_model: RuckigTimeParameterizer(
                robot_model, backend.ruckig
            ),
        ),
    )

    assert result.success, result.message
    egress_attempts = tuple(
        attempt
        for attempt in result.attempts
        if attempt.stage is TrajectoryStage.EGRESS
        and attempt.error_code is None
        and attempt.reason.startswith("EGRESS followed")
    )
    assert len(egress_attempts) == 1
    assert egress_portal_calls == 1
    assert lift_continuation_calls == 0
    assert lift_pose_translations
    grasp_translation = np.asarray(
        compose(
            inverse(request.T_world_base),
            request.candidates[0].T_world_tcp,
        ).matrix,
        dtype=np.float64,
    )[:3, 3]
    assert lift_pose_translations[0][:2] == pytest.approx(grasp_translation[:2])
    assert lift_pose_translations[0][2] - grasp_translation[2] == pytest.approx(
        request.object_config.payload_initial_vertical_clearance_m
    )
    assert any(
        not np.allclose(translation[:2], grasp_translation[:2])
        for translation in lift_pose_translations[1:]
    )
    assert lift_pose_step_lengths_m
    assert max(lift_pose_step_lengths_m) <= 0.03 + 1e-9
    assert max(lift_pose_step_lengths_m) > 0.01
    assert egress_attempts[0].reason == (
        "EGRESS followed the fixed portal-to-transition, vertical exit, and "
        "Cartesian transfer"
    )
    assert all(
        "adaptive portal" not in attempt.reason
        for attempt in result.attempts
    )


def test_pregrasp_shortcut_keeps_the_farthest_collision_valid_edges() -> None:
    path = tuple((float(index),) for index in range(5))

    class _NoContinuousModel:
        @staticmethod
        def unwrap_positions(positions, reference):
            del reference
            return positions

    class _ShortcutVamp:
        @staticmethod
        def motion_valid(start, goal, environment):
            del environment
            return (int(start[0]), int(goal[0])) in {(0, 3), (3, 4)}

    assert _farthest_collision_valid_shortcut(
        path, _NoContinuousModel(), _ShortcutVamp(), {}, 10.0
    ) == (
        path[0],
        path[3],
        path[4],
    )


def test_shortcut_uses_final_validation_resolution() -> None:
    path = ((0.0, 0.0), (1.0, 1.0), (2.0, 0.0))

    class _NoContinuousModel:
        @staticmethod
        def unwrap_positions(positions, reference):
            del reference
            return positions

    class _ResolutionVamp:
        @staticmethod
        def motion_valid(start, goal, environment):
            del environment
            maximum_delta = max(abs(b - a) for a, b in zip(start, goal))
            if maximum_delta > 0.75:
                return True
            crosses_center = min(start[0], goal[0]) <= 1.0 <= max(start[0], goal[0])
            return not (crosses_center and max(abs(start[1]), abs(goal[1])) < 0.1)

    assert _farthest_collision_valid_shortcut(
        path, _NoContinuousModel(), _ResolutionVamp(), {}, 0.5
    ) == path


def test_shortcut_executes_the_same_short_periodic_edge_that_vamp_validates() -> None:
    request = _request()
    start = request.initial_state.positions_rad
    accumulated = list(start)
    accumulated[0] += 2.0 * np.pi + 0.2
    path = (start, tuple(accumulated))

    shortcut = _farthest_collision_valid_shortcut(
        path,
        request.robot_model,
        _FakeVamp(request.robot_model),
        {},
        0.1,
    )

    assert shortcut[-1][0] == pytest.approx(start[0] + 0.2)
    assert _maximum_joint_path_edge(shortcut) == pytest.approx(0.2)


def test_vamp_rejects_invalid_simplification_without_raw_path_fallback() -> None:
    backend = object.__new__(VampBackend)
    simplified = ((0.0,), (1.0,))

    def path_valid(path, environment):
        del environment
        return False

    backend.path_valid = path_valid  # type: ignore[method-assign]

    with pytest.raises(BackendOperationError, match="simplified path failed"):
        backend._validated_native_path(simplified, object())


def test_pregrasp_detour_ratio_is_a_hard_route_limit(monkeypatch) -> None:
    request = _request()
    request = replace(
        request,
        candidates=(request.candidates[0],),
        planning_policy=replace(
            request.planning_policy,
            pregrasp_joint_path_detour_ratio_limit=1.01,
        ),
    )

    class _DetouringPregraspVamp(_FakeVamp):
        def __init__(self, robot_model):
            super().__init__(robot_model)
        def plan(self, start, goal, environment, *, sample_offset):
            if not self.plan_calls:
                self.plan_environments.append(environment)
                self.plan_calls.append(
                    (tuple(start), tuple(goal), environment, sample_offset)
                )
                midpoint = list(
                    0.5 * (left + right) for left, right in zip(start, goal)
                )
                midpoint[0] += 1.0
                return VampPlan(
                    (tuple(start), tuple(midpoint), tuple(goal)),
                    0.001,
                    1,
                )
            return super().plan(
                start,
                goal,
                environment,
                sample_offset=sample_offset,
            )

    vamp = _DetouringPregraspVamp(request.robot_model)
    monkeypatch.setattr(
        "task2.planning._farthest_collision_valid_shortcut",
        lambda path, *args, **kwargs: path,
    )
    dependencies = replace(
        _fake_dependencies(),
        create_vamp=lambda robot_model: vamp,
    )

    result = plan(request, dependencies=dependencies)

    assert not result.success
    assert result.error_code is PlanningErrorCode.VAMP_SEARCH_FAILED
    assert "detour ratio limit" in result.message


def test_vamp_backend_returns_a_valid_direct_motion_before_running_rrtc() -> None:
    model = _request().robot_model
    start = model.default_joint_positions
    raw_goal = list(start)
    raw_goal[0] += 2.0 * np.pi
    raw_goal[1] += 0.1
    expected_goal = model.unwrap_positions(tuple(raw_goal), start)

    class _DirectRobot:
        def __init__(self):
            self.validated = []

        def validate_motion(self, left, right, environment, check_bounds):
            self.validated.append((left, right, environment, check_bounds))
            return True

        def rrtc(self, *args, **kwargs):
            raise AssertionError("RRTC must not run for a valid direct motion")

    robot = _DirectRobot()
    backend = object.__new__(VampBackend)
    backend._model = model
    backend._robot = robot

    result = backend.plan(start, tuple(raw_goal), object(), sample_offset=19)

    assert result.path == (start, expected_goal)
    assert result.iterations == 0
    assert len(robot.validated) == 1


def test_vamp_backend_multi_goal_direct_motion_selects_the_shortest_goal() -> None:
    model = _request().robot_model
    start = model.default_joint_positions
    farther = list(start)
    farther[1] += 0.2
    nearer = list(start)
    nearer[1] += 0.05

    class _DirectRobot:
        @staticmethod
        def validate_motion(left, right, environment, check_bounds):
            del left, right, environment, check_bounds
            return True

        @staticmethod
        def rrtc(*args, **kwargs):
            raise AssertionError("RRTC must not run when direct goals are valid")

    backend = object.__new__(VampBackend)
    backend._model = model
    backend._robot = _DirectRobot()

    result = backend.plan_to_any(
        start,
        (tuple(farther), tuple(nearer)),
        object(),
        sample_offset=19,
    )

    assert result.path == (start, tuple(nearer))
    assert result.selected_goal_index == 1
    assert result.iterations == 0


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
        budgets=PlanningBudgets(1, 1, 5.0, 20.0),
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


def test_orientation_free_placement_selects_later_lower_cost_payload_route() -> None:
    request = _two_endpoint_free_placement_request()
    candidate = request.candidates[0]
    vamp = _LiftEndpointCostVamp(
        request.robot_model,
        np.asarray(candidate.T_world_tcp.matrix, dtype=np.float64)[:3, :3],
        minimum_is_better=False,
    )
    dependencies = replace(
        _fake_dependencies(),
        create_vamp=lambda robot_model: vamp,
    )

    result = plan(request, dependencies=dependencies)

    assert result.success, result.message
    assert result.trajectory is not None
    assert len(vamp.egress_starts) == 2
    lift = next(
        phase for phase in result.trajectory.phases
        if phase.stage is TrajectoryStage.LIFT
    )
    assert result.trajectory.knots[lift.last_knot].positions_rad == pytest.approx(
        vamp.egress_starts[1]
    )


def test_orientation_free_placement_keeps_minimum_lower_cost_payload_route() -> None:
    request = _two_endpoint_free_placement_request()
    candidate = request.candidates[0]
    vamp = _LiftEndpointCostVamp(
        request.robot_model,
        np.asarray(candidate.T_world_tcp.matrix, dtype=np.float64)[:3, :3],
        minimum_is_better=True,
    )
    dependencies = replace(
        _fake_dependencies(),
        create_vamp=lambda robot_model: vamp,
    )

    result = plan(request, dependencies=dependencies)

    assert result.success, result.message
    assert result.trajectory is not None
    assert len(vamp.egress_starts) == 2
    lift = next(
        phase for phase in result.trajectory.phases
        if phase.stage is TrajectoryStage.LIFT
    )
    assert result.trajectory.knots[lift.last_knot].positions_rad == pytest.approx(
        vamp.egress_starts[0]
    )


def test_orientation_free_placement_tries_higher_lift_after_minimum_fails() -> None:
    request = _two_endpoint_free_placement_request()
    candidate = request.candidates[0]
    vamp = _LiftEndpointVamp(
        request.robot_model,
        np.asarray(candidate.T_world_tcp.matrix, dtype=np.float64)[:3, :3],
        reject_minimum=True,
    )
    dependencies = replace(
        _fake_dependencies(),
        create_vamp=lambda robot_model: vamp,
    )

    result = plan(request, dependencies=dependencies)

    assert result.success, result.message
    assert result.trajectory is not None
    assert len(vamp.loaded_plan_starts) == 3
    minimum_start, selected_start = vamp.loaded_plan_starts[:2]
    assert selected_start != minimum_start
    assert any(
        knot.stage is TrajectoryStage.LIFT
        and knot.positions_rad == pytest.approx(selected_start)
        for knot in result.trajectory.knots
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
        dependencies=_approach_contact_dependencies(
            contact_distance_m=(
                request.planning_policy.approach_contact_transition_m + 0.002
            )
        ),
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

    object_transition_request = replace(
        request,
        object_config=replace(
            request.object_config,
            approach_contact_transition_m=0.03,
        ),
    )
    object_terminal = plan(
        object_transition_request,
        dependencies=_approach_contact_dependencies(contact_distance_m=0.012),
    )
    assert object_terminal.success, object_terminal.message

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


@pytest.mark.parametrize(
    ("proxy_contact_distance_m", "physical_contact", "expected_success"),
    (
        (0.008, False, True),
        (
            CONFIG.planning_policy.approach_contact_transition_m + 0.002,
            False,
            False,
        ),
        (0.008, True, False),
    ),
)
def test_approach_source_support_uses_physical_pad_hull_only_in_terminal_transition(
    proxy_contact_distance_m, physical_contact, expected_success
) -> None:
    request = _request()
    result = plan(
        request,
        dependencies=_approach_support_proxy_dependencies(
            proxy_contact_distance_m,
            physical_contact=physical_contact,
        ),
    )

    assert result.success is expected_success
    if not expected_success:
        assert result.attempts[-1].stage is TrajectoryStage.APPROACH
        assert result.attempts[-1].error_code in (
            PlanningErrorCode.CONTINUOUS_IK_FAILED,
            PlanningErrorCode.STAGE_COLLISION_VIOLATION,
        )


def test_planning_uses_calibrated_vamp_range_and_contact_transition() -> None:
    vamp = CONFIG.planning_backends[PROFILE.planning_backend_profile].vamp
    assert vamp.range_rad == 0.5
    assert vamp.dynamic_domain is False
    assert vamp.simplify_iterations == 4
    assert vamp.robot_sphere_count == 436
    assert tuple(
        (
            group.group,
            group.first_sphere,
            group.sphere_count,
            group.translation_per_closed_fraction_m,
            group.sweep_radius_inflation_m,
            group.reference_closed_fraction,
        )
        for group in vamp.collision_sphere_groups
    ) == (
        ("left_carriage_structure", 148, 4, (0.0, 0.042, 0.0), 0.0, 1.0),
        ("right_carriage_structure", 152, 4, (0.0, -0.042, 0.0), 0.0, 1.0),
        ("left_finger_structure", 156, 12, (0.0, 0.042, 0.0), 0.0, 0.0),
        ("right_finger_structure", 168, 12, (0.0, -0.042, 0.0), 0.0, 0.0),
        ("left_pad", 180, 128, (0.0, 0.042, 0.0), 0.021, None),
        ("right_pad", 308, 128, (0.0, -0.042, 0.0), 0.021, None),
    )
    assert tuple(
        (item.sphere_index, item.center_at_default_m, item.radius_m)
        for item in vamp.collision_sphere_calibrations
    ) == (
        (22, (0.114356, -0.034096, 0.706641), 0.037193),
        (45, (0.27569, -0.019505, 0.428311), 0.035516),
        (46, (0.287652, -0.005146, 0.399621), 0.010515),
        (180, (0.643443, -0.008751, 0.405724), 0.026419),
        (307, (0.668818, 0.005874, 0.411724), 0.024911),
        (308, (0.643443, -0.008751, 0.461724), 0.026419),
        (435, (0.668818, 0.005874, 0.455724), 0.024911),
    )
    assert tuple(
        (item.group, item.translation_per_closed_fraction_m)
        for item in vamp.source_support_contact_hulls
    ) == (
        ("left_pad", (0.0, 0.042, 0.0)),
        ("right_pad", (0.0, -0.042, 0.0)),
    )
    assert CONFIG.planning_policy.approach_contact_transition_m == 0.01
    assert vamp.source_support_contact_hull_clearance_m == 0.0001
    assert (
        CONFIG.planning_object("masking_tape").approach_contact_transition_m
        == 0.03
    )
    assert CONFIG.planning_object("marker").approach_contact_transition_m is None
    marker_geometry = CONFIG.planning_object("marker").attached_geometry
    assert len(marker_geometry) == 1
    assert isinstance(marker_geometry[0].geometry, Capsule)
    assert marker_geometry[0].geometry.radius_m == pytest.approx(0.009)
    assert marker_geometry[0].geometry.length_m == pytest.approx(0.122)
    assert (
        CONFIG.planning_object("masking_tape").approach_motion_strategy
        is LocalMotionStrategy.STAGED_PORTAL
    )
    assert CONFIG.planning_object("masking_tape").payload_motion_strategy is (
        LocalMotionStrategy.STAGED_PORTAL
    )
    assert CONFIG.planning_object("marker").approach_motion_strategy is (
        LocalMotionStrategy.CONTINUOUS_CARTESIAN
    )
    assert {
        CONFIG.planning_object(object_type).approach_motion_strategy
        for object_type in ("marker", "rivet_gun", "standard_part")
    } == {LocalMotionStrategy.CONTINUOUS_CARTESIAN}
    assert CONFIG.planning_object("marker").payload_motion_strategy is (
        LocalMotionStrategy.ADAPTIVE_PORTAL
    )
    assert (
        CONFIG.planning_object("marker").payload_initial_vertical_clearance_m
        == 0.010
    )
    assert (
        CONFIG.planning_object("drill_bit").payload_initial_vertical_clearance_m
        == 0.020
    )
    assert all(
        CONFIG.planning_object(object_type).payload_initial_vertical_clearance_m
        == 0.0
        for object_type in ("masking_tape", "rivet_gun", "standard_part")
    )
    assert {
        CONFIG.planning_object(object_type).payload_motion_strategy
        for object_type in ("rivet_gun", "standard_part")
    } == {LocalMotionStrategy.CONTINUOUS_CARTESIAN}
    assert CONFIG.planning_policy.pregrasp_joint_path_detour_ratio_limit == 3.0
    assert CONFIG.planning_policy.pregrasp_tcp_path_detour_ratio_limit == 2.25
    assert CONFIG.planning_policy.egress_tcp_path_detour_ratio_limit == 2.25
    assert CONFIG.planning_policy.egress_portal_maximum_joint_edge_rad == 1.9
    assert CONFIG.planning_policy.payload_tcp_max_angular_velocity_rad_s == 0.66
    assert CONFIG.planning_policy.payload_tcp_max_translation_velocity_m_s == 0.30
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


def test_native_attachment_spheres_apply_object_local_geometry_pose() -> None:
    fake_backend = SimpleNamespace(
        tcp_pose=lambda _configuration: RigidTransform(
            "base",
            "tcp",
            (
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
        ),
        _vamp=SimpleNamespace(Sphere=lambda center, radius: (tuple(center), radius)),
    )
    T_tcp_object = RigidTransform(
        "tcp",
        "object",
        (
            (1.0, 0.0, 0.0, 0.10),
            (0.0, 1.0, 0.0, 0.20),
            (0.0, 0.0, 1.0, 0.30),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    attached = AttachedGeometry(
        "offset_sphere",
        RigidTransform(
            "object",
            "offset_sphere",
            (
                (1.0, 0.0, 0.0, 0.01),
                (0.0, 1.0, 0.0, 0.02),
                (0.0, 0.0, 1.0, 0.03),
                (0.0, 0.0, 0.0, 1.0),
            ),
        ),
        Sphere(0.04),
    )

    spheres = VampBackend._attachment_spheres(
        fake_backend,
        (0.0,),
        T_tcp_object,
        (attached,),
    )

    assert len(spheres) == 1
    assert spheres[0][0] == pytest.approx((0.11, 0.22, 0.33))
    assert spheres[0][1] == pytest.approx(0.04)


def test_semantic_attachment_contacts_apply_object_local_geometry_pose_once() -> None:
    class NativeSphere:
        def __init__(self, center, radius):
            self.x, self.y, self.z = center
            self.r = radius

    identity = _pose("base", "tcp")
    backend = SimpleNamespace(
        _attachment_contact_cache={},
        _semantic_polyhedra={},
        _model=SimpleNamespace(base_frame="base"),
        _vamp=SimpleNamespace(Sphere=NativeSphere),
        _tcp_matrix=lambda _configuration: np.eye(4, dtype=np.float64),
        _geometries_in_base=lambda geometries, _T_world_base: geometries,
        _attachment_spheres=None,
    )
    backend._attachment_spheres = lambda configuration, T_tcp_object, attached: (
        VampBackend._attachment_spheres(
            backend, configuration, T_tcp_object, attached
        )
    )
    T_tcp_object = RigidTransform(
        "tcp",
        "object",
        (
            (1.0, 0.0, 0.0, 0.10),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    attached = AttachedGeometry(
        "offset_sphere",
        RigidTransform(
            "object",
            "offset_sphere",
            (
                (1.0, 0.0, 0.0, 0.01),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
        ),
        Sphere(0.001),
    )
    obstacle = SceneGeometry(
        "obstacle",
        "tank",
        "source_support",
        RigidTransform(
            "base",
            "obstacle",
            (
                (1.0, 0.0, 0.0, 0.11),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
        ),
        Box((0.002, 0.002, 0.002)),
    )

    contacts = VampBackend.semantic_attachment_contacts(
        backend,
        (0.0,),
        T_tcp_object,
        (attached,),
        (obstacle,),
        identity,
    )

    assert contacts == frozenset(("source_support",))


def test_semantic_finger_structure_spheres_follow_commanded_opening() -> None:
    config = CONFIG.planning_backends[PROFILE.planning_backend_profile].vamp
    robot_spheres = tuple(
        SimpleNamespace(x=0.0, y=0.0, z=0.0, r=0.01)
        for _ in range(config.robot_sphere_count)
    )
    backend = SimpleNamespace(
        _config=config,
        _collision_group_by_sphere=tuple(
            next(
                (
                    group
                    for group in config.collision_sphere_groups
                    if group.first_sphere <= index
                    < group.first_sphere + group.sphere_count
                ),
                None,
            )
            for index in range(config.robot_sphere_count)
        ),
        _tcp_matrix=lambda _configuration: np.eye(4, dtype=np.float64),
        robot_spheres=lambda _configuration: robot_spheres,
        _vamp=SimpleNamespace(
            Sphere=lambda center, radius: SimpleNamespace(
                x=center[0], y=center[1], z=center[2], r=radius
            )
        ),
    )

    open_spheres = VampBackend._semantic_robot_spheres(backend, (0.0,), 0.084)
    closed_spheres = VampBackend._semantic_robot_spheres(backend, (0.0,), 0.0)

    assert open_spheres[148][0].y == pytest.approx(-0.042)
    assert closed_spheres[148][0].y == pytest.approx(0.0)
    assert open_spheres[152][0].y == pytest.approx(0.042)
    assert closed_spheres[152][0].y == pytest.approx(0.0)
    assert open_spheres[148][1] == "left_carriage_structure"
    assert open_spheres[152][1] == "right_carriage_structure"
    assert open_spheres[156][0].y == pytest.approx(0.0)
    assert closed_spheres[156][0].y == pytest.approx(0.042)
    assert open_spheres[168][0].y == pytest.approx(0.0)
    assert closed_spheres[168][0].y == pytest.approx(-0.042)
    assert open_spheres[156][1] == "left_finger_structure"
    assert open_spheres[168][1] == "right_finger_structure"


def test_half_arm_proxy_calibration_retains_the_previous_physical_surface_coverage() -> None:
    spheres = yaml.safe_load(
        (ROOT / "configs/curobo/gen3/task2_active_wrist_parallel_spheres.yml").read_text()
    )["collision_spheres"]["half_arm_2_link"]
    centers = np.asarray([item["center"] for item in spheres], dtype=np.float64)
    new_effective_radii = np.asarray(
        [float(item["radius"]) + 0.001 for item in spheres], dtype=np.float64
    )
    old_physical_radii = np.asarray(
        [float(item["radius"]) for item in spheres], dtype=np.float64
    )
    old_physical_radii[1] = 0.037698

    mesh_path = ROOT / "examples/mujoco/kinova_gen3/assets/half_arm_2_link.stl"
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
    surface_points = []
    for triangle in np.asarray(records["vertices"], dtype=np.float64):
        for first_weight in range(7):
            for second_weight in range(7 - first_weight):
                weights = np.asarray(
                    (first_weight, second_weight, 6 - first_weight - second_weight),
                    dtype=np.float64,
                ) / 6.0
                surface_points.append(weights @ triangle)
    points = np.asarray(surface_points, dtype=np.float64)
    distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
    previously_covered = np.any(distances <= old_physical_radii[None, :], axis=1)

    assert np.any(previously_covered)
    assert np.all(
        np.any(
            distances[previously_covered] <= new_effective_radii[None, :],
            axis=1,
        )
    )


def test_forearm_proxy_split_preserves_the_previous_physical_surface_coverage() -> None:
    spheres = yaml.safe_load(
        (ROOT / "configs/curobo/gen3/task2_active_wrist_parallel_spheres.yml").read_text()
    )["collision_spheres"]["forearm_link"]
    centers = np.asarray([item["center"] for item in spheres], dtype=np.float64)
    radii = np.asarray([item["radius"] for item in spheres], dtype=np.float64)
    old_centers = np.vstack(
        (
            (-0.002957, -0.173842, -0.00483),
            (0.0, -0.185257, -0.006375),
            centers[6:],
        )
    )
    old_radii = np.concatenate(((0.049268, 0.035643), radii[6:]))

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
    surface_points = []
    for triangle in np.asarray(records["vertices"], dtype=np.float64):
        for first_weight in range(7):
            for second_weight in range(7 - first_weight):
                weights = np.asarray(
                    (first_weight, second_weight, 6 - first_weight - second_weight),
                    dtype=np.float64,
                ) / 6.0
                surface_points.append(weights @ triangle)
    points = np.asarray(surface_points, dtype=np.float64)
    old_distances = np.linalg.norm(
        points[:, None, :] - old_centers[None, :, :], axis=2
    )
    new_distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
    previously_covered = np.any(old_distances <= old_radii[None, :], axis=1)

    assert len(spheres) == 14
    assert max(radii[:6]) < 0.049268
    assert np.any(previously_covered)
    assert np.all(np.any(new_distances[previously_covered] <= radii[None, :], axis=1))


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
        ("left_jaw", "task2_tool_left_pad_collision", "left_pad", 1.0),
        ("right_jaw", "task2_tool_right_pad_collision", "right_pad", -1.0),
    )
    support_hulls = {
        item.group: item
        for item in CONFIG.planning_backends[
            PROFILE.planning_backend_profile
        ].vamp.source_support_contact_hulls
    }
    for body_name, sphere_link, contact_group, closed_direction in groups:
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
        configured_hull = np.asarray(
            support_hulls[contact_group].vertices_at_maximum_opening_m,
            dtype=np.float64,
        )
        vertex_distances = np.linalg.norm(
            configured_hull[:, None, :] - open_vertices[None, :, :], axis=2
        )
        assert np.all(np.min(vertex_distances, axis=1) <= 1e-12)
        assert np.all(np.min(vertex_distances, axis=0) <= 1e-12)
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
        budgets=replace(request.budgets, search_attempts_per_candidate=2),
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
    assert grasp_phase.first_knot < grasp_phase.last_knot
    assert place_phase.first_knot <= place_phase.last_knot
    assert all(
        knot.velocities_rad_s
        == pytest.approx((0.0,) * len(request.robot_model.joint_names))
        for knot in result.trajectory.knots
    )
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


def test_payload_retiming_only_slows_the_edge_that_exceeds_tcp_limit() -> None:
    baseline = _request()
    request = replace(
        baseline,
        planning_policy=replace(
            baseline.planning_policy,
            payload_tcp_max_angular_velocity_rad_s=100.0,
            payload_tcp_max_translation_velocity_m_s=0.3,
            payload_tcp_max_translation_acceleration_m_s2=100.0,
        ),
    )
    backend = CONFIG.planning_backends[PROFILE.planning_backend_profile]
    ruckig = RuckigTimeParameterizer(request.robot_model, backend.ruckig)
    positions = [list(request.initial_state.positions_rad) for _ in range(8)]
    positions[5][0] += 0.2
    positions[6][0] += 0.2
    positions[6][1] += 0.2
    positions[7] = list(positions[6])
    channels = {
        PROFILE.planning.actuator_position_channel: 0.01,
        PROFILE.planning.actuator_force_limit_channel: 1.0,
    }
    stages = (
        TrajectoryStage.PREGRASP,
        TrajectoryStage.PREGRASP,
        TrajectoryStage.APPROACH,
        TrajectoryStage.GRASP,
        TrajectoryStage.LIFT,
        TrajectoryStage.EGRESS,
        TrajectoryStage.EGRESS,
        TrajectoryStage.PLACE,
    )
    waypoints = tuple(
        GeometricWaypoint(
            stage,
            tuple(position),
            dict(channels),
            minimum_duration_s=(
                0.01
                if index > 0 and position == positions[index - 1]
                else 0.0
            ),
        )
        for index, (stage, position) in enumerate(zip(stages, positions, strict=True))
    )
    route = _Route(
        waypoints,
        (),
        _pose("world", "object_goal"),
        0.0,
    )
    nominal = ruckig.parameterize(waypoints, ())

    retimed = _retime_payload_motion(
        request,
        route,
        nominal,
        _EdgeSpeedVamp(request.robot_model),
        ruckig,
        monotonic(),
    )

    assert retimed.segments[4].duration_s == pytest.approx(
        nominal.segments[4].duration_s
    )
    assert retimed.segments[5].duration_s > nominal.segments[5].duration_s


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
        object_config=replace(
            baseline.object_config,
            payload_motion_strategy=LocalMotionStrategy.CONTINUOUS_CARTESIAN,
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
        assert trajectory.knots[lift.last_knot].velocities_rad_s == pytest.approx(
            (0.0,) * len(request.robot_model.joint_names)
        )
        assert trajectory.knots[egress.last_knot].velocities_rad_s == pytest.approx(
            (0.0,) * len(request.robot_model.joint_names)
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


def test_box_attachment_semantic_collision_uses_exact_oriented_geometry() -> None:
    slender = SceneGeometry(
        "slender",
        "attached_object",
        "attached_object",
        _pose("base", "slender"),
        Box((0.08, 0.01, 0.01)),
    )
    separated = SceneGeometry(
        "separated",
        "tank",
        "tank_environment",
        _pose("base", "separated", (0.0, 0.02, 0.0)),
        Box((0.08, 0.01, 0.01)),
    )
    intersecting = SceneGeometry(
        "intersecting",
        "tank",
        "tank_environment",
        _pose("base", "intersecting", (0.0, 0.009, 0.0)),
        Box((0.08, 0.01, 0.01)),
    )

    slender_polyhedron = _prepared_convex_polyhedron(slender)
    separated_polyhedron = _prepared_convex_polyhedron(separated)
    intersecting_polyhedron = _prepared_convex_polyhedron(intersecting)
    assert slender_polyhedron is not None
    assert separated_polyhedron is not None
    assert intersecting_polyhedron is not None
    assert not _prepared_polyhedra_intersect(
        slender_polyhedron, separated_polyhedron
    )
    assert _prepared_polyhedra_intersect(
        slender_polyhedron, intersecting_polyhedron
    )
    assert not _prepared_polyhedra_within_clearance(
        slender_polyhedron,
        separated_polyhedron,
        0.004,
    )
    assert _prepared_polyhedra_within_clearance(
        slender_polyhedron,
        separated_polyhedron,
        0.011,
    )


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
        create_ik=lambda robot_model: _WrappedContinuousIK(
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
    request = replace(
        baseline,
        candidates=(baseline.candidates[0],),
        object_config=replace(
            baseline.object_config,
            payload_motion_strategy=LocalMotionStrategy.CONTINUOUS_CARTESIAN,
        ),
    )
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
    )
    result = plan(request, dependencies=CONFIG.planning_dependencies(PROFILE))
    assert result.success, f"{result.error_code}: {result.message}"
    assert result.trajectory is not None
    assert tuple(phase.stage for phase in result.trajectory.phases) == tuple(TrajectoryStage)
    request.validate()
