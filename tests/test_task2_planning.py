from __future__ import annotations

from dataclasses import replace
import json
import subprocess
import sys

import pytest

from task2 import plan, target
from task2.backends.errors import BackendUnavailableError
from task2.backends.vamp import VampBackend, VampPlan
from task2.composition import load_planning_config, load_targeting_config
from task2.contracts import (
    AllowedContact,
    AttachedGeometry,
    Box,
    Capsule,
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
    StaticTriangleMesh,
    TargetingRequest,
    TrajectoryStage,
)
from task2.ik import IKResult, IKSolution
from task2.backends.vamp import _attached_sphere_specs


IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


class _FakeVamp:
    version = "0.6.4"

    def __init__(self, robot_model, planning_config):
        self.model = robot_model

    def environment(self, scene, **kwargs):
        return kwargs

    def plan(self, start, goal, environment, *, seed):
        return VampPlan((tuple(start), tuple(goal)), 0.001, 1)

    def configuration_valid(self, configuration, environment):
        return True

    def motion_valid(self, start, goal, environment):
        return True

    def semantic_target_contacts(self, configuration, geometries, opening_m, T_world_base):
        if geometries and geometries[0].collision_group == "target" and opening_m < 0.02:
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


class _ApproachPadContactVamp(_FakeVamp):
    def semantic_target_contacts(self, configuration, geometries, opening_m, T_world_base):
        if (
            geometries
            and geometries[0].collision_group == "target"
            and configuration[2] < 0.02
        ):
            return frozenset(("left_pad", "right_pad"))
        return frozenset()


class _SeedSelectionVamp(_FakeVamp):
    def plan(self, start, goal, environment, *, seed):
        if seed % 2 == 0:
            return VampPlan((tuple(start), tuple(goal)), 0.001, 1)
        midpoint = tuple(
            left + 0.5 * (right - left) + (0.05 if index == 0 else 0.0)
            for index, (left, right) in enumerate(zip(start, goal))
        )
        return VampPlan((tuple(start), midpoint, tuple(goal)), 0.001, 1)


class _FakeIK:
    version = "0.8.0"

    def __init__(self, robot_model, planning_config, *, project_root, random_seed):
        self.model = robot_model

    def solve(self, targets, seed_states, constraints):
        matrix = targets[0].matrix
        values = list(seed_states[0].positions_rad)
        values[0] = 0.1 * matrix[0][3]
        values[1] = 0.1 * matrix[1][3]
        values[2] = 0.1 * matrix[2][3]
        state = RobotState(self.model.joint_names, tuple(values))
        return IKResult((IKSolution(state, 0.0, 0.0),), None, "fake IK succeeded")


def _transform(parent: str, child: str, xyz=(0.0, 0.0, 0.0)) -> RigidTransform:
    rows = [list(row) for row in IDENTITY]
    for index, value in enumerate(xyz):
        rows[index][3] = value
    return RigidTransform(parent, child, tuple(tuple(row) for row in rows))  # type: ignore[arg-type]


def _request() -> PlanningRequest:
    targeting = load_targeting_config(
        "configs/task2/objects.yaml", "configs/task2/end_effectors.yaml"
    )
    composition = load_planning_config(
        "configs/task2/robot.yaml",
        "configs/task2/planning.yaml",
        "configs/task2/objects.yaml",
        "configs/task2/end_effectors.yaml",
    )
    targeting_result = target(
        TargetingRequest(
            object_type="marker",
            T_world_object=_transform("world", "object", (0.5, 0.0, 0.1)),
            object_config=targeting.objects[0],
            end_effector_config=targeting.end_effectors[0],
            continuous_yaw_resolution_rad=0.5,
            maximum_candidates=2,
        )
    )
    assert targeting_result.success
    scene = SceneSnapshot(
        "regression_scene",
        "world",
        (
            SceneGeometry(
                "target_sphere",
                "target_object",
                "target",
                _transform("world", "target_sphere", (0.5, 0.0, 0.1)),
                Sphere(0.01),
            ),
            SceneGeometry(
                "floor",
                "tank",
                "source_support",
                _transform("world", "floor", (0.0, 0.0, -1.0)),
                Box((2.0, 2.0, 0.1)),
            ),
        ),
        (AllowedContact("target", "source_support"),),
    )
    return PlanningRequest(
        request_id="known_regression",
        candidates=targeting_result.candidates[:1],
        initial_state=RobotState(
            composition.robot_model.joint_names,
            composition.robot_model.default_joint_positions,
        ),
        robot_model=composition.robot_model,
        T_world_base=_transform("world", composition.robot_model.base_frame),
        scene=scene,
        target_object_id="target_object",
        source_support_group="source_support",
        placement_goal=PlacementGoal(
            "outside",
            PlacementGoalKind.EXACT,
            _transform("world", composition.robot_model.tcp_frame, (0.6, 0.0, 0.6)),
        ),
        object_config=composition.get_object("marker"),
        end_effector_config=composition.get_end_effector("parallel_gripper"),
        planning_config=composition.planning_config,
        random_seed=7,
        budgets=PlanningBudgets(1, 1, 2.0, 10.0),
    )


def _install_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("task2.planning.VampBackend", _FakeVamp)
    monkeypatch.setattr(
        "task2.planning.create_ik_solver",
        lambda backend, robot_model, planning_config, **kwargs: _FakeIK(
            robot_model, planning_config, **kwargs
        ),
    )


def test_plan_exposes_six_phases_named_channels_stops_and_ruckig_limits(monkeypatch):
    _install_fakes(monkeypatch)
    request = _request()
    result = plan(request)
    assert result.success, result.message
    assert result.trajectory is not None
    trajectory = result.trajectory
    json.dumps(result.to_dict())
    assert tuple(phase.stage for phase in trajectory.phases) == tuple(TrajectoryStage)
    assert tuple(event.event for event in trajectory.events) == ("grip", "release")
    assert len(trajectory.knots) == len(trajectory.segments) + 1
    assert len(trajectory.knots) < 100
    expected_channels = {
        request.end_effector_config.actuator_position_channel,
        request.end_effector_config.actuator_force_limit_channel,
    }
    assert all(set(knot.actuator_channels) == expected_channels for knot in trajectory.knots)
    for stage in (TrajectoryStage.GRASP, TrajectoryStage.PLACE):
        phase = next(item for item in trajectory.phases if item.stage is stage)
        assert max(abs(value) for value in trajectory.knots[phase.last_knot].velocities_rad_s) == 0.0
    for knot in trajectory.knots:
        for index, limits in enumerate(request.robot_model.joint_limits):
            assert abs(knot.velocities_rad_s[index]) <= limits.maximum_velocity_rad_s + 1e-7
            assert abs(knot.accelerations_rad_s2[index]) <= limits.maximum_acceleration_rad_s2 + 1e-7
    for segment in trajectory.segments:
        assert segment.sections
        assert sum(section.duration_s for section in segment.sections) == pytest.approx(
            segment.duration_s
        )
        sampled = segment.sample(0.5 * segment.duration_s)
        assert sampled.time_s == pytest.approx(
            segment.start_time_s + 0.5 * segment.duration_s
        )
        for section in segment.sections:
            _, velocity, acceleration = section.sample(0.5 * section.duration_s)
            for index, limits in enumerate(request.robot_model.joint_limits):
                assert abs(velocity[index]) <= limits.maximum_velocity_rad_s + 1e-7
                assert abs(acceleration[index]) <= limits.maximum_acceleration_rad_s2 + 1e-7
                assert abs(6.0 * section.position_coefficients[index][3]) <= (
                    limits.maximum_jerk_rad_s3 + 1e-7
                )
        for left, right in zip(segment.sections, segment.sections[1:]):
            left_state = left.sample(left.duration_s)
            right_state = right.sample(0.0)
            for left_values, right_values in zip(left_state, right_state):
                assert left_values == pytest.approx(right_values)

    position_channel = request.end_effector_config.actuator_position_channel
    grasp_segments = [
        segment for segment in trajectory.segments if segment.stage is TrajectoryStage.GRASP
    ]
    command = request.candidates[0].grasp_command
    assert len(grasp_segments) == 2
    assert grasp_segments[0].start_actuator_channels[position_channel] == pytest.approx(
        command.pregrasp_opening_m
    )
    assert grasp_segments[0].end_actuator_channels[position_channel] == pytest.approx(
        command.target_opening_m
    )
    expected_closing_duration = (
        command.pregrasp_opening_m - command.target_opening_m
    ) / command.closing_speed_m_s
    assert grasp_segments[0].duration_s + 1e-9 >= expected_closing_duration
    assert grasp_segments[1].duration_s >= command.dwell_s
    release = trajectory.events[-1]
    assert release.actuator_channels[position_channel] == pytest.approx(
        request.end_effector_config.open_position_m
    )
    assert release.time_s == pytest.approx(trajectory.phases[-1].end_time_s)


def test_plan_enforces_candidate_declared_grasp_contacts(monkeypatch):
    _install_fakes(monkeypatch)
    request = _request()
    bad_candidate = replace(request.candidates[0], expected_contacts=("left_pad",))
    result = plan(replace(request, candidates=(bad_candidate,)))
    assert result.error_code is PlanningErrorCode.STAGE_COLLISION_VIOLATION
    assert "forbidden target contacts" in result.message


def test_plan_allows_candidate_declared_pad_contact_during_approach(monkeypatch):
    monkeypatch.setattr("task2.planning.VampBackend", _ApproachPadContactVamp)
    monkeypatch.setattr(
        "task2.planning.create_ik_solver",
        lambda backend, robot_model, planning_config, **kwargs: _FakeIK(
            robot_model, planning_config, **kwargs
        ),
    )
    result = plan(_request())
    assert result.success, result.message


def test_plan_maps_unavailable_backend_without_fallback(monkeypatch):
    class _Unavailable:
        def __init__(self, *args, **kwargs):
            raise BackendUnavailableError("patched VAMP is unavailable")

    monkeypatch.setattr("task2.planning.VampBackend", _Unavailable)
    result = plan(_request())
    assert result.error_code is PlanningErrorCode.BACKEND_UNAVAILABLE
    assert result.trajectory is None
    assert "patched VAMP" in result.message


def test_plan_distinguishes_invalid_config_from_invalid_request():
    request = _request()
    invalid = replace(
        request,
        planning_config=replace(request.planning_config, vamp_version="0.6.3"),
    )
    result = plan(invalid)
    assert result.error_code is PlanningErrorCode.INVALID_CONFIG


def test_plan_selects_shorter_route_across_same_candidate_seeds(monkeypatch):
    monkeypatch.setattr("task2.planning.VampBackend", _SeedSelectionVamp)
    monkeypatch.setattr(
        "task2.planning.create_ik_solver",
        lambda backend, robot_model, planning_config, **kwargs: _FakeIK(
            robot_model, planning_config, **kwargs
        ),
    )
    request = _request()
    result = plan(
        replace(request, budgets=PlanningBudgets(1, 2, 2.0, 10.0))
    )
    assert result.success, result.message
    assert result.metadata["selected_seed"] == request.random_seed + 1


def test_known_planning_regression_is_deterministic(monkeypatch):
    _install_fakes(monkeypatch)
    first = plan(_request())
    second = plan(_request())
    assert first.success and second.success
    assert first.trajectory == second.trajectory
    assert first.metadata == second.metadata


def test_importing_public_task2_does_not_load_native_backends():
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
    )
    assert completed.returncode == 0, completed.stderr


def test_plan_dispatches_ik_backend_from_planning_config(monkeypatch):
    selected = []

    def create(backend, robot_model, planning_config, **kwargs):
        selected.append(backend)
        return _FakeIK(robot_model, planning_config, **kwargs)

    monkeypatch.setattr("task2.planning.VampBackend", _FakeVamp)
    monkeypatch.setattr("task2.planning.create_ik_solver", create)
    request = _request()
    result = plan(
        replace(
            request,
            planning_config=replace(request.planning_config, ik_backend="isolated_test"),
        )
    )
    assert result.success, result.message
    assert selected == ["isolated_test"]


def test_plan_maps_unregistered_ik_backend_to_backend_unavailable(monkeypatch):
    monkeypatch.setattr("task2.planning.VampBackend", _FakeVamp)
    request = _request()
    result = plan(
        replace(
            request,
            planning_config=replace(request.planning_config, ik_backend="not_registered"),
        )
    )
    assert result.error_code is PlanningErrorCode.BACKEND_UNAVAILABLE
    assert "not registered" in result.message


def test_scene_snapshot_supports_convex_and_static_triangle_mesh_contracts():
    tetrahedron = ConvexHull(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)),
    )
    mesh = StaticTriangleMesh(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ((0, 1, 2),),
    )
    scene = SceneSnapshot(
        "mesh_scene",
        "world",
        (
            SceneGeometry("convex", "one", "obstacle", _transform("world", "convex"), tetrahedron),
            SceneGeometry("mesh", "two", "obstacle", _transform("world", "mesh"), mesh),
        ),
    )
    scene.validate()
    backend = object.__new__(VampBackend)
    with pytest.raises(BackendUnavailableError, match="explicit PointCloud"):
        backend._add_scene_geometry(object(), scene.geometries[0])
    with pytest.raises(BackendUnavailableError, match="explicit PointCloud"):
        backend._add_scene_geometry(object(), scene.geometries[1])


def test_vamp_attachment_decomposes_capsules_and_boxes_into_covering_spheres():
    capsule = AttachedGeometry(
        "capsule",
        _transform("object", "capsule", (0.1, 0.0, 0.0)),
        Capsule(0.01, 0.05),
    )
    box = AttachedGeometry(
        "box",
        _transform("object", "box", (0.0, 0.2, 0.0)),
        Box((0.02, 0.04, 0.02)),
    )
    capsule_spheres = _attached_sphere_specs(capsule)
    box_spheres = _attached_sphere_specs(box)
    assert len(capsule_spheres) > 2
    assert len(box_spheres) == 2
    assert all(radius >= capsule.geometry.radius_m for _, radius in capsule_spheres)
    assert all(center[0] == pytest.approx(0.1) for center, _ in capsule_spheres)
    assert all(center[1] > 0.18 for center, _ in box_spheres)
