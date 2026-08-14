from __future__ import annotations

from dataclasses import replace
import subprocess
import sys
from pathlib import Path

import pytest

from task2 import plan, target
from task2.backends.errors import BackendUnavailableError
from task2.backends.ruckig import RuckigTimeParameterizer
from task2.backends.vamp import VampPlan
from task2.config import load_task2_config
from task2.contracts import (
    AllowedContact,
    Box,
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
    for stage in (TrajectoryStage.GRASP, TrajectoryStage.PLACE):
        phase = next(item for item in trajectory.phases if item.stage is stage)
        assert all(
            abs(value) < 1e-8
            for value in trajectory.knots[phase.last_knot].velocities_rad_s
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
