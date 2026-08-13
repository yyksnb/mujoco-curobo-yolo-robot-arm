from __future__ import annotations

import ast
from dataclasses import replace
from math import pi
from pathlib import Path

import pytest

from task2.composition import (
    get_end_effector_targeting_config,
    get_targeting_object_config,
    load_targeting_config,
)
from task2.contracts import RigidTransform, TargetingErrorCode, TargetingRequest, compose
from task2.targeting import target


REPO_ROOT = Path(__file__).resolve().parents[1]
OBJECTS_CONFIG = REPO_ROOT / "configs/task2/objects.yaml"
END_EFFECTORS_CONFIG = REPO_ROOT / "configs/task2/end_effectors.yaml"


def _world_pose(x_m: float = 0.4) -> RigidTransform:
    return RigidTransform(
        parent_frame="world",
        child_frame="object",
        matrix=(
            (1.0, 0.0, 0.0, x_m),
            (0.0, 1.0, 0.0, 0.1),
            (0.0, 0.0, 1.0, 0.02),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )


def _request(object_type: str, *, x_m: float = 0.4) -> TargetingRequest:
    config = load_targeting_config(OBJECTS_CONFIG, END_EFFECTORS_CONFIG)
    return TargetingRequest(
        object_type=object_type,
        T_world_object=_world_pose(x_m),
        object_config=get_targeting_object_config(config, object_type),
        end_effector_config=get_end_effector_targeting_config(config, "parallel_gripper"),
        continuous_yaw_resolution_rad=pi / 2.0,
        maximum_candidates=16,
    )


def test_four_calibrated_templates_are_deterministic_and_executor_compatible() -> None:
    expected = {
        "marker": (2, 0.013014206, 0.75),
        "masking_tape": (4, 0.074333332, 2.0),
        "rivet_gun": (2, 0.011946974, 24.0),
        "standard_part": (2, 0.002745694, 0.30),
    }

    for object_type, (candidate_count, target_opening_m, grip_force_limit_n) in expected.items():
        request = _request(object_type)
        result = target(request)
        repeated = target(request)
        moved = target(replace(request, T_world_object=_world_pose(0.7)))

        assert result.success
        assert result == repeated
        assert len(result.candidates) == candidate_count
        calibrated_translation = tuple(
            request.object_config.templates[0].T_object_tcp.matrix[row][3] for row in range(3)
        )
        assert [item.candidate_id for item in result.candidates] == [
            item.candidate_id for item in moved.candidates
        ]
        for candidate in result.candidates:
            command = candidate.grasp_command
            capabilities = request.end_effector_config.capabilities
            assert command.target_opening_m == pytest.approx(target_opening_m)
            assert command.grip_force_limit_n == pytest.approx(grip_force_limit_n)
            assert command.calibration == "simulation_nominal"
            assert capabilities.opening_command_range_m[0] <= command.target_opening_m
            assert command.pregrasp_opening_m <= capabilities.opening_command_range_m[1]
            assert capabilities.grip_force_limit_range_n[0] <= command.grip_force_limit_n
            assert command.grip_force_limit_n <= capabilities.grip_force_limit_range_n[1]
            assert command.closing_speed_m_s <= capabilities.maximum_closing_speed_m_s
            assert set(candidate.expected_contacts) <= set(capabilities.contact_groups)
            assert candidate.quality_score == candidate.score_components.template_priority
            assert candidate.score_components.to_dict() == {"template_priority": 1.0}
            assert tuple(candidate.T_object_tcp.matrix[row][3] for row in range(3)) == pytest.approx(
                calibrated_translation
            )
            assert compose(request.T_world_object, candidate.T_object_tcp) == candidate.T_world_tcp
            identity = compose(candidate.T_object_tcp, candidate.T_tcp_object).matrix
            expected_identity = (
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
            assert all(
                identity[row][column] == pytest.approx(expected_identity[row][column], abs=1e-8)
                for row in range(4)
                for column in range(4)
            )


def test_continuous_yaw_is_lower_inclusive_upper_exclusive() -> None:
    result = target(_request("masking_tape"))
    limited = target(
        replace(
            _request("masking_tape"),
            continuous_yaw_resolution_rad=1e-12,
            maximum_candidates=3,
        )
    )

    assert result.success
    yaws = [candidate.symmetry_yaw_rad for candidate in result.candidates]
    assert yaws == pytest.approx([-pi, -pi / 2.0, 0.0, pi / 2.0])
    assert len(yaws) == len(set(yaws))
    assert all(yaw < pi for yaw in yaws)
    assert limited.success
    assert len(limited.candidates) == 3
    assert len({item.candidate_id for item in limited.candidates}) == 3


def test_targeting_failure_codes_preserve_public_failure_semantics() -> None:
    request = _request("marker")

    unknown = target(replace(request, object_type="unknown_part"))
    incompatible = target(
        replace(
            request,
            end_effector_config=replace(request.end_effector_config, family="vacuum"),
        )
    )
    invalid_config = target(replace(request, maximum_candidates=0))
    invalid_pose = target(
        replace(
            request,
            T_world_object=replace(request.T_world_object, child_frame="detected_object"),
        )
    )
    invalid_resolution = target(replace(request, continuous_yaw_resolution_rad=1e-30))

    assert unknown.error_code is TargetingErrorCode.UNKNOWN_OBJECT
    assert incompatible.error_code is TargetingErrorCode.INCOMPATIBLE_END_EFFECTOR
    assert invalid_config.error_code is TargetingErrorCode.INVALID_CONFIGURATION
    assert invalid_pose.error_code is TargetingErrorCode.INVALID_INPUT_POSE
    assert invalid_resolution.error_code is TargetingErrorCode.INVALID_CONFIGURATION
    assert not unknown.candidates
    assert not incompatible.candidates
    assert not invalid_config.candidates
    assert not invalid_pose.candidates
    assert not invalid_resolution.candidates


def test_targeting_module_does_not_cross_runtime_module_boundaries() -> None:
    targeting_path = REPO_ROOT / "src/task2/targeting.py"
    tree = ast.parse(targeting_path.read_text(encoding="utf-8"))
    imported_modules = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    forbidden_roots = (
        "task1",
        "robot_arm_pipeline",
        "task2.planning",
        "task2.execution",
        "mujoco",
        "curobo",
        "ruckig",
        "torch",
        "vamp",
    )

    assert not any(
        module == root or module.startswith(f"{root}.")
        for module in imported_modules
        for root in forbidden_roots
    )
    assert set(TargetingRequest.__dataclass_fields__) == {
        "object_type",
        "T_world_object",
        "object_config",
        "end_effector_config",
        "continuous_yaw_resolution_rad",
        "maximum_candidates",
    }
