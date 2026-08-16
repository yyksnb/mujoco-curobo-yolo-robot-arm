from __future__ import annotations

import ast
from dataclasses import replace
from math import cos, pi, radians, sin
from pathlib import Path

import pytest

from task2.config import load_task2_config
from task2.contracts import RigidTransform, TargetingErrorCode, TargetingRequest, compose
from task2.targeting import target


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_task2_config(ROOT / "configs/task2/pipeline.yaml")
PROFILE = CONFIG.end_effectors[0]


def _world_pose(x_m: float = 0.4) -> RigidTransform:
    return RigidTransform(
        "world",
        "object",
        (
            (1.0, 0.0, 0.0, x_m),
            (0.0, 1.0, 0.0, 0.1),
            (0.0, 0.0, 1.0, 0.02),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )


def _request(object_type: str, *, x_m: float = 0.4) -> TargetingRequest:
    return TargetingRequest(
        object_type,
        _world_pose(x_m),
        CONFIG.targeting_object(object_type),
        PROFILE.targeting,
        pi / 2.0,
        16,
    )


def test_calibrated_templates_preserve_candidates_and_commands() -> None:
    expected = {
        "marker": (2, 0.013014206, 0.75, "marker_parallel_body"),
        "masking_tape": (4, 0.074333332, 2.0, "masking_tape_parallel_outer_wall"),
        "rivet_gun": (2, 0.011946974, 24.0, "rivet_gun_parallel_handle"),
        "standard_part": (4, 0.002745694, 0.30, "standard_part_parallel_shaft"),
    }
    for object_type, (count, opening, force, template_id) in expected.items():
        request = _request(object_type)
        result = target(request)
        repeated = target(request)
        moved = target(replace(request, T_world_object=_world_pose(0.7)))
        assert result.success and result == repeated
        assert len(result.candidates) == count
        assert [item.candidate_id for item in result.candidates] == [
            item.candidate_id for item in moved.candidates
        ]
        calibrated = [
            candidate
            for candidate in result.candidates
            if candidate.template_id == template_id
        ]
        assert calibrated
        assert all(
            candidate.candidate_id.startswith(
                f"{object_type}.parallel_gripper.{template_id}.yaw_f64_"
            )
            for candidate in calibrated
        )
        assert all(
            candidate.grasp_command.target_opening_m == pytest.approx(opening)
            and candidate.grasp_command.grip_force_limit_n == pytest.approx(force)
            for candidate in calibrated
        )
        for candidate in result.candidates:
            assert candidate.grasp_command.calibration == "simulation_nominal"
            assert candidate.quality_score == candidate.score_components.template_priority
            assert compose(request.T_world_object, candidate.T_object_tcp) == candidate.T_world_tcp
            assert set(candidate.expected_contacts) <= set(
                request.end_effector_config.capabilities.contact_groups
            )


def test_standard_part_mid_pos10_template_contract() -> None:
    template_id = "standard_part_parallel_shaft_mid_pos10"
    result = target(_request("standard_part"))
    candidates = [
        candidate
        for candidate in result.candidates
        if candidate.template_id == template_id
    ]
    assert [candidate.candidate_id for candidate in candidates] == [
        "standard_part.parallel_gripper.standard_part_parallel_shaft_mid_pos10."
        "yaw_f64_0000000000000000",
        "standard_part.parallel_gripper.standard_part_parallel_shaft_mid_pos10."
        "yaw_f64_400921fb54524550",
    ]

    yaw_zero = candidates[0]
    angle = radians(10.0)
    expected_T_object_tcp = (
        (0.0, 1.0, 0.0, 0.0),
        (cos(angle), 0.0, sin(angle), -0.002826351822),
        (sin(angle), 0.0, -cos(angle), 0.002265192247),
        (0.0, 0.0, 0.0, 1.0),
    )
    matrix = yaw_zero.T_object_tcp.matrix
    assert tuple(value for row in matrix for value in row) == pytest.approx(
        tuple(value for row in expected_T_object_tcp for value in row)
    )
    closing_axis_object = tuple(matrix[row][1] for row in range(3))
    standard_part_axis_object = (0.0, 1.0, 0.0)
    assert sum(
        left * right
        for left, right in zip(closing_axis_object, standard_part_axis_object)
    ) == pytest.approx(0.0, abs=1e-9)

    assert yaw_zero.grasp_command.target_opening_m == pytest.approx(0.002501621)
    assert yaw_zero.grasp_command.grip_force_limit_n == pytest.approx(0.30)


def test_continuous_yaw_is_lower_inclusive_upper_exclusive() -> None:
    result = target(_request("masking_tape"))
    assert result.success
    assert [item.symmetry_yaw_rad for item in result.candidates] == pytest.approx(
        [-pi, -pi / 2.0, 0.0, pi / 2.0]
    )
    limited = target(
        replace(
            _request("masking_tape"),
            continuous_yaw_resolution_rad=1e-12,
            maximum_candidates=3,
        )
    )
    assert limited.success and len(limited.candidates) == 3


def test_targeting_failure_codes_are_stable() -> None:
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
    assert unknown.error_code is TargetingErrorCode.UNKNOWN_OBJECT
    assert incompatible.error_code is TargetingErrorCode.INCOMPATIBLE_END_EFFECTOR
    assert invalid_config.error_code is TargetingErrorCode.INVALID_CONFIGURATION
    assert invalid_pose.error_code is TargetingErrorCode.INVALID_INPUT_POSE
    assert not unknown.candidates and not incompatible.candidates


def test_targeting_module_stays_within_its_boundary() -> None:
    tree = ast.parse((ROOT / "src/task2/targeting.py").read_text(encoding="utf-8"))
    modules = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    forbidden = ("task2.planning", "task2.execution", "mujoco", "curobo", "ruckig", "vamp")
    assert not any(
        module == root or module.startswith(f"{root}.")
        for module in modules
        for root in forbidden
    )
    assert set(TargetingRequest.__dataclass_fields__) == {
        "object_type",
        "T_world_object",
        "object_config",
        "end_effector_config",
        "continuous_yaw_resolution_rad",
        "maximum_candidates",
    }
