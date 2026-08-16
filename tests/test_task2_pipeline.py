from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from task2 import Task2Pipeline
from task2.artifacts import Task2Artifacts
from task2.config import load_task2_config
from task2.contracts import (
    AllowedContact,
    Box,
    DynamicsDiagnostics,
    ExecutionResult,
    ExecutionStatus,
    PlacementGoal,
    PlacementGoalKind,
    PlacementOrientationConstraint,
    PlannedTrajectory,
    PlanningResult,
    RigidTransform,
    RobotState,
    SceneGeometry,
    SceneSnapshot,
    Sphere,
    Task2Request,
    TrajectoryEvent,
    TrajectoryKnot,
    TrajectoryPhase,
    TrajectoryPolynomialSection,
    TrajectorySegment,
    TrajectoryStage,
)
from task2.runtime import ActivatedEndEffector
from task2.targeting import target


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_task2_config(ROOT / "configs/task2/pipeline.yaml")
PROFILE = CONFIG.end_effectors[0]


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


def _request(request_id: str = "pipeline_contract") -> Task2Request:
    object_pose = _pose("world", "object", (0.5, 0.0, 0.1))
    scene = SceneSnapshot(
        "pipeline_scene",
        "world",
        (
            SceneGeometry(
                "target",
                "target_object",
                "target",
                _pose("world", "target", (0.5, 0.0, 0.1)),
                Sphere(0.01),
            ),
            SceneGeometry(
                "support",
                "tank",
                "source_support",
                _pose("world", "support", (0.0, 0.0, -0.1)),
                Box((2.0, 2.0, 0.1)),
            ),
        ),
        (AllowedContact("target", "source_support"),),
    )
    return Task2Request(
        request_id,
        "marker",
        object_pose,
        RobotState(CONFIG.base_robot.joint_names, CONFIG.base_robot.default_positions),
        _pose("world", CONFIG.base_robot.base_frame),
        scene,
        "target_object",
        "source_support",
        PlacementGoal(
            "place",
            PlacementGoalKind.EXACT,
            _pose("world", PROFILE.tcp_frame, (0.6, 0.0, 0.6)),
            PlacementOrientationConstraint.EXACT,
        ),
        7,
    )


def _trajectory(joint_names: tuple[str, ...]) -> PlannedTrajectory:
    channels = {
        "parallel_gripper.opening_m": 0.02,
        "parallel_gripper.force_limit_n": 1.0,
    }
    knots = []
    segments = []
    phases = []
    dof = len(joint_names)
    for index in range(7):
        stage = tuple(TrajectoryStage)[min(index, 5)]
        knots.append(
            TrajectoryKnot(stage, index * 0.1, (0.0,) * dof, (0.0,) * dof, (0.0,) * dof, channels)
        )
    for index, stage in enumerate(TrajectoryStage):
        segments.append(
            TrajectorySegment(
                stage,
                index * 0.1,
                0.1,
                index,
                index + 1,
                (
                    TrajectoryPolynomialSection(
                        0.0,
                        0.1,
                        tuple((0.0, 0.0, 0.0, 0.0) for _ in joint_names),
                    ),
                ),
                channels,
                channels,
            )
        )
        phases.append(
            TrajectoryPhase(stage, index, index + 1, index, index, index * 0.1, (index + 1) * 0.1)
        )
    return PlannedTrajectory(
        joint_names,
        tuple(knots),
        tuple(segments),
        tuple(phases),
        (
            TrajectoryEvent("grip", TrajectoryStage.GRASP, 0.3, channels),
            TrajectoryEvent("release", TrajectoryStage.PLACE, 0.6, channels),
        ),
    )


class _UnusedAdapter:
    adapter_id = "fake_pipeline_adapter"


class _FakeRuntime:
    def activate(self, profile, base_robot_state):
        model = CONFIG.robot_model(profile)
        state = RobotState(model.joint_names, model.default_joint_positions)
        return ActivatedEndEffector(profile, model, state, _UnusedAdapter)


def _fake_plan(request):
    assert request.candidates
    return PlanningResult.succeeded(
        _trajectory(request.robot_model.joint_names),
        _pose("world", "object", (0.6, 0.0, 0.6)),
        (),
        {},
    )


def _fake_execute(request):
    Path(request.control_log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(request.control_log_path).write_text("planned_time_s\n", encoding="utf-8")
    Path(request.diagnostic_log_path).write_text("{}\n", encoding="utf-8")
    return ExecutionResult(
        ExecutionStatus.COMPLETED,
        None,
        "fake execution completed",
        DynamicsDiagnostics.not_evaluated("fake pipeline executor has no dynamics"),
        0.6,
        0.6,
        7,
        1,
        1,
        0.01,
    )


def _pipeline(tmp_path: Path, *, targeting_stage=target) -> Task2Pipeline:
    return Task2Pipeline(
        CONFIG,
        tmp_path,
        targeting_stage,
        _fake_plan,
        _fake_execute,
        _FakeRuntime(),
    )


def test_pipeline_hands_off_only_through_formal_artifacts(tmp_path: Path) -> None:
    result = _pipeline(tmp_path).run(_request())
    assert result.success
    assert result.completed_stages == ("setup", "targeting", "planning", "execution")
    run_dir = tmp_path / "pipeline_contract"
    assert set(result.artifact_paths) == {
        "request",
        "setup",
        "targeting",
        "planning",
        "execution",
        "control_log",
        "dynamics_log",
    }
    assert not list(run_dir.glob("*/request.json"))
    assert not json.loads((run_dir / "result.json").read_text()).get("candidates")
    assert "trajectory" not in json.loads((run_dir / "result.json").read_text())
    planning = json.loads((run_dir / "planning/result.json").read_text())
    assert planning["metadata"] == {}
    assert planning["T_world_object_goal"]["parent_frame"] == "world"
    assert planning["T_world_object_goal"]["child_frame"] == "object"
    execution = json.loads((run_dir / "execution/result.json").read_text())
    assert execution["execution_status"] == "completed"
    assert not execution["dynamics_diagnostics"]["success"]


def test_request_artifact_round_trip_preserves_free_placement_orientation(
    tmp_path: Path,
) -> None:
    baseline = _request("free_placement_artifact")
    request = replace(
        baseline,
        placement_goal=replace(
            baseline.placement_goal,
            orientation_constraint=PlacementOrientationConstraint.FREE,
        ),
    )
    artifacts = Task2Artifacts(tmp_path, request.request_id)

    artifacts.write_request(request)

    persisted = artifacts.read_request()
    assert (
        persisted.placement_goal.orientation_constraint
        is PlacementOrientationConstraint.FREE
    )
    payload = json.loads((artifacts.run_dir / "request.json").read_text())
    assert payload["placement_goal"]["orientation_constraint"] == "free"


def test_request_artifact_rejects_missing_placement_orientation(
    tmp_path: Path,
) -> None:
    request = _request("missing_placement_orientation")
    artifacts = Task2Artifacts(tmp_path, request.request_id)
    artifacts.write_request(request)
    path = artifacts.run_dir / "request.json"
    payload = json.loads(path.read_text())
    del payload["placement_goal"]["orientation_constraint"]
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="orientation_constraint"):
        artifacts.read_request()


def test_cli_steps_resume_from_formal_artifacts(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    request = _request("cli_steps")
    targeting = pipeline._run_cli_step("targeting", request=request)
    assert targeting.completed_stages == ("setup", "targeting")

    run_dir = tmp_path / request.request_id
    planning = pipeline._run_cli_step("planning", run_dir=run_dir)
    assert planning.completed_stages == ("setup", "targeting", "planning")

    execution = pipeline._run_cli_step("execution", run_dir=run_dir)
    assert execution.success
    assert execution.completed_stages == (
        "setup",
        "targeting",
        "planning",
        "execution",
    )


def test_failed_stage_stops_and_removes_stale_downstream_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "pipeline_contract"
    (run_dir / "planning").mkdir(parents=True)
    (run_dir / "planning/result.json").write_text("{}", encoding="utf-8")

    def fail_target(request):
        from task2.contracts import TargetingErrorCode, TargetingResult

        return TargetingResult.failed(TargetingErrorCode.NO_CANDIDATES, "no candidates")

    result = _pipeline(tmp_path, targeting_stage=fail_target).run(_request())
    assert not result.success
    assert result.failed_stage == "targeting"
    assert result.error_code.value == "no_candidates"
    assert (run_dir / "targeting/result.json").is_file()
    assert not (run_dir / "planning/result.json").exists()
    assert not (run_dir / "execution/result.json").exists()
    persisted = Task2Artifacts(tmp_path, "pipeline_contract").read_result()
    assert persisted == result


def test_pipeline_boundary_failure_uses_single_pipeline_error_code(tmp_path: Path) -> None:
    request = _request("invalid_profile")
    result = _pipeline(tmp_path).run(
        Task2Request(
            request.request_id,
            request.object_type,
            request.T_world_object,
            request.base_robot_state,
            request.T_world_base,
            request.scene,
            request.target_object_id,
            request.source_support_group,
            request.placement_goal,
            request.random_seed,
            "vacuum",
        )
    )
    assert not result.success
    assert result.failed_stage == "setup"
    assert result.error_code.value == "pipeline_failed"


def test_setup_failure_removes_previous_run_artifacts(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    request = _request("reused_request")
    assert pipeline.run(request).success

    result = pipeline.run(replace(request, end_effector_profile_id="vacuum"))
    assert not result.success
    assert result.failed_stage == "setup"
    assert result.error_code.value == "pipeline_failed"
    assert result.artifact_paths == {}
    run_dir = tmp_path / request.request_id
    assert {
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.is_file()
    } == {"result.json"}


def test_invalid_request_id_returns_pipeline_failure(tmp_path: Path) -> None:
    result = _pipeline(tmp_path).run(
        replace(_request(), request_id="../outside_output_root")
    )
    assert not result.success
    assert result.failed_stage == "setup"
    assert result.error_code.value == "pipeline_failed"
    assert result.artifact_paths == {}


def test_cli_missing_upstream_artifact_returns_pipeline_failure(tmp_path: Path) -> None:
    run_dir = tmp_path / "missing_upstream"
    result = _pipeline(tmp_path)._run_cli_step("planning", run_dir=run_dir)
    assert not result.success
    assert result.failed_stage == "planning"
    assert result.error_code.value == "pipeline_failed"
    assert result.completed_stages == ()
    assert (run_dir / "result.json").is_file()


def test_cli_upstream_failure_removes_stale_downstream_artifacts(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    cases = (
        (
            "planning",
            "setup.json",
            {
                "planning/result.json",
                "execution/result.json",
                "execution/control.csv",
                "execution/dynamics.jsonl",
            },
            {"request", "setup", "targeting"},
        ),
        (
            "execution",
            "request.json",
            {
                "execution/result.json",
                "execution/control.csv",
                "execution/dynamics.jsonl",
            },
            {"request", "setup", "targeting", "planning"},
        ),
    )
    for step, corrupted, removed, expected_artifacts in cases:
        request = _request(f"corrupt_upstream_{step}")
        assert pipeline.run(request).success
        run_dir = tmp_path / request.request_id
        (run_dir / corrupted).write_text("{", encoding="utf-8")

        result = pipeline._run_cli_step(step, run_dir=run_dir)
        assert not result.success
        assert result.failed_stage == step
        assert result.error_code.value == "pipeline_failed"
        assert set(result.artifact_paths) == expected_artifacts
        assert all(not (run_dir / relative).exists() for relative in removed)
        assert (run_dir / "result.json").is_file()


def test_result_write_failure_overrides_stage_error(
    tmp_path: Path, monkeypatch
) -> None:
    from task2.contracts import TargetingErrorCode, TargetingResult

    def fail_target(request):
        return TargetingResult.failed(TargetingErrorCode.NO_CANDIDATES, "no candidates")

    def fail_write_result(self, result):
        raise OSError("result storage unavailable")

    monkeypatch.setattr(Task2Artifacts, "write_result", fail_write_result)
    result = _pipeline(tmp_path, targeting_stage=fail_target).run(_request())
    assert not result.success
    assert result.failed_stage == "result"
    assert result.error_code.value == "pipeline_failed"
    assert "result storage unavailable" in result.message
    assert not (tmp_path / "pipeline_contract/result.json").exists()
