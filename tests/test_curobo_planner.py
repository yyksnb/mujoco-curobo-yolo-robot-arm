from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from robot_arm_pipeline.perception import (
    fake_yolo_object_poses,
    load_bodex_grasp_target,
)
from robot_arm_pipeline.pipeline import default_robot_state
from robot_arm_pipeline.planning import (
    MotionPlanResult,
    MotionPlanSegment,
    PoseTarget,
    offset_pose_target_along_local_z,
)
from robot_arm_pipeline.planning.curobo_conversions import (
    collision_scene_to_curobo_world_config,
    joint_trajectory_to_planned_trajectory,
    transform_to_curobo_pose,
)
from robot_arm_pipeline.planning.curobo_planner import (
    MOTION_PLAN_RESULT_SCHEMA,
    CuroboPlanner,
    filter_feasible_graph_goals,
)
from robot_arm_pipeline.scene import build_collision_scene
from robot_arm_pipeline.types import ObjectPose, PlanningRequest


REPO_ROOT = Path(__file__).resolve().parents[1]
BODEX_EXAMPLE = REPO_ROOT / "examples" / "bodex_grasp_target.json"
JOINT_NAMES = ("joint_1", "joint_2")


def test_curobo_planner_reports_missing_configuration_as_planning_failure(
    tmp_path: Path,
) -> None:
    grasp_target = load_bodex_grasp_target(BODEX_EXAMPLE)
    object_pose = ObjectPose(
        object_id=grasp_target.object_id,
        label="test_target",
        pose=grasp_target.pose,
        T_world_object=grasp_target.T_world_pregrasp,
    )
    request = PlanningRequest(
        object_pose=object_pose,
        grasp_target=grasp_target,
        robot_state=default_robot_state(),
    )
    planner = CuroboPlanner(
        repo_root=REPO_ROOT,
        robot_config_path=tmp_path / "missing_robot.yml",
    )
    assert planner.name == planner.planner_name == "curobo_planner"
    with pytest.raises(ValueError, match="at least one target"):
        planner.plan_pose_route((), request.robot_state)

    result = planner.plan(request)

    assert result.success is False
    assert result.trajectory is None
    assert "robot configuration does not exist" in result.message


def test_run_curobo_planner_script_generates_failure_report(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_curobo_planner.py"),
            "--bodex-grasp",
            str(BODEX_EXAMPLE),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(
        (tmp_path / "reports" / "curobo_planner_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert "success=False" in result.stdout
    assert report["success"] is False
    assert report["trajectory_available"] is False
    assert report["message"]


def test_pose_target_and_motion_plan_contracts_preserve_legacy_routes() -> None:
    target = PoseTarget(
        "survey_0000",
        (0.1, 0.2, 0.3),
        (1.0, 0.0, 0.0, 0.0),
    )
    assert offset_pose_target_along_local_z(target, 0.1).target_position == (
        0.1,
        0.2,
        0.4,
    )
    with pytest.raises(ValueError, match="normalized"):
        PoseTarget("invalid", (0.1, 0.2, 0.3), (2.0, 0.0, 0.0, 0.0))

    segment = MotionPlanSegment(
        target_id=target.target_id,
        success=True,
        message="planned",
        planning_time_s=0.25,
        waypoint_count=2,
        trajectory=((0.0, 0.1), (0.2, 0.3)),
        trajectory_time_s=(0.0, 0.1),
        trajectory_velocity=((0.0, 0.0), (2.0, 2.0)),
        target_position_error_m=0.001,
        target_orientation_error_rad=0.002,
        planning_strategy="cartesian_continuation",
        continuation_offset_m=0.1,
    )
    plan = MotionPlanResult(
        success=True,
        planner_name="test_planner",
        joint_names=JOINT_NAMES,
        segments=(segment,),
        failed_target_id=None,
        message="planned route",
        reached_target_ids=(target.target_id,),
    )
    payload = plan.to_dict()
    assert payload["schema"] == MOTION_PLAN_RESULT_SCHEMA
    assert MotionPlanResult.from_dict(payload) == plan

    legacy = plan.to_dict()
    legacy["schema"] = "camera_route_plan"
    legacy_segment = legacy["segments"][0]
    legacy_segment["planning_strategy"] = "portal_continuation"
    legacy_segment["portal_offset_m"] = legacy_segment.pop(
        "continuation_offset_m"
    )
    assert MotionPlanResult.from_dict(legacy) == plan

    malformed = plan.to_dict()
    malformed["segments"][0]["waypoint_count"] = 3
    with pytest.raises(ValueError, match="waypoint_count"):
        MotionPlanResult.from_dict(malformed)


class _GraphPlannerStub:
    def check_samples_feasibility(self, states: np.ndarray) -> np.ndarray:
        return states[:, 0] >= 0.0


def test_graph_query_filters_infeasible_goals_and_fails_when_none_remain() -> None:
    start = np.asarray([[0.0, 0.0]])
    queries = filter_feasible_graph_goals(
        _GraphPlannerStub(),
        start,
        np.asarray([[1.0, 0.1], [-1.0, 0.2], [2.0, 0.3]]),
    )
    assert queries is not None
    starts, goals = queries
    np.testing.assert_array_equal(starts, [[0.0, 0.0], [0.0, 0.0]])
    np.testing.assert_array_equal(goals, [[1.0, 0.1], [2.0, 0.3]])
    assert filter_feasible_graph_goals(
        _GraphPlannerStub(), start, np.asarray([[-1.0, 0.1]])
    ) is None


def test_curobo_conversion_helpers_validate_schema() -> None:
    transform = (
        (1.0, 0.0, 0.0, 0.45),
        (0.0, 1.0, 0.0, 0.05),
        (0.0, 0.0, 1.0, 0.20),
        (0.0, 0.0, 0.0, 1.0),
    )
    assert transform_to_curobo_pose(transform) == [
        0.45,
        0.05,
        0.2,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    object_pose = ObjectPose(
        object_id="object_001",
        label="cube",
        pose=fake_yolo_object_poses()[0].pose,
    )
    world = collision_scene_to_curobo_world_config(
        build_collision_scene((object_pose,))
    )
    assert "object_001" in world["cuboid"]

    trajectory = joint_trajectory_to_planned_trajectory(
        joint_names=JOINT_NAMES,
        joint_positions=((0.0, 0.1), (0.2, 0.3)),
        target_object_id="object_001",
        interpolation_dt=0.1,
    )
    assert trajectory.planner_name == "curobo_planner"
    assert trajectory.waypoints[-1].time_s == 0.1
