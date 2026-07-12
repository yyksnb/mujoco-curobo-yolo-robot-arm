from __future__ import annotations

import numpy as np
import pytest

from robot_arm_pipeline.planning import CameraRoutePlan, CameraRouteSegment, CameraRouteTarget
from robot_arm_pipeline.planning.curobo_camera_route import (
    CAMERA_ROUTE_PLAN_SCHEMA,
    filter_feasible_graph_goals,
)


JOINT_NAMES = ("joint_1", "joint_2")


def _plan() -> CameraRoutePlan:
    segment = CameraRouteSegment(
        target_id="survey_0000",
        success=True,
        message="planned",
        planning_time_s=0.25,
        waypoint_count=2,
        trajectory=((0.0, 0.1), (0.2, 0.3)),
        trajectory_time_s=(0.0, 0.1),
        trajectory_velocity=((0.0, 0.0), (2.0, 2.0)),
        target_position_error_m=0.001,
        target_orientation_error_rad=0.002,
    )
    return CameraRoutePlan(
        success=True,
        planner_name="test_planner",
        joint_names=JOINT_NAMES,
        segments=(segment,),
        failed_target_id=None,
        message="planned route",
        reached_target_ids=("survey_0000",),
    )


def test_camera_route_target_uses_explicit_normalized_pose() -> None:
    target = CameraRouteTarget("survey_0000", (0.1, 0.2, 0.3), (1.0, 0.0, 0.0, 0.0))

    assert target.target_position == (0.1, 0.2, 0.3)
    assert target.target_quaternion_wxyz == (1.0, 0.0, 0.0, 0.0)

    with pytest.raises(ValueError, match="normalized"):
        CameraRouteTarget("survey_0000", (0.1, 0.2, 0.3), (2.0, 0.0, 0.0, 0.0))


def test_camera_route_plan_round_trip_and_rejects_malformed_artifacts() -> None:
    plan = _plan()

    payload = plan.to_dict()

    assert payload["schema"] == CAMERA_ROUTE_PLAN_SCHEMA
    assert payload["segments"][0]["trajectory_time_s"] == [0.0, 0.1]
    assert payload["segments"][0]["trajectory_velocity"] == [[0.0, 0.0], [2.0, 2.0]]
    assert CameraRoutePlan.from_dict(payload) == plan

    invalid_schema = _plan().to_dict()
    invalid_schema["schema"] = "unsupported"
    with pytest.raises(ValueError, match="unsupported"):
        CameraRoutePlan.from_dict(invalid_schema)

    invalid_segment = _plan().segments[0].to_dict()
    invalid_segment["waypoint_count"] = 3
    with pytest.raises(ValueError, match="waypoint_count"):
        CameraRouteSegment.from_dict(invalid_segment)


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
