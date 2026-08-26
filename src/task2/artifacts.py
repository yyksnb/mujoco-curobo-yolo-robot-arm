from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .contracts.execution import (
    BodyKinematicState,
    ContactRecord,
    DynamicsDiagnostics,
    ExecutionErrorCode,
    ExecutionResult,
    ExecutionStatus,
)
from .contracts.geometry import RigidTransform
from .contracts.pipeline import (
    PipelineErrorCode,
    PipelineStatus,
    Task2PipelineResult,
    Task2Request,
)
from .contracts.planning import (
    AllowedContact,
    Box,
    Capsule,
    ConvexHull,
    PlacementGoal,
    PlacementGoalKind,
    PlacementOrientationConstraint,
    PlannedTrajectory,
    PlanningAttempt,
    PlanningErrorCode,
    PlanningResult,
    PointCloud,
    RobotState,
    SceneGeometry,
    SceneSnapshot,
    Sphere,
    StaticTriangleMesh,
    TrajectoryEvent,
    TrajectoryKnot,
    TrajectoryPhase,
    TrajectoryPolynomialSection,
    TrajectorySegment,
    TrajectoryStage,
    VerticalTransitCorridor,
)
from .contracts.targeting import (
    GraspCandidate,
    ParallelGripperCommand,
    ScoreComponents,
    TargetingDiagnosticsConfig,
    TargetingErrorCode,
    TargetingResult,
)


@dataclass(frozen=True)
class SetupArtifact:
    end_effector_profile_id: str
    end_effector_family: str
    robot_model_id: str
    initial_state: RobotState


class Task2Artifacts:
    REQUEST = "request.json"
    SETUP = "setup.json"
    TARGETING = "targeting/result.json"
    PLANNING = "planning/result.json"
    EXECUTION = "execution/result.json"
    CONTROL_LOG = "execution/control.csv"
    DYNAMICS_LOG = "execution/dynamics.jsonl"
    RESULT = "result.json"

    def __init__(self, output_root: str | Path, request_id: str):
        if not request_id or Path(request_id).name != request_id or request_id in {".", ".."}:
            raise ValueError("request_id must be one path-safe name")
        self.output_root = Path(output_root).resolve()
        self.run_dir = self.output_root / request_id

    def reset(self) -> None:
        for relative in (
            self.REQUEST,
            self.SETUP,
            self.TARGETING,
            self.PLANNING,
            self.EXECUTION,
            self.CONTROL_LOG,
            self.DYNAMICS_LOG,
            self.RESULT,
        ):
            (self.run_dir / relative).unlink(missing_ok=True)

    def clear_from(self, stage: str) -> None:
        downstream = {
            "targeting": (
                self.TARGETING,
                self.PLANNING,
                self.EXECUTION,
                self.CONTROL_LOG,
                self.DYNAMICS_LOG,
                self.RESULT,
            ),
            "planning": (
                self.PLANNING,
                self.EXECUTION,
                self.CONTROL_LOG,
                self.DYNAMICS_LOG,
                self.RESULT,
            ),
            "execution": (
                self.EXECUTION,
                self.CONTROL_LOG,
                self.DYNAMICS_LOG,
                self.RESULT,
            ),
        }
        if stage not in downstream:
            raise ValueError(f"unknown Task2 stage {stage!r}")
        for relative in downstream[stage]:
            (self.run_dir / relative).unlink(missing_ok=True)

    def write_request(self, request: Task2Request) -> None:
        self._write(self.REQUEST, _request_to_dict(request))

    def read_request(self) -> Task2Request:
        return _request_from_dict(self._read(self.REQUEST))

    def write_setup(self, setup: SetupArtifact) -> None:
        self._write(
            self.SETUP,
            {
                "end_effector_profile_id": setup.end_effector_profile_id,
                "end_effector_family": setup.end_effector_family,
                "robot_model_id": setup.robot_model_id,
                "initial_state": _robot_state_to_dict(setup.initial_state),
            },
        )

    def read_setup(self) -> SetupArtifact:
        value = self._read(self.SETUP)
        return SetupArtifact(
            _string(value, "end_effector_profile_id"),
            _string(value, "end_effector_family"),
            _string(value, "robot_model_id"),
            _robot_state_from_dict(_mapping(value, "initial_state")),
        )

    def write_targeting(self, result: TargetingResult) -> None:
        self._write(self.TARGETING, result.to_dict())

    def read_targeting(self) -> TargetingResult:
        return _targeting_from_dict(self._read(self.TARGETING))

    def write_planning(self, result: PlanningResult) -> None:
        self._write(self.PLANNING, result.to_dict())

    def read_planning(self) -> PlanningResult:
        return _planning_from_dict(self._read(self.PLANNING))

    def write_execution(self, result: ExecutionResult) -> None:
        payload = result.to_dict()
        payload["control_log_path"] = self.CONTROL_LOG
        payload["dynamics_log_path"] = self.DYNAMICS_LOG
        self._write(self.EXECUTION, payload)

    def read_execution(self) -> ExecutionResult:
        return _execution_from_dict(self._read(self.EXECUTION))

    def write_result(self, result: Task2PipelineResult) -> None:
        self._write(self.RESULT, result.to_dict())

    def read_result(self) -> Task2PipelineResult:
        value = self._read(self.RESULT)
        raw_code = value.get("error_code")
        failed_stage = _optional_string(value.get("failed_stage"))
        code = _pipeline_result_error_code(raw_code, failed_stage)
        return Task2PipelineResult(
            PipelineStatus(_string(value, "status")),
            tuple(_strings(value, "completed_stages")),
            failed_stage,
            _optional_string(value.get("selected_end_effector_profile_id")),
            {
                key: float(item)
                for key, item in _mapping(value, "stage_elapsed_s").items()
            },
            {
                key: str(item)
                for key, item in _mapping(value, "artifact_paths").items()
            },
            code,
            _string(value, "message"),
        )

    @property
    def control_log_path(self) -> Path:
        return self.run_dir / self.CONTROL_LOG

    @property
    def dynamics_log_path(self) -> Path:
        return self.run_dir / self.DYNAMICS_LOG

    def _write(self, relative: str, payload: Mapping[str, object]) -> None:
        path = self.run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _read(self, relative: str) -> dict[str, Any]:
        path = self.run_dir / relative
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read Task2 artifact {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Task2 artifact {path} must contain a JSON object")
        return value


def _pipeline_result_error_code(raw_code: object, failed_stage: str | None):
    if raw_code is None:
        return None
    error_types = {
        "targeting": TargetingErrorCode,
        "planning": PlanningErrorCode,
        "execution": ExecutionErrorCode,
    }
    error_type = error_types.get(failed_stage, PipelineErrorCode)
    try:
        return error_type(raw_code)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"unknown Task2 {failed_stage or 'pipeline'} error code {raw_code!r}"
        ) from exc


def load_request_file(path: str | Path) -> Task2Request:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read Task2 request {source}: {exc}") from exc
    return _request_from_dict(_as_mapping(value, str(source)))


def _request_to_dict(request: Task2Request) -> dict[str, object]:
    return {
        "request_id": request.request_id,
        "object_type": request.object_type,
        "T_world_object": request.T_world_object.to_dict(),
        "base_robot_state": _robot_state_to_dict(request.base_robot_state),
        "T_world_base": request.T_world_base.to_dict(),
        "scene": _scene_to_dict(request.scene),
        "target_object_id": request.target_object_id,
        "source_support_group": request.source_support_group,
        "placement_goal": _placement_to_dict(request.placement_goal),
        "random_seed": request.random_seed,
        "end_effector_profile_id": request.end_effector_profile_id,
        "vertical_transit_corridor": (
            _vertical_transit_corridor_to_dict(request.vertical_transit_corridor)
            if request.vertical_transit_corridor is not None
            else None
        ),
    }


def _request_from_dict(value: dict[str, Any]) -> Task2Request:
    if "vertical_transit_corridor" not in value:
        raise ValueError("artifact field 'vertical_transit_corridor' is required")
    corridor = value["vertical_transit_corridor"]
    result = Task2Request(
        _string(value, "request_id"),
        _string(value, "object_type"),
        _transform_from_dict(_mapping(value, "T_world_object")),
        _robot_state_from_dict(_mapping(value, "base_robot_state")),
        _transform_from_dict(_mapping(value, "T_world_base")),
        _scene_from_dict(_mapping(value, "scene")),
        _string(value, "target_object_id"),
        _string(value, "source_support_group"),
        _placement_from_dict(_mapping(value, "placement_goal")),
        _integer(value, "random_seed"),
        _optional_string(value.get("end_effector_profile_id")),
        (
            _vertical_transit_corridor_from_dict(
                _as_mapping(corridor, "vertical_transit_corridor")
            )
            if corridor is not None
            else None
        ),
    )
    result.validate()
    return result


def _scene_to_dict(scene: SceneSnapshot) -> dict[str, object]:
    return {
        "snapshot_id": scene.snapshot_id,
        "world_frame": scene.world_frame,
        "geometries": [
            {
                "geometry_id": item.geometry_id,
                "object_id": item.object_id,
                "collision_group": item.collision_group,
                "T_world_geometry": item.T_world_geometry.to_dict(),
                "geometry": _geometry_to_dict(item.geometry),
            }
            for item in scene.geometries
        ],
        "allowed_contacts": [
            {"first_group": item.first_group, "second_group": item.second_group}
            for item in scene.allowed_contacts
        ],
    }


def _scene_from_dict(value: dict[str, Any]) -> SceneSnapshot:
    geometries = []
    for raw in _list(value, "geometries"):
        item = _as_mapping(raw, "scene geometry")
        geometries.append(
            SceneGeometry(
                _string(item, "geometry_id"),
                _string(item, "object_id"),
                _string(item, "collision_group"),
                _transform_from_dict(_mapping(item, "T_world_geometry")),
                _geometry_from_dict(_mapping(item, "geometry")),
            )
        )
    contacts = tuple(
        AllowedContact(
            _string(item, "first_group"),
            _string(item, "second_group"),
        )
        for item in (
            _as_mapping(raw, "allowed contact")
            for raw in _list(value, "allowed_contacts")
        )
    )
    result = SceneSnapshot(
        _string(value, "snapshot_id"),
        _string(value, "world_frame"),
        tuple(geometries),
        contacts,
    )
    result.validate()
    return result


def _geometry_to_dict(value: object) -> dict[str, object]:
    if isinstance(value, Sphere):
        return {"kind": "sphere", "radius_m": value.radius_m}
    if isinstance(value, Capsule):
        return {"kind": "capsule", "radius_m": value.radius_m, "length_m": value.length_m}
    if isinstance(value, Box):
        return {"kind": "box", "size_m": list(value.size_m)}
    if isinstance(value, PointCloud):
        return {
            "kind": "point_cloud",
            "points_m": [list(item) for item in value.points_m],
            "point_radius_m": value.point_radius_m,
        }
    if isinstance(value, ConvexHull):
        return {
            "kind": "convex_hull",
            "vertices_m": [list(item) for item in value.vertices_m],
            "triangles": [list(item) for item in value.triangles],
        }
    if isinstance(value, StaticTriangleMesh):
        return {
            "kind": "static_triangle_mesh",
            "vertices_m": [list(item) for item in value.vertices_m],
            "triangles": [list(item) for item in value.triangles],
        }
    raise TypeError(f"unsupported scene geometry {type(value).__name__}")


def _geometry_from_dict(value: dict[str, Any]):
    kind = _string(value, "kind")
    if kind == "sphere":
        return Sphere(_number(value, "radius_m"))
    if kind == "capsule":
        return Capsule(_number(value, "radius_m"), _number(value, "length_m"))
    if kind == "box":
        return Box(_vector(value, "size_m"))
    if kind == "point_cloud":
        return PointCloud(
            tuple(_as_vector(item, "point") for item in _list(value, "points_m")),
            _number(value, "point_radius_m"),
        )
    if kind in {"convex_hull", "static_triangle_mesh"}:
        vertices = tuple(_as_vector(item, "vertex") for item in _list(value, "vertices_m"))
        triangles = tuple(
            tuple(int(index) for index in _as_list(item, "triangle"))
            for item in _list(value, "triangles")
        )
        return ConvexHull(vertices, triangles) if kind == "convex_hull" else StaticTriangleMesh(vertices, triangles)
    raise ValueError(f"unsupported scene geometry kind {kind!r}")


def _placement_to_dict(value: PlacementGoal) -> dict[str, object]:
    return {
        "goal_id": value.goal_id,
        "kind": value.kind.value,
        "orientation_constraint": value.orientation_constraint.value,
        "T_world_tcp": value.T_world_tcp.to_dict(),
        "size_m": list(value.size_m) if value.size_m is not None else None,
        "radius_m": value.radius_m,
        "height_range_m": list(value.height_range_m),
    }


def _vertical_transit_corridor_to_dict(
    value: VerticalTransitCorridor,
) -> dict[str, object]:
    return {
        "corridor_id": value.corridor_id,
        "portal_world_xy_m": list(value.portal_world_xy_m),
        "ingress_bottom_offset_m": value.ingress_bottom_offset_m,
        "ingress_top_offset_m": value.ingress_top_offset_m,
        "minimum_vertical_lift_m": value.minimum_vertical_lift_m,
        "egress_transition_offset_m": value.egress_transition_offset_m,
        "egress_top_offset_m": value.egress_top_offset_m,
        "ingress_waypoint_step_m": value.ingress_waypoint_step_m,
        "egress_waypoint_step_m": value.egress_waypoint_step_m,
    }


def _vertical_transit_corridor_from_dict(
    value: dict[str, Any],
) -> VerticalTransitCorridor:
    portal = _list(value, "portal_world_xy_m")
    return VerticalTransitCorridor(
        _string(value, "corridor_id"),
        (float(portal[0]), float(portal[1])),
        _number(value, "ingress_bottom_offset_m"),
        _number(value, "ingress_top_offset_m"),
        _number(value, "minimum_vertical_lift_m"),
        _number(value, "egress_transition_offset_m"),
        _number(value, "egress_top_offset_m"),
        _number(value, "ingress_waypoint_step_m"),
        _number(value, "egress_waypoint_step_m"),
    )


def _placement_from_dict(value: dict[str, Any]) -> PlacementGoal:
    size = value.get("size_m")
    radius = value.get("radius_m")
    height = _list(value, "height_range_m")
    return PlacementGoal(
        _string(value, "goal_id"),
        PlacementGoalKind(_string(value, "kind")),
        _transform_from_dict(_mapping(value, "T_world_tcp")),
        PlacementOrientationConstraint(_string(value, "orientation_constraint")),
        tuple(float(item) for item in _as_list(size, "size_m")) if size is not None else None,
        float(radius) if radius is not None else None,
        (float(height[0]), float(height[1])),
    )


def _targeting_from_dict(value: dict[str, Any]) -> TargetingResult:
    code = value.get("error_code")
    candidates = tuple(
        _candidate_from_dict(_as_mapping(item, "candidate"))
        for item in _list(value, "candidates")
    )
    return TargetingResult(
        candidates,
        TargetingErrorCode(code) if code is not None else None,
        _string(value, "message"),
    )


def _candidate_from_dict(value: dict[str, Any]) -> GraspCandidate:
    diagnostics = _mapping(value, "diagnostics")
    command = _mapping(value, "grasp_command")
    score = _mapping(value, "score_components")
    return GraspCandidate(
        _string(value, "candidate_id"),
        _string(value, "object_type"),
        _string(value, "end_effector_profile_id"),
        _string(value, "template_id"),
        _number(value, "symmetry_yaw_rad"),
        _transform_from_dict(_mapping(value, "T_object_tcp")),
        _transform_from_dict(_mapping(value, "T_world_tcp")),
        _transform_from_dict(_mapping(value, "T_world_pregrasp")),
        _transform_from_dict(_mapping(value, "T_tcp_object")),
        _number(value, "quality_score"),
        ScoreComponents(_number(score, "template_priority")),
        TargetingDiagnosticsConfig(
            _number(diagnostics, "contact_margin_m"),
            _number(diagnostics, "opening_margin_m"),
            _number(diagnostics, "force_margin_n"),
        ),
        ParallelGripperCommand(
            _number(command, "pregrasp_opening_m"),
            _number(command, "target_opening_m"),
            _number(command, "grip_force_limit_n"),
            _number(command, "closing_speed_m_s"),
            _number(command, "dwell_s"),
            _string(command, "calibration"),
        ),
        tuple(_strings(value, "expected_contacts")),
    )


def _planning_from_dict(value: dict[str, Any]) -> PlanningResult:
    raw_code = value.get("error_code")
    trajectory = value.get("trajectory")
    if "T_world_object_goal" not in value:
        raise ValueError("artifact field 'T_world_object_goal' is required")
    raw_goal = value["T_world_object_goal"]
    attempts = tuple(
        PlanningAttempt(
            _string(item, "candidate_id"),
            _integer(item, "seed"),
            TrajectoryStage(_string(item, "stage")),
            _number(item, "elapsed_s"),
            PlanningErrorCode(code) if (code := item.get("error_code")) is not None else None,
            _string(item, "reason"),
        )
        for item in (
            _as_mapping(raw, "planning attempt") for raw in _list(value, "attempts")
        )
    )
    result = PlanningResult(
        _trajectory_from_dict(_as_mapping(trajectory, "trajectory")) if trajectory is not None else None,
        _transform_from_dict(_as_mapping(raw_goal, "T_world_object_goal"))
        if raw_goal is not None
        else None,
        PlanningErrorCode(raw_code) if raw_code is not None else None,
        _string(value, "message"),
        attempts,
        _mapping(value, "metadata"),
    )
    result.validate()
    return result


def _trajectory_from_dict(value: dict[str, Any]) -> PlannedTrajectory:
    knots = tuple(
        TrajectoryKnot(
            TrajectoryStage(_string(item, "stage")),
            _number(item, "time_s"),
            _numbers(item, "positions_rad"),
            _numbers(item, "velocities_rad_s"),
            _numbers(item, "accelerations_rad_s2"),
            _number_mapping(item, "actuator_channels"),
        )
        for item in (_as_mapping(raw, "trajectory knot") for raw in _list(value, "knots"))
    )
    segments = []
    for raw in _list(value, "segments"):
        item = _as_mapping(raw, "trajectory segment")
        sections = tuple(
            TrajectoryPolynomialSection(
                _number(section, "start_time_s"),
                _number(section, "duration_s"),
                tuple(
                    tuple(float(coefficient) for coefficient in _as_list(row, "coefficients"))
                    for row in _list(section, "position_coefficients")
                ),
            )
            for section in (
                _as_mapping(raw_section, "trajectory section")
                for raw_section in _list(item, "sections")
            )
        )
        segments.append(
            TrajectorySegment(
                TrajectoryStage(_string(item, "stage")),
                _number(item, "start_time_s"),
                _number(item, "duration_s"),
                _integer(item, "start_knot"),
                _integer(item, "end_knot"),
                sections,
                _number_mapping(item, "start_actuator_channels"),
                _number_mapping(item, "end_actuator_channels"),
            )
        )
    phases = tuple(
        TrajectoryPhase(
            TrajectoryStage(_string(item, "stage")),
            _integer(item, "first_knot"),
            _integer(item, "last_knot"),
            _integer(item, "first_segment"),
            _integer(item, "last_segment"),
            _number(item, "start_time_s"),
            _number(item, "end_time_s"),
        )
        for item in (_as_mapping(raw, "trajectory phase") for raw in _list(value, "phases"))
    )
    events = tuple(
        TrajectoryEvent(
            _string(item, "event"),
            TrajectoryStage(_string(item, "stage")),
            _number(item, "time_s"),
            _number_mapping(item, "actuator_channels"),
        )
        for item in (_as_mapping(raw, "trajectory event") for raw in _list(value, "events"))
    )
    return PlannedTrajectory(
        tuple(_strings(value, "joint_names")),
        knots,
        tuple(segments),
        phases,
        events,
    )


def _execution_from_dict(value: dict[str, Any]) -> ExecutionResult:
    raw_code = value.get("error_code")
    diagnostics = _mapping(value, "dynamics_diagnostics")
    return ExecutionResult(
        ExecutionStatus(_string(value, "execution_status")),
        ExecutionErrorCode(raw_code) if raw_code is not None else None,
        _string(value, "message"),
        DynamicsDiagnostics(
            _boolean(diagnostics, "evaluated"),
            _boolean(diagnostics, "success"),
            _boolean(diagnostics, "target_object_present"),
            _boolean(diagnostics, "departed_source_support"),
            _optional_body_state(diagnostics.get("pre_release_state")),
            _optional_body_state(diagnostics.get("post_release_state")),
            _boolean(diagnostics, "stable_after_release"),
            _optional_number(diagnostics.get("target_position_error_m")),
            tuple(
                _contact_from_dict(_as_mapping(item, "contact"))
                for item in _list(diagnostics, "contact_pairs")
            ),
            _number(diagnostics, "maximum_penetration_m"),
            _number(diagnostics, "maximum_normal_force_n"),
            _number(diagnostics, "maximum_impact_impulse_ns"),
            tuple(_strings(diagnostics, "failure_reasons")),
        ),
        _number(value, "planned_duration_s"),
        _number(value, "executed_planned_duration_s"),
        _integer(value, "control_sample_count"),
        _integer(value, "control_log_sample_count"),
        _integer(value, "diagnostic_log_sample_count"),
        _number(value, "elapsed_s"),
    )


def _optional_body_state(value: object) -> BodyKinematicState | None:
    if value is None:
        return None
    item = _as_mapping(value, "body state")
    return BodyKinematicState(
        _transform_from_dict(_mapping(item, "T_world_body")),
        _vector(item, "linear_velocity_m_s"),
        _vector(item, "angular_velocity_rad_s"),
    )


def _contact_from_dict(value: dict[str, Any]) -> ContactRecord:
    return ContactRecord(
        _string(value, "first_geom"),
        _string(value, "second_geom"),
        _number(value, "maximum_penetration_m"),
        _number(value, "maximum_normal_force_n"),
        _number(value, "maximum_impact_impulse_ns"),
    )


def _transform_from_dict(value: dict[str, Any]) -> RigidTransform:
    rows = _list(value, "matrix")
    matrix = tuple(
        tuple(float(item) for item in _as_list(row, "transform row")) for row in rows
    )
    result = RigidTransform(
        _string(value, "parent_frame"),
        _string(value, "child_frame"),
        matrix,  # type: ignore[arg-type]
    )
    result.validate()
    return result


def _robot_state_to_dict(value: RobotState) -> dict[str, object]:
    return {"joint_names": list(value.joint_names), "positions_rad": list(value.positions_rad)}


def _robot_state_from_dict(value: dict[str, Any]) -> RobotState:
    return RobotState(tuple(_strings(value, "joint_names")), _numbers(value, "positions_rad"))


def _mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in value:
        raise ValueError(f"artifact field {key!r} is required")
    return _as_mapping(value[key], key)


def _as_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed object")
    return value


def _list(value: dict[str, Any], key: str) -> list[object]:
    if key not in value:
        raise ValueError(f"artifact field {key!r} is required")
    return _as_list(value[key], key)


def _as_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _string(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"artifact field {key!r} must be a non-empty string")
    return item


def _strings(value: dict[str, Any], key: str) -> tuple[str, ...]:
    result = tuple(item for item in _list(value, key) if isinstance(item, str))
    if len(result) != len(_list(value, key)):
        raise ValueError(f"artifact field {key!r} must contain strings")
    return result


def _integer(value: dict[str, Any], key: str) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int):
        raise ValueError(f"artifact field {key!r} must be an integer")
    return item


def _number(value: dict[str, Any], key: str) -> float:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        raise ValueError(f"artifact field {key!r} must be a number")
    return float(item)


def _numbers(value: dict[str, Any], key: str) -> tuple[float, ...]:
    return tuple(float(item) for item in _list(value, key))


def _number_mapping(value: dict[str, Any], key: str) -> dict[str, float]:
    return {name: float(item) for name, item in _mapping(value, key).items()}


def _boolean(value: dict[str, Any], key: str) -> bool:
    item = value.get(key)
    if not isinstance(item, bool):
        raise ValueError(f"artifact field {key!r} must be a boolean")
    return item


def _vector(value: dict[str, Any], key: str) -> tuple[float, float, float]:
    return _as_vector(_list(value, key), key)


def _as_vector(value: object, label: str) -> tuple[float, float, float]:
    items = _as_list(value, label)
    if len(items) != 3:
        raise ValueError(f"{label} must have three values")
    return tuple(float(item) for item in items)  # type: ignore[return-value]


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("optional artifact string must be non-empty")
    return value


def _optional_number(value: object) -> float | None:
    return None if value is None else float(value)
