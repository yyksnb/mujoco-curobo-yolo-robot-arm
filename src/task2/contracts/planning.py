from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Mapping

from .geometry import RigidTransform, Vector3
from .targeting import GraspCandidate


class PlanningErrorCode(str, Enum):
    INVALID_REQUEST = "invalid_request"
    INVALID_CONFIG = "invalid_config"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    IK_CANDIDATES_EXHAUSTED = "ik_candidates_exhausted"
    VAMP_SEARCH_FAILED = "vamp_search_failed"
    CONTINUOUS_IK_FAILED = "continuous_ik_failed"
    STAGE_COLLISION_VIOLATION = "stage_collision_violation"
    RUCKIG_FAILED = "ruckig_failed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    BACKEND_ERROR = "backend_error"


class TrajectoryStage(str, Enum):
    PREGRASP = "PREGRASP"
    APPROACH = "APPROACH"
    GRASP = "GRASP"
    LIFT = "LIFT"
    EGRESS = "EGRESS"
    PLACE = "PLACE"


@dataclass(frozen=True)
class JointLimits:
    minimum_position_rad: float
    maximum_position_rad: float
    maximum_velocity_rad_s: float
    maximum_acceleration_rad_s2: float
    maximum_jerk_rad_s3: float

    def validate(self, joint_name: str) -> None:
        values = (
            self.minimum_position_rad,
            self.maximum_position_rad,
            self.maximum_velocity_rad_s,
            self.maximum_acceleration_rad_s2,
            self.maximum_jerk_rad_s3,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError(f"limits for {joint_name} must be finite")
        if self.minimum_position_rad >= self.maximum_position_rad:
            raise ValueError(f"position limits for {joint_name} must be increasing")
        if min(values[2:]) <= 0.0:
            raise ValueError(f"dynamic limits for {joint_name} must be positive")


@dataclass(frozen=True)
class RobotModel:
    model_id: str
    base_frame: str
    tcp_frame: str
    joint_names: tuple[str, ...]
    joint_limits: tuple[JointLimits, ...]
    default_joint_positions: tuple[float, ...]

    def validate(self) -> None:
        if not self.model_id or not self.base_frame or not self.tcp_frame:
            raise ValueError("RobotModel identifiers and frames must be non-empty")
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("RobotModel joint names must be unique and non-empty")
        if len(self.joint_limits) != len(self.joint_names):
            raise ValueError("RobotModel must provide one limit set per joint")
        if len(self.default_joint_positions) != len(self.joint_names):
            raise ValueError("RobotModel default state does not match joint order")
        if not all(isfinite(value) for value in self.default_joint_positions):
            raise ValueError("RobotModel default joint positions must be finite")
        for name, limits, value in zip(
            self.joint_names, self.joint_limits, self.default_joint_positions
        ):
            limits.validate(name)
            if not limits.minimum_position_rad <= value <= limits.maximum_position_rad:
                raise ValueError(f"RobotModel default position for {name} is outside limits")


@dataclass(frozen=True)
class RobotState:
    joint_names: tuple[str, ...]
    positions_rad: tuple[float, ...]

    def validate(self, robot_model: RobotModel) -> None:
        if self.joint_names != robot_model.joint_names:
            raise ValueError("robot state joint order must exactly match RobotModel")
        if len(self.positions_rad) != len(self.joint_names) or not all(
            isfinite(value) for value in self.positions_rad
        ):
            raise ValueError("robot state positions must match joint order and be finite")
        for name, value, limits in zip(
            self.joint_names, self.positions_rad, robot_model.joint_limits
        ):
            if not limits.minimum_position_rad <= value <= limits.maximum_position_rad:
                raise ValueError(f"robot state position for {name} is outside limits")


@dataclass(frozen=True)
class Sphere:
    radius_m: float

    def validate(self) -> None:
        if not isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise ValueError("sphere radius must be finite and positive")


@dataclass(frozen=True)
class Capsule:
    radius_m: float
    length_m: float

    def validate(self) -> None:
        if not all(isfinite(value) and value > 0.0 for value in (self.radius_m, self.length_m)):
            raise ValueError("capsule radius and length must be finite and positive")


@dataclass(frozen=True)
class Box:
    size_m: Vector3

    def validate(self) -> None:
        if len(self.size_m) != 3 or not all(isfinite(value) and value > 0.0 for value in self.size_m):
            raise ValueError("box size must contain three finite positive values")


@dataclass(frozen=True)
class PointCloud:
    points_m: tuple[Vector3, ...]
    point_radius_m: float

    def validate(self) -> None:
        if not self.points_m or not isfinite(self.point_radius_m) or self.point_radius_m <= 0.0:
            raise ValueError("point cloud needs points and a positive point radius")
        if any(len(point) != 3 or not all(isfinite(value) for value in point) for point in self.points_m):
            raise ValueError("point cloud points must be finite xyz triples")


@dataclass(frozen=True)
class ConvexHull:
    vertices_m: tuple[Vector3, ...]
    triangles: tuple[tuple[int, int, int], ...]

    def validate(self) -> None:
        _validate_triangle_geometry(self.vertices_m, self.triangles, "convex hull", 4)


@dataclass(frozen=True)
class StaticTriangleMesh:
    vertices_m: tuple[Vector3, ...]
    triangles: tuple[tuple[int, int, int], ...]

    def validate(self) -> None:
        _validate_triangle_geometry(self.vertices_m, self.triangles, "static triangle mesh", 3)


def _validate_triangle_geometry(
    vertices: tuple[Vector3, ...],
    triangles: tuple[tuple[int, int, int], ...],
    label: str,
    minimum_vertices: int,
) -> None:
    if len(vertices) < minimum_vertices or any(
        len(vertex) != 3 or not all(isfinite(value) for value in vertex)
        for vertex in vertices
    ):
        raise ValueError(f"{label} vertices must contain finite xyz triples")
    if not triangles:
        raise ValueError(f"{label} must contain triangles")
    for triangle in triangles:
        if (
            len(triangle) != 3
            or len(set(triangle)) != 3
            or any(not isinstance(index, int) or index < 0 or index >= len(vertices) for index in triangle)
        ):
            raise ValueError(f"{label} triangles must contain three distinct valid vertex indices")


CollisionGeometry = Sphere | Capsule | Box | PointCloud | ConvexHull | StaticTriangleMesh


@dataclass(frozen=True)
class SceneGeometry:
    geometry_id: str
    object_id: str
    collision_group: str
    T_world_geometry: RigidTransform
    geometry: CollisionGeometry

    def validate(self, world_frame: str) -> None:
        if not self.geometry_id or not self.object_id or not self.collision_group:
            raise ValueError("scene geometry identifiers must be non-empty")
        self.T_world_geometry.validate(label=f"scene geometry {self.geometry_id} pose")
        if self.T_world_geometry.parent_frame != world_frame:
            raise ValueError(f"scene geometry {self.geometry_id} must be expressed in {world_frame}")
        self.geometry.validate()


@dataclass(frozen=True)
class AllowedContact:
    first_group: str
    second_group: str

    def validate(self) -> None:
        if not self.first_group or not self.second_group or self.first_group == self.second_group:
            raise ValueError("allowed contact must name two distinct collision groups")

    def normalized(self) -> tuple[str, str]:
        return tuple(sorted((self.first_group, self.second_group)))  # type: ignore[return-value]


@dataclass(frozen=True)
class SceneSnapshot:
    snapshot_id: str
    world_frame: str
    geometries: tuple[SceneGeometry, ...]
    allowed_contacts: tuple[AllowedContact, ...] = ()

    def validate(self) -> None:
        if not self.snapshot_id or not self.world_frame:
            raise ValueError("scene snapshot ID and world frame must be non-empty")
        ids = [item.geometry_id for item in self.geometries]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("scene geometry IDs must be unique and non-empty")
        for item in self.geometries:
            item.validate(self.world_frame)
        contacts = []
        for contact in self.allowed_contacts:
            contact.validate()
            contacts.append(contact.normalized())
        if len(contacts) != len(set(contacts)):
            raise ValueError("scene allowed contacts must be unique")


class PlacementGoalKind(str, Enum):
    EXACT = "exact"
    RECTANGLE = "rectangle"
    CIRCLE = "circle"


@dataclass(frozen=True)
class PlacementGoal:
    goal_id: str
    kind: PlacementGoalKind
    T_world_tcp: RigidTransform
    size_m: tuple[float, float] | None = None
    radius_m: float | None = None
    height_range_m: tuple[float, float] = (0.0, 0.0)

    def validate(self, world_frame: str, tcp_frame: str) -> None:
        if not self.goal_id:
            raise ValueError("placement goal ID must be non-empty")
        self.T_world_tcp.validate(label="placement goal pose")
        if self.T_world_tcp.parent_frame != world_frame or self.T_world_tcp.child_frame != tcp_frame:
            raise ValueError("placement goal pose frames do not match request frames")
        low, high = self.height_range_m
        if not all(isfinite(value) for value in (low, high)) or low > high:
            raise ValueError("placement height range must be finite and increasing")
        if self.kind is PlacementGoalKind.EXACT:
            if self.size_m is not None or self.radius_m is not None or low != 0.0 or high != 0.0:
                raise ValueError("exact placement cannot declare a sampling region")
        elif self.kind is PlacementGoalKind.RECTANGLE:
            if self.size_m is None or len(self.size_m) != 2 or not all(
                isfinite(value) and value > 0.0 for value in self.size_m
            ) or self.radius_m is not None:
                raise ValueError("rectangle placement needs two positive dimensions")
        elif self.kind is PlacementGoalKind.CIRCLE:
            if self.radius_m is None or not isfinite(self.radius_m) or self.radius_m <= 0.0 or self.size_m is not None:
                raise ValueError("circle placement needs a positive radius")


@dataclass(frozen=True)
class AttachedGeometry:
    geometry_id: str
    T_object_geometry: RigidTransform
    geometry: Sphere | Capsule | Box

    def validate(self) -> None:
        if not self.geometry_id:
            raise ValueError("attached geometry ID must be non-empty")
        self.T_object_geometry.validate(label=f"attached geometry {self.geometry_id} pose")
        if self.T_object_geometry.parent_frame != "object":
            raise ValueError("attached geometry pose parent must be object")
        self.geometry.validate()


@dataclass(frozen=True)
class PlanningObjectConfig:
    object_type: str
    attached_geometry: tuple[AttachedGeometry, ...]

    def validate(self) -> None:
        if not self.object_type or not self.attached_geometry:
            raise ValueError("Planning object config requires type and attached geometry")
        ids = [item.geometry_id for item in self.attached_geometry]
        if len(ids) != len(set(ids)):
            raise ValueError("attached geometry IDs must be unique")
        for item in self.attached_geometry:
            item.validate()


@dataclass(frozen=True)
class PlanningEndEffectorConfig:
    profile_id: str
    actuator_position_channel: str
    actuator_force_limit_channel: str
    open_position_m: float
    contact_groups: tuple[str, ...]

    def validate(self) -> None:
        if not self.profile_id or not self.actuator_position_channel or not self.actuator_force_limit_channel:
            raise ValueError("Planning end-effector IDs and channels must be non-empty")
        if not isfinite(self.open_position_m) or self.open_position_m < 0.0:
            raise ValueError("Planning end-effector open position must be finite and non-negative")
        if not self.contact_groups or len(set(self.contact_groups)) != len(self.contact_groups):
            raise ValueError("Planning end-effector contact groups must be unique")


@dataclass(frozen=True)
class PlanningPolicy:
    ik_position_tolerance_m: float
    ik_orientation_tolerance_rad: float
    cartesian_translation_step_m: float
    cartesian_rotation_step_rad: float
    lift_distance_m: float
    preplace_offset_m: float
    joint_limit_margin_rad: float

    def validate(self) -> None:
        positive_floats = (
            self.ik_position_tolerance_m,
            self.ik_orientation_tolerance_rad,
            self.cartesian_translation_step_m,
            self.cartesian_rotation_step_rad,
            self.lift_distance_m,
            self.preplace_offset_m,
            self.joint_limit_margin_rad,
        )
        if not all(isfinite(value) and value > 0.0 for value in positive_floats):
            raise ValueError("Planning distances, tolerances, margins, and periods must be positive")


@dataclass(frozen=True)
class PlanningBudgets:
    maximum_candidates: int
    seeds_per_candidate: int
    stage_timeout_s: float
    total_timeout_s: float

    def validate(self) -> None:
        if self.maximum_candidates <= 0 or self.seeds_per_candidate <= 0:
            raise ValueError("Planning candidate and seed budgets must be positive")
        if not all(isfinite(value) and value > 0.0 for value in (self.stage_timeout_s, self.total_timeout_s)):
            raise ValueError("Planning time budgets must be finite and positive")
        if self.stage_timeout_s > self.total_timeout_s:
            raise ValueError("Planning stage timeout cannot exceed total timeout")


@dataclass(frozen=True)
class PlanningRequest:
    request_id: str
    candidates: tuple[GraspCandidate, ...]
    initial_state: RobotState
    robot_model: RobotModel
    T_world_base: RigidTransform
    scene: SceneSnapshot
    target_object_id: str
    source_support_group: str
    placement_goal: PlacementGoal
    object_config: PlanningObjectConfig
    end_effector_config: PlanningEndEffectorConfig
    planning_policy: PlanningPolicy
    random_seed: int
    budgets: PlanningBudgets

    def validate(self) -> None:
        if not self.request_id or not self.candidates or not self.target_object_id or not self.source_support_group:
            raise ValueError("Planning request IDs, candidates, and source support must be explicit")
        if self.random_seed < 0:
            raise ValueError("Planning random seed must be non-negative")
        self.robot_model.validate()
        self.initial_state.validate(self.robot_model)
        self.scene.validate()
        self.T_world_base.validate(label="T_world_base")
        if (
            self.T_world_base.parent_frame != self.scene.world_frame
            or self.T_world_base.child_frame != self.robot_model.base_frame
        ):
            raise ValueError("T_world_base frames must connect SceneSnapshot to RobotModel")
        self.placement_goal.validate(self.scene.world_frame, self.robot_model.tcp_frame)
        self.object_config.validate()
        self.end_effector_config.validate()
        self.planning_policy.validate()
        self.budgets.validate()
        object_ids = {item.object_id for item in self.scene.geometries}
        if self.target_object_id not in object_ids:
            raise ValueError("Planning target object is absent from SceneSnapshot")
        scene_groups = {item.collision_group for item in self.scene.geometries}
        if self.source_support_group not in scene_groups:
            raise ValueError("Planning source support group is absent from SceneSnapshot")
        target_groups = {
            item.collision_group
            for item in self.scene.geometries
            if item.object_id == self.target_object_id
        }
        allowed = {item.normalized() for item in self.scene.allowed_contacts}
        if not any(
            tuple(sorted((group, self.source_support_group))) in allowed
            for group in target_groups
        ):
            raise ValueError("SceneSnapshot must declare the target/source-support contact")
        if self.object_config.object_type != self.candidates[0].object_type or any(
            candidate.object_type != self.object_config.object_type for candidate in self.candidates
        ):
            raise ValueError("Planning candidates and object config must name one object type")
        if any(candidate.end_effector_profile_id != self.end_effector_config.profile_id for candidate in self.candidates):
            raise ValueError("Planning candidates and end-effector config must use one profile")


@dataclass(frozen=True)
class PlanningAttempt:
    candidate_id: str
    seed: int
    stage: TrajectoryStage
    elapsed_s: float
    error_code: PlanningErrorCode | None
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "seed": self.seed,
            "stage": self.stage.value,
            "elapsed_s": self.elapsed_s,
            "error_code": self.error_code.value if self.error_code is not None else None,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TrajectoryKnot:
    stage: TrajectoryStage
    time_s: float
    positions_rad: tuple[float, ...]
    velocities_rad_s: tuple[float, ...]
    accelerations_rad_s2: tuple[float, ...]
    actuator_channels: Mapping[str, float]


@dataclass(frozen=True)
class TrajectoryPhase:
    stage: TrajectoryStage
    first_knot: int
    last_knot: int
    first_segment: int
    last_segment: int
    start_time_s: float
    end_time_s: float


@dataclass(frozen=True)
class TrajectoryPolynomialSection:
    start_time_s: float
    duration_s: float
    position_coefficients: tuple[tuple[float, float, float, float], ...]

    def sample(
        self, local_time_s: float
    ) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
        if not 0.0 <= local_time_s <= self.duration_s:
            raise ValueError("trajectory section sample time is outside the section")
        positions = []
        velocities = []
        accelerations = []
        for c0, c1, c2, c3 in self.position_coefficients:
            positions.append(c0 + local_time_s * (c1 + local_time_s * (c2 + local_time_s * c3)))
            velocities.append(c1 + local_time_s * (2.0 * c2 + local_time_s * 3.0 * c3))
            accelerations.append(2.0 * c2 + local_time_s * 6.0 * c3)
        return tuple(positions), tuple(velocities), tuple(accelerations)


@dataclass(frozen=True)
class TrajectorySegment:
    stage: TrajectoryStage
    start_time_s: float
    duration_s: float
    start_knot: int
    end_knot: int
    sections: tuple[TrajectoryPolynomialSection, ...]
    start_actuator_channels: Mapping[str, float]
    end_actuator_channels: Mapping[str, float]

    def sample(self, local_time_s: float) -> TrajectoryKnot:
        if not 0.0 <= local_time_s <= self.duration_s:
            raise ValueError("trajectory segment sample time is outside the segment")
        section = self.sections[-1]
        for candidate in self.sections:
            if local_time_s <= candidate.start_time_s + candidate.duration_s:
                section = candidate
                break
        section_time = min(
            section.duration_s, max(0.0, local_time_s - section.start_time_s)
        )
        positions, velocities, accelerations = section.sample(section_time)
        fraction = local_time_s / self.duration_s
        if set(self.start_actuator_channels) != set(self.end_actuator_channels):
            raise ValueError("trajectory segment actuator channel names are inconsistent")
        channels = {
            name: value + fraction * (self.end_actuator_channels[name] - value)
            for name, value in self.start_actuator_channels.items()
        }
        return TrajectoryKnot(
            self.stage,
            self.start_time_s + local_time_s,
            positions,
            velocities,
            accelerations,
            channels,
        )


@dataclass(frozen=True)
class TrajectoryEvent:
    event: str
    stage: TrajectoryStage
    time_s: float
    actuator_channels: Mapping[str, float]


@dataclass(frozen=True)
class PlannedTrajectory:
    joint_names: tuple[str, ...]
    knots: tuple[TrajectoryKnot, ...]
    segments: tuple[TrajectorySegment, ...]
    phases: tuple[TrajectoryPhase, ...]
    events: tuple[TrajectoryEvent, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "joint_names": list(self.joint_names),
            "knots": [
                {
                    "stage": knot.stage.value,
                    "time_s": knot.time_s,
                    "positions_rad": list(knot.positions_rad),
                    "velocities_rad_s": list(knot.velocities_rad_s),
                    "accelerations_rad_s2": list(knot.accelerations_rad_s2),
                    "actuator_channels": dict(knot.actuator_channels),
                }
                for knot in self.knots
            ],
            "segments": [
                {
                    "stage": segment.stage.value,
                    "start_time_s": segment.start_time_s,
                    "duration_s": segment.duration_s,
                    "start_knot": segment.start_knot,
                    "end_knot": segment.end_knot,
                    "sections": [
                        {
                            "start_time_s": section.start_time_s,
                            "duration_s": section.duration_s,
                            "position_coefficients": [
                                list(coefficients)
                                for coefficients in section.position_coefficients
                            ],
                        }
                        for section in segment.sections
                    ],
                    "start_actuator_channels": dict(segment.start_actuator_channels),
                    "end_actuator_channels": dict(segment.end_actuator_channels),
                }
                for segment in self.segments
            ],
            "phases": [
                {
                    "stage": phase.stage.value,
                    "first_knot": phase.first_knot,
                    "last_knot": phase.last_knot,
                    "first_segment": phase.first_segment,
                    "last_segment": phase.last_segment,
                    "start_time_s": phase.start_time_s,
                    "end_time_s": phase.end_time_s,
                }
                for phase in self.phases
            ],
            "events": [
                {
                    "event": event.event,
                    "stage": event.stage.value,
                    "time_s": event.time_s,
                    "actuator_channels": dict(event.actuator_channels),
                }
                for event in self.events
            ],
        }


@dataclass(frozen=True)
class PlanningResult:
    trajectory: PlannedTrajectory | None
    T_world_object_goal: RigidTransform | None
    error_code: PlanningErrorCode | None
    message: str
    attempts: tuple[PlanningAttempt, ...]
    metadata: Mapping[str, object]

    @property
    def success(self) -> bool:
        return self.error_code is None

    @classmethod
    def succeeded(
        cls,
        trajectory: PlannedTrajectory,
        T_world_object_goal: RigidTransform,
        attempts: tuple[PlanningAttempt, ...],
        metadata: Mapping[str, object],
    ) -> PlanningResult:
        result = cls(
            trajectory,
            T_world_object_goal,
            None,
            "planning succeeded",
            attempts,
            metadata,
        )
        result.validate()
        return result

    @classmethod
    def failed(
        cls,
        error_code: PlanningErrorCode,
        message: str,
        attempts: tuple[PlanningAttempt, ...] = (),
        metadata: Mapping[str, object] | None = None,
    ) -> PlanningResult:
        return cls(None, None, error_code, message, attempts, metadata or {})

    def validate(self) -> None:
        if self.success:
            if self.trajectory is None or self.T_world_object_goal is None:
                raise ValueError(
                    "successful PlanningResult requires a trajectory and T_world_object_goal"
                )
            self.T_world_object_goal.validate(label="PlanningResult.T_world_object_goal")
            if self.T_world_object_goal.child_frame != "object":
                raise ValueError("PlanningResult.T_world_object_goal must end at object")
        elif self.trajectory is not None or self.T_world_object_goal is not None:
            raise ValueError(
                "failed PlanningResult cannot contain a trajectory or T_world_object_goal"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "success": self.success,
            "error_code": self.error_code.value if self.error_code is not None else None,
            "message": self.message,
            "trajectory": self.trajectory.to_dict() if self.trajectory is not None else None,
            "T_world_object_goal": (
                self.T_world_object_goal.to_dict()
                if self.T_world_object_goal is not None
                else None
            ),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "metadata": dict(self.metadata),
        }
