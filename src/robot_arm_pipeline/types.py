from __future__ import annotations

from dataclasses import asdict, dataclass


Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]
BBoxXYXY = tuple[float, float, float, float]
TransformMatrix = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


@dataclass(frozen=True)
class Pose3D:
    position: Vector3
    orientation_xyzw: Quaternion = (0.0, 0.0, 0.0, 1.0)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ObjectDetection:
    object_id: str
    class_name: str
    confidence: float
    bbox_xyxy: BBoxXYXY
    T_world_object: TransformMatrix

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ObjectPose:
    object_id: str
    label: str
    pose: Pose3D
    T_world_object: TransformMatrix | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GraspTarget:
    object_id: str
    pose: Pose3D
    approach_vector: Vector3
    gripper_width_m: float
    T_world_pregrasp: TransformMatrix | None = None
    T_world_grasp: TransformMatrix | None = None
    hand_joint_goal: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CollisionObject:
    object_id: str
    label: str
    pose: Pose3D
    size_m: Vector3

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CollisionScene:
    frame_id: str
    objects: tuple[CollisionObject, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RobotState:
    joint_names: tuple[str, ...]
    joint_positions: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.joint_names) != len(self.joint_positions):
            raise ValueError("joint_names and joint_positions must have the same length")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrajectoryWaypoint:
    time_s: float
    joint_positions: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PlannedTrajectory:
    joint_names: tuple[str, ...]
    waypoints: tuple[TrajectoryWaypoint, ...]
    planner_name: str
    target_object_id: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PlanningRequest:
    object_pose: ObjectPose
    grasp_target: GraspTarget
    robot_state: RobotState

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PlanningResult:
    success: bool
    trajectory: PlannedTrajectory | None
    message: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    executor_name: str
    duration_s: float
    final_state: RobotState
    message: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationReport:
    success: bool
    target_object_id: str
    waypoint_count: int
    duration_s: float
    message: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationResult:
    success: bool
    object_id: str
    planning_success: bool
    execution_success: bool
    trajectory_path: str
    execution_report_path: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
