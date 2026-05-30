from __future__ import annotations

from dataclasses import asdict, dataclass


Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


@dataclass(frozen=True)
class Pose3D:
    position: Vector3
    orientation_xyzw: Quaternion = (0.0, 0.0, 0.0, 1.0)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ObjectPose:
    object_id: str
    label: str
    pose: Pose3D

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GraspTarget:
    object_id: str
    pose: Pose3D
    approach_vector: Vector3
    gripper_width_m: float

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

