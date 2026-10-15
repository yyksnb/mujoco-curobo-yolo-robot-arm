from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Mapping

from .execution import ExecutionErrorCode
from .geometry import RigidTransform
from .planning import (
    PlacementGoal,
    PlacementOrientationConstraint,
    PlanningErrorCode,
    RobotState,
    SceneSnapshot,
    VerticalTransitCorridor,
)
from .targeting import TargetingErrorCode


class PipelineErrorCode(str, Enum):
    PIPELINE_FAILED = "pipeline_failed"


class PipelineStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


StageErrorCode = (
    PipelineErrorCode | TargetingErrorCode | PlanningErrorCode | ExecutionErrorCode
)


@dataclass(frozen=True)
class Task2Request:
    request_id: str
    object_type: str
    T_world_object: RigidTransform
    base_robot_state: RobotState
    T_world_base: RigidTransform
    scene: SceneSnapshot
    target_object_id: str
    source_support_group: str
    placement_goal: PlacementGoal
    end_effector_profile_id: str | None = None
    vertical_transit_corridor: VerticalTransitCorridor | None = None

    def validate(self) -> None:
        if not self.request_id or not self.object_type:
            raise ValueError("Task2 request and object type must be explicit")
        if not self.target_object_id or not self.source_support_group:
            raise ValueError("Task2 target object and source support must be explicit")
        self.T_world_object.validate(label="T_world_object")
        if (
            self.T_world_object.parent_frame != "world"
            or self.T_world_object.child_frame != "object"
        ):
            raise ValueError("Task2 object pose must be world -> object")
        self.T_world_base.validate(label="T_world_base")
        self.scene.validate()
        if self.T_world_base.parent_frame != self.scene.world_frame:
            raise ValueError("T_world_base must start at the SceneSnapshot world frame")
        if len(self.base_robot_state.joint_names) != len(
            self.base_robot_state.positions_rad
        ) or len(self.base_robot_state.joint_names) != len(
            set(self.base_robot_state.joint_names)
        ):
            raise ValueError("base robot state must contain unique named positions")
        if not all(isfinite(value) for value in self.base_robot_state.positions_rad):
            raise ValueError("base robot state positions must be finite")
        object_ids = {item.object_id for item in self.scene.geometries}
        groups = {item.collision_group for item in self.scene.geometries}
        if self.target_object_id not in object_ids:
            raise ValueError("Task2 target object is absent from SceneSnapshot")
        if self.source_support_group not in groups:
            raise ValueError("Task2 source support is absent from SceneSnapshot")
        if self.vertical_transit_corridor is not None:
            self.vertical_transit_corridor.validate()
            if (
                self.placement_goal.orientation_constraint
                is not PlacementOrientationConstraint.FREE
            ):
                raise ValueError(
                    "vertical transit corridor requires free placement orientation"
                )


@dataclass(frozen=True)
class Task2PipelineResult:
    status: PipelineStatus
    completed_stages: tuple[str, ...]
    failed_stage: str | None
    selected_end_effector_profile_id: str | None
    stage_elapsed_s: Mapping[str, float]
    artifact_paths: Mapping[str, str]
    error_code: StageErrorCode | None
    message: str

    @property
    def success(self) -> bool:
        return self.status is PipelineStatus.COMPLETED

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "completed_stages": list(self.completed_stages),
            "failed_stage": self.failed_stage,
            "selected_end_effector_profile_id": self.selected_end_effector_profile_id,
            "stage_elapsed_s": dict(self.stage_elapsed_s),
            "artifact_paths": dict(self.artifact_paths),
            "error_code": self.error_code.value if self.error_code is not None else None,
            "message": self.message,
        }
