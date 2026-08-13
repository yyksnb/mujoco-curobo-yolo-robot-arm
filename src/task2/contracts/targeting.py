from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from .config import EndEffectorTargetingConfig, TargetingDiagnosticsConfig, TargetingObjectConfig
from .geometry import RigidTransform


class TargetingErrorCode(str, Enum):
    UNKNOWN_OBJECT = "unknown_object"
    INCOMPATIBLE_END_EFFECTOR = "incompatible_end_effector"
    INVALID_CONFIGURATION = "invalid_configuration"
    INVALID_INPUT_POSE = "invalid_input_pose"
    NO_CANDIDATES = "no_candidates"


@dataclass(frozen=True)
class ParallelGripperCommand:
    pregrasp_opening_m: float
    target_opening_m: float
    grip_force_limit_n: float
    closing_speed_m_s: float
    dwell_s: float
    calibration: str

    def validate(self) -> None:
        values = (
            self.pregrasp_opening_m,
            self.target_opening_m,
            self.grip_force_limit_n,
            self.closing_speed_m_s,
            self.dwell_s,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("parallel-gripper command values must be finite")
        if self.target_opening_m < 0.0 or self.pregrasp_opening_m < self.target_opening_m:
            raise ValueError("parallel-gripper openings must satisfy 0 <= target <= pregrasp")
        if self.grip_force_limit_n <= 0.0 or self.closing_speed_m_s <= 0.0:
            raise ValueError("parallel-gripper force limit and closing speed must be positive")
        if self.dwell_s < 0.0:
            raise ValueError("parallel-gripper dwell must be non-negative")
        if self.calibration != "simulation_nominal":
            raise ValueError("parallel-gripper command must identify simulation_nominal calibration")

    def to_dict(self) -> dict[str, object]:
        return {
            "pregrasp_opening_m": self.pregrasp_opening_m,
            "target_opening_m": self.target_opening_m,
            "grip_force_limit_n": self.grip_force_limit_n,
            "closing_speed_m_s": self.closing_speed_m_s,
            "dwell_s": self.dwell_s,
            "calibration": self.calibration,
        }


@dataclass(frozen=True)
class ScoreComponents:
    template_priority: float

    def to_dict(self) -> dict[str, float]:
        return {"template_priority": self.template_priority}


@dataclass(frozen=True)
class GraspCandidate:
    candidate_id: str
    object_type: str
    end_effector_profile_id: str
    template_id: str
    symmetry_yaw_rad: float
    T_object_tcp: RigidTransform
    T_world_tcp: RigidTransform
    T_world_pregrasp: RigidTransform
    T_tcp_object: RigidTransform
    quality_score: float
    score_components: ScoreComponents
    diagnostics: TargetingDiagnosticsConfig
    grasp_command: ParallelGripperCommand
    expected_contacts: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "object_type": self.object_type,
            "end_effector_profile_id": self.end_effector_profile_id,
            "template_id": self.template_id,
            "symmetry_yaw_rad": self.symmetry_yaw_rad,
            "T_object_tcp": self.T_object_tcp.to_dict(),
            "T_world_tcp": self.T_world_tcp.to_dict(),
            "T_world_pregrasp": self.T_world_pregrasp.to_dict(),
            "T_tcp_object": self.T_tcp_object.to_dict(),
            "quality_score": self.quality_score,
            "score_components": self.score_components.to_dict(),
            "diagnostics": {
                "contact_margin_m": self.diagnostics.contact_margin_m,
                "opening_margin_m": self.diagnostics.opening_margin_m,
                "force_margin_n": self.diagnostics.force_margin_n,
            },
            "grasp_command": self.grasp_command.to_dict(),
            "expected_contacts": list(self.expected_contacts),
        }


@dataclass(frozen=True)
class TargetingRequest:
    object_type: str
    T_world_object: RigidTransform
    object_config: TargetingObjectConfig
    end_effector_config: EndEffectorTargetingConfig
    continuous_yaw_resolution_rad: float
    maximum_candidates: int


@dataclass(frozen=True)
class TargetingResult:
    candidates: tuple[GraspCandidate, ...]
    error_code: TargetingErrorCode | None
    message: str

    @property
    def success(self) -> bool:
        return self.error_code is None

    @classmethod
    def succeeded(cls, candidates: tuple[GraspCandidate, ...]) -> TargetingResult:
        if not candidates:
            raise ValueError("successful TargetingResult requires candidates")
        return cls(candidates=candidates, error_code=None, message="targeting succeeded")

    @classmethod
    def failed(cls, error_code: TargetingErrorCode, message: str) -> TargetingResult:
        return cls(candidates=(), error_code=error_code, message=message)

    def to_dict(self) -> dict[str, object]:
        return {
            "success": self.success,
            "error_code": self.error_code.value if self.error_code is not None else None,
            "message": self.message,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }
