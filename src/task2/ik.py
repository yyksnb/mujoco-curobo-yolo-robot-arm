from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Callable, Protocol

from .backends.errors import BackendUnavailableError
from .contracts import PlanningPolicy, RigidTransform, RobotModel, RobotState


class IKErrorCode(str, Enum):
    NO_SOLUTION = "no_solution"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    BACKEND_ERROR = "backend_error"


@dataclass(frozen=True)
class IKConstraints:
    position_tolerance_m: float
    orientation_tolerance_rad: float
    maximum_solutions: int
    solution_validator: Callable[[RobotState], bool] | None = None


@dataclass(frozen=True)
class IKSolution:
    state: RobotState
    position_error_m: float
    orientation_error_rad: float


@dataclass(frozen=True)
class IKResult:
    solutions: tuple[IKSolution, ...]
    error_code: IKErrorCode | None
    message: str

    @property
    def success(self) -> bool:
        return self.error_code is None


class IKSolver(Protocol):
    @property
    def version(self) -> str: ...

    def solve(
        self,
        targets: tuple[RigidTransform, ...],
        seed_states: tuple[RobotState, ...],
        constraints: IKConstraints,
    ) -> IKResult: ...


def create_ik_solver(
    backend: str,
    robot_model: RobotModel,
    planning_policy: PlanningPolicy,
    backend_config: object,
) -> IKSolver:
    if backend == "curobo":
        from .backends.curobo_ik import CuroboIKConfig, CuroboIKSolver

        if not isinstance(backend_config, CuroboIKConfig):
            raise BackendUnavailableError("cuRobo IK backend config has the wrong type")
        return CuroboIKSolver(
            robot_model,
            planning_policy,
            backend_config,
        )
    raise BackendUnavailableError(f"IK backend {backend!r} is not registered")


def validate_ik_result(result: IKResult, robot_model: RobotModel) -> None:
    if result.success and not result.solutions:
        raise ValueError("successful IK result must contain solutions")
    for solution in result.solutions:
        solution.state.validate(robot_model)
        if not all(
            isfinite(value) and value >= 0.0
            for value in (solution.position_error_m, solution.orientation_error_rad)
        ):
            raise ValueError("IK residuals must be finite and non-negative")
