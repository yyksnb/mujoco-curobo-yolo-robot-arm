from __future__ import annotations

from pathlib import Path

from ..contracts import PlanningConfig, RobotModel
from ..ik import IKSolver
from .errors import BackendUnavailableError


def create_ik_solver(
    backend: str,
    robot_model: RobotModel,
    planning_config: PlanningConfig,
    *,
    project_root: Path,
    random_seed: int,
) -> IKSolver:
    if backend == "curobo":
        from .curobo_ik import CuroboIKSolver

        return CuroboIKSolver(
            robot_model,
            planning_config,
            project_root=project_root,
            random_seed=random_seed,
        )
    raise BackendUnavailableError(f"IK backend {backend!r} is not registered")
