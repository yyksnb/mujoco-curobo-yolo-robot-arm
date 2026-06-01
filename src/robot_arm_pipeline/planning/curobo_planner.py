from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from robot_arm_pipeline.planning.curobo_conversions import (
    collision_scene_to_curobo_world_config,
    transform_to_curobo_pose,
)
from robot_arm_pipeline.scene import build_collision_scene
from robot_arm_pipeline.types import PlanningRequest, PlanningResult


CUROBO_UNAVAILABLE_MESSAGE = (
    "cuRobo is not installed or not configured. "
    "Use MockPlanner or set up Linux CUDA environment."
)


@dataclass(frozen=True)
class CuroboPlannerConfig:
    robot_config_path: Path | None = None
    world_config_path: Path | None = None
    ee_link: str | None = None
    base_link: str | None = None
    joint_names: tuple[str, ...] = ()
    use_cuda: bool = True
    interpolation_dt: float = 0.01


@dataclass(frozen=True)
class OptionalCuroboModules:
    curobo: Any | None
    torch: Any | None
    message: str | None = None


class CuroboPlanner:
    name = "curobo_planner"

    def __init__(
        self,
        *,
        robot_config_path: Path | str | None = None,
        world_config_path: Path | str | None = None,
        ee_link: str | None = None,
        base_link: str | None = None,
        joint_names: tuple[str, ...] = (),
        use_cuda: bool = True,
        interpolation_dt: float = 0.01,
    ) -> None:
        self.config = CuroboPlannerConfig(
            robot_config_path=Path(robot_config_path) if robot_config_path else None,
            world_config_path=Path(world_config_path) if world_config_path else None,
            ee_link=ee_link,
            base_link=base_link,
            joint_names=joint_names,
            use_cuda=use_cuda,
            interpolation_dt=interpolation_dt,
        )

    @staticmethod
    def is_available() -> bool:
        return find_spec("curobo") is not None and find_spec("torch") is not None

    def plan(self, request: PlanningRequest) -> PlanningResult:
        modules = self._load_optional_modules()
        if modules.message:
            return PlanningResult(success=False, trajectory=None, message=modules.message)

        cuda_message = self._validate_cuda(modules.torch)
        if cuda_message:
            return PlanningResult(success=False, trajectory=None, message=cuda_message)

        config_message = self._validate_config()
        if config_message:
            return PlanningResult(success=False, trajectory=None, message=config_message)

        schema_message = self._validate_request_schema(request)
        if schema_message:
            return PlanningResult(success=False, trajectory=None, message=schema_message)

        return PlanningResult(
            success=False,
            trajectory=None,
            message=(
                "CuroboPlanner skeleton validated inputs, but real CUDA motion "
                "planning is not implemented in Stage 4.1."
            ),
        )

    def _load_optional_modules(self) -> OptionalCuroboModules:
        if find_spec("curobo") is None or find_spec("torch") is None:
            return OptionalCuroboModules(curobo=None, torch=None, message=CUROBO_UNAVAILABLE_MESSAGE)

        import importlib

        try:
            curobo = importlib.import_module("curobo")
            torch = importlib.import_module("torch")
        except ImportError:
            return OptionalCuroboModules(curobo=None, torch=None, message=CUROBO_UNAVAILABLE_MESSAGE)
        return OptionalCuroboModules(curobo=curobo, torch=torch)

    def _validate_cuda(self, torch: Any | None) -> str | None:
        if not self.config.use_cuda:
            return None
        if torch is None or not torch.cuda.is_available():
            return "torch CUDA is not available. Set up Linux CUDA + PyTorch for cuRobo planning."
        return None

    def _validate_config(self) -> str | None:
        if self.config.robot_config_path is None:
            return "cuRobo robot_config_path is required before real planning can run."
        if not self.config.robot_config_path.exists():
            return f"cuRobo robot_config_path does not exist: {self.config.robot_config_path}"
        if self.config.world_config_path is not None and not self.config.world_config_path.exists():
            return f"cuRobo world_config_path does not exist: {self.config.world_config_path}"
        return None

    def _validate_request_schema(self, request: PlanningRequest) -> str | None:
        if request.grasp_target.T_world_pregrasp is None:
            return "PlanningRequest.grasp_target.T_world_pregrasp is required for cuRobo goal pose conversion."
        scene = build_collision_scene((request.object_pose,))
        collision_scene_to_curobo_world_config(scene)
        transform_to_curobo_pose(request.grasp_target.T_world_pregrasp)
        return None
