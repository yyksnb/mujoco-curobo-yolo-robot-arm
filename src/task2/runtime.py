from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from .adapters.mujoco import (
    MuJoCoExecutorAdapter,
    read_activated_joint_state,
)
from .config import EndEffectorProfile, Task2Config
from .contracts.execution import ExecutorAdapter
from .contracts.planning import RobotModel, RobotState


@dataclass(frozen=True)
class ActivatedEndEffector:
    profile: EndEffectorProfile
    robot_model: RobotModel
    initial_state: RobotState
    executor_factory: Callable[[], ExecutorAdapter]


class EndEffectorRuntime(Protocol):
    def activate(
        self,
        profile: EndEffectorProfile,
        base_robot_state: RobotState,
    ) -> ActivatedEndEffector: ...


class ActiveWristParallelRuntime:
    def __init__(self, config: Task2Config):
        self._config = config

    def activate(
        self,
        profile: EndEffectorProfile,
        base_robot_state: RobotState,
    ) -> ActivatedEndEffector:
        if profile.profile_id not in self._config.mujoco_robots:
            raise ValueError(f"no runtime provider for profile {profile.profile_id!r}")
        robot_model = self._config.robot_model(profile)
        robot_config = self._config.mujoco_robots[profile.profile_id]
        end_effector_config = self._config.mujoco_end_effectors[profile.profile_id]
        initial_state = read_activated_joint_state(
            robot_config,
            robot_model,
            base_robot_state,
        )
        return ActivatedEndEffector(
            profile,
            robot_model,
            initial_state,
            lambda: MuJoCoExecutorAdapter(
                robot_config,
                end_effector_config,
                self._config.mujoco_objects,
            ),
        )
