from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from pathlib import Path
from typing import Callable

from .artifacts import SetupArtifact, Task2Artifacts
from .config import EndEffectorProfile, Task2Config, load_task2_config
from .contracts.execution import (
    DynamicsDiagnosticsConfig,
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    SceneObjectBinding,
)
from .contracts.geometry import reframe
from .contracts.pipeline import (
    PipelineErrorCode,
    PipelineStatus,
    StageErrorCode,
    Task2PipelineResult,
    Task2Request,
)
from .contracts.planning import PlanningRequest, PlanningResult
from .contracts.targeting import (
    TargetingAccessContext,
    TargetingRequest,
    TargetingResult,
)
from .execution import execute
from .planning import plan
from .runtime import (
    ActiveWristParallelRuntime,
    ActivatedEndEffector,
    EndEffectorRuntime,
)
from .targeting import target


TargetingStage = Callable[[TargetingRequest], TargetingResult]
PlanningStage = Callable[[PlanningRequest], PlanningResult]
ExecutionStage = Callable[[ExecutionRequest], ExecutionResult]


@dataclass(frozen=True)
class _ConfiguredPlanner:
    config: Task2Config

    def __call__(self, request: PlanningRequest) -> PlanningResult:
        profile_ids = {item.end_effector_profile_id for item in request.candidates}
        if len(profile_ids) != 1:
            raise ValueError("Planning candidates must identify one end-effector profile")
        profile_id = next(iter(profile_ids))
        profile = next(
            (item for item in self.config.end_effectors if item.profile_id == profile_id),
            None,
        )
        if profile is None:
            raise ValueError(f"Planning profile {profile_id!r} is not configured")
        return plan(request, dependencies=self.config.planning_dependencies(profile))


class Task2Pipeline:
    def __init__(
        self,
        config: Task2Config,
        output_root: str | Path,
        targeting_stage: TargetingStage,
        planning_stage: PlanningStage,
        execution_stage: ExecutionStage,
        runtime: EndEffectorRuntime,
    ):
        self._config = config
        self._output_root = Path(output_root).resolve()
        self._target = targeting_stage
        self._plan = planning_stage
        self._execute = execution_stage
        self._runtime = runtime

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        output_root: str | Path,
        *,
        targeting_stage: TargetingStage | None = None,
        planning_stage: PlanningStage | None = None,
        execution_stage: ExecutionStage | None = None,
        runtime: EndEffectorRuntime | None = None,
    ) -> Task2Pipeline:
        config = load_task2_config(config_path)
        return cls(
            config,
            output_root,
            targeting_stage or target,
            planning_stage or _ConfiguredPlanner(config),
            execution_stage or execute,
            runtime or ActiveWristParallelRuntime(config),
        )

    def run(self, request: Task2Request) -> Task2PipelineResult:
        elapsed: dict[str, float] = {}
        completed: list[str] = []
        profile: EndEffectorProfile | None = None
        activated: ActivatedEndEffector | None = None
        try:
            artifacts = Task2Artifacts(self._output_root, request.request_id)
        except (OSError, TypeError, ValueError) as exc:
            return self._unpersisted_failure("setup", str(exc))
        try:
            artifacts.reset()
        except OSError as exc:
            return self._unpersisted_failure(
                "setup", f"could not reset Task2 artifacts: {exc}"
            )
        try:
            setup_started = monotonic()
            request.validate()
            profile = self._config.select_end_effector(
                request.object_type,
                request.end_effector_profile_id,
            )
            activated = self._activate(request, profile)
            artifacts.write_request(request)
            artifacts.write_setup(
                SetupArtifact(
                    profile.profile_id,
                    profile.family,
                    activated.robot_model.model_id,
                    activated.initial_state,
                )
            )
            artifacts.read_request()
            artifacts.read_setup()
            elapsed["setup"] = monotonic() - setup_started
            completed.append("setup")
        except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
            return self._finish(
                artifacts,
                completed,
                elapsed,
                profile,
                "setup",
                PipelineErrorCode.PIPELINE_FAILED,
                str(exc),
            )

        assert profile is not None and activated is not None
        try:
            targeting_result = self._run_targeting(artifacts, profile, elapsed)
        except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
            return self._finish(
                artifacts,
                completed,
                elapsed,
                profile,
                "targeting",
                PipelineErrorCode.PIPELINE_FAILED,
                str(exc),
            )
        if not targeting_result.success:
            return self._finish(
                artifacts,
                completed,
                elapsed,
                profile,
                "targeting",
                targeting_result.error_code,
                targeting_result.message,
            )
        completed.append("targeting")

        try:
            planning_result = self._run_planning(artifacts, profile, elapsed)
        except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
            return self._finish(
                artifacts,
                completed,
                elapsed,
                profile,
                "planning",
                PipelineErrorCode.PIPELINE_FAILED,
                str(exc),
            )
        if not planning_result.success:
            return self._finish(
                artifacts,
                completed,
                elapsed,
                profile,
                "planning",
                planning_result.error_code,
                planning_result.message,
            )
        completed.append("planning")

        try:
            execution_result = self._run_execution(
                artifacts,
                profile,
                activated,
                elapsed,
            )
        except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
            return self._finish(
                artifacts,
                completed,
                elapsed,
                profile,
                "execution",
                PipelineErrorCode.PIPELINE_FAILED,
                str(exc),
            )
        if execution_result.execution_status is not ExecutionStatus.COMPLETED:
            return self._finish(
                artifacts,
                completed,
                elapsed,
                profile,
                "execution",
                execution_result.error_code,
                execution_result.message,
            )
        completed.append("execution")
        return self._finish(
            artifacts,
            completed,
            elapsed,
            profile,
            None,
            None,
            "Task2 pipeline completed",
        )

    def _run_cli_step(
        self,
        step: str,
        *,
        request: Task2Request | None = None,
        run_dir: str | Path | None = None,
    ) -> Task2PipelineResult:
        if step == "full":
            if request is None:
                return self._unpersisted_failure(
                    "setup", "full step requires a Task2Request"
                )
            return self.run(request)
        if step == "targeting":
            if request is None:
                return self._unpersisted_failure(
                    "setup", "targeting step requires a Task2Request"
                )
            elapsed: dict[str, float] = {}
            completed: list[str] = []
            profile: EndEffectorProfile | None = None
            try:
                artifacts = Task2Artifacts(self._output_root, request.request_id)
            except (OSError, TypeError, ValueError) as exc:
                return self._unpersisted_failure("setup", str(exc))
            try:
                artifacts.reset()
            except OSError as exc:
                return self._unpersisted_failure(
                    "setup", f"could not reset Task2 artifacts: {exc}"
                )
            try:
                started = monotonic()
                request.validate()
                profile = self._config.select_end_effector(
                    request.object_type, request.end_effector_profile_id
                )
                activated = self._activate(request, profile)
                artifacts.write_request(request)
                artifacts.write_setup(
                    SetupArtifact(
                        profile.profile_id,
                        profile.family,
                        activated.robot_model.model_id,
                        activated.initial_state,
                    )
                )
                elapsed["setup"] = monotonic() - started
                completed.append("setup")
                result = self._run_targeting(artifacts, profile, elapsed)
            except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
                return self._finish(
                    artifacts,
                    completed,
                    elapsed,
                    profile,
                    "targeting" if completed else "setup",
                    PipelineErrorCode.PIPELINE_FAILED,
                    str(exc),
                )
            if not result.success:
                return self._finish(
                    artifacts,
                    completed,
                    elapsed,
                    profile,
                    "targeting",
                    result.error_code,
                    result.message,
                )
            completed.append("targeting")
            return self._finish(
                artifacts, completed, elapsed, profile, None, None, "targeting completed"
            )
        if step not in {"planning", "execution"} or run_dir is None:
            failed_stage = step if step in {"planning", "execution"} else "setup"
            return self._unpersisted_failure(
                failed_stage,
                "planning/execution steps require an existing run directory",
            )
        try:
            path = Path(run_dir).resolve()
            artifacts = Task2Artifacts(path.parent, path.name)
            if artifacts.run_dir != path:
                raise ValueError("run directory is invalid")
        except (OSError, TypeError, ValueError) as exc:
            return self._unpersisted_failure(step, str(exc))
        try:
            artifacts.clear_from(step)
        except OSError as exc:
            return self._unpersisted_failure(
                step, f"could not clear stale {step} artifacts: {exc}"
            )
        elapsed: dict[str, float] = {}
        completed: list[str] = []
        profile = None
        try:
            root = artifacts.read_request()
            profile = self._config.select_end_effector(
                root.object_type, root.end_effector_profile_id
            )
            activated = self._activate(root, profile)
            setup = artifacts.read_setup()
            self._validate_setup(setup, profile)
            if setup.initial_state != activated.initial_state:
                raise ValueError(
                    "setup.json state does not match the currently activated profile"
                )
            completed.append("setup")
            if step == "planning":
                result = self._run_planning(artifacts, profile, elapsed)
                completed.append("targeting")
                if not result.success:
                    return self._finish(
                        artifacts,
                        completed,
                        elapsed,
                        profile,
                        "planning",
                        result.error_code,
                        result.message,
                    )
                completed.append("planning")
                return self._finish(
                    artifacts,
                    completed,
                    elapsed,
                    profile,
                    None,
                    None,
                    "planning completed",
                )
            result = self._run_execution(artifacts, profile, activated, elapsed)
            completed.extend(("targeting", "planning"))
            if result.execution_status is not ExecutionStatus.COMPLETED:
                return self._finish(
                    artifacts,
                    completed,
                    elapsed,
                    profile,
                    "execution",
                    result.error_code,
                    result.message,
                )
            completed.append("execution")
            return self._finish(
                artifacts,
                completed,
                elapsed,
                profile,
                None,
                None,
                "execution completed",
            )
        except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
            return self._finish(
                artifacts,
                completed,
                elapsed,
                profile,
                step,
                PipelineErrorCode.PIPELINE_FAILED,
                str(exc),
            )

    def _activate(
        self,
        request: Task2Request,
        profile: EndEffectorProfile,
    ) -> ActivatedEndEffector:
        if request.base_robot_state.joint_names != self._config.base_robot.joint_names:
            raise ValueError("base robot state joint order does not match robot.yaml")
        for name, value, limits in zip(
            request.base_robot_state.joint_names,
            request.base_robot_state.positions_rad,
            self._config.base_robot.joint_limits,
        ):
            if limits.position_bounded and not (
                limits.minimum_position_rad <= value <= limits.maximum_position_rad
            ):
                raise ValueError(f"base robot state position for {name} is outside limits")
        activated = self._runtime.activate(profile, request.base_robot_state)
        if request.T_world_base.child_frame != activated.robot_model.base_frame:
            raise ValueError("T_world_base child frame does not match the activated RobotModel")
        request.placement_goal.validate(
            request.scene.world_frame,
            activated.robot_model.tcp_frame,
        )
        return activated

    def _run_targeting(
        self,
        artifacts: Task2Artifacts,
        profile: EndEffectorProfile,
        elapsed: dict[str, float],
    ) -> TargetingResult:
        artifacts.clear_from("targeting")
        started = monotonic()
        root = artifacts.read_request()
        setup = artifacts.read_setup()
        self._validate_setup(setup, profile)
        result = self._target(
            TargetingRequest(
                root.object_type,
                root.T_world_object,
                self._config.targeting_object(root.object_type),
                profile.targeting,
                self._config.continuous_yaw_resolution_rad,
                self._config.maximum_targeting_candidates,
                (
                    None
                    if root.vertical_transit_corridor is None
                    else TargetingAccessContext(
                        root.vertical_transit_corridor.portal_world_xy_m,
                        root.vertical_transit_corridor.egress_top_offset_m
                        - root.vertical_transit_corridor.ingress_bottom_offset_m,
                    )
                ),
            )
        )
        artifacts.write_targeting(result)
        persisted = artifacts.read_targeting()
        elapsed["targeting"] = monotonic() - started
        return persisted

    def _run_planning(
        self,
        artifacts: Task2Artifacts,
        profile: EndEffectorProfile,
        elapsed: dict[str, float],
    ) -> PlanningResult:
        artifacts.clear_from("planning")
        started = monotonic()
        root = artifacts.read_request()
        setup = artifacts.read_setup()
        self._validate_setup(setup, profile)
        targeting_result = artifacts.read_targeting()
        if not targeting_result.success:
            raise ValueError("Planning requires a successful targeting/result.json")
        robot_model = self._config.robot_model(profile)
        setup.initial_state.validate(robot_model)
        result = self._plan(
            PlanningRequest(
                root.request_id,
                targeting_result.candidates,
                setup.initial_state,
                robot_model,
                root.T_world_base,
                root.scene,
                root.target_object_id,
                root.source_support_group,
                root.placement_goal,
                self._config.planning_object(root.object_type),
                profile.planning,
                self._config.planning_policy,
                root.random_seed,
                self._config.planning_budgets,
                root.vertical_transit_corridor,
            )
        )
        result.validate()
        artifacts.write_planning(result)
        persisted = artifacts.read_planning()
        elapsed["planning"] = monotonic() - started
        return persisted

    def _run_execution(
        self,
        artifacts: Task2Artifacts,
        profile: EndEffectorProfile,
        activated: ActivatedEndEffector,
        elapsed: dict[str, float],
    ) -> ExecutionResult:
        artifacts.clear_from("execution")
        started = monotonic()
        root = artifacts.read_request()
        setup = artifacts.read_setup()
        self._validate_setup(setup, profile)
        planning_result = artifacts.read_planning()
        if (
            not planning_result.success
            or planning_result.trajectory is None
            or planning_result.T_world_object_goal is None
        ):
            raise ValueError("Execution requires a successful planning/result.json")
        settings = self._config.execution_settings
        if planning_result.T_world_object_goal.parent_frame != settings.world_frame:
            raise ValueError(
                "Planning object goal frame does not match Execution diagnostics world frame"
            )
        target_position = tuple(
            planning_result.T_world_object_goal.matrix[index][3]
            for index in range(3)
        )
        result = self._execute(
            ExecutionRequest(
                root.request_id,
                planning_result,
                activated.executor_factory(),
                (
                    SceneObjectBinding(
                        root.target_object_id,
                        root.object_type,
                        reframe(
                            root.T_world_object,
                            child_frame=root.target_object_id,
                        ),
                    ),
                ),
                root.target_object_id,
                settings.control_frequency_hz,
                settings.control_log_frequency_hz,
                settings.diagnostic_log_frequency_hz,
                str(artifacts.control_log_path),
                str(artifacts.dynamics_log_path),
                DynamicsDiagnosticsConfig(
                    settings.world_frame,
                    settings.source_support_geom_names,
                    settings.minimum_departure_height_m,
                    target_position,  # type: ignore[arg-type]
                    settings.target_position_tolerance_m,
                    settings.post_release_observation_s,
                    settings.settled_linear_speed_m_s,
                    settings.settled_angular_speed_rad_s,
                ),
            )
        )
        artifacts.write_execution(result)
        persisted = artifacts.read_execution()
        elapsed["execution"] = monotonic() - started
        return persisted

    @staticmethod
    def _validate_setup(setup: SetupArtifact, profile: EndEffectorProfile) -> None:
        if (
            setup.end_effector_profile_id != profile.profile_id
            or setup.end_effector_family != profile.family
            or setup.robot_model_id != profile.combined_model_id
        ):
            raise ValueError("setup.json does not match the activated end-effector profile")

    def _finish(
        self,
        artifacts: Task2Artifacts,
        completed: list[str],
        elapsed: dict[str, float],
        profile: EndEffectorProfile | None,
        failed_stage: str | None,
        error_code: StageErrorCode | None,
        message: str,
    ) -> Task2PipelineResult:
        produced = {
            name: relative
            for name, relative in (
                ("request", artifacts.REQUEST),
                ("setup", artifacts.SETUP),
                ("targeting", artifacts.TARGETING),
                ("planning", artifacts.PLANNING),
                ("execution", artifacts.EXECUTION),
                ("control_log", artifacts.CONTROL_LOG),
                ("dynamics_log", artifacts.DYNAMICS_LOG),
            )
            if (artifacts.run_dir / relative).is_file()
        }
        result = Task2PipelineResult(
            PipelineStatus.COMPLETED if failed_stage is None else PipelineStatus.FAILED,
            tuple(completed),
            failed_stage,
            profile.profile_id if profile is not None else None,
            dict(elapsed),
            produced,
            error_code,
            message,
        )
        try:
            artifacts.write_result(result)
        except (OSError, TypeError, ValueError) as exc:
            return Task2PipelineResult(
                PipelineStatus.FAILED,
                tuple(completed),
                "result",
                result.selected_end_effector_profile_id,
                dict(elapsed),
                produced,
                PipelineErrorCode.PIPELINE_FAILED,
                f"could not write result.json: {exc}",
            )
        return result

    @staticmethod
    def _unpersisted_failure(
        failed_stage: str,
        message: str,
    ) -> Task2PipelineResult:
        return Task2PipelineResult(
            PipelineStatus.FAILED,
            (),
            failed_stage,
            None,
            {},
            {},
            PipelineErrorCode.PIPELINE_FAILED,
            message,
        )
