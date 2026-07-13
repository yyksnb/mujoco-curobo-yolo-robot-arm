from __future__ import annotations

import json
import multiprocessing
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from robot_arm_pipeline.planning import CameraRoutePlan
from task1.final.processing import (
    DEFAULT_FINAL_CONFIG_PATH,
    FINAL_REPORT_SCHEMA,
    FinalCandidate,
    FinalConfig,
    FinalProcessor,
    load_final_config,
    load_survey_candidates,
    make_final_failure_report,
)
from task1.detection import YoloDetector
from task1.scene import (
    DEFAULT_OBJECT_COUNT,
    load_world_pose_in_base,
    make_random_target_object_pose_payload,
)
from task1.survey.manifest import localize_capture_manifest
from task1.survey.simulation import MujocoSurveySimulation, SimulationConfig
from task1.survey.route import (
    DEFAULT_SURVEY_ROUTE_PLAN_PATH,
    load_survey_route_plan,
)
from task1.survey.config import (
    DEFAULT_CONFIG_PATH,
    SurveyDetectionConfig,
    load_survey_detection_config,
)


@dataclass(frozen=True)
class PipelineOptions:
    repo_root: Path
    output_dir: Path
    seed: int | None = None
    step: str | None = None
    layout_path: Path | None = None
    capture_manifest: Path | None = None
    survey_config: Path = DEFAULT_CONFIG_PATH
    final_config: Path = DEFAULT_FINAL_CONFIG_PATH
    survey_report_path: Path | None = None
    retain_survey_depth: bool = False


@dataclass(frozen=True)
class PipelineResult:
    status: str
    run_dir: Path
    completed_stages: tuple[str, ...]
    failed_stage: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "run_dir": str(self.run_dir),
            "completed_stages": list(self.completed_stages),
            "failed_stage": self.failed_stage,
        }


@dataclass(frozen=True)
class PipelineStageEvent:
    stage: str
    status: str
    elapsed_s: float | None = None


@dataclass(frozen=True)
class _Stage:
    name: str
    run: Callable[[], bool]


def _run_survey_worker(
    *,
    simulation_config: SimulationConfig,
    survey_config: SurveyDetectionConfig,
    layout_path: Path,
    survey_dir: Path,
    route: CameraRoutePlan,
    planner_artifact: str,
) -> None:
    """Persist Survey production output before simulation-only evaluation."""
    simulation = MujocoSurveySimulation(
        simulation_config,
        layout_path=layout_path,
        output_dir=survey_dir,
        detector=YoloDetector(survey_config),
        yolo_evaluation_policy=survey_config.evaluation.yolo,
    )
    report = simulation.run(
        route,
        planner_artifact=planner_artifact,
        policy=survey_config.localization,
    )
    report_path = survey_dir / "survey_report.json"
    _write_json(report_path, report)
    if report.get("status") != "success":
        return

    try:
        report.update(simulation.evaluate(report, policy=survey_config.localization))
    except Exception as exc:
        report.update(
            _survey_evaluation_error(
                primary_stage="evaluation_module",
                message=f"{type(exc).__name__}: {exc}",
            )
        )
    _write_json(report_path, report)


def _run_final_worker(
    *,
    repo_root: Path,
    final_config: FinalConfig,
    layout_path: Path,
    final_dir: Path,
    candidates: tuple[FinalCandidate, ...],
    start_joint_positions: tuple[float, ...],
    source_survey_report: Path,
) -> None:
    """Keep native CUDA/OpenGL runtimes out of the orchestrator process."""
    report_path = final_dir / "final_report.json"
    capture = None
    try:
        from robot_arm_pipeline.planning import (
            CameraRoutePlanningPolicy,
            CuroboCameraRoutePlanner,
        )
        from task1.final.simulation import MujocoFinalCapture

        capture = MujocoFinalCapture(
            repo_root=repo_root,
            layout_path=layout_path,
            output_dir=final_dir,
            image_width=final_config.camera.image_width,
            image_height=final_config.camera.image_height,
            camera_name=final_config.simulation.camera_name,
            model_path=final_config.simulation.model_path,
            ground_z_m=final_config.simulation.ground_z_m,
        )
        processor = FinalProcessor(
            planner=CuroboCameraRoutePlanner(
                repo_root=repo_root,
                robot_config_path=final_config.planning.robot_config_path,
                world_config_path=final_config.planning.world_config_path,
                graph_config_path=final_config.planning.graph_config_path,
                planning_policy=CameraRoutePlanningPolicy(
                    max_attempts=final_config.planning.max_attempts,
                    enable_graph_attempt=final_config.planning.enable_graph_attempt,
                    num_ik_seeds=final_config.planning.num_ik_seeds,
                    num_trajopt_seeds=final_config.planning.num_trajopt_seeds,
                    random_seed=final_config.planning.random_seed,
                    position_tolerance=final_config.planning.position_tolerance_m,
                    orientation_tolerance=final_config.planning.orientation_tolerance_rad,
                ),
                ik_batch_size=final_config.planning.ik_batch_size,
                ik_solutions_per_target=final_config.planning.ik_solutions_per_target,
            ),
            capture=capture,
            detector=YoloDetector(final_config.detection),
            config=final_config,
            world_pose_base=load_world_pose_in_base(final_config.simulation.model_path),
            output_dir=final_dir,
        )
        report = processor.run(
            candidates,
            start_joint_positions=start_joint_positions,
            source_survey_report=source_survey_report,
        )
        report["simulation"] = _final_simulation_metadata(final_config)
    except Exception as exc:
        report = make_final_failure_report(
            failure_stage="final_worker",
            message=f"{type(exc).__name__}: {exc}",
            source_survey_report=source_survey_report,
            final_config_path=final_config.config_path,
            expected_object_count=final_config.evaluation.expected_object_count,
            candidate_count=len(candidates),
        )
        report["simulation"] = _final_simulation_metadata(final_config)
    final_dir.mkdir(parents=True, exist_ok=True)
    _write_json(report_path, report)

    try:
        from task1.final.evaluation import evaluate_final_simulation

        traces = (
            capture.capture_evaluation_traces()
            if capture is not None and report.get("failure_stage") != "final_worker"
            else ()
        )
        evaluation = evaluate_final_simulation(
            final_report=report,
            candidates=candidates,
            layout_path=layout_path,
            traces=traces,
            policy=final_config.evaluation,
            image_width=final_config.camera.image_width,
            image_height=final_config.camera.image_height,
        )
    except Exception as evaluation_exc:
        evaluation = _final_evaluation_error(
            primary_stage="evaluation_module",
            message=f"{type(evaluation_exc).__name__}: {evaluation_exc}",
        )
    _attach_final_evaluation(final_dir, report, evaluation)
    _write_json(report_path, report)


def _final_simulation_metadata(config: FinalConfig) -> dict[str, Any]:
    return {
        "enabled": True,
        "renderer": "mujoco",
        "model_path": str(config.simulation.model_path),
        "camera_name": config.simulation.camera_name,
        "ground_z_m": config.simulation.ground_z_m,
        "image_width": config.camera.image_width,
        "image_height": config.camera.image_height,
    }


def _final_evaluation_error(*, primary_stage: str, message: str) -> dict[str, Any]:
    return {
        "schema": "task1_final_simulation_evaluation",
        "status": "error",
        "success": False,
        "evaluation_only": True,
        "used_for_production_control": False,
        "diagnosis": {
            "status": "issue_found",
            "issue_kind": "internal_likely",
            "primary_stage": primary_stage,
            "message": message,
        },
    }


def _survey_evaluation_error(*, primary_stage: str, message: str) -> dict[str, Any]:
    return {
        "candidate_position_evaluation": None,
        "yolo_evaluation": None,
        "detection_diagnosis": None,
        "simulation_evaluation": {
            "status": "error",
            "evaluation_only": True,
            "used_for_production_control": False,
            "diagnosis": {
                "primary_stage": primary_stage,
                "message": message,
            },
        },
    }


def _attach_final_evaluation(
    final_dir: Path,
    report: dict[str, Any],
    evaluation: dict[str, Any],
) -> None:
    evaluation_path = final_dir / "final_evaluation.json"
    _write_json(evaluation_path, evaluation)
    report["simulation_evaluation"] = evaluation
    report["simulation_evaluation_path"] = str(evaluation_path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_valid_final_report(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    required_fields = {
        "failure_stage",
        "candidate_count",
        "processed_candidate_count",
        "successful_candidate_count",
        "failed_candidate_count",
        "unprocessed_candidate_count",
        "candidate_processing_order",
        "candidate_count_evaluation",
        "artifact_generation_failure_count",
        "results",
        "stable_objects",
        "simulation",
        "simulation_evaluation",
        "simulation_evaluation_path",
    }
    if (
        not isinstance(value, dict)
        or value.get("schema") != FINAL_REPORT_SCHEMA
        or value.get("stage") != "final"
        or value.get("status") not in {"success", "partial", "failed"}
        or not required_fields.issubset(value)
    ):
        return None
    return value


def _load_valid_survey_report(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    required_fields = {
        "planner_artifact",
        "detection_source",
        "candidate_localization_policy",
        "simulation",
        "artifact_retention",
        "layout_path",
        "views",
        "observations",
        "candidates",
        "localization_failures",
        "artifact_generation_failure_count",
        "candidate_position_evaluation",
        "yolo_evaluation",
        "detection_diagnosis",
        "fusion_diagnostics",
        "simulation_evaluation",
    }
    if (
        not isinstance(value, dict)
        or value.get("schema") != "task1_survey_report"
        or value.get("stage") != "survey"
        or value.get("status") not in {"success", "partial", "failed"}
        or not required_fields.issubset(value)
    ):
        return None
    return value


class Task1Pipeline:
    """Task1 stage orchestrator. Add later stages to `_stages` in dependency order."""

    def __init__(
        self,
        options: PipelineOptions,
        *,
        stage_observer: Callable[[PipelineStageEvent], None] | None = None,
    ) -> None:
        self.options = options
        self.stage_observer = stage_observer
        self.repo_root = options.repo_root.resolve()
        self.run_dir = self._make_run_dir()
        self.layout_path: Path | None = None
        self.survey_report_path: Path | None = None
        self._stages = (
            _Stage("layout", self._run_layout),
            _Stage("survey", self._run_survey),
            _Stage("final", self._run_final),
        )

    @property
    def stage_names(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self._stages)

    def run(self) -> PipelineResult:
        selected = self._selected_stages()
        completed: list[str] = []
        for stage in selected:
            self._emit_stage_event(PipelineStageEvent(stage.name, "started"))
            started = time.perf_counter()
            try:
                succeeded = stage.run()
            except Exception:
                self._emit_stage_event(
                    PipelineStageEvent(stage.name, "failed", time.perf_counter() - started)
                )
                raise
            self._emit_stage_event(
                PipelineStageEvent(
                    stage.name,
                    "success" if succeeded else "failed",
                    time.perf_counter() - started,
                )
            )
            if not succeeded:
                return PipelineResult("failed", self.run_dir, tuple(completed), stage.name)
            completed.append(stage.name)
        return PipelineResult("success", self.run_dir, tuple(completed))

    def _emit_stage_event(self, event: PipelineStageEvent) -> None:
        if self.stage_observer is not None:
            self.stage_observer(event)

    def _selected_stages(self) -> tuple[_Stage, ...]:
        if self.options.step is None:
            return self._stages
        for stage in self._stages:
            if stage.name == self.options.step:
                return (stage,)
        available = ", ".join(self.stage_names)
        raise ValueError(f"unknown Task1 step {self.options.step!r}; available steps: {available}")

    def _run_layout(self) -> bool:
        output = self.run_dir / "layout" / "target_object_poses.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        if self.options.layout_path is not None:
            source = self.options.layout_path.resolve()
            if not source.exists():
                raise FileNotFoundError(f"Task1 layout does not exist: {source}")
            if source != output.resolve():
                shutil.copyfile(source, output)
        else:
            if self.options.seed is None:
                raise ValueError("Task1 layout requires --seed or --layout")
            payload = make_random_target_object_pose_payload(
                model_path=self.repo_root / "examples/mujoco/gen3_with_tank.xml",
                seed=self.options.seed,
                object_count=DEFAULT_OBJECT_COUNT,
            )
            output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.layout_path = output
        return True

    def _run_survey(self) -> bool:
        if self.layout_path is None:
            self._run_layout()
        if self.layout_path is None:
            raise RuntimeError("Task1 layout stage did not produce an artifact")

        survey_dir = self.run_dir / "survey"
        survey_dir.mkdir(parents=True, exist_ok=True)
        source_route_path = self.repo_root / DEFAULT_SURVEY_ROUTE_PLAN_PATH
        route = load_survey_route_plan(source_route_path, repo_root=self.repo_root)
        route_path = survey_dir / "curobo_route_plan.json"
        if source_route_path.resolve() != route_path.resolve():
            shutil.copyfile(source_route_path, route_path)
        planner_artifact = route_path.name
        report_path = survey_dir / "survey_report.json"
        survey_config = load_survey_detection_config(
            self.options.survey_config, repo_root=self.repo_root
        )

        if self.options.capture_manifest is not None:
            if self.options.retain_survey_depth:
                raise ValueError(
                    "--retain-survey-depth applies only to generated MuJoCo depth; "
                    "capture-manifest depth files remain owned by their source manifest"
                )
            report = {
                **localize_capture_manifest(
                    self.options.capture_manifest,
                    detector=YoloDetector(survey_config),
                    annotated_image_dir=survey_dir / "annotated",
                    policy=survey_config.localization,
                ),
                "schema": "task1_survey_report",
                "stage": "survey",
                "planner_artifact": planner_artifact,
            }
            _write_json(report_path, report)
        else:
            worker = multiprocessing.get_context("spawn").Process(
                target=_run_survey_worker,
                kwargs={
                    "simulation_config": SimulationConfig(
                        repo_root=self.repo_root,
                        candidate_position_tolerance_m=(
                            survey_config.evaluation.candidate.position_tolerance_m
                        ),
                        retain_depth_artifacts=self.options.retain_survey_depth,
                    ),
                    "survey_config": survey_config,
                    "layout_path": self.layout_path.resolve(),
                    "survey_dir": survey_dir,
                    "route": route,
                    "planner_artifact": planner_artifact,
                },
            )
            worker.start()
            worker.join()
            report = _load_valid_survey_report(report_path)
            if worker.exitcode != 0:
                if report is None:
                    raise RuntimeError(
                        "Task1 Survey worker exited abnormally with code "
                        f"{worker.exitcode} before producing a valid production report."
                    )
                if report["status"] == "success":
                    report.update(
                        _survey_evaluation_error(
                            primary_stage="evaluation_worker_process",
                            message=(
                                "Survey production completed, but its evaluation process "
                                f"exited abnormally with code {worker.exitcode}."
                            ),
                        )
                    )
                    _write_json(report_path, report)
            if report is None:
                raise ValueError("Task1 Survey worker produced an invalid report contract")

        self.survey_report_path = report_path
        return report["status"] == "success"

    def _run_final(self) -> bool:
        report_path = self.options.survey_report_path or self.survey_report_path
        if report_path is None:
            raise ValueError("Task1 Final requires --survey-report when Survey was not run in this process")
        report_path = report_path.resolve()
        survey_report, candidates = load_survey_candidates(report_path)
        final_config = load_final_config(
            self.options.final_config,
            repo_root=self.repo_root,
        )

        planner_artifact = survey_report.get("planner_artifact")
        if not isinstance(planner_artifact, str) or not planner_artifact:
            raise ValueError("Survey report planner_artifact must be a non-empty string")
        route_path = Path(planner_artifact)
        if not route_path.is_absolute():
            route_path = report_path.parent / route_path
        route = load_survey_route_plan(route_path, repo_root=self.repo_root)
        if not route.segments or not route.segments[-1].trajectory:
            raise ValueError("Survey route has no terminal joint state for Task1 Final")
        start_joint_positions = route.segments[-1].trajectory[-1]

        layout_value = survey_report.get("layout_path")
        if not isinstance(layout_value, str) or not layout_value:
            raise ValueError("Survey report layout_path must be a non-empty string")
        layout_path = Path(layout_value)
        if not layout_path.is_absolute():
            layout_path = self.repo_root / layout_path
        if not layout_path.is_file():
            raise FileNotFoundError(f"Task1 Final layout does not exist: {layout_path}")

        final_dir = self.run_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        final_report_path = final_dir / "final_report.json"
        worker = multiprocessing.get_context("spawn").Process(
            target=_run_final_worker,
            kwargs={
                "repo_root": self.repo_root,
                "final_config": final_config,
                "layout_path": layout_path.resolve(),
                "final_dir": final_dir,
                "candidates": candidates,
                "start_joint_positions": start_joint_positions,
                "source_survey_report": report_path,
            },
        )
        worker.start()
        worker.join()
        if worker.exitcode != 0:
            persisted_report = _load_valid_final_report(final_report_path)
            if persisted_report is not None:
                evaluation = _final_evaluation_error(
                    primary_stage="evaluation_worker_process",
                    message=(
                        "Final production completed, but the evaluation worker "
                        f"exited abnormally with code {worker.exitcode}."
                    ),
                )
                _attach_final_evaluation(final_dir, persisted_report, evaluation)
                _write_json(final_report_path, persisted_report)
                return persisted_report["status"] == "success"
            report = make_final_failure_report(
                failure_stage="final_worker_process",
                message=f"Final worker exited abnormally with code {worker.exitcode}.",
                source_survey_report=report_path,
                final_config_path=final_config.config_path,
                expected_object_count=final_config.evaluation.expected_object_count,
                candidate_count=len(candidates),
            )
            report["simulation"] = _final_simulation_metadata(final_config)
            _write_json(final_report_path, report)
            return False
        if not final_report_path.is_file():
            raise RuntimeError("Task1 Final worker did not produce final_report.json")
        report = _load_valid_final_report(final_report_path)
        if report is None:
            raise ValueError("Task1 Final worker produced an invalid report contract")
        return report["status"] == "success"

    def _make_run_dir(self) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = f"_seed{self.options.seed}" if self.options.seed is not None else "_layout"
        return self.options.output_dir / f"{timestamp}{suffix}"
