from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from task1.layout import DEFAULT_OBJECT_COUNT, make_random_target_object_pose_payload
from task1.simulation.mujoco import MujocoSurveySimulation, SimulationConfig
from task1.survey.manifest import localize_capture_manifest
from task1.survey.route import DEFAULT_SURVEY_ROUTE_PLAN_PATH, load_survey_route_plan
from task1.survey.detection import YoloSurveyDetector
from task1.survey.detection_config import DEFAULT_CONFIG_PATH, load_survey_detection_config


@dataclass(frozen=True)
class PipelineOptions:
    repo_root: Path
    output_dir: Path
    seed: int | None = None
    step: str | None = None
    layout_path: Path | None = None
    capture_manifest: Path | None = None
    survey_config: Path = DEFAULT_CONFIG_PATH
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
class _Stage:
    name: str
    run: Callable[[], bool]


class Task1Pipeline:
    """Task1 stage orchestrator. Add later stages to `_stages` in dependency order."""

    def __init__(self, options: PipelineOptions) -> None:
        self.options = options
        self.repo_root = options.repo_root.resolve()
        self.run_dir = self._make_run_dir()
        self.layout_path: Path | None = None
        self._stages = (
            _Stage("layout", self._run_layout),
            _Stage("survey", self._run_survey),
        )

    @property
    def stage_names(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self._stages)

    def run(self) -> PipelineResult:
        selected = self._selected_stages()
        completed: list[str] = []
        for stage in selected:
            if not stage.run():
                return PipelineResult("failed", self.run_dir, tuple(completed), stage.name)
            completed.append(stage.name)
        return PipelineResult("success", self.run_dir, tuple(completed))

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

        if self.options.capture_manifest is not None:
            if self.options.retain_survey_depth:
                raise ValueError(
                    "--retain-survey-depth applies only to generated MuJoCo depth; "
                    "capture-manifest depth files remain owned by their source manifest"
                )
            survey_config = load_survey_detection_config(
                self.options.survey_config, repo_root=self.repo_root
            )
            report = {
                **localize_capture_manifest(
                    self.options.capture_manifest,
                    detector=YoloSurveyDetector(survey_config),
                    annotated_image_dir=survey_dir / "annotated",
                    policy=survey_config.localization,
                ),
                "schema": "task1_survey_report",
                "stage": "survey",
                "planner_artifact": planner_artifact,
            }
        else:
            survey_config = load_survey_detection_config(
                self.options.survey_config, repo_root=self.repo_root
            )
            simulation = MujocoSurveySimulation(
                SimulationConfig(
                    repo_root=self.repo_root,
                    candidate_position_tolerance_m=(
                        survey_config.evaluation.candidate.position_tolerance_m
                    ),
                    retain_depth_artifacts=self.options.retain_survey_depth,
                ),
                layout_path=self.layout_path,
                output_dir=survey_dir,
                detector=YoloSurveyDetector(survey_config),
                yolo_evaluation_policy=survey_config.evaluation.yolo,
            )
            report = simulation.run(
                route,
                planner_artifact=planner_artifact,
                policy=survey_config.localization,
            )

        report_path = survey_dir / "survey_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report["status"] == "success"

    def _make_run_dir(self) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = f"_seed{self.options.seed}" if self.options.seed is not None else "_layout"
        return self.options.output_dir / f"{timestamp}{suffix}"
