from __future__ import annotations

from pathlib import Path

from task2.targeting import (
    GraspGeneratorFactoryRegistry,
    TargetingReport,
    TargetingStage,
    build_targeting_stage,
)


class Task2Pipeline:
    """Task2 orchestration boundary.

    Targeting is the only implemented Task2 stage. Planning, execution, and
    verification will be composed here when their formal stage interfaces exist.
    """

    def __init__(self, targeting_stage: TargetingStage) -> None:
        self.targeting_stage = targeting_stage

    def run(
        self,
        manifest_path: Path | str,
        *,
        output_path: Path | str | None = None,
    ) -> TargetingReport:
        return self.targeting_stage.run(manifest_path, output_path=output_path)


def build_task2_pipeline(
    config_path: Path | str,
    *,
    end_effector_id: str | None = None,
    factory_registry: GraspGeneratorFactoryRegistry | None = None,
) -> Task2Pipeline:
    targeting_stage = build_targeting_stage(
        config_path,
        end_effector_id=end_effector_id,
        factory_registry=factory_registry,
    )
    return Task2Pipeline(targeting_stage)
