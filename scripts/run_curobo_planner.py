from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path

add_src_to_path()

from robot_arm_pipeline.perception import load_bodex_grasp_target  # noqa: E402
from robot_arm_pipeline.pipeline import default_robot_state  # noqa: E402
from robot_arm_pipeline.planning.curobo_planner import (  # noqa: E402
    DEFAULT_GRAPH_CONFIG,
    DEFAULT_ROBOT_CONFIG,
    DEFAULT_WORLD_CONFIG,
    CuroboPlanner,
)
from robot_arm_pipeline.types import ObjectPose, PlanningRequest  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the shared cuRobo planner.")
    parser.add_argument("--bodex-grasp", type=Path, default=Path("examples/bodex_grasp_target.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/curobo/example_planner_config.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    config = _load_config(args.config)
    grasp_target = load_bodex_grasp_target(args.bodex_grasp)
    request = PlanningRequest(
        object_pose=ObjectPose(
            object_id=grasp_target.object_id,
            label="bodex_target",
            pose=grasp_target.pose,
            T_world_object=grasp_target.T_world_pregrasp,
        ),
        grasp_target=grasp_target,
        robot_state=default_robot_state(),
    )

    planner = CuroboPlanner(
        repo_root=REPO_ROOT,
        robot_config_path=Path(config.get("robot_config_path", DEFAULT_ROBOT_CONFIG)),
        world_config_path=Path(config.get("world_config_path", DEFAULT_WORLD_CONFIG)),
        graph_config_path=Path(config.get("graph_config_path", DEFAULT_GRAPH_CONFIG)),
    )
    result = planner.plan(request)
    report_path = _save_report(args.output_dir, args.bodex_grasp, args.config, result)
    print(f"success={result.success} message={result.message} report={report_path}")


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_report(output_dir: Path, bodex_path: Path, config_path: Path, result: Any) -> Path:
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "curobo_planner_report.json"
    payload = {
        "success": result.success,
        "bodex_grasp_path": str(bodex_path),
        "config_path": str(config_path),
        "trajectory_available": result.trajectory is not None,
        "message": result.message,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
