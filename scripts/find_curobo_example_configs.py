from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_CUROBO_ROOT = Path("/home/yyk/projects/curobo")


def main() -> None:
    parser = argparse.ArgumentParser(description="Find local cuRobo demo robot and world configs.")
    parser.add_argument("--curobo-root", type=Path, default=DEFAULT_CUROBO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    report = build_report(args.curobo_root)
    report_path = save_report(args.output_dir, report)
    print(
        f"success={report['success']} robot_configs={len(report['robot_configs'])} "
        f"world_configs={len(report['world_configs'])} report={report_path}"
    )


def build_report(curobo_root: Path) -> dict[str, object]:
    robot_dir = curobo_root / "curobo" / "content" / "configs" / "robot"
    scene_dir = curobo_root / "curobo" / "content" / "configs" / "scene"
    example_dir = curobo_root / "curobo" / "examples"

    robot_configs = _find_files(robot_dir, ("*.yml", "*.yaml", "*.xrdf"))
    world_configs = _find_files(scene_dir, ("*.yml", "*.yaml"))
    urdfs = _find_files(curobo_root / "curobo" / "content" / "assets" / "robot", ("*.urdf",))
    motion_examples = _find_files(example_dir, ("*.py",))

    selected_robot = _prefer(robot_configs, "franka.yml")
    selected_world = _prefer(world_configs, "collision_table.yml") or _prefer(world_configs, "collision_test.yml")

    return {
        "success": curobo_root.exists(),
        "curobo_root": str(curobo_root),
        "is_demo_inventory": True,
        "robot_configs": [str(path) for path in robot_configs],
        "world_configs": [str(path) for path in world_configs],
        "urdfs": [str(path) for path in urdfs],
        "motion_examples": [str(path) for path in motion_examples if "motion" in path.name.lower()],
        "selected_demo_robot_config": str(selected_robot) if selected_robot else None,
        "selected_demo_world_config": str(selected_world) if selected_world else None,
        "notes": (
            "These are local cuRobo example resources for smoke demos only. "
            "They are not the final real robot configuration for this project."
        ),
    }


def save_report(output_dir: Path, report: dict[str, object]) -> Path:
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "curobo_example_configs_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def _find_files(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    if not root.exists():
        return []
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))
    return sorted(path for path in files if path.is_file())


def _prefer(paths: list[Path], name: str) -> Path | None:
    for path in paths:
        if path.name == name:
            return path
    return paths[0] if paths else None


if __name__ == "__main__":
    main()
