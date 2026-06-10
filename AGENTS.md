# AGENTS.md

This file intentionally mirrors the project instructions in `AGENT` and records
the current implementation stage.

Stage 1 is implemented:

* Use fake object poses.
* Use fake BODex grasp targets.
* Use a simple mock planner.
* Use a simple mock executor.
* Save trajectory and evaluation results under `outputs/`.

Stage 2 currently adds a minimal MuJoCo trajectory executor skeleton:

* Accept a Stage 1 trajectory JSON file.
* Accept a MuJoCo MJCF/XML model path.
* Fail gracefully when the model path is missing.
* Fail gracefully when the optional `mujoco` package is unavailable.
* Save execution logs under `outputs/logs/`.
* Save execution reports under `outputs/reports/`.

Stage 3 standardizes upstream interfaces without integrating upstream projects:

* Read YOLO-style detections from JSON.
* Require `T_world_object`; do not estimate 6D pose from 2D bbox.
* Read BODex-style grasp targets from JSON.
* Use `T_world_pregrasp` as the current mock planning target.
* Check YOLO and BODex `object_id` values before planning.
* Reuse the mock planner and MuJoCo executor.

Stage 4.1 adds a cuRobo planner skeleton without real CUDA planning:

* Keep `MockPlanner` as the default planner.
* Do not import cuRobo, torch, or CUDA libraries at module import time.
* Return clear `PlanningResult` failures when cuRobo, torch CUDA, or config is
  unavailable.
* Keep cuRobo conversion helpers pure Python and testable.

Later stages validate real cuRobo smoke demos, but YOLO and BODex remain
upstream modules. Do not fake real robot parameters, and keep cuRobo, torch, and
CUDA imports lazy outside explicit validation/demo entry points.

## Codex Collaboration Rules

* Do not modify `main` directly. Create an independent branch for each task,
  such as `feature/xxx`, `fix/xxx`, `docs/xxx`, or `experiment/xxx`.
* Before making changes, read `README.md`, `AGENTS.md`, and `docs/HANDOFF.md`.
* Keep changes scoped to the requested task and preserve the existing project
  structure.
* After changes, run the existing tests when the environment supports them.
* If tests fail, do not claim success. Summarize the failed command, key error,
  likely cause, and next step.
* Do not implement YOLO training in this repository. Treat YOLO as an upstream
  module that provides detection and pose inputs.
* Do not deeply modify BODex in this repository. Treat BODex as an upstream
  module that provides grasp targets.
* Prefer mature open-source libraries and the current project structure over
  rebuilding existing planning, simulation, parsing, or robotics logic from
  scratch.
* Do not commit large datasets, model weights, generated `outputs/`,
  `__pycache__/`, or `.pytest_cache/`.
