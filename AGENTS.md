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

Do not install or integrate cuRobo, YOLO, or BODex yet.
