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

Do not install or integrate cuRobo, YOLO, or BODex yet.
