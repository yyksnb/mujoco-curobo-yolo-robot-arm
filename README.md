# mujoco-curobo-yolo-robot-arm

Codex-assisted robot arm simulation pipeline with clean interfaces for scene
construction, grasp target conversion, motion planning, execution, and
evaluation.

## Current Stage

This repository currently implements Stage 1 only:

- fake object poses instead of YOLO output
- fake BODex grasp targets
- a deterministic mock planner
- a deterministic mock executor
- JSON trajectory and evaluation outputs under `outputs/`

cuRobo, MuJoCo, YOLO, and BODex are not installed or integrated yet. Their
interfaces are intentionally kept separate so later stages can replace the mock
implementations without changing the pipeline contract.

## Setup

Use Python 3.10 or newer.

```powershell
python -m pip install -r requirements.txt
```

No runtime dependencies are required for Stage 1.

## Run Stage 1 Mock Pipeline

```powershell
python scripts/run_mock_pipeline.py
```

Expected outputs:

```text
outputs/
  trajectories/object_001_trajectory.json
  reports/object_001_evaluation.json
```

To print saved reports:

```powershell
python scripts/evaluate_outputs.py
```

## Test

```powershell
pytest -q
```

The tests cover:

- collision scene construction from object poses
- mock planner trajectory contract
- mock executor result contract
- end-to-end mock pipeline output generation

## Repository Layout

```text
scripts/                         runnable entry points
src/robot_arm_pipeline/types.py   internal typed data contracts
src/robot_arm_pipeline/perception fake YOLO/BODex adapters for Stage 1
src/robot_arm_pipeline/scene      collision scene construction
src/robot_arm_pipeline/planning   mock planner and cuRobo placeholder
src/robot_arm_pipeline/execution  mock executor and MuJoCo placeholder
src/robot_arm_pipeline/evaluation metrics and JSON output helpers
tests/                           pytest coverage for Stage 1 contracts
outputs/                         generated artifacts
```
