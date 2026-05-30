# mujoco-curobo-yolo-robot-arm

Codex-assisted robot arm simulation pipeline with clean interfaces for scene
construction, grasp target conversion, motion planning, execution, and
evaluation.

## Current Stage

This repository currently implements Stage 1 and a minimal Stage 2 MuJoCo
executor skeleton.

Stage 1 includes:

- fake object poses instead of YOLO output
- fake BODex grasp targets
- a deterministic mock planner
- a deterministic mock executor
- JSON trajectory and evaluation outputs under `outputs/`

Stage 2 currently includes:

- a MuJoCo trajectory executor interface
- graceful failure when the MJCF model path is missing
- graceful failure when the optional `mujoco` package is not installed
- a minimal MJCF example model for executor flow testing
- CSV execution logs and JSON execution reports under `outputs/`

cuRobo, YOLO, and BODex are not installed or integrated yet. Their interfaces
are intentionally kept separate so later stages can replace the mock
implementations without changing the pipeline contract.

## Setup

Use Python 3.10 or newer.

```powershell
python -m pip install -r requirements.txt
```

No runtime dependencies are required for Stage 1.

MuJoCo is optional for Stage 2:

```powershell
python -m pip install mujoco
```

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

## Run Stage 2 MuJoCo Executor

Generate the Stage 1 trajectory first:

```powershell
python scripts/run_mock_pipeline.py
```

Then run the MuJoCo executor skeleton:

```powershell
python scripts/run_mujoco_executor.py
```

By default it reads:

```text
outputs/trajectories/object_001_trajectory.json
examples/mujoco/minimal_six_joint_arm.xml
```

Expected outputs:

```text
outputs/
  logs/mujoco_execution_log.csv
  reports/mujoco_execution_report.json
```

If `mujoco` is not installed, the script still exits cleanly and writes a
failure report with an installation message:

```powershell
python -m pip install mujoco
```

You can also provide explicit paths:

```powershell
python scripts/run_mujoco_executor.py `
  --trajectory outputs/trajectories/object_001_trajectory.json `
  --model examples/mujoco/minimal_six_joint_arm.xml
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
- MuJoCo executor missing-model behavior
- MuJoCo availability detection
- MuJoCo executor script report generation

## Repository Layout

```text
scripts/                         runnable entry points
src/robot_arm_pipeline/types.py   internal typed data contracts
src/robot_arm_pipeline/perception fake YOLO/BODex adapters for Stage 1
src/robot_arm_pipeline/scene      collision scene construction
src/robot_arm_pipeline/planning   mock planner and cuRobo placeholder
src/robot_arm_pipeline/execution  mock executor and MuJoCo executor skeleton
src/robot_arm_pipeline/evaluation metrics and JSON output helpers
examples/mujoco/                  minimal MJCF models
tests/                           pytest coverage for Stage 1 contracts
outputs/                         generated artifacts
```
