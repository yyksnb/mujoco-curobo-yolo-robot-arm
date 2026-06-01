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

The requirements file installs `numpy`, `pytest`, and `mujoco`. Stage 1 does
not need MuJoCo at runtime, but Stage 2 uses it for real MJCF execution when it
is available in the environment.

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
python -m pip install -r requirements.txt
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
python -m pip install -r requirements.txt
```

You can also provide explicit paths:

```powershell
python scripts/run_mujoco_executor.py `
  --trajectory outputs/trajectories/object_001_trajectory.json `
  --model examples/mujoco/minimal_six_joint_arm.xml
```

## Run Stage 3 Upstream Interface Pipeline

Stage 3 standardizes the upstream JSON contracts for YOLO and BODex without
integrating either project directly. It still uses the current mock planner and
the MuJoCo executor.

```powershell
python scripts/run_stage3_pipeline.py
```

By default it reads:

```text
examples/yolo_detection.json
examples/bodex_grasp_target.json
examples/mujoco/minimal_six_joint_arm.xml
```

Expected outputs:

```text
outputs/
  trajectories/object_001_trajectory.json
  logs/mujoco_execution_log.csv
  reports/mujoco_execution_report.json
  reports/stage3_evaluation_report.json
```

### YOLO Input Contract

The YOLO-side adapter reads `examples/yolo_detection.json`. The YOLO group
should provide:

```json
{
  "object_id": "object_001",
  "class_name": "mock_cube",
  "confidence": 0.97,
  "bbox_xyxy": [120.0, 80.0, 220.0, 180.0],
  "T_world_object": [
    [1.0, 0.0, 0.0, 0.45],
    [0.0, 1.0, 0.0, 0.05],
    [0.0, 0.0, 1.0, 0.08],
    [0.0, 0.0, 0.0, 1.0]
  ]
}
```

`T_world_object` is required. The current project does not estimate a 6D object
pose from `bbox_xyxy`; if the field is missing, the adapter raises a clear
error.

### BODex Input Contract

The BODex-side adapter reads `examples/bodex_grasp_target.json`. The BODex group
should provide:

```json
{
  "object_id": "object_001",
  "T_world_pregrasp": [
    [1.0, 0.0, 0.0, 0.45],
    [0.0, 1.0, 0.0, 0.05],
    [0.0, 0.0, 1.0, 0.20],
    [0.0, 0.0, 0.0, 1.0]
  ],
  "T_world_grasp": [
    [1.0, 0.0, 0.0, 0.45],
    [0.0, 1.0, 0.0, 0.05],
    [0.0, 0.0, 1.0, 0.10],
    [0.0, 0.0, 0.0, 1.0]
  ],
  "approach_vector_world": [0.0, 0.0, -1.0],
  "hand_joint_goal": [0.02, 0.02, 0.02, 0.02]
}
```

The Stage 3 planner target currently uses `T_world_pregrasp`. `T_world_grasp`
and `hand_joint_goal` are preserved in the internal `GraspTarget` for future
controller and grasp execution stages.

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
- Stage 3 YOLO/BODex JSON adapters
- Stage 3 object_id validation
- Stage 3 pipeline script report generation

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
examples/*.json                   upstream interface examples
tests/                           pytest coverage for Stage 1 contracts
outputs/                         generated artifacts
```

## Troubleshooting

If `python -m pip install -r requirements.txt` cannot install `mujoco`, check
that the Python version and platform are supported by the published MuJoCo wheel.
The executor will still import and run in this case, but
`python scripts/run_mujoco_executor.py` will write a failure report explaining
that MuJoCo is unavailable instead of crashing.
