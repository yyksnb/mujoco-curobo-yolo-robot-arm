# mujoco-curobo-yolo-robot-arm

Codex-assisted robot arm simulation pipeline with clean interfaces for scene
construction, grasp target conversion, motion planning, execution, and
evaluation.

## Quick Navigation / 快速导航

| Topic | Start here |
|---|---|
| New teammate onboarding | [docs/START_HERE.md](docs/START_HERE.md) |
| Team task split | [docs/team_task_split.md](docs/team_task_split.md) |
| Git branch and PR flow | [docs/team_git_flow.md](docs/team_git_flow.md) |
| End-to-end integration workflow | [docs/integration_workflow.md](docs/integration_workflow.md) |
| Data format policy | [docs/interface_contracts/data_format_policy.md](docs/interface_contracts/data_format_policy.md) |
| YOLO detection contract | [docs/interface_contracts/yolo_detection_contract.md](docs/interface_contracts/yolo_detection_contract.md) |
| Robot and scene contract | [docs/interface_contracts/robot_scene_contract.md](docs/interface_contracts/robot_scene_contract.md) |
| Object asset contract | [docs/interface_contracts/object_asset_contract.md](docs/interface_contracts/object_asset_contract.md) |
| Demo recording checklist | [docs/demo_recording_checklist.md](docs/demo_recording_checklist.md) |

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

For Linux CUDA cuRobo validation, first install a PyTorch CUDA build using the
official PyTorch command for your platform and CUDA runtime. Then install this
project's extra cuRobo Linux runtime requirements:

```bash
python -m pip install -r requirements-curobo-linux.txt
```

This currently installs `cuda-core[cu12]>=0.7`, which is required by cuRobo V2
MotionPlanner on the validated Linux CUDA environment.

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

### Convert Raw YOLO Output to Stage 3 Contract

YOLO teammates may keep their own raw output format, but raw bbox output must
not enter the main planning pipeline directly. Convert it to the canonical Stage
3 detection contract first:

```powershell
python scripts/convert_yolo_raw_to_stage3_contract.py `
  --raw examples/yolo_raw_output_sample.json `
  --output outputs/tmp_yolo_detection_converted.json `
  --allow-mock-pose
```

`--allow-mock-pose` inserts a clearly marked mock `T_world_object` for adapter
tests only. Without a real pose, bbox-only YOLO output is not enough for cuRobo
planning. Generated files under `outputs/` are ignored and should not be
committed.

## Run the CuroboPlanner

`robot_arm_pipeline.planning.CuroboPlanner` is the single collision-aware
cuRobo implementation shared by task pipelines. Its generic boundary accepts
base-frame tool poses and ordered `RobotState` values, exposes batched IK and
pose-route planning, and converts the canonical Stage 3 `PlanningRequest` into
a timed `PlannedTrajectory`. Task-specific candidate fields and evaluation
truth stay outside the planner.

```powershell
python scripts/run_curobo_planner.py
```

By default it reads:

```text
examples/bodex_grasp_target.json
configs/curobo/example_planner_config.json
```

The example config intentionally points to placeholder files. With missing
configuration, cuRobo, or CUDA, the command writes an explicit failure report:

```text
outputs/
  reports/curobo_planner_report.json
```

The configured collision world is fixed for one planner instance. Callers must
express targets in the robot config's base frame and provide joint names in the
exact configured order. A future Task2 adapter must define any world-to-base
transform and dynamic collision-world updates in its own formal input contract;
the shared planner does not infer either from task-internal fields.

## Check cuRobo Environment

Stage 4.2 adds an environment check for future real cuRobo validation. It does
not run MotionGen or MotionPlanner.

```powershell
python scripts/check_curobo_environment.py
```

The script prints a JSON-style report with Python/platform details, PyTorch and
CUDA availability, cuRobo import status, CUDA tensor allocation status, and a
recommended next step. On Windows or when CUDA is unavailable, it recommends
using Ubuntu 22.04 or WSL2 Ubuntu 22.04 with NVIDIA CUDA, a PyTorch CUDA build,
and cuRobo installed from the official repository.

See `docs/curobo_environment_setup.md` for the setup checklist.

## Run Stage 4.3 MotionGen Demo Adapter

Stage 4.3 adds a minimal cuRobo MotionGen demo adapter entry point. It fixes the
configuration, input, output, and fallback contract for future Linux CUDA
validation. It still does not run real CUDA MotionGen planning in the current
Windows environment.

```powershell
python scripts/run_curobo_motiongen_demo.py
```

By default it reads:

```text
configs/curobo/minimal_motiongen_demo.json
```

Expected Windows/no-cuRobo behavior:

```text
outputs/
  reports/curobo_motiongen_demo_report.json
```

The report should contain `success: false`, no trajectory, and a clear message
that cuRobo is not installed or configured. In a future Ubuntu 22.04/WSL2 CUDA
environment, this script is intended to become the real MotionGen smoke-test
entry point.

See `docs/curobo_motiongen_demo_plan.md` for the input/output mapping.

## Stage 4.4 Linux CUDA Validation Workflow

Stage 4.4 defines the order for real cuRobo validation on Ubuntu 22.04, WSL2
Ubuntu 22.04, or another Linux CUDA machine. It does not force cuRobo validation
on Windows.

Start with:

```powershell
python scripts/save_curobo_environment_report.py
```

This writes:

```text
outputs/reports/curobo_environment_report.json
```

In a real Linux CUDA environment, confirm the report shows `torch_installed`,
`cuda_available`, `can_allocate_cuda_tensor`, and `curobo_installed` are all
true before running the MotionGen demo adapter. On Windows, graceful failure and
a Linux/WSL2 recommendation are expected.

See `docs/stage4_4_linux_cuda_validation.md` for the full checklist and failure
diagnostics.

## Stage 4.5 Linux cuRobo Environment Validation Result

Stage 4.5 was validated on Ubuntu 22.04.5 LTS with an NVIDIA GeForce RTX 4090
using Python 3.10.12, PyTorch 2.11.0+cu128, and cuRobo
0.8.0.post1.dev33. The environment check reports that PyTorch CUDA is available,
a CUDA tensor can be allocated, and cuRobo imports successfully.

The current cuRobo planner and MotionGen demo scripts now fail at the expected
next boundary:

```text
cuRobo robot_config_path does not exist: configs/curobo/replace_with_robot_config.yml
```

This means the Linux CUDA/cuRobo environment is ready, and the next work is to
prepare a real robot config, world config, joint order, `ee_link`, `base_link`,
and collision spheres. See
`docs/stage4_5_linux_curobo_validation_result.md` for the validation details.

## Stage 5 Pick-and-Lift Demo

Stage 5 adds a demo closure for pick-and-lift validation without changing the
Stage 1/2/3 pipeline behavior.

```powershell
python scripts/find_curobo_example_configs.py
python scripts/run_curobo_motiongen_smoke.py
python scripts/run_curobo_pick_lift_demo.py
python scripts/run_mujoco_physical_grasp_demo.py
python scripts/run_pick_lift_full_demo.py
```

Stage 5.1 validates the real cuRobo V2 `MotionPlanner` API with the official
Franka demo resources. The smoke script writes:

```text
outputs/reports/curobo_motiongen_smoke_report.json
outputs/trajectories/curobo_motiongen_smoke_trajectory.json
```

The cuRobo pick-lift side uses demo configuration under
`configs/curobo/example_pick_lift_demo.json` and local cuRobo example resources.
It can run four demo planning stages with the official Franka config on the
validated Linux CUDA environment. These are still demo trajectories, not final
real-arm trajectories.

Current validated Stage 5.1 result:

```text
run_curobo_motiongen_smoke.py -> success=True, motiongen_api_called=True
run_curobo_pick_lift_demo.py -> success=True, four planning stages succeeded
run_mujoco_physical_grasp_demo.py -> success=True
run_pick_lift_full_demo.py -> overall_status=success
```

The MuJoCo physical demo uses a toy two-finger gripper and cube model:

```text
examples/mujoco/two_finger_grasp_cube.xml
```

This model is a physical grasp smoke demo, not a real robot arm. The integrated
full demo allows partial success, for example when MuJoCo physics is ready but
cuRobo planning still needs a real robot/world config.

See `docs/stage5_pick_lift_demo.md` for details and the real integration inputs
needed next.

## Team Collaboration / 多人协作

`main` is the stable branch. `main` 是稳定分支，应该保持可运行、可复现实验。

Team members should create their own feature branches from the latest `main`.
队友开发时请从最新 `main` 新建自己的分支，例如 `feature/xxx`、`fix/xxx`、
`docs/xxx` 或 `experiment/xxx`。

All changes should be merged through Pull Request. 所有代码、文档、实验结果
都必须通过 PR 合并，不要直接向 `main` 提交。

Before merging, run the relevant tests and ask at least one teammate to review
the PR. 合并前请运行相关测试，并至少让一名成员 review。

If there is a conflict, sync the latest `main` first, then resolve the conflict
on your own branch. 如果出现冲突，先同步最新 `main`，再在自己的分支上解决冲突。

See `CONTRIBUTING.md` for the full workflow.

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
- shared CuroboPlanner pose/IK/trajectory contracts and graceful failure behavior
- cuRobo conversion helper schema validation
- cuRobo environment check script output and recommended next step
- cuRobo MotionGen demo adapter graceful fallback
- cuRobo environment report saving and Linux CUDA validation workflow

## Repository Layout

```text
scripts/                         runnable entry points
src/robot_arm_pipeline/types.py   internal typed data contracts
src/robot_arm_pipeline/perception fake YOLO/BODex adapters for Stage 1
src/robot_arm_pipeline/scene      collision scene construction
src/robot_arm_pipeline/planning   mock planner and shared cuRobo runtime
src/robot_arm_pipeline/execution  mock executor and MuJoCo executor skeleton
src/robot_arm_pipeline/evaluation metrics and JSON output helpers
configs/curobo/                   cuRobo robot, world, and planning config
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
