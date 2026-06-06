# Handoff

## Project Goal

Build a pick-and-lift pipeline across YOLO/BODex inputs, grasp conversion,
cuRobo planning, MuJoCo execution, and evaluation. Demos are smoke tests only.

## Current Environment

Validated on Ubuntu 22.04.5 LTS physical Linux with RTX 4090. Use Python env
`/home/yyk/projects/.venvs/curobo310-stage45`.

```text
Python 3.10.12
torch 2.11.0+cu128, CUDA available
cuRobo 0.8.0.post1.dev33
cuRobo import path: /home/yyk/projects/curobo/curobo/__init__.py
```

cuRobo uses V2 `MotionPlanner` / `MotionPlannerCfg`, not old
`curobo.wrap.reacher.motion_gen`. Real planning needed `cuda-core[cu12]>=0.7`
in `requirements-curobo-linux.txt`. Install PyTorch CUDA separately from the
official PyTorch command.

## Completed Stages

Stage 1: mock perception/grasp/planner/executor and JSON outputs.
Stage 2: MuJoCo trajectory executor skeleton.
Stage 3: YOLO/BODex JSON contracts; default planner stays mock.
Stage 4: cuRobo docs, lazy-import skeleton, environment checks.
Stage 5: pick-lift scaffold plus MuJoCo toy physical grasp demo.
Stage 5.1: real cuRobo V2 MotionPlanner smoke test with official Franka demo.

## Important Scripts

```bash
/home/yyk/projects/.venvs/curobo310-stage45/bin/python <script>
```

```text
scripts/check_curobo_environment.py
scripts/find_curobo_example_configs.py
scripts/run_curobo_motiongen_smoke.py
scripts/run_curobo_pick_lift_demo.py
scripts/run_mujoco_physical_grasp_demo.py
scripts/run_pick_lift_full_demo.py
scripts/run_stage3_pipeline.py
scripts/run_mujoco_executor.py
```

## Current Verified Results

```text
motiongen_smoke -> success=True, motiongen_api_called=True, trajectory_available=True
pick_lift_demo -> success=True, four stages success=True
mujoco_physical_grasp -> success=True, lifted_distance about 0.114554
full_demo -> overall_status=success
stage3_pipeline -> success=True
mujoco_executor -> success=True
pytest -q -> 28 passed
```

## Known Constraints

Do not break Stage 1/2/3 behavior. Do not remove `MockPlanner`. Keep
torch/cuRobo/CUDA imports lazy. Do not commit `outputs/`, `__pycache__`, or
`.pytest_cache`.

Official Franka files are demo resources. The MuJoCo two-finger model is a toy
demo. MJCF is not a cuRobo robot config; cuRobo needs planning kinematics,
collision spheres, self-collision rules, links, and frames.

## Current Next Step

Stage 6.0 can start after this commit. Replace demo resources with real
robot/world setup: arm model, URDF/XRDF/meshes, joint order, `base_link`,
`ee_link`, gripper joints, collision spheres, self-collision settings, and
aligned frames.

## How To Resume

```bash
cd /home/yyk/projects/mujoco-curobo-yolo-robot-arm
git status
/home/yyk/projects/.venvs/curobo310-stage45/bin/python scripts/run_curobo_motiongen_smoke.py
/home/yyk/projects/.venvs/curobo310-stage45/bin/python scripts/run_pick_lift_full_demo.py
/home/yyk/projects/.venvs/curobo310-stage45/bin/pytest -q
```

If green, proceed to Stage 6.0 without changing the default Stage 1/2/3
pipeline.
