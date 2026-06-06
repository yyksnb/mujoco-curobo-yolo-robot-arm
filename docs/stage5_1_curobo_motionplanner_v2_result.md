# Stage 5.1 cuRobo MotionPlanner V2 Result

Stage 5.1 validates real cuRobo V2 motion planning on the Linux CUDA machine.
It uses cuRobo's official Franka demo resources only; this is not the final
real robot configuration for the project.

## Environment

- OS: Ubuntu 22.04.5 LTS on physical Linux.
- GPU: NVIDIA GeForce RTX 4090.
- Python environment: `/home/yyk/projects/.venvs/curobo310-stage45`.
- Python version: 3.10.12.
- PyTorch version: 2.11.0+cu128.
- PyTorch CUDA: available.
- cuRobo version: 0.8.0.post1.dev33.

The installed cuRobo version does not expose the old
`curobo.wrap.reacher.motion_gen` API. The working API is:

```text
curobo.motion_planner.MotionPlanner
curobo.motion_planner.MotionPlannerCfg
```

## Linux cuRobo Runtime Dependency

cuRobo imported successfully before Stage 5.1, but real MotionPlanner
construction failed with:

```text
No module named 'cuda.core'
```

The fix was to install the cuRobo V2 CUDA runtime dependency:

```bash
python -m pip install -r requirements-curobo-linux.txt
```

`requirements-curobo-linux.txt` currently contains:

```text
cuda-core[cu12]>=0.7
```

PyTorch CUDA itself should still be installed from the official PyTorch command
for the target platform and CUDA build. This project does not pin PyTorch in
`requirements.txt`.

## Smoke Test

Run:

```bash
python scripts/run_curobo_motiongen_smoke.py
```

Verified result:

```text
success=True
motiongen_api_called=True
failure_category=None
trajectory_available=True
```

Outputs:

```text
outputs/reports/curobo_motiongen_smoke_report.json
outputs/trajectories/curobo_motiongen_smoke_trajectory.json
```

The trajectory uses the official cuRobo Franka demo config and
`collision_table.yml`. It is a smoke-test trajectory, not a final real-arm
trajectory.

## Pick-Lift Demo

Run:

```bash
python scripts/run_curobo_pick_lift_demo.py
```

Verified result:

```text
success=True
failure_category=None
stages=4
```

The four demo planning stages succeeded:

- `start_to_pregrasp`
- `pregrasp_to_grasp`
- `grasp_to_lift`
- `lift_to_retreat`

Each stage calls the real cuRobo V2 `MotionPlanner.plan_pose` path and reports
`trajectory_available=True`.

## Integrated Demo Result

Run:

```bash
python scripts/run_pick_lift_full_demo.py
```

Verified result:

```text
overall_status=success
```

The planning side succeeds with cuRobo V2, and the MuJoCo physical demo still
successfully lifts the toy cube. The next real integration step is to bridge
cuRobo trajectory output into MuJoCo execution with matching real robot and
gripper joint semantics.

## Remaining Boundary

Stage 5.1 proves the real cuRobo API can run in this environment. It does not
prove a final robot-arm deployment model. Stage 6.0 should prepare the real
robot config, world config, joint order, `base_link`, `ee_link`, gripper joints,
collision spheres, and frame conventions before replacing demo resources.
