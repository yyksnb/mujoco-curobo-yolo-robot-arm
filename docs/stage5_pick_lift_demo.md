# Stage 5 Pick-and-Lift Demo

Stage 5 builds two runnable demo lines and an integration scaffold:

- a cuRobo pick-lift planning demo scaffold,
- a MuJoCo physical grasp smoke demo,
- a full pick-lift report that summarizes both.

This stage is a demo closure, not the final real robot integration.

## Stage 5.1 MotionPlanner Smoke Result

On Ubuntu 22.04.5 with RTX 4090, PyTorch 2.11.0+cu128, and cuRobo
0.8.0.post1.dev33, the real cuRobo V2 `MotionPlanner` path has been validated
with the official Franka demo config.

The environment required the cuRobo V2 CUDA runtime dependency:

```bash
python -m pip install "cuda-core[cu12]>=0.7"
```

Without this package, cuRobo imports but MotionPlanner construction can fail
with `No module named 'cuda.core'`.

Run:

```bash
python scripts/run_curobo_motiongen_smoke.py
```

Expected Linux CUDA result:

```text
success=True
motiongen_api_called=True
trajectory_available=True
```

The script writes:

```text
outputs/reports/curobo_motiongen_smoke_report.json
outputs/trajectories/curobo_motiongen_smoke_trajectory.json
```

The smoke trajectory is a demo trajectory for cuRobo's official Franka example
robot. It is not a trajectory for the final real arm.

## cuRobo Planning Demo

`scripts/run_curobo_pick_lift_demo.py` reads
`configs/curobo/example_pick_lift_demo.json` and validates the planning entry
points for a grasp-like sequence:

1. `start_to_pregrasp`
2. `pregrasp_to_grasp`
3. `grasp_to_lift`
4. `lift_to_retreat`

The default config uses cuRobo's local Franka example resources when available:

```text
/home/yyk/projects/curobo/curobo/content/configs/robot/franka.yml
/home/yyk/projects/curobo/curobo/content/configs/scene/collision_table.yml
```

These are demo resources only. They are not the final real robot model for this
project. The Stage 5.1 implementation uses the real cuRobo V2 `MotionPlanner`
API. If the local cuRobo runtime, robot config, scene config, start state, or
goal pose is invalid, the script writes a graceful failure report with a
specific `failure_category`.

## MuJoCo Physical Demo

`scripts/run_mujoco_physical_grasp_demo.py` loads:

```text
examples/mujoco/two_finger_grasp_cube.xml
```

The model is a toy two-finger gripper and cube. It is meant to test MuJoCo
contact, friction, gripper closing, and lifting. It is not a real arm or final
gripper model.

The script writes:

```text
outputs/logs/mujoco_physical_grasp_demo_log.csv
outputs/reports/mujoco_physical_grasp_demo_report.json
```

The report includes whether the simulation ran, whether finger-cube contact was
detected, and how far the cube moved in z.

## Integrated Scaffold

`scripts/run_pick_lift_full_demo.py` runs the cuRobo pick-lift demo and the
MuJoCo physical grasp demo, then writes:

```text
outputs/reports/pick_lift_full_demo_report.json
```

For this stage, the bridge is intentionally loose. The full demo does not force
cuRobo output directly into MuJoCo. It records whether planning and physics are
individually ready. With the Stage 5.1 Linux CUDA environment, the expected
status is:

```text
success
```

If planning fails but MuJoCo physics still succeeds, the status can be:

```text
partial_success_physics_ready_planning_needs_robot_config
```

## Why MJCF Is Not a cuRobo Robot Config

MuJoCo MJCF describes a simulation model: bodies, joints, geoms, actuators,
mass, friction, contacts, and solver properties. cuRobo robot configs require a
planning model: kinematics, joint limits, collision spheres, self-collision
ignore rules, base and end-effector links, and frame conventions.

A MuJoCo MJCF cannot be used directly as a cuRobo robot config. A real bridge
needs a validated URDF/XRDF/YAML planning model and a separate MuJoCo execution
model with matching joint semantics.

## Real Integration Inputs Needed Next

Stage 5 uses a demo robot and toy gripper because the final real robot
description is not validated yet. Stage 5.1 or Stage 6 should prepare:

- real robot arm model,
- URDF, XRDF, and meshes,
- `joint_names` and `joint_order`,
- `base_link` and `ee_link`,
- gripper joints,
- collision spheres,
- world frame alignment,
- BODex `T_world_pregrasp` and `T_world_grasp` mapping,
- MuJoCo physical parameters: mass, friction, contact, actuator gains.

## Commands

```bash
python scripts/find_curobo_example_configs.py
python scripts/run_curobo_motiongen_smoke.py
python scripts/run_curobo_pick_lift_demo.py
python scripts/run_mujoco_physical_grasp_demo.py
python scripts/run_pick_lift_full_demo.py
```
