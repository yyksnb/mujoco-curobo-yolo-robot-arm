# Stage 5 Pick-and-Lift Demo

Stage 5 builds two runnable demo lines and an integration scaffold:

- a cuRobo pick-lift planning demo scaffold,
- a MuJoCo physical grasp smoke demo,
- a full pick-lift report that summarizes both.

This stage is a demo closure, not the final real robot integration.

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
project. If the current project-side cuRobo adapter cannot yet enter the
official MotionPlanner/MotionGen API, the script writes a graceful failure
report with `failure_category: motiongen_api`.

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
individually ready. A common expected state is:

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
python scripts/run_curobo_pick_lift_demo.py
python scripts/run_mujoco_physical_grasp_demo.py
python scripts/run_pick_lift_full_demo.py
```
