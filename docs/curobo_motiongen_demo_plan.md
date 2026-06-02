# cuRobo MotionGen Demo Adapter Plan

Stage 4.3 adds a minimal demo adapter entry point for future cuRobo MotionGen
validation. It does not implement real CUDA motion planning in the current
Windows development environment.

## Demo Inputs

The demo reads `configs/curobo/minimal_motiongen_demo.json`.

Required input shape:

- `robot_config_path`: path to a validated cuRobo robot YAML/XRDF config.
- `world_config_path`: optional path to a cuRobo world config.
- `start_joint_state`: ordered joint positions in radians.
- `goal_pose`: cuRobo-style target pose `[x, y, z, qw, qx, qy, qz]`.
- `ee_link`: end-effector link name.
- `base_link`: robot base link name.
- `joint_names`: ordered names matching `start_joint_state`.

The placeholder config in this repository is only a schema fixture. It is not a
valid production robot config.

## Demo Outputs

The script writes:

```text
outputs/reports/curobo_motiongen_demo_report.json
```

Report fields:

- `success`
- `message`
- `planning_time_s`
- `trajectory_available`
- `trajectory_path`
- `config_path`
- `robot_config_path`
- `world_config_path`

In the current no-cuRobo environment, `success` is expected to be `false` with a
clear fallback message.

## PlanningRequest Mapping

The demo adapter maps config fields to this project's internal request shape:

- `start_joint_state` + `joint_names` -> `RobotState`
- `goal_pose` translation -> `GraspTarget.pose.position`
- `goal_pose` quaternion `[qw, qx, qy, qz]` -> `Pose3D.orientation_xyzw`
- `goal_pose` -> `GraspTarget.T_world_pregrasp`
- a synthetic `ObjectPose` is created at the goal translation so the existing
  collision scene builder and planner contract can be reused

This is intentionally minimal. Real Stage 4.4+ integration should map actual
Stage 3 YOLO/BODex inputs and a real collision scene into cuRobo world config.

## cuRobo Trajectory Mapping

Future cuRobo output should map back to `PlannedTrajectory` using the existing
conversion helper:

```text
joint_names + joint position waypoints + interpolation_dt
-> PlannedTrajectory
```

That format is already consumable by the MuJoCo executor.

## Fallback Behavior

The script calls `CuroboPlanner`, which lazy-imports cuRobo and torch. If either
package is unavailable, CUDA is unavailable, or config paths are placeholders,
the script writes a failure report and exits normally.

Expected Windows development behavior:

```text
success=False
message=cuRobo is not installed or not configured. Use MockPlanner or set up Linux CUDA environment.
```

Real MotionGen validation should be done in Ubuntu 22.04 or WSL2 Ubuntu 22.04
with NVIDIA CUDA, a PyTorch CUDA build, and cuRobo installed from the official
NVlabs/curobo repository.
