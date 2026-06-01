# cuRobo Configuration Placeholder

Stage 4.1 does not include a production robot model. A real cuRobo deployment
must provide validated robot and world configuration files from the target robot
package or cuRobo robot builder.

Expected fields for this repository's planner configuration:

- `robot_config_path`: path to the cuRobo robot YAML/XRDF config.
- `world_config_path`: optional path to a cuRobo world config template.
- `ee_link`: end-effector link name in the robot model.
- `base_link`: base link name in the robot model.
- `joint_names`: ordered joint names matching this project's `RobotState`.
- `use_cuda`: whether real planning requires CUDA.

The placeholder JSON in this directory documents the shape only. It is not a
valid cuRobo robot model.
