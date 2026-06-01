# cuRobo Integration Notes

Stage: 4.0 research and design only. Do not implement a real cuRobo planner in
this stage.

## Official Sources Checked

- cuRobo latest documentation: https://nvlabs.github.io/curobo/latest/
- Latest installation instructions: https://nvlabs.github.io/curobo/latest/getting-started/installation.html
- Latest motion planning tutorial: https://nvlabs.github.io/curobo/latest/getting-started/motion_planning.html
- Latest robot model tutorial: https://nvlabs.github.io/curobo/latest/getting-started/build_robot_model.html
- Latest runtime configuration reference: https://nvlabs.github.io/curobo/latest/reference/runtime_configuration.html
- NVlabs/curobo GitHub repository: https://github.com/NVlabs/curobo
- Legacy cuRobo Python examples: https://curobo.org/get_started/2a_python_examples.html
- Legacy cuRobo install page: https://curobo.org/get_started/1_install_instructions.html

## Positioning

cuRobo is a CUDA-accelerated robotics library built around PyTorch and custom
CUDA kernels. The official GitHub README describes cuRoboV2 as covering
forward/inverse kinematics, collision checking, trajectory optimization,
geometric planning, and collision-free motion generation.

For this repository, cuRobo should be treated as the planning backend that
eventually replaces `MockPlanner`, not as a perception, grasp synthesis, or
execution backend. YOLO and BODex remain upstream. MuJoCo remains the local
execution/simulation backend.

Relevant cuRobo capabilities for this project:

- IK: turn an end-effector goal pose into feasible joint configurations.
- Collision checking: check robot/world and self-collision against a configured
  world model.
- Trajectory optimization: optimize smooth, feasible joint-space trajectories.
- Motion generation/planning: combine IK, collision checking, graph/geometric
  planning, and trajectory optimization to produce a trajectory.

## API Version Note

The latest GitHub README says cuRoboV2 is a significant rewrite and that the
public API changed from cuRobo v1. Older official docs use `MotionGen`,
`MotionGenConfig`, and `MotionGenPlanConfig`. The latest documentation describes
motion planning around `MotionPlanner` and tutorial commands such as:

```bash
python -m curobo.examples.getting_started.motion_planning
```

Stage 4.1 should first decide whether to target latest cuRoboV2 or pin the
older v1 API. The safer default is to target latest cuRoboV2 for new work, while
keeping this repository's public planner interface independent of cuRobo class
names.

## Recommended Installation Direction

Latest docs recommend an Ubuntu + CUDA + PyTorch environment and installing from
the NVlabs/curobo repository. They recommend `uv` and CUDA-specific extras:

```bash
git clone https://github.com/NVlabs/curobo && cd curobo
uv venv --python 3.11
source .venv/bin/activate

# For CUDA 12.x, fresh environment including PyTorch:
uv pip install .[cu12-torch]

# For CUDA 12.x, when PyTorch is already installed:
uv pip install .[cu12]

python -c "import curobo; print(curobo.__version__)"
pytest --pyargs curobo.tests
```

Legacy docs recommended editable install with `pip install -e .
--no-build-isolation` after installing git-lfs. That remains useful historical
context, but the latest docs should drive Stage 4.1 unless we deliberately pin
to a v1 tag.

## Environment Notes

From official docs and current project constraints:

- OS: latest docs target Ubuntu >= 20.04 and document a guaranteed path on
  Ubuntu 22.04. Legacy docs mention experimental Windows support, but Windows is
  a risk for CUDA extension build/debug.
- GPU: NVIDIA GPU is required. Latest docs require a GPU newer than Turing and
  at least 4 GB VRAM.
- Driver/CUDA: latest docs require an NVIDIA driver that supports at least CUDA
  12 and provide separate install extras for CUDA 12.x and CUDA 13.x.
- Python: latest docs recommend a Python 3.11 virtual environment and say
  Python > 3.13 is not validated. Legacy docs recommended Python 3.10.
- PyTorch: cuRobo is PyTorch-based. Install the CUDA-matching PyTorch stack via
  cuRobo extras or verify the existing PyTorch build before installing.
- git-lfs: legacy install docs require git-lfs before cloning because robot
  assets/configs may depend on large files. Keep git-lfs in the Stage 4.1
  environment checklist.
- Runtime flags: cuRobo has module-level runtime flags such as `cuda_graphs`,
  `torch_compile`, `cache_dir`, and debugging/profiling flags. Stage 4.1 should
  avoid changing these globally unless exposed through explicit config.

## MotionGen / Motion Planning Minimal Integration Shape

The legacy MotionGen example establishes the conceptual flow:

1. Build a world config containing cuboids/meshes.
2. Load a robot config such as `ur5e.yml`.
3. Construct a motion generation/planning object.
4. Warm it up.
5. Convert the current robot state to a cuRobo `JointState`.
6. Convert the goal end-effector pose to a cuRobo `Pose`.
7. Plan a trajectory.
8. Extract an interpolated joint trajectory.

The latest motion planning tutorial describes the same planning concepts with
newer naming: initialize planner, set start `JointState`, set goal `Pose`, plan,
then use the returned optimized/interpolated trajectory.

For this repository, the first usable Stage 4.1 path should be pose-to-pose
planning only:

```text
PlanningRequest
  object_pose
  grasp_target.pose from T_world_pregrasp
  robot_state
  collision scene

-> CuroboPlanner.plan(...)
-> PlannedTrajectory
-> save_trajectory(...)
-> MujocoExecutor.execute(...)
```

Grasp-specific chained planning can wait. Stage 3 already preserves
`T_world_grasp`, `approach_vector_world`, and `hand_joint_goal`, so Stage 4.2 can
extend from pregrasp-only planning to approach/grasp/lift.

## Data Relationships

### Robot config

cuRobo needs a robot configuration that includes:

- kinematic tree from URDF or a generated config,
- base link and end-effector link names,
- joint names and joint limits,
- collision spheres per link,
- self-collision ignore matrix.

The latest robot model tutorial states that a plain URDF is not enough for full
collision-aware planning because cuRobo needs collision spheres and a
self-collision ignore matrix. It can generate YAML/XRDF configs from a URDF.

### World config

The world config should come from this repository's `CollisionScene`. For the
first cuRobo skeleton:

- map `CollisionObject` boxes to cuRobo cuboids,
- use dimensions from `CollisionObject.size_m`,
- use poses from `CollisionObject.pose`,
- keep units in meters,
- use cuRobo pose convention `[x, y, z, qw, qx, qy, qz]` where applicable.

Meshes and depth/ESDF worlds should be deferred.

### Start joint state

`RobotState` should map to cuRobo's joint state:

- `RobotState.joint_names` must exactly match the cuRobo robot config joint
  names or be explicitly reordered.
- `RobotState.joint_positions` should be radians.
- Missing joints or name mismatch should fail with a clear `PlanningResult`
  error, not with a raw cuRobo exception.

### Goal pose

Stage 3 BODex input provides:

- `T_world_pregrasp`
- `T_world_grasp`
- `approach_vector_world`
- `hand_joint_goal`

Current planning should use `T_world_pregrasp` as the goal pose. The transform
must be converted to cuRobo's pose type. Pay attention to quaternion ordering:
the project stores `Pose3D.orientation_xyzw`; cuRobo examples commonly use
`[x, y, z, qw, qx, qy, qz]`. Stage 4.1 needs an explicit conversion helper.

### Trajectory output

cuRobo output should be converted back into the existing `PlannedTrajectory`:

- preserve this repository's `joint_names`,
- one `TrajectoryWaypoint` per interpolated timestep,
- `time_s = index * interpolation_dt`,
- `planner_name = "curobo_planner"`,
- `target_object_id = grasp_target.object_id`.

This keeps Stage 2 MuJoCo execution unchanged because MuJoCo already consumes
the saved Stage 1/3 trajectory JSON format.

## Upgrade Path from MockPlanner to CuroboPlanner

Keep the existing public shape:

```python
planner.plan(scene: CollisionScene, robot_state: RobotState, grasp_target: GraspTarget) -> PlannedTrajectory
```

Stage 4.1 should implement `CuroboPlanner` behind the same method signature:

1. Lazy import cuRobo and torch inside `CuroboPlanner.__init__` or `plan`, not at
   module import time.
2. Accept config paths through constructor arguments:
   - robot config YAML/XRDF,
   - optional world config template,
   - interpolation dt,
   - end-effector frame if not encoded in robot config.
3. Convert `CollisionScene` to cuRobo world config.
4. Convert `RobotState` to cuRobo `JointState`.
5. Convert `GraspTarget.T_world_pregrasp` to cuRobo goal pose.
6. Run the cuRobo planner.
7. Convert result to `PlannedTrajectory`.
8. If cuRobo is unavailable or planning fails, return/raise a project-level
   clear error that scripts can report cleanly.

Do not change `MockPlanner`. Add tests that skip or assert graceful failure when
cuRobo is not installed.

## Keeping Stage 1/2/3 Independent

To avoid breaking existing flows:

- Do not import cuRobo at top level in `robot_arm_pipeline.planning`.
- Keep cuRobo out of `requirements.txt` until the environment strategy is
  decided. Use an optional dependency group later, e.g. `[project.optional-
  dependencies].curobo`.
- Preserve `scripts/run_mock_pipeline.py`, `scripts/run_mujoco_executor.py`, and
  `scripts/run_stage3_pipeline.py`.
- Add a new script for cuRobo, e.g. `scripts/run_curobo_stage3_pipeline.py`, or
  add an explicit `--planner curobo` option only after tests prove the default
  remains mock.
- Tests that require cuRobo/CUDA should be skipped unless cuRobo and CUDA are
  available.

## Risk Register

- Windows: current development is on Windows. cuRobo latest install path targets
  Ubuntu and CUDA-specific wheels/extensions. Use WSL2 or Linux CI for real
  integration unless Windows is explicitly validated.
- CUDA/driver mismatch: install extras must match the CUDA version visible to
  `nvidia-smi`. CUDA graph behavior may need runtime flags during debugging.
- PyTorch mismatch: cuRobo depends on PyTorch and CUDA extension compatibility.
  Existing project deps include MuJoCo/numpy, but no PyTorch.
- numpy version: project currently allows modern numpy. Confirm cuRobo's tested
  stack before pinning.
- Python version: local environment is Python 3.12; latest cuRobo docs recommend
  Python 3.11 and say >3.13 is not validated. Validate Python 3.12 before
  relying on it.
- Robot assets: cuRobo needs a robot YAML/XRDF with collision spheres and a
  self-collision matrix. A MuJoCo MJCF is not sufficient.
- Joint name ordering: this repository's `RobotState` uses simple names
  (`joint_1` ... `joint_6`); a real URDF config will use robot-specific joint
  names. A mapping layer is required.
- Pose convention: this project currently stores orientation as xyzw, while
  cuRobo examples use wxyz in pose lists.
- Collision world dimensions: `CollisionObject.size_m` currently uses a mock
  cube size. Real planning needs accurate object/table dimensions and frame
  alignment.
- Stage 3 input quality: YOLO must provide `T_world_object`; this project still
  does not estimate 6D pose from 2D bbox.
- MotionGen vs MotionPlanner API: old docs and examples refer to MotionGen;
  latest docs use MotionPlanner and note v2 API changes. Pin the API target
  before writing code.

