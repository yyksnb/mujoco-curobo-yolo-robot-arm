# Stage 4 cuRobo Plan

Stage 4.0 is documentation only. Do not change planner code in this stage.

## Goal

Add a cuRobo-backed planner that can replace the current mock planner without
breaking Stage 1, Stage 2, or Stage 3.

The cuRobo integration should keep the existing project contract:

```text
CollisionScene + RobotState + GraspTarget -> PlannedTrajectory
```

## Proposed Stage 4.1 Scope

Stage 4.1 should be a skeleton, not full production planning:

1. Keep `MockPlanner` as the default planner.
2. Implement `CuroboPlanner` with lazy imports and clear unavailable errors.
3. Add config files for cuRobo robot/world planning, but do not assume they are
   present in every environment.
4. Add a script that exercises cuRobo only when explicitly requested.
5. Add tests for graceful failure when cuRobo is unavailable.
6. Add optional dependency metadata, not a mandatory dependency.

## Files to Add or Modify in Stage 4.1

### Modify: `src/robot_arm_pipeline/planning/curobo_planner.py`

Replace the current placeholder with a real class shell:

- `__init__(robot_config_path, world_config_path=None, interpolation_dt=0.01)`
- `is_available() -> bool`
- `plan(scene, robot_state, grasp_target) -> PlannedTrajectory`
- private helpers:
  - `_load_curobo_modules`
  - `_build_world_config`
  - `_to_curobo_joint_state`
  - `_to_curobo_goal_pose`
  - `_to_planned_trajectory`

Imports of `curobo` and `torch` must happen inside helper methods or
constructor code paths, not at module import time.

### Add: `src/robot_arm_pipeline/planning/curobo_conversions.py`

Keep conversion logic testable and separate:

- transform matrix -> pose list `[x, y, z, qw, qx, qy, qz]`
- `Pose3D.orientation_xyzw` -> cuRobo wxyz
- `CollisionScene` -> cuRobo world dictionary
- `RobotState` -> ordered joint positions

This file should not import cuRobo unless unavoidable. Pure conversion tests
can run without CUDA.

### Add: `configs/curobo/`

Initial contents:

- `README.md` explaining expected robot config files.
- Example world config only if it does not pretend to be robot-specific.

Do not invent a fake robot config that looks production-ready. cuRobo needs
robot collision spheres and self-collision data, so a real robot config should
come from the cuRobo robot builder or a validated upstream robot package.

### Add or Modify: scripts

Preferred new script:

- `scripts/run_curobo_stage3_pipeline.py`

Behavior:

- read Stage 3 YOLO/BODex examples,
- require explicit `--robot-config`,
- use `CuroboPlanner`,
- save trajectory in existing format,
- optionally call `MujocoExecutor` after planning,
- fail clearly if cuRobo/CUDA/robot config is unavailable.

Avoid changing `scripts/run_stage3_pipeline.py` default behavior in Stage 4.1.
The default should remain mock planner until cuRobo is stable.

### Modify: `pyproject.toml`

Add optional dependency metadata only after deciding the supported install path.
Because latest cuRobo installation is source/extras based, do not blindly add
`curobo` to mandatory requirements.

Possible shape:

```toml
[project.optional-dependencies]
curobo = [
    # Document source install separately if no stable PyPI package is used.
]
```

The actual install command should remain documented in `docs/` and README.

### Modify: `README.md`

Add a short Stage 4 section only after the skeleton exists:

- how to install cuRobo in a CUDA environment,
- how to run the cuRobo script,
- how it fails when cuRobo is unavailable.

### Add Tests

Add tests without requiring CUDA:

- importing `robot_arm_pipeline.planning.curobo_planner` does not require cuRobo,
- `CuroboPlanner.is_available()` returns bool,
- missing cuRobo returns clear error,
- conversion helpers map pose/quaternion/joint order correctly,
- existing Stage 1/2/3 tests still pass.

CUDA integration tests should be opt-in and skipped unless:

- `curobo` imports,
- `torch.cuda.is_available()` is true,
- a robot config path is provided through an environment variable.

## Implementation Order

1. Add conversion helpers and unit tests.
2. Replace `CuroboPlanner` placeholder with lazy-import skeleton and graceful
   unavailable behavior.
3. Add a cuRobo-only script with explicit config arguments.
4. Add documentation for environment setup and robot config requirements.
5. Run existing Stage 1/2/3 commands to confirm no regression.
6. In a Linux CUDA environment, run the official cuRobo install tests before
   enabling real planning tests.

## Done Criteria for Stage 4.1

- `pytest -q` passes without cuRobo installed.
- `python scripts/run_mock_pipeline.py` still works.
- `python scripts/run_stage3_pipeline.py` still uses mock planning by default.
- Importing the package does not require cuRobo, torch, CUDA, or a robot config.
- Running the cuRobo script without cuRobo gives a clear actionable error.
- Running the cuRobo script with cuRobo but without robot config gives a clear
  config error.

