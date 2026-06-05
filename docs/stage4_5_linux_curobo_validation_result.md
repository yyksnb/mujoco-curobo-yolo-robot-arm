# Stage 4.5 Linux cuRobo Validation Result

Stage 4.5 validates the real Ubuntu/Linux CUDA environment for future cuRobo
MotionGen integration. It does not add real robot planning yet.

## System

- OS: Ubuntu 22.04.5 LTS on physical Linux, not WSL2.
- GPU: NVIDIA GeForce RTX 4090.
- Python environment: `/home/yyk/projects/.venvs/curobo310-stage45`.
- Python version: 3.10.12.
- PyTorch version: 2.11.0+cu128.
- PyTorch CUDA runtime: 12.8.
- CUDA available from PyTorch: true.
- CUDA tensor allocation: true.
- cuRobo version: 0.8.0.post1.dev33.
- cuRobo import path: `/home/yyk/projects/curobo/curobo/__init__.py`.

## Environment Check

`python scripts/check_curobo_environment.py` reports:

```text
torch_installed: true
torch_version: 2.11.0+cu128
cuda_available: true
cuda_device_count: 1
cuda_device_name: NVIDIA GeForce RTX 4090
curobo_installed: true
curobo_import_path: /home/yyk/projects/curobo/curobo/__init__.py
can_allocate_cuda_tensor: true
recommended_next_step: Environment looks ready for cuRobo smoke tests with a real robot config.
```

The saved report path is:

```text
outputs/reports/curobo_environment_report.json
```

## cuRobo Script Results

`python scripts/run_curobo_planner.py` currently exits cleanly with:

```text
success=False
message=cuRobo robot_config_path does not exist: configs/curobo/replace_with_robot_config.yml
```

`python scripts/run_curobo_motiongen_demo.py` currently exits cleanly with:

```text
success=False
message=cuRobo robot_config_path does not exist: configs/curobo/replace_with_robot_config.yml
```

This is expected for the current repository state. The validation has moved
past the environment-readiness layer: Python, PyTorch CUDA, CUDA tensor
allocation, and cuRobo import all work. The current blocker is now real robot
configuration and world configuration.

## Next Step

Stage 4.6 should prepare real cuRobo planning inputs:

- validated `robot_config_path` for the target arm,
- validated `world_config_path` or generated world config,
- joint order matching the robot config,
- `ee_link` and `base_link` names,
- collision spheres and self-collision settings,
- frame and pose convention checks between YOLO, BODex, cuRobo, and MuJoCo.
