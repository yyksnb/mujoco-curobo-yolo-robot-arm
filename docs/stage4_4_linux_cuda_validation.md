# Stage 4.4 Linux CUDA cuRobo Validation Workflow

Stage 4.4 prepares a repeatable workflow for validating cuRobo in a real Linux
CUDA environment. It does not implement complex MotionGen planning.

## Recommended Machines

Use one of:

- Ubuntu 22.04 on a physical Linux workstation with an NVIDIA GPU.
- WSL2 Ubuntu 22.04 on Windows with NVIDIA GPU support enabled.
- A remote Linux CUDA machine or CI runner with NVIDIA GPU access.

Plain Windows Python remains a graceful-failure development environment. It is
not the baseline for real cuRobo validation.

## Validation Order

1. Confirm NVIDIA driver and GPU visibility:

   ```bash
   nvidia-smi
   ```

2. Create and activate a Linux Python environment.
3. Install a PyTorch CUDA build that matches the driver/CUDA runtime.
4. Install `git-lfs` and clone/install cuRobo from the official NVlabs/curobo
   repository.
5. Run the cuRobo upstream import/tests recommended by official docs.
6. Run this repository's environment check:

   ```bash
   python scripts/check_curobo_environment.py
   python scripts/save_curobo_environment_report.py
   ```

7. Inspect `outputs/reports/curobo_environment_report.json`.
8. Only after the report shows CUDA and cuRobo are available, run the demo:

   ```bash
   python scripts/run_curobo_motiongen_demo.py
   ```

## Environment Check Report

`scripts/check_curobo_environment.py` prints JSON with:

- `python_version`
- `platform`
- `is_windows`
- `is_linux`
- `torch_installed`
- `torch_version`
- `cuda_available`
- `cuda_device_count`
- `cuda_device_name`
- `curobo_installed`
- `curobo_import_path`
- `can_allocate_cuda_tensor`
- `recommended_next_step`

`scripts/save_curobo_environment_report.py` writes the same report to:

```text
outputs/reports/curobo_environment_report.json
```

## Readiness Criteria

The environment is ready for a real cuRobo smoke test when:

- `is_linux` is `true`
- `torch_installed` is `true`
- `cuda_available` is `true`
- `cuda_device_count` is greater than `0`
- `can_allocate_cuda_tensor` is `true`
- `curobo_installed` is `true`
- `curobo_import_path` points to the intended environment

If any of those are false, do not debug MotionGen first. Fix the environment.

## Common Failures and Next Steps

- `is_windows=true`: use Ubuntu 22.04, WSL2 Ubuntu 22.04, or a Linux CUDA
  machine for real cuRobo validation.
- `torch_installed=false`: install PyTorch before cuRobo.
- `cuda_available=false`: the PyTorch build is CPU-only, the NVIDIA driver is
  not visible, or WSL2 GPU passthrough is not configured.
- `can_allocate_cuda_tensor=false`: CUDA may be visible but unusable from the
  active Python process; check driver/runtime mismatch.
- `curobo_installed=false`: install cuRobo from the official NVlabs/curobo
  repository in the same environment.
- `curobo_import_path` points elsewhere: deactivate conflicting environments
  and verify `python`, `pip`, and shell activation.
- MotionGen demo still fails after environment readiness: check
  `robot_config_path`, `world_config_path`, joint names, goal pose convention,
  and robot collision-sphere configuration.

## Windows Expected Behavior

On this project's current Windows development environment, cuRobo scripts are
expected to fail gracefully. That means:

- scripts exit without crashing,
- reports are written under `outputs/reports/`,
- `success` is `false`,
- `recommended_next_step` points to Ubuntu 22.04/WSL2 + NVIDIA CUDA + PyTorch
  CUDA + official cuRobo install.

This is a valid local development result and should not be treated as a failed
real cuRobo validation.
