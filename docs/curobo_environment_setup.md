# cuRobo Environment Setup

Stage 4.2 prepares environment validation for future real cuRobo planning. It
does not implement MotionGen or MotionPlanner integration.

## Recommended Environment

Use one of:

- Ubuntu 22.04 on a machine with an NVIDIA GPU.
- WSL2 Ubuntu 22.04 on Windows with NVIDIA GPU passthrough enabled.

Do not treat a normal Windows Python environment as the baseline for real cuRobo
validation. cuRobo depends on CUDA, PyTorch CUDA builds, and compiled CUDA
extensions. Windows may work for some pieces, but the official installation path
and lowest-risk debugging path are Linux-based.

## CUDA and NVIDIA Driver Checks

Before installing PyTorch or cuRobo, verify the GPU and driver:

```bash
nvidia-smi
```

Confirm:

- an NVIDIA GPU is visible,
- the driver supports the CUDA version you plan to use,
- WSL2 can see the GPU if running from Windows,
- the GPU has enough VRAM for cuRobo smoke tests.

## PyTorch CUDA Notes

cuRobo is PyTorch-based. Install a PyTorch build that matches the CUDA runtime
you intend to use. A CPU-only PyTorch build is not enough for real cuRobo motion
planning.

After installing PyTorch, verify:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

## git-lfs Requirement

Install `git-lfs` before cloning or using robot assets that may be stored as
large files:

```bash
git lfs install
```

This is especially important when working with robot meshes, URDF-related
assets, and upstream cuRobo examples.

## cuRobo Install Flow

Follow the official NVlabs/curobo repository and latest docs for the exact
version and CUDA extra. The general shape is:

```bash
git clone https://github.com/NVlabs/curobo
cd curobo

# Create a Linux Python environment. The official docs currently recommend uv.
uv venv --python 3.11
source .venv/bin/activate

# Choose the CUDA extra that matches your environment.
uv pip install .[cu12-torch]

python -c "import curobo; print(curobo.__version__)"
pytest --pyargs curobo.tests
```

If PyTorch CUDA is already installed, use the official cuRobo install extra that
does not reinstall PyTorch. Check the latest official docs before pinning the
exact command.

## Why Not Plain Windows Python

This project is developed on Windows and must keep Stage 1/2/3/4.1 usable
without cuRobo. Real cuRobo validation is different:

- it requires CUDA-visible NVIDIA hardware,
- it depends on a CUDA-enabled PyTorch build,
- it may build or load CUDA extensions,
- robot assets and git-lfs are easier to validate in a Linux workflow,
- official cuRobo docs target Ubuntu/WSL2-style environments.

Use Windows for interface development and graceful-failure tests. Use
Ubuntu/WSL2 + CUDA for real planner validation.

## Environment Check Script

Run:

```powershell
python scripts/check_curobo_environment.py
```

The script prints a JSON-style report with:

- Python and platform details,
- whether the OS is Windows or Linux,
- whether PyTorch imports,
- CUDA availability and device details,
- whether a CUDA tensor can be allocated,
- whether cuRobo imports,
- the recommended next step.

On Windows or when CUDA is unavailable, the recommended next step is:

```text
Use Ubuntu 22.04/WSL2 with NVIDIA CUDA, install PyTorch CUDA build, then install cuRobo from the official repository.
```
