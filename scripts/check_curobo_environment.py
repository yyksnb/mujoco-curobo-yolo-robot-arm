from __future__ import annotations

import importlib.util
import json
import platform
import sys
from typing import Any


CUDA_SETUP_MESSAGE = (
    "Use Ubuntu 22.04/WSL2 with NVIDIA CUDA, install PyTorch CUDA build, "
    "then install cuRobo from the official repository."
)


def main() -> None:
    report = check_environment()
    print(json.dumps(report, indent=2))


def check_environment() -> dict[str, Any]:
    system = platform.system()
    torch_info = _check_torch()
    curobo_info = _check_curobo()

    cuda_available = bool(torch_info["cuda_available"])
    is_windows = system == "Windows"
    if is_windows or not cuda_available:
        recommended_next_step = CUDA_SETUP_MESSAGE
    elif not curobo_info["curobo_installed"]:
        recommended_next_step = "Install cuRobo from the official NVlabs/curobo repository and rerun this check."
    else:
        recommended_next_step = "Environment looks ready for cuRobo smoke tests with a real robot config."

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "is_windows": is_windows,
        "is_linux": system == "Linux",
        "torch_installed": torch_info["torch_installed"],
        "torch_version": torch_info["torch_version"],
        "cuda_available": cuda_available,
        "cuda_device_count": torch_info["cuda_device_count"],
        "cuda_device_name": torch_info["cuda_device_name"],
        "curobo_installed": curobo_info["curobo_installed"],
        "curobo_import_path": curobo_info["curobo_import_path"],
        "can_allocate_cuda_tensor": torch_info["can_allocate_cuda_tensor"],
        "recommended_next_step": recommended_next_step,
    }


def _check_torch() -> dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {
            "torch_installed": False,
            "torch_version": None,
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_device_name": None,
            "can_allocate_cuda_tensor": False,
        }

    try:
        import torch
    except Exception:
        return {
            "torch_installed": False,
            "torch_version": None,
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_device_name": None,
            "can_allocate_cuda_tensor": False,
        }

    cuda_available = False
    cuda_device_count = 0
    cuda_device_name = None
    can_allocate_cuda_tensor = False

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False

    if cuda_available:
        try:
            cuda_device_count = int(torch.cuda.device_count())
        except Exception:
            cuda_device_count = 0
        try:
            cuda_device_name = str(torch.cuda.get_device_name(0)) if cuda_device_count else None
        except Exception:
            cuda_device_name = None
        try:
            tensor = torch.zeros((1,), device="cuda")
            can_allocate_cuda_tensor = bool(tensor.is_cuda)
        except Exception:
            can_allocate_cuda_tensor = False

    return {
        "torch_installed": True,
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_device_name": cuda_device_name,
        "can_allocate_cuda_tensor": can_allocate_cuda_tensor,
    }


def _check_curobo() -> dict[str, Any]:
    spec = importlib.util.find_spec("curobo")
    if spec is None:
        return {"curobo_installed": False, "curobo_import_path": None}

    try:
        import curobo  # noqa: F401
    except Exception:
        return {"curobo_installed": False, "curobo_import_path": None}

    return {
        "curobo_installed": True,
        "curobo_import_path": spec.origin,
    }


if __name__ == "__main__":
    sys.exit(main())
