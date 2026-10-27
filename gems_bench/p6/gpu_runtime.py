"""P6 full GPU runtime: CUDA detection, explicit GPU requirement, determinism, dependency record.

    info = detect()                          # torch / CUDA / device / memory facts (never raises)
    dev = require_gpu(info, cfg["gpu"])      # GPURequired when CUDA is absent and the run requires it (no CPU fallback)
    seed_everything(2026, deterministic=True)
    deps = dependency_record()               # versions of the numerical stack (every manifest records it)

Precision: CNN inference uses the precision declared by each checkpoint spec (``p3.cnn3._amp_dtype``: bf16 when the
device supports it, else fp16, only on CUDA); logits are cast to float32 before the sigmoid
(``inference.full_raster.predict_heads``). Nothing here changes a model's numerics.
"""
from __future__ import annotations

import os
import platform
import random
import time

import numpy as np

PACKAGES = ("numpy", "scipy", "pandas", "pyarrow", "rasterio", "lightgbm", "scikit-learn", "scikit-image",
            "matplotlib", "torch", "psutil", "pyyaml")


class GPURequired(RuntimeError):
    """A GPU stage was requested without a usable CUDA device: it stops instead of running on the CPU."""


def detect() -> dict:
    """Facts about the accelerator (no side effect besides importing torch)."""
    out = {"torch": None, "cuda_available": False, "device": "cpu", "name": None, "capability": None,
           "total_gb": None, "free_gb": None, "bf16": False, "cudnn": None, "cuda_version": None, "n_devices": 0}
    try:
        import torch
    except ImportError as e:
        out["error"] = f"torch not importable: {e}"
        return out
    out["torch"], out["cuda_version"] = torch.__version__, torch.version.cuda
    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(i)
        out.update(cuda_available=True, device=f"cuda:{i}", name=torch.cuda.get_device_name(i),
                   capability=".".join(map(str, torch.cuda.get_device_capability(i))),
                   total_gb=round(total / 2**30, 2), free_gb=round(free / 2**30, 2),
                   bf16=bool(torch.cuda.is_bf16_supported()), cudnn=torch.backends.cudnn.version(),
                   n_devices=torch.cuda.device_count())
    return out


def gpu_decision(info: dict, min_free_gb: float = 0.0) -> tuple[bool, str]:
    """(GPU usable?, reason), pure."""
    if not info.get("cuda_available"):
        return False, "no CUDA device" + (f" ({info['error']})" if info.get("error") else "")
    free = info.get("free_gb")
    if free is not None and float(free) < float(min_free_gb):
        return False, f"free GPU memory {free} GB < {float(min_free_gb):g} GB"
    return True, f"{info.get('name')} ({free} of {info.get('total_gb')} GB free)"


def require_gpu(info: dict, gpu_cfg: dict) -> str | None:
    """CUDA device string, or None when the run does not require a GPU and none is usable. Raises ``GPURequired``
    when ``gpu_cfg.required`` is set: expensive models are never silently moved to the CPU."""
    ok, why = gpu_decision(info, float(gpu_cfg.get("min_free_gb", 0.0)))
    if ok:
        return str(info["device"])
    if gpu_cfg.get("required"):
        raise GPURequired(f"GPU stages required but unavailable: {why}. Colab: Runtime > Change runtime type > GPU, "
                          "or run FULL_GPU_CONFIG['mode'] = 'recovery' (no training, CPU only)")
    return None


def gpu_stages(gpu_cfg: dict, device: str | None) -> dict:
    """Which GPU stages run: each needs its switch AND a device (``require_gpu`` already raised if mandatory)."""
    on = device is not None
    return {k: bool(gpu_cfg.get(k)) and on for k in ("train_seeds", "tta", "fold_average_maps")}


def seed_everything(seed: int, deterministic: bool = True) -> dict:
    """Python / NumPy / torch seeds; deterministic cuDNN + cuBLAS when asked. Returns what was set."""
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed) % 2**32)
    out = {"seed": int(seed), "deterministic": bool(deterministic), "torch": False}
    try:
        import torch
    except ImportError:
        return out
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)     # warn (not fail) on ops without one
    out.update(torch=True, cudnn_deterministic=bool(torch.backends.cudnn.deterministic))
    return out


def dependency_record() -> dict:
    """Installed versions of the numerical stack + interpreter / platform (absent packages recorded as None)."""
    from importlib import metadata

    vers = {}
    for p in PACKAGES:
        try:
            vers[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            vers[p] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": vers,
            "recorded": time.strftime("%Y-%m-%dT%H:%M:%S")}


def memory_report() -> dict:
    """Host RAM and GPU memory now (peak GPU allocation since the last reset)."""
    out = {}
    try:
        import psutil

        vm = psutil.virtual_memory()
        out.update(ram_total_gb=round(vm.total / 2**30, 2), ram_available_gb=round(vm.available / 2**30, 2),
                   rss_gb=round(psutil.Process().memory_info().rss / 2**30, 2))
    except ImportError:
        out["ram"] = "psutil absent"
    try:
        import torch

        if torch.cuda.is_available():
            out.update(gpu_allocated_gb=round(torch.cuda.memory_allocated() / 2**30, 3),
                       gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 3))
    except ImportError:
        pass
    return out


def reset_gpu_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass
