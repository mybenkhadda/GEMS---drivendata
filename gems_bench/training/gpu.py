"""Runtime GPU detection and conservative auto-configuration (Colab GPUs vary: T4, L4, V100, A100, none).

The configuration never changes the *objective*: effective batch size (batch x grad_accum), patch size,
epochs and steps are preserved; only memory / throughput knobs move. bf16 is chosen only on compute
capability >= 8.0 (Ampere+): on a T4 recent PyTorch reports bf16 "supported" through slow emulation.
"""
from __future__ import annotations

import time


def detect() -> dict:
    info = {"cuda": False, "name": None, "vram_gb": 0.0, "capability": None, "count": 0}
    try:
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info.update(cuda=True, name=p.name, vram_gb=round(p.total_memory / 2**30, 1),
                        capability=f"{p.major}.{p.minor}", count=torch.cuda.device_count(), major=p.major)
    except Exception as e:                       # torch missing or broken driver
        info["error"] = str(e)
    return info


def autoconfig(gpu: dict | None = None, *, patch_px: int = 256, effective_batch: int = 16, n_channels: int = 30) -> dict:
    """Spec overrides for this GPU (merge under explicit YAML ``runtime.gpu`` overrides)."""
    g = gpu or detect()
    if not g.get("cuda"):
        return {"amp": False, "data_on_gpu": False, "batch_size": min(effective_batch, 4),
                "grad_accum": max(effective_batch // 4, 1), "infer_batch": 4, "cudnn_benchmark": False}
    vram, major = g["vram_gb"], g.get("major", 7)
    per_sample = (patch_px / 256) ** 2 * (0.35 + 0.004 * n_channels)        # GiB, measured ~0.4 on P4_B_gated
    fit = max(int((vram * 0.6) // per_sample), 2)
    bs = min(effective_batch, 1 << (fit.bit_length() - 1))
    stack_gb = n_channels * 12.3e6 * 2 / 2**30
    return {"amp": True, "amp_dtype": "bf16" if major >= 8 else "fp16", "batch_size": bs,
            "grad_accum": max(effective_batch // bs, 1), "infer_batch": 32 if vram >= 15 else 16,
            "data_on_gpu": stack_gb < 0.35 * vram, "channels_last": True, "cudnn_benchmark": True}


def memory_snapshot() -> dict:
    try:
        import torch

        if torch.cuda.is_available():
            return {"allocated_mb": torch.cuda.memory_allocated() / 2**20,
                    "peak_mb": torch.cuda.max_memory_allocated() / 2**20,
                    "reserved_mb": torch.cuda.memory_reserved() / 2**20, "t": time.time()}
    except Exception:
        pass
    return {}
