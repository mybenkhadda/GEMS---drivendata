"""Colab / local runtime detection, conservative resource configuration and per-stage resource tracking.

Nothing here changes a scientific parameter: ``configure_runtime`` only adapts batch sizes, worker counts and
chunk sizes to the hardware, and every adjustment is recorded (``adjustments``) so it is visible in the reports.
"""
from __future__ import annotations

import gc
import os
import platform
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

GIB = 1024 ** 3

# Conservative defaults when the hardware is unknown (also the CPU-only values)
SAFE = {"batch_size": 4, "workers": 0, "aggregate_chunk_cells": 32, "tile_px": 256, "max_ram_gb": 10.0}
GPU_PROFILES = {  # name substring -> (batch_size, amp)
    "A100": (32, True), "H100": (32, True), "L4": (16, True), "T4": (8, True), "V100": (16, True), "P100": (8, False),
}


def _ram():
    try:
        import psutil

        vm = psutil.virtual_memory()
        return vm.total / GIB, vm.available / GIB
    except ImportError:
        try:
            pages, size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
            return pages * size / GIB, None
        except (ValueError, OSError, AttributeError):
            return None, None


def detect_runtime(drive_root: str | None = None, local_root: str | None = None) -> dict:
    """Hardware / environment facts (no side effects). Unknown values are None, never guessed."""
    import sys

    info = {"python": platform.python_version(), "platform": platform.platform(),
            "in_colab": "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules,
            "cpu_count": os.cpu_count()}
    total, avail = _ram()
    info.update(ram_total_gb=total, ram_available_gb=avail)
    info["gpu"] = False
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info.update(gpu=True, gpu_name=p.name, gpu_memory_gb=p.total_memory / GIB, cuda_version=torch.version.cuda,
                        gpu_count=torch.cuda.device_count())
    except ImportError:
        info["torch"] = None
    for key, root in (("local", local_root or "/content"), ("drive", drive_root)):
        if root and Path(root).exists():
            du = shutil.disk_usage(root)
            info[f"{key}_disk_free_gb"] = du.free / GIB
    info["drive_mounted"] = bool(drive_root and Path(drive_root).is_dir())
    return info


def configure_runtime(cfg: dict, info: dict | None = None, use_gpu: bool = True, mixed_precision: bool = True
                      ) -> tuple[dict, list[str]]:
    """Return (cfg, adjustments): batch size / AMP / workers / chunk sizes adapted to the detected hardware.

    Scientific parameters (scales, targets, splits, thresholds, epochs) are never modified.
    """
    import copy

    info = info or detect_runtime(cfg.get("data_root"))
    c = copy.deepcopy(cfg)
    adj = []

    def setv(path, value, why):
        d = c
        for k in path[:-1]:
            d = d[k]
        if d.get(path[-1]) != value:
            adj.append(f"{'.'.join(path)}: {d.get(path[-1])} -> {value} ({why})")
            d[path[-1]] = value

    if info.get("gpu") and use_gpu:
        name = info.get("gpu_name", "")
        bs, amp = next((v for k, v in GPU_PROFILES.items() if k in name), (SAFE["batch_size"], True))
        setv(("training", "batch_size"), bs, f"GPU {name}")
        setv(("training", "amp"), bool(amp and mixed_precision), f"GPU {name}")
        setv(("hardware", "device"), "cuda", "GPU available")
    else:
        setv(("training", "batch_size"), SAFE["batch_size"], "no GPU (or use_gpu=False): conservative CPU default")
        setv(("training", "amp"), False, "AMP needs CUDA")
        setv(("hardware", "device"), "cpu", "no GPU")
    ram = info.get("ram_total_gb")
    if ram is None:
        setv(("dem", "aggregate_chunk_cells"), SAFE["aggregate_chunk_cells"], "unknown RAM: conservative chunk")
        setv(("hardware", "max_ram_gb"), SAFE["max_ram_gb"], "unknown RAM")
    else:
        setv(("hardware", "max_ram_gb"), round(0.75 * ram, 1), f"75 % of {ram:.1f} GiB")
        if ram < 16:
            setv(("dem", "aggregate_chunk_cells"), min(int(c["dem"]["aggregate_chunk_cells"]), 48),
                 f"{ram:.0f} GiB RAM: smaller DEM chunks")
    setv(("hardware", "workers"), min(2, max((info.get("cpu_count") or 1) - 1, 0)), "bounded worker pool")
    return c, adj


def estimate_stage_memory(stage: str, metadata: dict) -> dict:
    """Rough peak-RAM ESTIMATE (GiB) for a stage from grid / chunk / batch metadata. Clearly an estimate."""
    h, w = metadata.get("grid_shape", (3730, 3292))
    px = h * w
    f32 = 4 * px / GIB
    if stage == "features":
        est = 18 * 2 * f32                                    # ~15-18 float64 full-grid arrays per (field, scale)
    elif stage == "dem_aggregate":
        cells = int(metadata.get("chunk_cells", 64)) + 2 * int(metadata.get("halo_cells", 10))
        f = int(round(100 / float(metadata.get("res_m", 2.0))))
        est = 30 * 4 * (cells * f) ** 2 / GIB + 30 * f32       # ~30 work layers per chunk + output grids
    elif stage == "eda":
        est = 8 * f32
    elif stage in ("train_pixel", "train"):
        n, k = int(metadata.get("n_samples", 400_000)), int(metadata.get("n_features", 300))
        est = 3 * 4 * n * k / GIB + 4 * f32
    elif stage == "train_deep":
        bs, p, c = int(metadata.get("batch_size", 8)), int(metadata.get("patch_px", 128)), int(metadata.get("channels", 16))
        est = int(metadata.get("channels", 16)) * f32 + 50 * bs * c * p * p * 4 / GIB
    elif stage == "infer":
        est = 12 * f32
    else:
        est = 6 * f32
    return {"stage": stage, "estimated_peak_ram_gb": round(est, 2), "basis": metadata, "note": "ESTIMATE"}


def cleanup_runtime() -> dict:
    """Garbage-collect and release cached GPU memory."""
    freed = gc.collect()
    out = {"gc_objects": freed}
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            out["gpu_allocated_gb"] = torch.cuda.memory_allocated() / GIB
    except ImportError:
        pass
    return out


class StageTracker:
    """Wall time, peak process RSS (sampled) and peak GPU memory per stage; ``records`` -> report."""

    def __init__(self):
        self.records: list[dict] = []

    @contextmanager
    def track(self, stage: str, log=print):
        import threading

        peak = {"rss": 0.0}
        stop = threading.Event()
        try:
            import psutil

            proc = psutil.Process()

            def sample():
                while not stop.is_set():
                    peak["rss"] = max(peak["rss"], proc.memory_info().rss / GIB)
                    stop.wait(1.0)

            th = threading.Thread(target=sample, daemon=True)
            th.start()
        except ImportError:
            th = None
        gpu = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                gpu = True
        except ImportError:
            pass
        t0 = time.time()
        rec = {"stage": stage, "status": "RUNNING", "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            yield rec
            rec["status"] = rec.get("status_override", "COMPLETE")
        except Exception as e:
            rec["status"] = "FAILED"
            rec["error"] = f"{type(e).__name__}: {e}"[:500]
            raise
        finally:
            stop.set()
            if th is not None:
                th.join(timeout=2)
            rec["seconds"] = round(time.time() - t0, 1)
            rec["peak_ram_gb"] = round(peak["rss"], 2) if peak["rss"] else None
            if gpu:
                import torch

                rec["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / GIB, 2)
            self.records.append(rec)
            log(f"[{stage}] {rec['status']} in {rec['seconds']:.0f} s | peak RAM {rec['peak_ram_gb']} GiB"
                + (f" | peak GPU {rec.get('peak_gpu_gb')} GiB" if gpu else ""))
            cleanup_runtime()


def save_runtime_report(path, info: dict | None = None, adjustments=None, tracker: StageTracker | None = None) -> Path:
    import json

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {"detected": info or detect_runtime(), "adjustments": list(adjustments or []),
           "stages": tracker.records if tracker else [], "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    tmp = p.with_name(p.name + ".partial")
    tmp.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    tmp.replace(p)
    return p
