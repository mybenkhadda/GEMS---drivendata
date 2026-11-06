"""Tiled inference over any rectangle of the padded grid.

Tiles of ``patch_px`` with ``infer_overlap`` (stride a multiple of the pooling factor) are blended with a
separable sin^2 window (floor 1e-3, so every pixel has positive weight). Only the fault head runs. The result is
sigmoid probabilities on the requested rectangle; callers mask it to the footprint.
"""
from __future__ import annotations

import numpy as np
import torch

from .datasets import E6Data


def _starts(lo: int, hi: int, P: int, stride: int, limit: int, pool: int) -> list[int]:
    """Tile starts (multiples of ``pool``) covering [lo, hi) with every tile inside [0, limit)."""
    first = max(min(lo, limit - P), 0)
    out = list(range(first, max(hi - P, first) + 1, stride))
    last = max(min(-(-(hi - P) // pool) * pool, limit - P), 0)
    if out[-1] < last:
        out.append(last)
    return sorted(set(out))


def window_weight(P: int) -> np.ndarray:
    w = np.sin(np.pi * (np.arange(P) + 0.5) / P) ** 2
    return np.maximum(np.outer(w, w), 1e-3).astype(np.float32)


def bbox_of(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    if not len(rows):
        raise ValueError("empty region")
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


@torch.no_grad()
def predict(model_fn, data: E6Data, region: tuple[int, int, int, int], cfg: dict, device, amp_dtype=None) -> np.ndarray:
    """Probabilities on rows r0:r1, cols c0:c1. ``model_fn(batch_tensors) -> logits (B, 1, P, P)``."""
    t = cfg["training"]
    P, pool = int(t["patch_px"]), data.pool
    stride = max(int(round(P * (1 - float(t["infer_overlap"])) / pool)) * pool, pool)
    ext = int(cfg["model"]["context_extent"])
    r0, r1, c0, c1 = region
    r0, c0 = (r0 // pool) * pool, (c0 // pool) * pool
    rs = _starts(r0, r1, P, stride, data.Hp, pool)
    cs = _starts(c0, c1, P, stride, data.Wp, pool)
    R0, R1 = rs[0], rs[-1] + P
    C0, C1 = cs[0], cs[-1] + P
    acc = np.zeros((R1 - R0, C1 - C0), np.float32)
    wsum = np.zeros_like(acc)
    win = window_weight(P)
    pos = [(r, c) for r in rs for c in cs]
    bs = int(t["infer_batch"])
    for i in range(0, len(pos), bs):
        chunk = pos[i:i + bs]
        items = [data.inputs(r, c, P, ext) for r, c in chunk]
        b = {k: torch.from_numpy(np.stack([it[k] for it in items]).astype(np.float32)).to(device) for k in items[0]}
        b["gate_cov"] = b["cov"]
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logit = model_fn(b)
        prob = torch.sigmoid(logit.float()).cpu().numpy()[:, 0]
        for (r, c), p in zip(chunk, prob):
            acc[r - R0:r - R0 + P, c - C0:c - C0 + P] += p * win
            wsum[r - R0:r - R0 + P, c - C0:c - C0 + P] += win
    out = acc / np.maximum(wsum, 1e-12)
    return out[region[0] - R0:region[1] - R0, region[2] - C0:region[3] - C0]


def reference_fn(model):
    return lambda b: model(b["terrain"], b["field"], b["ctx"], aux=False)["logit"]


def system_fn(model):
    return lambda b: model(b["terrain"], b["field"], b["dem"], b["cov"], b["gate_cov"], b["ctx"], aux=False)["logit"]
