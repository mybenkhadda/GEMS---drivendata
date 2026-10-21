"""Full-raster inference for Phase-5 models: all heads, fold averaging, tile-artefact probes.

``predict_heads``             ``phase4_training.predict_tiled4`` generalised to every head (seg, dist, orient)
``full_region_map``           fold-model average over the whole valid footprint (no retraining) - the default
                              submission map; ``full_retrain`` (one model on all labels) is a separate unit
``shifted_tile_consistency``  predicts a window twice with tile grids offset by ``shift_px`` and compares them:
                              large differences reveal tile-edge artefacts (watch long-range / Fourier variants)
"""
from __future__ import annotations

import numpy as np

from ..cnn import _window, bbox


def predict_heads(model, gstack, box, spec, device, gctx=None) -> dict[str, np.ndarray]:
    import torch

    from ..p3 import cnn3
    from ..phase4_training import _forward, context_crop

    patch, stride = spec.patch_px, int(spec.patch_px * spec.infer_stride_frac)
    r0, r1, c0, c1 = box
    C, H, W = gstack.shape
    m = patch // 4
    R0, R1, C0, C1 = r0 - m, r1 + m, c0 - m, c1 + m
    R1, C1 = max(R1, R0 + patch), max(C1, C0 + patch)
    ys = list(range(R0, max(R1 - patch, R0) + 1, stride))
    xs = list(range(C0, max(C1 - patch, C0) + 1, stride))
    if ys[-1] + patch < R1:
        ys.append(R1 - patch)
    if xs[-1] + patch < C1:
        xs.append(C1 - patch)
    win = torch.from_numpy(_window(patch)).to(device)
    acc: dict = {}
    wsum = torch.zeros((R1 - R0, C1 - C0), dtype=torch.float32, device=device)
    amp = cnn3._amp_dtype(spec, device)
    tiles = [(y, x) for y in ys for x in xs]
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(tiles), spec.infer_batch):
            chunk = tiles[i:i + spec.infer_batch]
            xb = torch.zeros((len(chunk), C, patch, patch), dtype=gstack.dtype, device=gstack.device)
            for j, (y, x) in enumerate(chunk):
                ya, yb, xa, xb_ = max(y, 0), min(y + patch, H), max(x, 0), min(x + patch, W)
                if ya < yb and xa < xb_:
                    xb[j, :, ya - y:yb - y, xa - x:xb_ - x] = gstack[:, ya:yb, xa:xb_]
            cb = torch.stack([context_crop(gctx, y, x, patch, spec.context_factor) for y, x in chunk]).to(device).float() \
                if gctx is not None else None
            out = _forward(model, xb.to(device).float(), cb, spec, device, amp)
            heads = {"seg": torch.sigmoid(out["seg"].float())}
            if "dist" in out:
                heads["dist"] = torch.sigmoid(out["dist"].float())
            if "orient" in out:
                o = torch.tanh(out["orient"].float())
                heads["ocos"], heads["osin"] = o[:, :1], o[:, 1:2]
            for name, t in heads.items():
                a = acc.setdefault(name, torch.zeros_like(wsum))
                for j, (y, x) in enumerate(chunk):
                    a[y - R0:y - R0 + patch, x - C0:x - C0 + patch] += t[j, 0] * win
            for y, x in chunk:
                wsum[y - R0:y - R0 + patch, x - C0:x - C0 + patch] += win
    return {k: (v / wsum.clamp_min(1e-6))[m:m + (r1 - r0), m:m + (c1 - c0)].cpu().numpy() for k, v in acc.items()}


def predict_heads_mask(model, gstack, mask: np.ndarray, spec, device, gctx=None) -> dict[str, np.ndarray]:
    out = {}
    if not mask.any():
        return out
    r0, r1, c0, c1 = bbox(mask)
    sub = mask[r0:r1, c0:c1]
    for k, v in predict_heads(model, gstack, (r0, r1, c0, c1), spec, device, gctx).items():
        a = np.full(mask.shape, np.nan, np.float32)
        a[r0:r1, c0:c1][sub] = v[sub]
        out[k] = a
    return out


def full_region_map(models: list, gstack, valid: np.ndarray, device, gctx=None, head: str = "seg") -> np.ndarray:
    """Mean of fold models over the full valid footprint; ``models`` = [(spec, model)]."""
    acc = np.zeros(valid.shape, np.float64)
    for spec, m in models:
        acc += np.nan_to_num(predict_heads_mask(m, gstack, valid, spec, device, gctx)[head])
    out = (acc / max(len(models), 1)).astype(np.float32)
    out[~valid] = np.nan
    return out


def shifted_tile_consistency(model, gstack, box, spec, device, shift_px: int = 64, gctx=None) -> dict:
    r0, r1, c0, c1 = box
    a = predict_heads(model, gstack, box, spec, device, gctx)["seg"]
    b = predict_heads(model, gstack, (r0 + shift_px, r1 + shift_px, c0 + shift_px, c1 + shift_px), spec, device, gctx)["seg"]
    x, y = a[shift_px:, shift_px:], b[:-shift_px, :-shift_px]
    ok = np.isfinite(x) & np.isfinite(y)
    d = np.abs(x[ok] - y[ok])
    return {"shift_px": shift_px, "mean_abs_diff": float(d.mean()) if d.size else np.nan,
            "p99_abs_diff": float(np.quantile(d, 0.99)) if d.size else np.nan,
            "corr": float(np.corrcoef(x[ok], y[ok])[0, 1]) if d.size > 2 else np.nan}
