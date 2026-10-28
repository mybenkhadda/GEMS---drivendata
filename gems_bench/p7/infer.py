"""Tiled multi-head inference for GeoFaultFormer (held-out masks and the full region).

``predict_box``   one window: overlapping P x P patches (stride ``stride_frac`` x P) with a P/4 margin read around
                  the window, inputs scaled with the MODEL'S OWN unit scalers, outputs blended with the M9 2-D Hann
                  window (patch borders, where zero padding acts, get almost no weight). Window-attention padding /
                  masks are handled inside the network, so any P works. Optional rot180 TTA (same checkpoint) and
                  MC-dropout samples (``seg_std`` = across-sample std of the fault probability: a dispersion, not a
                  calibrated uncertainty).
``predict_mask``  ``predict_box`` on the bounding box of a mask; NaN outside; several models are averaged.
``full_region``   the whole grid in ``tile_px`` tiles with a ``halo_px`` halo, linear ramp over the halo (m9.infer
                  ``_ramp``), every finished tile saved before the next starts (resumable).
``centerline_reconcile``  crest pixels of the MOSAICKED centerline probability (computed once on the global map,
                  so no tile seam can create or break a centerline).
The cube never moves to the GPU: batches of patches are built on the CPU and transferred.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SIGMOID_HEADS = ("seg", "center", "dist", "ep", "jn")


def heads_of(model) -> list[str]:
    out = []
    for k in model.head_keys:
        out += ["ocos", "osin"] if k == "orient" else [k]
    return out


def _enable_dropout(model, on: bool) -> None:
    import torch.nn as nn

    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train(on)


def _forward_maps(model, xb, device, use_amp: bool, amp_dtype) -> dict:
    import torch

    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        out = model(torch.from_numpy(xb).to(device, non_blocking=True))
    res = {}
    for k, v in out.items():
        v = v.float()
        if k == "orient":
            o = torch.tanh(v).cpu().numpy()
            res["ocos"], res["osin"] = o[:, 0], o[:, 1]
        else:
            res[k] = torch.sigmoid(v).cpu().numpy()[:, 0]
    return res


def predict_box(model, cube, extra, sc, esc, box, *, P: int, stride: int, device, amp: bool = True, amp_dtype=None,
                batch: int = 16, rot180: bool = False, mc_samples: int = 0) -> dict:
    """{head: (r1 - r0, c1 - c0) float32} for ``box`` = (r0, r1, c0, c1) of the full grid."""
    from ..m9.infer import _hann

    r0, r1, c0, c1 = box
    C, H, W = cube.shape
    E = 0 if extra is None else extra.shape[0]
    m = P // 4
    R0, R1, C0, C1 = r0 - m, r1 + m, c0 - m, c1 + m
    R1, C1 = max(R1, R0 + P), max(C1, C0 + P)
    ys = list(range(R0, max(R1 - P, R0) + 1, stride))
    xs = list(range(C0, max(C1 - P, C0) + 1, stride))
    if ys[-1] + P < R1:
        ys.append(R1 - P)
    if xs[-1] + P < C1:
        xs.append(C1 - P)
    names = heads_of(model)
    if mc_samples:
        names = names + ["seg_std"]
    acc = {h: np.zeros((R1 - R0, C1 - C0), np.float32) for h in names}
    wsum = np.zeros((R1 - R0, C1 - C0), np.float32)
    win = _hann(P)
    tiles = [(y, x) for y in ys for x in xs]
    model.eval()
    use_amp = bool(amp) and device.type == "cuda" and amp_dtype is not None
    for i in range(0, len(tiles), batch):
        chunk = tiles[i:i + batch]
        xb = np.zeros((len(chunk), C + E, P, P), np.float32)
        for j, (y, x) in enumerate(chunk):
            ya, yb, xa, xb_ = max(y, 0), min(y + P, H), max(x, 0), min(x + P, W)
            if ya < yb and xa < xb_:
                xb[j, :C, ya - y:yb - y, xa - x:xb_ - x] = sc.apply(cube[:, ya:yb, xa:xb_])
                if E:
                    xb[j, C:, ya - y:yb - y, xa - x:xb_ - x] = esc.apply(extra[:, ya:yb, xa:xb_])
        if mc_samples:
            _enable_dropout(model, True)
            samp = [_forward_maps(model, xb, device, use_amp, amp_dtype) for _ in range(int(mc_samples))]
            _enable_dropout(model, False)
            res = {k: np.mean([s[k] for s in samp], 0) for k in samp[0]}
            res["seg_std"] = np.std([s["seg"] for s in samp], 0)
        else:
            res = _forward_maps(model, xb, device, use_amp, amp_dtype)
        if rot180:
            rr = _forward_maps(model, np.ascontiguousarray(xb[:, :, ::-1, ::-1]), device, use_amp, amp_dtype)
            for k, v in rr.items():                         # rot180 leaves (cos 2t, sin 2t) unchanged
                res[k] = 0.5 * (res[k] + v[:, ::-1, ::-1])
        for j, (y, x) in enumerate(chunk):
            s = (slice(y - R0, y - R0 + P), slice(x - C0, x - C0 + P))
            for k in names:
                acc[k][s] += res[k][j] * win
            wsum[s] += win
    ws = np.maximum(wsum, 1e-6)
    return {h: (a / ws)[m:m + (r1 - r0), m:m + (c1 - c0)] for h, a in acc.items()}


def predict_mask(models: list, cube, extra, mask: np.ndarray, **kw) -> dict:
    """Average of ``models`` = [(model, scaler, extra scaler)] on the bounding box of ``mask``; NaN outside."""
    from ..m9.infer import bbox

    if not mask.any():
        return {}
    box = bbox(mask)
    acc = None
    for model, sc, esc in models:
        r = predict_box(model, cube, extra, sc, esc, box, **kw)
        acc = r if acc is None else {h: acc[h] + r[h] for h in acc}
    out = {}
    r0, r1, c0, c1 = box
    sub = mask[r0:r1, c0:c1]
    for h, a in acc.items():
        o = np.full(mask.shape, np.nan, np.float32)
        o[r0:r1, c0:c1][sub] = (a / len(models))[sub]
        out[h] = o
    return out


def full_region(models: list, cube, extra, valid: np.ndarray, tile_dir: Path, *, tile_px: int, halo_px: int,
                log=print, **kw) -> dict:
    """Resumable full-grid inference; returns {head: (H, W) float32}, NaN outside ``valid``."""
    from ..artifacts.atomic import atomic_save
    from ..m9.infer import _ramp

    tile_dir = Path(tile_dir)
    tile_dir.mkdir(parents=True, exist_ok=True)
    H, W = valid.shape
    T, h = int(tile_px), int(halo_px)
    plan = [(i, j) for i in range(0, H, T) for j in range(0, W, T)]
    names = None
    for k, (i, j) in enumerate(plan):
        p = tile_dir / f"{i}_{j}.npz"
        if p.exists():
            continue
        R0, R1, C0, C1 = max(i - h, 0), min(i + T + h, H), max(j - h, 0), min(j + T + h, W)
        if not valid[R0:R1, C0:C1].any():
            atomic_save(p, lambda t: np.savez_compressed(t, empty=np.array(1)))
            continue
        acc = None
        for model, sc, esc in models:
            r = predict_box(model, cube, extra, sc, esc, (R0, R1, C0, C1), **kw)
            acc = r if acc is None else {hh: acc[hh] + r[hh] for hh in acc}
        wt = _ramp(R1 - R0, C1 - C0, h, (R0 > 0, R1 < H, C0 > 0, C1 < W))
        arr = {hh: (a / len(models) * wt).astype(np.float16) for hh, a in acc.items()}
        atomic_save(p, lambda t: np.savez_compressed(t, box=np.array([R0, R1, C0, C1]), w=wt.astype(np.float16), **arr))
        log(f"[p7] tile {k + 1}/{len(plan)} ({i}, {j}) done")
    acc, ws = {}, np.zeros((H, W), np.float32)
    for i, j in plan:
        z = np.load(tile_dir / f"{i}_{j}.npz")
        if "empty" in z:
            continue
        R0, R1, C0, C1 = (int(v) for v in z["box"])
        ws[R0:R1, C0:C1] += z["w"].astype(np.float32)
        names = names or [f for f in z.files if f not in ("box", "w")]
        for hh in names:
            acc.setdefault(hh, np.zeros((H, W), np.float32))[R0:R1, C0:C1] += z[hh].astype(np.float32)
    return {hh: np.where(valid, a / np.maximum(ws, 1e-6), np.nan).astype(np.float32) for hh, a in acc.items()}


def centerline_reconcile(center: np.ndarray, valid: np.ndarray, sigma_px: float = 1.0) -> np.ndarray:
    """Crest pixels of the mosaicked centerline probability (global, seam-free)."""
    from ..gems3.representation import crest_map

    return crest_map(np.nan_to_num(center, nan=0.0), np.asarray(valid, bool), float(sigma_px))
