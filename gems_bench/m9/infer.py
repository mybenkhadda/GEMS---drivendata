"""Sections 08 / 09 / 17: tiled multi-head inference (Hann-blended patches, halo tiles, resumable, CPU / GPU).

``predict_box``     one window: overlapping P x P patches (stride ``stride_frac`` x P) blended with a 2-D Hann
                    window, a P/4 margin read around the window, inputs scaled with the MODEL'S OWN unit scaler,
                    low-resolution context crop, map-context channel + flag. Returns sigmoid seg / ep / jn and the
                    blended (cos 2t, sin 2t) orientation field.
``predict_mask``    ``predict_box`` on the bounding box of a mask (NaN outside).
``full_region``     the whole grid in ``tile_px`` tiles with a ``halo_px`` halo; every tile's contribution is
                    weighted by a linear ramp over its halo (seams blend smoothly) and saved as
                    ``inference/tiles/<run>/<i>_<j>.npz`` before the next tile starts, so an interrupted session
                    resumes at the first missing tile. Several models (fold average) are averaged per tile.
The full cube is never moved to the GPU: each batch of patches is built on the CPU and transferred.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

HEAD_NAMES = ("seg", "ocos", "osin", "ep", "jn")


def device_of(pref: str = "auto"):
    import torch

    if pref == "cpu":
        return torch.device("cpu")
    from ..cnn import device_of as _dev

    return _dev()


def _hann(P: int) -> np.ndarray:
    h = np.hanning(P + 2)[1:-1]
    return (np.outer(h, h) + 1e-3).astype(np.float32)


def _ramp(h: int, w: int, halo: int, edges: tuple[bool, bool, bool, bool]) -> np.ndarray:
    """Tile weight: 1 inside, linear ramp over the halo (no ramp on grid edges)."""
    def prof(n, a, b):
        x = np.ones(n, np.float32)
        if halo > 0:
            r = np.linspace(1.0 / (halo + 1), 1.0, halo, dtype=np.float32)
            if a and n > halo:
                x[:halo] = r
            if b and n > halo:
                x[-halo:] = r[::-1]
        return x
    top, bottom, left, right = edges
    return np.outer(prof(h, top, bottom), prof(w, left, right))


def predict_box(model, cube, pooled, sc, psc, box, *, ctx_map=None, flag: float = 0.0, P: int, f: int, stride: int,
                device, amp: bool = True, batch: int = 16) -> dict:
    """{head: (r1-r0, c1-c0) float32} for ``box`` = (r0, r1, c0, c1) of the full grid."""
    import torch

    from .dataset import context_crop

    r0, r1, c0, c1 = box
    C, H, W = cube.shape
    m = P // 4
    R0, R1, C0, C1 = r0 - m, r1 + m, c0 - m, c1 + m
    R1, C1 = max(R1, R0 + P), max(C1, C0 + P)
    ys = list(range(R0, max(R1 - P, R0) + 1, stride))
    xs = list(range(C0, max(C1 - P, C0) + 1, stride))
    if ys[-1] + P < R1:
        ys.append(R1 - P)
    if xs[-1] + P < C1:
        xs.append(C1 - P)
    acc = {h: np.zeros((R1 - R0, C1 - C0), np.float32) for h in HEAD_NAMES}
    wsum = np.zeros((R1 - R0, C1 - C0), np.float32)
    win = _hann(P)
    tiles = [(y, x) for y in ys for x in xs]
    model.eval()
    use_amp = amp and device.type == "cuda"
    with torch.no_grad():
        for i in range(0, len(tiles), batch):
            chunk = tiles[i:i + batch]
            xb = np.zeros((len(chunk), C, P, P), np.float32)
            mb = np.zeros((len(chunk), 2, P, P), np.float32)
            cb = np.stack([context_crop(pooled, psc, y, x, P, f) for y, x in chunk])
            mb[:, 1] = flag
            for j, (y, x) in enumerate(chunk):
                ya, yb, xa, xb_ = max(y, 0), min(y + P, H), max(x, 0), min(x + P, W)
                if ya < yb and xa < xb_:
                    xb[j, :, ya - y:yb - y, xa - x:xb_ - x] = sc.apply(cube[:, ya:yb, xa:xb_])
                    if ctx_map is not None:
                        mb[j, 0, ya - y:yb - y, xa - x:xb_ - x] = ctx_map[ya:yb, xa:xb_]
            t = lambda a: torch.from_numpy(a).to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(t(xb), t(mb), t(cb))
            seg = torch.sigmoid(out["seg"].float()).cpu().numpy()[:, 0]
            ori = torch.tanh(out["orient"].float()).cpu().numpy()
            ep = torch.sigmoid(out["ep"].float()).cpu().numpy()[:, 0]
            jn = torch.sigmoid(out["jn"].float()).cpu().numpy()[:, 0]
            for j, (y, x) in enumerate(chunk):
                s = (slice(y - R0, y - R0 + P), slice(x - C0, x - C0 + P))
                acc["seg"][s] += seg[j] * win
                acc["ocos"][s] += ori[j, 0] * win
                acc["osin"][s] += ori[j, 1] * win
                acc["ep"][s] += ep[j] * win
                acc["jn"][s] += jn[j] * win
                wsum[s] += win
    ws = np.maximum(wsum, 1e-6)
    return {h: (a / ws)[m:m + (r1 - r0), m:m + (c1 - c0)] for h, a in acc.items()}


def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def predict_mask(models: list, cube, pooled, mask: np.ndarray, *, ctx_map=None, flag: float = 0.0, P: int, f: int,
                 stride: int, device, amp: bool = True, batch: int = 16) -> dict:
    """Average of ``models`` = [(model, scaler, pooled scaler)] on the bounding box of ``mask``; NaN outside."""
    out = {h: np.full(mask.shape, np.nan, np.float32) for h in HEAD_NAMES}
    if not mask.any():
        return out
    box = bbox(mask)
    acc = None
    for model, sc, psc in models:
        r = predict_box(model, cube, pooled, sc, psc, box, ctx_map=ctx_map, flag=flag, P=P, f=f, stride=stride,
                        device=device, amp=amp, batch=batch)
        acc = r if acc is None else {h: acc[h] + r[h] for h in HEAD_NAMES}
    r0, r1, c0, c1 = box
    sub = mask[r0:r1, c0:c1]
    for h in HEAD_NAMES:
        o = out[h][r0:r1, c0:c1]
        o[sub] = (acc[h] / len(models))[sub]
    return out


def full_region(models: list, cube, pooled, valid: np.ndarray, tile_dir: Path, *, ctx_map=None, flag: float = 0.0,
                P: int, f: int, stride: int, tile_px: int, halo_px: int, device, amp: bool = True, batch: int = 16,
                log=print) -> dict:
    """Resumable full-grid inference; returns {head: (H, W) float32} with NaN outside ``valid``."""
    from ..artifacts.atomic import atomic_save

    tile_dir = Path(tile_dir)
    tile_dir.mkdir(parents=True, exist_ok=True)
    H, W = valid.shape
    T, h = int(tile_px), int(halo_px)
    plan = [(i, j) for i in range(0, H, T) for j in range(0, W, T)]
    for k, (i, j) in enumerate(plan):
        p = tile_dir / f"{i}_{j}.npz"
        if p.exists():
            continue
        R0, R1, C0, C1 = max(i - h, 0), min(i + T + h, H), max(j - h, 0), min(j + T + h, W)
        if not valid[R0:R1, C0:C1].any():
            atomic_save(p, lambda t: np.savez_compressed(t, empty=np.array(1)))
            continue
        acc = None
        for model, sc, psc in models:
            r = predict_box(model, cube, pooled, sc, psc, (R0, R1, C0, C1), ctx_map=ctx_map, flag=flag, P=P, f=f,
                            stride=stride, device=device, amp=amp, batch=batch)
            acc = r if acc is None else {hh: acc[hh] + r[hh] for hh in HEAD_NAMES}
        wt = _ramp(R1 - R0, C1 - C0, h, (R0 > 0, R1 < H, C0 > 0, C1 < W))
        arr = {hh: (acc[hh] / len(models) * wt).astype(np.float16) for hh in HEAD_NAMES}
        atomic_save(p, lambda t: np.savez_compressed(t, box=np.array([R0, R1, C0, C1]), w=wt.astype(np.float16), **arr))
        log(f"[m9] tile {k + 1}/{len(plan)} ({i}, {j}) done")
    acc = {hh: np.zeros((H, W), np.float32) for hh in HEAD_NAMES}
    ws = np.zeros((H, W), np.float32)
    for i, j in plan:
        z = np.load(tile_dir / f"{i}_{j}.npz")
        if "empty" in z:
            continue
        R0, R1, C0, C1 = (int(v) for v in z["box"])
        ws[R0:R1, C0:C1] += z["w"].astype(np.float32)
        for hh in HEAD_NAMES:
            acc[hh][R0:R1, C0:C1] += z[hh].astype(np.float32)
    out = {hh: np.where(valid, a / np.maximum(ws, 1e-6), np.nan).astype(np.float32) for hh, a in acc.items()}
    return out


def strike_of(ocos: np.ndarray, osin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(strike deg in [0, 180), orientation confidence = |(c, s)|) of a blended doubled-angle field."""
    from ..geology.orientation import axial_angle

    return axial_angle(np.nan_to_num(ocos), np.nan_to_num(osin)), np.hypot(np.nan_to_num(ocos), np.nan_to_num(osin))
