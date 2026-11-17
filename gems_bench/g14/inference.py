"""Memory-efficient tiled inference on the 100 m grid (Phase 12).

* Local tiles of ``tile_px`` (default 128) on the centre lattice, stride ``tile_px - 2 * halo_px`` rounded down to the
  lattice step, so every tile's regional / global views are exactly aligned (see ``multiscale``).
* Only the tile core (halo removed) is kept, blended with a Hann window over the overlap: no seams.
* At the study-area boundary the coarser views extend beyond the raster with value 0 and validity 0, the same
  convention as training, so the model always knows where context is missing.
* Optional MC-dropout samples (mean and std maps) and rot180 test-time augmentation.
* Output: float32 probability on the full grid, NaN outside the requested mask. Nothing is written unless asked.

UNTESTED: written without execution.
"""
from __future__ import annotations

import time

import numpy as np

from .multiscale import LEVEL_NAMES


def tile_origins(mask: np.ndarray, tile: int, halo: int, step_lattice: int) -> list[tuple[int, int]]:
    """Top-left corners covering every ``mask`` pixel with the tile cores; centres on the lattice."""
    H, W = mask.shape
    core = tile - 2 * halo
    stride = max(step_lattice, (core // step_lattice) * step_lattice)
    rows = list(range(-halo, H, stride))
    cols = list(range(-halo, W, stride))
    out = []
    for r in rows:
        for c in cols:
            cy, cx = r + tile // 2, c + tile // 2
            cy -= cy % step_lattice
            cx -= cx % step_lattice
            t, l = cy - tile // 2, cx - tile // 2
            if mask[max(t, 0):max(min(t + tile, H), 0), max(l, 0):max(min(l + tile, W), 0)].any():
                out.append((t, l))
    return sorted(set(out))


def hann2d(n: int, halo: int) -> np.ndarray:
    w = np.ones(n, np.float32)
    if halo > 0:
        ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 2 * halo, dtype=np.float32)))
        w[:2 * halo] = np.minimum(w[:2 * halo], ramp)
        w[-2 * halo:] = np.minimum(w[-2 * halo:], ramp[::-1])
    return np.outer(w, w) + 1e-4


def predict_full(model, inputs, cfg, mask: np.ndarray, batch_size: int = 16, mc_samples: int = 0, tta: bool = False,
                 log=print) -> dict:
    """Probability map (and MC std if requested) over ``mask``. Requires inference_enabled."""
    import torch

    from .train import resolve_device

    if not cfg.inference_enabled:
        raise PermissionError("inference requires inference_enabled=True")
    device = resolve_device(cfg.device)
    model = model.to(device).eval()
    if mc_samples:
        model.enable_mc_dropout(True)
    d = cfg.data
    L = d.local.size_px
    views = [(1, L), (int(round(d.regional.resolution_m / d.local.resolution_m)), d.regional.size_px),
             (int(round(d.global_.resolution_m / d.local.resolution_m)), d.global_.size_px)]
    step = max(f for f, _ in views)
    halo = cfg.eval.halo_px
    origins = tile_origins(mask, L, halo, step)
    H, W = mask.shape
    acc = np.zeros((H, W), np.float64)
    acc2 = np.zeros((H, W), np.float64)
    wsum = np.zeros((H, W), np.float64)
    win = hann2d(L, halo)
    t0 = time.perf_counter()
    n_pass = max(1, mc_samples)
    with torch.no_grad():
        for b0 in range(0, len(origins), batch_size):
            chunk = origins[b0:b0 + batch_size]
            batch = {k: [] for k in ("local", "regional", "global", "local_mask", "regional_mask", "global_mask")}
            for t, l in chunk:
                cy, cx = t + L // 2, l + L // 2
                for name, (f, S) in zip(LEVEL_NAMES, views):
                    x, m = inputs.window(f, cy // f - S // 2, cx // f - S // 2, S)
                    batch[name].append(x)
                    batch[f"{name}_mask"].append(m)
            tb = {k: torch.from_numpy(np.stack(v)).to(device) for k, v in batch.items()}
            preds = []
            for _ in range(n_pass):
                p = torch.sigmoid(model(tb)["fault"][:, 0].float())
                if tta:
                    tr = {k: torch.rot90(v, 2, (-2, -1)) for k, v in tb.items()}
                    # rot180 leaves axial / vector channels' values changed: (cos2t, sin2t) is invariant under 180
                    # degrees and vectors flip sign; G14 schema vectors are aspect only
                    tr = _rot180_values(tr, inputs.kinds)
                    p2 = torch.rot90(torch.sigmoid(model(tr)["fault"][:, 0].float()), 2, (-2, -1))
                    p = 0.5 * (p + p2)
                preds.append(p.cpu().numpy())
            P = np.stack(preds)                                          # (n_pass, B, L, L)
            mean, sq = P.mean(0), (P ** 2).mean(0)
            for i, (t, l) in enumerate(chunk):
                r0, c0, r1, c1 = max(t, 0), max(l, 0), min(t + L, H), min(l + L, W)
                if r1 <= r0 or c1 <= c0:
                    continue
                sr, sc = r0 - t, c0 - l
                w = win[sr:sr + r1 - r0, sc:sc + c1 - c0]
                acc[r0:r1, c0:c1] += mean[i, sr:sr + r1 - r0, sc:sc + c1 - c0] * w
                acc2[r0:r1, c0:c1] += sq[i, sr:sr + r1 - r0, sc:sc + c1 - c0] * w
                wsum[r0:r1, c0:c1] += w
            if b0 % (batch_size * 50) == 0:
                log(f"[g14.infer] {b0 + len(chunk)}/{len(origins)} tiles ({time.perf_counter() - t0:.0f}s)")
    model.enable_mc_dropout(False)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(wsum > 0, acc / wsum, np.nan)
        std = np.sqrt(np.maximum(np.where(wsum > 0, acc2 / wsum, np.nan) - mean ** 2, 0)) if mc_samples else None
    mean[~mask] = np.nan
    out = {"prob": mean.astype(np.float32), "tiles": len(origins), "seconds": time.perf_counter() - t0}
    if std is not None:
        std[~mask] = np.nan
        out["mc_std"] = std.astype(np.float32)
    if device.type == "cuda":
        out["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2**30
    return out


def _rot180_values(tb: dict, kinds: list[str]) -> dict:
    """After a 180 degree rotation: axial (cos 2t, sin 2t) unchanged; (east, north) vectors negate."""
    import torch

    idx = [i for i, k in enumerate(kinds) if k in ("vec_x", "vec_y")]
    if not idx:
        return tb
    out = dict(tb)
    for name in LEVEL_NAMES:
        x = out[name].clone()
        x[:, idx] = -x[:, idx]
        out[name] = x
    return out


def heldout_maps(cfg, stack, data, system, units, out_dir, log=print) -> dict:
    """Honest maps: each trained unit's model predicts its own held-out zone only (OOF / discovery construction)."""
    from .train import load_unit_model

    maps = {}
    for u in units:
        model, inputs, _ = load_unit_model(cfg, stack, data, u, out_dir)
        r = predict_full(model, inputs, cfg, u.heldout & data.valid, mc_samples=0, tta=cfg.eval.tta, log=log)
        maps[u.key] = r["prob"]
    return maps
