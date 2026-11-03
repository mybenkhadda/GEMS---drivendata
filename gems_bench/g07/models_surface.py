"""System A - surface (DEM) models.

Classical (100 m, DEM-covered cells only): G07-A1 = DEM geomorphometry channels; G07-A2 = A1 + terrain-structure
channels (scarps, lineaments, truncation, persistence, drainage) + terrain-vs-geophysics orientation agreement.
Random Forest / LightGBM through the G06 pixel machinery (same pools, visibilities, early stopping, imputation).

Compact U-Net (G07-C1, opt-in): ``g06.models_patch.build_small_unet`` with 3 outputs - centreline logit and the
doubled-angle strike (cos 2t, sin 2t) - on DEM patches ALIGNED to the 100 m grid at the surface work resolution.
Loss: weighted BCE (PU weights: IGNORE 0, UNLABELED low, LIKELY_BACKGROUND 1, POSITIVE high) + soft Dice on the
centreline + orientation MSE on positive pixels where the strike is stable. Augmentations keep directional targets
consistent: a 90 deg rotation maps (cos 2t, sin 2t) -> (-cos 2t, -sin 2t); a horizontal or vertical mirror maps
(cos 2t, sin 2t) -> (cos 2t, -sin 2t). Early stopping on inner spatial validation patches; predictions are averaged
back onto the 100 m grid.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd


def train_surface_random_forest(X, y, w, params: dict, seed: int):
    from ..g06.models_pixel import fit_pixel_model

    return fit_pixel_model("random_forest", X, y, w, params, seed)


def train_surface_lightgbm(X, y, w, params: dict, seed: int, inner=None, inner_buffer=None):
    from ..g06.models_pixel import fit_pixel_model

    return fit_pixel_model("lightgbm", X, y, w, params, seed, inner, inner_buffer)


# =============================================================================== U-Net
def build_surface_unet(in_ch: int, base: int = 16, depth: int = 3, orientation: bool = True):
    from ..g06.models_patch import build_small_unet

    return build_small_unet(in_ch + 1, base, depth, 3 if orientation else 1)


def augment(x: np.ndarray, y: np.ndarray, w: np.ndarray, o: np.ndarray, rng, k: int | None = None,
            mirror: bool | None = None) -> tuple:
    """Random 90 deg rotation + mirror with consistent orientation targets. x (C, P, P); y, w (P, P);
    o (3, P, P) = (cos 2t, sin 2t, validity mask)."""
    k = int(rng.integers(4)) if k is None else int(k)
    mirror = bool(rng.random() < 0.5) if mirror is None else bool(mirror)
    x, y, w, o = np.rot90(x, k, (1, 2)), np.rot90(y, k), np.rot90(w, k), np.rot90(o, k, (1, 2)).copy()
    if k % 2:
        o[:2] = -o[:2]                                           # theta + 90 deg: doubled angle + 180 deg
    if mirror:
        x, y, w, o = x[:, :, ::-1], y[:, ::-1], w[:, ::-1], o[:, :, ::-1].copy()
        o[1] = -o[1]                                             # mirror: theta -> -theta (mod 180)
    return (np.ascontiguousarray(x), np.ascontiguousarray(y), np.ascontiguousarray(w), np.ascontiguousarray(o))


def patch_targets(visible: np.ndarray, classes: np.ndarray, orient: dict, r0: int, c0: int, cells: int, f: int,
                  cfg: dict) -> tuple:
    """Upsampled (nearest) centreline / weight / orientation targets of one patch from 100 m label products."""
    from ..g06.models_patch import loss_weights

    sl = (slice(r0, r0 + cells), slice(c0, c0 + cells))
    up = lambda a: np.repeat(np.repeat(np.asarray(a)[sl], f, 0), f, 1)    # noqa: E731
    y = up(visible).astype(np.float32)
    w = up(loss_weights(classes, cfg)).astype(np.float32)
    o = np.stack([up(np.nan_to_num(orient["orient_cos2"])), up(np.nan_to_num(orient["orient_sin2"]))])
    om = up(np.isfinite(orient["orient_cos2"]))
    return y, w, o.astype(np.float32), om.astype(np.float32)


def train_surface_unet(mm_path, index: pd.DataFrame, vis, classes: np.ndarray, orient: dict, inner: np.ndarray,
                       cfg: dict, f: int, use_gpu: bool, seed: int, progress=print) -> tuple:
    """(model on CPU, scaler, history). Patches inside the fit's training region only."""
    import torch
    import torch.nn.functional as Fnn

    from ..g06.models_patch import FoldScaler, _device

    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))
    mm = np.load(mm_path, mmap_mode="r")
    cells = int(index.cells.iloc[0])
    region = np.asarray(vis.region, bool)
    keep = [i for i, r in enumerate(index.itertuples())
            if region[r.r0:r.r0 + cells, r.c0:r.c0 + cells].mean() >= float(cfg.get("min_region_share", 0.7))]
    if not keep:
        raise ValueError(f"{vis.fit_id}: no surface patch inside the training region")
    I = index.iloc[keep].assign(mm_row=keep)
    I["inner"] = [int(inner[r.r0 + cells // 2, r.c0 + cells // 2]) for r in I.itertuples()]
    I["has_pos"] = [bool(vis.visible[r.r0:r.r0 + cells, r.c0:r.c0 + cells].any()) for r in I.itertuples()]
    tr, va = I[I.inner != 0], I[I.inner == 0]
    if not len(tr) or not len(va):
        raise ValueError(f"{vis.fit_id}: inner split left {len(tr)} / {len(va)} patches")
    sample = np.asarray(mm[tr.mm_row.to_numpy()[: min(64, len(tr))]], np.float32)
    scaler = FoldScaler().fit(sample.transpose(1, 0, 2, 3).reshape(sample.shape[1], -1).T)
    dev = _device(use_gpu)
    model = build_surface_unet(mm.shape[1], int(cfg.get("base_channels", 16)), int(cfg.get("depth", 3))).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-3)), weight_decay=float(
        cfg.get("weight_decay", 1e-4)))
    B, lam_d, lam_o = int(cfg.get("batch", 8)), float(cfg.get("dice_weight", 0.5)), float(cfg.get("orient_weight", 0.2))

    def batch(rows, aug):
        xs, ys, ws, os_ = [], [], [], []
        for r in rows.itertuples():
            x = scaler.transform(np.asarray(mm[r.mm_row], np.float32))
            y, w, o, om = patch_targets(vis.visible, classes, orient, r.r0, r.c0, cells, f, cfg)
            o = np.concatenate([o, om[None]])
            if aug:
                x, y, w, o = augment(x, y, w, o, rng)
            xs.append(x)
            ys.append(y[None])
            ws.append(w[None])
            os_.append(o)
        return (torch.from_numpy(np.stack(a)).to(dev) for a in (xs, ys, ws, os_))

    def loss_of(x, y, w, o):
        out = model(x)
        lg = out[:, :1]
        bce = (Fnn.binary_cross_entropy_with_logits(lg, y, reduction="none") * w).sum() / w.sum().clamp_min(1.0)
        p = torch.sigmoid(lg) * (w > 0)
        dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
        om = o[:, 2:3] * y
        ori = (((out[:, 1:3] - o[:, :2]) ** 2) * om).sum() / om.sum().clamp_min(1.0)
        return bce + lam_d * dice + lam_o * ori

    hist, best, best_state, bad = [], np.inf, None, 0
    trp, trn = tr[tr.has_pos], tr[~tr.has_pos]
    for ep in range(int(cfg.get("epochs", 30))):
        model.train()
        t0, tl = time.perf_counter(), []
        for _ in range(int(cfg.get("steps_per_epoch", 60))):
            npos = int(round(B * float(cfg.get("pos_patch_share", 0.5)))) if len(trp) else 0
            rows = pd.concat([trp.sample(npos, replace=True, random_state=int(rng.integers(1 << 31))) if npos else
                              trp.iloc[:0], (trn if len(trn) else tr).sample(B - npos, replace=True,
                                                                              random_state=int(rng.integers(1 << 31)))])
            x, y, w, o = batch(rows, True)
            opt.zero_grad()
            loss = loss_of(x, y, w, o)
            loss.backward()
            opt.step()
            tl.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            vl = [float(loss_of(*batch(va.iloc[i:i + B], False)).cpu()) for i in range(0, len(va), B)]
        v = float(np.mean(vl))
        hist.append({"epoch": ep, "train_loss": float(np.mean(tl)), "inner_val_loss": v,
                     "seconds": round(time.perf_counter() - t0, 1)})
        if progress:
            progress(f"  surface U-Net {vis.fit_id} epoch {ep}: train {hist[-1]['train_loss']:.4f} inner {v:.4f}")
        if v < best - 1e-5:
            best, bad, best_state = v, 0, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(cfg.get("patience", 5)):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu().eval(), scaler, {"history": hist, "best_inner_val_loss": best, "n_train": len(tr),
                                        "n_val": len(va)}


def predict_surface_tiles(model, scaler, mm_path, index: pd.DataFrame, grid_shape, f: int, use_gpu: bool = False,
                          batch: int = 8) -> np.ndarray:
    """Centreline probability of every indexed patch averaged onto the 100 m grid (NaN where no patch)."""
    import torch

    from ..g06.models_patch import _device

    dev = _device(use_gpu)
    model = model.to(dev).eval()
    mm = np.load(mm_path, mmap_mode="r")
    acc = np.zeros(grid_shape, np.float64)
    cnt = np.zeros(grid_shape, np.float64)
    cells = int(index.cells.iloc[0]) if len(index) else 0
    rows = list(index.itertuples())
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            grp = rows[i:i + batch]
            x = np.stack([scaler.transform(np.asarray(mm[int(r.Index)], np.float32)) for r in grp])
            p = torch.sigmoid(model(torch.from_numpy(x).to(dev))[:, 0]).cpu().numpy()
            for r, pp in zip(grp, p):
                blk = pp.reshape(cells, f, cells, f).mean((1, 3))
                acc[r.r0:r.r0 + cells, r.c0:r.c0 + cells] += blk
                cnt[r.r0:r.r0 + cells, r.c0:r.c0 + cells] += 1
    model.cpu()
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan).astype(np.float32)
