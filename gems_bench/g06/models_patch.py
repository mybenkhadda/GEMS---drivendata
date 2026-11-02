"""Compact patch segmentation (G06-U1): a small U-Net on selected geological channels (opt-in, GPU in Colab).

* Inputs: ``patch.channels`` store features, written ONCE to a channel memmap ``datasets/patches/stack_<h>.npy``
  (C, H, W float32; label-free) + a validity channel. Robust scaling (median / IQR) is fitted per fit on TRAINING
  region pixels only (``FoldScaler``) and stored with the model.
* Targets (per training visibility, visible faults only): centreline = visible fault pixels dilated by ``dilate_px``;
  per-pixel loss weights from the PU classes (POSITIVE x ``positive_weight``, UNLABELED x ``unlabeled_weight``,
  LIKELY_BACKGROUND x 1, IGNORE x 0 - so pixels outside the training region never contribute).
* Training patches lie inside the training region (``min_region_share``); a share of every batch comes from patches
  with visible positives (rare-positive sampling); early stopping on the INNER spatial validation patches.
* Inference: overlapping tiles blended with a separable Hann window (``predict_patch_model``) - no seams.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .supervision import IGNORE, LIKELY_BACKGROUND, POSITIVE, UNLABELED, validate_label_visibility


class FoldScaler:
    """Per-channel median / IQR fitted on training pixels; NaN -> 0 after scaling (+ explicit validity channel)."""

    def __init__(self, med=None, iqr=None):
        self.med, self.iqr = med, iqr

    def fit(self, X: np.ndarray) -> "FoldScaler":
        with np.errstate(all="ignore"):
            q = np.nanpercentile(X, [25, 50, 75], axis=0)
        self.med = np.nan_to_num(q[1]).astype(np.float32)
        self.iqr = np.where(np.isfinite(q[2] - q[0]) & ((q[2] - q[0]) > 0), q[2] - q[0], 1.0).astype(np.float32)
        return self

    def transform(self, A: np.ndarray) -> np.ndarray:
        """(C, h, w) or (B, C, h, w) -> scaled + validity channel appended on the channel axis."""
        ax = 0 if A.ndim == 3 else 1
        shp = [1] * A.ndim
        shp[ax] = -1
        valid = np.all(np.isfinite(A), axis=ax, keepdims=True).astype(np.float32)
        Z = (A - self.med.reshape(shp)) / self.iqr.reshape(shp)
        Z = np.clip(np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0), -10, 10).astype(np.float32)
        return np.concatenate([Z, valid], axis=ax)

    def to_dict(self) -> dict:
        return {"med": self.med.tolist(), "iqr": self.iqr.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "FoldScaler":
        return cls(np.asarray(d["med"], np.float32), np.asarray(d["iqr"], np.float32))


# =============================================================================== data
def create_patch_dataset(store, channels: list[str], out_dir, tile_px: int = 512) -> Path:
    """Write the channel memmap once (reused while the channel list and the store schema are unchanged)."""
    from .persistence import read_json, stable_hash, write_json

    validate_label_visibility("patch", channels)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h = stable_hash(store.schema_hash, list(channels), n=10)
    p, meta = out_dir / f"stack_{h}.npy", out_dir / f"stack_{h}.json"
    m = read_json(meta)
    if p.exists() and isinstance(m, dict) and m.get("complete"):
        return p
    H, W = store.grid.shape
    mm = np.lib.format.open_memmap(p, mode="w+", dtype=np.float32, shape=(len(channels), H, W))
    for r in range(0, H, tile_px):
        for c in range(0, W, tile_px):
            win = (r, min(r + tile_px, H), c, min(c + tile_px, W))
            mm[:, win[0]:win[1], win[2]:win[3]] = store.load(channels, win)
    mm.flush()
    del mm
    write_json(meta, {"channels": list(channels), "schema_hash": store.schema_hash, "shape": [len(channels), H, W],
                      "complete": True})
    return p


def loss_weights(classes: np.ndarray, cfg: dict) -> np.ndarray:
    w = np.zeros(classes.shape, np.float32)
    w[classes == POSITIVE] = float(cfg.get("positive_weight", 5.0))
    w[classes == UNLABELED] = float(cfg.get("unlabeled_weight", 0.3))
    w[classes == LIKELY_BACKGROUND] = 1.0
    w[classes == IGNORE] = 0.0
    return w


def centreline_target(visible: np.ndarray, dilate_px: int) -> np.ndarray:
    from scipy import ndimage

    from ..postprocess.topology import _disk

    v = np.asarray(visible, bool)
    return (ndimage.binary_dilation(v, structure=_disk(int(dilate_px))) if dilate_px > 0 else v).astype(np.float32)


def select_patches(index: pd.DataFrame, region: np.ndarray, inner: np.ndarray, visible: np.ndarray, P: int,
                   min_region_share: float) -> pd.DataFrame:
    """Training-region patches with inner fold of the centre and visible-positive flag."""
    rows = []
    for r in index.itertuples():
        sl = (slice(r.r0, r.r0 + P), slice(r.c0, r.c0 + P))
        share = float(region[sl].mean())
        if share < min_region_share:
            continue
        rows.append({"patch_id": r.patch_id, "r0": r.r0, "c0": r.c0, "region_share": share,
                     "inner": int(inner[r.r0 + P // 2, r.c0 + P // 2]), "has_pos": bool(visible[sl].any())})
    return pd.DataFrame(rows)


def batch_arrays(mm, scaler: FoldScaler, target: np.ndarray, weights: np.ndarray, sel: pd.DataFrame, P: int,
                 rng, augment: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, ws = [], [], []
    for r in sel.itertuples():
        sl = (slice(r.r0, r.r0 + P), slice(r.c0, r.c0 + P))
        x = scaler.transform(np.asarray(mm[:, sl[0], sl[1]]))
        y, w = target[sl], weights[sl]
        if augment:
            k = int(rng.integers(4))
            x, y, w = np.rot90(x, k, (1, 2)), np.rot90(y, k), np.rot90(w, k)
            if rng.random() < 0.5:
                x, y, w = x[:, :, ::-1], y[:, ::-1], w[:, ::-1]
        xs.append(np.ascontiguousarray(x))
        ys.append(np.ascontiguousarray(y)[None])
        ws.append(np.ascontiguousarray(w)[None])
    return np.stack(xs), np.stack(ys), np.stack(ws)


# =============================================================================== model
def build_small_unet(in_ch: int, base: int = 16, depth: int = 3, out_ch: int = 1):
    """Compact U-Net (conv-BN-ReLU x 2 per level, max-pool down, transposed-conv up, skip concatenation)."""
    import torch
    from torch import nn

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            chs = [base * 2 ** i for i in range(depth + 1)]
            self.down = nn.ModuleList([block(in_ch if i == 0 else chs[i - 1], chs[i]) for i in range(depth)])
            self.pool = nn.MaxPool2d(2)
            self.mid = block(chs[depth - 1], chs[depth])
            self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2)
                                     for i in reversed(range(depth))])
            self.dec = nn.ModuleList([block(2 * chs[i], chs[i]) for i in reversed(range(depth))])
            self.head = nn.Conv2d(chs[0], out_ch, 1)

        def forward(self, x):
            skips = []
            for d in self.down:
                x = d(x)
                skips.append(x)
                x = self.pool(x)
            x = self.mid(x)
            for up, dec, s in zip(self.up, self.dec, reversed(skips)):
                x = up(x)
                x = dec(torch.cat([x, s], 1))
            return self.head(x)

    return UNet()


def _device(use_gpu: bool):
    import torch

    return torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")


def train_patch_model(mm_path, index: pd.DataFrame, vis, classes: np.ndarray, inner: np.ndarray, cfg: dict,
                      use_gpu: bool, seed: int, progress=print) -> tuple[object, FoldScaler, dict]:
    """Train one fit; returns (model on CPU, scaler, history)."""
    import torch
    import torch.nn.functional as Fnn

    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))
    P = int(cfg["patch_px"])
    mm = np.load(mm_path, mmap_mode="r")
    region = np.asarray(vis.region, bool)
    rr, cc = np.nonzero(region)
    pick = rng.choice(rr.size, min(int(cfg.get("scaler_sample", 200000)), rr.size), replace=False)
    scaler = FoldScaler().fit(np.asarray(mm[:, rr[pick], cc[pick]]).T)
    target = centreline_target(vis.visible, int(cfg.get("dilate_px", 1)))
    weights = loss_weights(classes, cfg)
    S = select_patches(index, region, inner, vis.visible, P, float(cfg.get("min_region_share", 0.7)))
    if not len(S):
        raise ValueError(f"{vis.fit_id}: no patch inside the training region")
    val = S[S.inner == 0]
    tr = S[S.inner != 0]
    if not len(tr) or not len(val):
        raise ValueError(f"{vis.fit_id}: inner split left {len(tr)} training / {len(val)} validation patches")
    dev = _device(use_gpu)
    model = build_small_unet(mm.shape[0] + 1, int(cfg.get("base_channels", 16)), int(cfg.get("depth", 3))).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-3)),
                            weight_decay=float(cfg.get("weight_decay", 1e-4)))
    B = int(cfg.get("batch", 16))
    pos_share = float(cfg.get("pos_patch_share", 0.5))
    trp, trn = tr[tr.has_pos], tr[~tr.has_pos]
    hist, best, best_state, bad = [], np.inf, None, 0
    steps = int(cfg.get("steps_per_epoch", 100))

    def loss_of(x, y, w):
        logits = model(x)
        return (Fnn.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).sum() / w.sum().clamp_min(1.0)

    for ep in range(int(cfg.get("epochs", 30))):
        model.train()
        t0, tl = time.perf_counter(), []
        for _ in range(steps):
            npos = int(round(B * pos_share)) if len(trp) else 0
            sel = pd.concat([trp.sample(npos, replace=len(trp) < npos, random_state=int(rng.integers(1 << 31)))
                             if npos else trp.iloc[:0],
                             (trn if len(trn) else tr).sample(B - npos, replace=True,
                                                              random_state=int(rng.integers(1 << 31)))])
            x, y, w = batch_arrays(mm, scaler, target, weights, sel, P, rng, True)
            x, y, w = (torch.from_numpy(a).to(dev) for a in (x, y, w))
            opt.zero_grad()
            loss = loss_of(x, y, w)
            loss.backward()
            opt.step()
            tl.append(float(loss.detach().cpu()))
        model.eval()
        vl = []
        with torch.no_grad():
            for i in range(0, len(val), B):
                x, y, w = batch_arrays(mm, scaler, target, weights, val.iloc[i:i + B], P, rng, False)
                x, y, w = (torch.from_numpy(a).to(dev) for a in (x, y, w))
                vl.append(float(loss_of(x, y, w).cpu()))
        v = float(np.mean(vl)) if vl else np.nan
        hist.append({"epoch": ep, "train_loss": float(np.mean(tl)), "inner_val_loss": v,
                     "seconds": round(time.perf_counter() - t0, 1)})
        progress and progress(f"  U-Net {vis.fit_id} epoch {ep}: train {hist[-1]['train_loss']:.4f} "
                              f"inner-val {v:.4f}")
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(cfg.get("patience", 5)):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.cpu().eval()
    return model, scaler, {"history": hist, "best_inner_val_loss": best, "n_train_patches": len(tr),
                           "n_val_patches": len(val), "channels": int(mm.shape[0])}


def hann2d(P: int) -> np.ndarray:
    w = np.hanning(P + 2)[1:-1]
    return np.outer(w, w).astype(np.float32) + 1e-3


def predict_patch_model(model, mm_path, scaler: FoldScaler, window: tuple, mask: np.ndarray, P: int,
                        overlap: float = 0.5, use_gpu: bool = False, batch: int = 16) -> np.ndarray:
    """Overlapping-tile inference blended with a Hann window (``reconstruct_patch_predictions``)."""
    import torch

    dev = _device(use_gpu)
    model = model.to(dev).eval()
    mm = np.load(mm_path, mmap_mode="r")
    H, W = mm.shape[1:]
    r0, r1, c0, c1 = (int(x) for x in window)
    step = max(int(P * (1 - overlap)), 1)
    rs = list(range(max(r0 - P // 4, 0), max(r1 - P, r0) + 1, step)) + [max(min(r1, H) - P, 0)]
    cs = list(range(max(c0 - P // 4, 0), max(c1 - P, c0) + 1, step)) + [max(min(c1, W) - P, 0)]
    origins = sorted({(min(max(r, 0), max(H - P, 0)), min(max(c, 0), max(W - P, 0))) for r in rs for c in cs})
    tiles = []
    with torch.no_grad():
        for i in range(0, len(origins), batch):
            grp = origins[i:i + batch]
            x = np.stack([scaler.transform(np.asarray(mm[:, r:r + P, c:c + P])) for r, c in grp])
            p = torch.sigmoid(model(torch.from_numpy(x).to(dev))).cpu().numpy()[:, 0]
            tiles += [((r, c), p[j]) for j, (r, c) in enumerate(grp)]
    model.cpu()
    out = reconstruct_patch_predictions(tiles, (r0, r1, c0, c1), P)
    return np.where(np.asarray(mask, bool), out, np.nan).astype(np.float32)


def reconstruct_patch_predictions(tiles, window: tuple, P: int) -> np.ndarray:
    """Weighted average of overlapping tile predictions over ``window`` (Hann weights)."""
    r0, r1, c0, c1 = window
    acc = np.zeros((r1 - r0, c1 - c0), np.float64)
    wsum = np.zeros_like(acc)
    hw = hann2d(P)
    for (r, c), p in tiles:
        ra, rb = max(r, r0), min(r + P, r1)
        ca, cb = max(c, c0), min(c + P, c1)
        if ra >= rb or ca >= cb:
            continue
        acc[ra - r0:rb - r0, ca - c0:cb - c0] += (p * hw)[ra - r:rb - r, ca - c:cb - c]
        wsum[ra - r0:rb - r0, ca - c0:cb - c0] += hw[ra - r:rb - r, ca - c:cb - c]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(wsum > 0, acc / wsum, np.nan).astype(np.float32)


def save_patch_model(model, scaler: FoldScaler, info: dict, d) -> Path:
    import torch

    from .persistence import atomic_write, write_json

    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    atomic_write(d / "unet.pt", lambda t: torch.save(model.state_dict(), t))
    write_json(d / "unet.json", {"scaler": scaler.to_dict(), **{k: v for k, v in info.items() if k != "history"}})
    pd.DataFrame(info.get("history", [])).to_csv(d / "history.csv", index=False)
    return d


def load_patch_model(d, in_ch: int, cfg: dict):
    import torch

    d = Path(d)
    meta = json.loads((d / "unet.json").read_text(encoding="utf-8"))
    model = build_small_unet(in_ch + 1, int(cfg.get("base_channels", 16)), int(cfg.get("depth", 3)))
    model.load_state_dict(torch.load(d / "unet.pt", map_location="cpu"))
    return model.eval(), FoldScaler.from_dict(meta["scaler"])
