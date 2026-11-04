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

import json
import time
from pathlib import Path

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


def prepare_targets(visible: np.ndarray, classes: np.ndarray, orient: dict, cfg: dict) -> dict:
    """Full-grid 100 m target products of one training visibility (computed once per fit, never per sample)."""
    from ..g06.models_patch import loss_weights

    return {"y": np.asarray(visible, np.float32), "w": loss_weights(classes, cfg).astype(np.float32),
            "cos2": np.nan_to_num(orient["orient_cos2"]).astype(np.float32),
            "sin2": np.nan_to_num(orient["orient_sin2"]).astype(np.float32),
            "om": np.isfinite(orient["orient_cos2"]).astype(np.float32)}


def patch_targets(T: dict, r0: int, c0: int, cells: int, f: int) -> tuple:
    """Upsampled (nearest) centreline / weight / orientation targets of one patch; a target cell maps exactly onto its
    ``f`` x ``f`` input pixels (the patches are aligned to the analysis grid)."""
    sl = (slice(r0, r0 + cells), slice(c0, c0 + cells))
    up = lambda a: np.repeat(np.repeat(a[sl], f, 0), f, 1)    # noqa: E731
    return up(T["y"]), up(T["w"]), np.stack([up(T["cos2"]), up(T["sin2"])]), up(T["om"])


def select_surface_patches(index: pd.DataFrame, vis, inner: np.ndarray, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """(patches of one fit with their role, readiness facts). Uses TRAINING-VISIBLE faults and the fit's region only.

    A patch is usable when >= ``min_region_share`` of it lies in the training region. Validation patches lie entirely
    in one inner fold; training patches touch no cell of it; straddling patches are dropped so no validation cell is
    trained on. The validation fold is the inner fold with the best balance of visible-positive patches: the inner
    folds balance the whole footprint, so a DEM that covers part of it can miss any one given fold."""
    region, visible = np.asarray(vis.region, bool), np.asarray(vis.visible, bool)
    rows = []
    for i, r in enumerate(index.itertuples()):
        c = int(r.cells)
        sl = (slice(r.r0, r.r0 + c), slice(r.c0, r.c0 + c))
        share = float(region[sl].mean())
        iv = inner[sl]
        iv = iv[iv >= 0]
        if share < float(cfg.get("min_region_share", 0.7)) or not iv.size:
            continue
        bits = int(np.bitwise_or.reduce(1 << np.unique(iv).astype(int)))
        rows.append({"mm_row": i, "r0": int(r.r0), "c0": int(r.c0), "cells": c, "region_share": share,
                     "has_pos": bool(visible[sl].any()), "fold_bits": bits,
                     "dem_share": float(getattr(r, "dem_share", np.nan))})
    I = pd.DataFrame(rows, columns=["mm_row", "r0", "c0", "cells", "region_share", "has_pos", "fold_bits", "dem_share"])
    facts = {"val_fold": None, "n_train": 0, "n_val": 0, "n_train_pos": 0, "n_val_pos": 0, "n_train_neg": 0,
             "n_val_neg": 0, "n_region": int(len(I)), "status": "BLOCKED", "reason": ""}
    if not len(I):
        return I.assign(role=pd.Series(dtype=str)), {**facts, "reason": f"{vis.fit_id}: none of the {len(index)} patches "
                                                     "has enough of its area in the training region"}
    best, best_key = None, None
    for f in np.unique(inner[inner >= 0]):
        bit = 1 << int(f)
        va, tr = I[I.fold_bits == bit], I[(I.fold_bits & bit) == 0]
        key = (min(int(va.has_pos.sum()), int(tr.has_pos.sum())), min(len(va), len(tr)), -int(f))
        if best_key is None or key > best_key:
            best, best_key = int(f), key
    bit = 1 << best
    I["role"] = np.where(I.fold_bits == bit, "val", np.where((I.fold_bits & bit) == 0, "train", "buffer"))
    tr, va = I[I.role == "train"], I[I.role == "val"]
    facts.update(val_fold=best, n_train=int(len(tr)), n_val=int(len(va)), n_train_pos=int(tr.has_pos.sum()),
                 n_val_pos=int(va.has_pos.sum()), n_train_neg=int((~tr.has_pos).sum()), n_val_neg=int((~va.has_pos).sum()))
    if not len(tr) or not len(va):
        facts["reason"] = f"{vis.fit_id}: inner split left {len(tr)} training / {len(va)} validation patches"
    elif not facts["n_train_pos"]:
        facts["reason"] = f"{vis.fit_id}: no training patch contains a visible fault"
    elif not facts["n_val_pos"]:
        facts["reason"] = f"{vis.fit_id}: no inner-validation patch contains a visible fault"
    else:
        facts["status"] = "READY"
    return I, facts


class GPURuntimeError(RuntimeError):
    """A GPU was requested and is not usable: there is no silent CPU fallback."""


def validate_gpu_runtime(requested: bool = True, require_cuda: bool = True) -> dict:
    """What the runtime offers for the U-Net (the classical models always run on CPU: see their registry ``device``).
    Raises ``GPURuntimeError`` when CUDA is requested and required but unavailable."""
    info = {"requested": bool(requested), "require_cuda": bool(require_cuda), "torch_version": None,
            "torch_cuda_version": None, "cuda_available": False, "gpu_name": None, "vram_total_gib": None,
            "selected_device": "cpu", "readiness": "CPU_ONLY"}
    try:
        import torch
    except ImportError as e:
        if requested and require_cuda:
            raise GPURuntimeError(f"PyTorch is not installed ({e}): the U-Net cannot train") from e
        return {**info, "readiness": "NO_TORCH"}
    info.update(torch_version=torch.__version__, torch_cuda_version=torch.version.cuda,
                cuda_available=bool(torch.cuda.is_available()))
    if info["cuda_available"]:
        p = torch.cuda.get_device_properties(0)
        info.update(gpu_name=torch.cuda.get_device_name(0), vram_total_gib=round(p.total_memory / 2**30, 2))
    if requested and info["cuda_available"]:
        info.update(selected_device="cuda:0", readiness="READY")
    elif requested and require_cuda:
        raise GPURuntimeError("use_gpu=True but CUDA is unavailable (torch "
                              f"{info['torch_version']}, CUDA build {info['torch_cuda_version']}). In Colab: Runtime -> "
                              "Change runtime type -> GPU, then run from the top. There is no silent CPU fallback; set "
                              "CONFIG['use_gpu']=False only if a slow CPU run is really wanted.")
    return info


def resolve_device(use_gpu: bool):
    """The training / inference device; ``use_gpu`` without CUDA raises instead of falling back to CPU."""
    import torch

    return torch.device(validate_gpu_runtime(bool(use_gpu), bool(use_gpu))["selected_device"])


OOM_GUIDANCE = ("CUDA out of memory in the surface U-Net. Lower OVERRIDES['surface']['unet']['batch'] (e.g. 4 -> 2), "
                "then base_channels (16 -> 12); restart the runtime to release the GPU. Finished fits are reused and an "
                "interrupted fit resumes from its checkpoint.")


def train_surface_unet(mm_path, index: pd.DataFrame, vis, classes: np.ndarray, orient: dict, inner: np.ndarray,
                       cfg: dict, f: int, use_gpu: bool, seed: int, progress=print, checkpoint=None, on_epoch=None
                       ) -> tuple:
    """(model on CPU, scaler, info with the history). Patch roles: ``select_surface_patches``. Inputs, targets and
    weights all live on the selected device; a non-finite loss or gradient stops the fit with its epoch and step. With
    ``checkpoint`` (path) the state is saved every epoch and an interrupted fit resumes from it."""
    import torch
    import torch.nn.functional as Fnn

    from ..g06.contracts import LeakageError
    from ..g06.models_patch import FoldScaler
    from ..g06.persistence import atomic_write

    if (np.asarray(vis.visible, bool) & np.asarray(vis.hidden, bool)).any():
        raise LeakageError(f"{vis.fit_id}: a hidden fault is visible to the U-Net targets")
    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))
    mm = np.load(mm_path, mmap_mode="r")
    I, facts = select_surface_patches(index, vis, inner, cfg)
    if facts["status"] != "READY":
        raise ValueError(facts["reason"])
    tr, va = I[I.role == "train"], I[I.role == "val"]
    cells = int(tr.cells.iloc[0])
    pick = np.sort(rng.choice(tr.mm_row.to_numpy(), min(64, len(tr)), replace=False))
    sample = np.asarray(mm[pick], np.float32).transpose(1, 0, 2, 3).reshape(mm.shape[1], -1).T   # training patches only
    sample = sample[rng.choice(len(sample), min(500_000, len(sample)), replace=False)]
    scaler = FoldScaler().fit(sample)
    del sample
    dev = resolve_device(use_gpu)
    model = build_surface_unet(mm.shape[1], int(cfg.get("base_channels", 16)), int(cfg.get("depth", 3))).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-3)), weight_decay=float(
        cfg.get("weight_decay", 1e-4)))
    B, lam_d, lam_o = int(cfg.get("batch", 8)), float(cfg.get("dice_weight", 0.5)), float(cfg.get("orient_weight", 0.2))
    T = prepare_targets(vis.visible, classes, orient, cfg)
    cap = int(cfg.get("max_val_patches", 48))
    shuffled = va.sample(frac=1.0, random_state=int(seed))                     # a fixed, positive-rich validation set
    va_eval = pd.concat([shuffled[shuffled.has_pos].head(cap // 2 + cap % 2), shuffled[~shuffled.has_pos].head(cap // 2)])

    def batch(rows, aug):
        xs, ys, ws, os_ = [], [], [], []
        for r in rows.itertuples():
            x = scaler.transform(np.asarray(mm[r.mm_row], np.float32))
            y, w, o, om = patch_targets(T, r.r0, r.c0, cells, f)
            w = w * x[-1]                                                      # no loss where a channel has no DEM
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

    hist, best, best_state, bad, start, best_epoch = [], np.inf, None, 0, 0, -1
    if checkpoint is not None and Path(checkpoint).exists():
        ck = torch.load(checkpoint, map_location=dev, weights_only=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        hist, best, best_state, bad, start, best_epoch = (ck["hist"], ck["best"], ck["best_state"], ck["bad"],
                                                          ck["epoch"] + 1, ck["best_epoch"])
        rng = np.random.default_rng(int(seed) + start)
        if progress:
            progress(f"  surface U-Net {vis.fit_id}: resumed after epoch {ck['epoch']}")
    trp, trn = tr[tr.has_pos], tr[~tr.has_pos]
    use_cuda = dev.type == "cuda"
    grad_clip = float(cfg.get("grad_clip", 5.0))
    stopped_early = bad >= int(cfg.get("patience", 5))
    try:
        for ep in range(start, int(cfg.get("epochs", 30))):
            if stopped_early:
                break
            model.train()
            t0, tl = time.perf_counter(), []
            if use_cuda:
                torch.cuda.reset_peak_memory_stats(dev)
            for step in range(int(cfg.get("steps_per_epoch", 60))):
                npos = int(round(B * float(cfg.get("pos_patch_share", 0.5))))
                rows = pd.concat([trp.sample(npos, replace=True, random_state=int(rng.integers(1 << 31))),
                                  (trn if len(trn) else tr).sample(B - npos, replace=True,
                                                                   random_state=int(rng.integers(1 << 31)))])
                x, y, w, o = batch(rows, True)
                opt.zero_grad()
                loss = loss_of(x, y, w, o)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{vis.fit_id}: non-finite loss at epoch {ep}, step {step}")
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if not torch.isfinite(gn):
                    raise FloatingPointError(f"{vis.fit_id}: non-finite gradient at epoch {ep}, step {step}")
                opt.step()
                tl.append(float(loss.detach().cpu()))
            model.eval()
            with torch.no_grad():
                vl = [float(loss_of(*batch(va_eval.iloc[i:i + B], False)).cpu()) for i in range(0, len(va_eval), B)]
            v = float(np.mean(vl))
            if not np.isfinite(v):
                raise FloatingPointError(f"{vis.fit_id}: non-finite inner-validation loss at epoch {ep}")
            if v < best - 1e-5:
                best, bad, best_epoch = v, 0, ep
                best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
            hist.append({"fit_id": vis.fit_id, "epoch": ep, "train_loss": float(np.mean(tl)), "inner_val_loss": v,
                         "lr": float(opt.param_groups[0]["lr"]),
                         "gpu_allocated_mb": round(torch.cuda.max_memory_allocated(dev) / 2**20, 1) if use_cuda else 0.0,
                         "gpu_reserved_mb": round(torch.cuda.max_memory_reserved(dev) / 2**20, 1) if use_cuda else 0.0,
                         "seconds": round(time.perf_counter() - t0, 1), "best_epoch": best_epoch})
            if progress:
                progress(f"  surface U-Net {vis.fit_id} epoch {ep} [{dev}]: train {hist[-1]['train_loss']:.4f} "
                         f"inner {v:.4f} best@{best_epoch} ({hist[-1]['seconds']} s)")
            if checkpoint is not None:
                state = {"epoch": ep, "model": model.state_dict(), "opt": opt.state_dict(), "hist": hist, "best": best,
                         "best_state": best_state, "bad": bad, "best_epoch": best_epoch}
                atomic_write(checkpoint, lambda t, s=state: torch.save(s, t))
            if on_epoch:
                on_epoch(hist)
            stopped_early = bad >= int(cfg.get("patience", 5))
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise RuntimeError(f"{vis.fit_id}: {OOM_GUIDANCE} ({str(e)[:200]})") from e
        raise
    if best_state is None:
        raise ValueError(f"{vis.fit_id}: training produced no valid model state")
    model.load_state_dict(best_state)                                          # the best epoch, not the last one
    gpu = torch.cuda.get_device_name(dev) if use_cuda else None
    return model.cpu().eval(), scaler, {"history": hist, "best_inner_val_loss": best, "best_epoch": best_epoch,
                                        "epochs_run": len(hist), "n_train": int(len(tr)), "n_val": int(len(va)),
                                        "n_val_used": int(len(va_eval)), "n_train_pos": facts["n_train_pos"],
                                        "n_val_pos": facts["n_val_pos"], "val_fold": facts["val_fold"],
                                        "device": str(dev), "gpu_name": gpu}


def save_surface_unet(model, scaler, info: dict, d, channels: list[str], cfg: dict, res_m: float, cells: int) -> Path:
    """Weights + scaler + the input contract (channel order, resolution, patch size, architecture) next to each other."""
    from ..g06.models_patch import save_patch_model

    contract = {"channels": list(channels), "res_m": float(res_m), "cells": int(cells), "out_channels": 3,
                "base_channels": int(cfg.get("base_channels", 16)), "depth": int(cfg.get("depth", 3))}
    return save_patch_model(model, scaler, {**info, **contract}, d)


def load_surface_unet(d) -> tuple:
    """(model on CPU, scaler, saved metadata) rebuilt from the persisted input contract (3 output channels)."""
    import torch

    from ..g06.models_patch import FoldScaler

    meta = json.loads((Path(d) / "unet.json").read_text(encoding="utf-8"))
    model = build_surface_unet(len(meta["channels"]), int(meta["base_channels"]), int(meta["depth"]))
    model.load_state_dict(torch.load(Path(d) / "unet.pt", map_location="cpu", weights_only=True))
    return model.eval(), FoldScaler.from_dict(meta["scaler"]), meta


def predict_surface_tiles(model, scaler, mm_path, index: pd.DataFrame, grid_shape, f: int, use_gpu: bool = False,
                          batch: int = 8) -> np.ndarray:
    """Centreline probability of every indexed patch on the 100 m grid (NaN where no patch or no DEM). Same scaler and
    validity channel as training; overlapping patches blend with a Hann window. ``index`` must be in memmap order."""
    import torch

    from ..g06.models_patch import hann2d

    dev = resolve_device(use_gpu)
    model = model.to(dev).eval()
    mm = np.load(mm_path, mmap_mode="r")
    if mm.shape[1] != len(scaler.med):
        raise ValueError(f"patch memmap has {mm.shape[1]} channels, the model's scaler {len(scaler.med)}")
    acc = np.zeros(grid_shape, np.float64)
    cnt = np.zeros(grid_shape, np.float64)
    cells = int(index.cells.iloc[0]) if len(index) else 0
    hw = hann2d(cells).astype(np.float64) if cells else None
    rows = [(i, r.r0, r.c0) for i, r in enumerate(index.itertuples())]
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            grp = rows[i:i + batch]
            x = np.stack([scaler.transform(np.asarray(mm[j], np.float32)) for j, _, _ in grp])
            p = torch.sigmoid(model(torch.from_numpy(x).to(dev))[:, 0]).cpu().numpy()
            if not np.isfinite(p).all():
                raise FloatingPointError("non-finite U-Net prediction")
            for (_, r0, c0), pp, xx in zip(grp, p, x):
                blk = pp.reshape(cells, f, cells, f).mean((1, 3))
                ok = xx[-1].reshape(cells, f, cells, f).mean((1, 3)) >= 0.5       # the validity channel
                sl = (slice(r0, r0 + cells), slice(c0, c0 + cells))
                acc[sl] += np.where(ok, blk * hw, 0.0)
                cnt[sl] += np.where(ok, hw, 0.0)
    model.cpu()
    return np.where(cnt > 0, acc / np.maximum(cnt, 1e-12), np.nan).astype(np.float32)
