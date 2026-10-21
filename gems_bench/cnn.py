"""Patch-based U-Net segmentation baseline with geographic CV.

Organizer reference (github.com/drivendataorg/gems-prize-reference-solution,
``unet-mc-cv-reference-solution.ipynb``) — what could be verified from its visible cells:
``segmentation_models_pytorch`` U-Net, all 19 bands min-max normalised to [0, 1] per channel,
labels ``y < 1 -> 0``, ``TverskyLoss`` weighted towards false negatives, random train/test
splits repeated as CV iterations, predictions averaged across iterations.
Encoder, patch size, epochs, learning rate and Tversky alpha/beta were NOT visible (notebook
outputs hide them); ``reference_spec()`` uses documented guesses — diff them against the
notebook before quoting it as "the official baseline". Our version replaces random splits by
the geographic folds.

Leakage rules: a training patch must lie entirely inside ``train`` (val and buffer pixels
forbidden, checked with an integral image); invalid pixels inside a patch are zero-filled
and weight 0; pseudo-discovery ignore pixels get weight 0 (not negatives). Checkpoints are
selected on ``inner_val`` DTI, never on the scored fold.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from gems_eda.io import write_geotiff

from .config import BenchConfig
from .data import NativeData
from .features import FeatureStore
from .folds import GeoFolds
from .metric import MaskedScorer, full_diagnostics
from .persist import ExperimentRun


@dataclass
class CNNSpec:
    name: str
    channels: list[str]
    encoder: str = "resnet34"            # smp encoder; "builtin" = small U-Net below
    base_width: int = 32                 # builtin U-Net only
    loss: str = "bce_dice"               # "bce_dice" | "tversky" | "focal_tversky"
    tversky_alpha: float = 0.2           # FP weight
    tversky_beta: float = 0.8            # FN weight
    pos_weight: float = 5.0              # BCE positive weight
    target: str = "label"                # "label" | "kernel" (soft k(d) target, distance-aware)
    norm: str = "robust"                 # "robust" (cached z, clipped) | "minmax" (organizer style)
    patch_px: int = 256
    augment: bool = False                # flips change strike families; off by default
    epochs: int = 30
    steps_per_epoch: int = 200
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    pos_patch_frac: float = 0.7
    min_valid_frac: float = 0.3
    amp: bool = True
    seed: int = 2026
    notes: str = ""


def reference_spec(channels: list[str]) -> CNNSpec:
    return CNNSpec("ref_unet", channels, encoder="resnet34", loss="tversky", tversky_alpha=0.3, tversky_beta=0.7,
                   norm="minmax", notes="organizer-style: min-max inputs, Tversky loss; encoder/epochs/alpha "
                                        "unverified vs reference notebook")


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def build_model(spec: CNNSpec):
    import torch.nn as nn

    if spec.encoder != "builtin":
        try:
            import segmentation_models_pytorch as smp

            return smp.Unet(encoder_name=spec.encoder, encoder_weights=None,
                            in_channels=len(spec.channels), classes=1)
        except ImportError:
            print("segmentation_models_pytorch not installed -> builtin U-Net")

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self, c_in, w):
            super().__init__()
            ws = [w, 2 * w, 4 * w, 8 * w, 16 * w]
            self.down = nn.ModuleList([block(c_in, ws[0])] + [block(ws[i], ws[i + 1]) for i in range(4)])
            self.pool = nn.MaxPool2d(2)
            self.up = nn.ModuleList([nn.ConvTranspose2d(ws[i + 1], ws[i], 2, stride=2) for i in reversed(range(4))])
            self.dec = nn.ModuleList([block(2 * ws[i], ws[i]) for i in reversed(range(4))])
            self.head = nn.Conv2d(ws[0], 1, 1)

        def forward(self, x):
            skips = []
            for i, d in enumerate(self.down):
                x = d(x)
                if i < 4:
                    skips.append(x)
                    x = self.pool(x)
            for up, dec, s in zip(self.up, self.dec, reversed(skips)):
                x = dec(__import__("torch").cat([up(x), s], 1))
            return self.head(x)

    return UNet(len(spec.channels), spec.base_width)


# ---------------------------------------------------------------------------
# losses (all masked by a per-pixel weight map)
# ---------------------------------------------------------------------------
def masked_loss(logits, target, weight, spec: CNNSpec):
    import torch
    import torch.nn.functional as F

    p = torch.sigmoid(logits)
    wsum = weight.sum().clamp_min(1.0)
    if spec.loss == "bce_dice":
        pw = torch.tensor(spec.pos_weight, device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, target, weight=weight, pos_weight=pw, reduction="sum") / wsum
        inter = (p * target * weight).sum()
        dice = 1 - (2 * inter + 1) / ((p * weight).sum() + (target * weight).sum() + 1)
        return bce + dice
    tp = (p * target * weight).sum()
    fp = (p * (1 - target) * weight).sum()
    fn = ((1 - p) * target * weight).sum()
    tv = 1 - (tp + 1) / (tp + spec.tversky_alpha * fp + spec.tversky_beta * fn + 1)
    if spec.loss == "focal_tversky":
        return tv.clamp_min(1e-6) ** 0.75
    return tv


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def integral(mask: np.ndarray) -> np.ndarray:
    ii = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), np.int32)
    ii[1:, 1:] = np.cumsum(np.cumsum(mask.astype(np.int32), 0), 1)
    return ii


def box_sum(ii: np.ndarray, r0: int, c0: int, size: int) -> int:
    return int(ii[r0 + size, c0 + size] - ii[r0, c0 + size] - ii[r0 + size, c0] + ii[r0, c0])


def kernel_target(labels: np.ndarray, grid, radius_m: float = 300.0) -> np.ndarray:
    from scipy import ndimage

    d = ndimage.distance_transform_edt(~labels, sampling=(grid.pixel_km_y * 1000, grid.pixel_km_x * 1000))
    return np.clip(1 - d / radius_m, 0, 1).astype(np.float32)


def normalise_stack(stack: np.ndarray, valid: np.ndarray, norm: str) -> tuple[np.ndarray, dict]:
    """In-place-ish normalisation of a (C, H, W) float16 stack; returns (stack, stats)."""
    stats = {"norm": norm, "lo": [], "hi": []}
    for c in range(stack.shape[0]):
        a = stack[c]
        if norm == "minmax":
            v = a[valid].astype(np.float32)
            lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
            a[...] = ((a.astype(np.float32) - lo) / max(hi - lo, 1e-6)).astype(stack.dtype)
        else:
            lo, hi = -8.0, 8.0
            np.clip(a, lo, hi, out=a)
        stats["lo"].append(lo)
        stats["hi"].append(hi)
        a[~np.isfinite(a)] = 0
        a[~valid] = 0
    return stack, stats


class PatchSampler:
    """Random patches fully inside ``allowed`` (forbidden-pixel count must be 0)."""

    def __init__(self, allowed: np.ndarray, valid: np.ndarray, positives: np.ndarray, size: int,
                 min_valid_frac: float, rng: np.random.Generator):
        self.size, self.rng, self.min_valid = size, rng, min_valid_frac
        self.forbid = integral(~allowed & valid)      # invalid pixels are allowed (weight 0)
        self.valid_ii = integral(valid)
        self.H, self.W = allowed.shape
        self.pos = np.argwhere(positives & allowed)

    def ok(self, r0: int, c0: int) -> bool:
        s = self.size
        if r0 < 0 or c0 < 0 or r0 + s > self.H or c0 + s > self.W:
            return False
        return box_sum(self.forbid, r0, c0, s) == 0 and box_sum(self.valid_ii, r0, c0, s) >= self.min_valid * s * s

    def draw(self, positive: bool, tries: int = 200) -> tuple[int, int] | None:
        s = self.size
        for _ in range(tries):
            if positive and len(self.pos):
                r, c = self.pos[self.rng.integers(len(self.pos))]
                r0, c0 = r - self.rng.integers(s // 8, s - s // 8), c - self.rng.integers(s // 8, s - s // 8)
            else:
                r0, c0 = self.rng.integers(0, self.H - s), self.rng.integers(0, self.W - s)
            if self.ok(int(r0), int(c0)):
                return int(r0), int(c0)
        return None


def make_batch(stack, target, weight, sampler: PatchSampler, spec: CNNSpec, rng):
    import torch

    s, xs, ys, ws = spec.patch_px, [], [], []
    while len(xs) < spec.batch_size:
        rc = sampler.draw(rng.random() < spec.pos_patch_frac) or sampler.draw(False, 1000)
        if rc is None:
            raise RuntimeError("no admissible patch: training zone too fragmented for this patch size")
        r0, c0 = rc
        x = stack[:, r0:r0 + s, c0:c0 + s].astype(np.float32)
        y = target[r0:r0 + s, c0:c0 + s][None].astype(np.float32)
        w = weight[r0:r0 + s, c0:c0 + s][None].astype(np.float32)
        if spec.augment:
            if rng.random() < 0.5:
                x, y, w = x[:, :, ::-1], y[:, :, ::-1], w[:, :, ::-1]
            if rng.random() < 0.5:
                x, y, w = x[:, ::-1], y[:, ::-1], w[:, ::-1]
        xs.append(np.ascontiguousarray(x))
        ys.append(np.ascontiguousarray(y))
        ws.append(np.ascontiguousarray(w))
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)), torch.from_numpy(np.stack(ws))


# ---------------------------------------------------------------------------
# tiled inference
# ---------------------------------------------------------------------------
def _window(size: int) -> np.ndarray:
    h = np.hanning(size + 2)[1:-1]
    return (np.outer(h, h) + 1e-3).astype(np.float32)


def predict_tiled(model, stack: np.ndarray, box: tuple[int, int, int, int], patch: int, stride: int,
                  device, amp: bool = True, batch: int = 8) -> np.ndarray:
    """Overlapping Hann-blended prediction of ``box`` = (r0, r1, c0, c1); context read around it."""
    import torch

    r0, r1, c0, c1 = box
    C, H, W = stack.shape
    m = patch // 4
    R0, R1, C0, C1 = r0 - m, r1 + m, c0 - m, c1 + m
    R1, C1 = max(R1, R0 + patch), max(C1, C0 + patch)          # box smaller than one patch
    ys = list(range(R0, max(R1 - patch, R0) + 1, stride))
    xs = list(range(C0, max(C1 - patch, C0) + 1, stride))
    if ys[-1] + patch < R1:
        ys.append(R1 - patch)
    if xs[-1] + patch < C1:
        xs.append(C1 - patch)
    acc = np.zeros((R1 - R0, C1 - C0), np.float32)
    wsum = np.zeros_like(acc)
    win = _window(patch)
    tiles = [(y, x) for y in ys for x in xs]
    model.eval()
    with torch.no_grad():
        for i in range(0, len(tiles), batch):
            chunk = tiles[i:i + batch]
            xb = np.zeros((len(chunk), C, patch, patch), np.float32)
            for j, (y, x) in enumerate(chunk):
                ya, yb, xa, xb_ = max(y, 0), min(y + patch, H), max(x, 0), min(x + patch, W)
                if ya < yb and xa < xb_:
                    xb[j, :, ya - y:yb - y, xa - x:xb_ - x] = stack[:, ya:yb, xa:xb_]
            t = torch.from_numpy(xb).to(device)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                p = torch.sigmoid(model(t)).float().cpu().numpy()[:, 0]
            for j, (y, x) in enumerate(chunk):
                acc[y - R0:y - R0 + patch, x - C0:x - C0 + patch] += p[j] * win
                wsum[y - R0:y - R0 + patch, x - C0:x - C0 + patch] += win
    out = acc / np.maximum(wsum, 1e-6)
    return out[m:m + (r1 - r0), m:m + (c1 - c0)]


def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def predict_mask(model, stack, mask, spec: CNNSpec, device, stride_frac: float = 0.5) -> np.ndarray:
    out = np.full(mask.shape, np.nan, np.float32)
    if not mask.any():
        return out
    r0, r1, c0, c1 = bbox(mask)
    p = predict_tiled(model, stack, (r0, r1, c0, c1), spec.patch_px, int(spec.patch_px * stride_frac), device, spec.amp)
    sub = out[r0:r1, c0:c1]
    sub[mask[r0:r1, c0:c1]] = p[mask[r0:r1, c0:c1]]
    return out


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def device_of():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one(spec: CNNSpec, stack, target, weight, allowed, positives, valid, inner_val_mask, truth,
              device, rng, log_prefix="", ckpt_path=None):
    """Train on patches inside ``allowed``; keep the checkpoint with best inner-val DTI."""
    import torch

    torch.manual_seed(spec.seed)
    model = build_model(spec).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=spec.lr, total_steps=spec.epochs * spec.steps_per_epoch)
    use_amp = spec.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    sampler = PatchSampler(allowed, valid, positives, spec.patch_px, spec.min_valid_frac, rng)
    scorer = MaskedScorer(truth, inner_val_mask) if inner_val_mask is not None and inner_val_mask.any() else None
    best, best_state, hist = -1.0, None, []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for ep in range(spec.epochs):
        model.train()
        t0, losses = time.perf_counter(), []
        for _ in range(spec.steps_per_epoch):
            x, y, w = (t.to(device, non_blocking=True) for t in make_batch(stack, target, weight, sampler, spec, rng))
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
            loss = masked_loss(logits.float(), y, w, spec)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(loss.detach()))
        row = {"epoch": ep + 1, "train_loss": float(np.mean(losses)), "lr": float(sched.get_last_lr()[0]),
               "epoch_s": round(time.perf_counter() - t0, 1)}
        if scorer is not None:
            pv = predict_mask(model, stack, inner_val_mask, spec, device)
            c = scorer.components(pv)
            row.update(inner_val_dti=c.dti, inner_val_recall_w=c.tp / max(c.n_truth, 1))
            if c.dti > best:
                best = c.dti
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if device.type == "cuda":
            row["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
        hist.append(row)
        print(f"{log_prefix} ep {ep + 1:3d} loss {row['train_loss']:.4f} inner DTI {row.get('inner_val_dti', float('nan')):.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    if ckpt_path is not None:
        torch.save({"state_dict": model.state_dict(), "spec": asdict(spec), "best_inner_dti": best}, ckpt_path)
    return model, pd.DataFrame(hist)


def prepare_inputs(spec: CNNSpec, data: NativeData, store: FeatureStore):
    stack = store.load_stack(spec.channels, dtype=np.float16)
    stack, stats = normalise_stack(stack, data.valid, spec.norm)
    target = kernel_target(data.labels, data.grid) if spec.target == "kernel" else data.labels.astype(np.float32)
    return stack, stats, target


def run_cnn_cv(exp_id: str, spec: CNNSpec, data: NativeData, store: FeatureStore, folds: GeoFolds,
               cfg: BenchConfig, *, input_config: str = "", overwrite: bool = False) -> ExperimentRun:
    import torch

    from .gbm import _finish_cv

    run = ExperimentRun(data.paths, exp_id, "cnn", {**asdict(spec), "input_config": input_config,
                                                    "folds": data.paths.rel(folds.directory) if folds.directory else None})
    if run.done and not overwrite:
        print(f"[{exp_id}] already done -> {run.dir}")
        return run
    run.start()
    device = device_of()
    rng = np.random.default_rng(spec.seed)
    stack, stats, target = prepare_inputs(spec, data, store)
    (run.models_dir / "normalisation.json").write_text(json.dumps({"channels": spec.channels, **stats}), encoding="utf-8")
    run.register("normalisation", run.models_dir / "normalisation.json", "preprocessing")
    oof = np.full(data.shape, np.nan, np.float32)
    hist_all, met_rows = [], []
    for k in range(folds.n_folds):
        allowed = folds.inner_train(k)
        # labels outside the training zone must not shape the loss: weight = training zone only
        weight = (allowed & data.valid).astype(np.float32)
        ck = run.models_dir / f"fold{k}.pt"
        with run.timer(f"fold{k}_train"):
            model, hist = train_one(spec, stack, target, weight, allowed, data.labels, data.valid,
                                    folds.inner_val(k), data.labels, device, rng, f"[{exp_id} f{k}]", ck)
        hist["fold"] = k
        hist_all.append(hist)
        val = folds.val(k)
        with run.timer(f"fold{k}_predict"):
            oof[val] = predict_mask(model, stack, val, spec, device, cfg.infer_stride_frac)[val]
        diag = full_diagnostics(oof, data.labels, val & data.valid, rng, cfg.recall_tolerances_m, cfg.report_threshold)
        met_rows.append({"fold": k, "best_inner_dti": float(hist.inner_val_dti.max()) if "inner_val_dti" in hist else np.nan,
                         "gpu_peak_mb": float(hist.gpu_peak_mb.max()) if "gpu_peak_mb" in hist else np.nan, **diag})
        print(f"[{exp_id}] fold {k}: DTI={diag['dti']:.4f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_geotiff(run.oof_path, {"oof_probability": oof}, data.grid)
    run.register("oof", run.oof_path, "raster")
    run.config["channels"] = spec.channels
    return _finish_cv(run, oof, data, folds, cfg, hist_all, [], [], met_rows, [], rng)


def run_cnn_full(exp_id: str, spec: CNNSpec, data: NativeData, store: FeatureStore, *,
                 ignore_mask: np.ndarray | None = None, predict_area: np.ndarray | None = None,
                 overwrite: bool = False, extra_config: dict | None = None) -> ExperimentRun:
    """One model on the whole valid area minus ``ignore_mask`` (weight 0, patches allowed over it).

    Used for pseudo-discovery: withheld systems stay visible as *inputs* but carry no supervision.
    Fixed epoch budget (no inner validation), last state kept. Writes ``pred.tif``.
    """
    run = ExperimentRun(data.paths, exp_id, "cnn", {**asdict(spec), **(extra_config or {})})
    pred_path = run.dir / "pred.tif"
    if run.state.get("status") == "done" and pred_path.exists() and not overwrite:
        return run
    run.start()
    device = device_of()
    rng = np.random.default_rng(spec.seed)
    stack, stats, target = prepare_inputs(spec, data, store)
    allowed = data.valid.copy()
    weight = data.valid.astype(np.float32)
    positives = data.labels.copy()
    if ignore_mask is not None:
        weight[ignore_mask] = 0.0
        positives &= ~ignore_mask
    with run.timer("train"):
        model, hist = train_one(spec, stack, target, weight, allowed, positives, data.valid, None, data.labels,
                                device, rng, f"[{exp_id}]", run.models_dir / "model.pt")
    run.save_table(hist, "history")
    area = data.valid if predict_area is None else predict_area
    with run.timer("predict"):
        pred = predict_mask(model, stack, area, spec, device)
    write_geotiff(pred_path, {"probability": pred}, data.grid)
    run.register("pred", pred_path, "raster")
    run.state.update(status="done")
    run._flush()
    return run


def load_fold_models(run: ExperimentRun, device=None):
    import torch

    device = device or device_of()
    models = []
    for ck in sorted(run.models_dir.glob("fold*.pt")):
        obj = torch.load(ck, map_location=device)
        spec = CNNSpec(**obj["spec"])
        m = build_model(spec).to(device)
        m.load_state_dict(obj["state_dict"])
        models.append((spec, m))
    return models
