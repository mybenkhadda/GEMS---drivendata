"""Compact U-Net benchmark (Phase 3E) built on cnn.py (model, patch sampler, tiled inference).

Additions over Phase 2
----------------------
* PU ignore band: pixels 1..``ignore_px`` from a visible label get weight 0 (thin-label
  localisation uncertainty), as in the GBM sampler; withheld systems + ring get weight 0 and are
  never positive-patch centres (pseudo-discovery positives are never relabelled as negatives).
* Targets are built from *visible* labels only (fold training zone / non-withheld).
* ``loss="dti"``: differentiable DTI surrogate — TP via the metric's own kernel max over offsets,
  FP weighted by (1 - k(dist)), FN = n - TP — plus a small BCE term for stable early training.
  ``loss="bce_dice"`` is the reference objective.
* Checkpoint / early stopping on inner-val *binary-area* DTI (scale-free; training-side labels),
  with patience; raw inner-val DTI also logged.
* Optional rot180 augmentation (preserves strike; flips do not).
* torch.amp API, deterministic seeds, per-epoch time / GPU peak / RSS logged.

GPU throughput (CUDA only; the objective, sampler RNG stream and patches are unchanged)
------------------------------------------------------------------------------------
* ``data_on_gpu``: the normalised stack (fp16) and target / kernel / weight rasters live on the GPU;
  patches are sliced there, so no per-step numpy copy or host->device transfer. Falls back to the
  host path if the stack does not fit.
* bf16 autocast on Ampere+ (no GradScaler), fp16 + GradScaler otherwise; channels_last convs;
  cuDNN autotuning + TF32 (``cudnn_benchmark=False`` restores bit-reproducible kernels).
* Loss is accumulated on the device (no per-step sync); the DTI surrogate uses a running max over
  cached kernel offsets instead of stacking ~29 shifted copies.
* Inference (inner-val each epoch, OOF, discovery) runs fully on the GPU: tiles are cut from the
  resident stack and Hann-blended in a device accumulator, ``infer_batch`` tiles per forward.
* ``eval_every``: inner-val evaluation every n epochs (patience then counts evaluations).

Dependencies: torch >= 2.1 (CUDA build for a GPU), optional segmentation-models-pytorch
(``encoder="builtin"`` needs neither smp nor pretrained weights).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import ndimage

from gems_eda.io import write_geotiff

from ..cnn import CNNSpec, PatchSampler, _window, bbox, build_model, device_of, normalise_stack, predict_mask
from ..metric import kernel_offsets
from ..persist import ExperimentRun, peak_rss_mb
from .context import P3Context
from .curves import BinaryCurve


@dataclass
class CNN3Spec(CNNSpec):
    ignore_px: int = 2
    patience: int = 6
    min_epochs: int = 5
    bce_weight: float = 0.1               # BCE share inside loss="dti"
    rot180: bool = True
    select_by: str = "binary_area"        # inner-val checkpoint metric: "binary_area" | "raw"
    infer_stride_frac: float = 0.5
    # GPU throughput only (see module docstring)
    data_on_gpu: bool = True
    amp_dtype: str = "auto"               # "auto" (bf16 if supported) | "bf16" | "fp16"
    channels_last: bool = True
    cudnn_benchmark: bool = True          # False = deterministic cuDNN kernels (slower)
    eval_every: int = 1                   # inner-val evaluation every n epochs
    infer_batch: int = 32


def cnn_a_spec(channels, **kw) -> CNN3Spec:
    base = dict(encoder="builtin", base_width=32, loss="bce_dice", patch_px=256, epochs=40, steps_per_epoch=150,
                batch_size=16, lr=1e-3, pos_patch_frac=0.5, norm="robust")
    return CNN3Spec(kw.pop("name", "CNN_A"), list(channels), **{**base, **kw})


# ---------------------------------------------------------------------------
# targets, weights, loss
# ---------------------------------------------------------------------------
def build_targets(labels_visible: np.ndarray, grid, ignore_px: int, radius_m: float = 300.0):
    """(binary target, kernel target k(d), ignore band) from visible labels only."""
    px = (grid.pixel_km_y * 1000, grid.pixel_km_x * 1000)
    d = ndimage.distance_transform_edt(~labels_visible, sampling=px)
    kern = np.clip(1 - d / radius_m, 0, 1).astype(np.float16)
    band = (d > 0) & (d <= ignore_px * min(px) + 1e-6)
    return labels_visible.astype(np.float16), kern, band


@lru_cache(maxsize=None)
def _offsets(pixel_m: tuple[float, float]):
    offs = kernel_offsets(300.0, pixel_m)
    return offs, max(max(abs(a), abs(b)) for a, b, _ in offs)


def dti_surrogate(logits, truth, kern, weight, alpha=0.2, beta=0.8, bce_weight=0.1, pixel_m=(100.0, 100.0)):
    import torch
    import torch.nn.functional as F

    p = torch.sigmoid(logits)
    offs, r = _offsets(tuple(pixel_m))
    H, W = p.shape[-2:]
    P = F.pad(p, (r, r, r, r))
    M = None
    for dr, dc, k in offs:                 # running max: same value as stack(...).amax(0), no stacked copy
        t = k * P[..., r + dr:r + dr + H, r + dc:r + dc + W]
        M = t if M is None else torch.maximum(M, t)
    tw = truth * weight
    tp = (M * tw).sum()
    n = tw.sum()
    fp = (p * (1 - kern) * weight).sum()
    loss = 1 - (tp + 1) / (tp + alpha * fp + beta * (n - tp) + 1)
    if bce_weight:
        bce = F.binary_cross_entropy_with_logits(logits, truth, weight=weight, reduction="sum") / weight.sum().clamp_min(1)
        loss = loss + bce_weight * bce
    return loss


def loss_fn(logits, y, k, w, spec: CNN3Spec):
    from ..cnn import masked_loss

    if spec.loss == "dti":
        return dti_surrogate(logits, y, k, w, spec.tversky_alpha, spec.tversky_beta, spec.bce_weight)
    return masked_loss(logits, y, w, spec)


def make_batch3(stack, target, kern, weight, sampler: PatchSampler, spec: CNN3Spec, rng):
    import torch

    s, xs, ys, ks, ws = spec.patch_px, [], [], [], []
    while len(xs) < spec.batch_size:
        rc = sampler.draw(rng.random() < spec.pos_patch_frac) or sampler.draw(False, 1000)
        if rc is None:
            raise RuntimeError("no admissible patch: training zone too fragmented for this patch size")
        r0, c0 = rc
        arrs = [stack[:, r0:r0 + s, c0:c0 + s], target[None, r0:r0 + s, c0:c0 + s],
                kern[None, r0:r0 + s, c0:c0 + s], weight[None, r0:r0 + s, c0:c0 + s]]
        if spec.rot180 and rng.random() < 0.5:
            arrs = [a[:, ::-1, ::-1] for a in arrs]
        x, y, k, w = (np.ascontiguousarray(a, dtype=np.float32) for a in arrs)
        xs.append(x), ys.append(y), ks.append(k), ws.append(w)
    return tuple(torch.from_numpy(np.stack(a)) for a in (xs, ys, ks, ws))


def make_batch_gpu(gstack, gaux, sampler: PatchSampler, spec: CNN3Spec, rng):
    """``make_batch3`` on device-resident tensors; same RNG call order, hence the same patches.

    ``gaux`` = (3, H, W) [target, kern, weight]. Returns float32 (x, y, k, w) on the device.
    """
    import torch

    s, xs, auxs = spec.patch_px, [], []
    while len(xs) < spec.batch_size:
        rc = sampler.draw(rng.random() < spec.pos_patch_frac) or sampler.draw(False, 1000)
        if rc is None:
            raise RuntimeError("no admissible patch: training zone too fragmented for this patch size")
        r0, c0 = rc
        x, a = gstack[:, r0:r0 + s, c0:c0 + s], gaux[:, r0:r0 + s, c0:c0 + s]
        if spec.rot180 and rng.random() < 0.5:
            x, a = x.flip((-2, -1)), a.flip((-2, -1))
        xs.append(x), auxs.append(a)
    x = torch.stack(xs).float()
    a = torch.stack(auxs).float()
    return x, a[:, 0:1], a[:, 1:2], a[:, 2:3]


def device_stack(stack, spec: CNN3Spec, device):
    """Move the (C, H, W) fp16 stack to the GPU once; numpy stays numpy if disabled / no CUDA / OOM."""
    import torch

    if torch.is_tensor(stack) or not (spec.data_on_gpu and device.type == "cuda"):
        return stack
    try:
        return torch.from_numpy(np.ascontiguousarray(stack, dtype=np.float16)).to(device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"stack ({stack.nbytes / 2**30:.2f} GiB) does not fit on the GPU -> host-side patch path")
        return stack


def _amp_dtype(spec: CNN3Spec, device):
    import torch

    if not (spec.amp and device.type == "cuda"):
        return None
    if spec.amp_dtype == "bf16" or (spec.amp_dtype == "auto" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    return torch.float16


def _to_model_layout(x, spec: CNN3Spec, device):
    import torch

    return x.contiguous(memory_format=torch.channels_last) if spec.channels_last and device.type == "cuda" else x


def predict_tiled_gpu(model, gstack, box, patch: int, stride: int, spec: CNN3Spec, device) -> np.ndarray:
    """``cnn.predict_tiled`` with tiles cut from a device-resident stack and blended on the device."""
    import torch

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
    acc = torch.zeros((R1 - R0, C1 - C0), dtype=torch.float32, device=device)
    wsum = torch.zeros_like(acc)
    win = torch.from_numpy(_window(patch)).to(device)
    tiles = [(y, x) for y in ys for x in xs]
    amp = _amp_dtype(spec, device)
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(tiles), spec.infer_batch):
            chunk = tiles[i:i + spec.infer_batch]
            xb = torch.zeros((len(chunk), C, patch, patch), dtype=gstack.dtype, device=device)
            for j, (y, x) in enumerate(chunk):
                ya, yb, xa, xb_ = max(y, 0), min(y + patch, H), max(x, 0), min(x + patch, W)
                if ya < yb and xa < xb_:
                    xb[j, :, ya - y:yb - y, xa - x:xb_ - x] = gstack[:, ya:yb, xa:xb_]
            with torch.autocast(device_type="cuda", dtype=amp or torch.float16, enabled=amp is not None):
                p = torch.sigmoid(model(_to_model_layout(xb.float(), spec, device))).float()[:, 0]
            for j, (y, x) in enumerate(chunk):
                acc[y - R0:y - R0 + patch, x - C0:x - C0 + patch] += p[j] * win
                wsum[y - R0:y - R0 + patch, x - C0:x - C0 + patch] += win
    out = acc / wsum.clamp_min(1e-6)
    return out[m:m + (r1 - r0), m:m + (c1 - c0)].cpu().numpy()


def predict_mask3(model, stack, mask, spec: CNN3Spec, device, stride_frac: float = 0.5) -> np.ndarray:
    """``cnn.predict_mask`` that dispatches to the GPU path when ``stack`` is a device tensor."""
    import torch

    if not torch.is_tensor(stack):
        return predict_mask(model, stack, mask, spec, device, stride_frac)
    out = np.full(mask.shape, np.nan, np.float32)
    if not mask.any():
        return out
    r0, r1, c0, c1 = bbox(mask)
    p = predict_tiled_gpu(model, stack, (r0, r1, c0, c1), spec.patch_px, int(spec.patch_px * stride_frac), spec, device)
    sub = out[r0:r1, c0:c1]
    sub[mask[r0:r1, c0:c1]] = p[mask[r0:r1, c0:c1]]
    return out


def _seed_all(seed: int, benchmark: bool = False):
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = not benchmark
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cuda.matmul.allow_tf32 = benchmark
    torch.backends.cudnn.allow_tf32 = benchmark


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train3(spec: CNN3Spec, stack, target, kern, weight, allowed, positives, valid, device, rng, *,
           inner_mask=None, inner_truth=None, fixed_epochs: int | None = None, log_prefix="", ckpt_path=None):
    """Train; with ``inner_mask`` keep the best inner-val checkpoint and stop after ``patience`` stale epochs."""
    import torch

    _seed_all(spec.seed, spec.cudnn_benchmark and device.type == "cuda")
    model = build_model(spec).to(device)
    if spec.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)
    epochs = fixed_epochs or spec.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=spec.lr, total_steps=epochs * spec.steps_per_epoch)
    amp = _amp_dtype(spec, device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16)     # bf16 needs no loss scaling
    sampler = PatchSampler(allowed, valid, positives, spec.patch_px, spec.min_valid_frac, rng)
    on_gpu = torch.is_tensor(stack)
    if on_gpu:
        gaux = torch.from_numpy(np.stack([target, kern, weight]).astype(np.float16)).to(device)
    best, best_ep, best_state, stale, hist = -1.0, 0, None, 0, []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for ep in range(epochs):
        model.train()
        t0 = time.perf_counter()
        loss_sum = torch.zeros((), device=device)
        for _ in range(spec.steps_per_epoch):
            if on_gpu:
                x, y, k, w = make_batch_gpu(stack, gaux, sampler, spec, rng)
            else:
                x, y, k, w = (t.to(device, non_blocking=True) for t in make_batch3(stack, target, kern, weight, sampler, spec, rng))
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp or torch.float16, enabled=amp is not None):
                logits = model(_to_model_layout(x, spec, device))
            loss = loss_fn(logits.float(), y, k, w, spec)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.detach()
        row = {"epoch": ep + 1, "train_loss": float(loss_sum) / spec.steps_per_epoch, "lr": float(sched.get_last_lr()[0]),
               "epoch_s": round(time.perf_counter() - t0, 1), "rss_mb": peak_rss_mb()}
        evaluate = (ep + 1) % max(spec.eval_every, 1) == 0 or ep + 1 == epochs
        if evaluate and inner_mask is not None and inner_mask.any():
            t1 = time.perf_counter()
            pv = predict_mask3(model, stack, inner_mask, spec, device, spec.infer_stride_frac)
            c = BinaryCurve(pv, inner_truth, inner_mask)
            b = c.best(np.geomspace(0.003, 0.5, 30))
            from ..metric import dti_components

            row.update(inner_dti_binary_best=b["dti"], inner_best_frac=b["target_frac"],
                       inner_dti_raw=dti_components(pv, inner_truth, eval_mask=inner_mask).dti,
                       inner_eval_s=round(time.perf_counter() - t1, 1))
            score = row["inner_dti_binary_best"] if spec.select_by == "binary_area" else row["inner_dti_raw"]
            if score > best:
                best, best_ep, stale = score, ep + 1, 0
                best_state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
            else:
                stale += 1
        if device.type == "cuda":
            row["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
        hist.append(row)
        print(f"{log_prefix} ep {ep + 1:3d} loss {row['train_loss']:.4f} inner {row.get('inner_dti_binary_best', float('nan')):.4f}"
              f"  ({row['epoch_s']:.0f}s train, {row.get('inner_eval_s', 0):.0f}s eval)")
        if inner_mask is not None and ep + 1 >= spec.min_epochs and stale >= spec.patience:
            break
    if on_gpu:
        del gaux
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_ep = len(hist)
    if ckpt_path is not None:
        torch.save({"state_dict": model.state_dict(), "spec": asdict(spec), "best_inner": best, "best_epoch": best_ep}, ckpt_path)
    return model, pd.DataFrame(hist), best_ep


def prepare_stack(spec: CNN3Spec, ctx: P3Context):
    stack = ctx.store.load_stack(spec.channels, dtype=np.float16)
    return normalise_stack(stack, ctx.data.valid, spec.norm)


def run_cnn3_cv(ctx: P3Context, exp_id: str, spec: CNN3Spec, *, input_config: str = "", overwrite: bool = False,
                stack=None) -> ExperimentRun:
    import torch

    from .lgbm3 import finish_variant

    run = ExperimentRun(ctx.paths, exp_id, "cnn", {**asdict(spec), "input_config": input_config,
                                                   "folds": ctx.bench.rel(ctx.folds.directory)})
    if run.done and not overwrite:
        print(f"[{exp_id}] already done -> {run.dir}")
        return run
    run.start()
    device = device_of()
    rng = np.random.default_rng(spec.seed)
    if stack is None:
        with run.timer("load_stack"):
            stack, stats = prepare_stack(spec, ctx)
        (run.models_dir / "normalisation.json").write_text(json.dumps({"channels": spec.channels, **stats}), encoding="utf-8")
    stack = device_stack(stack, spec, device)
    print(f"[{exp_id}] device={device} | stack on GPU: {torch.is_tensor(stack)} | amp={_amp_dtype(spec, device)}")
    data = ctx.data
    oof = np.full(data.shape, np.nan, np.float32)
    hist_all, met = [], []
    for k in ctx.fold_ids:
        allowed = ctx.folds.inner_train(k)
        visible = data.labels & ctx.folds.train(k)
        target, kern, band = build_targets(visible, data.grid, spec.ignore_px)
        weight = (allowed & data.valid & ~band).astype(np.float16)
        with run.timer(f"fold{k}_train"):
            model, hist, best_ep = train3(spec, stack, target, kern, weight, allowed, visible, data.valid, device, rng,
                                          inner_mask=ctx.folds.inner_val(k) & data.valid, inner_truth=visible,
                                          log_prefix=f"[{exp_id} f{k}]", ckpt_path=run.models_dir / f"fold{k}.pt")
        hist["fold"] = k
        hist_all.append(hist)
        val = ctx.folds.val(k)
        with run.timer(f"fold{k}_predict"):
            oof[val] = predict_mask3(model, stack, val, spec, device, spec.infer_stride_frac)[val]
        from ..metric import full_diagnostics

        diag = full_diagnostics(oof, data.labels, val & data.valid, rng, ctx.cfg.recall_tolerances_m, ctx.cfg.report_threshold)
        met.append({"fold": k, "best_iteration": best_ep, "epochs_run": len(hist),
                    "gpu_peak_mb": float(hist.gpu_peak_mb.max()) if "gpu_peak_mb" in hist else np.nan, **diag})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_geotiff(run.oof_path, {"oof_probability": oof}, data.grid)
    run.register("oof", run.oof_path, "raster")
    return finish_variant(ctx, run, oof, hist_all, [], met, [])


def run_cnn3_discovery(ctx: P3Context, exp_id: str, spec: CNN3Spec, d: int, epochs: int, *, overwrite: bool = False,
                       stack=None) -> np.ndarray:
    """Whole-map model for discovery split d: withheld systems + ring weight 0, fixed epoch budget."""
    from ..ensemble import read_map

    run = ExperimentRun(ctx.paths, exp_id, "cnn", {**asdict(spec), "discovery_split": d, "epochs_fixed": epochs})
    pred_path = run.dir / "pred.tif"
    if run.state.get("status") == "done" and pred_path.exists() and not overwrite:
        return read_map(pred_path)
    run.start()
    device = device_of()
    rng = np.random.default_rng(spec.seed + d)
    if stack is None:
        stack, _ = prepare_stack(spec, ctx)
    stack = device_stack(stack, spec, device)
    data, ign = ctx.data, ctx.splits.ignore(d)
    visible = data.labels & ~ign
    target, kern, band = build_targets(visible, data.grid, spec.ignore_px)
    weight = (data.valid & ~ign & ~band).astype(np.float16)
    with run.timer("train"):
        model, hist, _ = train3(spec, stack, target, kern, weight, data.valid, visible, data.valid, device, rng,
                                fixed_epochs=epochs, log_prefix=f"[{exp_id}]", ckpt_path=run.models_dir / "model.pt")
    run.save_table(hist, "history")
    zone = ctx.splits.eval_zone(d) & data.valid
    with run.timer("predict"):
        pred = predict_mask3(model, stack, zone, spec, device, spec.infer_stride_frac)
    write_geotiff(pred_path, {"probability": pred}, data.grid)
    run.register("pred", pred_path, "raster")
    run.state.update(status="done")
    run._flush()
    return pred


def load_cnn3_fold_models(run: ExperimentRun, device=None):
    import torch

    device = device or device_of()
    out = []
    for ck in sorted(run.models_dir.glob("fold*.pt")):
        obj = torch.load(ck, map_location=device, weights_only=False)
        spec = CNN3Spec(**obj["spec"])
        m = build_model(spec).to(device)
        m.load_state_dict(obj["state_dict"])
        m.eval()
        out.append((spec, m))
    return out
