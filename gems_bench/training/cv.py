"""Geographic CV and pseudo-discovery drivers for Phase-5 CNNs, writing into a Phase-5 RunDir.

Protocol identical to Phase 4 (``run_cnn4_cv`` / ``run_cnn4_discovery``):
* CV fold k: patches inside ``folds.inner_train(k)``; targets from labels visible in ``folds.train(k)``;
  checkpoint on ``folds.inner_val(k)``; OOF on ``folds.val(k)``. Context branches see a strict
  (training-zone-only) pooled stack during training, the full stack at OOF inference.
* discovery split d: whole valid area, withheld systems + 1 km ring at weight 0, fixed epochs = median best
  CV epoch (Phase-3 rule), prediction inside ``splits.eval_zone(d)``.

Units (``cv/fold<k>``, ``disc/<d>``) are resumable and skipped when their final model and prediction raster
exist, so a disconnect costs at most ``ckpt_every`` epochs of one unit.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from ..artifacts.store import RunDir
from ..cnn import device_of
from ..metric import full_diagnostics
from ..p3 import cnn3
from ..phase4_training import pooled_context, predict_mask4, to_tensor_stack
from .checkpoint import CheckpointManager
from .spec import spec_hash
from .trainer import load_unit_model, train5


def ensure_channels(ctx, spec) -> None:
    missing = [c for c in spec.channels if c in ctx.store.specs and not ctx.store.has(c)]
    unknown = [c for c in spec.channels if c not in ctx.store.specs]
    if unknown:
        raise KeyError(f"channels unknown to the feature store: {unknown}")
    if missing:
        print(f"[channels] building {len(missing)} missing features")
        ctx.store.build(missing, verbose=True)


def prepare_stack5(ctx, spec, rd: RunDir | None = None):
    ensure_channels(ctx, spec)
    stack, stats = cnn3.prepare_stack(spec, ctx)
    if rd is not None:
        rd.save_json({"channels": spec.channels, **stats}, "normalisation", sub="checkpoints")
    return stack


def background_weight_map(ctx, spec) -> np.ndarray | None:
    if spec.pu != "bg_weight":
        return None
    from ..losses.pu import background_weights, resolve_evidence

    ev = resolve_evidence(ctx, spec.bg_evidence)
    return background_weights(ev, ctx.data.valid, lam=spec.bg_lambda, q0=spec.bg_q0, w_min=spec.bg_min_weight)


def _sync_unit(sync, rd: RunDir) -> None:
    if sync is not None:
        try:
            sync.push([rd.path.resolve().relative_to(sync.local_root.resolve()).as_posix()])
        except OSError as e:
            print(f"[sync] push failed ({e}); artifacts remain local until the next push")


def run_cv5(ctx, rd: RunDir, spec, *, sync=None, deadline: float | None = None, stack=None,
            mark=None) -> np.ndarray:
    """Train / resume every CV fold; returns the stitched OOF map (predictions/oof.tif)."""
    import torch

    from ..losses.compose import build_targets5

    device = device_of()
    data, folds = ctx.data, ctx.folds
    sh = spec_hash(spec)
    gstack = to_tensor_stack(stack if stack is not None else prepare_stack5(ctx, spec, rd), device)
    gctx_full = pooled_context(gstack, spec.context_factor) if spec.context else None
    bgw = background_weight_map(ctx, spec)
    for k in ctx.fold_ids:
        unit = f"cv_fold{k}"
        ck = CheckpointManager(rd.checkpoints, unit, sync=sync, sync_every_min=spec.ckpt_sync_min, spec_hash=sh)
        if ck.has_final() and rd.raster_path(f"oof_fold{k}").exists():
            print(f"[{unit}] done -> skip")
            continue
        t0 = time.perf_counter()
        allowed = folds.inner_train(k)
        visible = data.labels & folds.train(k)
        aux, names = build_targets5(visible, data.grid, spec, allowed & data.valid, bg_weights=bgw)
        gaux = torch.from_numpy(aux).to(gstack.device)
        gctx_train = gctx_full
        if spec.context and spec.context_strict:
            gctx_train = pooled_context(gstack, spec.context_factor, folds.train(k) & data.valid)
        if ck.has_final():
            spec_k, model = load_unit_model(ck, device)
            hist, best_ep = pd.DataFrame(), ck.load_final().get("best_epoch")
        else:
            model, hist, best_ep = train5(spec, gstack, gaux, names, allowed, visible, data.valid, device,
                                          np.random.default_rng(spec.seed + 1000 * k),
                                          gctx_train=gctx_train, gctx_eval=gctx_train,
                                          inner_mask=folds.inner_val(k) & data.valid, inner_truth=visible,
                                          log_prefix=f"[{rd.path.name} f{k}]", ckpt=ck, deadline=deadline)
        train_s = time.perf_counter() - t0
        val = folds.val(k)
        t1 = time.perf_counter()
        pred = predict_mask4(model, gstack, val, spec, device, gctx_full)
        rd.save_raster({"oof_probability": np.where(val, pred, np.nan).astype(np.float32)}, f"oof_fold{k}", data.grid)
        diag = full_diagnostics(np.where(val, pred, np.nan), data.labels, val & data.valid,
                                np.random.default_rng(spec.seed), ctx.cfg.recall_tolerances_m, ctx.cfg.report_threshold)
        if len(hist):
            rd.append_table(hist.assign(unit=unit, fold=k), "training_history", key=["unit", "epoch"])
        rd.append_table(pd.DataFrame([{"unit": unit, "fold": k, "best_epoch": best_ep, "epochs_run": len(hist),
                                       "train_s": round(train_s, 1), "predict_s": round(time.perf_counter() - t1, 1),
                                       "gpu_peak_mb": float(hist.gpu_peak_mb.max()) if "gpu_peak_mb" in hist else np.nan,
                                       **diag}]), "fold_metrics", key=["unit"])
        if mark:
            mark(unit, {"best_epoch": best_ep})
        del model, gaux
        if spec.context and spec.context_strict:
            del gctx_train
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _sync_unit(sync, rd)
    oof = np.full(data.shape, np.nan, np.float32)
    for k in ctx.fold_ids:
        p = rd.read_raster(f"oof_fold{k}")
        v = folds.val(k)
        oof[v] = p[v]
    rd.save_raster({"oof_probability": oof}, "oof", data.grid)
    return oof


def discovery_epochs(rd: RunDir, min_epochs: int) -> int:
    fm = rd.load_table("fold_metrics")
    it = [x for x in fm.best_epoch.dropna().astype(float) if x == x]
    return max(min_epochs, int(np.median(it))) if it else min_epochs


def run_discovery5(ctx, rd: RunDir, spec, epochs: int, *, sync=None, deadline: float | None = None, stack=None,
                   mark=None) -> dict[int, np.ndarray]:
    import torch

    from ..losses.compose import build_targets5

    device = device_of()
    data, splits = ctx.data, ctx.splits
    sh = spec_hash(spec)
    gstack = to_tensor_stack(stack if stack is not None else prepare_stack5(ctx, spec, rd), device)
    gctx = pooled_context(gstack, spec.context_factor) if spec.context else None
    bgw = background_weight_map(ctx, spec)
    out = {}
    for d in range(splits.n_splits):
        unit = f"disc{d}"
        ck = CheckpointManager(rd.checkpoints, unit, sync=sync, sync_every_min=spec.ckpt_sync_min, spec_hash=sh)
        if ck.has_final() and rd.raster_path(unit).exists():
            out[d] = rd.read_raster(unit)
            continue
        ign = splits.ignore(d)
        visible = data.labels & ~ign
        aux, names = build_targets5(visible, data.grid, spec, data.valid & ~ign, bg_weights=bgw)
        gaux = torch.from_numpy(aux).to(gstack.device)
        t0 = time.perf_counter()
        if ck.has_final():
            _, model = load_unit_model(ck, device)
            hist = pd.DataFrame()
        else:
            model, hist, _ = train5(spec, gstack, gaux, names, data.valid, visible, data.valid, device,
                                    np.random.default_rng(spec.seed + d), gctx_train=gctx, fixed_epochs=epochs,
                                    log_prefix=f"[{rd.path.name} disc{d}]", ckpt=ck, deadline=deadline)
        zone = splits.eval_zone(d) & data.valid
        pred = predict_mask4(model, gstack, zone, spec, device, gctx)
        rd.save_raster({"probability": pred}, unit, data.grid)
        if len(hist):
            rd.append_table(hist.assign(unit=unit, fold=-1), "training_history", key=["unit", "epoch"])
        rd.append_table(pd.DataFrame([{"unit": unit, "split": d, "epochs_fixed": epochs,
                                       "train_s": round(time.perf_counter() - t0, 1)}]), "discovery_runs", key=["unit"])
        if mark:
            mark(unit, {"epochs": epochs})
        out[d] = pred
        del model, gaux
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _sync_unit(sync, rd)
    return out


def runtime_tables(rd: RunDir) -> None:
    """runtime_profile.csv and gpu_memory_profile.csv from the epoch history (regenerable at any time)."""
    if not rd.has_table("training_history"):
        return
    h = rd.load_table("training_history")
    agg = {"epochs": ("epoch", "count"), "train_s": ("epoch_s", "sum"), "mean_epoch_s": ("epoch_s", "mean")}
    if "inner_eval_s" in h:
        agg["inner_eval_s"] = ("inner_eval_s", "sum")
    rd.save_table(h.groupby("unit").agg(**agg).reset_index(), "runtime_profile")
    cols = [c for c in ("unit", "epoch", "gpu_peak_mb", "gpu_reserved_mb", "batch_size", "grad_accum", "oom_events") if c in h]
    rd.save_table(h[cols], "gpu_memory_profile")
    rd.save_table(h, "epoch_metrics")


def write_spec(rd: RunDir, spec) -> None:
    from dataclasses import asdict

    rd.save_text(json.dumps(asdict(spec), indent=2, default=str), "spec.json")
