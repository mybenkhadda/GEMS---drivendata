"""Staged PyTorch training engine (Phase 8).

One ``fit_unit`` call trains one model for one ``splits.TrainUnit``:

* deterministic seeding (python, numpy, torch, cuDNN flags), per-epoch sample lists derived from the seed;
* AMP (bf16 when supported, else fp16 + GradScaler), gradient accumulation, gradient clipping, AdamW,
  linear warm-up + cosine decay;
* inner validation on the unit's ``inner_val`` pixels (inside the training geography, never the held-out zone);
  checkpoint selection on the inner metric only;
* atomic checkpoints (model, optimiser, scheduler, scaler, RNG states, epoch, config and schema hashes);
  resume refuses a checkpoint whose hashes differ; early stopping; a wall-clock budget per unit;
* per-term loss history, peak GPU memory, samples per second, model metadata.

UNTESTED: written without execution.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np

from .contract import schema_hash
from .losses import CompositeLoss, dti_surrogate
from .model import build_model, count_parameters
from .multiscale import MultiScaleDataset, UnitInputs, collate, unit_targets


def unit_seed(base: int, unit_key: str) -> int:
    """Stable per-unit seed (crc32, not Python's salted ``hash``)."""
    import zlib

    return int(base) + zlib.crc32(unit_key.encode()) % 10_000


def seed_everything(seed: int, deterministic: bool = True) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(pref: str = "auto"):
    import torch

    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is not available (no silent CPU fallback)")
    return torch.device(pref)


def amp_setup(device, enabled: bool):
    import torch

    if not enabled or device.type != "cuda":
        return None, None
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda") if dtype == torch.float16 else None
    return dtype, scaler


def lr_lambda(total_steps: int, warmup_frac: float):
    warm = max(1, int(total_steps * warmup_frac))

    def f(step):
        if step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
    return f


def _to(batch, device):
    out = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k not in ("targets", "meta")}
    out["targets"] = {k: v.to(device, non_blocking=True) for k, v in batch["targets"].items()}
    return out


class Checkpoint:
    """Atomic save / resume with hash checks."""

    def __init__(self, root: Path, unit_key: str, cfg_hash: str, schema: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.unit, self.cfg_hash, self.schema = unit_key, cfg_hash, schema

    def path(self, tag: str) -> Path:
        return self.root / f"{self.unit}_{tag}.pt"

    def save(self, tag: str, state: dict) -> Path:
        import torch

        state = {**state, "cfg_hash": self.cfg_hash, "schema_hash": self.schema, "unit": self.unit,
                 "saved_unix": time.time()}
        p = self.path(tag)
        tmp = p.with_suffix(".partial")
        torch.save(state, tmp)
        os.replace(tmp, p)
        return p

    def load(self, tag: str, map_location="cpu"):
        import torch

        p = self.path(tag)
        if not p.exists():
            return None
        st = torch.load(p, map_location=map_location, weights_only=False)
        if st.get("cfg_hash") != self.cfg_hash or st.get("schema_hash") != self.schema:
            raise RuntimeError(f"{p.name}: config / schema hash mismatch; refusing to resume a different experiment")
        return st


def fit_unit(cfg, stack, data, unit, out_dir: Path, log=print, resume: bool = True) -> dict:
    """Train one unit. Returns metadata (paths, history, best epoch, timings). Requires train_enabled."""
    import torch
    from torch.utils.data import DataLoader

    if not (cfg.train_enabled and cfg.run_mode == "TRAIN"):
        raise PermissionError("training requires run_mode='TRAIN' and train_enabled=True")
    t_start = time.perf_counter()
    T = cfg.train
    seed = unit_seed(T.seed, unit.key)
    seed_everything(seed, T.deterministic)
    device = resolve_device(cfg.device)
    inputs = UnitInputs(stack, data, unit, cfg, seed)
    tg = unit_targets(unit.labels & (unit.train | unit.inner_val), cfg.loss.kernel_radius_px)
    strict = None
    if cfg.split.strict_context_embargo:
        from scipy import ndimage

        reach = int(cfg.data.regional.extent_m / 2 / cfg.data.local.resolution_m)
        strict = ndimage.distance_transform_edt(~unit.heldout) <= reach
    model = build_model(cfg, inputs).to(device)
    params = count_parameters(model)
    sh = schema_hash(stack.channels)
    ck = Checkpoint(Path(out_dir) / "checkpoints", unit.key, cfg.semantic_hash(), sh)
    opt = torch.optim.AdamW(model.parameters(), lr=T.lr, weight_decay=T.weight_decay)
    steps_per_epoch = max(1, cfg.data.samples_per_epoch // (T.batch_size * T.accumulation))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(T.epochs * steps_per_epoch, T.warmup_frac))
    amp_dtype, scaler = amp_setup(device, T.amp)
    loss_fn = CompositeLoss(cfg)
    start_epoch, best, best_epoch, bad, history = 0, -math.inf, -1, 0, []
    if resume:
        st = ck.load("latest", map_location=device)
        if st is not None:
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["optimizer"])
            sched.load_state_dict(st["scheduler"])
            if scaler is not None and st.get("scaler"):
                scaler.load_state_dict(st["scaler"])
            torch.set_rng_state(st["rng_torch"].cpu())          # map_location may have moved it to the GPU
            np.random.set_state(st["rng_numpy"])
            start_epoch, best, best_epoch, bad, history = st["epoch"] + 1, st["best"], st["best_epoch"], st["bad"], \
                st["history"]
            log(f"[g14.train] {unit.key}: resumed at epoch {start_epoch}")
    val_ds = MultiScaleDataset(inputs, tg, unit.inner_val, unit.ignore, cfg, cfg.data.val_samples, seed + 999,
                               augment=False, positive_frac=0.5)
    val_dl = DataLoader(val_ds, batch_size=T.batch_size, shuffle=False, num_workers=cfg.data.num_workers,
                        collate_fn=collate)
    for epoch in range(start_epoch, T.epochs):
        if (time.perf_counter() - t_start) / 60 > T.max_minutes_per_unit:
            log(f"[g14.train] {unit.key}: wall-clock budget reached at epoch {epoch}; stopping cleanly")
            break
        ds = MultiScaleDataset(inputs, tg, unit.train, unit.ignore, cfg, cfg.data.samples_per_epoch,
                               seed + 17 * epoch, augment=True, strict_context=strict)
        dl = DataLoader(ds, batch_size=T.batch_size, shuffle=True, num_workers=cfg.data.num_workers,
                        collate_fn=collate, drop_last=True,
                        generator=torch.Generator().manual_seed(seed + epoch))
        model.train()
        sums, n, t0 = {}, 0, time.perf_counter()
        opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for it, batch in enumerate(dl):
            b = _to(batch, device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(b)
            loss, logs = loss_fn(out, b["targets"])
            loss = loss / T.accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{unit.key}: non-finite loss at epoch {epoch} iter {it}: {logs}")
            (scaler.scale(loss) if scaler is not None else loss).backward()
            if (it + 1) % T.accumulation == 0:
                if scaler is not None:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), T.grad_clip)
                if scaler is not None:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
            for k, v in logs.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
        train_logs = {f"train_{k}": v / max(n, 1) for k, v in sums.items()}
        val = evaluate_inner(model, val_dl, loss_fn, device, amp_dtype, cfg)
        rec = {"epoch": epoch, **train_logs, **val, "lr": opt.param_groups[0]["lr"],
               "samples_per_s": n * T.batch_size / max(time.perf_counter() - t0, 1e-6),
               "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0}
        history.append(rec)
        score = val["val_dti_surrogate"] if T.select_metric == "val_dti_surrogate" else -val["val_loss"]
        if score > best:
            best, best_epoch, bad = score, epoch, 0
            ck.save("best", {"model": model.state_dict(), "epoch": epoch, "score": score})
        else:
            bad += 1
        ck.save("latest", {"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                           "scaler": scaler.state_dict() if scaler is not None else None,
                           "rng_torch": torch.get_rng_state(), "rng_numpy": np.random.get_state(), "epoch": epoch,
                           "best": best, "best_epoch": best_epoch, "bad": bad, "history": history})
        log(f"[g14.train] {unit.key} epoch {epoch}: loss {train_logs.get('train_total', float('nan')):.4f} "
            f"val {val['val_loss']:.4f} surrogate {val['val_dti_surrogate']:.4f} (best {best_epoch})")
        if bad >= T.early_stopping_patience:
            log(f"[g14.train] {unit.key}: early stop at epoch {epoch}")
            break
    meta = {"unit": unit.key, "experiment": cfg.experiment, "cfg_hash": cfg.semantic_hash(), "schema_hash": sh,
            "channels": inputs.channel_names(), "modalities": inputs.modalities, "normalizer": inputs.norm.to_json(),
            "parameters": params, "best_epoch": best_epoch, "best_inner_score": best, "history": history,
            "seconds": time.perf_counter() - t_start, "device": str(device),
            "amp_dtype": str(amp_dtype) if amp_dtype is not None else None,
            "checkpoint_best": str(ck.path("best")), "selection_metric": T.select_metric,
            "selection_scope": "inner validation inside the unit's training geography"}
    p = Path(out_dir) / "manifests" / f"{unit.key}_train.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, indent=1, default=str))
    return meta


def evaluate_inner(model, dl, loss_fn, device, amp_dtype, cfg) -> dict:
    """Inner-validation loss and the DTI surrogate (diagnostic; the official scorer is used for held-out zones)."""
    import torch

    model.eval()
    tot, n, dsum = 0.0, 0, 0.0
    with torch.no_grad():
        for batch in dl:
            b = _to(batch, device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(b)
            loss, _ = loss_fn(out, b["targets"])
            tg = b["targets"]
            p = torch.sigmoid(out["fault"][:, 0].float())
            dsum += float(1 - dti_surrogate(p, tg["trace"], tg["loss_mask"], cfg.loss.kernel_radius_px))
            tot += float(loss)
            n += 1
    return {"val_loss": tot / max(n, 1), "val_dti_surrogate": dsum / max(n, 1)}


def load_unit_model(cfg, stack, data, unit, out_dir: Path, tag: str = "best"):
    """Rebuild the model of a trained unit from its manifest and checkpoint (same normaliser, same channels)."""
    import torch

    from .contract import UnitNormalizer

    meta = json.loads((Path(out_dir) / "manifests" / f"{unit.key}_train.json").read_text())
    inputs = UnitInputs(stack, data, unit, cfg, unit_seed(cfg.train.seed, unit.key))
    if meta["channels"] != inputs.channel_names():
        raise RuntimeError(f"{unit.key}: channel order differs from the trained model")
    nj = meta["normalizer"]                       # restore the exact training statistics
    inputs.norm = UnitNormalizer(np.asarray(nj["median"], np.float32), np.asarray(nj["iqr"], np.float32),
                                 nj["kinds"], nj["fitted_on"])
    model = build_model(cfg, inputs)
    ck = Checkpoint(Path(out_dir) / "checkpoints", unit.key, cfg.semantic_hash(), schema_hash(stack.channels))
    st = ck.load(tag)
    if st is None:
        raise FileNotFoundError(ck.path(tag))
    model.load_state_dict(st["model"])
    return model, inputs, meta
