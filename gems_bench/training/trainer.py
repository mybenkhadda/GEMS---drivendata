"""Phase-5 trainer: ``phase4_training.train4`` generalised, resumable and Colab-robust.

Unchanged from train4 (so a P5Spec with Phase-5 features off is a Phase-4 run): PatchSampler (patches fully
inside the allowed zone), RNG call order of the batch builder (``make_batch4``), OneCycle, AdamW with an
encoder learning-rate group, bf16/fp16 autocast, channels_last, inner-val *binary-area* DTI checkpointing
with patience, tiled Hann-blended inference (``predict_mask4``).

Added
-----
* objective ``losses.loss5`` (PU / geometry terms) and an optional EMA mean teacher (B3);
* full-state resume from ``CheckpointManager`` (model, optimizer, scheduler, scaler, teacher, NumPy / torch /
  CUDA RNG, best-so-far, history, current batch and accumulation): an interrupted Colab session continues at
  the last saved epoch with an identical sample stream;
* checkpoint selection on the best *finite* inner-validation score (a NaN epoch counts as stale, never best);
  without an inner-validation zone (pseudo-discovery: fixed epochs) the final model is the last epoch;
* ``deadline`` (``time.monotonic`` seconds): stop at an epoch boundary, checkpoint, raise TrainingInterrupted;
* CUDA OOM back-off: halve the batch, double ``grad_accum`` (effective batch unchanged), up to ``oom_retries``;
* collapse guard: stop when the mean probability on unlabeled training pixels exceeds
  ``collapse_max_unlabeled_mean`` (a model drifting towards "fault everywhere"; matters for B1/B2);
* per-epoch GPU memory, loss parts and unlabeled-mean logging.
"""
from __future__ import annotations

import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from ..cnn import PatchSampler
from ..p3 import cnn3
from ..p3.curves import BinaryCurve
from ..phase4_training import _forward, make_batch4, predict_mask4


class TrainingInterrupted(RuntimeError):
    """Raised at an epoch boundary when the runtime budget is spent; the checkpoint is already saved."""


def improved(score, best) -> bool:
    """Checkpoint-selection rule: only a finite score strictly above the best so far (NaN / inf never wins)."""
    try:
        return bool(np.isfinite(score) and score > best)
    except TypeError:
        return False


def _rng_state(rng) -> dict:
    import torch

    s = {"numpy": rng.bit_generator.state, "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def _set_rng_state(rng, s: dict) -> None:
    import torch

    rng.bit_generator.state = s["numpy"]
    torch.set_rng_state(s["torch"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def train5(spec, gstack, gaux, aux_names: list[str], allowed, positives, valid, device, rng, *,
           gctx_train=None, gctx_eval=None, inner_mask=None, inner_truth=None, fixed_epochs: int | None = None,
           log_prefix: str = "", ckpt=None, deadline: float | None = None, pretrained_path: str | None = None):
    """Returns (model with the best inner-val state or the last state, history DataFrame, best epoch)."""
    import torch

    from ..losses.compose import loss5
    from ..losses.consistency import EMATeacher, consistency_loss, rampup, shifted_view
    from ..metric import dti_components
    from ..models.factory import build_model5, count_params

    cnn3._seed_all(spec.seed, spec.cudnn_benchmark and device.type == "cuda")
    model = build_model5(spec)
    if pretrained_path and spec.model == "geounet":
        from ..geological_encoders import load_encoder_weights

        load_encoder_weights(model, pretrained_path, spec.channels)
    model = model.to(device)
    if spec.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    enc_ids = {id(p) for p in model.encoder_parameters()}
    enc = [p for p in model.parameters() if id(p) in enc_ids]
    rest = [p for p in model.parameters() if id(p) not in enc_ids]
    groups = [{"params": enc, "lr": spec.lr * spec.encoder_lr_mult}, {"params": rest, "lr": spec.lr}]
    opt = torch.optim.AdamW(groups, lr=spec.lr, weight_decay=spec.weight_decay)
    epochs = fixed_epochs or spec.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups],
                                                total_steps=epochs * spec.steps_per_epoch)
    amp = cnn3._amp_dtype(spec, device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16)
    sampler = PatchSampler(allowed, valid, positives, spec.patch_px, spec.min_valid_frac, rng)
    teacher = EMATeacher(model, spec.ema_decay) if spec.consistency > 0 else None
    it, iw = aux_names.index("target"), aux_names.index("weight")

    st = {"epoch": 0, "best": -1.0, "best_ep": 0, "best_state": None, "stale": 0, "hist": [],
          "batch_size": spec.batch_size, "grad_accum": max(int(spec.grad_accum), 1), "oom_events": 0}
    if ckpt is not None:
        s = ckpt.latest()
        if s is not None:
            model.load_state_dict(s["model"])
            opt.load_state_dict(s["optimizer"])
            sched.load_state_dict(s["scheduler"])
            scaler.load_state_dict(s["scaler"])
            if teacher is not None and s.get("teacher"):
                teacher.load_state_dict(s["teacher"])
            _set_rng_state(rng, s["rng"])
            st.update({k: s[k] for k in st if k in s})
            print(f"{log_prefix} resumed at epoch {st['epoch']} (best {st['best']:.4f} @ {st['best_ep']})")
    run_spec = replace(spec, batch_size=st["batch_size"])

    def checkpoint(force_sync=False):
        if ckpt is None:
            return
        ckpt.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                   "scaler": scaler.state_dict(), "teacher": teacher.state_dict() if teacher else None,
                   "rng": _rng_state(rng), **{k: st[k] for k in st}}, st["epoch"], force_sync=force_sync)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for ep in range(st["epoch"], epochs):
        frozen = ep < spec.freeze_encoder_epochs
        for p in enc:
            p.requires_grad_(not frozen)
        model.train()
        t0 = time.perf_counter()
        loss_sum = torch.zeros((), device=device)
        part_sum: dict = {}
        step = 0
        while step < spec.steps_per_epoch:
            try:
                opt.zero_grad(set_to_none=True)
                for _a in range(st["grad_accum"]):
                    x, a, c = make_batch4(gstack, gaux, sampler, run_spec, rng, device, gctx_train)
                    out = _forward(model, x, c, run_spec, device, amp)
                    loss, parts = loss5(out, a, aux_names, run_spec, epoch=ep, return_parts=True)
                    if teacher is not None:
                        v, sh = shifted_view(x, rng, spec.consistency_shift_px, spec.consistency_noise)
                        with torch.no_grad():
                            tout = _forward(teacher.model, v, c, run_spec, device, amp)
                        lc = consistency_loss(out["seg"], tout["seg"], sh, a[:, it:it + 1], a[:, iw:iw + 1])
                        loss = loss + spec.consistency * rampup(ep, spec.consistency_rampup) * lc
                        parts["consistency"] = lc.detach()
                    loss = loss / st["grad_accum"]
                    scaler.scale(loss).backward()
                    loss_sum += loss.detach()
                    for k, v_ in parts.items():
                        part_sum[k] = part_sum.get(k, 0) + v_ / st["grad_accum"]
                scaler.step(opt)
                scaler.update()
                sched.step()
                if teacher is not None:
                    teacher.update(model)
                step += 1
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if st["oom_events"] >= spec.oom_retries or st["batch_size"] <= 2:
                    raise
                st["oom_events"] += 1
                st["batch_size"] //= 2
                st["grad_accum"] *= 2
                run_spec = replace(run_spec, batch_size=st["batch_size"])
                print(f"{log_prefix} CUDA OOM -> batch {st['batch_size']} x accum {st['grad_accum']}")
        row = {"epoch": ep + 1, "train_loss": float(loss_sum) / spec.steps_per_epoch,
               "lr": float(sched.get_last_lr()[-1]), "encoder_frozen": frozen,
               "epoch_s": round(time.perf_counter() - t0, 1), "batch_size": st["batch_size"],
               "grad_accum": st["grad_accum"], "oom_events": st["oom_events"],
               **{f"loss_{k}": float(v) / spec.steps_per_epoch for k, v in part_sum.items()}}
        if hasattr(model, "context_weight") and model.context_weight() is not None:
            row["context_gate"] = model.context_weight()
        evaluate = (ep + 1) % max(spec.eval_every, 1) == 0 or ep + 1 == epochs
        if evaluate and inner_mask is not None and inner_mask.any():
            t1 = time.perf_counter()
            pv = predict_mask4(model, gstack, inner_mask, run_spec, device, gctx_eval)
            b = BinaryCurve(pv, inner_truth, inner_mask).best(np.geomspace(0.003, 0.5, 30))
            row.update(inner_dti_binary_best=b["dti"], inner_best_frac=b["target_frac"],
                       inner_dti_raw=dti_components(pv, inner_truth, eval_mask=inner_mask).dti,
                       inner_eval_s=round(time.perf_counter() - t1, 1))
            score = row["inner_dti_binary_best"] if spec.select_by == "binary_area" else row["inner_dti_raw"]
            row["inner_score_valid"] = bool(np.isfinite(score))
            if improved(score, st["best"]):
                st.update(best=score, best_ep=ep + 1, stale=0,
                          best_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            else:
                st["stale"] += 1
        if device.type == "cuda":
            row["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
            row["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 2**20
        collapsed = ep + 1 >= spec.min_epochs and row.get("loss_unlabeled_mean_p", 0) > spec.collapse_max_unlabeled_mean
        row["collapsed"] = bool(collapsed)
        st["hist"].append(row)
        st["epoch"] = ep + 1
        print(f"{log_prefix} ep {ep + 1:3d} loss {row['train_loss']:.4f} inner {row.get('inner_dti_binary_best', float('nan')):.4f}"
              f" unl.p {row.get('loss_unlabeled_mean_p', float('nan')):.3f} ({row['epoch_s']:.0f}s)")
        early = inner_mask is not None and ep + 1 >= spec.min_epochs and st["stale"] >= spec.patience
        last = ep + 1 == epochs or early or collapsed
        out_of_time = deadline is not None and time.monotonic() > deadline and not last
        if (ep + 1) % max(spec.ckpt_every, 1) == 0 or out_of_time:
            checkpoint(force_sync=out_of_time)
        if out_of_time:
            raise TrainingInterrupted(f"{log_prefix} runtime budget reached after epoch {ep + 1}; resumable")
        if collapsed:
            print(f"{log_prefix} STOP: mean p on unlabeled pixels {row['loss_unlabeled_mean_p']:.3f} > "
                  f"{spec.collapse_max_unlabeled_mean} (prediction collapse)")
            break
        if early:
            break
    if st["best_state"] is not None:
        model.load_state_dict(st["best_state"])
        best_ep = st["best_ep"]
    else:
        best_ep = len(st["hist"])
    if ckpt is not None:
        ckpt.save_final({"state_dict": model.state_dict(), "spec": asdict(spec), "best_inner": st["best"],
                         "best_epoch": best_ep, "n_params": count_params(model), "epochs_run": len(st["hist"]),
                         "run_batch_size": st["batch_size"], "run_grad_accum": st["grad_accum"]})
        ckpt.clear_resume()
    return model, pd.DataFrame(st["hist"]), best_ep


def load_unit_model(ckpt, device=None):
    """(spec, model in eval mode) from ``<unit>_model.pt``."""
    import torch

    from ..cnn import device_of
    from ..models.factory import build_model5
    from .spec import spec_from_dict

    device = device or device_of()
    obj = ckpt.load_final(map_location=device)
    spec = spec_from_dict(obj["spec"])
    m = build_model5(spec).to(device)
    m.load_state_dict(obj["state_dict"])
    m.eval()
    return spec, m
