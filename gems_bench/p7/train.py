"""Resumable GeoFaultFormer training per (experiment, unit) on a CUDA GPU (Colab L4 / A100 / T4).

* Seeds: unit seed = seed + hash(experiment / unit); sampler RNG lives on the single prefetch thread (fixed order).
* Mixed precision (bf16 where supported, else fp16 + GradScaler), micro-batch x accumulation = the semantic
  effective batch (preset-dependent split, same model), gradient clipping, OneCycle or cosine schedule, AdamW.
* Inner validation (cv units only): official-style DTI of the binary top ``select_frac`` map on the inner zone
  (labels of the inner zone only; the inner zone lies inside the training fold, never in the held-out fold).
  Every epoch records an explicit status:
      NOT_CONFIGURED          the unit has no inner zone (discovery units, by design)
      NO_ELIGIBLE_VALIDATION  the inner zone has no scored truth pixel (DTI undefined)
      INVALID_NUMERIC         the score is not finite (NaN / inf) -> logged, never an improvement
      VALID                   finite DTI on a zone with truth
  Checkpoint selection and early stopping use VALID epochs only; NaN is never read as a score.
* ``CheckpointManager`` (best_and_latest; atomic, sha256-verified, Drive-synchronised) holds the full resume state;
  a resume file written by another spec hash is refused. The spec hash covers the semantic experiment
  configuration, network spec, input contract (channel names + cube key + extra-layer key), unit scaler, fold
  provenance and the M9 target configuration: resuming with a changed feature contract is impossible.
* Saved per unit: ``<unit>_model.pt`` (selected), ``<unit>_last.pt`` (last epoch, model only), resume files until
  the final model exists, ``<unit>_selection.json`` (policy, selected epoch, statuses), per-epoch history CSV with
  losses, lr, inner score + status, GPU peak memory, samples / s.
* CUDA OOM: halve the micro-batch, double the accumulation (effective batch unchanged), ``oom_retries`` times.
* Collapse guard: stop when the mean probability on unlabeled training pixels exceeds the configured maximum.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd

STATUSES = ("NOT_CONFIGURED", "VALID", "INVALID_NUMERIC", "NO_ELIGIBLE_VALIDATION")
P7_TRAIN_CONTRACT = "p7-train.1"


def inner_status(score: float | None, n_truth: int | None, configured: bool) -> str:
    if not configured:
        return "NOT_CONFIGURED"
    if not n_truth:
        return "NO_ELIGIBLE_VALIDATION"
    if score is None or not np.isfinite(score):
        return "INVALID_NUMERIC"
    return "VALID"


def better(status: str, score: float | None, best: float) -> bool:
    return status == "VALID" and score is not None and np.isfinite(score) and score > best


def unit_seed(seed: int, exp: str, unit: str) -> int:
    return int(seed) + int(hashlib.sha1(f"{exp}/{unit}".encode()).hexdigest()[:6], 16) % 100_000


def spec_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def amp_dtype(name: str | None, device):
    import torch

    if device.type != "cuda" or name in (None, "none"):
        return None
    if name == "bfloat16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def inner_score(model, sc, esc, cube, extra, tu, labels, ecfg: dict, rt: dict, device) -> tuple[float | None, int]:
    """(DTI or None, scored truth pixels) of the binary top-frac map on ``tu.inner``."""
    from ..gems3.official import FastScorer
    from ..gems3.representation import top_mask
    from ..m9.infer import bbox
    from .infer import predict_mask

    P = int(ecfg["data"]["patch_px"])
    out = predict_mask([(model, sc, esc)], cube, extra, tu.inner, P=P, stride=max(int(P * float(rt["stride_frac"])), 1),
                       device=device, amp=bool(ecfg["training"]["amp"]), amp_dtype=rt.get("amp_dtype"),
                       batch=int(rt["infer_batch"]))
    if not out:
        return None, 0
    s = np.nan_to_num(out["seg"])
    q = top_mask(s, tu.inner, float(ecfg["training"]["select_frac"])).astype(np.float32)
    r0, r1, c0, c1 = bbox(tu.inner)
    sl = (slice(max(r0 - 4, 0), r1 + 4), slice(max(c0 - 4, 0), c1 + 4))
    fs = FastScorer(np.asarray(labels, bool)[sl], np.asarray(tu.inner, bool)[sl])
    if not fs.n:
        return None, 0
    return float(fs.components(q[sl]).dti), int(fs.n)


def train_unit(p7, exp_id: str, ecfg: dict, tu, inputs: dict, rt: dict, *, fixed_epochs: int | None = None,
               deadline: float | None = None) -> dict:
    """Train (or resume / reuse) one unit; returns its registry record. ``inputs``: cube, extra, meta, registry,
    groups, extra_names, extra_key. ``rt``: runtime (device, micro_batch, grad_accum, infer_batch, amp_dtype,
    oom_retries, ckpt_sync_min, stride_frac)."""
    import torch

    from ..artifacts.atomic import atomic_torch_save, atomic_write_json, sha256_file
    from ..m9.dataset import FoldScaler
    from ..m9.train import Prefetcher
    from ..training.checkpoint import CheckpointManager
    from ..training.trainer import TrainingInterrupted, _rng_state, _set_rng_state
    from . import losses as LS
    from .config import semantic_view
    from .data import P7Batch, fold_provenance
    from .model import build, count_params, net_spec, param_breakdown
    from .targets import build_p7_targets

    tc = ecfg["training"]
    dev = rt["device"]
    name = ecfg["experiment"]["name"]
    seed = unit_seed(int(ecfg["seed"]), name, tu.name)
    valid = np.asarray(p7.ctx.data.valid, bool)
    cube, extra, meta = inputs["cube"], inputs["extra"], inputs["meta"]
    scfg = p7.m9.cfg["features"]["scaler"]
    sc = FoldScaler.fit(cube, meta["local"], tu.allowed & valid, int(scfg["sample_px"]), float(scfg["clip"]), seed)
    esc = FoldScaler.fit(extra, inputs["extra_names"], tu.allowed & valid, int(scfg["sample_px"]), float(scfg["clip"]),
                         seed) if extra is not None else None
    channels = list(meta["local"]) + list(inputs["extra_names"])
    nspec = net_spec(channels, inputs["groups"], ecfg, meta["key"])
    epochs = int(fixed_epochs or tc["epochs"])
    steps = int(tc["steps_per_epoch"])
    prov = fold_provenance(tu)
    sem = {"contract": P7_TRAIN_CONTRACT, "experiment": semantic_view(ecfg), "net": nspec,
           "extra_key": inputs.get("extra_key"), "scaler": sc.to_json()["hash"],
           "extra_scaler": esc.to_json()["hash"] if esc is not None else None, "fold": prov,
           "m9_targets": spec_hash({k: p7.m9.cfg["data"].get(k) for k in ("kernel_radius_m", "orientation", "topology",
                                                                          "pu")}),
           "unit": tu.name, "epochs": epochs, "steps": steps}
    shash = spec_hash(sem)
    ckdir = p7.exp_dir(exp_id) / "checkpoints"
    sync = getattr(p7.m9.pc, "sync", None) if p7.m9.pc is not None else None
    ck = CheckpointManager(ckdir, tu.name, sync=sync, sync_every_min=float(rt["ckpt_sync_min"]), spec_hash=shash,
                           policy="best_and_latest")
    rec = {"experiment": name, "experiment_id": exp_id, "unit": tu.name, "kind": tu.kind, "zone": tu.zone,
           "spec_hash": shash, "scaler_hash": sem["scaler"], "cube_key": meta["key"]}
    if ck.final_ok():
        obj = ck.load_final()
        p7.say(f"{exp_id}/{tu.name}: final model exists (spec {shash}) -> reused")
        return {**rec, **{k: obj.get(k) for k in ("best_epoch", "best_inner", "epochs_run", "status", "n_params")},
                "checkpoint": str(ck.final_path), "sha256": sha256_file(ck.final_path), "reused": True}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    targets = build_p7_targets(p7.m9, tu, ecfg)
    builder = P7Batch(p7, tu, ecfg, cube, extra, sc, esc, targets, inputs["groups"], rng)
    model = build(nspec).to(dev)
    n_params = count_params(model)
    adt = rt.get("amp_dtype")
    use_amp = bool(tc["amp"]) and dev.type == "cuda" and adt is not None
    from ..m9.train import _grad_scaler

    scaler = _grad_scaler(use_amp and adt == torch.float16)
    bs, acc = int(rt["micro_batch"]), int(rt["grad_accum"])
    opt = torch.optim.AdamW(model.parameters(), lr=float(tc["lr"]), weight_decay=float(tc["weight_decay"]))
    total = max(epochs * steps, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(tc["lr"]), total_steps=total) \
        if tc["schedule"] == "onecycle" else torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    start, best, best_ep, best_state, hist, bad = 0, -np.inf, None, None, [], 0
    st = ck.latest(map_location=dev)
    if st is not None:
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        if st.get("scaler"):
            scaler.load_state_dict(st["scaler"])
        _set_rng_state(rng, st["rng"])
        start, best, best_ep, best_state = int(st["epoch"]), float(st["best"]), st.get("best_ep"), st.get("best_state")
        hist, bad = list(st.get("history", [])), int(st.get("bad", 0))
        bs, acc = int(st.get("bs", bs)), int(st.get("acc", acc))
        p7.say(f"{exp_id}/{tu.name}: resumed at epoch {start} (best {best:.4f} @ {best_ep})")
    labels = np.asarray(p7.ctx.data.labels, bool)
    configured = tu.inner is not None and bool(np.asarray(tu.inner).any())
    pf = Prefetcher(builder, bs)
    t_unit = time.perf_counter()
    status = "complete"
    try:
        for ep in range(start, epochs):
            model.train()
            t0, parts, oom_left, n_samples = time.perf_counter(), [], int(rt["oom_retries"]), 0
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            builder.stats.clear()
            step = 0
            while step < steps:
                try:
                    opt.zero_grad(set_to_none=True)
                    for _ in range(acc):
                        b = {k: v.to(dev, non_blocking=True) for k, v in pf.next().items()}
                        with torch.autocast(device_type=dev.type, dtype=adt, enabled=use_amp):
                            out = model(b["x"])
                        loss, pr = LS.compute(out, b, ecfg["loss"])
                        scaler.scale(loss / acc).backward()
                        parts.append(pr)
                        n_samples += int(b["x"].shape[0])
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(tc["clip_grad"]))
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                    step += 1
                except torch.cuda.OutOfMemoryError:
                    if oom_left <= 0 or bs <= 1:
                        raise
                    oom_left -= 1
                    torch.cuda.empty_cache()
                    bs, acc = max(bs // 2, 1), acc * 2
                    pf.resize(bs)
                    p7.say(f"{tu.name}: CUDA OOM -> micro-batch {bs} x accumulation {acc}")
            dt = time.perf_counter() - t0
            Pd = pd.DataFrame(parts)
            row = {"epoch": ep + 1, "lr": float(sched.get_last_lr()[0]), "micro_batch": bs, "accum": acc,
                   "epoch_s": round(dt, 1), "samples_per_s": round(n_samples / max(dt, 1e-9), 2),
                   **{f"loss_{c}": float(Pd[c].mean()) for c in Pd.columns},
                   **{f"aug_{k}": v for k, v in builder.stats.items()}}
            if dev.type == "cuda":
                row["gpu_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            score, n_truth = None, None
            if not configured:
                row["inner_status"] = "NOT_CONFIGURED"
            elif (ep + 1) % int(tc["inner_every"]) == 0 or ep + 1 == epochs:
                t1 = time.perf_counter()
                score, n_truth = inner_score(model, sc, esc, cube, extra, tu, labels, ecfg, rt, dev)
                row["inner_s"] = round(time.perf_counter() - t1, 1)
                row["inner_status"] = inner_status(score, n_truth, True)
            else:
                row["inner_status"] = "SKIPPED_EPOCH"
            row["inner_dti"] = score if score is not None else np.nan
            row["inner_truth_px"] = n_truth
            if better(row["inner_status"], score, best):
                best, best_ep, bad = float(score), ep + 1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            elif row["inner_status"] == "VALID":
                bad += 1
            hist.append(row)
            p7.say(f"{exp_id}/{tu.name} ep {ep + 1}/{epochs} loss {row.get('loss_total', np.nan):.4f} "
                   f"inner {row['inner_dti']:.4f} [{row['inner_status']}] ({dt:.0f}s, {row['samples_per_s']} samples/s)")
            ck.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                     "scaler": scaler.state_dict(), "rng": _rng_state(rng), "epoch": ep + 1, "best": best,
                     "best_ep": best_ep, "best_state": best_state, "history": hist, "bad": bad, "bs": bs, "acc": acc},
                    ep + 1)
            if row.get("loss_unlabeled_mean_p", 0.0) > float(tc["collapse_max_unlabeled_mean"]):
                status = "stopped_collapse"
                break
            if configured and ep + 1 >= int(tc["min_epochs"]) and bad >= int(tc["patience"]):
                status = "early_stopped"
                break
            if deadline is not None and time.monotonic() > deadline and ep + 1 < epochs:
                raise TrainingInterrupted(f"{exp_id}/{tu.name}: runtime budget reached after epoch {ep + 1}")
    finally:
        pf.close()
    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    atomic_torch_save({"state_dict": last_state, "net_spec": nspec, "epoch": len(hist), "spec_hash": shash},
                      ckdir / f"{tu.name}_last.pt")
    statuses = pd.Series([h["inner_status"] for h in hist]).value_counts().to_dict() if hist else {}
    if configured and best_state is None:
        selection = "last_epoch (no VALID inner epoch: flagged)"
        status = f"{status}_no_valid_inner"
    elif configured:
        selection = "best_valid_inner"
        model.load_state_dict(best_state)
    else:
        selection = "fixed_epochs_last"
    final = {"state_dict": model.state_dict(), "net_spec": nspec, "scaler": sc.to_json(),
             "extra_scaler": esc.to_json() if esc is not None else None, "best_epoch": best_ep if configured else len(hist),
             "best_inner": best if np.isfinite(best) else None, "epochs_run": len(hist), "status": status,
             "selection": selection, "history": hist, "n_params": n_params, "params": param_breakdown(model),
             "semantic": sem, "fold": prov}
    ck.save_final(final)
    sel = {"unit": tu.name, "experiment_id": exp_id, "policy": selection, "configured_inner": configured,
           "selected_epoch": final["best_epoch"], "best_inner_dti": final["best_inner"], "status_counts": statuses,
           "epochs_run": len(hist), "fixed_epochs": fixed_epochs, "spec_hash": shash, "fold": prov,
           "rule": "cv: best VALID inner DTI; disc: fixed epochs = median best cv epoch, last epoch kept"}
    atomic_write_json(ckdir / f"{tu.name}_selection.json", sel)
    ck.clear_resume()
    p7.ws.save_table(pd.DataFrame(hist), f"history_{tu.name}", p7.exp_sub(exp_id, "metrics"))
    p7.ws.save_json(json.loads(json.dumps({"scaler": sc.to_json(), "extra_scaler": final["extra_scaler"]}, default=str)),
                    f"{p7.exp_sub(exp_id, 'manifests')}/{tu.name}_scaler.json")
    return {**rec, "best_epoch": final["best_epoch"], "best_inner": final["best_inner"], "epochs_run": len(hist),
            "status": status, "selection": selection, "n_params": n_params,
            "seconds": round(time.perf_counter() - t_unit, 1), "checkpoint": str(ck.final_path),
            "sha256": sha256_file(ck.final_path), "reused": False, "inner_status_counts": json.dumps(statuses)}


def load_unit(p7, exp_id: str, unit: str, device):
    """(model, scaler, extra scaler, payload) of a finished unit (eval mode), or None."""
    import torch

    from ..m9.dataset import FoldScaler
    from .model import build

    p = p7.exp_dir(exp_id) / "checkpoints" / f"{unit}_model.pt"
    if not p.exists():
        p7.stage_in([p])
    if not p.exists():
        return None
    obj = torch.load(p, map_location=device, weights_only=False)
    model = build(obj["net_spec"]).to(device)
    model.load_state_dict(obj["state_dict"])
    model.eval()
    esc = FoldScaler.from_json(obj["extra_scaler"]) if obj.get("extra_scaler") else None
    return model, FoldScaler.from_json(obj["scaler"]), esc, obj


def discovery_epochs(records: list[dict], min_epochs: int, default: int) -> int:
    """Predeclared rule: median selected epoch of the experiment's cv units with a VALID selection, >= min_epochs."""
    eps = [int(r["best_epoch"]) for r in records if r.get("kind") == "cv" and r.get("best_epoch")
           and str(r.get("selection", "")).startswith("best_valid")]
    return max(int(np.median(eps)), int(min_epochs)) if eps else int(default)
