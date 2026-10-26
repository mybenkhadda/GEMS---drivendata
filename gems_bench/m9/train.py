"""Section 08: resumable GeoFaultNet training per (variant, unit).

* Deterministic seeds (unit seed = seed + hash(variant / unit)); patches built on the CPU by a single prefetch
  thread with its own RNG (fixed sample order; after a resume, the one batch that was prefetched when the checkpoint
  was written may differ); mixed precision (bf16 on Ampere+, fp16 + GradScaler on T4);
  gradient accumulation; gradient clipping; OneCycle schedule; AdamW.
* Curriculum (losses.stage_of): A dense + reconstruction, B + orientation / topology, C + enabled Stage-C terms.
* Checkpoint selection (cv units): best FINITE inner-validation score = official-style DTI of the binary top
  ``select_frac`` map on the inner-validation zone, predicted WITHOUT map context (independent discovery); patience
  after ``min_epochs``. Discovery / full units: fixed epochs = median best epoch of the variant's cv units.
* ``CheckpointManager`` (best_and_latest): atomic, sha256-verified, Drive-synchronised full resume state (model,
  optimiser, scheduler, scaler, EMA teacher, NumPy / torch / CUDA RNG, epoch, best, history); a checkpoint written by
  a different spec is refused. ``max_runtime_min`` stops cleanly at an epoch boundary (TrainingInterrupted).
* CUDA OOM: halve the micro-batch and double the accumulation (effective batch unchanged), ``oom_retries`` times.
* Collapse guard: stop when the mean probability on unlabeled training pixels exceeds
  ``collapse_max_unlabeled_mean`` (a model drifting towards "fault everywhere").
* Every epoch: loss decomposition (L1..L6, consistency), stage, lr, inner score, unlabeled mean p, removed-pixel
  mean p, GPU peak memory, corruption operation counts -> ``training/logs/<variant>__<unit>.csv``.
"""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd


def spec_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def semantic(vcfg: dict, tu_name: str, nspec: dict, scaler_hash: str, epochs: int, steps: int) -> dict:
    """Everything that changes the trained weights (runtime knobs excluded)."""
    tr = {k: v for k, v in vcfg["training"].items() if k not in ("ckpt_sync_min", "max_runtime_min", "infer_batch",
                                                               "oom_retries")}
    return {"unit": tu_name, "net": {k: v for k, v in nspec.items() if k != "groups"}, "groups": nspec["groups"],
            "loss": vcfg["loss"], "corruption": vcfg["corruption"], "data": vcfg["data"], "training": tr,
            "seed": vcfg["seed"], "scaler": scaler_hash, "epochs": int(epochs), "steps": int(steps)}


def _amp_dtype():
    import torch

    from ..training.gpu import detect

    g = detect()
    return torch.bfloat16 if g.get("cuda") and g.get("major", 7) >= 8 else torch.float16


def _grad_scaler(enabled: bool):
    import torch

    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


class Prefetcher:
    """Builds the next batch on a single background thread while the GPU works (sample order unchanged)."""

    def __init__(self, builder, n: int):
        self.b, self.n = builder, n
        self.ex = ThreadPoolExecutor(max_workers=1)
        self.fut = self.ex.submit(builder.batch, n)

    def next(self):
        out = self.fut.result()
        self.fut = self.ex.submit(self.b.batch, self.n)
        return out

    def resize(self, n: int) -> None:
        self.fut.result()
        self.n = n
        self.fut = self.ex.submit(self.b.batch, n)

    def close(self) -> None:
        try:
            self.fut.result()
        except Exception:
            pass
        self.ex.shutdown(wait=True)


def inner_score(models, cube, pooled, tu, m9, vcfg, device) -> float:
    """Official-style DTI of the binary top-frac map on the inner-validation zone, no map context."""
    from ..gems3.representation import top_mask
    from ..metric import MaskedScorer
    from .infer import predict_mask

    dc, tc = vcfg["data"], vcfg["training"]
    P = int(dc["patch_px"])
    out = predict_mask(models, cube, pooled, tu.inner, flag=0.0, P=P, f=int(dc["context_factor"]),
                       stride=int(P * float(tc["stride_frac"])), device=device, amp=bool(tc["amp"]),
                       batch=int(tc["infer_batch"]))
    s = np.nan_to_num(out["seg"])
    q = top_mask(s, tu.inner, float(tc["select_frac"])).astype(np.float32)
    lab = np.asarray(m9.ctx.data.labels, bool)
    from .infer import bbox

    r0, r1, c0, c1 = bbox(tu.inner)
    sl = (slice(max(r0 - 4, 0), r1 + 4), slice(max(c0 - 4, 0), c1 + 4))
    return float(MaskedScorer(lab[sl], tu.inner[sl]).components(q[sl]).dti)


def train_unit(m9, vname: str, vcfg: dict, tu, meta: dict, cube, pooled, groups: dict, *, fixed_epochs: int | None = None,
               deadline: float | None = None) -> dict:
    """Train (or resume / reuse) one unit; returns its registry record."""
    import torch

    from ..training.checkpoint import CheckpointManager
    from ..training.trainer import TrainingInterrupted, _rng_state, _set_rng_state, improved
    from . import losses as LS
    from .dataset import BatchBuilder, FoldScaler, build_targets, pooled_scaler
    from .infer import device_of
    from .model import build, count_params, net_spec

    tc, dc, lc, cc = vcfg["training"], vcfg["data"], vcfg["loss"], vcfg["corruption"]
    prof = m9.profile
    seed = int(vcfg["seed"]) + int(hashlib.sha1(f"{vname}/{tu.name}".encode()).hexdigest()[:6], 16) % 100_000
    valid = np.asarray(m9.ctx.data.valid, bool)
    sc = FoldScaler.fit(cube, meta["local"], tu.allowed & valid, int(vcfg["features"]["scaler"]["sample_px"]),
                        float(vcfg["features"]["scaler"]["clip"]), seed)
    f = int(dc["context_factor"])
    hp, wp = pooled.shape[1:]
    alp = tu.allowed[:hp * f, :wp * f].reshape(hp, f, wp, f).mean((1, 3)) > 0.5
    psc = pooled_scaler(sc, meta["pooled"], pooled, alp, seed)
    nspec = net_spec(meta, groups, vcfg)
    epochs = int(fixed_epochs or prof["epochs"])
    steps = int(prof["steps_per_epoch"])
    sem = semantic(vcfg, tu.name, nspec, sc.to_json()["hash"], epochs, steps)
    shash = spec_hash(sem)
    ckdir = m9.ws.root / "models" / "checkpoints" / vname
    ck = CheckpointManager(ckdir, tu.name, sync=getattr(m9.pc, "sync", None) if m9.pc is not None else None,
                           sync_every_min=float(tc["ckpt_sync_min"]), spec_hash=shash, policy=tc["ckpt_policy"])
    rec = {"variant": vname, "unit": tu.name, "kind": tu.kind, "zone": tu.zone, "spec_hash": shash,
           "scaler_hash": sc.to_json()["hash"], "cube_key": meta["key"]}
    if ck.final_ok():
        from ..artifacts.atomic import sha256_file

        obj = ck.load_final()
        m9.say(f"s08 {vname}/{tu.name}: final model exists (spec {shash}) -> reused")
        return {**rec, **{k: obj.get(k) for k in ("best_epoch", "best_inner", "epochs_run", "status", "n_params")},
                "checkpoint": str(ck.final_path), "sha256": sha256_file(ck.final_path), "reused": True}
    dev = device_of(m9.cfg["inference"]["device"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)                      # sampler RNG (prefetch thread only)
    rng_c = np.random.default_rng(seed + 1)                # consistency views (main thread): never shared
    targets = build_targets(m9, tu)
    builder = BatchBuilder(m9, tu, vcfg, cube, pooled, sc, psc, targets, rng)
    model = build(nspec).to(dev)
    n_params = count_params(model)
    use_amp = bool(tc["amp"]) and dev.type == "cuda"
    amp_dtype = _amp_dtype() if use_amp else None
    scaler = _grad_scaler(use_amp and amp_dtype == torch.float16)
    bs, acc = int(tc["batch_size"]), max(int(tc["grad_accum"]), 1)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tc["lr"]), weight_decay=float(tc["weight_decay"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(tc["lr"]), total_steps=max(epochs * steps, 1))
    teacher = None
    cons = lc["consistency"]
    if cons.get("enabled"):
        from ..losses.consistency import EMATeacher

        teacher = EMATeacher(model, float(cons["ema_decay"]))
    start, best, best_ep, best_state, hist, bad = 0, -np.inf, None, None, [], 0
    st = ck.latest(map_location=dev)
    if st is not None:
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        if st.get("scaler"):
            scaler.load_state_dict(st["scaler"])
        if teacher is not None and st.get("teacher"):
            teacher.load_state_dict(st["teacher"])
        _set_rng_state(rng, st["rng"])
        start, best, best_ep, best_state = int(st["epoch"]), float(st["best"]), st.get("best_ep"), st.get("best_state")
        hist, bad = list(st.get("history", [])), int(st.get("bad", 0))
        bs, acc = int(st.get("bs", bs)), int(st.get("acc", acc))
        m9.say(f"s08 {vname}/{tu.name}: resumed at epoch {start} (best {best:.4f} @ {best_ep})")
    pf = Prefetcher(builder, bs)
    t_unit = time.perf_counter()
    status = "complete"
    try:
        for ep in range(start, epochs):
            stage = LS.stage_of(ep, epochs, lc)
            model.train()
            t0, parts_acc, n_ok, oom_left = time.perf_counter(), [], 0, int(tc["oom_retries"])
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            builder.stats.clear()
            step = 0
            while step < steps:
                try:
                    opt.zero_grad(set_to_none=True)
                    for _ in range(acc):
                        b = {k: v.to(dev, non_blocking=True) for k, v in pf.next().items()}
                        with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                            out = model(b["x"], b["m"], b["xc"])
                        loss, parts = LS.compute(out, b, lc, cc, stage)
                        if teacher is not None:
                            from ..losses.consistency import consistency_loss, rampup, shifted_view

                            xv, sh = shifted_view(b["x"], rng_c, int(cons["shift_px"]), float(cons["noise"]))
                            mv = torch.roll(b["m"], shifts=sh, dims=(-2, -1))
                            with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                                tout = teacher.model(xv, mv, b["xc"])
                            lcons = consistency_loss(out["seg"], tout["seg"], sh, b["seg"], b["w"])
                            wc = float(lc["weights"]["l7"]) * rampup(ep, int(cons["rampup"]))
                            loss = loss + wc * lcons
                            parts["l7"] = float(lcons.detach())
                        scaler.scale(loss / acc).backward()
                        parts_acc.append(parts)
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(tc["clip_grad"]))
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                    if teacher is not None:
                        teacher.update(model)
                    step += 1
                    n_ok += 1
                except torch.cuda.OutOfMemoryError:
                    if oom_left <= 0 or bs <= 1:
                        raise
                    oom_left -= 1
                    torch.cuda.empty_cache()
                    bs, acc = max(bs // 2, 1), acc * 2
                    pf.resize(bs)
                    m9.say(f"s08 {tu.name}: CUDA OOM -> batch {bs} x accumulation {acc}")
            P = pd.DataFrame(parts_acc)
            row = {"epoch": ep + 1, "stage": stage, "lr": float(sched.get_last_lr()[0]), "batch": bs, "accum": acc,
                   "epoch_s": round(time.perf_counter() - t0, 1), **{f"loss_{c}": float(P[c].mean()) for c in P.columns},
                   **{f"ops_{k}": v for k, v in builder.stats.items()}}
            if dev.type == "cuda":
                row["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
            if tu.inner is not None and tu.inner.any():
                row["inner_dti"] = inner_score([(model, sc, psc)], cube, pooled, tu, m9, vcfg, dev)
                if improved(row["inner_dti"], best):
                    best, best_ep, bad = row["inner_dti"], ep + 1, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
            hist.append(row)
            m9.say(f"s08 {vname}/{tu.name} ep {ep + 1}/{epochs} [{stage}] loss_l1 {row.get('loss_l1', np.nan):.4f} "
                   f"inner {row.get('inner_dti', float('nan')):.4f} ({row['epoch_s']:.0f}s)")
            ck.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                     "scaler": scaler.state_dict(), "teacher": teacher.state_dict() if teacher is not None else None,
                     "rng": _rng_state(rng), "epoch": ep + 1, "best": best, "best_ep": best_ep, "best_state": best_state,
                     "history": hist, "bad": bad, "bs": bs, "acc": acc}, ep + 1)
            if row.get("loss_unlabeled_mean_p", 0.0) > float(tc["collapse_max_unlabeled_mean"]):
                status = "stopped_collapse"
                break
            if tu.inner is not None and ep + 1 >= int(tc["min_epochs"]) and bad >= int(tc["patience"]):
                status = "early_stopped"
                break
            if deadline is not None and time.monotonic() > deadline and ep + 1 < epochs:
                raise TrainingInterrupted(f"{vname}/{tu.name}: runtime budget reached after epoch {ep + 1}")
    finally:
        pf.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    final = {"state_dict": model.state_dict(), "net_spec": nspec, "scaler": sc.to_json(), "pooled_scaler": psc.to_json(),
             "best_epoch": best_ep, "best_inner": best if np.isfinite(best) else None, "epochs_run": len(hist),
             "status": status, "history": hist, "n_params": n_params, "semantic": sem}
    ck.save_final(final)
    ck.clear_resume()
    H = pd.DataFrame(hist)
    m9.ws.save_table(H, f"{vname}__{tu.name}", "training/logs")
    m9.ws.save_json({"scaler": sc.to_json(), "pooled_scaler": psc.to_json()}, f"models/{vname}__{tu.name}_scaler.json")
    from ..artifacts.atomic import sha256_file

    return {**rec, "best_epoch": best_ep, "best_inner": final["best_inner"], "epochs_run": len(hist), "status": status,
            "n_params": n_params, "seconds": round(time.perf_counter() - t_unit, 1), "checkpoint": str(ck.final_path),
            "sha256": sha256_file(ck.final_path), "reused": False}


def load_unit(m9, vname: str, unit: str, device=None):
    """(model, scaler, pooled scaler, payload) of a finished unit (eval mode)."""
    import torch

    from .dataset import FoldScaler
    from .infer import device_of
    from .model import build

    p = m9.ws.root / "models" / "checkpoints" / vname / f"{unit}_model.pt"
    if not p.exists():
        m9.stage_in([p.relative_to(m9.local).as_posix()])
    if not p.exists():
        return None
    dev = device or device_of(m9.cfg["inference"]["device"])
    obj = torch.load(p, map_location=dev, weights_only=False)
    model = build(obj["net_spec"]).to(dev)
    model.load_state_dict(obj["state_dict"])
    model.eval()
    return model, FoldScaler.from_json(obj["scaler"]), FoldScaler.from_json(obj["pooled_scaler"]), obj


def discovery_epochs(m9, vname: str, default: int) -> int:
    """Median best epoch of the variant's finished cv units (the Phase-3 rule), else ``default``."""
    import torch

    eps = []
    d = m9.ws.root / "models" / "checkpoints" / vname
    for p in sorted(d.glob("geo_f*_model.pt")) if d.exists() else []:
        try:
            e = torch.load(p, map_location="cpu", weights_only=False).get("best_epoch")
            if e:
                eps.append(int(e))
        except Exception:
            continue
    return int(np.median(eps)) if eps else int(default)
