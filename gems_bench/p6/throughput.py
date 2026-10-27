"""W3: where do GeoFaultNet's ~450 s per epoch go? Component timings on the real M9 objects, then an exit decision.

Timed on one training unit with the production builder, model, losses and scaler (nothing is trained to completion):

    patch_read     cube[:, rows, cols] for one patch (memory-mapped vs RAM copy)
    scale          FoldScaler.apply on the patch
    context        low-resolution context crop (pooled cube)
    corruption     map-context corruption of the patch labels
    batch          BatchBuilder.batch(n) end to end (CPU; overlapped with the GPU by the single prefetch thread)
    h2d            host -> device copy of one batch (pageable vs pinned)
    fwd_bwd        forward + losses + backward + optimiser step (AMP as configured)
    inner_val      one inner-validation prediction + official-style DTI (once per epoch in M9)

Epoch estimate = steps x accumulation x max(batch, h2d + fwd_bwd) + inner_val / inner_every (the prefetcher overlaps
data and GPU work; the slower side bounds the step). Exit: <= ``max_epoch_s`` -> one ``core`` model on limited
folds is allowed; otherwise M9 training stops and the CNN champion stays. Optimisations are applied only to the
component that dominates, after this measurement.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd


def epoch_estimate(t: dict, steps: int, accum: int, inner_every: int = 1) -> float:
    """Seconds per epoch from per-step component timings (pure)."""
    step = max(float(t["batch"]), float(t["h2d"]) + float(t["fwd_bwd"]))
    return steps * max(accum, 1) * step + float(t.get("inner_val", 0.0)) / max(int(inner_every), 1)


def decide(rows: pd.DataFrame, max_epoch_s: float) -> dict:
    """Best configuration and the W3 exit decision (pure)."""
    if not len(rows):
        return {"decision": "STOP_M9", "reason": "no timing"}
    r = rows.sort_values("epoch_s_est").iloc[0]
    ok = float(r.epoch_s_est) <= float(max_epoch_s)
    dom = max(("batch", "h2d", "fwd_bwd", "inner_val"), key=lambda k: float(r.get(k, 0.0)))
    return {"decision": "PERMIT_ONE_CORE" if ok else "STOP_M9", "config": r.config,
            "epoch_s_est": float(r.epoch_s_est), "dominant": dom, "max_epoch_s": float(max_epoch_s)}


def _timeit(fn, n: int, sync=None) -> float:
    fn()                                                        # warm-up (allocations, caches, cuDNN autotune)
    if sync:
        sync()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if sync:
        sync()
    return (time.perf_counter() - t0) / n


def profile(p6, in_ram: bool, pinned: bool) -> dict:
    import torch

    from ..m9 import losses as LS
    from ..m9.dataset import BatchBuilder, FoldScaler, build_cube, build_targets, open_cube, pooled_scaler, training_units
    from ..m9.features import channel_groups
    from ..m9.infer import device_of
    from ..m9.model import build, net_spec
    from ..m9.train import _amp_dtype, _grad_scaler, inner_score

    tc_ = p6.cfg["throughput"]
    m9 = p6.m9
    vcfg = m9.variant_cfg(tc_["variant"])
    groups = channel_groups(m9, vcfg)
    meta = build_cube(m9, groups)                              # reuses the cached cube (never rebuilt when present)
    cube, pooled = open_cube(meta, in_ram=in_ram)
    tu = next(t for t in training_units(m9) if t.name == tc_["unit"])
    dc, tc = vcfg["data"], vcfg["training"]
    seed = int(vcfg["seed"])
    valid = np.asarray(m9.ctx.data.valid, bool)
    sc = FoldScaler.fit(cube, meta["local"], tu.allowed & valid, int(vcfg["features"]["scaler"]["sample_px"]),
                        float(vcfg["features"]["scaler"]["clip"]), seed)
    f = int(dc["context_factor"])
    hp, wp = pooled.shape[1:]
    alp = tu.allowed[:hp * f, :wp * f].reshape(hp, f, wp, f).mean((1, 3)) > 0.5
    psc = pooled_scaler(sc, meta["pooled"], pooled, alp, seed)
    rng = np.random.default_rng(seed)
    builder = BatchBuilder(m9, tu, vcfg, cube, pooled, sc, psc, build_targets(m9, tu), rng)
    P, bs = builder.P, int(tc["batch_size"])
    n = int(tc_["steps"])
    out = {}
    rc = [builder.sampler.draw(False, 1000) for _ in range(n)]
    rc = [x for x in rc if x is not None]
    t0 = time.perf_counter()                                   # no warm-up: a second pass would hit the page cache
    for r, c in rc:
        np.ascontiguousarray(cube[:, r:r + P, c:c + P])
    out["patch_read"] = (time.perf_counter() - t0) / max(len(rc), 1) * bs
    x0 = np.ascontiguousarray(cube[:, rc[0][0]:rc[0][0] + P, rc[0][1]:rc[0][1] + P])
    out["scale"] = _timeit(lambda: sc.apply(x0), n) * bs
    out["context"] = _timeit(lambda: builder.context_crop(*rc[0]), n) * bs
    seg = np.asarray(builder.t["seg"][rc[0][0]:rc[0][0] + P, rc[0][1]:rc[0][1] + P]) > 0
    sm = np.asarray(builder.t["sysmap"][rc[0][0]:rc[0][0] + P, rc[0][1]:rc[0][1] + P])
    out["corruption"] = _timeit(lambda: builder.corr(seg, sm), n) * bs
    out["batch"] = _timeit(lambda: builder.batch(bs), max(n // 4, 2))
    dev = device_of(m9.cfg["inference"]["device"])
    sync = torch.cuda.synchronize if dev.type == "cuda" else None
    b = builder.batch(bs)
    if pinned and dev.type == "cuda":
        b = {k: v.pin_memory() for k, v in b.items()}
    out["h2d"] = _timeit(lambda: {k: v.to(dev, non_blocking=True) for k, v in b.items()}, n, sync)
    model = build(net_spec(meta, groups, vcfg)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tc["lr"]))
    use_amp = bool(tc["amp"]) and dev.type == "cuda"
    amp_dtype = _amp_dtype() if use_amp else None
    scaler = _grad_scaler(use_amp and amp_dtype == torch.float16)
    bd = {k: v.to(dev) for k, v in b.items()}
    stage = LS.stage_of(0, 1, vcfg["loss"])

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
            o = model(bd["x"], bd["m"], bd["xc"])
        loss, _ = LS.compute(o, bd, vcfg["loss"], vcfg["corruption"], stage)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    out["fwd_bwd"] = _timeit(step, n, sync)
    if tu.inner is not None and tu.inner.any():
        t0 = time.perf_counter()
        inner_score([(model, sc, psc)], cube, pooled, tu, m9, vcfg, dev)
        out["inner_val"] = time.perf_counter() - t0
    else:
        out["inner_val"] = 0.0
    out.update(steps=int(m9.profile["steps_per_epoch"]), accum=int(tc["grad_accum"]), batch_size=bs,
               cube_bytes=int(getattr(cube, "nbytes", 0)), cube_in_ram=not isinstance(cube, np.memmap))
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return out


def run(p6) -> pd.DataFrame:
    tc_ = p6.cfg["throughput"]
    rows = []
    for in_ram in tc_["cube_in_ram"]:
        for pinned in tc_["pinned"]:
            t = profile(p6, bool(in_ram), bool(pinned))
            for ie in tc_["inner_every"]:
                rows.append({"config": f"ram{int(bool(in_ram))}_pin{int(bool(pinned))}_inner{int(ie)}", **t,
                             "inner_every": int(ie),
                             "epoch_s_est": epoch_estimate(t, t["steps"], t["accum"], int(ie))})
            p6.ws.save_table(pd.DataFrame(rows), "timings", "throughput")
    T = pd.DataFrame(rows)
    d = decide(T, float(tc_["max_epoch_s"]))
    p6.ws.save_json(d, "throughput/decision.json")
    p6.say(f"W3: {d}")
    return T
