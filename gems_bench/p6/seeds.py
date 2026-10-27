"""W1: cheap improvement of the champion: <= 3 seeds per member family, rank averaging, optional 180 deg TTA.

* Members are existing Phase-5 runs (``seeds.families.<f>.existing``) plus extra seeds of the same committed
  experiment (``pipeline.experiments.run_experiment`` with ``{seed, name}`` overrides; nothing else changes). Extra
  seeds are trained only while ``budget.workstreams.w1.gpu`` allows (``allocate``); existing checkpoints are reused.
* Family map = mean of per-unit ranks of its members (rank averaging is scale-free, like the champion's fusion).
* TTA (``rot180``): every honest map is re-predicted on the 180 deg rotated input by the SAME checkpoint that made
  it (``cv_fold<k>`` for geo fold k, ``disc<d>`` for split d) and averaged with the stored map, so TTA maps stay
  held-out. Cached per (member, unit) under ``seeds/maps``.
* Evaluation = the champion's own grid (thin reps x fusion weights x area fractions, eligible budget, ring 0) and
  nested selection on each task; paired with the fallback's per-zone rows from M9 section 01.

Promotion (``seeds.promote``): discovery DTI >= fallback + 0.005, geographic DTI and worst discovery zone do not
regress. Otherwise the original B0 + B1 fusion stays.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd


# =============================================================================== pure helpers
def allocate(families: dict, existing: dict, cost_h: dict, left_h: float, priority: list[str], max_members: int) -> list:
    """Extra (family, seed) runs that fit in ``left_h`` GPU hours, filling families in ``priority`` order
    round-robin (one extra seed per family per round) so both components gain diversity before either reaches 3."""
    want = {f: [s for s in families[f].get("extra_seeds", [])][: max(max_members - len(existing.get(f, [])), 0)]
            for f in priority}
    out, spent = [], 0.0
    while any(want.values()):
        progressed = False
        for f in priority:
            if not want[f]:
                continue
            c = float(cost_h.get(f, np.inf))
            if spent + c <= left_h + 1e-9:
                out.append((f, want[f].pop(0)))
                spent += c
                progressed = True
            else:
                want[f] = []
        if not progressed:
            break
    return out


def rank_average(maps: list[np.ndarray], ev: np.ndarray) -> np.ndarray:
    """Mean of per-unit ranks (``gems3.representation.rank01``) of several members inside ``ev``."""
    from ..gems3.representation import rank01

    if not maps:
        raise ValueError("no member map")
    return (sum(rank01(m, ev) for m in maps) / len(maps)).astype(np.float32)


def rot180_average(predict, x: np.ndarray) -> np.ndarray:
    """0.5 (f(x) + R f(R x)) with R the 180 deg rotation of the last two axes (pure; for any array predictor)."""
    r = np.flip(x, axis=(-2, -1))
    return (0.5 * (predict(x) + np.flip(predict(np.ascontiguousarray(r)), axis=(-2, -1)))).astype(np.float32)


def member_name(stem: str, seed: int) -> str:
    return f"{stem}_s{int(seed)}"


# =============================================================================== Colab side
def _have(ctx, model: str) -> bool:
    try:
        return ctx.oof(model) is not None and ctx.has_discovery(model)
    except (FileNotFoundError, KeyError):
        return False


def members(p6) -> dict:
    """{family: [model names with OOF + every discovery map]} (existing runs + finished extra seeds)."""
    sc = p6.cfg["seeds"]
    out = {}
    for f, fc in sc["families"].items():
        names = list(fc.get("existing", [])) + [member_name(fc["stem"], s) for s in fc.get("extra_seeds", [])]
        out[f] = [n for n in names if _have(p6.ctx, n)][: int(sc["max_members_per_family"])]
    return out


def run_cost_h(p6, family: str) -> float:
    """Measured GPU hours of one run of the family's experiment (sum of stage seconds), else the default."""
    from ..pipeline.experiments import experiment_config, open_experiment

    fc = p6.cfg["seeds"]["families"][family]
    try:
        exp = open_experiment(p6.m9.pc, experiment_config(p6.m9.pc, fc["stem"]))
        st = exp.status()
        s = float(pd.to_numeric(st.get("seconds", pd.Series(dtype=float)), errors="coerce").sum())
        if s > 0:
            return s / 3600.0
    except Exception as e:                                     # cost estimate never blocks the plan
        p6.say(f"W1 cost of {fc['stem']}: {type(e).__name__}: {e}; default used")
    return float(p6.cfg["seeds"]["run_hours_default"])


def plan(p6) -> pd.DataFrame:
    sc = p6.cfg["seeds"]
    have = members(p6)
    cost = {f: run_cost_h(p6, f) for f in sc["families"]}
    left = p6.budget.left_h("w1", "gpu")
    runs = allocate(sc["families"], have, cost, left, list(sc["priority"]), int(sc["max_members_per_family"]))
    runs = [(f, s) for f, s in runs if member_name(sc["families"][f]["stem"], s) not in have.get(f, [])]
    t = pd.DataFrame([{"family": f, "stem": sc["families"][f]["stem"], "seed": s,
                       "name": member_name(sc["families"][f]["stem"], s), "est_gpu_h": cost[f]} for f, s in runs])
    p6.ws.save_table(t, "train_plan", "seeds")
    p6.ws.save_json({"members": have, "cost_h": cost, "gpu_left_h": left}, "seeds/plan.json")
    return t


def train_missing(p6) -> pd.DataFrame:
    """Train the planned extra seeds (resumable experiments; one at a time; budget re-checked before each)."""
    from ..pipeline.experiments import run_experiment

    t = plan(p6)
    rows = []
    for r in t.itertuples():
        p6.budget.check("w1", "gpu", float(r.est_gpu_h))
        t0 = time.perf_counter()
        out = run_experiment(p6.m9.pc, r.stem, overrides={"seed": int(r.seed), "name": r.name,
                                                          "model_name": r.name})
        rows.append({**r._asdict(), **out, "seconds": round(time.perf_counter() - t0, 1)})
        p6.ws.save_table(pd.DataFrame(rows), "train_log", "seeds")
    return pd.DataFrame(rows)


def run_dir(ctx, model: str):
    from pathlib import Path

    z = getattr(ctx, "zoo5", {}).get(model) or {}
    p = z.get("oof")
    return Path(p).parent if p else None


def tta_unit_map(p6, model: str, u, device=None) -> np.ndarray | None:
    """Honest 180 deg TTA map of ``model`` on unit ``u`` (cropped), cached under seeds/maps."""
    from ..gems3.official import unit_score

    path = p6.ws.root / "seeds" / "maps" / f"{model}__{u.task}_{u.zone}__rot180.npy"
    if path.exists():
        return np.load(path)
    base = unit_score(p6.ctx, model, u)
    rd = run_dir(p6.ctx, model)
    if base is None or rd is None:
        return None
    import torch

    from ..cnn import device_of
    from ..inference.full_raster import predict_heads_mask
    from ..phase4_training import pooled_context, to_tensor_stack
    from ..training.checkpoint import CheckpointManager
    from ..training.cv import prepare_stack5
    from ..training.trainer import load_unit_model

    unit = f"cv_fold{u.zone}" if u.task == "geo" else f"disc{u.zone}"
    dev = device or device_of()
    spec, model_t = load_unit_model(CheckpointManager(rd / "checkpoints", unit), dev)
    g = torch.flip(to_tensor_stack(prepare_stack5(p6.ctx, spec), dev), dims=(1, 2))
    gctx = pooled_context(g, spec.context_factor) if spec.context else None
    m = np.zeros(p6.ctx.data.shape, bool)
    m[u.sl] = u.ev
    rot = predict_heads_mask(model_t, g, np.ascontiguousarray(np.flip(m, (0, 1))), spec, dev, gctx)["seg"]
    back = np.flip(rot, (0, 1))[u.sl]
    out = np.where(u.ev, 0.5 * (np.asarray(base, np.float32) + np.nan_to_num(back)), np.nan).astype(np.float32)
    np.save(path, out)
    del g, gctx, model_t
    return out


def tta_all(p6) -> pd.DataFrame:
    """GPU step: TTA maps of every member on every geo / primary unit (cached; budget re-checked per member)."""
    rows = []
    for f, names in members(p6).items():
        for n in names:
            p6.budget.check("w1", "gpu")
            for t in ("geo", p6.prim):
                for u in p6.m9.m7.units_of(t):
                    t0 = time.perf_counter()
                    ok = tta_unit_map(p6, n, u) is not None
                    rows.append({"family": f, "member": n, "unit": u.key, "ok": ok,
                                 "seconds": round(time.perf_counter() - t0, 1)})
            p6.ws.save_table(pd.DataFrame(rows), "tta_log", "seeds")
    return pd.DataFrame(rows)


def family_maps(p6, u, fam: dict, tta: bool) -> dict | None:
    from ..gems3.official import unit_score

    out = {}
    for f, names in fam.items():
        maps = []
        for n in names:
            a = tta_unit_map(p6, n, u) if tta else unit_score(p6.ctx, n, u)
            if a is not None:
                maps.append(a)
        if not maps:
            return None
        out[f] = rank_average(maps, u.ev)
    return out


def stitch(shape, parts: list[tuple[tuple, np.ndarray, np.ndarray]]) -> np.ndarray:
    """Full-grid map from (crop slices, crop ev, crop map) of disjoint units; NaN where no unit (pure)."""
    out = np.full(shape, np.nan, np.float32)
    for sl, ev, a in parts:
        view = out[sl]
        view[ev] = np.asarray(a, np.float32)[ev]
    return out


def full_region(p6) -> dict:
    """Full-region GeoTIFF of a PROMOTED seed ensemble: per member, the geo-fold honest maps (TTA when chosen)
    stitched like the fallback's ``oof_stitched`` source, family rank average, all-zone (rep, w), eligible budget."""
    from ..gems3 import representation as R
    from ..gems3 import submission as S
    from ..gems3.official import unit_score
    from ..m7 import export as E

    cand = json.loads((p6.ws.root / "seeds" / "candidate.json").read_text(encoding="utf-8"))
    if cand.get("gate") != "PROMOTE":
        raise RuntimeError("W1 full region refused: the seed ensemble did not pass its gate")
    tta = cand["name"].endswith("_tta")
    data = p6.ctx.data
    valid = np.asarray(data.valid, bool)
    geo = list(p6.m9.m7.units_of("geo"))
    fam = {}
    for f, names in cand["members"].items():
        maps = [stitch(data.shape, [(u.sl, u.ev, tta_unit_map(p6, n, u) if tta else unit_score(p6.ctx, n, u))
                                    for u in geo]) for n in names]
        fam[f] = rank_average(maps, valid)
    pa = cand["params_all"][p6.prim]
    s = R.fuse(fam["b0"], fam["b1"], float(pa["w"]))
    src = p6.m9.sh["submission"]["area_frac_source"]
    task = "geo" if src == "geo" else p6.prim
    frac = float(src) if isinstance(src, (int, float)) else float(cand["params_all"][task]["frac"])
    el = S.eligible_mask(data, 0, "eligible")
    rep = p6.m9.sh["baseline"]["representations"][pa["rep"]]
    q = R.apply(rep, s, valid, frac, known=data.labels, sigma_px=float(p6.m9.sh["evaluation"]["ridge_sigma_px"]),
                eligible=el)
    led = [S.area_ledger(q, data, 0, f"seed ensemble {pa['rep']} w {pa['w']} frac {frac:g}")]
    info = E._write(p6.view, f"p6_{cand['name'][3:]}_fusion", q, led,
                    {"model_id": cand["name"], "parameters": {**pa, "frac": frac, "ring": 0}, "members": cand["members"]},
                    0)
    cand["submission_file"] = info["submission"]
    p6.ws.save_json(json.loads(json.dumps(cand, default=float)), "seeds/candidate.json")
    return info


def evaluate(p6) -> dict:
    """Grid + nested selection of the seed ensemble (with and without TTA), gate, candidate.json."""
    from ..gems3 import representation as R
    from ..gems3.metric_opt import nested_select
    from ..m7.protocol import score_row
    from .select import fallback_reference, gate

    sc, m7c = p6.cfg["seeds"], p6.m9.sh
    sel = p6.selection()
    fam = members(p6)
    reps = m7c["baseline"]["representations"]
    thin = list(m7c["baseline"]["thinning_reps"])
    sigma = float(m7c["evaluation"]["ridge_sigma_px"])
    variants = {"seeds": False} | ({"seeds_tta": True} if sc.get("tta_rot180") else {})
    rows = []
    for t in ("geo", p6.prim):
        for u in p6.m9.m7.units_of(t):
            el, pk = p6.m9.elig(u, 0), p6.m9.pkey(u)
            for vname, tta in variants.items():
                fm = family_maps(p6, u, fam, tta)
                if fm is None:
                    continue
                for w in sc["fusion_weights"]:
                    s = R.fuse(fm["b0"], fm["b1"], float(w))
                    crest = R.crest_map(s, u.ev, sigma)
                    for rn in thin:
                        for f in m7c["evaluation"]["area_fracs"]:
                            q = R.apply(reps[rn], s, u.ev, float(f), known=u.known, crest=crest, sigma_px=sigma,
                                        eligible=el)
                            rows.append(score_row(u, q, el, pk, method=vname, rep=rn, w=float(w), frac=float(f)))
    G = pd.DataFrame(rows)
    p6.ws.save_table(G, "grid", "seeds")
    summ = []
    for (vname, task), g in G.groupby(["method", "task"]) if len(G) else []:
        per, s = nested_select(g, ["rep", "w", "frac"])
        p6.ws.save_table(per, f"per_zone_{vname}_{task}", "seeds")
        summ.append({"method": vname, "task": task, "dti_crossfit": s["dti_crossfit"], "fold_min": s["fold_min"],
                     "params_all": json.dumps(s["params_all"], default=float)})
    S = pd.DataFrame(summ)
    p6.ws.save_table(S, "summary", "seeds")
    ref = fallback_reference(p6, sel)
    best = None
    for vname in variants:
        d = S[(S.method == vname) & (S.task == p6.prim)] if len(S) else S
        gg = S[(S.method == vname) & (S.task == "geo")] if len(S) else S
        if not len(d):
            continue
        c = {"name": f"p6_{vname}", "model_source": f"p6_{vname}", "members": fam,
             "disc_dti": float(d.dti_crossfit.iloc[0]), "disc_worst": float(d.fold_min.iloc[0]),
             "geo_dti": float(gg.dti_crossfit.iloc[0]) if len(gg) else None, "unique_systems": 0,
             "params_all": {t: json.loads(S[(S.method == vname) & (S.task == t)].params_all.iloc[0])
                            for t in ("geo", p6.prim) if ((S.method == vname) & (S.task == t)).any()}}
        pr = sc["promote"]
        ok, why = gate(c, ref, {"min_gain_over_ref": pr["min_disc_gain"], "max_geo_drop": pr["max_geo_drop"],
                                "max_worst_drop": pr["max_worst_drop"]})
        c.update(gate="PROMOTE" if ok else "REJECT", gate_reasons=why)
        if best is None or c["disc_dti"] > best["disc_dti"]:
            best = c
    if best is not None:
        p6.ws.save_json(json.loads(json.dumps(best, default=float)), "seeds/candidate.json")
    return {"summary": S, "candidate": best, "members": fam}
