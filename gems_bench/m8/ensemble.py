"""Section 14: Systems A / B / C, cross-fitted fusion, comparison and champion selection (Gate 6).

    System A  existing CNN champion: B0/B1 per-zone rank fusion + crest thinning (section 01, ``s01_fusion_thin``)
    System B  physics-guided discovery: ranked candidates, uncertainty-aware rasterisation (section 13, ``m8_physics``)
    System C  network completion: stochastic + deterministic hypotheses (section 08, ``m8_completion``; disc only)

Compatible score scales: every system contributes a per-zone PERCENTILE score in [0, 1] (A: rank fusion of B0 / B1 at
its own per-zone weight; B: candidate score painted on its lines; C: completion score painted on its hypotheses; 0 off
the lines). Combination s = wA SA + wB SB + wC SC on the simplex grid of step ``weight_step``; representation
(``binary`` / ``crest_1px``), budget fraction and weights are chosen on the OTHER zones (nested), so the fusion weights
are cross-fitted. Methods: ens_A, ens_B, ens_C, ens_A+B, ens_A+C, ens_A+B+C (geo: C is unavailable).

Comparison: M7's independent re-scoring of every saved held-out map (geographic DTI, official-style pseudo-discovery,
retained-FP alternative, worst zone, system / length / short-fault recall, density, regional DTI, known-fault
proximity, paired zone deltas, paired spatial block bootstrap) + candidate false-length density (predicted pixels
farther than ``false_length_dist_km`` from ANY mapped fault per 100 km^2: unlabelled, an upper bound, not proven
false). Verdicts: M7 rules (PROMOTE / RETAIN / REJECT / INCONCLUSIVE). Three folds / three splits: descriptive paired
evidence, never statistical certainty from millions of correlated pixels. Oracle numbers are never used here.
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
from scipy import ndimage

COMBOS = {"ens_A": (1, 0, 0), "ens_B": (0, 1, 0), "ens_C": (0, 0, 1), "ens_A+B": (1, 1, 0), "ens_A+C": (1, 0, 1),
          "ens_A+B+C": (1, 1, 1)}


def simplex(step: float, active: tuple) -> list[tuple]:
    n = int(round(1.0 / step))
    out = []
    for a, b in itertools.product(range(n + 1), repeat=2):
        c = n - a - b
        if c < 0:
            continue
        w = (a / n, b / n, c / n)
        if all((wi > 0) == bool(act) for wi, act in zip(w, active)):
            out.append(w)
    return out


def _paint_rank(cs, scores, ev) -> np.ndarray:
    from ..m7.rasterize import score_raster

    sr = score_raster(cs, scores, ev.shape)
    on = np.isfinite(sr) & ev
    out = np.zeros(ev.shape, np.float32)
    if on.any():
        out[on] = pd.Series(sr[on]).rank(pct=True).to_numpy()
    return out


def system_scores(m8, u, mm, T: pd.DataFrame | None, col: str | None, comp: dict) -> dict:
    """{A, B, C} per-zone percentile scores of one unit (missing systems absent)."""
    from ..gems3 import representation as R

    out = {}
    sel = (m8.baseline_selection() or {}).get(u.task, {}).get(m8.cfg["baseline"]["champion"])
    maps = mm.get(u)
    if sel is not None and maps is not None:
        z = next((r for r in sel["per_zone"] if int(r["zone"]) == u.zone), None)
        w = float(z["w"]) if z is not None else float(sel["params_all"].get("w", 0.5))
        out["A"] = np.where(u.ev, R.fuse(maps["r0"], maps["r1"], w), 0.0).astype(np.float32)
    cs = m8.pool(u.key)
    if T is not None and col and cs is not None and len(cs):
        sub = T[(T.task == u.task) & (T.zone == u.zone)].sort_values("cand_id")
        if len(sub) == len(cs) and col in sub:
            out["B"] = _paint_rank(cs, sub[col].fillna(-1).to_numpy(), u.ev)
    c = comp.get(u.key)
    if c is not None and len(c):
        out["C"] = _paint_rank(c, c.table.completion_score.to_numpy(), u.ev)
    return out


def run(m8, T: pd.DataFrame | None, col: str | None, comp_sets: dict) -> dict:
    from ..gems3 import representation as R
    from ..gems3.metric_opt import nested_select
    from ..m7.baseline import MemberMaps
    from ..m7.protocol import score_row
    from ..m7.rasterize import save_heldout

    ec = m8.cfg["ensemble"]
    reps = {k: m8.sh["baseline"]["representations"][k] for k in ec["representations"]}
    fracs = m8.sh["rasterization"]["area_fracs"]
    sigma = float(m8.sh["evaluation"]["ridge_sigma_px"])
    mm = MemberMaps(m8.ctx, m8.models)
    rows, cache = [], {}
    for task in ("geo", m8.prim):
        for u in m8.units_of(task):
            S = system_scores(m8, u, mm, T, col, comp_sets)
            if not S:
                continue
            cache[u.key] = S
            for name, act in COMBOS.items():
                if any(a and k not in S for a, k in zip(act, "ABC")):
                    continue
                for w in simplex(float(ec["weight_step"]), act):
                    s = sum(wi * S[k] for wi, k in zip(w, "ABC") if wi > 0)
                    crest = R.crest_map(s, u.ev, sigma) if any(r.get("select") == "crest" for r in reps.values()) else None
                    for rn, rep in reps.items():
                        for f in fracs:
                            q = R.apply(rep, s, u.ev, float(f), known=u.known, crest=crest, sigma_px=sigma,
                                        eligible=m8.elig(u, 0))
                            rows.append(score_row(u, q, m8.elig(u, 0), m8.pkey(u), method=name, w=json.dumps(w),
                                                  rep=rn, frac=float(f)))
            mm.release(u)
            m8.say(f"s14 fusion grid {u.key}: done")
    grid = pd.DataFrame(rows)
    summ, per_all = [], []
    for (task, mth), g in grid.groupby(["task", "method"]) if len(grid) else []:
        pz, s = nested_select(g, ["w", "rep", "frac"])
        summ.append({"task": task, "method": mth, "dti_crossfit": s["dti_crossfit"], "fold_min": s["fold_min"],
                     "optimism": s["optimism"], "params_all": json.dumps(s["params_all"], default=float)})
        per_all.append(pz.assign(task=task, method=mth))
        for r in pz.itertuples():
            u = m8.unit(f"{task}:{r.zone}")
            S = cache.get(u.key)
            if S is None:
                continue
            w = json.loads(r.w)
            s = sum(wi * S[k] for wi, k in zip(w, "ABC") if wi > 0)
            rep = reps[r.rep]
            crest = R.crest_map(s, u.ev, sigma) if rep.get("select") == "crest" else None
            q = R.apply(rep, s, u.ev, float(r.frac), known=u.known, crest=crest, sigma_px=sigma, eligible=m8.elig(u, 0))
            save_heldout(m8.ws, mth, u, q, m8.pkey(u))
    return {"grid": grid, "summary": pd.DataFrame(summ),
            "per_zone": pd.concat(per_all, ignore_index=True) if per_all else pd.DataFrame()}


def false_length(m8, methods: list[str]) -> pd.DataFrame:
    """Predicted pixels farther than ``false_length_dist_km`` from ANY mapped fault, per 100 km^2 (unlabelled)."""
    from ..m7.rasterize import load_heldout

    lab = np.asarray(m8.ctx.data.labels, bool)
    dist = m8.cache_get("dist_any_fault_km", lambda: ndimage.distance_transform_edt(~lab, sampling=(0.1, 0.1)))
    lim = float(m8.cfg["ensemble"]["false_length_dist_km"])
    rows = []
    for mth in methods:
        for task in ("geo", m8.prim):
            for u in m8.units_of(task):
                q = load_heldout(m8.ws, mth, u)
                if q is None:
                    continue
                pos = (q > 0) & u.ev
                far = pos & (dist[u.sl] > lim)
                rows.append({"task": task, "zone": u.zone, "method": mth,
                             "far_px_per_100km2": float(far.sum() / max(u.ev.sum() * 0.01, 1e-9) * 100.0),
                             "far_share_of_prediction": float(far.sum() / max(pos.sum(), 1))})
    d = pd.DataFrame(rows)
    return d.groupby(["task", "method"])[["far_px_per_100km2", "far_share_of_prediction"]].mean().reset_index() if len(d) else d


def compare(m8) -> dict:
    """M7's comparison + verdicts on every held-out map saved by M8 (one protocol, champion = s01_fusion_thin)."""
    from ..m7 import comparison as CP
    from ..m7 import decision as DC
    from ..m7.rasterize import saved_methods

    cmp = CP.evaluate(m8.view)
    cmp["false_length"] = false_length(m8, saved_methods(m8.ws))
    ver = DC.verdicts(m8.view, cmp)
    return {**cmp, "verdicts": ver}


def gate6(m8, ver: pd.DataFrame, pooled: pd.DataFrame) -> pd.DataFrame:
    if not len(ver):
        return pd.DataFrame([{"gate": "G6_deployable", "decision": "RETAIN_CHAMPION", "reason": "no comparison"}])
    new = ver[ver.method.str.startswith(("m8_", "ens_")) & (ver.verdict == "PROMOTE")]
    ref = ver[ver.verdict == "RETAIN"].method.iloc[0] if (ver.verdict == "RETAIN").any() else "s01_fusion_thin"
    if len(new):
        best = new.sort_values("dti_pooled", ascending=False).iloc[0]
        return pd.DataFrame([{"gate": "G6_deployable", "task": m8.prim, "decision": "PROMOTE", "method": best.method,
                              "dti_pooled": best.dti_pooled, "reference": ref, "pooled_gain": best.get("pooled_gain"),
                              "worst_zone_delta": best.get("worst_zone_delta"), "geo_drop": best.get("geo_drop"),
                              "profile": m8.profile["name"], "provisional": bool(m8.profile["provisional"]),
                              "reason": "every promotion criterion passed (nested, geographic, paired)"}])
    return pd.DataFrame([{"gate": "G6_deployable", "task": m8.prim, "decision": "RETAIN_CHAMPION", "method": ref,
                          "profile": m8.profile["name"], "provisional": bool(m8.profile["provisional"]),
                          "reason": "no M8 system or fusion passed the promotion rules; the B0/B1 champion stays"}])
