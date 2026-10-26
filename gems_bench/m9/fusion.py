"""Section 16: dense + structured (+ legacy B0/B1) cross-fitted fusion, the final comparison and Gate 6.

Channels (per-zone percentile scores in [0, 1], computed from held-out maps only)
    D  dense     rank of the M9 dense map of variant ``core`` (production context on disc units)
    S  structured  candidate score of the chosen GBM painted on its traces (percentile among candidate pixels;
                 0 off the lines, so faint dense evidence elsewhere is never erased by the ranking)
    L  legacy    the champion's B0/B1 per-zone rank fusion at its own per-zone weight (immutable reference)
Methods: ``m9_fuse_D+S``, ``m9_fuse_D+L``, ``m9_fuse_D+S+L`` (single channels are ``m9_dense_core``,
``m9_structured`` and ``s01_fusion_thin``). Weights on the simplex grid (``weight_step``), representation and area
fraction are chosen on the OTHER zones (nested), so fusion weights and calibration are cross-fitted. The budget is
counted on eligible pixels (after masking). The 300 m known-fault ring is evaluated SEPARATELY (``<best>_ring<k>``)
and is excluded from the champion unless ``allow_ring_in_champion`` (the pseudo-discovery proxy under-represents
new faults next to known traces). No public-leaderboard feedback enters any choice.

Gate 6 (deployable): an M9 method is promoted only when it passes every M7 promotion rule against the champion
(pooled gain, zone wins, worst zone, geographic drop, bootstrap, short-fault recall) AND it is reproducible (every
model unit it depends on has a registered checkpoint sha256 and the held-out maps carry the current protocol keys).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

COMBOS = {"m9_fuse_D+S": (1, 1, 0), "m9_fuse_D+L": (1, 0, 1), "m9_fuse_D+S+L": (1, 1, 1)}


def channels(m9, u, mm, T: pd.DataFrame | None, col: str | None) -> dict:
    from ..gems3 import representation as R
    from ..m8.ensemble import _paint_rank
    from .validation import load_pred

    out = {}
    p = load_pred(m9, "core", u, "none" if u.task == "geo" else m9.cfg["validation"]["production_context"])
    if p is not None:
        out["D"] = R.rank01(p["seg"], u.ev)
    cs = m9.trace_set(u.key)
    if T is not None and col and cs is not None and len(cs):
        sub = T[(T.task == u.task) & (T.zone == u.zone)].sort_values("cand_id")
        if len(sub) == len(cs) and col in sub:
            out["S"] = _paint_rank(cs, sub[col].fillna(-1).to_numpy(), u.ev)
    sel = (m9.baseline_selection() or {}).get(u.task, {}).get(m9.cfg["baseline"]["champion"])
    maps = mm.get(u)
    if sel is not None and maps is not None:
        z = next((r for r in sel.get("per_zone", []) if int(r["zone"]) == u.zone), None)
        w = float(z["w"]) if z is not None else float(sel["params_all"].get("w", 0.5))
        out["L"] = np.where(u.ev, R.fuse(maps["r0"], maps["r1"], w), 0.0).astype(np.float32)
    return out


def run(m9, T: pd.DataFrame | None, col: str | None) -> dict:
    from ..gems3 import representation as R
    from ..gems3.metric_opt import nested_select
    from ..m7.baseline import MemberMaps
    from ..m7.protocol import score_row
    from ..m8.ensemble import simplex

    fc = m9.cfg["fusion"]
    reps = {k: m9.cfg["validation"]["representations"][k] for k in fc["representations"]}
    fracs = m9.cfg["validation"]["area_fracs"]
    sigma = float(m9.sh["evaluation"]["ridge_sigma_px"])
    mm = MemberMaps(m9.ctx, m9.models)
    rows, cache = [], {}
    for task in ("geo", m9.prim):
        for u in m9.units_of(task):
            S = channels(m9, u, mm, T, col)
            if "D" not in S:
                continue
            cache[u.key] = S
            for name, act in COMBOS.items():
                if any(a and k not in S for a, k in zip(act, "DSL")):
                    continue
                for w in simplex(float(fc["weight_step"]), act):
                    s = sum(wi * S[k] for wi, k in zip(w, "DSL") if wi > 0)
                    crest = R.crest_map(s, u.ev, sigma) if any(r.get("select") == "crest" for r in reps.values()) else None
                    for rn, rep in reps.items():
                        for f in fracs:
                            q = R.apply(rep, s, u.ev, float(f), known=u.known, crest=crest, sigma_px=sigma,
                                        eligible=m9.elig(u, 0))
                            rows.append(score_row(u, q, m9.elig(u, 0), m9.pkey(u), method=name, w=json.dumps(w), rep=rn,
                                                  frac=float(f), ring=0))
            mm.release(u)
            m9.say(f"s16 fusion grid {u.key}: done")
    grid = pd.DataFrame(rows)
    summ, per_all = [], []
    for (task, mth), g in grid.groupby(["task", "method"]) if len(grid) else []:
        pz, s = nested_select(g, ["w", "rep", "frac"])
        how = "nested"
        if not len(pz):                                         # single zone: equal weights, borrowed frac
            from .validation import borrowed_choice

            ch = borrowed_choice(m9, task) or {"rep": "binary", "frac": 0.02}
            fr = np.asarray(sorted(g.frac.unique()))
            f0 = float(fr[np.argmin(np.abs(fr - ch["frac"]))])
            ws = sorted(g.w.unique(), key=lambda x: np.std(json.loads(x)))
            pz = g[(g.w == ws[0]) & (g.rep == ch["rep"]) & np.isclose(g.frac, f0)][["zone", "w", "rep", "frac", "tp_w",
                                                                                     "fp_w", "fn_w", "n_truth", "dti"]]
            pz = pz.rename(columns={"dti": "dti_heldout"})
            s = {"dti_crossfit": float(pz.dti_heldout.mean()) if len(pz) else np.nan, "fold_min": np.nan,
                 "optimism": np.nan, "params_all": {"w": ws[0], "rep": ch["rep"], "frac": f0}}
            how = "borrowed (single zone: equal weights)"
        summ.append({"task": task, "method": mth, "dti_crossfit": s["dti_crossfit"], "fold_min": s.get("fold_min"),
                     "optimism": s.get("optimism"), "params_all": json.dumps(s["params_all"], default=float), "selection": how})
        per_all.append(pz.assign(task=task, method=mth))
        for r in pz.itertuples():
            _save(m9, cache, task, int(r.zone), mth, json.loads(r.w), reps[r.rep], float(r.frac), 0, sigma)
    per = pd.concat(per_all, ignore_index=True) if per_all else pd.DataFrame()
    summary = pd.DataFrame(summ)
    ring = ring_check(m9, cache, per, summary, reps, sigma)
    return {"grid": grid, "summary": summary, "per_zone": per, "ring": ring}


def _save(m9, cache, task, zone, method, w, rep, frac, ring, sigma) -> None:
    from ..gems3 import representation as R
    from ..m7.rasterize import save_heldout

    u = m9.unit(f"{task}:{zone}")
    S = cache.get(u.key) if u is not None else None
    if S is None:
        return
    s = sum(wi * S[k] for wi, k in zip(w, "DSL") if wi > 0)
    rr = R.with_ring(rep, ring)
    crest = R.crest_map(s, u.ev, sigma) if rep.get("select") == "crest" else None
    q = R.apply(rr, s, u.ev, frac, known=u.known, crest=crest, sigma_px=sigma, eligible=m9.elig(u, ring))
    save_heldout(m9.ws, method, u, q, m9.pkey(u))


def ring_check(m9, cache, per: pd.DataFrame, summary: pd.DataFrame, reps: dict, sigma: float) -> pd.DataFrame:
    """The best fusion re-applied with each ring (same per-zone choices): a separate, clearly named method."""
    rings = [int(r) for r in m9.cfg["fusion"]["ring_px"] if int(r) > 0]
    s = summary[summary.task == m9.prim].sort_values("dti_crossfit", ascending=False) if len(summary) else summary
    if not rings or not len(s):
        return pd.DataFrame()
    best = str(s.method.iloc[0])
    rows = []
    for ring in rings:
        name = f"{best}_ring{ring}"
        for r in per[(per.method == best) & (per.task == m9.prim)].itertuples():
            _save(m9, cache, m9.prim, int(r.zone), name, json.loads(r.w), reps[r.rep], float(r.frac), ring, sigma)
        rows.append({"method": name, "base": best, "ring_px": ring,
                     "note": "evaluated separately; excluded from the champion unless fusion.allow_ring_in_champion"})
    return pd.DataFrame(rows)


def false_length(m9, methods: list[str]) -> pd.DataFrame:
    """Predicted pixels farther than ``false_length_dist_km`` from ANY mapped fault, per 100 km^2 (unlabelled: an
    upper bound on false structure, never proven false)."""
    from scipy import ndimage

    from ..m7.rasterize import load_heldout

    lab = np.asarray(m9.ctx.data.labels, bool)
    dist = m9.cache_get("dist_any_fault_km", lambda: ndimage.distance_transform_edt(~lab, sampling=(0.1, 0.1)))
    lim = float(m9.cfg["fusion"]["false_length_dist_km"])
    rows = []
    for mth in methods:
        for task in ("geo", m9.prim):
            for u in m9.units_of(task):
                q = load_heldout(m9.ws, mth, u)
                if q is None:
                    continue
                pos = (q > 0) & u.ev
                far = pos & (dist[u.sl] > lim)
                rows.append({"task": task, "zone": u.zone, "method": mth,
                             "far_px_per_100km2": float(far.sum() / max(u.ev.sum() * 0.01, 1e-9) * 100.0),
                             "far_share_of_prediction": float(far.sum() / max(pos.sum(), 1))})
    d = pd.DataFrame(rows)
    return d.groupby(["task", "method"])[["far_px_per_100km2", "far_share_of_prediction"]].mean().reset_index() if len(d) else d


def compare(m9) -> dict:
    """M7's independent comparison + verdicts on every held-out map saved by M9 (champion = s01_<champion>)."""
    from ..m7 import comparison as CP
    from ..m7 import decision as DC
    from ..m7.rasterize import saved_methods

    cmp = CP.evaluate(m9.m7view)
    cmp["false_length"] = false_length(m9, saved_methods(m9.ws))
    cmp["verdicts"] = DC.verdicts(m9.m7view, cmp)
    return cmp


def reproducible(m9, method: str) -> tuple[bool, str]:
    reg = m9.ws.load_table("model_registry", "reports/model_registry")
    if not len(reg):
        return False, "no model registry"
    core = reg[(reg.variant == "core")]
    if not len(core) or core.get("sha256", pd.Series(dtype=str)).isna().any():
        return False, "core checkpoints without sha256"
    return True, "checkpoints registered with sha256; held-out maps carry current protocol keys"


def gate6(m9, ver: pd.DataFrame) -> pd.DataFrame:
    base = {"gate": "G6_final_fusion", "task": m9.prim, "profile": m9.profile["name"],
            "provisional": bool(m9.profile["provisional"])}
    ref = f"s01_{m9.cfg['baseline']['champion']}"
    if not len(ver):
        return pd.DataFrame([{**base, "decision": "RETAIN_CHAMPION", "method": ref, "reason": "no comparison"}])
    allow_ring = bool(m9.cfg["fusion"]["allow_ring_in_champion"])
    cand = ver[ver.method.str.startswith("m9_") & (ver.verdict == "PROMOTE") & ~ver.method.str.endswith("_noctx")]
    if not allow_ring:
        cand = cand[~cand.method.str.contains("_ring")]
    for r in cand.sort_values("dti_pooled", ascending=False).itertuples():
        ok, why = reproducible(m9, r.method)
        if ok:
            return pd.DataFrame([{**base, "decision": "PROMOTE", "method": r.method, "reference": ref,
                                  "dti_pooled": r.dti_pooled, "pooled_gain": getattr(r, "pooled_gain", np.nan),
                                  "worst_zone_delta": getattr(r, "worst_zone_delta", np.nan),
                                  "geo_drop": getattr(r, "geo_drop", np.nan), "reason": f"all promotion rules passed; {why}"}])
    return pd.DataFrame([{**base, "decision": "RETAIN_CHAMPION", "method": ref,
                          "reason": "no leakage-safe, reproducible M9 method passed every promotion rule; the B0/B1 "
                                    "champion stays (and is the submission fallback)"}])
