"""Section 13: uncertainty-aware rasterisation of ranked candidates on the exact 100 m competition grid.

Budget: the candidate score is painted on its pixels (max where lines cross), converted to a per-unit percentile and
optionally blended with a REGIONAL percentile (rank within ``regional_block_km`` blocks, weight ``regional_beta``), so
one dense region cannot consume the whole budget. The top ``frac`` x |eligible| pixels are kept (eligible = scored,
template-finite, not a known fault, not in the optional ring): masked pixels never consume the budget.

Representations (``rasterization.representations``)::

    centerline          1-px lines (value 1)
    occupancy           selection on score x (alpha + (1 - alpha) MC occupancy): Monte Carlo geometry samples move
                        the budget towards positions that are stable under perturbation (outside the MC region the
                        occupancy is unknown and alpha = 1, i.e. plain centerline)
    uncertainty         kept core spread across its candidate's MC positional dispersion, w = round(pos_std / 100 m)
                        <= max_px, values 1 - d / (w + 1) (narrow, decaying; candidates without MC keep 1 px)
    confidence_width    the top quantile of kept pixels gets one extra pixel of width (binary)
    crest               selection restricted to crest pixels of the smoothed candidate score raster
    + optional ring     zero within ring_px of known faults (an experiment, not an organizer exclusion)

The 300 m tolerance of the metric is NOT a reason to predict 300 m-wide corridors. Every choice (rep, regional beta,
frac, ring) is nested over zones; the area after every mask and the final probability mass are recorded.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import ndimage


def regional_rank(sr: np.ndarray, ev: np.ndarray, block_px: int, beta: float) -> np.ndarray:
    on = np.isfinite(sr) & ev
    out = np.full(sr.shape, -np.inf, np.float32)
    if not on.any():
        return out
    v = sr[on]
    g = pd.Series(v).rank(pct=True).to_numpy()
    if beta <= 0:
        out[on] = g
        return out
    rr, cc = np.nonzero(on)
    b = (rr // block_px) * 100000 + cc // block_px
    loc = pd.Series(v).groupby(b).rank(pct=True).to_numpy()
    out[rr, cc] = (1.0 - beta) * g + beta * loc
    return out


def _spread(core: np.ndarray, sr_ids: np.ndarray, widths: np.ndarray, max_px: int) -> np.ndarray:
    out = core.astype(np.float32)
    for w in range(1, max_px + 1):
        sel = core & (widths[np.clip(sr_ids, 0, len(widths) - 1)] >= w) & (sr_ids >= 0)
        if not sel.any():
            continue
        d = ndimage.distance_transform_edt(~sel)
        out = np.maximum(out, np.where(d <= w, 1.0 - d / (w + 1.0), 0.0))
    return out.astype(np.float32)


def predict(u, cs, scores: np.ndarray, rep: dict, frac: float, ring_px: int, beta: float, elig_of, *,
            occupancy: np.ndarray | None = None, pos_std_m: np.ndarray | None = None, block_px: int = 200):
    """(prediction on the unit crop, area / mass ledger)."""
    from ..gems3 import representation as R
    from ..gems3.submission import ring_zone
    from ..m7.protocol import unit_ledger
    from ..m7.rasterize import score_raster, select_core, shape_core

    el = elig_of(u, ring_px)
    sr = score_raster(cs, scores, u.ev.shape)
    s = regional_rank(np.where(np.isfinite(sr), sr, np.nan), u.ev, block_px, beta)
    kind = rep.get("kind", "centerline")
    if kind == "occupancy" and occupancy is not None:
        a = float(rep.get("alpha", 0.5))
        occ = np.where(np.isfinite(occupancy), occupancy, 1.0)
        s = np.where(np.isfinite(s), s * (a + (1.0 - a) * occ), s)
    if kind == "crest":
        fill = np.where(np.isfinite(s), s, 0.0)
        cm = R.crest_map(ndimage.gaussian_filter(fill, 1.0), u.ev, 1.0)
        s = np.where(cm | (ndimage.maximum_filter(fill, 3) == fill), s, -np.inf)
    core = select_core(np.where(np.isfinite(s), s, -np.inf), el, frac)
    led = [unit_ledger(core, u, el, "core selection (top-frac of eligible candidate pixels)", ring_px)]
    if kind == "uncertainty":
        ids = np.full(u.ev.shape, -1, np.int32)
        for i, p in enumerate(cs.paths):
            ids[p[:, 0], p[:, 1]] = i
        w = np.zeros(len(cs), int) if pos_std_m is None else np.clip(np.nan_to_num(np.rint(pos_std_m / 100.0), nan=0),
                                                                      0, int(rep.get("max_px", 2))).astype(int)
        q = _spread(core, ids, w, int(rep.get("max_px", 2)))
    elif kind in ("confidence_width",):
        q = shape_core(core, np.where(np.isfinite(s), s, 0.0), {"kind": "confidence_width",
                                                              "hi_quantile": rep.get("hi_quantile", 0.8),
                                                              "extra_px": rep.get("extra_px", 1)})
    else:
        q = core.astype(np.float32)
    led.append(unit_ledger(q, u, el, f"representation {kind}", ring_px))
    if ring_px and u.known is not None:
        q = np.where(ring_zone(u.known, ring_px), 0.0, q).astype(np.float32)
        led.append(unit_ledger(q, u, el, f"ring {ring_px} px", ring_px))
    q = np.where(u.ev, q, 0.0).astype(np.float32)
    led.append({**unit_ledger(q, u, el, "final (scorer mask applied)", ring_px), "mass_eligible": float(q[el].sum()),
                "mass_total": float(q.sum())})
    return q, led


def unit_inputs(m8, u, T: pd.DataFrame, col: str, stab: pd.DataFrame):
    sub = T[(T.task == u.task) & (T.zone == u.zone)].sort_values("cand_id")
    scores = sub[col].fillna(-1).to_numpy()
    occ = m8.mc_occupancy()
    occ = None if occ is None else np.asarray(occ[u.sl], np.float32)
    pos = None
    if len(stab):
        s = stab[stab.unit == u.key].set_index("cand_id").mc_pos_std_m
        pos = s.reindex(sub.cand_id.to_numpy()).to_numpy()
    return scores, occ, pos


def run(m8, T: pd.DataFrame, sets: dict, col: str, stab: pd.DataFrame) -> dict:
    from ..gems3.metric_opt import nested_select
    from ..m7.protocol import score_row
    from ..m7.rasterize import save_heldout

    rz = m8.cfg["rasterization"]
    fracs = m8.sh["rasterization"]["area_fracs"]
    block = max(int(round(float(rz["regional_block_km"]) / 0.1)), 10)
    rows, inputs = [], {}
    for task in ("geo", m8.prim):
        for u in m8.units_of(task):
            cs = sets.get(u.key)
            if cs is None or not len(cs):
                continue
            sc, occ, pos = unit_inputs(m8, u, T, col, stab)
            inputs[u.key] = (u, cs, sc, occ, pos)
            rings = [int(r) for r in rz["ring_px"]] if u.is_disc else [0]
            for name, rep in rz["representations"].items():
                if rep["kind"] == "occupancy" and occ is None:
                    continue
                for beta in rz["regional_beta"]:
                    for f in fracs:
                        for ring in rings:
                            q, led = predict(u, cs, sc, rep, float(f), ring, float(beta), m8.elig, occupancy=occ,
                                             pos_std_m=pos, block_px=block)
                            rows.append(score_row(u, q, m8.elig(u, ring), m8.pkey(u), method="m8_physics", rep=name,
                                                  beta=float(beta), frac=float(f), ring=int(ring),
                                                  mass_eligible=led[-1]["mass_eligible"],
                                                  nonzero_frac_eligible=led[-1]["frac_of_eligible"]))
            m8.say(f"s13 {u.key}: done")
    grid = pd.DataFrame(rows)
    summ, per_all, abl = [], [], []
    for task, g in grid.groupby("task") if len(grid) else []:
        pz, s = nested_select(g, ["rep", "beta", "frac", "ring"])
        summ.append({"task": task, "method": "m8_physics", "score": col, "dti_crossfit": s["dti_crossfit"],
                     "dti_insample": s["dti_insample"], "optimism": s["optimism"], "fold_min": s["fold_min"],
                     "params_all": json.dumps(s["params_all"], default=float),
                     **{f"dti_zone{r.zone}": r.dti_heldout for r in pz.itertuples()}})
        per_all.append(pz.assign(task=task))
        for rep, gg in g.groupby("rep"):
            _, s2 = nested_select(gg, ["beta", "frac", "ring"])
            abl.append({"task": task, "rep": rep, "dti_crossfit": s2["dti_crossfit"], "fold_min": s2["fold_min"],
                        "mass_eligible_mean": float(gg.mass_eligible.mean()),
                        "nonzero_frac_eligible_mean": float(gg.nonzero_frac_eligible.mean())})
        for beta, gg in g.groupby("beta"):
            _, s3 = nested_select(gg, ["rep", "frac", "ring"])
            abl.append({"task": task, "rep": f"regional_beta={beta:g} (best rep)", "dti_crossfit": s3["dti_crossfit"],
                        "fold_min": s3["fold_min"]})
    per = pd.concat(per_all, ignore_index=True) if per_all else pd.DataFrame()
    ledger = []
    for r in per.itertuples() if len(per) else []:
        u, cs, sc, occ, pos = inputs[f"{r.task}:{r.zone}"]
        q, led = predict(u, cs, sc, rz["representations"][r.rep], float(r.frac), int(r.ring), float(r.beta), m8.elig,
                         occupancy=occ, pos_std_m=pos, block_px=block)
        save_heldout(m8.ws, "m8_physics", u, q, m8.pkey(u))
        ledger += [{**x, "rep": r.rep, "beta": r.beta, "frac": r.frac} for x in led]
    return {"grid": grid, "summary": pd.DataFrame(summ), "per_zone": per, "ablation": pd.DataFrame(abl),
            "ledger": pd.DataFrame(ledger)}
