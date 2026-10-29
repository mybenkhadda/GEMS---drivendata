"""One evaluation contract for every model (A_submitted_W2, A_W2_*, A_seg, B, C_*, D) and every scenario.

Selection protocols: ``frozen_rows`` (the submitted W2 cell, no selection), ``fixed_policy`` (ONE rule for every
scenario: the outer unit's cell is chosen on non-overlapping development units only; used for model selection) and
``summarize`` (per-scenario nested choice: a DIAGNOSTIC, never used for selection).

* Raster: a candidate model's score map holds its segment score on the segment's pixels and NaN elsewhere (only
  candidate pixels can be selected; deterministic tie-breaks, ``score_map``). Policies = the predeclared
  representations x area fractions (config ``evaluation``), applied by ``gems3.representation.apply`` on the unit's
  eligible domain: the SAME budget for every model; whole segments are kept, so the area can exceed the frac by at
  most one segment (<= ``max_segment_km``), which ``budget_rows`` checks against the export tolerance.
* Score: ``m7.protocol.score_row`` (official FastScorer components, pixel-exact known-fault exclusion for V1 / V2).
* Selection: (rep, frac) nested leave-one-zone-out inside each scenario (``gems3.metric_opt.nested_select``): the
  outer zone's cell is chosen on the other zones of the same scenario only. Pooled DTI = summed components.
* Uncertainty: paired spatial block bootstrap (``m7.protocol.paired_block_bootstrap``), descriptive only.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .scenarios import BANDS, banded_components, system_recovery

COMP = ["tp_w", "fp_w", "fn_w", "n_truth"]
TIE_EPS, TIE_EPS2 = 1e-7, 1e-10


def dti(tp, fp, fn) -> float:
    """Official DTI from summed components (same expression as ``p6.boost.dti`` / ``nested_select``)."""
    return float(tp / (tp + 0.2 * fp + 0.8 * fn + 1e-6))


# =============================================================================== pure: rasterisation
def score_map(lab: np.ndarray, labels: np.ndarray, scores: np.ndarray, n_px: np.ndarray | None = None) -> np.ndarray:
    """Segment score on its pixels, NaN elsewhere (one label per pixel: no duplicated mass), float64. Exact ties
    (e.g. ECDF saturation at 1) are broken first by segment length (``TIE_EPS``), then by label (``TIE_EPS2``), so
    no two segments share a value and the top-k budget overshoots by at most one segment."""
    n = int(lab.max())
    v = np.full(n + 1, np.nan, np.float64)
    lb = np.asarray(labels, int)
    s = np.asarray(scores, float)
    if n_px is not None and len(n_px):
        s = s + TIE_EPS * np.asarray(n_px, float) / max(float(np.max(n_px)), 1.0)
    s = s + TIE_EPS2 * (1.0 - lb / max(n, 1))
    v[lb] = s
    v[0] = np.nan
    return v[lab]


def budget_rows(rows: pd.DataFrame, tol: float) -> pd.DataFrame:
    """Fair-budget check: area of every evaluated cell <= frac x (1 + tol) of the eligible domain."""
    R = rows.copy()
    R["budget_ok"] = R.area_frac_eligible <= R.frac * (1.0 + float(tol)) + 1e-9
    return R


# =============================================================================== pure: tables
def pooled(G: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = G.groupby(by, dropna=False)[COMP].sum().reset_index()
    g["dti"] = [dti(a, b, c) for a, b, c in zip(g.tp_w, g.fp_w, g.fn_w)]
    g["recall_w"] = g.tp_w / (g.tp_w + g.fn_w).clip(lower=1e-9)
    g["precision_w"] = g.tp_w / (g.tp_w + g.fp_w).clip(lower=1e-9)
    return g


def nested(G: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Nested (rep, frac) choice per held-out zone on one (scenario, model) grid."""
    from ..gems3.metric_opt import nested_select

    if not len(G):
        return pd.DataFrame(), {"dti_crossfit": np.nan, "fold_min": np.nan}
    per, summ = nested_select(G, ["rep", "frac"])
    return per, summ


def summarize(G: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(summary per scenario x model, per-zone held-out rows) from the long grid."""
    S, P = [], []
    for (sc, m), g in G.groupby(["scenario", "model"]):
        per, s = nested(g)
        if not len(per):
            continue
        z = {int(r.zone): float(r.dti_heldout) for r in per.itertuples()}
        tp, fp, fn = (float(per[c].sum()) for c in ("tp_w", "fp_w", "fn_w"))
        S.append({"scenario": sc, "model": m, "dti": s["dti_crossfit"], "worst_zone_dti": s["fold_min"],
                  "worst_zone": min(z, key=z.get) if z else None, "insample_dti": s["dti_insample"],
                  "optimism": s["optimism"], "params_all": json.dumps(s["params_all"]), "tp_w": tp, "fp_w": fp,
                  "fn_w": fn, "per_zone": json.dumps(z), "n_zones": len(z), "fold_std": float(np.std(list(z.values())))})
        P.append(per.assign(scenario=sc, model=m))
    return pd.DataFrame(S), (pd.concat(P, ignore_index=True) if P else pd.DataFrame())


def overlap_share(a, b) -> float:
    """Share of unit ``a``'s evaluation pixels that lie inside unit ``b``'s evaluation zone (full-grid geometry)."""
    r0, r1 = max(a.sl[0].start, b.sl[0].start), min(a.sl[0].stop, b.sl[0].stop)
    c0, c1 = max(a.sl[1].start, b.sl[1].start), min(a.sl[1].stop, b.sl[1].stop)
    n = int(a.ev.sum())
    if r0 >= r1 or c0 >= c1 or not n:
        return 0.0
    ea = a.ev[r0 - a.sl[0].start:r1 - a.sl[0].start, c0 - a.sl[1].start:c1 - a.sl[1].start]
    eb = b.ev[r0 - b.sl[0].start:r1 - b.sl[0].start, c0 - b.sl[1].start:c1 - b.sl[1].start]
    return float((ea & eb).sum() / n)


def dev_units(units: list, max_overlap: float) -> tuple[dict, dict]:
    """({outer unit key: [development unit keys]}, {outer key: rule used}). Development units = every OTHER unit of
    any scenario whose evaluation zone overlaps the outer one by at most ``max_overlap`` in both directions (V2 / V3
    of the same fold always overlap, so they are never development data for each other). When no unit qualifies,
    the established within-scenario nested rule is used (the other zones of the same scenario) and flagged
    ``DEV_OVERLAP_FALLBACK`` (disclosed in every table: the outer unit then shares geography with its development
    data)."""
    out, rule = {}, {}
    for a in units:
        d = [b.key for b in units if b.key != a.key and overlap_share(a.unit, b.unit) <= max_overlap and
             overlap_share(b.unit, a.unit) <= max_overlap]
        if d:
            out[a.key], rule[a.key] = d, "CROSS_SCENARIO_NON_OVERLAPPING"
        else:
            out[a.key] = [b.key for b in units if b.scenario == a.scenario and b.key != a.key]
            rule[a.key] = "DEV_OVERLAP_FALLBACK"
    return out, rule


def policy_score(G: pd.DataFrame) -> pd.Series:
    """Selection criterion of one (rep, frac) cell on development rows: the MEAN over scenarios of the pooled DTI
    (equal weight per scenario, so V3's larger truth does not dominate)."""
    g = G.groupby(["rep", "frac", "scenario"])[["tp_w", "fp_w", "fn_w"]].sum()
    d = (g.tp_w / (g.tp_w + 0.2 * g.fp_w + 0.8 * g.fn_w + 1e-6)).rename("dti").reset_index()
    return d.groupby(["rep", "frac"]).dti.mean()


def fixed_policy(G: pd.DataFrame, dev: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """ONE selection rule for every scenario (no per-scenario retuning): the outer unit's (rep, frac) maximises
    ``policy_score`` on its development units (``dev_units``), then is applied to the outer unit. Returns (summary per
    scenario x model, per-unit held-out rows, {model: all-unit cell = the deployable policy})."""
    S, P, deploy = [], [], {}
    for m, g in G.groupby("model"):
        rows = []
        for key, h in g.groupby("unit"):
            d = g[g.unit.isin(dev.get(key, []))]
            if not len(d):
                rows.append({"unit": key, "status": "NO_DEVELOPMENT_UNIT"})
                continue
            ps = policy_score(d)
            rep, frac = ps.idxmax()
            r = h[(h.rep == rep) & (h.frac == frac)]
            if not len(r):
                continue
            r0 = r.iloc[0]
            rows.append({"unit": key, "scenario": r0.scenario, "zone": int(r0.zone), "model": m, "rep": rep,
                         "frac": float(frac), "dev_policy_score": float(ps.max()), "dev_units": ";".join(dev[key]),
                         **{c: float(r0[c]) for c in COMP}, "dti_heldout": dti(r0.tp_w, r0.fp_w, r0.fn_w),
                         "status": "HELD_OUT"})
        R = pd.DataFrame(rows)
        R = R[R.status == "HELD_OUT"] if len(R) and "status" in R else R
        if len(R):
            P.append(R)
            for sc, x in R.groupby("scenario"):
                z = dict(zip(x.zone.astype(int), x.dti_heldout))
                tp, fp, fn = (float(x[c].sum()) for c in ("tp_w", "fp_w", "fn_w"))
                S.append({"scenario": sc, "model": m, "dti": dti(tp, fp, fn), "worst_zone_dti": min(z.values()),
                          "worst_zone": min(z, key=z.get), "tp_w": tp, "fp_w": fp, "fn_w": fn,
                          "per_zone": json.dumps({int(k): float(v) for k, v in z.items()}), "n_zones": len(z),
                          "fold_std": float(np.std(list(z.values()))), "protocol": "fixed_policy"})
        ps_all = policy_score(g)
        if len(ps_all):
            deploy[m] = {"rep": str(ps_all.idxmax()[0]), "frac": float(ps_all.idxmax()[1]),
                         "policy_score_in_sample": float(ps_all.max())}
    S, P = pd.DataFrame(S), (pd.concat(P, ignore_index=True) if P else pd.DataFrame())
    for m, c in deploy.items():
        if len(S):
            S.loc[S.model == m, "params_all"] = json.dumps({"rep": c["rep"], "frac": c["frac"]})
    return S, P, deploy


def frozen_rows(G: pd.DataFrame, model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summary + per-unit rows of a model evaluated at ONE frozen cell (no selection at all)."""
    g = G[G.model == model]
    if not len(g):
        return pd.DataFrame(), pd.DataFrame()
    P = g.assign(dti_heldout=[dti(a, b, c) for a, b, c in zip(g.tp_w, g.fp_w, g.fn_w)], status="FROZEN_POLICY")
    S = []
    for sc, x in P.groupby("scenario"):
        z = dict(zip(x.zone.astype(int), x.dti_heldout))
        tp, fp, fn = (float(x[c].sum()) for c in ("tp_w", "fp_w", "fn_w"))
        S.append({"scenario": sc, "model": model, "dti": dti(tp, fp, fn), "worst_zone_dti": min(z.values()),
                  "worst_zone": min(z, key=z.get), "tp_w": tp, "fp_w": fp, "fn_w": fn,
                  "per_zone": json.dumps({int(k): float(v) for k, v in z.items()}), "n_zones": len(z),
                  "fold_std": float(np.std(list(z.values()))), "protocol": "frozen_submitted_policy",
                  "params_all": json.dumps({"rep": str(x.rep.iloc[0]), "frac": float(x.frac.iloc[0])})})
    return pd.DataFrame(S), P


def subtype_components(u, q: np.ndarray, sysmap: np.ndarray, events: pd.DataFrame) -> pd.DataFrame:
    """V2 per-subtype official components: truth = withheld pixels of one class; the other withheld classes are
    masked like the retained faults (pixel-exact, same convention), so each class is scored under its own mask."""
    from ..gems3.official import FastScorer

    if events is None or not len(events):
        return pd.DataFrame()
    cls_of = dict(zip(events.event_id.astype(int), events["class"]))
    ids = np.asarray(sysmap)
    truth = u.eval_truth()
    rows = []
    for c in sorted(set(cls_of.values())):
        mine = np.isin(ids, [k for k, v in cls_of.items() if v == c]) & truth
        other = truth & ~mine
        ex = (u.exclude if u.exclude is not None else np.zeros(truth.shape, bool)) | other
        s = FastScorer(mine, u.ev, ex, exclude_supports_tp=u.supports_tp)
        k = s.components(np.where(u.ev, np.nan_to_num(np.asarray(q, np.float64)), 0.0)).as_dict()
        rows.append({"subtype": c, "n_events": int(sum(1 for v in cls_of.values() if v == c)),
                     "truth_px": int(mine.sum()), **{x: k[x] for x in ("tp_w", "fp_w", "fn_w", "n_truth")},
                     "dti": dti(k["tp_w"], k["fp_w"], k["fn_w"])})
    return pd.DataFrame(rows)


def guard_oracle(G: pd.DataFrame) -> None:
    """Oracle rows never reach a selection."""
    if "role" in G and (G.role.astype(str) == "oracle").any():
        raise ValueError("oracle rows present in a selection table: oracle predictions are diagnostics only")


# =============================================================================== Colab side
def policy_q(G, u, smap: np.ndarray, rep: str, frac: float, crest=None) -> np.ndarray:
    from ..gems3 import representation as R

    sh = G.sh
    return R.apply(sh["baseline"]["representations"][rep], smap, u.ev, float(frac), known=u.known, crest=crest,
                   sigma_px=float(sh["evaluation"]["ridge_sigma_px"]), eligible=G.elig(u))


def eval_grid(G, su, smap: np.ndarray, model: str, extra: dict | None = None, cells=None) -> list[dict]:
    """Every predeclared (rep, frac) cell of one score map on one scenario unit (``cells``: an explicit list, e.g.
    the one frozen submitted cell)."""
    from ..gems3 import representation as R

    ec = G.cfg["evaluation"]
    u = su.unit
    cells = list(cells) if cells is not None else [(r, f) for r in ec["reps"] for f in ec["area_fracs"]]
    crest = R.crest_map(np.nan_to_num(smap, nan=0.0), u.ev, float(G.sh["evaluation"]["ridge_sigma_px"])) \
        if any(r.startswith("crest") for r, _ in cells) else None
    rows = []
    for rep, f in cells:
        q = policy_q(G, u, smap, rep, f, crest if rep.startswith("crest") else None)
        r = G.score(u, q)
        hs = system_recovery(u, q, su.sysmap, G.sys_support, G.sys_frac)
        rec = sorted(int(x) for x in hs.system_id[hs.recovered]) if len(hs) else []
        rows.append({**r, "scenario": su.scenario, "unit": su.key, "model": model, "rep": rep, "frac": float(f),
                     "n_systems": int(len(hs)), "n_recovered": len(rec), "systems": ";".join(map(str, rec)),
                     "role": "heldout", **(extra or {})})
    return rows


def chosen_details(G, su, smap: np.ndarray, model: str, rep: str, frac: float, short_ids: set,
                   dist_km: np.ndarray) -> dict:
    """Held-out chosen cell of one unit: binary map, spatial blocks, distance bands, systems."""
    from ..m7.protocol import block_components

    u = su.unit
    q = policy_q(G, u, smap, rep, frac)
    B = block_components(u, q, int(G.cfg["evaluation"]["block_px"])).assign(method=model, scenario=su.scenario)
    D = banded_components(u, q, dist_km, G.cfg["distance_bands_km"]).assign(scenario=su.scenario, zone=su.zone,
                                                                           model=model)
    hs = system_recovery(u, q, su.sysmap, G.sys_support, G.sys_frac).assign(scenario=su.scenario, zone=su.zone,
                                                                           model=model)
    hs["short"] = hs.system_id.isin(short_ids)
    return {"q": np.asarray(q) > 0, "blocks": B, "bands": D, "systems": hs}


def paired(blocks: pd.DataFrame, pairs: list[tuple[str, str]], n_boot: int, seed: int) -> pd.DataFrame:
    from ..m7.protocol import paired_block_bootstrap

    rows = []
    for sc, g in blocks.groupby("scenario"):
        have = set(g.method)
        for m, ref in pairs:
            if m in have and ref in have:
                rows.append({"scenario": sc, **paired_block_bootstrap(g, m, ref, int(n_boot), int(seed))})
    return pd.DataFrame(rows)


def band_table(D: pd.DataFrame) -> pd.DataFrame:
    if not len(D):
        return D
    g = D.groupby(["scenario", "model", "band"])[["tp_w", "fp_w", "fn_w", "n_truth"]].sum().reset_index()
    g["dti"] = [dti(a, b, c) for a, b, c in zip(g.tp_w, g.fp_w, g.fn_w)]
    g["recall_w"] = g.tp_w / (g.tp_w + g.fn_w).clip(lower=1e-9)
    g["band"] = pd.Categorical(g.band, BANDS, ordered=True)
    return g.sort_values(["scenario", "model", "band"])
