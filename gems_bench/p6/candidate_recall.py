"""P6.1 X2: candidate-pool recall ceiling and failure taxonomy (P6_RECOVERY_HANDOFF.md).

Question: is the remaining DTI ceiling caused by systems absent from every candidate map, by weak ranking, by poor
geometry, by excess false-positive coverage or by calibration? Saved / honest maps only (B0, B1, fallback fusion, W2,
their union, optional G02-C); no training.

Three numbers are kept apart in every table:

* ``existing``   the deployable ranker (X1 chosen policy, nested): a valid estimate.
* ``reachable``  systems recovered by ANY inventory map at a budget, and the best fixed source cell: in-sample
                 diagnostics (a max over sources), never a validation score.
* ``oracle``     segments of the inventory pool chosen WITH held-out labels (``role = oracle_*``): a diagnostic
                 upper bound; ``export_parity.guard_oracle`` refuses these rows in every selection.

Failure taxonomy (per missed system, evidence-based, first matching category; every flag is saved so other
priorities can be recomputed): insufficient_spatial_coverage, known_fault_masking, budget_limitation,
ranked_too_low, fragmented, displaced, orientation_inconsistency, no_candidate_support, other_unexplained.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from . import boost as B
from . import stack as ST

CATEGORIES = ("recovered", "insufficient_spatial_coverage", "known_fault_masking", "budget_limitation",
              "ranked_too_low", "fragmented", "displaced", "orientation_inconsistency", "no_candidate_support",
              "other_unexplained")
GROUPS = {"absent": ("no_candidate_support",), "ranking": ("budget_limitation", "ranked_too_low"),
          "geometry": ("fragmented", "displaced", "orientation_inconsistency"),
          "masking_coverage": ("known_fault_masking", "insufficient_spatial_coverage"),
          "unexplained": ("other_unexplained",)}
NEXT = {"absent": "structural candidate generation (lineament generation, graph tracing, endpoint continuation, "
                  "multiscale structural features)",
        "ranking": "feature engineering and ranker improvement (label-safe structural evidence, bounded experiments)",
        "geometry": "line tracing and geometric reconstruction (segment tracing, connectivity, representation)",
        "masking_coverage": "known-fault masking / eligibility handling (no new model)",
        "unexplained": "inspect examples before choosing a direction"}


# =============================================================================== pure helpers
def system_pixels(ctx, u) -> pd.DataFrame:
    """Scored truth pixels of the unit with their system id (crop coordinates; -1 ids dropped). Evaluation only."""
    from ..m7.protocol import system_ids

    gr, gc = u.scorer.g_rc
    sid = np.asarray(system_ids(ctx, u))
    d = pd.DataFrame({"system_id": sid, "r": np.asarray(gr), "c": np.asarray(gc)})
    return d[d.system_id >= 0].reset_index(drop=True)


def axial_diff_deg(a, b) -> np.ndarray:
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) % np.pi
    return np.degrees(np.minimum(d, np.pi - d))


def _moments(r, c):
    r, c = np.asarray(r, float), np.asarray(c, float)
    if r.size < 2:
        return np.nan, 1.0
    vrr, vcc, vrc = r.var(), c.var(), np.mean((r - r.mean()) * (c - c.mean()))
    th = float(np.mod(0.5 * np.arctan2(2 * vrc, vcc - vrr), np.pi))
    tr, det = vrr + vcc, vrr * vcc - vrc ** 2
    disc = np.sqrt(max(tr * tr / 4 - det, 0.0))
    l1, l2 = tr / 2 + disc, max(tr / 2 - disc, 0.0)
    return th, (1.0 - l2 / l1) if l1 > 0 else 1.0


def system_attributes(px: pd.DataFrame, layers: dict, known: np.ndarray | None, ref: np.ndarray | None,
                      elig: np.ndarray, px_km: float = 0.1, density_radius_px: int = 10) -> pd.DataFrame:
    """Descriptive, label-derived attributes of every truth system (evaluation only): length, orientation,
    curvature, fragmentation, distance to permitted known faults, eligible share, mean of each layer, local
    prediction density of ``ref`` (share of positive pixels in a (2R+1)^2 window)."""
    if not len(px):
        return pd.DataFrame(columns=["system_id", "n_px", "length_km"])
    shape = elig.shape
    dk = ndimage.distance_transform_edt(~np.asarray(known, bool)) * px_km if known is not None and \
        np.asarray(known, bool).any() else np.full(shape, np.inf)
    dens = ndimage.uniform_filter(np.asarray(ref, bool).astype(np.float32), 2 * int(density_radius_px) + 1,
                                  mode="constant") if ref is not None else None
    rows = []
    for sid, g in px.groupby("system_id"):
        r, c = g.r.to_numpy(), g.c.to_numpy()
        th, lin = _moments(r, c)
        r0, c0 = r.min(), c.min()
        box = np.zeros((r.max() - r0 + 1, c.max() - c0 + 1), bool)
        box[r - r0, c - c0] = True
        _, nfrag = ndimage.label(box, structure=np.ones((3, 3), bool))
        row = {"system_id": int(sid), "n_px": int(len(g)), "length_km": float(len(g) * px_km),
               "orientation_deg": float(np.degrees(th)) if np.isfinite(th) else np.nan,
               "linearity": float(lin), "curvature": float(1.0 - lin), "fragments": int(nfrag),
               "known_dist_km_median": float(np.median(dk[r, c])), "eligible_share": float(elig[r, c].mean()),
               "pred_density": float(dens[r, c].mean()) if dens is not None else np.nan}
        for k, a in layers.items():
            if a is not None:
                row[f"mean_{k}"] = float(np.nanmean(np.asarray(a, np.float64)[r, c])) if np.isfinite(
                    np.asarray(a, np.float64)[r, c]).any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def pool_geometry(px: pd.DataFrame, pool: np.ndarray, tol_px: float, factor: float) -> pd.DataFrame:
    """Per system, against the candidate pool (union of every inventory map at the largest budget): share of truth
    pixels within ``tol_px`` and within ``factor * tol_px`` of the pool, number of pool components touching the
    system (within ``tol_px``), and the median axial angle between the system and the nearest pool pixels."""
    cols = ["system_id", "near_tol_share", "near_far_share", "pool_components", "orient_diff_deg"]
    if not len(px):
        return pd.DataFrame(columns=cols)
    pool = np.asarray(pool, bool)
    if not pool.any():
        return pd.DataFrame({"system_id": sorted(px.system_id.unique()), "near_tol_share": 0.0, "near_far_share": 0.0,
                             "pool_components": 0, "orient_diff_deg": np.nan})
    d, (ir, ic) = ndimage.distance_transform_edt(~pool, return_indices=True)
    lab, _ = ndimage.label(pool, structure=np.ones((3, 3), bool))
    th_pool = B.axial_orientation(pool)
    rows = []
    for sid, g in px.groupby("system_id"):
        r, c = g.r.to_numpy(), g.c.to_numpy()
        dd = d[r, c]
        near, far = dd <= tol_px, dd <= factor * tol_px
        th_sys, _ = _moments(r, c)
        nr, nc = ir[r, c][far], ic[r, c][far]
        od = float(np.nanmedian(axial_diff_deg(th_sys, th_pool[nr, nc]))) if far.any() and np.isfinite(th_sys) else np.nan
        rows.append({"system_id": int(sid), "near_tol_share": float(near.mean()), "near_far_share": float(far.mean()),
                     "pool_components": int(len(set(lab[ir[r, c][near], ic[r, c][near]].tolist()) - {0})),
                     "orient_diff_deg": od})
    return pd.DataFrame(rows, columns=cols)


def classify(r: dict, t: dict) -> str:
    """One category per system (priority order of ``CATEGORIES``); ``r`` holds the flags of ``system_flags``."""
    if r.get("recovered_ref"):
        return "recovered"
    if r.get("eligible_share", 1.0) < float(t["coverage_min"]):
        return "insufficient_spatial_coverage"
    if r.get("known_dist_km_median", np.inf) <= float(t["known_near_km"]):
        return "known_fault_masking"
    if r.get("recovered_ref_ranker_larger_budget"):
        return "budget_limitation"
    if r.get("recovered_any"):
        return "ranked_too_low"
    if r.get("best_hit_share", 0.0) >= float(t["partial_hit_min"]) and \
            r.get("pool_components", 0) >= int(t["fragment_min_components"]):
        return "fragmented"
    if r.get("near_far_share", 0.0) >= 0.5 and r.get("near_tol_share", 0.0) < 0.5:
        return "displaced"
    if r.get("near_far_share", 0.0) >= 0.5 and np.isfinite(r.get("orient_diff_deg", np.nan)) and \
            r["orient_diff_deg"] > float(t["orientation_max_deg"]):
        return "orientation_inconsistency"
    if r.get("best_support", 0.0) < float(t["support_low"]) and r.get("near_far_share", 0.0) < 0.5:
        return "no_candidate_support"
    return "other_unexplained"


def group_of(cat: str) -> str | None:
    return next((g for g, cs in GROUPS.items() if cat in cs), None)


def system_flags(inv: pd.DataFrame, ref_frac: dict, ranker: str = "w2") -> pd.DataFrame:
    """Per (unit, system) recovery flags from the long inventory (source, rep, frac, support, hit share, recovered):
    the reference map (``source == 'reference'``), the reference RANKER at a budget larger than the unit's reference
    budget (``ref_frac``: unit key -> frac), any inventory source at any budget, the best support / hit share."""
    if not len(inv):
        return pd.DataFrame()
    key = ["unit", "zone", "system_id"]
    src = inv[inv.source != "reference"]
    best = src.groupby(key).agg(best_support=("support_mean", "max"), best_hit_share=("hit_share", "max"),
                                recovered_any=("recovered", "any"), n_px=("n_px", "first")).reset_index()
    ref = inv[inv.source == "reference"].groupby(key).agg(recovered_ref=("recovered", "any"),
                                                          ref_support=("support_mean", "max")).reset_index()
    rf = src.unit.map(lambda k: float(ref_frac.get(k, np.inf)))
    big = src[(src.source == ranker) & (src.frac > rf + 1e-12)]
    lb = big.groupby(key).recovered.any().rename("recovered_ref_ranker_larger_budget").reset_index()
    rec_by = src[src.recovered].groupby(key).source.agg(lambda s: ";".join(sorted(set(s)))).rename(
        "recovered_by").reset_index()
    out = best.merge(ref, on=key, how="left").merge(lb, on=key, how="left").merge(rec_by, on=key, how="left")
    for c in ("recovered_ref", "recovered_ref_ranker_larger_budget"):
        out[c] = out[c].astype("boolean").fillna(False).astype(bool)
    return out


def taxonomy(flags: pd.DataFrame, attrs: pd.DataFrame, geom: pd.DataFrame, t: dict) -> pd.DataFrame:
    key = ["unit", "system_id"]
    T = flags.merge(attrs, on=key, how="left", suffixes=("", "_attr")).merge(geom, on=key, how="left")
    T["category"] = [classify(r, t) for r in T.to_dict("records")]
    T["group"] = T.category.map(group_of)
    T["missed"] = T.category != "recovered"
    return T


def recall_decision(T: pd.DataFrame, t: dict, min_systems: int, margin: float) -> dict:
    """Dominant failure group among missed systems (count and length-weighted). INCONCLUSIVE below
    ``min_systems`` missed systems; MIXED when the top two groups are within ``margin``."""
    M = T[T.missed] if len(T) else T
    n = int(len(M))
    if not n:
        return {"verdict": "NO_MISSED_SYSTEMS", "n_missed": 0}
    by_n = M.group.value_counts(normalize=True).to_dict()
    by_len = (M.groupby("group").length_km.sum() / max(M.length_km.sum(), 1e-9)).to_dict()
    order = sorted(by_len, key=by_len.get, reverse=True)
    top = order[0]
    second = order[1] if len(order) > 1 else None
    if n < int(min_systems):
        verdict = "INCONCLUSIVE"
    elif second is not None and by_len[top] - by_len[second] < float(margin):
        verdict = "MIXED"
    else:
        verdict = top.upper()
    return {"verdict": verdict, "n_missed": n, "n_systems": int(len(T)), "dominant_group": top,
            "second_group": second, "share_by_count": by_n, "share_by_length": by_len,
            "next_direction": NEXT.get(top), "next_if_mixed": NEXT.get(second) if verdict == "MIXED" else None}


def sensitivity(flags: pd.DataFrame, attrs: pd.DataFrame, geom_by_factor: dict, t: dict, min_systems: int,
                margin: float) -> list[dict]:
    """Decision under alternative thresholds (support_low x {0.5, 1, 2}; displacement factor from
    ``geom_by_factor``): a conclusion that flips is reported, never hidden."""
    out = []
    for f, geom in geom_by_factor.items():
        for m in (0.5, 1.0, 2.0):
            tt = {**t, "support_low": float(t["support_low"]) * m}
            d = recall_decision(taxonomy(flags, attrs, geom, tt), tt, min_systems, margin)
            out.append({"displaced_factor": f, "support_low": tt["support_low"], "verdict": d["verdict"],
                        "dominant_group": d.get("dominant_group"), "share_by_length": d.get("share_by_length")})
    return out


def oracle_select(pool: np.ndarray, truth: np.ndarray, k: int | None, tol_px: int = 3) -> tuple[np.ndarray, int]:
    """DIAGNOSTIC ONLY (held-out labels): pool branches ordered by the share of their pixels within ``tol_px`` of the
    truth (then size), kept while the area stays <= ``k`` (``None``: every branch with any truth support)."""
    from ..postprocess.topology import _disk

    lab = B.branches(pool)
    n = int(lab.max())
    if not n:
        return np.zeros(np.shape(pool), bool), 0
    near = ndimage.binary_dilation(np.asarray(truth, bool), structure=_disk(int(tol_px))) if tol_px > 0 else \
        np.asarray(truth, bool)
    size = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    hit = np.bincount(lab[near & (lab > 0)], minlength=n + 1)[1:]
    prec = hit / np.maximum(size, 1)
    ids = np.arange(1, n + 1)
    keep = ids[prec > 0]
    order = keep[np.lexsort((-size[keep - 1], -prec[keep - 1]))]
    m = B.fill_budget(lab, order, int(k) if k is not None else int(size.sum()))
    return m, int(len(np.unique(lab[m & (lab > 0)])))


def attribute_contrast(T: pd.DataFrame, cols: list[str], min_n: int = 10) -> pd.DataFrame:
    """Recovered vs missed systems per attribute: n, medians, Mann-Whitney p (two-sided); ``insufficient`` when
    either side has fewer than ``min_n`` systems (no conclusion drawn from a few examples)."""
    from scipy.stats import mannwhitneyu

    rows = []
    for c in cols:
        if c not in T:
            continue
        a = T.loc[~T.missed, c].astype(float).dropna().to_numpy()
        b = T.loc[T.missed, c].astype(float).dropna().to_numpy()
        p = float(mannwhitneyu(a, b).pvalue) if a.size and b.size and np.ptp(np.r_[a, b]) > 0 else np.nan
        rows.append({"attribute": c, "n_recovered": a.size, "n_missed": b.size,
                     "median_recovered": float(np.median(a)) if a.size else np.nan,
                     "median_missed": float(np.median(b)) if b.size else np.nan, "mannwhitney_p": p,
                     "insufficient": bool(min(a.size, b.size) < min_n)})
    return pd.DataFrame(rows)


# =============================================================================== Colab side
def reference_cells(rc) -> tuple[dict, str]:
    """{disc zone: cell} of the deployed reference map: X1 chosen policy's held-out cells, else boost W2
    rep+frac nested choices, else the configured cell. Returns (cells, source)."""
    bst = rc.bst
    dec = rc.read_json("export_parity", "decision") or {}
    pz = rc.load_table("per_zone_metrics", "export_parity")
    if dec.get("chosen_policy") and len(pz):
        p = pz[(pz.policy == dec["chosen_policy"]) & (pz.objective == dec["objective"]) & (pz.task == bst.prim)]
        if len(p):
            return ({int(r.zone): {"candidate": r.candidate, "blend": float(r.blend), "rep": r.rep, "frac": float(r.frac)}
                     for r in p.itertuples()}, f"X1 {dec['chosen_policy']} ({dec['objective']})")
    n = bst.load_table("nested_per_zone", "benchmark")
    if len(n) and "space" in n:
        p = n[(n.space == "w2|rep+frac") & (n.task == bst.prim)]
        if len(p):
            return ({int(r.zone): {"candidate": "w2", "blend": 1.0, "rep": r.rep, "frac": float(r.frac)}
                     for r in p.itertuples()}, "boost benchmark w2|rep+frac nested choices")
    c = rc.cfg["candidate_recall"]["reference_fallback_cell"]
    return {int(u.zone): {"candidate": "w2", **c} for u in bst.disc_units()}, "configured fallback cell"


def _score(bst, u, srcs, cell):
    lam = float(cell["blend"])
    if cell["candidate"] == "fallback_fusion" or lam == 0.0:
        return srcs["fallback_fusion"]
    return ((1.0 - lam) * srcs["fallback_fusion"] + lam * srcs["w2"]).astype(np.float32) if lam < 1.0 else srcs["w2"]


def run(rc) -> dict:
    """X2 stage (``rc.candidate_recall()``)."""
    from ..gems3 import representation as R

    bst, p6 = rc.bst, rc.p6
    cfg = rc.cfg["candidate_recall"]
    stage = "candidate_recall"
    refs, ref_src = reference_cells(rc)
    fp = rc.fp(stage, cfg, rc.input_hashes(), refs)
    if rc._done(stage, fp):
        return rc.load(stage)
    t0 = rc._begin(stage, fp)
    budgets, reps = [float(f) for f in cfg["budgets"]], list(cfg["reps"])
    tol = int(cfg["oracle"]["tol_px"])
    pool_f = float(cfg["oracle"]["pool_frac"])
    cov, inv, orc, attrs, geoms, red = [], [], [], [], {f: [] for f in cfg["taxonomy"]["sensitivity_factors"]}, []
    avail = []
    for u in bst.units():
        srcs = bst.unit_sources(u)
        raw = bst.raw(u) if (u.is_disc or "g02c" in cfg["sources"]) else {}
        if "g02c" in cfg["sources"]:
            if "g02c" in raw:
                srcs["g02c"] = R.rank01(np.nan_to_num(raw["g02c"], nan=0.0), u.ev)
            avail.append({"unit": u.key, "source": "g02c", "available": "g02c" in raw})
        want = [s for s in cfg["sources"] if s in srcs]
        crest = {s: bst._crest(u, srcs[s]) for s in want}
        pool = np.zeros(u.ev.shape, bool)
        for rep in reps:
            for f in budgets:
                maps = {s: bst.binary(u, srcs[s], rep, f, crest[s]) for s in want}
                if "w2_fallback_union" in cfg["sources"]:
                    maps["w2_fallback_union"] = maps["w2"] | maps["fallback_fusion"]
                if abs(f - pool_f) < 1e-12:
                    pool |= np.logical_or.reduce([maps[s] for s in want])
                for s, q in maps.items():
                    role = "union (area up to 2x budget)" if s == "w2_fallback_union" else "inventory"
                    cov.append(bst._eval_q(u, q, candidate=s, rep=rep, frac=f, role=role))
                    if u.is_disc:
                        hs = bst._last_hits
                        inv += [{"unit": u.key, "zone": int(u.zone), "source": s, "rep": rep, "frac": f,
                                 "system_id": int(h.system_id), "n_px": int(h.n_px), "support_mean": float(h.support_mean),
                                 "hit_share": float(h.hit_share), "recovered": bool(h.recovered)}
                                for h in hs.itertuples()] if len(hs) else []
                names = list(maps)
                for i, a in enumerate(names):
                    for b in names[i + 1:]:
                        red.append({"unit": u.key, "rep": rep, "frac": f, "a": a, "b": b,
                                    "inter_px": int((maps[a] & maps[b]).sum()), "union_px": int((maps[a] | maps[b]).sum())})
        if u.is_disc:
            cell = refs.get(int(u.zone))
            if cell is not None:
                q = bst.binary(u, _score(bst, u, srcs, cell), cell["rep"], cell["frac"])
                row = bst._eval_q(u, q, candidate="reference", rep=cell["rep"], frac=cell["frac"], role="reference",
                                  blend=cell["blend"])
                cov.append(row)
                hs = bst._last_hits
                inv += [{"unit": u.key, "zone": int(u.zone), "source": "reference", "rep": cell["rep"],
                         "frac": cell["frac"], "system_id": int(h.system_id), "n_px": int(h.n_px),
                         "support_mean": float(h.support_mean), "hit_share": float(h.hit_share),
                         "recovered": bool(h.recovered)} for h in hs.itertuples()] if len(hs) else []
            else:
                q = np.zeros(u.ev.shape, bool)
            px = system_pixels(p6.ctx, u)
            lay = {"mag_hgrad_rank": ST.rank_in(raw["mag_hgrad"], u.ev) if "mag_hgrad" in raw else None,
                   "grav_hgrad_rank": ST.rank_in(raw["grav_hgrad"], u.ev) if "grav_hgrad" in raw else None,
                   "cnn_agreement": 1.0 - np.asarray(raw["cnn_disagree"]) if "cnn_disagree" in raw else None,
                   "mg_orient_agree": raw.get("mg_orient_agree"), "g02c": raw.get("g02c"),
                   "w2_rank": srcs["w2"], "fallback_rank": srcs["fallback_fusion"]}
            at = system_attributes(px, lay, u.known, q, bst.elig(u), float(rc.cfg["px_km"]),
                                   int(round(float(cfg["taxonomy"]["density_radius_km"]) / float(rc.cfg["px_km"]))))
            attrs.append(at.assign(unit=u.key, zone=int(u.zone)))
            for ff in geoms:
                geoms[ff].append(pool_geometry(px, pool, tol, float(ff)).assign(unit=u.key))
            if cfg["oracle"]["enabled"]:
                truth = u.eval_truth()
                for f in budgets:
                    k = int(round(f * int(bst.elig(u).sum())))
                    m, nseg = oracle_select(pool, truth, k, tol)
                    orc.append({**bst._eval_q(u, m, candidate="oracle_inventory_constrained", rep="pool", frac=f,
                                              role="oracle_diagnostic"), "segments": nseg})
                m, nseg = oracle_select(pool, truth, None, tol)
                orc.append({**bst._eval_q(u, m, candidate="oracle_inventory_unconstrained", rep="pool", frac=np.nan,
                                          role="oracle_diagnostic (no area limit)"), "segments": nseg})
        bst._raw.pop(u.key, None)
        u.release()
    C = pd.DataFrame(cov)
    I = pd.DataFrame(inv)
    O = pd.DataFrame(orc)
    A = pd.concat(attrs, ignore_index=True) if attrs else pd.DataFrame()
    t = cfg["taxonomy"]
    GM = {f: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for f, v in geoms.items()}
    F = system_flags(I, {u.key: refs[int(u.zone)]["frac"] for u in bst.disc_units() if int(u.zone) in refs}, "w2")
    main_f = float(t["displaced_tol_factor"])
    TX = taxonomy(F, A.drop(columns=["zone"], errors="ignore"), GM.get(main_f, pd.DataFrame()), t) if len(F) else \
        pd.DataFrame()
    dec = recall_decision(TX, t, int(cfg["min_systems_for_conclusion"]), float(cfg["decision_margin"])) if len(TX) \
        else {"verdict": "NO_SYSTEMS"}
    sens = sensitivity(F, A.drop(columns=["zone"], errors="ignore"), GM, t, int(cfg["min_systems_for_conclusion"]),
                       float(cfg["decision_margin"])) if len(F) else []
    tables = summarize(C, I, O, red, bst.prim, rc)
    for name, df in tables.items():
        rc.save_table(df, name, stage)
    rc.save_parquet(I, "system_inventory", stage)
    rc.save_table(TX, "missed_system_taxonomy", stage)
    rc.save_table(attribute_contrast(TX, list(cfg["contrast_attributes"]), int(cfg["min_systems_for_conclusion"]))
                  if len(TX) else pd.DataFrame(), "attribute_contrast", stage)
    rc.save_table(pd.DataFrame(avail), "source_availability", stage)
    rc.fig_coverage(tables["coverage_curves_data"], stage)
    examples(rc, TX, refs, stage)
    reach = tables["coverage_by_budget"]
    r5 = reach[(reach.source == "w2_fallback_union") & np.isclose(reach.frac, max(budgets))] if len(reach) else reach
    d = {"stage": stage, **dec, "sensitivity": sens, "reference": ref_src, "reference_cells": refs,
         "union_recall_at_max_budget": r5.system_recall.max() if len(r5) and "system_recall" in r5 else None,
         "rule": "absent -> candidate generation; ranking -> features / ranker; geometry -> tracing / reconstruction",
         "oracle_note": "oracle rows use held-out labels: diagnostic upper bound only, never a selection input"}
    rc.save_json(d, stage, "decision")
    rc._finish(stage, fp, t0, {"inputs": rc.input_hashes(), "reference": ref_src}, {"verdict": dec.get("verdict")},
               schema=list(cfg["sources"]), selection={"reference_cells": refs})
    return rc.load(stage)


def summarize(C: pd.DataFrame, I: pd.DataFrame, O: pd.DataFrame, red: list, prim: str, rc) -> dict:
    """Coverage by budget / zone, source attribution, redundancy, oracle upper bound and the curve data."""
    key = ["task", "candidate", "rep", "frac"]
    C = C[C.candidate != "reference"] if len(C) else C
    P = B.pooled(C, key).rename(columns={"candidate": "source"}) if len(C) else pd.DataFrame()
    PZ = B.pooled(C, key + ["zone"]).rename(columns={"candidate": "source"}) if len(C) else pd.DataFrame()

    def sysrec(df, by):
        if not len(df):
            return pd.DataFrame()
        g = df.groupby(by).agg(n_systems=("system_id", "size"), n_recovered=("recovered", "sum"),
                               len_total=("n_px", "sum"),
                               len_recovered=("n_px", lambda s: s[df.loc[s.index, "recovered"]].sum()),
                               completely_missed=("support_mean", lambda s: int((s <= 0).sum()))).reset_index()
        g["system_recall"] = g.n_recovered / g.n_systems.clip(lower=1)
        g["length_weighted_recall"] = g.len_recovered / g.len_total.clip(lower=1)
        return g

    Iv = I[I.source != "reference"] if len(I) else I
    SB = sysrec(Iv, ["source", "rep", "frac"])
    SZ = sysrec(Iv, ["zone", "source", "rep", "frac"])
    CB = P.merge(SB.assign(task=prim), on=["task", "source", "rep", "frac"], how="left") if len(P) else P
    CZ = PZ.merge(SZ.assign(task=prim), on=["task", "zone", "source", "rep", "frac"], how="left") if len(PZ) else PZ
    att = []
    if len(Iv):
        R = Iv[Iv.recovered].assign(sid=lambda d: d.zone.astype(str) + ":" + d.system_id.astype(str))
        for (rep, f), g in R.groupby(["rep", "frac"]):
            sets = {s: set(x.sid) for s, x in g.groupby("source")}
            for s, v in sets.items():
                others = set().union(*[w for k, w in sets.items() if k != s]) if len(sets) > 1 else set()
                att.append({"rep": rep, "frac": f, "source": s, "recovered": len(v), "only_this_source": len(v - others),
                            "not_recovered_by_w2": len(v - sets.get("w2", set())),
                            "only_ids": ";".join(sorted(v - others))})
    RD = pd.DataFrame(red)
    if len(RD):
        RD = RD.groupby(["rep", "frac", "a", "b"])[["inter_px", "union_px"]].sum().reset_index()
        RD["pixel_jaccard"] = RD.inter_px / RD.union_px.clip(lower=1)
        if len(Iv):
            S = {k: set(g.zone.astype(str) + ":" + g.system_id.astype(str)) for k, g in
                 Iv[Iv.recovered].groupby(["rep", "frac", "source"])}
            RD["system_jaccard"] = [len(S.get((r.rep, r.frac, r.a), set()) & S.get((r.rep, r.frac, r.b), set())) /
                                    max(len(S.get((r.rep, r.frac, r.a), set()) | S.get((r.rep, r.frac, r.b), set())), 1)
                                    for r in RD.itertuples()]
    ub = []
    dec = rc.read_json("export_parity", "decision") or {}
    ex = (dec.get("chosen_summary") or {}).get("disc_dti")
    OP = B.pooled(O, ["candidate", "frac"]) if len(O) else pd.DataFrame()
    for f in sorted({float(x) for x in CB.frac}) if len(CB) else []:
        d = CB[(CB.task == prim) & np.isclose(CB.frac, f) & (CB.source != "w2_fallback_union")]
        o = OP[(OP.candidate == "oracle_inventory_constrained") & np.isclose(OP.frac.astype(float), f)] if len(OP) else OP
        un = CB[(CB.source == "w2_fallback_union") & np.isclose(CB.frac, f)]
        ub.append({"frac": f, "existing_ranker_nested_disc_dti": ex, "existing_note": "X1 chosen deployable policy "
                   "(nested; its own budget choice, constant across rows)",
                   "best_fixed_source_cell_dti": float(d.dti.max()) if len(d) else np.nan,
                   "best_fixed_source_cell": (f"{d.loc[d.dti.idxmax(), 'source']}|{d.loc[d.dti.idxmax(), 'rep']}"
                                              if len(d) else None),
                   "best_fixed_note": "in-sample max over sources and reps (diagnostic, not a validation score)",
                   "union_system_recall": float(un.system_recall.max()) if len(un) and "system_recall" in un else np.nan,
                   "oracle_constrained_dti": float(o.dti.iloc[0]) if len(o) else np.nan,
                   "oracle_note": "held-out labels choose pool segments within the same eligible area: DIAGNOSTIC "
                                  "upper bound, role = oracle_diagnostic"})
    if len(OP):
        o = OP[OP.candidate == "oracle_inventory_unconstrained"]
        if len(o):
            ub.append({"frac": np.nan, "oracle_unconstrained_dti": float(o.dti.iloc[0]),
                       "oracle_note": "unconstrained diagnostic bound: every truth-touching pool segment, no area limit"})
    curves = CB[[c for c in ("task", "source", "rep", "frac", "dti", "system_recall", "length_weighted_recall",
                             "precision_w", "recall_w") if c in CB]].copy() if len(CB) else pd.DataFrame()
    if len(OP):
        curves = pd.concat([curves, OP.rename(columns={"candidate": "source"}).assign(task=prim, rep="pool")
                            [["task", "source", "rep", "frac", "dti", "precision_w", "recall_w"]]], ignore_index=True)
    return {"coverage_by_budget": CB, "coverage_by_zone": CZ, "candidate_source_attribution": pd.DataFrame(att),
            "candidate_redundancy": RD, "oracle_upper_bound": pd.DataFrame(ub), "coverage_curves_data": curves}


def examples(rc, TX: pd.DataFrame, refs: dict, stage: str) -> None:
    """``missed_system_examples.png`` (+ data CSV): one panel per category of interest (recovered, missed with
    candidate support, missed without, fallback-only, W2-only, fragmented / displaced). The reference annotation
    is drawn in magenta and labelled EVALUATION-ONLY; it never enters any map."""
    import matplotlib.pyplot as plt

    bst = rc.bst
    if not len(TX):
        rc.save_fig(None, "missed_system_examples", pd.DataFrame(), stage)
        return
    rb = TX.recovered_by.fillna("")
    picks = {"recovered": TX[TX.category == "recovered"],
             "missed, candidate support": TX[TX.category.isin(["ranked_too_low", "budget_limitation"])],
             "missed, no candidate support": TX[TX.category == "no_candidate_support"],
             "fallback-only": TX[rb.str.contains("fallback_fusion") & ~rb.str.contains(r"(^|;)w2(;|$)")],
             "W2-only": TX[rb.str.contains(r"(^|;)w2(;|$)") & ~rb.str.contains("fallback_fusion")],
             "fragmented / displaced": TX[TX.category.isin(["fragmented", "displaced"])]}
    chosen = [(k, v.sort_values("length_km", ascending=False).iloc[0]) for k, v in picks.items() if len(v)]
    pad = int(rc.cfg["candidate_recall"]["examples"]["pad_px"])
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), squeeze=False)
    rows = []
    units = {u.key: u for u in bst.disc_units()}
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (label, r) in zip(axes.ravel(), chosen):
        u = units.get(r.unit)
        if u is None:
            continue
        px = system_pixels(rc.p6.ctx, u)
        g = px[px.system_id == int(r.system_id)]
        H, W = u.ev.shape
        r0, r1 = max(int(g.r.min()) - pad, 0), min(int(g.r.max()) + pad + 1, H)
        c0, c1 = max(int(g.c.min()) - pad, 0), min(int(g.c.max()) + pad + 1, W)
        srcs = bst.unit_sources(u)
        cell = refs.get(int(u.zone), {"candidate": "w2", "blend": 1.0, "rep": "crest_1px", "frac": 0.01})
        qw = bst.binary(u, _score(bst, u, srcs, cell), cell["rep"], cell["frac"])
        qf = bst.binary(u, srcs["fallback_fusion"], cell["rep"], cell["frac"])
        ax.imshow(srcs["w2"][r0:r1, c0:c1], cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        for m, col, lab in ((qf, "#1f77b4", "fallback map"), (qw, "#ff7f0e", "deployed W2 map")):
            rr, cc = np.nonzero(m[r0:r1, c0:c1])
            ax.scatter(cc, rr, s=2, c=col, label=lab, linewidths=0)
        ax.scatter(g.c - c0, g.r - r0, s=3, c="magenta", marker="s", label="EVALUATION-ONLY reference", linewidths=0)
        ax.set_title(f"{label}\n{r.unit} system {int(r.system_id)} ({r.category}, {r.length_km:.1f} km)", fontsize=8)
        ax.legend(fontsize=6, loc="lower right")
        rows.append({"panel": label, "unit": r.unit, "system_id": int(r.system_id), "category": r.category,
                     "length_km": r.length_km, "window": [r0, r1, c0, c1], "reference_cell": cell,
                     "background": "W2 held-out rank (grey)", "truth_use": "evaluation-only overlay"})
        u.release()
    fig.suptitle("Candidate recall examples (magenta = held-out reference, EVALUATION ONLY; never used for inference)")
    rc.save_fig(fig, "missed_system_examples", pd.DataFrame(rows), stage)
