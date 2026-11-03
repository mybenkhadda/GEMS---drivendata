"""Canonical evaluation with exact budgets (the GEMS scorer, masks and protocol keys; G07 exact-k selection).

Every model is scored on the same units with the same policy list and fractions; the selection report (requested /
actual fraction, cutoff, ties, legacy count) is stored with every row, so a tie-inflated area can never pass
unnoticed (``exact_budget_ok``). Pooled DTI from summed components; worst zone, per zone, systems recovered / lost
vs W2, candidate recall, promotion gates. Rows use the G06 column names (``rep`` = policy) so the G06 aggregation,
comparison and promotion helpers are reused unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..g06 import evaluation as _E6
from .representation import apply_policy

# canonical helpers reused unchanged from G06 (re-exported under the G07 namespace)
aggregate_components, compare_with_w2, dti = _E6.aggregate_components, _E6.compare_with_w2, _E6.dti
nested_policy_selection, per_zone, system_recovery_report = (_E6.nested_policy_selection, _E6.per_zone,
                                                             _E6.system_recovery_report)
_g06_promotion = _E6.promotion_decision
REFERENCE = "W2"


def unit_protocol(vu, template_crop: np.ndarray) -> dict:
    from ..g06.evaluation import unit_protocol as up

    return up(vu, template_crop)


def evaluate_map(vu, score: np.ndarray, policies: dict, fracs, proto: dict, tags: dict, cfg: dict,
                 secondary: np.ndarray | None = None, width_full: int | None = None) -> tuple[list, list]:
    """Rows (components + selection report) and system rows of one map on one unit."""
    from ..geomoe.scenarios import system_recovery
    from ..m7.protocol import score_row

    u = vu.unit
    tb = str((cfg.get("representation") or {}).get("tie_break", "secondary_then_hash"))
    seed = int((cfg.get("notebook") or {}).get("seed", 2026))
    ev = cfg.get("evaluation") or {}
    rows, srows = [], []
    s = np.where(u.ev, np.asarray(score, np.float64), np.nan)
    for pn, rep in policies.items():
        for f in fracs:
            q, rp = apply_policy(rep, s, u.ev, float(f), eligible=proto["elig"], known=vu.known, secondary=secondary,
                                 offset=(u.sl[0].start, u.sl[1].start), width_full=width_full, tie_break=tb, seed=seed)
            r = score_row(u, q, proto["elig"], proto["pkey"], rep=pn, frac=float(f), unit=vu.key,
                          scenario=vu.scenario, **tags)
            r["dti"] = dti(r["tp_w"], r["fp_w"], r["fn_w"])
            r["prediction_mass"] = float(q[u.ev].sum())
            r.update({f"sel_{k}": v for k, v in rp.items() if not isinstance(v, dict)})
            rows.append(r)
            S = system_recovery(u, q, vu.sysmap, float(ev.get("system_recovered_support", 0.5)),
                                float(ev.get("system_recovered_frac", 0.5)))
            for x in S.itertuples():
                srows.append({"unit": vu.key, "scenario": vu.scenario, "zone": vu.zone, "rep": pn, "frac": float(f),
                              "system_id": int(x.system_id), "n_px": int(x.n_px),
                              "support_mean": float(x.support_mean), "hit_share": float(x.hit_share),
                              "recovered": bool(x.recovered), **tags})
    return rows, srows


def evaluate_scenario(units: dict, maps: dict, scenario: str, policies: dict, fracs, templates: dict, cfg: dict,
                      strictness: dict, secondary: dict | None = None, width_full: int | None = None,
                      progress=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All models x units of one scenario (V1 discovery / V3 geographic / V2 completion)."""
    rows, srows = [], []
    for key, vu in sorted(units.items()):
        if vu.scenario != scenario:
            continue
        proto = unit_protocol(vu, templates[key])
        for m, d in maps.items():
            if key not in d:
                continue
            sec = (secondary or {}).get(key) if m != REFERENCE else None
            r, s = evaluate_map(vu, d[key], policies, fracs, proto, {"model": m, "strictness": strictness.get(m, "")},
                                cfg, sec, width_full)
            rows += r
            srows += s
        vu.unit.release()
        if progress:
            progress(f"  evaluated {key}")
    return pd.DataFrame(rows), pd.DataFrame(srows)


def evaluate_discovery(units, maps, policies, fracs, templates, cfg, strictness, **kw):
    """V1: complete withheld fault systems (primary)."""
    return evaluate_scenario(units, maps, "V1", policies, fracs, templates, cfg, strictness, **kw)


def evaluate_geographic(units, maps, policies, fracs, templates, cfg, strictness, **kw):
    """V3: geographic folds."""
    return evaluate_scenario(units, maps, "V3", policies, fracs, templates, cfg, strictness, **kw)


def evaluate_completion(units, maps, policies, fracs, templates, cfg, strictness, **kw):
    """V2: network completion (diagnostic)."""
    return evaluate_scenario(units, maps, "V2", policies, fracs, templates, cfg, strictness, **kw)


def compare_to_w2(summary: pd.DataFrame) -> pd.DataFrame:
    return compare_with_w2(summary, REFERENCE)


def exact_budget_ok(rows: pd.DataFrame, tol_px: int = 0) -> pd.DataFrame:
    """Per model: every top-policy row selected exactly k pixels (no tie inflation, no shortfall)."""
    if not len(rows) or "sel_k" not in rows:
        return pd.DataFrame(columns=["model", "rows", "exact", "max_excess", "max_shortfall", "tied_rows"])
    R = rows[rows.sel_policy.astype(str).str.startswith("g07.rep")] if "sel_policy" in rows else rows
    g = R.assign(excess=R.sel_actual_px - R.sel_k).groupby("model").agg(
        rows=("excess", "size"), max_excess=("excess", "max"), max_shortfall=("sel_shortfall", "max"),
        tied_rows=("sel_n_tied_at_cutoff", lambda x: int((x > 1).sum())))
    g["exact"] = (g.max_excess.abs() <= tol_px) & (g.max_shortfall <= tol_px)
    return g.reset_index()


def evaluate_candidate_recall(vu, lab: np.ndarray, pools: dict, tol_px: float = 3.0, share: float = 0.5
                              ) -> pd.DataFrame:
    """Geometric recall of the unit's truth systems by each pool, by the candidate inventory and by the union
    (EVALUATION SIDE - diagnostic, never a selection input)."""
    from scipy import ndimage

    t = vu.eval_truth()
    ids = vu.sysmap[t & (vu.sysmap >= 0)]
    if not ids.size:
        return pd.DataFrame()
    rows = []
    masks = {**{f"pool:{k}": np.asarray(v, bool) for k, v in pools.items()}, "inventory": np.asarray(lab) > 0}
    if pools:
        masks["union_pools"] = np.any(np.stack([np.asarray(v, bool) for v in pools.values()]), 0)
    tr, tc = np.nonzero(t & (vu.sysmap >= 0))
    sid = vu.sysmap[tr, tc]
    n_el = max(int(vu.unit.scored_mask().sum()), 1)
    for name, m in masks.items():
        if not m.any():
            near = np.zeros(sid.size, bool)
        else:
            near = (ndimage.distance_transform_edt(~m) <= tol_px)[tr, tc]
        df = pd.DataFrame({"sid": sid, "near": near}).groupby("sid").near.mean()
        rows.append({"unit": vu.key, "scenario": vu.scenario, "source": name, "n_systems": int(df.size),
                     "n_covered": int((df >= share).sum()), "recall": float((df >= share).mean()),
                     "px_share_near": float(near.mean()), "area_px": int(m.sum()),
                     "area_frac": float(m.sum() / n_el), "covered_ids": ";".join(map(str, df.index[df >= share]))})
    return pd.DataFrame(rows)


# =============================================================================== DEM feasibility (diagnostic)
def dem_signal_table(units: dict, dem_store, w2_maps: dict, cfg: dict, min_cov: float = 0.5,
                     reference: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-feature signal table, per-unit facts) on DEM-covered cells of the V1 / V3 units.

    Fault-near = scored truth dilated by ``near_px``; background = scored cells >= ``bg_km`` from EVERY mapped fault.
    AUC per feature (direction-free: |AUC - 0.5| + 0.5), Spearman with ``dem_coverage`` (artefact screen: tile edges /
    partial cells), scale stability (mean |rho| between the same operator at different scales), orientation
    consistency (axial agreement of DEM strike with the truth strike at truth cells; 0.5 = random) and false-positive
    support (feature at W2 exact-k false-positive cells vs W2 true-positive cells). HELD-OUT TRUTH: a diagnostic of
    the data source only - never a model input, never used to tune a policy."""
    from scipy import ndimage
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    from ..g06.supervision import build_orientation_targets
    from .representation import exact_k_mask

    fc = cfg.get("feasibility") or {}
    near_px, bg_km = int(fc.get("near_px", 1)), float(fc.get("background_km", 2.0))
    names = [n for n in dem_store.names if n != "dem_coverage"]
    ref = reference or {}
    rows_pos, rows_neg, facts = [], [], []
    orient_rows, fp_rows = [], []
    for key, vu in sorted(units.items()):
        if vu.scenario not in ("V1", "V3"):
            continue
        u = vu.unit
        win = (u.sl[0].start, u.sl[0].stop, u.sl[1].start, u.sl[1].stop)
        cov = np.nan_to_num(dem_store.coverage(win)) >= min_cov
        sc = u.scored_mask() & cov
        t = vu.eval_truth()
        if not (t & sc).any():
            facts.append({"unit": key, "covered_scored_px": int(sc.sum()), "covered_truth_px": 0, "n_systems": 0})
            continue
        near = ndimage.binary_dilation(t, iterations=near_px) & sc
        alllab = vu.known | t
        far = (ndimage.distance_transform_edt(~alllab) * 0.1 >= bg_km) & sc
        A = dem_store.load(names, win)
        cv_arr = dem_store.coverage(win)
        pos_vals = A[:, near].T
        neg_idx = np.flatnonzero(far.ravel())
        rng = np.random.default_rng(0)
        if neg_idx.size > 20 * max(int(near.sum()), 1):
            neg_idx = rng.choice(neg_idx, 20 * max(int(near.sum()), 1), replace=False)
        neg_vals = A.reshape(len(names), -1)[:, neg_idx].T
        rows_pos.append(pd.DataFrame(pos_vals, columns=names).assign(unit=key, dem_coverage=cv_arr[near]))
        rows_neg.append(pd.DataFrame(neg_vals, columns=names).assign(unit=key,
                                                                     dem_coverage=cv_arr.ravel()[neg_idx]))
        for rk, m in ref.items():
            if key in m:
                rows_pos[-1][rk] = np.asarray(m[key])[near]
                rows_neg[-1][rk] = np.asarray(m[key]).ravel()[neg_idx]
        ids = vu.sysmap[t & cov & (vu.sysmap >= 0)]
        facts.append({"unit": key, "scenario": vu.scenario, "covered_scored_px": int(sc.sum()),
                      "covered_truth_px": int((t & cov).sum()), "n_systems": int(np.unique(ids).size)})
        o = build_orientation_targets(t, 2.0, 0.3)
        tt = t & cov & np.isfinite(o["orient_cos2"])
        for n in names:
            if n.endswith("_cos2") and n.replace("_cos2", "_sin2") in names and tt.any():
                c2 = A[names.index(n)][tt]
                s2 = A[names.index(n.replace("_cos2", "_sin2"))][tt]
                agree = 0.5 * (1 + c2 * o["orient_cos2"][tt] + s2 * o["orient_sin2"][tt])
                orient_rows.append({"unit": key, "feature": n[:-5], "agreement": float(np.nanmean(agree)),
                                    "n": int(np.isfinite(agree).sum())})
        if key in w2_maps:
            el = u.scored_mask()
            m, _ = exact_k_mask(w2_maps[key], el, float(fc.get("fp_frac", 0.01)), offset=(u.sl[0].start,
                                                                                         u.sl[1].start))
            dt = ndimage.distance_transform_edt(~t) * 0.1
            fp = m & cov & (dt > 1.0)
            tp = m & cov & (dt <= 0.3)
            if fp.any() and tp.any():
                for j, n in enumerate(names):
                    fp_rows.append({"unit": key, "feature": n, "mean_at_w2_fp": float(np.nanmean(A[j][fp])),
                                    "mean_at_w2_tp": float(np.nanmean(A[j][tp]))})
    if not rows_pos:
        return pd.DataFrame(), pd.DataFrame(facts)
    import re

    P, N = pd.concat(rows_pos, ignore_index=True), pd.concat(rows_neg, ignore_index=True)
    cols = names + [k for k in ref if k in P]
    out = []
    for n in cols:
        y = np.r_[np.ones(len(P)), np.zeros(len(N))]
        v = np.r_[P[n].to_numpy(float), N[n].to_numpy(float)]
        ok = np.isfinite(v)
        if ok.sum() < 50 or np.unique(y[ok]).size < 2:
            continue
        auc = float(roc_auc_score(y[ok], v[ok]))
        out.append({"feature": n, "auc": auc, "auc_abs": abs(auc - 0.5) + 0.5, "n_pos": int((y[ok] == 1).sum()),
                    "n_neg": int((y[ok] == 0).sum()), "is_reference": n in ref})
    S = pd.DataFrame(out)
    if len(S):
        allv = pd.concat([P, N], ignore_index=True)
        S["rho_with_coverage"] = [abs(spearmanr(allv[f], allv["dem_coverage"], nan_policy="omit").correlation)
                                  for f in S.feature]
        S["family"] = S.feature.str.replace(r"_s\d+.*$|_w\d+.*$|_(fine|coarse)$", "", regex=True)
        stab = {}
        for fam, g in S.groupby("family"):
            fs = list(g.feature)
            if len(fs) > 1:
                rhos = [abs(spearmanr(allv[a], allv[b], nan_policy="omit").correlation) for i, a in enumerate(fs)
                        for b in fs[i + 1:]]
                stab[fam] = float(np.nanmean(rhos))
        S["scale_stability"] = S.family.map(stab)
        O = pd.DataFrame(orient_rows)
        if len(O):
            ag = O.groupby("feature").agreement.mean().to_dict()
            S["truth_orient_agreement"] = [ag.get(re.sub("_cos2$|_sin2$|_resultant$", "", f)) for f in S.feature]
        F = pd.DataFrame(fp_rows)
        if len(F):
            S = S.merge(F.groupby("feature")[["mean_at_w2_fp", "mean_at_w2_tp"]].mean(), on="feature", how="left")
    return S.sort_values("auc_abs", ascending=False).reset_index(drop=True), pd.DataFrame(facts)


def feasibility_decision(coverage: pd.DataFrame, signal: pd.DataFrame, facts: pd.DataFrame, recall: pd.DataFrame,
                         licence_ok: bool, has_local: bool, cfg: dict) -> dict:
    """GO / NO_GO / INCONCLUSIVE / BLOCKED_INPUT with every criterion and its value (predeclared thresholds)."""
    g = (cfg.get("feasibility") or {}).get("go_if") or {}
    if not has_local:
        return {"decision": "BLOCKED_INPUT", "reasons": ["no usable local high-resolution DEM tile (headers verified)"]}
    if not licence_ok:
        return {"decision": "BLOCKED_INPUT", "reasons": ["dem.research_use_ok is False (licence / permitted use)"]}
    cov = coverage.set_index("region").local_dem_share.get("eligible", 0.0) if len(coverage) else 0.0
    n_sys = int(facts.n_systems.sum()) if len(facts) and "n_systems" in facts else 0
    clean = signal[(~signal.get("is_reference", False).astype(bool)) &
                   (signal.get("rho_with_coverage", pd.Series(0, index=signal.index)).fillna(0) <
                    float(g.get("max_rho_with_coverage", 0.5)))] if len(signal) else signal
    best = float(clean.auc_abs.max()) if len(clean) else 0.5
    w2_ref = signal[signal.feature == "W2"].auc_abs if len(signal) else pd.Series(dtype=float)
    incr = 0
    if len(recall):
        r = recall[recall.source.isin(["pool:dem_score", "pool:w2"])]
        dem = set(";".join(r[r.source == "pool:dem_score"].covered_ids.astype(str)).split(";")) - {""}
        w2 = set(";".join(r[r.source == "pool:w2"].covered_ids.astype(str)).split(";")) - {""}
        incr = len(dem - w2)
    crit = {"eligible_dem_share": (cov, cov >= float(g.get("min_eligible_share", 0.05))),
            "systems_in_coverage": (n_sys, n_sys >= int(g.get("min_systems", 5))),
            "best_clean_feature_auc": (best, best >= float(g.get("min_auc", 0.6))),
            "incremental_systems_vs_w2": (incr, incr >= int(g.get("min_incremental_systems", 2)))}
    if not crit["systems_in_coverage"][1]:
        dec = "INCONCLUSIVE"
    elif all(v[1] for v in crit.values()):
        dec = "GO"
    else:
        dec = "NO_GO"
    return {"decision": dec, "criteria": {k: {"value": v[0], "pass": bool(v[1])} for k, v in crit.items()},
            "w2_reference_auc": float(w2_ref.iloc[0]) if len(w2_ref) else None,
            "reasons": [k for k, v in crit.items() if not v[1]],
            "note": "decided on held-out V1 / V3 truth: the DEM research direction was chosen with this information, "
                    "so later DEM-model claims must be confirmed on V3 and never re-tune these thresholds"}


def evaluate_promotion(cmp: pd.DataFrame, sysrep: pd.DataFrame, strictness: dict, cfg: dict, primary: dict,
                       exact: pd.DataFrame, leakage_ok: bool, recall_gain: dict | None = None) -> pd.DataFrame:
    """G06 gates (V1 gain, zone drop, V3 guard, net systems, strict validation) + exact budget + leakage; export
    parity is checked at export on the written file. A screening-only model is never promoted."""
    D = _g06_promotion(cmp, sysrep, strictness, cfg, primary)
    if not len(D):
        return D
    ex = dict(zip(exact.model, exact.exact)) if len(exact) else {}
    out = []
    for r in D.to_dict("records"):
        extra = []
        if not ex.get(r["model"], False):
            extra.append("exact_budget")
        if not leakage_ok:
            extra.append("label_leakage_audit")
        if recall_gain is not None and int(recall_gain.get(r["model"], 1)) < 1 and cfg.get("require_recall_gain"):
            extra.append("candidate_recall_gain")
        if extra:
            r["reasons"] = ";".join([x for x in [r.get("reasons", "")] + extra if x])
            if r["status"] == "PROMOTE_CANDIDATE":
                r["status"] = "RETAIN_W2"
        r["deferred_gates"] = "full_region_parity;submission_provenance"
        out.append(r)
    return pd.DataFrame(out)
