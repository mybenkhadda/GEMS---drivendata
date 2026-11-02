"""Evaluation under the canonical competition protocol (the same scorer, masks and keys as W2).

* Units: the GEMS units (``gems3.official``): V1 = ``disc_official`` (retained faults masked pixel-exactly, masked
  predictions ignored), V3 = ``geo``, V2 = ``v2_net``. Components from ``m7.protocol.score_row`` (FastScorer, 300 m
  linear kernel); every row carries the M7 protocol key (rows with different keys are never pooled).
* Budget domain: ``m7.protocol.eligible`` (evaluation zone & template & not excluded). Representations:
  ``gems3.representation.apply`` with the configured reps; the FIXED primary policy (``evaluation.primary``) is the
  same for every model. Representation selection is a separate, nested (leave-one-zone-out) analysis.
* Pooled DTI = tp / (tp + 0.2 fp + 0.8 fn) on SUMMED components - never a mean of zone DTIs; the worst-zone DTI is
  reported with every pooled number. System recovery = canonical support >= 0.5 on >= 50 % of a system's pixels.
* PR-AUC / calibration are diagnostics only (never a selection criterion).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COMP = ("tp_w", "fp_w", "fn_w")
STRICT_OK = ("STRICT_NESTED", "STRICT_LABEL_FREE", "STRICT_ISOLATED")


def dti(tp: float, fp: float, fn: float) -> float:
    return float(tp / (tp + 0.2 * fp + 0.8 * fn + 1e-6)) if (tp + fp + fn) > 0 else float("nan")


def unit_protocol(vu, template_crop: np.ndarray) -> dict:
    from ..m7.protocol import eligible, protocol_key

    u = vu.unit
    return {"elig": eligible(u, template_crop, 0, "eligible", True),
            "pkey": protocol_key(u, template_crop, "eligible", True)}


def evaluate_prediction_grid(vu, score: np.ndarray, reps: dict, fracs, proto: dict, tags: dict,
                             sys_support: float = 0.5, sys_frac: float = 0.5) -> tuple[list[dict], list[dict]]:
    """Official components + area + systems of one score map on one unit for every (rep, frac)."""
    from ..gems3 import representation as R
    from ..geomoe.scenarios import system_recovery
    from ..m7.protocol import score_row

    u = vu.unit
    s = np.where(u.ev, np.nan_to_num(np.asarray(score, np.float32), nan=-1.0), -1.0)
    rows, srows = [], []
    for rn, rep in reps.items():
        for f in fracs:
            q = R.apply(rep, s, u.ev, float(f), known=vu.known, eligible=proto["elig"])
            r = score_row(u, q, proto["elig"], proto["pkey"], rep=rn, frac=float(f), unit=vu.key,
                          scenario=vu.scenario, **tags)
            r["dti"] = dti(r["tp_w"], r["fp_w"], r["fn_w"])
            r["prediction_mass"] = float(q[u.ev].sum())
            rows.append(r)
            S = system_recovery(u, q, vu.sysmap, sys_support, sys_frac)
            for x in S.itertuples():
                srows.append({"unit": vu.key, "scenario": vu.scenario, "zone": vu.zone, "rep": rn, "frac": float(f),
                              "system_id": int(x.system_id), "n_px": int(x.n_px), "support_mean": float(x.support_mean),
                              "hit_share": float(x.hit_share), "recovered": bool(x.recovered), **tags})
    return rows, srows


def pr_auc(score: np.ndarray, vu, elig: np.ndarray, n_max: int = 2_000_000, seed: int = 0) -> float:
    """Average precision of the score for the scored truth pixels inside the budget domain (diagnostic)."""
    from sklearn.metrics import average_precision_score

    t = vu.eval_truth() & elig
    m = elig & np.isfinite(score)
    if not t.any() or not m.any():
        return float("nan")
    idx = np.flatnonzero(m.ravel())
    if idx.size > n_max:
        idx = np.random.default_rng(seed).choice(idx, n_max, replace=False)
    y = t.ravel()[idx]
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, np.asarray(score).ravel()[idx]))


def calibration_table(prob: np.ndarray, vu, elig: np.ndarray, bins: int = 10, tol_px: float = 3.0) -> pd.DataFrame:
    """Reliability of a probability map: mean prediction vs share of pixels within ``tol_px`` of scored truth."""
    from scipy import ndimage

    t = vu.eval_truth()
    near = ndimage.distance_transform_edt(~t) <= tol_px if t.any() else np.zeros(t.shape, bool)
    m = elig & np.isfinite(prob)
    p, y = np.asarray(prob)[m], near[m]
    if not p.size:
        return pd.DataFrame()
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    b = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    df = pd.DataFrame({"bin": b, "p": p, "y": y}).groupby("bin").agg(n=("p", "size"), mean_pred=("p", "mean"),
                                                                      share_near_truth=("y", "mean")).reset_index()
    return df.assign(unit=vu.key)


def aggregate_components(rows: pd.DataFrame, by=("model", "scenario", "rep", "frac")) -> pd.DataFrame:
    """Pooled DTI from summed components + worst zone + per-zone DTIs (one row per group)."""
    from ..m7.protocol import assert_comparable

    if not len(rows):
        return pd.DataFrame()
    by = [b for b in by if b in rows.columns]
    assert_comparable(rows, by=("task", "zone"))         # one protocol key per unit across every model / cell
    out = []
    for key, g in rows.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        tp, fp, fn = (float(g[c].sum()) for c in COMP)
        z = g.assign(dti_z=[dti(a, b, c) for a, b, c in zip(g.tp_w, g.fp_w, g.fn_w)])
        worst = z.loc[z.dti_z.idxmin()] if len(z) else None
        out.append({**dict(zip(by, key)), "dti": dti(tp, fp, fn), "tp_w": tp, "fp_w": fp, "fn_w": fn,
                    "precision_w": tp / max(tp + fp, 1e-9), "recall_w": tp / max(tp + fn, 1e-9),
                    "worst_zone_dti": float(worst.dti_z) if worst is not None else np.nan,
                    "worst_zone": int(worst.zone) if worst is not None else -1, "n_units": int(len(g)),
                    "area_px": int(g.area_px.sum()), "area_frac_eligible": float(
                        g.area_px.sum() / max(g.eligible_px.sum(), 1)),
                    "prediction_mass": float(g.prediction_mass.sum()) if "prediction_mass" in g else np.nan,
                    "zone_dtis": ";".join(f"{int(a)}:{b:.4f}" for a, b in zip(z.zone, z.dti_z))})
    return pd.DataFrame(out)


def per_zone(rows: pd.DataFrame) -> pd.DataFrame:
    if not len(rows):
        return pd.DataFrame()
    return rows.assign(dti=[dti(a, b, c) for a, b, c in zip(rows.tp_w, rows.fp_w, rows.fn_w)])[
        [c for c in ("model", "experiment", "scenario", "unit", "zone", "rep", "frac", "dti", "tp_w", "fp_w", "fn_w",
                     "area_px", "area_frac_eligible", "prediction_mass", "strictness") if c in rows.columns]]


def compare_with_w2(summary: pd.DataFrame, ref: str = "W2") -> pd.DataFrame:
    """Every model vs the W2 reference at the same (scenario, rep, frac): pooled and worst-zone deltas."""
    if not len(summary) or ref not in set(summary.model):
        return pd.DataFrame()
    R = summary[summary.model == ref].set_index(["scenario", "rep", "frac"])
    out = []
    for r in summary[summary.model != ref].itertuples():
        k = (r.scenario, r.rep, r.frac)
        if k not in R.index:
            continue
        b = R.loc[k]
        out.append({"model": r.model, "scenario": r.scenario, "rep": r.rep, "frac": r.frac, "dti": r.dti,
                    "dti_w2": float(b.dti), "delta": r.dti - float(b.dti), "worst_zone_dti": r.worst_zone_dti,
                    "worst_zone_dti_w2": float(b.worst_zone_dti),
                    "delta_worst": r.worst_zone_dti - float(b.worst_zone_dti)})
    return pd.DataFrame(out)


def system_recovery_report(sys_rows: pd.DataFrame, ref: str = "W2") -> pd.DataFrame:
    """Per (model, unit, rep, frac): systems recovered, new vs W2, lost vs W2 (ids listed)."""
    if not len(sys_rows) or ref not in set(sys_rows.model):
        return pd.DataFrame()
    key = ["unit", "rep", "frac"]
    rec = sys_rows[sys_rows.recovered].groupby(["model"] + key).system_id.apply(set)
    n_sys = sys_rows.groupby(["model"] + key).system_id.nunique()
    out = []
    for (m, *k), n in n_sys.items():
        r = rec.get((m, *k), set())
        b = rec.get((ref, *k), set())
        out.append({"model": m, **dict(zip(key, k)), "n_systems": int(n), "n_recovered": len(r),
                    "n_recovered_w2": len(b), "n_new": len(r - b), "n_lost": len(b - r),
                    "new_ids": ";".join(map(str, sorted(r - b))), "lost_ids": ";".join(map(str, sorted(b - r)))})
    T = pd.DataFrame(out)
    T["scenario"] = T.unit.str.split(":").str[0]
    return T


def nested_policy_selection(rows: pd.DataFrame, model: str, scenario: str) -> pd.DataFrame:
    """Leave-one-zone-out representation selection: for each zone, the (rep, frac) maximising pooled DTI on the OTHER
    zones of the scenario, then that zone's components under the chosen cell. The result is an honest estimate of
    'select the representation, then deploy' - never the in-sample best cell."""
    R = rows[(rows.model == model) & (rows.scenario == scenario)]
    out = []
    for z in sorted(R.zone.unique()):
        dev = R[R.zone != z]
        if not len(dev):
            continue
        g = dev.groupby(["rep", "frac"])[list(COMP)].sum()
        g["dti"] = [dti(a, b, c) for a, b, c in g[list(COMP)].to_numpy()]
        rep, frac = g.dti.idxmax()
        held = R[(R.zone == z) & (R.rep == rep) & (R.frac == frac)]
        if len(held):
            h = held.iloc[0]
            out.append({"model": model, "scenario": scenario, "zone": int(z), "chosen_rep": rep,
                        "chosen_frac": float(frac), "tp_w": float(h.tp_w), "fp_w": float(h.fp_w),
                        "fn_w": float(h.fn_w), "dti_zone": dti(h.tp_w, h.fp_w, h.fn_w)})
    T = pd.DataFrame(out)
    if len(T):
        T["nested_pooled_dti"] = dti(T.tp_w.sum(), T.fp_w.sum(), T.fn_w.sum())
    return T


def promotion_decision(cmp: pd.DataFrame, sysrep: pd.DataFrame, strictness: dict, cfg: dict, primary: dict
                       ) -> pd.DataFrame:
    """Gates (all required) at the fixed primary cell; ``export_parity_verified`` is deferred to the export stage.
    SCREENING_ONLY models stop at PROMISING_SCREENING_ONLY; failures RETAIN_W2."""
    out = []
    rep, frac = primary["rep"], float(primary["frac"])
    for m in sorted(set(cmp.model)) if len(cmp) else []:
        c1 = cmp[(cmp.model == m) & (cmp.scenario == "V1") & (cmp.rep == rep) & (cmp.frac == frac)]
        c3 = cmp[(cmp.model == m) & (cmp.scenario == "V3") & (cmp.rep == rep) & (cmp.frac == frac)]
        if not len(c1):
            out.append({"model": m, "status": "NOT_EVALUATED", "reasons": "no V1 row at the primary cell"})
            continue
        a = c1.iloc[0]
        s = sysrep[(sysrep.model == m) & (sysrep.scenario == "V1") & (sysrep.rep == rep) & (sysrep.frac == frac)] \
            if len(sysrep) else pd.DataFrame()
        net = int(s.n_new.sum() - s.n_lost.sum()) if len(s) else 0
        st = strictness.get(m, "SCREENING_ONLY")
        gates = {"v1_gain": a.delta >= float(cfg["min_gain_v1"]),
                 "zone_drop": a.delta_worst >= -float(cfg["max_zone_drop"]),
                 "v3_guard": bool(len(c3)) and float(c3.iloc[0].delta) >= float(cfg["min_v3_delta"]),
                 "net_systems": net >= int(cfg["min_net_systems"]),
                 "strict_validation": st in STRICT_OK}
        reasons = [k for k, v in gates.items() if not v]
        if not reasons:
            status = "PROMOTE_CANDIDATE"
        elif reasons == ["strict_validation"]:
            status = "PROMISING_SCREENING_ONLY"
        else:
            status = "RETAIN_W2"
        out.append({"model": m, "strictness": st, "status": status, "v1_dti": a.dti, "v1_dti_w2": a.dti_w2,
                    "v1_gain": a.delta, "v1_worst_delta": a.delta_worst,
                    "v3_delta": float(c3.iloc[0].delta) if len(c3) else np.nan, "net_systems": net,
                    "recommended_gain_met": bool(a.delta >= float(cfg.get("recommend_gain_v1", 0.01))),
                    "deferred_gates": "export_parity_verified", "reasons": ";".join(reasons)})
    return pd.DataFrame(out)
