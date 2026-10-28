"""M1.3 budget / mask forensics: why a zone passes or fails the area budget, and how the deployed binary map differs
from the evaluated one. Pure functions + figures (PNG + plotting data); used by the preflight on the PERSISTED tables
of an earlier run and by the validation stage on the fresh zone-stitched build.

Budget contract (the one ``export_parity.audit_export`` applies to the written file and ``boost_audit.
budget_violations`` to the benchmark grid): ``frac`` is the share of the eligible domain the SELECTOR may take
(``representation._k``: pool = ceil(frac x eligible) score-ranked pixels); a thin representation then keeps part of
that pool (``skeleton_1px`` = top pool -> skeletonise; ``crest_1px`` = top crest pixels, all of them when fewer), so
the final area is <= frac by construction and the benchmark DTI was measured on exactly that thinned area. A zone
fails only on over-allocation, a positive pixel on a mapped fault (the official pixel-exact mask) or an empty map.

Observed on Drive (M1.2 run, 2026-10-28, ``validation/zone_budget.csv`` + ``per_zone_metrics.csv`` +
``parity/zone_stitched_parity.csv``): skeleton_1px @ 0.01, final area 0.46-0.49 % of eligible (pool yield ~0.47-0.49),
0 px on mapped faults, evaluated geo cells 0.39-0.41 %. The M1.1 check ``|area / frac - 1| <= tol`` rejected every
zone although no thin cell of the grid could ever satisfy it: CHECK_DEFINITION_MISMATCH, not a builder defect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ROOT_CAUSES = ("AREA_UNDER_ALLOCATION", "AREA_OVER_ALLOCATION", "KNOWN_FAULT_MASK_LEAK", "ELIGIBLE_MASK_MISMATCH",
               "RANKING_DOMAIN_MISMATCH", "MORPHOLOGY_BUDGET_MISMATCH", "FOLD_STITCHING_MISMATCH",
               "RASTER_ALIGNMENT_MISMATCH", "CHECK_DEFINITION_MISMATCH", "UNRESOLVED")
BUDGET_RULE = "upper_bound"                       # frac caps the selector pool; final area <= frac x (1 + tol)
# map codes of diagnostics/evaluated_vs_exported_maps.npz
CODE_NONE, CODE_BOTH, CODE_EVAL_ONLY, CODE_DEPLOY_ONLY = 0, 1, 2, 3


# =============================================================================== budget
def budget_table(zones: pd.DataFrame, frac: float, tol: float, evaluated: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-zone budget under the upper-bound contract, with the facts behind the verdict (pure).

    ``zones``: zone, frac_of_eligible, positive_on_known_px (+ eligible_px, positive_px, zone_px when known).
    ``evaluated``: zone, eval_positive_px, eval_area_frac_eligible of the evaluated geo cell (optional)."""
    z = zones.copy()
    f, t = float(frac), float(tol)
    z["budget_rule"] = BUDGET_RULE
    z["target_frac"] = f
    z["cap_frac"] = f * (1.0 + t)
    z["rel_error"] = (z.frac_of_eligible - f) / f                       # informational (final area vs frac)
    if "eligible_px" in z:
        z["pool_px"] = np.maximum(np.ceil(f * z.eligible_px.astype(float)), 1).astype(np.int64)
        z["cap_px"] = np.floor(z.cap_frac * z.eligible_px.astype(float) + 1e-9).astype(np.int64)
        if "positive_px" in z:
            z["pool_yield"] = z.positive_px / z.pool_px
            z["px_below_pool"] = z.pool_px - z.positive_px
    if evaluated is not None and len(evaluated):
        z = z.merge(evaluated, on="zone", how="left")
        if "positive_px" in z and "eval_positive_px" in z:
            z["deploy_vs_eval_px"] = z.positive_px / z.eval_positive_px.replace(0, np.nan)
    over = z.frac_of_eligible > z.cap_frac + 1e-9
    leak = z.positive_on_known_px.fillna(0) > 0
    empty = (z.positive_px <= 0) if "positive_px" in z else (z.frac_of_eligible <= 0)
    z["ok"] = ~(over | leak | empty)
    z["m1_1_two_sided_ok"] = (z.rel_error.abs() <= t) & ~leak             # the superseded M1.1 rule, for the record
    z["cause"] = [_cause(o, lk, e, ok, old, z.get("pool_yield", pd.Series(np.nan, index=z.index)).iloc[i])
                  for i, (o, lk, e, ok, old) in enumerate(zip(over, leak, empty, z.ok, z.m1_1_two_sided_ok))]
    return z


def _cause(over, leak, empty, ok, old_ok, yld) -> str:
    c = []
    if over:
        c.append("AREA_OVER_ALLOCATION")
    if leak:
        c.append("KNOWN_FAULT_MASK_LEAK")
    if empty:
        c.append("AREA_UNDER_ALLOCATION (empty prediction)")
    if ok and not old_ok:
        c.append("CHECK_DEFINITION_MISMATCH (M1.1 two-sided rule; final area below frac by design)")
    if ok and np.isfinite(yld) and yld < 1.0:
        c.append(f"thin representation keeps {yld:.3f} of the selector pool (by design)")
    return "; ".join(c) or "WITHIN_BUDGET"


def evaluated_area(V: pd.DataFrame, geo_task: str = "geo") -> pd.DataFrame:
    """zone, eval_positive_px, eval_area_frac_eligible of the evaluated geo cells (validation/per_zone_metrics)."""
    if V is None or not len(V) or "task" not in V:
        return pd.DataFrame(columns=["zone", "eval_positive_px", "eval_area_frac_eligible"])
    g = V[V.task.astype(str) == geo_task]
    out = pd.DataFrame({"zone": g.zone.astype(int).values})
    out["eval_positive_px"] = pd.to_numeric(g["area_px"], errors="coerce").values if "area_px" in g else np.nan
    out["eval_area_frac_eligible"] = (pd.to_numeric(g["area_frac_eligible"], errors="coerce").values
                                      if "area_frac_eligible" in g else np.nan)
    return out


def root_cause_summary(Z: pd.DataFrame) -> dict:
    """One verdict per zone + overall (pure): which ROOT_CAUSES the measured budget table supports."""
    zones = {}
    for r in Z.to_dict("records"):
        cls = [c for c in ROOT_CAUSES if c in str(r.get("cause", ""))]
        zones[str(int(r["zone"]))] = {"ok": bool(r["ok"]), "classes": cls or (["WITHIN_BUDGET"] if r["ok"] else
                                                                            ["UNRESOLVED"]),
                                      "cause": r.get("cause"), "frac_of_eligible": r.get("frac_of_eligible"),
                                      "pool_yield": r.get("pool_yield"), "positive_on_known_px":
                                      r.get("positive_on_known_px")}
    blocking = sorted({c for v in zones.values() if not v["ok"] for c in v["classes"]})
    return {"budget_rule": BUDGET_RULE, "zones": zones, "all_ok": all(v["ok"] for v in zones.values()),
            "blocking_classes": blocking}


# =============================================================================== evaluated vs deployed maps
def map_comparison(q_deploy: np.ndarray, q_eval: np.ndarray, zone_id: np.ndarray, eligible: np.ndarray,
                   known: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """(per-zone disagreement table, uint8 code map) between the deployed and the evaluated binary maps (pure).

    Attribution: an evaluated pixel absent from the deployed map that lies on a mapped fault was removed by the
    official mask (``masking``: the deployed eligible domain excludes mapped faults, the geo-unit evaluation counted
    them as truth); every other disagreement comes from the budget pool being re-ranked on the smaller deployed
    domain (``ranking_domain``). The representation and the score are the same (score parity is checked separately)."""
    D = np.nan_to_num(np.asarray(q_deploy, np.float32)) > 0
    E = np.nan_to_num(np.asarray(q_eval, np.float32)) > 0
    zid, el, kn = np.asarray(zone_id), np.asarray(eligible, bool), np.asarray(known, bool)
    code = np.full(D.shape, CODE_NONE, np.uint8)
    code[D & E] = CODE_BOTH
    code[E & ~D] = CODE_EVAL_ONLY
    code[D & ~E] = CODE_DEPLOY_ONLY
    rows = []
    for k in sorted(int(x) for x in np.unique(zid) if x >= 0):
        m = zid == k
        d, e = D & m, E & m
        both, eo, do = int((d & e).sum()), int((e & ~d).sum()), int((d & ~e).sum())
        eo_known = int((e & ~d & kn).sum())
        rows.append({"zone": k, "eval_px": int(e.sum()), "deploy_px": int(d.sum()), "both_px": both,
                     "eval_only_px": eo, "eval_only_on_mapped_fault_px": eo_known,
                     "eval_only_elsewhere_px": eo - eo_known, "deploy_only_px": do,
                     "deploy_outside_eligible_px": int((d & ~el).sum()),
                     "jaccard": both / max(both + eo + do, 1), "dice": 2 * both / max(int(e.sum() + d.sum()), 1),
                     "pixel_disagreement": eo + do,
                     "disagreement_masking_share": eo_known / max(eo + do, 1),
                     "disagreement_ranking_domain_share": (eo - eo_known + do) / max(eo + do, 1)})
    return pd.DataFrame(rows), code


def mask_alignment(Z: pd.DataFrame) -> pd.DataFrame:
    """Evaluated vs deployed eligible domain per zone (pure; needs eval_eligible_px / known_in_zone_px)."""
    cols = [c for c in ("zone", "zone_px", "eval_eligible_px", "eligible_px", "known_in_zone_px") if c in Z]
    M = Z[cols].copy()
    if {"eval_eligible_px", "eligible_px"} <= set(M):
        M["eval_minus_deploy_eligible_px"] = M.eval_eligible_px - M.eligible_px
        if "known_in_zone_px" in M:
            M["unexplained_by_mapped_faults_px"] = M.eval_minus_deploy_eligible_px - M.known_in_zone_px
    M["note"] = ("deployed eligible = valid & template finite & not a mapped fault (official pixel-exact mask); the "
                 "geo-unit evaluation keeps the zone's mapped faults as truth: the difference is by design")
    return M


# =============================================================================== figures
def _plt():
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def budget_figure(ws, Z: pd.DataFrame, rel: str = "figures/per_zone_area_budget.png") -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(Z))
    ax.bar(x - 0.2, Z.frac_of_eligible, 0.4, label="deployed final area / eligible")
    if "eval_area_frac_eligible" in Z:
        ax.bar(x + 0.2, Z.eval_area_frac_eligible, 0.4, label="evaluated cell area / eligible (grid)")
    ax.axhline(float(Z.target_frac.iloc[0]), color="k", ls="--", lw=1, label="frac (selector pool)")
    ax.axhline(float(Z.cap_frac.iloc[0]), color="r", ls=":", lw=1, label="cap frac x (1 + tol)")
    ax.set(xticks=x, xticklabels=Z.zone.astype(str), xlabel="geographic zone", ylabel="share of eligible px",
           title=f"Area budget ({BUDGET_RULE}): {'all zones OK' if Z.ok.all() else 'violations'}")
    ax.legend(fontsize=7)
    ws.save_fig(fig, rel, Z)


def mask_figure(ws, M: pd.DataFrame, rel: str = "figures/eligible_mask_comparison.png") -> None:
    if not {"eval_eligible_px", "eligible_px"} <= set(M):
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(M))
    ax.bar(x - 0.27, M.eval_eligible_px, 0.27, label="evaluation eligible px")
    ax.bar(x, M.eligible_px, 0.27, label="deployed eligible px")
    if "known_in_zone_px" in M:
        ax.bar(x + 0.27, M.known_in_zone_px, 0.27, label="mapped-fault px in zone")
    ax.set(xticks=x, xticklabels=M.zone.astype(str), yscale="log", xlabel="geographic zone",
           title="Eligible domain: evaluation vs deployment")
    ax.legend(fontsize=7)
    ws.save_fig(fig, rel, M)


def comparison_figure(ws, RC: pd.DataFrame, code: np.ndarray, step: int,
                      rel: str = "figures/evaluated_vs_exported.png") -> None:
    """Stacked disagreement per zone + a max-pooled code map (no fault layer is drawn)."""
    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.3]})
    x = np.arange(len(RC))
    b = np.zeros(len(RC))
    for col, lab in (("both_px", "both"), ("eval_only_on_mapped_fault_px", "evaluated only: removed by official mask"),
                     ("eval_only_elsewhere_px", "evaluated only: re-ranked pool"),
                     ("deploy_only_px", "deployed only: re-ranked pool")):
        a1.bar(x, RC[col], bottom=b, label=lab)
        b = b + RC[col].values
    a1.set(xticks=x, xticklabels=RC.zone.astype(str), xlabel="geographic zone", ylabel="px",
           title="Evaluated vs deployed binary map")
    a1.legend(fontsize=7)
    from matplotlib.colors import ListedColormap

    sm = code[::step, ::step]
    a2.imshow(sm, cmap=ListedColormap(["#f2f2f2", "#333333", "#1f77b4", "#d62728"]), vmin=0, vmax=3,
              interpolation="nearest")
    a2.set(xticks=[], yticks=[], title=f"grey: both | blue: evaluated only | red: deployed only (1/{step} sample)")
    ws.save_fig(fig, rel, RC)


def geo_figure(ws, PC: pd.DataFrame, floor: float, rel: str = "figures/geographic_score_comparison.png") -> None:
    if PC is None or not len(PC) or "metric" not in PC:
        return
    d = PC[PC.metric == "geo_dti"].copy()
    if not len(d):
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4))
    col = ["#2ca02c" if bool(c) else "#7f7f7f" for c in d.comparable_to_champion]
    ax.barh(d.value_id + " [" + d.protocol + "]", pd.to_numeric(d.value, errors="coerce"), color=col)
    ax.axvline(float(floor), color="r", ls="--", lw=1, label=f"internal geo floor {floor}")
    ax.set(xlabel="geographic DTI", title="Geographic DTI by protocol (green: comparable with the champion)")
    ax.legend(fontsize=7)
    ws.save_fig(fig, rel, d)
