"""Section 14: automatic decision report (rebuilt from the saved tables only; no recomputation).

Verdicts (``decision.promote_if`` / ``reject_if``), per method against the champion baseline on the primary task:

    PROMOTE       pooled gain >= min_pooled_gain AND wins >= min_zone_win_share of the zones AND worst-zone delta >=
                  worst_zone_delta AND geographic DTI not lower by more than max_geo_drop AND bootstrap
                  P(delta > 0) >= min_boot_p_positive AND short-fault recall not lower by more than max_short_recall_drop
    REJECT        pooled gain < pooled_gain_below OR worst-zone delta < worst_zone_delta_below
    RETAIN        the champion itself
    INCONCLUSIVE  everything else, and every method with fewer than ``comparison.min_zones_for_claim`` zones

No verdict rests on one aggregate: each row lists every criterion with its value and pass / fail.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd


def verdicts(m7, cmp: dict) -> pd.DataFrame:
    cfg = m7.cfg
    pi, ri = cfg["decision"]["promote_if"], cfg["decision"]["reject_if"]
    prim = cfg["evaluation"]["primary_task"]
    P, pz, bt = cmp["pooled"], cmp["paired"], cmp["bootstrap"]
    ref = cmp["champion_reference"]
    if not len(P):
        return pd.DataFrame()
    pp, pg = P[P.task == prim].set_index("method"), P[P.task == "geo"].set_index("method")
    rows = []
    for mth in pp.index:
        r = {"method": mth, "reference": ref, "task": prim}
        if mth == ref:
            rows.append({**r, "verdict": "RETAIN", "reason": "champion baseline (reference of every comparison)",
                         "dti_pooled": float(pp.loc[mth, "dti_pooled"])})
            continue
        gain = float(pp.loc[mth, "dti_pooled"] - pp.loc[ref, "dti_pooled"]) if ref in pp.index else np.nan
        z = pz[(pz.task == prim) & (pz.method == mth)] if len(pz) else pz
        win = float((z.delta_vs_ref > 0).mean()) if len(z) else np.nan
        worst = float(z.delta_vs_ref.min()) if len(z) else np.nan
        geo_drop = float(pg.loc[ref, "dti_pooled"] - pg.loc[mth, "dti_pooled"]) if (mth in pg.index and ref in pg.index) \
            else np.nan
        b = bt[(bt.task == prim) & (bt.method == mth)] if len(bt) else bt
        bp = float(b.p_delta_positive.iloc[0]) if len(b) else np.nan
        sdrop = float(pp.loc[ref, "short_recall_w"] - pp.loc[mth, "short_recall_w"]) if ref in pp.index else np.nan
        n_z = int(pp.loc[mth, "n_zones"])
        crit = {"pooled_gain": (gain, np.isfinite(gain) and gain >= pi["min_pooled_gain"]),
                "zone_win_share": (win, np.isfinite(win) and win >= pi["min_zone_win_share"]),
                "worst_zone_delta": (worst, np.isfinite(worst) and worst >= pi["worst_zone_delta"]),
                "geo_drop": (geo_drop, (not np.isfinite(geo_drop)) or geo_drop <= pi["max_geo_drop"]),
                "boot_p_positive": (bp, np.isfinite(bp) and bp >= pi["min_boot_p_positive"]),
                "short_recall_drop": (sdrop, (not np.isfinite(sdrop)) or sdrop <= pi["max_short_recall_drop"])}
        rej = (np.isfinite(gain) and gain < ri["pooled_gain_below"]) or (np.isfinite(worst) and
                                                                          worst < ri["worst_zone_delta_below"])
        if n_z < int(cfg["comparison"]["min_zones_for_claim"]):
            v, why = "INCONCLUSIVE", f"only {n_z} zones"
        elif all(ok for _, ok in crit.values()):
            v, why = "PROMOTE", "every promotion criterion passed"
        elif rej:
            v, why = "REJECT", "pooled loss or a large worst-zone loss"
        else:
            v, why = "INCONCLUSIVE", "failed: " + ", ".join(k for k, (_, ok) in crit.items() if not ok)
        rows.append({**r, "verdict": v, "reason": why, "dti_pooled": float(pp.loc[mth, "dti_pooled"]), "n_zones": n_z,
                     **{k: val for k, (val, _) in crit.items()}, **{f"pass_{k}": ok for k, (_, ok) in crit.items()}})
    return pd.DataFrame(rows)


def _t(m7, name, sub) -> pd.DataFrame:
    return m7.ws.load_table(name, sub)


def answers(m7, cmp: dict, ver: pd.DataFrame) -> list[dict]:
    cfg = m7.cfg
    prim = cfg["evaluation"]["primary_task"]
    out = []
    ref_rec = (cfg.get("recorded_reference") or {}).get("candidate_system_recall")
    od = _t(m7, "gate_decision", "candidate_generation/oracle")
    dd = _t(m7, "decision", "dem/candidates")
    mcd = _t(m7, "decision", "map_completion/metrics")
    mcs = _t(m7, "selection", "map_completion/metrics")
    rk = _t(m7, "ranking_summary", "candidate_ranking/models")
    P = cmp.get("pooled", pd.DataFrame())

    def add(q, a, ev):
        out.append({"question": q, "answer": a, "evidence": ev})

    if len(dd):
        r = dd.iloc[0]
        add("1. Did DEM add previously missed geological structures?",
            f"{r.decision}: {int(r.dem_only_systems)} systems covered only by DEM candidates, "
            f"{int(r.missed_by_b0b1_recovered_by_dem)} of them missed by B0 and B1",
            f"dem/candidates/decision.csv ({r.reason})")
    else:
        add("1. Did DEM add previously missed geological structures?", "NOT EVALUATED (sections 06-07 not run)", "-")
    po = od[od.task == prim] if len(od) else od
    if len(po):
        r = po.iloc[0]
        add("2. Did candidate coverage improve?",
            f"system recall at the gate density {r.system_recall_at_max_density:.3f} vs recorded {ref_rec}; "
            f"decision {r.decision}", "candidate_generation/oracle/gate_decision.csv")
    else:
        add("2. Did candidate coverage improve?", "NOT EVALUATED (section 04 not run)", "-")
    if len(mcs):
        b = mcs.sort_values("dti_crossfit", ascending=False).groupby("class").head(1)
        txt = "; ".join(f"{r['class']}: {r.completer} recall {r.recall_crossfit:.3f}" for _, r in b.iterrows())
        add("3. Did map completion recover useful fault geometry?",
            f"{mcd.decision.iloc[0] if len(mcd) else '?'} - {txt}", "map_completion/metrics/selection.csv")
    else:
        add("3. Did map completion recover useful fault geometry?", "NOT EVALUATED (section 09 not run)", "-")
    if len(rk):
        g = rk[rk.task == prim]
        lf = g[g.method == "label_free"].dti_crossfit
        best = g[~g.method.isin(["label_free", "neural_support"])].sort_values("dti_crossfit", ascending=False).head(1)
        if len(best) and len(lf):
            d = float(best.dti_crossfit.iloc[0] - lf.iloc[0])
            add("4. Did ranking improve beyond simple structural scores?",
                f"{'YES' if d > 0 else 'NO'}: {best.method.iloc[0]} {best.dti_crossfit.iloc[0]:.4f} vs label-free "
                f"{lf.iloc[0]:.4f} (delta {d:+.4f}, cross-fitted)", "candidate_ranking/models/ranking_summary.csv")
    if not any(o["question"].startswith("4.") for o in out):
        add("4. Did ranking improve beyond simple structural scores?", "NOT EVALUATED (section 11 not run / gated)", "-")
    pr = ver[ver.verdict == "PROMOTE"] if len(ver) else ver
    top = (pr.sort_values("dti_pooled", ascending=False).method.iloc[0] if len(pr) else
           (ver[ver.verdict != "RETAIN"].sort_values("dti_pooled", ascending=False).method.iloc[0]
            if len(ver[ver.verdict != "RETAIN"]) else None)) if len(ver) else None
    pz = cmp.get("paired", pd.DataFrame())
    if top and len(pz):
        z = pz[(pz.method == top)]
        add("5. Was the improvement robust across regions?",
            f"{top}: zone deltas " + ", ".join(f"{t}:{int(zz)} {d:+.4f}" for t, zz, d in zip(z.task, z.zone,
                                                                                         z.delta_vs_ref)),
            "validation/metrics/paired_zone_deltas.csv + regional_dti.csv")
    else:
        add("5. Was the improvement robust across regions?", "NO CANDIDATE to assess", "-")
    if len(P):
        pp = P[P.task == prim].set_index("method")
        ref = cmp["champion_reference"]
        if top in pp.index and ref in pp.index:
            add("6. Did short-fault recall improve?",
                f"{top}: {pp.loc[top, 'short_recall_w']:.3f} vs {ref}: {pp.loc[ref, 'short_recall_w']:.3f}",
                "validation/metrics/method_pooled.csv")
    if not any(o["question"].startswith("6.") for o in out):
        add("6. Did short-fault recall improve?", "NOT EVALUATED", "-")
    sy = cmp.get("systems", pd.DataFrame())
    s02 = _t(m7, "system_table", "failure_analysis/tables")
    if len(sy) and len(s02):
        m = sy[sy.task == prim].merge(s02[s02.task == prim][["zone", "system_id", "class"]], on=["zone", "system_id"])
        g = m[m["class"] == "neither"].groupby("method").recovered.sum().sort_values(ascending=False)
        add("7. Which method recovered faults missed by B0/B1?",
            ", ".join(f"{k}: {int(v)}" for k, v in g.head(6).items()) or "none",
            "validation/metrics/system_hits.csv x failure_analysis/tables/system_table.csv")
    else:
        add("7. Which method recovered faults missed by B0/B1?", "NOT EVALUATED", "-")
    pg = _t(m7, "pooled_groups", "candidate_generation/oracle")
    if len(pg):
        g = pg[(pg.task == prim) & (pg.group.isin(["provenance", "modality"]))].copy()
        g["recall_per_density"] = g.system_recall / g.density_km_per_100km2.clip(lower=1e-6)
        b = g.sort_values("recall_per_density", ascending=False).head(3)
        add("8. Which candidate generator has the best recall-density trade-off?",
            "; ".join(f"{r.group}:{r.value} recall {r.system_recall:.3f} at {r.density_km_per_100km2:.1f} km/100km2"
                      for r in b.itertuples()), "candidate_generation/oracle/pooled_groups.csv")
    else:
        add("8. Which candidate generator has the best recall-density trade-off?", "NOT EVALUATED", "-")
    if len(pr):
        add("9. Which model should be submitted next?",
            f"{pr.sort_values('dti_pooled', ascending=False).method.iloc[0]} (PROMOTE) as probe C; keep the champion "
            "probes A / B (ring on / off)", "reports/decision_verdicts.csv")
    else:
        add("9. Which model should be submitted next?",
            f"no new method passed its gate: submit the champion ({cmp.get('champion_reference')}) with the ring "
            "on / off pair (probes A / B) and the conservative fallback as probe C", "reports/decision_verdicts.csv")
    add("10. What is the next highest-value experiment?", next_experiment(m7, po, dd, mcd, pr), "rule-based")
    return out


def next_experiment(m7, po, dd, mcd, pr) -> str:
    md = _t(m7, "miss_summary", "candidate_generation/oracle")
    if len(po) and po.decision.iloc[0] == "NO_GO":
        hint = ""
        if len(md):
            top = md.sort_values("n_systems", ascending=False).iloc[0]
            hint = f" Most missed systems: '{top.hint}' ({int(top.n_systems)} systems)."
        return ("Candidate generation is the bottleneck: widen it before any ranker (more scales / modalities, lower "
                "gates for the modalities named in miss_diagnostics)." + hint)
    if len(dd) and dd.decision.iloc[0] in ("EXPAND", "EXPAND_TARGETED"):
        return "Expand the 1 m DEM processing (coverage-weighted), then re-run sections 07 and 10-13."
    if len(dd) and dd.decision.iloc[0] == "INCONCLUSIVE":
        return "Extend the DEM pilot to tiles that contain withheld systems missed by both models (section 05 quotas)."
    if len(mcd) and mcd.decision.iloc[0] == "TRAIN_LEARNED":
        return "Train the learned completer on damaged maps (systems split by fold before corruption)."
    if len(pr):
        return "Upload the promoted method as probe C, record it in leaderboard_log.csv, then iterate on its weakest zone."
    return ("Ring on / off leaderboard pair (probes A / B) to test the known-fault proximity bias, then revisit the "
            "candidate pool with the modalities flagged in miss_diagnostics.")


def report(m7, cmp: dict, ver: pd.DataFrame, ans: list[dict]) -> str:
    cfg = m7.cfg
    prim = cfg["evaluation"]["primary_task"]
    L = [f"# GEMS M7 discovery-first research report", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} from the saved tables of `{m7.ws.root}`. Every number is a "
         "nested-cross-fitted offline value (pseudo-discovery proxy); **none is a leaderboard score**. The zones were "
         "inspected in earlier sessions: these are not untouched test results.", "",
         f"Primary task: `{prim}` (withheld systems, known faults masked pixel-exactly, masked predictions ignored). "
         f"Budget domain: `{cfg['evaluation']['budget_domain']}` (known pixels, template nodata and the optional ring "
         "never consume the area budget).", ""]
    b = m7.ws.load_immutable_table("baseline_table", "recovered_baselines")
    if len(b):
        L += ["## Baselines (section 01, immutable record)", "", _md(b[["task", "method", "dti_crossfit", "fold_min",
                                                                         "optimism", "area_frac_eligible_mean"]]), ""]
    rec = m7.ws.load_table("recorded_vs_recomputed", "recovered_baselines")
    if len(rec):
        L += ["Recorded vs recomputed (reported, never reconciled):", "", _md(rec), ""]
    for title, (name, sub) in {"Failure analysis (section 02)": ("summary", "failure_analysis/tables"),
                               "Coverage gate (section 04)": ("gate_decision", "candidate_generation/oracle"),
                               "DEM increment (section 07)": ("decision", "dem/candidates"),
                               "Map completion (section 09)": ("decision", "map_completion/metrics"),
                               "Ranking (section 11)": ("ranking_summary", "candidate_ranking/models"),
                               "Rasterisation (section 12)": ("rasterization_summary", "candidate_ranking/predictions")}.items():
        t = m7.ws.load_table(name, sub)
        if len(t):
            L += [f"## {title}", "", _md(t.head(40)), ""]
    P = cmp.get("pooled", pd.DataFrame())
    if len(P):
        cols = [c for c in ("task", "method", "dti_pooled", "worst_zone_dti", "dti_retained_fp", "system_recall",
                            "short_recall_w", "recall_w", "area_frac_eligible_mean") if c in P]
        L += ["## Model comparison (section 13)", "", _md(P[cols]), ""]
    if len(ver):
        cols = [c for c in ("method", "verdict", "reason", "dti_pooled", "pooled_gain", "zone_win_share",
                            "worst_zone_delta", "geo_drop", "boot_p_positive", "short_recall_drop") if c in ver]
        L += ["## Verdicts (section 14)", "", _md(ver[cols]), ""]
    L += ["## Answers", ""] + [f"**{a['question']}** {a['answer']}  \n_evidence: {a['evidence']}_" for a in ans] + [""]
    L += ["## Caveats", "",
          "* Pseudo-discovery withholds whole systems: new faults next to known traces (corrections) are "
          "under-represented, so a known-fault ring can look better here than on the hidden test.",
          "* The hidden new-fault density is unknown: the submission area fraction is the largest unverified choice.",
          "* Oracle numbers read the truth: upper bounds conditional on the candidate pool, never scores.",
          "* DEM results cover only the pilot tiles and are scored on a restricted protocol (never pooled with "
          "full-zone numbers).", ""]
    return "\n".join(L)


def _md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False, floatfmt=".4f")
    except ImportError:
        return "```\n" + df.to_string(index=False) + "\n```"


def run(m7, cmp: dict) -> dict:
    ver = verdicts(m7, cmp)
    ans = answers(m7, cmp, ver)
    rep = report(m7, cmp, ver, ans)
    return {"verdicts": ver, "answers": pd.DataFrame(ans), "report": rep,
            "decision": json.loads(json.dumps({"verdicts": ver.to_dict("records") if len(ver) else [],
                                               "answers": ans}, default=str))}
