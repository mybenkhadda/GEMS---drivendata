"""Experiments 3 + 5: output representation of the EXACT frozen W2 score, and one honest baseline table.

Scores (per scenario unit, honest maps only): ``W2`` = the frozen submitted W2 score (``geomoe.run.frozen_smap``:
factory evaluation construction); ``A1`` = label-free lineament score (``physics.lineament_score``, ranked inside the
unit); ``A2`` / ``A3`` = B0 / B1 held-out ranks; ``A0`` = an organiser reference GeoTIFF only when its path is
configured (trained on every label: reported, never treated as held-out).

Representations (predeclared, config ``benchmark``): gems3 syntax ones (skeleton_1px, crest_1px, skeleton_3px,
halo_soft, confidence_width; added to the engine at attach and exportable through the factory writer) and two
diagnostic-only ops defined here: ``continuous_ramp`` (a linear confidence ramp whose probability MASS equals frac of
the eligible domain) and ``evidence_weighted_skeleton`` (skeleton at frac, value 0.5 + 0.5 x label-free lineament
rank: confidence from evidence the W2 labels did not define). Neither diagnostic op is exported.

Fairness: every row records realised area AND probability mass; ``area_response`` interpolates DTI at common
realised budgets so binary and soft outputs are compared at equal mass, never at equal nominal frac. Model selection
uses the GeoMoE fixed-policy rule (development units only); the reference is the frozen submitted cell (A4).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..geomoe import evaluate as EV
from ..geomoe.decide import model_checks

REF, A5 = "A4_submitted_W2", "A5_W2_representation"
DIAGNOSTIC_REPS = ("continuous_ramp", "evidence_weighted_skeleton")


# =============================================================================== pure: representations
def ramp_map(score: np.ndarray, eligible: np.ndarray, frac: float, iters: int = 40) -> np.ndarray:
    """Continuous q = clip((s - t) / (1 - t), 0, 1) on eligible pixels, t chosen so that sum(q) = frac x n_eligible
    (equal probability MASS to a binary map of area frac)."""
    s = np.where(eligible, np.nan_to_num(np.asarray(score, np.float64), nan=0.0), 0.0)
    v = s[eligible]
    target = float(frac) * max(int(eligible.sum()), 1)
    if not v.size:
        return np.zeros(s.shape, np.float32)
    lo, hi = float(v.min()) - 1.0, float(v.max())
    for _ in range(iters):
        t = 0.5 * (lo + hi)
        m = np.clip((v - t) / max(1.0 - t, 1e-9), 0, 1).sum()
        lo, hi = (t, hi) if m > target else (lo, t)
    t = 0.5 * (lo + hi)
    q = np.zeros(s.shape, np.float32)
    q[eligible] = np.clip((v - t) / max(1.0 - t, 1e-9), 0, 1)
    return q


def evidence_weighted(q_bin: np.ndarray, evidence_rank: np.ndarray) -> np.ndarray:
    q = np.asarray(q_bin) > 0
    e = np.clip(np.nan_to_num(np.asarray(evidence_rank, np.float32), nan=0.0), 0, 1)
    return np.where(q, 0.5 + 0.5 * e, 0.0).astype(np.float32)


def interpolate_budget(g: pd.DataFrame, budgets, x: str = "mass_frac") -> list[dict]:
    """Pooled DTI interpolated at common realised budgets (NaN outside the measured range: never extrapolated)."""
    g = g.sort_values(x)
    xs, ys = g[x].to_numpy(float), g.dti.to_numpy(float)
    out = []
    for b in budgets:
        ok = len(xs) >= 2 and xs.min() <= b <= xs.max()
        out.append({"budget": float(b), "budget_axis": x, "dti_interp": float(np.interp(b, xs, ys)) if ok else np.nan})
    return out


# =============================================================================== Colab
def install_representations(R) -> dict:
    """Add the configured gems3-syntax representations to the engine's shared config (in memory, recorded)."""
    reps = R.sh["baseline"]["representations"]
    rec = {}
    for k, v in R.rc["benchmark"]["representations"].items():
        if k in reps and reps[k] != v:
            rec[k] = "EXISTING_DIFFERS (kept the engine's own definition)"
        elif k in reps:
            rec[k] = "EXISTING"
        else:
            reps[k] = v
            rec[k] = "ADDED"
    return rec


def apply_rep(R, u, smap: np.ndarray, rep: str, frac: float, evidence: np.ndarray | None = None) -> np.ndarray:
    if rep == "continuous_ramp":
        return ramp_map(smap, R.gm.elig(u) & u.ev, frac)
    if rep == "evidence_weighted_skeleton":
        q = EV.policy_q(R.gm, u, smap, "skeleton_1px", frac)
        return evidence_weighted(q, evidence if evidence is not None else np.zeros(q.shape))
    return EV.policy_q(R.gm, u, smap, rep, frac)


def unit_scores(R, su, L: dict) -> dict:
    from ..gems3 import representation as REP
    from .physics import rank_valid

    u = su.unit
    out = {"W2": R.gm.frozen_smap(su)}
    b0, b1 = R.bst.cnn(su.base)
    out["A2_B0"], out["A3_B1"] = REP.rank01(b0, u.ev), REP.rank01(b1, u.ev)
    if "lineament" in L:
        out["A1_label_free"] = np.nan_to_num(rank_valid(L["lineament"][u.sl], u.ev), nan=0.0)
    ref = R.reference_crop(u)
    if ref is not None:
        out["A0_reference"] = ref
    return {k: v for k, v in out.items() if v is not None}


def row(R, su, q, model, rep, frac, extra=None) -> dict:
    from ..geomoe.scenarios import system_recovery

    r = R.gm.score(su.unit, q)
    hs = system_recovery(su.unit, q, su.sysmap, R.gm.sys_support, R.gm.sys_frac)
    rec = sorted(int(x) for x in hs.system_id[hs.recovered]) if len(hs) else []
    el = max(int(r["eligible_px"]), 1)
    return {**r, "scenario": su.scenario, "unit": su.key, "model": model, "rep": rep, "frac": float(frac),
            "mass_frac": float(r["pred_mass"]) / el, "n_systems": int(len(hs)), "n_recovered": len(rec),
            "systems": ";".join(map(str, rec)), "role": "heldout", **(extra or {})}


def grid(R, L: dict) -> pd.DataFrame:
    bc = R.rc["benchmark"]
    reps = list(bc["representations"]) + list(bc["diagnostic_representations"])
    exp = set(bc.get("exportable_representations") or ())
    fz = R.gm.frozen_policy()["cell"]
    rows = []
    for su in R.units:
        S = unit_scores(R, su, L)
        u = su.unit
        ev_rank = S.get("A1_label_free")
        if "W2" in S:
            q = EV.policy_q(R.gm, u, S["W2"], str(fz["rep"]), float(fz["frac"]))
            rows.append(row(R, su, q, REF, str(fz["rep"]), float(fz["frac"]), {"exportable": True}))
            for rp in reps:
                for f in bc["area_fracs"]:
                    q = apply_rep(R, u, S["W2"], rp, f, ev_rank)
                    rows.append(row(R, su, q, "W2", rp, f, {"exportable": rp in exp}))
        for m in ("A1_label_free", "A2_B0", "A3_B1", "A0_reference"):
            if m in S:
                for rp in bc["baseline_reps"]:
                    for f in bc["area_fracs"]:
                        rows.append(row(R, su, EV.policy_q(R.gm, u, S[m], rp, f), m, rp, f, {"exportable": False}))
        R.gm.release(su.base)
    return pd.DataFrame(rows)


def summaries(R, G: pd.DataFrame) -> dict:
    """Frozen A4, fixed-policy A5 (rep x frac), per-representation fixed policy (frac only), baselines."""
    dev, rule = EV.dev_units(R.units, float(R.cfg["selection"]["policy_max_overlap"]))
    Sf, Pf = EV.frozen_rows(G, REF)
    W = G[G.model == "W2"].copy()
    parts_S, parts_P, deploy = [Sf], [Pf], {}
    S5, P5, d5 = EV.fixed_policy(W.assign(model=A5), dev)
    parts_S.append(S5)
    parts_P.append(P5)
    deploy.update(d5)
    for rp, g in W.groupby("rep"):
        s, p, d = EV.fixed_policy(g.assign(model=f"W2|{rp}"), dev)
        parts_S.append(s.assign(diagnostic=True))
        parts_P.append(p)
        deploy.update(d)
    for m in ("A1_label_free", "A2_B0", "A3_B1"):
        g = G[G.model == m]
        if len(g):
            s, p, d = EV.fixed_policy(g, dev)
            parts_S.append(s)
            parts_P.append(p)
            deploy.update(d)
    S = pd.concat([x for x in parts_S if len(x)], ignore_index=True)
    P = pd.concat([x for x in parts_P if len(x)], ignore_index=True)
    if len(P):
        P["dev_rule"] = P.unit.map(rule)
    A0 = G[G.model == "A0_reference"]
    if len(A0):
        s, _, _ = EV.fixed_policy(A0, dev)
        S = pd.concat([S, s.assign(caveat="trained on every mapped fault: NOT a held-out estimate")],
                      ignore_index=True)
    return {"S": S, "P": P, "deploy": deploy, "dev_rule": rule}


def curves(G: pd.DataFrame, budgets) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = G.groupby(["scenario", "model", "rep", "frac"]).agg(
        tp_w=("tp_w", "sum"), fp_w=("fp_w", "sum"), fn_w=("fn_w", "sum"), pred_mass=("pred_mass", "sum"),
        area_px=("area_px", "sum"), eligible_px=("eligible_px", "sum")).reset_index()
    g["dti"] = [EV.dti(a, b, c) for a, b, c in zip(g.tp_w, g.fp_w, g.fn_w)]
    g["mass_frac"] = g.pred_mass / g.eligible_px.clip(lower=1)
    g["area_frac"] = g.area_px / g.eligible_px.clip(lower=1)
    rows = []
    for (sc, m, rp), h in g.groupby(["scenario", "model", "rep"]):
        for ax in ("mass_frac", "area_frac"):
            rows += [{"scenario": sc, "model": m, "rep": rp, **x} for x in interpolate_budget(h, budgets, ax)]
    return g, pd.DataFrame(rows)


def chosen(R, P: pd.DataFrame, L: dict) -> dict:
    """Held-out chosen cells -> spatial blocks, systems, distance bands (for paired comparisons)."""
    from scipy import ndimage

    from ..geomoe.scenarios import banded_components, system_recovery
    from ..m7.protocol import block_components

    keep = [REF, A5, "A1_label_free", "A2_B0", "A3_B1"]
    blocks, systems, bands = [], [], []
    for su in R.units:
        S = unit_scores(R, su, L)
        u = su.unit
        dist = (ndimage.distance_transform_edt(~su.known) * 0.1).astype(np.float32) if su.known.any() else \
            np.full(u.ev.shape, 99.0, np.float32)
        for m in keep:
            pz = P[(P.unit == su.key) & (P.model == m)]
            src = S.get("W2" if m in (REF, A5) else m)
            if not len(pz) or src is None:
                continue
            r = pz.iloc[0]
            q = apply_rep(R, u, src, str(r.rep), float(r.frac), S.get("A1_label_free"))
            blocks.append(block_components(u, q, int(R.cfg["evaluation"]["block_px"])).assign(method=m,
                                                                                            scenario=su.scenario))
            systems.append(system_recovery(u, q, su.sysmap, R.gm.sys_support, R.gm.sys_frac).assign(
                model=m, unit=su.key, scenario=su.scenario))
            bands.append(banded_components(u, q, dist, R.cfg["distance_bands_km"]).assign(model=m,
                                                                                          scenario=su.scenario))
        R.gm.release(su.base)
    cat = (lambda x: pd.concat(x, ignore_index=True) if x else pd.DataFrame())
    return {"blocks": cat(blocks), "systems": cat(systems), "bands": cat(bands)}


def complementarity(SY: pd.DataFrame, ref: str = REF) -> pd.DataFrame:
    """Per scenario: systems recovered by each source and NOT by the frozen W2 (and the reverse)."""
    if not len(SY):
        return SY
    rows = []
    for sc, g in SY.groupby("scenario"):
        base = set(zip(g[(g.model == ref) & g.recovered].unit, g[(g.model == ref) & g.recovered].system_id))
        for m, h in g.groupby("model"):
            s = set(zip(h[h.recovered].unit, h[h.recovered].system_id))
            rows.append({"scenario": sc, "model": m, "recovered": len(s), "recovered_not_by_w2": len(s - base),
                         "w2_only": len(base - s), "n_systems": int(h[["unit", "system_id"]].drop_duplicates()
                                                                   .shape[0])})
    return pd.DataFrame(rows)


def representation_decision(S: pd.DataFrame, PC: pd.DataFrame, sel: dict, deploy: dict,
                            exportable=("skeleton_1px", "crest_1px")) -> dict:
    """Is a representation change alone a credible improvement over the frozen submitted W2? (fixed policy, every
    scenario measured, paired bootstrap)."""
    if not len(S) or A5 not in set(S.model) or REF not in set(S.model):
        return {"status": "NOT_EVALUATED", "credible": None}
    c = model_checks(S, PC, A5, REF, ("V1", "V2", "V3"), sel)
    ok_p = [sc for sc, v in c["per_scenario"].items() if (v.get("p_delta_positive") or 0) >= float(sel["min_p_positive"])
            and (v.get("delta_dti") or 0) >= float(sel["min_gain_vs_w2"])]
    cred = bool(c["beats_ref"] and len(ok_p) >= int(sel["min_scenarios_improved"]))
    d = deploy.get(A5) or {}
    rep = d.get("rep")
    return {"status": "CREDIBLE" if cred else "NOT_CREDIBLE", "credible": cred, "checks": c,
            "scenarios_with_paired_support": ok_p, "deployable_cell": d,
            "exportable": rep is not None and rep in set(exportable),
            "rule": f"A5 beats A4 by >= {sel['min_gain_vs_w2']} on >= {sel['min_scenarios_improved']} scenarios "
                    f"with P(delta > 0) >= {sel['min_p_positive']}, no scenario drop > {sel['max_scenario_drop']}"}


def reviewed_agreement(R, P: pd.DataFrame, L: dict) -> pd.DataFrame:
    """A6: agreement of each model's chosen V3 held-out map with REVIEWED atlas candidates (fault vs hard-negative
    interpretations; evidence-only reviews flagged). Empty when no review exists (NOT_AVAILABLE)."""
    from scipy import ndimage

    from .atlas import latest_reviews

    rv = latest_reviews(R.load_reviews())
    rv = rv[rv.label_class.isin(["INDEPENDENT_REVIEWED", "DIAGNOSTIC_REVIEWED", "HARD_NEGATIVE"])] if len(rv) else rv
    if not len(rv):
        return pd.DataFrame()
    rows = []
    for su in [s for s in R.units if s.scenario == "V3"]:
        S = unit_scores(R, su, L)
        u = su.unit
        for m in (REF, A5, "A1_label_free"):
            pz = P[(P.unit == su.key) & (P.model == m)]
            src = S.get("W2" if m in (REF, A5) else m)
            if not len(pz) or src is None:
                continue
            q = apply_rep(R, u, src, str(pz.iloc[0].rep), float(pz.iloc[0].frac), S.get("A1_label_free")) > 0
            d = ndimage.distance_transform_edt(~q) if q.any() else np.full(q.shape, np.inf)
            for r in rv.itertuples():
                g = R.candidate_pixels(r.candidate_id)
                if g is None:
                    continue
                rr, cc = g[0] - u.sl[0].start, g[1] - u.sl[1].start
                ok = (rr >= 0) & (rr < q.shape[0]) & (cc >= 0) & (cc < q.shape[1])
                if not ok.any() or not u.ev[rr[ok], cc[ok]].any():
                    continue
                rows.append({"model": m, "unit": su.key, "candidate_id": r.candidate_id, "label_class": r.label_class,
                             "interpretation": r.interpretation, "independent": r.independent_of_original_fault_map,
                             "share_within_300m": float((d[rr[ok], cc[ok]] <= 3).mean())})
        R.gm.release(su.base)
    return pd.DataFrame(rows)


def run(R, L: dict) -> dict:
    added = install_representations(R)
    G = grid(R, L)
    G = EV.budget_rows(G, float(R.fg.cfg["submission"]["budget_tolerance"]))
    R.ws.save_table(G[G.model.isin([REF, "W2"])], "benchmark/representation_grid.csv")
    R.ws.save_table(G, "benchmark/full_grid.csv")
    sm = summaries(R, G)
    S, P = sm["S"], sm["P"]
    R.ws.save_table(S, "benchmark/baseline_metrics.csv")
    R.ws.save_table(S, "benchmark/model_comparison.csv")
    R.ws.save_table(P, "benchmark/per_zone_metrics.csv")
    cv, ar = curves(G, R.rc["benchmark"]["realized_budgets"])
    R.ws.save_table(cv, "benchmark/tp_fp_fn_curves.csv")
    R.ws.save_table(ar, "benchmark/area_response.csv")
    ch = chosen(R, P, L)
    pairs = [(m, REF) for m in (A5, "A1_label_free", "A2_B0", "A3_B1")]
    ec = R.cfg["evaluation"]
    PC = EV.paired(ch["blocks"], pairs, int(ec["n_boot"]), int(R.rc["atlas"]["seed"])) if len(ch["blocks"]) else \
        pd.DataFrame()
    R.ws.save_table(PC, "benchmark/paired_comparison.csv")
    R.ws.save_table(ch["systems"], "benchmark/fault_system_recovery.csv")
    R.ws.save_table(EV.band_table(ch["bands"]) if len(ch["bands"]) else pd.DataFrame(),
                    "benchmark/near_known_metrics.csv")
    R.ws.save_table(complementarity(ch["systems"]), "benchmark/source_complementarity.csv")
    AG = reviewed_agreement(R, P, L)
    R.ws.save_table(AG, "benchmark/reviewed_agreement.csv")
    dec = representation_decision(S, PC, R.rc["challenger"]["selection"], sm["deploy"],
                                  R.rc["benchmark"].get("exportable_representations") or ())
    dec.update(representations=added, frozen_policy=R.gm.frozen_policy(), dev_rules=dict(pd.Series(
        sm["dev_rule"]).value_counts()), budget_check=bool(G.budget_ok.all()), A6=(
        "reviewed agreement available" if len(AG) else "NOT_AVAILABLE (no reviewed candidate)"))
    R.ws.save_json(json.loads(json.dumps(dec, default=str)), "benchmark/selected_policy.json")
    R.ws.save_json(sm["deploy"], "benchmark/deployable_policies.json")
    return {"decision": dec["status"], "note": f"representation change {dec['status']}; A5 cell "
                                               f"{dec.get('deployable_cell')}"}
