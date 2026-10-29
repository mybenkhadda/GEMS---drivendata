"""Stage 11: which full-region construction keeps the measured advantage? (V3 geographic units, cross-fitted)

Three constructions of the SAME model, each evaluated on held-out geo zone k at the SAME nested (rep, frac) cell
that the zone-stitched evaluation chose for zone k, so only the construction differs:

* ``zone_stitched``   zone k <- model fitted on the other zones, every layer ranked inside its own geo zone (the
                      validated construction; re-derived here with ``DiscoveryGeoMoE.fit_score`` and checked against
                      the stage 08-10 scores: Spearman 1 expected).
* ``fold_average``    zone k <- mean of one model per other zone j (each normalised on zone j only): what averaging
                      the per-fold models would deploy.
* ``footprint_rank``  zone k <- model fitted on the other zones, but every rank-domain layer (CNN, W2, potential
                      fields, G02) ranked over the WHOLE geographic footprint of the stitched held-out maps instead of
                      inside each zone: the inference transformation of a full-region refit. The held-out maps stay
                      honest (no model saw zone k labels); only the ranking domain changes.

No construction uses zone-k labels; a construction that loses more than ``dti_tol`` or whose candidate scores
disagree with the zone-stitched ones (Spearman < ``parity_min_spearman``) is reported as NOT EQUIVALENT and is never
used for export.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import evaluate as EV
from . import experts as EX

FOOTPRINT_LAYERS = ("cnn_b0", "cnn_b1", "w2", "mag_hgrad", "grav_hgrad", "g02c", "g02a", "g01_persistence")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    return float(spearmanr(a[m], b[m]).correlation)


def deploy_model(G) -> str | None:
    """Selected model, else the fused model with the best pooled V3 DTI (diagnostic choice; nothing is promoted)."""
    sel = G.ws.read_json("gating/selection.json") or {}
    if sel.get("selected") not in (None, "RETAIN_W2"):
        return sel["selected"]
    S = G.ws.load_table("gating/model_summary.csv")
    S = S[(S.scenario == "V3") & S.model.isin(["B", "C_all", "D"])] if len(S) else S
    return str(S.sort_values("dti", ascending=False).model.iloc[0]) if len(S) else None


def footprint_table(G, units: list) -> pd.DataFrame:
    """V3 expert table with every FOOTPRINT_LAYERS mean recomputed from footprint-ranked layers."""
    from ..p6 import stack as ST
    from .candidates import layer_means

    X = G.expert_table()
    X3 = X[X.scenario == "V3"].copy()
    shape = G.ctx.data.shape
    foot = np.zeros(shape, bool)
    vals = {k: [] for k in FOOTPRINT_LAYERS}
    for su in units:
        b = su.base
        foot[b.sl] |= b.ev
        raw = G.bst.raw(b)
        for k in FOOTPRINT_LAYERS:
            a = None
            if k == "w2":
                try:
                    a = np.asarray(G.bst.w2_map(b), np.float32)
                except (FileNotFoundError, ValueError):
                    a = None
            elif k in raw:
                a = np.asarray(raw[k], np.float32)
            if a is not None:
                vals[k].append((b.sl, b.ev, a[b.ev]))
        G.release(b)
    for k, parts in vals.items():
        if not parts:
            continue
        full = np.full(shape, np.nan, np.float32)
        for sl, ev, v in parts:
            view = full[sl]
            view[ev] = v
        rk = ST.rank_in(full, foot & np.isfinite(full))
        for su in units:
            lab = G.lab(su)
            M = layer_means(lab, {k: np.where(su.base.ev, rk[su.base.sl], np.nan)})
            idx = X3.unit == su.key
            m = X3.loc[idx, ["label"]].merge(M, on="label", how="left")
            X3.loc[idx, f"mean_{k}"] = m[f"mean_{k}"].to_numpy()
            X3.loc[idx, f"fin_{k}"] = m[f"fin_{k}"].to_numpy()
        del full, rk
    X3 = X3.drop(columns=[c for c in X3 if c.startswith(("raw_", "avail_", "members_"))])
    return pd.concat([X3, EX.expert_raw(X3, G.cfg["experts"])], axis=1)


def run(G) -> tuple[str, dict]:
    dc = G.cfg["deployment"]
    m = deploy_model(G)
    if m is None:
        return "SKIPPED", {"note": "no fused model was evaluated on V3"}
    units = [su for su in G.units if su.scenario == "V3"]
    P = G.ws.load_table("validation/per_zone_metrics.csv")
    P = P[(P.scenario == "V3") & (P.model == m)]
    cells = {int(r.zone): (str(r.rep), float(r.frac)) for r in P.itertuples()}
    stored = pd.read_parquet(G.ws.path(f"gating/scores_{m}.parquet"))
    Xf = footprint_table(G, units)
    rows, par = [], []
    for su in units:
        if su.zone not in cells:
            continue
        rep, frac = cells[su.zone]
        lab = G.lab(su)
        v = G.views(su)
        T = v["T"][v["held"]]
        a = G.fit_score(m, v)
        st = stored[stored.unit == su.key].set_index("label").score.reindex(T.label.to_numpy()).to_numpy(float)
        others = sorted(int(z) for z in v["train_zones"])
        fold = [G.fit_score(m, G.views(su, only_zone=j)) for j in others]
        b = np.nanmean(np.stack(fold), 0) if fold else np.full(len(T), np.nan)
        vf = G.views(su, X=Xf)
        c = G.fit_score(m, vf)
        c = pd.Series(c, index=vf["T"].label.to_numpy()[vf["held"]]).reindex(T.label.to_numpy()).to_numpy(float)
        for name, sc in (("zone_stitched", a), ("fold_average", b), ("footprint_rank", c)):
            sm = EV.score_map(lab, T.label.to_numpy(), sc, T.n_px.to_numpy())
            q = EV.policy_q(G, su.unit, sm, rep, frac) if np.isfinite(sm).any() else np.zeros(lab.shape, np.float32)
            r = G.score(su.unit, q)
            rows.append({"construction": name, "model": m, "zone": su.zone, "rep": rep, "frac": frac,
                         **{k: r[k] for k in ("tp_w", "fp_w", "fn_w", "n_truth", "dti", "area_frac_eligible")}})
        par += [{"zone": su.zone, "check": "zone_stitched re-derivation == stage scores", "spearman": spearman(a, st)},
                {"zone": su.zone, "check": "fold_average vs zone_stitched", "spearman": spearman(b, a)},
                {"zone": su.zone, "check": "footprint_rank vs zone_stitched", "spearman": spearman(c, a)}]
        su.unit.release()
    R = pd.DataFrame(rows)
    Q = pd.DataFrame(par)
    pooled = R.groupby("construction")[["tp_w", "fp_w", "fn_w"]].sum()
    pooled["dti"] = [EV.dti(x.tp_w, x.fp_w, x.fn_w) for x in pooled.itertuples()]
    ref = float(pooled.loc["zone_stitched", "dti"]) if "zone_stitched" in pooled.index else np.nan
    tol, smin = float(dc["dti_tol"]), float(dc["parity_min_spearman"])
    verdict = {}
    for name in pooled.index:
        sp = Q[Q.check == f"{name} vs zone_stitched"].spearman if name != "zone_stitched" else \
            Q[Q.check.str.startswith("zone_stitched")].spearman
        eq = bool(abs(float(pooled.loc[name, "dti"]) - ref) <= tol and (sp.min() if len(sp) else np.nan) >= smin)
        verdict[name] = {"pooled_dti": float(pooled.loc[name, "dti"]), "delta_vs_zone_stitched": float(
            pooled.loc[name, "dti"]) - ref, "min_spearman": float(sp.min()) if len(sp) else None,
            "equivalent": eq if name != "zone_stitched" else bool((sp.min() if len(sp) else 0) >= smin)}
    G.ws.save_table(R.assign(pooled_dti=R.construction.map(lambda n: verdict[n]["pooled_dti"])),
                    "gating/deployment_comparison.csv")
    G.ws.save_table(Q, "gating/deployment_parity.csv")
    zs_ok = verdict.get("zone_stitched", {}).get("equivalent", False)
    dec = {"model": m, "constructions": verdict, "export_construction": dc["export_construction"],
           "zone_stitched_self_consistent": zs_ok, "cells": {str(k): v for k, v in cells.items()},
           "rule": f"equivalent = |delta DTI| <= {tol} and Spearman >= {smin}; export uses the zone-stitched "
                   "construction only, and only when it is self-consistent"}
    G.ws.save_json(json.loads(json.dumps(dec, default=str)), "gating/deployment_decision.json")
    return ("COMPLETE" if zs_ok else "FAILED"), {"note": f"{m}: zone_stitched self-consistent {zs_ok}; "
                                                         f"{ {k: round(v['pooled_dti'], 4) for k, v in verdict.items()} }"}
