"""Section 8: choose ONE challenger from measured evidence (no architecture chosen in advance).

``decide`` reads the persisted benchmark / review / terrain / physics outputs and returns the first case whose
evidence condition holds, in the brief's order:
    CASE_1_ALLOCATION   a representation / allocation change of the frozen W2 score alone is CREDIBLE
                        (benchmark/selected_policy.json) -> no new model; exported through the factory writer
    CASE_2_VERIFIER     enough REVIEWED fault and hard-negative candidates in TRAIN regions -> small regularised
                        logistic fault-vs-nonfault verifier on candidate descriptors, evaluated on EVAL regions
    CASE_3_TERRAIN      the DEM pilot passed its stop rule -> the same verifier with terrain descriptors
    CASE_4_REPRESENTATION every source localises poorly (localization floor) and no case above holds ->
                        self-supervised geophysical representation learning is RECOMMENDED; not implemented here
                        (needs a separately authorised session: no neural training in this iteration)
    NO_CHALLENGER       none of the above: W2 stays the reference
A condition that could not be evaluated is NOT_EVALUATED (never a negative). Verifier outputs are candidate rankings:
they are not rasterised into a submission in this iteration (stated in selected_challenger.json).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

CASES = ("CASE_1_ALLOCATION", "CASE_2_VERIFIER", "CASE_3_TERRAIN", "CASE_4_REPRESENTATION", "NO_CHALLENGER")
FAULT = ("LIKELY_FAULT", "POSSIBLE_FAULT")


def review_counts(latest: pd.DataFrame, hard_negative) -> dict:
    if latest is None or not len(latest):
        return {"train_pos": 0, "train_neg": 0, "eval_pos": 0, "eval_neg": 0, "independent_share": None}
    pos = latest.interpretation.isin(FAULT)
    neg = latest.label_class.eq("HARD_NEGATIVE") & latest.interpretation.isin(hard_negative)
    tr = latest.review_split.eq("TRAIN")
    return {"train_pos": int((pos & tr).sum()), "train_neg": int((neg & tr).sum()),
            "eval_pos": int((pos & ~tr).sum()), "eval_neg": int((neg & ~tr).sum()),
            "independent_share": float(latest.independent_of_original_fault_map.astype(str).eq("True").mean())}


def decide(rep: dict | None, reviews: dict, terrain: dict | None, loc: pd.DataFrame | None, cc: dict) -> dict:
    """Pure case selection (first matching case, every condition recorded with its evidence)."""
    cond = {}
    cond["CASE_1_ALLOCATION"] = ("NOT_EVALUATED" if not rep or rep.get("credible") is None else
                                 bool(rep["credible"] and rep.get("exportable")))
    cond["CASE_2_VERIFIER"] = bool(reviews["train_pos"] >= int(cc["min_reviewed_positive"]) and
                                   reviews["train_neg"] >= int(cc["min_reviewed_negative"]))
    cond["CASE_3_TERRAIN"] = ("NOT_EVALUATED" if not terrain or terrain.get("decision") not in ("STOP", "CONTINUE")
                              else terrain["decision"] == "CONTINUE")
    if loc is None or not len(loc) or "truth_within_0.3km" not in loc:
        cond["CASE_4_REPRESENTATION"] = "NOT_EVALUATED"
        best = None
    else:
        best = float(loc.groupby("source")["truth_within_0.3km"].mean().max())
        cond["CASE_4_REPRESENTATION"] = bool(best < float(cc["localization_floor"]))
    pick = next((c for c in CASES[:4] if cond.get(c) is True), "NO_CHALLENGER")
    if pick == "CASE_2_VERIFIER" and cond["CASE_3_TERRAIN"] is True:
        pick = "CASE_3_TERRAIN"                         # same verifier, with the independent terrain evidence
    return {"selected_case": pick, "conditions": cond, "reviews": reviews, "best_truth_within_300m": best,
            "implemented": pick in ("CASE_1_ALLOCATION", "CASE_2_VERIFIER", "CASE_3_TERRAIN", "NO_CHALLENGER"),
            "note": {"CASE_1_ALLOCATION": "export the selected representation of the frozen W2 score",
                     "CASE_2_VERIFIER": "candidate verifier (rankings only in iteration 1)",
                     "CASE_3_TERRAIN": "candidate verifier with terrain descriptors (rankings only)",
                     "CASE_4_REPRESENTATION": "recommended; NOT implemented (requires an authorised GPU session)",
                     "NO_CHALLENGER": "W2 stays the reference"}[pick]}


def verifier_table(flags: pd.DataFrame, feats: pd.DataFrame, latest: pd.DataFrame, catalog: pd.DataFrame,
                   hard_negative) -> pd.DataFrame:
    """Candidate descriptors + reviewed target (1 fault, 0 hard negative; everything else excluded, never 0)."""
    T = catalog[["candidate_id", "category", "region_id", "review_split", "distance_to_known_fault_km",
                 "length_km"]].merge(flags, on=["candidate_id", "category"], how="left")
    if len(feats):
        T = T.merge(feats, on="candidate_id", how="left")
    lr = latest.set_index("candidate_id") if len(latest) else pd.DataFrame()
    y = []
    for c in T.candidate_id:
        if not len(lr) or c not in lr.index:
            y.append(np.nan)
            continue
        r = lr.loc[c]
        y.append(1.0 if r.interpretation in FAULT else 0.0 if (r.label_class == "HARD_NEGATIVE" and
                                                                r.interpretation in hard_negative) else np.nan)
    T["target"] = y
    T["target_provenance"] = ["reviewed (atlas)" if np.isfinite(v) else "unreviewed: excluded" for v in y]
    return T


def fit_verifier(T: pd.DataFrame, feature_cols: list[str], seed: int = 2026) -> dict:
    """L2 logistic regression on TRAIN regions, scored on EVAL regions (geographic split predeclared in config)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score

    D = T[np.isfinite(T.target)]
    tr, te = D[D.review_split == "TRAIN"], D[D.review_split == "EVAL"]
    if tr.target.nunique() < 2:
        return {"status": "INSUFFICIENT_TRAIN_CLASSES"}
    X = tr[feature_cols].astype(float)
    mu, sd = X.mean(), X.std().replace(0, 1).fillna(1)
    m = LogisticRegression(C=1.0, max_iter=500, random_state=seed)
    m.fit(((X - mu) / sd).fillna(0).to_numpy(), tr.target.to_numpy())
    out = {"status": "FITTED", "n_train": int(len(tr)), "n_eval": int(len(te)), "features": feature_cols,
           "coef": dict(zip(feature_cols, m.coef_[0].round(4).tolist())), "intercept": float(m.intercept_[0]),
           "standardize": {"mean": mu.to_dict(), "std": sd.to_dict()}}
    if len(te):
        p = m.predict_proba(((te[feature_cols].astype(float) - mu) / sd).fillna(0).to_numpy())[:, 1]
        out["heldout"] = pd.DataFrame({"candidate_id": te.candidate_id, "target": te.target, "p_fault": p})
        if te.target.nunique() == 2:
            out.update(eval_auc=float(roc_auc_score(te.target, p)), eval_ap=float(average_precision_score(te.target,
                                                                                                          p)))
        else:
            out["eval_note"] = "EVAL regions hold one class only: no AUC"
    return out


def run(R) -> dict:
    from .atlas import latest_reviews

    cc = R.rc["challenger"]
    rep = R.ws.read_json("benchmark/selected_policy.json")
    terr = R.ws.read_json("terrain/pilot_decision.json")
    loc = R.ws.load_table("physics/localization_error.csv")
    lat = latest_reviews(R.load_reviews())
    d = decide(rep, review_counts(lat, cc["hard_negative"]), terr, loc, cc)
    out = {"decision": d}
    if d["selected_case"] in ("CASE_2_VERIFIER", "CASE_3_TERRAIN"):
        flags = R.load_parquet("physics/structural_evidence.parquet")
        feats = R.load_parquet("physics/magnetic_gravity_features.parquet")
        T = verifier_table(flags, feats, lat, R.ws.load_table("atlas/atlas_index.csv"), cc["hard_negative"])
        cols = [c for c in T.columns if c not in ("candidate_id", "category", "region_id", "review_split", "target",
                                                  "target_provenance") and pd.api.types.is_numeric_dtype(T[c])]
        if d["selected_case"] == "CASE_2_VERIFIER":
            cols = [c for c in cols if not c.startswith("terrain_")]
        R.save_parquet(T, "challenger/training_features.parquet")
        f = fit_verifier(T, cols, int(R.rc["atlas"]["seed"]))
        if "heldout" in f:
            R.ws.save_table(f.pop("heldout"), "challenger/heldout_predictions/verifier_eval_regions.csv")
        R.ws.save_json(f, "challenger/model_artifacts/verifier.json")
        out["verifier"] = {k: f.get(k) for k in ("status", "n_train", "n_eval", "eval_auc", "eval_ap", "eval_note")}
        R.ws.save_table(pd.DataFrame([{"model": "verifier", **out["verifier"]}]), "challenger/comparison.csv")
        abl = []
        for drop in ("terrain_", "magnetic_", "gravity_"):
            k = [c for c in cols if not c.startswith(drop)]
            if k and k != cols:
                g = fit_verifier(T, k, int(R.rc["atlas"]["seed"]))
                abl.append({"dropped_prefix": drop, **{x: g.get(x) for x in ("status", "eval_auc", "eval_ap")}})
        R.ws.save_table(pd.DataFrame(abl), "challenger/ablation.csv")
    elif d["selected_case"] == "CASE_1_ALLOCATION":
        R.ws.save_table(R.ws.load_table("benchmark/model_comparison.csv"), "challenger/comparison.csv")
    sel = {**d, "export": ("factory writer, zone-stitched frozen W2 score with the selected representation"
                           if d["selected_case"] == "CASE_1_ALLOCATION" else
                           "none in iteration 1 (verifier = candidate ranking; CASE_4 not implemented)"),
           "acceptance": "beats the frozen submitted W2 under the fixed-policy cross-fitted rule, no geographic "
                         "collapse, reproducible zone-stitched inference, technically valid GeoTIFF"}
    R.ws.save_json(json.loads(json.dumps(sel, default=str)), "challenger/selected_challenger.json")
    return {"selected_case": d["selected_case"], "note": f"{d['selected_case']}: {d['note']}"}
