"""Stage G/H: cross-fitting of the rankers and the strict-nesting audit.

Cross-fitting (every learned model): for a held-out scenario unit (scenario s, zone j) the model is fitted on the
OTHER zones of scenario s only, minus every candidate whose centroid lies in zone j dilated by the training buffer
(``geomoe.gating.training_rows``), with ``assert_disjoint``; lambda is chosen on inner training zones; the held-out
unit is scored once. That makes the held-out TRUTH unseen by the fit.

It does NOT make the result strictly nested, and this module says so: the W2 / CNN values and the permitted-known
context of the training rows (zone k) come from models / masks built with zone j's withheld systems as ordinary
labels (``p6.stack.permitted_known``: labels & ~withheld(k)). A learned model therefore sees features shaped by
zone-j truth -> every learned result is ``SCREENING_ONLY``. Strict nesting needs W2 / CNN refits per (j, k) pair and
is ``STRICT_NESTING_BLOCKED`` here (``retraining_plan``). The S1 fixed blend learns nothing (predeclared weight) and
uses only zone j's own honest W2 map: it is the one ``STRICT_LABEL_FREE`` challenger.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import rankers as RK

SCREENING, STRICT_FREE, BLOCKED = "SCREENING_ONLY", "STRICT_LABEL_FREE", "STRICT_NESTING_BLOCKED"
STRICTNESS = {"G0": STRICT_FREE, "S1": STRICT_FREE, **{m: SCREENING for m in RK.LEARNED}}


class CrossfitLeakError(RuntimeError):
    pass


def training_mask(T: pd.DataFrame, held_zone: int, zone_mask_full) -> np.ndarray:
    from ..geomoe import gating as GA

    tr = GA.training_rows(T, held_zone, zone_mask_full)
    if "eligible_share" in T:
        tr &= T.eligible_share.to_numpy(float) > 0
    GA.assert_disjoint(T.zone[tr].unique(), held_zone)
    return tr


def crossfit_unit(R: pd.DataFrame, key: str, scenario: str, zone: int, zone_mask_full, models, rk: dict, lc: dict,
                  feats: dict) -> tuple[dict, dict, list[dict]]:
    """({model: score frame}, fit records, leakage rows) of every model on one held-out unit. ``feats`` =
    {'nograph': [...], 'graph': [...]} registry feature lists."""
    T = R[R.scenario == scenario].reset_index(drop=True)
    held = (T.unit == key).to_numpy()
    tr = training_mask(T, zone, zone_mask_full)
    if (tr & held).any():
        raise CrossfitLeakError(f"{key}: a held-out row is among the training rows")
    Th = T[held]
    out, rec, leak = {}, {}, []
    base = Th[["unit", "label", "n_px"]].reset_index(drop=True)
    seed = int(rk["seed"])
    Ttr = T[tr].reset_index(drop=True)
    L = RK.make_labels(Ttr, lc)
    info = {"train_zones": sorted(int(z) for z in Ttr.zone.unique()), "n_train": int(len(Ttr)),
            "n_positive": int((L.cls == "positive").sum()), "n_hard_negative": int((L.cls == "hard_negative").sum()),
            "n_ambiguous": int((L.cls == "ambiguous").sum()), "n_unlabeled": int((L.cls == "unlabeled").sum())}
    leak.append({"unit": key, "check": "held-out zone not among training zones", "status": "PASS" if zone not in
                 info["train_zones"] else "FAIL", "detail": str(info["train_zones"]), "method": "assert_disjoint"})
    leak.append({"unit": key, "check": "no truth-derived feature", "status": "PASS",
                 "detail": "registry.assert_label_free on every feature list", "method": "executed in feature_columns"})
    for m in models:
        if m == "G0":
            sc, r = Th.w2_unit_rank.to_numpy(float), {"status": "NO_FIT"}
        elif m == "S1":
            sc, r = RK.strict_blend(Th, float(rk["strict_fixed_blend"])), {"status": "NO_FIT",
                                                                            "blend": float(rk["strict_fixed_blend"])}
        else:
            spec = RK.model_spec(m, R, feats["nograph"], feats["graph"])
            if spec is None:
                rec[m] = {**info, "status": "UNAVAILABLE (inputs absent)", "strictness": STRICTNESS[m]}
                continue
            if info["n_positive"] < 5 or info["n_train"] < 50:
                rec[m] = {**info, "status": "NOT_FITTED (too few training rows / positives)",
                          "strictness": STRICTNESS[m]}
                continue
            lam = RK.choose_lambda(spec, Ttr, L, rk, rk["gate"], seed)
            _, c = RK._fit_predict(spec, Ttr, L, Th, rk, seed)
            if spec.get("gated"):
                c = c * RK.gate_mask(Th, rk["gate"])
            sc = RK.logit(Th.w2_unit_rank.to_numpy()) + lam["lambda"] * c
            r = {"status": "FITTED", "lambda": lam["lambda"], "lambda_status": lam["status"],
                 "inner_ap": lam["inner"], "features": spec["features"], "n_features": len(spec["features"])}
        out[m] = base.assign(score=np.asarray(sc, float), model=m)
        rec[m] = {**info, **r, "strictness": STRICTNESS[m]}
    return out, rec, leak


# =============================================================================== nesting audit
def nesting_audit(units: list[tuple[str, str, int]], models) -> pd.DataFrame:
    """Per (model, held-out unit): what is and is not nested. ``units`` = [(key, scenario, zone)]."""
    rows = []
    for key, sc, z in units:
        for m in models:
            learned = m in RK.LEARNED
            base = {"model": m, "unit": key, "scenario": sc, "zone": z}
            rows += [
                {**base, "check": "held-out truth excluded from the fit", "status": "PASS" if learned else
                 "NOT_APPLICABLE (no fit)", "detail": "training_rows + assert_disjoint + buffer"},
                {**base, "check": "training-row W2 / CNN values independent of held-out truth",
                 "status": "FAIL_STRICT" if learned else "NOT_APPLICABLE",
                 "detail": "W2 / CNN of training zone k were fitted with zone-j systems as labels"},
                {**base, "check": "training-row known context excludes held-out truth",
                 "status": "FAIL_STRICT" if learned else "NOT_APPLICABLE",
                 "detail": "permitted_known(k) keeps zone-j truth pixels as known faults"},
                {**base, "check": "held-out unit scored with its own honest W2 map",
                 "status": "PASS", "detail": "W2 of zone j excludes zone-j truth (discovery / OOF maps)"},
                {**base, "check": "classification", "status": STRICTNESS.get(m, SCREENING),
                 "detail": "screening numbers are never reported as nested DTI" if learned else
                 "no learning: strict by construction"}]
    return pd.DataFrame(rows)


def retraining_plan(disc_zones, geo_zones) -> pd.DataFrame:
    """What strict nesting of a learned residual ranker would require (not executed here)."""
    rows = []
    for tag, zones in (("disc_official", disc_zones), ("geo", geo_zones)):
        for j in zones:
            for k in zones:
                if k == j:
                    continue
                rows.append({"task": tag, "outer_zone": j, "inner_training_zone": k,
                             "labels_removed": f"withheld({j}) | withheld({k})" if tag == "disc_official" else
                             f"fold {j} | fold {k}",
                             "refit_w2_stack": True, "refit_cnn_b0_b1": True, "recompute_known_features": True,
                             "recompute_g02": False, "compute": "W2: CPU minutes per refit; CNN: GPU hours per refit "
                             "(refused in this notebook)", "status": BLOCKED})
    return pd.DataFrame(rows)
