"""G05.1 cross-fitting on a synthetic registry: held-out truth never changes held-out scores."""
import numpy as np
import pandas as pd
import pytest

from gems_bench.g05_1 import crossfit as XF
from gems_bench.g05_1 import rankers as RK

pytest.importorskip("lightgbm")

RKC = {"seed": 7, "lgbm": {"n_estimators": 20, "learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 10,
                           "subsample": 0.8, "colsample_bytree": 1.0, "reg_lambda": 1.0},
       "lambda_grid": [0.0, 0.5, 1.0], "pair_group_km": 20.0, "gate": {"w2_q_max": 0.7, "evidence_q_min": 0.5},
       "strict_fixed_blend": 0.25}
LC = {"positive_share": 0.5, "ambiguous_share": 0.1, "hard_negative_w2_q": 0.9, "hard_negative_weight": 1.0,
      "unlabeled_weight": 0.3}
FEATS = {"nograph": ["mean_cnn_b0", "mean_mag_hgrad"], "graph": ["mean_cnn_b0", "mean_mag_hgrad", "g_support"]}


def registry(seed=0, n=300):
    rng = np.random.default_rng(seed)
    parts = []
    for z in range(3):
        cnn = rng.random(n)
        y = np.where(cnn > 0.8, 0.9, np.where(cnn > 0.7, 0.3, 0.0))
        parts.append(pd.DataFrame({"scenario": "V1", "zone": z, "unit": f"V1:{z}", "label": np.arange(1, n + 1),
                                   "n_px": rng.integers(5, 40, n), "row_full": 1000 * z + rng.integers(0, 500, n),
                                   "col_full": rng.integers(0, 500, n), "eligible_share": 1.0,
                                   "w2_unit_rank": rng.random(n), "mean_cnn_b0": cnn,
                                   "mean_mag_hgrad": rng.random(n), "g_support": rng.random(n),
                                   "y_near_truth": y}))
    return pd.concat(parts, ignore_index=True)


def run(R, models=("G0", "S1", "G1", "G2", "G3", "G4", "G5")):
    return XF.crossfit_unit(R, "V1:0", "V1", 0, None, list(models), RKC, LC, FEATS)


def test_models_scored_and_recorded():
    out, rec, leak = run(registry())
    assert {"G0", "S1", "G1", "G2", "G3", "G5"} <= set(out)
    assert all(len(s) == 300 for s in out.values())
    assert rec["G1"]["status"] == "FITTED" and rec["G1"]["train_zones"] == [1, 2]
    assert rec["G1"]["lambda"] in RKC["lambda_grid"]
    assert rec["G4"]["status"].startswith("UNAVAILABLE")           # no domain column
    assert rec["G1"]["strictness"] == XF.SCREENING and rec["S1"]["strictness"] == XF.STRICT_FREE
    assert all(r["status"] == "PASS" for r in leak)


def test_heldout_truth_does_not_change_heldout_scores():
    R = registry()
    a, _, _ = run(R, ("G1", "G2", "S1"))
    R2 = R.copy()
    m = R2.zone == 0
    R2.loc[m, "y_near_truth"] = np.random.default_rng(99).permutation(R2.loc[m, "y_near_truth"].to_numpy())
    b, _, _ = run(R2, ("G1", "G2", "S1"))
    for k in a:
        assert np.allclose(a[k].score.to_numpy(), b[k].score.to_numpy()), k


def test_g0_is_w2_rank():
    R = registry()
    out, _, _ = run(R, ("G0",))
    assert np.allclose(out["G0"].score.to_numpy(), R[R.zone == 0].w2_unit_rank.to_numpy())


def test_lambda_zero_returns_w2_order():
    R = registry()
    T = R[R.zone != 0].reset_index(drop=True)
    L = RK.make_labels(T, LC)
    spec = RK.model_spec("G1", R, FEATS["nograph"], FEATS["graph"])
    lam = RK.choose_lambda(spec, T, L, {**RKC, "lambda_grid": [0.0]}, RKC["gate"], 7)
    assert lam["lambda"] == 0.0 and lam["status"] == "INNER_ZONE_SELECTED"


def test_gate_mask_bounds():
    T = pd.DataFrame({"unit": ["u"] * 4, "w2_unit_rank": [0.1, 0.9, 0.2, 0.5], "g_support": [9.0, 9.0, 0.0, 5.0]})
    g = RK.gate_mask(T, {"w2_q_max": 0.7, "evidence_q_min": 0.5})    # evidence ranks .875 .875 .25 .5
    assert list(g) == [True, False, False, True]
