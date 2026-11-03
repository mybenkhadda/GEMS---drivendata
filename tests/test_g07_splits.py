"""G07 crossfit isolation: outer folds, upstream honesty, held-out truth never used, synthetic rows train-only."""
import numpy as np
import pandas as pd
import pytest

from gems_bench.g07 import crossfit as CX
from gems_bench.g07.models_candidates import candidate_feature_columns


def test_outer_folds():
    f = CX.build_outer_folds(3, 3, ("V1", "V3"), True)
    assert "V1:0" in f and "V3:2" in f and "V1:0|1" in f and "V1:1|1" not in f


@pytest.mark.parametrize("inputs,nested,expected", [({"w2": True}, True, "SCREENING_ONLY"),
                                                    ({"subsurface": True}, True, "STRICT_NESTED"),
                                                    ({"subsurface": True}, False, "SCREENING_ONLY"),
                                                    ({}, True, "STRICT_NESTED")])
def test_upstream_honesty(inputs, nested, expected):
    assert CX.validate_upstream_honesty(inputs, nested)[0] == expected


def candidates(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for z in range(3):
        for syn in (False, True):
            n = 80
            g = rng.normal(size=n)
            rows.append(pd.DataFrame({"unit": f"V1:{z}", "scenario": "V1", "zone": z,
                                      "label": np.arange(1, n + 1) + (1_000_000 if syn else 0),
                                      "geo_mean_x": g, "seg_length_km": 1.0, "label_conf": 1.0,
                                      "ev_w2_rank_mean": rng.random(n), "target": (g > 0).astype(float),
                                      "system_id": rng.integers(0, 5, n), "synthetic": syn,
                                      "r_full": z * 10 + 1, "c_full": 1}))
    return pd.concat(rows, ignore_index=True)


def zm():
    out = {}
    for z in range(3):
        a = np.zeros((40, 5), bool)
        a[z * 10:(z + 1) * 10] = True
        out[f"V1:{z}"] = a
    return out


def test_heldout_truth_and_synthetic_rows_never_leak():
    pytest.importorskip("lightgbm")
    D = candidates()
    prm = {"n_estimators": 30, "num_leaves": 7, "min_child_samples": 5}
    cols = lambda F: candidate_feature_columns(F, False)                     # noqa: E731
    s1, info = CX.crossfit_candidate_ranker("D1", D, cols, "lightgbm", prm, zm(), 0, progress=None)
    D2 = D.copy()
    m = D2.unit == "V1:0"
    D2.loc[m, "target"] = 1 - D2.loc[m, "target"]                           # held-out truth permuted
    s2, _ = CX.crossfit_candidate_ranker("D1", D2, cols, "lightgbm", prm, zm(), 0, progress=None)
    a = s1[s1.unit == "V1:0"].sort_values("label").score.to_numpy()
    b = s2[s2.unit == "V1:0"].sort_values("label").score.to_numpy()
    np.testing.assert_allclose(a, b)
    assert (s1.label < 1_000_000).all()                                     # synthetic candidates are never scored
    assert info["units"]["V1:0"]["n_synthetic"] > 0                         # ... but they do train other units
    assert info["strictness"] == "STRICT_NESTED"                            # no supervised upstream input declared


def test_require_strict_blocks_missing_nested_inputs():
    pytest.importorskip("lightgbm")
    D = candidates()
    s, info = CX.crossfit_candidate_ranker("D1", D, lambda F: candidate_feature_columns(F, False), "lightgbm",
                                           {"n_estimators": 10}, zm(), 0, unit_columns=lambda k, o: None,
                                           inputs={"subsurface": True}, require_strict=True, progress=None)
    assert info["strictness"] == "BLOCKED_UPSTREAM" and not len(s)


def test_system_balanced_weights():
    T = pd.DataFrame({"label_conf": 1.0, "seg_length_km": 1.0, "system_id": [1, 1, 1, 2, -1, -1]})
    y = np.array([1, 1, 1, 1, 0, 0])
    w = CX.system_balanced_weights(T, y)
    assert w[:3].sum() == pytest.approx(w[3])                               # one system = one unit of weight
    assert w[y == 1].sum() == pytest.approx(w[y == 0].sum())
