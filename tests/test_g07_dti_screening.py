"""Synthetic, label-free regression checks for G07 DTI screening.

Authoring-only: run in Colab, not as part of the notebook creation step.
"""
import numpy as np
import pandas as pd

from gems_bench.g07.dti_screening import _pooled, _rank_map
from gems_bench.g07.representation import exact_k_mask
from gems_bench.gems3.official import FastScorer


def test_feature_rank_directions_are_predeclared_and_label_free():
    x = np.array([[1., -2., np.nan], [3., 0., -4.]], dtype=float)
    valid = np.ones(x.shape, bool)
    high = _rank_map(x, valid, "high")
    absolute = _rank_map(x, valid, "abs")
    assert not np.isfinite(high[0, 2])
    assert high[1, 0] > high[0, 0]
    assert absolute[1, 2] > absolute[0, 0]
    assert np.isnan(_rank_map(x, np.zeros(x.shape, bool))[0, 0])


def test_full_exact_k_budget_matches_original_reference_area():
    rng = np.random.default_rng(2026)
    shape = 80, 80
    eligible = rng.random(shape) > 0.3
    w2 = rng.normal(size=shape)
    dem = rng.random(shape)
    dem_valid = np.zeros(shape, bool)
    dem_valid[10:30, 10:30] = True
    w2_r = _rank_map(w2, eligible)
    dem_r = _rank_map(dem, dem_valid)
    fusion = w2_r.copy()
    valid = eligible & dem_valid
    fusion[valid] = 0.85 * w2_r[valid] + 0.15 * dem_r[valid]
    assert np.array_equal(fusion[~valid], w2_r[~valid], equal_nan=True)
    for frac in (0.005, 0.01, 0.02):
        base, info_b = exact_k_mask(w2, eligible, frac)
        fused, info_f = exact_k_mask(fusion, eligible, frac)
        assert base.sum() == fused.sum() == int(np.ceil(frac * eligible.sum()))
        assert info_b["shortfall"] == info_f["shortfall"] == 0


def test_pooled_dti_computed_from_components_not_mean_of_dti():
    d = pd.DataFrame([
        dict(unit="V1:0", model="W2", scenario="V1", frac=.01,
             tp_w=10., fp_w=8., fn_w=2., n_truth=12),
        dict(unit="V1:1", model="W2", scenario="V1", frac=.01,
             tp_w=20., fp_w=16., fn_w=4., n_truth=24),
    ])
    got = _pooled(d, ("model", "scenario", "frac"), {"V1": 3})
    row = got.iloc[0]
    expected = 30. / (30. + .2 * 24. + .8 * 6. + 1e-6)
    assert np.isclose(row.dti, expected)
    assert row.complete == False
    assert row.n_expected_units == 3


def test_restricted_coverage_changes_protocol_not_competition_truth():
    truth = np.zeros((15, 15), bool)
    truth[7, 7] = True
    valid = np.ones_like(truth)
    cov = np.zeros_like(truth)
    cov[:, 7:] = True
    q = np.zeros((15, 15), np.float32)
    q[7, 7] = 1.
    full = FastScorer(truth, valid, exclude_supports_tp=False).components(q)
    local = FastScorer(truth, valid & cov, exclude_supports_tp=False).components(q)
    assert np.isclose(full.dti, local.dti)
    cov[7, 7] = False
    restricted = FastScorer(truth, valid & cov, exclude_supports_tp=False).components(q)
    assert restricted.n_truth == 0
