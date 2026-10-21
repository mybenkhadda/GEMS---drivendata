"""Synthetic scoring tests for Phase 3 (authored, run with: python -m pytest tests/test_phase3_scoring.py -q).

Analytic values on a vertical 30-px fault line (100 m pixels, R = 300 m, alpha 0.2, beta 0.8):

* shift s px:            DTI = 1 - s/3            (TP = n k, FP = n (1 - k), FN = n (1 - k))
* binary width 3 / 5:    DTI = 30/34, 30/42
* soft k(d) corridor:    DTI = 1 / (1 + 0.2 * 8/9)
"""
import itertools

import numpy as np
import pytest

from gems_bench.metric import MaskedScorer, dti_components
from gems_bench.p3.curves import BinaryCurve, crossfit_select, pooled, zone_curves

N, ROWS, COL = 40, slice(5, 35), 20


def line(col=COL, value=1.0, shape=(N, N)):
    a = np.zeros(shape)
    a[ROWS, col] = value
    return a


TRUTH = line().astype(bool)


@pytest.mark.parametrize("s,expected", [(0, 1.0), (1, 2 / 3), (2, 1 / 3), (3, 0.0)])
def test_displacement(s, expected):
    assert dti_components(line(COL + s), TRUTH).dti == pytest.approx(expected, abs=1e-4)


def test_width_expansion():
    w3 = line(19) + line(20) + line(21)
    w5 = w3 + line(18) + line(22)
    assert dti_components(w3, TRUTH).dti == pytest.approx(30 / 34, abs=1e-4)
    assert dti_components(w5, TRUTH).dti == pytest.approx(30 / 42, abs=1e-4)


def test_soft_corridor_below_exact_line_above_wide_binary():
    corr = line(20) + line(19, 2 / 3) + line(21, 2 / 3) + line(18, 1 / 3) + line(22, 1 / 3)
    v = dti_components(corr, TRUTH).dti
    assert v == pytest.approx(1 / (1 + 0.2 * 8 / 9), abs=1e-4)
    assert 30 / 42 < v < 1.0


def test_parallel_prediction_far_away_scores_zero():
    assert dti_components(line(30), TRUTH).dti == 0.0


def test_exact_exclusion():
    other = np.zeros((N, N), bool)
    other[ROWS, 5] = True
    truth = TRUTH | other
    pred = line(5)                                   # predicts only the excluded fault
    c = dti_components(pred, truth, exclude_mask=other)
    assert c.n_truth == 30 and c.tp == 0 and c.fp == pytest.approx(0.0)
    c2 = dti_components(pred, truth)                 # without exclusion it is a true positive
    assert c2.tp == pytest.approx(30.0)


def test_withheld_policies():
    withheld = TRUTH
    retained = np.zeros((N, N), bool)
    retained[ROWS, 5] = True
    pred = line(5) + line(20, 0.5)
    excl = dti_components(pred, withheld, exclude_mask=retained)
    as_fp = dti_components(pred, withheld)
    all_truth = dti_components(pred, withheld | retained)
    assert as_fp.fp > excl.fp                         # predictions on known faults cost FP under 'retained_fp'
    assert all_truth.dti > as_fp.dti
    assert excl.dti > as_fp.dti


def test_valid_masking_and_nan():
    pred = line(20)
    pred[:, 30:] = 1.0                               # outside the evaluation mask: ignored entirely
    ev = np.zeros((N, N), bool)
    ev[:, :28] = True
    assert dti_components(pred, TRUTH, eval_mask=ev).dti == pytest.approx(1.0, abs=1e-4)
    p_nan = line(20)
    p_nan[0, 0] = np.nan
    assert dti_components(p_nan, TRUTH).dti == pytest.approx(1.0, abs=1e-4)


def test_pooled_differs_from_mean_of_regions():
    t = np.zeros((N, 2 * N), bool)
    t[ROWS, 10] = True
    t[ROWS, 60] = True
    p = np.zeros_like(t, float)
    p[ROWS, 10] = 1.0                                # region A perfect
    p[:, 45:] = 0.2                                  # region B: diffuse
    zones = np.zeros(t.shape, int)
    zones[:, N:] = 1
    a = dti_components(p, t, eval_mask=zones == 0)
    b = dti_components(p, t, eval_mask=zones == 1)
    pooled_c = a + b
    assert pooled_c.dti != pytest.approx(0.5 * (a.dti + b.dti), abs=1e-3)


def test_binary_curve_matches_masked_scorer():
    rng = np.random.default_rng(0)
    truth = np.zeros((60, 60), bool)
    truth[10:50, 30] = True
    truth[20, 5:40] = True
    p = np.clip(rng.random((60, 60)) * 0.6 + 0.4 * np.exp(-np.abs(np.arange(60) - 31)[None, :] / 3), 0, 1)
    ev = np.ones_like(truth)
    ev[:, 55:] = False
    ex = np.zeros_like(truth)
    ex[18:23, 5:15] = True
    curve = BinaryCurve(p, truth, ev, ex)
    sc = MaskedScorer(truth, ev, ex)
    for t in (0.2, 0.5, 0.7, 0.9):
        row = curve.at([t]).iloc[0]
        c = sc.components((p >= t).astype(float))
        assert row.tp_w == pytest.approx(c.tp, rel=1e-9, abs=1e-9)
        assert row.fp_w == pytest.approx(c.fp, rel=1e-9, abs=1e-9)
        assert row.dti == pytest.approx(c.dti, rel=1e-9, abs=1e-12)


def test_recall_tolerances():
    c = BinaryCurve(line(21), TRUTH, np.ones((N, N), bool))
    r = c.at([0.5]).iloc[0]
    assert r["recall@100m"] == 1.0
    c2 = BinaryCurve(line(22), TRUTH, np.ones((N, N), bool))
    r2 = c2.at([0.5]).iloc[0]
    assert r2["recall@100m"] == 0.0 and r2["recall@200m"] == 1.0


def test_binary_is_dti_optimal_on_small_instance():
    """Brute force: no soft map beats the best binary mask (vertex argument in curves.py)."""
    rng = np.random.default_rng(1)
    truth = np.zeros((3, 4), bool)
    truth[1, 0] = truth[1, 3] = True
    best_bin = max(dti_components(np.array(m, float).reshape(3, 4), truth).dti
                   for m in itertools.product((0, 1), repeat=12))
    for _ in range(300):
        soft = rng.random((3, 4)) ** rng.uniform(0.3, 3)
        assert dti_components(soft, truth).dti <= best_bin + 1e-9


def test_zone_curves_and_crossfit():
    rng = np.random.default_rng(2)
    t = np.zeros((60, 90), bool)
    for c in (10, 40, 70):
        t[5:55, c] = True
    p = rng.random(t.shape) * 0.5
    p[t] += 0.5
    zones = np.repeat(np.arange(3), 30)[None, :].repeat(60, 0)
    valid = np.ones_like(t)
    zc = zone_curves(p, t, zones, valid, fracs=(0.02, 0.05, 0.2))
    per, s = crossfit_select(zc, ["target_frac"])
    assert set(per.zone) == {0, 1, 2}
    assert s["dti_insample"] >= s["dti_crossfit"] - 1e-12
    pl = pooled(zc, ["target_frac"])
    assert (pl.area_frac > 0).all()
