"""Synthetic checks of the official distance-weighted Tversky index (gems_bench.metric).

Run later with:  python -m pytest tests/test_metric.py -q

Geometry used throughout: 100 m pixels, R = 300 m, so k(100 m) = 2/3, k(200 m) = 1/3,
k(300 m) = 0. A vertical truth line of n pixels displaced by j pixels gives
TP = n k, FN = n (1 - k), FP = n (1 - k), hence DTI = k / (k + (alpha + beta)(1 - k)) = k
because alpha + beta = 1.
"""
import numpy as np
import pytest

from gems_bench.metric import (
    DTIComponents, MaskedScorer, apply_output_transform, cross_fitted_transform, dti, dti_components,
    kernel_offsets, local_weighted_max, pool, tolerant_recall_precision, transform_component_table,
)

H, W = 60, 60
ROWS = slice(10, 50)         # 40-pixel vertical line
N = 40
COL = 20
TOL = 1e-4


def line(col=COL, rows=ROWS, shape=(H, W)):
    a = np.zeros(shape, bool)
    a[rows, col] = True
    return a


def test_kernel_offsets_support():
    offs = {(dr, dc): k for dr, dc, k in kernel_offsets(300.0, (100.0, 100.0))}
    assert offs[(0, 0)] == pytest.approx(1.0)
    assert offs[(0, 1)] == pytest.approx(2 / 3)
    assert offs[(0, 2)] == pytest.approx(1 / 3)
    assert offs[(1, 1)] == pytest.approx(1 - np.sqrt(2) / 3)
    assert (0, 3) not in offs and (3, 0) not in offs          # d = R -> k = 0, excluded
    assert (2, 2) in offs                                    # d = 283 m < R


def test_perfect_prediction():
    t = line()
    c = dti_components(t.astype(float), t)
    assert c.tp == pytest.approx(N) and c.fn == pytest.approx(0) and c.fp == pytest.approx(0)
    assert c.dti == pytest.approx(1.0, abs=TOL)


def test_empty_prediction():
    t = line()
    c = dti_components(np.zeros((H, W)), t)
    assert c.tp == 0 and c.fp == 0 and c.fn == pytest.approx(N)
    assert c.dti == pytest.approx(0.0)


def test_empty_truth_and_prediction_is_zero_not_nan():
    c = dti_components(np.zeros((H, W)), np.zeros((H, W), bool))
    assert c.dti == 0.0


def test_missing_fault_segment():
    a, b = line(col=10), line(col=45)
    c = dti_components(a.astype(float), a | b)
    assert c.tp == pytest.approx(N) and c.fn == pytest.approx(N) and c.fp == pytest.approx(0)
    assert c.dti == pytest.approx(N / (N + 0.8 * N), abs=TOL)


@pytest.mark.parametrize("shift,k", [(1, 2 / 3), (2, 1 / 3)])
def test_pixel_displacement(shift, k):
    t = line()
    p = line(col=COL + shift).astype(float)
    c = dti_components(p, t)
    assert c.tp == pytest.approx(N * k)
    assert c.fn == pytest.approx(N * (1 - k))
    assert c.fp == pytest.approx(N * (1 - k))
    assert c.dti == pytest.approx(k, abs=TOL)


@pytest.mark.parametrize("shift", [3, 4, 10])
def test_outside_support(shift):
    t = line()
    p = line(col=COL + shift).astype(float)
    c = dti_components(p, t)
    assert c.tp == pytest.approx(0.0)
    assert c.fp == pytest.approx(N)
    assert c.dti == pytest.approx(0.0)


def test_soft_prediction():
    t = line()
    c = dti_components(0.5 * t, t)
    assert c.tp == pytest.approx(0.5 * N) and c.fn == pytest.approx(0.5 * N) and c.fp == pytest.approx(0)
    assert c.dti == pytest.approx(0.5 / (0.5 + 0.8 * 0.5), abs=TOL)


def test_local_max_not_sum():
    t = np.zeros((H, W), bool)
    t[30, 30] = True
    p = np.zeros((H, W))
    p[30, 29] = p[30, 31] = 1.0                              # two hits at 100 m
    c = dti_components(p, t)
    assert c.tp == pytest.approx(2 / 3)                      # max, not 4/3
    assert c.fp == pytest.approx(2 * (1 / 3))


def test_broad_corridor():
    """7-px wide corridor: full recall, FP per row = 2 (1/3 + 2/3 + 1) = 4."""
    t = line()
    p = np.zeros((H, W))
    p[ROWS, COL - 3:COL + 4] = 1.0
    c = dti_components(p, t)
    assert c.tp == pytest.approx(N) and c.fn == pytest.approx(0)
    assert c.fp == pytest.approx(4 * N)
    assert c.dti == pytest.approx(1 / (1 + 0.2 * 4), abs=TOL)


def test_known_fault_exclusion():
    a, b = line(col=10), line(col=45)
    p = (a | b).astype(float)
    # with A excluded: A is neither truth nor FP -> perfect on B
    c = dti_components(p, a | b, exclude_mask=a)
    assert c.n_truth == N and c.fp == pytest.approx(0) and c.dti == pytest.approx(1.0, abs=TOL)
    # same prediction, truth = B only, no exclusion -> A becomes full-weight FP
    c2 = dti_components(p, b)
    assert c2.fp == pytest.approx(N)
    assert c2.dti == pytest.approx(N / (N + 0.2 * N), abs=TOL)


def test_exclusion_does_not_remove_tp_support():
    """Excluded predictions still support nearby truth (only G and FP are masked)."""
    t = line()
    p = line(col=COL + 1).astype(float)
    ex = line(col=COL + 1)
    c = dti_components(p, t, exclude_mask=ex)
    assert c.tp == pytest.approx(N * 2 / 3) and c.fp == pytest.approx(0)


def test_nodata_and_eval_mask():
    t = line()
    ev = np.zeros((H, W), bool)
    ev[:, :40] = True
    p = t.astype(float)
    p[:, 40:] = 1.0                                          # garbage outside eval
    p[0, 0] = np.nan                                         # nodata inside eval
    truth = t.copy()
    truth[ROWS, 50] = True                                   # truth outside eval is ignored
    c = dti_components(p, truth, eval_mask=ev)
    assert c.n_truth == N and c.fp == pytest.approx(0)
    assert c.dti == pytest.approx(1.0, abs=TOL)


def test_predictions_outside_eval_cannot_create_tp():
    t = line(col=38)
    ev = np.zeros((H, W), bool)
    ev[:, :39] = True
    p = line(col=39).astype(float)                           # 100 m away, but outside eval
    assert dti_components(p, t, eval_mask=ev).tp == pytest.approx(0.0)


def test_nan_equals_zero_and_clipping():
    t = line()
    p_nan = np.full((H, W), np.nan)
    assert dti(p_nan, t) == pytest.approx(dti(np.zeros((H, W)), t))
    assert dti(2.0 * t, t) == pytest.approx(1.0, abs=TOL)


def test_crop_is_exact():
    rng = np.random.default_rng(0)
    t = rng.random((H, W)) > 0.97
    p = rng.random((H, W))
    ev = np.zeros((H, W), bool)
    ev[5:35, 8:50] = True
    a = dti_components(p, t, eval_mask=ev, crop=True)
    b = dti_components(p, t, eval_mask=ev, crop=False)
    assert a.tp == pytest.approx(b.tp) and a.fp == pytest.approx(b.fp) and a.fn == pytest.approx(b.fn)


def test_brute_force_definition():
    """Compare with a literal O(|G| |X|) evaluation of the organizer formula."""
    rng = np.random.default_rng(1)
    h, w = 14, 16
    t = rng.random((h, w)) > 0.85
    p = rng.random((h, w)) * (rng.random((h, w)) > 0.5)
    ys, xs = np.mgrid[0:h, 0:w]
    xy = np.c_[ys.ravel(), xs.ravel()] * 100.0
    g = xy[t.ravel()]
    d = np.sqrt(((xy[:, None, :] - g[None, :, :]) ** 2).sum(-1))      # (X, G)
    k = np.clip(1 - d / 300.0, 0, None)
    tp = (p.ravel()[:, None] * k).max(0).sum()
    fp = (p.ravel() * (1 - k.max(1))).sum()
    fn = len(g) - tp
    ref = tp / (tp + 0.2 * fp + 0.8 * fn + 1e-6)
    c = dti_components(p, t)
    assert c.tp == pytest.approx(tp) and c.fp == pytest.approx(fp) and c.fn == pytest.approx(fn)
    assert c.dti == pytest.approx(ref)


def test_masked_scorer_matches_function():
    rng = np.random.default_rng(3)
    t = rng.random((H, W)) > 0.95
    ev = np.zeros((H, W), bool)
    ev[3:40, 5:55] = True
    ex = rng.random((H, W)) > 0.97
    s = MaskedScorer(t, ev, ex)
    for _ in range(3):
        p = rng.random((H, W)) ** 3
        a, b = s.components(p), dti_components(p, t, eval_mask=ev, exclude_mask=ex)
        assert a.tp == pytest.approx(b.tp) and a.fp == pytest.approx(b.fp) and a.fn == pytest.approx(b.fn)


def test_pooling_sums_components():
    a = DTIComponents(2.0, 1.0, 1.0, 3, 3.0)
    b = DTIComponents(1.0, 3.0, 2.0, 3, 4.0)
    c = pool([a, b])
    assert (c.tp, c.fp, c.fn, c.n_truth) == (3.0, 4.0, 3.0, 6)
    assert c.dti == pytest.approx(3 / (3 + 0.2 * 4 + 0.8 * 3 + 1e-6))


def test_local_weighted_max_border():
    p = np.zeros((5, 5))
    p[0, 0] = 1.0
    M = local_weighted_max(p, 300.0, (100.0, 100.0))
    assert M[0, 0] == 1.0 and M[0, 1] == pytest.approx(2 / 3) and M[0, 3] == 0.0


def test_tolerant_recall():
    t = line()
    p = line(col=COL + 2).astype(float)
    ev = np.ones((H, W), bool)
    assert tolerant_recall_precision(p, t, ev, 100.0, 0.5)["recall"] == 0.0
    assert tolerant_recall_precision(p, t, ev, 200.0, 0.5)["recall"] == 1.0


def test_output_transform_and_crossfit():
    p = np.array([0.0, 0.04, 0.5, 1.0, np.nan])
    q = apply_output_transform(p, gamma=1.0, scale=2.0, floor=0.05)
    assert q[1] == 0.0 and q[2] == pytest.approx(1.0) and np.isnan(q[4])
    t = line()
    oof = 0.3 * line(col=COL).astype(float)
    folds = np.zeros((H, W), int)
    folds[:, 30:] = 1
    folds[:30, :] = np.where(folds[:30, :] == 0, 2, folds[:30, :])
    tab = transform_component_table(oof, t, folds, np.ones((H, W), bool),
                                    grid={"gamma": (1.0,), "scale": (1.0, 2.0), "floor": (0.0,)})
    per_fold, final = cross_fitted_transform(tab)
    assert final["scale"] == 2.0                             # brighter soft prediction scores higher
