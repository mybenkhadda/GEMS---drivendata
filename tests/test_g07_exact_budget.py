"""G07 exact-k: ties, determinism, crop invariance, secondary tie-break, shortfall, portfolio budget."""
import numpy as np
import pandas as pd
import pytest

from gems_bench.g07 import representation as R


def test_constant_scores_select_exactly_k_and_legacy_does_not():
    s = np.ones((50, 40))
    el = np.ones_like(s, bool)
    m, rep = R.exact_k_mask(s, el, 0.01)
    k = int(np.ceil(0.01 * el.sum()))
    assert m.sum() == k == rep["actual_px"] and rep["n_tied_at_cutoff"] == el.sum()
    assert rep["legacy_px"] == el.sum()                                   # the old rule selects everything
    lm, _ = R.exact_k_mask(s, el, 0.01, tie_break="legacy")
    assert lm.sum() == el.sum()


def test_deterministic_and_crop_invariant():
    rng = np.random.default_rng(0)
    s = np.round(rng.random((60, 60)), 1)                                  # many ties
    el = np.ones_like(s, bool)
    a, _ = R.exact_k_mask(s, el, k=100, width_full=60)
    b, _ = R.exact_k_mask(s, el, k=100, width_full=60)
    np.testing.assert_array_equal(a, b)
    crop = (slice(10, 50), slice(5, 55))
    elc = np.zeros_like(el)
    elc[crop] = True
    full, _ = R.exact_k_mask(np.where(elc, s, np.nan), elc, k=200, width_full=60)
    sub, _ = R.exact_k_mask(s[crop], el[crop], k=200, offset=(10, 5), width_full=60)
    np.testing.assert_array_equal(full[crop], sub)                         # same pixels whatever the crop


def test_secondary_breaks_ties_label_free():
    s = np.zeros((10, 10))
    sec = np.arange(100, dtype=float).reshape(10, 10)
    m, _ = R.exact_k_mask(s, np.ones_like(s, bool), k=5, secondary=sec)
    assert set(np.flatnonzero(m.ravel())) == set(range(95, 100))


def test_shortfall_reported_and_ineligible_never_selected():
    s = np.full((10, 10), np.nan)
    s[0, :3] = 1.0
    el = np.ones_like(s, bool)
    el[0, 0] = False
    m, rep = R.exact_k_mask(s, el, k=10)
    assert m.sum() == 2 and rep["shortfall"] == 8 and not m[0, 0]


def test_apply_policy_post_ops_and_report():
    rng = np.random.default_rng(1)
    s = rng.random((40, 40))
    ev = np.ones_like(s, bool)
    q, rep = R.apply_policy({"select": "top", "post": [["dilate", {"radius_px": 1}]]}, s, ev, 0.01, eligible=ev)
    assert rep["actual_px"] == rep["k"] and rep["final_px"] >= rep["k"]


def test_portfolio_is_exactly_k_and_keeps_w2_base():
    rng = np.random.default_rng(2)
    w2 = rng.random((50, 50))
    el = np.ones_like(w2, bool)
    lab = np.zeros((50, 50), np.int32)
    lab[5, 5:25] = 1
    lab[40, 5:45] = 2
    U = pd.DataFrame({"label": [1, 2], "utility": [5.0, -1.0]})
    m, rep = R.candidate_portfolio_selection(w2, el, 0.04, lab, U, 0.8)
    k = R.budget_k(0.04, el.sum())
    base, _ = R.exact_k_mask(w2, el, k=int(np.floor(0.8 * k)))
    assert m.sum() == k
    assert (m[base]).all()                                                  # W2's strongest pixels are preserved
    assert m[lab == 1].all()                                                # positive-utility candidate promoted whole
    assert rep["added_candidates"] == 1                                     # the negative-utility one is not
    assert rep["base_px"] + rep["candidate_px"] + rep["w2_fill_px"] == k


@pytest.mark.parametrize("frac", [0.005, 0.02])
def test_budget_counts_eligible_only(frac):
    s = np.random.default_rng(3).random((30, 30))
    el = np.zeros_like(s, bool)
    el[:, :15] = True
    m, rep = R.exact_k_mask(s, el, frac)
    assert m.sum() == R.budget_k(frac, int(el.sum())) and not m[:, 15:].any()
