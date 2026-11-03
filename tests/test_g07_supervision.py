"""G07 supervision: visible-only labels, distance targets, PU / improved weights, hard negatives, system groups."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gems_bench.g07 import datasets as DS
from gems_bench.g07 import training_labels as TL
from gems_bench.g07.contracts import LeakageError


def test_visible_labels_refuse_overlap():
    v = np.zeros((10, 10), bool)
    v[2, :] = True
    TL.visible_fault_labels(SimpleNamespace(fit_id="f", visible=v, hidden=np.zeros_like(v)))
    with pytest.raises(LeakageError):
        TL.visible_fault_labels(SimpleNamespace(fit_id="f", visible=v, hidden=v))


def test_distance_targets_monotone_and_bounded():
    v = np.zeros((40, 40), bool)
    v[:, 20] = True
    t = TL.fault_distance_targets(v, 0.3, 2.0)
    p = t["proximity"][10]
    assert p[20] == pytest.approx(1.0) and p[21] < p[20] and p[25] < p[21] and (p >= 0).all() and (p <= 1).all()
    assert t["dist_km"].max() <= 2.0


def test_improved_weights_balance_systems_and_grade_unlabeled():
    L = pd.DataFrame({"cls": [1] * 30 + [0, 0, 2], "system_id": [1] * 25 + [2] * 5 + [-1, -1, -1],
                      "dist_km": [0] * 30 + [0.4, 2.5, 4.0], "stratum": ["label"] * 30 + ["background"] * 3})
    w = DS.improved_weights(L, {"likely_background_km": 3.0, "unlabeled_min_weight": 0.1})
    assert w[:25].sum() == pytest.approx(w[25:30].sum())             # equal total per system
    assert w[30] < w[31] < w[32] + 1e-9                              # closer unlabelled pixels count less
    assert w[30:].sum() == pytest.approx(w[:30].sum(), rel=1e-5)


def test_hard_negatives_far_from_visible():
    v = np.zeros((50, 50), bool)
    v[:, 25] = True
    e = np.random.default_rng(0).random((50, 50))
    idx = TL.make_hard_negative_pool(e, v, np.ones_like(v), 1.0, 0.5, np.random.default_rng(1), 100)
    r, c = np.unravel_index(idx, v.shape)
    assert idx.size > 0 and (np.abs(c - 25) >= 10).all()


def test_system_groups_visible_only():
    v = np.zeros((20, 20), bool)
    v[:, 3] = True
    sm = np.full((20, 20), -1)
    sm[:, 3], sm[:, 15] = 4, 9
    g = TL.build_fault_system_groups(v, sm, 1.0)
    assert set(np.unique(g["system_map"])) == {-1, 4}                # hidden system 9 never appears
