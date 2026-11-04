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


def two_system_scene():
    """Training unit whose truth holds system 1 (visible to the fit) and system 2 (held out from the fit)."""
    n = 60
    t = np.zeros((n, n), bool)
    t[5:55, 15] = True
    t[5:55, 45] = True
    sm = np.full((n, n), -1, np.int32)
    sm[:, 15][t[:, 15]] = 1
    sm[:, 45][t[:, 45]] = 2
    visible, hidden = t.copy(), np.zeros((n, n), bool)
    visible[:, 45], hidden[:, 45] = False, t[:, 45]
    vis = SimpleNamespace(fit_id="V1:2", visible=visible, hidden=hidden)
    unit = SimpleNamespace(scored_mask=lambda: np.ones((n, n), bool))
    vu = SimpleNamespace(key="V1:0", sl=(slice(0, n), slice(0, n)), eval_truth=lambda: t, known=np.zeros((n, n), bool),
                         sysmap=sm, unit=unit)
    return vis, vu


def test_synthetic_candidates_exclude_withheld_truth():
    vis, vu = two_system_scene()
    S = TL.synthetic_training_candidates(vis, vu, {"per_system": 6, "max_shift_px": 3, "max_rot_deg": 5,
                                                   "n_negatives": 40}, 0)
    assert S["geometry"]["systems"] == [1]                           # the held-out system is not a training system
    assert not (S["geometry"]["truth"] & vis.hidden).any()
    lab, T = S["syn_pos"]
    assert lab.max() > 0 and set(T.system_id) == {1}
    assert np.abs(np.nonzero(lab)[1] - 15).max() <= 10               # all perturbations stay around system 1


def test_synthetic_candidates_do_not_depend_on_withheld_truth():
    vis, vu = two_system_scene()
    a = TL.synthetic_training_candidates(vis, vu, {"per_system": 4, "n_negatives": 40}, 3)
    other = vu.sysmap.copy()
    t2 = vu.eval_truth().copy()
    t2[:, 45] = False
    t2[10:30, 40] = True                                             # the hidden fault moves, the fit cannot know
    hidden = np.zeros_like(t2)
    hidden[10:30, 40] = True
    vis2 = SimpleNamespace(fit_id="V1:2", visible=vis.visible, hidden=hidden)
    vu2 = SimpleNamespace(key=vu.key, sl=vu.sl, eval_truth=lambda: t2, known=vu.known, sysmap=other, unit=vu.unit)
    b = TL.synthetic_training_candidates(vis2, vu2, {"per_system": 4, "n_negatives": 40}, 3)
    np.testing.assert_array_equal(a["syn_pos"][0], b["syn_pos"][0])
    np.testing.assert_array_equal(a["syn_neg"], b["syn_neg"])


def test_overlap_with_held_out_geometry_fails_closed():
    vis, vu = two_system_scene()
    vis.visible[:, 45] = vu.eval_truth()[:, 45]                      # a broken visibility shows the hidden system
    with pytest.raises(LeakageError):
        TL.synthetic_training_geometry(vis, vu)
    with pytest.raises(LeakageError):
        TL.assert_no_heldout_geometry(np.ones((3, 3), bool), np.eye(3, dtype=bool), "V1:0", "x")


def test_candidate_provenance_record():
    p = TL.candidate_provenance("V1", 2, 0, "V1:2", "perturbed_visible_truth", True, [1, 4], {"w2": "x"})
    assert p["outer_fold"] == 2 and p["inner_fold"] == 0 and p["training_visibility"] == "V1:2"
    assert p["uses_mapped_geometry"] is True and p["training_systems"] == "[1, 4]"
    from gems_bench.g07.models_candidates import column_family

    assert all(column_family(k) == "meta" for k in p)                # provenance can never become a model feature


def test_system_groups_visible_only():
    v = np.zeros((20, 20), bool)
    v[:, 3] = True
    sm = np.full((20, 20), -1)
    sm[:, 3], sm[:, 15] = 4, 9
    g = TL.build_fault_system_groups(v, sm, 1.0)
    assert set(np.unique(g["system_map"])) == {-1, 4}                # hidden system 9 never appears
