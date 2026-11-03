"""G07 candidate generation: label-independence, exact pool budgets, provenance, geometric augmentation."""
from types import SimpleNamespace

import numpy as np
import pytest

from gems_bench.g07 import candidate_generation as CG
from gems_bench.g07 import training_labels as TL

H = W = 60


class Unit:
    def __init__(self, truth):
        self.ev = np.ones((H, W), bool)
        self.truth = truth
        self.sl = (slice(0, H), slice(0, W))
        self.task, self.zone = "disc_official", 0

    def scored_mask(self):
        return self.ev.copy()

    def eval_truth(self):
        return self.truth & self.ev


class Store:
    def __init__(self, arrays):
        self.arrays, self.names = arrays, list(arrays)

    def load(self, names, window=None):
        r0, r1, c0, c1 = window or (0, H, 0, W)
        return np.stack([self.arrays[n][r0:r1, c0:c1] for n in names])


def scene(truth_col=30):
    r, c = np.indices((H, W))
    w2 = np.exp(-((c - 30) ** 2) / 8.0) + 0.01 * np.random.default_rng(0).random((H, W))
    mag = np.exp(-((r - 0.5 * c - 10) ** 2) / 8.0)
    truth = np.zeros((H, W), bool)
    truth[5:55, truth_col] = True
    eng = SimpleNamespace(template_valid=lambda: np.ones((H, W), bool), w2_rank=lambda b: w2,
                          provenance=lambda b: {"w2": "p6_stack_map:test", "cnn_b0": "oof"},
                          cnn=lambda b: (w2, w2))
    u = Unit(truth)
    vu = SimpleNamespace(unit=u, base=u, known=np.zeros((H, W), bool), sysmap=np.where(truth, 7, -1), key="V1:0",
                         scenario="V1", zone=0, sl=u.sl, eval_truth=u.eval_truth)
    cfg = {"pools": {"w2": {"frac": 0.03}, "magnetic": {"feature": "mag", "frac": 0.03}}, "max_len_px": 60,
           "min_px": 5}
    return eng, Store({"mag": mag}), vu, cfg


def test_pools_have_exact_budget_and_provenance():
    eng, st, vu, cfg = scene()
    out = CG.generate_unit_candidates(eng, st, vu, cfg, None, 0, W)
    k = int(np.ceil(0.03 * H * W))
    for name, m in out["pools"].items():
        assert int(m.sum()) == k, name
    assert out["lab"].max() > 0
    assert {"on_w2", "on_magnetic", "n_sources", "dominant_source"} <= set(out["dedup"].columns)
    assert out["provenance"]["magnetic"] == "store:mag"


def test_generation_ignores_truth():
    a = CG.generate_unit_candidates(*scene(30)[:3], scene(30)[3], None, 0, W)
    b = CG.generate_unit_candidates(*scene(10)[:3], scene(10)[3], None, 0, W)   # truth moved, scores identical
    np.testing.assert_array_equal(a["lab"], b["lab"])


def test_perturbed_candidates_stay_near_training_truth():
    truth = np.zeros((H, W), bool)
    truth[5:55, 30] = True
    lab, T = TL.perturbed_candidates(truth, np.where(truth, 3, -1), np.random.default_rng(1), 5, 3.0, 10.0)
    assert lab.max() == len(T) > 0 and (T.system_id == 3).all()
    rr, cc = np.nonzero(lab)
    assert np.abs(cc - 30).max() <= 3 + 50 * np.tan(np.radians(10)) + 1
    lab2, _ = TL.perturbed_candidates(truth, np.where(truth, 3, -1), np.random.default_rng(1), 5, 3.0, 10.0)
    np.testing.assert_array_equal(lab, lab2)                        # deterministic


def test_random_negatives_far_from_truth():
    from scipy import ndimage

    truth = np.zeros((H, W), bool)
    truth[:, 30] = True
    lab = TL.random_negative_candidates(np.ones((H, W), bool), truth, np.random.default_rng(0), 30, 10, 8.0)
    d = ndimage.distance_transform_edt(~truth)
    assert lab.max() > 0 and (d[lab > 0] >= 8.0).all()


@pytest.mark.parametrize("frac", [0.01, 0.05])
def test_surface_pool_respects_finite_mask(frac):
    eng, st, vu, cfg = scene()
    s = np.full((H, W), np.nan)
    s[:, :20] = np.random.default_rng(2).random((H, 20))
    pools = CG.generate_surface_candidates({"surface": s}, np.ones((H, W), bool), {"pools": {"surface": {"frac": frac}}},
                                           offset=(0, 0), width_full=W, seed=0)
    assert not pools["surface"][:, 20:].any()                       # never selects where the DEM is missing
