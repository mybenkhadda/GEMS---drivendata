import numpy as np
import pandas as pd
import pytest

from gems_bench.g11 import fault_labels as FL
from gems_bench.g11 import spatial_splits as SS


@pytest.fixture(scope="module")
def split(synth, grid):
    mask = FL.label_raster(synth["raw"] / "existing_faults.tif", grid)
    return SS.make_splits(grid.valid, mask, grid, synth["cfg"]), mask


def test_embargo_width_from_components(cfg):
    cfg["spatial_splits"]["embargo_m"] = None
    w, parts = SS.embargo_width_m(cfg)
    assert w == max(parts.values()) and "dem_halo_m" in parts and "candidate_link_radius_m" in parts


def test_disjoint_and_embargo(split, grid):
    res, _ = split
    assert not SS.check_disjoint(res)
    assert not SS.check_embargo(res, grid)
    assert res.mask("embargo").any()
    assert ((res.code > 0) == grid.valid).all()                    # every valid pixel is in exactly one state


def test_fault_system_not_split(split, grid):
    res, mask = split
    codes = set(np.unique(res.code[mask])) - {0, SS.CODES["embargo"]}
    assert len(codes) == 1                                         # the single fault system lives in one split


def test_patch_leakage_detected(split):
    res, _ = split
    ok_rows = []
    tr = np.argwhere(res.mask("train"))
    r, c = tr[len(tr) // 2]
    ok_rows.append({"row0": max(r - 2, 0), "col0": max(c - 2, 0), "size": 3, "split": "train"})
    assert not SS.check_patch_leakage(pd.DataFrame(ok_rows), res)
    bad = pd.DataFrame([{"row0": 0, "col0": 0, "size": res.code.shape[0], "split": "train"}])
    assert SS.check_patch_leakage(bad, res)


def test_segment_groups_and_predictors():
    seg = pd.DataFrame({"group": [1, 1, 2], "split": ["train", "test", "train"]})
    assert SS.check_segment_groups(seg)
    assert SS.check_predictor_names(["a", "Y_DISTANCE"], ["Y_DISTANCE"])


def test_normalizer_fit_only_on_train():
    X = np.random.default_rng(0).normal(size=(100, 3))
    with pytest.raises(SS.LeakageError):
        SS.TrainOnlyNormalizer().fit(X, np.full(100, SS.CODES["test"]))
    n = SS.TrainOnlyNormalizer().fit(X, np.full(100, SS.CODES["train"]))
    assert np.allclose(np.median(n.transform(X), 0), 0, atol=1e-9)


def test_oof_provenance():
    assert SS.check_oof_provenance({"path": "p", "alignment_confidence": "UNLOCATED"}, "test")
    assert SS.check_oof_provenance({"path": "p", "alignment_confidence": "DOCUMENTED", "split": "train"}, "test")
    assert not SS.check_oof_provenance({"path": "p", "alignment_confidence": "DOCUMENTED", "split": "oof"}, "validation")


def test_leakage_table(split, grid):
    res, _ = split
    t = SS.run_leakage_checks(res, grid, predictor_names=["f1"], label_derived=["Y_DISTANCE"])
    assert t.passed.all()
