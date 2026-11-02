"""G06 splits: training visibilities, fold isolation, discovery withholding, nested fits, inner folds (fake engine)."""
from types import SimpleNamespace

import numpy as np
import pytest

from gems_bench.folds import ROLE
from gems_bench.g06 import splits as SP
from gems_bench.g06.contracts import LeakageError

H, W = 60, 60


class FakeUnit:
    def __init__(self, task, zone, ev_full, truth_full):
        rr, cc = np.nonzero(ev_full)
        self.sl = (slice(rr.min(), rr.max() + 1), slice(cc.min(), cc.max() + 1))
        self.task, self.zone = task, zone
        self.ev, self.truth = ev_full[self.sl], truth_full[self.sl]

    def eval_truth(self):
        return self.truth & self.ev


def fake_engine():
    lab = np.zeros((H, W), bool)
    for c in (10, 30, 50):
        lab[5:55, c] = True
    lab[20, 22:28] = True
    fold_map = (np.arange(W)[None, :] // 20).repeat(H, 0).astype(np.int16)
    roles = np.zeros((3, H, W), np.int8)
    for k in range(3):
        val = fold_map == k
        near = np.zeros_like(val)
        for d in range(1, 6):
            near |= np.roll(val, d, 1) | np.roll(val, -d, 1)
        roles[k] = ROLE["train"]
        roles[k][near & ~val] = ROLE["buffer"]
        roles[k][val] = ROLE["val"]
    wh = {0: lab & (np.arange(W)[None, :] == 10), 1: lab & (np.arange(W)[None, :] == 50)}
    ez = {0: np.arange(W)[None, :].repeat(H, 0) < 20, 1: np.arange(W)[None, :].repeat(H, 0) >= 40}
    splits = SimpleNamespace(n_splits=2, withheld=lambda d: wh[d], eval_zone=lambda d: ez[d],
                             system_map=np.where(lab, 1, -1))
    folds = SimpleNamespace(fold_map=fold_map, roles=roles, n_folds=3)
    valid = np.ones((H, W), bool)
    eng = SimpleNamespace(labels=lambda: lab, valid=lambda: valid, ctx=SimpleNamespace(splits=splits, folds=folds),
                          shape=(H, W))
    units = {}
    for d in range(2):
        u = FakeUnit("disc_official", d, ez[d], wh[d])
        units[f"V1:{d}"] = SP.ValidationUnit("V1", d, u, u, (lab & ~wh[d])[u.sl], np.where(u.truth, 1, -1))
    for k in range(3):
        t = lab & (fold_map == k)
        u = FakeUnit("geo", k, fold_map == k, t)
        units[f"V3:{k}"] = SP.ValidationUnit("V3", k, u, u, (lab & (fold_map != k))[u.sl], np.where(u.truth, 1, -1))
    return eng, units, lab, wh, fold_map


VAL = {"v1_training_region": "all_visible", "v1_eval_buffer_km": 0.5}


def test_parse_and_list_fits():
    assert SP.parse_fit_id("V1:2") == ("V1", 2, None)
    assert SP.parse_fit_id("V3:0|2") == ("V3", 0, 2)
    assert SP.parse_fit_id("FULL") == ("FULL", None, None)
    f = SP.fit_ids(3, 3, ("V1", "V3"), True)
    assert "V1:0|1" in f and "V3:2|0" in f and "V1:1|1" not in f
    assert len(f) == 3 + 3 + 6 + 6


def test_discovery_withholding():
    eng, units, lab, wh, _ = fake_engine()
    v = SP.build_training_visibility(eng, "V1:0", VAL)
    assert not (v.visible & wh[0]).any() and (v.visible == (lab & ~wh[0])).all()
    assert v.predicts == ("V1:0",)
    rows = SP.verify_fold_isolation(eng, v, units)
    assert all(r["status"] == "PASS" for r in rows)
    strict = SP.build_training_visibility(eng, "V1:0", {**VAL, "v1_training_region": "outside_eval_zone"})
    assert not (strict.region & (np.arange(W)[None, :] < 20)).any()


def test_geographic_holdout_region_excludes_fold_and_buffer():
    eng, units, lab, _, fm = fake_engine()
    v = SP.build_training_visibility(eng, "V3:1", VAL)
    assert not (v.region & (fm == 1)).any()
    assert not (v.visible & (fm == 1)).any()
    assert not v.region[:, 15:45].any()                                  # 5-px buffer on both sides of fold 1
    assert v.predicts == ("V3:1", "V2:1")
    rows = SP.verify_fold_isolation(eng, v, units)
    assert all(r["status"] == "PASS" for r in rows if "V3:1" in r["check"] or "disjoint" in r["check"])


def test_nested_fit_hides_outer_and_inner_truth():
    eng, units, lab, wh, _ = fake_engine()
    v = SP.build_training_visibility(eng, "V1:0|1", VAL)
    assert not (v.visible & (wh[0] | wh[1])).any()
    assert v.predicts == ("V1:1",)
    rows = SP.verify_fold_isolation(eng, v, units, outer_unit=units["V1:0"])
    assert all(r["status"] == "PASS" for r in rows)


def test_isolation_detects_leak():
    eng, units, lab, wh, _ = fake_engine()
    v = SP.build_training_visibility(eng, "V1:0", VAL)
    v.visible = v.visible | wh[0]                                        # inject the held-out truth
    rows = SP.verify_fold_isolation(eng, v, units)
    assert any(r["status"] == "FAIL" for r in rows)
    with pytest.raises(LeakageError):
        SP.assert_isolated(rows)


def test_inner_folds_cover_region_and_balance():
    region = np.ones((200, 200), bool)
    visible = np.zeros_like(region)
    visible[:, ::10] = True
    m = SP.make_inner_folds(region, visible, 2.0, 3, 0)
    assert set(np.unique(m)) == {0, 1, 2}
    pos = [int((visible & (m == f)).sum()) for f in range(3)]
    assert max(pos) - min(pos) <= 0.25 * max(pos)
    b = SP.inner_buffer_mask(m, 0.5)
    assert b.any() and not b[m < 0].any()
    assert (SP.make_inner_folds(region, visible, 2.0, 3, 0) == m).all()  # deterministic
