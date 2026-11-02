"""G06 datasets: leakage-aware pool, per-fit labels and weights, reload fidelity, segment / connection rows."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gems_bench.g06 import datasets as DS
from gems_bench.g06 import sampling as SA
from gems_bench.g06.contracts import ProvenanceError, assert_label_free_columns
from gems_bench.g06.persistence import write_json, write_table
from gems_bench.g06.supervision import POSITIVE

SUP = {"ignore_km": 0.3, "likely_background_km": 1.0, "unlabeled_weight": 0.3, "likely_background_weight": 1.0}
SAMP = {"seed": 3, "block_km": 2.0, "label_spacing_px": 2, "per_system_cap": 50, "label_max": 1000,
        "background_total": 600, "background_min_per_block": 2, "evidence_quantile": 0.9, "evidence_total": 100,
        "region_balance_alpha": 0.5, "region_km": 3.0}


class FakeStore:
    def __init__(self, arrays):
        self.arrays, self.names = arrays, list(arrays)

    def load(self, names, window=None):
        h, w = next(iter(self.arrays.values())).shape
        r0, r1, c0, c1 = window or (0, h, 0, w)
        return np.stack([self.arrays[n][r0:r1, c0:c1] for n in names]).astype(np.float32)


def world():
    lab = np.zeros((80, 80), bool)
    lab[5:75, 20] = True                  # system 1 (visible)
    lab[5:75, 60] = True                  # system 2 (hidden in the fit below)
    sysm = np.where(lab, 0, -1)
    sysm[:, 60] = np.where(lab[:, 60], 1, -1)
    fp = np.ones_like(lab)
    ev = np.random.default_rng(0).random(lab.shape)
    return lab, sysm, fp, ev


def test_pool_is_deterministic_and_stratified():
    lab, sysm, fp, ev = world()
    a, b = SA.build_pool(lab, sysm, fp, ev, SAMP), SA.build_pool(lab, sysm, fp, ev, SAMP)
    pd.testing.assert_frame_equal(a, b)
    assert a.flat.is_unique
    assert lab.ravel()[a.flat[a.stratum == "label"]].all()
    assert set(a.stratum) <= {"label", "evidence", "background"}
    assert (a.incl_prob > 0).all() and (a.incl_prob <= 1).all()


def test_fit_labels_drop_hidden_label_rows_and_balance_systems():
    lab, sysm, fp, ev = world()
    pool = SA.build_pool(lab, sysm, fp, ev, SAMP)
    hidden = lab & (np.arange(80)[None, :] == 60)
    vis = SimpleNamespace(fit_id="V1:0", visible=lab & ~hidden, hidden=hidden, region=np.ones_like(lab))
    L = DS.label_pool_for_fit(pool, vis, fp, sysm, SUP, SAMP, {"block_km": 2.0, "n_folds": 2, "buffer_km": 0.2}, 0)
    J = L.merge(pool[["row_id", "row", "col"]], on="row_id")
    on_hidden = hidden[J.row, J.col]
    assert not (J.stratum[on_hidden] == "label").any()                 # oversampled hidden rows are dropped
    assert (J.cls[on_hidden] != POSITIVE).all()                        # hidden faults never positive
    assert (vis.visible[J.row[J.cls == POSITIVE], J.col[J.cls == POSITIVE]]).all()
    pos = L[L.cls == POSITIVE]
    assert pos.weight.sum() == pytest.approx(1.0, rel=1e-5)
    assert L.loc[L.cls != POSITIVE, "weight"].sum() == pytest.approx(1.0, rel=1e-4)   # neg_to_pos = 1
    assert (L.weight >= 0).all() and set(L.inner_fold.unique()) <= {-1, 0, 1}


def test_long_system_does_not_dominate():
    T = pd.DataFrame({"cls": [1] * 110, "stratum": ["label"] * 110, "system_id": [0] * 100 + [1] * 10,
                      "system_px": [1000] * 100 + [100] * 10, "row": 0, "col": 0})
    w = SA.compute_sample_weights(T, {"system_weight": "equal"})
    assert w[:100].sum() == pytest.approx(w[100:].sum())


def test_reload_fidelity_and_schema_guard(tmp_path):
    meta = pd.DataFrame({"row_id": [0, 1], "row": [1, 2], "col": [3, 4], "x": [0.5, 1.5], "y": [2.0, 3.0],
                         "stratum": ["label", "background"], "in_label": [True, False], "in_evidence": [False, False],
                         "in_background": [False, True], "block": [0, 0], "incl_prob": [0.5, 0.1]})
    X = pd.DataFrame({"row_id": [0, 1], "f1": np.array([0.1, np.nan], np.float32), "f2": np.array([1, 2], np.float32)})
    L = pd.DataFrame({"row_id": [0, 1], "fit_id": "V1:0", "cls": [1, 0], "target": [1, 0], "weight": [1.0, 1.0]})
    write_table(tmp_path / "datasets/pixels/pool_meta.parquet", meta)
    write_table(tmp_path / "datasets/pixels/pool_features.parquet", X)
    write_table(tmp_path / "datasets/pixels/labels/V1_0.parquet", L)
    write_json(tmp_path / "datasets/dataset_manifests/pixel.json", {"feature_schema_hash": "abc"})
    d = DS.load_dataset(tmp_path, "pixel", "V1:0", columns=["f1"], expected_schema_hash="abc")
    pd.testing.assert_frame_equal(d["X"], X[["row_id", "f1"]])
    pd.testing.assert_frame_equal(d["labels"], L)
    with pytest.raises(ProvenanceError):
        DS.load_dataset(tmp_path, "pixel", expected_schema_hash="other")


def test_segment_rows_are_label_free_and_geometric():
    lab = np.zeros((40, 40), np.int32)
    lab[10, 5:25] = 1
    lab[5:30, 30] = 2
    r, c = np.indices(lab.shape)
    store = FakeStore({"mag_h": (c * 1.0).astype(np.float32), "phys_rtp": np.where(r > 10, 10.0, 0.0).astype(
        np.float32), "oc": np.ones(lab.shape, np.float32), "os": np.zeros(lab.shape, np.float32)})
    cfg = {"geology_features": ["mag_h"], "std_features": ["mag_h"], "orientation_pairs": {"rtp": ["oc", "os"]},
           "jump_fields": ["phys_rtp"], "jump_offset_px": 2}
    known = {k: np.zeros(lab.shape, np.float32) for k in ("known_dist_km", "known_density", "tip_along_cos",
                                                          "known_net_density", "tip_dist_km", "known_orient_cos2",
                                                          "known_orient_sin2")}
    T = DS.segment_features(lab, store, (0, 40, 0, 40), cfg, known, {"w2_rank": np.full(lab.shape, 0.5)})
    assert list(T.label) == [1, 2]
    assert T.seg_n_px.tolist() == [20, 25]
    assert T.loc[0, "geo_mean_mag_h"] == pytest.approx(14.5)
    assert T.loc[0, "geo_jump_phys_rtp"] == pytest.approx(10.0)          # E-W segment on a N-S step
    assert T.loc[1, "geo_jump_phys_rtp"] < 1e-6
    # E-W segment, gradient axis east: segment parallel to the gradient -> strike agreement 0
    assert T.loc[0, "ori_rtp_agree"] == pytest.approx(0.0, abs=1e-6)
    assert T.loc[1, "ori_rtp_agree"] == pytest.approx(1.0, abs=1e-6)
    assert_label_free_columns([c for c in T.columns if c not in ("label", "row", "col")])
    assert {DS.segment_family(c) for c in T.columns if c.startswith(("geo_", "kn_", "ev_", "seg_", "ori_"))} == \
        {"geology", "context_known", "model_evidence", "geometry", "orientation"}


def test_patch_index_label_free():
    fp = np.zeros((300, 300), bool)
    fp[:, :200] = True
    P = DS.build_patch_index(fp, 64, 64, None, None, 0.5)
    assert (P.footprint_share >= 0.5).all() and (P.c0 + 64 <= 256).all()
    assert not any("label" in c or "truth" in c for c in P.columns)
