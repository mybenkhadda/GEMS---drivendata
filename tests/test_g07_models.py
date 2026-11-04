"""G07 models: directional augmentation, proximity regressor, GPU refusal, patch roles, U-Net training contract,
candidate features, utility."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gems_bench.g06.supervision import build_orientation_targets
from gems_bench.g07 import models_candidates as MC
from gems_bench.g07 import models_surface as MS
from gems_bench.g07.contracts import LeakageError


def line(angle_deg, n=48):
    """A straight line through the centre at ``angle_deg`` (from east, counter-clockwise, north up)."""
    m = np.zeros((n, n), bool)
    t = np.linspace(-18, 18, 200)
    a = np.radians(angle_deg)
    rr, cc = np.round(n / 2 - t * np.sin(a)).astype(int), np.round(n / 2 + t * np.cos(a)).astype(int)
    m[rr, cc] = True
    return m


@pytest.mark.parametrize("k,mirror", [(1, False), (2, False), (3, True), (0, True)])
def test_augmentation_keeps_orientation_consistent(k, mirror):
    y = line(30)
    o = build_orientation_targets(y, 2.0, 0.3)
    O = np.stack([np.nan_to_num(o["orient_cos2"]), np.nan_to_num(o["orient_sin2"]), np.isfinite(o["orient_cos2"])])
    x = np.zeros((1, *y.shape), np.float32)
    _, ya, _, oa = MS.augment(x, y.astype(np.float32), np.ones(y.shape, np.float32), O.astype(np.float32),
                              np.random.default_rng(0), k=k, mirror=mirror)
    ref = build_orientation_targets(ya > 0.5, 2.0, 0.3)                      # recompute on the transformed label
    m = (oa[2] > 0) & np.isfinite(ref["orient_cos2"])
    assert m.sum() > 10
    np.testing.assert_allclose(oa[0][m], ref["orient_cos2"][m], atol=0.15)
    np.testing.assert_allclose(oa[1][m], ref["orient_sin2"][m], atol=0.15)


def test_proximity_regressor_bounds():
    pytest.importorskip("lightgbm")
    from gems_bench.g07.models_subsurface import train_fault_distance_model

    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 3)).astype(np.float32)
    t = np.clip(np.exp(-np.abs(X[:, 0])), 0, 1).astype(np.float32)
    inner = (np.arange(600) % 3).astype(np.int8)
    m, _, info = train_fault_distance_model(X, t, np.ones(600, np.float32), {"n_estimators": 40, "num_leaves": 7,
                                                                             "early_stopping_rounds": 5}, 0, inner)
    p = m.predict_proba(X)
    assert p.shape == (600, 2) and (p >= 0).all() and (p <= 1).all() and info.get("best_iteration")


def test_unet_has_centreline_and_orientation_heads():
    torch = pytest.importorskip("torch")
    net = MS.build_surface_unet(4, base=4, depth=2)
    out = net(torch.zeros(2, 5, 32, 32))
    assert tuple(out.shape) == (2, 3, 32, 32)


def test_gpu_request_refuses_cpu_fallback(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(MS.GPURuntimeError):
        MS.validate_gpu_runtime(requested=True, require_cuda=True)
    with pytest.raises(MS.GPURuntimeError):
        MS.resolve_device(True)
    r = MS.validate_gpu_runtime(requested=False, require_cuda=False)
    assert r["selected_device"] == "cpu" and r["readiness"] == "CPU_ONLY" and r["cuda_available"] is False
    assert str(MS.resolve_device(False)) == "cpu"


def test_gpu_report_when_cuda_is_present():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA here")
    r = MS.validate_gpu_runtime(True, True)
    assert r["readiness"] == "READY" and r["selected_device"] == "cuda:0" and r["gpu_name"] and r["vram_total_gib"] > 0


CELLS, F = 4, 2


def patch_scene():
    """8 x 16 analysis cells cut into 8 patches; a visible fault crosses both inner folds, a hidden one is elsewhere."""
    H, W = 2 * CELLS, 4 * CELLS
    visible, hidden = np.zeros((H, W), bool), np.zeros((H, W), bool)
    visible[1, :] = True
    hidden[6, :] = True
    idx = pd.DataFrame([{"r0": r, "c0": c, "cells": CELLS, "dem_share": 1.0} for r in (0, 4) for c in (0, 4, 8, 12)])
    inner = np.where(np.indices((H, W))[1] < 8, 0, 1).astype(np.int8)
    vis = SimpleNamespace(fit_id="V1:0", visible=visible, hidden=hidden, region=np.ones((H, W), bool))
    return vis, idx, inner


UCFG = {"epochs": 2, "steps_per_epoch": 2, "batch": 2, "base_channels": 4, "depth": 2, "patience": 5,
        "min_region_share": 0.7, "pos_patch_share": 0.5, "positive_weight": 5.0, "unlabeled_weight": 0.3}


def test_patch_roles_keep_validation_cells_out_of_training():
    vis, idx, inner = patch_scene()
    I, facts = MS.select_surface_patches(idx, vis, inner, UCFG)
    assert facts["status"] == "READY" and facts["n_val"] and facts["n_train"]
    for r in I[I.role == "train"].itertuples():
        cells = inner[r.r0:r.r0 + CELLS, r.c0:r.c0 + CELLS]
        assert not (cells == facts["val_fold"]).any()
    assert facts["n_val_pos"] >= 1 and facts["n_train_pos"] >= 1


def test_straddling_patch_is_dropped():
    vis, idx, inner = patch_scene()
    inner = np.where(np.indices(inner.shape)[1] < 6, 0, 1).astype(np.int8)
    I, _ = MS.select_surface_patches(idx, vis, inner, UCFG)
    assert (I.role == "buffer").any()


def test_validation_fold_follows_the_covered_positives():
    """The DEM covers only inner fold 2: a fixed validation fold 0 would have left no validation patch at all."""
    vis, idx, _ = patch_scene()
    inner = np.where(np.indices(vis.visible.shape)[1] < 8, 1, 2).astype(np.int8)
    inner[0, 0] = 0
    I, facts = MS.select_surface_patches(idx.iloc[1:].reset_index(drop=True), vis, inner, UCFG)
    assert facts["status"] == "READY" and facts["val_fold"] in (1, 2)


@pytest.mark.parametrize("change,word", [("no_region", "training region"), ("no_positive", "visible fault")])
def test_missing_patches_are_reported_with_the_reason(change, word):
    vis, idx, inner = patch_scene()
    if change == "no_region":
        vis.region = np.zeros_like(vis.region)
    else:
        vis.visible = np.zeros_like(vis.visible)
    _, facts = MS.select_surface_patches(idx, vis, inner, UCFG)
    assert facts["status"] == "BLOCKED" and word in facts["reason"]


def test_prediction_targets_align_with_patch_cells():
    vis, idx, inner = patch_scene()
    classes = np.where(vis.visible, 1, 0).astype(np.int8)
    o = build_orientation_targets(vis.visible, 1.0, 0.1)
    T = MS.prepare_targets(vis.visible, classes, o, UCFG)
    y, w, oo, om = MS.patch_targets(T, 0, 4, CELLS, F)
    assert y.shape == w.shape == om.shape == (CELLS * F, CELLS * F) and oo.shape == (2, CELLS * F, CELLS * F)
    np.testing.assert_array_equal(y[2:4, :], 1.0)                           # analysis row 1 = input rows 2, 3
    np.testing.assert_array_equal(y[:2, :], 0.0)
    assert (w[2:4] == 5.0).all() and np.allclose(w[:2], 0.3)


def fit_small_unet(tmp_path, cfg=UCFG):
    pytest.importorskip("torch")
    vis, idx, inner = patch_scene()
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(len(idx), 3, CELLS * F, CELLS * F)).astype(np.float16)
    arr[0, 1, :2, :2] = np.nan                                              # no DEM here: no loss, validity channel 0
    mm = tmp_path / "patches.npy"
    np.save(mm, arr)
    classes = np.where(vis.visible, 1, 0).astype(np.int8)
    o = build_orientation_targets(vis.visible, 1.0, 0.1)
    out = MS.train_surface_unet(mm, idx, vis, classes, o, inner, cfg, F, False, 0, None, tmp_path / "ck.pt")
    return out, mm, idx, vis


def test_unet_trains_with_finite_losses_saves_contract_and_predicts(tmp_path):
    (model, scaler, info), mm, idx, vis = fit_small_unet(tmp_path)
    H = pd.DataFrame(info["history"])
    assert len(H) == 2 and np.isfinite(H.train_loss).all() and np.isfinite(H.inner_val_loss).all()
    assert {"lr", "gpu_allocated_mb", "gpu_reserved_mb", "seconds", "best_epoch", "fit_id"} <= set(H.columns)
    assert info["device"] == "cpu" and 0 <= info["best_epoch"] <= 1 and info["n_val"] > 0
    MS.save_surface_unet(model, scaler, info, tmp_path / "m", ["a", "b", "c"], UCFG, 50.0, CELLS)
    m2, s2, meta = MS.load_surface_unet(tmp_path / "m")
    assert meta["channels"] == ["a", "b", "c"] and meta["res_m"] == 50.0 and meta["out_channels"] == 3
    p1 = MS.predict_surface_tiles(model, scaler, mm, idx, vis.visible.shape, F, False, 2)
    p2 = MS.predict_surface_tiles(m2, s2, mm, idx, vis.visible.shape, F, False, 2)
    assert p1.shape == vis.visible.shape and np.nanmin(p1) >= 0 and np.nanmax(p1) <= 1 and np.isfinite(p1).any()
    np.testing.assert_allclose(p1, p2, atol=1e-5)                           # saved model = trained model


def test_unet_restores_the_best_epoch_and_resumes(tmp_path):
    fit_small_unet(tmp_path)
    (_, _, info2), *_ = fit_small_unet(tmp_path, {**UCFG, "epochs": 3})
    H = info2["history"]
    assert [h["epoch"] for h in H] == [0, 1, 2]                             # resumed, not restarted
    losses = [h["inner_val_loss"] for h in H]
    assert info2["best_epoch"] == int(np.argmin(losses)) and info2["best_inner_val_loss"] == min(losses)


def test_channel_mismatch_is_refused_at_inference(tmp_path):
    (model, scaler, info), mm, idx, vis = fit_small_unet(tmp_path)
    np.save(tmp_path / "wrong.npy", np.zeros((len(idx), 2, CELLS * F, CELLS * F), np.float16))
    with pytest.raises(ValueError):
        MS.predict_surface_tiles(model, scaler, tmp_path / "wrong.npy", idx, vis.visible.shape, F, False)


def test_hidden_visible_overlap_refused(tmp_path):
    pytest.importorskip("torch")
    vis, idx, inner = patch_scene()
    vis.hidden = vis.visible.copy()
    np.save(tmp_path / "p.npy", np.zeros((len(idx), 2, CELLS * F, CELLS * F), np.float16))
    with pytest.raises(LeakageError):
        MS.train_surface_unet(tmp_path / "p.npy", idx, vis, np.zeros_like(inner), build_orientation_targets(
            vis.visible, 1.0, 0.1), inner, UCFG, F, False, 0, None)


def test_geology_only_candidates_refuse_w2():
    T = pd.DataFrame({"geo_mean_dem_scarp_s10": [1.0], "ev_w2_rank_mean": [0.5], "ev_subsurface_rank_mean": [0.2],
                      "on_w2": [1.0], "seg_length_km": [1.0], "target": [1], "system_id": [3]})
    g = MC.candidate_feature_columns(T, w2_aware=False)
    assert "ev_w2_rank_mean" not in g and "on_w2" not in g and "ev_subsurface_rank_mean" in g
    assert "target" not in g and "system_id" not in g
    assert "ev_w2_rank_mean" in MC.candidate_feature_columns(T, w2_aware=True)
    with pytest.raises(LeakageError):
        MC.candidate_feature_columns(T, w2_aware=False, families=["model_evidence"])


def test_utility_rewards_new_relevant_area_only():
    lab = np.zeros((20, 20), np.int32)
    lab[2, 2:12] = 1                                                        # new area
    lab[10, 2:12] = 2                                                       # overlaps the base
    base = np.zeros((20, 20), bool)
    base[10, :] = True
    S = pd.DataFrame({"label": [1, 2], "p": [0.9, 0.9]})
    U = MC.estimate_candidate_utility(lab, S, base, np.ones((20, 20), bool), 0.25, 1.0).set_index("label")
    assert U.loc[1, "utility"] > 0 and U.loc[2, "n_new"] == 0
    S2 = pd.DataFrame({"label": [1, 2], "p": [0.1, 0.1]})
    assert MC.estimate_candidate_utility(lab, S2, base, np.ones((20, 20), bool), 0.25).set_index("label").loc[
        1, "utility"] < 0
