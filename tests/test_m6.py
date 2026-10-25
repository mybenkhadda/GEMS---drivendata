"""M6: audit, leakage-safe selection, equal-area comparison, postprocessing, gate, resume / atomic outputs.

Tiny synthetic rasters only (no competition data). A 60 x 60 grid at 100 m with three vertical fold stripes
separated by buffer columns (fold -1) stands in for the geographic design.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage

H = W = 60


# ---------------------------------------------------------------------------- tiny context
@dataclass
class _Grid:
    height: int = H
    width: int = W
    pixel_km_x: float = 0.1
    pixel_km_y: float = 0.1
    crs: object = None

    @property
    def pixel_km(self):
        return 0.1

    @property
    def shape(self):
        return (self.height, self.width)

    @property
    def transform(self):
        from affine import Affine

        return Affine(100.0, 0.0, 0.0, 0.0, -100.0, 6000.0)


@dataclass
class _Data:
    labels: np.ndarray
    valid: np.ndarray
    grid: _Grid = field(default_factory=_Grid)

    @property
    def shape(self):
        return self.labels.shape


@dataclass
class _Folds:
    fold_map: np.ndarray
    params: dict = field(default_factory=lambda: {"block_km": 2.0})

    @property
    def n_folds(self):
        return 3


class TinyCtx:
    def __init__(self, labels=None, seed=0):
        self.data = _Data(labels if labels is not None else _labels(), np.ones((H, W), bool))
        fm = np.full((H, W), -1, np.int16)
        fm[:, 0:18], fm[:, 21:39], fm[:, 42:60] = 0, 1, 2
        self.folds = _Folds(fm)
        self._maps = {}
        self.seg = self.seg_tab = None

    @property
    def fold_ids(self):
        return [0, 1, 2]

    def dist_to_labels_km(self):
        return ndimage.distance_transform_edt(~self.data.labels, sampling=(0.1, 0.1)).astype(np.float32)

    def oof(self, name):
        return self._maps[name]

    @staticmethod
    def disc_key(m, d):
        return f"_disc:{m}:{d}"


def _labels():
    lab = np.zeros((H, W), bool)
    for c in (5, 12, 27, 33, 48, 55):
        lab[5:50, c] = True                        # N-S traces in every fold
    lab[30, 2:16] = True                           # an E-W trace in fold 0
    return lab


def _score(lab, seed, noise=0.6):
    rng = np.random.default_rng(seed)
    d = ndimage.distance_transform_edt(~lab)
    return np.clip(np.exp(-d / 2.0) + noise * rng.random(lab.shape), 0, None).astype(np.float32) / (1 + noise)


def _perturb_fold(lab, fold_cols):
    """Different labels inside one fold only (the held-out fold)."""
    out = lab.copy()
    out[:, fold_cols] = False
    out[10:40, fold_cols.start + 3] = True
    return out


# ---------------------------------------------------------------------------- 1. provenance compatibility
def test_compatibility_blocks_different_folds_and_protocol():
    from gems_bench.evaluation.comparative import compatibility_table

    inv = pd.DataFrame([
        {"model": "A", "status": "COMPLETE", "folds_fp": "f1", "discovery_fp": "d1", "protocol_hash": "p1",
         "feature_version": "x", "git_commit": "c1", "seed": 2026},
        {"model": "B", "status": "COMPLETE", "folds_fp": "f1", "discovery_fp": "d1", "protocol_hash": "p1",
         "feature_version": "y", "git_commit": "c2", "seed": 2026},
        {"model": "C", "status": "COMPLETE", "folds_fp": "f2", "discovery_fp": "d1", "protocol_hash": "p2",
         "feature_version": "x", "git_commit": "c1", "seed": 2026}])
    t = compatibility_table(inv).set_index(["model_a", "model_b"])
    assert bool(t.loc[("A", "B"), "comparable"])                 # features / commit differ: informative only
    assert not bool(t.loc[("A", "B"), "same_feature_version"])
    assert not bool(t.loc[("A", "C"), "comparable"])
    assert set(t.loc[("A", "C"), "blocking"].split(",")) == {"folds_fp", "protocol_hash"}


# ---------------------------------------------------------------------------- 2. grid alignment + missing maps
def _tif(path: Path, arr: np.ndarray, x0: float = 0.0):
    import rasterio
    from affine import Affine

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype="float32",
                       transform=Affine(100.0, 0.0, x0, 0.0, -100.0, 6000.0), crs="EPSG:32611") as dst:
        dst.write(arr.astype("float32"), 1)


def _experiment(root: Path, stem="b0_reference", model="P5_B0_gated", shape=(H, W), n_disc=2, folds_fp="F",
                with_oof=True):
    d = root / "experiments" / f"{stem}__abcd1234"
    maps = {"oof": "predictions/oof.tif" if with_oof else None,
            "disc": {str(k): f"predictions/disc{k}.tif" for k in range(n_disc)}}
    if with_oof:
        _tif(d / "predictions" / "oof.tif", np.zeros(shape))
    for k in range(n_disc):
        _tif(d / "predictions" / f"disc{k}.tif", np.zeros(shape))
    m = {"experiment_id": d.name, "name": stem, "config_hash": "h", "git_commit": "c", "maps": {model: maps},
         "units": {f"cv_fold{k}": {"status": "done"} for k in range(3)},
         "stages": {s: {"status": "done"} for s in ("prepare", "train", "evaluate", "export")},
         "protocol": {"folds_fp": folds_fp, "discovery_fp": "D", "protocol_hash": "P"}}
    (d / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    (d / "config.yaml").write_text(f"name: {stem}\n", encoding="utf-8")
    return d


def _ref_grid():
    return {"height": H, "width": W, "crs": "EPSG:32611", "transform": (100.0, 0.0, 0.0, 0.0, -100.0, 6000.0)}


def test_inventory_statuses_grid_and_fold_consistency(tmp_path):
    from gems_bench.evaluation.comparative import ModelSpec, M6ArtifactError, artifact_inventory, require_complete

    _experiment(tmp_path, "b0_reference", "P5_B0_gated")
    _experiment(tmp_path, "b1_bg_weight", "P5_B1_bgw", shape=(H, W + 4))                 # other grid
    _experiment(tmp_path, "b2_nnpu_p010", "P5_B2", n_disc=1)                             # discovery split 1 missing
    _experiment(tmp_path, "c1_geometry_heads", "P5_C1", folds_fp="OTHER")                # other fold files
    _experiment(tmp_path, "e1_gsn_parity", "P5_E1", with_oof=False)                      # no OOF
    specs = [ModelSpec("b0_reference", "P5_B0_gated"), ModelSpec("b1_bg_weight", "P5_B1_bgw"),
             ModelSpec("b2_nnpu_p010", "P5_B2"), ModelSpec("c1_geometry_heads", "P5_C1"),
             ModelSpec("e1_gsn_parity", "P5_E1"), ModelSpec("nothing_here", "P5_X")]
    tabs = artifact_inventory(tmp_path, specs, ref_grid=_ref_grid(), n_folds=3, n_splits=2,
                              current_protocol={"folds_fp": "F", "discovery_fp": "D"})
    st = tabs["inventory"].set_index("model").status
    assert st["P5_B0_gated"] == "COMPLETE"
    assert st["P5_B1_bgw"] == "INCOMPATIBLE"
    assert st["P5_B2"] == "PARTIAL"
    assert st["P5_C1"] == "INCOMPATIBLE"
    assert st["P5_E1"] == "MISSING" and st["P5_X"] == "MISSING"
    miss = tabs["missing"]
    assert {"P5_E1", "P5_X", "P5_B2"} <= set(miss.model)
    require_complete(tabs["inventory"], ["P5_B0_gated"], n_splits=2)
    with pytest.raises(M6ArtifactError) as e:
        require_complete(tabs["inventory"], ["P5_B0_gated", "P5_B2", "P5_E1"], n_splits=2)
    assert "P5_B2" in str(e.value) and "P5_E1" in str(e.value)


def test_prediction_cache_missing_and_memmap(tmp_path):
    from gems_bench.evaluation.comparative import M6ArtifactError
    from gems_bench.pipeline.m6 import PredictionCache

    c = PredictionCache(tmp_path / "cache")
    with pytest.raises(M6ArtifactError):
        c.load(tmp_path / "absent.tif")
    arr = np.arange(H * W, dtype=np.float32).reshape(H, W)
    _tif(tmp_path / "p.tif", arr)
    a = c.load(tmp_path / "p.tif")
    assert isinstance(a, np.memmap) and np.array_equal(np.asarray(a), arr)
    b = c.load(tmp_path / "p.tif")                                   # cached copy reused
    assert np.array_equal(np.asarray(a), np.asarray(b)) and len(list((tmp_path / "cache").glob("*.npy"))) == 1


# ---------------------------------------------------------------------------- 3. exact block decomposition
def test_block_components_sum_to_binary_curve():
    from gems_bench.evaluation.comparative import block_components, block_ids
    from gems_bench.p3.curves import BinaryCurve

    ctx = TinyCtx()
    p = _score(ctx.data.labels, 1)
    ev = ctx.folds.fold_map == 1
    c = BinaryCurve(p, ctx.data.labels, ev)
    thr = float(c.threshold_for_area([0.1])[0])
    ref = c.at([thr]).iloc[0]
    b = block_components(p, ctx.data.labels, ev, thr, block_ids((H, W), 10))
    assert np.isclose(b.tp_w.sum(), ref.tp_w) and np.isclose(b.fp_w.sum(), ref.fp_w) and np.isclose(b.fn_w.sum(), ref.fn_w)


def test_block_bootstrap_is_paired_and_deterministic():
    from gems_bench.evaluation.comparative import block_bootstrap

    rows = []
    for z in range(3):
        for b in range(5):
            rows.append({"model": "a", "zone": z, "block": b, "tp_w": 5.0, "fp_w": 10.0, "fn_w": 5.0})
            rows.append({"model": "b", "zone": z, "block": b, "tp_w": 4.0, "fp_w": 10.0, "fn_w": 6.0})
    comp = pd.DataFrame(rows)
    r1 = block_bootstrap(comp, "a", "b", n_boot=200, seed=3)
    r2 = block_bootstrap(comp, "a", "b", n_boot=200, seed=3)
    assert r1 == r2 and r1["p_delta_positive"] == 1.0 and r1["observed_delta"] > 0


def test_paired_deltas_direction_and_effect_size():
    from gems_bench.evaluation.comparative import paired_deltas

    t = pd.DataFrame({"model": ["r"] * 3 + ["c"] * 3, "fold": [0, 1, 2] * 2,
                      "dti_heldout": [0.10, 0.20, 0.30, 0.12, 0.21, 0.29]})
    per, s = paired_deltas(t, "r", "fold")
    assert np.allclose(per.sort_values("fold").delta, [0.02, 0.01, -0.01])
    r = s.iloc[0]
    assert r.units_improved == 2 and r.sign_pattern == "++-" and np.isclose(r.mean_delta, 0.02 / 3)


# ---------------------------------------------------------------------------- 4. leakage-safe calibration
def test_calibration_choice_for_a_fold_ignores_that_folds_labels():
    from gems_bench.evaluation.calibration import calibration_summary, calibration_table, threshold_transfer

    fams = ("raw", "rank", "linear", "hard_threshold")
    a, b = TinyCtx(), TinyCtx(labels=_perturb_fold(_labels(), slice(0, 18)))
    p = _score(a.data.labels, 2)
    _, per_a = calibration_summary(calibration_table(a, "m", p, families=fams))
    _, per_b = calibration_summary(calibration_table(b, "m", p, families=fams))
    ka = per_a[per_a.zone == 0].set_index("family").param
    kb = per_b[per_b.zone == 0].set_index("family").param
    assert ka.equals(kb)                                             # fold 0's choice never reads fold 0's labels
    assert (per_a.selection_zones[per_a.zone == 0] == "1,2").all() and not per_a.uses_heldout_labels.any()
    ta, tb = threshold_transfer(a, "m", p), threshold_transfer(b, "m", p)
    assert ta[ta.heldout_fold == 0].parameter.tolist() == tb[tb.heldout_fold == 0].parameter.tolist()
    assert all("0" not in r.split(",") for r in ta[ta.heldout_fold == 0].selection_zones)


def test_rank_normalisation_does_not_change_binary_masks():
    from gems_bench.evaluation.calibration import calibration_table, invariance_check

    ctx = TinyCtx()
    p = np.round(_score(ctx.data.labels, 4), 6)
    inv = invariance_check(calibration_table(ctx, "m", p, families=("hard_threshold",)))
    assert len(inv) == 1 and inv.max_abs_diff.iloc[0] < 1e-6


# ---------------------------------------------------------------------------- 5. nested ensemble weights
def test_nested_weight_selection_excludes_heldout_labels():
    from gems_bench.discovery.nested_selection import nested_geo_selection, normalised_members

    maps_a = {"B0": _score(_labels(), 5), "B1": _score(_labels(), 6)}
    a, b = TinyCtx(), TinyCtx(labels=_perturb_fold(_labels(), slice(21, 39)))
    sa = nested_geo_selection(a, normalised_members(a, maps_a, "rank"), ["B0", "B1"], refine=True)
    sb = nested_geo_selection(b, normalised_members(b, maps_a, "rank"), ["B0", "B1"], refine=True)
    ra, rb = sa["per_fold"].set_index("zone"), sb["per_fold"].set_index("zone")
    assert ra.loc[1, "w"] == rb.loc[1, "w"] and ra.loc[1, "target_frac"] == rb.loc[1, "target_frac"]
    assert not sa["audit"].heldout_labels_used_for_selection.any()
    assert set(sa["audit"].set_index("heldout_zone").loc[1, "selection_zones"].split(",")) == {"0", "2"}


def test_weight_grid_contains_single_members_and_audit_rejects_leakage():
    from gems_bench.discovery.nested_selection import LeakageError, index_audit, weight_grid

    g = weight_grid(["B0", "B1"])
    assert {"B0": 1.0, "B1": 0.0} in g and {"B0": 0.0, "B1": 1.0} in g and len(g) == 5
    assert all(abs(sum(w.values()) - 1) < 1e-9 for w in weight_grid(["a", "b", "c"], simplex_step=0.25))
    with pytest.raises(LeakageError):
        index_audit(pd.DataFrame([{"zone": 0, "selection_zones": "0,1", "w": "{}", "target_frac": 0.05}]), "geo", [0, 1, 2])


def test_stitched_ensemble_uses_each_folds_own_weights():
    from gems_bench.discovery.nested_selection import stitched

    ctx = TinyCtx()
    U = {"a": np.full((H, W), 0.2, np.float32), "b": np.full((H, W), 0.8, np.float32)}
    per = pd.DataFrame({"zone": [0, 1, 2], "w": [json.dumps({"a": 1.0, "b": 0.0}), json.dumps({"a": 0.0, "b": 1.0}),
                                                   json.dumps({"a": 0.5, "b": 0.5})]})
    e = stitched(ctx, U, per)
    assert np.allclose(e[:, 0:18], 0.2) and np.allclose(e[:, 21:39], 0.8) and np.allclose(e[:, 42:60], 0.5)
    assert np.isnan(e[:, 19]).all()                                  # buffer columns stay unscored


# ---------------------------------------------------------------------------- 6. equal-area comparison
def test_pair_statistics_compare_identical_areas():
    from gems_bench.evaluation.complementarity import pair_statistics

    ctx = TinyCtx()
    maps = {"a": _score(ctx.data.labels, 7), "b": _score(ctx.data.labels, 8)}
    t = pair_statistics(ctx, maps, "a", "b", budgets=(0.05, 0.10), sample=2000)
    for r in t.itertuples():
        assert abs(r.area_a_km2 - r.area_b_km2) <= 0.01 * 3 + 1e-9    # same per-fold quantile cut (ties aside)
        assert 0.0 <= r.iou <= 1.0
    same = pair_statistics(ctx, {"a": maps["a"], "b": maps["a"].copy()}, "a", "b", budgets=(0.05,), sample=2000)
    assert np.allclose(same.iou, 1.0) and np.allclose(same.a_only_km2, 0.0)


# ---------------------------------------------------------------------------- 7. topology operations
def test_component_filtering_is_deterministic():
    from gems_bench.postprocess.topology import remove_small

    m = np.zeros((20, 20), bool)
    m[2, 2] = m[10, 3:12] = True
    m[15:17, 15:17] = True
    a, b = remove_small(m, 4), remove_small(m, 4)
    assert np.array_equal(a, b) and not a[2, 2] and a[10, 3:12].all() and a[15:17, 15:17].all()


def test_endpoint_linking_respects_direction():
    from gems_bench.postprocess.topology import link_endpoints

    m = np.zeros((30, 40), bool)
    m[15, 2:15] = True
    m[15, 19:35] = True                                              # collinear, gap of 4 px
    out = link_endpoints(m, max_gap_px=6, max_angle_deg=30)
    assert out[15, 15:19].all()
    p = np.zeros((30, 40), bool)
    p[15, 2:15] = True
    p[17:29, 17] = True                                              # perpendicular piece
    assert np.array_equal(link_endpoints(p, max_gap_px=6, max_angle_deg=30), p)


def test_min_length_skeleton_and_orientation_filter():
    from gems_bench.postprocess.topology import min_length, orientation_filter, skeleton_width

    m = np.zeros((40, 40), bool)
    m[5, 2:30] = True                                                # 2.8 km line
    m[20, 2:6] = True                                                # 0.4 km piece
    m[30:37, 25:32] = True                                           # 7 x 7 blob
    ml = min_length(m, 1.0, 0.1)
    assert ml[5, 2:30].all() and not ml[20, 2:6].any()
    of = orientation_filter(m, 0.5, min_px=10)
    assert of[5, 2:30].all() and not of[30:37, 25:32].any() and of[20, 2:6].all()   # tiny piece left alone
    sw = skeleton_width(m[:10], 0)
    assert sw.sum() <= m[:10].sum()


def test_operations_never_cross_fold_zones():
    from gems_bench.postprocess.topology import apply_per_zone

    ctx = TinyCtx()
    m = np.zeros((H, W), bool)
    m[30, 10:18] = True
    m[30, 21:28] = True                                              # pieces on both sides of a buffer
    out = apply_per_zone(m, ctx.folds.fold_map, ctx.data.valid, {0: [("close_gaps", {"radius_px": 3})],
                                                                   1: [("close_gaps", {"radius_px": 3})]}, 0.1)
    assert not out[:, 18:21].any()


def test_postprocessing_selection_ignores_heldout_labels():
    from gems_bench.postprocess.topology import ablation

    grid = {"remove_small": [{"min_px": 2}, {"min_px": 6}], "close_gaps": [{"radius_px": 1}]}
    p = _score(_labels(), 9)
    a, b = TinyCtx(), TinyCtx(labels=_perturb_fold(_labels(), slice(42, 60)))
    sa = ablation(a, "m", p, fracs=(0.05, 0.10), grid=grid)["selection"].set_index(["operation", "heldout_zone"])
    sb = ablation(b, "m", p, fracs=(0.05, 0.10), grid=grid)["selection"].set_index(["operation", "heldout_zone"])
    for op in ("identity", "remove_small", "close_gaps"):
        assert sa.loc[(op, 2), "chosen_param"] == sb.loc[(op, 2), "chosen_param"]
        assert sa.loc[(op, 2), "chosen_frac"] == sb.loc[(op, 2), "chosen_frac"]
    assert not sa.uses_heldout_labels.any()


# ---------------------------------------------------------------------------- 8. promotion guardrails
def _sc(cand: dict, ctrl: dict, splits_better=True):
    base = {"geo_dti": 0.16, "geo_fold_min": 0.15, "disc_dti_excl": 0.20, "disc_fold_min_excl": 0.18,
            "disc_dti_retained_fp": 0.09, "short_recall_w": 0.18, "geo_far_share_at_match": 0.3, "geo_frac": 0.05,
            "disc_frac_excl": 0.05}
    score = pd.DataFrame([{"model": "C", **base, **cand}, {"model": "K", **base, **ctrl}])
    ps = pd.DataFrame([{"model": m, "policy": "excl", "zone": z, "dti_heldout": v + (0.01 if (m == "C" and splits_better) else 0)}
                       for m in ("C", "K") for z, v in enumerate((0.19, 0.20, 0.21))])
    uniq = pd.DataFrame([{"model": "C", "ref": "K", "policy": "excl", "n_unique_systems": 2}])
    return {"score": score, "disc_per_split": ps, "unique": uniq}


EV = {"provenance_ok": True, "members_full": True, "leakage_ok": True, "boot_p": {"K": 0.9}, "cost": 2.0}


def test_gate_promotes_only_with_every_criterion():
    from gems_bench.evaluation.selection_gate import gate_one

    sc = _sc({"disc_dti_excl": 0.21, "geo_dti": 0.165, "disc_dti_retained_fp": 0.091, "disc_fold_min_excl": 0.19},
             {})
    assert gate_one(sc, "C", "K", rules={}, gate={}, evidence=EV)["verdict"] == "PROMOTE"
    worse_short = _sc({"disc_dti_excl": 0.21, "short_recall_w": 0.15}, {})
    r = gate_one(worse_short, "C", "K", rules={}, gate={}, evidence=EV)
    assert r["verdict"] == "REVIEW" and r["criteria"]["5_short_fault_recall"] == "FAIL"
    big_area = _sc({"disc_dti_excl": 0.21, "geo_frac": 0.10}, {})
    assert gate_one(big_area, "C", "K", rules={}, gate={}, evidence=EV)["criteria"]["6_area_calibration"] == "FAIL"


def test_gate_rejects_leakage_and_non_full_and_flags_missing_evidence():
    from gems_bench.evaluation.selection_gate import gate, gate_one

    sc = _sc({"disc_dti_excl": 0.21}, {})
    assert gate_one(sc, "C", "K", rules={}, gate={}, evidence={**EV, "leakage_ok": False})["verdict"] == "REJECT"
    assert gate_one(sc, "C", "K", rules={}, gate={}, evidence={**EV, "members_full": False})["verdict"] == "REJECT"
    nan = _sc({"disc_dti_retained_fp": np.nan}, {})
    assert gate_one(nan, "C", "K", rules={}, gate={}, evidence=EV)["verdict"] == "INSUFFICIENT_EVIDENCE"
    regress = _sc({"disc_dti_excl": 0.19, "geo_dti": 0.15}, {})
    assert gate_one(regress, "C", "K", rules={}, gate={}, evidence=EV)["verdict"] == "REJECT"
    table, rec = gate(regress, ["C"], ["K"], rules={}, gate_cfg={}, evidence={"C": EV})
    assert rec["recommended"] is None and rec["verdict"] == "NO_CANDIDATE_PASSES"
    assert table[table.control == "ALL"].verdict.iloc[0] == "REJECT"


# ---------------------------------------------------------------------------- 9. config, resume, atomic outputs
def test_m6_config_validation():
    from gems_bench.pipeline.m6 import M6ConfigError, load_m6_config

    cfg = load_m6_config({"ensemble": {"refine": False}, "reference_observations": {"NEW": {"geo_dti": 0.1}}})
    assert cfg["ensemble"]["refine"] is False and cfg["ensemble"]["weights"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    for bad in ({"train_models": True}, {"mode": "TRAIN"}, {"ensembel": {}}, {"primary_pair": ["b0_reference"]},
                {"postprocessing": {"grid": {"blur": [{}]}}}, {"save_rasters": "some"}):
        with pytest.raises(M6ConfigError):
            load_m6_config(bad)


def test_workspace_units_resume_and_invalidate(tmp_path):
    from gems_bench.pipeline.m6 import Workspace

    ws = Workspace(tmp_path / "m6", {"save_csv": True, "save_parquet": False})
    assert not ws.valid("u", "fp1")
    ws.begin("u")
    ws.save_table(pd.DataFrame({"a": [1]}), "t")
    ws.finish("u", "fp1")
    assert ws.valid("u", "fp1") and not ws.valid("u", "fp2")        # changed inputs -> recompute
    (tmp_path / "m6" / "tables" / "t.csv").unlink()
    assert not ws.valid("u", "fp1")                                  # an output disappeared -> recompute
    ws2 = Workspace(tmp_path / "m6", {"overwrite_existing": True})
    ws2.begin("v")
    ws2.finish("v", "x")
    assert not ws2.valid("v", "x")                                   # overwrite_existing forces recomputation


def test_atomic_table_write_keeps_previous_file_on_failure(tmp_path):
    from gems_bench.artifacts.atomic import atomic_save
    from gems_bench.pipeline.m6 import Workspace

    ws = Workspace(tmp_path / "m6", {"save_parquet": False})
    p = ws.save_table(pd.DataFrame({"a": [1, 2]}), "t")

    def boom(t):
        Path(t).write_text("partial", encoding="utf-8")
        raise RuntimeError("disconnect")

    with pytest.raises(RuntimeError):
        atomic_save(p, boom)
    assert pd.read_csv(p).a.tolist() == [1, 2]
    assert not [f for f in p.parent.iterdir() if ".tmp-" in f.name]


def test_reference_discrepancies_report_without_reconciling():
    from gems_bench.evaluation.comparative import reference_discrepancies

    obs = {"A": {"geo_dti": 0.16456, "short_recall": 0.19}}
    persisted = pd.DataFrame([{"model": "A", "geo_dti_area": 0.16456, "short_recall_w": 0.20}])
    t = reference_discrepancies(obs, persisted, None).set_index("metric")
    assert t.loc["geo_dti_area", "status"] == "MATCH" and t.loc["short_recall_w", "status"] == "DIFFERENT"
    assert np.isclose(t.loc["short_recall_w", "persisted"], 0.20)
