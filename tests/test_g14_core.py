"""G14 tests: data contract, splits, multiscale geometry, augmentation, normalisation, ensemble constraints, schema.

NEVER EXECUTED by the authoring session. Run: ``pytest -q tests/test_g14_core.py tests/test_g14_model.py``.
Synthetic fixtures only (``gems_bench.g13.synthetic``); torch-dependent tests live in test_g14_model.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from gems_bench.g14.config import G14Config
from gems_bench.g14.experiments import REGISTRY, resolve


# =============================================================================== fixtures
@pytest.fixture(scope="module")
def data(tmp_path_factory):
    from gems_bench.g13.contracts import load
    from gems_bench.g13.synthetic import make

    root = tmp_path_factory.mktemp("g14data")
    return load(make(root), tmp_path_factory.mktemp("g14cache"))


def small_cfg(**kw) -> G14Config:
    """Geometry scaled to the 400 px synthetic grid: local 32 px, regional 64 px @ x2, global 64 px @ x8."""
    base = {"data": {"local": {"resolution_m": 100.0, "size_px": 32},
                     "regional": {"resolution_m": 200.0, "size_px": 64},
                     "global": {"resolution_m": 800.0, "size_px": 64},
                     "samples_per_epoch": 16, "val_samples": 8, "min_train_frac": 0.3,
                     "g13_features_dir": "/nonexistent", "known_context": True},
            "model": {"local_widths": (8, 16, 16, 32), "swin_embed": 16, "swin_depths": (1, 1, 1, 1),
                      "swin_heads": (1, 2, 2, 4), "swin_window": 4, "swin_patch": 4, "global_dim": 32,
                      "global_tokens": 4, "global_layers": 1, "modality_dim": 8},
            "split": {"backend": "g13"}}
    for k, v in kw.items():
        base.setdefault(k, {})
        if isinstance(v, dict):
            base[k].update(v)
        else:
            base[k] = v
    return G14Config.from_dict(base)


@pytest.fixture(scope="module")
def system(data):
    from gems_bench.g13.splits import build_splits
    from gems_bench.g13.synthetic import small_config
    from gems_bench.g14.splits import UnitSystem

    sp = build_splits(data, small_config("/tmp/u", "/tmp/u"))
    return UnitSystem("g13", data, g13_splits=sp, embargo_m=2000.0)


@pytest.fixture(scope="module")
def stack(data, tmp_path_factory):
    from gems_bench.g14.contract import build_stack

    return build_stack(data, small_cfg(), tmp_path_factory.mktemp("g14stack"), None, log=lambda *a: None)


# =============================================================================== config / registry
def test_config_validation_and_hash():
    c = G14Config()
    assert c.run_mode == "AUDIT_ONLY" and not c.train_enabled and not c.submission_enabled
    with pytest.raises(KeyError, match="did you mean"):
        G14Config.from_dict({"modell": {}})
    with pytest.raises(ValueError, match="train_enabled"):
        G14Config.from_dict({"train_enabled": True})
    with pytest.raises(ValueError, match="submission_enabled"):
        G14Config.from_dict({"submission_enabled": True, "run_mode": "TRAIN"})
    a, b = G14Config(), G14Config.from_dict({"device": "cpu", "drive_root": "/x"})
    assert a.semantic_hash() == b.semantic_hash()                  # paths / device are not semantic
    assert a.semantic_hash() != G14Config.from_dict({"loss": {"pos_weight": 3.0}}).semantic_hash()


def test_registry_entries_resolve_and_keep_eval_fixed():
    for name, e in REGISTRY.items():
        cfg = resolve(name)
        if e.group != "validation":
            assert cfg.split == G14Config().split and cfg.eval == G14Config().eval, name
    assert resolve("T1_local_cnn").model.regional_encoder == "none"
    assert resolve("T5_topology").loss.aux_weights["connectivity"] > 0


# =============================================================================== contract
def test_schema_order_reproducible_and_flip_unsafe_excluded(data):
    from gems_bench.g14.contract import build_schema, schema_hash

    c1, _ = build_schema(data, small_cfg())
    c2, _ = build_schema(data, small_cfg())
    assert [c.name for c in c1] == [c.name for c in c2] and schema_hash(c1) == schema_hash(c2)
    names = {c.name for c in c1}
    assert "raw:rtp" in names and "raw:tmi" not in names and "raw:mag_anom" not in names
    # axial / vector channels come in adjacent (cos, sin) / (x, y) pairs
    kinds = [c.kind for c in c1]
    for i, k in enumerate(kinds):
        if k in ("axial_cos", "vec_x"):
            assert kinds[i + 1] in ("axial_sin", "vec_y")


def test_stack_shapes_masks_and_nan_policy(stack, data):
    from gems_bench.g14.contract import check_contract

    for f in stack.factors:
        a = stack.level(f)
        assert a.shape[1:] == (-(-data.shape[0] // f), -(-data.shape[1] // f))
        assert stack.masks(f).shape[1:] == a.shape[1:]
    a1 = np.asarray(stack.level(1), np.float32)
    assert np.isnan(a1[:, ~data.valid]).all()                      # NaN outside the footprint, never zero-filled
    chk = check_contract(stack, data, small_cfg())
    assert (chk.status != "FAIL").all(), chk[chk.status == "FAIL"]


def test_block_nanmean_conserves_and_ignores_nan():
    from gems_bench.g14.contract import block_fraction, block_nanmean

    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    a[0, 0] = np.nan
    m = block_nanmean(a, 2)
    assert m.shape == (2, 2) and np.isclose(m[0, 0], (1 + 4 + 5) / 3)
    assert np.isnan(block_nanmean(np.full((2, 2), np.nan, np.float32), 2)).all()
    assert block_fraction(np.ones((3, 3), bool), 2).shape == (2, 2)


def test_train_only_normalisation(stack, data, system):
    from gems_bench.g14.multiscale import UnitInputs

    u = system.geo_unit(system.cv_zones[0])
    inp = UnitInputs(stack, data, u, small_cfg())
    assert inp.norm.fitted_on["mask"] == "unit.train"
    # changing values in the held-out zone must not change the statistics
    a = np.asarray(stack.level(1)).copy()
    b = a.copy()
    b[:, u.heldout] = b[:, u.heldout] * 100 + 7
    from gems_bench.g14.contract import UnitNormalizer

    ids = np.flatnonzero(u.train.ravel())[:5000]
    s1 = UnitNormalizer.fit(a.reshape(a.shape[0], -1)[:, ids].T, inp.kinds[:a.shape[0]], {})
    s2 = UnitNormalizer.fit(b.reshape(b.shape[0], -1)[:, ids].T, inp.kinds[:a.shape[0]], {})
    assert np.allclose(s1.med, s2.med) and np.allclose(s1.iqr, s2.iqr)


def test_missing_modality_masks_zero_where_absent(stack, data, system):
    from gems_bench.g14.multiscale import UnitInputs

    inp = UnitInputs(stack, data, system.geo_unit(system.cv_zones[0]), small_cfg())
    j = inp.modalities.index("conductivity")
    x, m = inp.window(1, 300, 20, 32)                              # synthetic conductivity gap at rows 300-340
    assert m[j].max() == 0.0 and np.isfinite(x).all()
    x, m = inp.window(1, -10, -10, 32)                            # outside the raster: zero values, zero validity
    assert m[:, :10, :10].max() == 0.0 and (x[:, :10, :10] == 0).all()


# =============================================================================== splits
def test_units_disjoint_embargo_and_no_heldout_in_known_context(system):
    units = system.units_for("spatial_cv_embargo") + system.units_for("fault_system_holdout")
    system.assert_no_leakage(units)
    a = system.audit(units)
    assert (a.train_heldout_overlap == 0).all() and (a.train_test_overlap == 0).all()
    assert (a.heldout_truth_supervised == 0).all() and (a.permitted_known_in_heldout == 0).all()
    assert (a.inner_heldout_overlap == 0).all()


def test_fault_system_grouping_requires_orientation():
    from gems_bench.g14.splits import fault_system_groups

    lab = np.zeros((60, 60), bool)
    lab[10, 5:25] = True
    lab[10, 28:50] = True          # collinear, 3 px gap -> one system
    lab[30, 5:25] = True
    lab[33:55, 27] = True          # perpendicular, nearby -> separate systems
    sysmap, comps = fault_system_groups(lab, link_m=500.0, angle_deg=30.0)
    assert sysmap[10, 10] == sysmap[10, 40]
    assert sysmap[30, 10] != sysmap[40, 27]


def test_locked_test_once_only(system, tmp_path, data):
    from gems_bench.g14.evaluation import evaluate_locked_test

    m = np.random.default_rng(0).random(data.shape).astype(np.float32)
    evaluate_locked_test("a", m, system, "top", 0.02, tmp_path)
    with pytest.raises(PermissionError):
        evaluate_locked_test("b", m, system, "top", 0.02, tmp_path)
    rec = evaluate_locked_test("b", m, system, "top", 0.02, tmp_path, force_second_look=True)
    assert rec["status"] == "SECOND_LOOK"


# =============================================================================== multiscale + augmentation
def test_views_share_centre_and_exact_fraction(stack, data, system):
    from gems_bench.g14.multiscale import MultiScaleDataset, UnitInputs, unit_targets

    cfg = small_cfg()
    u = system.geo_unit(system.cv_zones[0])
    inp = UnitInputs(stack, data, u, cfg)
    ds = MultiScaleDataset(inp, unit_targets(u.labels), u.train, u.ignore, cfg, 4, 0, augment=False)
    for (f, t, l, S) in ds.view_tops(ds.samples[0].top, ds.samples[0].left):
        cy = (t + S // 2) * f
        assert cy == ds.samples[0].top + cfg.data.local.size_px // 2      # identical geographic centre
    item = ds[0]
    assert item["local"].shape[1:] == (32, 32) and item["regional"].shape[1:] == (64, 64)
    assert item["targets"]["loss_mask"].max() <= 1 and np.isfinite(item["local"]).all()


def test_axial_and_vector_transforms():
    from gems_bench.g14.multiscale import axial_update, transform_channels, vector_update

    th = np.array([0.3])
    for k in range(4):
        for flip in (False, True):
            c, s = axial_update(np.cos(2 * th), np.sin(2 * th), k, flip)
            t2 = th + k * np.pi / 2
            t2 = np.pi - t2 if flip else t2
            assert np.allclose([c[0], s[0]], [np.cos(2 * t2[0]), np.sin(2 * t2[0])], atol=1e-6)
            vx, vy = vector_update(np.array([1.0]), np.array([0.0]), k, flip)
            ang = k * np.pi / 2
            ex, ey = np.cos(ang), np.sin(ang)
            ex = -ex if flip else ex
            assert np.allclose([vx[0], vy[0]], [ex, ey], atol=1e-6)
    # a straight east-west line image + its axial channel stay consistent after a quarter turn
    img = np.zeros((2, 8, 8), np.float32)
    img[0, 4, :] = 1.0                    # value channel
    img[1] = 1.0                          # cos 2theta of an E-W strike
    out = transform_channels(np.vstack([img, np.zeros((1, 8, 8), np.float32)]), ["scalar", "axial_cos", "axial_sin"],
                             1, False)
    assert out[0, :, 4].sum() == 8 and np.allclose(out[1], -1.0)     # now N-S: cos 2theta = -1


def test_targets_from_visible_labels_only(system):
    from gems_bench.g14.multiscale import unit_targets

    u = system.disc_unit(0)
    t = unit_targets(u.labels)
    wh = system.sp.withheld(0)
    assert t["trace"][wh].sum() == 0                                 # withheld systems are not targets


# =============================================================================== ensemble / schema
def test_ensemble_weights_chosen_on_other_zones(system, data):
    from gems_bench.g14 import ensemble as E

    rng = np.random.default_rng(0)
    maps = {}
    for u in system.eval_units(("geo",)):
        maps[(u.task, u.zone)] = rng.random(data.shape).astype(np.float32)
    members = {"a": maps, "b": {k: v * 0.5 for k, v in maps.items()}}
    ranks = E.ranked_members(members, system)
    fused, choice = E.weighted_crossfit(ranks, ["a", "b"], system, small_cfg(), "geo")
    assert set(choice.zone) == {u.zone for u in system.eval_units(("geo",))}
    assert (choice.dti_other.notna()).all()


def test_submission_guarded(data, tmp_path):
    from gems_bench.g14.submission import prepare_submission

    cfg = G14Config()
    with pytest.raises(PermissionError):
        prepare_submission(cfg, data, np.zeros(data.shape), "top", 0.02, tmp_path, name="x", provenance={})
    cfg = G14Config.from_dict({"run_mode": "SUBMIT", "submission_enabled": True})
    with pytest.raises(PermissionError, match="confirm"):
        prepare_submission(cfg, data, np.zeros(data.shape), "top", 0.02, tmp_path, name="x", provenance={})
    q = np.random.default_rng(0).random(data.shape).astype(np.float32)
    rec = prepare_submission(cfg, data, q, "top", 0.02, tmp_path, name="x", confirm="WRITE_SUBMISSION",
                             provenance={"model": "m", "construction": "c", "selection": "cross-fitted", "cfg_hash": "h"})
    v = rec["validation"]
    assert v["valid"], v["checks_failed"]
