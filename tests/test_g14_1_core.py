"""G14.1 tests: configuration / modes / stage plan, geometry, representations, discovery, metric audit, artifacts,
submission guards, ensemble leakage.

WRITTEN BUT NEVER EXECUTED by the authoring session. Run in Colab:
``python -m pytest -q tests/test_g14_1_core.py tests/test_g14_1_torch.py tests/test_g14_1_regressions.py``.
Pure-numpy tests use tiny synthetic rasters; data-dependent tests use ``gems_bench.g13.synthetic`` (400 px grid).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from gems_bench.g14_1.config import G141Config


# =============================================================================== config / modes / plan
def test_defaults_are_safe_and_valid():
    c = G141Config()
    assert c.mode == "AUDIT_ONLY" and not (c.allow_inference or c.allow_training or c.allow_submission)
    assert "artifact_inventory" in c.enabled_stages() and "train_models" not in c.enabled_stages()
    assert not c.stage_enabled("submission")


def test_run_mode_alias_and_typo_message():
    assert G141Config.from_dict({"run_mode": "DIAGNOSE"}).mode == "DIAGNOSE"
    with pytest.raises(KeyError, match="did you mean"):
        G141Config.from_dict({"profle": "FAST"})


def test_incompatible_stage_and_mode_is_an_error_not_a_skip():
    with pytest.raises(ValueError, match="capability 'training'"):
        G141Config.from_dict({"mode": "SUBMIT", "stages": {"train_models": True}})
    with pytest.raises(ValueError, match="capability 'submission'"):
        G141Config.from_dict({"mode": "TRAIN", "stages": {"submission": True}})
    c = G141Config.for_mode("SUBMIT")
    assert c.stage_enabled("submission") and not c.allow_training


def test_protocol_vs_primary_task():
    with pytest.raises(ValueError, match="not produced by protocol"):
        G141Config.from_dict({"validation": {"protocol": "geo_only", "primary_task": "disc_official"}})
    assert G141Config.from_dict({"validation": {"protocol": "geo_only", "primary_task": "geo"}}).tasks == ("geo",)
    with pytest.raises(ValueError, match="locked_test"):
        G141Config.from_dict({"validation": {"backend": "phase5", "locked_test": True}})


def test_sweep_validation_and_hash():
    with pytest.raises(ValueError, match="increasing"):
        G141Config.from_dict({"sweep": {"area_fracs": [0.1, 0.01]}})
    with pytest.raises(ValueError, match="start with 0.0"):
        G141Config.from_dict({"sweep": {"g14_weights": [0.1, 0.2]}})
    a, b = G141Config(), G141Config.from_dict({"mode": "DIAGNOSE", "drive_root": "/x", "profile": "FULL"})
    assert a.semantic_hash() == b.semantic_hash()


def test_stage_plan_dependencies_and_profiles():
    from gems_bench.g14_1.experiments import plan

    p = plan(G141Config()).set_index("stage")
    for s in ("R00", "R01", "R02", "R03", "R04", "R05"):
        assert p.loc[s, "action"] == "RUN", s
    for s in ("R08", "R09", "R10", "R13", "R14", "R15"):
        assert p.loc[s, "action"] == "SKIP", s
    full = plan(G141Config.for_mode("TRAIN", "FULL")).set_index("stage")
    assert full.loc["R09", "action"] == "RUN" and full.loc["R17", "action"] == "SKIP"


def test_spec_registry_changes_one_factor():
    from dataclasses import asdict

    from gems_bench.g14_1.experiments import REGISTRY

    for name, (spec, ref, factor) in REGISTRY.items():
        spec.validate()
        if ref and ref in REGISTRY and not name.startswith(("M0", "D1", "M1", "M2", "E2_nnpu", "M5_hard")):
            a, b = asdict(spec), asdict(REGISTRY[ref][0])
            diff = {k for k in a if a[k] != b[k]} - {"name", "notes"}
            assert len(diff) <= 2, (name, diff)          # one factor (+ its required companion setting)


# =============================================================================== geometry
def test_kernel_radius_from_transform():
    from gems_bench.g14_1.geometry import assumption_audit, kernel_radius_px

    k = kernel_radius_px((100.0, 0, 500000.0, 0, -100.0, 4000000.0))
    # k(d) > 0 only for d < 300 m: (3, 0) has d = 300 m exactly and is excluded; (2, 2) = 283 m is the farthest
    assert k["radius_px_col"] == pytest.approx(3.0) and k["max_offset_px"] == 2
    k50 = kernel_radius_px((50.0, 0, 0, 0, -50.0, 0))
    assert k50["radius_px_col"] == pytest.approx(6.0)
    a = assumption_audit((50.0, 0, 0, 0, -50.0, 0))
    assert (a.status == "INCONSISTENT").any()
    assert not (assumption_audit((100.0, 0, 0, 0, -100.0, 0)).status == "INCONSISTENT").any()


def test_phase_correlation_detects_integer_and_subpixel_shift():
    from gems_bench.g14_1.geometry import phase_correlation

    rng = np.random.default_rng(0)
    a = rng.random((128, 96))
    b = np.roll(a, (-3, 5), (0, 1))                 # b = a moved up 3, right 5
    r = phase_correlation(a, b)
    assert round(r["dy_px"]) == 3 and round(r["dx_px"]) == -5 and r["peak"] > 0.5
    assert abs(phase_correlation(a, a)["dy_px"]) < 1e-6


def test_d4_match_detects_flip():
    from gems_bench.g14_1.geometry import d4_match

    a = np.random.default_rng(1).random((40, 40))
    assert d4_match(a, a[::-1]).transform.iloc[0] == "flip_ud"
    assert d4_match(a, a.T).transform.iloc[0] == "transpose"
    assert d4_match(a, a).transform.iloc[0] == "identity"


# =============================================================================== representations
def _unit_like(n=60):
    ev = np.ones((n, n), bool)
    known = np.zeros((n, n), bool)
    known[30, 5:55] = True
    s = np.zeros((n, n), np.float32)
    for d in range(-3, 4):
        s[30 + d, 5:55] = 1 - abs(d) / 4          # a ridge on the known fault
    s[10, 5:55] = 0.6                              # a secondary ridge elsewhere
    return s, ev, known, ev & ~known


def test_area_accounting_and_known_zero():
    from gems_bench.g14_1.representations import ALL, apply_rep

    s, ev, known, scored = _unit_like()
    for rep in ALL:
        for order in ("pixel_exact", "mask_before", "mask_after", "mask_both", "ring1"):
            q, acc = apply_rep(rep, s, ev, known, scored, None if rep in ("raw", "power2", "power4") else 0.05, order)
            assert not (q[known] > 0).any(), (rep, order)          # the scorer's pixel-exact mask, applied
            assert acc["scored_px"] <= acc["final_px"]
    q, acc = apply_rep("crest_w1", s, ev, known, scored, 0.02)
    assert acc["dilation_factor"] > 1.5                            # the stated area is the PRE-dilation area
    q0, acc0 = apply_rep("crest_w0", s, ev, known, scored, 0.02)
    assert acc0["final_px"] <= acc0["k_requested"]


def test_mask_after_lets_known_compete_for_budget():
    from gems_bench.g14_1.representations import apply_rep

    s, ev, known, scored = _unit_like()
    _, a_after = apply_rep("top", s, ev, known, scored, 0.02, "mask_after")
    _, a_exact = apply_rep("top", s, ev, known, scored, 0.02, "pixel_exact")
    assert a_after["on_known_px"] > 0 and a_after["scored_px"] < a_exact["scored_px"]


def test_sparse_soft_keeps_probabilities():
    from gems_bench.g14_1.representations import apply_rep

    s, ev, known, scored = _unit_like()
    q, _ = apply_rep("sparse_soft", s, ev, known, scored, 0.05)
    v = q[q > 0]
    assert v.size and v.max() <= 1 and len(np.unique(v)) > 1


# =============================================================================== discovery / leakage (synthetic data)
@pytest.fixture(scope="module")
def data(tmp_path_factory):
    from gems_bench.g13.contracts import load
    from gems_bench.g13.synthetic import make

    return load(make(tmp_path_factory.mktemp("g141")), tmp_path_factory.mktemp("g141c"))


@pytest.fixture(scope="module")
def system(data):
    from gems_bench.g13.splits import build_splits
    from gems_bench.g13.synthetic import small_config
    from gems_bench.g14.splits import UnitSystem

    return UnitSystem("g13", data, g13_splits=build_splits(data, small_config("/tmp/u1", "/tmp/u1")), embargo_m=2000.0)


def test_original_g14_units_are_flagged_as_shortcut(system):
    from gems_bench.g14_1.leakage import target_context_overlap

    for u in [system.geo_unit(k) for k in system.cv_zones]:
        r = target_context_overlap(u)
        assert r["status"] == "SHORTCUT" and r["train_pos_in_context"] == pytest.approx(1.0)


def test_geophysics_only_removes_context(system):
    from gems_bench.g14_1.discovery import geophysics_only
    from gems_bench.g14_1.leakage import target_context_overlap

    u = geophysics_only(system.geo_unit(system.cv_zones[0]))
    assert not u.permitted_known.any() and target_context_overlap(u)["status"] == "DISJOINT"


def test_conditional_draws_are_disjoint_and_change(system):
    from gems_bench.g14_1.discovery import ConditionalDiscoveryUnit

    u = system.geo_unit(system.cv_zones[0])
    cu = ConditionalDiscoveryUnit.build(u, 1000.0, 30.0, 100.0)
    c1, t1, r1 = cu.draw(1)
    c2, t2, r2 = cu.draw(2)
    assert not (c1 & t1).any() and not (c2 & t2).any()
    assert t1.any() and c1.any()
    assert not (t1 & u.inner_val).any() and not (c1 & u.inner_val).any()     # inner faults are never in a draw
    assert not (cu.eval_context() & u.labels & u.inner_val).any()


def test_separated_targets_never_use_inner_labels(system):
    from gems_bench.g14_1.discovery import separate_targets, target_leak_check

    u = system.geo_unit(system.cv_zones[0])
    region, tr, iv = separate_targets(u, u.labels, 3.0, 300.0, 100.0)
    chk = target_leak_check(u, region, tr, 3.0)
    assert chk["inner_label_px_in_train_targets"] == 0
    assert not (region & u.inner_val).any()


def test_supervision_masks_context_carries_no_loss():
    from gems_bench.g14_1.discovery import supervision_masks

    n = 50
    region = np.ones((n, n), bool)
    ctx = np.zeros((n, n), bool)
    ctx[10, :] = True
    tgt = np.zeros((n, n), bool)
    tgt[40, :] = True
    s = supervision_masks(region, region, ctx, tgt, np.zeros_like(ctx), 100.0)
    assert s["loss_mask"][ctx].max() == 0
    assert s["neg_weight"][9].min() == 1 and s["neg_weight"][11].min() == 1   # shoulders of context: negatives
    assert s["neg_weight"][38:43].max() == 0 or s["positive"][40].min() == 1    # target ring: no negatives


def test_protocol_check_messages():
    from gems_bench.g14_1.discovery import check_protocol

    t = check_protocol("geo_only", "disc_official", {"geo": 3}, submission=True)
    assert "ERROR" in set(t.level)
    t = check_protocol("geo_and_disc", "disc_official", {"geo": 3, "disc_official": 0}, submission=True)
    assert "PROVISIONAL" in set(t.level)


# =============================================================================== metric audit
def test_ratio_of_sums_differs_from_mean_of_ratios():
    from gems_bench.g14_1.metric_audit import ratio_of_sums_vs_mean_of_ratios

    b = pd.DataFrame({"tp_w": [9.0, 0.0], "fp_w": [1.0, 100.0], "fn_w": [1.0, 0.0], "n_truth": [10, 0]})
    r = ratio_of_sums_vs_mean_of_ratios(b)
    assert r["mean_of_ratios"] != pytest.approx(r["ratio_of_sums"])
    assert r["ratio_of_sums"] < 0.5


def test_region_components_match_metric():
    from gems_bench.g14_1.metric_audit import region_components
    from gems_bench.metric import dti_components

    rng = np.random.default_rng(0)
    t = np.zeros((40, 40), bool)
    t[20, 5:35] = True
    p = rng.random((40, 40)) * 0.2
    ev = np.ones_like(t)
    a = region_components(p, t, ev, None, (100.0, 100.0))
    b = dti_components(p, t, eval_mask=ev, exclude_supports_tp=False)
    assert a.dti == pytest.approx(b.dti, abs=1e-9)


def test_failure_classes_need_evidence():
    from gems_bench.g14_1.metric_audit import classify

    out = classify({"alpha_fp": 10.0, "beta_fn": 1.0, "pred_px": 1000, "n_truth": 10, "dti": 0.05,
                    "fragmentation_per_100km": np.nan})
    st = {r["class"]: r["status"] for r in out}
    assert st["F01"] == "FIRED" and st["F03"] == "not_assessed" and st["F09"] == "not_assessed"


# =============================================================================== artifacts / ledger / submission
def test_discovery_prefers_local_and_ledger_resumes(tmp_path):
    from gems_bench.g14_1.artifacts import Roots, StageLedger, best_copy, discover

    exp = "T1_local_cnn__abc"
    loc = tmp_path / "local" / "experiments" / exp / "maps"
    drv = tmp_path / "drive" / "outputs" / "G14_GEOFAULTFORMER" / "experiments" / exp / "maps"
    for d in (loc, drv):
        d.mkdir(parents=True)
    np.save(drv / "full.npy", np.zeros((4, 5), np.float32))
    np.save(drv / "geo_1.npy", np.zeros((4, 5), np.float32))
    np.save(loc / "geo_1.npy", np.zeros((4, 5), np.float32))
    roots = Roots(tmp_path / "local", tmp_path / "drive", exp, tmp_path / "drive" / "outputs" / "G14_GEOFAULTFORMER",
                  g13_local=tmp_path / "none")
    inv = discover(roots, (4, 5))
    assert best_copy(inv, "g14_full_map").parts[-6] == "drive" or "drive" in str(best_copy(inv, "g14_full_map"))
    assert "local" in str(best_copy(inv, "g14_unit_maps", "maps/geo_1.npy"))
    assert inv.grid_ok.all()
    led = StageLedger(tmp_path / "stages.json", "h")
    t0 = led.start("R00", "d1")
    led.finish("R00", t0, [tmp_path / "stages.json"])
    assert StageLedger(tmp_path / "stages.json", "h").fresh("R00", "d1")
    assert not StageLedger(tmp_path / "stages.json", "h").fresh("R00", "d2")


def test_experiment_hash_check(tmp_path):
    from gems_bench.g14_1.artifacts import experiment_hash_check

    m = tmp_path / "geo_1_train.json"
    m.write_text(json.dumps({"unit": "geo_1", "cfg_hash": "zzz", "channels": []}))
    assert experiment_hash_check("T1_local_cnn__abc", [m]).status.iloc[0] == "HASH_MISMATCH"


def test_compatibility_rules():
    from gems_bench.g14_1.artifacts import compatible

    assert not compatible({"shape": (10, 10), "protocol": "g13"}, {"shape": (10, 11), "protocol": "g13"})[0]
    assert not compatible({"shape": (10, 10), "protocol": "g13"}, {"shape": (10, 10), "protocol": "phase5"})[0]
    assert not compatible({"shape": (10, 10), "protocol": "full"}, {"shape": (10, 10), "protocol": "g13"})[0]
    assert compatible({"shape": (10, 10), "protocol": "g13"}, {"shape": (10, 10), "protocol": "g13"})[0]


def test_missing_final_map_and_scorecard_give_actionable_errors(tmp_path):
    from gems_bench.g14_1.submission import SubmissionError, find_full_map, find_operating_point

    inv = pd.DataFrame(columns=["artifact", "rel", "path", "location", "bytes"])
    with pytest.raises(SubmissionError, match="predict_final"):
        find_full_map("g14", inv, tmp_path)
    assert find_operating_point("g14", inv, tmp_path) is None


def test_submission_refuses_wrong_mode(tmp_path):
    from gems_bench.g14_1.submission import SubmissionError, generate_submission

    with pytest.raises(SubmissionError, match="SUBMIT"):
        generate_submission(G141Config(), None, pd.DataFrame())


def test_operating_point_preference(tmp_path):
    from gems_bench.g14_1.submission import find_operating_point

    (tmp_path / "calibration").mkdir()
    (tmp_path / "calibration" / "selected_operating_points.json").write_text(json.dumps(
        {"g14": {"geo": {"rep": "top", "frac": 0.02}, "disc_official": {"rep": "sparse_soft", "frac": 0.005}}}))
    op = find_operating_point("g14", pd.DataFrame(columns=["artifact"]), tmp_path)
    assert op["status"] == "DISCOVERY_CALIBRATED" and op["rep"] == "sparse_soft"


# =============================================================================== ensemble leakage
def test_blend_weight_for_a_zone_is_chosen_on_other_zones():
    from gems_bench.g14_1.champion import crossfit_blend

    rows = []
    for z in (0, 1, 2):
        for w in (0.0, 0.5):
            good = (w == 0.5) != (z == 0)          # zone 0 alone prefers w = 0, the others prefer w = 0.5
            rows.append({"task": "geo", "zone": z, "kind": "champion" if w == 0 else "blend", "w": w,
                         "target_frac": 0.02, "tp_w": 10.0 if good else 1.0, "fp_w": 1.0, "fn_w": 1.0, "dti": 0})
    res = crossfit_blend(pd.DataFrame(rows))["geo"]
    z0 = res["per_zone"][res["per_zone"].zone == 0].iloc[0]
    assert float(z0.w) == 0.5                      # its own preference (w = 0) could not be used
