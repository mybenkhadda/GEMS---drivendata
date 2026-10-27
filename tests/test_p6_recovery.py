"""P6.1 recovery: synthetic checks of the deployable-policy evaluation (one cell per zone for both tasks, no
held-out leakage, matched comparisons), the single export contract, parity statistics, the candidate-recall
inventory / oracle isolation / taxonomy, segment deduplication and graph topology, the G01 / G02 channel registry
classification, the promotion gate and restartability (no Drive, no real data, no trained model).

Written for the P6.1 brief; NOT executed in the authoring session (code-authoring-only mode)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


# ============================================================================ helpers
def _grid(cells: dict, zones=(0, 1, 2), tasks=("disc_official", "geo")) -> pd.DataFrame:
    """cells: {(candidate, blend, rep, frac): {task: [tp per zone]}} with fp = fn = 10 everywhere. ``systems`` follows
    the ``system_hits`` contract: ";"-joined INTEGER ids (here one id per zone and rep, plus one shared id)."""
    rows = []
    for (cand, lam, rep, frac), per in cells.items():
        for t in tasks:
            for z in zones:
                tp = per[t][z] if isinstance(per[t], (list, tuple)) else per[t]
                sid = 10 * z + (1 if rep.startswith("crest") else 2)
                rows.append({"task": t, "zone": z, "candidate": cand, "blend": lam, "rep": rep, "frac": frac,
                             "tp_w": float(tp), "fp_w": 10.0, "fn_w": 10.0, "n_truth": 20.0,
                             "systems": f"{sid};{100 + z}" if t == "disc_official" else None, "area_px": 100,
                             "area_frac_eligible": frac, "length_km": 1.0, "n_segments": 3, "short_share": 0.0})
    return pd.DataFrame(rows)


GATES = {"geo_floor": 0.5, "worst_floor": 0.0, "slot_min_gain": 0.01, "budget_tolerance": 0.02,
         "w2_recorded_disc": 0.1}


# ============================================================================ X1: policies
def test_policy_one_cell_per_zone_for_both_tasks():
    from gems_bench.p6.export_parity import evaluate_policy

    G = _grid({("w2", 1.0, "crest_1px", 0.01): {"disc_official": 10, "geo": 30},
               ("w2", 1.0, "skeleton_1px", 0.01): {"disc_official": 12, "geo": 5}})
    s, P = evaluate_policy(G, "D", {"candidates": ["w2"]}, "disc", "disc_official", 0.5, ["disc_official", "geo"])
    per = P.groupby("zone").rep.nunique()
    assert (per == 1).all()                                  # the geo number comes from the same cell as disc
    assert set(P.rep) == {"skeleton_1px"} and s["deploy_cell"]["rep"] == "skeleton_1px" and s["deployable"]
    assert s["geo_dti"] < s["disc_dti"]                      # geo is what that one policy actually achieves


def test_policy_never_selects_on_heldout_zone():
    from gems_bench.p6.export_parity import evaluate_policy

    G = _grid({("w2", 1.0, "crest_1px", 0.01): {"disc_official": [100, 1, 1], "geo": 10},
               ("w2", 1.0, "crest_1px", 0.03): {"disc_official": 10, "geo": 10}})
    _, P = evaluate_policy(G, "B", {"candidates": ["w2"]}, "disc", "disc_official", 0.5, ["disc_official", "geo"])
    p = P[P.task == "disc_official"].set_index("zone")
    assert p.loc[0, "frac"] == pytest.approx(0.03)          # the spike lives on zone 0 only
    assert "0" not in p.loc[0, "selection_disc_zones"].split(";")


def test_disc_geo_floor_objective_and_separate_not_deployable():
    from gems_bench.p6.export_parity import evaluate_policy, select_cell

    G = _grid({("w2", 1.0, "crest_1px", 0.01): {"disc_official": 10, "geo": 20},     # geo dti 0.5 (meets floor)
               ("w2", 1.0, "skeleton_1px", 0.01): {"disc_official": 14, "geo": 2}})  # better disc, geo below floor
    S = G.assign(blend=G.blend.astype(float))
    c, info = select_cell(S, [0, 1], [0, 1], "disc_geo_floor", "disc_official", 0.45)
    assert c["rep"] == "crest_1px" and info["floor_met"] is True
    c2, _ = select_cell(S, [0, 1], [0, 1], "disc", "disc_official", 0.45)
    assert c2["rep"] == "skeleton_1px"
    s, _ = evaluate_policy(G, "D", {"candidates": ["w2"]}, "separate", "disc_official", 0.45, ["disc_official", "geo"])
    assert s["deployable"] is False                         # disc and geo picked different cells


def test_fixed_policy_is_the_fixed_cell():
    from gems_bench.p6.export_parity import evaluate_policy

    G = _grid({("w2", 1.0, "skeleton_1px", 0.05): {"disc_official": 5, "geo": 5},
               ("w2", 1.0, "skeleton_1px", 0.01): {"disc_official": 50, "geo": 50}})
    s, P = evaluate_policy(G, "E", {"candidates": ["w2"], "fixed": {"rep": "skeleton_1px", "frac": 0.05, "blend": 1.0}},
                           "disc_geo_floor", "disc_official", 0.5, ["disc_official", "geo"])
    assert s["fixed"] and (P.frac == 0.05).all() and s["deploy_cell"]["frac"] == pytest.approx(0.05)


def test_oracle_rows_refused_by_selection():
    from gems_bench.p6.export_parity import OracleLeak, evaluate_policy

    G = _grid({("w2", 1.0, "crest_1px", 0.01): {"disc_official": 10, "geo": 10}}).assign(role="inventory")
    G.loc[0, "role"] = "oracle_diagnostic"
    with pytest.raises(OracleLeak):
        evaluate_policy(G, "B", {"candidates": ["w2"]}, "disc", "disc_official", 0.5, ["disc_official", "geo"])


def test_policy_tables_choice_meta_and_reproduction():
    from gems_bench.p6.export_parity import policy_tables

    G = _grid({("w2", 1.0, "crest_1px", 0.01): {"disc_official": 12, "geo": 20},
               ("w2", 1.0, "skeleton_1px", 0.01): {"disc_official": 14, "geo": 4},
               ("fallback_fusion", 0.0, "crest_1px", 0.01): {"disc_official": 5, "geo": 20}})
    cfg = {"objective": "disc_geo_floor", "sensitivity_objectives": ["disc"],
           "policies": {"B": {"candidates": ["w2"], "reps": ["crest_1px"], "benchmark_space": "w2|crest_1px"},
                        "C": {"candidates": ["w2"], "reps": ["skeleton_1px"]}},
           "references": {"R": {"candidates": ["fallback_fusion"]}}}
    bench = pd.DataFrame([{"space": "w2|crest_1px", "disc_dti": 12 / (12 + 2 + 8 + 1e-6),
                           "geo_dti": 20 / (20 + 2 + 8 + 1e-6)}])
    T = policy_tables(G, cfg, "disc_official", GATES, ["disc_official", "geo"], benchmark=bench)
    assert T["chosen"] == "B"                                # C misses the geo floor
    S = T["comparison"]
    assert bool(S[(S.policy == "B") & (S.objective == "separate")].reproduces_benchmark.iloc[0])
    assert set(T["meta"]["chosen_per_zone"].values()) <= {"B", "C"}
    assert {"nested_policy"} <= set(T["frontier"].kind)


def test_overlap_exclusion_falls_back_when_everything_overlaps():
    from gems_bench.p6.export_parity import overlap_exclusion

    ov = pd.DataFrame([{"task_a": a, "zone_a": za, "task_b": b, "zone_b": zb, "overlap_px": 5}
                       for a in ("disc_official", "geo") for b in ("disc_official", "geo")
                       for za in range(3) for zb in range(3)])
    ex, ok = overlap_exclusion(ov, [0, 1, 2], "disc_official")
    assert ok is False and ex[0] == {"disc": {0}, "geo": {0}}
    ov2 = ov.assign(overlap_px=np.where(ov.zone_a == ov.zone_b, 5, 0))
    ex2, ok2 = overlap_exclusion(ov2, [0, 1, 2], "disc_official")
    assert ok2 and ex2[1] == {"disc": {1}, "geo": {1}}


# ============================================================================ X1: export contract / parity
def _recipe(**over):
    from gems_bench.p6.export_parity import build_recipe

    pol = {"selection_id": "B|disc_geo_floor|all_zones", "space": "w2 crest", "objective": "disc_geo_floor",
           "rep": "crest_1px", "frac": 0.03, "blend": 1.0, **over}
    return build_recipe(candidate="w2_B", kind="w2_policy", policy=pol, model={"features": ["a", "b"]},
                        sources={"score": "abc"}, feature_schema=["a", "b"],
                        preprocessing={"eligibility": "eligible", "known": "labels"})


def test_recipe_rejects_mixed_selection_sources():
    from gems_bench.p6.export_parity import PolicyMismatch, plan_audit

    with pytest.raises(PolicyMismatch):                      # the historical W2 defect
        _recipe(frac_source="geo nested choice")
    with pytest.raises(ValueError):
        _recipe(frac=None)
    r = _recipe()
    assert r["policy"]["rep_source"] == r["policy"]["frac_source"] == r["policy"]["selection_id"]
    assert plan_audit(r)["status"] == "PASS" and r["fingerprint"] == _recipe()["fingerprint"]


def test_audit_export_checks_grid_area_and_sidecar(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    from gems_bench.p6.export_parity import audit_export

    H, W = 20, 30
    tr, crs = from_origin(500000, 4000000, 100, 100), CRS.from_epsg(32611)
    tv = np.ones((H, W), bool)
    tv[:, -2:] = False
    data = SimpleNamespace(template_profile={"crs": crs, "transform": tr, "height": H, "width": W},
                           template_valid=tv, valid=tv.copy(), labels=np.zeros((H, W), bool))
    rec = _recipe()
    el = tv.copy()
    a = np.zeros((H, W), np.float32)
    a[~tv] = np.nan
    a[5, 2:10] = 1.0                                         # 8 px of 540 eligible: 1.5 % <= 3 %
    p = tmp_path / "sub.tif"
    with rasterio.open(p, "w", driver="GTiff", height=H, width=W, count=1, dtype="float32", crs=crs, transform=tr) as d:
        d.write(a, 1)
    p.with_suffix(".json").write_text(json.dumps({"parameters": {"rep": "crest_1px", "frac": 0.03, "blend": 1.0},
                                                  "recipe_fp": rec["fingerprint"], "features": ["a", "b"]}))
    ok = audit_export(p, rec, data, el)
    assert ok["status"] == "PASS" and ok["sha256"]
    a[0:10, 0:20] = 1.0                                      # far above the budget
    with rasterio.open(p, "w", driver="GTiff", height=H, width=W, count=1, dtype="float32", crs=crs, transform=tr) as d:
        d.write(a, 1)
    bad = audit_export(p, rec, data, el)
    assert bad["status"] == "FAILED" and any("area" in f for f in bad["failed"])
    assert audit_export(tmp_path / "absent.tif", rec, data, el)["status"] == "FAILED"


def test_parity_agreement_and_classification():
    pytest.importorskip("scipy")
    from gems_bench.p6.export_parity import agreement, classify_parity, map_agreement

    rng = np.random.default_rng(0)
    a = rng.random((50, 50)).astype(np.float32)
    m = np.ones_like(a, bool)
    pc = {"warn_rank_corr": 0.6, "block_rank_corr": 0.2}
    same = agreement(a, a, m)
    assert same["spearman"] == pytest.approx(1.0) and same["top1_jaccard"] == pytest.approx(1.0)
    assert classify_parity(same, pc) == "PASS"
    inv = agreement(a, 1 - a, m)
    assert classify_parity(inv, pc) == "BLOCKED"
    assert classify_parity(inv, pc, expected=True) == "WARNING"   # by-design differences never BLOCK
    assert map_agreement(a > 0.9, a > 0.9)["jaccard"] == pytest.approx(1.0)


# ============================================================================ X2: candidate recall
def _unit(z=0, shape=(40, 60)):
    from gems_bench.gems3.official import Unit

    H, W = shape
    truth = np.zeros(shape, bool)
    truth[10, 5:55] = True                                  # system 1
    truth[30, 10:20] = True                                 # system 2 (short)
    return Unit("disc_official", z, (slice(0, H), slice(0, W)), np.ones(shape, bool), truth, None, False, None)


def test_system_pixels_and_attributes():
    from gems_bench.p6.candidate_recall import system_attributes, system_pixels

    u = _unit()
    sm = np.full(u.ev.shape, -1)
    sm[10, 5:55], sm[30, 10:20] = 1, 2
    px = system_pixels(SimpleNamespace(splits=SimpleNamespace(system_map=sm)), u)
    assert set(px.system_id) == {1, 2}
    A = system_attributes(px, {"x": np.ones(u.ev.shape)}, None, None, np.ones(u.ev.shape, bool)).set_index("system_id")
    assert A.loc[1, "length_km"] > A.loc[2, "length_km"] and A.loc[1, "fragments"] == 1
    assert A.loc[1, "orientation_deg"] == pytest.approx(0.0, abs=1e-6) and A.loc[1, "eligible_share"] == 1.0


def test_oracle_respects_budget_and_is_labelled():
    from gems_bench.p6.candidate_recall import oracle_select

    u = _unit()
    pool = np.zeros(u.ev.shape, bool)
    pool[10, 5:55] = True                                   # on the truth
    pool[20, 5:55] = True                                   # false positive line
    m, n = oracle_select(pool, u.truth, 30, tol_px=1)
    assert m.sum() <= 30 and not m[20].any()                # only truth-supported pieces, within the budget
    m2, _ = oracle_select(pool, u.truth, None, tol_px=1)
    assert m2[10].sum() > 0 and not m2[20].any()


def test_taxonomy_categories_and_decision():
    from gems_bench.p6.candidate_recall import classify, recall_decision

    t = {"coverage_min": 0.5, "known_near_km": 0.3, "partial_hit_min": 0.1, "fragment_min_components": 3,
         "orientation_max_deg": 30, "support_low": 0.05}
    base = {"eligible_share": 1.0, "known_dist_km_median": 5.0}
    assert classify({**base, "recovered_ref": True}, t) == "recovered"
    assert classify({**base, "recovered_ref_ranker_larger_budget": True}, t) == "budget_limitation"
    assert classify({**base, "recovered_any": True}, t) == "ranked_too_low"
    assert classify({**base, "best_support": 0.0, "near_far_share": 0.0}, t) == "no_candidate_support"
    assert classify({**base, "near_far_share": 0.8, "near_tol_share": 0.1}, t) == "displaced"
    assert classify({**base, "eligible_share": 0.2}, t) == "insufficient_spatial_coverage"
    T = pd.DataFrame({"missed": [True] * 4, "group": ["absent"] * 4, "length_km": [1.0] * 4})
    assert recall_decision(T, t, 10, 0.1)["verdict"] == "INCONCLUSIVE"        # never concluded from 4 examples
    T = pd.DataFrame({"missed": [True] * 12, "group": ["absent"] * 9 + ["ranking"] * 3, "length_km": [1.0] * 12})
    assert recall_decision(T, t, 10, 0.1)["verdict"] == "ABSENT"


def test_system_flags_budget_and_any_source():
    from gems_bench.p6.candidate_recall import system_flags

    inv = pd.DataFrame([
        {"unit": "d:0", "zone": 0, "source": "reference", "rep": "c", "frac": 0.01, "system_id": 1, "n_px": 5,
         "support_mean": 0.1, "hit_share": 0.1, "recovered": False},
        {"unit": "d:0", "zone": 0, "source": "w2", "rep": "c", "frac": 0.05, "system_id": 1, "n_px": 5,
         "support_mean": 0.9, "hit_share": 0.9, "recovered": True},
        {"unit": "d:0", "zone": 0, "source": "reference", "rep": "c", "frac": 0.01, "system_id": 2, "n_px": 5,
         "support_mean": 0.0, "hit_share": 0.0, "recovered": False},
        {"unit": "d:0", "zone": 0, "source": "cnn_b0", "rep": "c", "frac": 0.01, "system_id": 2, "n_px": 5,
         "support_mean": 0.8, "hit_share": 0.8, "recovered": True}])
    F = system_flags(inv, {"d:0": 0.01}).set_index("system_id")
    assert F.loc[1, "recovered_ref_ranker_larger_budget"] and not F.loc[1, "recovered_ref"]
    assert F.loc[2, "recovered_any"] and not F.loc[2, "recovered_ref_ranker_larger_budget"]


# ============================================================================ X3: segments
def test_skeleton_graph_junction_and_branches():
    from gems_bench.p6.segment_recovery import skeleton_graph

    m = np.zeros((30, 30), bool)
    m[15, 2:28] = True
    m[2:28, 15] = True                                      # a cross: one junction, four branches
    m[2, 2:5] = True                                        # a short isolated piece (kept)
    lab, T, d = skeleton_graph(m)
    assert d["n_junction_clusters"] == 1 and d["n_edges"] >= 5 and d["n_components"] == 2
    assert (T.degree_class == "isolated").sum() >= 1       # short pieces are never dropped


def test_dedup_removes_parallel_duplicate_keeps_crossing():
    from gems_bench.p6.segment_recovery import dedup_inventory

    a = np.zeros((40, 40), bool)
    a[20, 5:35] = True
    b = np.zeros((40, 40), bool)
    b[21, 6:34] = True                                      # parallel copy 1 px away -> duplicate
    b[5:35, 10] = True                                      # crossing line -> kept
    lab, D = dedup_inventory({"w2": a, "fallback_fusion": b}, ["w2", "fallback_fusion"], 2, 0.6, 20)
    fb = D[D.source == "fallback_fusion"]
    assert fb.duplicate.any() and (~fb.duplicate).any()
    kept = fb[~fb.duplicate]
    assert (np.abs(np.degrees(kept.theta) - 90) < 20).any()   # the vertical (crossing) piece survives


def _apply_top(score, frac):
    s = np.asarray(score, float)
    k = max(int(round(frac * s.size)), 1)
    thr = np.partition(s.ravel(), -k)[-k]
    return s >= thr


def test_strategies_never_exceed_h0_area():
    from gems_bench.p6.boost_diagnostics import load_recovery_config
    from gems_bench.p6.segment_recovery import strategy_map

    cfg = load_recovery_config()["segments"]
    rng = np.random.default_rng(1)
    H, W = 60, 80
    w2 = rng.random((H, W)) * 0.5
    w2[20, 5:75] = 0.99                                     # two W2 lines absorb (almost) the whole 3 % budget
    w2[50, 5:75] = 0.97
    fb = rng.random((H, W)) * 0.5
    fb[40, 5:30] = 0.98                                     # short fallback-only line (fits the 25 % non-core share),
                                                            #   both CNNs confident there
    c0, c1 = fb.copy(), fb.copy()
    inp = {"w2": w2, "fallback_fusion": fb, "cnn_b0": c0, "cnn_b1": c1}
    q0, _, d0 = strategy_map("H0", inp, _apply_top, 0.03, cfg)
    for name in ("H1", "H2"):
        q, T, d = strategy_map(name, inp, _apply_top, 0.03, cfg)
        assert q.sum() <= q0.sum() and d["px"] <= d0["px"]
    q1, _, _ = strategy_map("H1", inp, _apply_top, 0.03, cfg)
    assert q1[40].any()                                     # the confident fallback-only line was added


# ============================================================================ X4: G01 / G02 registry
def test_virtual_channels_are_registry_mismatch_not_missing():
    from gems_bench.p6.boost_diagnostics import blocker_kinds, channel_class, repair_status

    assert channel_class("virtual_channel", True, False, "virtual:thdr:mag@500m", "virtual:thdr:mag@500m") == \
        "REGISTRY_MISMATCH"
    assert channel_class("cached_layer", False, False, "x", None) == "MISSING_SOURCE"
    assert channel_class("cached_layer", True, True, "a", "b") == "SOURCE_INCOMPATIBILITY"
    C = pd.DataFrame([{"name": "g01:edge", "cls": "REGISTRY_MISMATCH"}, {"name": "g01:surface",
                                                                         "cls": "REGISTRY_MISMATCH"},
                      {"name": "m9:mag_energy@4km", "cls": "OK"}])
    U = pd.DataFrame([{"unit": "geo:2", "raster_exists": False}, {"unit": "geo:0", "raster_exists": True}])
    k = blocker_kinds(True, C, U)
    assert k["feature_registry_mismatch"] == ["g01:edge", "g01:surface"] and k["missing_saved_artifact"] == ["geo:2"]
    assert repair_status(k) == "READY_FOR_BACKFILL"         # registry mismatches never block
    assert repair_status({**k, "true_dependency_missing": ["m9:x"]}) == "BLOCKED"
    assert repair_status(blocker_kinds(False, C, U)) == "BLOCKED"


# ============================================================================ gate / restart / config
def test_recovery_gate_rules():
    from gems_bench.p6.boost_diagnostics import recovery_gate

    g = {**GATES, "geo_floor": 0.1758, "worst_floor": 0.153414, "w2_recorded_disc": 0.22464}
    good = {"disc_dti": 0.24, "disc_worst": 0.19, "geo_dti": 0.18, "deployable": True, "provenance_ok": True,
            "parity_ok": True, "recipe_ok": True, "role": "deploy_candidate"}
    ok, why, slot = recovery_gate(good, g, 0.23)
    assert ok and slot
    assert not recovery_gate({**good, "geo_dti": 0.1688}, g, 0.23)[0]          # skeleton-like geo shortfall
    assert not recovery_gate({**good, "deployable": False}, g, 0.23)[0]        # split disc / geo policy
    assert not recovery_gate({**good, "role": "historical"}, g, 0.23)[0]       # the uploaded fixed cell
    assert not recovery_gate({**good, "role": "oracle_diagnostic"}, g, 0.23)[0]
    assert not recovery_gate({**good, "disc_dti": 0.229}, g, 0.23)[0]          # must beat matched W2 too
    assert recovery_gate({**good, "disc_dti": 0.235}, g, 0.23)[2] is False      # +0.005: no leaderboard slot


def test_missing_outputs_block_reload(tmp_path):
    from gems_bench.p6.boost_diagnostics import OUTPUTS, missing_outputs

    d = tmp_path / "export_parity"
    d.mkdir()
    assert set(missing_outputs(d, "export_parity")) == set(OUTPUTS["export_parity"])
    for n in OUTPUTS["export_parity"]:
        (d / n).write_text("x")
    (d / "decision.json").write_text("")                    # empty = partially written
    assert missing_outputs(d, "export_parity") == ["decision.json"]


def test_system_set_survives_csv_float_reload():
    from gems_bench.p6.boost import system_set

    assert system_set("3;7") == {3, 7} and system_set(3.0) == {3} and system_set("nan") == set()
    assert system_set(None) == set() and system_set("") == set() and system_set(" 4 ; 9 ") == {4, 9}


def test_system_set_rejects_non_integer_ids():
    """Regression: the fixture once wrote systems = f'{zone}{rep[0]}' ('0c'), which reached float() in system_set.
    Invalid identifiers must raise clearly: never coerced to 0, never dropped, never NaN."""
    from gems_bench.p6.boost import system_set

    for bad in ("0c", "3;0c", "3.5", "inf"):
        with pytest.raises(ValueError, match="not an integer system id"):
            system_set(bad)


def test_policy_tables_refuses_the_exact_failing_systems_cell():
    from gems_bench.p6.export_parity import policy_tables

    G = _grid({("w2", 1.0, "crest_1px", 0.01): {"disc_official": 12, "geo": 20},
               ("fallback_fusion", 0.0, "crest_1px", 0.01): {"disc_official": 5, "geo": 20}})
    G.loc[G.task == "disc_official", "systems"] = G.loc[G.task == "disc_official", "zone"].astype(str) + "c"
    cfg = {"objective": "disc", "policies": {"B": {"candidates": ["w2"]}},
           "references": {"R": {"candidates": ["fallback_fusion"]}}}
    with pytest.raises(ValueError, match="'0c'"):
        policy_tables(G, cfg, "disc_official", GATES, ["disc_official", "geo"])


def test_grid_types_survive_csv_reload(tmp_path):
    """Numeric parameters stay numeric, identifiers stay strings, systems keep their integer identities after the
    CSV round trip used by every stage (read with dtype={'systems': str}, as Recovery.benchmark_grid does, or not)."""
    from gems_bench.p6.boost import system_set
    from gems_bench.p6.export_parity import evaluate_policy

    G = _grid({("w2", 1.0, "crest_1px", 0.01): {"disc_official": 12, "geo": 20},
               ("w2", 1.0, "skeleton_1px", 0.03): {"disc_official": 14, "geo": 4}})
    G.loc[(G.task == "disc_official") & (G.zone == 1), "systems"] = "7"           # single id: float on reload
    p = tmp_path / "grid.csv"
    G.to_csv(p, index=False)
    for R_ in (pd.read_csv(p, dtype={"systems": str}), pd.read_csv(p)):
        assert R_.frac.dtype.kind == "f" and R_.blend.dtype.kind == "f" and R_.zone.dtype.kind == "i"
        assert R_.rep.dtype == object and set(R_.rep) == {"crest_1px", "skeleton_1px"}
        d = R_[R_.task == "disc_official"]
        assert [system_set(s) for s in d.systems] == [system_set(s) for s in G[G.task == "disc_official"].systems]
        s0, P0 = evaluate_policy(G, "D", {"candidates": ["w2"]}, "disc", "disc_official", 0.5, ["disc_official", "geo"])
        s1, P1 = evaluate_policy(R_, "D", {"candidates": ["w2"]}, "disc", "disc_official", 0.5,
                                 ["disc_official", "geo"])
        assert s0["disc_dti"] == pytest.approx(s1["disc_dti"]) and s0["deploy_cell"] == s1["deploy_cell"]


def test_separate_distinct_cells_per_zone_count():
    """The groupby replacement keeps the exact number of distinct selection keys per zone."""
    from gems_bench.p6.export_parity import evaluate_policy

    same = _grid({("w2", 1.0, "crest_1px", 0.01): {"disc_official": 12, "geo": 20},
                  ("w2", 1.0, "skeleton_1px", 0.01): {"disc_official": 10, "geo": 4}})
    s, P = evaluate_policy(same, "D", {"candidates": ["w2"]}, "separate", "disc_official", 0.5,
                           ["disc_official", "geo"])
    assert s["deployable"] is True and s["n_distinct_cells"] == 1    # both tasks pick crest in every zone
    split = _grid({("w2", 1.0, "crest_1px", 0.01): {"disc_official": 10, "geo": 20},
                   ("w2", 1.0, "skeleton_1px", 0.01): {"disc_official": 14, "geo": 2}})
    s2, _ = evaluate_policy(split, "D", {"candidates": ["w2"]}, "separate", "disc_official", 0.5,
                            ["disc_official", "geo"])
    assert s2["deployable"] is False and s2["n_distinct_cells"] == 2


def test_recovery_config_bounded_and_defaults():
    from gems_bench.p6.boost_diagnostics import load_recovery_config
    from gems_bench.p6.run import load_p6_config

    c = load_recovery_config()
    assert c["export_parity"]["objective"] == "disc_geo_floor" and c["gates"]["worst_floor"] == pytest.approx(0.153414)
    assert c["candidate_recall"]["taxonomy"]["displaced_tol_factor"] in c["candidate_recall"]["taxonomy"][
        "sensitivity_factors"]
    assert "E_w2_uploaded_cell" in c["export_parity"]["policies"] and c["workstream"] == "wb"
    with pytest.raises(ValueError):
        load_recovery_config({"segments": {"strategies": ["H0", "H1", "H2", "H3"]}})
    with pytest.raises(ValueError):
        load_recovery_config({"export_parity": {"objective": "separate"}})
    assert load_p6_config()["stack"]["full_region"]["area_frac_source"] == "primary"   # no mixed W2 re-export
