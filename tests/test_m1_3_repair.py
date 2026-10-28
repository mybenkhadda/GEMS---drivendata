"""M1.3 repair: budget contract, forensics, Drive preflight, technical vs promotion decisions, exploratory export.
Synthetic inputs only: no Drive, competition data or GPU. Written in the authoring session; NOT executed there (the
brief forbade any local execution)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gems_bench.leaderboard_focus import config as CF
from gems_bench.leaderboard_focus import forensics as FO
from gems_bench.leaderboard_focus import gates as GT
from gems_bench.leaderboard_focus import preflight as PF
from gems_bench.leaderboard_focus import submission as SU
from gems_bench.leaderboard_focus import validation as VA

# validation/zone_budget.csv of the M1.2 Colab run (read from Drive in the authoring session, 2026-10-28)
M12_ZONES = pd.DataFrame({"zone": [0, 1, 2], "zone_px": [1887813, 1548292, 1729747],
                          "eligible_px": [1868064, 1526003, 1709351], "positive_px": [9232, 7001, 8089],
                          "frac_of_eligible": [9232 / 1868064, 7001 / 1526003, 8089 / 1709351],
                          "positive_on_known_px": [0, 0, 0]})
M12_EVAL = pd.DataFrame({"zone": [0, 1, 2], "eval_positive_px": [7648, 6009, 6806],
                         "eval_area_frac_eligible": [0.004053486, 0.003881755, 0.003935173]})


# =============================================================================== budget contract
def test_m12_budget_passes_upper_bound_and_records_old_failure():
    Z = VA.budget_rows(M12_ZONES, 0.01, 0.02, M12_EVAL)
    assert Z.ok.all() and not Z.m1_1_two_sided_ok.any()            # M1.1 rejected all three zones
    assert all("CHECK_DEFINITION_MISMATCH" in c for c in Z.cause)
    assert list(Z.pool_px) == [18681, 15261, 17094]                 # ceil(0.01 x eligible)
    assert ((Z.pool_yield > 0.45) & (Z.pool_yield < 0.5)).all()
    assert (Z.deploy_vs_eval_px > 1).all()                          # deployed pool re-ranked off mapped faults
    s = FO.root_cause_summary(Z)
    assert s["all_ok"] and s["blocking_classes"] == [] and "CHECK_DEFINITION_MISMATCH" in s["zones"]["0"]["classes"]


def test_target_vs_observed_area_allocation_failures():
    base = M12_ZONES.iloc[[0]].copy()
    over = base.assign(positive_px=19100, frac_of_eligible=19100 / 1868064)     # 1.022 % > 1.02 %
    leak = base.assign(positive_on_known_px=3)
    empty = base.assign(positive_px=0, frac_of_eligible=0.0)
    for Z, cls in ((over, "AREA_OVER_ALLOCATION"), (leak, "KNOWN_FAULT_MASK_LEAK"),
                   (empty, "AREA_UNDER_ALLOCATION")):
        out = VA.budget_rows(Z, 0.01, 0.02)
        assert not out.ok.any() and cls in out.cause.iloc[0]
        assert cls in FO.root_cause_summary(out)["blocking_classes"]


def test_budget_without_pixel_columns_keeps_old_contract_shape():
    Z = pd.DataFrame({"zone": [0, 1], "frac_of_eligible": [0.0151, 0.02], "positive_on_known_px": [0, 0]})
    assert list(VA.budget_rows(Z, 0.015, 0.02).ok) == [True, False]


def test_per_zone_integer_rounding():
    Z = pd.DataFrame({"zone": [0], "eligible_px": [101], "positive_px": [1], "frac_of_eligible": [1 / 101],
                      "positive_on_known_px": [0]})
    out = VA.budget_rows(Z, 0.01, 0.02)
    assert int(out.pool_px.iloc[0]) == 2 and int(out.cap_px.iloc[0]) == 1 and out.ok.iloc[0]
    out2 = VA.budget_rows(Z.assign(positive_px=2, frac_of_eligible=2 / 101), 0.01, 0.02)
    assert not out2.ok.iloc[0]                                      # 2 px > floor(1.0302): over the cap


def test_skeletonization_reduces_area_below_frac_and_is_accepted():
    from gems_bench.gems3 import representation as R

    yy, xx = np.mgrid[0:80, 0:80]
    score = np.exp(-((xx - 40) ** 2) / 50.0).astype(np.float32)       # one vertical ridge
    ev = np.ones(score.shape, bool)
    top = R.apply({"select": "top"}, score, ev, 0.1, eligible=ev)
    sk = R.apply({"select": "top", "post": [["skeleton", {"width_px": 0}]]}, score, ev, 0.1, eligible=ev)
    assert (sk > 0).sum() < (top > 0).sum() <= int(np.ceil(0.1 * ev.sum())) + 80   # ties may add one column
    Z = pd.DataFrame({"zone": [0], "eligible_px": [int(ev.sum())], "positive_px": [int((sk > 0).sum())],
                      "frac_of_eligible": [float((sk > 0).sum() / ev.sum())], "positive_on_known_px": [0]})
    out = VA.budget_rows(Z, 0.1, 0.02)
    assert out.ok.iloc[0] and out.pool_yield.iloc[0] < 0.5 and not out.m1_1_two_sided_ok.iloc[0]


def test_known_fault_exclusion_never_selects_known_pixels():
    from gems_bench.gems3 import representation as R

    score = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)
    ev = np.ones(score.shape, bool)
    known = np.zeros(score.shape, bool)
    known[-1] = True                                                 # the highest scores are mapped faults
    q = R.apply({"select": "top"}, score, ev, 0.1, known=known, eligible=ev & ~known)
    assert (q > 0).sum() == R._k(0.1, ev & ~known) and not ((q > 0) & known).any()   # pool counted on eligible


# =============================================================================== forensics
def test_map_comparison_attributes_masking_and_ranking_domain():
    zid = np.full((4, 6), -1, np.int16)
    zid[:, :3], zid[:, 3:] = 0, 1
    known = np.zeros(zid.shape, bool)
    known[0, 0] = True
    E = np.zeros(zid.shape, np.float32)
    D = np.zeros(zid.shape, np.float32)
    E[0, 0] = E[1, 1] = E[2, 2] = 1                                   # zone 0 evaluated
    D[1, 1] = D[3, 2] = 1                                            # zone 0 deployed: (0,0) masked, (2,2)->(3,2)
    el = ~known
    RC, code = FO.map_comparison(D, E, zid, el, known)
    z0 = RC[RC.zone == 0].iloc[0]
    assert (z0.both_px, z0.eval_only_on_mapped_fault_px, z0.eval_only_elsewhere_px, z0.deploy_only_px) == (1, 1, 1, 1)
    assert z0.deploy_outside_eligible_px == 0 and abs(z0.jaccard - 0.25) < 1e-12
    assert code[0, 0] == FO.CODE_EVAL_ONLY and code[3, 2] == FO.CODE_DEPLOY_ONLY and code[1, 1] == FO.CODE_BOTH


def test_eligible_mask_mismatch_is_explained_by_mapped_faults():
    Z = pd.DataFrame({"zone": [0, 1], "zone_px": [100, 100], "eval_eligible_px": [98, 97], "eligible_px": [95, 97],
                      "known_in_zone_px": [3, 0]})
    M = FO.mask_alignment(Z)
    assert list(M.unexplained_by_mapped_faults_px) == [0, 0]
    M2 = FO.mask_alignment(Z.assign(eligible_px=[90, 97]))
    assert list(M2.unexplained_by_mapped_faults_px) == [5, 0]       # an unexplained 5 px is visible


def test_incorrect_fold_stitching_is_reported_as_uncovered():
    from gems_bench.p6 import final_submission as FS

    fm = np.array([[0, 0, 1], [1, 2, 9]])
    valid = np.ones(fm.shape, bool)
    zid, cov = FS.zone_partition(fm, valid, [0, 1, 2])
    assert cov["uncovered_px"] == 1 and zid[1, 2] == -1
    S = FS.stitch(fm.shape, [((slice(0, 2), slice(0, 3)), zid == 0, np.full(fm.shape, 0.5)),
                             ((slice(0, 2), slice(0, 3)), zid == 1, np.full(fm.shape, 1.0))])
    assert S[0, 0] == 0.5 and S[0, 2] == 1.0 and S[1, 2] == 0.0


def test_evaluated_area_reads_geo_rows_only():
    V = pd.DataFrame({"task": ["geo", "disc_official"], "zone": [0, 0], "area_px": [7648, 6897],
                      "area_frac_eligible": [0.00405, 0.0039]})
    E = FO.evaluated_area(V)
    assert list(E.eval_positive_px) == [7648] and len(FO.evaluated_area(pd.DataFrame())) == 0


# =============================================================================== preflight
def _tif(p: Path, transform, w=4, h=3):
    import rasterio

    p.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(p, "w", driver="GTiff", width=w, height=h, count=1, dtype="float32", crs="EPSG:32611",
                       transform=transform, nodata=np.nan) as d:
        d.write(np.ones((1, h, w), np.float32))


def _pf_cfg(extra=()):
    arts = [{"id": "template", "path": "data/raw/geodawn/example_submission.tif", "required": True, "staged": False},
            {"id": "b0_maps", "path": "experiments/b0_x__1/predictions/*.tif", "required": True, "staged": False,
             "required_files": ["oof.tif", "disc0.tif"], "expected_id": "b0_x__1"},
            {"id": "boost_a_grid", "path": "{p6}/boost/benchmark/grid.csv", "required": True, "staged": True},
            {"id": "w2_maps", "path": "{p6}/stack/maps/*_[0-9].npy", "required": True, "staged": True,
             "required_files": ["geo_0.npy"]}, *extra]
    return CF.load_m1_config({"artifacts": arts, "outputs": {"sync_to_drive": False, "save_figures": False}})


def _pf_tree(tmp_path: Path, b0_transform=None) -> Path:
    from rasterio.transform import Affine

    root = tmp_path / "drive"
    t = Affine(100.0, 0, 243350.0, 0, -100.0, 4508550.0)
    _tif(root / "data/raw/geodawn/example_submission.tif", t)
    for n in ("oof.tif", "disc0.tif", "oof_fold0.tif"):                # an extra valid prediction file
        _tif(root / "experiments/b0_x__1/predictions" / n, b0_transform or t)
    g = root / "outputs/p6_fast_track/boost/benchmark/grid.csv"
    g.parent.mkdir(parents=True)
    pd.DataFrame({"task": ["geo"], "zone": [0], "candidate": ["w2"], "blend": [1.0], "rep": ["skeleton_1px"],
                  "frac": [0.01], "tp_w": [1.0], "fp_w": [1.0], "fn_w": [1.0]}).to_csv(g, index=False)
    m = root / "outputs/p6_fast_track/stack/maps"
    m.mkdir(parents=True)
    np.save(m / "geo_0.npy", np.array([[0.1, np.nan], [0.3, 0.9]], np.float32))
    return root


def test_preflight_inventory_reads_metadata_and_accepts_extra_files(tmp_path):
    root = _pf_tree(tmp_path)
    cfg = _pf_cfg()
    T = PF.inventory(root, cfg)
    fz = T[T.scope == "frozen_source"]
    assert (fz.status == PF.OK).all() and len(fz[fz.id == "b0_maps"]) == 3
    npy = json.loads(fz[fz.rel.str.endswith("geo_0.npy")].meta.iloc[0])
    assert npy["shape"] == [2, 2] and npy["nonfinite_sampled"] == 1
    tif = json.loads(fz[fz.id == "template"].meta.iloc[0])
    assert tif["epsg"] == 32611 and tif["res"] == [100.0, 100.0] and "finite_share" in tif
    assert (T[T.scope == "m1_output"].status == PF.MISSING).all()   # downstream outputs: documented, not fatal
    C = PF.compatibility(T, root, cfg, {})
    assert (C[C.critical].status == "PASS").all()


def test_preflight_detects_missing_required_file_and_misaligned_grid(tmp_path):
    from rasterio.transform import Affine

    root = _pf_tree(tmp_path, b0_transform=Affine(100.0, 0, 0.0, 0, -100.0, 4508550.0))
    (root / "experiments/b0_x__1/predictions/disc0.tif").unlink()
    cfg = _pf_cfg()
    T = PF.inventory(root, cfg)
    assert ((T.id == "b0_maps") & (T.status == PF.MISSING)).any()
    C = PF.compatibility(T, root, cfg, {})
    assert (C[C.check.str.contains("RASTER_ALIGNMENT")].status == "FAIL").any()
    assert (C[C.check == "required frozen sources readable"].status == "FAIL").all()


def test_preflight_schema_mismatch_and_empty_csvs(tmp_path):
    root = _pf_tree(tmp_path)
    out = root / "outputs/leaderboard_focus/m1_champion_recovery/validation"
    out.mkdir(parents=True)
    (out / "zone_budget.csv").write_text("zone,frac_of_eligible\n0,0.1\n", encoding="utf-8")   # schema mismatch
    (out / "per_zone_metrics.csv").write_bytes(b"")                                          # 0-byte file
    (out / "official_scores.csv").write_text("candidate\n", encoding="utf-8")                # header only
    g = root / "outputs/p6_fast_track/boost/benchmark/grid.csv"
    g.write_text("task,zone\n", encoding="utf-8")                                            # required, wrong schema
    T = PF.inventory(root, _pf_cfg())
    S = PF.schema_checks(T)
    st = dict(zip(S.id, S.status))
    assert st["validation/zone_budget.csv"] == PF.SCHEMA and st["validation/per_zone_metrics.csv"] == PF.EMPTY
    assert st["validation/official_scores.csv"] == PF.OK and st["boost_a_grid"] == PF.SCHEMA


def test_preflight_run_blocks_on_missing_required_source(tmp_path):
    root = _pf_tree(tmp_path)
    (root / "data/raw/geodawn/example_submission.tif").unlink()
    nb = {"drive_root": str(root), "local_root": str(tmp_path / "local")}
    m = PF.run(nb, {"artifacts": _pf_cfg()["artifacts"], "outputs": {"sync_to_drive": False,
                                                                      "save_figures": False}})
    assert m["status"] == CF.BLOCKED and any("required frozen sources" in r for r in m["reasons"])
    for rel in PF.OUT:
        assert (tmp_path / "local/outputs/leaderboard_focus/m1_champion_recovery" / rel).exists()
    assert not (root / "outputs/leaderboard_focus/m1_champion_recovery/diagnostics").exists()   # Drive untouched


# =============================================================================== decisions + exploratory
def _gates(**over):
    st = {"SELECTION_COMPLETE": "PASS", "DISCOVERY_ELIGIBLE": "PASS", "GEOGRAPHIC_GATE_PASS": CF.BLOCKED,
          "VALIDATION_PASS": "PASS", "PARITY_PASS": "PASS", "READY_FOR_EXPORT": CF.BLOCKED, **over}
    return pd.DataFrame({"gate": list(st), "status": list(st.values())})


def test_technical_validity_is_separate_from_internal_promotion():
    d = GT.decisions(_gates())
    assert d["technical_validity"] == "PASS" and d["internal_promotion"] == CF.BLOCKED and d["exploratory_eligible"]
    d2 = GT.decisions(_gates(PARITY_PASS="PENDING"))
    assert d2["technical_validity"] == "PENDING" and not d2["exploratory_eligible"]
    d3 = GT.decisions(_gates(VALIDATION_PASS=CF.BLOCKED))
    assert d3["technical_validity"] == CF.BLOCKED and not d3["exploratory_eligible"]
    d4 = GT.decisions(_gates(GEOGRAPHIC_GATE_PASS="PASS", READY_FOR_EXPORT="PASS"))
    assert d4["internal_promotion"] == "PASS" and not d4["exploratory_eligible"]


def test_exploratory_default_off_and_reason_mandatory():
    cfg = CF.load_m1_config()
    assert cfg["exploratory"]["enabled"] is False and cfg["promotion"]["geographic_gate"] == "mandatory"
    with pytest.raises(CF.M1ConfigError):
        CF.load_m1_config({"exploratory": {"enabled": True, "reason": "  "}})
    with pytest.raises(CF.M1ConfigError):
        CF.load_m1_config({"exploratory": {"enabled": True, "reason": "x", "prefix": cfg["submission"]["prefix"]}})
    st, why = SU.exploratory_precheck(cfg["exploratory"], {}, {}, True, True)
    assert st == CF.NOT_REQUIRED


def test_exploratory_precheck_requires_technical_validity_and_geo_only_failure():
    ex = {"enabled": True, "reason": "leaderboard baseline", "prefix": "exploratory_m1"}
    ok = {s: CF.COMPLETE for s in SU.TECH_STAGES}
    prom = {"status": CF.BLOCKED, **GT.decisions(_gates()), "gates": _gates().to_dict("records")}
    assert SU.exploratory_precheck(ex, prom, ok, True, True) == (None, [])
    st, _ = SU.exploratory_precheck(ex, prom, {**ok, "parity": CF.INCOMPLETE}, True, True)
    assert st == CF.INCOMPLETE                                       # no export when technical parity fails
    assert SU.exploratory_precheck(ex, prom, ok, True, False)[0] == CF.INCOMPLETE      # stale promotion
    bad = {"status": CF.BLOCKED, **GT.decisions(_gates(DISCOVERY_ELIGIBLE=CF.BLOCKED)), "gates": []}
    assert SU.exploratory_precheck(ex, bad, ok, True, True)[0] == CF.BLOCKED
    good = {"status": CF.COMPLETE, **GT.decisions(_gates(GEOGRAPHIC_GATE_PASS="PASS", READY_FOR_EXPORT="PASS"))}
    assert SU.exploratory_precheck(ex, good, ok, True, True)[0] == CF.NOT_REQUIRED     # use the champion path
    assert SU.exploratory_precheck(ex, prom, ok, False, True)[0] == CF.NOT_REQUIRED


def test_geographic_floor_still_blocks_the_champion_export():
    st, why = SU.precheck({"artifacts": CF.COMPLETE, "selection": CF.COMPLETE, "validation": CF.COMPLETE,
                           "parity": CF.COMPLETE, "promotion": CF.BLOCKED}, True)
    assert st == CF.BLOCKED and "promotion BLOCKED" in why[0]
    assert CF.EXPLORATORY_READY != CF.READY


def test_exploratory_readme_never_calls_it_the_champion():
    t = SU.exploratory_readme({"status": CF.EXPLORATORY_READY, "file": "exploratory_m1_w2_x.tif", "sha256": "ab",
                               "internal_promotion": CF.BLOCKED, "geo_dti": 0.1057, "geo_floor": 0.1758,
                               "technical_validity": "PASS", "reason": "baseline"})
    assert "NOT the validated champion" in t and "READY_FOR_UPLOAD" not in t


# =============================================================================== files, cache, runner
def test_submission_checksum_and_non_overwrite(tmp_path):
    from gems_bench.p6 import final_submission as FS

    p, reuse = FS.submission_path(tmp_path, "exploratory_m1_w2_r", "fp_one_123456")
    assert p.name == "exploratory_m1_w2_r.tif" and not reuse
    p.write_bytes(b"x")
    p.with_suffix(".json").write_text(json.dumps({"recipe_fp": "fp_one_123456"}), encoding="utf-8")
    assert FS.submission_path(tmp_path, "exploratory_m1_w2_r", "fp_one_123456") == (p, True)
    q, reuse = FS.submission_path(tmp_path, "exploratory_m1_w2_r", "fp_two_abcdef")
    assert q.name == "exploratory_m1_w2_r__fp_two_a.tif" and not reuse
    q.write_bytes(b"y")
    q.with_suffix(".json").write_text(json.dumps({"recipe_fp": "other"}), encoding="utf-8")
    with pytest.raises(FileExistsError):
        FS.submission_path(tmp_path, "exploratory_m1_w2_r", "fp_two_abcdef")
    a = np.array([[np.nan, 1.0]], np.float32)
    assert VA.array_sha(a) == VA.array_sha(a.copy()) and VA.array_sha(a) != VA.array_sha(np.array([[np.nan, 0.0]],
                                                                                                   np.float32))


def test_cache_invalidation_tracks_the_new_modules():
    from gems_bench.leaderboard_focus import session as SE

    assert "leaderboard_focus/forensics.py" in SE.IMPL_MODULES["validation"]
    assert SE.IMPL_MODULES["exploratory"] == SE.IMPL_MODULES["submissions"]
    assert SE.impl_fp("validation") and SE.impl_fp("preflight") is None


def _runner(tmp_path, m1=None):
    from gems_bench.leaderboard_focus.workflow import Runner

    (tmp_path / "drive").mkdir(exist_ok=True)
    (tmp_path / "local").mkdir(exist_ok=True)
    nb = {"drive_root": str(tmp_path / "drive"), "local_root": str(tmp_path / "local"), "sync_to_drive": False,
          "save_figures": False}
    return Runner(nb, "deadbeef", m1)


def test_runner_exploratory_disabled_needs_no_engine(tmp_path):
    R = _runner(tmp_path)
    m = R.step("exploratory")
    assert m["status"] == CF.NOT_REQUIRED and R.sess is None


def test_runner_refuses_engine_stages_after_blocked_preflight(tmp_path):
    R = _runner(tmp_path)
    R.results["preflight"] = {"status": CF.BLOCKED, "reasons": ["required frozen sources readable: template MISSING"]}
    m = R.step("validation")
    assert m["status"] == CF.INCOMPLETE and "preflight BLOCKED" in m["reasons"][0] and R.sess is None
