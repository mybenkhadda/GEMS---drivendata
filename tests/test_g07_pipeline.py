"""G07 orchestration: training gate, feasibility metrics, strict stage checks, registry completeness, partial evaluation,
promotion of exploratory results (synthetic; no engine, no real data)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gems_bench.g07 import crossfit as CX
from gems_bench.g07 import evaluation as EV
from gems_bench.g07 import persistence as PS
from gems_bench.g07.contracts import GridSpec
from gems_bench.g07.pipeline import G07Pipeline, StageError

GRID = GridSpec(48, 48, "EPSG:32611", (100.0, 0.0, 500000.0, 0.0, -100.0, 4004800.0), 100.0)


# =============================================================================== gate
@pytest.mark.parametrize("decision,mode,n,status,tier", [
    ("GO", "STRICT_RESEARCH", 5, "READY", "STRICT_ELIGIBLE"),
    ("NO_GO", "STRICT_RESEARCH", 5, "BLOCKED", "NONE"),
    ("INCONCLUSIVE", "STRICT_RESEARCH", 5, "BLOCKED", "NONE"),
    ("NO_GO", "EXPLORATORY_TRAINING", 5, "READY", "SCREENING_ONLY"),
    ("INCONCLUSIVE", "EXPLORATORY_TRAINING", 5, "READY", "SCREENING_ONLY"),
    ("GO", "EXPLORATORY_TRAINING", 5, "READY", "SCREENING_ONLY"),
    ("BLOCKED_INPUT", "EXPLORATORY_TRAINING", 5, "BLOCKED", "NONE"),        # exploratory never bypasses invalid inputs
    ("BLOCKED_INPUT", "STRICT_RESEARCH", 0, "BLOCKED", "NONE"),
    ("NO_GO", "EXPLORATORY_TRAINING", 0, "BLOCKED", "NONE"),                # no valid tile: nothing to explore
])
def test_training_gate_matrix(decision, mode, n, status, tier):
    g = EV.training_gate({"decision": decision, "reasons": ["x"]}, mode, n)
    assert g["status"] == status and g["result_tier"] == tier
    assert g["gpu_training_can_proceed"] is (status == "READY")
    assert g["reason"] and g["corrective_action"] and g["input_requirements"] and g["decision"] == decision


def test_gate_without_a_decision_blocks():
    g = EV.training_gate(None, "EXPLORATORY_TRAINING", 5)
    assert g["status"] == "BLOCKED" and "DEM_FEASIBILITY" in g["corrective_action"]


def test_strict_block_explains_the_exploratory_route_and_keeps_thresholds():
    g = EV.training_gate({"decision": "NO_GO", "reasons": ["v1_incremental_systems_vs_w2"]}, "STRICT_RESEARCH", 3)
    assert "require_go" in g["corrective_action"] and "lower" in g["corrective_action"]


def test_blocked_input_carries_the_validation_remedy():
    g = EV.training_gate({"decision": "BLOCKED_INPUT", "reasons": ["r"]}, "STRICT_RESEARCH", 0,
                         {"remedy": ["set vertical_unit_override"]})
    assert "vertical_unit_override" in g["corrective_action"]


# =============================================================================== promotion of exploratory results
def promotion_inputs(strict):
    rows = [{"model": "M", "scenario": s, "rep": "top", "frac": 0.01, "dti": 0.3, "dti_w2": 0.2, "delta": 0.1,
             "worst_zone_dti": 0.2, "worst_zone_dti_w2": 0.1, "delta_worst": 0.1} for s in ("V1", "V3")]
    sysrep = pd.DataFrame([{"model": "M", "scenario": "V1", "rep": "top", "frac": 0.01, "n_new": 3, "n_lost": 0}])
    cfg = {"min_gain_v1": 0.005, "max_zone_drop": 0.01, "min_v3_delta": -0.005, "min_net_systems": 1}
    return pd.DataFrame(rows), sysrep, {"M": strict}, cfg, {"rep": "top", "frac": 0.01}, \
        pd.DataFrame({"model": ["M"], "exact": [True]}), True


def test_exploratory_result_cannot_be_promoted_however_good():
    assert EV.evaluate_promotion(*promotion_inputs("SCREENING_ONLY")).status.iloc[0] == "PROMISING_SCREENING_ONLY"
    assert EV.evaluate_promotion(*promotion_inputs("STRICT_ISOLATED")).status.iloc[0] == "PROMOTE_CANDIDATE"


# =============================================================================== feasibility metrics
FCFG = {"feasibility": {"recall_fracs": [0.01, 0.02], "primary_recall_frac": 0.01,
                        "go_if": {"min_eligible_share": 0.05, "min_systems": 2, "min_auc": 0.6,
                                  "min_incremental_systems": 2, "max_rho_with_coverage": 0.5}}}
COV = pd.DataFrame({"region": ["eligible"], "local_dem_share": [0.5]})
FACTS = pd.DataFrame({"scenario": ["V1", "V1", "V3"], "unit": ["V1:0", "V1:1", "V3:0"], "n_systems": [2, 2, 50]})


def signal(*rows):
    return pd.DataFrame([{"feature": f, "auc_abs": a, "rho_with_coverage": r, "is_reference": False} for f, a, r in rows])


def recall(*rows):
    return pd.DataFrame([{"source": s, "scenario": u.split(":")[0], "unit": u, "frac": f, "covered_ids": c}
                         for s, u, f, c in rows])


GOOD_RECALL = recall(("pool:dem_score", "V1:0", 0.01, "1;2"), ("pool:w2", "V1:0", 0.01, ""))


def test_negative_coverage_correlation_is_an_artefact_too():
    D = EV.feasibility_decision(COV, signal(("a", 0.9, -0.9), ("b", 0.55, 0.1)), FACTS, GOOD_RECALL, True, True, FCFG)
    assert D["criteria"]["v1_best_clean_feature_auc"]["value"] == pytest.approx(0.55) and D["decision"] == "NO_GO"
    D2 = EV.feasibility_decision(COV, signal(("a", 0.9, -0.1)), FACTS, GOOD_RECALL, True, True, FCFG)
    assert D2["decision"] == "GO"


def test_same_numeric_system_in_two_units_are_two_structures():
    R = recall(("pool:dem_score", "V1:0", 0.01, "3"), ("pool:dem_score", "V1:1", 0.01, "3"),
               ("pool:w2", "V1:0", 0.01, ""), ("pool:w2", "V1:1", 0.01, ""))
    D = EV.feasibility_decision(COV, signal(("a", 0.9, 0.0)), FACTS, R, True, True, FCFG)
    assert D["criteria"]["v1_incremental_systems_vs_w2"]["value"] == 2


def test_w2_recovery_in_another_unit_does_not_cancel_a_system():
    R = recall(("pool:dem_score", "V1:0", 0.01, "3"), ("pool:w2", "V1:1", 0.01, "3"))
    D = EV.feasibility_decision(COV, signal(("a", 0.9, 0.0)), FACTS, R, True, True, FCFG)
    assert D["criteria"]["v1_incremental_systems_vs_w2"]["value"] == 1


def test_v3_is_reported_never_pooled_into_the_gate():
    R = recall(("pool:dem_score", "V3:0", 0.01, "1;2;3"), ("pool:w2", "V3:0", 0.01, ""))
    D = EV.feasibility_decision(COV, signal(("a", 0.9, 0.0)), FACTS, R, True, True, FCFG)
    assert D["criteria"]["v1_incremental_systems_vs_w2"]["value"] == 0
    assert D["v3_report"]["incremental_systems"] == 3 and D["v3_report"]["n_systems"] == 50
    assert D["criteria"]["v1_systems_in_coverage"]["value"] == 4            # V3's 50 systems never count


def test_only_the_primary_fraction_gates():
    R = recall(("pool:dem_score", "V1:0", 0.02, "1;2"), ("pool:w2", "V1:0", 0.02, ""))
    D = EV.feasibility_decision(COV, signal(("a", 0.9, 0.0)), FACTS, R, True, True, FCFG)
    assert D["criteria"]["v1_incremental_systems_vs_w2"]["value"] == 0


def test_blocked_input_reports_why_every_file_was_excluded():
    D = EV.feasibility_decision(COV, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), True, False, FCFG,
                                {"exclusion_codes": {"UNKNOWN_VERTICAL_UNIT": 7}, "remedy": ["set the override"]})
    assert D["decision"] == "BLOCKED_INPUT" and "UNKNOWN_VERTICAL_UNIT" in D["reasons"][0]
    assert D["remedy"] == ["set the override"]


# =============================================================================== require_stage
def bare_pipeline(tmp_path, **status):
    P = G07Pipeline.__new__(G07Pipeline)
    P.root = tmp_path
    P.status = status
    return P


def test_valid_statuses_pass(tmp_path):
    P = bare_pipeline(tmp_path, a=PS.COMPLETE, b=PS.REUSED)
    PS.save_manifest(tmp_path, "a", PS.COMPLETE, "fp", [], {"note": "n"})
    assert P.require_stage("a")["status"] == PS.COMPLETE
    assert P.require_stage("b") == {}


@pytest.mark.parametrize("status,info,word", [
    (PS.FAILED, {"error": "boom", "traceback": "line 7"}, "boom"),
    (PS.BLOCKED_INPUT, {"note": "no DEM"}, "no DEM"),
    (PS.BLOCKED_UPSTREAM, {"reason": "upstream [('datasets', 'FAILED')]"}, "datasets"),
    (PS.NOT_REQUESTED, {}, "not a success"),
    (PS.PARTIAL, {"note": "2/3 fits"}, "2/3 fits"),
    (PS.NO_GO, {"note": "decision NO_GO"}, "NO_GO"),
    (PS.NOT_RUN, {}, "did not run"),
])
def test_every_other_status_raises_with_a_reason(tmp_path, status, info, word):
    P = bare_pipeline(tmp_path, s=status)
    PS.save_manifest(tmp_path, "s", status, "fp", [], info)
    with pytest.raises(StageError, match=word):
        P.require_stage("s")


def test_allow_list_is_explicit(tmp_path):
    P = bare_pipeline(tmp_path, s=PS.PARTIAL)
    P.require_stage("s", allow=(PS.PARTIAL,))
    with pytest.raises(StageError):
        P.require_stage("s", allow=(PS.NO_GO,))


def test_a_stage_that_never_ran_is_not_success(tmp_path):
    with pytest.raises(StageError):
        bare_pipeline(tmp_path).require_stage("train_surface_unet")


# =============================================================================== registry completeness
def unit(key):
    ev = np.ones((6, 6), bool)
    return SimpleNamespace(key=key, scenario=key.split(":")[0], unit=SimpleNamespace(ev=ev))


UNITS = {k: unit(k) for k in ("V1:0", "V1:1", "V1:2", "V3:0", "V2:0")}


def write_fit(root, exp, fid, key, fp="abc", arr=None):
    from gems_bench.g06.datasets import unit_file
    from gems_bench.g06.models_pixel import oof_path

    p = oof_path(root, exp, fid, key, fp)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, np.full((6, 6), 0.5, np.float32) if arr is None else arr)
    return {"experiment": exp, "fit_id": fid, "status": "COMPLETE", "fingerprint": fp, "strictness": "STRICT_ISOLATED",
            "device": "cpu", "model_dir": f"models/{exp}/{unit_file(fid)}"}


def loader(_d):
    return SimpleNamespace(schema_hash="s")


def expected():
    return {"E": ["V1:0", "V1:1", "V1:2"]}


def test_missing_fold_is_detected(tmp_path):
    R = pd.DataFrame([write_fit(tmp_path, "E", f"V1:{d}", f"V1:{d}") for d in (0, 1)])
    V = CX.validate_model_registry(tmp_path, R, expected(), UNITS, loader, "s")
    assert V.valid.tolist() == [True, True, False] and V.registry_status.iloc[2] == "MISSING"
    E = CX.summarize_experiments(V)
    assert E.status.iloc[0] == "PARTIAL" and E.n_valid_fits.iloc[0] == 2 and E.invalid_fits.iloc[0] == "V1:2"


def test_complete_registry_passes(tmp_path):
    R = pd.DataFrame([write_fit(tmp_path, "E", f"V1:{d}", f"V1:{d}") for d in range(3)])
    V = CX.validate_model_registry(tmp_path, R, expected(), UNITS, loader, "s")
    assert V.valid.all() and CX.summarize_experiments(V).status.iloc[0] == "COMPLETE"


@pytest.mark.parametrize("arr,word", [(np.zeros((5, 6), np.float32), "shape"),
                                      (np.full((6, 6), np.nan, np.float32), "no finite"),
                                      (np.full((6, 6), 1.7, np.float32), "leaves"),
                                      (np.full((6, 6), np.inf, np.float32), "infinite")])
def test_invalid_prediction_arrays_fail_the_fit(tmp_path, arr, word):
    R = pd.DataFrame([write_fit(tmp_path, "E", "V1:0", "V1:0", arr=arr)])
    V = CX.validate_model_registry(tmp_path, R, {"E": ["V1:0"]}, UNITS, loader, "s")
    assert not V.valid.iloc[0] and word in V.problem.iloc[0]


def test_missing_prediction_file_and_unreadable_model_fail(tmp_path):
    from gems_bench.g06.models_pixel import oof_path

    row = write_fit(tmp_path, "E", "V1:0", "V1:0")
    oof_path(tmp_path, "E", "V1:0", "V1:0", "abc").unlink()

    def broken(_d):
        raise OSError("corrupt")

    V = CX.validate_model_registry(tmp_path, pd.DataFrame([row]), {"E": ["V1:0"]}, UNITS, broken, "s")
    assert "missing" in V.problem.iloc[0] and "unreadable" in V.problem.iloc[0]


def test_schema_mismatch_and_missing_strictness_fail(tmp_path):
    row = write_fit(tmp_path, "E", "V1:0", "V1:0")
    V = CX.validate_model_registry(tmp_path, pd.DataFrame([row]), {"E": ["V1:0"]}, UNITS, loader, "other")
    assert "schema" in V.problem.iloc[0]
    row["strictness"] = None
    V = CX.validate_model_registry(tmp_path, pd.DataFrame([row]), {"E": ["V1:0"]}, UNITS, loader, "s")
    assert "strictness" in V.problem.iloc[0]


def test_failed_and_blocked_fits_keep_their_reason(tmp_path):
    R = pd.DataFrame([{"experiment": "E", "fit_id": "V1:0", "status": "FAILED", "detail": "OOM"},
                      {"experiment": "E", "fit_id": "V1:1", "status": "BLOCKED_INPUT", "detail": "no val patch"}])
    V = CX.validate_model_registry(tmp_path, R, {"E": ["V1:0", "V1:1"]}, UNITS, loader)
    assert V.problem.tolist() == ["OOM", "no val patch"] and not V.valid.any()
    assert CX.summarize_experiments(V).status.iloc[0] == "FAILED"


def test_fit_prediction_units():
    assert CX.fit_prediction_units("V1:1", UNITS) == ["V1:1"]
    assert CX.fit_prediction_units("V3:0", UNITS) == ["V3:0", "V2:0"]
    assert CX.fit_prediction_units("V1:0|2", UNITS) == ["V1:2"]
    assert CX.fit_prediction_units("FULL", UNITS) == []


def test_partial_folds_never_look_fully_evaluated():
    maps = {"W2": {k: 1 for k in UNITS}, "PART": {"V1:0": 1, "V1:1": 1}, "UNET": {"V1:0": 1, "V1:1": 1, "V1:2": 1}}
    E = EV.evaluation_status(maps, UNITS, {"W2": "REFERENCE", "PART": "STRICT_ISOLATED", "UNET": "SCREENING_ONLY"},
                             ["V1", "V3"]).set_index(["model", "scenario"])
    assert E.loc[("PART", "V1"), "status"] == "PARTIALLY_EVALUATED" and E.loc[("PART", "V1"), "missing_units"] == "V1:2"
    assert E.loc[("UNET", "V1"), "status"] == "FULLY_EVALUATED"
    assert E.loc[("UNET", "V1"), "validation_tier"] == "SCREENING_ONLY"
    assert E.loc[("UNET", "V3"), "status"] == "NOT_EVALUATED"
    assert E.loc[("PART", "V1"), "validation_tier"] == "STRICTLY_VALIDATED"
    assert E.loc[("W2", "V1"), "validation_tier"] == "REFERENCE"


# =============================================================================== surface patch artifacts
UNET = {"channels": ["a", "b"], "res_m": 10.0, "patch_cells": 4, "fits": ["V1:0"], "min_region_share": 0.5}


def write_patches(d, n=4, channels=("a", "b"), res=10.0, cells=4, complete=True, shape=None):
    from gems_bench.g06.persistence import write_json, write_table

    d.mkdir(parents=True, exist_ok=True)
    P = cells * int(round(100 / res))
    np.save(d / "s.npy", np.zeros(shape or (n, len(channels), P, P), np.float16))
    dem = d / "coarse.tif"
    dem.write_bytes(b"x")
    idx = pd.DataFrame([{"patch_id": f"p{i}", "r0": 0, "c0": i * cells, "cells": cells, "dem_share": 1.0}
                        for i in range(n)])
    write_table(d / "patch_index.parquet", idx)
    write_json(d / "surface_manifest.json", {"memmap": "s.npy", "channels": list(channels), "res_m": res, "cells": cells,
                                             "n_patches": n, "dem_path": str(dem), "fingerprint": "h",
                                             "complete": complete})


def test_valid_patch_dataset_has_no_problem(tmp_path):
    from gems_bench.g07.datasets import validate_surface_patch_artifacts as check

    write_patches(tmp_path / "p")
    man, idx, probs = check(tmp_path / "p", UNET, GRID)
    assert not probs and len(idx) == 4 and man["fingerprint"] == "h"


@pytest.mark.parametrize("kw,word", [({"channels": ("b", "a")}, "channels"), ({"res": 20.0}, "resolution"),
                                     ({"shape": (3, 2, 40, 40)}, "shape"), ({"complete": False}, "manifest")])
def test_inconsistent_patch_dataset_is_reported(tmp_path, kw, word):
    from gems_bench.g07.datasets import validate_surface_patch_artifacts as check

    write_patches(tmp_path / "p", **kw)
    _, _, probs = check(tmp_path / "p", UNET, GRID)
    assert any(word in p for p in probs)


def test_patch_outside_the_grid_is_reported(tmp_path):
    from gems_bench.g06.persistence import write_table
    from gems_bench.g07.datasets import validate_surface_patch_artifacts as check

    write_patches(tmp_path / "p")
    idx = pd.read_parquet(tmp_path / "p" / "patch_index.parquet").assign(c0=46)
    write_table(tmp_path / "p" / "patch_index.parquet", idx)
    assert any("outside" in p for p in check(tmp_path / "p", UNET, GRID)[2])


def test_missing_patches_block_the_unet_cleanly(tmp_path):
    from gems_bench.g07.datasets import surface_patch_readiness, validate_surface_patch_artifacts as check

    man, idx, probs = check(tmp_path / "absent", UNET, GRID)
    assert man is None and probs
    R = surface_patch_readiness(["V1:0", "V1:1"], idx, man, probs, None, UNET)
    assert R.status.tolist() == ["BLOCKED", "BLOCKED"] and "BUILD_DATASETS" in R.reason.iloc[0]


def test_readiness_row_per_fit(tmp_path):
    from gems_bench.g07.datasets import surface_patch_readiness, validate_surface_patch_artifacts as check

    write_patches(tmp_path / "p")
    man, idx, probs = check(tmp_path / "p", UNET, GRID)
    visible = np.zeros(GRID.shape, bool)
    visible[1, :] = True
    inner = np.where(np.indices(GRID.shape)[1] < 8, 0, 1).astype(np.int8)
    vis = SimpleNamespace(fit_id="V1:0", visible=visible, region=np.ones(GRID.shape, bool))
    R = surface_patch_readiness(["V1:0"], idx, man, probs, lambda f: (vis, inner), UNET).iloc[0]
    assert R.status == "READY" and R.n_train == 2 and R.n_val == 2 and R.n_train_pos == 2 and R.n_val_pos == 2
    assert R.input_channels == 2 and R.res_m == 10.0 and R.mean_dem_share == 1.0


# =============================================================================== exact budget of ready-made selections
def test_portfolio_selection_passes_the_exact_budget_audit():
    from gems_bench.g07.representation import apply_policy

    ev = np.ones((50, 50), bool)
    q = np.zeros((50, 50), np.float32)
    q[:5, :5] = 1.0                                                         # exactly 1 % of 2500
    _, rep = apply_policy({"select": "raw", "post": []}, q, ev, 0.01, eligible=ev)
    assert rep["policy"].startswith("g07.rep") and rep["actual_px"] == rep["k"] == 25 and rep["shortfall"] == 0
    q[6, 6] = 1.0
    _, rep2 = apply_policy({"select": "raw", "post": []}, q, ev, 0.01, eligible=ev)
    R = pd.DataFrame([{"model": "D2", **{f"sel_{k}": v for k, v in rep2.items()}}])
    assert not bool(EV.exact_budget_ok(R).exact.iloc[0])
