"""Synthetic tests of the fault-discovery reset (gems_bench/reset). No data, no Drive, no GPU.

Run by notebooks/colab/GEMS_FAULT_DISCOVERY_RESET.ipynb stage [00] (enforced). Written, not run locally.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")
pytest.importorskip("skimage")

from gems_bench.geomoe import decide as DE  # noqa: E402
from gems_bench.reset import atlas as AT  # noqa: E402
from gems_bench.reset import audit as AU  # noqa: E402
from gems_bench.reset import benchmark as BM  # noqa: E402
from gems_bench.reset import physics as PH  # noqa: E402
from gems_bench.reset import run as RR  # noqa: E402
from gems_bench.reset import selftest as RST  # noqa: E402


def test_runtime_selftests_all_pass():
    T = RST.run_selftests()
    assert not len(T[T.status != "PASS"]), T[T.status != "PASS"].to_dict("records")


def test_reset_config_guards():
    with pytest.raises(RR.ResetConfigError):
        RR.reset_config({"challenger": {"train_large_neural_model": True}})
    with pytest.raises(RR.ResetConfigError):
        RR.reset_config({"dem": {"allow_download": True}})              # needs dem.enabled
    with pytest.raises(RR.ResetConfigError):
        RR.reset_config({"mode": "TRAIN_EVERYTHING"})
    with pytest.raises(RR.ResetConfigError):
        RR.reset_config({"export": {"manual_upload_only": False}})
    c = RR.reset_config({"mode": "resume"})
    assert c["mode"] == "RESUME" and c["reuse_cached"] and not c["dem"]["enabled"]


def test_full_mode_never_trains_or_downloads_by_default():
    c = RR.reset_config({"mode": "FULL"})
    assert not c["dem"]["enabled"] and not c["dem"]["allow_download"] and not c["export"]["exploratory"]
    assert "09_export" in RR.MODES["FULL"] and "06_benchmark" not in RR.MODES["ATLAS"]


def test_reset_dependencies_use_its_own_tables():
    st = {"00_tests": DE.COMPLETE, "03_scenarios": DE.COMPLETE, "04_atlas": DE.FAILED}
    D, dec = RR.FaultDiscoveryReset.DEPENDS, RR.FaultDiscoveryReset.DECISION_STAGES
    assert DE.dependency_status("05_physics", st, True, D, dec)[0] == DE.BLOCKED_UPSTREAM
    assert DE.dependency_status("08_challenger", {"06_benchmark": DE.FAILED}, True, D, dec)[0] == DE.NOT_EVALUATED
    assert DE.dependency_status("04_atlas", {"00_tests": DE.TESTS_FAILED, "03_scenarios": DE.COMPLETE}, True, D,
                                dec)[0] == DE.BLOCKED_UPSTREAM


def test_yaml_config_loads():
    pytest.importorskip("yaml")
    rc = RR.load_reset_config({"atlas": {"n_examples": 40}})
    assert rc["atlas"]["n_examples"] == 40 and set(rc["benchmark"]["exportable_representations"]) <= set(
        rc["benchmark"]["representations"])
    assert set(rc["atlas"]["eval_regions"]) <= set(range(rc["atlas"]["region_grid"][0] * rc["atlas"]["region_grid"][1]))


def test_eda_audit_detects_the_recorded_defects():
    class R:
        drive = Path("/nonexistent")

    eda = Path(__file__).resolve().parents[1] / AU.EDA
    if not eda.exists():
        pytest.skip("EDA report not in this checkout")
    assert AU.verify(R, "bound_method_repr")[0] == "CONFIRMED_BUG"
    assert AU.verify(R, "saturated_evidence")[0] == "CONFIRMED_SATURATION"
    assert AU.verify(R, "header_grid")[0] == "CONFIRMED"
    assert {p[3] for p in AU.PRIOR} <= {AU.V, AU.R_, AU.Q, AU.H, AU.NA}


def test_region_ids_and_split():
    r = AT.region_ids(np.array([0, 50, 99]), np.array([0, 50, 99]), (0, 100, 0, 100), (3, 3))
    assert list(r) == [0, 4, 8]
    assert AT.review_split(4, [0, 4, 8]) == "EVAL" and AT.review_split(1, [0, 4, 8]) == "TRAIN"


def test_unreviewed_is_never_a_negative():
    assert AT.label_class("NOT_REVIEWED", "EVIDENCE_ONLY", "", 1.0)[1] == "UNVERIFIED"
    assert AT.label_class("INSUFFICIENT_EVIDENCE", "EVIDENCE_ONLY", "", 1.0)[1] == "UNVERIFIED"
    assert AT.label_class("LIKELY_CONTACT", "EVIDENCE_ONLY", "", "LOW", ("LIKELY_CONTACT",))[1] == \
        "UNVERIFIED_NONFAULT"


def test_ramp_has_equal_mass_to_binary_budget():
    rng = np.random.default_rng(3)
    s = rng.random((80, 80))
    el = rng.random((80, 80)) > 0.2
    q = BM.ramp_map(s, el, 0.01)
    assert abs(q.sum() / el.sum() - 0.01) < 2e-4


def test_complementarity_counts():
    SY = pd.DataFrame({"scenario": ["V1"] * 4, "model": [BM.REF, BM.REF, "A1_label_free", "A1_label_free"],
                       "unit": ["u"] * 4, "system_id": [1, 2, 1, 3], "recovered": [True, False, False, True]})
    C = BM.complementarity(SY).set_index("model")
    assert C.loc["A1_label_free", "recovered_not_by_w2"] == 1 and C.loc["A1_label_free", "w2_only"] == 1


def test_localization_and_profiles():
    a = np.tile(np.r_[np.zeros(20), np.ones(20)], (40, 1))
    p = PH.perpendicular_profile(a, 20.0, 20.0, np.pi / 2, 10)                    # trace along rows
    assert len(p) == 21 and np.isfinite(p.value).all()
    st = PH.step_and_texture(a, np.arange(5, 35, dtype=float), np.full(30, 19.5), np.pi / 2, 5)
    assert st["step_z"] > 1.0


def test_lineament_score_is_label_free_and_bounded():
    r1, r3 = np.random.default_rng(0).random((10, 10)), np.random.default_rng(1).random((10, 10))
    s = PH.lineament_score({"mag": [r1, r3]}, {"mag": np.full((10, 10), 0.5)})
    assert np.allclose(s, 0.5 * np.fmin(r1, r3))
    with pytest.raises(ValueError):
        PH.lineament_score({}, {})
