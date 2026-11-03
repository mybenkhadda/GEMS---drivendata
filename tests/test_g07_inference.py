"""G07 full-region construction, export parity and the gated submission writer (synthetic)."""
import numpy as np
import pandas as pd

from gems_bench.g07 import inference as INF


def setup(n=60):
    rng = np.random.default_rng(0)
    w2 = rng.random((n, n)).astype(np.float32)
    el = np.ones((n, n), bool)
    labels = np.zeros((n, n), bool)
    labels[:, 30] = True
    el &= ~labels
    return w2, el, labels


def test_w2_exact_construction_and_parity():
    w2, el, labels = setup()
    q, rep = INF.construct_final_prediction(w2, el, 0.02, "w2_exact")
    par = INF.verify_export_parity(q, el, 0.02, labels)
    assert par["status"] == "PASS" and par["checks"]["nothing_on_known_faults"]
    assert set(np.unique(q)) <= {0.0, 1.0}


def test_portfolio_construction_exact():
    w2, el, labels = setup()
    lab = np.zeros(w2.shape, np.int32)
    lab[5, 2:12] = 1
    U = pd.DataFrame({"label": [1], "utility": [3.0]})
    q, rep = INF.construct_final_prediction(w2, el, 0.02, "portfolio", lab, U, 0.8)
    assert INF.verify_export_parity(q, el, 0.02, labels)["status"] == "PASS"
    assert rep["added_candidates"] == 1


def test_parity_detects_budget_and_known_fault_violations():
    w2, el, labels = setup()
    q, _ = INF.construct_final_prediction(w2, el, 0.02, "w2_exact")
    q2 = q.copy()
    q2[0, 30] = 1.0                                                        # on a known fault
    par = INF.verify_export_parity(q2, el, 0.02, labels)
    assert par["status"] == "FAIL" and not par["checks"]["nothing_on_known_faults"]
    q3 = q.copy()
    q3[np.flatnonzero(q3.ravel() == 0)[:5] // q3.shape[1], np.flatnonzero(q3.ravel() == 0)[:5] % q3.shape[1]] = 1.0
    assert not INF.verify_export_parity(q3, el, 0.02, labels)["checks"]["exact_budget"]


def test_writer_refuses_when_any_gate_fails(tmp_path):
    w2, el, labels = setup()
    q, _ = INF.construct_final_prediction(w2, el, 0.02, "w2_exact")
    res = INF.write_submission_candidate(q, None, tmp_path / "c.tif", {}, {"EXPORT_SUBMISSION": True,
                                                                          "full_region_parity": False})
    assert res["status"] == "REFUSED" and res["uploaded"] is False and not (tmp_path / "c.tif").exists()
