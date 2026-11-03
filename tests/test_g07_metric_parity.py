"""G07 metric parity: canonical weighted TP / FP / FN, exact area in evaluation rows, pooled components, audit."""
import numpy as np
import pandas as pd
import pytest

from gems_bench.gems3.official import Unit
from gems_bench.g06.splits import ValidationUnit
from gems_bench.g07 import evaluation as EV
from gems_bench.metric import dti_components

CFG = {"representation": {"tie_break": "secondary_then_hash"}, "notebook": {"seed": 1}, "evaluation": {}}


def unit():
    ev = np.ones((50, 50), bool)
    truth = np.zeros((50, 50), bool)
    truth[5:45, 20] = True
    known = np.zeros((50, 50), bool)
    known[5:45, 40] = True
    u = Unit("disc_official", 0, (slice(0, 50), slice(0, 50)), ev, truth, known, False, known)
    return ValidationUnit("V1", 0, u, u, known, np.where(truth, 1, -1).astype(np.int32))


def test_weighted_components_match_reference():
    vu = unit()
    q = np.zeros((50, 50), np.float32)
    q[5:30, 21] = 1.0
    q[10:12, 0:5] = 0.6
    a = vu.unit.components(q)
    b = dti_components(q, vu.unit.truth, eval_mask=vu.unit.ev, exclude_mask=vu.unit.exclude, exclude_supports_tp=False)
    assert (a.tp, a.fp, a.fn) == pytest.approx((b.tp, b.fp, b.fn), rel=1e-9)


def test_rows_carry_exact_area_even_with_ties():
    vu = unit()
    proto = EV.unit_protocol(vu, np.ones((50, 50), bool))
    s = np.ones((50, 50))                                                  # every score tied
    rows, srows = EV.evaluate_map(vu, s, {"top": {"select": "top", "post": []}}, [0.02], proto, {"model": "M"}, CFG)
    r = rows[0]
    k = int(np.ceil(0.02 * proto["elig"].sum()))
    assert r["area_px"] == k == r["sel_actual_px"] and r["sel_legacy_px"] == proto["elig"].sum()
    assert r["masked_area_px"] == 0
    E = EV.exact_budget_ok(pd.DataFrame(rows))
    assert bool(E.exact.iloc[0])


def test_exact_budget_audit_flags_inflation():
    R = pd.DataFrame({"model": ["A", "A"], "sel_policy": ["g07.rep.1"] * 2, "sel_k": [10, 10],
                      "sel_actual_px": [10, 14], "sel_shortfall": [0, 0], "sel_n_tied_at_cutoff": [1, 5]})
    assert not bool(EV.exact_budget_ok(R).exact.iloc[0])


def test_pooled_dti_from_summed_components():
    R = pd.DataFrame({"model": "M", "scenario": "V1", "rep": "top", "frac": 0.01, "task": "disc_official",
                      "zone": [0, 1], "protocol_key": ["a", "b"], "tp_w": [10.0, 2.0], "fp_w": [5.0, 50.0],
                      "fn_w": [2.0, 20.0], "area_px": 10, "eligible_px": 100, "prediction_mass": 10.0})
    S = EV.aggregate_components(R)
    assert S.dti.iloc[0] == pytest.approx(EV.dti(12.0, 55.0, 22.0))


def test_candidate_recall_is_geometric_and_per_pool():
    vu = unit()
    pools = {"good": np.zeros((50, 50), bool), "bad": np.zeros((50, 50), bool)}
    pools["good"][5:45, 21] = True
    pools["bad"][5:45, 2] = True
    R = EV.evaluate_candidate_recall(vu, np.zeros((50, 50), np.int32), pools).set_index("source")
    assert R.loc["pool:good", "recall"] == 1.0 and R.loc["pool:bad", "recall"] == 0.0
    assert R.loc["union_pools", "recall"] == 1.0
