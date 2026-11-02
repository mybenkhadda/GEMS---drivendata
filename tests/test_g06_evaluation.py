"""G06 evaluation: canonical DTI parity, actual-area accounting, pooled components, W2 comparison, promotion gates,
nested representation selection (synthetic units on the real GEMS scorer)."""
import numpy as np
import pandas as pd
import pytest

from gems_bench.gems3.official import Unit
from gems_bench.g06 import evaluation as EV
from gems_bench.g06.splits import ValidationUnit
from gems_bench.metric import dti_components


def unit(seed=0, zone=0, task="disc_official"):
    rng = np.random.default_rng(seed)
    ev = np.ones((60, 60), bool)
    truth = np.zeros((60, 60), bool)
    truth[10:50, 20] = True
    truth[30, 30:55] = True
    known = np.zeros((60, 60), bool)
    known[5:55, 45] = True
    u = Unit(task, zone, (slice(0, 60), slice(0, 60)), ev, truth, known, False, known)
    sysmap = np.where(truth, 1, -1).astype(np.int32)
    sysmap[30, 30:55] = 2
    vu = ValidationUnit("V1", zone, u, u, known, sysmap)
    score = rng.random((60, 60)).astype(np.float32)
    score[truth] += 1.0
    return vu, score


def test_fast_scorer_matches_reference_metric():
    vu, score = unit()
    q = (score > 1.2).astype(np.float32)
    a = vu.unit.components(q)
    b = dti_components(q, vu.unit.truth, eval_mask=vu.unit.ev, exclude_mask=vu.unit.exclude,
                       exclude_supports_tp=False)
    assert a.tp == pytest.approx(b.tp, rel=1e-9) and a.fp == pytest.approx(b.fp, rel=1e-9)
    assert a.fn == pytest.approx(b.fn, rel=1e-9) and a.dti == pytest.approx(b.dti, rel=1e-9)


def test_grid_rows_area_and_systems():
    vu, score = unit()
    proto = EV.unit_protocol(vu, np.ones((60, 60), bool))
    rows, srows = EV.evaluate_prediction_grid(vu, score, {"top": {"select": "top", "post": []}}, [0.02], proto,
                                              {"model": "M"})
    r = rows[0]
    k = int(np.ceil(0.02 * proto["elig"].sum()))
    assert r["area_px"] == k                                                   # actual final area accounted
    assert r["masked_area_px"] == 0                                            # nothing on the known faults
    assert r["dti"] == pytest.approx(EV.dti(r["tp_w"], r["fp_w"], r["fn_w"]))
    assert {s["system_id"] for s in srows} == {1, 2}


def test_pooled_dti_sums_components():
    R = pd.DataFrame({"model": "M", "scenario": "V1", "rep": "top", "frac": 0.01, "task": "disc_official",
                      "zone": [0, 1], "protocol_key": ["a", "b"], "tp_w": [10.0, 1.0], "fp_w": [10.0, 100.0],
                      "fn_w": [5.0, 50.0], "area_px": [10, 10], "eligible_px": [100, 100],
                      "prediction_mass": [10.0, 10.0]})
    S = EV.aggregate_components(R)
    assert S.dti.iloc[0] == pytest.approx(EV.dti(11.0, 110.0, 55.0))
    mean_of_zones = np.mean([EV.dti(10, 10, 5), EV.dti(1, 100, 50)])
    assert S.dti.iloc[0] != pytest.approx(mean_of_zones)
    assert S.worst_zone.iloc[0] == 1


def test_protocol_mismatch_refused():
    from gems_bench.m7.protocol import ProtocolMismatch

    R = pd.DataFrame({"model": ["A", "B"], "scenario": "V1", "rep": "top", "frac": 0.01, "task": "disc_official",
                      "zone": [0, 0], "protocol_key": ["a", "z"], "tp_w": 1.0, "fp_w": 1.0, "fn_w": 1.0,
                      "area_px": 1, "eligible_px": 10, "prediction_mass": 1.0})
    with pytest.raises(ProtocolMismatch):
        EV.aggregate_components(R)


def _summary(dti_m, dti_w2, worst_m, worst_w2, v3_m, v3_w2):
    return pd.DataFrame([
        {"model": "W2", "scenario": "V1", "rep": "top", "frac": 0.01, "dti": dti_w2, "worst_zone_dti": worst_w2},
        {"model": "M", "scenario": "V1", "rep": "top", "frac": 0.01, "dti": dti_m, "worst_zone_dti": worst_m},
        {"model": "W2", "scenario": "V3", "rep": "top", "frac": 0.01, "dti": v3_w2, "worst_zone_dti": v3_w2},
        {"model": "M", "scenario": "V3", "rep": "top", "frac": 0.01, "dti": v3_m, "worst_zone_dti": v3_m}])


CFG = {"min_gain_v1": 0.005, "recommend_gain_v1": 0.01, "max_zone_drop": 0.01, "min_v3_delta": -0.005,
       "min_net_systems": 1}
PRIM = {"rep": "top", "frac": 0.01}
SR = pd.DataFrame({"model": ["M"], "scenario": ["V1"], "rep": ["top"], "frac": [0.01], "n_new": [2], "n_lost": [0]})


def test_promotion_gates():
    cmp = EV.compare_with_w2(_summary(0.30, 0.28, 0.20, 0.20, 0.13, 0.13))
    d = EV.promotion_decision(cmp, SR, {"M": "STRICT_NESTED"}, CFG, PRIM)
    assert d.status.iloc[0] == "PROMOTE_CANDIDATE"
    d = EV.promotion_decision(cmp, SR, {"M": "SCREENING_ONLY"}, CFG, PRIM)
    assert d.status.iloc[0] == "PROMISING_SCREENING_ONLY"
    cmp2 = EV.compare_with_w2(_summary(0.30, 0.28, 0.17, 0.20, 0.13, 0.13))     # worst zone drops 0.03
    assert EV.promotion_decision(cmp2, SR, {"M": "STRICT_NESTED"}, CFG, PRIM).status.iloc[0] == "RETAIN_W2"
    cmp3 = EV.compare_with_w2(_summary(0.30, 0.28, 0.20, 0.20, 0.10, 0.13))     # geographic guard fails
    assert EV.promotion_decision(cmp3, SR, {"M": "STRICT_NESTED"}, CFG, PRIM).status.iloc[0] == "RETAIN_W2"


def test_nested_policy_selection_uses_other_zones():
    rows = []
    for z, best in ((0, "a"), (1, "a"), (2, "b")):
        for rep in ("a", "b"):
            tp = 10.0 if rep == best else 5.0
            rows.append({"model": "M", "scenario": "V1", "zone": z, "rep": rep, "frac": 0.01, "tp_w": tp,
                         "fp_w": 10.0, "fn_w": 10.0})
    T = EV.nested_policy_selection(pd.DataFrame(rows), "M", "V1")
    chosen = dict(zip(T.zone, T.chosen_rep))
    assert chosen[2] == "a"                     # zone 2's own best ('b') is never chosen with its own truth
    assert chosen[0] in ("a", "b") and "nested_pooled_dti" in T


def test_system_report_new_and_lost():
    S = pd.DataFrame({"model": ["W2", "W2", "M", "M"], "unit": "V1:0", "rep": "top", "frac": 0.01,
                      "system_id": [1, 2, 1, 2], "recovered": [True, False, False, True]})
    R = EV.system_recovery_report(S)
    m = R[R.model == "M"].iloc[0]
    assert m.n_new == 1 and m.n_lost == 1 and m.new_ids == "2" and m.lost_ids == "1"
