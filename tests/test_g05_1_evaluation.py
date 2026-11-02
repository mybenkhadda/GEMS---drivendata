"""G05.1 evaluation helpers: recall tables, recovery matrix, promotion gates, atlas categories."""
import numpy as np
import pandas as pd

from gems_bench.g05_1 import atlas as AT
from gems_bench.g05_1 import recall as RC
from gems_bench.g05_1 import selection as SL
from gems_bench.g05_1 import selftest as ST


def rows():
    base = {"scenario": "V1", "kind": "expert", "role": "final", "tp_w": 1.0, "fp_w": 1.0, "fn_w": 1.0,
            "n_truth": 2, "n_systems": 3, "area_frac_eligible": 0.01}
    return pd.DataFrame([{**base, "unit": "V1:0", "construction": "W2", "frac": 0.01, "systems": "1;2",
                          "n_recovered": 2},
                         {**base, "unit": "V1:0", "construction": "Full", "kind": "union_max_rank", "frac": 0.01,
                          "systems": "1;3", "n_recovered": 2, "tp_w": 2.0},
                         {**base, "unit": "V1:0", "construction": "Full-minus-CNN", "kind": "union_max_rank",
                          "frac": 0.01, "systems": "1", "n_recovered": 1}])


def test_recall_union():
    ST.t_recall_union()


def test_new_lost():
    N = RC.new_lost(rows())
    f = N[N.construction == "Full"].iloc[0]
    assert f.n_new == 1 and f.n_lost == 1 and f.new_ids == "3" and f.lost_ids == "2"


def test_pooled_sums_components_not_dtis():
    P = RC.pooled(rows())
    w = P[P.construction == "W2"].iloc[0]
    assert np.isclose(w.dti, 1.0 / (1.0 + 0.2 + 0.8 + 1e-6))


def test_marginal():
    M = RC.marginal(RC.pooled(rows()))
    r = M.iloc[0]
    assert r.expert == "CNN" and r.marginal_systems == 1 and r.marginal_dti > 0


def test_recovery_matrix_targets():
    Mx = RC.recovery_matrix(rows(), {"V1:0": [1, 2, 3]}, {0: [3]})
    t = Mx[Mx.target]
    assert set(t.system_id) == {3}
    assert bool(t[t.construction == "Full"].recovered.iloc[0]) and not bool(t[t.construction == "W2"].recovered.iloc[0])


def test_expert_scores_rank_inside_unit():
    T = pd.DataFrame({"mean_cnn_b0": [0.1, 0.5, 0.9], "mean_cnn_b1": [0.9, 0.5, 0.1]})
    E = RC.expert_scores(T, {"CNN": ["cnn_b0", "cnn_b1", "fusion"], "None": ["absent"]})
    assert np.allclose(E.CNN, [2 / 3, 2 / 3, 2 / 3]) and E["None"].isna().all()


def test_promotion_gates():
    ST.t_promotion()


def test_attach_systems():
    P = pd.DataFrame([{"unit": "V1:0", "model": "S1", "rep": "top_area", "frac": 0.01, "tp_w": 1.0}])
    G = pd.DataFrame([{"unit": "V1:0", "model": "S1", "rep": "top_area", "frac": 0.01, "systems": "4",
                       "n_recovered": 1, "n_systems": 3}])
    assert SL.attach_systems(P, G).systems.iloc[0] == "4"


def test_decision_table_serialises_gates():
    S, P, rS, rP = ST.decision_frames()
    d = SL.promotion("S1", "STRICT_LABEL_FREE", S, P, rS, rP, {"dti": 0.1}, {"dti": 0.1}, ST.PC,
                     {"provenance_verified": True, "upstream_honesty_verified": True})
    D = SL.decision_table([d])
    assert "gates" in D and isinstance(D.gates.iloc[0], str)


def test_atlas_categories_and_controls():
    assert AT.category(False, False, 0, False) == "NOT_FOUND_IN_UNIT"
    assert AT.category(True, False, 0, False) == "NOT_IN_INVENTORY"
    assert AT.category(True, True, 0, False) == "IN_INVENTORY_NOT_SELECTED"
    assert AT.category(True, True, 0.2, False) == "PARTIAL_SELECTION"
    assert AT.category(True, True, 0.9, True) == "RECOVERED"
    A = pd.DataFrame({"unit": "V1:0", "system_id": [1, 2, 3, 4], "length_km": [5.0, 4.0, 9.0, 5.5]})
    rec = pd.DataFrame({"unit": "V1:0", "system_id": [2, 3, 4], "recovered": [True, True, False]})
    assert AT.matched_controls(A, rec, "V1:0", 1, 2) == [2, 3]
    assert AT.hypothesis({"category": "NOT_IN_INVENTORY"}).startswith("HYPOTHESIS")
