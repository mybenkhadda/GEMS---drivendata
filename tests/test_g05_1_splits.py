"""G05.1 split discipline: training rows, disjointness, nesting audit, run ids, RUN validation."""
import numpy as np
import pandas as pd
import pytest

from gems_bench.g05_1 import config as CF
from gems_bench.g05_1 import crossfit as XF
from gems_bench.g05_1 import selftest as ST


def _T():
    return pd.DataFrame({"zone": [0, 0, 1, 1, 2, 2], "row_full": [1, 2, 50, 51, 90, 91],
                         "col_full": [1, 2, 50, 51, 90, 91], "eligible_share": [1, 1, 1, 0, 1, 1]})


def test_training_mask_excludes_held_zone_buffer_and_ineligible():
    T = _T()
    zm = np.zeros((100, 100), bool)
    zm[85:95, 85:95] = True                          # zone 2 dilated by the buffer
    tr = XF.training_mask(T, 2, zm)
    assert list(tr) == [True, True, True, False, False, False]


def test_training_mask_buffer_removes_other_zone_rows_inside():
    T = _T()
    zm = np.zeros((100, 100), bool)
    zm[48:53, 48:53] = True                          # held zone 0's buffer reaches the zone-1 rows
    tr = XF.training_mask(T, 0, zm)
    assert not tr[:4].any() and tr[4:].all()


def test_nesting_audit_learned_is_screening():
    A = XF.nesting_audit([("V1:0", "V1", 0)], ["G0", "S1", "G1"])
    cls = A[A.check == "classification"].set_index("model").status
    assert cls["G1"] == XF.SCREENING and cls["S1"] == XF.STRICT_FREE and cls["G0"] == XF.STRICT_FREE
    assert (A[(A.model == "G1") & A.check.str.contains("independent")].status == "FAIL_STRICT").all()


def test_retraining_plan_pairs():
    P = XF.retraining_plan([0, 1, 2], [0, 1, 2])
    assert len(P) == 12 and (P.status == XF.BLOCKED).all()
    assert not ((P.outer_zone == P.inner_training_zone).any())


def test_run_id_deterministic():
    ST.t_run_id()


@pytest.mark.parametrize("run,msg", [({"selection": True}, "needs RUN['training']"),
                                     ({"export": True}, "needs RUN['reconstruction']"),
                                     ({"bogus": True}, "unknown RUN keys")])
def test_run_validation(run, msg):
    with pytest.raises(CF.G051ConfigError, match=msg.replace("[", r"\[").replace("]", r"\]")):
        CF.notebook({}, run)


def test_manual_upload_cannot_be_disabled():
    with pytest.raises(CF.G051ConfigError):
        CF.notebook({"manual_upload_only": False}, {})


def test_first_run_defaults():
    _, r = CF.notebook({}, {})
    assert r["preflight"] and r["atlas"] and r["recall"] and r["registry"] and r["graph"]
    assert not (r["training"] or r["strict"] or r["ablations"] or r["reconstruction"] or r["export"] or r["selection"])
