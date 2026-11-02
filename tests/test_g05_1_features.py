"""G05.1 registry / band / label features (synthetic; no real data)."""
import numpy as np
import pandas as pd
import pytest

from gems_bench.g05_1 import bands as BD
from gems_bench.g05_1 import displacement as DP
from gems_bench.g05_1 import rankers as RK
from gems_bench.g05_1 import registry as RG
from gems_bench.g05_1 import selftest as ST


def test_exact_band_resolution_refuses_substring():
    ST.t_bands_exact()
    rows = BD.resolution_table(["iso_grav_anom", "nope"], ["iso_grav_anom_hg - g", "iso_grav_anom - a"])
    assert rows[0]["status"] == "RESOLVED" and rows[0]["band_index"] == 2
    assert rows[1]["status"] == "MISSING"


def test_identifier_strips_description():
    assert BD.identifier("rtp - reduced to pole") == "rtp" and BD.identifier(" tmi_hg ") == "tmi_hg"


def test_feature_columns_exclude_base_and_truth():
    ST.t_leakage_guard()


def test_feature_columns_ablation_drops_family():
    R = pd.DataFrame({c: [0.0] for c in ("mean_cnn_b0", "mean_mag_hgrad", "g_support", "chain_length_km")})
    f = RG.feature_columns(R, drop_families=["D2_magnetic"])
    assert "mean_mag_hgrad" not in f and "mean_cnn_b0" in f and "chain_length_km" in f


def test_dictionary_and_provenance_cover_every_feature():
    D, P = RG.dictionary(), RG.provenance()
    assert set(RG.FAMILY) <= set(D.feature) and set(RG.FAMILY) == set(P.feature)
    assert (D[D.feature == "y_near_truth"].label_free == False).all()  # noqa: E712


def test_local_contrast_zero_on_flat_field():
    a = np.ones((20, 20), np.float32)
    ev = np.ones_like(a, bool)
    assert np.allclose(RG.local_contrast(a, ev, 3), 0.0, atol=1e-6)


def test_persistence_count_monotone():
    lab = ST.line_lab((40, 40), (((5, 5), (5, 30)), ((30, 5), (30, 30))))
    w2 = np.zeros((40, 40), np.float32)
    w2[5, 5:31] = 1.0
    el = np.ones_like(w2, bool)
    pc = RG.persistence_count(lab, w2, el, [0.01, 0.02, 0.05])
    assert pc[0] >= pc[1] and pc[0] >= 1


def test_labels_classes():
    ST.t_labels()


def test_strict_blend_never_penalises_missing_evidence():
    ST.t_strict_blend()


def test_segment_pass_columns():
    lab = ST.line_lab()
    xc = {"rtp": DP.prepare(ST.shifted_field())}
    D = RG.segment_pass(lab, [1], {}, xc, ST.DC, set(), {"survey_axes_deg": [0, 90], "survey_axis_tol_deg": 2.0,
                                                          "short_km": 0.5})
    assert {"chain_length_km", "xc_rtp", "survey_artifact", "short"} <= set(D.columns)
    assert bool(D.survey_artifact.iloc[0])            # strike 90 = a grid axis: flagged, never a label


@pytest.mark.parametrize("col", ["y_near_truth", "truth_share", "system_id"])
def test_assert_label_free_rejects(col):
    with pytest.raises(RG.LeakageError):
        RG.assert_label_free([col])


def test_evidence_rank_is_unitwise():
    T = pd.DataFrame({"unit": ["a", "a", "b", "b"], "g_support": [1.0, 2.0, 100.0, 200.0]})
    e = RK.evidence_rank(T)
    assert np.allclose(e, [0.5, 1.0, 0.5, 1.0])
