"""G06 supervision: PU classes from visible labels only, label scope, segment / connection targets."""
import numpy as np
import pandas as pd
import pytest

from gems_bench.g06 import supervision as SU
from gems_bench.g06.contracts import LeakageError

CFG = {"ignore_km": 0.3, "likely_background_km": 1.0, "target_dist_cap_km": 2.0}


def test_pixel_classes_from_visible_labels():
    lab = np.zeros((50, 50), bool)
    lab[:, 10] = True                      # visible fault
    hidden = np.zeros_like(lab)
    hidden[:, 40] = True                   # withheld fault: NOT passed to the supervision
    region = np.ones_like(lab)
    region[:, 45:] = False
    sup = SU.build_pixel_supervision(lab, region, np.ones_like(lab), CFG)
    c = sup["classes"]
    assert (c[:, 10] == SU.POSITIVE).all()
    assert (c[:, 11:13] == SU.IGNORE).all()                 # <= 300 m halo of a visible fault
    assert (c[:, 14:20] == SU.UNLABELED).all()
    assert (c[:, 30] == SU.LIKELY_BACKGROUND).all()
    assert (c[:, 40] == SU.LIKELY_BACKGROUND).all()         # a hidden fault is just an unlabelled pixel
    assert (c[:, 45:] == SU.IGNORE).all()                   # outside the training region
    assert sup["target_dist_km"].max() <= 2.0


def test_hidden_truth_does_not_change_classes():
    lab = np.zeros((40, 40), bool)
    lab[:, 5] = True
    a = SU.build_pixel_supervision(lab, np.ones_like(lab), np.ones_like(lab), CFG)["classes"]
    b = SU.build_pixel_supervision(lab, np.ones_like(lab), np.ones_like(lab), CFG)["classes"]
    assert (a == b).all()


def test_label_visibility_rules():
    SU.validate_label_visibility("segment", ["kn_known_dist_km_mean", "geo_mean_x"], ["target"])
    with pytest.raises(LeakageError):
        SU.validate_label_visibility("pixel", ["mag_rtp_hgm_s300", "known_dist_km"])
    with pytest.raises(LeakageError):
        SU.validate_label_visibility("pixel", ["tip_dist_km"])
    with pytest.raises(LeakageError):
        SU.validate_label_visibility("segment", ["near_share", "geo_mean_x"], ["target"])


def test_segment_supervision_classes():
    lab = np.zeros((30, 30), np.int32)
    lab[5, 2:12] = 1                   # on truth
    lab[20, 2:12] = 2                  # far
    lab[10, 2:12] = 3                  # half near
    truth = np.zeros((30, 30), bool)
    truth[5, :] = True
    truth[13, 2:5] = True              # 3 of label 3's 10 pixels within 3 px -> ambiguous (0.1 <= 0.3 < 0.5)
    L = SU.build_segment_supervision(lab, truth, {"tol_px": 3, "positive_share": 0.5, "negative_share": 0.1})
    m = dict(zip(L.label, L.cls))
    assert m[1] == SU.SEG_POS and m[2] == SU.SEG_NEG and m[3] == SU.SEG_AMB
    assert np.isnan(L.set_index("label").loc[3, "target"])


def test_connection_targets():
    lab = np.zeros((30, 30), np.int32)
    lab[5, 2:10] = 1
    lab[5, 14:22] = 2
    lab[25, 2:10] = 3
    truth = np.zeros((30, 30), bool)
    truth[5, 0:25] = True
    sysm = np.where(truth, 7, -1)
    L = SU.build_segment_supervision(lab, truth, {"tol_px": 2, "positive_share": 0.5, "negative_share": 0.1})
    dom = SU.dominant_system(lab, sysm, truth, 2)
    assert dom[0] == 7 and dom[1] == 7 and dom[2] == -1
    P = pd.DataFrame({"label_a": [1, 1], "label_b": [2, 3]})
    out = SU.build_connection_supervision(P, L, dom, np.array([1.0, 0.0]), {"gap_positive_share": 0.5})
    assert list(out.target) == [1.0, 0.0]


def test_orientation_targets_axial():
    lab = np.zeros((40, 40), bool)
    lab[:, 20] = True                  # N-S trace
    o = SU.build_orientation_targets(lab, 2.0, 0.3)
    c2 = np.nanmean(o["orient_cos2"])
    assert c2 == pytest.approx(-1.0, abs=0.05)              # strike 90 deg from east -> cos 2theta = -1
