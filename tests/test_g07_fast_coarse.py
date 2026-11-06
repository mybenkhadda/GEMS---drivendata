"""Authoring-only synthetic checks for the cached G07 FAST 10m research track.

These are NOT run during code authoring; run selectively in Colab.
"""
import numpy as np

from gems_bench.g07.fast_coarse import DIRECTIONS, FEATURES, extract_fast_features
from gems_bench.g07.dti_targets import visible_fault_soft_target


def test_fast_feature_schema_and_finite_interior():
    y, x = np.mgrid[:256, :256]
    # Smooth synthetic topography with a central artificial slope break.
    z = (0.02 * x + 0.005 * y + 1.7 * (x > 128)).astype(np.float32)
    features = extract_fast_features(z, 10.0)
    assert tuple(features) == FEATURES
    assert set(DIRECTIONS) == set(FEATURES)
    for name, values in features.items():
        assert values.shape == z.shape
        assert np.isfinite(values[128, 128]), name


def test_fast_features_do_not_create_fault_edges_from_nodata():
    y, x = np.mgrid[:256, :256]
    z = (0.02 * x + 0.005 * y).astype(np.float32)
    z[120:136, 120:136] = np.nan
    f = extract_fast_features(z)
    for name, a in f.items():
        assert np.isnan(a[127, 127]), name
        assert np.isnan(a[119, 127]), name  # guard the gap boundary
        assert np.isfinite(a[30, 30]), name


def test_soft_target_uses_only_training_visible_fault_labels():
    known = np.zeros((31, 31), bool)
    known[15, 15] = True
    mask = np.ones((31, 31), bool)
    mask[:3] = False
    t, m = visible_fault_soft_target(known, mask, pixel_m=100., radius_m=300.)
    assert t[15, 15] == 1.
    assert np.isclose(t[15, 16], 2./3.)
    assert t[15, 18] == 0.
    assert (t[~m] == 0).all()


def test_masked_withheld_fault_does_not_enter_soft_target():
    truth = np.zeros((31, 31), bool)
    truth[15, 15] = True
    visible = np.ones((31, 31), bool)
    visible[12:19, 12:19] = False
    t, _ = visible_fault_soft_target(truth, visible, pixel_m=100.)
    assert np.all(t == 0)
