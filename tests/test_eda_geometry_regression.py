"""Geometry numerical stability (gems_eda.morphology geom.2) - the Colab e03 failure
``ProductInvalid: tables/fault_geometry.csv: schema rules violated ['strike_in_0_180', 'linearity_in_0_1']``
(19 strikes == 180.0, one linearity 1.0000000000000002). Synthetic only; NOT executed at authoring time."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gems_eda.morphology import GEOMETRY_REV, GeometryPrecisionError, axial_deg, path_geometry, pca_axis


def _grid():
    from gems_eda.discovery.selftest import synthetic_grid

    return synthetic_grid(60, 60)


@pytest.mark.parametrize("theta,expected", [(180.0, 0.0), (-1e-14, 0.0), (360.0, 0.0), (179.9999999999, 0.0),
                                            (90.0, 90.0), (-90.0, 90.0), (270.0, 90.0), (45.0, 45.0)])
def test_axial_deg_range(theta, expected):
    a = axial_deg(theta)
    assert 0.0 <= a < 180.0 and abs(a - expected) < 1e-9


def test_axial_deg_vectorised():
    a = axial_deg(np.array([180.0, 0.0, -0.0, 359.999999999999, 1e-300]))
    assert ((a >= 0) & (a < 180)).all() and a.shape == (5,)


@pytest.mark.parametrize("pts,strike", [
    ((np.arange(10.0), np.zeros(10)), 90.0),                                   # E-W (east varies)
    ((np.zeros(10), np.arange(10.0)), 0.0),                                    # N-S: must be 0, never 180
    ((np.arange(10.0), np.arange(10.0)), 45.0),                                # NE
    ((np.arange(10.0), -np.arange(10.0)), 135.0),                              # SE
])
def test_pca_axis_straight_traces(pts, strike):
    s, lin = pca_axis(*pts)
    assert 0 <= s < 180 and min(abs(s - strike), 180 - abs(s - strike)) < 1e-6
    assert 0.0 <= lin <= 1.0 and lin > 1 - 1e-9                               # collinear: linearity exactly <= 1


def test_pca_axis_curved_and_loop():
    t = np.linspace(0, np.pi, 50)
    s, lin = pca_axis(np.cos(t), np.sin(t))                                    # half circle
    assert 0 <= s < 180 and 0 < lin < 1
    t = np.linspace(0, 2 * np.pi, 50)
    s, lin = pca_axis(np.cos(t), np.sin(t))                                    # closed loop: isotropic
    assert 0 <= s < 180 and 0 <= lin < 0.1


def test_pca_axis_degenerate_covariance():
    assert all(np.isnan(v) for v in pca_axis(np.ones(5), np.ones(5)))          # identical points


def test_pca_axis_tiny_negative_eigenvalue_corrected(monkeypatch):
    real = np.linalg.eigh

    def fake(cov):
        w, v = real(cov)
        return np.array([-1e-14 * w[1], w[1]]), v                              # machine-precision negative
    monkeypatch.setattr(np.linalg, "eigh", fake)
    s, lin = pca_axis(np.arange(10.0), np.zeros(10))
    assert lin == 1.0                                                          # 1.0000000000000002 -> 1.0


def test_pca_axis_substantial_negative_eigenvalue_raises(monkeypatch):
    real = np.linalg.eigh

    def fake(cov):
        w, v = real(cov)
        return np.array([-1e-3 * w[1], w[1]]), v                               # a real defect
    monkeypatch.setattr(np.linalg, "eigh", fake)
    with pytest.raises(GeometryPrecisionError):
        pca_axis(np.arange(10.0), np.zeros(10))


def test_path_geometry_bounds_on_many_paths():
    g = _grid()
    rng = np.random.default_rng(0)
    for k in range(200):
        n = int(rng.integers(3, 40))
        ang = rng.uniform(0, np.pi)
        r = np.round(30 + np.arange(n) * np.sin(ang)).astype(int)
        c = np.round(30 + np.arange(n) * np.cos(ang)).astype(int)
        path = np.unique(np.c_[r.clip(0, 59), c.clip(0, 59)], axis=0)
        if len(path) < 2:
            continue
        out = path_geometry(path, g)
        assert 0 <= out["azimuth"] < 180, out
        assert np.isnan(out["linearity"]) or 0 <= out["linearity"] <= 1, out
        assert out["chord_km"] <= out["length_km"] + 1e-9


def test_vertical_two_pixel_path_is_zero_not_180():
    out = path_geometry(np.array([[10, 5], [11, 5]]), _grid())               # southward step: atan2 = 180
    assert out["azimuth"] == 0.0


def test_fault_geometry_rules_still_strict():
    from gems_eda.discovery import products as PR

    T = pd.DataFrame([{**dict.fromkeys(PR.GEOMETRY_COLUMNS, 0), "seg_id": 0, "system_id": 1, "length_km": 1.0,
                       "chord_km": 1.0, "strike_deg": 180.0, "linearity": 1.0000000000000002}])
    v = PR.rule_violations(T, PR.GEOMETRY_RULES)
    assert {"strike_in_0_180", "linearity_in_0_1"} <= set(v)                  # the table rule never clips


def test_network_and_products_consistent():
    """The fault network, fault_geometry.csv and its provenance agree (ids, axial strike, revision)."""
    from gems_eda.discovery import products as PR
    from gems_eda.discovery import selftest as ST

    class _E:
        pass
    import tempfile
    from pathlib import Path

    from gems_bench.leaderboard_focus.session import Workspace

    with tempfile.TemporaryDirectory() as d:
        E = _E()
        E.D = ST.synthetic_data()
        E.ws = Workspace(Path(d))
        E.domains = lambda: np.zeros(E.D.grid.shape, np.int16)
        T = PR.write_fault_geometry(E, PR.fault_geometry(E))
        assert list(T.seg_id) == list(E.D.net.segments.seg_id)
        assert (T.strike_deg.to_numpy() == E.D.net.segments.azimuth.to_numpy()).all()
        assert E.ws.read_json("tables/fault_geometry.provenance.json")["geometry_rev"] == GEOMETRY_REV
