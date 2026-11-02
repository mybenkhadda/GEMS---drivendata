"""G05.1 geometry, graph and displacement-style evidence on synthetic traces."""
import numpy as np

from gems_bench.g05_1 import displacement as DP
from gems_bench.g05_1 import geometry as GM
from gems_bench.g05_1 import graph as GR
from gems_bench.g05_1 import selftest as ST


def test_line_geometry():
    ST.t_geometry_line()


def test_diagonal_geometry():
    ST.t_geometry_diagonal()


def test_path_order_endpoints_on_bent_trace():
    lab = np.zeros((30, 30), np.int32)
    lab[5, 5:20] = 1
    lab[5:20, 19] = 1
    rr, cc = np.nonzero(lab)
    o = GM.path_order(rr, cc)
    ends = {(int(rr[o[0]]), int(cc[o[0]])), (int(rr[o[-1]]), int(cc[o[-1]]))}
    assert ends == {(5, 5), (19, 19)}
    g = GM.features(rr, cc)
    assert g["tortuosity"] > 1.3 and g["curvature_deg_per_km"] > 0


def test_survey_artifact_axes():
    s = np.array([0.5, 45.0, 89.0, 179.5, np.nan])
    assert list(GM.survey_artifact(s, [0, 90], 2.0)) == [True, False, True, True, False]


def test_axial_dispersion_zero_for_constant():
    assert GM.axial_dispersion_deg([30.0] * 10) < 1e-6


def test_graph_continuity_and_isolation():
    ST.t_graph_continuity()


def test_regional_agreement_parallel_vs_perpendicular():
    lab = ST.line_lab(segs=(((10, 10), (10, 40)), ((20, 10), (20, 40)), ((30, 10), (30, 40))))
    G = GM.geometry_table(lab)
    assert np.allclose(GR.regional_agreement(G, 10.0), 1.0)
    lab2 = ST.line_lab(segs=(((10, 10), (10, 40)), ((15, 60), (45, 60))))
    G2 = GM.geometry_table(lab2)
    assert np.allclose(GR.regional_agreement(G2, 10.0), -1.0)


def test_displacement_recovers_known_shift():
    ST.t_displacement_shift()


def test_displacement_negative_control():
    ST.t_displacement_control()


def test_best_shift_symmetry():
    x = np.sin(np.arange(60) / 3.0)
    k, cb, c0 = DP.best_shift(x[3:], x[:-3], 8)       # A[s] = x[s + 3] = B[s + 3]
    assert k == 3 and cb > 0.999 and c0 < cb


def test_sample_nan_aware():
    f = np.ones((5, 5))
    f[2, 2] = np.nan
    v = DP.sample(DP.prepare(f), np.array([0.0, 2.0, 2.5, 10.0]), np.array([0.0, 2.0, 2.0, 0.0]))
    assert v[0] == 1.0 and np.isnan(v[1]) and np.isnan(v[2]) and np.isnan(v[3])
