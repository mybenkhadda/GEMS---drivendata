"""Tests of the exhaustive geological discovery EDA (gems_eda.discovery). Synthetic inputs only; run by stage 00 of
notebooks/colab/GEMS_EXHAUSTIVE_GEOLOGICAL_EDA.ipynb (a failure blocks every downstream stage)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from gems_eda.discovery import selftest as ST

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "notebooks/colab/GEMS_EXHAUSTIVE_GEOLOGICAL_EDA.ipynb"


@pytest.mark.parametrize("name,critical,fn", ST.TESTS, ids=[t[0] for t in ST.TESTS])
def test_selftest(name, critical, fn):
    fn()


def test_run_selftests_table():
    T = ST.run_selftests()
    assert list(T.columns) == ["test", "critical", "status", "detail"] and len(T) == len(ST.TESTS)


def test_stage_tables_consistent():
    from gems_eda.discovery.run import STAGE_SWITCH, ExhaustiveEDA

    S = set(ExhaustiveEDA.STAGES)
    assert set(ExhaustiveEDA.DEPENDS) == S and set(ExhaustiveEDA.LABELS) == S
    assert all(set(d) <= S for d in ExhaustiveEDA.DEPENDS.values())
    assert set(ExhaustiveEDA.OUTPUTS) <= S and set(STAGE_SWITCH) <= S
    order = {s: i for i, s in enumerate(ExhaustiveEDA.STAGES)}
    assert all(order[d] < order[s] for s, ds in ExhaustiveEDA.DEPENDS.items() for d in ds)


def test_figure_ids_unique_and_parts_complete():
    from gems_eda.discovery import parts_ab, parts_cd, parts_efg, parts_hi

    fns = (parts_ab.PART_A + parts_ab.PART_B + parts_cd.PART_C + parts_cd.PART_D + parts_efg.PART_E +
           parts_efg.PART_F + parts_efg.PART_G + parts_hi.PART_H + parts_hi.PART_I + parts_hi.PART_DEM)
    names = [f.__name__ for f in fns]
    assert len(names) == len(set(names)) and len(names) >= 90


def test_hypothesis_and_question_sources_exist():
    from gems_eda.discovery import synthesis as SY

    src = "".join(p.read_text(encoding="utf-8") for p in (REPO / "gems_eda/discovery").glob("parts_*.py"))
    cited = {f for _, _, figs in SY.HYPOTHESES for f in figs} | {f for _, _, figs in SY.QUESTIONS for f in figs}
    missing = [f for f in sorted(cited) if f'"{f}"' not in src]
    assert not missing, f"figures cited by the synthesis but never produced: {missing}"
    assert len(SY.QUESTIONS) == 15 and len({q for q, _, _ in SY.QUESTIONS}) == 15


# =============================================================================== gde.1-r2 regression tests (A - J)
# Synthetic, deterministic, no Drive / GeoTIFF / model weights. Each docstring names the Colab failure it protects.
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


class _Fake:
    """An ExhaustiveEDA-like object for the pure producers: workspace, config, seed, synthetic data, domains."""

    def __init__(self, root: Path):
        from gems_bench.leaderboard_focus.session import Workspace
        from gems_eda.discovery import core as K

        self.ws = Workspace(root)
        self.c = K.eda_config({"figures": {"format": ["png"], "dpi": 50}})
        self.seed, self.n_boot, self.min_effect = 2026, 50, 0.1
        self.D = ST.synthetic_data()
        self.current_stage = "test"
        self.fps = {}

    def table(self, rel):
        p = self.ws.path(rel)
        return pd.DataFrame() if not p.exists() else (pd.read_parquet(p) if p.suffix == ".parquet"
                                                      else self.ws.load_table(rel))

    def domains(self):
        return np.zeros(self.D.grid.shape, np.int16)


def _runner(root: Path):
    """A real ExhaustiveEDA stage machine without __init__ (no factory, no Drive, no data load)."""
    from gems_bench.leaderboard_focus.session import Workspace
    from gems_eda.discovery import core as K
    from gems_eda.discovery.run import ExhaustiveEDA

    E = ExhaustiveEDA.__new__(ExhaustiveEDA)
    E.c = K.eda_config({})
    E.nb = {"reuse_cached": True, "strict_dependency_checks": True}
    E.cfg, E.head, E.drive, E.out = {}, "test", root, "eda"
    E.ws = Workspace(root / "eda")
    E.status = {s: "NOT_RUN" for s in E.STAGES}
    E.fps, E.errors, E.current_stage, E._D = {}, [], None, None
    E.GRID_SHAPE = (2, 2)                                 # synthetic grid for the NPZ grid-shape checks
    E.F, E.local = None, root
    return E


def valid_band_row() -> dict:
    from gems_eda.discovery import products as PR

    r = dict.fromkeys(PR.BAND_COLUMNS, 1.0)
    r.update(band="rtp", modality="magnetic", provenance_class="processed_survey", units="nT", description="",
             parent_band="", measured_in="test", limitations="none", valid_frac_footprint=1.0, p01=-1.0, p05=-0.5,
             p25=0.0, p50=0.5, p75=0.7, p95=0.9, p99=1.0, n_valid_px=4, n_footprint_px=4)
    return r


def valid_geometry_row(seg_id: int = 0) -> dict:
    from gems_eda.discovery import products as PR

    r = dict.fromkeys(PR.GEOMETRY_COLUMNS, 0)
    r.update(seg_id=seg_id, system_id=1, length_km=2.0, chord_km=1.9, sinuosity=1.05, strike_deg=0.0,
             linearity=0.99, system_kind="single_trace", region_name="NW", crs="EPSG:32611", x_mid_utm=300000.0,
             y_mid_utm=4300000.0)
    return r


def _write_e02(E):
    E.save_parquet(pd.DataFrame({"domain": [0, 1]}), "datasets/physical_regions.parquet")
    E.save_npz("rasters/spatial_domains/domains.npz", domain=np.zeros((2, 2), np.int16))
    E.ws.save_table(pd.DataFrame([valid_band_row()]), "tables/band_characterization.csv")


def _write_e03(E):
    E.save_parquet(pd.DataFrame({"seg_id": [0], "length_km": [2.0]}), "datasets/fault_segments.parquet")
    E.ws.save_table(pd.DataFrame([valid_geometry_row()]), "tables/fault_geometry.csv")


def test_A_band_characterization_contract(tmp_path):
    """e02 FAILED in Colab: tables/band_characterization.csv had no producer."""
    from gems_eda.discovery import products as PR
    from gems_eda.discovery.run import ExhaustiveEDA

    E = _Fake(tmp_path)
    T = PR.write_band_characterization(E, PR.band_characterization(E))
    req = dict(ExhaustiveEDA.OUTPUTS["e02_landscape"])["tables/band_characterization.csv"]
    R = pd.read_csv(tmp_path / "tables/band_characterization.csv")
    assert set(req) <= set(R.columns) and list(R.columns) == list(PR.BAND_COLUMNS)
    assert len(R) == len(E.D.stack.names) and R.band.is_unique
    r = R.set_index("band").loc["rtp"]
    assert r.valid_frac_footprint == 1.0 and r.p01 <= r.p50 <= r.p99 and r.robust_scale_mad >= 0
    assert R.set_index("band").loc["iso_grav_anom"].robust_scale_mad == 0          # flat field: no invented spread
    assert R.corr_length_km.isna().all()                                            # not measured -> NaN, not filled
    assert (R.units == "not available (e01 metadata absent)").all()
    E.ws.save_table(pd.DataFrame({"modality": ["magnetic"], "band": ["rtp"], "direction": ["E-W"],
                                  "corr_length_km": [2.5]}), "figures/data/F005_autocorrelation.csv")
    T2 = PR.enrich_band_characterization(E, T)
    assert T2.set_index("band").corr_length_km["rtp"] == 2.5 and "F005" in T2.measured_in.iloc[0]
    assert "corridor context only" in T2.set_index("band").limitations["rtp"]


def test_A_band_characterization_rejects_invalid(tmp_path):
    from gems_eda.discovery import products as PR

    E = _Fake(tmp_path)
    T = PR.band_characterization(E)
    T.loc[0, "valid_frac_footprint"] = 1.5
    with pytest.raises(PR.ProductInvalid):
        PR.write_band_characterization(E, T)
    assert not (tmp_path / "tables/band_characterization.csv").exists()            # never written when invalid


def test_B_fault_geometry_contract(tmp_path):
    """e03 FAILED in Colab: tables/fault_geometry.csv had no producer."""
    from gems_eda.discovery import products as PR
    from gems_eda.discovery.run import ExhaustiveEDA

    E = _Fake(tmp_path)
    T = PR.write_fault_geometry(E, PR.fault_geometry(E))
    req = dict(ExhaustiveEDA.OUTPUTS["e03_geometry"])["tables/fault_geometry.csv"]
    R = pd.read_csv(tmp_path / "tables/fault_geometry.csv")
    assert set(req) <= set(R.columns) and len(R) == len(E.D.net.segments) >= 1
    assert R.seg_id.is_unique and (R.system_id > 0).all() and (R.length_km > 0).all()
    assert (R.chord_km <= R.length_km + 1e-6).all() and R.strike_deg.between(0, 180).all()
    main = R.loc[R.length_km.idxmax()]
    assert abs(main.strike_deg - 90) < 2 and main.length_km > 15                      # 20 km E-W synthetic trace
    assert (R.crs == "EPSG:32611").all() and R.x_mid_utm.between(500_000, 525_000).all()
    assert len(T) == len(R)


def test_C_missing_mandatory_artifact_fails_stage(tmp_path):
    """A stage whose figures all succeed is FAILED (never COMPLETE) when a contracted product is missing."""
    E = _runner(tmp_path)
    E.status["e01_inventory"] = "COMPLETE"

    def body():                                          # writes everything except the band table
        E.save_parquet(pd.DataFrame({"domain": [0]}), "datasets/physical_regions.parquet")
        E.save_npz("rasters/spatial_domains/domains.npz", domain=np.zeros((2, 2), np.int16))
        return {"note": "12/12 figures; failures []"}
    m = E.stage("e02_landscape", [], body)
    assert m["status"] == "FAILED" and "band_characterization.csv: missing" in m["error"]


def test_C_empty_or_corrupt_npz_is_invalid(tmp_path):
    E = _runner(tmp_path)
    E.ws.path("rasters/spatial_domains").mkdir(parents=True, exist_ok=True)
    E.ws.path("rasters/spatial_domains/domains.npz").write_bytes(b"")
    _, bad = E.check_outputs([("rasters/spatial_domains/domains.npz", ("domain",))])
    assert bad and "missing" in bad[0]
    E.save_npz("rasters/spatial_domains/domains.npz", other=np.zeros(1))
    _, bad = E.check_outputs([("rasters/spatial_domains/domains.npz", ("domain",))])
    assert bad and "absent" in bad[0]


def test_H_cache_invalidation_on_missing_artifact(tmp_path):
    """A valid manifest + matching fingerprint is NOT reused when a contracted output disappeared."""
    E = _runner(tmp_path)
    E.status["e01_inventory"] = "COMPLETE"
    calls = []

    def good():
        calls.append(1)
        _write_e02(E)
        return {"note": "ok"}
    assert E.stage("e02_landscape", [], good)["status"] == "COMPLETE"
    assert E.stage("e02_landscape", [], good)["status"] == "REUSED_VALID_CACHE" and len(calls) == 1
    E.ws.path("tables/band_characterization.csv").unlink()
    m = E.stage("e02_landscape", [], lambda: (calls.append(1), {"note": "nothing written"})[1])
    assert m["status"] == "FAILED" and len(calls) == 2                               # recomputed, not reused


def test_H_contract_enters_fingerprint(tmp_path):
    E = _runner(tmp_path)
    a = E.fp("e02_landscape", *E.stage_parts("e02_landscape", []))
    E.STAGE_SCHEMA = {**E.STAGE_SCHEMA, "e02_landscape": "test-bump"}
    b = E.fp("e02_landscape", *E.stage_parts("e02_landscape", []))
    c = E.fp("e03_geometry", *E.stage_parts("e03_geometry", []))
    E.STAGE_SCHEMA = {**E.STAGE_SCHEMA, "e02_landscape": "2"}
    assert a != b and c == E.fp("e03_geometry", *E.stage_parts("e03_geometry", []))   # targeted invalidation


def test_I_dependency_recovery(tmp_path):
    """e04 is BLOCKED_UPSTREAM while e02 / e03 are FAILED and runs once they satisfy their contracts."""
    E = _runner(tmp_path)
    E.status.update({"e01_inventory": "COMPLETE", "e02_landscape": "FAILED", "e03_geometry": "FAILED"})
    ran = []
    m = E.stage("e04_profiles", [], lambda: ran.append(1) or {"note": ""})
    assert m["status"] == "BLOCKED_UPSTREAM" and not ran
    assert E.stage("e02_landscape", [], lambda: _write_e02(E) or {"note": ""})["status"] == "COMPLETE"
    assert E.stage("e03_geometry", [], lambda: _write_e03(E) or {"note": ""})["status"] == "COMPLETE"
    m = E.stage("e04_profiles", [], lambda: ran.append(1) or {"note": ""})
    assert ran and m["status"] == "FAILED" and "matched_controls.parquet: missing" in m["error"]   # it RAN


class _Atlas:
    ATLAS_STAGE = "e12_candidates"

    def __init__(self, root, status="COMPLETE"):
        from gems_bench.leaderboard_focus.session import Workspace

        self.ws, self.status = Workspace(root), {"e12_candidates": status}


def _atlas(root, ids=("GDE000_A", "GDE001_B"), cols=("candidate_id", "category", "review_split"), panels=True):
    A = pd.DataFrame({"candidate_id": list(ids), "category": "A", "review_split": "TRAIN"})
    (root / "atlas/panels").mkdir(parents=True, exist_ok=True)
    A[[c for c in cols if c in A]].to_csv(root / "atlas/atlas_index.csv", index=False)
    if panels:
        for i in ids:
            (root / f"atlas/panels/{i}_evidence_only.png").write_bytes(b"\x89PNG")


def test_D_widget_without_atlas(tmp_path, capsys):
    from gems_bench.reset.atlas import review_readiness, review_widget

    R = _Atlas(tmp_path)
    assert not review_readiness(R)["ready"] and review_widget(R) is None
    assert "absent" in capsys.readouterr().out


def test_E_widget_atlas_without_candidate_id(tmp_path):
    """Colab: AttributeError 'DataFrame' object has no attribute 'candidate_id'."""
    from gems_bench.reset.atlas import review_readiness, review_widget

    _atlas(tmp_path, cols=("category", "review_split"))
    R = _Atlas(tmp_path)
    rd = review_readiness(R)
    assert not rd["ready"] and "candidate_id" in rd["reason"] and review_widget(R) is None
    (tmp_path / "atlas/atlas_index.csv").write_text("")                                # empty file
    assert not review_readiness(R)["ready"]


def test_F_widget_blocked_stage_hides_stale_atlas(tmp_path):
    from gems_bench.reset.atlas import review_readiness, review_widget

    _atlas(tmp_path)                                     # a valid atlas left by an EARLIER run
    for st in ("BLOCKED_UPSTREAM", "FAILED", "NOT_RUN"):
        R = _Atlas(tmp_path, st)
        rd = review_readiness(R)
        assert not rd["ready"] and st in rd["reason"] and review_widget(R) is None


def test_G_widget_valid_atlas(tmp_path):
    from gems_bench.reset.atlas import review_readiness, review_widget

    _atlas(tmp_path, ids=("GDE000_A", "GDE001_B", "GDE002_C"))
    (tmp_path / "atlas/panels/GDE002_C_evidence_only.png").unlink()
    for st in ("COMPLETE", "REUSED_VALID_CACHE"):
        rd = review_readiness(_Atlas(tmp_path, st))
        assert rd["ready"] and list(rd["candidates"].candidate_id) == ["GDE000_A", "GDE001_B"]
        assert rd["missing_panels"] == ["GDE002_C"]
    pytest.importorskip("ipywidgets")
    pytest.importorskip("IPython")
    W = review_widget(_Atlas(tmp_path))
    assert W is not None


def test_J_review_keys_dtype_stability():
    """pandas FutureWarning at reset/atlas.py:167 (fillna("") on float / bool columns of a reloaded CSV)."""
    from gems_bench.reset.atlas import review_keys

    X = pd.DataFrame({"candidate_id": ["c1", "c2"], "confidence": [np.nan, 0.66], "flag": [True, np.nan],
                      "notes": [None, "x"]})
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        k = review_keys(X, ["candidate_id", "confidence", "flag", "notes", "absent_col"])
    assert k == [("c1", "", "True", "", ""), ("c2", "0.66", "", "x", "")]


def test_J_topology_has_loop_dtype_stability():
    """pandas FutureWarning at morphology.py:296 (fillna(False) on an object column)."""
    from gems_eda.morphology import build_fault_network, component_diagnostics

    D = ST.synthetic_data()
    lab = D.labels.copy()
    lab[30, 30] = True                                   # isolated pixel: a component without a traced segment
    net = build_fault_network(lab, D.grid, (3, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        C = component_diagnostics(net, D.grid, 2.0)
    assert C.has_loop.dtype == bool and not C.has_loop.any() and len(C) == 2


def test_recorded_grid_matches_repository_evidence():
    """check_shape mismatch in Colab: RECORDED had width / height swapped (rows = 3730, cols = 3292)."""
    from gems_eda.discovery.data import RECORDED

    assert (RECORDED["height"], RECORDED["width"]) == (3730, 3292)
    p = REPO / "outputs/benchmark/tables/native_grid_summary.csv"
    if p.exists():
        g = pd.read_csv(p).set_index("key").value
        assert (int(g["height"]), int(g["width"])) == (RECORDED["height"], RECORDED["width"])


def test_every_contracted_output_has_a_producer():
    """Static contract audit: every contracted path appears in the producing source (no contract without writer)."""
    from gems_eda.discovery.run import ExhaustiveEDA

    src = "".join(p.read_text(encoding="utf-8") for p in (REPO / "gems_eda/discovery").glob("*.py")) + \
        (REPO / "gems_bench/geomoe/run.py").read_text(encoding="utf-8") + \
        (REPO / "gems_bench/reset/atlas.py").read_text(encoding="utf-8")
    eda = [s for s in ExhaustiveEDA.STAGES if s.startswith("e") or s == "00_tests"]
    miss = [rel for s in eda for rel, _ in ExhaustiveEDA.OUTPUTS.get(s, []) if f'"{rel}"' not in src]
    assert not miss, f"contracted outputs without a writer: {miss}"


def _code_cells():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_structure():
    # Saved Colab outputs are allowed (the committed notebook keeps its run history); only the code is checked.
    # IPython-only lines (!cmd / %magic) are dropped before parsing.
    code = _code_cells()
    for c in code:
        ast.parse("\n".join(ln for ln in c.splitlines() if not ln.lstrip().startswith(("!", "%"))))
    allc = "\n".join(code)
    assert "'train_models': False" in allc and "'mode': 'FULL_EDA'" in allc
    assert "E.s00_tests()" in allc and "review_widget(E)" in allc and "E.s_synthesis()" in allc


def test_notebook_revision_contract_and_production_path():
    """Notebook EXPECT == package rev (gde.1-r1 vs r2 mismatch in Colab) and the production cells are in order."""
    import re

    from gems_eda.discovery import EDA_CODE_REV

    allc = "\n".join(_code_cells())
    assert re.search(r"EXPECT = \{'code_rev': '([^']+)'", allc).group(1) == EDA_CODE_REV
    marks = ["GR_EDA.ExhaustiveEDA(", "# 2b. Revision contract", "# 3 + 4. Preflight", "# 5. Full pipeline",
             "# 6. Final health", "# 7. Recovery", "E.s00_tests()"]
    pos = [allc.index(m) for m in marks]
    assert pos == sorted(pos), dict(zip(marks, pos))


def test_notebook_patcher_idempotent_and_keeps_outputs():
    """The patcher repairs every target independently, never touches outputs, and is a no-op on the repo notebook."""
    import copy
    import importlib.util

    spec = importlib.util.spec_from_file_location("patcher", REPO / "scripts/eda/patch_gems_eda_notebook.py")
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    nb = json.loads(NB.read_text(encoding="utf-8"))
    outs = [copy.deepcopy(c.get("outputs")) for c in nb["cells"]]
    log = P.patch(nb, P.code_rev())
    assert not [x for x in log if x.startswith(("WOULD_FIX", "MISSING_ANCHOR"))], log
    assert [c.get("outputs") for c in nb["cells"]] == outs
    # a diverged copy (old rev, production cells removed, an old health cell present) is repaired, outputs kept
    old = json.loads(NB.read_text(encoding="utf-8"))
    for c in old["cells"]:
        if "EXPECT = {" in "".join(c["source"]):
            c["source"] = ["".join(c["source"]).replace(P.code_rev(), "gde.1-r1")]
    old["cells"] = [c for c in old["cells"] if not any(m in "".join(c["source"]) for m, *_ in P.PRODUCTION)]
    old["cells"].append({"cell_type": "code", "id": "gde-health", "metadata": {}, "outputs": [{"x": 1}],
                         "execution_count": 1, "source": ["H = diag()"]})
    outs = {c.get("id"): copy.deepcopy(c.get("outputs")) for c in old["cells"]}
    log = P.patch(old, P.code_rev())
    assert sum(x.startswith("WOULD_FIX insert") for x in log) == len(P.PRODUCTION) and "WOULD_FIX config EXPECT" \
        in " ".join(log)
    assert all(c.get("outputs") == outs[c.get("id")] for c in old["cells"] if c.get("id") in outs)
    assert not [x for x in P.patch(old, P.code_rev()) if x.startswith("WOULD_FIX")]           # second pass no-op
