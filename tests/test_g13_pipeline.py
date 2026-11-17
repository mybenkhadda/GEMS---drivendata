"""End-to-end synthetic smoke runs of the G13 engine: FAST_BASELINE milestone, resume, FINAL_SUBMISSION reuse.

The research stages (EDA, regimes, ablation, fusion, topology, failure modes) are exercised on a reduced feature set
so the suite stays fast; the full FULL_RESEARCH synthetic run is done by ``scripts/g13_smoke.py``.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from gems_bench.g13.engine import Engine
from gems_bench.g13.synthetic import make, small_config

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("g13pipe")
    make(root / "data")
    cfg = small_config(root / "work", root / "data", "FAST_BASELINE", out_dir=str(root / "drive"))
    eng = Engine(cfg, log=lambda *a: None)
    st = eng.run()
    return root, eng, st


def test_fast_baseline_completes_and_exports_valid_s0(run):
    root, eng, st = run
    assert st.complete.all(), st
    sub = root / "work" / "submissions"
    for f in ("submission_g13_baseline.tif", "submission_g13_champion.tif", "predicted_faults.tif"):
        assert (sub / f).exists()
    meta = json.loads((sub / "submission_g13_baseline.json").read_text())
    assert meta["validation"]["valid"] and meta["milestone"] == "S0"
    champ = json.loads((sub / "champion.json").read_text())
    assert "not available" in champ["leaderboard_score"]  # never claims a leaderboard score
    # persisted to the Drive mirror
    assert (root / "drive" / "submissions" / "predicted_faults.tif").exists()
    assert (root / "drive" / "reports" / "report.md").exists()


def test_resume_reuses_every_stage(run):
    root, eng, _ = run
    msgs = []
    eng2 = Engine(eng.cfg, log=msgs.append)
    st = eng2.run()
    assert st.complete.all()
    assert all("reused" in m for m in msgs if m.startswith("[") and "running" not in m), msgs
    assert not any("running" in m for m in msgs)


def test_config_change_invalidates_downstream(run):
    root, eng, _ = run
    cfg = small_config(root / "work", root / "data", "FAST_BASELINE", out_dir=str(root / "drive"), area_fracs=(0.02, 0.04))
    eng2 = Engine(cfg, log=lambda *a: None)
    st = eng2.status().set_index("stage")
    assert st.loc["baseline", "complete"] and not st.loc["dti_opt", "complete"] and not st.loc["submission", "complete"]


def test_final_submission_mode_reuses_models_on_fresh_runtime(run):
    root, eng, _ = run
    # simulate a new Colab runtime: empty local work dir, Drive mirror intact
    cfg = small_config(root / "work_fresh", root / "data", "FINAL_SUBMISSION", out_dir=str(root / "drive"))
    msgs = []
    eng2 = Engine(cfg, log=msgs.append)
    st = eng2.run()
    assert st.complete.all(), st
    assert not any("[model]" in m for m in msgs)  # no model retrained
    assert (root / "work_fresh" / "submissions" / "predicted_faults.tif").exists()


def test_research_stages_on_small_feature_set(tmp_path):
    make(tmp_path / "data")
    cfg = small_config(
        tmp_path / "work",
        tmp_path / "data",
        "FULL_RESEARCH",
        families=("magnetic", "terrain", "cross_modal", "validity"),
        pu_strategies=("standard", "soft_negative"),
        tabular_models=("xgboost",),
    )
    eng = Engine(cfg, log=lambda *a: None)
    todo = (
        "inventory",
        "splits",
        "features",
        "quality",
        "regimes",
        "baseline",
        "ablation",
        "pu_learning",
        "fusion",
        "topology",
        "dti_opt",
        "failure_modes",
        "submission",
        "report",
    )
    st = eng.run(todo)
    failed = st[st.stage.isin(todo) & ~st.complete]
    assert failed.empty, failed
    w = tmp_path / "work"
    board = pd.read_csv(w / "tables" / "scoreboard.csv")
    assert {"S0_raw_xgboost", "A0_raw_xgboost", "PU_soft_negative_xgboost"} <= set(board.model)
    assert board.disc_proxy_dti.between(0, 1).all()
    cands = pd.read_csv(w / "topology" / "candidates.csv")
    assert set(cands["class"]) <= {
        "mapped_trace",
        "known_extension",
        "parallel",
        "intersecting",
        "unverified_network",
        "isolated",
    }
    sel = json.loads((w / "dti_opt" / "selection.json").read_text())
    assert 0 <= sel["test"]["test_proxy_dti"] <= 1
    comp = pd.read_csv(w / "submissions" / "candidate_comparison.csv")
    assert comp.valid.all() and set(comp.candidate) >= {"A", "B", "C"}
    rep = (w / "reports" / "report.md").read_text()
    assert "proxies" in rep and "## 15. Recommended next experiments" in rep
    fig = pd.read_csv(w / "figures" / "figure_index.csv")
    assert fig.takeaway.str.len().gt(10).all()


def test_synthetic_physics_beats_chance(run):
    """Sanity: the raw model ranks the synthetic withheld faults far above chance (not a performance claim)."""
    root, eng, _ = run
    board = pd.read_csv(root / "work" / "tables" / "scoreboard.csv")
    assert board.disc_proxy_dti.max() > 0.1 and np.isfinite(board.geo_proxy_dti).all()
