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


def test_notebook_structure():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for c in code:
        ast.parse(c)
    allc = "\n".join(code)
    assert "EXPECT = {'code_rev': 'gde.1-r1'" in allc and "'train_models': False" in allc
    assert "E.s00_tests()" in allc and "review_widget(E)" in allc and "E.s_synthesis()" in allc
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code"), "notebook committed with outputs"


def test_code_rev_matches_notebook():
    from gems_eda.discovery import EDA_CODE_REV

    assert EDA_CODE_REV in NB.read_text(encoding="utf-8")
