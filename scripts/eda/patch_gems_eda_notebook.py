"""Idempotent patch of notebooks/colab/GEMS_EXHAUSTIVE_GEOLOGICAL_EDA.ipynb for EDA code rev gde.1-r2.

    git pull                                              # get the Colab-saved notebook (outputs) first
    python scripts/eda/patch_gems_eda_notebook.py --check # report what would change
    python scripts/eda/patch_gems_eda_notebook.py         # write in place (every existing output is kept)

Why a script: the committed notebook carries Colab outputs and cells added in Colab, so it is edited cell by cell
(located by content, never by position) instead of being rewritten. Changes:
* config cell: ``EXPECT['code_rev']`` gde.1-r1 -> gde.1-r2 (nothing else in CONFIG is touched: mode, resolution,
  caching, Drive sync and frozen-model switches keep their saved values);
* new cell after the environment cell: ``diag()`` helper (stage status, missing contracted outputs, upstream stages,
  root error, figure failures, recovery hint, from ``E.health()``);
* stage cells: a trailing ``diag([...])`` call for the stages they run;
* review cell: the widget only when ``review_readiness(E)`` says the atlas of THIS run is reviewable;
* new health-check section after the synthesis cell.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

NB = Path(__file__).resolve().parents[2] / "notebooks" / "colab" / "GEMS_EXHAUSTIVE_GEOLOGICAL_EDA.ipynb"
MARK = "gde-health"


def _src(text: str) -> list[str]:
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def code(cid: str, text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "id": cid, "metadata": {"id": cid}, "outputs": [],
            "source": _src(text)}


def md(cid: str, text: str) -> dict:
    return {"cell_type": "markdown", "id": cid, "metadata": {"id": cid}, "source": _src(text)}


DIAG = code("gde-diag-fn", """
# Stage diagnostics (cheap: reads manifests and checks the output contract; recomputes nothing)
def diag(stages=None, all_rows=False):
    H = E.health(stages)
    bad = H if all_rows else H[H.recovery != '']
    if not len(bad):
        print('diag: all listed stages valid:', list(H.stage)); return H
    for r in bad.itertuples():
        print(f'[{r.stage}] status={r.status} (manifest {r.manifest_status})')
        for k in ('missing_or_invalid', 'upstream_not_valid', 'error', 'affected_file', 'figure_failures'):
            v = getattr(r, k)
            if isinstance(v, str) and v:
                print(f'    {k}: {v}')
        print(f'    manifest: {OUT}/manifests/stages/{r.stage}.json | figure failures: {OUT}/manifests/figure_failures/{r.stage}.json')
        print(f'    recovery: {r.recovery}')
    return H
""")

REVIEW = """
from gems_bench.reset.atlas import review_readiness, review_widget
RD = review_readiness(E)                 # atlas stage of THIS run COMPLETE / REUSED + index, columns, rows, panels
print('atlas review:', 'READY' if RD['ready'] else 'SKIPPED', '|', RD['stage'], '=', RD['status'], '|', RD['reason'])
if RD['ready']:
    W = review_widget(E)                 # re-checks the same conditions itself
    if W is not None: display(W)
# Alternative: fill atlas/review_template.csv (copy it on Drive) and run E.import_reviews('<path to the filled copy>')
# One review in code: E.record_review({'candidate_id': '...', 'interpretation': 'POSSIBLE_FAULT', 'confidence': 'MEDIUM',
#                                      'review_mode': 'EVIDENCE_ONLY', 'reviewer': 'name', 'evidence_sources': 'magnetic;gravity',
#                                      'supporting_evidence': '...', 'alternative_explanation': '...', 'data_license': 'n/a'})
"""

HEALTH_MD = md("gde-health-md", """
### Health check (optional, cheap)

One row per stage: status in this session, last manifest status, missing / invalid contracted outputs, upstream
stages that are not valid, root error, figure failures and the recovery step. A stage is COMPLETE only when every
contracted output exists and validates; figures alone never make a stage complete.
""")

HEALTH = code(MARK, """
H = diag(all_rows=False)
display(H[['stage', 'status', 'manifest_status', 'missing_or_invalid', 'upstream_not_valid', 'figure_failures', 'recovery']])
""")

STAGE_DIAG = {"E.s_landscape()": "['e02_landscape', 'e03_geometry']",
              "E.s_profiles()": "['e04_profiles', 'e05_signatures', 'e06_scales']",
              "'e07_crossmodal'": "['e07_crossmodal', 'e08_label_bias', 'e09_regional']",
              "E.s_forensics()": "['01_attach', '03_scenarios', 'e11_forensics']",
              "E.s_candidates()": "['e12_candidates']"}


def text(c: dict) -> str:
    return "".join(c.get("source", []))


def find(cells: list, needle: str, kind: str = "code") -> int:
    hits = [i for i, c in enumerate(cells) if c["cell_type"] == kind and needle in text(c)]
    if len(hits) != 1:
        raise KeyError(f"expected exactly one {kind} cell containing {needle!r}, found {len(hits)}: the notebook "
                       "differs from the expected revision (patch nothing; inspect it)")
    return hits[0]


def patch(nb: dict) -> list[str]:
    cells, log = nb["cells"], []
    ids = {c.get("id") or c.get("metadata", {}).get("id") for c in cells}
    if MARK in ids:
        return ["already patched (health-check cell present): nothing to do"]
    i = find(cells, "EXPECT = {")
    s = text(cells[i])
    if "'code_rev': 'gde.1-r1'" in s:
        cells[i]["source"] = _src(s.replace("'code_rev': 'gde.1-r1'", "'code_rev': 'gde.1-r2'"))
        log.append("config cell: EXPECT code_rev gde.1-r2 (CONFIG untouched)")
    for needle, stages in STAGE_DIAG.items():
        try:
            j = find(cells, needle)
        except KeyError as e:
            log.append(f"stage diag skipped: {e}")
            continue
        s = text(cells[j])
        if "diag(" not in s:
            cells[j]["source"] = _src(s.rstrip("\n") + f"\ndiag({stages})")
            log.append(f"stage cell [{needle}]: + diag({stages}) (outputs kept)")
    j = find(cells, "review_widget(E)")
    cells[j]["source"] = _src(REVIEW)                    # outputs of the earlier run are kept as history
    log.append("review cell: gated by review_readiness (outputs kept)")
    k = find(cells, "GR_EDA.ExhaustiveEDA(")
    cells[k + 1:k + 1] = [DIAG]
    log.append("inserted gde-diag-fn after the environment cell")
    m = find(cells, "E.s_synthesis()")
    cells[m + 1:m + 1] = [HEALTH_MD, HEALTH]
    log.append("inserted health-check section after the synthesis cell")
    return log


def main() -> int:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    log = patch(nb)
    print("\n".join(log))
    if "--check" in sys.argv or log[0].startswith("already"):
        return 0
    tmp = NB.with_suffix(".ipynb.tmp")
    tmp.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(NB)
    print(f"written {NB}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
