"""Idempotent, per-target repair of notebooks/colab/GEMS_EXHAUSTIVE_GEOLOGICAL_EDA.ipynb (keeps every saved output).

    python scripts/eda/patch_gems_eda_notebook.py --check   # report each target: OK / WOULD_FIX / MISSING_ANCHOR
    python scripts/eda/patch_gems_eda_notebook.py           # apply (atomic write); re-running is a no-op

gde.1-r3 notebook already contains these cells (edited in source); this script repairs a copy that diverged (e.g. a
Colab-saved notebook from an older revision). Every target is checked independently - an existing cell never stops
the others (r2's early return on the health cell is gone):
* config cell: ``EXPECT['code_rev']`` = the checkout's ``gems_eda/discovery/__init__.py`` EDA_CODE_REV (the single
  revision contract; no hard-coded rev here), nothing else in CONFIG is touched;
* title markdown: "(code rev ...)" updated;
* production cells after the bootstrap cell, each identified by its first-line marker, inserted when absent:
  2b revision contract + diag, 3-4 preflight, 5 run, 6 health, 7 recovery, step-by-step heading.
Existing cells are never replaced or reordered, and outputs are never modified.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NB = REPO / "notebooks" / "colab" / "GEMS_EXHAUSTIVE_GEOLOGICAL_EDA.ipynb"


def code_rev() -> str:
    m = re.search(r'EDA_CODE_REV = "([^"]+)"', (REPO / "gems_eda/discovery/__init__.py").read_text(encoding="utf-8"))
    if not m:
        raise SystemExit("EDA_CODE_REV not found in gems_eda/discovery/__init__.py")
    return m.group(1)


def _src(text: str) -> list[str]:
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def _cell(kind: str, cid: str, text: str) -> dict:
    c = {"cell_type": kind, "id": cid, "metadata": {"id": cid}, "source": _src(text)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


VERIFY = """
# 2b. Revision contract + diagnostics helper (cheap). Raises on a stale install / EXPECT mismatch.
print(GR_EDA.verify_checkout(CODE, EXPECT, HEAD))

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
        print(f'    manifest: {OUT}/manifests/stages/{r.stage}.json | history: {OUT}/manifests/history/stage_events.jsonl')
        print(f'    recovery: {r.recovery}')
    return H
"""
PREFLIGHT = """
# 3 + 4. Preflight (metadata, manifests, cached-product validation; no EDA computation) + readiness summary.
#        Saved: audit/preflight_*.csv, audit/preflight_report.json, reports/PREFLIGHT_READINESS.md
P = E.preflight()
pd.set_option('display.max_colwidth', 160)
display(P.summary())
display(P.inputs[[c for c in ('input', 'exists', 'ok', 'height', 'width', 'epsg', 'count', 'dtype', 'positive_px', 'note') if c in P.inputs]])
display(P.geometry)
display(P.models)
print('blocking problems:', P.problems or 'none')
"""
RUN = """
# 5. Full pipeline (one call). Order: tests -> inventory -> A..G -> frozen engine (01/03, only with a factory)
#    -> H forensics -> I atlas -> DEM (optional) -> integrity -> synthesis. Valid caches are reused; a failure is
#    printed once with its category, file:line, affected output, downstream impact and recovery step.
P.require_ready()                                 # stops here (before any computation) if preflight found a blocker
result = E.run()
"""
HEALTH = """
# 6. Final health + artifact validation (cheap; safe to re-run in a later session after the bootstrap)
H = diag(all_rows=False)
display(H[['stage', 'status', 'manifest_status', 'missing_or_invalid', 'upstream_not_valid', 'figure_failures', 'recovery']])
from gems_eda.discovery import diagnostics as DG
display(DG.current_errors(E))                     # errors whose stage is still not valid (history excluded)
show('audit/stage_integrity.csv', ['stage', 'actual_status', 'missing_outputs', 'failure_reason', 'scientific_conclusion_allowed'])
for r in ('MODELING_DECISION', 'EXECUTIVE_RESEARCH_SUMMARY'):
    p = OUT / f'reports/{r}.md'
    if p.exists(): display(Markdown(p.read_text()[:6000]))
"""
RECOVERY = """
# 7. Recovery after an interruption / runtime restart / a pushed fix:
#    re-run the config cell, the bootstrap cell and cell 2b (pin CONFIG['ref'] to the fixed SHA), then:
# P = E.preflight(); display(P.summary())         # shows exactly which stages are reused / recomputed
# E.resume()                                      # same ordered run; valid stages are REUSED_VALID_CACHE
# Single stage after a fix (its upstream stages must be valid in this session first, e.g. via E.resume()):
# E.s_geometry(); diag(['e03_geometry'])
# Current (not historical) errors:
# from gems_eda.discovery import diagnostics as DG; display(DG.current_errors(E)); display(DG.latest_state(E))
"""
STEPS_MD = """
---
## Step-by-step stages (optional)

The cells below run the same stages one part at a time, with figures shown inline (their earlier saved outputs are history). Use them for inspection; the production path is the cells above (preflight -> `E.run()` -> health). A stage already valid is REUSED, not recomputed.
"""
# (marker = first line, kind, id, text) in notebook order after the bootstrap cell
PRODUCTION = [("# 2b. Revision contract", "code", "gde-verify", VERIFY),
              ("# 3 + 4. Preflight", "code", "gde-preflight", PREFLIGHT),
              ("# 5. Full pipeline", "code", "gde-run", RUN),
              ("# 6. Final health", "code", "gde-final-health", HEALTH),
              ("# 7. Recovery", "code", "gde-recovery", RECOVERY),
              ("## Step-by-step stages (optional)", "markdown", "gde-steps-md", STEPS_MD)]


def text(c: dict) -> str:
    return "".join(c.get("source", []))


def find(cells, needle, kind=None) -> list[int]:
    return [i for i, c in enumerate(cells) if needle in text(c) and (kind is None or c["cell_type"] == kind)]


def patch(nb: dict, rev: str) -> list[str]:
    cells, log = nb["cells"], []
    # 1. EXPECT (config cell)
    hits = find(cells, "EXPECT = {", "code")
    if len(hits) != 1:
        log.append(f"MISSING_ANCHOR config cell (found {len(hits)} cells with 'EXPECT = {{')")
    else:
        s = text(cells[hits[0]])
        new = re.sub(r"'code_rev': '[^']*'", f"'code_rev': '{rev}'", s)
        log.append("OK config EXPECT" if new == s else f"WOULD_FIX config EXPECT -> {rev}")
        cells[hits[0]]["source"] = _src(new)
    # 2. title
    hits = find(cells, "# GEMS exhaustive geological discovery EDA (code rev", "markdown")
    for i in hits:
        s = text(cells[i])
        new = re.sub(r"\(code rev [^)]*\)", f"(code rev {rev})", s)
        log.append("OK title" if new == s else f"WOULD_FIX title -> {rev}")
        cells[i]["source"] = _src(new)
    # 3. production cells
    b = find(cells, "GR_EDA.ExhaustiveEDA(", "code")
    if len(b) != 1:
        log.append(f"MISSING_ANCHOR bootstrap cell (found {len(b)}): production cells not checked")
        return log
    at = b[0]
    for marker, kind, cid, body in PRODUCTION:
        h = find(cells, marker, kind)
        if h:
            at = h[0]
            log.append(f"OK {cid}")
            continue
        cells.insert(at + 1, _cell(kind, cid, body))
        at += 1
        log.append(f"WOULD_FIX insert {cid}")
    return log


def main() -> int:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    before = json.dumps(nb, sort_keys=True)
    log = patch(nb, code_rev())
    print("\n".join(log))
    changed = json.dumps(nb, sort_keys=True) != before
    if "--check" in sys.argv or not changed:
        print("no change written" + (" (check mode)" if "--check" in sys.argv else " (already up to date)"))
        return 0
    tmp = NB.with_suffix(".ipynb.tmp")
    tmp.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(NB)
    print(f"written {NB}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
