"""One-off, idempotent patch of notebooks/colab/GEMS_P6_FAST_TRACK.ipynb for the P6 boost r2 workflow.

    python scripts/p6/patch_p6_boost_notebook.py            # writes the notebook in place (outputs preserved)
    python scripts/p6/patch_p6_boost_notebook.py --check    # only reports what would change

Changes (by cell id, nothing else is touched; every existing output is kept):
* configuration cell: ``BOOST_RUN`` switches + ``EXPECT`` (notebook version, code rev, optional pinned commit);
* new start-up check cell after the clone cell (requested ref, resolved SHA, module path, config fingerprint);
* section 3B: preflight cell, optional G02 backfill cell, report cell; every stage cell gated by ``BOOST_RUN``;
  export only through ``BOOST_RUN['export']``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

NB = Path(__file__).resolve().parents[2] / "notebooks" / "colab" / "GEMS_P6_FAST_TRACK.ipynb"
MARK = "p6boost-start"


def _src(text: str) -> list[str]:
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def code(cid: str, text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {"id": cid}, "outputs": [], "source": _src(text)}


def md(cid: str, text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {"id": cid}, "source": _src(text)}


CONFIG_ADD = """
BOOST_RUN = {'preflight': True, 'benchmark': True, 'backfill_g02': False, 'geology': False, 'robustness': False,
             'hybrid': False, 'selection': False, 'export': False, 'report': True}   # section 3B switches (in order)
EXPECT = {'notebook_version': 'p6-nb.3', 'code_rev': 'p6b.1-r2',
          'commit': None}   # set to the pushed SHA prefix (also put it in CONFIG['ref']) for an immutable run
"""

START = code(MARK, """
# 0b. Start-up check: the code that runs is the code that was requested (raises; never falls back to main).
import json
from gems_bench.p6.boost_audit import startup_check
START = startup_check(CONFIG['ref'], HEAD, {**EXPECT, 'code_root': CODE,
                      'artifact_root': f"{CONFIG['drive_root']}/outputs/p6_fast_track/boost"}, BOOST_CONFIG)
print(json.dumps(START, indent=1))
""")

MD0 = md("p6boost-md0", """
## 3B. P6 boost — preflight, matched benchmark, G02 structure, weak-zone robustness, hybrids (CPU, ≤ 4 h)

`gems_bench/p6/boost.py` · `gems_bench/p6/boost_audit.py` · `configs/p6/boost.yaml` · guides `docs/p6/P6_BOOST.md`,
`docs/p6/P6_BOOST_HANDOFF.md` · outputs `boost/`. W0–W2 outputs are read-only inputs (W2 is frozen).
Order: 3B.0 → P (preflight) → A → B0 (optional G02 backfill) → B → C → D → E → R (report). Each cell runs only when
its `BOOST_RUN` switch is on; a stage reloads instead of recomputing while its fingerprint holds and its outputs exist.
`boost/<stage>/status.json` says PENDING / BLOCKED / RUNNING / COMPLETE / FAILED.
""")

PRE = code("p6boost-cP", """
if BOOST_RUN['preflight']:                 # cheap: metadata, hashes, headers; no training, no map scoring
    pre = bst.preflight()
    print(json.dumps(pre['areas'], indent=1))
    display(bst.load_table('feature_compatibility', 'preflight'))
    sp = (bst.read_json('preflight', 'submission_provenance') or {}).get('w2', {})
    print(json.dumps({k: sp.get(k) for k in ('status', 'rep', 'frac', 'blend', 'selected_cell', 'interpretation')},
                     indent=1, default=str))
    print(json.dumps(bst.read_json('preflight', 'g02_backfill_plan'), indent=1, default=str)[:2500])
""")

A = code("p6boost-c1", """
if BOOST_RUN['benchmark']:
    a = bst.a_benchmark()
    display(a['ranking_gain'])
    display(a['nested'][['space', 'disc_dti', 'disc_worst', 'geo_dti', 'systems_gained', 'systems_lost']])
    print(json.dumps(a['verdict'], indent=1, default=str))
""")

BF_MD = md("p6boost-mdB0", """
### 3B-B0. Optional G02 zone-2 backfill (explicit opt-in)

Runs only with `BOOST_RUN['backfill_g02'] = True`. Read `preflight/g02_backfill_plan.json` first: status must be
PENDING / READY (input drift ok, channels present) and the estimate must fit the `wb` budget. Existing zone-0/1 rasters
are never rewritten; an interrupted unit resumes from its finished tiles. Then rerun 3B.P and 3B-B.
""")

BF = code("p6boost-c2a", """
if BOOST_RUN['backfill_g02']:
    plan = bst.read_json('preflight', 'g02_backfill_plan') or {}
    assert plan.get('status') in ('PENDING', 'READY'), f"backfill plan status {plan.get('status')}: {plan.get('reason') or plan.get('input_drift')}"
    BOOST_CONFIG = {**BOOST_CONFIG, 'geology': {**BOOST_CONFIG.get('geology', {}), 'backfill_missing': True}}
    bst = p6.boost(BOOST_CONFIG)
    done = bst.section('geology_backfill', bst.backfill)
    print('backfilled:', done)
""")

B = code("p6boost-c2", """
if BOOST_RUN['geology']:
    b = bst.b_geology()
    print(json.dumps({k: v for k, v in b['attach'].items() if k != 'root_cause'}, indent=1, default=str))
    display(b['audit'].pivot_table(index='unit', columns='feature', values='key_present', aggfunc='first'))
    display(b['ablation'][['candidate', 'status', 'decision', 'disc_dti', 'disc_worst', 'geo_dti', 'gain_vs_base',
                           'reasons']])
    print('chosen:', b['decision']['chosen'], '| stage status:', bst.read_json('geology', 'status'))
""")

C = code("p6boost-c3", """
if BOOST_RUN['robustness']:
    c = bst.c_robustness()
    for n in ('zone_ranking', 'composition', 'prediction_geometry'):
        display(c['diagnostics'][n])
    display(c['ablation'][['candidate', 'decision', 'disc_dti', 'disc_worst', 'geo_dti', 'reasons']])
""")

D = code("p6boost-c4", """
if BOOST_RUN['hybrid']:
    d = bst.d_hybrid()
    display(d['ablation'][['candidate', 'decision', 'disc_dti', 'disc_worst', 'geo_dti', 'systems_gained',
                           'systems_lost', 'fp_increase', 'reasons']])
    display(d['d3'])
""")

E = code("p6boost-c5", """
if BOOST_RUN['selection']:                 # export only with BOOST_RUN['export'] AND a PASS gate; reruns reuse the file
    e = bst.e_select(export=BOOST_RUN['export'])
    print(json.dumps({k: v for k, v in e.items() if k != 'champion'}, indent=1, default=str)[:3000])
    display(bst.load_table('candidates', 'selection'))
    display(bst.load_table('leaderboard_compare', 'selection'))
""")

R = code("p6boost-cR", """
if BOOST_RUN['report']:                    # saved outputs only; safe after any stage
    rep = bst.report()
    display(rep['status'])
    display(rep['summary'])
    print(json.dumps(rep['readiness']['questions'], indent=1, default=str)[:5000])
""")


def patch(nb: dict) -> list[str]:
    cells, log = nb["cells"], []
    ids = [c.get("metadata", {}).get("id") for c in cells]
    if MARK in ids:
        return ["already patched (start-up cell present): nothing to do"]

    def at(cid):
        if cid not in ids:
            raise KeyError(f"cell {cid} not found: the notebook differs from the expected revision")
        return ids.index(cid)

    cfg = cells[at("6s1AmP_x-37y")]
    src = "".join(cfg["source"]).rstrip("\n") + "\n" + CONFIG_ADD.strip("\n")
    cfg["source"] = _src(src)
    log.append("config cell: BOOST_RUN + EXPECT")
    for cid, new in (("p6boost-md0", MD0), ("p6boost-c1", A), ("p6boost-c2", B), ("p6boost-c3", C),
                     ("p6boost-c4", D), ("p6boost-c5", E)):
        i = at(cid)
        new["outputs" if new["cell_type"] == "code" else "source"] = (cells[i].get("outputs", [])
                                                                      if new["cell_type"] == "code" else new["source"])
        cells[i] = new
        log.append(f"replaced {cid} (outputs kept)")
    ins = [("TPiW4rXq-37z", [START]), ("p6boost-c0", [PRE]), ("p6boost-c1", [BF_MD, BF]), ("p6boost-c5", [R])]
    for after, new in ins:
        ids = [c.get("metadata", {}).get("id") for c in cells]
        i = ids.index(after) + 1
        cells[i:i] = new
        log.append(f"inserted {[c['metadata']['id'] for c in new]} after {after}")
    return log


def main() -> int:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    log = patch(nb)
    print("\n".join(log))
    if "--check" in sys.argv or log[0].startswith("already"):
        return 0
    tmp = NB.with_suffix(".ipynb.tmp")
    tmp.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(NB)
    print(f"written {NB}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
