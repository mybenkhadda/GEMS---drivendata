"""Failure diagnostics, append-only stage history and latest stage state.

* ``classify``: error category from the exception text / traceback (input, source code, numerical, schema, insufficient
  data, missing model artifact, resource, visualisation).
* ``downstream``: every stage transitively blocked by a stage (from ``DEPENDS``).
* ``on_record``: called by ``ExhaustiveEDA.record`` after every stage: appends one event to
  ``manifests/history/stage_events.jsonl`` (never rewritten), marks earlier session errors of a stage that now
  succeeded as resolved, refreshes ``manifests/latest_stage_state.csv`` (latest per-stage manifest = current truth;
  ``manifests/run.json`` is only the snapshot of one session) and prints a concise diagnosis for a FAILED stage.
"""
from __future__ import annotations

import json
import re
import time

import pandas as pd

CATEGORIES = ("INPUT_ERROR", "SOURCE_CODE_ERROR", "NUMERICAL_ERROR", "SCHEMA_VIOLATION", "INSUFFICIENT_DATA",
              "MISSING_MODEL_ARTIFACT", "RESOURCE_FAILURE", "OPTIONAL_UNAVAILABLE", "VISUALIZATION_FAILURE",
              "UPSTREAM")
_PATTERNS = (
    ("RESOURCE_FAILURE", r"MemoryError|No space left|Errno 28|Killed|out of memory|Cannot allocate"),
    ("INPUT_ERROR", r"FileNotFoundError|RasterioIOError|not on the features grid|raw input|No such file"),
    ("SCHEMA_VIOLATION", r"ProductInvalid|required outputs invalid|schema rules violated|columns .* absent|rule "),
    ("INSUFFICIENT_DATA", r"InsufficientData"),
    ("NUMERICAL_ERROR", r"GeometryPrecisionError|FloatingPointError|LinAlgError|overflow|non-finite"),
    ("MISSING_MODEL_ARTIFACT", r"honest maps unavailable|frozen engine|factory|not attached|honest-map grid"),
    ("OPTIONAL_UNAVAILABLE", r"dem\.enabled|DEM coverage|ABSENT \("),
)
RECOVERY = {
    "INPUT_ERROR": "fix the raw input on Drive (see audit/preflight_inputs.csv); never resample silently",
    "SOURCE_CODE_ERROR": "code defect: send the traceback (manifests/history/stage_events.jsonl); re-run the stage "
                         "after the fix (upstream caches are reused)",
    "NUMERICAL_ERROR": "numerical defect in a producer: inspect the sample in the error; do not clip in the table",
    "SCHEMA_VIOLATION": "a product broke its contract: the violated rule and sample rows are in the error",
    "INSUFFICIENT_DATA": "the data cannot support this measurement with the current configuration: change the "
                         "sampling (documented) or accept the limitation; nothing is fabricated",
    "MISSING_MODEL_ARTIFACT": "attach the frozen engine (E.s01_attach(); E.s03_scenarios()) or report the stage "
                              "as unavailable; never train / infer to fill it",
    "RESOURCE_FAILURE": "restart on a larger runtime or free memory; then E.resume() (finished stages are reused)",
    "OPTIONAL_UNAVAILABLE": "optional data absent: reported, not a conclusion",
    "VISUALIZATION_FAILURE": "figures only: see manifests/figure_failures/<stage>.json; core products are valid",
    "UPSTREAM": "fix the first failing upstream stage (listed), then re-run",
}


def classify(error: str, traceback: str = "") -> str:
    """The exception message decides first; the traceback's last line only when the message matches nothing (code
    lines quoted in a traceback must not reclassify an error)."""
    last = (traceback or "").strip().splitlines()[-1:] or [""]
    for text in (error or "", last[0]):
        for cat, pat in _PATTERNS:
            if re.search(pat, text):
                return cat
    return "SOURCE_CODE_ERROR"


def downstream(stage: str, depends: dict) -> list[str]:
    out, frontier = [], [stage]
    while frontier:
        s = frontier.pop()
        for k, ds in depends.items():
            if s in ds and k not in out:
                out.append(k)
                frontier.append(k)
    order = list(depends)
    return sorted(out, key=lambda k: order.index(k) if k in order else 99)


def where(traceback: str) -> str:
    """'file:line in function' of the innermost repository frame of a traceback."""
    frames = re.findall(r'File "([^"]+)", line (\d+), in (\S+)', traceback or "")
    repo = [f for f in frames if "gems_" in f[0]] or frames
    if not repo:
        return ""
    f, ln, fn = repo[-1]
    return f"{f.split('GEMS/')[-1]}:{ln} in {fn}"


def latest_state(E) -> pd.DataFrame:
    rows = []
    for s in E.STAGES:
        m = E.manifest(s)
        rows.append({"stage": s, "status": m.get("status", "NEVER_RUN"), "written": m.get("written"),
                     "fingerprint": m.get("fingerprint"), "code_rev": m.get("eda_code_rev") or m.get("code_rev"),
                     "commit": m.get("commit"), "error": str(m.get("error") or "")[:300]})
    return pd.DataFrame(rows)


def on_record(E, stage: str, st: str, m: dict) -> None:
    from gems_bench.geomoe.decide import FAILED, PRODUCED

    info = {k: m.get(k) for k in ("error", "affected_file", "reason", "note", "figure_failures")}
    tb = str(m.get("traceback") or "")
    cat = classify(str(info.get("error") or ""), tb) if st == FAILED else (
        "UPSTREAM" if st in ("BLOCKED_UPSTREAM", "NOT_EVALUATED") else None)
    ev = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "stage": stage, "status": st, "category": cat,
          "fingerprint": m.get("fingerprint"), "commit": E.head, "eda_code_rev": m.get("eda_code_rev"),
          "where": where(tb), **{k: (str(v)[:2000] if v is not None else None) for k, v in info.items()},
          "traceback": tb[-4000:] if tb else None}
    p = E.ws.path("manifests/history/stage_events.jsonl")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:            # append-only history (one JSON object per line)
        fh.write(json.dumps(ev, default=str) + "\n")
    if st in PRODUCED:                                    # earlier errors of this stage are history now
        for e in E.errors:
            if e.get("stage") == stage and not e.get("resolved"):
                e["resolved"] = f"resolved by {st} at {ev['time']}"
        E.ws.save_json(E.errors, "manifests/runtime_errors.json")
    E.ws.save_table(latest_state(E), "manifests/latest_stage_state.csv")
    if st == FAILED:
        print(failure_text(E, stage, m, cat))


def failure_text(E, stage: str, m: dict, cat: str) -> str:
    blocked = downstream(stage, E.DEPENDS)
    lines = [f"  >>> {stage} FAILED [{cat}]",
             f"      error: {str(m.get('error') or '')[:600]}",
             f"      where: {where(str(m.get('traceback') or '')) or 'n/a'}",
             f"      affected output: {m.get('affected_file')}",
             f"      blocks downstream: {blocked or 'none'}",
             f"      recovery: {RECOVERY.get(cat, '')}",
             f"      manifest: {E.ws.path(f'manifests/stages/{stage}.json')} | history: "
             f"{E.ws.path('manifests/history/stage_events.jsonl')}"]
    return "\n".join(lines)


def current_errors(E) -> pd.DataFrame:
    """Errors that still matter: the stage's latest manifest is not COMPLETE / REUSED / SKIPPED."""
    from gems_bench.geomoe.decide import VALID

    L = latest_state(E).set_index("stage").status
    errs = E.ws.read_json("manifests/runtime_errors.json") or []
    rows = [e for e in errs if not e.get("resolved") and L.get(e.get("stage")) not in VALID]
    return pd.DataFrame(rows)
