"""GEMS submission factory: every enabled frozen candidate -> its own validated, read-back competition GeoTIFF.

    F = Factory(CONFIG, HEAD)          # notebooks/leaderboard_focus/GEMS_SUBMISSION_FACTORY.ipynb
    F.selftest()                       # synthetic runtime self-tests (a critical failure disables every export)
    F.run()                            # RUN_ALL: preflight -> engine -> candidates -> ranking -> reports
    F.record_result(ENTRY)             # manual leaderboard result (after a manual upload)

Thin orchestration over existing code; nothing is trained, re-inferred, tuned or uploaded:
    preflight        preflight.run + artifacts.run_environment (read-only Drive facts, selective staging) + the
                     candidate dependency table below (a missing artifact blocks only the candidates that need it)
    engine           session.M1Session.attach (P6 engine on the CPU, mode recovery; root = OUTPUT_DIR)
    candidate table  champion.recover: nested single-policy numbers of EVERY candidate from the frozen boost-A grid
                     (same folds, eligible area, scorer and protocol for all; the M1 champion rule picks nothing here)
    evaluation       validation.recompute: official components re-scored from the frozen maps vs the saved grid
    map              p6.final_submission.build_map: geo zone k from the maps evaluated on zone k (zone-stitched)
    file             p6.final_submission.recipe / write_file / validate (gems3.submission.finalise + validate,
                     export_parity.audit_export) + the read-back checks of this module on the file on disk
Verdicts per candidate, never conflated: TECHNICAL_VALIDATION, EXPORT_PARITY, OFFICIAL_FORMAT (hard) and
GEOGRAPHIC_TARGET_MET (internal research target: reported; it blocks only in ``geographic_target_mode: strict``).
READY_FOR_UPLOAD = every hard check passed on the file READ BACK from disk. RECOMMENDATION_RANK = the predeclared
``rank`` rule among READY files: an offline ordering, never a leaderboard claim.

Fusion fallback (b0_b1_fallback): the benchmark scored the B0 + B1 fusion with the nested per-zone champion weight
of each geo zone (``Boost.fusion``); the file is built with exactly those weights (``build_map(fusion_w='per_zone')``)
so it is the evaluated prediction. Nothing is re-tuned: the weights are read from the frozen M9 selection.
"""
from __future__ import annotations

import gc
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from .config import READY

FACTORY_VERSION = "sf.1"
FACTORY_CODE_REV = "sf.1-r1"                      # checked by the notebook start-up cell (not fingerprinted)
OUTPUT_DIR = "outputs/leaderboard_focus/submission_factory"
RUN_MODES = ("FULL", "EXPORT_ONLY", "DIAGNOSTIC", "RESUME")
TARGET_MODES = ("report_only", "strict")
PASS, WARNING, FAIL, NOT_RUN, UNAVAILABLE, BLOCKED = "PASS", "WARNING", "FAIL", "NOT_RUN", "UNAVAILABLE", "BLOCKED"
# candidate status (READY = READY_FOR_UPLOAD)
VALIDATED, HELD, DUPLICATE, INVALID, FAILED = ("VALIDATED_NOT_WRITTEN", "HELD_BY_STRICT_TARGET", "DUPLICATE",
                                               "INVALID", "FAILED")
REUSABLE = (READY, VALIDATED, HELD)
DEFAULTS = {
    "repo": "mybenkhadda/GEMS---drivendata", "ref": "main",
    "drive_root": "/content/drive/MyDrive/GEMS_2", "local_root": "/content/gems_work",
    "run_mode": "FULL", "reuse_cached": True, "overwrite_existing": False,
    "device": "cpu", "train_models": False,
    "candidates": ["w2_champion", "w2_crest", "w2_skeleton", "b0_b1_fallback"],
    "evaluate_official_metric": True, "validate_export_parity": True,
    "geographic_target": 0.1758, "geographic_target_mode": "report_only",
    "export_all_technically_valid": True, "max_submission_files": 4,
    "save_png": True, "save_csv": True, "save_parquet": True, "save_geotiff": True, "sync_to_drive": True,
    "manual_upload_only": True,
}
W2_ARTIFACTS = ("w2_candidate", "w2_rows", "w2_stack_grid", "w2_maps")
SHARED_ARTIFACTS = ("boost_a_grid", "boost_a_manifest", "m9_selection", "b0_manifest", "b1_manifest", "b0_maps",
                    "b1_maps", "geo_folds", "disc_splits", "template")
# Candidate registry: source = the configs/p6/full_gpu.yaml candidate whose nested row (champion.recover) defines the
# recipe. historical = recorded values with their protocol (context only; never used as inputs or as claims).
CANDIDATES = {
    "w2_champion": {
        "source": "a_w2_rep_frac", "family": "w2", "fusion_w": "all_zones", "cell": "w2", "reps": None,
        "artifacts": SHARED_ARTIFACTS + W2_ARTIFACTS, "label": "W2 stacker, nested rep x frac (the M1 champion)",
        "historical": {"disc_dti": 0.242156, "disc_worst": 0.198579, "geo_dti": 0.105687, "protocol": "single_policy",
                       "recipe_id": "0a8692b2ab", "source": "M1 Colab run (recorded)"}},
    "w2_crest": {
        "source": "a_w2_crest", "family": "w2", "fusion_w": "all_zones", "cell": "w2", "reps": ["crest_1px"],
        "artifacts": SHARED_ARTIFACTS + W2_ARTIFACTS, "label": "W2 stacker, crest_1px representation",
        "historical": {"disc_dti": 0.238882, "disc_worst": 0.192575, "geo_dti_separate": 0.185609,
                       "protocol": "separate", "source": "boost A (recorded)",
                       "note": "geo 0.185609 used a geo cell chosen on geo labels: NOT a single-policy claim"}},
    "w2_skeleton": {
        "source": "a_w2_skeleton", "family": "w2", "fusion_w": "all_zones", "cell": "w2", "reps": ["skeleton_1px"],
        "artifacts": SHARED_ARTIFACTS + W2_ARTIFACTS, "label": "W2 stacker, skeleton_1px representation",
        "historical": {"disc_dti": 0.242156, "disc_worst": 0.198579, "geo_dti_separate": 0.168841,
                       "protocol": "separate", "source": "boost A (recorded)",
                       "note": "geo 0.168841 used a geo cell chosen on geo labels: NOT a single-policy claim"}},
    "b0_b1_fallback": {
        "source": "ref_m9_fallback", "family": "fusion", "fusion_w": "per_zone", "cell": "fallback_fusion",
        "reps": None, "artifacts": SHARED_ARTIFACTS, "label": "B0 + B1 rank fusion (M9 fallback, no W2)",
        "historical": {"disc_dti": 0.162482, "disc_worst": 0.126160, "geo_dti": 0.158416,
                       "protocol": "single_policy", "source": "P6.1 recovery run (recorded)"}},
}
STAGES = ("Drive preflight", "Frozen models", "Candidate recovery", "Technical validation", "GeoTIFF export",
          "Submission readback")
GRID_COLUMNS = ("task", "zone", "candidate", "blend", "rep", "frac", "tp_w", "fp_w", "fn_w")


class FactoryConfigError(ValueError):
    pass


class CandidateUnavailable(RuntimeError):
    """The candidate cannot be reconstructed from the frozen artifacts (recorded, the run continues)."""


# =============================================================================== pure: configuration
def factory_config(CONFIG: dict | None) -> dict:
    """The notebook CONFIG with defaults, checked (no training, CPU only, manual upload only)."""
    c = {**DEFAULTS, **(CONFIG or {})}
    c["run_mode"] = str(c["run_mode"]).upper()
    errs = []
    if c["run_mode"] not in RUN_MODES:
        errs.append(f"run_mode {c['run_mode']!r} not in {RUN_MODES}")
    if c["train_models"]:
        errs.append("train_models=True: the factory reuses frozen artifacts only (nothing is trained)")
    if str(c["device"]).lower() != "cpu":
        errs.append(f"device {c['device']!r}: every candidate is rebuilt from frozen maps; no GPU is needed or used")
    cands = list(c["candidates"] or [])
    bad = [x for x in cands if x not in CANDIDATES]
    if not cands or bad or len(set(cands)) != len(cands):
        errs.append(f"candidates {cands}: non-empty, distinct, from {list(CANDIDATES)} (unknown: {bad})")
    if c["geographic_target_mode"] not in TARGET_MODES:
        errs.append(f"geographic_target_mode {c['geographic_target_mode']!r} not in {TARGET_MODES}")
    if not 0.0 < float(c["geographic_target"]) < 1.0:
        errs.append("geographic_target must be in (0, 1)")
    if int(c["max_submission_files"]) < 1:
        errs.append("max_submission_files >= 1")
    if not c["save_csv"]:
        errs.append("save_csv=False is not supported: the CSV tables are the stage contract")
    if not c["manual_upload_only"]:
        errs.append("manual_upload_only must stay True: nothing is ever uploaded automatically")
    if errs:
        raise FactoryConfigError("; ".join(errs))
    c["candidates"] = cands
    if c["run_mode"] == "RESUME":
        c["reuse_cached"] = True
    return c


def plan(c: dict) -> dict:
    """What a run mode does (pure): evaluate = official recomputation, write = GeoTIFF files."""
    m = c["run_mode"]
    return {"evaluate": bool(c["evaluate_official_metric"]) and m != "EXPORT_ONLY",
            "write": bool(c["save_geotiff"]) and m != "DIAGNOSTIC"}


def m1_overrides(c: dict) -> dict:
    """configs/leaderboard_focus/m1.yaml overrides: the factory root, W2 artifacts optional (only the W2 candidates
    depend on them: a missing W2 map must not block the B0 / B1 fallback), the boost-A grid only."""
    from .config import load_m1_config

    arts = [dict(a, required=False) if a["id"] in W2_ARTIFACTS else dict(a) for a in load_m1_config()["artifacts"]]
    return {"output_dir": OUTPUT_DIR, "artifacts": arts,
            "champion": {"grids": ["boost_a"], "required_grids": ["boost_a"]},
            "outputs": {"save_figures": bool(c["save_png"]), "save_parquet": bool(c["save_parquet"]),
                        "save_geotiff": bool(c["save_geotiff"]), "sync_to_drive": bool(c["sync_to_drive"])}}


def m1_notebook(c: dict) -> dict:
    """The CONFIG shape the reused M1 functions read."""
    return {**{k: c[k] for k in ("repo", "ref", "drive_root", "local_root", "reuse_cached")},
            "strict_provenance": True}


def submission_name(cid: str, recipe_id: str) -> str:
    return f"submission_{cid}_{recipe_id}"


# =============================================================================== pure: preflight
def pf_row(check: str, scope: str, candidate: str, artifact: str, status: str, detail: str = "",
           action: str = "") -> dict:
    return {"check": check, "scope": scope, "candidate": candidate, "artifact": artifact, "status": status,
            "detail": str(detail)[:600], "action": action}


def dependency_rows(inv: pd.DataFrame, cands: list[str]) -> list[dict]:
    """Artifact inventory -> preflight rows. An artifact every enabled candidate needs is a shared row (blocks all);
    one only some need (W2) blocks only those."""
    st = {r.id: (r.status, r.note or r.pattern) for r in inv.itertuples()} if len(inv) else {}
    need = {a: [c for c in cands if a in CANDIDATES[c]["artifacts"]] for c0 in cands for a in CANDIDATES[c0]["artifacts"]}
    rows = []
    for aid, who in need.items():
        s, note = st.get(aid, ("NOT_INSPECTED", "artifact id absent from the inventory"))
        ok = s == "OK"
        act = "" if ok else f"restore '{aid}' on Drive (pattern: configs/leaderboard_focus/m1.yaml artifacts)"
        for cid in (["*"] if len(who) == len(cands) else who):
            rows.append(pf_row(f"artifact {aid}", "shared" if cid == "*" else "candidate", cid, aid,
                               PASS if ok else BLOCKED, f"{s} {note}".strip(), act))
    return rows


def compatibility_rows(C: pd.DataFrame, cands: list[str]) -> list[dict]:
    """M1 preflight cross-checks -> preflight rows: critical FAIL blocks every candidate, a failing W2 check blocks the
    W2 candidates, any other FAIL / UNVERIFIED is a WARNING (the named later stage checks it)."""
    rows = []
    w2 = [c for c in cands if "w2_maps" in CANDIDATES[c]["artifacts"]]
    for r in (C.to_dict("records") if C is not None and len(C) else []):
        chk, st = str(r["check"]), str(r["status"])
        if st == "PASS":
            rows.append(pf_row(chk, "shared", "*", "", PASS, r.get("detail", "")))
        elif st == "FAIL" and str(chk).startswith("W2 "):
            rows += [pf_row(chk, "candidate", c, "w2_maps", BLOCKED, r.get("detail", ""), "restore the W2 maps")
                     for c in w2]
        elif st == "FAIL" and str(r.get("critical")).lower() == "true":
            rows.append(pf_row(chk, "shared", "*", "", BLOCKED, r.get("detail", ""), "see DRIVE_READINESS_REPORT.md"))
        else:
            rows.append(pf_row(chk, "shared", "*", "", WARNING, f"{st}: {r.get('detail', '')}"))
    return rows


def grid_rows(G: pd.DataFrame | None, facts: dict, cands: list[str]) -> list[dict]:
    """Boost-A grid contract (shared) + the cells each candidate is evaluated on (per candidate)."""
    if G is None:
        return [pf_row("boost-A grid readable with required columns", "shared", "*", "boost_a_grid", BLOCKED,
                       f"{facts.get('status')}: {facts.get('detail')}", "restore outputs/p6_fast_track/boost/"
                       "benchmark/grid.csv")]
    rows = [pf_row("boost-A grid readable with required columns", "shared", "*", "boost_a_grid", PASS,
                   f"{len(G)} rows x {len(G.columns)} columns")]
    if (len(G), len(G.columns)) != (300, 28):
        rows.append(pf_row("boost-A grid size vs the recorded 300 x 28", "shared", "*", "boost_a_grid", WARNING,
                           f"now {len(G)} x {len(G.columns)}: inspect before trusting the numbers"))
    fin = np.isfinite(G[["tp_w", "fp_w", "fn_w"]].apply(pd.to_numeric, errors="coerce").to_numpy())
    rows.append(pf_row("boost-A components finite", "shared", "*", "boost_a_grid", PASS if fin.all() else WARNING,
                       f"{int((~fin).sum())} non-finite component cells"))
    tz = {(t, int(z)) for t, z in zip(G.task, G.zone)}
    for cid in cands:
        sp = CANDIDATES[cid]
        d = G[G.candidate.astype(str) == sp["cell"]]
        if sp["reps"]:
            d = d[d.rep.isin(sp["reps"])]
        have = {(t, int(z)) for t, z in zip(d.task, d.zone)}
        ok = bool(len(d)) and have == tz
        rows.append(pf_row(f"grid rows for cell {sp['cell']}" + (f" {sp['reps']}" if sp["reps"] else ""),
                           "candidate", cid, "boost_a_grid", PASS if ok else BLOCKED,
                           f"{len(d)} rows; task/zone units {sorted(have)} of {sorted(tz)}",
                           "" if ok else "the candidate cannot be evaluated from the frozen grid"))
    return rows


def collision_rows(sub_dir: Path, cands: list[str]) -> list[dict]:
    """Earlier files per candidate (never overwritten: a different recipe gets its own name)."""
    rows = []
    for cid in cands:
        ex = sorted(p.name for p in Path(sub_dir).glob(f"submission_{cid}_*.tif")) if Path(sub_dir).exists() else []
        rows.append(pf_row("existing output files", "candidate", cid, "submissions/", PASS,
                           f"{len(ex)} earlier file(s) kept: {ex[:4]}" if ex else "none"))
    return rows


def blockers(rows: list[dict], cands: list[str]) -> dict:
    """{candidate: [blocking reasons]} (a shared BLOCKED row blocks every candidate)."""
    out = {c: [] for c in cands}
    for r in rows:
        if r["status"] != BLOCKED:
            continue
        for c in (cands if r["candidate"] == "*" else [r["candidate"]]):
            if c in out:
                out[c].append(f"{r['check']}: {r['detail']}")
    return out


# =============================================================================== pure: metrics, verdicts, ranking
def dti(tp, fp, fn) -> float:
    from .validation import dti as _dti

    return _dti(tp, fp, fn)


def saved_metrics(P: pd.DataFrame, prim: str) -> dict:
    """Pooled / worst-zone official DTI of the candidate's nested per-zone rows (frozen grid components)."""
    out = {}
    for task, d in P.groupby("task"):
        pre = "disc" if task == prim else "geo"
        tp, fp, fn = (float(pd.to_numeric(d[c]).sum()) for c in ("tp_w", "fp_w", "fn_w"))
        z = [dti(r.tp_w, r.fp_w, r.fn_w) for r in d.itertuples()]
        out.update({f"{pre}_dti": dti(tp, fp, fn), f"{pre}_worst": float(min(z)) if z else np.nan,
                    f"{pre}_worst_zone": int(d.zone.iloc[int(np.argmin(z))]) if z else None,
                    f"{pre}_tp_w": tp, f"{pre}_fp_w": fp, f"{pre}_fn_w": fn,
                    f"{pre}_n_truth": float(pd.to_numeric(d["n_truth"]).sum()) if "n_truth" in d else np.nan})
    return out


def chk(group: str, check: str, ok, detail="", hard: bool = True) -> dict:
    return {"group": group, "check": check, "ok": bool(ok), "hard": hard, "detail": str(detail)[:800]}


def verdict(checks: list[dict]) -> str:
    """FAIL on any failed hard check, WARNING on a failed soft one, PASS otherwise; NOT_RUN without checks."""
    if not checks:
        return NOT_RUN
    if any(c["hard"] and not c["ok"] for c in checks):
        return FAIL
    return WARNING if any(not c["ok"] for c in checks) else PASS


def decide(tech: str, parity: str, fmt: str, geo_met, target_mode: str, written: bool,
           duplicate_of: str | None = None) -> str:
    """Candidate status from the independent verdicts (pure). The geographic target blocks nothing unless strict."""
    if tech not in (PASS, WARNING) or parity not in (PASS, WARNING):
        return INVALID
    if duplicate_of:
        return DUPLICATE
    if not written:
        return VALIDATED
    if fmt not in (PASS, WARNING):
        return INVALID
    if target_mode == "strict" and geo_met is not True:
        return HELD
    return READY


def rank(rows: list[dict]) -> list[dict]:
    """Predeclared upload ranking (pure; fixed before any public score exists).
    1. only READY_FOR_UPLOAD files (one per distinct array: duplicates never take a slot);
    2. p6g-eligible (discovery rule incl. worst-zone floor, selection table) before ineligible;
    3. higher nested discovery DTI (the predeclared objective 'disc'), then higher worst discovery zone, then
       registry order. PRIMARY = rank 1.
    ALTERNATIVE = the remaining READY file with the highest single-policy geographic DTI, only when it beats the
    primary's: it tests on the leaderboard whether geographic robustness or discovery strength transfers. Other
    READY files: OPTIONAL; everything else DO_NOT_UPLOAD."""
    order = list(CANDIDATES)
    ready = [r for r in rows if r.get("status") == READY]
    key = (lambda r: (not bool(r.get("p6g_eligible")), -_f(r.get("disc_dti")), -_f(r.get("disc_worst")),
                      order.index(r["candidate"]) if r["candidate"] in order else 99))
    ready.sort(key=key)
    out = {r["candidate"]: {"candidate": r["candidate"], "recommendation_rank": None, "upload_priority":
                            "DO_NOT_UPLOAD", "rationale": f"status {r.get('status')}"} for r in rows}
    if not ready:
        return list(out.values())
    p = ready[0]
    for i, r in enumerate(ready, 1):
        out[r["candidate"]].update(recommendation_rank=i, upload_priority="OPTIONAL",
                                   rationale="READY; lower in the predeclared order")
    out[p["candidate"]].update(upload_priority="PRIMARY", rationale=(
        f"rank 1 of the predeclared rule: p6g eligible={bool(p.get('p6g_eligible'))}, disc DTI "
        f"{_f(p.get('disc_dti')):.6f}, worst disc zone {_f(p.get('disc_worst')):.6f}"))
    rest = [r for r in ready[1:] if _f(r.get("geo_dti")) > _f(p.get("geo_dti"))]
    if rest:
        a = max(rest, key=lambda r: _f(r.get("geo_dti")))
        out[a["candidate"]].update(upload_priority="ALTERNATIVE", rationale=(
            f"highest single-policy geo DTI {_f(a.get('geo_dti')):.6f} > primary {_f(p.get('geo_dti')):.6f} (disc "
            f"{_f(a.get('disc_dti')):.6f} vs {_f(p.get('disc_dti')):.6f}): its public score tells whether the "
            "leaderboard rewards geographic robustness or discovery strength"))
    return list(out.values())


def _f(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("-inf")
    return v if np.isfinite(v) else float("-inf")


def readback_checks(facts: dict, tpl: dict, exp: dict, size_mb: float, max_mb: float, validated_sha: str,
                    written_sha: str, side_fp, rec_fp: str, file_sha) -> list[dict]:
    """Checks on the GeoTIFF read back from disk (pure; ``facts`` = submission.raster_facts)."""
    a = facts["array"]
    v = a[np.isfinite(a)]
    tr = list(tpl["transform"])[:6]
    nod = facts["nodata"]
    return [
        chk("readback", "file non-empty with a sha256", bool(file_sha) and size_mb > 0, f"{size_mb:.3f} MB"),
        chk("readback", f"band count == {exp['count']}", facts["count"] == int(exp["count"]), facts["count"]),
        chk("readback", f"dtype == {exp['dtype']}", facts["dtype"] == exp["dtype"], facts["dtype"]),
        chk("readback", f"EPSG == {exp['epsg']}", facts["epsg"] == int(exp["epsg"]), facts["epsg"]),
        chk("readback", f"resolution == {exp['res_m']} m", all(abs(r - float(exp["res_m"])) < 1e-6 for r in
                                                               facts["res"]), facts["res"]),
        chk("readback", "shape == template", list(facts["shape"]) == [int(tpl["height"]), int(tpl["width"])],
            facts["shape"]),
        chk("readback", "transform (bounds) == template", np.allclose(facts["transform"], tr), facts["transform"]),
        chk("readback", "nodata tag is NaN", nod is not None and isinstance(nod, float) and np.isnan(nod), nod),
        chk("readback", "values in [0, 1]", bool(v.size) and float(v.min()) >= 0.0 and float(v.max()) <= 1.0,
            f"{v.min() if v.size else None}..{v.max() if v.size else None}"),
        chk("readback", "non-empty prediction", bool((v > 0).any()), f"{int((v > 0).sum())} positive px"),
        chk("readback", f"file size <= {max_mb} MB", size_mb <= float(max_mb), f"{size_mb:.2f} MB"),
        chk("readback_parity", "read-back band == validated array (sha256, NaN pattern included)",
            written_sha == validated_sha, f"{str(written_sha)[:12]} vs {str(validated_sha)[:12]}"),
        chk("readback", "sidecar recipe fingerprint == recipe", side_fp == rec_fp, f"{side_fp} vs {rec_fp}"),
    ]


# =============================================================================== Colab
class Factory:
    def __init__(self, CONFIG: dict, head: str):
        from ..artifacts.sync import DriveSync
        from .config import load_m1_config
        from .session import Workspace

        self.c = factory_config(CONFIG)
        self.head = head
        self.plan = plan(self.c)
        self.overrides = m1_overrides(self.c)
        self.cfg = load_m1_config(self.overrides)
        self.local, self.drive = Path(self.c["local_root"]), Path(self.c["drive_root"])
        self.ws = Workspace(self.local / OUTPUT_DIR, DriveSync(self.local, self.drive) if self.c["sync_to_drive"]
                            else None, self.local, save_figures=bool(self.c["save_png"]),
                            save_parquet=bool(self.c["save_parquet"]))
        self.sess = None
        self.errors: list[dict] = []
        self.stage: dict[str, str] = {s: NOT_RUN for s in STAGES}
        self.selftests: pd.DataFrame | None = None
        self.pf_rows: list[dict] = []
        self.block: dict[str, list[str]] = {}
        self.records: dict[str, dict] = {}
        self.seen: dict[str, str] = {}                # validated array sha -> first candidate (dedup)
        self.n_files = 0

    # ------------------------------------------------------------------------------- plumbing
    def say(self, i: int, name: str, status: str, note: str = "") -> None:
        print(f"[{i}/6] {name} {'.' * max(2, 22 - len(name))} {status}" + (f" | {note}" if note else ""))

    def error(self, cid: str, stage: str, ftype: str, artifact: str, explanation: str, tb: str = "") -> None:
        self.errors.append({"candidate": cid, "stage": stage, "failure_type": ftype, "artifact": artifact,
                            "explanation": str(explanation)[:2000], "traceback": tb[-3000:],
                            "time": time.strftime("%Y-%m-%dT%H:%M:%S")})

    def drive_path(self, rel: str) -> str:
        return str(self.drive / OUTPUT_DIR / rel)

    # ------------------------------------------------------------------------------- self-tests
    def selftest(self) -> pd.DataFrame:
        """Synthetic runtime self-tests (no Drive, no GPU); a failed critical test disables every export."""
        import tempfile

        from .factory_selftest import run_selftests

        with tempfile.TemporaryDirectory() as d:
            T = run_selftests(Path(d))
        self.selftests = T
        self.ws.save_table(T, "diagnostics/selftests.csv")
        bad = T[(T.status != PASS) & T.critical.astype(bool)]
        print(f"[self-test] {int((T.status == PASS).sum())}/{len(T)} PASS" +
              (f" | CRITICAL FAIL: {list(bad.test)} -> exports disabled" if len(bad) else ""))
        return T

    def selftest_ok(self) -> bool:
        T = self.selftests
        return T is not None and not ((T.status != PASS) & T.critical.astype(bool)).any()

    # ------------------------------------------------------------------------------- [1/6] preflight
    def preflight(self) -> pd.DataFrame:
        from ..artifacts.sync import DriveSync
        from ..p6 import full_benchmark as FB
        from . import artifacts as AR
        from . import preflight as PF

        cands, nb = self.c["candidates"], m1_notebook(self.c)
        if (self.drive / OUTPUT_DIR).exists():
            DriveSync(self.local, self.drive).stage_in([OUTPUT_DIR])          # earlier factory outputs (resume)
        rows = []
        try:
            pf = PF.run(nb, self.overrides)
            rows += compatibility_rows(self.ws.load_table("diagnostics/artifact_compatibility.csv"), cands)
            env = AR.run_environment(nb, self.head, self.overrides)
            inv = env["inventory"]
            self.ws.save_table(inv, "diagnostics/artifact_status.csv")
            rows += dependency_rows(inv, cands)
            note = f"M1 preflight {pf.get('status')}, inventory {env['summary']['status']}"
        except Exception as e:
            self.error("*", "preflight", "EXCEPTION", "Drive", f"{type(e).__name__}: {e}", traceback.format_exc())
            rows.append(pf_row("Drive preflight ran", "shared", "*", "Drive", BLOCKED, f"{type(e).__name__}: {e}",
                               "is Drive mounted at drive_root? see diagnostics/runtime_errors.csv"))
            note = "preflight raised"
        gp = self.drive / self.cfg["p6_output_dir"] / "boost" / "benchmark" / "grid.csv"
        G, facts = FB.read_csv_checked(gp, GRID_COLUMNS)
        rows += grid_rows(G, facts, cands)
        rows += collision_rows(self.local / OUTPUT_DIR / "submissions", cands)
        if self.selftests is None:
            rows.append(pf_row("runtime self-tests", "shared", "*", "", WARNING, "not run: call F.selftest() first "
                               "(F.run() runs them)"))
        elif not self.selftest_ok():
            rows.append(pf_row("runtime self-tests", "shared", "*", "", BLOCKED, "a critical self-test failed "
                               "(diagnostics/selftests.csv): exports disabled"))
        else:
            rows.append(pf_row("runtime self-tests", "shared", "*", "", PASS, f"{len(self.selftests)} tests"))
        self.pf_rows, self.block = rows, blockers(rows, cands)
        T = pd.DataFrame(rows)
        self.ws.save_table(T, "diagnostics/preflight.csv")
        self.ws.save_json({"status": self._pf_status(), "blockers": self.block, "rows": rows}, "diagnostics/"
                          "preflight.json")
        self.ws.save_text(self.preflight_md(T), "reports/PREFLIGHT.md")
        self.ws.push()
        self.stage[STAGES[0]] = self._pf_status()
        n_ok = sum(1 for c in cands if not self.block[c])
        self.say(1, STAGES[0], self.stage[STAGES[0]], f"{n_ok}/{len(cands)} candidates unblocked; {note}")
        for c, why in self.block.items():
            if why:
                print(f"      {c}: BLOCKED - {why[0]}")
        return T

    def _pf_status(self) -> str:
        n = [c for c in self.c["candidates"] if self.block.get(c)]
        if len(n) == len(self.c["candidates"]):
            return BLOCKED
        return WARNING if n or any(r["status"] == WARNING for r in self.pf_rows) else PASS

    def preflight_md(self, T: pd.DataFrame) -> str:
        from .reporting import _md

        return "\n".join(["# Submission factory preflight", "", f"Status: **{self._pf_status()}**", "",
                          "PASS / WARNING / BLOCKED. A shared BLOCKED row blocks every candidate; a candidate row "
                          "blocks only that candidate.", "", "## Blockers", "",
                          *[f"* `{c}`: " + ("; ".join(w) if w else "none") for c, w in self.block.items()], "",
                          "## Checks", "", _md(T[["status", "scope", "candidate", "check", "detail", "action"]]
                                              if len(T) else T), ""])

    # ------------------------------------------------------------------------------- [2/6] frozen models
    def attach(self) -> bool:
        """Engine + the shared candidate table (champion.recover) + template spec + fold geometry."""
        from ..p6 import final_submission as FS
        from . import champion as CH
        from .session import M1Session

        live = [c for c in self.c["candidates"] if not self.block.get(c)]
        if not live:
            self.stage[STAGES[1]] = NOT_RUN
            self.say(2, STAGES[1], NOT_RUN, "every candidate is blocked by the preflight")
            return False
        try:
            self.sess = M1Session.attach(m1_notebook(self.c), self.overrides)
            sel = CH.recover(self.sess)
            self.C = self.ws.load_table("selection/champion_candidates.csv")
            self.PZ = self.ws.load_table("selection/candidate_per_zone.csv")
            if not len(self.C) or not len(self.PZ):
                raise CandidateUnavailable(f"shared candidate table absent (selection {sel.get('status')}: "
                                           f"{(sel.get('reasons') or [''])[0]})")
            self.sel = self.ws.read_json("selection/champion_selection.json") or {}
            fg = self.sess.fg
            self.objective = self.sel.get("objective") or fg.cfg["selection"]["objective"]
            self.w_all = float(fg.bst.sel[fg.prim][fg.p6.cfg["probes"]["champion"]]["params_all"]["w"])
            self.sel_fp = self.sess.manifest("selection").get("fingerprint")
            self.data_fp = fg.data_fingerprint()
            self.spec = FS.inspect_spec(fg)              # raises on a template / grid mismatch (blocks every export)
            self.geometry = self.fold_geometry()
            self.ws.save_table(pd.concat([self.ws.load_table("diagnostics/artifact_compatibility.csv").assign(
                source="preflight"), pd.DataFrame(self.geometry).assign(source="fold geometry")], ignore_index=True),
                "diagnostics/source_compatibility.csv")
        except Exception as e:
            self.error("*", "frozen models", "EXCEPTION" if not isinstance(e, CandidateUnavailable) else "UNAVAILABLE",
                       "engine / selection / template", f"{type(e).__name__}: {e}", traceback.format_exc())
            for c in live:
                self.block[c].append(f"frozen models: {type(e).__name__}: {e}")
            self.stage[STAGES[1]] = FAIL
            self.say(2, STAGES[1], FAIL, f"{type(e).__name__}: {str(e)[:200]}")
            return False
        geo_bad = [g for g in self.geometry if not g["ok"]]
        self.stage[STAGES[1]] = PASS if not geo_bad else FAIL
        self.say(2, STAGES[1], self.stage[STAGES[1]], f"selection {sel.get('status')}, objective {self.objective}, "
                 f"template EPSG {self.spec.get('epsg')} {self.spec.get('shape')}" +
                 (f"; fold geometry FAIL {[g['check'] for g in geo_bad]}" if geo_bad else ""))
        return True

    def fold_geometry(self) -> list[dict]:
        """Fold assignment per geo unit, duplicated / missing pixels across zones (shared by every candidate)."""
        from ..p6.final_submission import zone_partition
        from . import parity as PA

        fg = self.sess.fg
        data, fm = fg.p6.ctx.data, np.asarray(fg.p6.ctx.folds.fold_map)
        valid = np.asarray(data.valid, bool)
        geo = [u for u in fg.bst.units() if u.task == "geo"]
        out, masks = [], {}
        for u in geo:
            fa = PA.fold_assignment(u.ev, valid[u.sl], fm[u.sl], u.zone)
            out.append(chk("geometry", f"fold assignment {u.key} (unit mask == valid & fold_map == k)",
                           fa["status"] == "PASS", f"{fa['mismatch_px']} px differ"))
            m = np.zeros(valid.shape, bool)
            m[u.sl] = u.ev
            masks[u.zone] = m
            u.release()
        ov = PA.zone_overlaps(masks, valid)
        _, cov = zone_partition(fm, valid, [u.zone for u in geo])
        tol = float(self.cfg["validation"]["max_uncovered_frac"])
        out.append(chk("geometry", "no duplicated pixel across geo zones", ov["duplicated_px"] == 0, json.dumps(ov)))
        out.append(chk("geometry", f"uncovered valid pixels <= {tol:g}", cov["uncovered_frac"] <= tol,
                       json.dumps(cov, default=str)))
        del masks
        gc.collect()
        return out

    # ------------------------------------------------------------------------------- one candidate
    def candidate(self, cid: str) -> dict:
        """Recover, evaluate, build, validate, write and read back one candidate (exceptions isolated)."""
        spec = CANDIDATES[cid]
        rec = {"candidate": cid, "source_candidate": spec["source"], "family": spec["family"], "label": spec["label"],
               "status": BLOCKED, "stage_reached": "preflight", "reasons": [], "warnings": [],
               "TECHNICAL_VALIDATION": NOT_RUN, "EXPORT_PARITY": NOT_RUN, "OFFICIAL_FORMAT": NOT_RUN,
               "GEOGRAPHIC_TARGET_MET": None, "READY_FOR_UPLOAD": False, "evaluation": NOT_RUN,
               "historical": spec["historical"], "uploaded": False}
        if self.block.get(cid):
            rec["reasons"] = list(self.block[cid])
            self.error(cid, "preflight", "BLOCKED_DEPENDENCY", "; ".join(spec["artifacts"]), rec["reasons"][0])
            return self.persist(rec)
        st = "recovery"
        try:
            row, P, cell, recipe = self.recover(cid)
            rec.update(recipe_id=recipe["recipe_id"], recipe=recipe, stage_reached=st)
            fp = self.fp(cid, recipe)
            rec["fingerprint"] = fp
            cached = self.cached(cid, fp)
            if cached is not None:
                return self.persist(cached, push=False)
            st = "evaluation"
            rec.update(self.metrics(row, P, cid), stage_reached=st)
            ev = self.evaluate(P, row)
            rec.update(evaluation=ev["status"], evaluation_detail=ev["detail"], per_zone=ev["per_zone"])
            st = "build"
            out = self.build_and_check(cid, rec, row, P, cell, recipe, ev)
            rec.update(out)
            return self.persist(rec)
        except CandidateUnavailable as e:
            rec.update(status=UNAVAILABLE, stage_reached=st, reasons=[str(e)])
            self.error(cid, st, "UNAVAILABLE", spec["source"], str(e))
        except FileExistsError as e:
            rec.update(status=BLOCKED, stage_reached=st, reasons=[f"output collision: {e}"])
            self.error(cid, st, "WRITE_COLLISION", "submissions/", str(e))
        except Exception as e:                           # isolated: the next candidate still runs
            rec.update(status=FAILED, stage_reached=st, reasons=[f"{type(e).__name__}: {e}"])
            self.error(cid, st, "EXCEPTION", spec["source"], f"{type(e).__name__}: {e}", traceback.format_exc())
        finally:
            gc.collect()
        return self.persist(rec)

    def recover(self, cid: str):
        """(selection row, nested per-zone rows, deploy cell, frozen recipe) of one candidate."""
        from . import champion as CH

        spec, fg = CANDIDATES[cid], self.sess.fg
        C = self.C[self.C.candidate_id == spec["source"]]
        if not len(C):
            raise CandidateUnavailable(f"{spec['source']} absent from selection/champion_candidates.csv")
        row = C.iloc[0].to_dict()
        if row.get("status") != "EVALUATED" or not isinstance(row.get("deploy_cell"), str):
            raise CandidateUnavailable(f"{spec['source']}: status {row.get('status')} "
                                       f"({row.get('status_reason') or 'no all-zone deploy cell'})")
        if row.get("deploy") not in ("stack", "fusion"):
            raise CandidateUnavailable(f"{spec['source']}: deploy kind {row.get('deploy')} has no frozen builder")
        cell = json.loads(row["deploy_cell"])
        P = self.PZ[self.PZ.name == spec["source"]].reset_index(drop=True)
        if not len(P):
            raise CandidateUnavailable(f"{spec['source']}: no nested per-zone rows in selection/candidate_per_zone.csv")
        recipe = CH.recipe_spec({"name": spec["source"], "deploy": row["deploy"]}, cell,
                                CH.map_sources(self.sess, row["deploy"]), self.objective, self.w_all, fg.p6.m9.sh)
        if spec["fusion_w"] == "per_zone":
            recipe["policy"]["fusion_w_mode"] = "per_zone"
            recipe["score"] = ("w_k rank01(B0) + (1 - w_k) rank01(B1) inside each geo zone k; w_k = the frozen nested "
                               "per-zone champion weight (exactly as the benchmark evaluated the fusion)")
        recipe["recipe_id"] = self.sess.fp("recipe", recipe)[:10]
        return row, P, cell, recipe

    def fp(self, cid: str, recipe: dict) -> str:
        """Cache key: recipe (source hashes inside) + shared selection (inventory, grid, folds) + data + code + the
        settings that change the verdicts."""
        from ..pipeline.m6 import fingerprint
        from .session import impl_fp

        keys = ("geographic_target", "geographic_target_mode", "validate_export_parity")
        return fingerprint(FACTORY_VERSION, cid, recipe, self.sel_fp, self.data_fp, impl_fp("factory"), self.plan,
                           {k: self.c[k] for k in keys}, self.cfg["validation"], self.cfg["submission"],
                           self.sess.fg.cfg["submission"], self.selftest_ok())

    def cached(self, cid: str, fp: str) -> dict | None:
        """A reusable earlier record (same fingerprint, READY / VALIDATED / HELD, file unchanged on disk)."""
        from ..artifacts.atomic import sha256_file

        if not self.c["reuse_cached"] or self.c["overwrite_existing"]:
            return None
        m = self.ws.read_json(f"manifests/candidates/{cid}.json") or {}
        if m.get("fingerprint") != fp or m.get("status") not in REUSABLE:
            return None
        vs = m.get("validated_array_sha256")
        if vs in self.seen:                             # an earlier candidate of this run has the same array
            return None
        if m.get("file"):
            p = self.ws.path(f"submissions/{m['file']}")
            if not p.exists() or sha256_file(p) != m.get("sha256"):
                print(f"      {cid}: cached file missing or changed -> rebuilt")
                return None
            self.n_files += 1
        self.seen[vs] = cid
        print(f"      {cid}: SKIP_CACHED (fingerprint {fp[:10]}, status {m['status']})")
        return {**m, "cache": "SKIP_CACHED"}

    def metrics(self, row: dict, P: pd.DataFrame, cid: str) -> dict:
        prim = self.sess.fg.prim
        s = saved_metrics(P, prim)
        geo, tgt = s.get("geo_dti"), float(self.c["geographic_target"])
        return {**{k: v for k, v in s.items()}, "metric_source": "frozen boost-A grid: official components of the "
                "nested single-policy held-out cells (same folds, eligible area, scorer for every candidate)",
                "nested_disc_dti": row.get("disc_dti"), "nested_geo_dti": row.get("geo_dti"),
                "p6g_eligible": str(row.get("eligible")).strip().lower() == "true",
                "p6g_eligibility_reasons": row.get("eligibility_reasons"), "deploy_cell": json.loads(row["deploy_cell"]),
                "deploy": row["deploy"], "geographic_target": tgt,
                "GEOGRAPHIC_TARGET_MET": bool(np.isfinite(geo) and geo >= tgt) if geo is not None else None}

    def evaluate(self, P: pd.DataFrame, row: dict) -> dict:
        """Official recomputation (validation.recompute) vs the frozen grid + pooling vs the nested table."""
        from . import validation as VA

        Z = P[["task", "zone", "cell_candidate", "blend", "rep", "frac", "tp_w", "fp_w", "fn_w"]].copy()
        Z["dti"] = [dti(r.tp_w, r.fp_w, r.fn_w) for r in Z.itertuples()]
        if not self.plan["evaluate"]:
            return {"status": NOT_RUN, "detail": f"run_mode {self.c['run_mode']}: frozen grid components used, not "
                    "recomputed", "per_zone": Z.to_dict("records"), "V": None}
        vc = self.cfg["validation"]
        V = VA.recompute(self.sess, P, vc)
        pl = VA.pooled(V, self.sess.fg.prim) if len(V) else {}
        why = [f"{r.task}:{r.zone} components differ from the frozen grid" for r in V.itertuples() if r.status != "PASS"]
        for pre in ("disc", "geo"):
            a, b = pl.get(f"{pre}_dti_recomputed"), row.get(f"{pre}_dti")
            if a is None or b is None or not VA.close(a, b, 0.0, float(vc["dti_atol"])):
                why.append(f"{pre} pooled recomputed {a} != nested {b}")
        Z = Z.merge(V[["task", "zone", "recomputed_dti", "status", "area_px", "area_frac_eligible"]].rename(
            columns={"status": "recompute_status"}), on=["task", "zone"], how="left")
        return {"status": FAIL if why else PASS, "detail": "; ".join(why) or "components == frozen grid (official "
                "scorer, rtol / atol from m1.yaml validation)", "per_zone": Z.to_dict("records"), "V": V}

    def build_and_check(self, cid: str, rec: dict, row: dict, P: pd.DataFrame, cell: dict, recipe: dict,
                        ev: dict) -> dict:
        from ..artifacts.atomic import sha256_file
        from ..gems3 import submission as SUB
        from ..p6 import export_parity as EP
        from ..p6 import final_submission as FS
        from . import forensics as FO
        from . import submission as SU
        from . import validation as VA

        spec, fg = CANDIDATES[cid], self.sess.fg
        data, pc = fg.p6.ctx.data, fg.cfg["submission"]
        brow = {"name": spec["source"], "deploy": row["deploy"], "deploy_cell": cell, "space": ""}
        built = FS.build_map(fg, brow, fusion_w=spec["fusion_w"])
        # ---------------------------------------------------------------- technical validation (in memory)
        tech = [chk("artifact", "candidate EVALUATED in the shared nested table", row.get("status") == "EVALUATED")]
        tech.append(chk("artifact", "official components recomputed == frozen grid", ev["status"] != FAIL,
                        ev["detail"], hard=ev["status"] != NOT_RUN))
        if ev["status"] == NOT_RUN:
            tech[-1].update(ok=False, hard=False)
        diff = VA.recipe_consistency(recipe, built)
        tech.append(chk("artifact", "builder cell / kind / maps == frozen recipe", not diff, "; ".join(diff)))
        if self.sel.get("champion") == spec["source"]:
            y = self._champion_recipe_id()
            tech.append(chk("artifact", "recipe id == selection-stage champion recipe", y == recipe["recipe_id"],
                            f"{recipe['recipe_id']} vs champion_recipe.yaml {y}"))
        h = spec["historical"].get("recipe_id")
        if h:
            tech.append(chk("artifact", f"recipe id == recorded {h}", recipe["recipe_id"] == h,
                            "a different id means a frozen source hash changed (recorded, not blocking)", hard=False))
        tech += [dict(g, group="geometry") for g in self.geometry]
        cov = built["coverage"]
        tech.append(chk("coverage", f"uncovered valid px <= {self.cfg['validation']['max_uncovered_frac']:g}",
                        cov["uncovered_frac"] <= float(self.cfg["validation"]["max_uncovered_frac"]),
                        json.dumps(cov, default=str)))
        Z = VA.budget_rows(built["zones"], float(cell["frac"]), float(pc["budget_tolerance"]),
                           FO.evaluated_area(ev["V"]) if ev["V"] is not None else None)
        tech.append(chk("budget", "per-zone area <= frac x (1 + tol), non-empty", bool(Z.ok.all()),
                        "; ".join(f"zone {int(r.zone)}: {r.cause}" for r in Z.itertuples())))
        known = int(built["zones"].positive_on_known_px.sum())
        tech.append(chk("mask", "no positive pixel on a mapped fault (official mask)", known == 0, f"{known} px"))
        arr = SUB.finalise(built["q"], data, pc["outside_valid"])
        tv = np.asarray(data.template_valid, bool)
        fin = np.isfinite(arr)
        tech.append(chk("array", "finalised array: finite exactly where the template is finite, values in [0, 1]",
                        bool(fin[tv].all()) and not bool(fin[~tv].any()) and float(np.nanmin(arr)) >= 0.0 and
                        float(np.nanmax(arr)) <= 1.0, f"{int(fin.sum())} finite px"))
        vsha = VA.array_sha(arr)
        pos = int((np.nan_to_num(arr) > 0).sum())
        tech.append(chk("array", "non-empty prediction", pos > 0, f"{pos} positive px"))
        # ---------------------------------------------------------------- export parity (in memory)
        par = built["parity"]
        zs = par[par.check == "zone score vs evaluation construction"]
        bm = par[par.check == "binary map vs evaluation map"]
        parity = [chk("parity", f"zone score == evaluated score {r.unit} (Spearman >= "
                      f"{pc['parity_min_spearman']})", r.status == "PASS", f"spearman {getattr(r, 'spearman', None)} "
                      f"({r.status})") for r in zs.itertuples()]
        parity += [chk("parity", f"binary map vs evaluation map {r.unit}", r.status == "PASS",
                       f"jaccard {getattr(r, 'jaccard', None)}: {r.note}", hard=False) for r in bm.itertuples()]
        if not self.c["validate_export_parity"]:
            parity = [dict(p, hard=False) for p in parity]
        out = {"zones": built["zones"].assign(fusion_w=[built["w_zone"].get(int(z), built["w_all"])
                                                        for z in built["zones"].zone]).to_dict("records"),
               "parity": par.drop(columns=[c for c in par.columns if c.startswith("q")], errors="ignore").to_dict(
                   "records"), "budget": Z.to_dict("records"), "coverage": cov, "positive_px": pos,
               "validated_array_sha256": vsha, "stage_reached": "technical"}
        tv_st, par_st = verdict(tech), verdict(parity)
        out.update(TECHNICAL_VALIDATION=tv_st, EXPORT_PARITY=par_st)
        geo_met = rec.get("GEOGRAPHIC_TARGET_MET")
        reasons = [f"{c['group']}: {c['check']} ({c['detail']})" for c in tech + parity if c["hard"] and not c["ok"]]
        warns = [f"{c['group']}: {c['check']} ({c['detail']})" for c in tech + parity if not c["hard"] and not c["ok"]]
        if geo_met is False:
            warns.append(f"GEOGRAPHIC_TARGET_MET False: single-policy geo DTI {rec.get('geo_dti'):.6f} < internal "
                         f"target {self.c['geographic_target']} (geographic-risk warning; not a format problem)")
        dup = self.seen.get(vsha) if tv_st in (PASS, WARNING) and par_st in (PASS, WARNING) else None
        write = self.plan["write"] and tv_st in (PASS, WARNING) and par_st in (PASS, WARNING) and not dup
        if write and not self.selftest_ok():
            write = False
            reasons.append("critical runtime self-test failed or not run: no file written")
        if write and self.n_files >= int(self.c["max_submission_files"]):
            write = False
            warns.append(f"max_submission_files {self.c['max_submission_files']} reached: validated, not written")
        if write and not self.c["export_all_technically_valid"] and self.n_files >= 1:
            write = False
            warns.append("export_all_technically_valid=False: only the first valid candidate is written")
        if tv_st in (PASS, WARNING) and par_st in (PASS, WARNING) and not dup:
            self.seen[vsha] = cid
        out["duplicate_of"] = dup
        if dup:
            warns.append(f"identical raster to {dup} (validated array sha256 {vsha[:12]}): no redundant file")
        checks = tech + parity
        n0 = len(checks)
        fmt, written = NOT_RUN, False
        if write:
            # ------------------------------------------------------------ GeoTIFF export
            out["stage_reached"] = "export"
            rp = FS.recipe(fg, brow, built, self.objective)
            pa = EP.plan_audit(rp)
            checks.append(chk("export", "recipe plan audit", pa["status"] == "PASS", json.dumps(pa["checks"])))
            if pa["status"] != "PASS":
                fmt = FAIL
            else:
                ledger = [SUB.area_ledger(built["q"], data, 0, f"{cid} zone-stitched {cell['rep']} @ "
                                                               f"{float(cell['frac']):g}")]
                name = submission_name(cid, recipe["recipe_id"])
                w = FS.write_file(fg, name, built["q"], rp, ledger)
                path = w["path"]
                written = True
                self.n_files += 1
                # -------------------------------------------------------- readback (the file on disk)
                out["stage_reached"] = "readback"
                v = FS.validate(fg, path, rp, built)
                checks += [chk(f"official:{g}", c["check"], c["ok"], c.get("detail", ""), c.get("hard", True))
                           for g, cs in v["groups"].items() for c in cs]
                facts = SU.raster_facts(path)
                wsha = VA.array_sha(facts["array"])
                side = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
                fsha = sha256_file(path)
                size_mb = path.stat().st_size / 2 ** 20
                rb = readback_checks(facts, data.template_profile, self.cfg["submission"]["expected"], size_mb,
                                     float(self.cfg["submission"]["max_size_mb"]), vsha, wsha, side.get("recipe_fp"),
                                     rp["fingerprint"], fsha)
                checks += rb
                fmt = verdict([c for c in checks if c["group"].startswith("official:") or c["group"] == "readback"])
                par_st = verdict(parity + [c for c in rb if c["group"] == "readback_parity"])
                out.update(EXPORT_PARITY=par_st)
                f_ = facts
                out.update(file=path.name, path_local=str(path), path_drive=self.drive_path(f"submissions/{path.name}"),
                           sha256=fsha, size_mb=round(size_mb, 4), shape=f_["shape"], crs=f"EPSG:{f_['epsg']}",
                           dtype=f_["dtype"], nodata=str(f_["nodata"]), reused_existing_file=w["reused"],
                           recipe_fp=rp["fingerprint"], written_array_sha256=wsha)
                del facts, f_
                self.ws.save_json({"checks": checks, "p6_validation": {k: v.get(k) for k in ("status", "failed",
                                                                                            "official_facts", "audit")}},
                                  f"submissions/{path.stem}_validation.json")
                if self.c["save_png"]:
                    try:
                        FS.preview(fg, path.stem, built["q"], f"{cid} ({cell['rep']} @ {float(cell['frac']):g})")
                    except Exception as e:                   # a preview never decides validity; recorded
                        warns.append(f"preview not written: {type(e).__name__}: {e}")
        reasons += [f"{c['group']}: {c['check']} ({c['detail']})" for c in checks[n0:] if c["hard"] and not c["ok"]]
        warns += [f"{c['group']}: {c['check']} ({c['detail']})" for c in checks[n0:] if not c["hard"] and not c["ok"]]
        status = decide(tv_st, out["EXPORT_PARITY"], fmt, geo_met, self.c["geographic_target_mode"],
                        written or fmt == FAIL, dup)
        if status == INVALID and written:
            reasons.append("file written but a hard check failed on read-back: INVALID, never recommended")
        out.update(OFFICIAL_FORMAT=fmt, status=status, READY_FOR_UPLOAD=status == READY, reasons=reasons,
                   warnings=warns, checks=checks)
        del built, arr
        return out

    def _champion_recipe_id(self) -> str | None:
        import yaml

        p = self.ws.path("selection/champion_recipe.yaml")
        try:
            y = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else None
        except yaml.YAMLError:
            y = None
        return y.get("recipe_id") if isinstance(y, dict) else None

    def persist(self, rec: dict, push: bool = True) -> dict:
        rec["written"] = rec.get("written") if rec.get("cache") else time.strftime("%Y-%m-%dT%H:%M:%S")
        self.records[rec["candidate"]] = rec
        self.ws.save_json(rec, f"manifests/candidates/{rec['candidate']}.json")
        if push:
            self.ws.push()                                  # persisted as soon as the candidate is done
        f = f" -> {rec.get('file')}" if rec.get("file") else ""
        why = (rec.get("reasons") or rec.get("warnings") or [""])[0]
        print(f"      {rec['candidate']:<15} {rec['status']}{f}" + (f" | {str(why)[:160]}" if why else ""))
        return rec

    # ------------------------------------------------------------------------------- RUN_ALL
    def run(self) -> dict:
        t0 = time.perf_counter()
        if self.selftests is None:
            self.selftest()
        self.preflight()
        ok = self.attach()
        for cid in self.c["candidates"]:
            if not ok and not self.block.get(cid):
                self.block[cid] = ["frozen models stage did not complete (diagnostics/runtime_errors.csv)"]
            self.candidate(cid)
        self.finish(round(time.perf_counter() - t0, 1))
        return self.records

    def finish(self, seconds: float | None = None) -> None:
        R = list(self.records.values())
        n = len(self.c["candidates"])

        def agg(i, name, ok_fn, ran_fn):
            ran = [r for r in R if ran_fn(r)]
            k = sum(1 for r in ran if ok_fn(r))
            st = NOT_RUN if not ran else PASS if k == n else FAIL if k == 0 else "PARTIAL"
            self.stage[name] = st
            self.say(i, name, f"{st} ({k}/{n})")

        agg(3, STAGES[2], lambda r: bool(r.get("recipe_id")), lambda r: r["stage_reached"] != "preflight")
        agg(4, STAGES[3], lambda r: r["TECHNICAL_VALIDATION"] in (PASS, WARNING) and r["EXPORT_PARITY"] in
            (PASS, WARNING), lambda r: r["TECHNICAL_VALIDATION"] != NOT_RUN)
        agg(5, STAGES[4], lambda r: bool(r.get("file")), lambda r: r["status"] not in (BLOCKED, UNAVAILABLE))
        agg(6, STAGES[5], lambda r: r["status"] == READY, lambda r: bool(r.get("file")))
        self.outputs(seconds)
        rk = {r["candidate"]: r for r in self.ws.load_table("submissions/submission_ranking.csv").to_dict("records")}
        ready = [r for r in R if r["status"] == READY]
        print("\nREADY_FOR_UPLOAD:" + ("" if ready else " none (reports/FAILURE_DIAGNOSTICS.md)"))
        for r in sorted(ready, key=lambda r: rk.get(r["candidate"], {}).get("recommendation_rank") or 99):
            print(f"  [{rk.get(r['candidate'], {}).get('upload_priority')}] {r['path_drive']}")
            print(f"      sha256 {r['sha256']} | disc {r.get('disc_dti'):.6f} | geo {r.get('geo_dti'):.6f} | "
                  f"geo target met: {r.get('GEOGRAPHIC_TARGET_MET')}")
        print(f"\nUpload instructions: {self.drive_path('submissions/README_UPLOAD.md')}")

    # ------------------------------------------------------------------------------- outputs
    def outputs(self, seconds: float | None = None) -> None:
        """Every table / figure / report from the PERSISTED candidate manifests (not from notebook variables)."""
        from . import factory_report as FR

        recs = [self.ws.read_json(f"manifests/candidates/{c}.json") or {"candidate": c, "status": NOT_RUN}
                for c in self.c["candidates"]]
        E = pd.DataFrame(self.errors, columns=["candidate", "stage", "failure_type", "artifact", "explanation",
                                               "traceback", "time"])
        self.ws.save_table(E, "diagnostics/runtime_errors.csv")
        self.ws.save_json(self.errors, "diagnostics/runtime_errors.json")
        FR.write_all(self.ws, recs, rank(recs), self.c, self.stage, {
            "factory_version": FACTORY_VERSION, "code_rev": FACTORY_CODE_REV, "commit": self.head,
            "run_mode": self.c["run_mode"], "plan": self.plan, "seconds": seconds,
            "selection_fingerprint": getattr(self, "sel_fp", None), "data_fingerprint": getattr(self, "data_fp", None),
            "objective": getattr(self, "objective", None), "official_spec": getattr(self, "spec", None),
            "selftests_ok": self.selftest_ok(), "n_errors": len(self.errors), "drive_dir": self.drive_path(""),
            "written": time.strftime("%Y-%m-%dT%H:%M:%S")})
        self.ws.push()

    # ------------------------------------------------------------------------------- manual leaderboard entry
    def record_result(self, entry: dict) -> dict:
        """Append one manual public score (after a manual upload) to leaderboard/leaderboard_results.csv. Refused
        when the file is not a READY factory file or its checksum differs. Never feeds back into any selection."""
        from . import reporting as RP

        p = self.ws.path("submissions/submission_manifest.csv")
        M = pd.read_csv(p, dtype={"recipe_id": str, "sha256": str}) if p.exists() and p.stat().st_size else \
            pd.DataFrame()
        if not entry or not entry.get("submission_id"):
            return {"status": NOT_RUN, "messages": ["empty entry: nothing recorded"]}
        m = M[(M.candidate == entry.get("candidate")) & (M.recipe_id.astype(str) == str(entry.get("recipe_id")))] \
            if len(M) else M
        if not len(m):
            return {"status": BLOCKED, "messages": [f"no manifest row for {entry.get('candidate')} / "
                                                    f"{entry.get('recipe_id')}"]}
        r = m.iloc[0].to_dict()
        man = {"status": r.get("status"), "file": r.get("file"), "sha256": r.get("sha256"), "recipe_id":
               r.get("recipe_id"), "candidate": r.get("candidate"), "offline": {"disc_dti": r.get("disc_dti"),
                                                                                "geo_dti": r.get("geo_dti")}}
        e = {"submission_id": entry.get("submission_id"), "submission_file": r.get("file"),
             "public_dti": entry.get("public_score"), "submitted_at": entry.get("submission_date"),
             "notes": entry.get("notes", "")}
        row, why = RP.leaderboard_row(e, man, entry.get("file_sha256") or None)
        if why:
            return {"status": BLOCKED, "messages": why}
        H = self.ws.load_table("leaderboard/leaderboard_results.csv")
        try:
            H = RP.append_history(H, row)
        except ValueError as x:
            return {"status": BLOCKED, "messages": [str(x)]}
        self.ws.save_table(H, "leaderboard/leaderboard_results.csv")
        self.ws.push()
        return {"status": "RECORDED", "messages": [f"{row['submission_id']}: public {row['public_dti']} (offline disc "
                                                   f"{row['validation_disc_dti']}, never used for selection)"]}
