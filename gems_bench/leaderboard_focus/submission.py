"""Stage 5: ONE validated, reproducible GeoTIFF of the champion (zone-stitched; no retraining, no new inference).

Gate order (the first failing gate decides; nothing is labelled ready by default):
    INCOMPLETE        an upstream stage (artifacts / selection / validation / parity / promotion) has not run or is
                      not usable (nothing is written)
    BLOCKED           the promotion stage is BLOCKED (READY_FOR_EXPORT not reached: nothing is written), or a check
                      below failed (the file, if written, stays on disk with status BLOCKED, never "ready")
    NOT_REQUIRED      ``outputs.save_geotiff`` is false (nothing is written)
    READY_FOR_UPLOAD  every check passed
Writer and official checks are the existing ones: ``final_submission.export_one`` (recipe -> plan audit ->
``write_file`` (never overwrites another recipe) -> ``validate`` = ``gems3.submission.validate`` + ``audit_export`` +
zone-stitched coverage / parity -> ``readiness``). M1 adds:
    * the written band equals the array validated in notebook 02 (sha256 of the finalised array);
    * the written recipe equals the notebook-01 champion recipe (rep, frac, blend, grid candidate, fusion w, kind);
    * file size <= ``submission.max_size_mb``; band count / dtype / EPSG / resolution from the config;
    * the p6 readiness is READY_FOR_UPLOAD.
File name: ``submissions/<prefix>_<family>_<recipe_id>.tif``; a different recipe never reuses the name.

Stage ``exploratory`` (M1.3, opt-in, default OFF): the same writer and the same checks for a candidate that is
technically valid but fails ONLY the internal geographic gate (``gates.decisions``: exploratory_eligible). It needs
``exploratory.enabled`` + a written ``exploratory.reason``, writes ``<exploratory.prefix>_<family>_<recipe_id>.tif``
with its own manifest / sha256 / README, and its best status is EXPLORATORY_READY, never READY_FOR_UPLOAD: the
champion submission stays BLOCKED and the candidate is never called the validated champion. Nothing is uploaded.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import BLOCKED, EXPLORATORY_READY, INCOMPLETE, NOT_REQUIRED, PASSING, READY

OUT = ("submissions/submission_manifest.json", "submissions/submission_validation.json",
       "submissions/submission_sha256.txt", "submissions/export_parity_report.json", "submissions/README.md")
EXPLORATORY_OUT = ("submissions/exploratory_manifest.json", "submissions/exploratory_validation.json",
                   "submissions/exploratory_sha256.txt", "submissions/EXPLORATORY_README.md")
TECH_STAGES = ("artifacts", "selection", "validation", "parity")


def submission_name(prefix: str, family: str, recipe_id: str) -> str:
    return f"{prefix}_{family}_{recipe_id}"


def gate(checks: list[dict], p6_readiness: str | None) -> tuple[str, list[str]]:
    """READY only when every check passed and the p6 readiness is READY_FOR_UPLOAD (pure)."""
    failed = [f"{c['check']}: {c.get('detail', '')}" for c in checks if not c["ok"]]
    if p6_readiness != READY:
        failed.append(f"p6 readiness {p6_readiness}")
    return (READY if not failed else BLOCKED), failed


def raster_facts(path: Path) -> dict:
    import rasterio

    with rasterio.open(path) as s:
        a = s.read(1)
        return {"count": s.count, "dtype": s.dtypes[0], "epsg": s.crs.to_epsg() if s.crs else None,
                "res": [abs(s.res[0]), abs(s.res[1])], "shape": [s.height, s.width], "nodata": s.nodata,
                "transform": list(s.transform)[:6], "array": a}


def m1_checks(facts: dict, size_mb: float, cfg: dict, validated_sha: str, written_sha: str, recipe_diff: list,
              rec_fp_side: str | None, rec_fp: str) -> list[dict]:
    e = cfg["submission"]["expected"]
    v = facts["array"][np.isfinite(facts["array"])]
    return [
        {"check": "single band", "ok": facts["count"] == int(e["count"]), "detail": str(facts["count"])},
        {"check": "dtype float32", "ok": facts["dtype"] == e["dtype"], "detail": facts["dtype"]},
        {"check": "EPSG", "ok": facts["epsg"] == int(e["epsg"]), "detail": str(facts["epsg"])},
        {"check": "resolution", "ok": all(abs(r - float(e["res_m"])) < 1e-6 for r in facts["res"]),
         "detail": str(facts["res"])},
        {"check": "values in [0, 1]", "ok": bool(v.size) and float(v.min()) >= 0 and float(v.max()) <= 1,
         "detail": f"{v.min() if v.size else None}..{v.max() if v.size else None}"},
        {"check": "non-empty prediction", "ok": bool((v > 0).any()), "detail": f"{int((v > 0).sum())} positive px"},
        {"check": f"file size <= {cfg['submission']['max_size_mb']} MB", "ok": size_mb <= float(
            cfg["submission"]["max_size_mb"]), "detail": f"{size_mb:.2f} MB"},
        {"check": "written band == array validated in notebook 02", "ok": written_sha == validated_sha,
         "detail": f"{written_sha[:12]} vs {validated_sha[:12]}"},
        {"check": "recipe == notebook-01 champion recipe", "ok": not recipe_diff, "detail": "; ".join(recipe_diff)},
        {"check": "sidecar recipe fingerprint == export recipe", "ok": rec_fp_side == rec_fp,
         "detail": f"{rec_fp_side} vs {rec_fp}"},
    ]


def precheck(statuses: dict, save_geotiff: bool) -> tuple[str | None, list[str]]:
    """(status, reasons) when the export must not run, (None, []) when it may (pure). ``statuses``: stage -> status of
    artifacts, selection, validation, parity, promotion."""
    if not save_geotiff:
        return NOT_REQUIRED, ["outputs.save_geotiff is false: no GeoTIFF requested"]
    prom = statuses.get("promotion")
    if prom == BLOCKED:
        return BLOCKED, ["promotion BLOCKED (READY_FOR_EXPORT not reached): see selection/promotion_gates.csv"]
    bad = {s: v for s, v in statuses.items() if v not in PASSING}
    if bad:
        return INCOMPLETE, [f"upstream stages not usable: {bad}"]
    return None, []


def exploratory_precheck(ex: dict, promotion: dict, statuses: dict, save_geotiff: bool,
                         promotion_fresh: bool) -> tuple[str | None, list[str]]:
    """(status, reasons) when the exploratory export must not run, (None, []) when it may (pure).
    ``promotion``: selection/promotion_gates.json; ``statuses``: stage -> status of TECH_STAGES."""
    if not ex.get("enabled"):
        return NOT_REQUIRED, ["exploratory.enabled is false (default): no exploratory file"]
    if not str(ex.get("reason") or "").strip():
        return BLOCKED, ["exploratory.reason is empty: a written reason is mandatory"]
    if not save_geotiff:
        return NOT_REQUIRED, ["outputs.save_geotiff is false: no GeoTIFF requested"]
    bad = {s: statuses.get(s) for s in TECH_STAGES if statuses.get(s) not in PASSING}
    if bad:
        return INCOMPLETE, [f"technical stages not usable: {bad} (an exploratory file needs the same technical "
                            "validation as the champion)"]
    if not promotion or not promotion_fresh:
        return INCOMPLETE, ["promotion gates absent or stale for the current selection / validation / parity: run the "
                            "promotion stage first"]
    if promotion.get("status") in PASSING:
        return NOT_REQUIRED, ["the champion passes promotion: export it with the submissions stage (champion file), "
                              "not the exploratory path"]
    if not promotion.get("exploratory_eligible"):
        gates = {g["gate"]: g["status"] for g in promotion.get("gates") or []}
        return BLOCKED, [f"exploratory export needs technical validity PASS and GEOGRAPHIC_GATE_PASS as the only failing "
                         f"gate: technical {promotion.get('technical_validity')}, internal "
                         f"{promotion.get('internal_promotion')}, gates {gates}"]
    return None, []


def exploratory_readme(man: dict) -> str:
    ok = man["status"] == EXPLORATORY_READY
    return "\n".join([
        "# M1 EXPLORATORY baseline (NOT the validated champion)", "",
        f"Status: **{man['status']}**" + ("" if ok else f" - reasons: {man.get('reasons')}"), "",
        f"* File: `{man.get('file')}` (sha256 `{man.get('sha256')}`)",
        f"* Candidate: `{man.get('candidate')}` | recipe `{man.get('recipe_id')}` | cell {man.get('cell')}",
        f"* Internal promotion: **{man.get('internal_promotion')}** (geographic DTI {man.get('geo_dti')} < floor "
        f"{man.get('geo_floor')}); technical validity: {man.get('technical_validity')}",
        f"* Reason given for this export: {man.get('reason')}",
        f"* Offline nested estimates (NOT leaderboard scores): {man.get('offline')}", "",
        "This file passed the same official-format, eligible-mask, parity and sha256 checks as a champion file. It "
        "exists to obtain a real leaderboard baseline; its public score must not be used to tune or select models.", "",
        "## Manual upload (your decision; only when the status is EXPLORATORY_READY)", "",
        "1. Download the file from `MyDrive/GEMS_2/outputs/leaderboard_focus/m1_champion_recovery/submissions/`.",
        "2. `sha256sum <file>` must equal `exploratory_sha256.txt`.",
        "3. Upload by hand on DrivenData (competition 306); record it in stage 6 with `notes: exploratory`.", ""])


def readme(man: dict) -> str:
    ready = man["status"] == READY
    return "\n".join([
        "# M1 submission", "",
        f"Status: **{man['status']}**" + ("" if ready else f" - reasons: {man.get('reasons')}"), "",
        f"* File: `{man.get('file')}` (sha256 `{man.get('sha256')}`)",
        f"* Candidate: `{man.get('candidate')}` | recipe `{man.get('recipe_id')}` | cell {man.get('cell')}",
        f"* Offline nested estimates (NOT leaderboard scores): {man.get('offline')}", "",
        "## Manual upload (only when the status is READY_FOR_UPLOAD)", "",
        "1. Download the file from Google Drive "
        "(`MyDrive/GEMS_2/outputs/leaderboard_focus/m1_champion_recovery/submissions/`).",
        "2. Check locally: `sha256sum <file>` must equal `submission_sha256.txt`.",
        "3. DrivenData -> competition 306 -> Submissions -> upload the .tif (3 scored submissions per rolling 7 days).",
        "4. Record the public score in notebook 04 (`LEADERBOARD_ENTRY`). Never tune on the public score.", ""])


def export(sess) -> dict:
    """Stage ``submissions``."""
    import yaml

    from .session import StageBlocked

    cfg, fg = sess.cfg, sess.fg
    ups = ("artifacts", "selection", "validation", "parity", "promotion")
    st0, why0 = precheck({s: sess.status(s) for s in ups}, bool((cfg.get("outputs") or {}).get("save_geotiff", True)))
    if st0 is None:
        try:
            sess.require(*ups)                           # also refuses results of different stage code
        except StageBlocked as e:
            st0, why0 = INCOMPLETE, [str(e)]
    if st0 is not None:
        man = {"status": st0, "reasons": why0, "uploaded": False, "file": None,
               "note": "no GeoTIFF written by this call; an earlier file (if any) keeps its own manifest"}
        sess.ws.save_json(man, "submissions/submission_manifest.json")
        sess.record("submissions", st0, "", reasons=why0)
        return man
    sel = sess.ws.read_json("selection/champion_selection.json")
    val = sess.ws.read_json("validation/validation_status.json")
    recipe = yaml.safe_load(sess.ws.path("selection/champion_recipe.yaml").read_text(encoding="utf-8"))
    fp = sess.fp("submissions", sess.manifest("parity").get("fingerprint"), recipe, cfg["submission"],
                 sess.manifest("promotion").get("fingerprint"))

    def run():
        w = _write_checked(sess, sel, val, recipe, cfg["submission"]["prefix"], "winner")
        status, failed, path, row, crow = w["status"], w["failed"], w["path"], w["row"], w["crow"]
        man6, rec, sha, checks, p6val = w["man6"], w["rec"], w["sha"], w["checks"], w["p6val"]
        written_sha, diff = w["written_sha"], w["diff"]
        man = {"status": status, "reasons": failed, "file": path.name,
               "path_local": str(path), "path_drive": f"{sess.cfg['output_dir']}/submissions/{path.name}",
               "sha256": sha, "candidate": crow["candidate_id"], "recipe_id": recipe["recipe_id"],
               "recipe_fp": rec["fingerprint"], "cell": row["deploy_cell"], "reused_existing_file": man6["reused"],
               "offline": {k: row.get(k) for k in ("disc_dti", "disc_worst", "geo_dti", "objective")},
               "p6_readiness": man6["readiness"], "p6_validation_status": man6["validation_status"],
               "coverage": man6["coverage"], "uploaded": False, "upload": "manual only (submissions/README.md)",
               "promotion": {k: (sess.ws.read_json("selection/promotion_gates.json") or {}).get(k)
                             for k in ("status", "warnings", "geographic_gate", "override_note")}}
        sess.ws.save_json(man, "submissions/submission_manifest.json")
        sess.ws.save_json({"m1_checks": checks, "p6_validation": p6val, "status": status},
                          "submissions/submission_validation.json")
        sess.ws.save_text(f"{sha}  {path.name}\n", "submissions/submission_sha256.txt")
        sess.ws.save_json({"validated_array_sha256": val["validated_array_sha256"], "written_array_sha256": written_sha,
                           "equal": written_sha == val["validated_array_sha256"], "recipe_diff": diff,
                           "zone_parity": (fg.load_table(f"{path.stem}_diagnostics", "submissions")
                                           .query("kind == 'parity'")[["unit", "check", "status"]]
                                           .to_dict("records"))},
                          "submissions/export_parity_report.json")
        sess.ws.save_json(man, f"submissions/{path.stem}_m1_manifest.json")      # versioned copy, never replaced
        sess.ws.save_text(readme(man), "submissions/README.md")
        return status if status != READY else "COMPLETE", failed, {"submission_status": status, "file": path.name,
                                                                   "sha256": sha}

    out = sess.run_stage("submissions", fp, OUT, run, inputs={"parity_fp": sess.manifest("parity").get("fingerprint"),
                                                              "recipe_id": recipe.get("recipe_id")})
    return sess.ws.read_json("submissions/submission_manifest.json") or out


def _write_checked(sess, sel: dict, val: dict, recipe: dict, prefix: str, role: str) -> dict:
    """Write the selected recipe with the P6 writer under ``<prefix>_<family>_<recipe_id>`` and run every M1 check
    (shared by the champion and the exploratory export; the gate verdict is the caller's)."""
    from ..p6 import final_submission as FS
    from .session import StageInvalid
    from .validation import array_sha, recipe_consistency

    cfg, fg = sess.cfg, sess.fg
    C = sess.ws.require_table("selection/champion_candidates.csv", ("candidate_id", "deploy", "deploy_cell"))
    if not (C.candidate_id == sel.get("champion")).any() or recipe.get("recipe_id") != sel.get("recipe_id"):
        raise StageInvalid(f"recipe ids disagree: champion {sel.get('champion')} / recipe yaml "
                           f"{recipe.get('recipe_id')} vs champion_selection.json {sel.get('recipe_id')}: re-run "
                           "selection")
    crow = C[C.candidate_id == sel["champion"]].iloc[0].to_dict()
    fam = crow.get("family") or "w2"
    name = submission_name(prefix, fam, recipe["recipe_id"])
    row = {"name": crow["candidate_id"], "label": crow.get("label"), "deploy": crow["deploy"],
           "deploy_cell": json.loads(crow["deploy_cell"]), "disc_dti": crow.get("disc_dti"),
           "disc_worst": crow.get("disc_worst"), "disc_worst_zone": crow.get("disc_worst_zone"),
           "geo_dti": crow.get("geo_dti"), "objective": sel["objective"], "space": ""}
    man6 = FS.export_one(fg, row, name, role, sel["objective"], {"decision": sel["decision"]})
    path = fg.root / man6["path"]
    facts = raster_facts(path)
    written_sha = array_sha(facts["array"])
    rec = man6["recipe"]
    side = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    built_like = {"cell": {"candidate": rec["policy"]["grid_candidate"], "rep": rec["policy"]["rep"],
                           "frac": rec["policy"]["frac"], "blend": rec["policy"]["grid_blend"]},
                  "w_all": rec["preprocessing"]["fusion_w"], "kind": rec["kind"].replace("zone_stitched_", ""),
                  "stack_dir": rec["model"]["maps"] if rec["model"]["kind"] == "stack" else None}
    diff = recipe_consistency(recipe, built_like)
    checks = m1_checks(facts, path.stat().st_size / 2**20, cfg, val["validated_array_sha256"], written_sha, diff,
                       side.get("recipe_fp"), rec["fingerprint"])
    status, failed = gate(checks, man6["readiness"]["status"])
    return {"status": status, "failed": failed, "path": path, "row": row, "crow": crow, "man6": man6, "rec": rec,
            "sha": man6["sha256"], "checks": checks, "written_sha": written_sha, "diff": diff,
            "p6val": fg.read_json("submissions", f"{path.stem}_validation") or {}}


def export_exploratory(sess) -> dict:
    """Stage ``exploratory`` (opt-in; see the module docstring). The champion submission is never touched."""
    import yaml

    from .session import StageBlocked

    cfg = sess.cfg
    ex = cfg.get("exploratory") or {}
    prom = sess.ws.read_json("selection/promotion_gates.json") or {}
    pman = sess.manifest("promotion")
    fresh = bool(pman) and (pman.get("inputs") or {}) == {s: sess.manifest(s).get("fingerprint")
                                                          for s in ("selection", "validation", "parity")}
    st0, why0 = exploratory_precheck(ex, prom, {s: sess.status(s) for s in TECH_STAGES},
                                     bool((cfg.get("outputs") or {}).get("save_geotiff", True)), fresh)
    if st0 is None:
        try:
            sess.require(*TECH_STAGES)
        except StageBlocked as e:
            st0, why0 = INCOMPLETE, [str(e)]
    if st0 is not None:
        man = {"status": st0, "reasons": why0, "uploaded": False, "file": None, "validated_champion": False,
               "note": "no exploratory GeoTIFF written by this call"}
        if ex.get("enabled"):
            sess.ws.save_json(man, "submissions/exploratory_manifest.json")
        sess.record("exploratory", st0, "", reasons=why0)
        return man
    sel = sess.ws.read_json("selection/champion_selection.json")
    val = sess.ws.read_json("validation/validation_status.json")
    recipe = yaml.safe_load(sess.ws.path("selection/champion_recipe.yaml").read_text(encoding="utf-8"))
    fp = sess.fp("exploratory", sess.manifest("parity").get("fingerprint"), recipe, cfg["submission"], ex,
                 pman.get("fingerprint"))

    def run():
        w = _write_checked(sess, sel, val, recipe, ex["prefix"], "exploratory")
        status = EXPLORATORY_READY if w["status"] == READY else BLOCKED
        path = w["path"]
        gates = {g["gate"]: g for g in prom.get("gates") or []}
        man = {"status": status, "reasons": w["failed"], "file": path.name, "validated_champion": False,
               "role": "exploratory baseline (fails the internal geographic gate)", "reason": ex["reason"],
               "technical_validity": prom.get("technical_validity"), "internal_promotion": prom.get(
                   "internal_promotion"), "geographic_gate": gates.get("GEOGRAPHIC_GATE_PASS"),
               "geo_dti": w["row"].get("geo_dti"), "geo_floor": cfg["selection"]["geo_floor"],
               "path_drive": f"{cfg['output_dir']}/submissions/{path.name}", "sha256": w["sha"],
               "candidate": w["crow"]["candidate_id"], "recipe_id": recipe["recipe_id"], "recipe_fp":
                   w["rec"]["fingerprint"], "cell": w["row"]["deploy_cell"], "reused_existing_file": w["man6"]["reused"],
               "offline": {k: w["row"].get(k) for k in ("disc_dti", "disc_worst", "geo_dti", "objective")},
               "written_equals_validated": w["written_sha"] == val["validated_array_sha256"],
               "p6_validation_status": w["man6"]["validation_status"], "uploaded": False,
               "upload": "manual only, the user's decision (submissions/EXPLORATORY_README.md)"}
        sess.ws.save_json(man, "submissions/exploratory_manifest.json")
        sess.ws.save_json({"m1_checks": w["checks"], "p6_validation": w["p6val"], "status": status},
                          "submissions/exploratory_validation.json")
        sess.ws.save_text(f"{w['sha']}  {path.name}\n", "submissions/exploratory_sha256.txt")
        sess.ws.save_json(man, f"submissions/{path.stem}_m1_manifest.json")      # versioned copy, never replaced
        sess.ws.save_text(exploratory_readme(man), "submissions/EXPLORATORY_README.md")
        return ("COMPLETE" if status == EXPLORATORY_READY else BLOCKED), w["failed"], {
            "submission_status": status, "file": path.name, "sha256": w["sha"], "validated_champion": False}

    out = sess.run_stage("exploratory", fp, EXPLORATORY_OUT, run, inputs={
        "parity_fp": sess.manifest("parity").get("fingerprint"), "recipe_id": recipe.get("recipe_id"),
        "reason": ex.get("reason")})
    return sess.ws.read_json("submissions/exploratory_manifest.json") or out
