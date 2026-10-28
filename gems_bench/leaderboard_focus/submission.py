"""Notebook 03: ONE validated, reproducible GeoTIFF of the champion (zone-stitched; no retraining, no new inference).

Gate order (the first failing gate decides; nothing is labelled ready by default):
    INCOMPLETE        artifacts / selection / validation / parity stage not COMPLETE (nothing is written)
    BLOCKED           a check below failed (the file, if written, stays on disk with status BLOCKED, never "ready")
    READY_FOR_UPLOAD  every check passed
Writer and official checks are the existing ones: ``final_submission.export_one`` (recipe -> plan audit ->
``write_file`` (never overwrites another recipe) -> ``validate`` = ``gems3.submission.validate`` + ``audit_export`` +
zone-stitched coverage / parity -> ``readiness``). M1 adds:
    * the written band equals the array validated in notebook 02 (sha256 of the finalised array);
    * the written recipe equals the notebook-01 champion recipe (rep, frac, blend, grid candidate, fusion w, kind);
    * file size <= ``submission.max_size_mb``; band count / dtype / EPSG / resolution from the config;
    * the p6 readiness is READY_FOR_UPLOAD.
File name: ``submissions/<prefix>_<family>_<recipe_id>.tif``; a different recipe never reuses the name.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import BLOCKED, INCOMPLETE, READY

OUT = ("submissions/submission_manifest.json", "submissions/submission_validation.json",
       "submissions/submission_sha256.txt", "submissions/export_parity_report.json", "submissions/README.md")


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
    """Notebook 03 stage ``submissions``."""
    import yaml

    from ..p6 import final_submission as FS
    from .validation import array_sha, recipe_consistency

    cfg, fg = sess.cfg, sess.fg
    try:
        sess.require("artifacts", "selection", "validation", "parity")
    except Exception as e:
        man = {"status": INCOMPLETE, "reasons": [str(e)], "uploaded": False}
        sess.ws.save_json(man, "submissions/submission_manifest.json")
        sess.record("submissions", INCOMPLETE, "", reasons=[str(e)])
        return man
    sel = sess.ws.read_json("selection/champion_selection.json")
    val = sess.ws.read_json("validation/validation_status.json")
    recipe = yaml.safe_load(sess.ws.path("selection/champion_recipe.yaml").read_text(encoding="utf-8"))
    fp = sess.fp("submissions", sess.manifest("parity").get("fingerprint"), recipe, cfg["submission"])

    def run():
        C = sess.ws.load_table("selection/champion_candidates.csv")
        crow = C[C.candidate_id == sel["champion"]].iloc[0].to_dict()
        fam = crow.get("family") or "w2"
        name = submission_name(cfg["submission"]["prefix"], fam, recipe["recipe_id"])
        row = {"name": crow["candidate_id"], "label": crow.get("label"), "deploy": crow["deploy"],
               "deploy_cell": json.loads(crow["deploy_cell"]), "disc_dti": crow.get("disc_dti"),
               "disc_worst": crow.get("disc_worst"), "disc_worst_zone": crow.get("disc_worst_zone"),
               "geo_dti": crow.get("geo_dti"), "objective": sel["objective"], "space": ""}
        man6 = FS.export_one(fg, row, name, "winner", sel["objective"], {"decision": sel["decision"]})
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
        p6val = fg.read_json("submissions", f"{path.stem}_validation") or {}
        sha = man6["sha256"]
        man = {"status": status, "reasons": failed, "file": path.name,
               "path_local": str(path), "path_drive": f"{sess.cfg['output_dir']}/submissions/{path.name}",
               "sha256": sha, "candidate": crow["candidate_id"], "recipe_id": recipe["recipe_id"],
               "recipe_fp": rec["fingerprint"], "cell": row["deploy_cell"], "reused_existing_file": man6["reused"],
               "offline": {k: row.get(k) for k in ("disc_dti", "disc_worst", "geo_dti", "objective")},
               "p6_readiness": man6["readiness"], "p6_validation_status": man6["validation_status"],
               "coverage": man6["coverage"], "uploaded": False, "upload": "manual only (submissions/README.md)"}
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
