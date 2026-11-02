"""Stage K: full-footprint reconstruction.

* ``reconstruction/research_score_<model>.tif`` - ALWAYS a research artifact (GeoTIFF tag ROLE =
  RESEARCH_NOT_A_SUBMISSION): zone-stitched V3 held-out scores (geo zone k <- candidates of zone k scored by the model
  fitted without zone k), ranked to [0, 1] inside each zone, 0 on valid non-candidate pixels, NaN outside the
  template footprint.
* ``submissions/validated_submission_candidate_<model>_<recipe>.tif`` - ONLY when Stage I returned
  PROMOTE_CANDIDATE for that model AND ``RUN['export']`` is on AND every hard check passes on the file read back
  from disk. Construction = ``geomoe.export`` (zone partition, official eligible mask, deployable fixed-policy cell,
  P6 writer / validator, factory read-back checks). Status READY_FOR_MANUAL_UPLOAD at best: nothing is uploaded.
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

READY, NONE = "READY_FOR_MANUAL_UPLOAD", "NO_VALIDATED_CANDIDATE"


def zone_rank(score: np.ndarray, dom: np.ndarray) -> np.ndarray:
    """Average rank in (0, 1] of the finite scores inside ``dom``; 0 on the rest of ``dom``; NaN outside."""
    out = np.where(dom, 0.0, np.nan).astype(np.float32)
    ok = dom & np.isfinite(score)
    if ok.any():
        out[ok] = (pd.Series(np.asarray(score)[ok]).rank(method="average").to_numpy() / ok.sum()).astype(np.float32)
    return out


def write_research_tif(path, arr: np.ndarray, template_profile: dict, tags: dict) -> None:
    import rasterio

    prof = {k: v for k, v in template_profile.items() if k not in ("blockxsize", "blockysize", "tiled")}
    prof.update(driver="GTiff", dtype="float32", count=1, nodata=np.nan, compress="deflate")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(np.float32), 1)
        dst.update_tags(**{k: str(v) for k, v in tags.items()})


def smap_of(R, su, S: pd.DataFrame, w2_pixel, partial: bool) -> np.ndarray:
    from ..geomoe import evaluate as EV
    from .selection import partial_map

    lab = R.lab(su)
    s = S[S.unit == su.key].dropna(subset=["score"])
    sm = EV.score_map(lab, s.label.to_numpy(), s.score.to_numpy(), s.n_px.to_numpy())
    return partial_map(sm, w2_pixel) if partial else sm


def research_score(R, model: str) -> dict:
    from ..p6 import final_submission as FS

    data = R.ctx.data
    S = R.load_parquet(f"rankers/scores_{model}.parquet")
    v3 = [su for su in R.units if su.scenario == "V3"]
    zid, cov = FS.zone_partition(R.ctx.folds.fold_map, data.valid, [su.zone for su in v3])
    full = np.where(np.asarray(data.template_valid, bool) & np.asarray(data.valid, bool), 0.0, np.nan).astype(
        np.float32)
    per = []
    for su in v3:
        u = su.unit
        sm = smap_of(R, su, S, R.w2_smap(su), bool(R.cfg51["selection"]["partial_segments"]))
        m = np.asarray(zid[u.sl] == su.zone) & u.ev
        z = zone_rank(sm, m)
        blk = full[u.sl]
        blk[m] = np.where(np.isfinite(blk[m]), z[m], np.nan)
        full[u.sl] = blk
        per.append({"zone": su.zone, "zone_px": int(m.sum()), "scored_px": int((m & np.isfinite(sm)).sum())})
        u.release()
    rel = f"reconstruction/research_score_{model}.tif"
    write_research_tif(R.ws.path(rel), full, data.template_profile,
                       {"ROLE": "RESEARCH_NOT_A_SUBMISSION", "MODEL": model, "RUN_ID": R.run_id,
                        "CONSTRUCTION": "zone-stitched V3 held-out scores, rank in zone",
                        "CODE_REV": R.code_rev})
    return {"file": rel, "zones": per, "coverage": cov}


def validated_candidate(R, model: str, decision: dict, cell: tuple) -> tuple[str, dict]:
    """Gated validated GeoTIFF (geomoe.export construction). Returns (status, info)."""
    from .selection import PROMOTE

    if decision.get("status") != PROMOTE:
        return "SKIPPED_BY_VALID_DECISION", {"status": NONE, "reason": f"Stage I status {decision.get('status')}"}
    from ..artifacts.atomic import sha256_file
    from ..gems3 import representation as REP
    from ..gems3 import submission as SUB
    from ..geomoe.export import chk
    from ..leaderboard_focus import factory as FA
    from ..leaderboard_focus import submission as SU
    from ..leaderboard_focus import validation as VA
    from ..p6 import export_parity as EP
    from ..p6 import final_submission as FS
    from .selection import apply_rep

    fg, data = R.fg, R.ctx.data
    pc = fg.cfg["submission"]
    rep_name, frac = str(cell[0]), float(cell[1])
    rep = R.cfg51["selection"]["reps"][rep_name]
    sigma = float(R.sh["evaluation"]["ridge_sigma_px"])
    v3 = [su for su in R.units if su.scenario == "V3"]
    zid, cov = FS.zone_partition(R.ctx.folds.fold_map, data.valid, [su.zone for su in v3])
    el_full = SUB.eligible_mask(data, 0, "eligible")
    labels = np.asarray(data.labels, bool)
    S = R.load_parquet(f"rankers/scores_{model}.parquet")
    qs, zrows, par = [], [], []
    for su in v3:
        u = su.unit
        sm = smap_of(R, su, S, R.w2_smap(su), bool(R.cfg51["selection"]["partial_segments"]))
        m = np.asarray(zid[u.sl] == su.zone) & u.ev
        el = el_full[u.sl] & m
        q = REP.apply(rep, sm, m, frac, known=labels[u.sl], sigma_px=sigma, eligible=el)
        qe = apply_rep(R, u, sm, rep, frac)
        ma = EP.map_agreement(np.asarray(qe) > 0, np.asarray(q) > 0)
        par.append({"unit": su.key, "check": "binary map vs evaluation map", **ma,
                    "status": "PASS" if (ma["jaccard"] or 0) >= 0.5 else "WARNING"})
        pos = np.asarray(q) > 0
        zrows.append({"zone": su.zone, "eligible_px": int(el.sum()), "positive_px": int(pos.sum()),
                      "frac_of_eligible": float(pos.sum() / max(int(el.sum()), 1)),
                      "positive_on_known_px": int((pos & labels[u.sl]).sum())})
        qs.append((u.sl, m, q))
        u.release()
    Q = FS.stitch(data.shape, qs)
    Z, P = pd.DataFrame(zrows), pd.DataFrame(par)
    tol = float(pc["budget_tolerance"])
    tech = [chk("budget", f"per-zone area <= frac x (1 + {tol:g}), non-empty",
                bool(((Z.frac_of_eligible <= frac * (1 + tol) + 1e-9) & (Z.positive_px > 0)).all()), Z.to_dict("records")),
            chk("mask", "no positive pixel on a mapped fault", int(Z.positive_on_known_px.sum()) == 0),
            chk("coverage", f"uncovered valid px <= {pc['max_uncovered_frac']:g}",
                cov["uncovered_frac"] <= float(pc["max_uncovered_frac"]), json.dumps(cov, default=str)),
            chk("decision", "Stage I PROMOTE_CANDIDATE (strict evaluation, every gate)", True, decision.get("gates"))]
    arr = SUB.finalise(Q, data, pc["outside_valid"])
    vsha = VA.array_sha(arr)
    rec = EP.build_recipe(
        candidate=f"g051_{model}", kind="zone_stitched_g051",
        policy={"selection_id": f"g051|{model}|fixed_policy", "space": "rep+frac", "objective": "fixed policy",
                "rep": rep_name, "frac": frac, "blend": 1.0, "grid_candidate": model, "grid_blend": 1.0},
        model={"kind": f"g051_{model}", "features": "rankers/model_registry.json", "refit": False,
               "maps": "V3 cross-fitted segment scores (rankers/scores_<model>.parquet)"},
        sources={"scores": sha256_file(R.ws.path(f"rankers/scores_{model}.parquet")),
                 "registry": sha256_file(R.ws.path("registry/fault_hypothesis_registry.parquet"))},
        feature_schema=[model], preprocessing={"construction": "zone_stitched", "representation": rep,
                                                "sigma_px": sigma, "partial_segments":
                                                    bool(R.cfg51["selection"]["partial_segments"])})
    pa = EP.plan_audit(rec)
    tech.append(chk("export", "recipe plan audit", pa["status"] == "PASS", json.dumps(pa["checks"])))
    rid = rec["fingerprint"][:10]
    name = f"validated_submission_candidate_{model}_{rid}"
    row = {"model": model, "recipe_id": rid, "rep": rep_name, "frac": frac, "validated_array_sha256": vsha}
    if FA.verdict(tech) == "FAIL":
        R.ws.save_json({"checks": tech, "recipe": rec}, f"submissions/{name}_validation.json")
        return "BLOCKED", {**row, "status": "INVALID", "reasons": [c["check"] for c in tech if c["hard"] and not c["ok"]]}
    ns = SimpleNamespace(p6=fg.p6, cfg=fg.cfg, root=R.ws.root)
    w = FS.write_file(ns, name, Q, rec, [SUB.area_ledger(Q, data, 0, f"g051 {model} {rep_name} @ {frac:g}")])
    path = w["path"]
    v = FS.validate(ns, path, rec, {"coverage": cov, "parity": P})
    checks = tech + [chk(f"official:{g}", c["check"], c["ok"], c.get("detail", ""), c.get("hard", True))
                     for g, cs in v["groups"].items() for c in cs]
    facts = SU.raster_facts(path)
    wsha = VA.array_sha(facts["array"])
    side = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    fsha = sha256_file(path)
    size_mb = path.stat().st_size / 2 ** 20
    mc = R.F.cfg["submission"]
    checks += FA.readback_checks(facts, data.template_profile, mc["expected"], size_mb, float(mc["max_size_mb"]), vsha,
                                 wsha, side.get("recipe_fp"), rec["fingerprint"], fsha)
    status = READY if FA.verdict(checks) in ("PASS", "WARNING") else "INVALID"
    R.ws.save_json({"checks": checks, "recipe": rec, "zones": zrows, "parity": P.to_dict("records")},
                   f"submissions/{path.stem}_validation.json")
    row.update(file=path.name, status=status, sha256=fsha, size_mb=round(size_mb, 4), path_local=str(path),
               written_array_sha256=wsha, export_parity_verified=bool(status == READY and vsha == wsha),
               uploaded=False, written=time.strftime("%Y-%m-%dT%H:%M:%S"))
    return ("COMPLETE" if status == READY else "FAILED"), row
