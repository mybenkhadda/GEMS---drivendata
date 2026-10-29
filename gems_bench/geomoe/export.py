"""Stage 12: a validated GeoTIFF of the selected model, only when the predeclared rule selected one.

Reused, not re-implemented: ``p6.final_submission.zone_partition / stitch / write_file / validate`` (template
finalise, official format checks, ``export_parity.audit_export``), ``export_parity.build_recipe / plan_audit /
agreement / map_agreement``, ``leaderboard_focus.factory.readback_checks / verdict``, ``leaderboard_focus.submission
.raster_facts``, ``leaderboard_focus.validation.array_sha``. The writer root is redirected to
``outputs/discovery_geomoe`` so no factory or P6 submission is touched.

Construction (``zone_stitched``, the evaluated one): geo zone k <- the V3 candidates of zone k scored by the model
fitted without zone k (stage 08-10 scores, unchanged), rasterised with ONE all-zone (rep, frac) cell = the model's
in-sample choice on the primary scenario (V1) over the eligible full-region domain (valid & template & not a mapped
fault: the official mask). No held-out label enters any score; mapped faults are used only as the official mask.

Status: READY_FOR_UPLOAD only when every hard check passed on the file read back from disk. A model that the rule did
not select is exported only with ``export.exploratory = True`` and is labelled EXPLORATORY (never the reference).
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import evaluate as EV

READY, NO_NEW = "READY_FOR_UPLOAD", "NO_NEW_SUBMISSION"


def chk(group, check, ok, detail="", hard=True) -> dict:
    return {"group": group, "check": check, "ok": bool(ok), "hard": hard, "detail": str(detail)[:800]}


def write_index(G, rows: list[dict], note: str) -> None:
    M = pd.DataFrame(rows)
    G.ws.save_table(M, "submissions/submission_manifest.csv")
    G.ws.save_json({"files": rows, "note": note, "uploaded": False, "upload": "manual only",
                    "written": time.strftime("%Y-%m-%dT%H:%M:%S")}, "submissions/submission_manifest.json")
    G.ws.save_text("".join(f"{r['sha256']}  {r['file']}\n" for r in rows if r.get("sha256")),
                   "submissions/checksums.txt")
    lines = ["# Discovery GeoMoE: upload instructions", "", note, ""]
    for r in rows:
        lines += [f"## {r.get('file') or '(no file)'}", "", f"* status: **{r.get('status')}** ({r.get('role')})",
                  f"* model: `{r.get('model')}`, recipe `{r.get('recipe_id')}`, cell {r.get('rep')} @ {r.get('frac')}",
                  f"* sha256: `{r.get('sha256')}`", f"* Drive: `{r.get('path_drive')}`", ""]
    lines += ["Check `sha256sum <file>` against `checksums.txt`, upload by hand on DrivenData, then record the public "
              "score with the submission factory's `record_result` (offline numbers are not leaderboard scores; a "
              "public score never changes a selection here).", ""]
    G.ws.save_text("\n".join(lines), "submissions/README_UPLOAD.md")


def cell_of(G, model: str) -> tuple[str, float]:
    """The model's deployable cell: the fixed-policy rule applied to ALL development units (the same rule whose
    held-out application produced the evaluated numbers; gating/deployable_policies.json)."""
    d = (G.ws.read_json("gating/deployable_policies.json") or {}).get(model)
    if not d:
        raise RuntimeError(f"{model}: no deployable policy in gating/deployable_policies.json")
    return str(d["rep"]), float(d["frac"])


def source_completeness(G, model: str) -> tuple[bool, str]:
    """Every V3 zone has candidates scored by the model and the same expert availability (no zone silently lost
    a family the model was evaluated with)."""
    A = G.ws.load_table("experts/source_availability.csv")
    S = pd.read_parquet(G.ws.path(f"gating/scores_{model}.parquet"))
    v3 = [su.key for su in G.units if su.scenario == "V3"]
    scored = {k: int(S[(S.unit == k)].score.notna().sum()) for k in v3}
    if not len(A):
        return False, "experts/source_availability.csv absent"
    a = A[(A.scenario == "V3") & (A.member == "*")].pivot_table(index="zone", columns="expert",
                                                               values="available_share")
    uneven = [c for c in a.columns if (a[c] >= 0.5).nunique() > 1]
    ok = all(v > 0 for v in scored.values())
    return ok, f"scored candidates per V3 zone {scored}; families with zone-dependent availability {uneven} " \
               "(handled by the consistent-availability rule)"


def run(G, sel: dict) -> tuple[str, dict]:
    from ..artifacts.atomic import sha256_file
    from ..gems3 import representation as R
    from ..gems3 import submission as SUB
    from ..leaderboard_focus import factory as FA
    from ..leaderboard_focus import submission as SU
    from ..leaderboard_focus import validation as VA
    from ..p6 import export_parity as EP
    from ..p6 import final_submission as FS
    from . import selftest as STT
    from .deployment import deploy_model

    ex = G.nb["export"]
    picked, role = sel.get("selected"), "SELECTED"
    if not sel.get("measured"):
        note = (f"No new submission: selection status {sel.get('status')} - no measured comparison exists "
                f"({'; '.join(sel.get('reasons') or [])}). This is NOT evidence for W2.")
        write_index(G, [{"file": None, "status": "NOT_SELECTED", "role": "none", "model": None}], note)
        return "SKIPPED_BY_VALID_DECISION", {"reason": note[:300]}
    if picked in (None, "RETAIN_W2"):
        if not ex.get("exploratory"):
            note = ("No new submission: the predeclared selection rule retained W2 on measured fixed-policy numbers "
                    f"({'; '.join(sel.get('reasons') or [])}). The reference file stays the submission factory's "
                    "w2_champion GeoTIFF.")
            write_index(G, [{"file": None, "status": "NOT_SELECTED", "role": "none", "model": "RETAIN_W2"}], note)
            return "SKIPPED_BY_VALID_DECISION", {"reason": note[:300]}
        picked, role = deploy_model(G), "EXPLORATORY"
        if picked is None:
            write_index(G, [{"file": None, "status": NO_NEW, "role": "none", "model": None}], "no fused model")
            return "SKIPPED_BY_VALID_DECISION", {"reason": "exploratory export asked but no fused model exists"}
    gates = []
    T = STT.run_selftests()
    G.ws.save_table(T, "audit/selftests.csv")
    bad = T[(T.status != "PASS") & T.critical.astype(bool)]
    gates.append(chk("gate", "GeoMoE synthetic self-tests (critical)", not len(bad), list(bad.test)))
    gates.append(chk("gate", "factory runtime self-tests", G.F.selftest_ok()))
    for k, v in (sel.get("hard_checks") or {}).items():
        gates.append(chk("gate", f"hard requirement {k}", v))
    dd = G.ws.read_json("gating/deployment_decision.json")
    if dd is not None:
        gates.append(chk("gate", "zone-stitched construction self-consistent (stage 11)",
                         dd.get("zone_stitched_self_consistent")))
    if role == "SELECTED":
        gates.append(chk("gate", "model selected by the predeclared rule", sel.get("status") == "SELECTED"))
    sc_ok, sc_detail = source_completeness(G, picked)
    gates.append(chk("gate", "candidate source completeness on every V3 zone", sc_ok, sc_detail))
    blocked = [g for g in gates if g["hard"] and not g["ok"]]
    G.ws.save_table(pd.DataFrame(gates), "submissions/export_gates.csv")
    if blocked:
        write_index(G, [{"file": None, "status": "BLOCKED", "role": role, "model": picked,
                         "reasons": "; ".join(g["check"] for g in blocked)}], "export blocked by a hard gate")
        return "BLOCKED", {"note": f"export blocked: {[g['check'] for g in blocked]}"}

    fg, data = G.fg, G.ctx.data
    pc = fg.cfg["submission"]
    rep, frac = cell_of(G, picked)
    sh = G.sh
    rp = sh["baseline"]["representations"][rep]
    sigma = float(sh["evaluation"]["ridge_sigma_px"])
    v3 = [su for su in G.units if su.scenario == "V3"]
    zid, cov = FS.zone_partition(G.ctx.folds.fold_map, data.valid, [su.zone for su in v3])
    el_full = SUB.eligible_mask(data, 0, "eligible")
    labels = np.asarray(data.labels, bool)
    S = pd.read_parquet(G.ws.path(f"gating/scores_{picked}.parquet"))
    qs, ss, par, zrows, seg_sha = [], [], [], [], {}
    for su in v3:
        u = su.unit
        lab = G.lab(su)
        s = S[S.unit == su.key].dropna(subset=["score"])
        sm = EV.score_map(lab, s.label.to_numpy(), s.score.to_numpy(), s.n_px.to_numpy())
        m = np.asarray(zid[u.sl] == su.zone) & u.ev
        el = el_full[u.sl] & m
        q = R.apply(rp, sm, m, frac, known=labels[u.sl], sigma_px=sigma, eligible=el)
        qe = EV.policy_q(G, u, sm, rep, frac)
        ag = EP.agreement(sm, sm, m, (0.01, 0.05), 200_000, int(G.cfg["v2"]["seed"]))
        ma = EP.map_agreement(qe, np.asarray(q) > 0)
        par.append({"unit": su.key, "check": "zone score vs evaluation construction", **ag,
                    "status": "PASS" if (ag.get("spearman") or 0) >= float(pc["parity_min_spearman"]) else "BLOCKED",
                    "note": "the exported zone score IS the evaluated V3 score (same candidates, same fit)"})
        par.append({"unit": su.key, "check": "binary map vs evaluation map", **ma,
                    "status": "PASS" if (ma["jaccard"] or 0) >= 0.5 else "WARNING",
                    "note": "eligibility differs by design: the file masks every mapped fault (official mask); the "
                            "V3 evaluation counts fold-k mapped faults as truth"})
        pos = np.asarray(q) > 0
        zrows.append({"zone": su.zone, "eligible_px": int(el.sum()), "positive_px": int(pos.sum()),
                      "frac_of_eligible": float(pos.sum() / max(int(el.sum()), 1)),
                      "positive_on_known_px": int((pos & labels[u.sl]).sum())})
        qs.append((u.sl, m, q))
        ss.append((u.sl, m, np.nan_to_num(sm, nan=0.0)))
        seg_sha[su.key] = sha256_file(G.ws.path(f"{G.cand_dir()}/{su.key.replace(':', '_')}.npz"))
        u.release()
    Q, SC = FS.stitch(data.shape, qs), FS.stitch(data.shape, ss)
    Z = pd.DataFrame(zrows)
    P = pd.DataFrame(par)
    tol = float(pc["budget_tolerance"])
    tech = [chk("budget", f"per-zone area <= frac x (1 + {tol:g}), non-empty", bool(((Z.frac_of_eligible <= frac * (
        1 + tol) + 1e-9) & (Z.positive_px > 0)).all()), Z.to_dict("records")),
            chk("mask", "no positive pixel on a mapped fault (official mask)", int(Z.positive_on_known_px.sum()) == 0),
            chk("coverage", f"uncovered valid px <= {pc['max_uncovered_frac']:g}", cov["uncovered_frac"] <=
                float(pc["max_uncovered_frac"]), json.dumps(cov, default=str)),
            chk("parity", "zone score == evaluated score (every zone)", not (P.status == "BLOCKED").any())]
    arr = SUB.finalise(Q, data, pc["outside_valid"])
    tv = np.asarray(data.template_valid, bool)
    fin = np.isfinite(arr)
    tech.append(chk("array", "finite exactly where the template is finite, values in [0, 1]",
                    bool(fin[tv].all()) and not bool(fin[~tv].any()) and float(np.nanmin(arr)) >= 0.0 and
                    float(np.nanmax(arr)) <= 1.0))
    vsha = VA.array_sha(arr)
    feats = sorted({m for ms in G.cfg["experts"].values() for m in ms})
    w2 = G.bst.w2_inputs()
    rec = EP.build_recipe(
        candidate=f"geomoe_{picked}", kind="zone_stitched_geomoe",
        policy={"selection_id": f"geomoe|{picked}|fixed_policy|all_units", "space": "rep+frac",
                "objective": "mean over scenarios of pooled DTI (fixed policy)", "rep": rep, "frac": frac,
                "blend": 1.0, "grid_candidate": picked,
                "grid_blend": 1.0},
        model={"kind": f"geomoe_{picked}", "features": feats, "refit": False, "maps": "V3 cross-fitted candidate "
               "scores (gating/scores_<model>.parquet)", "note": "zone k scored by the model fitted without zone k"},
        sources={"segments": seg_sha, "scores": sha256_file(G.ws.path(f"gating/scores_{picked}.parquet")),
                 "registry": sha256_file(G.ws.path("gating/model_registry.json")),
                 "normalization": sha256_file(G.ws.path("experts/source_normalization.json")),
                 "w2_maps": {k: v for k, v in (w2.get("maps_sha256") or {}).items() if k.startswith("geo_")} or
                 {"w2": "absent"}},
        feature_schema=feats,
        preprocessing={"construction": "zone_stitched", "ranking": "stack.rank_in inside each geo fold zone (W2 "
                       "convention); expert scores ECDF-normalised on the training zones", "sigma_px": sigma,
                       "representation": rp, "eligibility": "gems3.submission.eligible_mask(data, 0, 'eligible') "
                       "per zone", "known": "data.labels (official pixel-exact mask) passed to R.apply",
                       "ring_px": 0, "uncovered": "valid pixels outside every geo zone are written 0"})
    pa = EP.plan_audit(rec)
    tech.append(chk("export", "recipe plan audit", pa["status"] == "PASS", json.dumps(pa["checks"])))
    rid = rec["fingerprint"][:10]
    name = f"{'exploratory_' if role == 'EXPLORATORY' else ''}submission_geomoe_{picked}_{rid}"
    row = {"model": picked, "role": role, "recipe_id": rid, "rep": rep, "frac": frac, "validated_array_sha256": vsha}
    if FA.verdict(tech) == "FAIL":
        row.update(status="INVALID", reasons="; ".join(c["check"] for c in tech if c["hard"] and not c["ok"]))
        G.ws.save_json({"checks": tech, "recipe": rec}, f"submissions/{name}_validation.json")
        write_index(G, [row], "technical validation failed before writing: no file")
        return "BLOCKED", {"note": row["reasons"][:200]}
    ns = SimpleNamespace(p6=fg.p6, cfg=fg.cfg, root=G.ws.root)
    ledger = [SUB.area_ledger(Q, data, 0, f"geomoe {picked} zone-stitched {rep} @ {frac:g}")]
    w = FS.write_file(ns, name, Q, rec, ledger)
    path = w["path"]
    built = {"coverage": cov, "parity": P}
    v = FS.validate(ns, path, rec, built)
    checks = tech + [chk(f"official:{g}", c["check"], c["ok"], c.get("detail", ""), c.get("hard", True))
                     for g, cs in v["groups"].items() for c in cs]
    facts = SU.raster_facts(path)
    wsha = VA.array_sha(facts["array"])
    side = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    fsha = sha256_file(path)
    size_mb = path.stat().st_size / 2 ** 20
    mc = G.F.cfg["submission"]
    checks += FA.readback_checks(facts, data.template_profile, mc["expected"], size_mb, float(mc["max_size_mb"]), vsha,
                                 wsha, side.get("recipe_fp"), rec["fingerprint"], fsha)
    status = READY if FA.verdict(checks) in ("PASS", "WARNING") else "INVALID"
    G.save_npz(f"predictions/full_region/{path.stem}.npz", score=SC.astype(np.float16), binary=np.packbits(Q > 0),
               shape=np.asarray(Q.shape))
    G.ws.save_json({"checks": checks, "recipe": rec, "p6_validation": {k: v.get(k) for k in ("status", "failed",
                                                                                            "audit")},
                    "zones": zrows, "parity": P.to_dict("records")}, f"submissions/{path.stem}_validation.json")
    row.update(file=path.name, status=status, sha256=fsha, size_mb=round(size_mb, 4), path_local=str(path),
               path_drive=str(G.drive / G.out / "submissions" / path.name), written_array_sha256=wsha,
               reused_existing_file=w["reused"],
               reasons="; ".join(c["check"] for c in checks if c["hard"] and not c["ok"]),
               warnings="; ".join(c["check"] for c in checks if not c["hard"] and not c["ok"]))
    note = (f"{role}: {picked} passed the predeclared selection rule." if role == "SELECTED" else
            f"EXPLORATORY: {picked} did NOT pass the selection rule; W2 remains the reference. Upload only to test a "
            "hypothesis, never as the champion.")
    write_index(G, [row], note)
    return ("COMPLETE" if status == READY else "FAILED"), {"file": path.name, "status": status, "sha256": fsha,
                                                           "note": f"{path.name}: {status}"}
