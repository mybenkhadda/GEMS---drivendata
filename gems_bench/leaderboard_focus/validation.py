"""Stage 3: official validation of the recovered champion with the EXISTING scorer.

Four independent checks, every one with an explicit tolerance; a failure of 1, 2 or 4 makes the stage BLOCKED, a
recorded value that is not reproduced (3) is a WARNING unless ``validation.recorded_mismatch: blocking``:
1. Recomputation. For each outer zone and task, the champion's held-out cell is rebuilt from the frozen maps with
   the exact boost-A expressions (F = per-zone champion fusion, W = rank of the W2 held-out map; w2: W, hybrid /
   blends: (1 - l) F + l W, fallback: F, CNN: rank) and scored by ``Boost._eval_q`` (``m7.protocol.score_row`` ->
   ``gems3.official.FastScorer``). TP / FP / FN must equal the saved grid row (rtol / atol from the config).
2. Pooling. The pooled disc / geo DTI of the recomputed components must equal the nested summary of notebook 01.
3. Reproduction of recorded values. The recorded numbers (configs: same grid, same space, same objective) are
   recomputed from the frozen grids: ``separate`` = disc and geo selected independently (how boost A reported them),
   ``fixed`` = one historical cell, ``stack_grid`` = W2's own ``stack/grid.csv`` nested over (blend, rep, frac).
   Every row states its protocol and whether it is comparable with the champion's single-policy number
   (``protocol_comparison.csv``): a separate-protocol GEO value is not (its geo cell was chosen on geo labels), so a
   gap between it and the champion's geo DTI is a protocol difference, not evidence of a software defect.
4. Deployment dry run. ``final_submission.inspect_spec`` (template CRS, resolution, grid) and ``build_map`` (the
   zone-stitched construction, no file written): parity per geo zone (strict: every zone-score row PASS), coverage,
   per-zone area budget, no positive pixel on a mapped fault, recipe == notebook-01 recipe. The finalised array's
   sha256 is recorded; notebook 03 requires the written GeoTIFF to carry exactly this array.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from .config import BLOCKED, COMPLETE, INCOMPLETE, WARNING

COMP = ["tp_w", "fp_w", "fn_w", "n_truth"]
OUT = ("validation/official_scores.csv", "validation/per_zone_metrics.csv", "validation/weighted_components.csv",
       "validation/recorded_reproduction.csv", "validation/protocol_comparison.csv", "validation/validation_status.json",
       "validation/zone_budget.csv", "parity/zone_stitched_parity.csv", "reports/validation_report.md")
CHAMPION_PROTOCOL = "single_policy"
PROTOCOLS = {
    "single_policy": "one cell per outer zone chosen on the discovery labels of the other zones, applied unchanged to "
                     "both tasks (deployable: one file)",
    "separate": "discovery and geographic cells chosen independently, each on its own task's labels (not one file)",
    "fixed": "one fixed historical cell (no nested choice)",
}


# =============================================================================== pure: protocols
def comparability(protocol: str, metric: str) -> tuple[bool, str]:
    """Is a recorded ``metric`` under ``protocol`` comparable with the champion's single-policy number?"""
    if protocol == CHAMPION_PROTOCOL:
        return True, "same protocol"
    if protocol == "separate" and metric.startswith("disc"):
        return True, "same discovery selection criterion (the discovery cell is chosen on discovery labels in both)"
    if protocol == "separate":
        return False, ("geo cell chosen on geographic labels; the deployable policy applies the discovery-chosen "
                       "cell to the geographic zones, so a lower geo DTI is expected and not a defect")
    return False, "a fixed historical cell, not a nested selection"


def recorded_row(name: str, rec: dict, metric: str, got, tol: float) -> dict:
    """One recorded-vs-reproduced row: REPRODUCED / NOT_REPRODUCED / NOT_AVAILABLE + protocol facts (pure)."""
    proto = rec.get("protocol") or ("fixed" if rec.get("objective") == "fixed" else "separate")
    comp, why = comparability(proto, metric)
    v = None if got is None or not np.isfinite(float(got)) else float(got)
    st = "NOT_AVAILABLE" if v is None else "REPRODUCED" if abs(v - float(rec[metric])) <= tol else "NOT_REPRODUCED"
    return {"recorded_id": name, "source": rec["source"], "objective": rec.get("objective", "nested"),
            "protocol": proto, "metric": metric, "recorded": float(rec[metric]), "reproduced": v,
            "abs_diff": None if v is None else abs(v - float(rec[metric])), "tolerance": tol, "status": st,
            "comparable_to_champion": comp, "comparability_note": why}


def protocol_comparison(crow: dict, RR: pd.DataFrame, single: dict) -> pd.DataFrame:
    """The champion's nested numbers next to every recorded value, with protocol and comparability (pure)."""
    rows = [{"value_id": f"champion:{crow.get('candidate_id')}", "kind": "recomputed (this run)",
             "protocol": CHAMPION_PROTOCOL, "metric": m, "value": crow.get(m), "comparable_to_champion": True,
             "note": PROTOCOLS[CHAMPION_PROTOCOL]} for m in ("disc_dti", "disc_worst", "geo_dti")]
    for r in (RR.to_dict("records") if len(RR) else []):
        rows.append({"value_id": f"recorded:{r['recorded_id']}", "kind": f"recorded, {r['status']}",
                     "protocol": r["protocol"], "metric": r["metric"], "value": r["recorded"],
                     "comparable_to_champion": r["comparable_to_champion"], "note": r["comparability_note"]})
    for n, rec in (single or {}).items():
        for m in ("disc_dti", "disc_worst", "geo_dti"):
            if rec.get(m) is not None:
                rows.append({"value_id": f"recorded:{n}", "kind": "recorded (P6.1 run; context only)",
                             "protocol": CHAMPION_PROTOCOL, "metric": m, "value": float(rec[m]),
                             "comparable_to_champion": True, "note": "same protocol, recorded by the P6.1 recovery "
                             "run; the exclusion it used is not re-verified here"})
    return pd.DataFrame(rows)


# =============================================================================== pure
def dti(tp, fp, fn) -> float:
    return float(tp / (tp + 0.2 * fp + 0.8 * fn + 1e-6))


def close(a, b, rtol: float, atol: float) -> bool:
    a, b = float(a), float(b)
    if not (np.isfinite(a) and np.isfinite(b)):
        return False
    return abs(a - b) <= atol + rtol * abs(b)


def score_expression(orig: str, blend: float, F: np.ndarray | None, W: np.ndarray | None,
                     cnn: dict | None = None) -> np.ndarray:
    """The boost-A ranking score of one grid candidate (``Boost._a`` expressions, bit-for-bit)."""
    if orig == "fallback_fusion":
        return F
    if orig in ("cnn_b0", "cnn_b1"):
        return cnn[orig]
    if orig == "w2":
        return W if float(blend) == 1.0 else ((1.0 - float(blend)) * F + float(blend) * W).astype(np.float32)
    if orig == "hybrid":
        return (1.0 - float(blend)) * F + float(blend) * W
    raise ValueError(f"no boost-A expression for grid candidate {orig!r}")


def compare_rows(saved: dict, got: dict, rtol: float, atol: float) -> dict:
    out = {}
    ok = True
    for c in COMP:
        s, g = float(saved[c]), float(got[c])
        out[f"saved_{c}"], out[f"recomputed_{c}"] = s, g
        ok &= close(g, s, rtol, atol)
    out["saved_dti"] = dti(out["saved_tp_w"], out["saved_fp_w"], out["saved_fn_w"])
    out["recomputed_dti"] = dti(out["recomputed_tp_w"], out["recomputed_fp_w"], out["recomputed_fn_w"])
    out["status"] = "PASS" if ok else "BLOCKED"
    return out


def pooled(rows: pd.DataFrame, prim: str) -> dict:
    out = {}
    for task, d in rows.groupby("task"):
        pre = "disc" if task == prim else "geo"
        out[f"{pre}_dti_recomputed"] = dti(d.recomputed_tp_w.sum(), d.recomputed_fp_w.sum(), d.recomputed_fn_w.sum())
        out[f"{pre}_worst_recomputed"] = float(d.recomputed_dti.min())
        for c in ("tp_w", "fp_w", "fn_w", "n_truth"):
            out[f"{pre}_{c}"] = float(d[f"recomputed_{c}"].sum())
    return out


def recipe_consistency(recipe: dict, built: dict) -> list[str]:
    """Differences between the notebook-01 recipe and the cell / sources the builder used (empty = consistent)."""
    pol, cell = recipe["policy"], built["cell"]
    bad = []
    for k in ("rep", "frac", "blend"):
        a, b = pol[k], cell[k]
        if (float(a) != float(b)) if k != "rep" else (a != b):
            bad.append(f"{k}: recipe {a} != builder {b}")
    if pol["grid_candidate"] != cell["candidate"]:
        bad.append(f"grid candidate: recipe {pol['grid_candidate']} != builder {cell['candidate']}")
    if abs(float(pol["fusion_w_all_zones"]) - float(built["w_all"])) > 1e-12:
        bad.append(f"fusion w: recipe {pol['fusion_w_all_zones']} != builder {built['w_all']}")
    if recipe["deploy_kind"] != built["kind"]:
        bad.append(f"deploy kind: recipe {recipe['deploy_kind']} != builder {built['kind']}")
    if built["kind"] == "stack" and built.get("stack_dir") != "stack/maps":
        bad.append(f"stacker maps from {built.get('stack_dir')}, expected the frozen W2 stack/maps")
    return bad


def array_sha(a: np.ndarray) -> str:
    """sha256 of a float32 array's bytes, NaN canonicalised (so the written GeoTIFF band can be compared)."""
    x = np.asarray(a, np.float32).copy()
    x[np.isnan(x)] = np.float32(np.nan)
    return hashlib.sha256(np.ascontiguousarray(x).tobytes() + str(x.shape).encode()).hexdigest()


def budget_rows(zones: pd.DataFrame, frac: float, tol: float, evaluated: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-zone area budget under the upper-bound contract of ``audit_export`` (``forensics.budget_table``).

    M1.3: M1.1-M1.2 required ``|area / frac - 1| <= tol``; a thin representation keeps only part of the frac pool, so
    every zone of the M1.2 run failed (rel_error -0.51 to -0.54, 0 px on mapped faults) while the evaluated grid cells
    had the same thinned area. The rule is now the one the written file is audited with (area <= frac x (1 + tol), no
    positive mapped-fault pixel, non-empty); the old verdict is kept as ``m1_1_two_sided_ok``."""
    from .forensics import budget_table

    return budget_table(zones, frac, tol, evaluated)


# =============================================================================== Colab
def champion_rows(sess) -> tuple[dict, pd.DataFrame, dict]:
    from .session import StageBlocked, StageInvalid

    sel = sess.ws.read_json("selection/champion_selection.json") or {}
    name = sel.get("champion")
    if not name:
        raise StageBlocked("no champion recorded by the selection stage")
    C = sess.ws.require_table("selection/champion_candidates.csv", ("candidate_id", "deploy", "deploy_cell"))
    if not (C.candidate_id == name).any():
        raise StageInvalid(f"champion {name} absent from selection/champion_candidates.csv: re-run selection")
    row = C[C.candidate_id == name].iloc[0].to_dict()
    PZ = sess.ws.require_table("selection/candidate_per_zone.csv", ("name", "task", "zone", "rep", "frac"))
    P = PZ[PZ.name == name].reset_index(drop=True)
    if not len(P):
        raise StageInvalid(f"no per-zone rows for {name} in selection/candidate_per_zone.csv: re-run selection")
    recipe_id = (sess.manifest("selection") or {}).get("recipe_id")
    if recipe_id is not None and recipe_id != sel.get("recipe_id"):
        raise StageInvalid(f"recipe ids disagree: selection manifest {recipe_id} != champion_selection.json "
                           f"{sel.get('recipe_id')}: re-run selection")
    return sel, P, row


def recompute(sess, P: pd.DataFrame, tol: dict) -> pd.DataFrame:
    from ..gems3 import representation as R

    bst = sess.bst
    rows = []
    for r in P.to_dict("records"):
        u = sess.p6.m9.unit(f"{r['task']}:{int(r['zone'])}")
        orig = str(r["cell_candidate"]).partition(":")[2]
        F, W = bst.fusion(u), bst.stack_rank(u, bst.w2_map(u))
        b0, b1 = bst.cnn(u)
        s = score_expression(orig, float(r["blend"]) if pd.notna(r["blend"]) else np.nan, F, W,
                             {"cnn_b0": R.rank01(b0, u.ev), "cnn_b1": R.rank01(b1, u.ev)})
        q = bst._apply(u, s, r["rep"], float(r["frac"]), bst._crest(u, s))
        got = bst._eval_q(u, q, candidate=orig, rep=r["rep"], frac=float(r["frac"]))
        cmp = compare_rows(r, got, float(tol["component_rtol"]), float(tol["component_atol"]))
        rows.append({"task": r["task"], "zone": int(r["zone"]), "cell_candidate": r["cell_candidate"],
                     "blend": r["blend"], "rep": r["rep"], "frac": float(r["frac"]), **cmp,
                     "area_frac_eligible": got.get("area_frac_eligible"), "area_px": got.get("area_px"),
                     "n_systems": got.get("n_systems"), "n_recovered": got.get("n_recovered")})
        u.release()
    return pd.DataFrame(rows)


def reproduce_recorded(sess) -> pd.DataFrame:
    """Recorded values vs the same quantity recomputed from the frozen grids (tolerance ``recorded_atol``)."""
    from ..gems3.metric_opt import nested_select
    from ..p6 import export_parity as EP
    from ..p6 import full_benchmark as FB
    from .champion import candidate_specs
    from .session import StageInvalid

    cfg, fg = sess.cfg, sess.fg
    tol = float(cfg["validation"]["recorded_atol"])
    try:
        G, _ = FB.load_grids(fg, sources=cfg["champion"]["grids"], required=cfg["champion"]["required_grids"])
    except FB.GridUnavailable as e:
        raise StageInvalid(str(e)) from e
    G = G[G.source == "boost_a"]
    specs = candidate_specs(fg.cfg, cfg)
    rows = []
    for name, rec in cfg["recorded"].items():
        got, note = {}, ""
        if rec["source"] == "stack_grid":
            params = list(rec["params"])
            S, facts = FB.read_csv_checked(sess.p6.ws.root / "stack" / "grid.csv",
                                           ["task", "zone", *params, "tp_w", "fp_w", "fn_w"])
            note = "" if S is not None else f"stack/grid.csv {facts['status']} ({facts['detail']})"
        elif name in specs:                              # the boost / stack summaries used nested_select per task
            S, params = EP.policy_rows(G, FB.candidate_space(specs[name], G)), ["candidate", "blend", "rep", "frac"]
        else:
            S, note = None, f"{name}: no candidate spec to recompute it from"
        for task in (S.task.unique() if S is not None and len(S) else []):
            d = S[S.task == task]
            pre = "disc" if task == fg.prim else "geo"
            if rec.get("objective") == "fixed":
                got[f"{pre}_dti"] = dti(d.tp_w.sum(), d.fp_w.sum(), d.fn_w.sum())
            else:
                _, s = nested_select(d, params)
                got[f"{pre}_dti"], got[f"{pre}_worst"] = s.get("dti_crossfit"), s.get("fold_min")
        for k in ("disc_dti", "disc_worst", "geo_dti"):
            if rec.get(k) is not None:
                rows.append({**recorded_row(name, rec, k, got.get(k), tol), "note": note})
    return pd.DataFrame(rows)


def recorded_effect(RR: pd.DataFrame, mode: str) -> tuple[list[str], list[str]]:
    """(blocking reasons, warnings) of the recorded reproduction table under ``validation.recorded_mismatch``."""
    bad = RR[RR.status != "REPRODUCED"] if len(RR) else RR
    msgs = [f"{r.recorded_id}.{r.metric} [{r.protocol}] {r.status}: reproduced {r.reproduced} vs recorded "
            f"{r.recorded}" for r in bad.itertuples()]
    return (msgs, []) if mode == "blocking" else ([], msgs)


def forensics(sess, built: dict, Z: pd.DataFrame, cfg: dict) -> list[str]:
    """M1.3 diagnostics of the dry build (never gating): evaluated vs deployed map, eligible domains, budget root
    cause, geographic protocol figure. Returns warnings; an exception here is recorded (warning + traceback file),
    because a diagnostic must not hide the validation verdict, and it is never silently dropped."""
    import traceback

    from ..gems3 import submission as SUB
    from . import forensics as FO

    try:
        data = sess.fg.p6.ctx.data
        el = SUB.eligible_mask(data, 0, "eligible")
        RC, code = FO.map_comparison(built["q"], built["q_eval"], built["zone_id"], el, data.labels)
        sess.ws.save_table(RC, "diagnostics/representation_comparison.csv", parquet=True)
        from ..artifacts.atomic import atomic_save

        def _npz(t):                                     # a file object: np.savez appends .npz to a bare path
            with open(t, "wb") as fh:
                np.savez_compressed(fh, code=code, zone_id=np.asarray(built["zone_id"], np.int16),
                                    legend=np.array(["none", "both", "evaluated_only", "deployed_only"]))

        atomic_save(sess.ws.path("diagnostics/evaluated_vs_exported_maps.npz"), _npz)
        M = FO.mask_alignment(Z)
        sess.ws.save_table(M, "diagnostics/mask_alignment.csv")
        sess.ws.save_table(Z, "diagnostics/budget_root_cause.csv", parquet=True)
        summ = {**FO.root_cause_summary(Z), "source": "validation stage (fresh zone-stitched build)",
                "evidence": "OBSERVED", "map_comparison": RC.to_dict("records")}
        sess.ws.save_json(summ, "diagnostics/root_cause_summary.json")
        step = max(1, int(np.ceil(max(code.shape) / 1200)))
        FO.budget_figure(sess.ws, Z)
        FO.mask_figure(sess.ws, M)
        FO.comparison_figure(sess.ws, RC, code, step)
        FO.geo_figure(sess.ws, sess.ws.load_table("validation/protocol_comparison.csv"),
                      float(cfg["selection"]["geo_floor"]))
        sess.ws.save_json({"stage": "diagnostics", "status": "COMPLETE", "outputs": [
            "diagnostics/representation_comparison.csv", "diagnostics/evaluated_vs_exported_maps.npz",
            "diagnostics/mask_alignment.csv", "diagnostics/budget_root_cause.csv",
            "diagnostics/root_cause_summary.json"]}, "manifests/diagnostics.json")
        return []
    except Exception as e:
        sess.ws.save_text(traceback.format_exc(), "diagnostics/forensics_error.txt")
        sess.ws.save_json({"stage": "diagnostics", "status": "FAILED", "reason": f"{type(e).__name__}: {e}"},
                          "manifests/diagnostics.json")
        return [f"forensics diagnostics failed ({type(e).__name__}: {e}); see diagnostics/forensics_error.txt"]


def validate(sess) -> dict:
    """Stage ``validation``."""
    from ..gems3 import submission as SUB
    from ..p6 import final_submission as FS
    from .session import StageBlocked

    cfg, fg = sess.cfg, sess.fg
    vc = cfg["validation"]
    sel = sess.ws.read_json("selection/champion_selection.json") or {}
    fp = sess.fp("validation", sess.manifest("selection").get("fingerprint"), vc, cfg["recorded"],
                 cfg.get("recorded_single_policy"))
    try:
        sess.require("artifacts", "selection")
    except StageBlocked as e:
        return sess.record("validation", INCOMPLETE, fp, reasons=[str(e)])

    def run():
        import yaml

        sel_, P, crow = champion_rows(sess)
        reasons = []
        V = recompute(sess, P, vc)
        sess.ws.save_table(V, "validation/per_zone_metrics.csv", parquet=True)
        if not len(V) or (V.status != "PASS").any():
            reasons.append(f"recomputed components differ from the saved grid on {list(V[V.status != 'PASS'].zone)}")
        pl = pooled(V, fg.prim) if len(V) else {}
        summ = {"candidate": sel_["champion"], **pl, "disc_dti_nested": crow.get("disc_dti"),
                "disc_worst_nested": crow.get("disc_worst"), "geo_dti_nested": crow.get("geo_dti")}
        for pre in ("disc", "geo"):
            a, b = pl.get(f"{pre}_dti_recomputed"), crow.get(f"{pre}_dti")
            ok = a is not None and b is not None and close(a, b, 0.0, float(vc["dti_atol"]))
            summ[f"{pre}_pooling_status"] = "PASS" if ok else "BLOCKED"
            if not ok:
                reasons.append(f"{pre} pooled DTI recomputed {a} != nested {b}")
        sess.ws.save_table(pd.DataFrame([summ]), "validation/official_scores.csv")
        comp = V.groupby("task")[[f"recomputed_{c}" for c in COMP]].sum().reset_index() if len(V) else pd.DataFrame()
        sess.ws.save_table(comp, "validation/weighted_components.csv")
        RR = reproduce_recorded(sess)
        sess.ws.save_table(RR, "validation/recorded_reproduction.csv")
        sess.ws.save_table(protocol_comparison(crow, RR, cfg.get("recorded_single_policy")),
                           "validation/protocol_comparison.csv")
        rr_block, warns = recorded_effect(RR, vc.get("recorded_mismatch", "warning"))
        reasons += rr_block
        try:
            spec = FS.inspect_spec(fg)
        except ValueError as e:
            spec = {"problems": [str(e)]}
            reasons.append(f"official template check: {e}")
        recipe = yaml.safe_load(sess.ws.path("selection/champion_recipe.yaml").read_text(encoding="utf-8"))
        built = FS.build_map(fg, {"name": crow["candidate_id"], "deploy": crow["deploy"],
                                  "deploy_cell": json.loads(crow["deploy_cell"])})
        par = built["parity"]
        sess.ws.save_table(par, "parity/zone_stitched_parity.csv")
        zs = par[par.check == "zone score vs evaluation construction"]
        bad = zs[zs.status != "PASS"] if vc["strict_parity"] else zs[zs.status == "BLOCKED"]
        if len(bad):
            reasons.append(f"zone-stitched score parity not PASS on {list(bad.unit)} ({list(bad.status)}): "
                           "the deployed score is not the evaluated score")
        cov = built["coverage"]
        if cov["uncovered_frac"] > float(vc["max_uncovered_frac"]):
            reasons.append(f"uncovered valid pixels {cov['uncovered_frac']:.5f} > {vc['max_uncovered_frac']}")
        from . import forensics as FO

        Z = budget_rows(built["zones"], float(built["cell"]["frac"]), float(fg.cfg["submission"]["budget_tolerance"]),
                        FO.evaluated_area(V))
        sess.ws.save_table(Z, "validation/zone_budget.csv", parquet=True)
        if not Z.ok.all():
            bad_z = Z[~Z.ok]
            reasons.append(f"per-zone area budget / known-fault mask violated on zones {list(bad_z.zone)}: "
                           f"{dict(zip(bad_z.zone.astype(int), bad_z.cause))}")
        warns += forensics(sess, built, Z, cfg)
        rc = recipe_consistency(recipe, built)
        reasons += [f"recipe mismatch: {x}" for x in rc]
        arr = SUB.finalise(built["q"], fg.p6.ctx.data, fg.cfg["submission"]["outside_valid"])
        status = BLOCKED if reasons else WARNING if warns else COMPLETE
        st = {"status": status, "reasons": reasons, "warnings": warns, "champion": sel_["champion"],
              "recipe_id": sel_.get("recipe_id"), "validated_array_sha256": array_sha(arr),
              "positive_px": int((np.nan_to_num(arr) > 0).sum()), "coverage": cov, "official_spec": spec,
              "tolerances": vc, "summary": summ,
              "note": "discovery quality of the exported map is the nested disc DTI of the same policy on the "
                      "discovery units; the exported zone maps are the geographic held-out maps"}
        sess.ws.save_json(st, "validation/validation_status.json")
        from .reporting import validation_figures, validation_report

        validation_figures(sess, V, RR, Z, par, crow)
        sess.ws.save_text(validation_report(st, V, RR, Z, par), "reports/validation_report.md")
        return (BLOCKED if reasons else COMPLETE), reasons, {"validated_array_sha256": st["validated_array_sha256"],
                                                             "recipe_id": sel_.get("recipe_id")}, warns

    return sess.run_stage("validation", fp, OUT, run, inputs={"selection_fp": sess.manifest("selection").get(
        "fingerprint"), "champion": sel.get("champion")})
