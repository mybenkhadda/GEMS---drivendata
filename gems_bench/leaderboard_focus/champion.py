"""Notebook 01: the strongest VALID, DEPLOYABLE existing W2 configuration (no new model, no grid recomputation).

Procedure (every step is existing P6 code; M1 only restricts the candidate set and adds artifact gates):
1. Load the FROZEN boost-A grid (``full_benchmark.load_grids``) and check its protocol keys (``protocol_check``).
2. Candidate specs = the configured p6g candidate names (W2 family + the B0+B1 fusion fallback) + the historical
   uploaded cell (role ``historical``: evaluated for the record, never selectable).
3. Artifact gates per candidate (``deploy_artifacts``): the zone-stitched builder needs the three geographic held-out
   maps of its source (W2 cross-fitted maps; B0 / B1 OOF maps) with the unit crop shape. A missing map makes the
   candidate BLOCKED before any selection, so it can never win on a historical number.
4. Nested single-policy evaluation + the predeclared p6g rule (``full_benchmark.benchmark_tables`` / ``choose``):
   per outer zone ONE cell chosen on the other zones, applied to both tasks; eligibility = evaluated, deployable,
   worst-zone floor, geographic rule, ensemble rule, role. The 0.1758 geographic floor is reported per candidate and
   blocks only when ``selection.geo_floor_blocking`` (then the p6g ``gate`` mode is used).
5. The champion recipe (``champion_recipe.yaml``): the winner's all-zone deploy cell, its map sources and hashes, the
   zone-stitched construction. ``recipe_id`` = fingerprint of that recipe; it names the submission file.

Recorded historical values (configs) are kept in ``recorded_*`` columns, never in the recomputed ones.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import BLOCKED, COMPLETE

OUT = ("selection/champion_candidates.csv", "selection/champion_selection.json", "selection/champion_recipe.yaml",
       "selection/champion_provenance.json", "selection/blocked_candidates.csv", "reports/recovery_report.md")


# =============================================================================== pure
def candidate_specs(full_cfg: dict, m1_cfg: dict) -> dict:
    ch = m1_cfg["champion"]
    miss = [n for n in ch["candidates"] if n not in full_cfg["candidates"]]
    if miss:
        raise KeyError(f"M1 candidates absent from configs/p6/full_gpu.yaml: {miss}")
    specs = {n: dict(full_cfg["candidates"][n]) for n in ch["candidates"]}
    specs.update({n: dict(e) for n, e in (ch.get("extra_candidates") or {}).items()})
    return specs


def selection_rule(full_cfg: dict, m1_cfg: dict) -> dict:
    """p6g selection + M1 overrides; geo_floor_blocking switches the geographic rule to the p6g ``gate`` mode."""
    from .config import merge

    sel = merge(full_cfg["selection"], m1_cfg["selection"].get("overrides"))
    sel["references"] = [r for r in sel["references"] if r in m1_cfg["champion"]["candidates"]]
    if m1_cfg["selection"].get("geo_floor_blocking"):
        sel["geo_rule"] = {"mode": "gate", "floor": float(m1_cfg["selection"]["geo_floor"])}
    return sel


def apply_artifact_gates(specs: dict, gates: dict) -> dict:
    """Mark candidates whose deployment maps are missing as BLOCKED (``gates``: deploy kind -> list of problems)."""
    out = {}
    for n, sp in specs.items():
        sp = dict(sp)
        probs = gates.get(sp.get("deploy"), [])
        if probs and not sp.get("status"):
            sp.update(status="BLOCKED", reason="deployment maps unavailable: " + "; ".join(probs))
        out[n] = sp
    return out


def annotate(rows: list[dict], m1_cfg: dict) -> pd.DataFrame:
    """Candidate table: recomputed nested numbers, recorded values (separate columns), flags, decision reason."""
    rec = m1_cfg["recorded"]
    floor = float(m1_cfg["selection"]["geo_floor"])
    fam = m1_cfg["champion"]["family_of"]
    out = []
    for r in rows:
        g = pd.to_numeric(pd.Series([r.get("geo_dti")]), errors="coerce").iloc[0]
        rc = rec.get(r["name"], {})
        status = r.get("status")
        dec = ("CHAMPION" if r.get("is_winner") else "BASELINE" if r.get("is_baseline") else
               "ELIGIBLE_NOT_CHOSEN" if r.get("eligible") else "INELIGIBLE" if status == "EVALUATED" else status)
        out.append({"candidate_id": r["name"], "label": r.get("label", ""), "role": r.get("role"),
                    "family": fam.get(r.get("deploy"), r.get("deploy")), "source_grid": r.get("grid"),
                    "deploy": r.get("deploy"), "status": status, "status_reason": r.get("status_reason", ""),
                    "artifacts": "OK" if status != "BLOCKED" else "MISSING",
                    "disc_dti": r.get("disc_dti"), "disc_worst": r.get("disc_worst"),
                    "disc_worst_zone": r.get("disc_worst_zone"), "geo_dti": r.get("geo_dti"),
                    "objective": r.get("objective"), "deploy_cell": json.dumps(r.get("deploy_cell"), default=str)
                    if isinstance(r.get("deploy_cell"), dict) else r.get("deploy_cell"),
                    "deployable": r.get("deployable"), "geo_floor": floor,
                    "geo_floor_met": bool(np.isfinite(g) and g >= floor) if np.isfinite(g) else None,
                    "recorded_disc_dti": rc.get("disc_dti"), "recorded_disc_worst": rc.get("disc_worst"),
                    "recorded_geo_dti": rc.get("geo_dti"), "recorded_objective": rc.get("objective"),
                    "eligible": bool(r.get("eligible")), "eligibility_reasons": r.get("eligibility_reasons", ""),
                    "parity_status": "PENDING (notebook 02)", "export_feasible": bool(r.get("eligible")),
                    "decision": dec, "is_champion": bool(r.get("is_winner")),
                    "disc_per_zone": json.dumps(r.get("disc_per_zone")) if isinstance(r.get("disc_per_zone"), dict)
                    else r.get("disc_per_zone")})
    return pd.DataFrame(out)


def recipe_spec(row: dict, cell: dict, sources: dict, objective: str, fusion_w: float, sh: dict) -> dict:
    """The frozen deployment recipe of the champion (independent of any file written later)."""
    kind = row["deploy"]
    return {"recipe_version": "m1-recipe.1", "candidate": row["name"], "deploy_kind": kind,
            "selection": {"objective": objective, "rule": "configs/p6/full_gpu.yaml selection (+ M1 overrides)",
                          "selection_id": f"{row['name']}|{objective}|all_zones"},
            "policy": {"grid_candidate": cell["candidate"], "rep": cell["rep"], "frac": float(cell["frac"]),
                       "blend": float(cell["blend"]), "fusion_w_all_zones": float(fusion_w)},
            "representation": sh["baseline"]["representations"][cell["rep"]],
            "sigma_px": float(sh["evaluation"]["ridge_sigma_px"]),
            "score": {"stack": "(1 - blend) fusion + blend rank01(W2 held-out map) inside each geo zone",
                      "fusion": "w rank01(B0) + (1 - w) rank01(B1) inside each geo zone",
                      "cnn": "rank01(CNN OOF map) inside each geo zone"}.get(kind, kind),
            "construction": "zone_stitched: geographic fold zone k from the maps evaluated on zone k "
                            "(p6.final_submission.build_map); no refit, no new inference",
            "eligibility": "gems3.submission.eligible_mask(data, 0, 'eligible') per zone; mapped faults masked",
            "ranking": "gems3.representation.rank01 inside valid & fold_map == k", "calibration": "none (binary)",
            "raster": "gems3.submission.finalise(outside_valid=template) + submission.write_submission",
            "sources": sources}


# =============================================================================== Colab
def deploy_artifacts(sess) -> dict:
    """{deploy kind: [problems]} for the zone-stitched builder on every geographic unit."""
    bst, p6 = sess.bst, sess.p6
    probs = {"stack": [], "fusion": [], "cnn": []}
    for u in [x for x in bst.units() if x.task == "geo"]:
        p = p6.ws.root / "stack" / "maps" / f"{u.task}_{u.zone}.npy"
        if not p.exists():
            probs["stack"].append(f"{p.name} absent")
        else:
            a = np.load(p, mmap_mode="r")
            if a.shape != u.ev.shape:
                probs["stack"].append(f"{p.name} shape {a.shape} != unit crop {u.ev.shape}")
        try:
            bst.cnn(u)
        except FileNotFoundError as e:
            for k in probs:
                probs[k].append(str(e))
        u.release()
    return {k: v for k, v in probs.items() if v}


def map_sources(sess, kind: str) -> dict:
    from ..p6.boost_audit import file_hash

    root = sess.p6.ws.root
    out = {"cnn_models": list(sess.p6.m9.models)[:2], "m9_selection": file_hash(
        sess.p6.m9.local / sess.cfg["m9_output_dir"] / "baseline" / "baseline_selection.json")}
    if kind == "stack":
        out["w2_candidate"] = file_hash(root / "stack" / "candidate.json")
        out["w2_geo_maps"] = {p.name: file_hash(p) for p in sorted((root / "stack" / "maps").glob("geo_*.npy"))}
        out["w2_features"] = sess.bst.w2_features()
    out["boost_a_grid"] = file_hash(root / "boost" / "benchmark" / "grid.csv")
    return out


def recover(sess) -> dict:
    """Notebook 01 stage ``selection``."""
    from ..p6 import export_parity as EP
    from ..p6 import full_benchmark as FB

    fg, cfg = sess.fg, sess.cfg
    sess.require("artifacts")
    specs0 = candidate_specs(fg.cfg, cfg)
    sel = selection_rule(fg.cfg, cfg)
    fp = sess.fp("selection", sess.inventory_fp(), specs0, sel, cfg["recorded"],
                 fg.rc.read_json("export_parity", "decision"))

    def run():
        import yaml

        G, prov = FB.load_grids(fg)
        G = G[G.source.isin(cfg["champion"]["grids"])].reset_index(drop=True)
        specs = _resolve(fg, specs0, prov)
        gates = deploy_artifacts(sess)
        specs = apply_artifact_gates(specs, gates)
        PC, consistent = FB.protocol_check(G)
        zones = sorted(G.zone.unique()) if len(G) else []
        ov = fg.rc.load_table("unit_overlap", "export_parity")
        if not len(ov):
            ov = EP.unit_overlap(fg.bst)
        ex, feasible = EP.overlap_exclusion(ov, zones, fg.prim) if zones else (None, False)
        T = FB.benchmark_tables(G, specs, sel, fg.prim, fg.cfg["tasks"], ex, consistent, feasible)
        ch = T["selection"]
        meta = FB.meta_estimate(G, T["metrics"], specs, ch, sel, fg.prim, fg.cfg["tasks"], ex, T["objective"])
        C = annotate(ch["rows"], cfg)
        sess.ws.save_table(C, "selection/champion_candidates.csv", parquet=True)
        sess.ws.save_table(C[C.decision.isin(["BLOCKED", "NOT_RUN", "NOT_FOUND", "INCOMPATIBLE", "INCOMPLETE",
                                              "INELIGIBLE"])], "selection/blocked_candidates.csv")
        sess.ws.save_table(T["per_zone"].drop(columns=["systems"], errors="ignore"), "selection/candidate_per_zone.csv")
        sess.ws.save_table(PC, "selection/protocol_consistency.csv")
        win = ch["winner"]
        by = {r["name"]: r for r in ch["rows"]}
        reasons, recipe, rid = list(ch["reasons"]), None, None
        wrow = by.get(win) if win else None
        if wrow is None:
            reasons.append("no deployable candidate (no winner, no evaluated reference)")
        elif not wrow.get("eligible") and ch["decision"] != "NO_ELIGIBLE_CANDIDATE":
            reasons.append(f"winner {win} is not eligible: {wrow.get('eligibility_reasons')}")
        elif wrow.get("deploy_cell") is None:
            reasons.append(f"winner {win} has no all-zone deploy cell")
        if ch["decision"] == "NO_ELIGIBLE_CANDIDATE":
            reasons.append("no candidate passes eligibility; the strongest reference would be exported only after "
                           "you accept it explicitly (selection.overrides) - M1 does not do it silently")
        if wrow is not None and wrow.get("deploy_cell") is not None and not [r for r in reasons if "winner" in r]:
            cell = wrow["deploy_cell"] if isinstance(wrow["deploy_cell"], dict) else json.loads(wrow["deploy_cell"])
            w_all = float(fg.bst.sel[fg.prim][fg.p6.cfg["probes"]["champion"]]["params_all"]["w"])
            recipe = recipe_spec(wrow, cell, map_sources(sess, wrow["deploy"]), T["objective"], w_all, fg.p6.m9.sh)
            rid = sess.fp("recipe", recipe)[:10]
            recipe["recipe_id"] = rid
            sess.ws.save_text(yaml.safe_dump(json.loads(json.dumps(recipe, default=str)), sort_keys=False),
                              "selection/champion_recipe.yaml")
        else:
            sess.ws.save_text("# no champion recipe: see champion_selection.json -> reasons\n",
                              "selection/champion_recipe.yaml")
        status = COMPLETE if recipe is not None and ch["decision"] != "NO_ELIGIBLE_CANDIDATE" else BLOCKED
        out = {"status": status, "champion": win if status == COMPLETE else None, "decision": ch["decision"],
               "baseline": ch["baseline"], "best_eligible": ch["best_eligible"], "gain_vs_baseline":
               ch["gain_vs_baseline"], "objective": T["objective"], "objective_predeclared": T["objective_predeclared"],
               "overlap_exclusion_feasible": feasible, "rule": sel, "reasons": reasons, "recipe_id": rid,
               "meta_nested": {k: meta.get(k) for k in ("status", "disc_dti", "disc_worst", "candidates")},
               "artifact_gates": gates, "note": "nested offline numbers from the frozen boost-A grid; recorded "
               "values are in recorded_* columns; nothing here is a leaderboard score"}
        sess.ws.save_json(out, "selection/champion_selection.json")
        from .._version import provenance

        sess.ws.save_json({"grids": {k: v for k, v in prov.items() if k in cfg["champion"]["grids"]},
                           "w2_inputs": fg.bst.w2_inputs(), "cnn_members": _members(fg),
                           "data_fingerprint": fg.data_fingerprint(), "inventory_fp": sess.inventory_fp(),
                           "code": provenance(), "protocol_keys": fg.bst.protocol_keys()},
                          "selection/champion_provenance.json")
        from .reporting import recovery_report

        sess.ws.save_text(recovery_report(C, out), "reports/recovery_report.md")
        return status, reasons, {"champion": out["champion"], "recipe_id": rid}

    return sess.run_stage("selection", fp, OUT, run, inputs={"inventory_fp": sess.inventory_fp()})


def _resolve(fg, specs: dict, prov: dict) -> dict:
    """``full_benchmark.resolve_specs`` semantics for M1's own spec set (recovery choice, absent grids)."""
    from ..p6 import full_benchmark as FB

    saved = fg.cfg["candidates"]
    try:
        fg.cfg["candidates"] = specs
        return FB.resolve_specs(fg, prov)
    finally:
        fg.cfg["candidates"] = saved


def _members(fg) -> list:
    try:
        return [{"model": m.get("model"), "experiment_id": m.get("experiment_id"), "checkpoints": m.get("checkpoints")}
                for m in fg.p6.m9.m7.g3.members()]
    except Exception as e:                               # provenance never blocks; recorded
        return [{"error": f"{type(e).__name__}: {e}"}]
