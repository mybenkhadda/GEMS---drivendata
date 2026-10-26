"""Section 01: recover the baselines and establish the reference (System A = B0/B1 fusion + crest thinning).

Sources, in order:
1. M7 artifacts (read-only, staged from Drive): ``recovered_baselines/baseline_selection.json`` + the immutable
   ``baseline_table__<fp>.csv``, reused only when the protocol keys stored with them equal the current units' keys
   (same evaluation masks, scored truth, exclusions, TP convention, budget domain and template);
2. otherwise the same nested selection recomputed with M7's code (``m7.baseline.run``) under the shared protocol.

Either way the per-zone held-out maps of every method are rebuilt from the persisted per-zone choices
(``m7.baseline.heldout_map``) and saved under ``validation/maps/s01_<method>/`` stamped with their protocol key.

Recorded M7 offline values (``recorded_m7``) are reported next to the values found here; they are never reconciled,
and a match is never claimed unless the recomputation was executed in this run (``source`` column).

Compatibility checks (CRS, extent, transform, nodata, fold / split ids, model configuration, metric constants,
budget domain, score-map processing, map fingerprints): each row is MATCH, DIFFERS or UNKNOWN (M7 artifact absent).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

M7_TABLES = {"baseline_table": ("recovered_baselines", True), "baseline_per_zone": ("recovered_baselines", False),
             "area_ledger": ("recovered_baselines", False), "retained_fp_rescore": ("recovered_baselines", False),
             "protocol_keys": ("manifests", False), "model_inventory": ("manifests", False),
             "system_table": ("failure_analysis/tables", False), "system_attributes": ("failure_analysis/tables", False),
             "summary": ("failure_analysis/tables", False), "gate_decision": ("candidate_generation/oracle", False),
             "pooled_curve": ("candidate_generation/oracle", False), "incremental": ("candidate_generation/oracle", False),
             "candidate_inventory": ("candidate_generation/tables", False),
             "completion_baseline_summary": ("map_completion/metrics", False)}
M7_STAGE = ["recovered_baselines", "manifests", "failure_analysis/tables", "candidate_generation/oracle",
            "candidate_generation/tables", "candidate_generation/vectors/units", "map_completion/metrics",
            "map_completion/candidates", "dem/pilot", "dem/features", "dem/inventory"]


def recover_m7(m8) -> dict:
    """Every M7 table M8 can reuse (empty DataFrame when absent). Read-only."""
    m8.stage_m7(M7_STAGE)
    out = {}
    for name, (sub, immutable) in M7_TABLES.items():
        out[name] = m8.m7.ws.load_immutable_table(name, sub) if immutable else m8.m7.ws.load_table(name, sub)
    return out


def _keys_by_task(t: pd.DataFrame) -> dict:
    if not len(t) or "protocol_key" not in t:
        return {}
    return t.groupby("task").protocol_key.apply(lambda s: "|".join(sorted(set(s)))).to_dict()


def reuse_decision(m8, rec: dict, sel_m7: dict | None) -> tuple[bool, str]:
    if not m8.cfg["baseline"]["reuse_m7_selection"]:
        return False, "baseline.reuse_m7_selection = false"
    if not sel_m7:
        return False, "M7 baseline_selection.json absent"
    bt = rec["baseline_table"]
    if not len(bt) or "protocol_keys" not in bt:
        return False, "M7 baseline table absent or without protocol keys"
    now = _keys_by_task(m8.m7.protocol_keys())
    then = bt.dropna(subset=["protocol_keys"]).groupby("task").protocol_keys.first().to_dict()
    tasks = [t for t in sel_m7 if t in now]
    bad = [t for t in tasks if then.get(t) != now.get(t)]
    if not tasks:
        return False, "no common task"
    if bad:
        return False, f"protocol keys differ for {bad} (units / masks / budget domain changed)"
    return True, f"protocol keys match for {tasks}"


def recorded_table(table: pd.DataFrame, rec: dict, source: str) -> pd.DataFrame:
    rows = []
    for m, v in (rec.get("disc_official_dti") or {}).items():
        hit = table[(table.task == "disc_official") & (table.method == m)] if len(table) else table
        new = float(hit.dti_crossfit.iloc[0]) if len(hit) else np.nan
        rows.append({"task": "disc_official", "method": m, "recorded_m7_offline": float(v),
                     "value_this_run": new, "difference": new - float(v) if np.isfinite(new) else np.nan,
                     "source": source,
                     "note": "recorded = earlier M7 run (offline, not leaderboard); a difference is reported, never "
                             "reconciled"})
    return pd.DataFrame(rows)


def compatibility(m8, rec: dict) -> pd.DataFrame:
    from ..metric import DTI_ALPHA, DTI_BETA, DTI_RADIUS_M, OFFICIAL_EXCLUDE_SUPPORTS_TP

    ctx, d = m8.ctx, m8.ctx.data
    setup = m8.m7_json("manifests/setup.json") or {}
    m7cfg = m8.m7_json("manifests/m7_config.json") or {}
    rows = []

    def add(check, current, recorded, status=None, note=""):
        if status is None:
            status = "UNKNOWN (M7 artifact absent)" if recorded is None else ("MATCH" if str(current) == str(recorded)
                                                                              else "DIFFERS")
        rows.append({"check": check, "current": json.dumps(current, default=str),
                     "m7_recorded": json.dumps(recorded, default=str), "status": status, "note": note})

    epsg = d.grid.crs.to_epsg() if hasattr(d.grid.crs, "to_epsg") else str(d.grid.crs)
    add("crs_epsg", epsg, 32611, "MATCH" if epsg == 32611 else "DIFFERS", "required EPSG:32611")
    chk = setup.get("checks") or {}
    add("raster_shape", list(d.shape), chk.get("shape"))
    add("pixel_m", [d.grid.pixel_km_x * 1000, d.grid.pixel_km_y * 1000], chk.get("pixel_m"))
    add("template_same_grid (transform, extent, shape, CRS)", bool(d.template_matches()), True,
        "MATCH" if d.template_matches() else "DIFFERS")
    add("template_nodata", d.template_profile.get("nodata"), None, "RECORDED", "template nodata value (authority)")
    add("feature_nodata", d.profile.get("nodata"), None, "RECORDED", "")
    add("valid_frac", round(float(d.valid.mean()), 6), round(float(chk["valid_frac"]), 6) if "valid_frac" in chk else None)
    add("template_valid_frac", round(float(np.asarray(d.template_valid).mean()), 6),
        round(float(chk["template_valid_frac"]), 6) if "template_valid_frac" in chk else None)
    add("label_px", int(d.labels.sum()), chk.get("label_px"))
    add("fold_ids", list(ctx.fold_ids), list(range(int(chk["n_folds"]))) if "n_folds" in chk else None)
    add("split_ids", list(range(int(ctx.splits.n_splits))), list(range(int(chk["n_splits"]))) if "n_splits" in chk else None)
    pk = m8.m7.protocol_keys()
    pk7 = rec["protocol_keys"]
    if len(pk7) and "unit" in pk7:
        a = dict(zip(pk.unit, pk.protocol_key))
        b = dict(zip(pk7.unit, pk7.protocol_key))
        same = [u for u in a if b.get(u) == a[u]]
        add("unit_protocol_keys", f"{len(same)}/{len(a)} units identical", f"{len(b)} units",
            "MATCH" if len(same) == len(a) == len(b) else "DIFFERS", "evaluation masks, truth, exclusions, budget")
    else:
        add("unit_protocol_keys", f"{len(pk)} units", None)
    inv = m8.m7.g3.m6.inv
    mi = rec["model_inventory"]
    for m in m8.models:
        cur = inv[inv.model == m][["config_hash", "experiment_id"]].to_dict("records") if inv is not None and "config_hash" in inv else []
        old = mi[mi.model == m][["config_hash", "experiment_id"]].to_dict("records") if len(mi) and "config_hash" in mi else None
        add(f"model_config:{m}", cur, old)
    add("map_fingerprints", m8.m7.g3.m6.map_fp, setup.get("map_fingerprints"))
    add("metric_constants", {"alpha": DTI_ALPHA, "beta": DTI_BETA, "radius_m": DTI_RADIUS_M,
                             "official_exclude_supports_tp": OFFICIAL_EXCLUDE_SUPPORTS_TP}, None, "RECORDED",
        "the scorer of every M8 number (gems3.official.FastScorer)")
    ev, ev7 = m8.sh["evaluation"], (m7cfg.get("evaluation") or {})
    add("budget_domain", ev["budget_domain"], ev7.get("budget_domain") if ev7 else None)
    add("include_template_mask", ev["include_template_mask"], ev7.get("include_template_mask") if ev7 else None)
    add("score_processing", {"ridge_sigma_px": ev["ridge_sigma_px"], "fusion_weights": ev["fusion_weights"],
                             "representations": m8.sh["baseline"]["representations"]},
        {"ridge_sigma_px": ev7.get("ridge_sigma_px"), "fusion_weights": ev7.get("fusion_weights"),
         "representations": (m7cfg.get("baseline") or {}).get("representations")} if ev7 else None)
    add("area_fracs", ev["area_fracs"], ev7.get("area_fracs") if ev7 else None)
    return pd.DataFrame(rows)


def run(m8) -> dict:
    from ..m7 import baseline as B
    from ..m7 import failure as F

    view = m8.view
    rec = recover_m7(m8)
    sel_m7 = m8.m7_json("recovered_baselines/baseline_selection.json")
    reuse, why = reuse_decision(m8, rec, sel_m7)
    if reuse:
        selection = sel_m7
        table = rec["baseline_table"]
        per_zone, ledger, alt = rec["baseline_per_zone"], rec["area_ledger"], rec["retained_fp_rescore"]
        source = "reused M7 selection (protocol keys match); held-out maps rebuilt here"
    else:
        m8.say(f"s01 recomputing the baselines with M7's code ({why})")
        out = B.run(view)
        selection, table = out["selection"], out["table"]
        per_zone, ledger, alt = out["per_zone"], out["ledger"], out["retained_fp"]
        source = "recomputed in this run (M7 code, shared protocol)"
    m8.set_selection(selection)
    mm = B.MemberMaps(m8.ctx, m8.models)
    saved = []
    from ..m7.rasterize import save_heldout

    for task, ms in selection.items():
        for m in ms:
            for u in m8.units_of(task):
                q = B.heldout_map(view, task, m, u.zone, mm)
                if q is None:
                    continue
                save_heldout(m8.ws, f"s01_{m}", u, q, m8.pkey(u))
                saved.append({"method": f"s01_{m}", "task": task, "zone": u.zone,
                              "area_frac_eligible": float(((q > 0) & m8.elig(u, 0)).sum() / max(m8.elig(u, 0).sum(), 1))})
        for u in m8.units_of(task):
            mm.release(u)
    systab, attrs = rec["system_table"], rec["system_attributes"]
    sys_source = "M7 failure analysis (read-only)"
    if not len(systab) or not reuse:
        f = F.run(view)
        systab, attrs = f["systems"], f["attributes"]
        sys_source = "recomputed (M7 failure code, this selection)"
    comp = compatibility(m8, rec)
    m7_pool = rec["candidate_inventory"]
    m7_pool_summary = (m7_pool.groupby(["task", "zone"]).agg(n=("cand_id", "size"), km=("length_km", "sum")).reset_index()
                       if len(m7_pool) and {"task", "zone", "cand_id", "length_km"} <= set(m7_pool.columns) else pd.DataFrame())
    manifest = {"selection_source": source, "reuse_reason": why, "system_table_source": sys_source,
                "models": m8.models, "model_stems": list(m8.sh["models"]),
                "map_fingerprints": m8.m7.g3.m6.map_fp, "m7_experiment_ids": m8.m7_experiment_ids(),
                "champion": m8.cfg["baseline"]["champion"], "protocol": m8.sh["evaluation"]["budget_domain"],
                "compatibility_failures": comp[comp.status == "DIFFERS"].check.tolist()}
    return {"selection": selection, "table": table, "per_zone": per_zone, "ledger": ledger, "retained_fp": alt,
            "recorded": recorded_table(table, m8.cfg["recorded_m7"], source), "saved": pd.DataFrame(saved),
            "systems": systab, "attributes": attrs, "compatibility": comp, "m7_pool_summary": m7_pool_summary,
            "m7_gate": rec["gate_decision"], "manifest": manifest, "reused": reuse}
