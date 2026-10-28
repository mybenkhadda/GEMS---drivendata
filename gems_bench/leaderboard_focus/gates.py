"""Stage ``promotion``: may the selected champion be exported? Separate from "did the selection compute".

Gates (each PASS / WARNING / BLOCKED / PENDING, with its evidence):
    SELECTION_COMPLETE    selection stage COMPLETE, one recipe id agreeing across the selection files
    DISCOVERY_ELIGIBLE    the champion is eligible under the predeclared p6g rule and its worst discovery zone is
                          >= selection.worst_floor
    GEOGRAPHIC_GATE_PASS  champion geo DTI (nested single policy) >= selection.geo_floor. ``promotion.geographic_gate``:
                          mandatory -> BLOCKED below the floor; advisory -> WARNING with the recorded override note
    VALIDATION_PASS       validation stage COMPLETE (PASS) or WARNING (non-blocking caveats)
    PARITY_PASS           parity stage COMPLETE (every champion export-parity row PASS)
    READY_FOR_EXPORT      every gate above PASS / WARNING
PENDING = the evidence does not exist yet (an upstream stage has not run in this workspace) or is stale (a
validation / parity result produced for another selection / validation fingerprint). The stage status is
COMPLETE (READY_FOR_EXPORT PASS), WARNING (ready with warnings), BLOCKED or INCOMPLETE (something pending).
Nothing here changes the selection: the same champion stays recorded whatever the gates say.
M1.3: ``decisions`` separates technical validity (SELECTION / VALIDATION / PARITY) from internal promotion
(DISCOVERY / GEOGRAPHIC); both are recorded in ``promotion_gates.json`` and the promotion manifest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BLOCKED, COMPLETE, INCOMPLETE, PASSING, WARNING

GATES = ("SELECTION_COMPLETE", "DISCOVERY_ELIGIBLE", "GEOGRAPHIC_GATE_PASS", "VALIDATION_PASS", "PARITY_PASS",
         "READY_FOR_EXPORT")
PASS, PENDING = "PASS", "PENDING"
OUT = ("selection/promotion_gates.csv", "selection/promotion_gates.json")


def _num(x) -> float:
    v = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    return float(v) if pd.notna(v) else np.nan


def _stage_gate(name: str, status: str | None, reasons, warnings) -> dict:
    if status == COMPLETE:
        return {"gate": name, "status": PASS, "evidence": f"stage {status}"}
    if status == WARNING:
        return {"gate": name, "status": WARNING, "evidence": "; ".join(map(str, warnings or [])) or "stage WARNING"}
    if status in (None, INCOMPLETE):
        return {"gate": name, "status": PENDING, "evidence": f"stage {status or 'not run'}: "
                                                             + "; ".join(map(str, reasons or []))}
    return {"gate": name, "status": BLOCKED, "evidence": f"stage {status}: " + "; ".join(map(str, reasons or []))}


def evaluate(sel_manifest: dict, selection: dict, champion: dict, recipe_id_yaml: str | None, val_manifest: dict,
             par_manifest: dict, m1_cfg: dict, worst_floor: float) -> tuple[pd.DataFrame, str, list[str], list[str]]:
    """(gate table, stage status, blocking reasons, warnings) - pure."""
    rows = []
    sst = sel_manifest.get("status")
    rid = selection.get("recipe_id")
    if sst != COMPLETE or not selection.get("champion"):
        g = _stage_gate("SELECTION_COMPLETE", sst if sst != COMPLETE else BLOCKED, sel_manifest.get("reasons") or
                        selection.get("reasons"), None)
    elif rid is None or rid != recipe_id_yaml or (sel_manifest.get("recipe_id") not in (None, rid)):
        g = {"gate": "SELECTION_COMPLETE", "status": BLOCKED, "evidence": f"recipe ids disagree: selection {rid}, "
             f"recipe yaml {recipe_id_yaml}, manifest {sel_manifest.get('recipe_id')}: re-run selection"}
    else:
        g = {"gate": "SELECTION_COMPLETE", "status": PASS, "evidence": f"champion {selection['champion']}, recipe {rid}"}
    rows.append({**g, "mandatory": True})
    have = bool(champion) and g["status"] == PASS
    worst, geo = _num(champion.get("disc_worst")), _num(champion.get("geo_dti"))
    elig = str(champion.get("eligible")).strip().lower() == "true"
    if not have:
        rows.append({"gate": "DISCOVERY_ELIGIBLE", "status": PENDING, "evidence": "no selected champion",
                     "mandatory": True})
    else:
        ok = elig and np.isfinite(worst) and worst >= worst_floor
        rows.append({"gate": "DISCOVERY_ELIGIBLE", "status": PASS if ok else BLOCKED, "mandatory": True,
                     "evidence": f"eligible={elig}, disc_dti={champion.get('disc_dti')}, worst={worst} "
                                 f"(floor {worst_floor}); {champion.get('eligibility_reasons') or ''}".strip()})
    pr = m1_cfg["promotion"]
    floor = float(m1_cfg["selection"]["geo_floor"])
    mand = pr["geographic_gate"] == "mandatory"
    if not have:
        rows.append({"gate": "GEOGRAPHIC_GATE_PASS", "status": PENDING, "evidence": "no selected champion",
                     "mandatory": mand})
    elif np.isfinite(geo) and geo >= floor:
        rows.append({"gate": "GEOGRAPHIC_GATE_PASS", "status": PASS, "mandatory": mand,
                     "evidence": f"geo_dti {geo:.6f} >= floor {floor}"})
    else:
        ev = (f"geo_dti {geo} < floor {floor} (nested single policy). "
              f"{selection.get('n_evaluated_meeting_geo_floor', '?')} of {selection.get('n_evaluated', '?')} "
              "evaluated candidates reach the floor")
        rows.append({"gate": "GEOGRAPHIC_GATE_PASS", "status": BLOCKED if mand else WARNING, "mandatory": mand,
                     "evidence": ev if mand else f"{ev}; ADVISORY override: {pr.get('override_note')}"})
    links = (("VALIDATION_PASS", val_manifest, "selection_fp", sel_manifest),
             ("PARITY_PASS", par_manifest, "validation_fp", val_manifest))
    for name, man, key, up in links:
        g = _stage_gate(name, man.get("status"), man.get("reasons"), man.get("warnings"))
        if g["status"] in (PASS, WARNING) and (man.get("inputs") or {}).get(key) != up.get("fingerprint"):
            g = {"gate": name, "status": PENDING, "evidence": f"stale: produced for {key} "
                 f"{(man.get('inputs') or {}).get(key)}, current {up.get('fingerprint')}: re-run it"}
        rows.append({**g, "mandatory": True})
    st = [r["status"] for r in rows]
    if BLOCKED in st:
        final, status = BLOCKED, BLOCKED
    elif PENDING in st:
        final, status = PENDING, INCOMPLETE
    elif WARNING in st:
        final, status = WARNING, WARNING
    else:
        final, status = PASS, COMPLETE
    rows.append({"gate": "READY_FOR_EXPORT", "status": final, "mandatory": True,
                 "evidence": "every gate PASS" if final == PASS else
                 "; ".join(f"{r['gate']} {r['status']}" for r in rows if r["status"] != PASS)})
    T = pd.DataFrame(rows)[["gate", "status", "mandatory", "evidence"]]
    reasons = [f"{r.gate}: {r.evidence}" for r in T.itertuples() if r.status in (BLOCKED, PENDING)
               and r.gate != "READY_FOR_EXPORT"]
    warns = [f"{r.gate}: {r.evidence}" for r in T.itertuples() if r.status == WARNING and r.gate != "READY_FOR_EXPORT"]
    return T, status, reasons, warns


TECHNICAL = ("SELECTION_COMPLETE", "VALIDATION_PASS", "PARITY_PASS")
INTERNAL = ("DISCOVERY_ELIGIBLE", "GEOGRAPHIC_GATE_PASS")


def decisions(T: pd.DataFrame) -> dict:
    """Two independent verdicts from the gate table (pure). ``technical_validity``: can the recipe be written as a
    file that satisfies the official contract and reproduces the evaluated prediction (selection, validation,
    parity). ``internal_promotion``: does it meet our own model-quality rules (discovery eligibility, geographic
    floor). Only ``exploratory_eligible`` (technical PASS, discovery PASS, geographic gate the ONLY failure) allows the
    opt-in exploratory export; it never makes the candidate the validated champion."""
    st = dict(zip(T.gate, T.status))

    def verdict(names):
        v = [st.get(n, PENDING) for n in names]
        return BLOCKED if BLOCKED in v else PENDING if PENDING in v else WARNING if WARNING in v else PASS

    tech, internal = verdict(TECHNICAL), verdict(INTERNAL)
    return {"technical_validity": tech, "internal_promotion": internal,
            "exploratory_eligible": tech in (PASS, WARNING) and st.get("DISCOVERY_ELIGIBLE") in (PASS, WARNING)
            and st.get("GEOGRAPHIC_GATE_PASS") == BLOCKED,
            "technical_gates": list(TECHNICAL), "internal_gates": list(INTERNAL)}


def promote(sess) -> dict:
    """Stage ``promotion``: re-evaluated on every call from the persisted manifests (cheap; never cached)."""
    import yaml

    from ..p6.full_gpu import load_full_config

    cfg = sess.cfg
    sel = sess.ws.read_json("selection/champion_selection.json") or {}
    C = sess.ws.load_table("selection/champion_candidates.csv")
    crow = C[C.candidate_id == sel.get("champion")].iloc[0].to_dict() if sel.get("champion") and len(C) and \
        "candidate_id" in C and (C.candidate_id == sel.get("champion")).any() else {}
    rp = sess.ws.path("selection/champion_recipe.yaml")
    try:
        ry = yaml.safe_load(rp.read_text(encoding="utf-8")) if rp.exists() else None
    except yaml.YAMLError:
        ry = None
    rid = ry.get("recipe_id") if isinstance(ry, dict) else None
    fg_cfg = getattr(sess.fg, "cfg", None) or load_full_config()
    worst_floor = float(fg_cfg["selection"]["worst_floor"])
    T, status, reasons, warns = evaluate(sess.manifest("selection"), sel, crow, rid, sess.manifest("validation"),
                                         sess.manifest("parity"), cfg, worst_floor)
    sess.ws.save_table(T, "selection/promotion_gates.csv")
    out = {"status": status, "ready_for_export": status in PASSING, **decisions(T), "gates": T.to_dict("records"),
           "reasons": reasons, "warnings": warns, "champion": sel.get("champion"), "recipe_id": sel.get("recipe_id"),
           "geographic_gate": cfg["promotion"]["geographic_gate"],
           "override_note": cfg["promotion"].get("override_note") or None,
           "note": "promotion never changes the selection; a BLOCKED promotion stops the export only"}
    sess.ws.save_json(out, "selection/promotion_gates.json")
    upstream = {s: sess.manifest(s).get("fingerprint") for s in ("selection", "validation", "parity")}
    return sess.record("promotion", status, sess.fp("promotion", upstream, cfg["promotion"], cfg["selection"]),
                       reasons=reasons, warnings=warns, inputs=upstream, outputs=list(OUT),
                       ready_for_export=out["ready_for_export"], recipe_id=sel.get("recipe_id"),
                       **{k: out[k] for k in ("technical_validity", "internal_promotion", "exploratory_eligible")})
