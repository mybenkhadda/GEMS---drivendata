"""W6: transparent final selection among candidates that passed their own gate (no blending sweep).

Order (``select.order``): nested pooled discovery DTI -> worst-zone discovery DTI -> geographic DTI -> unique withheld
systems -> public score (tie-breaker only). At each criterion every candidate within ``tolerance`` of the best stays
in; the next criterion decides among them. The original fallback is always a candidate and always eligible, so the
procedure can only replace it with a gate-passing method.

Candidate rows come from persisted workstream summaries (``*/candidate.json``), so this section runs from files alone.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

CRITERIA = ("disc_dti", "disc_worst", "geo_dti", "unique_systems", "public_score")
ABS_TOL = {"unique_systems": 0.5}                 # counts: a tie is an equal count


def gate(c: dict, ref: dict, rules: dict) -> tuple[bool, list[str]]:
    """Generic promotion gate of one candidate against the reference (fallback). Missing numbers fail."""
    why = []
    d, w, g = (c.get(k) for k in ("disc_dti", "disc_worst", "geo_dti"))
    if d is None or not np.isfinite(d):
        return False, ["no discovery DTI"]
    need = max(float(rules.get("disc_min", -np.inf)), float(ref.get("disc_dti", -np.inf)) +
               float(rules.get("min_gain_over_ref", 0.0)))
    if d < need:
        why.append(f"disc {d:.4f} < {need:.4f}")
    wmin = max(float(rules.get("worst_min", -np.inf)), float(ref.get("disc_worst", -np.inf)) -
               float(rules.get("max_worst_drop", np.inf)))
    if w is None or not np.isfinite(w) or w < wmin:
        why.append(f"worst zone {w} < {wmin:.4f}")
    if "max_geo_drop" in rules and ref.get("geo_dti") is not None:
        gmin = float(ref["geo_dti"]) - float(rules["max_geo_drop"])
        if g is None or not np.isfinite(g) or g < gmin:
            why.append(f"geo {g} < {gmin:.4f}")
    return not why, why


def reference_from_selection(sel: dict, prim: str, champion: str) -> dict:
    """Fallback reference numbers recomputed by M9 section 01 (nested; same protocol as every candidate)."""
    out = {}
    for task, pre in ((prim, "disc"), ("geo", "geo")):
        s = (sel.get(task) or {}).get(champion) or {}
        if s.get("dti_crossfit") is not None:
            out[f"{pre}_dti"] = float(s["dti_crossfit"])
            out[f"{pre}_worst"] = float(s.get("fold_min", np.nan))
    return out


def fallback_reference(p6, sel: dict | None = None) -> dict:
    """Recomputed fallback numbers when the M9 selection has them, else the recorded values (labelled)."""
    rec = p6.cfg["recorded"]["fallback"]
    ref = {"disc_dti": rec["disc_official_dti"], "disc_worst": rec["disc_worst"], "geo_dti": rec["geo_dti"],
           "geo_worst": rec["geo_worst"], "source": "recorded (not recomputed)"}
    got = reference_from_selection(p6.selection() if sel is None else sel, p6.prim, p6.cfg["probes"]["champion"])
    if "disc_dti" in got:
        ref = {**ref, **got, "source": "M9 section 01 baseline_selection.json (recomputed)"}
    p6.ws.save_json(ref, "selection/fallback_reference.json")
    return ref


def choose(table: pd.DataFrame, order=CRITERIA, tolerance: float = 0.003) -> tuple[pd.Series, pd.DataFrame]:
    """(chosen row, trace of the elimination) among ``eligible`` rows."""
    t = table[table.eligible.astype(bool)].copy()
    if not len(t):
        raise ValueError("no eligible candidate (the fallback must always be one)")
    trace = []
    for c in order:
        if c not in t or len(t) == 1:
            continue
        v = pd.to_numeric(t[c], errors="coerce")
        if v.notna().sum() == 0:
            continue
        best = float(v.max())
        tol = ABS_TOL.get(c, tolerance)
        keep = v.fillna(-np.inf) >= best - tol
        trace.append({"criterion": c, "best": best, "tolerance": tol, "kept": ";".join(t.loc[keep, "name"]),
                      "dropped": ";".join(t.loc[~keep, "name"])})
        t = t[keep.to_numpy()]
    if len(t) > 1:                                     # still tied: prefer the fallback (fewest moving parts)
        fb = t[t.role == "fallback"]
        t = fb if len(fb) else t.iloc[:1]
    return t.iloc[0], pd.DataFrame(trace)


def candidates(p6) -> pd.DataFrame:
    """Fallback + every ``<workstream>/candidate.json`` written by W1 / W2 / W3 / W4 / W5."""
    ref_p = p6.ws.root / "selection" / "fallback_reference.json"
    try:
        ref = fallback_reference(p6)
    except FileNotFoundError:
        ref = json.loads(ref_p.read_text(encoding="utf-8")) if ref_p.exists() else fallback_reference(p6, {})
    champ = p6.cfg["probes"]["champion"]
    rows = [{"name": "fallback_fusion_thin", "role": "fallback", "eligible": True, "unique_systems": 0, **ref,
             "model_source": f"s01_{champ}", "submission_file": p6.cfg["probes"]["fallback_file"],
             "gate_reasons": "reference"}]
    for sub in ("seeds", "stack", "throughput", "dem", "completion"):
        p = p6.ws.root / sub / "candidate.json"
        if p.exists():
            c = json.loads(p.read_text(encoding="utf-8"))
            rows.append({**c, "role": "candidate", "eligible": c.get("gate") == "PROMOTE",
                         "gate_reasons": ";".join(c.get("gate_reasons") or [])})
    from .probes import read_log

    log = read_log(p6)
    if len(log):
        sc = log.dropna(subset=["public_score"]).groupby("model_source").public_score.last()
        for r in rows:
            r["public_score"] = float(sc.get(r.get("model_source", ""), np.nan))
    return pd.DataFrame(rows)


def segments(mask: np.ndarray, min_px: int = 3) -> list[np.ndarray]:
    """Ordered pixel paths of the 1-px lines of a submission (junctions split branches; pure)."""
    from ..p3.lineaments import split_branches, trace
    from ..postprocess.topology import skeleton

    lab, _, _ = split_branches(skeleton(np.asarray(mask, bool)), min_px)
    if not lab.any():
        return []
    objs = _objects(lab)
    return [trace(np.argwhere(lab[sl] == i) + [sl[0].start, sl[1].start]) for i, sl in objs]


def _objects(lab: np.ndarray) -> list[tuple[int, tuple]]:
    from scipy import ndimage

    return [(i + 1, sl) for i, sl in enumerate(ndimage.find_objects(lab)) if sl is not None]


def segment_table(paths: list[np.ndarray], layers: dict, source: str, pixel_km: float = 0.1) -> pd.DataFrame:
    """Per segment: length, mean of every supporting layer, a heuristic confidence (mean of ``confidence`` when
    present) and the provenance of the evidence (pure)."""
    rows = []
    for i, p in enumerate(paths):
        r, c = p[:, 0], p[:, 1]
        step = np.hypot(np.diff(r), np.diff(c)).sum() if len(p) > 1 else 1.0
        row = {"segment_id": i, "length_km": float(step * pixel_km), "n_px": int(len(p)), "source_model": source}
        for k, a in layers.items():
            row[f"mean_{k}"] = float(np.nanmean(np.asarray(a)[r, c]))
        row["segment_confidence"] = row.get("mean_confidence", np.nan)
        row["evidence"] = ";".join(layers)
        row["confidence_kind"] = "heuristic rank in [0, 1]; NOT a calibrated probability"
        rows.append(row)
    return pd.DataFrame(rows)


def export_lineaments(p6, decision: dict) -> str | None:
    """GPKG of the chosen submission's lines with supporting B0 / B1 / fused (/ stack) ranks."""
    import rasterio

    from ..gems3 import representation as R

    name = decision.get("submission")
    if not name:
        return None
    p = p6.m9_path(f"submissions/{name}") if decision.get("role") == "fallback" else p6.ws.root / "submissions" / name
    if not p.exists():
        p6.say(f"W6: {p} missing: no lineament export")
        return None
    with rasterio.open(p) as src:
        q = np.nan_to_num(src.read(1), nan=0.0) > 0
    valid = np.asarray(p6.ctx.data.valid, bool)
    maps, _ = p6.view.full_maps()
    r0, r1 = R.rank01(maps[0], valid), R.rank01(maps[1], valid)
    w = float(p6.selection()[p6.prim][p6.cfg["probes"]["champion"]]["params_all"]["w"])
    layers = {"cnn_b0_rank": r0, "cnn_b1_rank": r1, "confidence": R.fuse(r0, r1, w)}
    sp = p6.ws.root / "stack" / "maps" / "full_score.tif"
    if sp.exists():
        with rasterio.open(sp) as src:
            layers["stack_rank"] = R.rank01(np.nan_to_num(src.read(1), nan=0.0), valid)
            if decision.get("chosen") == "p6_stack":
                layers["confidence"] = layers["stack_rank"]
    paths = segments(q)
    T = segment_table(paths, layers, str(decision.get("chosen")))
    p6.ws.save_table(T, "final_lineaments_attributes", "vectors")
    out = p6.ws.save_lines(paths, T, "vectors/final_lineaments.gpkg", p6.ctx.data.grid, "lineaments")
    return str(out)


def run(p6) -> dict:
    t = candidates(p6)
    sc = p6.cfg["select"]
    row, trace = choose(t, sc["order"], float(sc["tolerance"]))
    p6.ws.save_table(t, "candidates", "selection")
    p6.ws.save_table(trace, "elimination_trace", "selection")
    out = {"chosen": row.get("name"), "role": row.get("role"), "submission": row.get("submission_file"),
           "criteria": list(sc["order"]), "tolerance": float(sc["tolerance"]),
           "note": "public score is a tie-breaker only; heuristic scores are not calibrated probabilities"}
    out["lineaments"] = export_lineaments(p6, out)
    p6.ws.save_json(json.loads(json.dumps(out, default=str)), "selection/decision.json")
    return out
