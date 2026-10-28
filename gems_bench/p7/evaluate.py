"""P7 evaluation through the EXISTING official protocol (no custom metric for any headline number).

Held-out maps (``heldout``): every trained unit predicts only its own evaluation unit (geo_f<k> -> geo:<k>, disc_s<d>
-> disc_official:<d>), stored as float16 crops ``predictions/units/<task>_<zone>.npz`` with the unit's M7 protocol
key, the checkpoint sha256 and ``labels_stored = False``; stitched GeoTIFFs: ``predictions/oof_<head>.tif`` (geo
folds) and ``predictions/disc_<head>.tif`` (one band per split).

Scoring (``grid``): each map is ranked inside the unit's evaluation mask (``rank01``) and scored by
``p6.boost.Boost.eval_score`` (``m7.protocol.score_row``: official TP / FP / FN on the unit, eligible budget domain,
known-fault masking, system hits) for every (rep, frac) cell of the boost grid. References (B0, B1, fallback fusion,
W2) are scored by the same calls on the same units (``reference_grid``), so every comparison is paired under
identical masks and area budgets.

Summaries (``summarize``): ``p6.export_parity.evaluate_policy`` per candidate (per outer zone ONE cell chosen on the
other zones, applied to that zone in both tasks; objective ``evaluation.objective``). Reported quantities are kept
apart: nested outer DTI (the estimate), matched-cell deltas (diagnostic, not a selection), inner checkpoint scores
(training logs only), the public leaderboard (never computed here).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from ..p6 import boost as B
from ..p6 import export_parity as EP

COMP = ["tp_w", "fp_w", "fn_w", "n_truth"]
CELL = ["task", "zone", "candidate", "blend", "rep", "frac"]
NOISE = 0.005                                   # three zones: smaller nested differences are not interpretable


class ProtocolMismatch(RuntimeError):
    """A saved prediction was made for a different evaluation unit (mask / truth / budget changed)."""


# =============================================================================== pure helpers
def cand_names(exp: str, heads: list[str]) -> dict:
    """{candidate name: head} of one experiment's rank maps."""
    out = {f"p7_{exp}": "seg"}
    if "center" in heads:
        out[f"p7_{exp}_center"] = "center"
    return out


def fill_blend(G: pd.DataFrame) -> pd.DataFrame:
    g = G.copy()
    g["blend"] = pd.to_numeric(g.blend, errors="coerce").fillna(-1.0).astype(float)
    g["frac"] = g.frac.astype(float)
    return g


def system_recall(G: pd.DataFrame, P: pd.DataFrame, prim: str) -> dict:
    """Pooled system recall of a policy's held-out disc choices (n_recovered / n_systems of the held rows)."""
    if not len(P) or "n_systems" not in G:
        return {"systems_recovered": np.nan, "systems_total": np.nan, "system_recall": np.nan}
    g = fill_blend(G[G.task == prim])
    p = fill_blend(P[(P.task == prim) & P.held_row_found.astype(bool)])
    m = p[CELL].merge(g[CELL + ["n_systems", "n_recovered"]], on=CELL, how="left")
    rec, tot = float(m.n_recovered.sum()), float(m.n_systems.sum())
    return {"systems_recovered": rec, "systems_total": tot, "system_recall": rec / tot if tot else np.nan}


def policy_summary(G: pd.DataFrame, cand: str, ecfg: dict, prim: str) -> tuple[dict, pd.DataFrame]:
    ev = ecfg
    s, P = EP.evaluate_policy(G, cand, {"candidates": [cand]}, ev["objective"], prim, float(ev["geo_floor"]),
                              list(ev["tasks"]))
    for task in ev["tasks"]:
        pre = "disc" if task == prim else "geo"
        p = P[(P.task == task) & P.held_row_found.astype(bool)] if len(P) else P
        s[f"{pre}_tp_w"], s[f"{pre}_fn_w"] = float(p.tp_w.sum()), float(p.fn_w.sum())
        s[f"{pre}_coverage_w"] = float(p.tp_w.sum() / max(p.n_truth.sum(), 1e-9)) if len(p) else np.nan
        s[f"{pre}_length_km"] = float(p.length_km.sum()) if len(p) and "length_km" in p else np.nan
        s[f"{pre}_short_share"] = float(p.short_share.mean()) if len(p) and "short_share" in p else np.nan
        s[f"{pre}_segments"] = float(p.n_segments.sum()) if len(p) and "n_segments" in p else np.nan
    s.update(system_recall(G, P, prim))
    return s, P


def per_zone_deltas(P: pd.DataFrame, R: pd.DataFrame, name: str, ref: str) -> pd.DataFrame:
    """Nested per-zone DTI of a candidate against a reference (same outer zones, same tasks)."""
    a = P[["task", "zone", "dti"]].rename(columns={"dti": "dti_cand"})
    b = R[["task", "zone", "dti"]].rename(columns={"dti": "dti_ref"})
    m = a.merge(b, on=["task", "zone"], how="outer")
    return m.assign(candidate=name, reference=ref, delta=m.dti_cand - m.dti_ref)


def matched_deltas(G: pd.DataFrame, cand: str, ref: str, ref_blend: float | None = None) -> pd.DataFrame:
    """Same (task, zone, rep, frac) cells: DTI of the candidate minus the reference (diagnostic only)."""
    g = fill_blend(G)
    g["dti"] = B.dti(g.tp_w, g.fp_w, g.fn_w)
    a = g[(g.candidate == cand) & np.isclose(g.blend, 1.0)]
    b = g[g.candidate == ref]
    if ref_blend is not None:
        b = b[np.isclose(b.blend, ref_blend)]
    k = ["task", "zone", "rep", "frac"]
    m = a[k + ["dti"]].merge(b[k + ["dti"]], on=k, suffixes=("_cand", "_ref"))
    return m.assign(candidate=cand, reference=ref, delta=m.dti_cand - m.dti_ref)


def recall_table(G: pd.DataFrame, cands: list[str], budgets: list[float], prim: str,
                 refs: dict[str, float | None]) -> pd.DataFrame:
    """Candidate recall at area budgets (disc units, blend-1 / reference rows) + systems new vs each reference at the
    same (zone, rep, frac). ``refs``: {reference candidate: blend to match (None = its NaN / -1 rows)}."""
    g = fill_blend(G[G.task == prim])
    rows = []
    for rep in sorted(g.rep.unique()):
        for f in budgets:
            cell = g[(g.rep == rep) & np.isclose(g.frac, float(f))]
            for c in cands + list(refs):
                r = cell[cell.candidate == c]
                r = r[np.isclose(r.blend, 1.0)] if c in cands else (r[np.isclose(r.blend, refs[c])] if refs[c] is not
                                                                      None else r[np.isclose(r.blend, -1.0)])
                if not len(r):
                    continue
                row = {"candidate": c, "rep": rep, "frac": float(f), "n_zones": int(r.zone.nunique()),
                       "systems_recovered": int(r.n_recovered.sum()), "systems_total": int(r.n_systems.sum()),
                       "recall": float(r.n_recovered.sum() / max(r.n_systems.sum(), 1)),
                       "coverage_w": float(r.tp_w.sum() / max(r.n_truth.sum(), 1e-9)),
                       "area_frac_eligible": float(r.area_frac_eligible.mean())}
                if c in cands:
                    for ref, bl in refs.items():
                        rr = cell[cell.candidate == ref]
                        rr = rr[np.isclose(rr.blend, bl)] if bl is not None else rr[np.isclose(rr.blend, -1.0)]
                        new = 0
                        for z in r.zone.unique():
                            a = B.system_set(r[r.zone == z].systems.iloc[0])
                            bz = rr[rr.zone == z]
                            bset = B.system_set(bz.systems.iloc[0]) if len(bz) else set()
                            new += len(a - bset)
                        row[f"new_vs_{ref}"] = new
                rows.append(row)
    return pd.DataFrame(rows)


def hypothesis_verdict(cand: dict | None, ctrl: dict | None, *, min_gain: float = NOISE, max_worst_drop: float = 0.01,
                       min_zone_wins: int = 2, zone_deltas: pd.DataFrame | None = None, prim_pre: str = "disc") -> dict:
    """SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE / INCOMPLETE for one rung against its control (nested disc DTI)."""
    if not cand or not ctrl or not np.isfinite(cand.get(f"{prim_pre}_dti", np.nan)) or \
            not np.isfinite(ctrl.get(f"{prim_pre}_dti", np.nan)):
        return {"verdict": "INCOMPLETE", "reason": "candidate or control has no complete nested score"}
    d = float(cand[f"{prim_pre}_dti"] - ctrl[f"{prim_pre}_dti"])
    dw = float(cand[f"{prim_pre}_worst"] - ctrl[f"{prim_pre}_worst"])
    wins = None
    if zone_deltas is not None and len(zone_deltas):
        z = zone_deltas[zone_deltas.task.str.startswith(prim_pre[:4])] if prim_pre == "disc" else zone_deltas
        wins = int((z.delta > 0).sum())
    out = {"delta_dti": d, "delta_worst": dw, "zone_wins": wins}
    if abs(d) < min_gain:
        return {**out, "verdict": "INCONCLUSIVE", "reason": f"|delta| {abs(d):.4f} < {min_gain} (3-zone noise)"}
    if d > 0 and dw >= -max_worst_drop and (wins is None or wins >= min_zone_wins):
        return {**out, "verdict": "SUPPORTED", "reason": ""}
    why = [] if d > 0 else ["no gain"]
    if dw < -max_worst_drop:
        why.append(f"worst zone drops {dw:.4f}")
    if wins is not None and wins < min_zone_wins:
        why.append(f"wins {wins} zones < {min_zone_wins}")
    return {**out, "verdict": "NOT_SUPPORTED", "reason": "; ".join(why)}


def promotion_verdict(cand: dict, w2: dict | None, pc: dict, new_systems: int | None, checks: dict) -> dict:
    """PROMOTE / AUXILIARY / REJECT under section-16 rules. ``checks``: booleans for leakage, schema, parity,
    deployable policy and export (False / missing = not satisfied)."""
    reasons = []
    disc, geo, worst = cand.get("disc_dti", np.nan), cand.get("geo_dti", np.nan), cand.get("disc_worst", np.nan)
    if not np.isfinite(disc):
        return {"decision": "REJECT", "reasons": ["no complete nested discovery score"], "auxiliary": False}
    w2d = (w2 or {}).get("disc_dti", np.nan)
    if not np.isfinite(w2d):
        reasons.append("W2 reference not recomputed on the same zones")
    elif disc < w2d + float(pc["min_gain_over_w2"]):
        reasons.append(f"disc {disc:.4f} < W2 {w2d:.4f} + {pc['min_gain_over_w2']}")
    if not np.isfinite(geo) or geo < float(pc["geo_min"]):
        reasons.append(f"geo {geo:.4f} < {pc['geo_min']}")
    w2w = (w2 or {}).get("disc_worst", np.nan)
    if np.isfinite(w2w) and np.isfinite(worst) and worst < w2w - float(pc["max_worst_drop"]):
        reasons.append(f"worst zone {worst:.4f} < W2 worst {w2w:.4f} - {pc['max_worst_drop']}")
    for k in ("no_leakage", "schema_compatible", "parity", "deployable_policy", "export_valid"):
        if not checks.get(k):
            reasons.append(f"check {k} not satisfied")
    aux = new_systems is not None and new_systems >= int(pc["complementarity"]["min_new_systems"])
    if not reasons:
        dec = "PROMOTE"
    elif aux and checks.get("no_leakage"):
        dec = "AUXILIARY"
    else:
        dec = "REJECT"
    return {"decision": dec, "reasons": reasons, "auxiliary": bool(aux), "new_systems": new_systems,
            "milestone_0_30": bool(disc > float(pc["milestone_disc"])),
            "target_0_40": bool(disc >= float(pc["target_disc"]))}


# =============================================================================== Colab side: held-out predictions
def pred_rel(p7, exp_id: str, u) -> str:
    return f"{p7.exp_sub(exp_id, 'predictions')}/units/{u.task}_{u.zone}.npz"


def load_pred(p7, exp_id: str, u) -> dict | None:
    p = p7.ws.root / pred_rel(p7, exp_id, u)
    if not p.exists():
        return None
    z = np.load(p)
    pk = str(z["protocol_key"]) if "protocol_key" in z else None
    if pk != p7.m9.pkey(u):
        raise ProtocolMismatch(f"{p.name}: protocol key {pk} != {p7.m9.pkey(u)} (masks / truth / budget changed)")
    if bool(z["labels_stored"]) if "labels_stored" in z else True:
        raise ProtocolMismatch(f"{p.name}: labels_stored is not False")
    return {k: z[k].astype(np.float32) for k in z.files if k not in ("protocol_key", "checkpoint_sha", "labels_stored",
                                                                       "unit", "task", "zone")}


def heldout(p7, exp_id: str, ecfg: dict, units: list, rt: dict) -> pd.DataFrame:
    """Predict every trained unit's evaluation mask (reuses existing files); returns an index table."""
    from .data import eval_key
    from .infer import predict_mask
    from .train import load_unit

    inp = p7.inputs()
    P = int(ecfg["data"]["patch_px"])
    ic = p7.cfg["inference"]
    rows = []
    for tu in units:
        u = p7.m9.unit(eval_key(p7, tu))
        if u is None:
            continue
        rel = pred_rel(p7, exp_id, u)
        if (p7.ws.root / rel).exists():
            rows.append({"unit": tu.name, "eval_unit": u.key, "status": "reused"})
            continue
        lu = load_unit(p7, exp_id, tu.name, rt["device"])
        if lu is None:
            rows.append({"unit": tu.name, "eval_unit": u.key, "status": "model missing"})
            continue
        model, sc, esc, obj = lu
        full = np.zeros(p7.ctx.data.shape, bool)
        full[u.sl] = u.ev
        t0 = time.perf_counter()
        out = predict_mask([(model, sc, esc)], inp["cube"], inp["extra"], full, P=P,
                           stride=max(int(P * float(ic["stride_frac"])), 1), device=rt["device"],
                           amp=bool(ecfg["training"]["amp"]), amp_dtype=rt.get("amp_dtype"), batch=int(rt["infer_batch"]),
                           rot180=bool(ic["tta_rot180"]),
                           mc_samples=int(ecfg["model"]["uncertainty"]["samples"])
                           if ecfg["model"]["uncertainty"]["method"] == "mc_dropout" else 0)
        sec = round(time.perf_counter() - t0, 1)
        arr = {h: np.nan_to_num(a[u.sl]).astype(np.float16) for h, a in out.items()}
        p7.ws.save_npz({**arr, "protocol_key": np.array(p7.m9.pkey(u)),
                        "checkpoint_sha": np.array(str(obj.get("spec_hash"))), "labels_stored": np.array(False),
                        "unit": np.array(tu.name), "task": np.array(u.task), "zone": np.array(int(u.zone))}, rel)
        rows.append({"unit": tu.name, "eval_unit": u.key, "status": "predicted", "seconds": sec,
                     "mass_mean_ev": float(np.nanmean(out["seg"][full])), "heads": ";".join(sorted(out))})
        del model
    T = pd.DataFrame(rows)
    p7.ws.save_table(T, "heldout_index", p7.exp_sub(exp_id, "predictions"))
    stitch(p7, exp_id)
    return T


def stitch(p7, exp_id: str) -> None:
    """GeoTIFFs of every head: OOF over geo folds, one band per discovery split."""
    H, W = p7.ctx.data.shape
    grid = p7.ctx.data.grid
    oof, disc = {}, {}
    for u in p7.m9.m7.units_of("geo"):
        p = load_pred(p7, exp_id, u)
        for h, a in (p or {}).items():
            o = oof.setdefault(h, np.full((H, W), np.nan, np.float32))
            sub = o[u.sl]
            sub[u.ev] = a[u.ev]
    for u in p7.m9.m7.units_of(p7.prim):
        p = load_pred(p7, exp_id, u)
        for h, a in (p or {}).items():
            b = np.full((H, W), np.nan, np.float32)
            b[u.sl] = np.where(u.ev, a, np.nan)
            disc.setdefault(h, {})[f"disc{u.zone}"] = b
    sub = p7.exp_sub(exp_id, "predictions")
    for h, a in oof.items():
        p7.ws.save_raster({f"oof_{h}": a}, f"{sub}/oof_{h}.tif", grid)
    for h, bands in disc.items():
        p7.ws.save_raster(bands, f"{sub}/disc_{h}.tif", grid)


# =============================================================================== Colab side: grids + summaries
def unit_ranks(p7, exp_id: str, exp: str, u, heads: list[str]) -> dict:
    from ..gems3 import representation as R

    p = load_pred(p7, exp_id, u)
    if p is None:
        return {}
    return {c: R.rank01(np.nan_to_num(p[h], nan=0.0), u.ev) for c, h in cand_names(exp, heads).items() if h in p}


def grid(p7, exp_id: str, exp: str) -> pd.DataFrame:
    """Official grid rows of one experiment (P7 alone per head + P7 fault blended with the fallback fusion)."""
    bst, ev = p7.bst, p7.cfg["evaluation"]
    heads = [h.replace("fault", "seg") for h in ev["heads_evaluated"]]
    rows = []
    for u in bst.units():
        rk = unit_ranks(p7, exp_id, exp, u, heads)
        if not rk:
            continue
        for c, s in rk.items():
            rows += bst.eval_score(u, s, candidate=c, blend=1.0, experiment_id=exp_id)
        F = bst.fusion(u)
        for lam in ev["fusion_blends"]:
            s = ((1.0 - float(lam)) * F + float(lam) * rk[f"p7_{exp}"]).astype(np.float32)
            rows += bst.eval_score(u, s, candidate=f"p7_{exp}+fusion", blend=float(lam), experiment_id=exp_id)
        u.release()
    G = pd.DataFrame(rows)
    p7.ws.save_table(G, "grid", p7.exp_sub(exp_id, "metrics"))
    return G


def reference_grid(p7) -> pd.DataFrame:
    """B0, B1, fallback fusion (blend NaN) and W2 (eval_variant: W2's own fusion blends) on every unit; cached by the
    protocol keys + frozen W2 input hashes."""
    bst = p7.bst
    fp = p7.fp("reference_grid", bst.protocol_keys(), bst.w2_inputs(), bst.cfg["grid"])
    sub = "comparisons"
    meta = p7.read_json(f"{sub}/reference_grid_manifest.json") or {}
    if meta.get("fingerprint") == fp:
        G = p7.ws.load_table("reference_grid", sub)
        if len(G):
            return G
    rows = []
    for u in bst.units():
        srcs = bst.unit_sources(u)
        for k in ("cnn_b0", "cnn_b1", "fallback_fusion"):
            rows += bst.eval_score(u, srcs[k], candidate=k, blend=np.nan)
        rows += bst.eval_variant(u, bst.w2_map(u), "w2")
        u.release()
    G = pd.DataFrame(rows)
    p7.ws.save_table(G, "reference_grid", sub)
    p7.ws.save_json({"fingerprint": fp, "written": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_rows": int(len(G))},
                    f"{sub}/reference_grid_manifest.json")
    return G


def summarize(p7, G: pd.DataFrame, cands: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ev, prim = p7.cfg["evaluation"], p7.prim
    S, Ps = [], []
    for c in cands:
        if c not in set(G.candidate):
            S.append({"policy": c, "status": "ABSENT"})
            continue
        s, P = policy_summary(G, c, ev, prim)
        complete = np.isfinite(s.get("disc_dti", np.nan)) and np.isfinite(s.get("geo_dti", np.nan))
        S.append({**{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in s.items()},
                  "status": "COMPLETE" if complete else "INCOMPLETE"})
        Ps.append(P)
    return pd.DataFrame(S), (pd.concat(Ps, ignore_index=True) if Ps else pd.DataFrame())


def unit_mass(p7, exp_id: str, exp: str) -> pd.DataFrame:
    """Prediction mass / sharpness per unit (raw fault probability inside the evaluation mask)."""
    rows = []
    for u in p7.bst.units():
        p = load_pred(p7, exp_id, u)
        if p is None:
            continue
        s = p["seg"][u.ev]
        rows.append({"unit": u.key, "mass_sum": float(s.sum()), "mass_mean": float(s.mean()),
                     "share_p_gt_0_5": float((s > 0.5).mean()), "p99": float(np.percentile(s, 99))})
    return pd.DataFrame(rows)
