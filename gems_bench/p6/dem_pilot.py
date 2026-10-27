"""W4: 1 m DEM pilot gate (20-30 tiles). Processing is M7's (``m7.dem_morph``: tile-streamed 1 m morphology aggregated
exactly to the 100 m grid, never a whole-DEM array); P6 only decides whether DEM evidence earns a place.

Order: reuse the cached M7 mosaic ``outputs/m7_discovery_first/dem/features/m7_dem_features_100m.tif``; only when it
is absent, run M7 sections 05 (inventory + pilot selection, capped at ``dem.max_tiles``) and 06 (morphology).

Evidence = mean per-band rank of ``dem.evidence`` bands inside covered cells (label-free). Coverage is a mask for
the gate only and is never an input to any score.

Gate (all three):
* trace recall: share of covered known-fault pixels within ``tolerance_m`` of the DEM selection when the selection
  is limited to ``density_km_per_100km2`` of 1-px crest lines over the covered area (>= 0.30);
* new systems: withheld systems of the discovery splits, covered for >= 50 % of their pixels, recovered by the DEM
  selection and NOT by the fallback's held-out map (>= 3);
* alignment: the integer shift that maximises evidence on known traces is within +-``max_abs_shift_px`` cells.
Faults without surface expression are not expected to appear: a low recall is a property of the area, not a bug.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import ndimage


def evidence_map(bands: dict, covered: np.ndarray) -> np.ndarray:
    """Mean per-band rank inside ``covered``; NaN elsewhere (pure)."""
    from .stack import rank_in

    rs = [rank_in(np.asarray(b, np.float32), covered) for b in bands.values()]
    if not rs:
        raise ValueError("no evidence band")
    st = np.stack(rs)
    n = np.isfinite(st).sum(0)
    out = np.where(n > 0, np.nansum(st, 0) / np.maximum(n, 1), np.nan)
    return np.where(covered, out, np.nan).astype(np.float32)


def best_shift(evidence: np.ndarray, known: np.ndarray, covered: np.ndarray, max_shift: int) -> dict:
    """Integer roll (dr, dc) of the evidence maximising its mean on covered known pixels (registration check); an
    evidence layer displaced by (+1, 0) cells returns (-1, 0)."""
    e = np.nan_to_num(evidence, nan=0.0)
    k = np.asarray(known, bool) & np.asarray(covered, bool)
    if not k.any():
        return {"dr": 0, "dc": 0, "gain": np.nan, "n_known": 0}
    rows = []
    for dr in range(-max_shift, max_shift + 1):
        for dc in range(-max_shift, max_shift + 1):
            rows.append((float(np.roll(e, (dr, dc), (0, 1))[k].mean()), dr, dc))
    base = float(e[k].mean())
    v, dr, dc = max(rows)
    return {"dr": int(dr), "dc": int(dc), "gain": v - base, "n_known": int(k.sum())}


def line_budget_px(covered: np.ndarray, km_per_100km2: float, px_km: float = 0.1) -> int:
    """Number of 1-px line cells allowed by a line-density budget over the covered area."""
    area_km2 = float(np.asarray(covered, bool).sum()) * px_km * px_km
    return int(round(km_per_100km2 * area_km2 / 100.0 / px_km))


def select_lines(evidence: np.ndarray, covered: np.ndarray, n_px: int, sigma_px: float = 1.0) -> np.ndarray:
    """Top-``n_px`` crest cells of the evidence inside ``covered``."""
    from ..gems3 import representation as R

    cov = np.asarray(covered, bool)
    s = np.nan_to_num(evidence, nan=0.0).astype(np.float32)
    if n_px <= 0 or not cov.any():
        return np.zeros(cov.shape, bool)
    crest = R.crest_map(s, cov, sigma_px)
    return R.crest_top(s, cov, n_px / max(int(cov.sum()), 1), crest)


def trace_recall(sel: np.ndarray, known: np.ndarray, covered: np.ndarray, tol_m: float, px_m: float = 100.0) -> float:
    k = np.asarray(known, bool) & np.asarray(covered, bool)
    if not k.any():
        return np.nan
    d = ndimage.distance_transform_edt(~np.asarray(sel, bool)) * px_m
    return float((d[k] <= tol_m).mean())


def gate(recall: float, new_systems: int, shift: dict, g: dict) -> tuple[str, list[str]]:
    why = []
    if not (np.isfinite(recall) and recall >= float(g["min_trace_recall"])):
        why.append(f"trace recall {recall} < {g['min_trace_recall']}")
    if new_systems < int(g["min_new_systems"]):
        why.append(f"{new_systems} new systems < {g['min_new_systems']}")
    if max(abs(shift["dr"]), abs(shift["dc"])) > int(g["max_abs_shift_px"]):
        why.append(f"alignment shift {(shift['dr'], shift['dc'])} beyond +-{g['max_abs_shift_px']}")
    return ("PASS" if not why else "STOP"), why


# =============================================================================== Colab side
def mosaic(p6):
    """Cached M7 mosaic, else M7 sections 05 / 06 with the pilot capped at ``dem.max_tiles``."""
    from ..m7.dem_morph import DemMosaic, mosaic_path

    m7 = p6.m9.m7
    p = mosaic_path(m7)
    if not p.exists():
        m7.stage_in([p.relative_to(m7.local).as_posix()])
    if p.exists() and p6.cfg["dem"]["reuse_m7_mosaic"]:
        return DemMosaic(p), "reused cached M7 mosaic"
    n = int(p6.cfg["dem"]["max_tiles"])                        # M7 caps its pilot at max_pilot_tiles / n_pilot_tiles
    m7.cfg["dem"].update(n_pilot_tiles=n, max_pilot_tiles=n, min_pilot_tiles=int(p6.cfg["dem"]["min_tiles"]))
    m7.s05_dem_inventory()
    m7.s06_dem_morphology()
    if not p.exists():
        raise FileNotFoundError("M7 DEM mosaic still missing after sections 05 / 06 (raw tiles not staged?)")
    return DemMosaic(p), f"processed by M7 sections 05 / 06 (<= {p6.cfg['dem']['max_tiles']} tiles)"


def run(p6) -> dict:
    from ..m7.baseline import heldout_map
    from ..m7.protocol import system_hits

    dc = p6.cfg["dem"]
    mos, how = mosaic(p6)
    cov = mos.coverage(float(dc["min_coverage"])) & np.asarray(p6.ctx.data.valid, bool)
    bands = {b: mos.band(b) for b in dc["evidence"] if mos.has(b)}
    ev = evidence_map(bands, cov)
    known = np.asarray(p6.ctx.data.labels, bool)
    sh = best_shift(ev, known, cov, int(dc["max_shift_px"]))
    n_px = line_budget_px(cov, float(dc["density_km_per_100km2"]))
    sel = select_lines(ev, cov, n_px)
    rec = trace_recall(sel, known, cov, float(dc["tolerance_m"]))
    rows = []
    for u in p6.m9.m7.units_of(p6.prim):
        cu = cov[u.sl]
        q = np.where(u.ev & cu, sel[u.sl], False).astype(np.float32)
        hd = system_hits(p6.ctx, u, q, p6.m9.sh)
        qf = heldout_map(p6.view, u.task, p6.cfg["probes"]["champion"], u.zone)
        hf = system_hits(p6.ctx, u, qf, p6.m9.sh) if qf is not None else hd.iloc[0:0]
        from ..m7.protocol import system_ids

        sid = system_ids(p6.ctx, u)
        gr, gc = u.scorer.g_rc
        cshare = pd.Series(cu[gr, gc]).groupby(sid).mean()
        covered_sys = set(cshare[cshare >= 0.5].index) - {-1}
        d_set = set(hd.system_id[hd.recovered]) & covered_sys if len(hd) else set()
        f_set = set(hf.system_id[hf.recovered]) if len(hf) else set()
        rows.append({"task": u.task, "zone": u.zone, "covered_systems": len(covered_sys), "dem_recovered": len(d_set),
                     "dem_only": len(d_set - f_set)})
    S = pd.DataFrame(rows)
    new = int(S.dem_only.sum()) if len(S) else 0
    decision, why = gate(rec, new, sh, dc["gate"])
    out = {"source": how, "bands": list(bands), "covered_km2": float(cov.sum()) * 0.01, "line_px": n_px,
           "trace_recall": rec, "new_systems": new, "shift": sh, "decision": decision, "reasons": why,
           "note": "coverage is a mask only; DEM evidence cannot see faults without surface expression"}
    p6.ws.save_table(S, "systems", "dem")
    p6.ws.save_json(json.loads(json.dumps(out, default=float)), "dem/gate.json")
    p6.ws.save_raster({"dem_evidence": ev, "dem_lines": sel.astype(np.float32), "covered": cov.astype(np.float32)},
                      "dem/dem_evidence_100m.tif", p6.ctx.data.grid)
    p6.say(f"W4 DEM gate: {decision} {why}")
    return out
