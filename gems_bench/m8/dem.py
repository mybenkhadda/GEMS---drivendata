"""Section 09: DEM pilot and incremental value (never blocks another section).

Reuse first: M7's eight-tile pilot (``dem/pilot/pilot_tiles.csv``, storage estimate ~2.61 GiB) and, when present, M7's
100 m DEM-morphology mosaic are read from the M7 outputs (read-only). ``dem_pilot.stream = True`` processes the pilot
tiles here with M7's tile-streaming code (``m7.dem_morph``: 1 m -> 2 m work resolution blocks with halo ->
multidirectional hillshade, slope / curvature, openness, sky-view, local relief, scarp strength, directional
discontinuity, multiscale scarp persistence -> sub-cell maxima / upper quartiles aggregated to 100 m), writing under
``dem/`` of M8. Downloading still needs ``shared.dem.download = True``. The full DEM inventory is never downloaded.

Comparison on IDENTICAL DEM-covered footprints (units restricted to covered pixels: a separate protocol, never pooled
with full-zone numbers) at the SAME candidate density (label-free structural order)::

    geophysics            section-06 pool, candidates with a geophysical family in their provenance
    dem_only              crests of the DEM evidence layers
    geophysics_dem        union (fused)
    cnn_geophysics        geophysical + B0 / B1 / disagreement families
    cnn_geophysics_dem    union with DEM (fused)

Gate 4: enough withheld systems inside the pilot, DEM-including sets cover >= ``additional_systems`` systems that the
same set without DEM misses, and the incremental system recall >= ``incremental_system_recall``. Download volume,
disk usage, processing time and coverage are logged with the projected full-coverage cost (M7 estimator).
"""
from __future__ import annotations

import json
import shutil
import time

import numpy as np
import pandas as pd

from ..gems3.candidates import CandidateSet


class DemUnavailable(RuntimeError):
    pass


def pilot(m8) -> tuple[pd.DataFrame, str]:
    sel = m8.ws.load_table("pilot_tiles", "dem/pilot")
    if len(sel):
        return sel, "m8"
    if m8.cfg["dem_pilot"]["reuse_m7"]:
        sel = m8.m7.ws.load_table("pilot_tiles", "dem/pilot")
        if len(sel):
            inv = m8.m7.ws.load_table("inventory", "dem/inventory")
            if len(inv):                                   # the M7 cost estimator reads dem/inventory/inventory
                m8.ws.save_table(inv, "inventory", "dem/inventory")
            for rel in ("dem/pilot/storage_estimate.json", "dem/pilot/dem_cost_estimate.json"):
                j = m8.m7_json(rel)
                if j is not None and m8.read_json(rel) is None:
                    m8.ws.save_json(j, rel)
            return sel, "m7 (read-only)"
    from ..m7 import dem_pilot as DP

    out = DP.run(m8.view)
    if out["status"] == "NO_INVENTORY":
        raise DemUnavailable(out["reason"])
    m8.ws.save_table(out["inventory"], "inventory", "dem/pilot")
    m8.ws.save_json({"status": out["status"], "provenance": out["provenance"], **out["storage"]},
                    "dem/pilot/storage_estimate.json")
    return out["pilot"], "m8 (selected with M7 code)"


def stream(m8, sel: pd.DataFrame) -> dict:
    from ..m7 import dem_morph as DM
    from ..m7 import dem_pilot as DP

    log = {}
    before = shutil.disk_usage(str(m8.local))
    t0 = time.perf_counter()
    if m8.sh["dem"]["download"]:
        dl = DP.download(m8.view, sel)
        m8.ws.save_table(dl, "download_log", "dem/pilot")
        log["download_rows"] = int(len(dl))
    out = DM.run(m8.view, sel)
    m8.ws.save_table(out["timing"], "tile_timing", "dem/pilot")
    m8.ws.save_json(out["estimate"], "dem/pilot/dem_cost_estimate.json")
    after = shutil.disk_usage(str(m8.local))
    log.update(seconds=round(time.perf_counter() - t0, 1), disk_used_delta_gib=(after.used - before.used) / 2**30,
               disk_free_gib=after.free / 2**30, mosaic=str(out["mosaic"]) if out["mosaic"] else None,
               tiles_processed=int((out["timing"].get("status", pd.Series(dtype=str)) == "processed").sum()))
    if out["mosaic"]:
        m8.ws._track(out["mosaic"])
    m8.forget_dem()
    return log


def _subset_by_family(cs: CandidateSet, names: set) -> CandidateSet:
    if cs is None or not len(cs) or "families" not in cs.table:
        return CandidateSet(pd.DataFrame(), [])
    idx = [i for i, f in enumerate(cs.table.families) if set(str(f).split(";")) & names]
    return cs.subset(idx)


def _union(a: CandidateSet, b: CandidateSet, shape, gc: dict) -> CandidateSet:
    from ..m7.generator import fuse, structural_score

    parts = [x for x in (a, b) if x is not None and len(x)]
    if not parts:
        return CandidateSet(pd.DataFrame(), [])
    t = pd.concat([x.table for x in parts], ignore_index=True)
    for c in ("uid", "source", "modality"):
        if c not in t:
            t[c] = ""
    if "scale_m" not in t:
        t["scale_m"] = np.nan
    t["scale_m"] = pd.to_numeric(t.scale_m, errors="coerce")
    raw = CandidateSet(t, [p for x in parts for p in x.paths])
    cs = fuse(raw, shape, gc)
    if len(cs):
        cs.table["length_km"] = [len(p) * 0.112 for p in cs.paths]
        cs.table["structural_score"] = structural_score(cs.table)
    return cs


def increment(m8, mos) -> dict:
    from ..gems3.oracle import coverage
    from ..m7 import dem_value as DV
    from ..m7.coverage import _ocfg
    from ..m7.generator import clip_to_unit
    from ..m7.protocol import restrict_unit, system_hits

    gc = m8.cfg["generation"]
    dp = m8.cfg["dem_pilot"]
    oo = _ocfg(m8.sh)
    covered = mos.coverage(m8.sh["dem"]["min_coverage"]) & np.asarray(m8.ctx.data.valid, bool)
    dem_full, dlog = DV.dem_candidates(m8.view, mos, covered)
    m8.set_dem_candidates(dem_full)
    geo_names = {s["name"] for s in gc["families"]}
    cnn_names = {s["name"] for s in gc["model_families"]}
    cap = float(dp["matched_density_km_per_100km2"])
    rows, sysr = [], []
    min_px = max(int(np.ceil(float(gc["min_len_km"]) / 0.112)), 2)
    for task in ("geo", m8.prim):
        for u in m8.units_of(task):
            cov = covered[u.sl]
            if not (cov & u.ev).any():
                continue
            ur = restrict_unit(u, cov, "_demcov")
            pool = m8.pool(u.key)
            dem = clip_to_unit(dem_full, u, min_px) if len(dem_full) else CandidateSet(pd.DataFrame(), [])
            if len(dem):
                dem.table["families"] = "dem"
            geo = _subset_by_family(pool, geo_names)
            cnn_geo = _subset_by_family(pool, geo_names | cnn_names)
            sets = {"geophysics": geo, "dem_only": dem, "geophysics_dem": _union(geo, dem, u.ev.shape, gc),
                    "cnn_geophysics": cnn_geo, "cnn_geophysics_dem": _union(cnn_geo, dem, u.ev.shape, gc)}
            for name, cs in sets.items():
                c = DV._clip(cs, ur, cov) if len(cs) else cs
                c = DV._cap(c, ur, cap) if len(c) else c
                rows.append({"task": u.task, "zone": u.zone, "set": name, "density_cap": cap,
                             "covered_eval_px": int(ur.ev.sum()), **coverage(m8.ctx, ur, c, oo)})
                h = system_hits(m8.ctx, ur, c.raster(ur.ev.shape) if len(c) else np.zeros(ur.ev.shape, np.float32), m8.sh)
                sysr.append(h.assign(task=u.task, zone=u.zone, set=name))
    res = pd.DataFrame(rows)
    sysd = pd.concat(sysr, ignore_index=True) if sysr else pd.DataFrame()
    return {"results": res, "systems": sysd, "dem_log": dlog, "covered_km2": float(covered.sum() * 0.01)}


def gate4(m8, res: pd.DataFrame, sysd: pd.DataFrame, est: dict | None, log: dict) -> pd.DataFrame:
    gi = m8.cfg["dem_pilot"]["go_if"]
    prim = m8.prim
    s = sysd[sysd.task == prim] if len(sysd) else sysd
    if not len(s):
        return pd.DataFrame([{"gate": "G4_dem", "decision": "INCONCLUSIVE", "reason": "no withheld system in the DEM footprint"}])
    p = s.pivot_table(index=["zone", "system_id"], columns="set", values="recovered", aggfunc="first").fillna(False).astype(bool)
    n_cov = int(len(p))
    add_geo = int((p.get("geophysics_dem", False) & ~p.get("geophysics", False)).sum())
    add_cnn = int((p.get("cnn_geophysics_dem", False) & ~p.get("cnn_geophysics", False)).sum())
    inc = add_cnn / max(n_cov, 1)
    if n_cov < int(gi["min_systems_covered"]):
        dec, why = "INCONCLUSIVE", f"only {n_cov} withheld systems inside the DEM pilot"
    elif max(add_geo, add_cnn) >= int(gi["additional_systems"]) and inc >= float(gi["incremental_system_recall"]):
        dec, why = "EXPAND_TARGETED", "DEM recovers systems the 100 m inputs miss"
    else:
        dec, why = "STOP", "no robust incremental coverage at matched density"
    return pd.DataFrame([{"gate": "G4_dem", "task": prim, "systems_in_footprint": n_cov,
                          "additional_vs_geophysics": add_geo, "additional_vs_cnn_geophysics": add_cnn,
                          "incremental_system_recall": inc, "decision": dec, "reason": why,
                          "projected_full_hours": (est or {}).get("projected_hours_all"),
                          "projected_full_raw_gib": (est or {}).get("projected_raw_gib_all"),
                          "profile": m8.profile["name"], "provisional": bool(m8.profile["provisional"]),
                          "log": json.dumps(log, default=str)}])


def run(m8) -> dict:
    sel, src = pilot(m8)
    m8.ws.save_table(sel, "pilot_tiles", "dem/pilot")
    est = m8.read_json("dem/pilot/dem_cost_estimate.json") or m8.m7_json("dem/pilot/dem_cost_estimate.json")
    storage = m8.read_json("dem/pilot/storage_estimate.json") or m8.m7_json("dem/pilot/storage_estimate.json")
    log = {"pilot_source": src, "n_pilot_tiles": int(len(sel)), "storage_estimate": storage}
    if m8.cfg["dem_pilot"]["stream"]:
        log.update(stream(m8, sel))
        est = m8.read_json("dem/pilot/dem_cost_estimate.json") or est
    mos = m8.dem_mosaic()
    if mos is None:
        raise DemUnavailable("no DEM mosaic (M8 or M7): set dem_pilot.stream = True (with shared.dem.download = True "
                             "or the tiles on disk) after reviewing dem/pilot/pilot_tiles.csv")
    log["mosaic"] = str(getattr(mos, "path", "")) or "m7 mosaic"
    inc = increment(m8, mos)
    gate = gate4(m8, inc["results"], inc["systems"], est, {**log, "covered_km2": inc["covered_km2"]})
    return {"pilot": sel, "results": inc["results"], "systems": inc["systems"], "dem_log": inc["dem_log"],
            "gate": gate, "log": pd.DataFrame([{k: json.dumps(v, default=str) if isinstance(v, (dict, list)) else v
                                                for k, v in {**log, "covered_km2": inc["covered_km2"]}.items()}])}
