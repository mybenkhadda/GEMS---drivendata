"""Experiment 4B: limited 1 m DEM pilot (opt-in; never a full download).

Reuses M7's DEM machinery end to end: ``m7.dem_pilot.run`` (tile inventory + diverse pilot selection: known faults,
fronts, basins, CNN misses, disagreement), ``m7.dem_pilot.download`` (explicit, storage-checked, only with
``dem.allow_download``), ``m7.dem_morph.run`` (tile-streamed 1 m morphology: hillshade statistics, slope, curvature,
relief / TPI-type openness, roughness, scarp density / length, aggregated exactly to the 100 m grid, coverage band
kept) -> the cached mosaic ``outputs/m7_discovery_first/dem/features/m7_dem_features_100m.tif``.

Order: (1) a cached mosaic on Drive is staged and used; (2) else, if ``dem.enabled``, the pilot is selected and only
tiles already on disk / Drive are processed; (3) downloads happen only with ``dem.allow_download`` and M7's storage
check. Missing coverage is NOT a negative observation (NaN, recorded).

Pilot comparison on DEM-covered pixels of the geo held-out units (restricted units: a new protocol): frozen W2, a
terrain-only score (mean rank of the evidence bands, ``p6.dem_pilot.evidence_map``) and W2 + terrain (mean of the two
ranks), same frac; unique terrain structures = terrain crest components farther than 300 m from the frozen W2 map
and from every mapped fault. Stop rule (config ``terrain``): covered share < min or unique structures < min -> STOP.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

EVIDENCE = ("scarp_density", "microrelief_rms", "slope_break_mean", "curv_profile_absmean")


def stop_rule(covered_share: float, n_unique: int, tc: dict) -> dict:
    why = []
    if not np.isfinite(covered_share) or covered_share < float(tc["min_covered_share"]):
        why.append(f"covered share {covered_share:.4f} < {tc['min_covered_share']}")
    if n_unique < int(tc["min_unique_structures"]):
        why.append(f"unique terrain structures {n_unique} < {tc['min_unique_structures']}")
    return {"decision": "STOP" if why else "CONTINUE", "reasons": why}


def mosaic(R) -> tuple[object | None, dict]:
    """(DemMosaic or None, provenance) without downloading unless allowed."""
    from ..m7.dem_morph import DemMosaic, mosaic_path

    m7 = R.gm.m7
    dc = R.rnb["dem"]
    p = mosaic_path(m7)
    info = {"mosaic": str(p)}
    if not p.exists():
        rel = p.relative_to(m7.local).as_posix()
        if (R.drive / rel).exists():
            m7.stage_in([rel])
    if p.exists():
        info["source"] = "cached M7 mosaic"
        return DemMosaic(p), info
    if not dc.get("enabled") or dc.get("cached_only", True) and not dc.get("process_cached_tiles", False):
        info["source"] = "ABSENT (dem.enabled False or cached_only without a cached mosaic)"
        return None, info
    from ..m7 import dem_morph as DM
    from ..m7 import dem_pilot as DP

    sel = DP.run(m7)
    info["pilot_status"] = sel.get("status")
    if sel.get("status") != "OK" or "pilot" not in sel:
        info["source"] = f"ABSENT (pilot {sel.get('status')}: {sel.get('reason', '')})"
        return None, info
    pilot = sel["pilot"].head(int(R.rc["terrain"]["max_tiles"]))
    R.ws.save_table(pilot, "terrain/pilot_inventory.csv")
    if dc.get("allow_download"):
        m7.cfg["dem"]["download"] = True
        try:
            dl = DP.download(m7, pilot)
            R.ws.save_table(pd.DataFrame(dl), "terrain/download_log.csv")
        finally:
            m7.cfg["dem"]["download"] = False
    out = DM.run(m7, pilot)
    R.ws.save_table(out["timing"], "terrain/tile_provenance.csv")
    if out.get("mosaic") is None:
        info["source"] = "ABSENT (no pilot tile available locally / on Drive; downloads not allowed)"
        return None, info
    info["source"] = "pilot mosaic processed this run"
    return DemMosaic(p), info


def run(R) -> dict:
    from ..gems3 import representation as REP
    from ..geomoe.evaluate import policy_q
    from ..m7.protocol import restrict_unit
    from ..p6.dem_pilot import evidence_map

    tc = R.rc["terrain"]
    mos, info = mosaic(R)
    if mos is None:
        R.ws.save_json({**info, "decision": "NOT_EVALUATED"}, "terrain/pilot_decision.json")
        return {"status": "SKIPPED_BY_VALID_DECISION", "reason": info["source"]}
    cov = mos.coverage(float(R.cfg["dem"]["min_coverage"])) & np.asarray(R.ctx.data.valid, bool)
    bands = {b: mos.band(b) for b in EVIDENCE if mos.has(b)}
    ev = evidence_map(bands, cov) if bands else np.full(cov.shape, np.nan, np.float32)
    share = float(cov.sum() / max(int(np.asarray(R.ctx.data.valid).sum()), 1))
    R.ws.save_table(pd.DataFrame([{"covered_px": int(cov.sum()), "covered_share_of_valid": share,
                                   "bands": ";".join(bands), "source": info["source"]}]), "terrain/tile_coverage.csv")
    fz = R.gm.frozen_policy()["cell"]
    lab = np.asarray(R.ctx.data.labels, bool)
    rows, uniq = [], []
    for su in [s for s in R.units if s.scenario == "V3"]:
        u = su.unit
        c = cov[u.sl] & u.ev
        if not c.any():
            continue
        ur = restrict_unit(u, c, "_dem")
        w = R.gm.frozen_smap(su)
        t = np.where(c, np.nan_to_num(ev[u.sl], nan=0.0), 0.0)
        tr = REP.rank01(t, c)
        scores = {"frozen_W2": w, "terrain_only": tr,
                  "W2_plus_terrain": 0.5 * (REP.rank01(np.nan_to_num(w, nan=0.0), c) + tr) if w is not None else None}
        for m, s in scores.items():
            if s is None:
                continue
            q = policy_q(R.gm, ur, s, str(fz["rep"]), float(tc["frac"]))
            r = R.gm.score(ur, q)
            rows.append({"unit": ur.key, "zone": su.zone, "model": m, "rep": fz["rep"], "frac": tc["frac"],
                         **{k: r[k] for k in ("tp_w", "fp_w", "fn_w", "n_truth", "dti", "area_px")}})
        qt = policy_q(R.gm, ur, tr, "crest_1px", float(tc["frac"])) > 0
        qw = policy_q(R.gm, ur, w, str(fz["rep"]), float(fz["frac"])) > 0 if w is not None else np.zeros_like(qt)
        far = (ndimage.distance_transform_edt(~qw) > 3 if qw.any() else np.ones_like(qt)) & \
              (ndimage.distance_transform_edt(~lab[u.sl]) > 3 if lab[u.sl].any() else np.ones_like(qt))
        lb, n = ndimage.label(qt & far, structure=np.ones((3, 3), bool))
        sizes = np.bincount(lb.ravel())[1:] if n else np.array([])
        uniq += [{"unit": ur.key, "component": i + 1, "length_km": float(s * 0.112)} for i, s in enumerate(sizes)
                 if s >= 5]
        R.gm.release(su.base)
    C = pd.DataFrame(rows)
    U = pd.DataFrame(uniq)
    R.ws.save_table(C, "terrain/pilot_comparison.csv")
    R.ws.save_table(U, "terrain/unique_terrain_structures.csv")
    d = stop_rule(share, int(len(U)), tc)
    d.update(info, covered_share=share, n_unique=int(len(U)),
             unique_length_km=float(U.length_km.sum()) if len(U) else 0.0,
             note="terrain gains are claimed only through the official-metric rows of pilot_comparison.csv")
    R.ws.save_json(d, "terrain/pilot_decision.json")
    return {"status": "COMPLETE", "note": f"DEM {d['decision']}: covered {share:.3%}, unique {len(U)}"}
