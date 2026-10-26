"""Section 03b: multiscale geological "worms" (scale-space edge trajectories).

For a potential field the physical scale-space is UPWARD CONTINUATION (Hornby et al. 1999 "worms"): the maxima of
the horizontal gradient at successive continuation heights trace source edges; a vertical contact keeps its worm in
place, a dipping contact makes the worm migrate down-dip with height. Terrain has no potential-field scale-space, so it
uses Gaussian smoothing. Levels below ``min_scale_factor`` x the survey's effective resolution are not computed.

Per level: THDR (Gaussian derivative at 0.7 px after continuation, or at the level's sigma), non-maximum suppression
across the edge, a LOCAL percentile threshold (``local_quantile`` in ``local_block_km`` blocks; a weak edge in a quiet
basin is not drowned by a strong range front elsewhere) and a structure-tensor coherence gate.

Per base-level worm pixel (the finest level kept), linked to every coarser level by the nearest worm pixel within
``link_tolerance_factor`` x level with an axial strike difference <= ``link_angle_deg``::

    persistence        share of levels with a linked worm (base included)
    run                consecutive levels linked from the base (a trajectory, not isolated hits)
    orient_stability   resultant length of the doubled strikes across linked levels (1 = constant strike)
    ridge_stability    1 - CV of the normalised strength across linked levels
    scale_max_m        level of maximum normalised strength (scale of maximum support)
    migration          signed lateral offset (m) of the linked worms per m of level above the base, along the base
                       gradient direction (potential fields: a dip indicator; 0 = vertical contact or unresolved)
    continuity_km      length of the 8-connected base-worm component

``worms:<family>`` (persistence x base strength) feeds the candidate generator. Nothing here reads labels; the
coverage comparison at the end of the section reads the units' truth for evaluation only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from . import potential_fields as PF

STATS = ("persistence", "run", "orient_stability", "ridge_stability", "scale_max_m", "migration", "continuity_km",
         "strike", "mask")


def _level(f: np.ndarray, valid: np.ndarray, level_m: float, space: str, wc: dict, px_km, pixel_m: float) -> dict:
    from ..m7.generator import _nms

    if space == "upward":
        g = PF.upward(f, level_m / 1000.0, px_km)
        sig = 0.7
    else:
        g = f
        sig = PF.sigma_px_of(level_m, pixel_m)
    ge, gn = PF.hgrad(g, sig, px_km)
    th = np.hypot(ge, gn).astype(np.float32)
    norm = np.maximum(th, 1e-12)
    nr, nc = (-gn / norm).astype(np.float32), (ge / norm).astype(np.float32)   # gradient direction in (row, col)
    nms = _nms(th, nr, nc) & valid
    coh = PF.coherence(ge, gn, max(2.0 * sig, 1.0))
    block = max(int(round(float(wc["local_block_km"]) * 1000.0 / pixel_m)), 8)
    thr = PF.block_quantile_map(th, nms, block, float(wc["local_quantile"]))
    mask = nms & (th >= thr) & (coh >= float(wc["coherence_min"]))
    v = th[mask]
    ref = float(np.quantile(v, 0.99)) if v.size else 1.0
    return {"mask": mask, "strike": PF.strike_from_gradient(ge, gn), "s": np.clip(th / max(ref, 1e-12), 0, 1),
            "nr": nr, "nc": nc}


def family_worms(store, fam: str, wc: dict) -> tuple[dict, pd.DataFrame]:
    """Full-grid worm statistics of one family (arrays NaN off the base worms) + a per-level log."""
    space = (wc.get("scale_space") or {}).get(fam, "gaussian")
    levels = store.scales(fam, wc["levels_m"])
    valid = store.valid(fam)
    f, _ = PF.fill(store.field(fam))
    px_km, pixel_m = store.px_km, store.pixel_m
    base = _level(f, valid, levels[0], space, wc, px_km, pixel_m)
    br, bc = np.nonzero(base["mask"])
    n = len(br)
    s0 = base["s"][br, bc].astype(np.float64)
    st0 = base["strike"][br, bc].astype(np.float64)
    n_r, n_c = base["nr"][br, bc], base["nc"][br, bc]
    cnt = np.ones(n, np.int16)
    run = np.ones(n, np.int16)
    alive = np.ones(n, bool)
    c2, s2 = np.cos(np.radians(2 * st0)), np.sin(np.radians(2 * st0))
    ssum, ssq = s0.copy(), s0 * s0
    best_s, best_l = s0.copy(), np.full(n, float(levels[0]))
    mig_sum, mig_n = np.zeros(n), np.zeros(n)
    log = [{"family": fam, "level_m": levels[0], "space": space, "worm_px": n, "linked_px": n}]
    del base
    for lv in levels[1:]:
        L = _level(f, valid, lv, space, wc, px_km, pixel_m)
        if not L["mask"].any():
            alive[:] = False
            log.append({"family": fam, "level_m": lv, "space": space, "worm_px": 0, "linked_px": 0})
            continue
        d, (ir, ic) = ndimage.distance_transform_edt(~L["mask"], return_indices=True)
        dd = d[br, bc]
        jr, jc = ir[br, bc], ic[br, bc]
        del d, ir, ic
        tol = max(float(wc["link_tolerance_factor"]) * lv / pixel_m, 1.0)
        stj = L["strike"][jr, jc].astype(np.float64)
        ok = (dd <= tol) & (PF.axial_diff(st0, stj) <= float(wc["link_angle_deg"]))
        cnt += ok
        alive &= ok
        run += alive
        c2 += np.where(ok, np.cos(np.radians(2 * stj)), 0.0)
        s2 += np.where(ok, np.sin(np.radians(2 * stj)), 0.0)
        sj = L["s"][jr, jc].astype(np.float64)
        ssum += np.where(ok, sj, 0.0)
        ssq += np.where(ok, sj * sj, 0.0)
        better = ok & (sj > best_s)
        best_s, best_l = np.where(better, sj, best_s), np.where(better, float(lv), best_l)
        off = (jr - br) * n_r + (jc - bc) * n_c                   # signed offset along the base gradient direction (px)
        mig_sum += np.where(ok, off * pixel_m / max(lv - levels[0], 1e-6), 0.0)
        mig_n += ok
        log.append({"family": fam, "level_m": lv, "space": space, "worm_px": int(L["mask"].sum()),
                    "linked_px": int(ok.sum()), "tolerance_px": tol})
        del L, jr, jc
    nl = float(len(levels))
    mean = ssum / cnt
    var = np.maximum(ssq / cnt - mean * mean, 0.0)
    H, W = valid.shape
    lab, _ = ndimage.label(_paint((H, W), br, bc), structure=np.ones((3, 3), bool))
    sizes = np.bincount(lab.ravel())
    out = {}

    def paint(vals, dtype=np.float32):
        a = np.full((H, W), np.nan, dtype)
        a[br, bc] = vals
        return a

    out["persistence"] = paint(cnt / nl)
    out["run"] = paint(run / nl)
    out["orient_stability"] = paint(np.hypot(c2, s2) / cnt)
    out["ridge_stability"] = paint(np.where(cnt >= 2, 1.0 - np.sqrt(var) / np.maximum(mean, 1e-9), np.nan))
    out["scale_max_m"] = paint(best_l)
    out["migration"] = paint(np.where(mig_n > 0, mig_sum / np.maximum(mig_n, 1), np.nan))
    out["continuity_km"] = paint(sizes[lab[br, bc]] * pixel_m / 1000.0)
    out["strike"] = paint((np.degrees(np.arctan2(s2, c2)) / 2.0) % 180.0)
    out["mask"] = _paint((H, W), br, bc).astype(np.float32)
    resp = np.where(valid, 0.0, np.nan).astype(np.float32)
    resp[br, bc] = (cnt / nl) * s0
    out["response"] = resp
    t = pd.DataFrame(log)
    t["levels_used"] = ";".join(f"{x:g}" for x in levels)
    t["share_persistent"] = float((cnt >= int(wc["min_levels"])).mean()) if n else np.nan
    return out, t


def _paint(shape, r, c) -> np.ndarray:
    m = np.zeros(shape, bool)
    m[r, c] = True
    return m


def run(m8) -> dict:
    """Every configured family: worm layers registered in the layer store + GeoTIFFs + logs."""
    wc = m8.cfg["worms"]
    key = m8.fp("worms", wc, m8.cfg["bands"]["families"])
    logs, summ = [], []
    for fam in wc["families"]:
        if not m8.layers.has_band(m8.layers.families[fam]["field"]):
            logs.append(pd.DataFrame([{"family": fam, "status": "field band missing"}]))
            continue
        arrs, lg = family_worms(m8.layers, fam, wc)
        for k in STATS:
            m8.layers.put(f"worms:{fam}:{k}", arrs[k], key)
        m8.layers.put(f"worms:{fam}", arrs["response"], key, {"definition": "persistence x base normalised THDR"})
        m8.ws.save_raster({k: arrs[k] for k in ("persistence", "run", "orient_stability", "ridge_stability",
                                                "scale_max_m", "migration", "continuity_km")},
                          f"geophysical_features/structural_worms/worms_{fam}.tif", m8.ctx.data.grid)
        msk = arrs["mask"] > 0
        summ.append({"family": fam, "worm_px": int(msk.sum()),
                     "share_persistent": float(np.nanmean(arrs["persistence"][msk] >= int(wc["min_levels"]) /
                                                          max(len(lg), 1))) if msk.any() else np.nan,
                     "median_orient_stability": float(np.nanmedian(arrs["orient_stability"][msk])) if msk.any() else np.nan,
                     "median_scale_max_m": float(np.nanmedian(arrs["scale_max_m"][msk])) if msk.any() else np.nan,
                     "levels_used": lg.levels_used.iloc[0]})
        logs.append(lg)
        del arrs
        m8.layers.release()
    return {"log": pd.concat(logs, ignore_index=True) if logs else pd.DataFrame(), "summary": pd.DataFrame(summ),
            "key": key}


def coverage_vs_fixed_scale(m8) -> pd.DataFrame:
    """EVALUATION (reads truth): withheld-system recall of persistent worms (>= min_levels) vs the single-scale gated
    crests of the same field at the base level, and the systems only the worms cover."""
    from ..m7.protocol import system_hits

    wc = m8.cfg["worms"]
    rows = []
    for fam in wc["families"]:
        name = f"worms:{fam}:persistence"
        if not m8.layers.has(name):
            continue
        pers = m8.layers.get(name)
        nl = max(len(m8.layers.scales(fam, wc["levels_m"])), 1)
        for u in m8.units_of(m8.prim):
            p = np.asarray(pers[u.sl], np.float32)
            worm = np.isfinite(p) & (p * nl >= int(wc["min_levels"]) - 1e-6)
            base = np.isfinite(p)
            h_w = system_hits(m8.ctx, u, worm.astype(np.float32), m8.sh)
            h_b = system_hits(m8.ctx, u, base.astype(np.float32), m8.sh)
            if not len(h_w):
                continue
            m = h_w.merge(h_b, on="system_id", suffixes=("_worm", "_base"))
            rows.append({"family": fam, "task": u.task, "zone": u.zone, "n_systems": int(len(m)),
                         "recall_persistent_worms": float(m.recovered_worm.mean()),
                         "recall_all_base_worms": float(m.recovered_base.mean()),
                         "worm_px_per_100km2": float(worm[u.ev].sum() / max(u.ev.sum() * 0.01, 1e-9) * 100.0),
                         "base_px_per_100km2": float(base[u.ev].sum() / max(u.ev.sum() * 0.01, 1e-9) * 100.0)})
    return pd.DataFrame(rows)
