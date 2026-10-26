"""Section 04: multimodal structural reasoning (label-free).

Magnetic-gravity agreement is spatially TOLERANT: a fault juxtaposing contrasting rocks produces a magnetic edge and a
gravity edge, but their gradient maxima need not coincide (different source depths, dip, gravity's ~2 km resolution).
For each tolerance t (``tolerances_m``) every magnetic worm pixel looks for the nearest gravity worm pixel: they agree
when the distance is <= t AND the axial strike difference is <= ``orient_tol_deg``. The agreement layer
``agree:mag_grav@t`` = (1 - d / t) x cos(strike difference) on agreeing pixels of BOTH modalities, 0 elsewhere.

Crest classes (codes of ``class:mag_grav``): 1 magnetic only, 2 gravity only, 3 agreeing (largest tolerance),
4 near but discordant strike (modality disagreement). Surface-subsurface classes (``class:surface_subsurface``):
1 surface-expressed (geophysical worm with a concordant terrain worm within ``terrain_tolerance_m``), 2 potentially
buried (geophysical only), 3 terrain-only lineament. A terrain lineament is NOT a confirmed fault.

Context (``context:<band>``): regional (``context_sigma_km``) percentile of strain-rate invariants, seismicity
densities, conductivity and basement depth: descriptors of the structural domain only. Scalar strain magnitude does
not give a fault orientation or slip tendency, and low seismicity never rejects a fault.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from . import potential_fields as PF


def _worm(m8, fam: str):
    L = m8.layers
    if not L.has(f"worms:{fam}:mask"):
        return None
    return {"mask": np.asarray(L.get(f"worms:{fam}:mask")) > 0, "strike": np.asarray(L.get(f"worms:{fam}:strike")),
            "pers": np.asarray(L.get(f"worms:{fam}:persistence"))}


def _nearest(target_mask: np.ndarray):
    d, (ir, ic) = ndimage.distance_transform_edt(~target_mask, return_indices=True)
    return d, ir, ic


def mag_grav(m8) -> tuple[dict, pd.DataFrame]:
    mc = m8.cfg["multimodal"]
    pixel_m = m8.layers.pixel_m
    mag, grav = _worm(m8, "magnetic"), _worm(m8, "gravity")
    if mag is None or grav is None or not mag["mask"].any() or not grav["mask"].any():
        return {}, pd.DataFrame([{"status": "worms missing (run section 03b)"}])
    valid = np.asarray(m8.ctx.data.valid, bool)
    tols = sorted(float(t) for t in mc["tolerances_m"])
    otol = float(mc["orient_tol_deg"])
    out, rows = {}, []
    code = np.zeros(valid.shape, np.uint8)
    pairs = {}
    for a, b, na in ((mag, grav, "magnetic"), (grav, mag, "gravity")):
        d, ir, ic = _nearest(b["mask"])
        r, c = np.nonzero(a["mask"])
        dm = d[r, c] * pixel_m
        od = PF.axial_diff(a["strike"][r, c], b["strike"][ir[r, c], ic[r, c]])
        pairs[na] = (r, c, dm, od, a["pers"][r, c] * b["pers"][ir[r, c], ic[r, c]])
        del d, ir, ic
    for t in tols:
        lay = np.where(valid, 0.0, np.nan).astype(np.float32)
        xs = np.where(valid, 0.0, np.nan).astype(np.float32)
        for na, (r, c, dm, od, pp) in pairs.items():
            ok = (dm <= t) & (od <= otol)
            sc = np.where(ok, (1.0 - dm / t) * np.cos(np.radians(od)), 0.0)
            lay[r, c] = np.maximum(lay[r, c], sc)
            xs[r, c] = np.maximum(xs[r, c], np.where(ok, pp, 0.0))
            rows.append({"tolerance_m": t, "from": na, "crest_px": int(len(r)), "agree_share": float(ok.mean()) if len(r) else np.nan,
                         "median_offset_m_agreeing": float(np.median(dm[ok])) if ok.any() else np.nan,
                         "discordant_near_share": float(((dm <= t) & (od > otol)).mean()) if len(r) else np.nan})
        out[f"agree:mag_grav@{int(t)}"] = lay
        out[f"agree:mag_grav_xscale@{int(t)}"] = xs
    tmax = tols[-1]
    for na, (r, c, dm, od, _) in pairs.items():
        own = 1 if na == "magnetic" else 2
        cls = np.where((dm <= tmax) & (od <= otol), 3, np.where(dm <= tmax, 4, own)).astype(np.uint8)
        code[r, c] = np.maximum(code[r, c], cls)
    out["class:mag_grav"] = code
    return out, pd.DataFrame(rows)


def surface_subsurface(m8) -> tuple[np.ndarray | None, pd.DataFrame]:
    mc = m8.cfg["multimodal"]
    pixel_m = m8.layers.pixel_m
    ter = _worm(m8, "terrain")
    geo = [w for w in (_worm(m8, "magnetic"), _worm(m8, "gravity")) if w is not None]
    if ter is None or not geo:
        return None, pd.DataFrame([{"status": "terrain or geophysical worms missing"}])
    gm = np.zeros(ter["mask"].shape, bool)
    gs = np.full(ter["mask"].shape, np.nan, np.float32)
    for w in geo:
        gs = np.where(w["mask"] & ~gm, w["strike"], gs)
        gm |= w["mask"]
    tol = float(mc["terrain_tolerance_m"]) / pixel_m
    otol = float(mc["orient_tol_deg"])
    code = np.zeros(gm.shape, np.uint8)
    d, ir, ic = _nearest(ter["mask"])
    r, c = np.nonzero(gm)
    ok = (d[r, c] <= tol) & (PF.axial_diff(gs[r, c], ter["strike"][ir[r, c], ic[r, c]]) <= otol)
    code[r, c] = np.where(ok, 1, 2)
    del d, ir, ic
    d2 = ndimage.distance_transform_edt(~gm)
    tr, tc = np.nonzero(ter["mask"])
    code[tr, tc] = np.where(d2[tr, tc] <= tol, np.maximum(code[tr, tc], 1), 3)
    names = {1: "surface_expressed", 2: "potentially_buried", 3: "terrain_only"}
    t = pd.DataFrame([{"class": names[k], "crest_px": int((code == k).sum()),
                       "crest_km": float((code == k).sum() * pixel_m / 1000.0)} for k in (1, 2, 3)])
    return code, t


def context_layers(m8) -> tuple[dict, pd.DataFrame]:
    """Regional percentile of each context band (descriptors, never rules)."""
    from gems_eda.enhance import nan_gaussian

    from ..rs_features import prank

    L = m8.layers
    valid = np.asarray(m8.ctx.data.valid, bool)
    sig = float(m8.cfg["multimodal"]["context_sigma_km"]) * 1000.0 / L.pixel_m
    out, rows = {}, []
    for b in m8.cfg["bands"]["context"]:
        if not L.has_band(b):
            rows.append({"band": b, "status": "missing"})
            continue
        a = nan_gaussian(L.band(b), sig)
        out[f"context:{b}"] = np.where(valid, prank(a, valid), np.nan).astype(np.float32)
        rows.append({"band": b, "status": "ok", "sigma_km": sig * L.pixel_m / 1000.0,
                     "note": "regional percentile: structural-domain descriptor only"})
    L.release()
    return out, pd.DataFrame(rows)


def by_region(m8, layers: dict) -> pd.DataFrame:
    from gems_eda.morphology import region_name

    from ..geo_eval import region_map

    reg = region_map(m8.ctx.data)
    rows = []
    for name, a in layers.items():
        if not name.startswith("class:"):
            continue
        for r in np.unique(reg[reg >= 0]):
            m = reg == r
            for k in np.unique(a[m & (a > 0)]):
                rows.append({"layer": name, "region": region_name(int(r), (3, 3)), "code": int(k),
                             "px": int((m & (a == k)).sum())})
    return pd.DataFrame(rows)


def run(m8) -> dict:
    key = m8.fp("multimodal", m8.cfg["multimodal"], m8.fp_of("s03b_structural_worms"))
    mg, t_mg = mag_grav(m8)
    ss, t_ss = surface_subsurface(m8)
    ctx_l, t_ctx = context_layers(m8)
    layers = dict(mg)
    if ss is not None:
        layers["class:surface_subsurface"] = ss
    for name, a in {**layers, **ctx_l}.items():
        m8.layers.put(name, a, key)
    grid = m8.ctx.data.grid
    agree = {k.replace(":", "_").replace("@", "_"): v for k, v in mg.items() if k.startswith("agree:")}
    if agree:
        m8.ws.save_raster(agree, "geophysical_features/multimodal/mag_grav_agreement.tif", grid)
    cls = {k.replace(":", "_"): v.astype(np.float32) for k, v in layers.items() if k.startswith("class:")}
    if cls:
        m8.ws.save_raster(cls, "geophysical_features/multimodal/crest_classes.tif", grid)
    if ctx_l:
        m8.ws.save_raster({k.replace(":", "_"): v for k, v in ctx_l.items()},
                          "geophysical_features/context/context_regional.tif", grid)
    return {"mag_grav": t_mg, "surface_subsurface": t_ss, "context": t_ctx, "regions": by_region(m8, layers),
            "key": key}
