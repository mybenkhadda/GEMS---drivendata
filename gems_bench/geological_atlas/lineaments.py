"""Candidate structural lineaments from fixed, label-free rules (UNVERIFIED geological structures).

1. E = xm_evidence_mean smoothed with sigma ``lineaments.smooth_sigma_px``;
2. mask = E >= its ``evidence_quantile`` over the footprint (a quantile of the evidence itself, not of labels);
3. skeleton of the mask; junction pixels (> 2 skeleton neighbours) removed so branches become segments;
4. 8-connected segments kept when length >= ``min_length_km`` and linearity (PCA l1 / (l1 + l2)) >=
   ``min_linearity``;
5. geometry = polyline of the segment pixels ordered along the principal axis (300 m bins), simplified.
Mapped faults are used ONLY afterwards, to annotate ``mapped_fault_distance_m`` (diagnostic, not a selector).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from .circular_statistics import axial_mean


def _junction_free(skel: np.ndarray) -> np.ndarray:
    nb = ndimage.convolve(skel.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - skel
    return skel & (nb <= 2)


def extract(E: np.ndarray, footprint: np.ndarray, cfg, px: float) -> tuple[np.ndarray, int, float]:
    from skimage.morphology import skeletonize

    lc = cfg["lineaments"]
    Es = ndimage.gaussian_filter(np.nan_to_num(E, nan=0.0), float(lc["smooth_sigma_px"]))
    Es[~footprint] = 0
    thr = float(np.quantile(Es[footprint & np.isfinite(E)], float(lc["evidence_quantile"])))
    skel = skeletonize(Es >= thr)
    seg = _junction_free(skel)
    lab, n = ndimage.label(seg, structure=np.ones((3, 3), bool))
    return lab, n, thr


def segments(lab: np.ndarray, n: int, cfg, grid) -> list[dict]:
    lc = cfg["lineaments"]
    px = grid.pixel_m
    objs = ndimage.find_objects(lab)
    out = []
    a, _, c, _, e, f = grid.transform[:6]
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        rr, cc = np.nonzero(lab[sl] == i)
        if len(rr) < 3:
            continue
        rr = rr + sl[0].start
        cc = cc + sl[1].start
        E_ = c + a * (cc + 0.5)
        N_ = f + e * (rr + 0.5)
        X = np.stack([E_ - E_.mean(), N_ - N_.mean()], 1)
        w, v = np.linalg.eigh(np.cov(X.T))
        l2, l1 = w
        lin = float(l1 / (l1 + l2 + 1e-12))
        d = v[:, 1]
        proj = X @ d
        length_km = float((proj.max() - proj.min() + px) / 1e3)
        if length_km < float(lc["min_length_km"]) or lin < float(lc["min_linearity"]):
            continue
        strike = float(np.degrees(np.arctan2(d[0], d[1])) % 180.0)        # azimuth of (dE, dN) from north
        bins = np.arange(proj.min(), proj.max() + 300.0, 300.0)
        idx = np.clip(np.digitize(proj, bins) - 1, 0, len(bins) - 1)
        pts = [(float(E_[idx == k].mean()), float(N_[idx == k].mean())) for k in np.unique(idx)]
        out.append({"label": i, "rows": rr, "cols": cc, "points": pts, "length_km": length_km, "linearity": lin,
                    "strike_deg": strike, "centroid_e": float(E_.mean()), "centroid_n": float(N_.mean())})
    return out


def _interpret(ev: dict, cov_q: float, seam_m: float) -> tuple[str, str]:
    groups = [g for g in ("magnetic", "gravity", "topography") if ev.get(g, 0) >= 0.8]
    if len(groups) >= 2 and ev.get("orientation_agreement", 0) >= 0.5:
        txt = (f"aligned {' + '.join(groups)} boundary with consistent strike: consistent with a fault-controlled "
               "boundary or a structurally controlled lithological contact")
    elif len(groups) >= 2:
        txt = f"co-located {' + '.join(groups)} edges without strike agreement: intersecting or unrelated structures"
    elif groups == ["topography"]:
        txt = "topographic lineament only: scarp, range front, drainage or erosional feature"
    elif groups == ["magnetic"]:
        txt = "magnetic boundary without surface expression: lithological contact or buried structure"
    elif groups == ["gravity"]:
        txt = "density boundary: possibly basin-bounding structure or deep contact"
    else:
        txt = "weak, diffuse evidence along the trace"
    lim = ["unverified candidate from fixed label-free rules", "100 m analysis grid"]
    if cov_q < 0.5:
        lim.append("little or no high-resolution DEM coverage")
    if seam_m < 500:
        lim.append(f"within {seam_m:.0f} m of a DEM project seam")
    return txt, "; ".join(lim)


def inventory(segs: list[dict], maps: dict, grid, fault_dist_m: np.ndarray, seam_dist_m: np.ndarray | None,
              dem_cov: np.ndarray | None, scale_ranks: dict) -> pd.DataFrame:
    """``maps``: magnetic / gravity / topography / dem ranks, orientation agreement, persistence, continuity."""
    from shapely.geometry import LineString

    rows = []
    for k, s in enumerate(segs):
        rr, cc = s["rows"], s["cols"]

        def m(name):
            a = maps.get(name)
            return float(np.nanmean(a[rr, cc])) if a is not None and np.isfinite(a[rr, cc]).any() else np.nan

        ev = {g: m(g) for g in ("magnetic", "gravity", "topography", "dem")}
        ev["orientation_agreement"] = m("orientation_agreement")
        best_scale, best = None, -1.0
        for sc, r in scale_ranks.items():
            v = float(np.nanmean(r[rr, cc])) if np.isfinite(r[rr, cc]).any() else -1
            if v > best:
                best_scale, best = sc, v
        cov_q = float(np.nanmean(dem_cov[rr, cc])) if dem_cov is not None else 0.0
        seam_m = float(np.nanmin(seam_dist_m[rr, cc])) if seam_dist_m is not None else np.inf
        txt, lim = _interpret(ev, cov_q, seam_m)
        geom = LineString(s["points"]) if len(s["points"]) >= 2 else None
        rows.append({"candidate_id": f"L{k + 1:05d}", "geometry": geom,
                     "centroid_e": s["centroid_e"], "centroid_n": s["centroid_n"], "length_km": s["length_km"],
                     "strike_deg": s["strike_deg"], "linearity": s["linearity"], "continuity": m("continuity"),
                     "dominant_scale": best_scale, "magnetic_evidence": ev["magnetic"], "gravity_evidence": ev["gravity"],
                     "terrain_evidence": m("terrain"), "dem_evidence": ev["dem"],
                     "orientation_agreement": ev["orientation_agreement"], "multiscale_persistence": m("persistence"),
                     "coverage_quality": cov_q, "seam_proximity_m": seam_m,
                     "mapped_fault_distance_m": float(np.nanmedian(fault_dist_m[rr, cc])),
                     "evidence_count": int(sum(ev.get(g, 0) >= 0.8 for g in ("magnetic", "gravity", "topography"))),
                     "interpretation": txt, "limitations": lim, "status": "UNVERIFIED geological structure"})
    df = pd.DataFrame(rows)
    if len(df):
        mean_strike, R = axial_mean(df.strike_deg, df.length_km)
        df.attrs["length_weighted_mean_strike"] = mean_strike
        df.attrs["strike_resultant"] = R
    return df


def export(df: pd.DataFrame, grid, out_dir, simplify_m: float) -> list:
    import geopandas as gpd

    from .cache import atomic_write_bytes, atomic_write_csv, atomic_write_text

    outs = []
    g = gpd.GeoDataFrame(df, geometry="geometry", crs=grid.crs)
    g["geometry"] = g.geometry.simplify(simplify_m)
    import io

    buf = io.BytesIO()
    g.to_parquet(buf, index=False)
    outs.append(atomic_write_bytes(out_dir / "candidate_lineaments.parquet", buf.getvalue()))
    outs.append(atomic_write_text(out_dir / "candidate_lineaments.geojson", g.to_crs(4326).to_json()))
    outs.append(atomic_write_csv(out_dir / "candidate_lineaments_summary.csv", pd.DataFrame(df.drop(columns="geometry"))))
    return outs
