"""Lineament-aware post-processing of probability maps into ranked candidate segments (Phase 3F).

These geometries are for *interpretation* (atlas, expert review). The official submission stays a
probability / binary raster: thinning a map to 1-px lines is only adopted if the DTI validates it
(``thinned_mask_dti`` gives the check).

Pipeline: smooth at a physical scale -> Hessian ridge strength and normal -> non-maximum
suppression across the ridge (splits wide blobs into crest lines) -> top-fraction gate ->
skeletonise -> remove junction pixels -> trace branches -> geometry (length, strike, sinuosity,
curvature), probability, multi-modal evidence and distance to mapped faults -> heuristic rank.
No dominant-strike filter is applied; geophysical edges are evidence, not confirmation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from gems_eda.enhance import nan_gaussian

from .context import P3Context

EVIDENCE = {"magnetic": "magnetic:thdr@2km", "gravity": "gravity:thdr@2km", "terrain": "terrain:curvature@1km",
            "terrain_ridge": "terrain:ridge@1km", "strain": "raw:geod_shearrate"}
NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def hessian_ridge(ps: np.ndarray, sigma_px: float):
    """Ridge strength (-lambda_min, >0 on bright lines) and unit normal (nr, nc) across the ridge."""
    a = np.nan_to_num(ps)
    hrr = ndimage.gaussian_filter(a, sigma_px, order=(2, 0))
    hcc = ndimage.gaussian_filter(a, sigma_px, order=(0, 2))
    hrc = ndimage.gaussian_filter(a, sigma_px, order=(1, 1))
    half = 0.5 * (hrr + hcc)
    root = np.sqrt((0.5 * (hrr - hcc)) ** 2 + hrc ** 2)
    lmin = half - root
    nr, nc = hrc, lmin - hrr                    # eigenvector of lambda_min (row, col)
    small = np.hypot(nr, nc) < 1e-12
    nr, nc = np.where(small, lmin - hcc, nr), np.where(small, hrc, nc)
    norm = np.maximum(np.hypot(nr, nc), 1e-12)
    return (-lmin).astype(np.float32), (nr / norm).astype(np.float32), (nc / norm).astype(np.float32)


def ridge_mask(p: np.ndarray, valid: np.ndarray, *, scale_m: float = 300.0, top_frac: float = 0.05,
               pixel_m: float = 100.0, min_px: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Thin crest lines of the smoothed map inside its top-``top_frac`` area. Returns (skeleton, smoothed)."""
    from skimage.morphology import remove_small_objects, skeletonize

    sig = max(scale_m / pixel_m, 0.5)
    ps = nan_gaussian(np.where(valid, p, np.nan).astype(np.float32), sig)
    strength, nr, nc = hessian_ridge(ps, sig)
    rr, cc = np.indices(ps.shape, dtype=np.float32)
    a = np.nan_to_num(ps, nan=-1.0)
    fwd = ndimage.map_coordinates(a, [rr + nr, cc + nc], order=1, mode="nearest")
    bwd = ndimage.map_coordinates(a, [rr - nr, cc - nc], order=1, mode="nearest")
    v = ps[valid & np.isfinite(ps)]
    thr = np.quantile(v, 1 - top_frac)
    m = valid & (a >= thr) & (strength > 0) & (a >= fwd) & (a >= bwd)
    m = ndimage.binary_closing(m, structure=np.ones((3, 3), bool)) & valid & (a >= thr)
    sk = skeletonize(m)
    sk = remove_small_objects(sk, min_size=min_px, connectivity=2)
    return sk, ps


def split_branches(skel: np.ndarray, min_px: int = 5) -> tuple[np.ndarray, int, np.ndarray]:
    """Remove junction pixels (>= 3 skeleton neighbours) -> labelled branches (8-connected)."""
    nb = ndimage.convolve(skel.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - skel
    junction = skel & (nb >= 3)
    branches = skel & ~ndimage.binary_dilation(junction, structure=np.ones((3, 3), bool))
    lab, n = ndimage.label(branches, structure=np.ones((3, 3), bool))
    sizes = ndimage.sum(branches, lab, np.arange(1, n + 1))
    keep = np.r_[False, sizes >= min_px]
    lab = np.where(keep[lab], lab, 0)
    return lab, n, junction


def trace(coords: np.ndarray) -> np.ndarray:
    """Order the pixels of one simple branch into a path (start at an end point; greedy nearest walk)."""
    pts = {tuple(x) for x in coords.tolist()}
    deg = {q: sum((q[0] + dr, q[1] + dc) in pts for dr, dc in NB8) for q in pts}
    start = min((q for q in pts if deg[q] <= 1), default=next(iter(pts)))
    path, seen, cur = [start], {start}, start
    while True:
        nxt = [(cur[0] + dr, cur[1] + dc) for dr, dc in NB8 if (cur[0] + dr, cur[1] + dc) in pts]
        nxt = [q for q in nxt if q not in seen]
        if not nxt:
            break
        cur = min(nxt, key=lambda q: abs(q[0] - path[-1][0]) + abs(q[1] - path[-1][1]))
        path.append(cur)
        seen.add(cur)
    return np.array(path)


def geometry(path: np.ndarray, pixel_m: float = 100.0, heading_step: int = 5) -> dict:
    y = -path[:, 0] * pixel_m / 1000.0          # km north
    x = path[:, 1] * pixel_m / 1000.0           # km east
    seg = np.hypot(np.diff(x), np.diff(y))
    length = float(seg.sum())
    chord = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    xy = np.c_[x - x.mean(), y - y.mean()]
    _, _, vt = np.linalg.svd(xy, full_matrices=False) if len(xy) > 1 else (None, None, np.eye(2))
    az = float(np.degrees(np.arctan2(vt[0, 0], vt[0, 1])) % 180.0)
    curv = np.nan
    if len(path) > 2 * heading_step:
        xs, ys = x[::heading_step], y[::heading_step]
        h = np.arctan2(np.diff(xs), np.diff(ys))
        dh = np.abs(np.angle(np.exp(1j * np.diff(h))))
        curv = float(np.degrees(dh).sum() / max(length, 1e-6))
    return {"length_km": length, "chord_km": chord, "sinuosity": length / max(chord, 1e-6),
            "strike_deg": az, "curvature_deg_per_km": curv}


def extract_candidates(ctx: P3Context, p: np.ndarray, *, scale_m: float = 300.0, top_frac: float = 0.05,
                       min_len_km: float = 1.0, model: str = "", evidence=EVIDENCE) -> tuple[pd.DataFrame, np.ndarray, list]:
    """Candidate segments of one probability map. Returns (table, branch-label raster, paths)."""
    data = ctx.data
    sk, ps = ridge_mask(p, data.valid, scale_m=scale_m, top_frac=top_frac)
    lab, _, _ = split_branches(sk)
    dist = ctx.dist_to_labels_km()
    ev_maps = {k: np.asarray(ctx.store.get(v)) for k, v in evidence.items() if ctx.store.has(v)}
    objs = ndimage.find_objects(lab)
    rows, paths = [], []
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        loc = np.argwhere(lab[sl] == i) + np.array([sl[0].start, sl[1].start])
        path = trace(loc)
        g = geometry(path, data.grid.pixel_km * 1000)
        if g["length_km"] < min_len_km:
            continue
        r, c = path[:, 0], path[:, 1]
        d = dist[r, c]
        row = {"cand_id": i, "model": model, "scale_m": scale_m, "top_frac": top_frac, "n_px": len(path), **g,
               "p_mean": float(np.nanmean(p[r, c])), "p_max": float(np.nanmax(p[r, c])), "ps_mean": float(np.nanmean(ps[r, c])),
               "dist_mapped_min_km": float(d.min()), "dist_mapped_median_km": float(np.median(d)),
               "r_mid": int(r[len(r) // 2]), "c_mid": int(c[len(c) // 2])}
        n_sup = 0
        for k, a in ev_maps.items():
            val = float(np.nanmean(np.abs(a[r, c])))
            row[f"ev_{k}"] = val
            n_sup += val >= 1.0                  # robust-z >= 1 along the trace
        row["n_supporting_modalities"] = n_sup
        rows.append(row)
        paths.append(path)
    df = pd.DataFrame(rows)
    if not len(df):
        return df, lab, paths
    df["status"] = pd.cut(df.dist_mapped_median_km, [-np.inf, 0.3, 2.0, np.inf],
                          labels=["on_mapped_fault", "near_mapped_extension", "new_candidate"]).astype(str)
    df["rank_score"] = df.ps_mean * np.sqrt(df.length_km) * (1 + 0.25 * df.n_supporting_modalities)
    df["rank_note"] = "heuristic: smoothed prob x sqrt(length) x (1 + 0.25 supporting modalities); not validated"
    x, y = data.grid.rc_to_xy(df.r_mid.to_numpy(), df.c_mid.to_numpy())
    df["x_mid"], df["y_mid"] = x, y
    try:
        from rasterio.warp import transform as warp

        lon, lat = warp(data.grid.crs, "EPSG:4326", list(x), list(y))
        df["lon_mid"], df["lat_mid"] = lon, lat
    except Exception:
        pass
    order = df.sort_values("rank_score", ascending=False).index
    df["rank_all"] = np.argsort(np.argsort(-df.rank_score.to_numpy())) + 1
    df["rank_new"] = np.nan
    new = df.status == "new_candidate"
    df.loc[new, "rank_new"] = df.loc[new, "rank_score"].rank(ascending=False, method="first")
    keep_paths = {int(cid): p_ for cid, p_ in zip(df.cand_id, paths)}
    return df.loc[order].reset_index(drop=True), lab, [keep_paths[int(c)] for c in df.loc[order].cand_id]


def write_lineaments(ctx: P3Context, df: pd.DataFrame, paths: list, path_gpkg) -> str:
    """LineStrings in the grid CRS -> GeoPackage (layer 'candidate_lineaments'); Parquet (+WKT) always."""
    from shapely.geometry import LineString

    geoms = []
    for pth in paths:
        x, y = ctx.data.grid.rc_to_xy(pth[:, 0], pth[:, 1])
        geoms.append(LineString(np.c_[x, y]) if len(pth) > 1 else None)
    out = df.copy()
    out["wkt"] = [g.wkt if g is not None else None for g in geoms]
    pq = str(path_gpkg).rsplit(".", 1)[0] + ".parquet"
    out.to_parquet(pq, index=False)
    try:
        import geopandas as gpd

        gpd.GeoDataFrame(df.copy(), geometry=geoms, crs=ctx.data.grid.crs).to_file(path_gpkg, driver="GPKG",
                                                                                  layer="candidate_lineaments")
        return str(path_gpkg)
    except ImportError:
        return pq


def blob_stats(p: np.ndarray, valid: np.ndarray, top_frac: float, pixel_km2: float = 0.01) -> dict:
    """Connected-component sizes of the plain top-frac mask (the 'enormous blob' diagnosis)."""
    from .budget import binary_mask

    m = binary_mask(p, valid, None, top_frac)
    lab, n = ndimage.label(m, structure=np.ones((3, 3), bool))
    if n == 0:
        return {"n_components": 0}
    sizes = np.sort(ndimage.sum(m, lab, np.arange(1, n + 1)))[::-1] * pixel_km2
    return {"top_frac": top_frac, "n_components": int(n), "largest_km2": float(sizes[0]),
            "share_in_largest": float(sizes[0] / sizes.sum()), "share_in_top10": float(sizes[:10].sum() / sizes.sum()),
            "median_km2": float(np.median(sizes))}


def thinned_mask_dti(ctx: P3Context, p: np.ndarray, skeleton: np.ndarray, widths_px=(0, 1, 2)) -> pd.DataFrame:
    """Geographic-fold DTI of the skeleton dilated by w px (binary) vs nothing else: does thinning pay?"""
    from ..metric import dti_components

    rows = []
    for w in widths_px:
        m = ndimage.binary_dilation(skeleton, structure=np.ones((3, 3), bool), iterations=w) if w else skeleton
        q = m.astype(np.float32)
        for k in ctx.fold_ids:
            c = dti_components(q, ctx.data.labels, eval_mask=ctx.data.valid & (ctx.folds.fold_map == k))
            rows.append({"width_px": w, "zone": k, "tp_w": c.tp, "fp_w": c.fp, "fn_w": c.fn, "area_px": int((m & ctx.data.valid & (ctx.folds.fold_map == k)).sum())})
    return pd.DataFrame(rows)


def atlas_figure(ctx: P3Context, df: pd.DataFrame, paths: list, p: np.ndarray, n: int = 12, half_km: float = 8.0,
                 status: str = "new_candidate"):
    """Ranked candidate atlas: hillshaded terrain + probability, and magnetic edge strength, per candidate."""
    import matplotlib.pyplot as plt

    from ..geo_eval import _hillshade

    sel = df[df.status == status].head(n) if status else df.head(n)
    idx = {int(c): i for i, c in enumerate(df.cand_id)}
    elev = np.asarray(ctx.store.get("raw:det_elev"))
    mag = np.asarray(ctx.store.get(EVIDENCE["magnetic"])) if ctx.store.has(EVIDENCE["magnetic"]) else None
    h = int(half_km / ctx.data.grid.pixel_km)
    ncol = 2 if mag is not None else 1
    fig, axs = plt.subplots(len(sel), ncol, figsize=(4.2 * ncol, 3.6 * max(len(sel), 1)), squeeze=False,
                            constrained_layout=True)
    lr, lc = np.nonzero(ctx.data.labels)
    for i, r in enumerate(sel.itertuples()):
        r0, c0 = max(r.r_mid - h, 0), max(r.c_mid - h, 0)
        r1, c1 = min(r.r_mid + h, elev.shape[0]), min(r.c_mid + h, elev.shape[1])
        pth = paths[idx[int(r.cand_id)]]
        inwin = (lr >= r0) & (lr < r1) & (lc >= c0) & (lc < c1)
        ax = axs[i, 0]
        ax.imshow(_hillshade(np.nan_to_num(elev[r0:r1, c0:c1])), cmap="gray")
        ax.imshow(np.ma.masked_invalid(p[r0:r1, c0:c1]), cmap="magma", alpha=0.45, vmin=0, vmax=np.nanquantile(p, 0.995))
        ax.plot(lc[inwin] - c0, lr[inwin] - r0, ",", color="#1baf7a")
        ax.plot(pth[:, 1] - c0, pth[:, 0] - r0, color="#2a78d6", lw=2)
        ax.set_title(f"#{int(r.rank_new) if r.rank_new == r.rank_new else i + 1}  L={r.length_km:.1f} km  strike {r.strike_deg:.0f}°  "
                     f"d={r.dist_mapped_median_km:.1f} km", fontsize=8, loc="left")
        ax.axis("off")
        if mag is not None:
            ax = axs[i, 1]
            ax.imshow(mag[r0:r1, c0:c1], cmap="cividis")
            ax.plot(pth[:, 1] - c0, pth[:, 0] - r0, color="#eb6834", lw=2)
            ax.set_title(f"magnetic THDR 2 km  | modalities >=1z: {r.n_supporting_modalities}", fontsize=8, loc="left")
            ax.axis("off")
    return fig
