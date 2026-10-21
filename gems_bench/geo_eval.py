"""Geological error analysis: per-segment recovery, stratified recall, candidates, error atlas.

Per-pixel recovery uses the metric's own TP support ``M(g) = max_x p(x) k(d(x, g))``, so a
segment's mean M is exactly its contribution to TP_w / n_px ("weighted recall"). A segment
counts as *recovered* when its mean M >= ``recovered_at`` (default 0.5).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import ndimage

from gems_eda.morphology import region_ids, region_name

from .data import NativeData
from .metric import clean_prediction, dti_components, local_weighted_max

STRATA = ["length_bin", "strike_sector", "sinuosity_bin", "curv_bin", "junction", "primary_archetype",
          "region_name", "fold"]


def segment_recovery(pred: np.ndarray, data: NativeData, seg_raster: np.ndarray, seg_table: pd.DataFrame,
                     fold_map: np.ndarray | None = None, threshold: float = 0.5,
                     tolerances=(100.0, 200.0, 300.0), recovered_at: float = 0.5) -> pd.DataFrame:
    p = clean_prediction(pred)
    p[~data.valid] = 0
    M = local_weighted_max(p, 300.0, (data.grid.pixel_km_y * 1000, data.grid.pixel_km_x * 1000))
    hit = p >= threshold
    d_hit = ndimage.distance_transform_edt(~hit, sampling=(data.grid.pixel_km_y * 1000, data.grid.pixel_km_x * 1000)) \
        if hit.any() else np.full(p.shape, np.inf)
    m = (seg_raster >= 0) & data.valid
    df = pd.DataFrame({"seg_id": seg_raster[m], "M": M[m], "p": p[m]})
    for t in tolerances:
        df[f"hit{t:g}"] = d_hit[m] <= t
    if fold_map is not None:
        df["fold"] = fold_map[m]
    agg = {"n_px": ("M", "size"), "mean_M": ("M", "mean"), "mean_p_on_trace": ("p", "mean"),
           **{f"recall@{t:g}m": (f"hit{t:g}", "mean") for t in tolerances}}
    if fold_map is not None:
        agg["fold"] = ("fold", lambda s: int(pd.Series(s).mode().iloc[0]))
    out = df.groupby("seg_id").agg(**agg).reset_index()
    out = out.merge(seg_table, on="seg_id", how="left")
    out["recovered"] = out.mean_M >= recovered_at
    out["region_name"] = [region_name(int(r), (3, 3)) if pd.notna(r) else "?" for r in out.get("region", np.nan)]
    return out


def stratified_recall(seg_rec: pd.DataFrame, strata=STRATA) -> pd.DataFrame:
    rows = []
    for col in strata:
        if col not in seg_rec:
            continue
        for val, g in seg_rec.groupby(col, observed=True):
            km = g.length_km.fillna(0)
            rows.append({"stratum": col, "value": str(val), "n_segments": len(g), "trace_km": float(km.sum()),
                         "weighted_recall": float(np.average(g.mean_M, weights=g.n_px)),
                         "recovered_share_km": float(km[g.recovered].sum() / max(km.sum(), 1e-9)),
                         "recall@300m": float(np.average(g["recall@300m"], weights=g.n_px)) if "recall@300m" in g else np.nan})
    return pd.DataFrame(rows)


def recovery_overlap(rec: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-segment recovered flags across models -> (wide table, pairwise summary with unique km)."""
    names = list(rec)
    wide = rec[names[0]][["seg_id", "length_km", "primary_archetype"]].copy()
    for n in names:
        wide = wide.merge(rec[n][["seg_id", "recovered", "mean_M"]].rename(
            columns={"recovered": f"rec_{n}", "mean_M": f"M_{n}"}), on="seg_id", how="outer")
    rec_cols = [f"rec_{n}" for n in names]
    wide[rec_cols] = wide[rec_cols].fillna(False).astype(bool)
    wide["n_models_recovering"] = wide[rec_cols].sum(1)
    wide["only_model"] = [names[r.index(True)] if sum(r) == 1 else "" for r in wide[rec_cols].values.tolist()]
    rows = []
    km = wide.length_km.fillna(0)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ra, rb = wide[f"rec_{a}"], wide[f"rec_{b}"]
            rows.append({"model_a": a, "model_b": b, "both_km": km[ra & rb].sum(), "only_a_km": km[ra & ~rb].sum(),
                         "only_b_km": km[~ra & rb].sum(), "neither_km": km[~ra & ~rb].sum(),
                         "jaccard_segments": float((ra & rb).sum() / max((ra | rb).sum(), 1))})
    for n in names:
        rows.append({"model_a": n, "model_b": "(unique vs all)", "only_a_km": km[wide.only_model == n].sum()})
    rows.append({"model_a": "(none)", "model_b": "(missed by all)", "neither_km": km[wide.n_models_recovering == 0].sum()})
    return wide, pd.DataFrame(rows)


def region_map(data: NativeData, region_grid=(3, 3)) -> np.ndarray:
    rr, cc = np.nonzero(data.valid)
    out = np.full(data.shape, -1, np.int16)
    out[rr, cc] = region_ids(rr, cc, data.grid, region_grid)
    return out


def dti_by_region(pred: np.ndarray, data: NativeData, regions: np.ndarray) -> pd.DataFrame:
    rows = []
    for r in np.unique(regions[regions >= 0]):
        c = dti_components(pred, data.labels, eval_mask=data.valid & (regions == r))
        rows.append({"region": int(r), "region_name": region_name(int(r), (3, 3)), **c.as_dict()})
    return pd.DataFrame(rows)


def block_dti(pred: np.ndarray, data: NativeData, block_km: float = 30.0, min_label_px: int = 300) -> pd.DataFrame:
    """DTI per square block (for objective best / worst atlas windows)."""
    bpx = int(round(block_km / data.grid.pixel_km))
    rows = []
    for r0 in range(0, data.shape[0] - bpx + 1, bpx):
        for c0 in range(0, data.shape[1] - bpx + 1, bpx):
            ev = np.zeros(data.shape, bool)
            ev[r0:r0 + bpx, c0:c0 + bpx] = data.valid[r0:r0 + bpx, c0:c0 + bpx]
            if ev.mean() == 0 or (data.labels & ev).sum() < min_label_px:
                continue
            c = dti_components(pred, data.labels, eval_mask=ev)
            rows.append({"r0": r0, "r1": r0 + bpx, "c0": c0, "c1": c0 + bpx, **c.as_dict()})
    return pd.DataFrame(rows)


def candidate_structures(pred: np.ndarray, data: NativeData, threshold: float, min_dist_km: float = 1.0,
                         min_len_km: float = 3.0, eda_candidates: pd.DataFrame | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Connected high-probability structures > ``min_dist_km`` from every mapped fault.

    These are *candidate unmapped structures* — exploratory, not confirmed faults.
    """
    from skimage.morphology import skeletonize

    g = data.grid
    p = clean_prediction(pred)
    d = ndimage.distance_transform_edt(~data.labels, sampling=(g.pixel_km_y, g.pixel_km_x))
    m = (p >= threshold) & data.valid & (d > min_dist_km)
    lab, n = ndimage.label(m, structure=np.ones((3, 3), bool))
    if n == 0:
        return pd.DataFrame(), np.zeros(data.shape, np.int32)
    sk = skeletonize(m)
    ids = np.arange(1, n + 1)
    length = ndimage.sum(sk, lab, ids) * g.pixel_km * 1.1
    area = ndimage.sum(m, lab, ids)
    pmean = ndimage.mean(p, lab, ids)
    pmax = ndimage.maximum(p, lab, ids)
    com = ndimage.center_of_mass(m, lab, ids)
    rc = np.array(com)
    x, y = g.rc_to_xy(rc[:, 0], rc[:, 1])
    df = pd.DataFrame({"cand_id": ids, "length_km": length, "area_px": area, "p_mean": pmean, "p_max": pmax,
                       "r": rc[:, 0], "c": rc[:, 1], "x": x, "y": y,
                       "dist_to_mapped_km": ndimage.minimum(d, lab, ids)})
    keep = df.length_km >= min_len_km
    df = df[keep].reset_index(drop=True)
    cand = np.where(np.isin(lab, df.cand_id.to_numpy()), lab, 0).astype(np.int32)
    if eda_candidates is not None and len(eda_candidates) and {"x_mid", "y_mid"} <= set(eda_candidates.columns):
        from scipy.spatial import cKDTree

        tree = cKDTree(eda_candidates[["x_mid", "y_mid"]].to_numpy())
        dd, ii = tree.query(df[["x", "y"]].to_numpy())
        df["nearest_eda_candidate_km"] = dd / 1000.0
        df["eda_candidate_idx"] = ii
    df["score"] = df.length_km * df.p_mean
    return df.sort_values("score", ascending=False).reset_index(drop=True), cand


def write_candidates_gpkg(cand_raster: np.ndarray, table: pd.DataFrame, data: NativeData, path) -> str:
    """Polygons of candidate structures (GeoPackage; Parquet+WKT fallback)."""
    from rasterio.features import shapes
    from shapely.geometry import shape

    geoms, ids = [], []
    for geom, val in shapes(cand_raster.astype(np.int32), mask=cand_raster > 0, transform=data.grid.transform):
        geoms.append(shape(geom))
        ids.append(int(val))
    df = pd.DataFrame({"cand_id": ids}).merge(table, on="cand_id", how="left")
    try:
        import geopandas as gpd

        gpd.GeoDataFrame(df, geometry=geoms, crs=data.grid.crs).to_file(path, driver="GPKG", layer="candidates")
        return str(path)
    except ImportError:
        df["wkt"] = [g.wkt for g in geoms]
        alt = str(path).rsplit(".", 1)[0] + ".parquet"
        df.to_parquet(alt, index=False)
        return alt


# ---------------------------------------------------------------------------
# error atlas
# ---------------------------------------------------------------------------
def native_windows(data: NativeData, eda_windows: pd.DataFrame) -> pd.DataFrame:
    """Convert Phase-1 windows (CRS bounds) to native pixel windows (same ground extent)."""
    inv = ~data.grid.transform
    rows = []
    for w in eda_windows.itertuples():
        c0, r0 = inv * (w.xmin, w.ymax)
        c1, r1 = inv * (w.xmax, w.ymin)
        rows.append({"name": w.name, "label": w.label, "r0": int(round(min(r0, r1))), "r1": int(round(max(r0, r1))),
                     "c0": int(round(min(c0, c1))), "c1": int(round(max(c0, c1))),
                     "xmin": w.xmin, "xmax": w.xmax, "ymin": w.ymin, "ymax": w.ymax})
    return pd.DataFrame(rows)


def objective_windows(blocks: pd.DataFrame, data: NativeData, n_each: int = 2) -> pd.DataFrame:
    if not len(blocks):
        return pd.DataFrame()
    b = blocks.sort_values("dti")
    picks = [("worst", r) for r in b.head(n_each).itertuples()] + [("best", r) for r in b.tail(n_each).itertuples()]
    rows = []
    for i, (kind, r) in enumerate(picks):
        (xmin, xmax, ymin, ymax) = data.grid.window_extent(r.r0, r.r1, r.c0, r.c1)
        rows.append({"name": f"O{i + 1}", "label": f"{kind} block DTI={r.dti:.3f}", "r0": r.r0, "r1": r.r1,
                     "c0": r.c0, "c1": r.c1, "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax})
    return pd.DataFrame(rows)


def _hillshade(a: np.ndarray, az=315.0, alt=45.0) -> np.ndarray:
    a = np.nan_to_num(a, nan=np.nanmedian(a) if np.isfinite(a).any() else 0)
    gy, gx = np.gradient(a)
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(az), np.radians(alt)
    return np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)


def atlas_figure(win, data: NativeData, primary: np.ndarray, gbm: np.ndarray, cnn: np.ndarray | None,
                 candidates: np.ndarray | None, primary_label: str = "det_elev (hillshade)",
                 names=("LightGBM", "U-Net"), vmax: float = 1.0):
    """Six panels A–F on a common extent with shared probability colour scale."""
    import matplotlib.pyplot as plt

    r0, r1, c0, c1 = int(win.r0), int(win.r1), int(win.c0), int(win.c1)
    ext = data.grid.window_extent(r0, r1, c0, c1)
    sl = (slice(r0, r1), slice(c0, c1))
    lab = data.labels[sl]
    fig, axs = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    a = axs.ravel()
    a[0].imshow(_hillshade(primary[sl]), cmap="gray", extent=ext)
    a[0].set_title(f"A  {primary_label}")
    a[1].imshow(np.where(lab, 1, np.nan), cmap="Greys", vmin=0, vmax=1, extent=ext, interpolation="nearest")
    a[1].set_title("B  mapped faults")
    for ax, m, t in ((a[2], gbm, f"C  {names[0]} P(fault)"), (a[3], cnn, f"D  {names[1]} P(fault)")):
        if m is None:
            ax.set_axis_off()
            ax.set_title(t + " (n/a)")
            continue
        im = ax.imshow(m[sl], cmap="magma", vmin=0, vmax=vmax, extent=ext)
        ax.contour(lab, levels=[0.5], colors="cyan", linewidths=0.4, extent=ext, origin="upper")
        ax.set_title(t)
    fig.colorbar(im, ax=[a[2], a[3]], shrink=0.6, label="probability (shared scale)")
    if cnn is not None:
        dd = np.nan_to_num(gbm[sl]) - np.nan_to_num(cnn[sl])
        im2 = a[4].imshow(dd, cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=ext)
        fig.colorbar(im2, ax=a[4], shrink=0.6, label=f"{names[0]} − {names[1]}")
    a[4].set_title("E  model disagreement")
    a[5].imshow(_hillshade(primary[sl]), cmap="gray", extent=ext, alpha=0.6)
    if candidates is not None:
        cm = np.where(candidates[sl] > 0, 1, np.nan)
        a[5].imshow(cm, cmap="autumn", vmin=0, vmax=1, extent=ext, alpha=0.9, interpolation="nearest")
    a[5].contour(lab, levels=[0.5], colors="k", linewidths=0.4, extent=ext, origin="upper")
    a[5].set_title("F  candidate unlabelled structures (> 1 km from mapped)")
    for ax in a:
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{win.name}: {win.label}")
    return fig


def strong_boundaries_without_labels(pred: np.ndarray, edge_feature: np.ndarray, data: NativeData,
                                     q_edge: float = 0.95, threshold: float = 0.5, min_dist_km: float = 2.0) -> dict:
    """Do confident predictions far from labels sit on strong geophysical edges?"""
    g = data.grid
    d = ndimage.distance_transform_edt(~data.labels, sampling=(g.pixel_km_y, g.pixel_km_x))
    p = clean_prediction(pred)
    far = data.valid & (d > min_dist_km)
    e = np.asarray(edge_feature)
    thr = np.nanquantile(e[data.valid][:: max(1, data.valid.sum() // 2_000_000)], q_edge)
    strong = far & (e >= thr)
    conf = far & (p >= threshold)
    return {"far_px": int(far.sum()), "confident_far_px": int(conf.sum()),
            "share_confident_on_strong_edge": float((conf & strong).sum() / max(conf.sum(), 1)),
            "base_rate_strong_edge": float(strong.sum() / max(far.sum(), 1)),
            "edge_quantile": q_edge}


def to_json(d: dict) -> str:
    return json.dumps(d, indent=2, default=float)
