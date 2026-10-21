"""Optional external Earth-observation experiments (EO01-EO03): interfaces and preprocessing only.

Nothing in this module downloads data or weights. The user places files under ``data/external/`` after
checking licence and competition eligibility (``ELIGIBILITY_CHECKLIST``), then the notebook calls these
functions on local paths.

EO01  1 m DEM (GEMS-provided links). ``select_aois`` proposes a few target windows from existing Phase-3/4
      outputs (strong mapped faults, short missed faults, weak-terrain faults, candidate lineaments, diverse
      regions). ``dem1m_features`` computes genuinely high-resolution scarp / slope / curvature statistics on the
      1 m DEM and aggregates them to the native 100 m grid with explicit statistics (max, p90, std, steep
      fraction) — never by upsampling the 100 m elevation. Coverage is partial, so EO01 is evaluated inside the
      AOIs only (``aoi_evidence``), not added to whole-map models.
EO02  Sentinel-2 L2A (user-supplied cloud-masked composites). Band-ratio geology proxies (SWIR clay/carbonate,
      ferric/ferrous iron), NDVI for vegetation masking, multispectral texture, spectral-boundary strength.
      A spectral boundary is surface-material contrast, not a fault.
EO03  Foundation-model embeddings (TerraMind Tiny / Prithvi-EO 2.0 Tiny): frozen feature extraction from the
      modalities those models were trained on only. Geophysical bands are rejected by design.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ELIGIBILITY_CHECKLIST = [
    "Dataset / weights licence permits use in a prize competition and redistribution of derived submissions",
    "GEMS rules list the source as allowed external data (or it is GEMS-provided, e.g. the 1 m DEM links)",
    "Acquisition dates do not post-date the competition data cutoff if the rules require it",
    "No external fault inventory or geological map is used as an input (label leakage of the hidden truth)",
    "Provenance (URL, version, date, checksum) recorded in data/external/SOURCES.md before use",
]


# ---------------------------------------------------------------------------
# EO01 — AOI selection and 1 m DEM features
# ---------------------------------------------------------------------------
def select_aois(ctx, support: dict[str, np.ndarray], reference: str, candidates: pd.DataFrame | None = None,
                n_per_kind: int = 2, half_km: float = 3.0) -> pd.DataFrame:
    """Small target windows (grid CRS bounds) for targeted 1 m DEM downloads.

    ``support`` = per-label-pixel TP support from phase4_eval.recovery_table (model -> array);
    ``candidates`` = Phase-3/4 candidate lineament table (optional, rank_new column).
    """
    from .phase4_eval import label_pixel_table

    lpt = label_pixel_table(ctx).assign(k=support[reference])
    seg = lpt.groupby("seg_id").agg(r=("r", "median"), c=("c", "median"), length_km=("length_km", "first"),
                                    k=("k", "mean"), archetype=("primary_archetype", "first"),
                                    region=("region", "first")).reset_index()
    seg = seg[seg.seg_id >= 0]
    picks = {
        "strong_mapped": seg[seg.k >= 0.8].sort_values("length_km", ascending=False),
        "short_missed": seg[(seg.length_km < 2) & (seg.k < 0.3)].sort_values("length_km", ascending=False),
        "weak_terrain_fault": seg[(seg.archetype != "terrain_visible") & (seg.k < 0.5)].sort_values("length_km", ascending=False),
    }
    rows = []
    for kind, s in picks.items():
        used_regions = set()
        for r in s.itertuples():
            if len([x for x in rows if x["kind"] == kind]) >= n_per_kind:
                break
            if r.region in used_regions:                       # one per region -> geological diversity
                continue
            used_regions.add(r.region)
            rows.append({"kind": kind, "seg_id": int(r.seg_id), "r": int(r.r), "c": int(r.c), "length_km": r.length_km,
                         "k_reference": r.k, "region": r.region})
    if candidates is not None and len(candidates) and "rank_new" in candidates:
        for r in candidates.dropna(subset=["rank_new"]).sort_values("rank_new").head(n_per_kind).itertuples():
            rows.append({"kind": "candidate_lineament", "seg_id": -1, "r": int(r.r_mid), "c": int(r.c_mid),
                         "length_km": r.length_km, "k_reference": np.nan, "region": None})
    df = pd.DataFrame(rows)
    if not len(df):
        return df
    g = ctx.data.grid
    x, y = g.rc_to_xy(df.r.to_numpy(), df.c.to_numpy())
    h = half_km * 1000
    df["xmin"], df["xmax"], df["ymin"], df["ymax"] = x - h, x + h, y - h, y + h
    df["crs"] = str(g.crs)
    df["dem_file"] = ""                                          # filled by the user after download
    return df


@dataclass
class DEMFeatureConfig:
    work_res_m: float = 2.0             # resample the 1 m DEM to this resolution before filtering
    sigmas_m: tuple = (4.0, 16.0)       # Gaussian scales for slope / curvature
    steep_deg: float = 25.0             # scarp-like slope threshold
    stats: tuple = ("max", "p90", "std", "steep_frac")


def dem1m_features(dem_path: Path, ctx, cfg: DEMFeatureConfig = DEMFeatureConfig()) -> dict[str, np.ndarray]:
    """High-resolution slope / curvature / scarp statistics aggregated to the native 100 m grid (NaN outside coverage).

    Aggregation is per 100 m cell over the true high-resolution pixels: max, 90th percentile and std of slope,
    fraction of pixels steeper than ``steep_deg`` (scarp density), and the same for |curvature|.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    from scipy import ndimage

    g = ctx.data.grid
    prof = ctx.data.profile
    with rasterio.open(dem_path) as src:
        scale = cfg.work_res_m / abs(src.transform.a)
        h, w = int(src.height / scale), int(src.width / scale)
        z = src.read(1, out_shape=(h, w), resampling=Resampling.average, masked=True).astype(np.float64)
        tr = src.transform * src.transform.scale(src.width / w, src.height / h)
        crs = src.crs
    m = ~np.ma.getmaskarray(z)
    z = np.ma.filled(z, np.nan)
    from gems_eda.enhance import fill_nan

    f = fill_nan(z)
    px = cfg.work_res_m
    layers = {}
    for s_m in cfg.sigmas_m:
        s = s_m / px
        dzr = ndimage.gaussian_filter(f, s, order=(1, 0)) / px
        dzc = ndimage.gaussian_filter(f, s, order=(0, 1)) / px
        slope = np.degrees(np.arctan(np.hypot(dzr, dzc)))
        lap = np.abs(ndimage.gaussian_laplace(f, s)) / px ** 2
        slope[~m], lap[~m] = np.nan, np.nan
        layers[f"slope@{s_m:g}m"] = slope
        layers[f"abs_laplacian@{s_m:g}m"] = lap
    out = {}
    cell = int(round(g.pixel_km * 1000 / px))                      # high-res pixels per 100 m cell (per axis)
    for name, a in layers.items():
        H, W = (a.shape[0] // cell) * cell, (a.shape[1] // cell) * cell
        blocks = a[:H, :W].reshape(H // cell, cell, W // cell, cell).transpose(0, 2, 1, 3).reshape(H // cell, W // cell, -1)
        stats = {"max": np.nanmax(blocks, -1), "p90": np.nanpercentile(blocks, 90, axis=-1), "std": np.nanstd(blocks, -1)}
        if name.startswith("slope"):
            stats["steep_frac"] = np.nanmean(blocks > cfg.steep_deg, -1)
        btr = tr * tr.scale(cell, cell)
        for st, arr in stats.items():
            if st not in cfg.stats:
                continue
            dst = np.full(ctx.data.shape, np.nan, np.float32)
            reproject(arr.astype(np.float32), dst, src_transform=btr, src_crs=crs, src_nodata=np.nan,
                      dst_transform=prof["transform"], dst_crs=prof["crs"], dst_nodata=np.nan,
                      resampling=Resampling.average)
            out[f"dem1m:{name}:{st}"] = dst
    return out


def aoi_evidence(ctx, feats: dict[str, np.ndarray], compare: dict[str, np.ndarray], aoi_mask: np.ndarray,
                 tol_m: float = 300.0) -> pd.DataFrame:
    """Within the AOIs only: AUC of each 1 m feature and of comparable 100 m features for 'within tol of a fault'."""
    from scipy import ndimage
    from sklearn.metrics import roc_auc_score

    g = ctx.data.grid
    near = ndimage.distance_transform_edt(~ctx.data.labels, sampling=(g.pixel_km_y * 1000, g.pixel_km_x * 1000)) <= tol_m
    rows = []
    for src, dd in (("dem1m", feats), ("native100m", compare)):
        for n, a in dd.items():
            m = aoi_mask & np.isfinite(a)
            y = near[m]
            if y.sum() < 10 or (~y).sum() < 10:
                continue
            rows.append({"source": src, "feature": n, "n_px": int(m.sum()), "pos_frac": float(y.mean()),
                         "auc": float(roc_auc_score(y, a[m]))})
    return pd.DataFrame(rows).sort_values("auc", ascending=False)


# ---------------------------------------------------------------------------
# EO02 — Sentinel-2 geology proxies (user-supplied L2A composites)
# ---------------------------------------------------------------------------
S2_SCL_INVALID = (0, 1, 3, 8, 9, 10, 11)    # no data, saturated, cloud shadow, cloud med/high, cirrus, snow


def s2_cloud_mask(scl: np.ndarray) -> np.ndarray:
    return ~np.isin(scl, S2_SCL_INVALID)


def s2_indices(b: dict[str, np.ndarray], ndvi_max: float = 0.4) -> dict[str, np.ndarray]:
    """Band-ratio proxies from surface reflectance (keys B02..B12). Vegetated pixels (NDVI > ndvi_max) are NaN.

    clay_carbonate = B11 / B12 (Al-OH / carbonate absorption near 2.2-2.3 um), ferric = B04 / B02,
    ferrous = (B12 / B08) + (B03 / B04), ndvi for masking only.
    """
    eps = 1e-6
    ndvi = (b["B08"] - b["B04"]) / (b["B08"] + b["B04"] + eps)
    veg = ndvi > ndvi_max
    out = {"clay_carbonate": b["B11"] / (b["B12"] + eps), "ferric": b["B04"] / (b["B02"] + eps),
           "ferrous": b["B12"] / (b["B08"] + eps) + b["B03"] / (b["B04"] + eps)}
    for k in out:
        out[k] = np.where(veg, np.nan, out[k]).astype(np.float32)
    out["ndvi"] = ndvi.astype(np.float32)
    return out


def spectral_boundary(index: np.ndarray, grid, scale_km: float = 0.3) -> np.ndarray:
    """Edge strength of a spectral index (THDR): a surface-material contrast, NOT fault evidence by itself."""
    from gems_eda.enhance import Enhancer

    return Enhancer(grid).thdr(index, scale_km)


def texture_std(a: np.ndarray, size_px: int = 5) -> np.ndarray:
    from scipy import ndimage

    m = np.isfinite(a)
    f = np.where(m, a, 0.0)
    n = ndimage.uniform_filter(m.astype(float), size_px)
    m1 = ndimage.uniform_filter(f, size_px) / np.maximum(n, 1e-6)
    m2 = ndimage.uniform_filter(f * f, size_px) / np.maximum(n, 1e-6)
    out = np.sqrt(np.clip(m2 - m1 * m1, 0, None))
    out[~m] = np.nan
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# EO03 — foundation-model embeddings (frozen; supported modalities only)
# ---------------------------------------------------------------------------
SUPPORTED_MODALITIES = {
    "terramind_tiny": {"S2L2A", "S1GRD", "DEM"},
    "prithvi_eo2_tiny": {"HLS"},           # HLS-harmonised optical bands only
}
GEOPHYSICAL_TOKENS = ("mag", "rtp", "tmi", "grav", "basement", "depth_to_base", "cond", "geod", "eq_")


class FoundationEmbeddingExtractor:
    """Frozen-encoder embedding extraction. Weights are loaded from a local path only (never downloaded)."""

    def __init__(self, model: str, weights_path: Path, modality: str, band_names: list[str]):
        if model not in SUPPORTED_MODALITIES:
            raise KeyError(f"unknown model {model}; supported: {list(SUPPORTED_MODALITIES)}")
        if modality not in SUPPORTED_MODALITIES[model]:
            raise ValueError(f"{model} was not trained on modality {modality!r}")
        bad = [b for b in band_names if any(t in b.lower() for t in GEOPHYSICAL_TOKENS)]
        if bad:
            raise ValueError(f"geophysical bands cannot be fed into spectral slots: {bad}")
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"place the licensed weights at {weights_path} (no automatic download)")
        self.model, self.weights_path, self.modality, self.band_names = model, Path(weights_path), modality, band_names
        self.encoder = None

    def load(self):
        """Model-specific loading is left to the user's installed package (terratorch / prithvi); kept explicit."""
        raise NotImplementedError("EO03 is deferred (P3): implement loading for the installed package once EO02 "
                                  "inputs exist and the licence check passes.")

    def embed(self, tiles: np.ndarray) -> np.ndarray:
        if self.encoder is None:
            self.load()
        raise NotImplementedError
