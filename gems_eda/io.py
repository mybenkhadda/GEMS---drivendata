"""Raster / vector I/O on a single shared analysis grid."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject


@dataclass
class Grid:
    """Georeferencing of the working grid plus ground-distance helpers."""

    transform: Affine
    crs: object
    height: int
    width: int
    pixel_km_x: float
    pixel_km_y: float
    is_geographic: bool

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def pixel_km(self) -> float:
        return math.sqrt(self.pixel_km_x * self.pixel_km_y)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return self.window_extent(0, self.height, 0, self.width)

    @property
    def aspect(self) -> float:
        """Matplotlib aspect giving equal ground distance on both axes."""
        if not self.is_geographic:
            return 1.0
        lat = 0.5 * (self.extent[2] + self.extent[3])
        return 1.0 / math.cos(math.radians(lat))

    @property
    def area_km2(self) -> float:
        return self.height * self.width * self.pixel_km_x * self.pixel_km_y

    def km_to_px(self, km: float) -> float:
        return km / self.pixel_km

    def window_extent(self, r0: int, r1: int, c0: int, c1: int) -> tuple[float, float, float, float]:
        """(left, right, bottom, top) of a pixel window, for imshow(extent=...)."""
        a = self.transform
        left, top = a * (c0, r0)
        right, bottom = a * (c1, r1)
        return (min(left, right), max(left, right), min(bottom, top), max(bottom, top))

    def rc_to_xy(self, rows, cols):
        """Pixel-centre (row, col) -> CRS (x, y); accepts fractional pixels."""
        a = self.transform
        rows = np.asarray(rows, dtype=float) + 0.5
        cols = np.asarray(cols, dtype=float) + 0.5
        return a.c + cols * a.a + rows * a.b, a.f + cols * a.d + rows * a.e

    def rc_to_km(self, rows, cols):
        """Local ground coordinates (east, north) in km, origin at the upper-left pixel."""
        return np.asarray(cols, float) * self.pixel_km_x, -np.asarray(rows, float) * self.pixel_km_y


def _pixel_km(transform: Affine, crs, height: int) -> tuple[float, float, bool]:
    dx, dy = abs(transform.a), abs(transform.e)
    if crs is not None and crs.is_geographic:
        lat = transform.f + transform.e * height / 2
        return dx * 111.320 * math.cos(math.radians(lat)), dy * 110.574, True
    factor = 1.0
    try:
        factor = float(crs.linear_units_factor[1])
    except Exception:  # unknown units: assume metres
        pass
    return dx * factor / 1000.0, dy * factor / 1000.0, False


def _out_shape(height: int, width: int, max_dim: int | None) -> tuple[int, int]:
    if not max_dim or max(height, width) <= max_dim:
        return height, width
    f = math.ceil(max(height, width) / max_dim)
    return math.ceil(height / f), math.ceil(width / f)


@dataclass
class RasterStack:
    """Multiband float32 stack with NaN for nodata, addressed by band name."""

    data: np.ndarray
    names: list[str]
    descriptions: dict[str, str]
    grid: Grid
    native_shape: tuple[int, int]
    signal_notes: dict[str, str] = field(default_factory=dict)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.data[self.names.index(name)]

    def __contains__(self, name: str) -> bool:
        return name in self.names

    @property
    def valid(self) -> np.ndarray:
        """Pixels where at least half of the bands carry data."""
        return np.isfinite(self.data).sum(0) >= math.ceil(0.5 * len(self.names))

    def band_table(self) -> pd.DataFrame:
        from .config import band_modality

        rows = []
        for i, n in enumerate(self.names):
            v = self.data[i]
            rows.append({
                "band": n,
                "modality": band_modality(n),
                "description": self.descriptions.get(n, ""),
                "valid_frac": float(np.isfinite(v).mean()),
            })
        return pd.DataFrame(rows)


def _band_names(src) -> tuple[list[str], dict[str, str]]:
    names, desc = [], {}
    for i in range(1, src.count + 1):
        tags = src.tags(i)
        d = src.descriptions[i - 1] or tags.get("DESCRIPTION") or ""
        name = tags.get("band_name") or (d.split(" - ")[0].strip() if d else f"band_{i}")
        names.append(name)
        desc[name] = tags.get("description", d)
    return names, desc


def load_feature_stack(path: Path, max_dim: int | None = None) -> RasterStack:
    """Read the multiband feature GeoTIFF, block-averaging to ``max_dim`` if needed."""
    with rasterio.open(path) as src:
        h, w = _out_shape(src.height, src.width, max_dim)
        data = src.read(out_shape=(src.count, h, w), resampling=Resampling.average, masked=True)
        arr = np.ma.filled(data.astype("float32"), np.nan)
        transform = src.transform * Affine.scale(src.width / w, src.height / h)
        names, desc = _band_names(src)
        crs, native = src.crs, (src.height, src.width)
    arr[~np.isfinite(arr) | (np.abs(arr) > 1e30)] = np.nan
    kx, ky, geo = _pixel_km(transform, crs, h)
    return RasterStack(arr, names, desc, Grid(transform, crs, h, w, kx, ky, geo), native)


def load_on_grid(path: Path, grid: Grid, resampling=Resampling.max) -> tuple[np.ndarray, dict]:
    """Read band 1 of ``path`` onto ``grid`` (max-resampling keeps thin fault traces)."""
    with rasterio.open(path) as src:
        info = {
            "file": Path(path).name,
            "native_shape": (src.height, src.width),
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
            "crs": str(src.crs),
            "bands": src.count,
        }
        raw = np.ma.filled(src.read(1, masked=True).astype("float32"), np.nan)
        if src.crs == grid.crs and src.transform == grid.transform and src.shape == grid.shape:
            return raw, info
        out = np.full(grid.shape, np.nan, dtype="float32")
        reproject(
            raw, out,
            src_transform=src.transform, src_crs=src.crs, src_nodata=np.nan,
            dst_transform=grid.transform, dst_crs=grid.crs, dst_nodata=np.nan,
            resampling=resampling,
        )
    return out, info


def write_geotiff(path: Path, arrays: dict[str, np.ndarray], grid: Grid) -> Path:
    """Write named float32 layers as one multiband, band-described GeoTIFF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff", height=grid.height, width=grid.width, count=len(arrays),
        dtype="float32", crs=grid.crs, transform=grid.transform, nodata=np.nan,
        compress="deflate", predictor=3, tiled=True, blockxsize=256, blockysize=256,
    )
    with rasterio.open(path, "w", **profile) as dst:
        for i, (name, arr) in enumerate(arrays.items(), start=1):
            dst.write(np.asarray(arr, dtype="float32"), i)
            dst.set_band_description(i, name)
    return path


def read_geotiff(path: Path) -> dict[str, np.ndarray]:
    """Reload a GeoTIFF written by :func:`write_geotiff` as {band description: array}."""
    with rasterio.open(path) as src:
        return {
            (src.descriptions[i] or f"band_{i + 1}"): np.ma.filled(src.read(i + 1, masked=True), np.nan)
            for i in range(src.count)
        }


def paths_to_lines(paths: list[np.ndarray], grid: Grid):
    """Ordered (row, col) pixel paths -> shapely LineStrings in the raster CRS."""
    from shapely.geometry import LineString

    lines = []
    for p in paths:
        x, y = grid.rc_to_xy(p[:, 0], p[:, 1])
        lines.append(LineString(np.column_stack([x, y])) if len(p) > 1 else None)
    return lines


def write_lines(path: Path, paths: list[np.ndarray], attrs: pd.DataFrame, grid: Grid, layer: str) -> Path:
    """Write traced polylines with attributes to GeoPackage (Parquet+WKT fallback)."""
    path = Path(path)
    lines = paths_to_lines(paths, grid)
    try:
        import geopandas as gpd

        gdf = gpd.GeoDataFrame(attrs.reset_index(drop=True), geometry=lines, crs=grid.crs)
        gdf = gdf[gdf.geometry.notna()]
        gdf.to_file(path, layer=layer, driver="GPKG")
        return path
    except ImportError:
        alt = path.with_suffix(".parquet")
        df = attrs.reset_index(drop=True).copy()
        df["wkt"] = [g.wkt if g is not None else None for g in lines]
        df.to_parquet(alt, index=False)
        return alt
