"""Readers for the raw GeoDAWN raster, the fault labels and the official template.

Rules (RAW_DATA_CONTRACT.md):
* band name = tag ``band_name`` or the band description up to `` - ``;
* a value is nodata when it is not finite, equals the declared nodata or has ``|x| >= raw.nodata_abs``;
* data footprint = pixels where >= ``raw.valid_min_band_share`` of the bands are valid;
* fault pixel = label value >= ``raw.label_min_value`` (reprojected nearest-neighbour if its grid differs);
* template-valid = finite template pixels; the template grid must equal the feature grid.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Grid:
    crs_wkt: str
    epsg: int | None
    transform: tuple            # affine (a, b, c, d, e, f)
    height: int
    width: int

    @property
    def pixel_m(self) -> float:
        return float(abs(self.transform[0]))

    @property
    def affine(self):
        from affine import Affine

        return Affine(*self.transform[:6])

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        a, _, c, _, e, f = self.transform[:6]
        return c, f + e * self.height, c + a * self.width, f

    def profile(self) -> dict:
        from rasterio.crs import CRS

        return {"crs": CRS.from_wkt(self.crs_wkt), "transform": self.affine, "height": self.height,
                "width": self.width}

    def as_dict(self) -> dict:
        return {"epsg": self.epsg, "transform": list(self.transform[:6]), "height": self.height,
                "width": self.width, "pixel_m": self.pixel_m}

    def same_as(self, other: "Grid", tol: float = 1e-6) -> bool:
        return (self.height, self.width) == (other.height, other.width) and \
            np.allclose(self.transform[:6], other.transform[:6], atol=tol) and self.epsg == other.epsg


def grid_of(src) -> Grid:
    epsg = src.crs.to_epsg() if src.crs else None
    return Grid(src.crs.to_wkt() if src.crs else "", epsg, tuple(src.transform)[:6], src.height, src.width)


def band_names(src) -> list[str]:
    out = []
    for i in range(1, src.count + 1):
        tags = src.tags(i)
        desc = src.descriptions[i - 1] or ""
        name = tags.get("band_name") or desc.split(" - ")[0].strip() or f"band_{i}"
        out.append(name)
    return out


def _clean(a: np.ndarray, nodata, nodata_abs: float) -> np.ndarray:
    a = a.astype(np.float32, copy=False)
    bad = ~np.isfinite(a) | (np.abs(a) >= nodata_abs)
    if nodata is not None and np.isfinite(nodata):
        bad |= a == np.float32(nodata)
    a[bad] = np.nan
    return a


class GeoDawn:
    """Lazy band access to the GeoDAWN numerical-features raster."""

    def __init__(self, path: Path, cfg: dict):
        import rasterio

        self.path = Path(path)
        self.nodata_abs = float(cfg["raw"]["nodata_abs"])
        self.min_share = float(cfg["raw"]["valid_min_band_share"])
        with rasterio.open(self.path) as src:
            self.grid = grid_of(src)
            self.names = band_names(src)
            self.nodata = src.nodata

    def read(self, name: str) -> np.ndarray:
        import rasterio

        idx = self.names.index(name) + 1
        with rasterio.open(self.path) as src:
            return _clean(src.read(idx), self.nodata, self.nodata_abs)

    def footprint(self) -> np.ndarray:
        import rasterio

        cnt = np.zeros((self.grid.height, self.grid.width), np.int16)
        with rasterio.open(self.path) as src:
            for i in range(1, src.count + 1):
                cnt += np.isfinite(_clean(src.read(i), self.nodata, self.nodata_abs))
        return cnt >= int(np.ceil(self.min_share * len(self.names)))


def read_on_grid(path: Path, grid: Grid, nearest: bool = True) -> tuple[np.ndarray, Grid, float | None]:
    """Band 1 of ``path`` on ``grid`` (reprojected when the grids differ). Returns (array, source grid, nodata)."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    with rasterio.open(path) as src:
        g = grid_of(src)
        nod = src.nodata
        if g.same_as(grid):
            return src.read(1).astype(np.float32), g, nod
        a = src.read(1).astype(np.float32)
        if nod is not None:
            a[a == np.float32(nod)] = np.nan
        out = np.full((grid.height, grid.width), np.nan, np.float32)
        reproject(a, out, src_transform=src.transform, src_crs=src.crs, src_nodata=np.nan, dst_transform=grid.affine,
                  dst_crs=grid.profile()["crs"], dst_nodata=np.nan,
                  resampling=Resampling.nearest if nearest else Resampling.bilinear)
        return out, g, np.nan


def read_labels(path: Path, grid: Grid, cfg: dict) -> np.ndarray:
    a, _, nod = read_on_grid(path, grid, nearest=True)
    if nod is not None and np.isfinite(nod):
        a = np.where(a == np.float32(nod), np.nan, a)
    return np.isfinite(a) & (a >= float(cfg["raw"]["label_min_value"]))


def read_template(path: Path, grid: Grid) -> tuple[np.ndarray, dict]:
    """Template-valid mask and the template's own profile (the submission is written with it)."""
    import rasterio

    with rasterio.open(path) as src:
        g = grid_of(src)
        if not g.same_as(grid):
            raise ValueError(f"template grid {g.as_dict()} differs from the GeoDAWN grid {grid.as_dict()}")
        a = src.read(1).astype(np.float32)
        nod = src.nodata
        prof = {"crs": src.crs, "transform": src.transform, "height": src.height, "width": src.width}
    valid = np.isfinite(a)
    if nod is not None and np.isfinite(nod):
        valid &= a != np.float32(nod)
    return valid, prof
