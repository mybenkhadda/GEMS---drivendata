"""The single 100 m analysis grid and the base masks.

Contract (``check_grid``): the GeoDAWN raster defines the grid; it must be north-up, EPSG:32611, 100 m; the
official template must have exactly the same grid; a different shape than documented is a warning. Fault labels
on another grid are reprojected with nearest neighbour (categorical). Every DEM summary is aggregated with exact
area weights (block statistics of an aligned work grid), never by nearest sampling.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .inventory import band_names


@dataclass(frozen=True)
class Grid:
    crs_wkt: str
    epsg: int | None
    transform: tuple
    height: int
    width: int

    @property
    def crs(self):
        from rasterio.crs import CRS

        return CRS.from_wkt(self.crs_wkt)

    @property
    def affine(self):
        from affine import Affine

        return Affine(*self.transform[:6])

    @property
    def pixel_m(self) -> float:
        return float(abs(self.transform[0]))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        a, _, c, _, e, f = self.transform[:6]
        return c, f + e * self.height, c + a * self.width, f

    def profile(self) -> dict:
        return {"crs": self.crs, "transform": self.affine}

    def as_dict(self) -> dict:
        return {"epsg": self.epsg, "transform": list(self.transform[:6]), "height": self.height, "width": self.width,
                "pixel_m": self.pixel_m, "bounds": list(self.bounds), "crs_wkt": self.crs_wkt}

    @classmethod
    def from_dict(cls, d: dict) -> "Grid":
        return cls(d["crs_wkt"], d["epsg"], tuple(d["transform"]), d["height"], d["width"])

    def same(self, o: "Grid") -> bool:
        return (self.height, self.width) == (o.height, o.width) and self.epsg == o.epsg and \
            np.allclose(self.transform[:6], o.transform[:6], atol=1e-6)

    def extent_km(self) -> tuple[float, float, float, float]:
        l, b, r, t = self.bounds
        return l / 1e3, r / 1e3, b / 1e3, t / 1e3


def grid_of(src) -> Grid:
    return Grid(src.crs.to_wkt() if src.crs else "", src.crs.to_epsg() if src.crs else None,
                tuple(src.transform)[:6], src.height, src.width)


def check_grid(geodawn: Path, template: Path, cfg) -> tuple[Grid, list[str], list[str]]:
    import rasterio

    errors, warnings = [], []
    with rasterio.open(geodawn) as s:
        g = grid_of(s)
    ge = cfg["grid_expected"]
    if g.epsg != ge["epsg"]:
        errors.append(f"GeoDAWN CRS EPSG:{g.epsg} != EPSG:{ge['epsg']}")
    if abs(g.transform[1]) > 0 or abs(g.transform[3]) > 0:
        errors.append("GeoDAWN grid is rotated")
    if abs(g.pixel_m - ge["pixel_m"]) > 1e-6 or abs(abs(g.transform[4]) - ge["pixel_m"]) > 1e-6:
        errors.append(f"GeoDAWN pixel {g.pixel_m} m != {ge['pixel_m']} m")
    if (g.height, g.width) != (ge["height"], ge["width"]):
        warnings.append(f"GeoDAWN shape {(g.height, g.width)} differs from documented {(ge['height'], ge['width'])}")
    with rasterio.open(template) as s:
        tg = grid_of(s)
    if not tg.same(g):
        errors.append(f"template grid {tg.as_dict()} != GeoDAWN grid")
    return g, errors, warnings


def window_of_bounds(b, grid: Grid, pad: int = 0):
    """(r0, r1, c0, c1) of grid cells intersecting bounds (in grid CRS), or None."""
    l, bo, r, t = b
    a, _, c, _, e, f = grid.transform[:6]
    c0 = int(np.floor((l - c) / a)) - pad
    c1 = int(np.ceil((r - c) / a)) + pad
    r0 = int(np.floor((t - f) / e)) - pad
    r1 = int(np.ceil((bo - f) / e)) + pad
    r0, c0 = max(r0, 0), max(c0, 0)
    r1, c1 = min(r1, grid.height), min(c1, grid.width)
    return (r0, r1, c0, c1) if r1 > r0 and c1 > c0 else None


class GeoDawn:
    def __init__(self, path: Path, cfg):
        import rasterio

        self.path = Path(path)
        self.nod_abs = float(cfg["raw"]["nodata_abs"])
        self.min_share = float(cfg["raw"]["valid_min_band_share"])
        with rasterio.open(self.path) as s:
            self.grid = grid_of(s)
            self.names = band_names(s)
            self.nodata = s.nodata

    def read(self, name: str) -> np.ndarray:
        import rasterio

        with rasterio.open(self.path) as s:
            a = s.read(self.names.index(name) + 1).astype(np.float32)
        bad = ~np.isfinite(a) | (np.abs(a) >= self.nod_abs)
        if self.nodata is not None and np.isfinite(self.nodata):
            bad |= a == np.float32(self.nodata)
        a[bad] = np.nan
        return a

    def footprint(self) -> np.ndarray:
        cnt = np.zeros((self.grid.height, self.grid.width), np.int16)
        for n in self.names:
            cnt += np.isfinite(self.read(n))
        return cnt >= int(np.ceil(self.min_share * len(self.names)))


def read_labels(path: Path, grid: Grid, cfg) -> tuple[np.ndarray, np.ndarray, str]:
    """(fault mask, label-raster valid footprint, resampling note) on the analysis grid."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    with rasterio.open(path) as s:
        g = grid_of(s)
        a = s.read(1).astype(np.float32)
        nod = s.nodata
        if nod is not None and np.isfinite(nod):
            a[a == np.float32(nod)] = np.nan
        if g.same(grid):
            out, note = a, "same grid (no resampling)"
        else:
            out = np.full((grid.height, grid.width), np.nan, np.float32)
            reproject(a, out, src_transform=s.transform, src_crs=s.crs, src_nodata=np.nan,
                      dst_transform=grid.affine, dst_crs=grid.crs, dst_nodata=np.nan, resampling=Resampling.nearest)
            note = "reprojected nearest-neighbour (categorical labels)"
    return np.isfinite(out) & (out >= float(cfg["raw"]["label_min_value"])), np.isfinite(out), note


def read_template(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as s:
        a = s.read(1).astype(np.float32)
        nod = s.nodata
    v = np.isfinite(a)
    if nod is not None and np.isfinite(nod):
        v &= a != np.float32(nod)
    return v


def distance_to_edge_m(mask: np.ndarray, px: float) -> np.ndarray:
    from scipy import ndimage

    return (ndimage.distance_transform_edt(mask) * px).astype(np.float32)
