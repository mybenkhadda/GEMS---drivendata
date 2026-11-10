"""100 m geophysical master grid: georeferencing, coordinate conversion and alignment.

Pixel convention: (row, col) index the pixel whose upper-left corner is ``transform * (col, row)``;
``pixel_to_world`` returns the pixel CENTRE, ``world_to_pixel`` returns the containing pixel (floor), so the
round trip ``world_to_pixel(*pixel_to_world(r, c)) == (r, c)`` holds exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

RESAMPLING_BY_KIND = {"continuous": "bilinear", "categorical": "nearest", "mask": "nearest",
                      "probability": "bilinear", "label": "nearest"}


class GridError(ValueError):
    pass


@dataclass
class GridSpec:
    crs: str                      # WKT or "EPSG:xxxx"
    transform: tuple              # affine (a, b, c, d, e, f)
    shape: tuple                  # (height, width)
    valid: np.ndarray | None = field(default=None, repr=False)

    @property
    def affine(self):
        from rasterio.transform import Affine

        return Affine(*self.transform[:6])

    @property
    def res(self) -> tuple[float, float]:
        return abs(self.transform[0]), abs(self.transform[4])

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        a, _, c, _, e, f = self.transform[:6]
        h, w = self.shape
        xs, ys = (c, c + a * w), (f, f + e * h)
        return min(xs), min(ys), max(xs), max(ys)

    @property
    def epsg(self) -> int | None:
        from rasterio.crs import CRS

        return CRS.from_user_input(self.crs).to_epsg()

    def profile(self, count: int = 1, dtype: str = "float32") -> dict:
        return {"driver": "GTiff", "crs": self.crs, "transform": self.affine, "height": self.shape[0],
                "width": self.shape[1], "count": count, "dtype": dtype}

    def as_dict(self) -> dict:
        return {"crs": self.crs, "transform": list(self.transform[:6]), "shape": list(self.shape),
                "bounds": list(self.bounds), "res": list(self.res), "epsg": self.epsg,
                "valid_px": int(self.valid.sum()) if self.valid is not None else None}

    def same_geometry(self, other: "GridSpec", atol: float = 1e-6) -> bool:
        from rasterio.crs import CRS

        return (tuple(self.shape) == tuple(other.shape)
                and np.allclose(self.transform[:6], other.transform[:6], atol=atol)
                and CRS.from_user_input(self.crs) == CRS.from_user_input(other.crs))


def load_master_grid(reference_raster, valid_min_band_share: float = 0.5, nodata_abs: float = 1e30,
                     expected: dict | None = None, window=None) -> GridSpec:
    """GridSpec of the reference (GeoDAWN) raster; valid = pixels finite in >= ``valid_min_band_share`` of bands.

    Raises ``GridError`` when the CRS is missing or the grid differs from ``expected`` (epsg / res / shape).
    ``window`` (r0, r1, c0, c1) returns the cropped sub-grid (SMOKE_TEST) after validating the FULL grid.
    """
    if window is not None:
        return crop_grid(load_master_grid(reference_raster, valid_min_band_share, nodata_abs, expected), window)
    import rasterio

    nodata_abs, valid_min_band_share = float(nodata_abs), float(valid_min_band_share)
    with rasterio.open(reference_raster) as ds:
        if ds.crs is None:
            raise GridError(f"{reference_raster}: raster has no CRS - cannot georeference the master grid")
        t = ds.transform
        if abs(t.b) > 1e-9 or abs(t.d) > 1e-9:
            raise GridError(f"{reference_raster}: rotated transform is not supported")
        cnt = np.zeros(ds.shape, np.int32)
        for b in range(1, ds.count + 1):
            a = ds.read(b, masked=True).astype(np.float32).filled(np.nan)
            cnt += (np.isfinite(a) & (np.abs(a) < nodata_abs)).astype(np.int32)
        valid = cnt >= max(1, int(np.ceil(valid_min_band_share * ds.count)))
        g = GridSpec(ds.crs.to_wkt(), tuple(t)[:6], tuple(ds.shape), valid)
    if expected:
        problems = []
        if expected.get("expected_epsg") and g.epsg != expected["expected_epsg"]:
            problems.append(f"EPSG {g.epsg} != {expected['expected_epsg']}")
        if expected.get("expected_res_m") and not np.allclose(g.res, expected["expected_res_m"]):
            problems.append(f"resolution {g.res} != {expected['expected_res_m']}")
        if expected.get("expected_shape") and tuple(g.shape) != tuple(expected["expected_shape"]):
            problems.append(f"shape {g.shape} != {tuple(expected['expected_shape'])}")
        if problems:
            raise GridError(f"{reference_raster}: master grid mismatch: " + "; ".join(problems))
    return g


def crop_grid(grid: GridSpec, window) -> GridSpec:
    """Sub-grid (r0, r1, c0, c1) of ``grid`` with the shifted transform (used by the SMOKE_TEST subset)."""
    r0, r1, c0, c1 = (int(x) for x in window)
    a, b, c, d, e, f = grid.transform[:6]
    t = (a, b, c + a * c0, d, e, f + e * r0)
    v = grid.valid[r0:r1, c0:c1].copy() if grid.valid is not None else None
    return GridSpec(grid.crs, t, (r1 - r0, c1 - c0), v)


def raster_window_on_grid(ds, grid: GridSpec):
    """rasterio Window of ``ds`` covering ``grid`` exactly (same CRS / resolution, integer offset), else GridError."""
    from rasterio.crs import CRS
    from rasterio.windows import Window

    if ds.crs is None or CRS.from_user_input(ds.crs) != CRS.from_user_input(grid.crs):
        raise GridError(f"{ds.name}: CRS differs from the grid")
    t = ds.transform
    if not np.allclose((t.a, t.e), (grid.transform[0], grid.transform[4]), atol=1e-6):
        raise GridError(f"{ds.name}: resolution {ds.res} differs from the grid {grid.res}")
    oc = (grid.transform[2] - t.c) / t.a
    orow = (grid.transform[5] - t.f) / t.e
    if abs(oc - round(oc)) > 1e-6 or abs(orow - round(orow)) > 1e-6:
        raise GridError(f"{ds.name}: sub-pixel offset to the grid")
    oc, orow = int(round(oc)), int(round(orow))
    h, w = grid.shape
    if orow < 0 or oc < 0 or orow + h > ds.height or oc + w > ds.width:
        raise GridError(f"{ds.name}: grid window ({orow}, {oc}, {h}, {w}) outside the raster {ds.shape}")
    return Window(oc, orow, w, h)


def read_on_grid(path, grid: GridSpec, band: int = 1, nodata_abs: float = 1e30) -> np.ndarray:
    """Band of ``path`` exactly on ``grid`` (full raster or the SMOKE sub-window); NaN = nodata."""
    import rasterio

    with rasterio.open(path) as ds:
        win = raster_window_on_grid(ds, grid)
        a = ds.read(band, window=win, masked=True).astype(np.float32).filled(np.nan)
    a[np.abs(a) >= nodata_abs] = np.nan
    return a


def pixel_to_world(row, col, grid: GridSpec):
    """Pixel centre coordinates (x, y) in the grid CRS (vectorised)."""
    a, _, c, _, e, f = grid.transform[:6]
    return c + a * (np.asarray(col) + 0.5), f + e * (np.asarray(row) + 0.5)


def world_to_pixel(x, y, grid: GridSpec):
    """(row, col) of the pixel containing (x, y); may fall outside the grid (check with ``inside``)."""
    a, _, c, _, e, f = grid.transform[:6]
    col = np.floor((np.asarray(x, float) - c) / a).astype(np.int64)
    row = np.floor((np.asarray(y, float) - f) / e).astype(np.int64)
    return row, col


def inside(row, col, grid: GridSpec):
    h, w = grid.shape
    row, col = np.asarray(row), np.asarray(col)
    return (row >= 0) & (row < h) & (col >= 0) & (col < w)


@dataclass
class AlignmentReport:
    aligned: bool
    same_crs: bool
    same_shape: bool
    transform_max_diff: float
    pixel_offset: tuple
    resolution_ratio: tuple
    notes: list

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def validate_alignment(source: GridSpec, target: GridSpec, atol: float = 1e-6) -> AlignmentReport:
    from rasterio.crs import CRS

    same_crs = CRS.from_user_input(source.crs) == CRS.from_user_input(target.crs)
    same_shape = tuple(source.shape) == tuple(target.shape)
    diff = float(np.max(np.abs(np.asarray(source.transform[:6]) - np.asarray(target.transform[:6]))))
    ta, ts = target.transform, source.transform
    off = ((ts[5] - ta[5]) / ta[4], (ts[2] - ta[2]) / ta[0])
    ratio = (abs(ts[4] / ta[4]), abs(ts[0] / ta[0]))
    notes = []
    if not same_crs:
        notes.append("different CRS: reprojection required")
    if same_crs and not np.allclose(ratio, np.round(ratio)) and not np.allclose(1 / np.array(ratio), np.round(1 / np.array(ratio))):
        notes.append("non-integer resolution ratio")
    if same_crs and not np.allclose(off, np.round(off), atol=1e-6):
        notes.append("sub-pixel offset: grids are not nested")
    return AlignmentReport(same_crs and same_shape and diff <= atol, same_crs, same_shape, diff, off, ratio, notes)


def align_raster(source, target: GridSpec, kind: str = "continuous", band: int = 1, nodata=None,
                 nodata_policy: str = "nan") -> np.ndarray:
    """Reproject one band of ``source`` (path or (array, GridSpec)) onto ``target``.

    ``kind`` selects the resampling (continuous / probability -> bilinear; categorical / mask / label -> nearest).
    Pixels without source support are NaN (``nodata_policy='nan'``) - never zero-filled.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    if kind not in RESAMPLING_BY_KIND:
        raise GridError(f"unknown alignment kind {kind}")
    if nodata_policy != "nan":
        raise GridError("only nodata_policy='nan' is supported (missing data is never filled)")
    rs = getattr(Resampling, RESAMPLING_BY_KIND[kind])
    dst = np.full(target.shape, np.nan, np.float32)
    if isinstance(source, tuple):
        arr, sg = source
        src = np.asarray(arr, np.float32)
        reproject(src, dst, src_transform=sg.affine, src_crs=sg.crs, dst_transform=target.affine, dst_crs=target.crs,
                  src_nodata=np.nan, dst_nodata=np.nan, resampling=rs)
        return dst
    with rasterio.open(source) as ds:
        if ds.crs is None:
            raise GridError(f"{source}: no CRS - cannot align")
        nd = ds.nodata if nodata is None else nodata
        src = ds.read(band).astype(np.float32)
        if nd is not None and np.isfinite(nd):
            src[src == nd] = np.nan
        reproject(src, dst, src_transform=ds.transform, src_crs=ds.crs, dst_transform=target.affine,
                  dst_crs=target.crs, src_nodata=np.nan, dst_nodata=np.nan, resampling=rs)
    return dst
