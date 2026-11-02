"""Safe reading and writing of large geospatial rasters.

* Bands are resolved by EXACT identifier (the part of the band description before ``" - "``), through
  ``gems_bench.g05_1.bands.resolve``: ``iso_grav_anom`` can never resolve to ``iso_grav_anom_hg``.
* Windows: a core tile plus a halo; reads are clipped to the raster and the missing halo is padded with NaN, so the
  arrays every feature function sees have the SAME geometry for edge and interior tiles. Feature functions treat NaN
  as "no data" (masked convolution), therefore processing tile by tile equals processing the whole raster as one
  padded tile, as long as the halo covers the operator support (``feature_store.required_halo_px``).
* Nodata: non-finite values and ``|x| > NODATA_ABS`` become NaN on read; outputs are float32 GeoTIFFs with NaN
  nodata, the source CRS and affine transform, DEFLATE + floating-point predictor, 256 px internal tiles.
* Tile cache: every computed tile is written to ``<dir>/<tile_id>.npz`` and THEN its manifest
  ``<tile_id>.json`` (sha256 of the npz, channel names, window). A tile counts as complete only when the manifest
  exists, its sha matches the file and its channel list matches - a crash between the two writes leaves an
  incomplete tile that is recomputed.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .contracts import ContractError, GridSpec

NODATA_ABS = 1e30


# =============================================================================== bands
def band_identifiers(dataset) -> list[str]:
    """Exact identifier of every band: tag ``band_name`` if present, else the description before ``" - "``."""
    from ..g05_1.bands import identifier

    out = []
    for i in range(1, dataset.count + 1):
        tags = dataset.tags(i)
        d = dataset.descriptions[i - 1] or tags.get("DESCRIPTION") or ""
        out.append(identifier(tags.get("band_name") or d or f"band_{i}"))
    return out


def band_descriptions(dataset) -> list[str]:
    return [dataset.descriptions[i - 1] or dataset.tags(i).get("DESCRIPTION") or "" for i in
            range(1, dataset.count + 1)]


def resolve_exact_band(dataset, name: str) -> int:
    """1-based index of band ``name`` by exact identifier equality (fails closed on missing / duplicate)."""
    from ..g05_1.bands import resolve

    return int(resolve([name], band_identifiers(dataset))[name.split(" - ")[0].strip()])


def resolve_bands(dataset, names) -> dict:
    from ..g05_1.bands import resolve

    return {k: int(v) for k, v in resolve(list(names), band_identifiers(dataset)).items()}


def clean(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float32)
    a[~np.isfinite(a) | (np.abs(a) > NODATA_ABS)] = np.nan
    return a


def read_band_full(path, index: int) -> np.ndarray:
    """One full band as float32 with NaN nodata (one band at a time: ~49 MB on the GeoDAWN grid)."""
    import rasterio

    with rasterio.open(path) as ds:
        a = ds.read(int(index), masked=True)
        return clean(np.ma.filled(a.astype(np.float32), np.nan))


# =============================================================================== windows
@dataclass(frozen=True)
class TileWindow:
    tile_id: str
    row0: int
    row1: int
    col0: int
    col1: int
    halo: int
    height: int                 # raster height (for clipping)
    width: int

    @property
    def core_shape(self) -> tuple[int, int]:
        return (self.row1 - self.row0, self.col1 - self.col0)

    @property
    def read_bounds(self) -> tuple[int, int, int, int]:
        """Halo window clipped to the raster: (r0, r1, c0, c1)."""
        return (max(self.row0 - self.halo, 0), min(self.row1 + self.halo, self.height),
                max(self.col0 - self.halo, 0), min(self.col1 + self.halo, self.width))

    @property
    def pads(self) -> tuple[int, int, int, int]:
        """(top, bottom, left, right) NaN padding that restores the full halo outside the raster."""
        r0, r1, c0, c1 = self.read_bounds
        return (r0 - (self.row0 - self.halo), (self.row1 + self.halo) - r1, c0 - (self.col0 - self.halo),
                (self.col1 + self.halo) - c1)

    @property
    def core(self) -> tuple[slice, slice]:
        """Core region inside the padded array (always offset by ``halo``)."""
        h, w = self.core_shape
        return (slice(self.halo, self.halo + h), slice(self.halo, self.halo + w))

    @property
    def grid_slice(self) -> tuple[slice, slice]:
        return (slice(self.row0, self.row1), slice(self.col0, self.col1))

    def to_dict(self) -> dict:
        return asdict(self)


def iter_windows(dataset_or_shape, tile_px: int, halo_px: int):
    """Row-major tiles of ``tile_px`` covering the raster (last row / column tiles are smaller)."""
    if hasattr(dataset_or_shape, "height"):
        h, w = int(dataset_or_shape.height), int(dataset_or_shape.width)
    else:
        h, w = (int(x) for x in dataset_or_shape)
    t = int(tile_px)
    for r in range(0, h, t):
        for c in range(0, w, t):
            yield TileWindow(f"r{r:05d}_c{c:05d}", r, min(r + t, h), c, min(c + t, w), int(halo_px), h, w)


def pad_window(a: np.ndarray, win: TileWindow) -> np.ndarray:
    """NaN-pad a read (bands, h, w) or (h, w) array to the full halo geometry."""
    top, bot, left, right = win.pads
    if not any((top, bot, left, right)):
        return a
    pw = ((top, bot), (left, right)) if a.ndim == 2 else ((0, 0), (top, bot), (left, right))
    return np.pad(a, pw, mode="constant", constant_values=np.nan)


def read_aligned_window(dataset, bands, win: TileWindow) -> np.ndarray:
    """(n_bands, core_h + 2 halo, core_w + 2 halo) float32, NaN nodata, NaN beyond the raster edge."""
    from rasterio.windows import Window

    r0, r1, c0, c1 = win.read_bounds
    a = dataset.read([int(b) for b in bands], window=Window(c0, r0, c1 - c0, r1 - r0), masked=True)
    a = clean(np.ma.filled(a.astype(np.float32), np.nan))
    return pad_window(a, win)


def crop_core(a: np.ndarray, win: TileWindow) -> np.ndarray:
    rs, cs = win.core
    return a[..., rs, cs]


# =============================================================================== writing
def geotiff_profile(grid: GridSpec, count: int, block_px: int = 256, compress: str = "deflate",
                    dtype: str = "float32") -> dict:
    return {"driver": "GTiff", "height": grid.height, "width": grid.width, "count": int(count), "dtype": dtype,
            "crs": grid.crs, "transform": grid.affine(), "nodata": np.nan if dtype.startswith("float") else None,
            "tiled": True, "blockxsize": int(block_px), "blockysize": int(block_px), "compress": compress,
            "predictor": 3 if dtype.startswith("float") else 2, "BIGTIFF": "IF_SAFER"}


def write_feature_raster(path, arrays: dict, grid: GridSpec, tags: dict | None = None, block_px: int = 256,
                         compress: str = "deflate") -> Path:
    """Full-grid float32 layers -> one multiband GeoTIFF (band description = layer name), written atomically."""
    import rasterio

    from .persistence import atomic_write

    names = list(arrays)
    for n in names:
        if np.shape(arrays[n]) != grid.shape:
            raise ContractError(f"{n}: shape {np.shape(arrays[n])} != grid {grid.shape}")

    def w(t):
        with rasterio.open(t, "w", **geotiff_profile(grid, len(names), block_px, compress)) as ds:
            for i, n in enumerate(names, 1):
                ds.write(np.asarray(arrays[n], np.float32), i)
                ds.set_band_description(i, n)
            ds.update_tags(**{k: str(v) for k, v in (tags or {}).items()})

    return atomic_write(Path(path), w)


class GroupWriter:
    """Window-wise writer of one multiband feature GeoTIFF (opened once, closed atomically)."""

    def __init__(self, path, names, grid: GridSpec, tags: dict | None = None, block_px: int = 256,
                 compress: str = "deflate"):
        import rasterio

        self.path = Path(path)
        self.tmp = self.path.with_name(self.path.stem + ".partial" + self.path.suffix)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.names = list(names)
        self.ds = rasterio.open(self.tmp, "w", **geotiff_profile(grid, len(self.names), block_px, compress))
        for i, n in enumerate(self.names, 1):
            self.ds.set_band_description(i, n)
        self.ds.update_tags(**{k: str(v) for k, v in (tags or {}).items()})

    def write(self, win: TileWindow, stack: np.ndarray) -> None:
        from rasterio.windows import Window

        if stack.shape != (len(self.names), *win.core_shape):
            raise ContractError(f"{win.tile_id}: stack {stack.shape} != ({len(self.names)}, {win.core_shape})")
        self.ds.write(np.asarray(stack, np.float32),
                      window=Window(win.col0, win.row0, win.col1 - win.col0, win.row1 - win.row0))

    def close(self) -> Path:
        import os

        self.ds.close()
        os.replace(self.tmp, self.path)
        return self.path

    def abort(self) -> None:
        try:
            self.ds.close()
        finally:
            self.tmp.unlink(missing_ok=True)


# =============================================================================== tile cache
class TileCache:
    """``<dir>/<tile_id>.npz`` + ``<tile_id>.json`` manifest (sha256, names, window); see module docstring."""

    def __init__(self, directory, names):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.names = list(names)

    def paths(self, tile_id: str) -> tuple[Path, Path]:
        return self.dir / f"{tile_id}.npz", self.dir / f"{tile_id}.json"

    def complete(self, win: TileWindow) -> bool:
        from .persistence import read_json, sha256_file

        npz, js = self.paths(win.tile_id)
        m = read_json(js)
        if not isinstance(m, dict) or not npz.exists():
            return False
        if m.get("names") != self.names or m.get("window") != win.to_dict():
            return False
        return m.get("sha256") == sha256_file(npz)

    def save(self, win: TileWindow, stack: np.ndarray) -> None:
        from .persistence import atomic_write, sha256_file, write_json

        if stack.shape != (len(self.names), *win.core_shape):
            raise ContractError(f"{win.tile_id}: tile stack {stack.shape} != ({len(self.names)}, {win.core_shape})")
        npz, js = self.paths(win.tile_id)
        atomic_write(npz, lambda t: np.savez(t, stack=np.asarray(stack, np.float32)))
        with np.load(npz) as z:                                   # verify before marking complete
            back = z["stack"]
            if back.shape != stack.shape or not np.array_equal(np.isnan(back), np.isnan(stack)):
                raise ContractError(f"{win.tile_id}: tile read-back differs from what was written")
        write_json(js, {"tile_id": win.tile_id, "names": self.names, "window": win.to_dict(),
                        "sha256": sha256_file(npz), "shape": list(stack.shape)})

    def load(self, win: TileWindow) -> np.ndarray:
        npz, _ = self.paths(win.tile_id)
        with np.load(npz) as z:
            return z["stack"]

    def clear(self) -> None:
        import shutil

        shutil.rmtree(self.dir, ignore_errors=True)


# =============================================================================== reading features
def read_window_bands(path, band_indices, window: tuple[int, int, int, int]) -> np.ndarray:
    """(n, h, w) float32 of 1-based ``band_indices`` over ``window`` = (r0, r1, c0, c1)."""
    import rasterio
    from rasterio.windows import Window

    r0, r1, c0, c1 = (int(x) for x in window)
    with rasterio.open(path) as ds:
        a = ds.read([int(b) for b in band_indices], window=Window(c0, r0, c1 - c0, r1 - r0))
    return np.asarray(a, np.float32)


def read_points(path, band_indices, rows: np.ndarray, cols: np.ndarray, block_px: int = 512) -> np.ndarray:
    """(n_points, n_bands) values at integer pixel coordinates, read block by block (no full-raster load)."""
    import rasterio
    from rasterio.windows import Window

    rows, cols = np.asarray(rows, np.int64), np.asarray(cols, np.int64)
    out = np.full((rows.size, len(band_indices)), np.nan, np.float32)
    if not rows.size:
        return out
    br, bc = rows // block_px, cols // block_px
    key = br * 100000 + bc
    order = np.argsort(key, kind="stable")
    ks = key[order]
    bounds = np.flatnonzero(np.diff(ks)) + 1
    with rasterio.open(path) as ds:
        for grp in np.split(order, bounds):
            r0, c0 = int(br[grp[0]] * block_px), int(bc[grp[0]] * block_px)
            r1, c1 = min(r0 + block_px, ds.height), min(c0 + block_px, ds.width)
            a = ds.read([int(b) for b in band_indices], window=Window(c0, r0, c1 - c0, r1 - r0))
            out[grp] = a[:, rows[grp] - r0, cols[grp] - c0].T
    return out


# =============================================================================== checks
def verify_raster_alignment(path, grid: GridSpec, names=None, epsg: int | None = None) -> list[str]:
    """Problems of one written raster against the grid contract (empty list = aligned)."""
    import rasterio

    from .contracts import validate_crs, validate_transform

    probs = []
    with rasterio.open(path) as ds:
        if (ds.height, ds.width) != grid.shape:
            probs.append(f"shape {(ds.height, ds.width)} != {grid.shape}")
        try:
            t = ds.transform
            validate_transform((t.a, t.b, t.c, t.d, t.e, t.f), grid.transform)
        except ContractError as e:
            probs.append(str(e))
        if epsg is not None:
            try:
                validate_crs(ds.crs.to_string() if ds.crs else "", epsg)
            except ContractError as e:
                probs.append(str(e))
        if names is not None and list(ds.descriptions) != list(names):
            probs.append(f"band names differ from the registry ({len(ds.descriptions)} vs {len(names)})")
        if ds.dtypes[0] != "float32":
            probs.append(f"dtype {ds.dtypes[0]} != float32")
    return probs


def compare_tile_seams(path, band: int, tile_px: int, max_lines: int = 64) -> dict:
    """Seam diagnostic of a tiled product: mean |difference| between adjacent pixel columns / rows ACROSS tile
    boundaries vs the same statistic one pixel inside the tiles. A ratio far above 1 flags a halo problem."""
    import rasterio

    with rasterio.open(path) as ds:
        a = ds.read(int(band)).astype(np.float64)
    h, w = a.shape
    cs = [c for c in range(tile_px, w, tile_px)][:max_lines]
    rs = [r for r in range(tile_px, h, tile_px)][:max_lines]

    def mad(x, y):
        d = np.abs(x - y)
        d = d[np.isfinite(d)]
        return float(d.mean()) if d.size else np.nan

    across = [mad(a[:, c - 1], a[:, c]) for c in cs] + [mad(a[r - 1], a[r]) for r in rs]
    inside = [mad(a[:, c - 2], a[:, c - 1]) for c in cs] + [mad(a[r - 2], a[r - 1]) for r in rs]
    acr, ins = np.nanmean(across) if across else np.nan, np.nanmean(inside) if inside else np.nan
    return {"band": int(band), "n_seams": len(across), "mad_across": float(acr), "mad_inside": float(ins),
            "ratio": float(acr / ins) if ins and np.isfinite(ins) and ins > 0 else np.nan}


def tags_json(tags: dict) -> dict:
    return {k: (json.dumps(v, default=str) if not isinstance(v, str) else v) for k, v in tags.items()}
