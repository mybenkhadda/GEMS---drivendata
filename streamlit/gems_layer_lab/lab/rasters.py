"""Bounded-window access to the competition feature raster and user-provided aligned layers."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window, transform as window_transform
from rasterio.enums import Resampling

FALLBACK_BANDS = [
    "mag_anom", "rtp", "tmi", "tmi_hg", "tmi_vg", "tc",
    "iso_grav_anom", "iso_grav_anom_hg", "iso_grav_anom_vg", "iso_grav_anom_slope",
    "det_elev", "det_elev_slope", "depth_to_base_surf", "cond_surf",
    "geod_2ndinv", "geod_shearrate", "geod_dilaterate", "deq_n100a15", "ieq_n100a15"
]
MAX_AOI_PIXELS = 1024 * 1024
MAX_NPZ_UNCOMPRESSED_BYTES = 160 * 1024 * 1024


@dataclass(frozen=True)
class LayerSource:
    name: str
    path: str
    band: int = 1
    key: str | None = None
    kind: str = "tif"


def _detect_name(src, idx):
    tags = src.tags(idx)
    desc = src.descriptions[idx - 1] or tags.get("DESCRIPTION") or ""
    title = tags.get("band_name") or desc.split(" - ")[0].strip()
    if not title and src.count == len(FALLBACK_BANDS):
        title = FALLBACK_BANDS[idx - 1]
    return title or f"band_{idx}"


def _grid_signature(src):
    return (src.height, src.width, src.crs, tuple(src.transform)[:6])


def aligned(src, target):
    return src.height == target.height and src.width == target.width and src.crs == target.crs and np.allclose(
        tuple(src.transform)[:6], tuple(target.transform)[:6], rtol=0, atol=1e-7
    )


def catalog(feature_tif: str, extra_entries: list[tuple[str, str, str | None]] | None = None):
    """Read raster metadata, not raster pixels. Extras: (name,path,npz_key)."""
    feature_tif = str(Path(feature_tif).expanduser())
    if not Path(feature_tif).is_file():
        raise FileNotFoundError(f"Feature TIFF not found: {feature_tif}")
    result = {}
    with rasterio.open(feature_tif) as src:
        details = {"height": src.height, "width": src.width, "crs": str(src.crs),
                   "resolution": (abs(src.transform.a), abs(src.transform.e)),
                   "transform": tuple(src.transform)[:6], "bands": src.count, "bounds": tuple(src.bounds)}
        for idx in range(1, src.count + 1):
            name = _detect_name(src, idx)
            if not name.isidentifier() or name.startswith("_") or name in result:
                raise ValueError(f"Duplicate or non-identifier band name {name!r} at {idx}; update source tags")
            result[name] = LayerSource(name, feature_tif, idx)
        for name, path, key in (extra_entries or []):
            if not name.isidentifier() or name.startswith("_") or name in result:
                raise ValueError(f"Invalid or duplicate external layer name: {name}")
            path = str(Path(path).expanduser())
            if not Path(path).is_file():
                raise FileNotFoundError(f"Missing extra layer {name}: {path}")
            ext = Path(path).suffix.lower()
            if ext in (".tif", ".tiff"):
                with rasterio.open(path) as extra:
                    if not aligned(extra, src):
                        raise ValueError(f"{name}: raster not aligned with feature TIFF (CRS/transform/shape)")
                    if extra.count != 1:
                        raise ValueError(f"{name}: extra TIFF must contain exactly one band")
                result[name] = LayerSource(name, path, kind="tif")
            elif ext == ".npy":
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                if array.shape != (src.height, src.width) or not np.issubdtype(array.dtype, np.number):
                    raise ValueError(f"{name}: NPY must be numeric with shape {(src.height, src.width)}")
                result[name] = LayerSource(name, path, kind="npy")
            elif ext == ".npz":
                # Inspect header to avoid decompressing huge arrays in preflight.
                with np.load(path, allow_pickle=False) as z:
                    if key is None or key not in z.files:
                        raise ValueError(f"{name}: specify an existing NPZ array key as name=path.npz::key")
                    # npz __getitem__ decompresses; allow only manageable files.
                    if Path(path).stat().st_size > MAX_NPZ_UNCOMPRESSED_BYTES:
                        raise ValueError(f"{name}: large NPZ compressed file; convert to aligned TIFF or NPY first")
                    array = z[key]
                    if array.shape != (src.height, src.width) or array.nbytes > MAX_NPZ_UNCOMPRESSED_BYTES:
                        raise ValueError(f"{name}: NPZ array is too large or does not match the native grid")
                result[name] = LayerSource(name, path, key=key, kind="npz")
            else:
                raise ValueError(f"{name}: supported formats: .tif, .tiff, .npy, .npz::key")
    return result, details


def parse_extra_entries(text: str):
    entries = []
    for n, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Extra layer line {n}: use name=/path/to/layer.tif")
        name, location = [s.strip() for s in line.split("=", 1)]
        path, key = (location.rsplit("::", 1) + [None])[:2] if "::" in location else (location, None)
        entries.append((name, path, key))
    return entries


def aoi_window(height, width, row, col, side_km, pixel_m=100):
    # Padded/cropped at borders. Pixel-metre value comes from source transform.
    side_px = max(10, int(round(float(side_km) * 1000.0 / pixel_m)))
    side_px = min(side_px, 1000, height, width)
    row_off = max(0, min(height - side_px, int(round(row - side_px / 2))))
    col_off = max(0, min(width - side_px, int(round(col - side_px / 2))))
    return Window(col_off=col_off, row_off=row_off, width=side_px, height=side_px)


def padded(window, height, width, halo):
    x0 = max(0, int(window.col_off) - halo)
    y0 = max(0, int(window.row_off) - halo)
    x1 = min(width, int(window.col_off + window.width) + halo)
    y1 = min(height, int(window.row_off + window.height) + halo)
    extended = Window(x0, y0, x1-x0, y1-y0)
    roi = (slice(int(window.row_off) - y0, int(window.row_off+window.height) - y0),
           slice(int(window.col_off) - x0, int(window.col_off+window.width) - x0))
    return extended, roi


def _masked_to_float(array):
    return np.ma.filled(array.astype(np.float32), np.nan) if np.ma.isMaskedArray(array) else np.asarray(array, np.float32)


def read_layers(sources: dict[str, LayerSource], names: list[str] | set[str], window: Window):
    if int(window.width) * int(window.height) > MAX_AOI_PIXELS * 3:
        raise ValueError("Selected AOI plus halo exceeds safe processing size")
    by_path = {}
    for name in names:
        source = sources[name]
        by_path.setdefault(source.path, []).append(source)
    out = {}
    for path, batch in by_path.items():
        if batch[0].kind == "tif":
            with rasterio.open(path) as src:
                for source in batch:
                    out[source.name] = _masked_to_float(src.read(source.band, window=window, masked=True))
        elif batch[0].kind == "npy":
            a = np.load(path, mmap_mode="r", allow_pickle=False)
            for source in batch:
                out[source.name] = np.asarray(a[int(window.row_off):int(window.row_off+window.height),
                                               int(window.col_off):int(window.col_off+window.width)], np.float32).copy()
        elif batch[0].kind == "npz":
            with np.load(path, allow_pickle=False) as z:
                for source in batch:
                    a = z[source.key]
                    out[source.name] = np.array(a[int(window.row_off):int(window.row_off+window.height),
                                                  int(window.col_off):int(window.col_off+window.width)], dtype=np.float32)
    return out


def read_faults(fault_tif: str, feature_tif: str, window: Window):
    if not fault_tif or not Path(fault_tif).is_file():
        return None
    with rasterio.open(feature_tif) as ref, rasterio.open(fault_tif) as src:
        if not aligned(src, ref):
            raise ValueError("Fault labels do not align with numerical features")
        a = src.read(1, masked=True, window=window)
        return (_masked_to_float(a) > 0) & np.isfinite(_masked_to_float(a))


def export_tiff(layer, window, feature_tif, name):
    from rasterio.io import MemoryFile
    layer = np.asarray(layer, np.float32)
    with rasterio.open(feature_tif) as src:
        tr = window_transform(window, src.transform)
        profile = dict(driver="GTiff", height=layer.shape[0], width=layer.shape[1], count=1,
                       dtype="float32", crs=src.crs, transform=tr, nodata=np.nan,
                       compress="deflate", predictor=3)
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(layer, 1)
            dst.set_band_description(1, name)
        return mem.read()


def overview(feature_tif: str, band: int, side: int = 240):
    with rasterio.open(feature_tif) as src:
        data = src.read(band, out_shape=(side, side), masked=True, resampling=Resampling.nearest)
    return _masked_to_float(data)
