"""Data inventory and geospatial contracts (sections 03-04).

Inputs are discovered by *content*, not filename alone: the GeoDAWN raster must carry the documented band
descriptions, the known-fault raster must be a single integer-like band on the same grid, and the template must
share the grid. No silent reprojection: any grid mismatch raises.

``GeoData`` holds the grid, masks and a memory-mapped float32 copy of the raw stack (NaN = nodata) on local disk,
so later stages read bands without keeping the whole cube in RAM.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import atomic_save_table, atomic_write_json, file_fingerprint

OFFICIAL_NAMES = {
    "raw": "gems-geodawn-numerical-features.tif",
    "known": "existing_faults.tif",
    "template": "example_submission.tif",
}
EPSG = 32611
PIXEL_M = 100.0

# Modality of every documented GeoDAWN band (names from the raster's GDAL descriptions; gems_eda.config).
MODALITY = {
    "mag_anom": "magnetic",
    "rtp": "magnetic",
    "tmi": "magnetic",
    "tmi_hg": "magnetic",
    "tmi_vg": "magnetic",
    "tc": "magnetic",
    "iso_grav_anom": "gravity",
    "iso_grav_anom_hg": "gravity",
    "iso_grav_anom_vg": "gravity",
    "iso_grav_anom_slope": "gravity",
    "det_elev": "terrain",
    "det_elev_slope": "terrain",
    "depth_to_base_surf": "depth",
    "cond_surf": "conductivity",
    "geod_2ndinv": "geodetic",
    "geod_shearrate": "geodetic",
    "geod_dilaterate": "geodetic",
    "deq_n100a15": "seismicity",
    "ieq_n100a15": "seismicity",
}

# Section 04: physical interpretation of every raw variable. ``status`` says how sure the interpretation is:
# "documented" = standard meaning of the field name in GeoDAWN / potential-field practice; "inferred" = from the
# name and the project EDA only; the organizers' data dictionary should be checked before publication.
BAND_DOCS = {
    "mag_anom": dict(
        meaning="Magnetic anomaly (TMI minus a reference field)",
        status="documented",
        fault_mechanism="Juxtaposition of rocks with different magnetisation; offset or truncated "
        "anomalies; demagnetised (altered) fault zones produce lows",
        alternatives="Lithological contacts, intrusions, volcanic flows, cultural noise",
        scale="0.5-20 km; deeper sources broader",
        noise="Line-levelling and survey-boundary steps",
    ),
    "rtp": dict(
        meaning="Reduced-to-pole magnetic anomaly",
        status="documented",
        fault_mechanism="Centres anomalies over sources, so gradients sit over contacts and faults",
        alternatives="Remanence breaks the RTP assumption and shifts edges",
        scale="0.5-20 km",
        noise="RTP amplifies N-S noise at low inclination (moderate here)",
    ),
    "tmi": dict(
        meaning="Total magnetic intensity",
        status="documented",
        fault_mechanism="Same physics as mag_anom, asymmetric at Nevada inclination (~63 deg)",
        alternatives="Regional field, survey merges",
        scale="0.5-50 km",
        noise="Survey boundaries",
    ),
    "tmi_hg": dict(
        meaning="Horizontal gradient magnitude of TMI",
        status="documented",
        fault_mechanism="Maxima over steep contacts",
        alternatives="Any contact, not only faults",
        scale="0.2-5 km",
        noise="Amplifies short-wavelength noise; common source with tmi",
    ),
    "tmi_vg": dict(
        meaning="Vertical gradient of TMI",
        status="documented",
        fault_mechanism="Sharpens shallow sources; zero crossings near contacts",
        alternatives="Shallow lithology, culture",
        scale="0.2-3 km",
        noise="Strong noise amplification",
    ),
    "tc": dict(
        meaning="Undocumented magnetic-family band (grouped with magnetics by the project EDA)",
        status="inferred",
        fault_mechanism="Unknown until documented",
        alternatives="Unknown",
        scale="Unknown",
        noise="Unknown",
    ),
    "iso_grav_anom": dict(
        meaning="Isostatic residual gravity anomaly",
        status="documented",
        fault_mechanism="Density contrast across basin-bounding faults (basin fill vs bedrock)",
        alternatives="Lithology, basin geometry without faulting",
        scale="2-30 km; station spacing limits resolution",
        noise="Sparse stations -> interpolation",
    ),
    "iso_grav_anom_hg": dict(
        meaning="Horizontal gradient of isostatic gravity",
        status="documented",
        fault_mechanism="Maxima over steep density contacts (range-front faults)",
        alternatives="Any density contact",
        scale="1-10 km",
        noise="Gridding artefacts between stations; common source with iso_grav_anom",
    ),
    "iso_grav_anom_vg": dict(
        meaning="Vertical gradient of isostatic gravity",
        status="documented",
        fault_mechanism="Shallow density edges",
        alternatives="Shallow lithology",
        scale="1-5 km",
        noise="High; common source",
    ),
    "iso_grav_anom_slope": dict(
        meaning="Slope (gradient magnitude) of isostatic gravity",
        status="inferred",
        fault_mechanism="As iso_grav_anom_hg",
        alternatives="Any density contact",
        scale="1-10 km",
        noise="Common source with iso_grav_anom",
    ),
    "det_elev": dict(
        meaning="Detrended elevation (regional trend removed)",
        status="inferred",
        fault_mechanism="Range fronts, scarps, linear valleys, offset ridges",
        alternatives="Erosion, drainage incision, ridgelines, lithology-controlled relief",
        scale="0.1-10 km",
        noise="100 m resampling removes scarps narrower than a pixel",
    ),
    "det_elev_slope": dict(
        meaning="Slope of detrended elevation",
        status="inferred",
        fault_mechanism="Slope breaks at range fronts and scarps",
        alternatives="Canyon walls, ridges",
        scale="0.1-5 km",
        noise="Common source with det_elev",
    ),
    "depth_to_base_surf": dict(
        meaning="Depth to basement (base of basin fill) surface",
        status="inferred",
        fault_mechanism="Steps in basement depth across basin-bounding faults",
        alternatives="Model artefacts from gravity inversion (shares source with gravity)",
        scale="2-30 km",
        noise="Model product; not independent of gravity",
    ),
    "cond_surf": dict(
        meaning="Surface electrical conductivity",
        status="inferred",
        fault_mechanism="Fluids or clay alteration along fault zones; lithology contrasts",
        alternatives="Playas, groundwater, clay-rich sediments",
        scale="0.5-20 km",
        noise="Survey coverage; no claim on permeability",
    ),
    "geod_2ndinv": dict(
        meaning="Second invariant of geodetic strain rate",
        status="inferred",
        fault_mechanism="Regional deformation localisation",
        alternatives="Interpolation of sparse GNSS",
        scale="tens of km (context only)",
        noise="Very smooth",
    ),
    "geod_shearrate": dict(
        meaning="Geodetic shear strain rate",
        status="inferred",
        fault_mechanism="Walker Lane shear zones",
        alternatives="Interpolation",
        scale="tens of km",
        noise="Very smooth",
    ),
    "geod_dilaterate": dict(
        meaning="Geodetic dilatation rate",
        status="inferred",
        fault_mechanism="Extensional domains",
        alternatives="Interpolation",
        scale="tens of km",
        noise="Very smooth",
    ),
    "deq_n100a15": dict(
        meaning="Earthquake density (name suggests 100-neighbour, 15 km kernel)",
        status="inferred",
        fault_mechanism="Active seismogenic structures",
        alternatives="Catalogue completeness, swarms, induced events",
        scale="~15 km kernel (context)",
        noise="Catalogue bias",
    ),
    "ieq_n100a15": dict(
        meaning="Second earthquake-density product (definition undocumented)",
        status="inferred",
        fault_mechanism="As deq_n100a15",
        alternatives="As deq_n100a15",
        scale="~15 km kernel",
        noise="Catalogue bias",
    ),
}

PRIMARY_FIELD = {
    "magnetic": "rtp",
    "gravity": "iso_grav_anom",
    "terrain": "det_elev",
    "depth": "depth_to_base_surf",
    "conductivity": "cond_surf",
}
# Bands that are derived from another band (common source): agreement between them is not independent support.
COMMON_SOURCE = {
    "tmi_hg": "tmi",
    "tmi_vg": "tmi",
    "rtp": "tmi",
    "mag_anom": "tmi",
    "iso_grav_anom_hg": "iso_grav_anom",
    "iso_grav_anom_vg": "iso_grav_anom",
    "iso_grav_anom_slope": "iso_grav_anom",
    "det_elev_slope": "det_elev",
    "depth_to_base_surf": "iso_grav_anom",
}


def make_grid(transform, crs, height: int, width: int):
    """A ``gems_eda.io.Grid`` (needed by ``gems_bench.folds``) for a projected metre grid."""
    from gems_eda.io import Grid

    return Grid(
        transform=transform,
        crs=crs,
        height=int(height),
        width=int(width),
        pixel_km_x=abs(transform.a) / 1000.0,
        pixel_km_y=abs(transform.e) / 1000.0,
        is_geographic=False,
    )


def check_grid(*srcs) -> None:
    """Identical CRS, transform and shape; EPSG:32611 at 100 m. Raises on any mismatch."""
    a = srcs[0]
    for b in srcs[1:]:
        if (a.width, a.height) != (b.width, b.height):
            raise ValueError(f"shape mismatch: {a.name} {(a.height, a.width)} vs {b.name} {(b.height, b.width)}")
        if a.transform != b.transform:
            raise ValueError(f"transform mismatch: {a.name} vs {b.name} (no silent reprojection)")
        if a.crs != b.crs:
            raise ValueError(f"CRS mismatch: {a.name} {a.crs} vs {b.name} {b.crs}")
    if a.crs is None or a.crs.to_epsg() != EPSG:
        raise ValueError(f"expected EPSG:{EPSG}, got {a.crs}")
    if not (abs(abs(a.transform.a) - PIXEL_M) < 0.1 and abs(abs(a.transform.e) - PIXEL_M) < 0.1):
        raise ValueError(f"expected {PIXEL_M:g} m pixels, got {a.transform.a}, {a.transform.e}")


# =============================================================================== discovery
def canonical_band_name(description: str | None, index: int) -> str:
    """Map descriptive GDAL band text to the stable GeoDAWN numerical-feature key.

    The official TIF may use 'rtp - reduced to pole magnetic data...'
    rather than 'rtp'. Original source metadata remains untouched.
    """
    raw = str(description or "").strip().lower()
    if not raw:
        return f"band_{index}"
    head = raw.split(" - ", 1)[0].strip()
    return head if head in MODALITY else raw


def _probe(path: Path) -> dict:
    import rasterio

    with rasterio.open(path) as src:
        desc = [canonical_band_name(d, i + 1) for i, d in enumerate(src.descriptions)]
        win = rasterio.windows.Window(0, 0, min(src.width, 512), min(src.height, 512))
        sample = src.read(1, window=win, masked=True).astype(np.float64).filled(np.nan)
        return {
            "path": str(path),
            "count": src.count,
            "descriptions": desc,
            "shape": (src.height, src.width),
            "crs": str(src.crs),
            "transform": tuple(src.transform)[:6],
            "dtype": src.dtypes[0],
            "sample_unique": int(len(np.unique(sample[np.isfinite(sample)])[:50])),
            "sample_integer_like": bool(np.all(np.mod(sample[np.isfinite(sample)], 1) == 0)),
        }


def _role_ok(role: str, info: dict) -> bool:
    if role == "raw":
        return info["count"] >= 10 and {"rtp", "iso_grav_anom"} <= set(info["descriptions"])
    if role == "known":
        return info["count"] == 1 and info["sample_integer_like"]
    if role == "template":
        return info["count"] == 1
    return False


def discover_inputs(root: str | Path, given: dict | None = None) -> dict:
    """Locate the three official files under ``root`` (explicit paths win). Content-verified; ambiguous -> raise."""
    given = {k: v for k, v in (given or {}).items() if v}
    out, notes = {}, []
    for role, name in OFFICIAL_NAMES.items():
        if role in given:
            p = Path(given[role])
            if not p.exists():
                raise FileNotFoundError(f"{role}: {p}")
            info = _probe(p)
            if not _role_ok(role, info):
                raise ValueError(
                    f"{role} file {p} does not look like {name}: {info['count']} bands {info['descriptions'][:5]}"
                )
            out[role] = str(p)
            continue
        cands = sorted(Path(root).rglob(name)) if Path(root).exists() else []
        cands = [c for c in cands if ".partial" not in c.name and "/.Trash" not in str(c)]
        good = [c for c in cands if _role_ok(role, _probe(c))]
        if not good:
            raise FileNotFoundError(f"no verified {name} under {root}; set the path explicitly")
        if len(good) > 1:
            fps = {str(c): file_fingerprint(c)["hash"] for c in good}
            if len(set(fps.values())) > 1:
                raise ValueError(f"several different {name} files under {root}: {list(fps)}; set the path explicitly")
            notes.append(f"{name}: {len(good)} identical copies, using {good[0]}")
        out[role] = str(good[0])
    out["notes"] = notes
    return out


# =============================================================================== data container
@dataclass
class GeoData:
    raw_path: str
    known_path: str
    template_path: str
    names: list
    shape: tuple
    transform: object
    crs: object
    profile: dict
    template_valid: np.ndarray  # template finite (competition bounds)
    valid: np.ndarray  # template finite AND at least one raw band finite
    labels: np.ndarray  # mapped faults (value >= 1), inside valid
    raw_cache: str = ""  # local float32 memmap (bands, h, w), NaN = nodata
    coverage: dict = field(default_factory=dict)  # band -> finite mask fraction
    _raw: np.ndarray | None = field(default=None, repr=False)

    @property
    def grid(self):
        return make_grid(self.transform, self.crs, *self.shape)

    @property
    def raw(self) -> np.ndarray:
        if self._raw is None:
            self._raw = np.load(self.raw_cache, mmap_mode="r")
        return self._raw

    def band(self, name: str) -> np.ndarray:
        return np.array(self.raw[self.names.index(name)], np.float32, copy=True)

    def has(self, name: str) -> bool:
        return name in self.names

    def modality_bands(self, modality: str) -> list[str]:
        return [n for n in self.names if MODALITY.get(n, "other") == modality]

    def band_valid(self, name: str) -> np.ndarray:
        return np.isfinite(self.band(name)) & self.valid


def load(paths: dict, work: Path, *, force: bool = False) -> GeoData:
    """Read the three rasters (grid-checked), cache the raw cube as a local memmap, build masks."""
    import rasterio

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    with rasterio.open(paths["raw"]) as r, rasterio.open(paths["known"]) as k, rasterio.open(paths["template"]) as t:
        check_grid(r, k, t)
        names = [canonical_band_name(d, i + 1) for i, d in enumerate(r.descriptions)]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate canonical band descriptions: {names}")
        shape, transform, crs, profile = (r.height, r.width), r.transform, r.crs, t.profile.copy()
        tv = np.isfinite(t.read(1, masked=True).astype(np.float32).filled(np.nan))
        kn = k.read(1, masked=True).filled(0)
        cache = work / "raw_stack_f32.npy"
        stamp = work / "raw_stack_f32.json"
        fp = file_fingerprint(Path(paths["raw"]))
        reuse = cache.exists() and stamp.exists() and not force and json.loads(stamp.read_text()).get("fp") == fp
        if not reuse:
            tmp = work / "raw_stack_f32.partial.npy"
            mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(r.count, *shape))
            for i in range(r.count):
                a = r.read(i + 1, masked=True).astype(np.float32).filled(np.nan)
                a[~np.isfinite(a) | (np.abs(a) > 1e20)] = np.nan
                mm[i] = a
            mm.flush()
            del mm
            tmp.replace(cache)
            atomic_write_json(stamp, {"fp": fp, "names": names})
    raw = np.load(cache, mmap_mode="r")
    any_finite = np.zeros(shape, bool)
    cov = {}
    for i, n in enumerate(names):
        f = np.isfinite(raw[i])
        any_finite |= f
        cov[n] = float(f[tv].mean()) if tv.any() else 0.0
    valid = tv & any_finite
    labels = (np.nan_to_num(kn.astype(np.float64)) >= 1) & valid
    return GeoData(
        str(paths["raw"]),
        str(paths["known"]),
        str(paths["template"]),
        names,
        shape,
        transform,
        crs,
        profile,
        tv,
        valid,
        labels,
        str(cache),
        cov,
    )


# =============================================================================== inventory (section 03)
def band_inventory(data: GeoData) -> pd.DataFrame:
    rows = []
    tv = data.template_valid
    for i, n in enumerate(data.names):
        a = np.asarray(data.raw[i])
        f = np.isfinite(a) & tv
        v = a[f].astype(np.float64)
        q = np.quantile(v, [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]) if v.size else [np.nan] * 9
        doc = BAND_DOCS.get(n, {})
        rows.append(
            {
                "band": i + 1,
                "name": n,
                "modality": MODALITY.get(n, "other"),
                "meaning": doc.get("meaning", "undocumented"),
                "interpretation_status": doc.get("status", "unknown"),
                "common_source_of": COMMON_SOURCE.get(n, ""),
                "finite_frac_in_template": float(f.sum() / max(tv.sum(), 1)),
                "missing_px_in_template": int(tv.sum() - f.sum()),
                "min": float(v.min()) if v.size else np.nan,
                "max": float(v.max()) if v.size else np.nan,
                "p001": q[0],
                "p01": q[1],
                "p05": q[2],
                "p25": q[3],
                "p50": q[4],
                "p75": q[5],
                "p95": q[6],
                "p99": q[7],
                "p999": q[8],
                "robust_range_p01_p99": q[7] - q[1],
                "iqr": q[5] - q[3],
                "n_unique_sample": int(len(np.unique(v[:: max(len(v) // 200000, 1)]))),
                "units": "not documented in raster metadata",
            }
        )
    return pd.DataFrame(rows)


def file_inventory(paths: dict, dem_dir: str = "") -> pd.DataFrame:
    import rasterio

    rows = []
    for role in ("raw", "known", "template"):
        p = Path(paths[role])
        t0 = time.perf_counter()
        with rasterio.open(p) as src:
            src.read(1, window=rasterio.windows.Window(0, 0, min(src.width, 1024), min(src.height, 1024)))
            rows.append(
                {
                    "role": role,
                    "file": p.name,
                    "path": str(p),
                    "size_mb": p.stat().st_size / 2**20,
                    "bands": src.count,
                    "height": src.height,
                    "width": src.width,
                    "dtype": src.dtypes[0],
                    "nodata": src.nodata,
                    "crs": str(src.crs),
                    "res_m": src.res[0],
                    "bounds": tuple(round(b, 1) for b in src.bounds),
                    "transform": tuple(src.transform)[:6],
                    "read_1024px_ms": 1000 * (time.perf_counter() - t0),
                    "fingerprint": file_fingerprint(p)["hash"][:16],
                }
            )
    if dem_dir and Path(dem_dir).exists():
        for p in sorted(Path(dem_dir).glob("*.tif"))[:200]:
            with rasterio.open(p) as src:
                rows.append(
                    {
                        "role": "dem_derived",
                        "file": p.name,
                        "path": str(p),
                        "size_mb": p.stat().st_size / 2**20,
                        "bands": src.count,
                        "height": src.height,
                        "width": src.width,
                        "dtype": src.dtypes[0],
                        "nodata": src.nodata,
                        "crs": str(src.crs),
                        "res_m": src.res[0],
                        "bounds": tuple(round(b, 1) for b in src.bounds),
                        "transform": tuple(src.transform)[:6],
                    }
                )
    return pd.DataFrame(rows)


def coverage_matrix(data: GeoData, blocks: int = 8) -> pd.DataFrame:
    """Finite fraction of every band in a ``blocks x blocks`` regional grid (inside the template)."""
    h, w = data.shape
    rb, cb = np.linspace(0, h, blocks + 1).astype(int), np.linspace(0, w, blocks + 1).astype(int)
    rows = []
    for i, n in enumerate(data.names):
        f = np.isfinite(np.asarray(data.raw[i]))
        for a in range(blocks):
            for b in range(blocks):
                tv = data.template_valid[rb[a] : rb[a + 1], cb[b] : cb[b + 1]]
                if tv.sum() == 0:
                    continue
                rows.append(
                    {
                        "band": n,
                        "modality": MODALITY.get(n, "other"),
                        "block_r": a,
                        "block_c": b,
                        "finite_frac": float(f[rb[a] : rb[a + 1], cb[b] : cb[b + 1]][tv].mean()),
                    }
                )
    return pd.DataFrame(rows)


def label_summary(data: GeoData) -> dict:
    from scipy import ndimage

    lab = data.labels
    comp, n = ndimage.label(lab, structure=np.ones((3, 3), bool))
    sizes = np.bincount(comp.ravel())[1:] if n else np.array([0])
    return {
        "fault_px": int(lab.sum()),
        "fault_frac_valid": float(lab.sum() / max(data.valid.sum(), 1)),
        "components_8conn": int(n),
        "component_px_median": float(np.median(sizes)),
        "component_px_p90": float(np.percentile(sizes, 90)),
        "valid_px": int(data.valid.sum()),
        "template_valid_px": int(data.template_valid.sum()),
        "valid_area_km2": float(data.valid.sum() * PIXEL_M**2 / 1e6),
    }


def run_inventory(data: GeoData, paths: dict, ws, dem_dir: str = "") -> dict:
    inv = band_inventory(data)
    files = file_inventory(paths, dem_dir)
    cov = coverage_matrix(data)
    docs = pd.DataFrame(
        [
            {"name": n, **BAND_DOCS.get(n, {"meaning": "undocumented"}), "modality": MODALITY.get(n, "other")}
            for n in data.names
        ]
    )
    outs = [
        atomic_save_table(ws.path("inventory", "band_inventory.csv"), inv),
        atomic_save_table(ws.path("inventory", "file_inventory.csv"), files),
        atomic_save_table(ws.path("inventory", "coverage_matrix.csv"), cov),
        atomic_save_table(ws.path("inventory", "band_interpretation.csv"), docs),
    ]
    summary = {
        **label_summary(data),
        "bands": data.names,
        "shape": data.shape,
        "crs": str(data.crs),
        "transform": tuple(data.transform)[:6],
        "notes": paths.get("notes", []),
        "undocumented_bands": [n for n in data.names if n not in BAND_DOCS],
        "low_coverage_bands": inv.loc[inv.finite_frac_in_template < 0.9, "name"].tolist(),
    }
    outs.append(atomic_write_json(ws.path("inventory", "inventory_summary.json"), summary))
    return {"outputs": outs, "summary": summary, "band_inventory": inv, "coverage": cov}


def hardware_report() -> dict:
    import os
    import platform
    import shutil as sh

    rep = {"python": platform.python_version(), "cpu_count": os.cpu_count(), "platform": platform.platform()}
    try:
        import psutil

        vm = psutil.virtual_memory()
        rep.update(ram_gb=vm.total / 2**30, ram_available_gb=vm.available / 2**30)
    except Exception:
        rep.update(ram_gb=np.nan, ram_available_gb=np.nan)
    rep["disk_free_gb"] = sh.disk_usage("/").free / 2**30
    try:
        import torch

        rep["torch"] = torch.__version__
        rep["cuda"] = bool(torch.cuda.is_available())
        if rep["cuda"]:
            p = torch.cuda.get_device_properties(0)
            rep.update(gpu=p.name, vram_gb=p.total_memory / 2**30)
    except Exception:
        rep["torch"], rep["cuda"] = None, False
    for mod in ("numpy", "scipy", "pandas", "rasterio", "sklearn", "xgboost", "lightgbm", "skimage"):
        try:
            rep[mod] = __import__(mod).__version__
        except Exception:
            rep[mod] = None
    return rep


def ram_budget_gb(cfg_gb: float = 0.0) -> float:
    if cfg_gb:
        return cfg_gb
    try:
        import psutil

        return psutil.virtual_memory().available / 2**30 * 0.6
    except Exception:
        return 4.0


def bytes_per_feature(shape: tuple, dtype: str) -> int:
    return int(math.prod(shape) * np.dtype(dtype).itemsize)
