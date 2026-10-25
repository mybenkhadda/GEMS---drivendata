"""Native-resolution (100 m) features, cached one-per-file, with a registry.

* Every feature is computed from input rasters only — never from labels. Phase-1
  ``label_context.tif`` is never read here.
* Filters reuse ``gems_eda.enhance.Enhancer`` on the *native* grid, so physical scales
  (km) mean the same thing as in the EDA even though pixel size halves.
* Optional Phase-1 rasters (lineament maps, cross-modal orientation) were written on the
  200 m EDA grid; they are bilinearly resampled to 100 m and flagged ``origin=eda_resampled``.
* Each feature is robust-standardised over the valid footprint (median / 1.4826 MAD, with
  signed-log compression for heavily skewed fields), exactly as in the EDA.
* Transform / scale choices come from the Phase-1 AUC screens, which used labels over the
  whole region: a mild, documented hyper-parameter selection effect (not pixel leakage).

Cache: ``outputs/benchmark/cache/features/<file>.npy`` (float32, memory-mapped on read) and
``outputs/benchmark/cache/features/feature_registry.csv``.
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from gems_eda.config import MODALITIES, PRIMARY_FIELD
from gems_eda.enhance import Enhancer, nan_gaussian, standardize_signal

from ..data import NativeData

STRIKE_FAMILIES = (17.5, 152.5)        # Phase-1 dominant strikes (fault_morphology_summary.json)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    groups: tuple[str, ...]
    modality: str
    source: str                         # raw band (or EDA raster:band)
    transform: str
    scale_km: float = 0.0
    kind: str = "local"                 # "local" (fault localisation) | "context" (regional)
    origin: str = "native"              # native | eda_resampled
    units: str = "robust-z"
    support_km: float = 0.0             # ~ radius of input influence (3 sigma or wavelet span)
    note: str = ""


def _support(transform: str, scale_km: float) -> float:
    s = scale_km / 2.0                  # Gaussian sigma (km), EDA convention
    return {"raw": 0.0, "smooth": 3 * s, "thdr": 3 * s, "dirderiv": 3 * s, "curvature": 3 * s,
            "mexhat": 3 * s, "ridge": 3 * s, "std": 3 * s, "relief": 3 * s,
            "coherence": 3 * s * 3.0, "tilt_edge": 6 * s, "tilt": 6 * s,
            "log_gabor": 2 * (2 * scale_km), "eda": 10.0}.get(transform, 3 * s)


def _spec(name, groups, modality, source, transform, scale_km=0.0, kind="local", origin="native", note=""):
    return FeatureSpec(name, tuple(groups), modality, source, transform, scale_km, kind, origin,
                       "robust-z", _support(transform, scale_km), note)


def build_specs(band_names: list[str], include_eda: bool = True) -> list[FeatureSpec]:
    """Compact, EDA-motivated feature catalogue. Groups: G1 terrain, G2 raw, G3 raw+eng, G4 G3+context."""
    S: list[FeatureSpec] = []
    mod_of = {b: m for m, bands in MODALITIES.items() for b in bands}
    for b in band_names:                                                    # G2 raw (all channels)
        g = ("G1", "G2", "G3", "G4") if b in ("det_elev", "det_elev_slope") else ("G2", "G3", "G4")
        kind = "context" if mod_of.get(b) in ("strain", "seismicity") else "local"
        S.append(_spec(f"raw:{b}", g, mod_of.get(b, "other"), b, "raw", 0.0, kind))
    T = ("G1", "G3", "G4")
    el = PRIMARY_FIELD["terrain"]
    for s in (0.5, 1.0, 2.0, 5.0):
        S.append(_spec(f"terrain:curvature@{s:g}km", T, "terrain", el, "curvature", s))
    for s in (2.0, 5.0):
        S.append(_spec(f"terrain:tilt_edge@{s:g}km", T, "terrain", el, "tilt_edge", s,
                       note="EDA best terrain transform @5km"))
        S.append(_spec(f"terrain:relief@{s:g}km", T, "terrain", el, "relief", s))
    for s in (0.5, 1.0):
        S.append(_spec(f"terrain:thdr@{s:g}km", T, "terrain", el, "thdr", s))
        S.append(_spec(f"terrain:ridge@{s:g}km", T, "terrain", el, "ridge", s, note="|Hessian| line response, both polarities"))
    for s in (1.0, 2.0):
        S.append(_spec(f"terrain:mexhat@{s:g}km", T, "terrain", el, "mexhat", s))
    for s in (1.0, 3.0):
        S.append(_spec(f"terrain:std@{s:g}km", T, "terrain", el, "std", s))
    S.append(_spec("terrain:coherence@1km", T, "terrain", el, "coherence", 1.0))

    E = ("G3", "G4")
    mag, grav, base, cond = (PRIMARY_FIELD[m] for m in ("magnetic", "gravity", "basement", "conductivity"))
    for s in (1.0, 2.0):
        S.append(_spec(f"magnetic:thdr@{s:g}km", E, "magnetic", mag, "thdr", s))
    for st in STRIKE_FAMILIES:
        S.append(_spec(f"magnetic:dirderiv_s{st:g}@2km", E, "magnetic", mag, "dirderiv", 2.0,
                       note=f"|d/dn| across strike {st} deg"))
        S.append(_spec(f"gravity:dirderiv_s{st:g}@2km", E, "gravity", grav, "dirderiv", 2.0,
                       note=f"|d/dn| across strike {st} deg"))
    for s in (2.0, 5.0):
        S.append(_spec(f"magnetic:log_gabor@{s:g}km", E, "magnetic", mag, "log_gabor", s,
                       note="EDA best magnetic transform @5km"))
    S.append(_spec("magnetic:tilt@2km", E, "magnetic", mag, "tilt", 2.0))
    S.append(_spec("magnetic:coherence@2km", E, "magnetic", mag, "coherence", 2.0))
    S.append(_spec("gravity:thdr@2km", E, "gravity", grav, "thdr", 2.0))
    S.append(_spec("gravity:log_gabor@2km", E, "gravity", grav, "log_gabor", 2.0, note="EDA best gravity transform"))
    S.append(_spec("gravity:tilt@2km", E, "gravity", grav, "tilt", 2.0))
    S.append(_spec("basement:thdr@5km", E, "basement", base, "thdr", 5.0))
    S.append(_spec("basement:tilt_edge@10km", E, "basement", base, "tilt_edge", 10.0, "context",
                   note="EDA best basement transform"))
    S.append(_spec("conductivity:thdr@5km", E, "conductivity", cond, "thdr", 5.0))

    C = ("G4",)
    for m in ("terrain", "magnetic", "gravity", "basement"):
        f = PRIMARY_FIELD[m]
        for s in (10.0, 20.0):
            S.append(_spec(f"{m}:smooth@{s:g}km", C, m, f, "smooth", s, "context"))
        S.append(_spec(f"{m}:thdr@10km", C, m, f, "thdr", 10.0, "context"))
    S.append(_spec("magnetic:coherence@10km", C, "magnetic", mag, "coherence", 10.0, "context"))
    S.append(_spec("gravity:coherence@10km", C, "gravity", grav, "coherence", 10.0, "context"))
    S.append(_spec("terrain:std@10km", C, "terrain", el, "std", 10.0, "context"))
    if include_eda:
        for raster, band in (("lineament_maps", "fused_score"), ("lineament_maps", "fused_fused"),
                             ("lineament_maps", "fused_support"), ("lineament_maps", "response_terrain"),
                             ("crossmodal_orientation", "edge_agreement_count")):
            S.append(_spec(f"eda:{band}", C, "fused", f"{raster}:{band}", "eda", 0.0, "context", "eda_resampled",
                           note="Phase-1 200 m raster, bilinear to 100 m; unsupervised (no label input)"))
    return S


GROUPS = ("G1", "G2", "G3", "G4")


def group_features(specs: list[FeatureSpec], group: str, available: set[str] | None = None) -> list[str]:
    names = [s.name for s in specs if group in s.groups]
    return [n for n in names if available is None or n in available]


def _fname(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name) + ".npy"


class FeatureStore:
    """Compute-once cache of native-grid features (float32 .npy, memory-mapped reads)."""

    def __init__(self, data: NativeData, specs: list[FeatureSpec] | None = None):
        self.data = data
        self.dir = data.paths.cache
        self.dir.mkdir(parents=True, exist_ok=True)
        self.specs = {s.name: s for s in (specs or build_specs(data.band_names))}
        self.enh = Enhancer(data.grid)
        self._src: dict[str, np.ndarray] = {}
        self._mm: dict[str, np.ndarray] = {}

    # -- lazy staging: (local_root, drive_root); a cache file missing locally but present on the Drive mirror
    # counts as available and is copied (atomic, size-verified) on first read. None = local files only.
    drive_mirror: tuple | None = None

    def _mirror_src(self, name: str) -> Path | None:
        if self.drive_mirror is None:
            return None
        local_root, drive_root = (Path(x) for x in self.drive_mirror)
        try:
            rel = self.path(name).resolve().relative_to(local_root.resolve())
        except ValueError:
            return None
        src = drive_root / rel
        return src if src.exists() else None

    def _staged(self, name: str) -> Path:
        p = self.path(name)
        if not p.exists():
            src = self._mirror_src(name)
            if src is not None:
                from ..artifacts.atomic import atomic_copy

                atomic_copy(src, p, "size")
        return p

    # -- paths / availability -------------------------------------------
    def path(self, name: str) -> Path:
        return self.dir / _fname(name)

    def has(self, name: str) -> bool:
        return self.path(name).exists() or self._mirror_src(name) is not None

    @property
    def available(self) -> set[str]:
        return {n for n in self.specs if self.has(n)}

    # -- computation ----------------------------------------------------
    def _source(self, band: str) -> np.ndarray:
        if band not in self._src:
            a = self.data.read_band(band)
            a[~self.data.valid] = np.nan
            self._src[band] = a
        return self._src[band]

    def _eda(self, source: str) -> np.ndarray:
        raster, band = source.split(":", 1)
        path = self.data.paths.eda / "features" / f"{raster}.tif"
        prof = self.data.profile
        out = np.full(self.data.shape, np.nan, np.float32)
        with rasterio.open(path) as src:
            idx = [d for d in src.descriptions].index(band) + 1
            a = np.ma.filled(src.read(idx, masked=True).astype(np.float32), np.nan)
            reproject(a, out, src_transform=src.transform, src_crs=src.crs, src_nodata=np.nan,
                      dst_transform=prof["transform"], dst_crs=prof["crs"], dst_nodata=np.nan,
                      resampling=Resampling.bilinear)
        return out

    def compute(self, spec: FeatureSpec) -> np.ndarray:
        e, s = self.enh, spec.scale_km
        if spec.origin == "eda_resampled":
            return self._eda(spec.source)
        a = self._source(spec.source)
        t = spec.transform
        if t == "raw":
            return a
        if t == "smooth":
            return nan_gaussian(a, e.sigma_px(s))
        if t == "thdr":
            return e.thdr(a, s)
        if t == "dirderiv":
            strike = float(re.search(r"_s([0-9.]+)@", spec.name).group(1))
            return np.abs(e.directional_derivative(a, s, strike + 90.0))
        if t == "curvature":
            return e.curvature(a, s)["max_abs"]
        if t == "tilt_edge":
            return 1.0 - np.abs(e.tilt(a, s)) / 90.0
        if t == "tilt":
            return e.tilt(a, s)
        if t == "relief":
            return a - nan_gaussian(a, e.sigma_px(s))
        if t == "ridge":
            return e.ridge(a, s, "both")[0]
        if t == "mexhat":
            return e.log_wavelet(a, s)
        if t == "std":
            sg = e.sigma_px(s)
            m1, m2 = nan_gaussian(a, sg), nan_gaussian(a * a, sg)
            return np.sqrt(np.clip(m2 - m1 * m1, 0, None))
        if t == "coherence":
            return e.structure_tensor(a, s)[1]
        if t == "log_gabor":
            return e.log_gabor(a, 2.0 * s)[0]
        raise KeyError(t)

    def build(self, names: list[str] | None = None, overwrite: bool = False, verbose: bool = True) -> pd.DataFrame:
        names = names or list(self.specs)
        log = []
        for n in names:
            if self.has(n) and not overwrite:
                continue
            t0 = time.perf_counter()
            arr = self.compute(self.specs[n]).astype(np.float32)
            arr[~self.data.valid] = np.nan
            z, note = standardize_signal(arr)
            np.save(self.path(n), z.astype(np.float32))
            self._mm.pop(n, None)
            log.append({"name": n, "seconds": round(time.perf_counter() - t0, 2), "standardization": note})
            if verbose:
                print(f"  feature {n:<40s} {log[-1]['seconds']:7.1f}s  {note}")
        self._src.clear()
        df = self.registry(extra=pd.DataFrame(log))
        return df

    # -- reading --------------------------------------------------------
    def get(self, name: str) -> np.ndarray:
        if name not in self._mm:
            self._mm[name] = np.load(self._staged(name), mmap_mode="r")
        return self._mm[name]

    def rows(self, names: list[str], r0: int, r1: int) -> np.ndarray:
        """(n_features, r1 - r0, W) float32 slab."""
        return np.stack([np.asarray(self.get(n)[r0:r1]) for n in names]).astype(np.float32)

    def window(self, names: list[str], r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
        return np.stack([np.asarray(self.get(n)[r0:r1, c0:c1]) for n in names]).astype(np.float32)

    def sample(self, names: list[str], flat_idx: np.ndarray) -> np.ndarray:
        """(n_samples, n_features) at flat pixel indices (sorted internally for I/O locality)."""
        order = np.argsort(flat_idx)
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        idx = flat_idx[order]
        out = np.empty((len(idx), len(names)), np.float32)
        for j, n in enumerate(names):
            out[:, j] = self.get(n).reshape(-1)[idx]
        return out[inv]

    def load_stack(self, names: list[str], dtype=np.float16) -> np.ndarray:
        """Whole-grid (C, H, W) stack for CNN training (float16 halves memory)."""
        return np.stack([np.asarray(self.get(n), dtype=dtype) for n in names])

    # -- registry -------------------------------------------------------
    def registry(self, extra: pd.DataFrame | None = None) -> pd.DataFrame:
        rows = []
        for n, s in self.specs.items():
            d = asdict(s)
            d["groups"] = ",".join(s.groups)
            d["path"] = self.data.paths.rel(self.path(n)) if self.has(n) else ""
            d["cached"] = self.has(n)
            rows.append(d)
        df = pd.DataFrame(rows)
        if extra is not None and len(extra):
            prev = self.dir / "feature_build_log.csv"
            log = pd.concat([pd.read_csv(prev), extra]) if prev.exists() else extra
            log.drop_duplicates("name", keep="last").to_csv(prev, index=False)
        df.to_csv(self.dir / "feature_registry.csv", index=False)
        return df


def leakage_audit(specs: list[FeatureSpec], buffer_km: float) -> pd.DataFrame:
    """Features whose input support exceeds the fold buffer (they see val-area *inputs*, never labels)."""
    df = pd.DataFrame([{"name": s.name, "kind": s.kind, "support_km": s.support_km,
                        "exceeds_buffer": s.support_km > buffer_km} for s in specs])
    return df.sort_values("support_km", ascending=False)
