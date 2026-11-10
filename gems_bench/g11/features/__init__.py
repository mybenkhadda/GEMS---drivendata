"""G11 feature families + the on-disk feature store (one float32 memmap per feature + a registry table)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


def write_geotiff(path, array: np.ndarray, grid, descriptions=None, nodata=np.nan) -> Path:
    """Atomic, compressed, tiled GeoTIFF on ``grid``; reopened to verify shape / CRS / transform."""
    import os

    import rasterio

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(array)
    if a.ndim == 2:
        a = a[None]
    flt = np.issubdtype(a.dtype, np.floating)
    prof = grid.profile(a.shape[0], "float32" if flt else a.dtype.name)
    prof.update(compress="deflate", tiled=True, blockxsize=256, blockysize=256, predictor=3 if flt else 2,
                nodata=nodata if flt else None, BIGTIFF="IF_SAFER")
    tmp = p.with_name(p.stem + ".partial" + p.suffix)
    with rasterio.open(tmp, "w", **prof) as ds:
        ds.write(a.astype(np.float32) if flt else a)
        for i, dsc in enumerate(descriptions or [], 1):
            ds.set_band_description(i, str(dsc))
    with rasterio.open(tmp) as ds:
        if ds.shape != tuple(grid.shape) or ds.count != a.shape[0] or not np.allclose(tuple(ds.transform)[:6],
                                                                                     grid.transform[:6]):
            raise RuntimeError(f"GeoTIFF readback mismatch: {tmp}")
    os.replace(tmp, p)
    return p


@dataclass
class FeatureCube:
    """In-memory named feature set on one grid (``arrays[name]`` float32; ``meta`` one row per feature)."""
    arrays: dict
    meta: pd.DataFrame
    grid: object = None
    notes: list = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return list(self.arrays)

    def __getitem__(self, k):
        return self.arrays[k]


def family_dir(family: str) -> str:
    """Sub-folder of a feature family: geophysical / terrain_100m / targets / other."""
    if family.startswith("geo") or family in ("raw", "g06", "evidence_rank", "external"):
        return "geophysical"
    if family in ("dem_100m", "multimodal"):
        return "terrain_100m"
    if family == "target":
        return "targets"
    return "other"


class FeatureStore:
    """``<root>/<family_dir>/<name>.npy`` (float32 memmap, NaN = missing) + ``<root>/feature_registry.parquet``.

    Columns of the registry: name, family, scale_m, unit, source, label_derived (bool: may NOT be used as a
    predictor), finite_fraction, fingerprint, relpath.
    """

    def __init__(self, root, grid):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.grid = grid
        self.reg_path = self.root / "feature_registry.parquet"
        self.reg = pd.read_parquet(self.reg_path) if self.reg_path.is_file() else pd.DataFrame(
            columns=["name", "family", "scale_m", "unit", "source", "label_derived", "finite_fraction", "fingerprint",
                     "relpath"])

    def path_of(self, name: str) -> Path:
        r = self.reg[self.reg.name == name]
        if len(r) and "relpath" in r and isinstance(r.relpath.iloc[-1], str) and r.relpath.iloc[-1]:
            return self.root / r.relpath.iloc[-1]
        return self.root / f"{name}.npy"

    def has(self, name: str, fingerprint: str | None = None) -> bool:
        r = self.reg[self.reg.name == name]
        if r.empty or not self.path_of(name).is_file():
            return False
        return fingerprint is None or r.fingerprint.iloc[-1] == fingerprint

    def put(self, name: str, arr: np.ndarray, family: str, scale_m=None, unit: str = "", source: str = "",
            label_derived: bool = False, fingerprint: str = "") -> Path:
        from ..registry import atomic_write_npy

        a = np.asarray(arr, np.float32)
        if a.shape != tuple(self.grid.shape):
            raise ValueError(f"{name}: shape {a.shape} != grid {self.grid.shape}")
        rel = f"{family_dir(family)}/{name}.npy"
        p = atomic_write_npy(self.root / rel, a)
        v = self.grid.valid if self.grid.valid is not None else np.ones(a.shape, bool)
        row = {"name": name, "family": family, "scale_m": scale_m, "unit": unit, "source": source,
               "label_derived": bool(label_derived), "finite_fraction": float(np.isfinite(a[v]).mean()) if v.any() else 0.0,
               "fingerprint": fingerprint, "relpath": rel}
        new = pd.DataFrame([row])
        old = self.reg[self.reg.name != name]
        self.reg = new if old.empty else pd.concat([old, new], ignore_index=True)
        self.reg["label_derived"] = self.reg["label_derived"].astype(bool)
        return p

    def flush(self) -> Path:
        from ..registry import atomic_write_parquet

        return atomic_write_parquet(self.reg_path, self.reg)

    def get(self, name: str, mmap: bool = True) -> np.ndarray:
        local = getattr(self, "_local", {}).get(name)
        p = local if local is not None else self.path_of(name)
        if not p.is_file():
            raise KeyError(f"feature {name} not in store {self.root}")
        return np.load(p, mmap_mode="r" if mmap else None)

    def localize(self, names, local_dir) -> dict:
        """Stage ``names`` to local scratch (sha256-verified copy + cache manifest); later ``get`` reads the copy.

        Used before GPU training so patches are not read from Drive at every step. Returns the cache manifest.
        """
        import json
        import shutil

        from ..registry import sha256_file

        d = Path(local_dir)
        d.mkdir(parents=True, exist_ok=True)
        man_p = d / "cache_manifest.json"
        man = json.loads(man_p.read_text()) if man_p.is_file() else {}
        self._local = getattr(self, "_local", {})
        for n in names:
            src = self.path_of(n)
            dst = d / f"{n}.npy"
            st = src.stat()
            ent = man.get(n)
            if not (dst.is_file() and ent and ent["size"] == st.st_size and ent["mtime_ns"] == st.st_mtime_ns):
                tmp = dst.with_name(dst.stem + ".partial.npy")
                shutil.copyfile(src, tmp)
                h_src, h_dst = sha256_file(src), sha256_file(tmp)
                if h_src != h_dst:
                    raise OSError(f"local staging checksum mismatch for {n}")
                tmp.replace(dst)
                man[n] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": h_src, "source": str(src)}
            self._local[n] = dst
        man_p.write_text(json.dumps(man, indent=1))
        return man

    def predictors(self, families=None) -> list[str]:
        ld = self.reg.label_derived.map(lambda v: v is True or v == 1 or str(v).lower() == "true")
        r = self.reg[~ld]
        if families:
            r = r[r.family.isin(families)]
        return r.name.tolist()

    def sample(self, names, rows, cols) -> np.ndarray:
        return np.stack([np.asarray(self.get(n)[rows, cols], np.float32) for n in names], 1)
