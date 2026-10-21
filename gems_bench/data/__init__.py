"""Native 100 m grid: feature bands, mapped-fault labels, valid footprint, submission template.

Everything in Phase 2 (targets, folds, OOF maps, scoring, submissions) lives on the
native grid of ``gems-geodawn-numerical-features.tif`` (EDA: 3730 x 3292 px, 100 m,
EPSG:32611). Phase-1 rasters were written on a 2x block-averaged 200 m grid; they are
only ever used here as *input features* resampled to 100 m (see ``features.py``),
never as labels.

Label convention follows the organizer reference solution: value >= 1 is a mapped fault.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from gems_eda.config import band_modality
from gems_eda.io import Grid, _band_names, _pixel_km

from ..config import BenchPaths

NODATA_ABS = 1e30


@dataclass
class NativeData:
    paths: BenchPaths
    grid: Grid
    profile: dict                    # rasterio profile of the feature file
    band_names: list[str]
    band_desc: dict[str, str]
    labels: np.ndarray               # bool, mapped fault pixels (>= 1)
    valid: np.ndarray                # bool, data footprint
    template_profile: dict           # rasterio profile of example_submission.tif
    template_valid: np.ndarray       # bool, finite pixels of the template (if it encodes one)
    label_info: dict

    @property
    def shape(self) -> tuple[int, int]:
        return self.grid.shape

    def read_band(self, name: str) -> np.ndarray:
        """One raw band as float32 with NaN nodata (read on demand, ~49 MB)."""
        i = self.band_names.index(name) + 1
        with rasterio.open(self.paths.raw("features")) as src:
            a = src.read(i).astype(np.float32)
        a[~np.isfinite(a) | (np.abs(a) > NODATA_ABS)] = np.nan
        return a

    def read_window(self, name: str, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
        from rasterio.windows import Window

        i = self.band_names.index(name) + 1
        with rasterio.open(self.paths.raw("features")) as src:
            a = src.read(i, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float32)
        a[~np.isfinite(a) | (np.abs(a) > NODATA_ABS)] = np.nan
        return a

    def summary(self) -> pd.DataFrame:
        g = self.grid
        rows = {
            "crs": str(g.crs), "height": g.height, "width": g.width,
            "pixel_m_x": g.pixel_km_x * 1000, "pixel_m_y": g.pixel_km_y * 1000,
            "n_bands": len(self.band_names), "valid_frac": float(self.valid.mean()),
            "valid_km2": float(self.valid.sum() * g.pixel_km_x * g.pixel_km_y),
            "label_px": int(self.labels.sum()), "label_frac_of_valid": float(self.labels[self.valid].mean()),
            "labels_outside_valid": int((self.labels & ~self.valid).sum()),
            "template_same_grid": bool(self.template_matches()),
            "template_valid_frac": float(self.template_valid.mean()),
            **{f"label_{k}": v for k, v in self.label_info.items()},
        }
        return pd.DataFrame({"key": list(rows), "value": [str(v) for v in rows.values()]})

    def template_matches(self) -> bool:
        t, p = self.template_profile, self.profile
        return (t["crs"] == p["crs"] and t["transform"] == p["transform"]
                and t["height"] == p["height"] and t["width"] == p["width"])

    def band_table(self) -> pd.DataFrame:
        return pd.DataFrame({"band": self.band_names,
                             "modality": [band_modality(b) for b in self.band_names],
                             "description": [self.band_desc.get(b, "") for b in self.band_names]})


def _read_on(path: Path, profile: dict, resampling=Resampling.nearest) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        a = src.read(1, masked=True).astype(np.float32)
        a = np.ma.filled(a, np.nan)
        info = {"file": Path(path).name, "dtype": src.dtypes[0], "nodata": src.nodata,
                "shape": (src.height, src.width), "same_grid": src.transform == profile["transform"]
                and src.shape == (profile["height"], profile["width"]) and src.crs == profile["crs"]}
        if info["same_grid"]:
            return a, info
        out = np.full((profile["height"], profile["width"]), np.nan, np.float32)
        reproject(a, out, src_transform=src.transform, src_crs=src.crs, src_nodata=np.nan,
                  dst_transform=profile["transform"], dst_crs=profile["crs"], dst_nodata=np.nan,
                  resampling=resampling)
    return out, info


def load_native(paths: BenchPaths | None = None) -> NativeData:
    paths = paths or BenchPaths.discover()
    with rasterio.open(paths.raw("features")) as src:
        profile = src.profile.copy()
        names, desc = _band_names(src)
        h, w = src.height, src.width
        finite = np.zeros((h, w), np.int16)
        for i in range(1, src.count + 1):                    # band-by-band: never holds the full stack
            a = src.read(i)
            finite += (np.isfinite(a) & (np.abs(a) < NODATA_ABS)).astype(np.int16)
        kx, ky, geo = _pixel_km(src.transform, src.crs, h)
        grid = Grid(src.transform, src.crs, h, w, kx, ky, geo)
    valid = finite >= math.ceil(0.5 * len(names))

    lab, linfo = _read_on(paths.raw("faults"), profile, Resampling.nearest)
    uniq = np.unique(lab[np.isfinite(lab)][:5_000_000])
    linfo["values_preview"] = [float(x) for x in uniq[:10]]
    labels = np.isfinite(lab) & (lab >= 1)

    with rasterio.open(paths.raw("submission")) as src:
        tprof = src.profile.copy()
        t = src.read(1, masked=True)
        tvalid = ~np.ma.getmaskarray(t) & np.isfinite(np.ma.filled(t.astype(np.float32), np.nan))
    return NativeData(paths, grid, profile, names, desc, labels, valid, tprof, tvalid, linfo)
