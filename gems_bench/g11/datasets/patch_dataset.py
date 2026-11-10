"""Patch datasets: 100 m geophysical patches and multiresolution (100 m + native DEM) patches.

Patches are cut ONLY inside one split: a centre is accepted if the whole ``patch_px`` window holds no pixel of
another split (embargo pixels allowed, masked from the loss). Items are dicts of NumPy arrays; ``collate`` stacks
them into torch tensors when torch is installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..spatial_splits import CODES


def patch_windows(centres_r, centres_c, size: int, shape, split_code: np.ndarray, split: str) -> pd.DataFrame:
    h, w = shape
    half = size // 2
    rows = []
    allowed = {0, CODES["embargo"], CODES[split]}
    for r, c in zip(centres_r, centres_c):
        r0 = int(np.clip(r - half, 0, h - size))
        c0 = int(np.clip(c - half, 0, w - size))
        sub = split_code[r0:r0 + size, c0:c0 + size]
        if set(np.unique(sub)) <= allowed and (sub == CODES[split]).any():
            rows.append({"row0": r0, "col0": c0, "size": size, "split": split})
    return pd.DataFrame(rows, columns=["row0", "col0", "size", "split"]).drop_duplicates().reset_index(drop=True)


class GeoPatchDataset:
    """``store`` channels -> (C, P, P) input; ``target`` -> (1, P, P); ``valid`` = own-split & finite pixels."""

    def __init__(self, store, channels, target: np.ndarray, windows: pd.DataFrame, split_code: np.ndarray,
                 split: str, normalizer=None, block_px: int = 100, distance: np.ndarray | None = None):
        self.store, self.channels, self.target = store, list(channels), target
        self.windows, self.code, self.split = windows.reset_index(drop=True), split_code, split
        self.norm = normalizer
        self.block_px = block_px
        self.distance = distance              # training-label distance TARGET (m) for the boundary loss, never an input
        self._mm = {c: store.get(c) for c in self.channels}

    def __len__(self):
        return len(self.windows)

    def _inputs(self, r0, c0, P):
        x = np.stack([np.asarray(self._mm[c][r0:r0 + P, c0:c0 + P], np.float32) for c in self.channels])
        if self.norm is not None:
            x = ((x - self.norm.center[:, None, None]) / self.norm.scale[:, None, None]).astype(np.float32)
        finite = np.isfinite(x)
        return np.where(finite, x, 0.0).astype(np.float32), finite

    def __getitem__(self, i) -> dict:
        w = self.windows.iloc[i]
        r0, c0, P = int(w.row0), int(w.col0), int(w["size"])
        x, finite = self._inputs(r0, c0, P)
        y = np.asarray(self.target[r0:r0 + P, c0:c0 + P], np.float32)
        own = self.code[r0:r0 + P, c0:c0 + P] == CODES[self.split]
        valid = own & np.isfinite(y) & finite.all(0)
        extra = {}
        if self.distance is not None:
            dd = np.asarray(self.distance[r0:r0 + P, c0:c0 + P], np.float32)
            extra["distance"] = np.nan_to_num(dd, nan=1e5, posinf=1e5)[None]
        return {**extra, "x": x, "y": np.nan_to_num(y)[None], "valid": valid[None].astype(np.float32),
                "geo_id": np.int64((r0 // self.block_px) * 10000 + c0 // self.block_px), "split": self.split,
                "weight": np.float32(1.0), "modality": np.array([1.0, 0.0], np.float32),
                "provenance": {"row0": r0, "col0": c0, "size": P}}


class MultiresolutionPatchDataset(GeoPatchDataset):
    """Adds a DEM patch read at ``dem_res_m`` for the same ground footprint; ``modality[1]`` = DEM present."""

    def __init__(self, *a, dem_reader=None, grid=None, dem_res_m: float = 2.0, dem_px: int = 256, **k):
        super().__init__(*a, **k)
        self.dem, self.grid, self.dem_res, self.dem_px = dem_reader, grid, dem_res_m, dem_px

    def __getitem__(self, i) -> dict:
        item = super().__getitem__(i)
        dem = np.zeros((1, self.dem_px, self.dem_px), np.float32)
        dmask = np.zeros((1, self.dem_px, self.dem_px), np.float32)
        has = 0.0
        if self.dem is not None:
            p = item["provenance"]
            a, _, c, _, e, f = self.grid.transform[:6]
            x0, y1 = c + a * p["col0"], f + e * p["row0"]
            x1, y0 = x0 + a * p["size"], y1 + e * p["size"]
            try:
                win = self.dem.read_dem_window((x0, y0, x1, y1), self.grid.crs, resolution=self.dem_res,
                                               resampling="average", snap=False)
                z = win.elevation
                if win.valid.mean() > 0.05:
                    from scipy.ndimage import zoom

                    zf = np.where(win.valid, z, np.nanmean(z[win.valid]))
                    zz = zoom(zf, (self.dem_px / zf.shape[0], self.dem_px / zf.shape[1]), order=1)
                    vv = zoom(win.valid.astype(np.float32), (self.dem_px / z.shape[0], self.dem_px / z.shape[1]), order=0)
                    zz = (zz - np.nanmean(zz)) / (np.nanstd(zz) + 1e-3)
                    dem[0], dmask[0], has = zz[: self.dem_px, : self.dem_px], vv[: self.dem_px, : self.dem_px], 1.0
                    item["provenance"]["dem_tiles"] = win.source_tiles
            except Exception as ex:                                        # noqa: BLE001
                item["provenance"]["dem_error"] = str(ex)
        item["dem"], item["dem_valid"] = dem, dmask
        item["modality"] = np.array([1.0, has], np.float32)
        return item


def collate(items: list[dict]):
    import torch

    out = {}
    for k in items[0]:
        if k in ("split", "provenance"):
            out[k] = [it[k] for it in items]
        else:
            out[k] = torch.as_tensor(np.stack([np.asarray(it[k]) for it in items]))
    return out
