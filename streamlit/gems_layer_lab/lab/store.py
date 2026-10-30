"""Persistent outputs of the Feature Discovery Studio: ``<GEMS_2>/outputs/geological_layer_lab/``.

    registry/feature_registry.json      saved features (lab.library)
    registry/dependency_graph.dot|json  dependency graph snapshots
    tables/<run>/*.csv|parquet          association, regional, complementarity tables
    tournaments/<id>/                   results.parquet / results.csv / manifest.json (resumable)
    transects/<run>/                    sampled profiles
    figures/<run>/*.png|svg + data/*.csv  figures and their source data
    exports/<name>_v<k>_<hash>/         native GeoTIFFs + export_manifest.json
    manifests/                          environment / session manifests

Every write is atomic (tmp + replace). Run directories are new per call (time stamp), so nothing is overwritten; the
original GEMS rasters are never written.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SUBDIRS = ("registry", "tables", "tournaments", "transects", "figures", "exports", "manifests", "plugins")


class OutputStore:
    def __init__(self, gems_root: str | Path, sub: str = "outputs/geological_layer_lab"):
        self.root = Path(gems_root) / sub
        for s in SUBDIRS:
            (self.root / s).mkdir(parents=True, exist_ok=True)

    @property
    def registry_path(self) -> Path:
        return self.root / "registry" / "feature_registry.json"

    def run_dir(self, kind: str, label: str = "") -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        d = self.root / kind / (f"{stamp}_{label}" if label else stamp)
        k = 1
        while d.exists():
            d = d.with_name(f"{d.name}_{k}")
            k += 1
        d.mkdir(parents=True)
        return d

    @staticmethod
    def _atomic(path: Path, writer) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        writer(tmp)
        os.replace(tmp, path)
        return path

    def table(self, df: pd.DataFrame, path: Path, parquet: bool = True) -> list[Path]:
        out = [self._atomic(path.with_suffix(".csv"), lambda t: df.to_csv(t, index=False))]
        if parquet and len(df):
            try:
                out.append(self._atomic(path.with_suffix(".parquet"), lambda t: _parquet_safe(df).to_parquet(
                    t, index=False)))
            except Exception:                            # parquet optional (pyarrow absent / mixed dtypes)
                pass
        return out

    def json(self, obj, path: Path) -> Path:
        return self._atomic(path, lambda t: Path(t).write_text(json.dumps(obj, indent=2, default=_js),
                                                               encoding="utf-8"))

    def text(self, s: str, path: Path) -> Path:
        return self._atomic(path, lambda t: Path(t).write_text(s, encoding="utf-8"))

    def environment(self) -> Path:
        mods = {}
        for m in ("numpy", "scipy", "pandas", "rasterio", "streamlit", "plotly", "matplotlib"):
            try:
                mods[m] = __import__(m).__version__
            except Exception:
                mods[m] = None
        return self.json({"python": sys.version, "platform": platform.platform(), "packages": mods,
                          "time": time.strftime("%Y-%m-%dT%H:%M:%S")},
                         self.root / "manifests" / f"environment_{time.strftime('%Y%m%d_%H%M%S')}.json")


def _js(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in d.columns:
        if d[c].dtype == object:
            d[c] = d[c].map(lambda v: v if v is None or isinstance(v, str) else json.dumps(v, default=_js))
    return d
