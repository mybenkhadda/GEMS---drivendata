"""reports/gems3 writer: the M6 ``Workspace`` (atomic files, per-unit fingerprints) with GEMS3 folders,
figures in PNG + SVG, Parquet next to every table and GeoPackage for vectors."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from ..artifacts.atomic import atomic_save, atomic_write_json
from ..pipeline.m6 import Workspace

GEMS3_VERSION = "gems3.1"


class G3Workspace(Workspace):
    SUBDIRS = ("manifests", "manifests/units", "audit", "metric_optimization", "metric_optimization/figures",
               "ensembles", "proposals", "ranker", "dem", "map_completion", "figures", "predictions", "submissions")

    def finish(self, name: str, fp: str, info: dict | None = None, seconds: float | None = None) -> None:
        atomic_write_json(self.root / "manifests" / "units" / f"{name}.json",
                          {"unit": name, "fingerprint": fp, "status": "done", "gems3_version": GEMS3_VERSION,
                           "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": seconds,
                           "outputs": sorted(set(self._written)), "info": info or {}})
        self._unit = None

    def save_table(self, df: pd.DataFrame, name: str, sub: str = "audit") -> Path:
        """CSV always (``save_csv``), plus Parquet when ``save_parquet``; ``sub`` is the report folder."""
        from ..pipeline.m6 import _parquet_safe

        df = df if df is not None else pd.DataFrame()
        p = self.root / sub / f"{name}.csv"
        if self.cfg.get("save_csv", True):
            atomic_save(p, lambda t: df.to_csv(t, index=False))
            self._track(p)
        if self.cfg.get("save_parquet") and len(df):
            q = p.with_suffix(".parquet")
            try:
                atomic_save(q, lambda t: _parquet_safe(df).to_parquet(t, index=False))
                self._track(q)
            except Exception as e:                                  # optional format: never block the analysis
                print(f"[gems3] parquet skipped for {name}: {type(e).__name__}: {e}")
        return p

    def load_table(self, name: str, sub: str = "audit") -> pd.DataFrame:
        for ext, reader in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
            p = self.root / sub / f"{name}{ext}"
            if p.exists():
                try:
                    return reader(p)
                except Exception:
                    continue
        return pd.DataFrame()

    def save_fig(self, fig, name: str, data=None, sub: str = "figures") -> Path | None:
        import matplotlib.pyplot as plt

        if fig is None:
            return None
        p = self.root / sub / f"{name}.png"
        if self.cfg.get("save_figures", True):
            for ext in ("png", "svg"):
                q = p.with_suffix(f".{ext}")
                atomic_save(q, lambda t, e=ext: fig.savefig(t, dpi=170, bbox_inches="tight", format=e))
                self._track(q)
        if data is not None:
            d = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            q = self.root / sub / f"{name}_data.csv"
            atomic_save(q, lambda t: d.to_csv(t, index=False))
            self._track(q)
        plt.close(fig)
        return p

    def save_raster(self, layers: dict, rel: str, grid) -> Path | None:
        if not self.cfg.get("save_rasters", True):
            return None
        return super().save_raster(layers, rel, grid)

    def save_lines(self, paths: list, attrs: pd.DataFrame, rel: str, grid, layer: str) -> Path:
        """GeoPackage (``write_lines`` falls back to Parquet + WKT without geopandas). A temporary .gpkg is
        written then renamed; the fallback writes its Parquet directly (it is a single atomic-enough file)."""
        from gems_eda.io import write_lines

        target = self.root / rel
        try:
            import geopandas  # noqa: F401
        except ImportError:
            return self._track(write_lines(target, paths, attrs, grid, layer))
        return self._track(atomic_save(target, lambda t: write_lines(Path(t), paths, attrs, grid, layer)))
