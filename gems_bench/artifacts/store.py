"""Experiment directory layout (Phase 5+) with atomic writers.

``experiments/<EXPERIMENT_ID>/``::

    config.yaml            resolved configuration (after ``extends``), written once
    manifest.json          identity, provenance, stage state, produced maps (experiment.manifest)
    checkpoints/           resumable training state (atomic, rotated; see training.checkpoint)
    metrics/               CSV / Parquet tables (REQUIRED_TABLES + experiment-specific ones)
    predictions/           OOF / discovery / full-region probability GeoTIFFs on the native grid
    figures/               PNG (+ SVG) figures, regenerable from metrics/ and predictions/
    logs/                  run.log (append-only text log, one line per event)
    reports/               experiment_summary.md, JSON decisions, protocol records
    runs/                  nested legacy-layout runs (persist.ExperimentRun) when Phase-3/4 code is reused

Figures are always derived from saved tables / rasters, so they can be regenerated without retraining.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .atomic import atomic_save, atomic_write_json, atomic_write_text

SUBDIRS = ("checkpoints", "metrics", "predictions", "figures", "logs", "reports")

# The per-experiment table contract (section 16 of the Phase-5 brief). Tables that do not apply to an
# experiment kind are written as a one-row "not_applicable" stub by ``RunDir.finalize_contract`` so a
# missing file always means "not produced yet", never "not relevant".
REQUIRED_TABLES = (
    "training_history", "epoch_metrics", "fold_metrics", "geographic_dti", "pseudo_discovery_dti",
    "subgroup_recall", "predicted_area_curve", "model_complementarity", "runtime_profile", "gpu_memory_profile",
)


class RunDir:
    def __init__(self, path: Path):
        self.path = Path(path)
        for s in SUBDIRS:
            (self.path / s).mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"RunDir({self.path})"

    def sub(self, name: str) -> Path:
        return self.path / name

    @property
    def checkpoints(self) -> Path:
        return self.path / "checkpoints"

    @property
    def metrics(self) -> Path:
        return self.path / "metrics"

    @property
    def predictions(self) -> Path:
        return self.path / "predictions"

    @property
    def figures(self) -> Path:
        return self.path / "figures"

    @property
    def reports(self) -> Path:
        return self.path / "reports"

    @property
    def runs(self) -> Path:
        p = self.path / "runs"
        p.mkdir(exist_ok=True)
        return p

    # -- tables -----------------------------------------------------------
    def table_path(self, name: str, sub: str = "metrics", ext: str = "csv") -> Path:
        return self.path / sub / f"{name}.{ext}"

    def has_table(self, name: str, sub: str = "metrics") -> bool:
        return self.table_path(name, sub).exists()

    def save_table(self, df: pd.DataFrame, name: str, sub: str = "metrics") -> Path:
        return atomic_save(self.table_path(name, sub), lambda t: df.to_csv(t, index=False))

    def append_table(self, df: pd.DataFrame, name: str, sub: str = "metrics", key: list[str] | None = None) -> Path:
        """Append rows (dedup on ``key``, latest wins) — used for per-fold / per-split incremental results."""
        p = self.table_path(name, sub)
        if p.exists():
            df = pd.concat([pd.read_csv(p), df], ignore_index=True)
            if key:
                df = df.drop_duplicates(key, keep="last")
        return self.save_table(df, name, sub)

    def load_table(self, name: str, sub: str = "metrics") -> pd.DataFrame:
        return pd.read_csv(self.table_path(name, sub))

    def save_parquet(self, df: pd.DataFrame, name: str, sub: str = "metrics") -> Path:
        return atomic_save(self.table_path(name, sub, "parquet"), lambda t: df.to_parquet(t, index=False))

    def save_json(self, obj, name: str, sub: str = "reports") -> Path:
        return atomic_write_json(self.path / sub / f"{name}.json", obj)

    def save_text(self, text: str, name: str, sub: str = "reports") -> Path:
        return atomic_write_text(self.path / sub / name, text)

    # -- figures / rasters ------------------------------------------------
    def save_fig(self, fig, name: str, svg: bool = True, dpi: int = 200) -> Path:
        png = self.figures / f"{name}.png"
        atomic_save(png, lambda t: fig.savefig(t, dpi=dpi, bbox_inches="tight", format="png"))
        if svg:
            atomic_save(self.figures / f"{name}.svg", lambda t: fig.savefig(t, bbox_inches="tight", format="svg"))
        return png

    def raster_path(self, name: str) -> Path:
        return self.predictions / f"{name}.tif"

    def save_raster(self, layers: dict[str, np.ndarray], name: str, grid) -> Path:
        from gems_eda.io import write_geotiff

        return atomic_save(self.raster_path(name), lambda t: write_geotiff(t, layers, grid))

    def read_raster(self, name: str) -> np.ndarray:
        from ..ensemble import read_map

        return read_map(self.raster_path(name))

    # -- log ----------------------------------------------------------------
    def log(self, msg: str, echo: bool = True) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
        with open(self.path / "logs" / "run.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if echo:
            print(line)

    # -- contract ---------------------------------------------------------
    def finalize_contract(self, kind: str) -> list[str]:
        """Write a stub for every REQUIRED_TABLE the experiment did not produce; returns the stub names."""
        stubs = []
        for t in REQUIRED_TABLES:
            if not self.has_table(t):
                self.save_table(pd.DataFrame([{"not_applicable": True, "experiment_kind": kind}]), t)
                stubs.append(t)
        return stubs
