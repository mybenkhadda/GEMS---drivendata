"""Experiment persistence: one folder per run, a run manifest, and a global registry.

Layout of ``outputs/benchmark/experiments/<exp_id>/``::

    config.json        model + data config, seed, feature list, fold file
    run.json           status, timings, peak memory, environment, artifact index
    metrics_fold.csv   DTI components and diagnostics per fold
    history.csv        training curves (all folds, long format)
    importance.csv     feature importance (GBM) — gain/split per fold
    shap_summary.csv   mean |contribution| per feature on a bounded sample
    oof.tif            out-of-fold probability map on the native 100 m grid
    models/            fold models / checkpoints / preprocessing
    figures/           per-run PNGs

``ExperimentRun.done`` lets notebook cells skip completed work; every artifact is
registered in ``run.json`` with a path relative to the project root.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BenchPaths


def environment_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": os.cpu_count()}
    for mod in ("numpy", "pandas", "scipy", "rasterio", "lightgbm", "torch", "segmentation_models_pytorch", "sklearn"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def peak_rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 2**20
    except Exception:
        return None


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return o.as_posix()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


@dataclass
class ExperimentRun:
    paths: BenchPaths
    exp_id: str
    family: str                     # "gbm" | "cnn" | "ensemble" | "baseline"
    config: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "models").mkdir(exist_ok=True)
        (self.dir / "figures").mkdir(exist_ok=True)
        if self.run_file.exists():
            self.state = json.loads(self.run_file.read_text(encoding="utf-8"))
        self.state.setdefault("exp_id", self.exp_id)
        self.state.setdefault("family", self.family)
        self.state.setdefault("status", "created")
        self.state.setdefault("artifacts", {})
        self.state.setdefault("timings_s", {})

    # -- locations ------------------------------------------------------
    @property
    def dir(self) -> Path:
        return self.paths.experiments / self.exp_id

    @property
    def run_file(self) -> Path:
        return self.dir / "run.json"

    @property
    def models_dir(self) -> Path:
        return self.dir / "models"

    @property
    def oof_path(self) -> Path:
        return self.dir / "oof.tif"

    @property
    def done(self) -> bool:
        return self.state.get("status") == "done" and self.oof_path.exists()

    # -- writing --------------------------------------------------------
    def save_config(self) -> None:
        self.state["seed"] = self.config.get("seed")
        (self.dir / "config.json").write_text(json.dumps(self.config, indent=2, default=_jsonable), encoding="utf-8")
        self.register("config", self.dir / "config.json", "config")

    def _flush(self) -> None:
        self.run_file.write_text(json.dumps(self.state, indent=2, default=_jsonable), encoding="utf-8")

    def register(self, name: str, path: Path, kind: str) -> Path:
        self.state["artifacts"][name] = {"path": self.paths.rel(path), "kind": kind}
        self._flush()
        return Path(path)

    def save_table(self, df: pd.DataFrame, name: str) -> Path:
        p = self.dir / f"{name}.csv"
        df.to_csv(p, index=False)
        return self.register(name, p, "table")

    def save_parquet(self, df: pd.DataFrame, name: str) -> Path:
        p = self.dir / f"{name}.parquet"
        df.to_parquet(p, index=False)
        return self.register(name, p, "dataset")

    def save_fig(self, fig, name: str, dpi: int = 130) -> Path:
        p = self.dir / "figures" / f"{name}.png"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        return self.register(name, p, "figure")

    @contextmanager
    def timer(self, key: str):
        t0 = time.perf_counter()
        yield
        self.state["timings_s"][key] = round(time.perf_counter() - t0, 2)
        rss = peak_rss_mb()
        if rss is not None:
            self.state.setdefault("rss_mb", {})[key] = round(rss, 1)
        self._flush()

    def start(self) -> None:
        self.state.update(status="running", started=time.strftime("%Y-%m-%dT%H:%M:%S"), env=environment_info())
        self.save_config()

    def finish(self, summary: dict) -> None:
        self.state.update(status="done", finished=time.strftime("%Y-%m-%dT%H:%M:%S"), summary=summary)
        self._flush()
        append_registry(self.paths, {"exp_id": self.exp_id, "family": self.family, **summary,
                                     "dir": self.paths.rel(self.dir)})

    # -- reading --------------------------------------------------------
    def artifact(self, name: str) -> Path:
        return self.paths.root / self.state["artifacts"][name]["path"]

    def table(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.artifact(name))


def append_registry(paths: BenchPaths, row: dict) -> None:
    """One row per finished run in outputs/benchmark/experiment_registry.csv (latest wins on reload)."""
    row = {k: (json.dumps(v, default=_jsonable) if isinstance(v, (dict, list, tuple)) else v) for k, v in row.items()}
    row["registered"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    df = pd.DataFrame([row])
    f = paths.registry_file
    if f.exists():
        df = pd.concat([pd.read_csv(f), df], ignore_index=True)
    df.to_csv(f, index=False)


def load_registry(paths: BenchPaths) -> pd.DataFrame:
    if not paths.registry_file.exists():
        return pd.DataFrame()
    return pd.read_csv(paths.registry_file).drop_duplicates("exp_id", keep="last")
