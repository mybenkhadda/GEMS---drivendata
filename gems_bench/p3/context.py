"""Phase-3 output layout and a single context object reloaded from Phase-1/2 artifacts.

Nothing here recomputes Phase-1/2 work: data, folds, discovery splits, segment rasters,
features (FeatureStore cache) and every finished OOF / discovery map are reloaded from disk.

Layout of ``outputs/phase3/``::

    experiments/<exp_id>/   one folder per new run (same schema as Phase 2, see persist.py)
    tables/                 audit and comparison CSVs
    figures/                PNG + SVG
    datasets/               Parquet / GeoPackage / GeoTIFF
    submissions/            leaderboard probe files (written, never uploaded)
    manifest.json           name -> {path, kind, description}
    experiment_registry.csv one row per finished Phase-3 run
    experiment_manifest.csv one row per experiment definition (id, params, folds, artifacts)
    *.csv / *.png / *.md    headline deliverables (calibration_audit.csv, PHASE3_FINDINGS.md, ...)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import BenchConfig, BenchManifest, BenchPaths
from ..data import NativeData, load_native
from ..ensemble import read_map
from ..features import FeatureStore, build_specs
from ..folds import DiscoverySplits, GeoFolds
from ..persist import ExperimentRun


@dataclass(frozen=True)
class Phase3Paths(BenchPaths):
    """BenchPaths whose run outputs live in outputs/phase3; the feature cache stays in Phase 2."""

    @property
    def out(self) -> Path:
        return self.root / "outputs" / "phase3"

    @property
    def bench(self) -> Path:
        return self.root / "outputs" / "benchmark"

    @property
    def cache(self) -> Path:                       # never duplicate the Phase-2 feature cache
        return self.bench / "cache" / "features"

    @property
    def datasets(self) -> Path:
        return self.out / "datasets"

    def ensure(self) -> "Phase3Paths":
        for p in (self.experiments, self.figures, self.tables, self.submissions, self.datasets):
            p.mkdir(parents=True, exist_ok=True)
        return self


# Phase-2 runs reused as the model zoo: name -> experiment folder (relative to outputs/benchmark/experiments)
PHASE2_RUNS = {
    "B0_lineament": "b0_lineament_fused_s2026",
    "B0_terrain": "b0_terrain_ridge_s2026",
    "G1": "lgbm_G1_s2026",
    "G2": "lgbm_G2_s2026",
    "G3": "lgbm_G3_s2026",
    "G4": "lgbm_G4_s2026",
}
PHASE2_ENSEMBLES = {"ens_mean": "ens_mean", "ens_rank_mean": "ens_rank_mean", "ens_weighted": "ens_weighted"}
DISCOVERY_MODELS = {"G1": "lgbm_G1_s2026", "G2": "lgbm_G2_s2026", "G3": "lgbm_G3_s2026"}

# fixed colour per model (validated default categorical order; colour follows the entity, never its rank)
MODEL_COLORS = {
    "G3": "#2a78d6", "G1": "#eb6834", "B0_lineament": "#1baf7a", "G4": "#eda100",
    "CNN_A": "#e87ba4", "CNN_B": "#008300", "G2": "#6250d6", "B0_terrain": "#e34948",
}
NEUTRAL = "#8a8985"


def model_color(name: str) -> str:
    for k, c in MODEL_COLORS.items():
        if name == k or name.startswith(k + "_") or name.startswith(k + ":"):
            return c
    return NEUTRAL


@dataclass
class P3Context:
    cfg: BenchConfig
    bench: BenchPaths                    # Phase-2 paths (features, folds, experiments)
    paths: Phase3Paths                   # Phase-3 outputs
    manifest: BenchManifest
    data: NativeData
    store: FeatureStore
    folds: GeoFolds
    splits: DiscoverySplits
    seg: np.ndarray | None = None        # native segment-id raster (-1 = none)
    seg_tab: pd.DataFrame | None = None
    _maps: dict = field(default_factory=dict)

    # -- model zoo ----------------------------------------------------------
    def phase2_run(self, name: str) -> ExperimentRun:
        exp = PHASE2_RUNS[name]
        family = "baseline" if name.startswith("B0") else "gbm"
        return ExperimentRun(self.bench, exp, family)

    def oof(self, name: str) -> np.ndarray:
        """Stitched OOF map (raw scale, NaN outside valid) of a Phase-2 run, ensemble or Phase-3 run."""
        if name not in self._maps:
            if name in PHASE2_RUNS:
                p = self.bench.experiments / PHASE2_RUNS[name] / "oof.tif"
            elif name in PHASE2_ENSEMBLES:
                p = self.bench.ensembles / f"{PHASE2_ENSEMBLES[name]}.tif"
            else:
                p = self.paths.experiments / name / "oof.tif"
            if not p.exists():                   # has_oof() relies on FileNotFoundError (rasterio raises its own)
                raise FileNotFoundError(f"no oof.tif for {name}: {p}")
            self._maps[name] = read_map(p)
        return self._maps[name]

    def register_map(self, name: str, arr: np.ndarray) -> None:
        self._maps[name] = arr

    def available_oof(self) -> list[str]:
        names = [k for k, v in PHASE2_RUNS.items() if (self.bench.experiments / v / "oof.tif").exists()]
        names += [k for k, v in PHASE2_ENSEMBLES.items() if (self.bench.ensembles / f"{v}.tif").exists()]
        return names

    def discovery_pred(self, model: str, d: int) -> np.ndarray | None:
        """Phase-2 pseudo-discovery prediction (raw scale, predicted inside the eval zone only)."""
        if model == "B0_lineament":
            return self.oof("B0_lineament")      # label-free map: valid for every split
        exp = DISCOVERY_MODELS.get(model)
        p = self.bench.experiments / f"disc{d}_{exp}" / "pred.tif" if exp else None
        if p is None or not p.exists():
            p = self.paths.experiments / f"disc{d}_{model}" / "pred.tif"
        return read_map(p) if p.exists() else None

    # -- small helpers ------------------------------------------------------
    @property
    def fold_ids(self) -> list[int]:
        return list(range(self.folds.n_folds))

    def dist_to_labels_km(self) -> np.ndarray:
        if "_dist_km" not in self._maps:
            from scipy import ndimage

            g = self.data.grid
            self._maps["_dist_km"] = ndimage.distance_transform_edt(
                ~self.data.labels, sampling=(g.pixel_km_y, g.pixel_km_x)).astype(np.float32)
        return self._maps["_dist_km"]

    def save_table(self, df: pd.DataFrame, name: str, desc: str = "", where: Path | None = None) -> Path:
        p = (where or self.paths.tables) / f"{name}.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        self.manifest.add(name, p, "table", desc)
        return p

    def save_fig(self, fig, name: str, desc: str = "", where: Path | None = None, svg: bool = True, dpi: int = 140) -> Path:
        d = where or self.paths.figures
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{name}.png"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        if svg:
            fig.savefig(d / f"{name}.svg", bbox_inches="tight")
        self.manifest.add(name, p, "figure", desc)
        return p

    def save_raster(self, layers: dict[str, np.ndarray], name: str, desc: str = "") -> Path:
        from gems_eda.io import write_geotiff

        p = self.paths.datasets / f"{name}.tif"
        write_geotiff(p, {k: np.asarray(v, np.float32) for k, v in layers.items()}, self.data.grid)
        self.manifest.add(name, p, "raster", desc)
        return p


def load_context(root: Path | None = None, with_segments: bool = True, *, fold_dir: Path | None = None,
                 disc_dir: Path | None = None) -> P3Context:
    """Reload everything Phase 3 needs from Phase-1/2 artifacts (no recomputation of folds or features).

    ``fold_dir`` / ``disc_dir`` select another (versioned) validation design; default = the Phase-2 protocol."""
    cfg = BenchConfig()
    bench = BenchPaths.discover(root)
    p3 = Phase3Paths(bench.root).ensure()
    man = BenchManifest(p3)
    data = load_native(bench)
    store = FeatureStore(data, build_specs(data.band_names, include_eda=True))
    fold_dir = Path(fold_dir) if fold_dir else \
        bench.folds / f"geo_k{cfg.n_folds}_b{cfg.block_km:g}km_buf{cfg.buffer_km:g}km_s{cfg.seed}"
    disc_dir = Path(disc_dir) if disc_dir else \
        bench.folds / f"discovery_d{cfg.n_discovery_folds}_link{cfg.system_link_km:g}km_s{cfg.seed}"
    if not (fold_dir / "fold_map.tif").exists() or not (disc_dir / "discovery_map.tif").exists():
        raise FileNotFoundError("Phase-2 folds / discovery splits missing: run the master pipeline stages "
                                "spatial_folds + discovery_splits (or notebook 02 sections 0-2)")
    folds = GeoFolds.load(fold_dir, data.grid)
    splits = DiscoverySplits.load(disc_dir, data.grid)
    seg, seg_tab = None, None
    if with_segments:
        from .. import segments

        sp = bench.out / "datasets" / "native_segment_ids.tif"
        if sp.exists():
            seg = segments.load_segment_raster(sp)
        tp = bench.out / "datasets" / "segment_attributes.parquet"
        seg_tab = pd.read_parquet(tp) if tp.exists() else segments.load_segment_table(bench.eda)
    return P3Context(cfg, bench, p3, man, data, store, folds, splits, seg, seg_tab)


def write_experiment_manifest(ctx: P3Context, rows: list[dict]) -> Path:
    """Append experiment definitions (id, kind, params, folds, artifact paths) to experiment_manifest.csv."""
    p = ctx.paths.out / "experiment_manifest.csv"
    df = pd.DataFrame([{k: (json.dumps(v, default=str) if isinstance(v, (dict, list, tuple)) else v)
                        for k, v in r.items()} for r in rows])
    if p.exists():
        df = pd.concat([pd.read_csv(p), df], ignore_index=True).drop_duplicates("exp_id", keep="last")
    df.to_csv(p, index=False)
    ctx.manifest.add("experiment_manifest", p, "table", "Phase-3 experiment definitions and artifact paths")
    return p
