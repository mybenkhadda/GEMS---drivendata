"""Phase-4 output layout, a feature store spanning the Phase-2 cache + Phase-4 RS features, and a context.

Layout of ``outputs/phase4/``::

    cache/features/         RS feature .npy files (+ rs_feature_registry.csv, rs_feature_build_log.csv)
    experiments/<exp_id>/   one folder per run (persist.ExperimentRun schema: config, run.json, history,
                            metrics_fold, oof.tif / pred.tif, models/, figures/)
    tables/  figures/  datasets/  submissions/
    manifest.json           name -> {path, kind, description}
    experiment_registry.csv one row per finished Phase-4 run
    experiment_manifest.csv one row per experiment definition (id, section, params, features, artifacts)
    *.csv / *.md            headline deliverables (feature_registry.csv, PHASE4_FINDINGS.md, ...)

Model zoo lookup (``P4Context.oof`` / ``discovery_pred``): Phase-2 names (G1, G3, ...) as in Phase 3;
any other name is looked up in outputs/phase4/experiments first, then outputs/phase3/experiments,
through the alias table of ``configs/phase4/experiments.yaml`` (``reuse``). Nothing is recomputed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from gems_eda.enhance import standardize_signal

from . import rs_features
from .config import BenchManifest, BenchPaths
from .ensemble import read_map
from .features import FeatureStore, _fname
from .p3.context import DISCOVERY_MODELS, PHASE2_ENSEMBLES, PHASE2_RUNS, P3Context, Phase3Paths, load_context


@dataclass(frozen=True)
class Phase4Paths(BenchPaths):
    @property
    def out(self) -> Path:
        return self.root / "outputs" / "phase4"

    @property
    def bench(self) -> Path:
        return self.root / "outputs" / "benchmark"

    @property
    def phase3(self) -> Path:
        return self.root / "outputs" / "phase3"

    @property
    def cache(self) -> Path:                       # RS features only; Phase-2 features stay in outputs/benchmark
        return self.out / "cache" / "features"

    @property
    def datasets(self) -> Path:
        return self.out / "datasets"

    def ensure(self) -> "Phase4Paths":
        for p in (self.experiments, self.figures, self.tables, self.submissions, self.datasets, self.cache):
            p.mkdir(parents=True, exist_ok=True)
        return self


class Phase4FeatureStore(FeatureStore):
    """Legacy specs read from (and built into) the Phase-2 cache; ``rs:*`` specs live in the Phase-4 cache.

    Same API as FeatureStore (get / sample / load_stack / build), so lgbm3 and cnn3 run unchanged.
    The Phase-2 ``feature_registry.csv`` is never rewritten: registries go to the Phase-4 cache.
    """

    def __init__(self, data, legacy_specs: list, rs_specs: list, rs_dir: Path):
        super().__init__(data, list(legacy_specs) + list(rs_specs))
        self.rs_dir = Path(rs_dir)
        self.rs_dir.mkdir(parents=True, exist_ok=True)
        self.rs_names = [s.name for s in rs_specs]
        self._rs_set = set(self.rs_names)

    def path(self, name: str) -> Path:
        return (self.rs_dir if name in self._rs_set else self.dir) / _fname(name)

    def compute(self, spec):
        if spec.name in self._rs_set:
            return rs_features.compute(self, spec)
        return super().compute(spec)

    def build(self, names: list[str] | None = None, overwrite: bool = False, verbose: bool = True) -> pd.DataFrame:
        names = names or list(self.specs)
        log = []
        try:
            for n in names:
                if self.has(n) and not overwrite:
                    continue
                spec = self.specs[n]
                t0 = time.perf_counter()
                arr = self.compute(spec).astype(np.float32)
                arr[~self.data.valid] = np.nan
                if getattr(spec, "standardize", True):
                    z, note = standardize_signal(arr)
                else:
                    z, note = arr, "none (bounded / count feature)"
                np.save(self.path(n), z.astype(np.float32))
                self._mm.pop(n, None)
                log.append({"name": n, "seconds": round(time.perf_counter() - t0, 2), "standardization": note})
                if verbose:
                    print(f"  feature {n:<52s} {log[-1]['seconds']:7.1f}s  {note}")
        finally:
            self._src.clear()
            rs_features.clear_memo(self)
        return self.registry(extra=pd.DataFrame(log))

    def registry(self, extra: pd.DataFrame | None = None) -> pd.DataFrame:
        rows = []
        for n in self.rs_names:
            s = self.specs[n]
            d = rs_features.spec_table([s]).iloc[0].to_dict()
            d["path"] = self.data.paths.rel(self.path(n)) if self.has(n) else ""
            d["cached"] = self.has(n)
            rows.append(d)
        df = pd.DataFrame(rows)
        if extra is not None and len(extra):
            prev = self.rs_dir / "rs_feature_build_log.csv"
            log = pd.concat([pd.read_csv(prev), extra]) if prev.exists() else extra
            log.drop_duplicates("name", keep="last").to_csv(prev, index=False)
        df.to_csv(self.rs_dir / "rs_feature_registry.csv", index=False)
        return df

    def full_registry(self) -> pd.DataFrame:
        """Legacy + RS registry with a 'phase' column (legacy rows from the Phase-2 spec objects)."""
        rows = []
        for n, s in self.specs.items():
            is_rs = n in self._rs_set
            d = rs_features.spec_table([s]).iloc[0].to_dict() if is_rs else {
                **{k: getattr(s, k) for k in ("name", "modality", "source", "transform", "scale_km", "kind", "origin",
                                              "units", "support_km", "note")}, "groups": ",".join(s.groups)}
            d["phase"] = "phase4_rs" if is_rs else "phase2"
            d["cached"] = self.has(n)
            d["path"] = self.data.paths.rel(self.path(n)) if self.has(n) else ""
            rows.append(d)
        return pd.DataFrame(rows)


@dataclass
class P4Context(P3Context):
    p3paths: Phase3Paths | None = None
    cfg4: dict = field(default_factory=dict)
    rs_specs: list = field(default_factory=list)

    # -- aliases ----------------------------------------------------------
    @property
    def oof_alias(self) -> dict:
        return dict(self.cfg4.get("reuse", {}).get("oof", {}))

    @property
    def disc_alias(self) -> dict:
        return dict(self.cfg4.get("reuse", {}).get("discovery", {}))

    def _exp_roots(self) -> list[Path]:
        return [self.paths.experiments] + ([self.p3paths.experiments] if self.p3paths else [])

    def run_dir(self, exp_id: str) -> Path:
        for root in self._exp_roots():
            if (root / exp_id).exists():
                return root / exp_id
        raise FileNotFoundError(f"experiment {exp_id} not found in {[str(r) for r in self._exp_roots()]}")

    # -- model zoo ----------------------------------------------------------
    def oof(self, name: str) -> np.ndarray:
        if name in self._maps:
            return self._maps[name]
        if name in PHASE2_RUNS or name in PHASE2_ENSEMBLES:
            return super().oof(name)
        exp = self.oof_alias.get(name, name)
        for root in self._exp_roots():
            p = root / exp / "oof.tif"
            if p.exists():
                self._maps[name] = read_map(p)
                return self._maps[name]
        raise FileNotFoundError(f"no oof.tif for {name} ({exp})")

    def has_oof(self, name: str) -> bool:
        try:
            self.oof(name)
            return True
        except FileNotFoundError:
            return False

    def _disc_path(self, model: str, d: int) -> Path | None:
        if model in DISCOVERY_MODELS:
            p = self.bench.experiments / f"disc{d}_{DISCOVERY_MODELS[model]}" / "pred.tif"
            if p.exists():
                return p
        key = self.disc_alias.get(model, model)
        for root in self._exp_roots():
            p = root / f"disc{d}_{key}" / "pred.tif"
            if p.exists():
                return p
        return None

    @staticmethod
    def disc_key(model: str, d: int) -> str:
        """``_maps`` key of an in-memory discovery map (M6 registers memory-mapped / derived maps under it)."""
        return f"_disc:{model}:{d}"

    def discovery_pred(self, model: str, d: int) -> np.ndarray | None:
        if self.disc_key(model, d) in self._maps:
            return self._maps[self.disc_key(model, d)]
        if model == "B0_lineament":
            return super().discovery_pred(model, d)       # label-free map, valid for every split
        p = self._disc_path(model, d)
        return read_map(p) if p is not None else None

    def has_discovery(self, model: str) -> bool:
        if model == "B0_lineament":
            return True
        return all(self.disc_key(model, d) in self._maps or self._disc_path(model, d) is not None
                   for d in range(self.splits.n_splits))


def load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_context4(root: Path | None = None, with_segments: bool = True, experiments_cfg: Path | None = None,
                  rs_cfg: Path | None = None, *, fold_dir: Path | None = None, disc_dir: Path | None = None) -> P4Context:
    """Reload Phase-1/2 data, folds, splits, segments (via the Phase-3 loader) and attach Phase-4 paths/store."""
    base = load_context(root, with_segments, fold_dir=fold_dir, disc_dir=disc_dir)
    p4 = Phase4Paths(base.bench.root).ensure()
    cfg_dir = base.bench.root / "configs" / "phase4"
    if not cfg_dir.is_dir():                       # Colab: data workspace and code checkout are separate
        from .paths import config_path

        cfg_dir = config_path("phase4")
    cfg4 = load_yaml(experiments_cfg or cfg_dir / "experiments.yaml")
    rs_specs = rs_features.build_rs_specs(base.data.band_names, rs_features.load_rs_config(rs_cfg or cfg_dir / "rs_features.yaml"))
    legacy = list(base.store.specs.values())
    store = Phase4FeatureStore(base.data, legacy, rs_specs, p4.cache)
    return P4Context(cfg=base.cfg, bench=base.bench, paths=p4, manifest=BenchManifest(p4), data=base.data, store=store,
                     folds=base.folds, splits=base.splits, seg=base.seg, seg_tab=base.seg_tab,
                     p3paths=base.paths, cfg4=cfg4, rs_specs=rs_specs)


def phase2_features(ctx: P4Context, name: str) -> list[str]:
    """Feature list of a Phase-2 GBM run (G1 / G3 ...) exactly as trained."""
    import json

    return json.loads((ctx.phase2_run(name).dir / "config.json").read_text(encoding="utf-8"))["features"]
