"""Phase-5 context: the Phase-4 context plus the Phase-5 experiment zoo, feature store and workspace paths.

Nothing from Phases 1-4 is recomputed. ``load_context5`` reloads data, folds, discovery splits, segments
and the Phase-2 + Phase-4 feature stores exactly as Phase 4 did, then

* adds Phase-5 feature families (fault-normal structural profiles, aggregated 1 m DEM morphology) through
  ``features.phase5_store.Phase5FeatureStore`` (cache: ``data/feature_store/phase5``);
* resolves model maps in this order: Phase-5 experiment manifests (``experiments/*/manifest.json`` ->
  ``maps``), then the Phase-4 / Phase-3 / Phase-2 zoo (aliases from ``configs/phase4/experiments.yaml``).

``ctx.paths`` stays the Phase-4 layout on purpose: completing a Phase-4 protocol (e.g. the gated dual
encoder's pseudo-discovery runs) with the Phase-4 functions writes to ``outputs/phase4/experiments`` under the
same naming, so the result is resolved exactly like DL00 / DL02 / RS05. New Phase-5 models write into their
own experiment folders (``ctx.paths5.experiments``) and are found through the manifests.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .ensemble import read_map
from .paths import ProjectPaths, config_path
from .phase4_context import P4Context, Phase4Paths, load_context4


@dataclass
class P5Context(P4Context):
    paths5: ProjectPaths | None = None
    zoo5: dict = field(default_factory=dict)          # name -> {"oof": Path|None, "disc": {d: Path}, "exp": id}

    # -- zoo ----------------------------------------------------------------
    def refresh_zoo(self) -> dict:
        """Re-scan Phase-5 experiment manifests for produced maps (call after a run finishes)."""
        self.zoo5 = scan_experiment_maps(self.paths5.experiments) if self.paths5 else {}
        return self.zoo5

    def oof(self, name: str) -> np.ndarray:
        if name in self._maps:
            return self._maps[name]
        z = self.zoo5.get(name)
        if z and z.get("oof") and Path(z["oof"]).exists():
            self._maps[name] = read_map(z["oof"])
            return self._maps[name]
        return super().oof(name)

    def _disc_path(self, model: str, d: int) -> Path | None:
        z = self.zoo5.get(model)
        if z and d in z.get("disc", {}) and Path(z["disc"][d]).exists():
            return Path(z["disc"][d])
        return super()._disc_path(model, d)

    def register_maps(self, name: str, oof: Path | None = None, disc: dict | None = None, exp: str = "") -> None:
        self.zoo5[name] = {"oof": oof, "disc": dict(disc or {}), "exp": exp}
        self._maps.pop(name, None)

    def model_names(self) -> list[str]:
        """Every name with a resolvable OOF map (Phase-2 zoo, configured aliases and Phase-5 manifests)."""
        names = list(self.available_oof()) + list(self.oof_alias) + list(self.zoo5)
        return [n for n in dict.fromkeys(names) if self.has_oof(n)]

    # -- workspace-relative helpers -----------------------------------------
    def legacy_root_rel(self, p: Path) -> str:
        return Path(p).resolve().relative_to(self.bench.root.resolve()).as_posix()


def scan_experiment_maps(exp_root: Path) -> dict:
    out = {}
    if not Path(exp_root).exists():
        return out
    for mf in sorted(Path(exp_root).glob("*/manifest.json")):
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        base = mf.parent
        for name, v in (m.get("maps") or {}).items():
            oof = base / v["oof"] if v.get("oof") else None
            disc = {int(d): base / p for d, p in (v.get("disc") or {}).items()}
            out[name] = {"oof": oof, "disc": disc, "exp": m.get("experiment_id", base.name)}
    return out


def load_context5(root: Path | None = None, *, with_segments: bool = True, paths5: ProjectPaths | None = None,
                  profile_cfg: Path | None = None, dem_dir: Path | None = None) -> P5Context:
    """Phase-4 context (unchanged loaders) + Phase-5 feature store + Phase-5 zoo."""
    from .features.phase5_store import Phase5FeatureStore, load_profile_specs

    paths5 = paths5 or ProjectPaths.discover(root)
    exp_cfg = paths5.root / "configs" / "phase4" / "experiments.yaml"
    rs_cfg = paths5.root / "configs" / "phase4" / "rs_features.yaml"
    base = load_context4(paths5.root, with_segments,
                         experiments_cfg=exp_cfg if exp_cfg.exists() else config_path("phase4", "experiments.yaml"),
                         rs_cfg=rs_cfg if rs_cfg.exists() else config_path("phase4", "rs_features.yaml"))
    prof_specs = load_profile_specs(base.data.band_names, profile_cfg)
    store = Phase5FeatureStore.from_phase4(base.store, prof_specs, paths5.feature_store / "phase5",
                                           dem_dir=dem_dir or paths5.processed / "dem_1m_100m")
    fields_ = {f: getattr(base, f) for f in base.__dataclass_fields__}
    fields_["store"] = store
    ctx = P5Context(**fields_, paths5=paths5)
    ctx.refresh_zoo()
    return ctx


@dataclass(frozen=True)
class NestedPaths(Phase4Paths):
    """Phase-4 layout whose run outputs (experiments/, registry, tables, figures) live under ``base``.

    Lets Phase-3/4 runners (``lgbm3.run_variant_cv``, ``phase4_training``) write *inside* a Phase-5 experiment
    folder (``experiments/<id>/runs``) while reading the shared Phase-2 / Phase-4 feature caches.
    """

    base: Path = Path(".")

    @property
    def out(self) -> Path:
        return self.base

    @property
    def cache(self) -> Path:
        return self.root / "outputs" / "phase4" / "cache" / "features"

    @property
    def datasets(self) -> Path:
        return self.base / "datasets"

    def ensure(self) -> "NestedPaths":
        for p in (self.experiments, self.figures, self.tables, self.submissions, self.datasets):
            p.mkdir(parents=True, exist_ok=True)
        return self


def nested_context(ctx: P5Context, base: Path) -> P5Context:
    """Shallow copy of ``ctx`` whose legacy runners write under ``base`` (shares data, store, folds, maps)."""
    from .config import BenchManifest

    np_ = NestedPaths(ctx.bench.root, base=Path(base)).ensure()
    return replace(ctx, paths=np_, manifest=BenchManifest(np_))
