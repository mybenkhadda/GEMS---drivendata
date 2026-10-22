"""Shared state of one pipeline session: configuration, roots, Drive sync, lazily loaded data and context.

Roots (same relative layout, ``gems_bench.paths``)::

    local  /content/gems_work            fast, ephemeral working copy (every stage reads and writes here)
    drive  /content/drive/MyDrive/GEMS_2 durable mirror (outputs pushed after every stage)
    code   /content/GEMS                 Git checkout (configs/)

Nothing here downloads, trains or deletes; stages do the work through the library.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .config import PipelineConfig


class PipelineStop(RuntimeError):
    """A stage cannot proceed; ``recovery`` tells the user exactly what to do."""

    def __init__(self, message: str, recovery: str = ""):
        super().__init__(message)
        self.recovery = recovery


class GPURequired(PipelineStop):
    def __init__(self, what: str):
        super().__init__(
            f"{what} needs a CUDA GPU but torch.cuda.is_available() is False in this runtime",
            "Colab menu: Runtime -> Change runtime type -> GPU (T4 / L4 / A100), reconnect, re-run the bootstrap "
            "cell and this stage. Do not reinstall torch / CUDA packages: Colab's preinstalled build is the right one. "
            "Alternatively set use_gpu=False to skip GPU stages (they are then reported as disabled).")


def fold_dir_name(v: dict, seed: int) -> str:
    """Folder of the geographic-fold design; non-default parameters that the Phase-2 name does not encode
    get an explicit suffix, so a changed protocol is always a new version (never an overwrite)."""
    name = f"geo_k{int(v['n_folds'])}_b{float(v['block_km']):g}km_buf{float(v['buffer_km']):g}km_s{seed}"
    extra = []
    if int(v["clusters_per_fold"]) != 3:
        extra.append(f"c{int(v['clusters_per_fold'])}")
    if float(v["inner_buffer_km"]) != 5.0:
        extra.append(f"ib{float(v['inner_buffer_km']):g}km")
    return name + ("_" + "_".join(extra) if extra else "")


def discovery_dir_name(v: dict, seed: int) -> str:
    name = f"discovery_d{int(v['n_discovery'])}_link{float(v['link_km']):g}km_s{seed}"
    dflt = {"frac_km": 0.12, "min_system_km": 5.0, "max_system_frac": 0.05, "ignore_km": 1.0, "eval_km": 10.0}
    tag = {"frac_km": "f", "min_system_km": "m", "max_system_frac": "x", "ignore_km": "i", "eval_km": "e"}
    extra = [f"{tag[k]}{float(v[k]):g}" for k, d in dflt.items() if float(v[k]) != d]
    return name + ("_" + "_".join(extra) if extra else "")


def fold_params(v: dict, seed: int) -> dict:
    return {"n_folds": int(v["n_folds"]), "block_km": float(v["block_km"]), "buffer_km": float(v["buffer_km"]),
            "clusters_per_fold": int(v["clusters_per_fold"]), "inner_buffer_km": float(v["inner_buffer_km"]),
            "seed": int(seed)}


def discovery_params(v: dict, seed: int) -> dict:
    return {"n_splits": int(v["n_discovery"]), "link_km": float(v["link_km"]), "frac_km": float(v["frac_km"]),
            "min_system_km": float(v["min_system_km"]), "max_system_frac": float(v["max_system_frac"]),
            "ignore_km": float(v["ignore_km"]), "eval_km": float(v["eval_km"]), "seed": int(seed)}


@dataclass
class PipelineContext:
    cfg: PipelineConfig
    urls: dict = field(default_factory=dict)          # runtime-only download URLs (never persisted)
    allow_download: bool = True
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.local = Path(self.cfg.local_root)
        self.local.mkdir(parents=True, exist_ok=True)
        dr = self.cfg.get("paths.drive_root") or ""            # "" = no Drive mirror (local / non-Colab run)
        d = Path(dr) if dr else None
        self.drive: Path | None = d if d is not None and d.exists() else None
        self.code = Path(self.cfg.code_root) if (Path(self.cfg.code_root) / "configs").is_dir() else None
        os.environ["GEMS_ROOT"] = str(self.local)
        if self.drive is not None:
            os.environ["GEMS_DRIVE_ROOT"] = str(self.drive)
        if self.code is not None:
            os.environ["GEMS_CODE_ROOT"] = str(self.code)
        from .manifests import StageManifests

        self.manifests = StageManifests(self.local, self.drive)

    # -- roots ----------------------------------------------------------------
    def root(self, which: str = "local") -> Path | None:
        return self.local if which == "local" else self.drive

    @property
    def code_root(self) -> Path:
        if self.code is not None:
            return self.code
        from ..paths import code_root

        return code_root()

    def config_file(self, *parts: str) -> Path:
        return self.code_root.joinpath("configs", *parts)

    @property
    def reports(self) -> Path:
        p = self.local / "reports" / "pipeline"
        (p / "figures").mkdir(parents=True, exist_ok=True)
        return p

    # -- Drive ------------------------------------------------------------------
    @property
    def sync(self):
        if self.drive is None:
            return None
        if "sync" not in self._cache:
            from ..artifacts.sync import DriveSync

            self._cache["sync"] = DriveSync(self.local, self.drive)
        return self._cache["sync"]

    def stage_in(self, rels: list[str]) -> None:
        if self.sync is None or not rels:
            return
        present = [r for r in rels if (self.drive / r).exists()]
        if present:
            self.sync.stage_in(present)

    def push(self, rels: list[str]) -> None:
        if self.sync is None or not rels or not self.cfg.get("sync.push_after_stage", True):
            return
        present = [r for r in rels if (self.local / r).exists()]
        if present:
            try:
                self.sync.push(present)
            except OSError as e:
                print(f"[pipeline] Drive push failed ({e}); re-run the stage or pipe.push_all() before the session ends")

    # -- protocol ---------------------------------------------------------------
    @property
    def seed(self) -> int:
        return int(self.cfg.get("seed"))

    @property
    def bench_cfg(self):
        from ..config import BenchConfig

        v = self.cfg.get("validation")
        return BenchConfig(seed=self.seed, n_folds=int(v["n_folds"]), block_km=float(v["block_km"]),
                           clusters_per_fold=int(v["clusters_per_fold"]), buffer_km=float(v["buffer_km"]),
                           n_discovery_folds=int(v["n_discovery"]), system_link_km=float(v["link_km"]),
                           discovery_frac_km=float(v["frac_km"]), discovery_min_system_km=float(v["min_system_km"]),
                           discovery_ignore_km=float(v["ignore_km"]), discovery_eval_km=float(v["eval_km"]))

    @property
    def fold_rel(self) -> str:
        return f"outputs/benchmark/folds/{fold_dir_name(self.cfg.get('validation'), self.seed)}"

    @property
    def discovery_rel(self) -> str:
        return f"outputs/benchmark/folds/{discovery_dir_name(self.cfg.get('validation'), self.seed)}"

    # -- native grid ----------------------------------------------------------
    def raw_path(self, key: str, root: Path | None = None) -> Path:
        from gems_eda.config import resolve_raw

        return resolve_raw(root or self.local, key)

    def grid(self, root: Path | None = None) -> dict | None:
        """Native grid (crs / transform / shape) from the features GeoTIFF header; None when unreadable."""
        key = f"grid:{root or self.local}"
        if key not in self._cache:
            from ..artifacts.validation import grid_of

            p = self.raw_path("features", root)
            try:
                self._cache[key] = grid_of(p) if p.exists() else None
            except Exception:
                self._cache[key] = None
        return self._cache[key]

    @property
    def data(self):
        """NativeData of the local workspace (bands read on demand; ~1 min the first time)."""
        if "data" not in self._cache:
            from ..config import BenchPaths
            from ..data import load_native

            self._cache["data"] = load_native(BenchPaths(self.local).ensure())
        return self._cache["data"]

    def ctx5(self, refresh: bool = False):
        """Full research context (Phase-4 loaders + Phase-5 store + zoo), shared by all experiments."""
        if refresh or "ctx5" not in self._cache:
            from ..models.zoo import register_aliases
            from ..paths import ProjectPaths
            from ..phase5_context import load_context5

            paths5 = ProjectPaths(self.local, self.code_root, self.drive).ensure()
            ctx = load_context5(self.local, paths5=paths5, fold_dir=self.local / self.fold_rel,
                                disc_dir=self.local / self.discovery_rel)
            register_aliases(ctx)
            self._cache["ctx5"] = ctx
        return self._cache["ctx5"]

    def project_paths(self):
        from ..paths import ProjectPaths

        return ProjectPaths(self.local, self.code_root, self.drive).ensure()

    def forget(self, *keys: str) -> None:
        for k in keys or list(self._cache):
            if k != "sync":
                self._cache.pop(k, None)

    # -- runtime --------------------------------------------------------------------
    def gpu(self) -> dict:
        if "gpu" not in self._cache:
            from ..training.gpu import detect

            self._cache["gpu"] = detect()
        return self._cache["gpu"]

    def require_gpu(self, what: str) -> None:
        if not self.gpu().get("cuda"):
            raise GPURequired(what)

    def runtime_overrides(self) -> dict:
        """Non-semantic runtime block merged into experiment configs (never changes experiment ids)."""
        g = self.cfg.get("gpu")
        return {"max_stage_minutes": g.get("max_stage_minutes", 0), "gpu": dict(g.get("overrides") or {})}
