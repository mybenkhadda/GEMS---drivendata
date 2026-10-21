"""Benchmark configuration and output layout (all paths relative to the project root)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from gems_eda.config import find_project_root, resolve_raw


@dataclass(frozen=True)
class BenchPaths:
    root: Path

    @classmethod
    def discover(cls, start: Path | None = None) -> "BenchPaths":
        return cls(find_project_root(start)).ensure()

    # -- Phase 1 (read only) --------------------------------------------
    @property
    def eda(self) -> Path:
        return self.root / "outputs" / "eda"

    def raw(self, key: str) -> Path:
        return resolve_raw(self.root, key)

    # -- Phase 2 ---------------------------------------------------------
    @property
    def out(self) -> Path:
        return self.root / "outputs" / "benchmark"

    @property
    def cache(self) -> Path:
        """Native-grid feature cache (one .npy memmap per feature)."""
        return self.out / "cache" / "features"

    @property
    def folds(self) -> Path:
        return self.out / "folds"

    @property
    def experiments(self) -> Path:
        return self.out / "experiments"

    @property
    def figures(self) -> Path:
        return self.out / "figures"

    @property
    def tables(self) -> Path:
        return self.out / "tables"

    @property
    def submissions(self) -> Path:
        return self.out / "submissions"

    @property
    def ensembles(self) -> Path:
        return self.out / "ensembles"

    @property
    def manifest_file(self) -> Path:
        return self.out / "manifest.json"

    @property
    def registry_file(self) -> Path:
        return self.out / "experiment_registry.csv"

    def ensure(self) -> "BenchPaths":
        for p in (self.cache, self.folds, self.experiments, self.figures, self.tables,
                  self.submissions, self.ensembles):
            p.mkdir(parents=True, exist_ok=True)
        return self

    def rel(self, p: Path) -> str:
        return Path(p).resolve().relative_to(self.root.resolve()).as_posix()


@dataclass
class BenchManifest:
    """name -> {path, kind, description}; same schema as outputs/eda/manifest.json."""

    paths: BenchPaths
    entries: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.paths.manifest_file.exists():
            self.entries = json.loads(self.paths.manifest_file.read_text(encoding="utf-8"))

    def add(self, name: str, path: Path, kind: str, description: str = "") -> Path:
        self.entries[name] = {"path": self.paths.rel(path), "kind": kind, "description": description}
        self.paths.manifest_file.write_text(json.dumps(self.entries, indent=2), encoding="utf-8")
        return Path(path)

    def path(self, name: str) -> Path:
        return self.paths.root / self.entries[name]["path"]


@dataclass
class BenchConfig:
    """Benchmark parameters. Distances in metres unless suffixed _km or _px."""

    seed: int = 2026
    # metric (official definition, DrivenData problem page)
    dti_alpha: float = 0.2
    dti_beta: float = 0.8
    dti_radius_m: float = 300.0
    dti_eps: float = 1e-6            # epsilon is not specified by the organizers
    # geographic folds
    n_folds: int = 3
    block_km: float = 20.0
    clusters_per_fold: int = 3
    buffer_km: float = 13.0          # >= half a 256 px patch (12.8 km) and >= local feature support
    inner_val_frac: float = 0.15     # share of training blocks used for early stopping
    # pseudo-discovery
    n_discovery_folds: int = 3
    system_link_km: float = 1.0      # mapped pieces closer than this form one "fault system"
    discovery_frac_km: float = 0.12  # share of mapped trace length withheld per discovery fold
    discovery_min_system_km: float = 5.0
    discovery_ignore_km: float = 1.0 # withheld traces + this buffer are ignored (never negative)
    discovery_eval_km: float = 10.0  # evaluation zone radius around withheld systems
    # tabular sampling
    n_pos: int = 150_000
    n_neg_uniform: int = 300_000
    n_neg_ring: int = 150_000
    ignore_px: int = 2               # 1..ignore_px around labels: localisation uncertainty, weight 0
    ring_km: tuple[float, float] = (0.5, 2.5)
    pu_weight: float = 0.3           # weight of negatives on unsupervised lineament candidates
    pu_quantile: float = 0.97
    # LightGBM
    lgbm_params: dict = field(default_factory=lambda: {
        "objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 200,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
        "max_bin": 255, "verbose": -1,
    })
    lgbm_rounds: int = 3000
    lgbm_early_stop: int = 100
    shap_rows: int = 20_000
    # CNN
    patch_px: int = 256
    batch_size: int = 16
    epochs: int = 30
    steps_per_epoch: int = 200
    lr: float = 1e-3
    pos_patch_frac: float = 0.7
    amp: bool = True
    num_workers: int = 0
    infer_stride_frac: float = 0.5
    # reporting
    recall_tolerances_m: tuple[float, ...] = (100.0, 200.0, 300.0)
    report_threshold: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)
