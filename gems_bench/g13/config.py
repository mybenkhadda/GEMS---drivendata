"""G13 configuration: execution modes, stage flags, resource budgets.

One ``G13Config`` drives the notebook and the CLI. ``from_mode`` returns the documented defaults of a mode;
anything can be overridden with a dict or a YAML file (``configs/g13/*.yaml``). Keys are validated: a typo raises
with the closest valid key, so a silently ignored flag cannot change an experiment.
"""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

MODES = ("FAST_BASELINE", "FEATURE_DISCOVERY", "FULL_RESEARCH", "FINAL_SUBMISSION")

# Stage order. Every stage reads only artifacts of earlier stages.
STAGES = (
    "inventory",  # 03 geospatial inventory and contracts
    "splits",  # 18 geographic CV + untouched test + whole-fault discovery splits
    "features",  # 09-13 physics features (raw only in FAST_BASELINE)
    "quality",  # 14 feature quality and registry
    "eda",  # 05-08 multiscale, contrasts, profiles, directional analysis
    "regimes",  # 15-16 geological regimes and fault archetypes
    "baseline",  # 22 tier-1 raw model -> S0 submission
    "ablation",  # 21 feature-family ablation
    "pu_learning",  # 23 incomplete-label strategies
    "deep",  # 22/24 optional tier-2 multitask U-Net (torch)
    "fusion",  # 27-28 disagreement + adaptive mixture of experts
    "topology",  # 25-26 candidate extraction, graphs, discovery atlas
    "dti_opt",  # 29-30 metric sensitivity + postprocessing tournament
    "failure_modes",  # 17 error taxonomy on held-out predictions
    "submission",  # 31-33 candidates A/B/C, validation, export
    "report",  # 39 findings + Markdown/HTML report
)

MODE_STAGES = {
    "FAST_BASELINE": ("inventory", "splits", "features", "baseline", "dti_opt", "submission", "report"),
    "FEATURE_DISCOVERY": (
        "inventory",
        "splits",
        "features",
        "quality",
        "eda",
        "regimes",
        "baseline",
        "dti_opt",
        "submission",
        "report",
    ),
    "FULL_RESEARCH": STAGES,
    "FINAL_SUBMISSION": ("inventory", "splits", "dti_opt", "submission", "report"),
}

FEATURE_FAMILIES = (
    "magnetic",
    "gravity",
    "conductivity",
    "depth",
    "terrain",
    "directional",
    "structural",
    "cross_modal",
    "regional_norm",
    "validity",
)


@dataclass
class G13Config:
    mode: str = "FAST_BASELINE"
    # ---- paths
    drive_root: str = "/content/drive/MyDrive/GEMS_2"
    work_dir: str = "/content/gems_g13_work"
    out_dir: str = "/content/drive/MyDrive/GEMS_2/outputs/G13_GEODISCOVERY"
    raw_path: str = ""  # gems-geodawn-numerical-features.tif (discovered when empty)
    known_path: str = ""  # existing_faults.tif
    template_path: str = ""  # example_submission.tif
    dem_dir: str = ""  # optional DEM-derived 100 m rasters (G07 dem_1m_100m)
    # ---- stage control
    stages: tuple = ()  # empty -> MODE_STAGES[mode]
    skip_stages: tuple = ()
    force_stages: tuple = ()  # recompute even if a verified artifact exists
    seed: int = 2026
    # ---- validation
    n_geo_folds: int = 4  # one fold becomes the untouched TEST partition
    block_km: float = 20.0
    buffer_km: float = 13.0  # embargo between training and held-out geography
    discovery_splits: int = 3
    discovery_link_km: float = 1.0
    discovery_frac_km: float = 0.12
    discovery_ignore_km: float = 1.0
    discovery_eval_km: float = 10.0
    primary_task: str = "disc_official"  # hidden-test analogue (gems3.official); "geo" is reported too
    # ---- features
    families: tuple = FEATURE_FAMILIES
    scales_m: tuple = (200, 400, 800, 1600, 3200)
    upward_continuation_m: tuple = (500, 1000, 2000, 4000)
    gabor_orientations: int = 8
    store_dtype: str = "float16"  # label-free robust-scaled storage; tree models are monotone-invariant
    # ---- sampling / models
    positive_radius_px: int = 1  # positives: within 100 m of a mapped trace
    ignore_ring_px: int = 3  # 100-300 m ring: unlabeled-near, weight 0 by default
    negative_ratio: float = 6.0
    max_positives: int = 120_000
    tabular_models: tuple = ("xgboost", "lightgbm")
    n_estimators: int = 400
    learning_rate: float = 0.05
    max_depth: int = 6
    n_jobs: int = -1
    pu_strategies: tuple = ("standard", "distance_stratified", "soft_negative", "hard_negative")
    run_deep: bool = False  # tier-2 network; opt-in (GPU)
    deep_epochs: int = 12
    deep_tile: int = 256
    # ---- postprocessing / submission
    area_fracs: tuple = (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12)
    known_ring_px: tuple = (0, 1, 2)
    allow_soft_outputs: bool = True
    submission_known_policy: str = "zero"  # zero on known pixels (neutral under the official pixel-exact mask)
    # ---- resources
    max_ram_gb: float = 0.0  # 0 -> detected
    chunk_px: int = 512
    halo_px: int = 96
    # ---- misc
    figures: bool = True
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_mode(cls, mode: str, **overrides) -> "G13Config":
        if mode not in MODES:
            raise ValueError(f"mode {mode!r} not in {MODES}")
        base = {"mode": mode}
        if mode == "FAST_BASELINE":
            base.update(families=("validity",), tabular_models=("xgboost",), pu_strategies=("standard",), n_estimators=300)
        elif mode == "FEATURE_DISCOVERY":
            base.update(tabular_models=("xgboost",), pu_strategies=("standard",))
        elif mode == "FINAL_SUBMISSION":
            base.update(pu_strategies=())
        base.update(overrides)
        return cls.from_dict(base)

    @classmethod
    def from_dict(cls, d: dict) -> "G13Config":
        valid = {f.name for f in fields(cls)}
        for k in d:
            if k not in valid:
                close = difflib.get_close_matches(k, valid, n=1)
                raise KeyError(f"unknown G13 config key {k!r}" + (f"; did you mean {close[0]!r}?" if close else ""))
        d = {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}
        cfg = cls(**d)
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> "G13Config":
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        mode = raw.pop("mode", overrides.pop("mode", "FAST_BASELINE"))
        raw.update(overrides)
        return cls.from_mode(mode, **raw)

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode {self.mode!r} not in {MODES}")
        bad = [s for s in (*self.stages, *self.skip_stages, *self.force_stages) if s not in STAGES]
        if bad:
            raise ValueError(f"unknown stages {bad}; valid: {STAGES}")
        bad = [f for f in self.families if f not in FEATURE_FAMILIES]
        if bad:
            raise ValueError(f"unknown feature families {bad}")
        if self.n_geo_folds < 3:
            raise ValueError("n_geo_folds >= 3 (two CV folds + one untouched test fold)")
        if self.primary_task not in ("disc_official", "geo"):
            raise ValueError("primary_task must be 'disc_official' or 'geo'")
        if self.submission_known_policy not in ("zero", "keep"):
            raise ValueError("submission_known_policy must be 'zero' or 'keep'")
        if self.store_dtype not in ("float16", "float32"):
            raise ValueError("store_dtype must be float16 or float32")

    # ------------------------------------------------------------------ helpers
    def active_stages(self) -> tuple:
        st = self.stages or MODE_STAGES[self.mode]
        return tuple(s for s in STAGES if s in st and s not in self.skip_stages)

    def as_dict(self) -> dict:
        return copy.deepcopy(asdict(self))

    def section(self, *keys: str) -> dict:
        d = self.as_dict()
        return {k: d[k] for k in keys}

    def fingerprint(self, *keys: str) -> str:
        """Stable hash of the semantic keys a stage depends on (paths and resource keys excluded)."""
        payload = json.dumps(self.section(*keys) if keys else self.as_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# Semantic configuration keys each stage depends on (a change invalidates that stage and its dependants).
STAGE_KEYS = {
    "inventory": (),
    "splits": (
        "seed",
        "n_geo_folds",
        "block_km",
        "buffer_km",
        "discovery_splits",
        "discovery_link_km",
        "discovery_frac_km",
        "discovery_ignore_km",
        "discovery_eval_km",
    ),
    "features": ("families", "scales_m", "upward_continuation_m", "gabor_orientations", "store_dtype"),
    "quality": (),
    "eda": ("seed",),
    "regimes": ("seed",),
    "baseline": (
        "seed",
        "positive_radius_px",
        "ignore_ring_px",
        "negative_ratio",
        "max_positives",
        "n_estimators",
        "learning_rate",
        "max_depth",
        "tabular_models",
    ),
    "ablation": (
        "seed",
        "positive_radius_px",
        "ignore_ring_px",
        "negative_ratio",
        "max_positives",
        "n_estimators",
        "learning_rate",
        "max_depth",
    ),
    "pu_learning": (
        "seed",
        "pu_strategies",
        "positive_radius_px",
        "ignore_ring_px",
        "negative_ratio",
        "max_positives",
        "n_estimators",
    ),
    "deep": ("seed", "run_deep", "deep_epochs", "deep_tile"),
    "fusion": ("seed",),
    "topology": ("seed", "area_fracs"),
    "dti_opt": ("area_fracs", "known_ring_px", "allow_soft_outputs", "primary_task"),
    "failure_modes": (),
    "submission": ("submission_known_policy",),
    "report": (),
}

STAGE_DEPS = {
    "inventory": (),
    "splits": ("inventory",),
    "features": ("inventory",),
    "quality": ("features", "splits"),
    "eda": ("features", "splits"),
    "regimes": ("features", "splits"),
    "baseline": ("features", "splits"),
    "ablation": ("features", "splits"),
    "pu_learning": ("features", "splits"),
    "deep": ("features", "splits"),
    "fusion": ("baseline",),
    "topology": ("baseline",),
    "dti_opt": ("baseline",),
    "failure_modes": ("baseline",),
    "submission": ("dti_opt",),
    "report": (),
}
