"""G14.1 configuration: named execution modes, profiles, stage switches and cross-field validation.

Design rules (they fix the G14 friction documented in docs/g14_1/STATIC_CODE_AUDIT.md, S05 / S15):

* A **mode** is a named capability set (may it run model inference? train? write a submission?). Choosing a mode
  sets every capability flag; the user never toggles three booleans that can contradict each other.
* A **profile** (FAST / STANDARD / FULL) chooses which research stages run by default and how large the sweeps are.
* ``stages`` can switch individual stages off, or on when the mode allows it; an enabled stage that the mode forbids
  is an error with an actionable message, never a silent skip.
* Validation protocol, evaluation tasks and the primary submission task are checked together: a discovery primary
  task needs discovery units (``disc_only`` or ``geo_and_disc``); a geographic-only calibration can only produce a
  PROVISIONAL submission.

UNTESTED: written without execution.
"""
from __future__ import annotations

import copy
import difflib
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass

# --------------------------------------------------------------------------- vocabularies
MODES = {
    # name: (model inference allowed, training allowed, submission write allowed)
    "AUDIT_ONLY": (False, False, False),        # read artifacts + static tables only
    "DIAGNOSE": (False, False, False),          # all CPU diagnostics on existing maps (R00-R05, R07)
    "COUNTERFACTUAL": (True, False, False),     # + inference with existing checkpoints (R06)
    "TRAIN": (True, True, False),               # + corrected training (R08-R15)
    "SUBMIT": (True, False, True),              # reuse maps / checkpoints, write candidate files (R17)
}
PROFILES = ("FAST", "STANDARD", "FULL")
PROTOCOLS = ("geo_only", "disc_only", "geo_and_disc")
TASKS = ("geo", "disc_official")
BACKENDS = ("g13", "phase5")

AREA_FRACS = (0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06,
              0.08, 0.10, 0.12, 0.15, 0.20, 0.30)
G14_WEIGHTS = (0.00, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50)

# stage switch -> capability it needs ("none" = static or CPU-only on existing artifacts)
STAGE_CAPABILITY = {
    "artifact_inventory": "none", "geometry_audit": "none", "prediction_diagnostics": "none",
    "leakage_static": "none", "metric_decomposition": "none", "representation_sweep": "none",
    "calibration": "none", "champion_comparison": "none", "complementarity": "none", "failure_atlas": "none",
    "discovery_splits": "none", "inner_validation_audit": "inference",
    "checkpoint_counterfactuals": "inference", "train_models": "training", "transformers": "training",
    "submission": "submission",
}
PROFILE_STAGES = {
    "FAST": {"artifact_inventory", "geometry_audit", "prediction_diagnostics", "leakage_static",
             "metric_decomposition", "representation_sweep", "champion_comparison", "complementarity"},
    "STANDARD": {"artifact_inventory", "geometry_audit", "prediction_diagnostics", "leakage_static",
                 "metric_decomposition", "representation_sweep", "calibration", "champion_comparison",
                 "complementarity", "failure_atlas", "discovery_splits", "checkpoint_counterfactuals",
                 "inner_validation_audit"},
    "FULL": set(STAGE_CAPABILITY) - {"submission"},
}


# --------------------------------------------------------------------------- sections
@dataclass
class ValidationCfg:
    primary_task: str = "disc_official"
    secondary_task: str = "geo"
    protocol: str = "geo_and_disc"
    backend: str = "g13"                        # g13 (locked TEST) | phase5 (units of P6 W2 / P7)
    locked_test: bool = True
    embargo_m: float = 13000.0
    fault_link_m: float = 1000.0
    fault_link_angle_deg: float = 30.0


@dataclass
class ResourcesCfg:
    device: str = "auto"
    num_workers: int = 2
    mixed_precision: bool = True
    cache_enabled: bool = True
    max_minutes_per_unit: float = 60.0


@dataclass
class SweepCfg:
    area_fracs: tuple = AREA_FRACS
    representations: tuple = ("raw", "top", "crest_w1", "crest_w0", "nms", "skeleton", "soft_skeleton",
                              "sharpen", "power", "hysteresis", "segment_rank", "sparse_soft")
    calibrations: tuple = ("identity", "temperature", "power", "quantile", "logistic", "isotonic", "regime")
    masking_orders: tuple = ("mask_before", "mask_after", "mask_both", "ring1", "pixel_exact")
    g14_weights: tuple = G14_WEIGHTS
    n_boot: int = 200
    max_units: int = 0                          # 0 = all units


@dataclass
class TrainingCfg:
    experiments: tuple = ("M1_geophysics_only", "M2_conditional_discovery")
    epochs: int = 30
    samples_per_epoch: int = 2000
    batch_size: int = 8
    accumulation: int = 2
    lr: float = 3e-4
    early_stopping_patience: int = 6
    selection_metric: str = "inner_full_region_dti"   # | inner_tiled_dti | inner_loss | enriched_surrogate
    inner_eval_every: int = 2
    context_fraction: float = 0.5               # share of training fault systems used as context (M2)
    target_resample_every_epoch: bool = True


@dataclass
class G141Config:
    repo: str = "mybenkhadda/GEMS---drivendata"
    ref: str = "g14-1-critical-recovery"
    drive_root: str = "/content/drive/MyDrive/GEMS_2"
    local_root: str = "/content/gems_work"
    out_subdir: str = "outputs/G14_1_CRITICAL_RECOVERY"
    g14_out_subdir: str = "outputs/G14_GEOFAULTFORMER"
    source_experiment: str = "T1_local_cnn__da3f44d05d84"
    profile: str = "FAST"
    mode: str = "AUDIT_ONLY"
    reuse_artifacts: bool = True
    resume: bool = True
    force_rebuild: bool = False
    stages: dict = field(default_factory=dict)          # overrides of the profile's default stage set
    validation: ValidationCfg = field(default_factory=ValidationCfg)
    resources: ResourcesCfg = field(default_factory=ResourcesCfg)
    sweep: SweepCfg = field(default_factory=SweepCfg)
    training: TrainingCfg = field(default_factory=TrainingCfg)
    seed: int = 2026

    # ------------------------------------------------------------------ capabilities
    @property
    def allow_inference(self) -> bool:
        return MODES[self.mode][0]

    @property
    def allow_training(self) -> bool:
        return MODES[self.mode][1]

    @property
    def allow_submission(self) -> bool:
        return MODES[self.mode][2]

    def capability(self, cap: str) -> bool:
        return {"none": True, "inference": self.allow_inference, "training": self.allow_training,
                "submission": self.allow_submission}[cap]

    def stage_enabled(self, stage: str) -> bool:
        if stage not in STAGE_CAPABILITY:
            raise KeyError(f"unknown stage {stage!r}; known: {sorted(STAGE_CAPABILITY)}")
        on = self.stages.get(stage, stage in PROFILE_STAGES[self.profile])
        if stage == "submission" and stage not in self.stages:
            on = self.mode == "SUBMIT"
        return bool(on) and self.capability(STAGE_CAPABILITY[stage])

    def enabled_stages(self) -> list[str]:
        return [s for s in STAGE_CAPABILITY if self.stage_enabled(s)]

    @property
    def tasks(self) -> tuple:
        return {"geo_only": ("geo",), "disc_only": ("disc_official",),
                "geo_and_disc": ("geo", "disc_official")}[self.validation.protocol]

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_dict(cls, d: dict | None = None) -> "G141Config":
        cfg = cls()
        if d:
            _merge(cfg, d, "")
        cfg.validate()
        return cfg

    @classmethod
    def for_mode(cls, mode: str, profile: str = "FAST", **overrides) -> "G141Config":
        """Named execution mode with compatible defaults (the recommended way to configure a run)."""
        return cls.from_dict({"mode": mode, "profile": profile, **overrides})

    def updated(self, overrides: dict) -> "G141Config":
        c = copy.deepcopy(self)
        _merge(c, overrides, "")
        c.validate()
        return c

    def validate(self) -> None:
        err = []
        if self.mode not in MODES:
            err.append(f"mode {self.mode!r} not in {list(MODES)}")
        if self.profile not in PROFILES:
            err.append(f"profile {self.profile!r} not in {PROFILES}")
        v = self.validation
        if v.protocol not in PROTOCOLS:
            err.append(f"validation.protocol {v.protocol!r} not in {PROTOCOLS}")
        if v.backend not in BACKENDS:
            err.append(f"validation.backend {v.backend!r} not in {BACKENDS}")
        for name, t in (("primary_task", v.primary_task), ("secondary_task", v.secondary_task)):
            if t and t not in TASKS:
                err.append(f"validation.{name} {t!r} not in {TASKS}")
        if v.protocol in PROTOCOLS and v.primary_task in TASKS and v.primary_task not in self.tasks:
            err.append(f"validation.primary_task={v.primary_task!r} is not produced by protocol {v.protocol!r} "
                       f"(it produces {self.tasks}); use protocol 'geo_and_disc' or change the primary task")
        if v.locked_test and v.backend == "phase5":
            err.append("validation.locked_test needs the g13 backend (phase5 has no locked TEST partition)")
        for s in self.stages:
            if s not in STAGE_CAPABILITY:
                close = difflib.get_close_matches(s, list(STAGE_CAPABILITY), n=1)
                err.append(f"unknown stage {s!r}" + (f"; did you mean {close[0]!r}?" if close else ""))
            elif self.stages[s] and self.mode in MODES and not self.capability(STAGE_CAPABILITY[s]):
                err.append(f"stage {s!r} needs capability {STAGE_CAPABILITY[s]!r}, which mode {self.mode!r} does not "
                           f"grant; use mode {_mode_for(STAGE_CAPABILITY[s])!r} or set stages[{s!r}]=False")
        bad = [f for f in self.sweep.area_fracs if not 0 < f <= 1]
        if bad:
            err.append(f"sweep.area_fracs outside (0, 1]: {bad}")
        if list(self.sweep.area_fracs) != sorted(self.sweep.area_fracs):
            err.append("sweep.area_fracs must be increasing (ties then prefer the smallest area)")
        if any(not 0 <= w <= 1 for w in self.sweep.g14_weights):
            err.append("sweep.g14_weights must lie in [0, 1]")
        if self.sweep.g14_weights and self.sweep.g14_weights[0] != 0:
            err.append("sweep.g14_weights must start with 0.0 (the champion alone is the reference)")
        if not 0 < self.training.context_fraction < 1:
            err.append("training.context_fraction must be in (0, 1)")
        if self.training.selection_metric not in ("inner_full_region_dti", "inner_tiled_dti", "inner_loss",
                                                  "enriched_surrogate"):
            err.append(f"unknown training.selection_metric {self.training.selection_metric!r}")
        if self.force_rebuild and not self.reuse_artifacts:
            pass                                         # consistent: rebuild everything
        if err:
            raise ValueError("invalid G14.1 config:\n  " + "\n  ".join(err))

    # ------------------------------------------------------------------ identity
    def as_dict(self) -> dict:
        return asdict(self)

    def semantic_hash(self) -> str:
        d = self.as_dict()
        for k in ("repo", "ref", "drive_root", "local_root", "out_subdir", "g14_out_subdir", "mode", "profile",
                  "reuse_artifacts", "resume", "force_rebuild", "stages", "resources"):
            d.pop(k, None)
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]

    @property
    def out_root(self) -> str:
        return f"{self.drive_root.rstrip('/')}/{self.out_subdir}"

    @property
    def g14_drive_root(self) -> str:
        return f"{self.drive_root.rstrip('/')}/{self.g14_out_subdir}"


def _mode_for(cap: str) -> str:
    return {"inference": "COUNTERFACTUAL", "training": "TRAIN", "submission": "SUBMIT", "none": "DIAGNOSE"}[cap]


def _merge(obj, d: dict, prefix: str) -> None:
    valid = {f.name for f in fields(obj)}
    for k, v in d.items():
        if k == "run_mode" and "mode" in valid:          # the prompt / G14 spelling
            k = "mode"
        if k not in valid:
            close = difflib.get_close_matches(k, list(valid), n=1)
            raise KeyError(f"unknown G14.1 config key {prefix + k!r}" + (f"; did you mean {close[0]!r}?" if close else ""))
        cur = getattr(obj, k)
        if is_dataclass(cur) and isinstance(v, dict):
            _merge(cur, v, prefix + k + ".")
        elif isinstance(cur, dict) and isinstance(v, dict):
            setattr(obj, k, {**cur, **v})
        else:
            setattr(obj, k, tuple(v) if isinstance(v, list) else v)
