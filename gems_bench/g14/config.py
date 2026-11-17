"""G14 configuration: nested dataclasses, strict validation, semantic hashing and run modes.

Rules
-----
* Unknown keys raise with the closest valid key (a typo cannot silently change an experiment).
* ``semantic_hash`` covers everything that changes a result (data views, channels, model, loss, training, splits) and
  excludes paths, device and resource knobs, so a run can resume on another GPU.
* Run modes gate side effects: training, inference and submission need an explicit mode *and* their flag.

UNTESTED: written without execution.
"""
from __future__ import annotations

import copy
import difflib
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass

RUN_MODES = ("AUDIT_ONLY", "VALIDATE_DATA", "TRAIN", "EVALUATE", "SUBMIT")
MODALITIES = ("magnetic", "gravity", "terrain", "conductivity", "geodesy", "depth", "seismicity", "structural",
              "known_context")
LOCAL_ENCODERS = ("unet", "resnet", "convnext")
REGIONAL_ENCODERS = ("swin", "mit", "none")
GLOBAL_ENCODERS = ("tokens", "none")
FUSION_MODES = ("concat", "gated", "cross_attention")
UNCERTAINTY = ("none", "mc_dropout", "heteroscedastic")
SPLIT_BACKENDS = ("g13", "phase5")
PROTOCOLS = ("spatial_cv", "spatial_cv_embargo", "leave_one_region_out", "leave_one_regime_out", "fault_system_holdout")


@dataclass
class ViewSpec:
    resolution_m: float
    size_px: int

    @property
    def extent_m(self) -> float:
        return self.resolution_m * self.size_px


@dataclass
class DataCfg:
    local: ViewSpec = field(default_factory=lambda: ViewSpec(100.0, 128))
    regional: ViewSpec = field(default_factory=lambda: ViewSpec(200.0, 256))
    global_: ViewSpec = field(default_factory=lambda: ViewSpec(800.0, 256))
    modalities: tuple = ("magnetic", "gravity", "terrain", "conductivity", "geodesy", "structural", "known_context")
    raw_only: bool = False                    # ablation: raw GeoDAWN bands (+ masks) only
    include_flip_unsafe: bool = False         # TMI / mag_anom are asymmetric at Nevada inclination
    g13_features_dir: str = "/content/gems_g13_work/features"   # G13 feature store (registry.csv + memmaps), local
    # G13 splits / regimes / tables / predictions, searched in order (local work tree first, then the Drive mirror)
    g13_artifact_dirs: tuple = ("/content/gems_g13_work", "/content/drive/MyDrive/GEMS_2/outputs/G13_GEODISCOVERY")
    build_g13_features_if_missing: bool = False
    known_context: bool = True                # permitted-known geometry channels (P6 known_features)
    store_dtype: str = "float16"
    positive_center_frac: float = 0.5         # share of local windows centred on visible faults
    min_train_frac: float = 0.6               # share of the local window inside the unit's training mask
    samples_per_epoch: int = 2000
    val_samples: int = 256
    augment: bool = True
    noise_std: float = 0.02
    modality_dropout: float = 0.15
    sensor_dropout: float = 0.05              # random rectangular dropout of one modality inside a window
    num_workers: int = 2


@dataclass
class ModelCfg:
    local_encoder: str = "unet"
    regional_encoder: str = "swin"
    global_encoder: str = "tokens"
    fusion: str = "gated"
    modality_dim: int = 32
    local_widths: tuple = (32, 64, 128, 256)
    convnext_depths: tuple = (2, 2, 6, 2)
    swin_embed: int = 64
    swin_depths: tuple = (2, 2, 4, 2)
    swin_heads: tuple = (2, 4, 8, 16)
    swin_window: int = 8
    swin_patch: int = 4
    global_tokens: int = 16
    global_dim: int = 128
    global_layers: int = 2
    context_stages: tuple = (2, 3)            # local encoder stages that receive aligned regional features
    film_global: bool = True                  # FiLM conditioning of the decoder by the global summary
    bottleneck_cross_attention: bool = True
    deep_supervision: bool = True
    heads: tuple = ("fault", "orientation")   # + "connectivity"; uncertainty via ``uncertainty``
    uncertainty: str = "mc_dropout"
    dropout: float = 0.1
    pretrained_local: str = ""                # local path to a timm-compatible checkpoint; never downloaded
    pretrained_regional: str = ""


@dataclass
class LossCfg:
    weights: dict = field(default_factory=lambda: {"masked_bce": 0.35, "soft_tversky": 0.25,
                                                   "distance_centerline": 0.20, "soft_cldice": 0.15,
                                                   "background_sparsity": 0.05})
    aux_weights: dict = field(default_factory=lambda: {"orientation": 0.1, "connectivity": 0.0, "dti_surrogate": 0.0,
                                                       "deep_supervision": 0.3})
    pos_weight: float = 8.0
    tversky_alpha: float = 0.2
    tversky_beta: float = 0.8
    cldice_iters: int = 5
    kernel_radius_px: float = 3.0             # DTI kernel radius (300 m at 100 m)
    ignore_ring_px: int = 3                   # unlabeled ring around visible faults: no negative supervision
    reliable_negative_px: int = 10            # sparsity term only beyond 1 km of any visible fault


@dataclass
class TrainCfg:
    epochs: int = 30
    batch_size: int = 8
    accumulation: int = 2
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    amp: bool = True
    early_stopping_patience: int = 6
    select_metric: str = "val_dti_surrogate"   # or "val_loss"
    seed: int = 2026
    deterministic: bool = True
    max_minutes_per_unit: float = 60.0


@dataclass
class SplitCfg:
    backend: str = "g13"                      # g13 (locked TEST) | phase5 (units shared with P6 / W2 / P7)
    protocol: str = "spatial_cv_embargo"
    embargo_m: float = 13000.0
    strict_context_embargo: bool = False      # True: embargo >= regional half-extent (label-free context allowed if False)
    fault_link_m: float = 1000.0
    fault_link_angle_deg: float = 30.0
    regime_k: int = 0                         # 0 -> reuse G13 regimes.npy if present
    tasks: tuple = ("geo", "disc_official")


@dataclass
class EvalCfg:
    area_fracs: tuple = (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3)
    representations: tuple = ("top", "crest_w1")
    primary_task: str = "disc_official"
    n_boot: int = 300
    tile_px: int = 128
    halo_px: int = 24
    tta: bool = False
    mc_samples: int = 8


@dataclass
class EnsembleCfg:
    members: tuple = ()                       # model ids with honest maps (OOF + discovery + full)
    methods: tuple = ("mean", "weighted", "oof_blend", "regime_gate", "spatial_moe")
    weight_step: float = 0.25


@dataclass
class G14Config:
    run_mode: str = "AUDIT_ONLY"
    train_enabled: bool = False
    inference_enabled: bool = False
    submission_enabled: bool = False
    repo: str = "mybenkhadda/GEMS---drivendata"
    ref: str = "g14-geofaultformer"
    drive_root: str = "/content/drive/MyDrive/GEMS_2"
    local_root: str = "/content/gems_work"
    out_subdir: str = "outputs/G14_GEOFAULTFORMER"
    device: str = "auto"
    seed: int = 2026
    experiment: str = "T1_local_cnn"
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    split: SplitCfg = field(default_factory=SplitCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    ensemble: EnsembleCfg = field(default_factory=EnsembleCfg)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_dict(cls, d: dict | None = None) -> "G14Config":
        cfg = cls()
        if d:
            _merge(cfg, d, "")
        cfg.validate()
        return cfg

    def updated(self, overrides: dict) -> "G14Config":
        c = copy.deepcopy(self)
        _merge(c, overrides, "")
        c.validate()
        return c

    def validate(self) -> None:
        err = []
        if self.run_mode not in RUN_MODES:
            err.append(f"run_mode {self.run_mode!r} not in {RUN_MODES}")
        if self.train_enabled and self.run_mode not in ("TRAIN",):
            err.append("train_enabled requires run_mode='TRAIN'")
        if self.inference_enabled and self.run_mode not in ("EVALUATE", "SUBMIT", "TRAIN"):
            err.append("inference_enabled requires run_mode TRAIN / EVALUATE / SUBMIT")
        if self.submission_enabled and self.run_mode != "SUBMIT":
            err.append("submission_enabled requires run_mode='SUBMIT'")
        m, d, lo, s = self.model, self.data, self.loss, self.split
        for name, val, allowed in (("model.local_encoder", m.local_encoder, LOCAL_ENCODERS),
                                   ("model.regional_encoder", m.regional_encoder, REGIONAL_ENCODERS),
                                   ("model.global_encoder", m.global_encoder, GLOBAL_ENCODERS),
                                   ("model.fusion", m.fusion, FUSION_MODES),
                                   ("model.uncertainty", m.uncertainty, UNCERTAINTY),
                                   ("split.backend", s.backend, SPLIT_BACKENDS), ("split.protocol", s.protocol, PROTOCOLS)):
            if val not in allowed:
                err.append(f"{name}={val!r} not in {allowed}")
        bad = [x for x in d.modalities if x not in MODALITIES]
        if bad:
            err.append(f"unknown modalities {bad}")
        if d.local.size_px % 16:
            err.append("data.local.size_px must be divisible by 16 (four stride-2 stages)")
        for v, nm in ((d.regional, "regional"), (d.global_, "global_")):
            if v.size_px % (m.swin_patch * 2 ** (len(m.swin_depths) - 1)):
                err.append(f"data.{nm}.size_px must be divisible by patch*2^(stages-1)")
            if v.extent_m < d.local.extent_m:
                err.append(f"data.{nm} must cover at least the local extent")
            if (v.resolution_m / d.local.resolution_m) % 1:
                err.append(f"data.{nm}.resolution_m must be an integer multiple of the local resolution")
        if (d.regional.size_px // m.swin_patch) % m.swin_window:
            err.append("regional tokens per side must be divisible by model.swin_window")
        if any(m.swin_embed * 2 ** i % h for i, h in enumerate(m.swin_heads)):
            err.append("swin stage dims must be divisible by their head counts")
        if len(m.swin_heads) != len(m.swin_depths):
            err.append("model.swin_heads and model.swin_depths lengths differ")
        if any(st < 0 or st >= len(m.local_widths) for st in m.context_stages):
            err.append("model.context_stages must index local encoder stages")
        if "orientation" not in m.heads and lo.aux_weights.get("orientation", 0):
            err.append("orientation loss weight > 0 but the orientation head is disabled")
        if "connectivity" not in m.heads and lo.aux_weights.get("connectivity", 0):
            err.append("connectivity loss weight > 0 but the connectivity head is disabled")
        unknown_l = set(lo.weights) - {"masked_bce", "soft_tversky", "distance_centerline", "soft_cldice",
                                       "background_sparsity"}
        unknown_a = set(lo.aux_weights) - {"orientation", "connectivity", "dti_surrogate", "deep_supervision"}
        if unknown_l or unknown_a:
            err.append(f"unknown loss terms {sorted(unknown_l | unknown_a)}")
        if sum(lo.weights.values()) <= 0:
            err.append("at least one core loss weight must be positive")
        if not 0 < lo.tversky_alpha + lo.tversky_beta:
            err.append("tversky alpha + beta must be positive")
        if s.strict_context_embargo and s.embargo_m < d.regional.extent_m / 2:
            err.append("strict_context_embargo needs split.embargo_m >= regional half-extent")
        if self.eval.primary_task not in s.tasks:
            err.append("eval.primary_task must be one of split.tasks")
        if err:
            raise ValueError("invalid G14 config:\n  " + "\n  ".join(err))

    # ------------------------------------------------------------------ hashing
    def as_dict(self) -> dict:
        return _to_dict(self)

    def semantic_dict(self) -> dict:
        d = self.as_dict()
        for k in ("run_mode", "train_enabled", "inference_enabled", "submission_enabled", "repo", "ref", "drive_root",
                  "local_root", "out_subdir", "device"):
            d.pop(k, None)
        d["data"].pop("num_workers", None)
        d["data"].pop("g13_features_dir", None)
        d["data"].pop("g13_artifact_dirs", None)
        d["train"].pop("max_minutes_per_unit", None)
        return d

    def semantic_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.semantic_dict(), sort_keys=True, default=str).encode()).hexdigest()[:12]

    @property
    def out_root(self) -> str:
        return f"{self.drive_root.rstrip('/')}/{self.out_subdir}"


# ===================================================================================== helpers
def _merge(obj, d: dict, prefix: str) -> None:
    valid = {f.name: f for f in fields(obj)}
    for k, v in d.items():
        key = "global_" if k == "global" else k
        if key not in valid:
            close = difflib.get_close_matches(key, list(valid), n=1)
            raise KeyError(f"unknown G14 config key {prefix + k!r}" + (f"; did you mean {close[0]!r}?" if close else ""))
        cur = getattr(obj, key)
        if is_dataclass(cur) and isinstance(v, dict):
            _merge(cur, v, prefix + key + ".")
        elif isinstance(cur, dict) and isinstance(v, dict):
            nd = dict(cur)
            nd.update(v)
            setattr(obj, key, nd)
        else:
            setattr(obj, key, tuple(v) if isinstance(v, list) else v)


def _to_dict(obj):
    if is_dataclass(obj):
        out = {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
        if isinstance(obj, ViewSpec):
            out["extent_m"] = obj.extent_m
        return out
    if isinstance(obj, (list, tuple)):
        return [_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


def asdict_safe(cfg) -> dict:
    return asdict(cfg)
