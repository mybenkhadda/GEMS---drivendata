"""P5Spec: the Phase-4 spec (P4Spec -> CNN3Spec -> CNNSpec) plus Phase-5 model, loss and runtime fields.

A P5Spec with default Phase-5 fields *is* a P4Spec run: ``model="geounet"``, ``pu="none"``,
``orient_loss="mse"`` and no geometry weights reproduce the Phase-4 objective and architecture exactly.
Specs are resolved from an experiment config by ``spec_from_experiment`` (base run + overrides) and stored
verbatim (``asdict``) in every checkpoint, so a checkpoint always carries the spec that produced it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace

from ..phase4_training import P4Spec


@dataclass
class P5Spec(P4Spec):
    # model
    model: str = "geounet"                 # geounet (Phase-4 GeoUNet) | geostructurenet
    gsn_groups: dict = field(default_factory=dict)   # {"terrain": [...], "potential": [...], "context": [...]}
    gsn_fusion: str = "gated"              # concat | gated | attention | crossscale
    gsn_long_range: str = "none"           # none | dilated | fourier | pyramid
    gsn_heads: int = 4
    # positive-unlabeled supervision (Experiment B)
    pu: str = "none"                       # none (B0) | bg_weight (B1) | nnpu (B2)
    pu_prior: float = 0.01
    bg_evidence: str = "map:B0_lineament"  # label-free only: map:<LABEL_FREE_MAPS> | feature:<name>
    bg_lambda: float = 0.5
    bg_q0: float = 0.8
    bg_min_weight: float = 0.2
    consistency: float = 0.0               # B3 mean-teacher weight (0 = off)
    ema_decay: float = 0.99
    consistency_rampup: int = 5
    consistency_shift_px: int = 8
    consistency_noise: float = 0.05
    # geometry (Experiment C); aux_dist / aux_orient weights are inherited from P4Spec
    orient_loss: str = "cosine"            # cosine | mse (Phase-4 parity)
    orient_junction_px: float = 3.0
    orient_min_length_km: float = 1.0
    w_centerline: float = 0.0
    w_thin: float = 0.0
    w_continuity: float = 0.0
    continuity_step_px: float = 1.0
    # runtime / robustness
    ckpt_every: int = 5                    # resumable checkpoint every n epochs (and at budget / end)
    ckpt_sync_min: float = 10.0            # push resume checkpoints to Drive at most every n minutes
    max_runtime_min: float = 0.0           # per-stage budget (0 = unlimited): stop cleanly, resumable
    oom_retries: int = 2                   # halve batch / double grad_accum on CUDA OOM
    collapse_max_unlabeled_mean: float = 0.35   # stop if mean p on unlabeled pixels exceeds this


P5_FIELDS = {f.name for f in fields(P5Spec)}


def spec_from_dict(d: dict) -> P5Spec:
    return P5Spec(**{k: v for k, v in d.items() if k in P5_FIELDS})


def base_spec_dict(ctx, base: str) -> dict:
    """config.json of a zoo run (exact Phase-3/4 settings); ``base`` is a zoo name or an experiment folder."""
    from ..models.zoo import BY_NAME, experiment_dir

    d = experiment_dir(ctx, base) if base in BY_NAME else ctx.run_dir(base)
    return json.loads((d / "config.json").read_text(encoding="utf-8"))


def spec_from_experiment(ctx, model_cfg: dict, name: str, seed: int) -> P5Spec:
    """Resolve ``model:`` of an experiment YAML.

    ``base``         zoo name whose config.json is the starting point (e.g. P4_B_gated)
    ``type``         geounet | geostructurenet
    ``add_channels`` {group: [feature names]} appended to the base groups (group "context" = context branch)
    ``drop_channels`` [names] removed everywhere
    ``overrides``    any P5Spec field
    """
    cfg = dict(base_spec_dict(ctx, model_cfg["base"])) if model_cfg.get("base") else {}
    groups = {k: list(v) for k, v in (cfg.get("groups") or {}).items()}
    for g, names in (model_cfg.get("add_channels") or {}).items():
        groups.setdefault(g, [])
        groups[g] += [n for n in names if n not in groups[g]]
    drop = set(model_cfg.get("drop_channels") or [])
    groups = {g: [c for c in v if c not in drop] for g, v in groups.items()}
    channels = [c for c in dict.fromkeys(c for v in groups.values() for c in v)] or \
               [c for c in cfg.get("channels", []) if c not in drop]
    typ = model_cfg.get("type", "geounet")
    kw = {**cfg, **(model_cfg.get("overrides") or {}), "name": name, "channels": channels, "seed": seed, "model": typ}
    if typ == "geostructurenet":
        kw["gsn_groups"] = groups
        kw["groups"] = {g: v for g, v in groups.items() if g != "context"}
        kw["context"] = bool(groups.get("context"))
        kw.setdefault("gsn_fusion", model_cfg.get("fusion", "gated"))
        kw["gsn_fusion"] = model_cfg.get("fusion", kw["gsn_fusion"])
        kw["gsn_long_range"] = model_cfg.get("long_range", "none")
    else:
        kw["groups"] = {g: v for g, v in groups.items() if g != "context"}
        if kw.get("arch") == "dual" and len(kw["groups"]) < 2:
            raise ValueError("dual GeoUNet needs two channel groups")
    return spec_from_dict(kw)


def with_gpu(spec: P5Spec, gpu_overrides: dict) -> P5Spec:
    return replace(spec, **{k: v for k, v in gpu_overrides.items() if k in P5_FIELDS})


def spec_hash(spec: P5Spec) -> str:
    import hashlib

    semantic = {k: v for k, v in asdict(spec).items()
                if k not in ("infer_batch", "data_on_gpu", "cudnn_benchmark", "channels_last", "eval_every",
                             "ckpt_every", "ckpt_sync_min", "max_runtime_min", "oom_retries", "notes")}
    return hashlib.sha256(json.dumps(semantic, sort_keys=True, default=str).encode()).hexdigest()[:12]
