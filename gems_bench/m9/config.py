"""M9 configuration: configs/m9/geofault_production.yaml + notebook overrides, validated.

Unknown keys are rejected (closest valid key suggested), except inside ``shared`` (handed to the M7 loader, which
validates it against configs/m7/discovery_first.yaml) and the free-form blocks in ``OPEN``. Profiles may only carry
computational-scope keys (``SCOPE_KEYS``). Variants are deep overrides of the semantic blocks in ``VARIANT_BLOCKS``
(a variant can change the model, the losses, the corruption mixture, the features; never the protocol).

``resolve_variant(cfg, name)`` returns the effective configuration of one training variant; every variant gets its
own fingerprint, checkpoints and held-out maps.
"""
from __future__ import annotations

import copy
import difflib
from pathlib import Path

CONFIG_PARTS = ("m9", "geofault_production.yaml")
OPEN = ("shared", "recorded", "profiles", "profile_overrides", "variants", "validation.representations",
        "features.families", "corruption.operators", "spectral.fields", "candidates.fields")
STAGES = ("PREPARE", "TRAIN", "VALIDATE", "TRACE", "RANK", "INFER", "SUBMIT")
SECTIONS = ("s01_baseline", "s02_features", "s03_spectral", "s04_dataset", "s05_corruption", "s06_architecture",
            "s07_losses", "s08_training", "s09_validation", "s10_dense_analysis", "s11_tracing", "s12_monte_carlo",
            "s13_rl", "s14_candidate_features", "s15_gbm_benchmark", "s16_fusion", "s17_inference", "s18_submission",
            "s19_reports")
PROFILES = ("PILOT", "FULL")
SCOPE_KEYS = ("train_folds", "train_splits", "epochs", "steps_per_epoch", "variants", "n_seeds", "max_seeds_per_unit",
              "stochastic_realizations", "mc_realizations", "rl_episodes", "n_boot", "full_region_traces",
              "audit_samples")
VARIANT_BLOCKS = ("features", "spectral", "data", "corruption", "model", "loss", "training")
CORRUPTION_OPS = ("endpoint_truncation", "interior_removal", "segment_omission", "branch_omission",
                  "local_displacement", "whole_system_removal")
GBM_LIBS = ("xgboost", "lightgbm", "catboost")


class M9ConfigError(ValueError):
    pass


def _open(path: str) -> bool:
    return any(path == p or path.startswith(p + ".") for p in OPEN)


def merge(base: dict, over: dict, path: str = "") -> dict:
    """Deep merge; unknown keys raise unless the path is open. Lists and scalars are replaced."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        here = f"{path}.{k}" if path else k
        if k not in base and not _open(path):
            close = difflib.get_close_matches(str(k), [str(x) for x in base], n=1)
            raise M9ConfigError(f"unknown M9 config key '{here}'" + (f" (did you mean '{close[0]}'?)" if close else ""))
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v, here)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_m9_config(overrides: dict | None = None, path: Path | None = None) -> dict:
    import yaml

    from ..paths import config_path

    p = Path(path) if path else config_path(*CONFIG_PARTS)
    base = yaml.safe_load(p.read_text(encoding="utf-8"))
    cfg = merge(base, overrides or {})
    validate(cfg)
    cfg["_profile"] = resolve_profile(cfg)
    return cfg


def resolve_profile(cfg: dict) -> dict:
    name = str(cfg["profile"]).upper()
    prof = dict(cfg["profiles"][name])
    prof.update(cfg.get("profile_overrides") or {})
    prof["name"] = name
    prof["provisional"] = name in (cfg.get("gates", {}).get("provisional_profiles") or [])
    return prof


def resolve_variant(cfg: dict, name: str) -> dict:
    """Effective configuration of one training variant (semantic blocks deep-merged, protocol untouched)."""
    if name not in cfg["variants"]:
        raise M9ConfigError(f"unknown variant {name!r}; defined: {sorted(cfg['variants'])}")
    over = cfg["variants"][name] or {}
    out = copy.deepcopy(cfg)
    for blk, v in over.items():
        out[blk] = merge(out[blk], v, blk)
    out["_variant"] = name
    return out


def m7_overrides(cfg: dict) -> dict:
    """The ``shared`` block as M7 overrides (M7 keeps its own output_dir: M9 only READS M7 artifacts there)."""
    sh = copy.deepcopy(cfg.get("shared") or {})
    sh["output_dir"] = cfg["m7_output_dir"]
    sh["legacy_gems3_dir"] = cfg["legacy_gems3_dir"]
    sh.setdefault("retrain_existing_cnns", False)
    return sh


def sections_of(cfg: dict, stage: str | None) -> list[str]:
    if stage in (None, "ALL"):
        return list(SECTIONS)
    st = str(stage).upper()
    if st not in STAGES:
        raise M9ConfigError(f"stage must be one of {STAGES} or ALL (got {stage!r})")
    return list(cfg["stages"][st])


def zones_of(spec, available: list[int]) -> list[int]:
    """Profile zone list (``all`` | list) intersected with the available fold / split ids."""
    if spec == "all":
        return list(available)
    return [int(z) for z in (spec or []) if int(z) in set(available)]


def validate(cfg: dict) -> None:
    if str(cfg.get("mode")).upper() != "GEOFAULT_PRODUCTION":
        raise M9ConfigError(f"mode must be GEOFAULT_PRODUCTION (got {cfg.get('mode')!r})")
    if str(cfg.get("profile")).upper() not in PROFILES:
        raise M9ConfigError(f"profile must be one of {PROFILES}")
    for k in ("m7_output_dir", "m8_output_dir", "legacy_gems3_dir"):
        if cfg["output_dir"] == cfg[k]:
            raise M9ConfigError(f"output_dir must differ from {k}: earlier artifacts are read-only")
    missing = [s for s in SECTIONS if s not in cfg["sections"]]
    if missing:
        raise M9ConfigError(f"sections missing switches: {missing}")
    staged = [s for st in STAGES for s in cfg["stages"].get(st, [])]
    if sorted(staged) != sorted(SECTIONS):
        raise M9ConfigError(f"stages must partition the sections exactly once; got {sorted(set(SECTIONS) ^ set(staged))}")
    keys = None
    for name in PROFILES:
        p = cfg["profiles"].get(name)
        if p is None:
            raise M9ConfigError(f"profiles.{name} missing")
        bad = [k for k in p if k not in SCOPE_KEYS]
        if bad:
            raise M9ConfigError(f"profiles.{name}: {bad} are not computational-scope keys ({SCOPE_KEYS})")
        if keys is None:
            keys = set(p)
        elif set(p) != keys:
            raise M9ConfigError("every profile must define the same scope keys")
        for v in p["variants"]:
            if v not in cfg["variants"]:
                raise M9ConfigError(f"profiles.{name}.variants: unknown variant {v!r}")
        if "core" not in p["variants"]:
            raise M9ConfigError(f"profiles.{name}.variants must contain 'core' (the production candidate)")
    bad = [k for k in (cfg.get("profile_overrides") or {}) if k not in SCOPE_KEYS]
    if bad:
        raise M9ConfigError(f"profile_overrides: {bad} are not scope keys")
    for v, over in cfg["variants"].items():
        bad = [b for b in (over or {}) if b not in VARIANT_BLOCKS]
        if bad:
            raise M9ConfigError(f"variants.{v}: blocks {bad} may not be changed by a variant (allowed {VARIANT_BLOCKS})")
        for blk, val in (over or {}).items():
            merge(cfg[blk], val, blk)                       # unknown keys inside a variant are refused too
    mx = cfg["corruption"]["mixture"]
    if set(mx) != {"none", "partial", "heavy"} or abs(sum(float(x) for x in mx.values()) - 1.0) > 1e-6:
        raise M9ConfigError("corruption.mixture needs none / partial / heavy summing to 1")
    bad = [o for o in cfg["corruption"]["operators"] if o not in CORRUPTION_OPS]
    if bad:
        raise M9ConfigError(f"corruption.operators: unknown {bad}; known {CORRUPTION_OPS}")
    for v, over in cfg["variants"].items():
        m = ((over or {}).get("corruption") or {}).get("mixture")
        if m is not None:
            full = {**mx, **m}
            if abs(sum(float(x) for x in full.values()) - 1.0) > 1e-6:
                raise M9ConfigError(f"variants.{v}.corruption.mixture must sum to 1")
    if not 0 < float(cfg["corruption"]["context_pixel_weight"]) <= 1:
        raise M9ConfigError("corruption.context_pixel_weight must lie in (0, 1]")
    if cfg["spectral"]["include_in_model"] not in ("auto", "always", "never"):
        raise M9ConfigError("spectral.include_in_model must be auto | always | never")
    if int(cfg["spectral"]["fft"]["window_px"]) < 16 or int(cfg["spectral"]["fft"]["stride_px"]) < 1:
        raise M9ConfigError("spectral.fft: window_px >= 16 and stride_px >= 1")
    if int(cfg["data"]["patch_px"]) % 16:
        raise M9ConfigError("data.patch_px must be a multiple of 16 (four 2x downsamplings)")
    if int(cfg["data"]["patch_px"]) // int(cfg["data"]["context_factor"]) < 16:
        raise M9ConfigError("data.patch_px / data.context_factor must be >= 16")
    if int(cfg["data"]["context_factor"]) not in (2, 4, 8, 16):
        raise M9ConfigError("data.context_factor must be 2, 4, 8 or 16 (the context branch aligns at stride 16)")
    if not 0 <= float(cfg["data"]["pu"]["unlabeled_weight"]) <= 1:
        raise M9ConfigError("data.pu.unlabeled_weight must lie in [0, 1]")
    if cfg["loss"]["l3_mode"] not in ("skeleton_recall", "cldice"):
        raise M9ConfigError("loss.l3_mode must be skeleton_recall | cldice")
    cur = cfg["loss"]["curriculum"]
    if not 0 <= float(cur["stage_a_end"]) <= float(cur["stage_b_end"]) <= 1:
        raise M9ConfigError("loss.curriculum: 0 <= stage_a_end <= stage_b_end <= 1")
    if cfg["model"]["fusion"] not in ("gated", "concat"):
        raise M9ConfigError("model.fusion must be gated | concat")
    if cfg["training"]["ckpt_policy"] not in ("rotating", "best_and_latest"):
        raise M9ConfigError("training.ckpt_policy must be rotating | best_and_latest")
    bad = [x for x in cfg["gbm"]["libraries"] if x not in GBM_LIBS]
    if bad:
        raise M9ConfigError(f"gbm.libraries: unknown {bad}; known {GBM_LIBS}")
    if cfg["rl"]["algorithm"] not in ("bc", "ppo", "dqn"):
        raise M9ConfigError("rl.algorithm must be bc | ppo | dqn")
    if cfg["inference"]["source"] not in ("fold_average", "full_model"):
        raise M9ConfigError("inference.source must be fold_average | full_model")
    if cfg["validation"]["production_context"] not in cfg["validation"]["context_modes"]:
        raise M9ConfigError("validation.production_context must be one of validation.context_modes")
    if not 0 < float(cfg["fusion"]["weight_step"]) <= 1:
        raise M9ConfigError("fusion.weight_step must lie in (0, 1]")
    if cfg["submission"]["export_m9_if"] not in ("PROMOTE", "NEVER"):
        raise M9ConfigError("submission.export_m9_if must be PROMOTE | NEVER")
    if (cfg.get("shared") or {}).get("retrain_existing_cnns"):
        raise M9ConfigError("shared.retrain_existing_cnns=True is refused: B0 / B1 are immutable references")
