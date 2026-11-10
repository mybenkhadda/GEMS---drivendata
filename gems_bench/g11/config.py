"""G11 configuration: YAML loading, validation, hashing and execution modes.

``load_config(path, overrides)`` returns a plain nested ``dict`` (validated); ``config_hash`` is the stable
identity recorded in every output manifest.
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from enum import Enum
from pathlib import Path

REQUIRED_SECTIONS = ("repository", "data_root", "output_root", "source_artifacts", "geophysical_grid", "dem", "faults",
                     "labels", "spatial_splits", "feature_engineering", "eda", "sampling", "models", "training",
                     "calibration", "inference", "candidate_generation", "ranking", "evaluation",
                     "official_submission", "hardware", "reproducibility")
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "g11_multiscale.yaml"


class Mode(str, Enum):
    AUDIT = "AUDIT"
    INGEST = "INGEST"
    FEATURES = "FEATURES"
    EDA = "EDA"
    TRAIN = "TRAIN"
    EVALUATE = "EVALUATE"
    INFER = "INFER"
    SUBMIT = "SUBMIT"
    FULL = "FULL"


MODE_STAGES = {
    Mode.AUDIT: ["audit", "report"],
    Mode.INGEST: ["audit", "ingest", "report"],
    Mode.FEATURES: ["audit", "ingest", "features", "report"],
    Mode.EDA: ["audit", "ingest", "features", "eda", "report"],
    Mode.TRAIN: ["audit", "ingest", "features", "splits", "train", "report"],
    Mode.EVALUATE: ["audit", "ingest", "features", "splits", "fusion", "evaluate", "report"],
    Mode.INFER: ["audit", "ingest", "features", "infer", "report"],
    Mode.SUBMIT: ["audit", "submit", "report"],
    Mode.FULL: ["audit", "ingest", "features", "eda", "splits", "train", "fusion", "evaluate", "infer", "submit",
                "report"],
}


class ConfigError(ValueError):
    pass


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Read the YAML config, merge ``overrides`` and validate."""
    import yaml

    p = Path(path) if path else DEFAULT_PATH
    if not p.is_file():
        raise ConfigError(f"G11 config not found: {p}")
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    cfg = deep_merge(cfg, overrides or {})
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ConfigError(f"config lacks sections {missing}")
    fr = cfg["spatial_splits"]["fractions"]
    if abs(sum(fr.values()) - 1.0) > 1e-6 or set(fr) != {"train", "validation", "stack", "test"}:
        raise ConfigError("spatial_splits.fractions must be train/validation/stack/test summing to 1")
    b = cfg["labels"]["buffers_m"]
    if sorted(b) != list(b) or min(b) <= 0:
        raise ConfigError("labels.buffers_m must be positive and increasing")
    mix = cfg["sampling"]["mix"]
    if abs(sum(mix.values()) - 1.0) > 1e-6:
        raise ConfigError("sampling.mix must sum to 1")
    if cfg["training"]["loss"] not in ("weighted_bce", "bce_dice", "focal", "bce_boundary"):
        raise ConfigError(f"unknown training.loss {cfg['training']['loss']}")
    if cfg["models"]["gated_fusion"]["mode"] not in ("geo_only", "dem_only", "concat", "gated", "cross_attention"):
        raise ConfigError("unknown gated_fusion.mode")
    for s in cfg["feature_engineering"]["terrain"]["scales_m"]:
        if s <= 0:
            raise ConfigError("terrain scales must be positive metres")


def stable_hash(obj, n: int = 16) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


def config_hash(cfg: dict) -> str:
    return stable_hash({k: v for k, v in cfg.items() if k not in ("data_root", "scratch_root")})


def code_commit(repo_dir: str | Path | None = None) -> str | None:
    """``git rev-parse HEAD`` of the checkout, or None when unavailable (never fabricated)."""
    d = Path(repo_dir) if repo_dir else Path(__file__).resolve().parents[2]
    try:
        return subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], check=True, capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def resolve_output_root(cfg: dict) -> Path:
    o = Path(cfg["output_root"])
    return o if o.is_absolute() else Path(cfg["data_root"]) / o
