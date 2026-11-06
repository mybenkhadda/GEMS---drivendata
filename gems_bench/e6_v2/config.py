"""Central configuration: one YAML file, validated, hashed per section for cache identities."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

CODE_VERSION = "e6v2.1"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "e6_v2" / "e6_v2.yaml"

REQUIRED_SECTIONS = ("paths", "raw", "grid_expected", "terrain", "potential", "dem", "normalization", "splits",
                     "model", "training", "evaluation", "export")
EXPORT_POLICIES = ("area_budget_binary", "continuous")
AREA_RULES = ("training_label_density", "fixed")


def deep_update(base: dict, upd: dict) -> dict:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Read the YAML, apply ``overrides`` (nested dict) and validate. Raises ``ValueError`` on bad values."""
    p = Path(path or DEFAULT_CONFIG)
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    deep_update(cfg, overrides or {})
    cfg["_source"] = str(p)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ValueError(f"config sections missing: {missing}")
    ex = cfg["export"]
    if ex["final"] not in EXPORT_POLICIES:
        raise ValueError(f"export.final must be one of {EXPORT_POLICIES}, got {ex['final']!r}")
    if ex["area_fraction_rule"] not in AREA_RULES:
        raise ValueError(f"export.area_fraction_rule must be one of {AREA_RULES}")
    if ex.get("fold_combination", "mean") != "mean":
        raise ValueError("export.fold_combination: only 'mean' is implemented")
    m = cfg["model"]
    if m["levels"] != 4 or len(m["dilations"]) != 4:
        raise ValueError("E6 V2 is defined with 4 encoder levels and 4 bottleneck dilations")
    t = cfg["training"]
    if m["context_pool"] != 2 ** (m["levels"] - 1):
        raise ValueError("model.context_pool must equal the bottleneck downsampling 2**(levels-1)")
    if t["patch_px"] % m["context_pool"] != 0:
        raise ValueError("training.patch_px must be a multiple of model.context_pool")
    if not 0.0 <= t["dem_dropout"] < 1.0:
        raise ValueError("training.dem_dropout must be in [0, 1)")
    d = cfg["dem"]
    largest = max(max(d["tpi_radii_m"]) * 2.0, max(d["structure_scales_m"]) * 4 + 4 * d["gradient_sigma_m"]
                  + d["scarp_offset_m"], 4 * d["curvature_sigma_m"], 8 * d["roughness_sigma_m"])
    if d["halo_m"] < largest:
        raise ValueError(f"dem.halo_m={d['halo_m']} is smaller than the largest operator support {largest} m")
    if abs(100.0 / d["work_res_m"] - round(100.0 / d["work_res_m"])) > 1e-9:
        raise ValueError("dem.work_res_m must divide the 100 m grid exactly")


def stable_hash(obj, n: int = 16) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:n]


def section_hash(cfg: dict, *sections: str) -> str:
    return stable_hash({s: cfg.get(s) for s in sections})


def module_hash(*names: str) -> str:
    """sha256 of the source files of e6_v2 modules - an edited extractor invalidates its cache."""
    here = Path(__file__).resolve().parent
    h = hashlib.sha256(CODE_VERSION.encode())
    for name in sorted(names):
        h.update(name.encode())
        h.update((here / f"{name}.py").read_bytes())
    return h.hexdigest()[:16]
