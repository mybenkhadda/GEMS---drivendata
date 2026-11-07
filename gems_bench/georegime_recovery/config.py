"""Configuration of the GeoRegime V2 recovery milestone (inference-only; training is forbidden).

Every threshold, band, transform and promotion gate is fixed HERE, before any held-out result is
computed. The resolved config and its hash are recorded in every stage manifest, so a change made after
looking at results is visible in the artifact trail.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

DEFAULTS: dict = {
    # --- notebook-level keys (section 00) ---
    "drive_root": "/content/drive/MyDrive/GEMS_2",
    "atlas_root": "GEMS_GEOLOGICAL_DATA_ATLAS_V1",
    "source_run": "GEMS_GEOREGIME_V1/runs/8dfa3b352417",
    "output_root": "GEMS_GEOREGIME_RECOVERY_V2",
    "scratch_root": "/content/georegime_recovery_v2",
    "repo": "mybenkhadda/GEMS---drivendata",
    "ref": "main",
    "seed": 2026,
    "train_models": False,
    "reuse_valid_cache": True,
    "compute_full_area_holdout_inference": True,
    "save_png": True,
    "save_svg": True,
    "save_csv": True,
    "save_parquet": True,
    "enable_submission_export": False,
    "code_commit": None,                       # set by the notebook bootstrap (git rev-parse HEAD)
    # --- units of the original V1 run ---
    "units": {"geographic": ["geo0", "geo1", "geo2"], "discovery": ["disc0", "disc1", "disc2"], "full": "full"},
    # --- metric (must equal gems_bench.metric constants; checked at start-up) ---
    "metric": {"radius_m": 300.0, "alpha": 0.2, "beta": 0.8},
    # --- known-fault exclusion conventions for the NEW full-area protocol ---
    # primary: exclude the mapped pixels the checkpoint could see in training (labels & train mask);
    # sensitivity: exclude every non-held mapped pixel (V1 narrow convention).
    "scoring": {"primary_exclusion": "training_visible", "sensitivity_exclusion": "all_non_held"},
    # --- diagnostics (sections 03, 04, 09, 11, 12) ---
    "diagnostics": {
        "distance_bands_m": [0, 300, 1000, 3000, None],
        "thresholds": [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
        "high_conf_quantile": 0.995,           # of eligible non-mapped pixels
        "fp_high_quantile": 0.99,              # FP-region seed threshold (per fold)
        "min_component_px": 5,
        "far_from_mapped_m": 3000.0,
        "endpoint_radius_m": 1500.0,
        "region_block_px": 100,                # 10 km blocks for regional tables and the bootstrap
        "fp_density_block_px": 50,             # 5 km blocks for FP-density maps
        "sample_block_px": 10,                 # 1 km: one sampled pixel per block for correlations
        "cumulative_max_m": 10000.0,
        "profile_max_m": 3000.0,
        "length_classes_km": [3.0, 10.0],      # short < 3 km <= medium < 10 km <= long
        "recall_missed": 0.2, "recall_correct": 0.5,
        "min_truth_per_block": 20,
        "n_regions_report": 5,
    },
    # --- expert diagnostics thresholds (descriptive labels, fixed in advance) ---
    "experts": {"redundant_spearman": 0.95, "redundant_jaccard": 0.7, "complementary_spearman": 0.8,
                "complementary_jaccard": 0.5, "weak_mixing_strength": 0.1, "material_change": 0.1,
                "top_fraction": 0.01},
    # --- full-area inference ---
    "inference": {"batch_size": 8, "snapshot_every_windows": 40, "allow_cpu": True, "reproduction_tol": 1e-2},
    # --- at most four predeclared continuous-score policies (section 14) ---
    "transforms": {
        "policies": [
            {"name": "identity", "kind": "identity"},
            {"name": "gamma_0.7", "kind": "gamma", "gamma": 0.7},
            {"name": "gamma_1.4", "kind": "gamma", "gamma": 1.4},
            {"name": "geo_support", "kind": "support", "a": 0.9, "b": 0.2,
             "support_features": ["xm_evidence_mean", "xm_multiscale_persistence"],
             "robust_quantiles": [0.05, 0.95], "missing_support": 0.5},
        ],
        "selection_protocol": "DISCOVERY_FULL_AREA",
    },
    # --- promotion gates (section 15), fixed before any held-out score exists ---
    "gates": {"min_relative_gain": 0.03, "max_fold_relative_loss": 0.05, "require_unanimous_choice": True,
              "max_geographic_relative_loss": 0.05, "require_ci_lower_above_zero": True,
              "n_bootstrap": 1000, "ci": 0.90},
    "figures": {"dpi": 200, "preview_factor": 4, "n_zoom": 3, "zoom_half_km": 8.0, "inline": True},
    # user-reported context from the brief; displayed next to recomputed values, never used as inputs
    "reference_values": {"leaderboard_dti": 0.03, "geographic_proxy_dti": 0.0995787,
                         "narrow_discovery_proxy_dti": 0.1006404, "eligible_outside_mapped": 5103418,
                         "mass_outside_mapped": 172864, "mass_within_1km": 0.7206, "area_within_1km": 0.2395,
                         "mass_beyond_3km": 0.1308, "mass_0_300": 0.4145, "mass_300_1000": 0.3061,
                         "mass_1000_3000": 0.1486, "n_lineaments": 184},
}

NOTEBOOK_KEYS = ("drive_root", "atlas_root", "source_run", "output_root", "scratch_root", "repo", "ref", "seed",
                 "train_models", "reuse_valid_cache", "compute_full_area_holdout_inference", "save_png", "save_svg",
                 "save_csv", "save_parquet", "enable_submission_export")


class ConfigError(ValueError):
    pass


def resolve(user: dict | None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    for k, v in (user or {}).items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    if cfg["train_models"] is not False:
        raise ConfigError("train_models must be False: this milestone never trains or fine-tunes a model")
    pols = cfg["transforms"]["policies"]
    if not 1 <= len(pols) <= 4:
        raise ConfigError("at most four predeclared policies are allowed")
    names = [p["name"] for p in pols]
    if "identity" not in names or len(set(names)) != len(names):
        raise ConfigError("policies must be uniquely named and include 'identity'")
    for p in pols:
        if p["kind"] not in ("identity", "gamma", "support"):
            raise ConfigError(f"unknown policy kind {p['kind']}")
        if p["kind"] == "support" and not (0 < p["a"] and p["a"] + p["b"] <= 1.5):
            raise ConfigError("support policy must stay a weak bounded adjustment")
    b = cfg["diagnostics"]["distance_bands_m"]
    if b[0] != 0 or b[-1] is not None or any(x >= y for x, y in zip(b[:-2], b[1:-1])):
        raise ConfigError("distance bands must start at 0, increase, and end open (None)")


def stable_hash(obj, n: int = 16) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


def config_hash(cfg: dict) -> str:
    semantic = {k: v for k, v in cfg.items() if k not in ("drive_root", "scratch_root", "code_commit")}
    return stable_hash(semantic)


def code_hash(files) -> str:
    h = hashlib.sha256()
    for f in sorted(Path(p) for p in files):
        h.update(f.name.encode())
        h.update(f.read_bytes() if f.is_file() else b"<missing>")
    return h.hexdigest()[:16]


def package_files() -> list[Path]:
    here = Path(__file__).resolve().parent
    root = here.parent
    return (sorted(here.glob("*.py")) + [root / "georegime" / n for n in ("model.py", "data.py", "run.py")]
            + [root / "metric.py"])
