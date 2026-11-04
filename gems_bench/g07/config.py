"""G07 configuration and reproducible identity.

``configs/g07/discovery.yaml`` (every scientific choice, predeclared) deep-merged with the notebook ``CONFIG`` /
``RUN``. The run directory is keyed by ``compute_run_id``: version + the DATA-ENGINEERING part of the configuration
(inputs, DEM processing, G06 store identity) + input fingerprints (+ the commit with ``run_id_mode='commit'``);
model / selection settings are cached per stage (``stage_config``) so a training change never recomputes DEM
features. ``auto_submit`` cannot be enabled.
"""
from __future__ import annotations

import copy
from pathlib import Path

from ..g06.config import deep_merge, stable_hash
from . import G07_VERSION

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "configs" / "g07" / "discovery.yaml"

NB_DEFAULTS = {
    "repo": "mybenkhadda/GEMS---drivendata", "ref": "main", "drive_root": "/content/drive/MyDrive/GEMS_2",
    "local_root": "/content/gems_work", "output_subdir": "outputs/g07_multiscale_fault_discovery", "seed": 2026,
    "dem_resolution_target_m": 1.0, "dem_tile_px": 512, "dem_halo_px": 128, "analysis_resolution_m": 100.0,
    "use_gpu": True, "reuse_g06_features": True, "reuse_w2_predictions": True, "strict_provenance": True,
    "save_figures": True, "auto_submit": False, "reuse_cached": True, "sync_to_drive": True, "run_id": None,
    "run_id_mode": "data", "run_synthetic_tests": True, "g06_run_dir": None,
}
RUN_DEFAULTS = {
    "PREFLIGHT": True, "DEM_INVENTORY": True, "DEM_COVERAGE": True, "DEM_FEASIBILITY": True, "DEM_FEATURES": True,
    "BUILD_DATASETS": True, "TRAIN_SURFACE_CLASSIC": True, "TRAIN_SUBSURFACE": True, "TRAIN_SURFACE_UNET": True,
    "GENERATE_CANDIDATES": False, "TRAIN_CANDIDATE_RANKER": False,
    "EVALUATE": True, "FULL_INFERENCE": False, "EXPORT_SUBMISSION": False,
    "REPORT": True,
}
VOLATILE = ("drive_root", "local_root", "reuse_cached", "sync_to_drive", "save_figures", "run_id", "use_gpu",
            "run_synthetic_tests", "ref", "repo", "auto_submit", "strict_provenance")
# dem keys that do not change a computed value: gating, licensing flags, catalog paths, processing geometry (tile sizes
# and halos above the operator support give identical features) and time budgets
DEM_NON_IDENTITY = ("require_go", "training_mode", "download", "catalog", "existing_products", "research_use_ok",
                    "competition_eligibility_verified", "tile_px", "halo_px", "coarse_tile_px", "fine_budget_hours",
                    "coarse_budget_hours", "stage_tile_local", "coverage")
TRAINING_MODES = ("STRICT_RESEARCH", "EXPLORATORY_TRAINING")
EXPERIMENTS = ("G07-A0", "G07-A1", "G07-A2", "G07-B0", "G07-B1", "G07-B2", "G07-C1", "G07-C2", "G07-D1", "G07-D2",
               "G07-E1")


class ConfigError(ValueError):
    pass


def _read_yaml(path: Path) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    if not isinstance(d, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return d


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Merged, validated configuration: ``overrides`` = {"notebook": CONFIG, "run": RUN, <yaml section>: {...}}."""
    ov = dict(overrides or {})
    nb = {**NB_DEFAULTS, **(ov.pop("notebook", None) or {})}
    run = {**RUN_DEFAULTS, **(ov.pop("run", None) or {})}
    y = deep_merge(_read_yaml(Path(path) if path else DEFAULT_CONFIG), ov)
    y.setdefault("dem", {})
    y["dem"]["tile_px"] = int(nb["dem_tile_px"])
    y["dem"]["halo_px"] = int(nb["dem_halo_px"])
    y["dem"]["target_resolution_m"] = float(nb["dem_resolution_target_m"])
    if nb.get("g06_run_dir"):
        y.setdefault("g06", {})["run_dir"] = nb["g06_run_dir"]
    cfg = {"version": G07_VERSION, **y, "notebook": nb, "run": run}
    validate_config(cfg)
    return cfg


def validate_config(config: dict) -> list[str]:
    """Raise ``ConfigError`` listing every problem; returns warnings."""
    errs, warn = [], []
    nb, run = config.get("notebook") or {}, config.get("run") or {}
    bad = set(run) - set(RUN_DEFAULTS)
    if bad:
        errs.append(f"unknown RUN flags {sorted(bad)}")
    if nb.get("auto_submit"):
        errs.append("auto_submit=True is refused: nothing is ever uploaded automatically")
    if float(nb.get("analysis_resolution_m", 100)) != 100.0:
        errs.append("analysis_resolution_m must be 100 (the competition grid)")
    dem = config.get("dem") or {}
    if int(dem.get("halo_px", 0)) < 0 or int(dem.get("tile_px", 0)) < 256:
        errs.append("dem_tile_px >= 256 and dem_halo_px >= 0")
    if float(dem.get("work_res_m", 1.0)) < float(dem.get("target_resolution_m", 1.0)) - 1e-9:
        errs.append("dem.work_res_m cannot be finer than the target resolution")
    for s in dem.get("scales_m") or []:
        if float(s) <= 0:
            errs.append(f"dem.scales_m: {s} must be > 0")
    rep = config.get("representation") or {}
    if not 0 < float(rep.get("primary_frac", 0.01)) < 1:
        errs.append("representation.primary_frac in (0, 1)")
    if rep.get("tie_break", "secondary_then_hash") not in ("secondary_then_hash", "hash", "legacy"):
        errs.append("representation.tie_break: secondary_then_hash | hash | legacy")
    pf = rep.get("portfolio") or {}
    if not 0 <= float(pf.get("w2_base_share", 0.85)) <= 1:
        errs.append("representation.portfolio.w2_base_share in [0, 1]")
    pr = config.get("promotion") or {}
    if float(pr.get("min_gain_v1", 0)) <= 0:
        errs.append("promotion.min_gain_v1 must be > 0")
    if run.get("EXPORT_SUBMISSION") and not run.get("FULL_INFERENCE"):
        warn.append("EXPORT_SUBMISSION needs a full-region prediction: FULL_INFERENCE (or its valid cache)")
    if run.get("TRAIN_SURFACE_UNET") and not nb.get("use_gpu"):
        warn.append("TRAIN_SURFACE_UNET with use_gpu=False: explicit CPU training (slow)")
    if dem.get("training_mode") not in (None,) + TRAINING_MODES:
        errs.append(f"dem.training_mode: {dem.get('training_mode')!r} is not one of {TRAINING_MODES} (or null)")
    if errs:
        raise ConfigError("; ".join(errs))
    return warn


def training_mode(config: dict) -> str:
    """``dem.training_mode`` when set, else ``dem.require_go`` (true = STRICT_RESEARCH, false = EXPLORATORY_TRAINING)."""
    dem = config.get("dem") or {}
    if dem.get("training_mode"):
        return str(dem["training_mode"])
    return "STRICT_RESEARCH" if dem.get("require_go", True) else "EXPLORATORY_TRAINING"


def dem_identity(dem: dict | None) -> dict:
    """The part of the ``dem`` section that determines DEM products."""
    return {k: v for k, v in (dem or {}).items() if k not in DEM_NON_IDENTITY}


def canonical_config(config: dict) -> dict:
    nb = {k: v for k, v in (config.get("notebook") or {}).items() if k not in VOLATILE}
    out = {k: copy.deepcopy(v) for k, v in config.items() if k not in ("notebook", "run", "config_dir")}
    out["notebook"] = nb
    return out


def data_identity(config: dict) -> dict:
    """What the run directory is keyed by: inputs, DEM processing, G06 store choice, seed."""
    dem = {k: v for k, v in dem_identity(config.get("dem")).items() if k != "vertical_unit_override"}
    return {"version": config.get("version"), "inputs": config.get("inputs"), "dem": dem,
            "g06": config.get("g06"), "seed": (config.get("notebook") or {}).get("seed")}


def compute_run_id(config: dict, input_fingerprints: dict | None, source_commit: str | None) -> str:
    nb = config.get("notebook") or {}
    if nb.get("run_id"):
        return str(nb["run_id"])
    fps = {k: (v or {}).get("sha256") or (v or {}).get("bytes") for k, v in (input_fingerprints or {}).items()}
    parts = [data_identity(config), fps]
    if nb.get("run_id_mode") == "commit":
        if not source_commit:
            raise ConfigError("run_id_mode='commit' needs the source commit")
        parts.append(str(source_commit))
    return "g07-" + stable_hash(*parts, n=10)


def stage_config(config: dict, stage: str) -> dict:
    c = config
    dem = c.get("dem") or {}
    unet = (c.get("surface") or {}).get("unet") or {}
    patches = {k: unet.get(k) for k in ("channels", "res_m", "patch_cells", "patch_stride_cells", "min_patch_coverage",
                                        "fits", "min_region_share")}
    m = {"preflight": {"inputs": c.get("inputs"), "g06": c.get("g06"), "w2": c.get("w2")},
         "dem_inventory": {"dem_catalog": dem.get("catalog"), "licenses": c.get("licenses"),
                           "vertical_unit_override": dem.get("vertical_unit_override"),
                           "max_accepted_res_m": dem.get("max_accepted_res_m")},
         "dem_coverage": {"dem": dem.get("coverage")},
         "dem_feasibility": {"dem": dem_identity(dem), "feasibility": c.get("feasibility"),
                             "research_use_ok": dem.get("research_use_ok")},
         "dem_features": {"dem": dem_identity(dem)},
         "g06_restore": {"g06": c.get("g06")},
         "datasets": {"datasets": c.get("datasets"), "labels": c.get("labels"), "validation": c.get("validation"),
                      "inner_cv": c.get("inner_cv"), "surface_patches": patches},
         "train_surface_classic": {"surface": {k: v for k, v in (c.get("surface") or {}).items() if k != "unet"},
                                   "models": c.get("models"), "validation": c.get("validation")},
         "train_subsurface": {"subsurface": c.get("subsurface"), "models": c.get("models"), "labels": c.get("labels"),
                              "validation": c.get("validation")},
         "train_surface_unet": {"unet": unet, "labels": c.get("labels"), "validation": c.get("validation"),
                                "inner_cv": c.get("inner_cv")},
         "generate_candidates": {"candidates": c.get("candidates")},
         "train_candidate_ranker": {"ranker": c.get("ranker"), "confidence": c.get("confidence"),
                                    "representation": c.get("representation")},
         "evaluate": {"evaluation": c.get("evaluation"), "representation": c.get("representation"),
                      "promotion": c.get("promotion")},
         "full_inference": {"inference": c.get("inference"), "representation": c.get("representation")},
         "export": {"promotion": c.get("promotion"), "export": c.get("export")}}
    return m.get(stage, {})
