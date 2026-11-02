"""Experiment configuration, feature selection switches and deterministic execution identity.

Three layers, deep-merged in this order::

    configs/g06/feature_store.yaml   -> cfg["feature_store"]   (sources, exact bands, grid, groups, scales, tiling)
    configs/g06/benchmarks.yaml      -> cfg["benchmarks"]      (validation, supervision, sampling, models, evaluation)
    notebook CONFIG / RUN            -> cfg["notebook"], cfg["run"]

Run identity (``compute_run_id``): the run directory is keyed by the DATA-ENGINEERING identity only - G06 version,
canonical feature-store configuration and the fingerprints of the source rasters - so changing a model or sampling
parameter never forces the feature store to be rebuilt. Everything downstream is cached per stage with its own
fingerprint (config section + upstream fingerprints + source hash of the modules the stage runs), and an earlier
successful output is archived under ``history/`` before it is replaced. ``run_id_mode='commit'`` additionally keys
the run directory by the repository commit (strict mode: a new commit = a new run directory).
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from . import G06_VERSION

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "configs" / "g06"
FEATURE_STORE_YAML = "feature_store.yaml"
BENCHMARKS_YAML = "benchmarks.yaml"

NB_DEFAULTS = {
    "repo": "mybenkhadda/GEMS---drivendata", "ref": "main", "drive_root": "/content/drive/MyDrive/GEMS_2",
    "local_root": "/content/gems_work", "output_subdir": "outputs/g06_geological_feature_store", "seed": 2026,
    "tile_px": 512, "halo_px": 64, "use_gpu": False, "reuse_cached": True, "strict_provenance": True,
    "sync_to_drive": True, "save_figures": True, "run_id": None, "run_id_mode": "data",
    "run_synthetic_tests": True, "manual_upload_only": True, "sync_tile_cache": False,
}
RUN_DEFAULTS = {
    "PREFLIGHT": True, "RAW_EDA": True, "FEATURE_ENGINEERING": True, "FEATURE_ATLAS": True,
    "BUILD_FEATURE_STORE": True, "BUILD_SPLITS": True, "BUILD_PIXEL_DATASET": True, "BUILD_SEGMENT_DATASET": True,
    "TRAIN_RANDOM_FOREST": False, "TRAIN_LIGHTGBM": False, "TRAIN_XGBOOST": False, "TRAIN_SEGMENT_MODEL": False,
    "TRAIN_PATCH_UNET": False, "TRAIN_META_MODEL": False,
    "EVALUATE": False, "FULL_INFERENCE": False, "EXPORT_SUBMISSION": False,
    "REPORT": True,
}
PIXEL_FAMILIES = ("raw", "physics", "texture", "structure", "context")
SEGMENT_FAMILIES = ("geometry", "geology", "orientation", "context_known", "model_evidence", "uncertainty")
PIXEL_MODELS = ("random_forest", "lightgbm", "xgboost")
SEGMENT_MODELS = ("lightgbm", "random_forest", "catboost", "ranker")
SCENARIOS = ("V1", "V2", "V3")
# keys that never change WHAT is computed (paths, sync, display, resumability) - excluded from identities
VOLATILE_NOTEBOOK_KEYS = ("drive_root", "local_root", "reuse_cached", "sync_to_drive", "save_figures", "run_id",
                          "run_synthetic_tests", "manual_upload_only", "sync_tile_cache", "use_gpu", "ref", "repo",
                          "strict_provenance")


class ConfigError(ValueError):
    """The configuration is inconsistent or would break a validation / safety rule."""


# =============================================================================== loading
def deep_merge(base: dict, over: dict | None) -> dict:
    """Recursive merge (``over`` wins; dicts merge, everything else is replaced). Inputs are not modified."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: Path) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    if not isinstance(d, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return d


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Merged, validated configuration.

    ``path``: the directory holding ``feature_store.yaml`` + ``benchmarks.yaml`` (default ``configs/g06``).
    ``overrides``: ``{"feature_store": {...}, "benchmarks": {...}, "notebook": CONFIG, "run": RUN}`` (each optional;
    the notebook's ``tile_px`` / ``halo_px`` / ``seed`` are mirrored into the feature-store tiling and sampling)."""
    d = Path(path) if path is not None else CONFIG_DIR
    ov = overrides or {}
    unknown = set(ov) - {"feature_store", "benchmarks", "notebook", "run"}
    if unknown:
        raise ConfigError(f"unknown override sections {sorted(unknown)}")
    fs = deep_merge(_read_yaml(d / FEATURE_STORE_YAML), ov.get("feature_store"))
    bm = deep_merge(_read_yaml(d / BENCHMARKS_YAML), ov.get("benchmarks"))
    nb = {**NB_DEFAULTS, **(ov.get("notebook") or {})}
    run = {**RUN_DEFAULTS, **(ov.get("run") or {})}
    fs.setdefault("tiling", {})
    fs["tiling"]["tile_px"] = int(nb["tile_px"])
    fs["tiling"]["halo_px"] = int(nb["halo_px"])
    bm.setdefault("sampling", {})["seed"] = int(nb["seed"])
    cfg = {"version": G06_VERSION, "feature_store": fs, "benchmarks": bm, "notebook": nb, "run": run,
           "config_dir": str(d)}
    validate_config(cfg)
    return cfg


# =============================================================================== validation
def _odd_windows(xs, name, errs) -> None:
    for w in xs or []:
        if int(w) < 3 or int(w) % 2 == 0:
            errs.append(f"{name}: window {w} must be odd and >= 3")


def validate_config(config: dict) -> list[str]:
    """Raise ``ConfigError`` listing every problem; returns the list of non-fatal warnings."""
    errs, warn = [], []
    fs, bm, nb, run = (config.get(k) or {} for k in ("feature_store", "benchmarks", "notebook", "run"))
    bad_run = set(run) - set(RUN_DEFAULTS)
    if bad_run:
        errs.append(f"unknown RUN flags {sorted(bad_run)} (known: {list(RUN_DEFAULTS)})")
    if not nb.get("manual_upload_only", True):
        errs.append("manual_upload_only must stay True: nothing is ever uploaded automatically")
    if nb.get("run_id_mode", "data") not in ("data", "commit"):
        errs.append("run_id_mode: 'data' | 'commit'")
    t = fs.get("tiling") or {}
    tile, halo, block = int(t.get("tile_px", 0)), int(t.get("halo_px", -1)), int(t.get("block_px", 256))
    if tile < 128 or tile % block:
        errs.append(f"tile_px {tile} must be >= 128 and a multiple of the GeoTIFF block size {block}")
    if halo < 0:
        errs.append("halo_px must be >= 0")
    g = fs.get("grid") or {}
    for k in ("height", "width", "epsg", "pixel_m", "n_bands"):
        if k not in g:
            errs.append(f"feature_store.grid.{k} missing")
    bands = fs.get("bands") or {}
    if len(bands) != int(g.get("n_bands", -1)):
        errs.append(f"feature_store.bands lists {len(bands)} identifiers, grid.n_bands = {g.get('n_bands')}")
    for b, spec in bands.items():
        if " - " in str(b):
            errs.append(f"band key {b!r}: give the bare identifier (the part before ' - ')")
        if (spec or {}).get("transform", "identity") not in ("identity", "asinh", "slog1p", "log1p", "log10"):
            errs.append(f"band {b}: unknown transform {spec.get('transform')}")
    groups = fs.get("groups") or {}
    for name, gc in groups.items():
        gc = gc or {}
        for k in ("scales_m", "extra_scales_m"):
            if any(float(s) <= 0 for s in gc.get(k) or []):
                errs.append(f"groups.{name}.{k}: scales must be > 0")
        for k in ("windows", "quantile_windows", "entropy_windows", "edge_windows", "orientation_windows"):
            _odd_windows(gc.get(k), f"groups.{name}.{k}", errs)
        for k in ("field", "fields", "sources", "extra_fields", "texture_fields", "continuity_fields",
                  "discontinuity_fields", "seismicity", "strain"):
            v = gc.get(k)
            vs = [v] if isinstance(v, str) else list(v or [])
            miss = [x for x in vs if x not in bands]
            if miss:
                errs.append(f"groups.{name}.{k}: {miss} are not declared band identifiers")
        for pair in gc.get("cross_modal_pairs") or []:
            if len(pair) != 2 or any(x not in bands for x in pair):
                errs.append(f"groups.{name}.cross_modal_pairs: {pair} must be two declared band identifiers")
    # benchmarks
    px = bm.get("pixel_experiments") or {}
    for eid, e in px.items():
        fam = list((e or {}).get("families") or [])
        if not fam or [f for f in fam if f not in PIXEL_FAMILIES]:
            errs.append(f"pixel experiment {eid}: families {fam} must be a non-empty subset of {PIXEL_FAMILIES}")
        if (e or {}).get("model") not in PIXEL_MODELS:
            errs.append(f"pixel experiment {eid}: model {e.get('model')!r} not in {PIXEL_MODELS}")
    for eid, e in (bm.get("segment_experiments") or {}).items():
        if (e or {}).get("model") not in SEGMENT_MODELS:
            errs.append(f"segment experiment {eid}: model {e.get('model')!r} not in {SEGMENT_MODELS}")
        if (e or {}).get("track") not in ("S1", "S2"):
            errs.append(f"segment experiment {eid}: track must be S1 (geology only) or S2 (geology + honest W2)")
    ev = bm.get("evaluation") or {}
    reps, fracs, prim = ev.get("reps") or {}, [float(x) for x in ev.get("fracs") or []], ev.get("primary") or {}
    if prim.get("rep") not in reps:
        errs.append(f"evaluation.primary.rep {prim.get('rep')!r} not among evaluation.reps {list(reps)}")
    if float(prim.get("frac", -1)) not in fracs:
        errs.append(f"evaluation.primary.frac {prim.get('frac')} not among evaluation.fracs {fracs}")
    if any(not 0 < f < 1 for f in fracs):
        errs.append("evaluation.fracs must lie in (0, 1)")
    val = bm.get("validation") or {}
    if val.get("v1_training_region", "all_visible") not in ("all_visible", "outside_eval_zone"):
        errs.append("validation.v1_training_region: 'all_visible' | 'outside_eval_zone'")
    bad_sc = [s for s in val.get("scenarios") or [] if s not in SCENARIOS]
    if bad_sc or "V1" not in (val.get("scenarios") or []):
        errs.append(f"validation.scenarios must include V1 (primary) and only {SCENARIOS}")
    for eid in val.get("nested_pixel_experiments") or []:
        if eid not in px:
            errs.append(f"validation.nested_pixel_experiments: {eid} is not a pixel experiment")
    sup = bm.get("supervision") or {}
    if float(sup.get("ignore_km", 0)) >= float(sup.get("likely_background_km", 1e9)):
        errs.append("supervision.ignore_km must be < likely_background_km")
    pr = bm.get("promotion") or {}
    if float(pr.get("min_gain_v1", 0)) <= 0:
        errs.append("promotion.min_gain_v1 must be > 0 (no promotion on a tie)")
    if run.get("EXPORT_SUBMISSION") and not run.get("EVALUATE"):
        warn.append("EXPORT_SUBMISSION without EVALUATE: the export gate reads the persisted evaluation decision; "
                    "without one it refuses")
    if run.get("TRAIN_PATCH_UNET") and not nb.get("use_gpu"):
        warn.append("TRAIN_PATCH_UNET with use_gpu=False: the U-Net trains on CPU (slow)")
    if errs:
        raise ConfigError("; ".join(errs))
    return warn


# =============================================================================== identity
def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def canonical_config(config: dict) -> dict:
    """The configuration that defines WHAT is computed: notebook paths / sync / display / resumability keys and the
    RUN flags are dropped (they select stages, they never change a stage's result)."""
    nb = {k: v for k, v in (config.get("notebook") or {}).items() if k not in VOLATILE_NOTEBOOK_KEYS}
    return json.loads(_canon({"version": config.get("version"), "feature_store": config.get("feature_store"),
                              "benchmarks": config.get("benchmarks"), "notebook": nb}))


def data_identity(config: dict) -> dict:
    """Data-engineering identity (run directory key): version + feature-store definition + seed."""
    fs = copy.deepcopy(config.get("feature_store") or {})
    (fs.get("tiling") or {}).pop("keep_tile_cache", None)
    return {"version": config.get("version"), "feature_store": fs,
            "seed": (config.get("notebook") or {}).get("seed")}


def stable_hash(*parts, n: int = 16) -> str:
    return hashlib.sha256(_canon(parts).encode()).hexdigest()[:n]


def compute_run_id(config: dict, artifact_fingerprints: dict | None, commit_sha: str | None) -> str:
    """``g06-<sha10>`` of the data-engineering identity and the source-raster fingerprints. The commit enters only
    with ``run_id_mode='commit'`` (otherwise code identity is tracked per stage, so a code change invalidates the
    stages whose modules changed, not the feature store of an unrelated edit). An explicit ``notebook.run_id`` wins."""
    nb = config.get("notebook") or {}
    if nb.get("run_id"):
        return str(nb["run_id"])
    fps = {k: (v or {}).get("sha256") or (v or {}).get("bytes") for k, v in (artifact_fingerprints or {}).items()}
    parts = [data_identity(config), fps]
    if nb.get("run_id_mode") == "commit":
        if not commit_sha:
            raise ConfigError("run_id_mode='commit' needs the repository commit sha")
        parts.append(str(commit_sha))
    return "g06-" + stable_hash(*parts, n=10)


def stage_config(config: dict, stage: str) -> dict:
    """The configuration sub-tree a stage's fingerprint depends on."""
    fs, bm = config.get("feature_store") or {}, config.get("benchmarks") or {}
    seed = (config.get("notebook") or {}).get("seed")
    m = {
        "preflight": {"grid": fs.get("grid"), "bands": fs.get("bands"), "sources": fs.get("sources"),
                      "reuse": fs.get("reuse")},
        "raw_eda": {"bands": fs.get("bands"), "nodata": fs.get("nodata")},
        "features": data_identity(config),
        "feature_atlas": {"atlas": fs.get("atlas")},
        "splits": {"validation": bm.get("validation"), "inner_cv": bm.get("inner_cv")},
        "pixel_dataset": {"supervision": bm.get("supervision"), "sampling": bm.get("sampling"),
                          "validation": bm.get("validation"), "patch": (bm.get("patch") or {}).get("index"),
                          "seed": seed},
        "segment_dataset": {"segment_candidates": bm.get("segment_candidates"),
                            "supervision": (bm.get("supervision") or {}).get("segment"),
                            "connection": (bm.get("supervision") or {}).get("connection"),
                            "validation": bm.get("validation"), "known": fs.get("known_context")},
        "train_pixel": {"pixel_experiments": bm.get("pixel_experiments"), "models": bm.get("models"),
                        "inner_cv": bm.get("inner_cv"), "validation": bm.get("validation"), "seed": seed},
        "train_segment": {"segment_experiments": bm.get("segment_experiments"), "models": bm.get("models"),
                          "segment_map": bm.get("segment_map"), "validation": bm.get("validation"), "seed": seed},
        "train_patch": {"patch": bm.get("patch"), "seed": seed},
        "train_meta": {"meta": bm.get("meta"), "models": bm.get("models"), "seed": seed},
        "evaluate": {"evaluation": bm.get("evaluation"), "promotion": bm.get("promotion")},
        "full_inference": {"inference": bm.get("inference"), "models": bm.get("models")},
        "export": {"promotion": bm.get("promotion"), "export": bm.get("export")},
    }
    return m.get(stage, {})
