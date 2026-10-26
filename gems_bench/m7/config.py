"""M7 configuration: configs/m7/discovery_first.yaml + notebook overrides, unknown keys rejected."""
from __future__ import annotations

from pathlib import Path

from ..gems3.config import merge

CONFIG_PARTS = ("m7", "discovery_first.yaml")
FREE_FORM = ("baseline.representations", "rasterization.representations", "ranking.lgbm", "ranking.catboost",
             "oracle.go_if", "dem.go_if", "dem.quotas", "dem.morphology", "map_completion.learned_if",
             "map_completion.learned", "decision.promote_if", "decision.reject_if", "recorded_reference",
             "recorded_reference.geo_dti", "recorded_reference.disc_official_dti", "failure.context_bands",
             "failure.gradient_features", "failure.morphology_features", "features.side_fields")
SECTIONS = ("s01_baseline", "s02_failure_analysis", "s03_candidates", "s04_oracle", "s05_dem_inventory",
            "s06_dem_morphology", "s07_dem_increment", "s08_network_analysis", "s09_corruption_benchmark",
            "s10_candidate_features", "s11_candidate_ranking", "s12_rasterization", "s13_comparison",
            "s14_decision_report", "s15_submission_export", "s16_probe_packages")
CANDIDATE_MODES = ("edge", "ridge", "valley", "ridge_valley")
RASTER_KINDS = ("centerline", "corridor", "confidence_width", "antialiased")


class M7ConfigError(ValueError):
    pass


def load_m7_config(overrides: dict | None = None, path: Path | None = None) -> dict:
    import yaml

    from ..paths import config_path

    p = Path(path) if path else config_path(*CONFIG_PARTS)
    base = yaml.safe_load(p.read_text(encoding="utf-8"))
    try:
        cfg = merge(base, overrides or {}, free_form=FREE_FORM)
    except ValueError as e:                                   # same message, M7 error type
        raise M7ConfigError(str(e).replace("GEMS3", "M7")) from None
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    from ..gems3.official import TASKS
    from ..gems3.representation import POST_OPS, SELECTORS

    if str(cfg.get("mode")).upper() != "DISCOVERY_FIRST":
        raise M7ConfigError(f"mode must be DISCOVERY_FIRST (got {cfg.get('mode')!r})")
    if cfg.get("retrain_existing_cnns"):
        raise M7ConfigError("retrain_existing_cnns=True is refused: M7 reuses the saved FULL B0 / B1 predictions; "
                            "retrain through the master pipeline (sections 14-15)")
    if len(cfg.get("models") or []) != 2 or len(set(cfg["models"])) != 2:
        raise M7ConfigError(f"models needs two distinct experiment stems, got {cfg.get('models')}")
    ev = cfg["evaluation"]
    bad = [t for t in ev["tasks"] if t not in TASKS]
    if bad:
        raise M7ConfigError(f"unknown evaluation tasks {bad}; known: {list(TASKS)}")
    if ev["primary_task"] not in ev["tasks"] or not ev["primary_task"].startswith("disc"):
        raise M7ConfigError("evaluation.primary_task must be a disc task listed in evaluation.tasks")
    if "geo" not in ev["tasks"]:
        raise M7ConfigError("evaluation.tasks must contain 'geo' (geographic DTI is reported for every method)")
    if ev["budget_domain"] not in ("eligible", "evaluation_zone"):
        raise M7ConfigError("evaluation.budget_domain must be eligible | evaluation_zone")
    if not all(0 < f < 1 for f in ev["area_fracs"]) or not all(0 < f < 1 for f in cfg["rasterization"]["area_fracs"]):
        raise M7ConfigError("area fractions must lie in (0, 1)")
    w = ev["fusion_weights"]
    if not w or min(w) < 0 or max(w) > 1 or 1.0 not in w or 0.0 not in w:
        raise M7ConfigError("evaluation.fusion_weights must lie in [0, 1] and contain 0 and 1 (the single members)")
    if 0 not in ev["ring_px"]:
        raise M7ConfigError("evaluation.ring_px must contain 0 (no ring is always a candidate)")
    for k in ("folds", "splits"):
        if ev[k] != "all" and not (isinstance(ev[k], list) and all(isinstance(x, int) for x in ev[k])):
            raise M7ConfigError(f"evaluation.{k} must be 'all' or a list of ints")
    sec = cfg["sections"]
    missing = [s for s in SECTIONS if s not in sec]
    if missing:
        raise M7ConfigError(f"sections missing switches: {missing}")
    for name, rep in (cfg["baseline"].get("representations") or {}).items():
        if rep.get("select") not in SELECTORS:
            raise M7ConfigError(f"baseline representation {name}: select must be one of {SELECTORS}")
        for op, _ in rep.get("post") or []:
            if op not in POST_OPS:
                raise M7ConfigError(f"baseline representation {name}: unknown post operation {op!r}")
    if "binary" not in cfg["baseline"]["representations"]:
        raise M7ConfigError("baseline.representations must define 'binary'")
    for t in cfg["baseline"]["thinning_reps"]:
        if t not in cfg["baseline"]["representations"]:
            raise M7ConfigError(f"baseline.thinning_reps: {t} is not a defined representation")
    names = [s["name"] for s in cfg["candidates"]["sources"]] + [s["name"] for s in cfg["candidates"]["model_sources"]]
    if len(names) != len(set(names)):
        raise M7ConfigError(f"candidate source names must be unique: {names}")
    for s in cfg["candidates"]["sources"]:
        if s.get("mode") not in CANDIDATE_MODES:
            raise M7ConfigError(f"candidate source {s['name']}: mode must be one of {CANDIDATE_MODES}")
        if bool(s.get("band")) == bool(s.get("feature")):
            raise M7ConfigError(f"candidate source {s['name']}: give exactly one of band / feature")
        if not str(s["name"]).replace("_", "").isalnum():
            raise M7ConfigError(f"candidate source name {s['name']!r}: alphanumeric + '_' only (feature names)")
    for s in cfg["candidates"]["model_sources"]:
        if s["model"] not in cfg["models"]:
            raise M7ConfigError(f"model source {s['name']}: {s['model']} is not one of models {cfg['models']}")
    for name, r in cfg["rasterization"]["representations"].items():
        if r.get("kind") not in RASTER_KINDS:
            raise M7ConfigError(f"rasterization representation {name}: kind must be one of {RASTER_KINDS}")
    for fw in cfg["rasterization"]["fusion_weights"]:
        if len(fw) != 3 or abs(sum(fw) - 1.0) > 1e-6 or min(fw) < 0:
            raise M7ConfigError(f"rasterization.fusion_weights rows must be 3 non-negative weights summing to 1: {fw}")
    d = cfg["dem"]
    if d["coverage_mode"] not in ("pilot", "expanded"):
        raise M7ConfigError("dem.coverage_mode must be pilot | expanded")
    if not (1 <= d["min_pilot_tiles"] <= d["n_pilot_tiles"] <= d["max_pilot_tiles"]):
        raise M7ConfigError("dem: need 1 <= min_pilot_tiles <= n_pilot_tiles <= max_pilot_tiles")
    sub = cfg["submission"]
    if sub["source"] not in ("fold_average", "oof_stitched"):
        raise M7ConfigError("submission.source must be fold_average | oof_stitched")
    afs = sub["area_frac_source"]
    if not (afs in ("geo", "primary") or (isinstance(afs, (int, float)) and 0 < afs < 1)):
        raise M7ConfigError("submission.area_frac_source must be geo | primary | a fraction in (0, 1)")
    if sub["outside_valid"] not in ("template", "nan"):
        raise M7ConfigError("submission.outside_valid must be template | nan")
    if int(cfg["probes"]["max_probes"]) > 3:
        raise M7ConfigError("probes.max_probes > 3: the competition allows 3 scored submissions per 7 days")
    for a in cfg["ranking"]["ablations"]:
        if a not in ("label_free", "neural_support", "geom_geophys", "geom_geophys_dem", "geom_geophys_dem_neural",
                     "plus_completion"):
            raise M7ConfigError(f"unknown ranking ablation {a!r}")
