"""GEMS3 configuration: configs/gems3/research.yaml + notebook overrides, unknown keys rejected."""
from __future__ import annotations

import copy
import difflib
from pathlib import Path

CONFIG_PARTS = ("gems3", "research.yaml")
FREE_FORM = ("metric_opt.representations", "ranker.lgbm", "oracle.continue_if")
STAGES = ("verify_official_metric", "run_metric_optimization", "prepare_submission_candidate",
          "run_candidate_generation", "run_candidate_oracle", "run_candidate_ranker", "run_dem_gate",
          "run_map_completion")


class GEMS3ConfigError(ValueError):
    pass


def merge(base: dict, over: dict, path: str = "", free_form=FREE_FORM) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        here = f"{path}.{k}" if path else k
        if k not in base and path not in free_form:
            close = difflib.get_close_matches(k, list(base), n=1)
            raise GEMS3ConfigError(f"unknown GEMS3 config key '{here}'" + (f" (did you mean '{close[0]}'?)" if close else ""))
        if isinstance(v, dict) and isinstance(out.get(k), dict) and here not in free_form:
            out[k] = merge(out[k], v, here, free_form)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_gems3_config(overrides: dict | None = None, path: Path | None = None) -> dict:
    import yaml

    from ..paths import config_path

    p = Path(path) if path else config_path(*CONFIG_PARTS)
    cfg = merge(yaml.safe_load(p.read_text(encoding="utf-8")), overrides or {})
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    from .official import TASKS
    from .representation import POST_OPS, SELECTORS

    if str(cfg.get("mode")).upper() != "FAST_DISCOVERY_RESEARCH":
        raise GEMS3ConfigError(f"mode must be FAST_DISCOVERY_RESEARCH (got {cfg.get('mode')!r})")
    if cfg.get("retrain_existing_cnns"):
        raise GEMS3ConfigError("retrain_existing_cnns=True is refused: GEMS3 reuses saved FULL predictions; "
                               "retrain through sections 14-15")
    if len(cfg.get("models") or []) != 2 or len(set(cfg["models"])) != 2:
        raise GEMS3ConfigError(f"models needs two distinct experiment stems, got {cfg.get('models')}")
    bad = [t for t in cfg["tasks"] if t not in TASKS]
    if bad:
        raise GEMS3ConfigError(f"unknown tasks {bad}; known: {list(TASKS)}")
    if cfg["primary_task"] not in cfg["tasks"]:
        raise GEMS3ConfigError("primary_task must be one of tasks")
    mo = cfg["metric_opt"]
    w = mo["fusion_weights"]
    if not w or min(w) < 0 or max(w) > 1:
        raise GEMS3ConfigError("metric_opt.fusion_weights must lie in [0, 1]")
    if not all(0 < f < 1 for f in mo["area_fracs"]):
        raise GEMS3ConfigError("metric_opt.area_fracs must lie in (0, 1)")
    for name, rep in (mo.get("representations") or {}).items():
        if rep.get("select") not in SELECTORS:
            raise GEMS3ConfigError(f"representation {name}: select must be one of {SELECTORS}")
        for op, _ in rep.get("post") or []:
            if op not in POST_OPS:
                raise GEMS3ConfigError(f"representation {name}: unknown post operation {op!r}; known {sorted(POST_OPS)}")
    sub = cfg["submission"]
    if sub["source"] not in ("fold_average", "oof_stitched"):
        raise GEMS3ConfigError("submission.source must be fold_average | oof_stitched")
    if sub["outside_valid"] not in ("template", "nan"):
        raise GEMS3ConfigError("submission.outside_valid must be template | nan")
    afs = sub["area_frac_source"]
    if not (afs in ("geo", "primary") or (isinstance(afs, (int, float)) and 0 < afs < 1)):
        raise GEMS3ConfigError("submission.area_frac_source must be geo | primary | a fraction in (0, 1)")
    if "geo" not in cfg["tasks"] and afs == "geo":
        raise GEMS3ConfigError("submission.area_frac_source=geo needs 'geo' in tasks")
