"""Validated master-pipeline configuration (``configs/pipeline/master.yaml`` + notebook overrides).

One schema instead of scattered notebook globals. ``PipelineConfig.load`` deep-merges, in order:
the YAML defaults, an optional user YAML, and a flat notebook ``CONFIG`` dict (keys such as ``run_eda`` or
``drive_root`` are routed to their section). Unknown keys, wrong types and invalid modes raise
``ConfigError`` with the closest valid key, so a typo never silently disables a safety switch.

Safety defaults: mode RESUME, model training / DEM download / full inference disabled, folds and
discovery splits fixed to the Phase-2 protocol (changing them creates a new versioned split).
"""
from __future__ import annotations

import copy
import difflib
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

MODES = ("SETUP_ONLY", "PREPARE_DATA", "BASELINES", "ADVANCED_RESEARCH", "FULL_PIPELINE", "RESUME", "SUBMISSION_ONLY")

FEATURE_FAMILIES = ("raw", "phase2", "phase2_eda", "phase4_rs_used", "phase4_rs_all", "phase5_profiles", "dem")

# Flat notebook keys -> dotted schema path
FLAT = {
    "repo": "repo", "ref": "ref", "seed": "seed", "run_mode": "run_mode",
    "drive_root": "paths.drive_root", "local_root": "paths.local_root", "code_root": "paths.code_root",
    "download_missing": "data.download_missing", "rebuild_missing": "rebuild_missing",
    "run_eda": "stages.run_eda", "run_baselines": "stages.run_baselines", "train_baselines": "stages.train_baselines",
    "run_advanced_features": "stages.run_advanced_features", "run_advanced_models": "stages.run_advanced_models",
    "run_dem_pilot": "stages.run_dem_pilot", "run_scoreboard": "stages.run_scoreboard",
    "run_ensemble": "stages.run_ensemble", "run_full_inference": "stages.run_full_inference",
    "run_submission": "stages.run_submission", "run_reports": "stages.run_reports",
    "use_gpu": "gpu.use_gpu", "max_stage_minutes": "gpu.max_stage_minutes",
    "save_figures": "reports.save_figures", "save_intermediate": "reports.save_intermediate",
    "urls": "data.urls", "advanced_experiments": "experiments.advanced",
    "feature_families": "features.families", "feature_stage": "features.stage",
    # experiment fidelity (SCREEN / VALIDATE / FULL): orthogonal to run_mode, never overloads it
    "experiment_profile": "experiments.profile", "fidelity_overrides": "experiments.fidelity_overrides",
    "screening_rules": "experiments.screening", "checkpoint": "experiments.checkpoint",
    "auto_provision_maps": "experiments.auto_provision_maps",
}

EXPERIMENT_PROFILES = ("SCREEN", "VALIDATE", "FULL")
CHECKPOINT_KEYS = ("policy", "save_every_epochs", "sync_every_minutes")

DEFAULTS: dict = {
    "version": 1,
    "repo": "mybenkhadda/GEMS---drivendata",
    "ref": "main",
    "seed": 2026,
    "run_mode": "RESUME",
    "rebuild_missing": True,
    "failure_policy": "batch",
    "paths": {"drive_root": "/content/drive/MyDrive/GEMS_2", "local_root": "/content/gems_work",
              "code_root": "/content/GEMS"},
    "data": {"download_missing": True, "use_colab_secrets": True, "urls": {}, "verify": "sha256"},
    "stages": {
        "run_eda": True, "run_baselines": True, "train_baselines": [], "run_advanced_features": True,
        "run_advanced_models": False, "run_dem_pilot": False, "run_scoreboard": True, "run_ensemble": True,
        "run_full_inference": False, "run_submission": False, "run_reports": True,
    },
    "validation": {                                   # Phase-2 protocol (gems_bench.config.BenchConfig defaults)
        "n_folds": 3, "block_km": 20.0, "buffer_km": 13.0, "clusters_per_fold": 3, "inner_buffer_km": 5.0,
        "n_discovery": 3, "link_km": 1.0, "frac_km": 0.12, "min_system_km": 5.0, "max_system_frac": 0.05,
        "ignore_km": 1.0, "eval_km": 10.0,
    },
    "eda": {"max_dim": 2048, "seed": 7},
    # stage: all (copy every required cache to local disk before the stage) | lazy (copy a cache file from Drive
    # the first time a run reads it; validation reads Drive headers)
    "features": {"families": ["raw", "phase2", "phase2_eda", "phase4_rs_used"], "extra": [], "stage": "all"},
    "dem": {"enabled": False, "mode": "pilot", "inventory": True, "download": False, "pilot_tiles": 24,
            "max_tiles": 30, "max_gib": 15.0, "download_only_intersecting": True, "compute_morphology": True,
            "aggregate_to_100m": True, "inspect_headers": 10, "experiment": "f1_dem_pilot"},
    "experiments": {"advanced": ["b0_reference", "b1_bg_weight", "b2_nnpu_p010", "c1_geometry_heads",
                                 "e1_gsn_parity"],
                    "scoreboard_models": ["G1", "G3", "RS05_G3_noSS_RS", "CNN_A_dti", "DL00_CNN_B_dti",
                                          "DL02_CNN_TMGB_dti", "P4_B_gated"],
                    "ensemble": "g1_ensemble", "submission": "z_submission", "stop_on_failure": False,
                    # fidelity of cnn5 research runs; FULL = committed YAMLs unchanged (historical behaviour)
                    "profile": "FULL", "fidelity_overrides": {}, "screening": {},
                    "auto_provision_maps": True,
                    "checkpoint": {"policy": "best_and_latest", "save_every_epochs": 5, "sync_every_minutes": 15}},
    "gpu": {"use_gpu": True, "max_stage_minutes": 0, "overrides": {}},
    "sync": {"push_after_stage": True},
    "reports": {"save_figures": True, "save_intermediate": True},
}

TYPES = {
    "version": int, "repo": str, "ref": str, "seed": int, "run_mode": str, "rebuild_missing": bool,
    "failure_policy": str,
}

# interactive: a failed critical stage prints its full traceback and raises StageFailed (stops "Run all").
# batch: independent stages continue; blocked stages are grouped under their root failure in one report.
FAILURE_POLICIES = ("interactive", "batch")


class ConfigError(ValueError):
    pass


def _deep_merge(base: dict, over: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        here = f"{path}.{k}" if path else k
        if k not in base and not _free_form(path):
            close = difflib.get_close_matches(k, list(base), n=1)
            raise ConfigError(f"unknown config key '{here}'" + (f" (did you mean '{close[0]}'?)" if close else ""))
        if isinstance(v, dict) and isinstance(out.get(k), dict) and not _free_form(here):
            out[k] = _deep_merge(out[k], v, here)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _free_form(path: str) -> bool:
    """Sections whose keys are user-defined (URLs per file, GPU overrides)."""
    return path in ("data.urls", "gpu.overrides", "experiments.fidelity_overrides", "experiments.screening",
                    "experiments.checkpoint")


def _set(d: dict, dotted: str, value) -> None:
    cur = d
    keys = dotted.split(".")
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def flat_to_nested(flat: dict) -> dict:
    out: dict = {}
    for k, v in (flat or {}).items():
        if k in FLAT:
            _set(out, FLAT[k], v)
        elif k in DEFAULTS:
            out[k] = v
        else:
            close = difflib.get_close_matches(k, list(FLAT) + list(DEFAULTS), n=1)
            raise ConfigError(f"unknown notebook CONFIG key '{k}'" + (f" (did you mean '{close[0]}'?)" if close else ""))
    return out


@dataclass
class PipelineConfig:
    raw: dict = field(default_factory=lambda: copy.deepcopy(DEFAULTS))
    source: str = "defaults"

    # -- construction ---------------------------------------------------------
    @classmethod
    def load(cls, yaml_path: str | Path | None = None, *, flat: dict | None = None, dem: dict | None = None,
             overrides: dict | None = None) -> "PipelineConfig":
        raw = copy.deepcopy(DEFAULTS)
        src = ["defaults"]
        if yaml_path:
            import yaml

            y = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8")) or {}
            raw = _deep_merge(raw, y)
            src.append(str(yaml_path))
        if flat:
            raw = _deep_merge(raw, flat_to_nested(flat))
            src.append("notebook CONFIG")
        if dem:
            raw = _deep_merge(raw, {"dem": dem})
            src.append("notebook DEM_CONFIG")
        if overrides:
            raw = _deep_merge(raw, overrides)
            src.append("overrides")
        cfg = cls(raw, " + ".join(src))
        cfg.validate()
        return cfg

    @classmethod
    def default_yaml(cls) -> Path | None:
        from ..paths import config_path

        try:
            p = config_path("pipeline", "master.yaml")
        except FileNotFoundError:
            return None
        return p if p.exists() else None

    # -- validation -------------------------------------------------------------
    def validate(self) -> None:
        r = self.raw
        for k, t in TYPES.items():
            if not isinstance(r[k], t) or (t is int and isinstance(r[k], bool)):
                raise ConfigError(f"'{k}' must be {t.__name__}, got {type(r[k]).__name__}")
        r["run_mode"] = r["run_mode"].upper()
        if r["run_mode"] not in MODES:
            raise ConfigError(f"run_mode '{r['run_mode']}' not in {MODES}")
        r["failure_policy"] = r["failure_policy"].lower()
        if r["failure_policy"] not in FAILURE_POLICIES:
            raise ConfigError(f"failure_policy '{r['failure_policy']}' not in {FAILURE_POLICIES}")
        for k, v in r["stages"].items():
            if k == "train_baselines":
                if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                    raise ConfigError("stages.train_baselines must be a list of baseline names")
            elif not isinstance(v, bool):
                raise ConfigError(f"stages.{k} must be true/false")
        fam = r["features"]["families"]
        bad = [f for f in fam if f not in FEATURE_FAMILIES]
        if bad:
            raise ConfigError(f"unknown feature families {bad}; valid: {FEATURE_FAMILIES}")
        if r["features"].get("stage", "all") not in ("all", "lazy"):
            raise ConfigError("features.stage must be 'all' or 'lazy'")
        v = r["validation"]
        if int(v["n_folds"]) < 2 or int(v["n_discovery"]) < 1:
            raise ConfigError("validation.n_folds must be >= 2 and n_discovery >= 1")
        if float(v["buffer_km"]) < 12.8:
            raise ConfigError("validation.buffer_km < 12.8 km lets 256 px CNN patches overlap validation blocks")
        d = r["dem"]
        if d["mode"] not in ("pilot", "inventory_only"):
            raise ConfigError("dem.mode must be 'pilot' or 'inventory_only' (full-collection download is not supported)")
        if int(d["pilot_tiles"]) > int(d["max_tiles"]):
            raise ConfigError("dem.pilot_tiles exceeds dem.max_tiles")
        if not isinstance(r["data"]["urls"], dict):
            raise ConfigError("data.urls must be a mapping {features|faults|submission|dem_json: url}")
        from ..data.acquisition import BY_KEY

        unknown = [k for k in r["data"]["urls"] if k not in BY_KEY]
        if unknown:
            raise ConfigError(f"data.urls keys {unknown} not in {list(BY_KEY)}")
        from ..models.zoo import BY_NAME

        from .baselines import RECIPES

        tb = [b for b in r["stages"]["train_baselines"] if b not in RECIPES]
        if tb:
            raise ConfigError(f"stages.train_baselines {tb} not in {list(RECIPES)}")
        sm = [m for m in r["experiments"]["scoreboard_models"] if m not in BY_NAME and m not in RECIPES]
        if sm:
            raise ConfigError(f"experiments.scoreboard_models {sm} unknown")
        self._validate_fidelity(r["experiments"])

    @staticmethod
    def _validate_fidelity(e: dict) -> None:
        from .fidelity import FidelityError, make_fidelity
        from .screening import DEFAULT_RULES

        if not isinstance(e.get("profile"), str):
            raise ConfigError("experiments.profile must be one of SCREEN / VALIDATE / FULL")
        e["profile"] = e["profile"].upper()
        if e["profile"] not in EXPERIMENT_PROFILES:
            raise ConfigError(f"experiments.profile '{e['profile']}' not in {EXPERIMENT_PROFILES} "
                              "(run_mode chooses the stages; the profile chooses the evidence level)")
        ov = e.get("fidelity_overrides") or {}
        if not isinstance(ov, dict):
            raise ConfigError("experiments.fidelity_overrides must be {PROFILE: {knob: value}}")
        for p, knobs in ov.items():
            if str(p).upper() not in EXPERIMENT_PROFILES or not isinstance(knobs, dict):
                raise ConfigError(f"experiments.fidelity_overrides key '{p}' must be a profile with a mapping")
            try:
                make_fidelity(str(p).upper(), knobs)
            except FidelityError as err:
                raise ConfigError(f"experiments.fidelity_overrides.{p}: {err}") from None
        e["fidelity_overrides"] = {str(p).upper(): v for p, v in ov.items()}
        bad = [k for k in (e.get("screening") or {}) if k not in DEFAULT_RULES]
        if bad:
            raise ConfigError(f"experiments.screening keys {bad} unknown; valid: {sorted(DEFAULT_RULES)}")
        ck = e.get("checkpoint") or {}
        bad = [k for k in ck if k not in CHECKPOINT_KEYS]
        if bad:
            raise ConfigError(f"experiments.checkpoint keys {bad} unknown; valid: {CHECKPOINT_KEYS}")
        if ck.get("policy", "best_and_latest") not in ("best_and_latest", "rotating"):
            raise ConfigError("experiments.checkpoint.policy must be 'best_and_latest' or 'rotating'")
        if not isinstance(e.get("auto_provision_maps"), bool):
            raise ConfigError("experiments.auto_provision_maps must be true/false")

    # -- access -------------------------------------------------------------------
    def get(self, dotted: str, default=None):
        cur = self.raw
        for k in dotted.split("."):
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    @property
    def mode(self) -> str:
        return self.raw["run_mode"]

    @property
    def local_root(self) -> Path:
        return Path(self.raw["paths"]["local_root"])

    @property
    def drive_root(self) -> Path:
        return Path(self.raw["paths"]["drive_root"])

    @property
    def code_root(self) -> Path:
        return Path(self.raw["paths"]["code_root"])

    def section_hash(self, *dotted: str) -> str:
        """sha256 of selected sections: the fingerprint a stage compares on resume. URLs are never hashed."""
        sub = {d: self.get(d) for d in dotted}
        return hashlib.sha256(json.dumps(sub, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def public(self) -> dict:
        """Copy safe to persist: URLs are replaced by their host (signed links never reach disk / Git)."""
        from ..data.acquisition import url_record

        r = copy.deepcopy(self.raw)
        r["data"]["urls"] = {k: url_record(v).get("host", "") for k, v in r["data"]["urls"].items() if v}
        return r

    def table(self):
        import pandas as pd

        rows = []

        def walk(d, p=""):
            for k, v in d.items():
                q = f"{p}.{k}" if p else k
                if isinstance(v, dict) and v and not _free_form(q):
                    walk(v, q)
                else:
                    rows.append({"key": q, "value": json.dumps(v, default=str)})

        walk(self.public())
        return pd.DataFrame(rows)
