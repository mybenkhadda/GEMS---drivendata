"""M8 configuration: configs/m8/physics_probabilistic_discovery.yaml + notebook overrides.

Unknown keys are rejected (closest valid key suggested), except inside ``shared`` (handed to the M7 loader, which
validates it against configs/m7/discovery_first.yaml) and inside the free-form blocks listed in ``OPEN``. Profiles
may only carry computational-scope keys (``SCOPE_KEYS``): a profile can never change units, masks, the metric, the
budget domain, operator definitions or gate thresholds.
"""
from __future__ import annotations

import copy
import difflib
from pathlib import Path

CONFIG_PARTS = ("m8", "physics_probabilistic_discovery.yaml")
OPEN = ("shared", "recorded_m7", "bands.verified", "bands.units", "bands.families", "profiles", "profile_overrides",
        "physics.operators", "worms.scale_space", "profiles_fn.fields", "monte_carlo.distributions",
        "monte_carlo.measured_noise", "rasterization.representations", "ensemble.systems")
SECTIONS = ("s01_baseline", "s02_signal_recovery", "s03_potential_fields", "s03b_structural_worms", "s04_multimodal",
            "s05_fault_normal_profiles", "s06_candidates", "s07_monte_carlo_signal", "s08_monte_carlo_completion",
            "s09_dem_pilot", "s10_candidate_features", "s11_ranking", "s12_graph", "s13_rasterization",
            "s14_ensemble", "s15_submission")
PROFILES = ("PILOT", "VALIDATE", "FULL")
SCOPE_KEYS = ("mc_realizations", "mc_window", "completion_samples", "corruption_realisations",
              "case_studies_per_class", "gallery_per_group", "persist_transects_max", "full_region_candidates",
              "signal_recovery_tasks", "n_boot", "run_pu_sensitivity")
OPERATORS = ("thdr", "tilt_edge", "tdx", "analytic_signal", "theta", "lcn", "tilt_edge_band", "analytic_signal_band",
             "tdx_band")
RASTER_KINDS = ("centerline", "occupancy", "uncertainty", "confidence_width", "crest")
ABLATIONS = ("label_free", "cnn_only", "physics_only", "physics_mc", "physics_dem", "physics_cnn", "all")
DISTS = ("uniform", "normal", "choice")


class M8ConfigError(ValueError):
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
            raise M8ConfigError(f"unknown M8 config key '{here}'" + (f" (did you mean '{close[0]}'?)" if close else ""))
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v, here)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_m8_config(overrides: dict | None = None, path: Path | None = None) -> dict:
    import yaml

    from ..paths import config_path

    p = Path(path) if path else config_path(*CONFIG_PARTS)
    base = yaml.safe_load(p.read_text(encoding="utf-8"))
    cfg = merge(base, overrides or {})
    validate(cfg)
    cfg["_profile"] = resolve_profile(cfg)
    return cfg


def resolve_profile(cfg: dict) -> dict:
    """Scope settings of the active profile (+ ``profile_overrides``), tagged with the profile name."""
    name = str(cfg["profile"]).upper()
    prof = dict(cfg["profiles"][name])
    prof.update(cfg.get("profile_overrides") or {})
    prof["name"] = name
    prof["provisional"] = name in (cfg.get("gates", {}).get("provisional_profiles") or [])
    return prof


def m7_overrides(cfg: dict) -> dict:
    """The ``shared`` block as M7 overrides (M7 keeps its own output_dir: M8 only READS M7 artifacts there)."""
    sh = copy.deepcopy(cfg.get("shared") or {})
    sh["output_dir"] = cfg["m7_output_dir"]
    sh["legacy_gems3_dir"] = cfg["legacy_gems3_dir"]
    sh.setdefault("retrain_existing_cnns", False)
    return sh


def _check_dist(name: str, d: dict) -> None:
    if not isinstance(d, dict) or d.get("dist") not in DISTS:
        raise M8ConfigError(f"monte_carlo.distributions.{name}: dist must be one of {DISTS}")
    if d["dist"] == "uniform" and not (float(d["lo"]) <= float(d["hi"])):
        raise M8ConfigError(f"monte_carlo.distributions.{name}: lo > hi")
    if d["dist"] == "normal" and float(d.get("sigma", -1)) < 0:
        raise M8ConfigError(f"monte_carlo.distributions.{name}: sigma must be >= 0")
    if d["dist"] == "choice" and name != "operator" and not d.get("values"):
        raise M8ConfigError(f"monte_carlo.distributions.{name}: choice needs values")


def validate(cfg: dict) -> None:
    if str(cfg.get("mode")).upper() != "PHYSICS_PROBABILISTIC_DISCOVERY":
        raise M8ConfigError(f"mode must be PHYSICS_PROBABILISTIC_DISCOVERY (got {cfg.get('mode')!r})")
    if str(cfg.get("profile")).upper() not in PROFILES:
        raise M8ConfigError(f"profile must be one of {PROFILES}")
    if cfg["output_dir"] == cfg["m7_output_dir"]:
        raise M8ConfigError("output_dir must differ from m7_output_dir: M7 artifacts are read-only")
    sec = cfg["sections"]
    missing = [s for s in SECTIONS if s not in sec]
    if missing:
        raise M8ConfigError(f"sections missing switches: {missing}")
    keys = None
    for name in PROFILES:
        p = cfg["profiles"].get(name)
        if p is None:
            raise M8ConfigError(f"profiles.{name} missing")
        bad = [k for k in p if k not in SCOPE_KEYS]
        if bad:
            raise M8ConfigError(f"profiles.{name}: {bad} are not computational-scope keys ({SCOPE_KEYS}); profiles "
                                "must not change evaluation definitions")
        if keys is None:
            keys = set(p)
        elif set(p) != keys:
            raise M8ConfigError("every profile must define the same scope keys")
    bad = [k for k in (cfg.get("profile_overrides") or {}) if k not in SCOPE_KEYS]
    if bad:
        raise M8ConfigError(f"profile_overrides: {bad} are not scope keys")
    for name in PROFILES:
        w = str(cfg["profiles"][name]["mc_window"])
        if not (w in ("full", "eval_units") or w.startswith("unit:") or w.startswith("bbox:")):
            raise M8ConfigError(f"profiles.{name}.mc_window must be full | eval_units | unit:<task>:<zone> | "
                                "bbox:r0,r1,c0,c1")
    fams = cfg["bands"]["families"]
    for f, spec in fams.items():
        if "field" not in spec or float(spec.get("effective_resolution_m", 0)) <= 0:
            raise M8ConfigError(f"bands.families.{f}: needs field and a positive effective_resolution_m")
    for f, ops in cfg["physics"]["operators"].items():
        if f not in fams:
            raise M8ConfigError(f"physics.operators.{f}: unknown family")
        bad = [o for o in ops if o not in OPERATORS]
        if bad:
            raise M8ConfigError(f"physics.operators.{f}: unknown operators {bad}; known {OPERATORS}")
    for f in cfg["worms"]["families"]:
        if f not in fams:
            raise M8ConfigError(f"worms.families: unknown family {f}")
        if cfg["worms"]["scale_space"].get(f, "gaussian") not in ("upward", "gaussian"):
            raise M8ConfigError(f"worms.scale_space.{f} must be upward | gaussian")
        if cfg["worms"]["scale_space"].get(f) == "upward" and not fams[f].get("potential_field"):
            raise M8ConfigError(f"worms.scale_space.{f}: upward continuation is only physical for potential fields")
    names = [s["name"] for s in cfg["generation"]["families"]] + [s["name"] for s in cfg["generation"]["model_families"]]
    if len(names) != len(set(names)):
        raise M8ConfigError(f"generation family names must be unique: {names}")
    for s in cfg["generation"]["families"]:
        if bool(s.get("band")) == bool(s.get("layer")):
            raise M8ConfigError(f"generation family {s['name']}: give exactly one of band / layer")
        if not str(s["name"]).replace("_", "").isalnum():
            raise M8ConfigError(f"generation family {s['name']!r}: alphanumeric + '_' only (feature names)")
        if s.get("band") and s.get("mode") not in ("edge", "ridge", "valley", "ridge_valley"):
            raise M8ConfigError(f"generation family {s['name']}: band families need a mode")
    models = list(cfg["shared"]["models"])
    for s in cfg["generation"]["model_families"]:
        if s.get("kind") != "disagreement" and s.get("model") not in models:
            raise M8ConfigError(f"model family {s['name']}: {s.get('model')} is not one of shared.models {models}")
    for n, d in cfg["monte_carlo"]["distributions"].items():
        _check_dist(n, d)
    for fam in cfg["monte_carlo"]["families"]:
        if fam["family"] not in fams:
            raise M8ConfigError(f"monte_carlo.families: unknown family {fam['family']}")
        bad = [o for o in fam["operators"] if o not in OPERATORS or o.endswith("_band")]
        if bad:
            raise M8ConfigError(f"monte_carlo.families.{fam['family']}: operators {bad} not allowed (band-derived "
                                "operators would perturb a derived band as if it were a separate sensor)")
    for name, r in cfg["rasterization"]["representations"].items():
        if r.get("kind") not in RASTER_KINDS:
            raise M8ConfigError(f"rasterization representation {name}: kind must be one of {RASTER_KINDS}")
    for a in cfg["ranking"]["ablations"]:
        if a not in ABLATIONS:
            raise M8ConfigError(f"unknown ranking ablation {a!r}; known {ABLATIONS}")
    if "label_free" not in cfg["ranking"]["ablations"] or "cnn_only" not in cfg["ranking"]["ablations"]:
        raise M8ConfigError("ranking.ablations must contain the label_free and cnn_only controls")
    if cfg["graph"]["edge_model"] not in ("heuristic", "lgbm"):
        raise M8ConfigError("graph.edge_model must be heuristic | lgbm")
    if not 0 < float(cfg["ensemble"]["weight_step"]) <= 1:
        raise M8ConfigError("ensemble.weight_step must lie in (0, 1]")
    if cfg["submission"]["export_physics_if"] not in ("PROMOTE", "NEVER"):
        raise M8ConfigError("submission.export_physics_if must be PROMOTE | NEVER")
    for t in ("shared",):
        if (cfg.get(t) or {}).get("retrain_existing_cnns"):
            raise M8ConfigError("shared.retrain_existing_cnns=True is refused: M8 reuses saved B0 / B1 predictions")
