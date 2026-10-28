"""P7 configuration: base YAML + bounded experiment ladder + runtime presets (pure; no torch, no data).

    cfg  = load_p7_config(overrides)                 # base + experiments + GPU presets, validated
    ecfg = resolve_experiment(cfg, "T2")             # deep-merged experiment configuration
    sem  = semantic_view(ecfg)                       # what changes trained weights (hashing / resume contract)

Runtime knobs (micro-batch, accumulation, inference batch, checkpoint sync, runtime caps) live in
``configs/p7/colab_gpu.yaml`` and never enter ``semantic_view``: the same experiment trained on a T4, L4 or A100 has
the same spec hash and may resume across them.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

P7_VERSION = "p7.1"
P7_CODE_REV = "p7.1-r1"                          # checked by the notebook start-up cell (not fingerprinted)
ROOT = Path(__file__).resolve().parents[2] / "configs" / "p7"
DEFAULT_BASE = ROOT / "geofaultformer.yaml"
DEFAULT_EXPERIMENTS = ROOT / "experiments.yaml"
DEFAULT_GPU = ROOT / "colab_gpu.yaml"
KINDS = ("reference", "train", "tracer", "stacker")
HEADS = ("fault", "center", "orient", "distance", "endpoint", "junction")
LOSS_HEAD = {"fault": "fault", "center": "center", "orient": "orient", "distance": "distance",
             "topology": "fault", "endpoint": "endpoint"}
SEMANTIC_KEYS = ("inputs", "data", "model", "loss", "training", "seed")
RUNTIME_TRAINING_KEYS = ("micro_batch", "grad_accum", "infer_batch", "ckpt_sync_min", "max_runtime_min",
                         "oom_retries", "amp_dtype")


class P7ConfigError(ValueError):
    """An experiment definition is inconsistent (e.g. a loss on a disabled head, an unknown parent)."""


def merge(base: dict, over: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _read(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_p7_config(config: dict | None = None, *, experiments: dict | None = None, gpu: dict | None = None,
                   base_path: Path | None = None, experiments_path: Path | None = None,
                   gpu_path: Path | None = None) -> dict:
    """Base + overrides; ``cfg['ladder']`` = experiment table, ``cfg['gpu']`` = runtime presets. Validated."""
    cfg = merge(_read(base_path or DEFAULT_BASE), config)
    cfg["ladder"] = merge(_read(experiments_path or DEFAULT_EXPERIMENTS), experiments)
    cfg["gpu"] = merge(_read(gpu_path or DEFAULT_GPU), gpu)
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    lad = cfg["ladder"]
    ex = lad.get("experiments") or {}
    if len(ex) > int(lad.get("max_experiments", 8)):
        raise P7ConfigError(f"{len(ex)} experiments > max_experiments {lad.get('max_experiments')}: the ladder is "
                            "bounded on purpose")
    for name, e in ex.items():
        if e.get("kind") not in KINDS:
            raise P7ConfigError(f"{name}: kind {e.get('kind')!r} not in {KINDS}")
        p = e.get("parent")
        if p not in (None, "best") and p not in ex:
            raise P7ConfigError(f"{name}: unknown parent {p!r}")
        if e["kind"] == "train":
            resolve_experiment(cfg, name)                      # raises on inconsistent heads / losses
    ev = cfg["evaluation"]
    if cfg["evaluation"]["objective"] not in ("disc", "disc_geo_floor", "joint", "geo"):
        raise P7ConfigError(f"evaluation.objective {ev['objective']!r}")
    if not set(ev["recall_budgets"]) <= set(ev["area_fracs"]):
        raise P7ConfigError("every recall budget must be an evaluated area fraction (rows are reused, not rebuilt)")
    if cfg["inference"]["full_region"]["construction"] not in ("zone_stitched", "fold_average"):
        raise P7ConfigError("inference.full_region.construction must be zone_stitched or fold_average")


def resolve_experiment(cfg: dict, name: str) -> dict:
    """Deep-merged configuration of one trained rung (+ ``experiment`` metadata). Raises ``P7ConfigError``."""
    ex = (cfg["ladder"].get("experiments") or {}).get(name)
    if ex is None:
        raise P7ConfigError(f"unknown experiment {name!r}")
    base = {k: copy.deepcopy(cfg[k]) for k in SEMANTIC_KEYS if k in cfg}
    e = merge(base, ex.get("overrides") or {})
    e["experiment"] = {k: ex.get(k) for k in ("kind", "hypothesis", "parent", "description", "requires")}
    e["experiment"]["name"] = name
    if ex.get("kind") != "train":
        return e
    heads, w = e["model"]["heads"], e["loss"]["weights"]
    if not heads.get("fault"):
        raise P7ConfigError(f"{name}: the fault head is mandatory (every candidate map is ranked from it)")
    for term, wt in w.items():
        if float(wt) > 0 and not heads.get(LOSS_HEAD.get(term, term)):
            raise P7ConfigError(f"{name}: loss {term} has weight {wt} but head {LOSS_HEAD.get(term, term)} is off")
    at = e["model"]["attention"]
    if at.get("enabled"):
        if not set(at["scales"]) <= {1, 2, 3, 4}:
            raise P7ConfigError(f"{name}: attention scales {at['scales']} must be encoder stages 1..4")
        if int(at["window"]) < 2 or int(at["heads"]) < 1:
            raise P7ConfigError(f"{name}: attention window / heads")
        for s in at["scales"]:
            if int(e["model"]["widths"][int(s) - 1]) % int(at["heads"]):
                raise P7ConfigError(f"{name}: width of stage {s} not divisible by {at['heads']} heads")
    g = e["model"]["convgru"]
    if g.get("enabled") and (int(g["steps"]) < 0 or int(g["scale"]) not in (1, 2, 3)):
        raise P7ConfigError(f"{name}: convgru steps >= 0 and scale in (1, 2, 3)")
    if e["data"]["augmentation"] not in ("conservative", "structural", "none"):
        raise P7ConfigError(f"{name}: augmentation preset {e['data']['augmentation']!r}")
    if e["model"]["uncertainty"]["method"] not in ("none", "mc_dropout"):
        raise P7ConfigError(f"{name}: uncertainty method (seed ensembles are separate experiments)")
    return e


def semantic_view(ecfg: dict) -> dict:
    """Everything that changes trained weights; runtime knobs removed."""
    out = {k: copy.deepcopy(ecfg[k]) for k in SEMANTIC_KEYS if k in ecfg}
    out["training"] = {k: v for k, v in out.get("training", {}).items() if k not in RUNTIME_TRAINING_KEYS}
    out["experiment"] = {"name": ecfg["experiment"]["name"]}
    return out


def trained_experiments(cfg: dict) -> list[str]:
    return [n for n, e in (cfg["ladder"].get("experiments") or {}).items() if e.get("kind") == "train"]


def experiment_id(name: str, fp: str) -> str:
    """Readable, collision-free directory name: a changed configuration gets a new directory (never overwritten)."""
    return f"p7_{name.lower()}_{fp[:10]}"


def gpu_preset(gcfg: dict, info: dict, select: str | None = None) -> tuple[str, dict]:
    """(preset name, preset) for the detected accelerator: explicit ``select`` wins, else the largest preset whose
    ``min_vram_gb`` fits the total VRAM; CPU when no CUDA device is present."""
    presets = gcfg["presets"]
    sel = select or gcfg.get("select", "auto")
    if sel != "auto":
        if sel not in presets:
            raise P7ConfigError(f"GPU preset {sel!r} not in {sorted(presets)}")
        return sel, dict(presets[sel])
    if not info.get("cuda_available"):
        return "CPU", dict(presets["CPU"])
    total = float(info.get("total_gb") or 0.0)
    fit = sorted(((float(p["min_vram_gb"]), n) for n, p in presets.items() if n != "CPU"
                  and float(p["min_vram_gb"]) <= total), reverse=True)
    name = fit[0][1] if fit else "T4"
    return name, dict(presets[name])


def micro_split(effective: int, micro: int) -> tuple[int, int]:
    """(micro-batch, accumulation) with micro x accumulation == effective (micro shrinks to a divisor)."""
    effective, micro = max(int(effective), 1), max(min(int(micro), int(effective)), 1)
    while effective % micro:
        micro -= 1
    return micro, effective // micro
