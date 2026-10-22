"""Phase-5 research experiments as pipeline work items: hypothesis, control, requirements, state, execution.

Each experiment is an ordinary ``configs/experiments/<stem>.yaml`` run by ``experiment.Experiment`` (four
restartable stages, unit-level resume, checkpoints pushed to Drive). The pipeline adds:

* a catalogue (scientific hypothesis, control, resource class, prerequisites) shown before spending GPU time;
* the session's validation design and non-semantic runtime knobs (stage budget, GPU overrides);
* prerequisite checks with exact recovery actions (GPU present, B0_lineament map for B1, ...);
* read-only state from ``experiments/<id>/manifest.json`` for plans / dashboards (no folders created).

Promotion is never automatic: candidates reach ``validated`` in the registry; ``promoted`` is a manual,
evidence-backed transition (``gems registry promote``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

GATE = "section-17 promotion gate (configs/validation/phase5_protocol.yaml): primary pseudo-discovery gain, no " \
       "alt-policy / geographic / fold-min regression, >= 1 unique recovered system, bounded far-field FP share"


@dataclass(frozen=True)
class ResearchExperiment:
    stem: str
    group: str
    hypothesis: str
    control: str
    gpu: bool = True
    requires_maps: tuple[str, ...] = ()
    note: str = ""


CATALOGUE = {e.stem: e for e in (
    ResearchExperiment("b0_reference", "B/C/E control",
                       "Re-training the gated dual encoder with the Phase-5 trainer reproduces P4_B_gated within seed "
                       "variance; it is the shared control for every B / C / E run.", "P4_B_gated"),
    ResearchExperiment("b1_bg_weight", "B: PU supervision",
                       "Down-weighting unlabeled pixels that carry strong label-free lineament evidence (instead of "
                       "treating every unmapped pixel as a confident negative) improves pseudo-discovery DTI without "
                       "losing geographic DTI.", "P5_B0_gated", requires_maps=("B0_lineament",)),
    ResearchExperiment("b2_nnpu_p005", "B: PU supervision",
                       "A non-negative PU risk with class prior 0.5 % treats unmapped pixels as unlabeled; it improves "
                       "rediscovery of withheld systems (prior sweep member).", "P5_B0_gated"),
    ResearchExperiment("b2_nnpu_p010", "B: PU supervision",
                       "nnPU with prior 1 % (central value of the sweep) improves pseudo-discovery DTI over B0.",
                       "P5_B0_gated"),
    ResearchExperiment("b2_nnpu_p020", "B: PU supervision",
                       "nnPU with prior 2 % (upper sweep member); collapse monitor guards against trivial positives.",
                       "P5_B0_gated"),
    ResearchExperiment("b3_consistency", "B: PU supervision",
                       "Mean-teacher consistency on unlabeled pixels adds to the winning PU variant (run only after B1/B2 "
                       "passes its gate).", "P5_B0_gated"),
    ResearchExperiment("c1_geometry_heads", "C: geometry",
                       "Auxiliary distance-to-fault and axial-orientation (cos 2θ, sin 2θ) heads regularise the decoder "
                       "towards thin, continuous, correctly oriented traces.", "P5_B0_gated"),
    ResearchExperiment("c2_thin_continuity", "C: geometry",
                       "Label-free thinness and along-strike continuity terms improve trace geometry over C1.",
                       "P5_C1_heads"),
    ResearchExperiment("c3_cldice", "C: geometry", "A small soft-clDice centreline term improves continuity over C1.",
                       "P5_C1_heads"),
    ResearchExperiment("d1_profiles_lgbm", "D: multiscale structure",
                       "Fault-normal profile contrasts (step, asymmetry, offset, persistence) add discriminative signal "
                       "to RS05 LightGBM.", "RS05_G3_noSS_RS", gpu=False),
    ResearchExperiment("d2_profiles_cnn", "D: multiscale structure",
                       "Profile channels that helped D1 also help the gated dual encoder.", "P5_B0_gated"),
    ResearchExperiment("e1_gsn_parity", "E: GeoStructureNet",
                       "GeoStructureNet in parity configuration (gated fusion) matches B0: the architecture is a "
                       "faithful generalisation.", "P5_B0_gated"),
    ResearchExperiment("e2_gsn_attention", "E: GeoStructureNet",
                       "Modality-wise attention fusion beats gated fusion at matched parameters.", "P5_E1_gsn_gated"),
    ResearchExperiment("e3_gsn_crossscale", "E: GeoStructureNet",
                       "Cross-scale fusion (coarse potential-field levels gating fine terrain levels) helps.",
                       "P5_E1_gsn_gated"),
    ResearchExperiment("e4_gsn_dilated", "E: long-range context",
                       "One dilated long-range bottleneck block captures along-strike context (limited ablation).",
                       "P5_E1_gsn_gated"),
    ResearchExperiment("e5_gsn_context", "E: GeoStructureNet",
                       "A near-zero-gated coarse strain / seismicity branch avoids the harm these fields did to tabular "
                       "models.", "P5_E1_gsn_gated"),
    ResearchExperiment("f4_g1_dem", "F: 1 m DEM", "1 m DEM morphology aggregates improve G1 on DEM-covered pixels.",
                       "G1", gpu=False, note="needs the DEM pilot mosaic (dem:* features)"),
    ResearchExperiment("f4_dual_dem", "F: 1 m DEM",
                       "DEM channels improve the gated dual encoder on DEM-covered pixels.", "P5_B0_gated",
                       note="needs the DEM pilot mosaic (dem:* features)"),
)}


def catalogue_table(stems: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for s in stems or list(CATALOGUE):
        e = CATALOGUE.get(s)
        rows.append({"experiment": s, "group": e.group if e else "?", "control": e.control if e else "",
                     "resource": "gpu" if (e and e.gpu) else "cpu", "hypothesis": e.hypothesis if e else "(not catalogued)",
                     "promotion": GATE if e else ""})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# configuration / state
# ---------------------------------------------------------------------------
def experiment_config(pc, stem: str, overrides: dict | None = None):
    """Experiment config with the session's validation design (semantic) and runtime knobs (non-semantic)."""
    from ..experiment.config import ExperimentConfig, deep_merge

    p = pc.config_file("experiments", f"{stem}.yaml")
    cfg = ExperimentConfig.from_yaml(p)
    raw = cfg.raw
    canonical_f = "outputs/benchmark/folds/geo_k3_b20km_buf13km_s2026"
    canonical_d = "outputs/benchmark/folds/discovery_d3_link1km_s2026"
    if pc.fold_rel != canonical_f:
        raw.setdefault("validation", {})["folds_dir"] = pc.fold_rel
    if pc.discovery_rel != canonical_d:
        raw.setdefault("validation", {})["discovery_dir"] = pc.discovery_rel
    rt = pc.runtime_overrides()
    raw.setdefault("runtime", {})
    if rt.get("max_stage_minutes"):
        raw["runtime"]["max_stage_minutes"] = rt["max_stage_minutes"]
    if rt.get("gpu"):
        raw["runtime"]["gpu"] = {**(raw["runtime"].get("gpu") or {}), **rt["gpu"]}
    if overrides:
        raw = deep_merge(raw, overrides)
    return ExperimentConfig(raw, p)


def experiment_state(root: Path, cfg) -> dict:
    """Read-only: experiment id + stage statuses from experiments/<id>/manifest.json under ``root``."""
    from ..experiment.manifest import STAGES

    eid = cfg.experiment_id()
    mf = Path(root) / "experiments" / eid / "manifest.json"
    st = {"experiment_id": eid, "kind": cfg.kind, **{s: "pending" for s in STAGES}, "complete": False}
    if mf.exists():
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
            for s in STAGES:
                st[s] = (m.get("stages", {}).get(s) or {}).get("status", "pending")
            st["complete"] = all(st[s] == "done" for s in STAGES)
            st["config_hash_ok"] = m.get("config_hash") == cfg.hash()
        except (OSError, json.JSONDecodeError):
            st["error"] = "unreadable manifest"
    return st


def open_experiment(pc, cfg):
    from ..experiment import Experiment

    # resume_policy="config": the config hash must match, a newer code commit is recorded as a warning
    return Experiment(cfg, pc.project_paths(), ctx=pc.ctx5(), resume_policy="config")


def check_prerequisites(pc, stem: str, cfg) -> None:
    from .context import PipelineStop

    e = CATALOGUE.get(stem)
    kind = cfg.kind
    if kind == "cnn5" or (e and e.gpu):
        if not pc.cfg.get("gpu.use_gpu"):
            raise PipelineStop(f"{stem} is a GPU experiment and use_gpu is False",
                               "set use_gpu=True on a GPU runtime, or remove it from experiments.advanced")
        pc.require_gpu(f"experiment {stem}")
    ctx = pc.ctx5()
    for m in (e.requires_maps if e else ()):
        if not ctx.has_oof(m):
            raise PipelineStop(f"{stem} needs the {m} map, which is not available",
                               f"add '{m}' to train_baselines (CPU, minutes) and re-run the baselines stage")
    if kind == "cnn5" and (cfg.get("model.base") or "") and not _base_spec_ok(ctx, cfg.get("model.base")):
        raise PipelineStop(f"{stem}: base spec {cfg.get('model.base')} not found",
                           "restore the legacy run folder or add a snapshot to configs/baselines/legacy_specs/")
    if stem.startswith("f4_") and not any(n.startswith("dem:") for n in ctx.store.specs):
        raise PipelineStop(f"{stem} needs dem:* features", "run the DEM pilot first (run_dem_pilot=True, dem.download=True)")


def _base_spec_ok(ctx, base: str) -> bool:
    from ..training.spec import base_spec_dict

    try:
        base_spec_dict(ctx, base)
        return True
    except FileNotFoundError:
        return False


def run_experiment(pc, stem: str, overrides: dict | None = None) -> dict:
    cfg = experiment_config(pc, stem, overrides)
    check_prerequisites(pc, stem, cfg)
    exp = open_experiment(pc, cfg)
    exp.run_all()
    pc.ctx5().refresh_zoo()
    st = exp.status()
    out = {"experiment": stem, "experiment_id": exp.id, **{r.stage: r.status for r in st.itertuples()}}
    if exp.rd.has_table("promotion_decisions"):
        d = exp.rd.load_table("promotion_decisions")
        out["promote_recommended"] = bool(d.get("promote", pd.Series([False])).any())
    return out
