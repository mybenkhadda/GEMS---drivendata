"""Phase-5 research experiments as pipeline work items: hypothesis, control, requirements, state, execution.

Each experiment is an ordinary ``configs/experiments/<stem>.yaml`` run by ``experiment.Experiment`` (four
restartable stages, unit-level resume, checkpoints pushed to Drive). The pipeline adds:

* a catalogue (scientific hypothesis, control, resource class, prerequisites) shown before spending GPU time;
* the session's validation design and non-semantic runtime knobs (stage budget, GPU overrides);
* prerequisite checks with exact recovery actions (GPU present, ...); label-free dependency maps (the
  ``B0_lineament`` evidence of B1) are found, verified and - if absent - built with their original CPU recipe
  (``ensure_label_free_map``) before the experiment starts;
* read-only state from ``experiments/<id>/manifest.json`` for plans / dashboards (no folders created);
* the experiment fidelity (``experiments.profile``: SCREEN / VALIDATE / FULL, ``pipeline.fidelity``) and a
  reuse-first plan (``plan_experiments``): a completed FULL run is never retrained because a cheaper profile is
  selected, and only missing work is scheduled.

Promotion is never automatic: candidates reach ``validated`` in the registry (FULL runs only); ``promoted`` is a
manual, evidence-backed transition (``gems registry promote``). SCREEN / VALIDATE shortlists
(``pipeline.screening``) only decide which experiment earns the next fidelity level.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

GATE = "section-17 promotion gate (configs/validation/phase5_protocol.yaml): primary pseudo-discovery gain, no " \
       "alt-policy / geographic / fold-min regression, >= 1 unique recovered system, bounded far-field FP share"

# role: control (shared reference) | candidate (a hypothesis) | parity_check (implementation equivalence test:
# expected to reproduce its control, never an independent improvement)
ROLES = ("control", "candidate", "parity_check")


@dataclass(frozen=True)
class ResearchExperiment:
    stem: str
    group: str
    hypothesis: str
    control: str
    gpu: bool = True
    requires_maps: tuple[str, ...] = ()
    note: str = ""
    role: str = "candidate"


CATALOGUE = {e.stem: e for e in (
    ResearchExperiment("b0_reference", "B/C/E control",
                       "Re-training the gated dual encoder with the Phase-5 trainer reproduces P4_B_gated within seed "
                       "variance; it is the shared control for every B / C / E run.", "P4_B_gated", role="control"),
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
                       "faithful generalisation.", "P5_B0_gated", role="parity_check",
                       note="identical modules, widths (24/32), construction order and seed as P4_B_gated / B0: "
                            "training is expected to be numerically identical to B0 (observed: identical fold-0 "
                            "losses and inner DTI for 14 epochs). An implementation check, not an architectural "
                            "improvement; run E2-E5 against it, not instead of B0"),
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
        rows.append({"experiment": s, "group": e.group if e else "?", "role": e.role if e else "?",
                     "control": e.control if e else "",
                     "resource": "gpu" if (e and e.gpu) else "cpu", "hypothesis": e.hypothesis if e else "(not catalogued)",
                     "promotion": GATE if e else "", "note": e.note if e else ""})
    return pd.DataFrame(rows)


def role_of(stem: str) -> str:
    e = CATALOGUE.get(stem)
    return e.role if e else "candidate"


def control_stem(stem: str) -> str | None:
    """Experiment stem that produces this experiment's control model under the same fidelity (B0 for B/C/E)."""
    ctrl = (CATALOGUE.get(stem).control if stem in CATALOGUE else "") or ""
    return {"P5_B0_gated": "b0_reference", "P5_C1_heads": "c1_geometry_heads",
            "P5_E1_gsn_gated": "e1_gsn_parity"}.get(ctrl)


# ---------------------------------------------------------------------------
# configuration / state
# ---------------------------------------------------------------------------
def session_fidelity(pc, profile: str | None = None):
    """The session's fidelity: ``experiments.profile`` + ``experiments.fidelity_overrides[profile]``."""
    from .fidelity import make_fidelity

    p = (profile or pc.cfg.get("experiments.profile") or "FULL").upper()
    ov = dict((pc.cfg.get("experiments.fidelity_overrides") or {}).get(p) or {})
    ck = pc.cfg.get("experiments.checkpoint.policy")
    if ck and "checkpoint_policy" not in ov:
        ov["checkpoint_policy"] = ck
    return make_fidelity(p, ov)


def experiment_config(pc, stem: str, overrides: dict | None = None, profile: str | None = None):
    """Experiment config with the session's validation design (semantic), fidelity (semantic below FULL) and
    runtime knobs (non-semantic). FULL leaves the committed YAML untouched: same id as every earlier run."""
    from ..experiment.config import ExperimentConfig, deep_merge
    from .fidelity import apply_fidelity

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
    ck = {k: v for k, v in (pc.cfg.get("experiments.checkpoint") or {}).items() if k != "policy"}
    if ck:
        raw["runtime"]["checkpoint"] = {**(raw["runtime"].get("checkpoint") or {}), **ck}
    if raw.get("kind") == "cnn5":
        raw = apply_fidelity(raw, session_fidelity(pc, profile))
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
        ensure_label_free_map(pc, m, requested_by=stem)
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


def ensure_label_free_map(pc, name: str, requested_by: str = "") -> dict:
    """Dependency map of an experiment (``B0_lineament`` for B1): reuse a verified copy, else build it.

    1. already resolvable in the session context -> reuse;
    2. its canonical folder exists on Drive -> stage it in, verify (run finished, native grid, fold provenance)
       and reuse;
    3. otherwise, when ``experiments.auto_provision_maps`` is true and the recipe is a label-free CPU map,
       build it with the original recipe (``baselines.train_baseline``: the same code path and committed spec
       snapshot as ``train_baselines``), push it, and verify again.

    Only maps in ``losses.pu.LABEL_FREE_MAPS`` are accepted: a supervised model's predictions are never
    substituted (they would confirm their own errors). An INCOMPATIBLE copy (other fold files) is never
    overwritten."""
    from ..losses.pu import LABEL_FREE_MAPS
    from . import baselines as B
    from .context import PipelineStop
    from .stages import folds_sha, splits_sha

    if name not in LABEL_FREE_MAPS or name not in B.RECIPES:
        raise PipelineStop(f"{requested_by}: dependency map {name} is not a label-free baseline recipe",
                           f"allowed label-free maps: {[m for m in LABEL_FREE_MAPS if m in B.RECIPES]}")
    ctx = pc.ctx5()
    if ctx.has_oof(name):
        return {"map": name, "action": "reused (session)"}
    r = B.RECIPES[name]
    n = pc.bench_cfg.n_discovery_folds
    pc.stage_in(B.all_rels(n, [name]))
    fs, ss = folds_sha(pc), splits_sha(pc)
    st = B.baseline_status(pc.local, name, grid=pc.grid(), n_splits=n, folds_sha=fs, splits_sha=ss)
    if st["status"] == B.INCOMPATIBLE:
        raise PipelineStop(f"{requested_by} needs {name}, but the existing copy was made on other fold files",
                           f"it is kept untouched; move {B.cv_rel(r)} aside on Drive and re-run to rebuild it")
    if st["status"] != B.AVAILABLE:
        if not pc.cfg.get("experiments.auto_provision_maps", True):
            raise PipelineStop(f"{requested_by} needs the {name} map, which is not available ({st['detail']})",
                               f"add '{name}' to train_baselines (CPU, minutes) or set auto_provision_maps=True")
        if r.gpu or r.kind != "map":
            raise PipelineStop(f"{requested_by} needs {name}; automatic provisioning covers label-free CPU maps only",
                               f"add '{name}' to train_baselines")
        print(f"[deps] {requested_by} needs {name} (label-free, CPU): building it with the original recipe")
        res = B.train_baseline(pc, name, folds_sha=fs, splits_sha=ss)
        pc.push(res.get("rels", []))
        st = B.baseline_status(pc.local, name, grid=pc.grid(), n_splits=n, folds_sha=fs, splits_sha=ss)
        if st["status"] != B.AVAILABLE:
            raise PipelineStop(f"{name} was built but does not validate: {st['detail']}",
                               f"inspect {B.cv_rel(r)} (run.json, oof.tif grid)")
        action = "built (original recipe)"
    else:
        action = f"reused ({st['origin'] or 'legacy'} copy, verified)"
    ctx._maps.pop(name, None)
    if not ctx.has_oof(name):
        raise PipelineStop(f"{name} validates on disk but the context cannot resolve it",
                           "re-run the notebook from the pipeline cell (stale context)")
    print(f"[deps] {name}: {action}")
    return {"map": name, "action": action}


def effective_profile(pc, stem: str, cfg=None) -> str:
    """The session profile for cnn5 experiments; CPU kinds (LightGBM, DEM) always run FULL."""
    cfg = cfg or experiment_config(pc, stem)
    return cfg.profile if cfg.kind == "cnn5" else "FULL"


def plan_experiments(pc, stems: list[str] | None = None, profile: str | None = None, root: Path | None = None,
                     inventory: pd.DataFrame | None = None) -> pd.DataFrame:
    """Reuse-first plan: one row per requested experiment with the action this session would take.

    ``REUSE_FULL``       a compatible FULL run is complete (never retrained for a cheaper profile)
    ``REUSE_HIGHER``     a compatible run at a higher fidelity than requested is complete
    ``DONE``             the requested profile is already complete
    ``RESUME``           the requested run exists and is incomplete (unit-level resume from its checkpoints)
    ``RUN``              nothing usable: train at the requested profile
    ``CONFIG_ERROR``     the YAML cannot be resolved

    A SCREEN / VALIDATE comparison needs the control under the *same* profile: when a candidate will run, its
    control stem is added (action as above). Read-only."""
    from .fidelity import RANK
    from .inventory import experiment_inventory

    stems = list(stems or pc.cfg.get("experiments.advanced") or [])
    want = (profile or pc.cfg.get("experiments.profile") or "FULL").upper()
    inv = inventory if inventory is not None else experiment_inventory(pc, root=root)
    rows, queue = [], list(stems)
    while queue:
        stem = queue.pop(0)
        if any(r["experiment"] == stem for r in rows):
            continue
        try:
            cfg = experiment_config(pc, stem, profile=want)
        except (FileNotFoundError, ValueError) as e:
            rows.append({"experiment": stem, "profile": want, "action": "CONFIG_ERROR", "reason": str(e)[:200]})
            continue
        prof = effective_profile(pc, stem, cfg)
        mine = inv[inv.experiment_id == cfg.experiment_id()] if len(inv) else inv
        same = inv[(inv.experiment == cfg.name) & (inv.config_compatible == True)] if len(inv) else inv  # noqa: E712
        full_done = same[(same.profile == "FULL") & same.complete] if len(same) else same
        higher = same[same.profile.map(lambda p: RANK.get(p, -1) > RANK[prof]) & same.complete] if len(same) else same
        row = {"experiment": stem, "role": role_of(stem), "profile": prof, "experiment_id": cfg.experiment_id()}
        if prof != "FULL" and len(full_done) and role_of(stem) != "control":
            row.update(action="REUSE_FULL", reason=f"FULL run {full_done.experiment_id.iloc[0]} is complete",
                       reuse_id=full_done.experiment_id.iloc[0])
        elif prof != "FULL" and len(higher) and role_of(stem) != "control":
            row.update(action="REUSE_HIGHER", reason=f"{higher.profile.iloc[0]} run {higher.experiment_id.iloc[0]} "
                       "is complete", reuse_id=higher.experiment_id.iloc[0])
        elif len(mine) and bool(mine.complete.iloc[0]):
            row.update(action="DONE", reason="requested profile already complete", reuse_id=cfg.experiment_id())
        elif len(mine):
            row.update(action="RESUME", reason=f"incomplete: {mine.next_action.iloc[0]}")
        else:
            row.update(action="RUN", reason="no compatible run found")
        rows.append(row)
        if prof != "FULL" and row["action"] in ("RUN", "RESUME"):
            ctrl = control_stem(stem)
            if ctrl and ctrl not in stems and not any(r["experiment"] == ctrl for r in rows):
                queue.append(ctrl)                     # matched control under the same screening protocol
    # a control trains at SCREEN / VALIDATE only for a candidate that needs the matched comparison
    for r in rows:
        if r.get("role") == "control" and r.get("profile") != "FULL" and r.get("action") in ("RUN", "RESUME"):
            needed = any(control_stem(o["experiment"]) == r["experiment"] and o.get("action") in ("RUN", "RESUME", "DONE")
                         for o in rows)
            if not needed:
                r.update(action="NOT_NEEDED", reason="no candidate at this profile needs the matched control")
    return pd.DataFrame(rows)


def run_experiment(pc, stem: str, overrides: dict | None = None, profile: str | None = None) -> dict:
    cfg = experiment_config(pc, stem, overrides, profile=profile)
    check_prerequisites(pc, stem, cfg)
    exp = open_experiment(pc, cfg)
    exp.run_all()
    pc.ctx5().refresh_zoo()
    st = exp.status()
    out = {"experiment": stem, "experiment_id": exp.id, "profile": cfg.profile,
           **{r.stage: r.status for r in st.itertuples()}}
    if exp.rd.has_table("promotion_decisions"):
        d = exp.rd.load_table("promotion_decisions")
        out["promote_recommended"] = bool(d.get("promote", pd.Series([False])).any())
    return out
