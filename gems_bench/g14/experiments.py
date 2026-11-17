"""Declarative experiment registry: the T0-T7 ladder and the ablation matrix.

Each entry is an override dict on ``G14Config`` plus metadata (hypothesis, control, kind). ``resolve(name)`` returns a
validated config; ``matrix()`` lists everything. Nothing here trains: the pipeline runs one entry when asked.

Every experiment shares the same folds, masks and evaluation definitions (``split`` and ``eval`` are never
overridden by an entry; ``resolve`` refuses an entry that tries).

UNTESTED: written without execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import G14Config

LOCAL_ONLY = {"model": {"regional_encoder": "none", "global_encoder": "none", "context_stages": (),
                        "bottleneck_cross_attention": False, "film_global": False}}
CORE_BCE = {"masked_bce": 1.0, "soft_tversky": 0.0, "distance_centerline": 0.0, "soft_cldice": 0.0,
            "background_sparsity": 0.0}


@dataclass
class Experiment:
    name: str
    kind: str                       # reference | train | refine | ensemble
    hypothesis: str
    control: str = ""
    overrides: dict = field(default_factory=dict)
    group: str = "ladder"


LADDER = [
    Experiment("T0_champion", "reference", "Reproduce the strongest existing honest maps (P6 W2, G13 champion) on the "
               "same units as every G14 model.", overrides={}),
    Experiment("T1_local_cnn", "train", "A local 100 m CNN segments faults better than tabular pixel models.",
               "T0_champion", {**LOCAL_ONLY, "model": {**LOCAL_ONLY["model"], "fusion": "concat"}}),
    Experiment("T2_local_regional", "train", "Regional (51 km) transformer context improves held-out recall.",
               "T1_local_cnn", {"model": {"global_encoder": "none", "film_global": False, "fusion": "concat"}}),
    Experiment("T3_local_regional_global", "train", "Global (205 km) context adds fault-system orientation priors.",
               "T2_local_regional", {"model": {"fusion": "concat"}}),
    Experiment("T4_gated_fusion", "train", "Mask-aware gated modality fusion beats early concatenation.",
               "T3_local_regional_global", {"model": {"fusion": "gated"}}),
    Experiment("T5_topology", "train", "Distance, orientation and connectivity objectives improve DTI and continuity.",
               "T4_gated_fusion", {"model": {"fusion": "gated", "heads": ("fault", "orientation", "connectivity")},
                                   "loss": {"aux_weights": {"orientation": 0.1, "connectivity": 0.1,
                                                            "dti_surrogate": 0.1}}}),
    Experiment("T6_graph_refinement", "refine", "Evidence-gated segment linking recovers fragmented systems.",
               "T5_topology"),
    Experiment("T7_ensemble", "ensemble", "Independently validated models are complementary.", "best single model"),
]

ABLATIONS = [
    # architecture
    Experiment("A_arch_cnn_only", "train", "architecture", "T3_local_regional_global", LOCAL_ONLY, "architecture"),
    Experiment("A_arch_transformer_only", "train", "architecture: regional transformer drives the prediction alone",
               "T3_local_regional_global", {"model": {"local_widths": (16, 32, 64, 128), "global_encoder": "none"}},
               "architecture"),
    Experiment("A_arch_cnn_transformer", "train", "architecture", "T3_local_regional_global",
               {"model": {"global_encoder": "none", "film_global": False}}, "architecture"),
    Experiment("A_arch_full_context", "train", "architecture", "", {}, "architecture"),
    # fusion
    *[Experiment(f"A_fusion_{f}", "train", "fusion", "T4_gated_fusion", {"model": {"fusion": f}}, "fusion")
      for f in ("concat", "gated", "cross_attention")],
    # loss
    Experiment("A_loss_bce", "train", "loss", "A_loss_full", {"loss": {"weights": CORE_BCE}}, "loss"),
    Experiment("A_loss_bce_tversky", "train", "loss", "A_loss_bce",
               {"loss": {"weights": {**CORE_BCE, "masked_bce": 0.6, "soft_tversky": 0.4}}}, "loss"),
    Experiment("A_loss_bce_distance", "train", "loss", "A_loss_bce",
               {"loss": {"weights": {**CORE_BCE, "masked_bce": 0.6, "distance_centerline": 0.4}}}, "loss"),
    Experiment("A_loss_bce_topology", "train", "loss", "A_loss_bce",
               {"loss": {"weights": {**CORE_BCE, "masked_bce": 0.6, "soft_cldice": 0.4},
                         "aux_weights": {"orientation": 0.1}}}, "loss"),
    Experiment("A_loss_full", "train", "loss", "", {}, "loss"),
    # feature family
    Experiment("A_feat_raw_only", "train", "features", "A_feat_all", {"data": {"raw_only": True}}, "features"),
    *[Experiment(f"A_feat_{m}_only", "train", "features", "A_feat_all",
                 {"data": {"modalities": (m,)}}, "features")
      for m in ("magnetic", "gravity", "terrain", "conductivity", "structural")],
    Experiment("A_feat_all", "train", "features", "", {}, "features"),
    Experiment("A_feat_no_known_context", "train", "features: is the permitted known-fault geometry what carries "
               "discovery?", "A_feat_all", {"data": {"known_context": False,
                                                    "modalities": ("magnetic", "gravity", "terrain", "conductivity",
                                                                   "geodesy", "structural")}}, "features"),
    # context
    Experiment("A_ctx_local", "train", "context", "A_ctx_all", LOCAL_ONLY, "context"),
    Experiment("A_ctx_local_regional", "train", "context", "A_ctx_all",
               {"model": {"global_encoder": "none", "film_global": False}}, "context"),
    Experiment("A_ctx_local_global", "train", "context", "A_ctx_all",
               {"model": {"regional_encoder": "none", "context_stages": ()}}, "context"),
    Experiment("A_ctx_all", "train", "context", "", {}, "context"),
    # recovery (post-processing on one trained model's maps)
    *[Experiment(f"A_rec_{r}", "refine", "recovery", "A_rec_none", {}, "recovery")
      for r in ("none", "directional_nms", "p6_recovery", "graph")],
    # validation (same model, different protocol: reported side by side, never mixed)
    *[Experiment(f"A_val_{p}", "train", "validation protocol", "", {"split": {"protocol": p}}, "validation")
      for p in ("spatial_cv", "spatial_cv_embargo", "fault_system_holdout", "leave_one_regime_out")],
]

REGISTRY = {e.name: e for e in LADDER + ABLATIONS}


def resolve(name: str, base: G14Config | None = None) -> G14Config:
    """Validated config of a registry entry (validation-protocol ablations are the only entries that touch split)."""
    if name not in REGISTRY:
        raise KeyError(f"unknown experiment {name!r}; known: {sorted(REGISTRY)}")
    e = REGISTRY[name]
    if ("split" in e.overrides or "eval" in e.overrides) and e.group != "validation":
        raise ValueError(f"{name}: experiments may not change split / eval definitions")
    cfg = (base or G14Config()).updated({**e.overrides, "experiment": name})
    return cfg


def matrix():
    import pandas as pd

    return pd.DataFrame([{"name": e.name, "group": e.group, "kind": e.kind, "hypothesis": e.hypothesis,
                          "control": e.control, "overrides": e.overrides} for e in REGISTRY.values()])
