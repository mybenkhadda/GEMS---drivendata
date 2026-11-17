"""G14 GeoFaultFormer: local-global, multimodal, topology-aware fault discovery.

STATUS: authored without any execution (no Python, test, notebook or training run). Every module is UNTESTED until
the notebook's audit cells and ``pytest tests/test_g14_*.py`` have run. See docs/g14/UNTESTED_RISKS.md.

Package map
-----------
config       G14Config (data views, model, loss, training, evaluation, ensemble, run modes) + validation + hashing
experiments  declarative registry: T0-T7 ladder and the ablation matrix (overrides on G14Config)
contract     modality registry, channel schema, availability masks, known-fault context, train-only normalisation,
             provenance manifest, contract checker
splits       UnitSystem over two backends (G13 splits with a locked TEST fold; Phase-5 folds shared with P6/W2/P7),
             orientation-aware fault-system grouping, regime holdout, fold manifest
multiscale   nan-aware pyramids, three aligned views (local 100 m, regional 200 m, global 800 m), deterministic
             sampling, D4 augmentation with directional-channel transforms
model        GeoFaultFormer: modality encoders, adaptive fusion, local CNN, regional Swin-style transformer, global
             token encoder, aligned local-global fusion, U-Net/FPN decoder, heads
losses       masked BCE, soft Tversky, distance-aware centreline, soft clDice, background sparsity, orientation,
             connectivity, DTI-inspired surrogate (NOT the official metric)
train        staged trainer (AMP, accumulation, clipping, AdamW, warmup-cosine, checkpoints, resume, early stopping)
inference    tiled halo inference with Hann blending and multiscale context at the study-area boundary
evaluation   one evaluator for every model (official components via gems3.official), nested area selection
graph        rule-based segment-graph refinement (P6 / G13 topology reused) with an optional edge classifier
ensemble     mean, cross-fitted weights, OOF blending, regime gating, spatial mixture of experts
submission   explicit, guarded export through the G13 writer / validator
pipeline     notebook orchestrator with run modes AUDIT_ONLY / VALIDATE_DATA / TRAIN / EVALUATE / SUBMIT
"""
from __future__ import annotations

G14_VERSION = "g14.0.1-untested"
SCHEMA_VERSION = "g14.channels.v1"

__all__ = ["G14_VERSION", "SCHEMA_VERSION"]
