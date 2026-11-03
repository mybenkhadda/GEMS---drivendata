# G07 - high-resolution multiscale fault discovery

Code rev `g07.1-r1`. **Status: authored, never executed** (`SESSION_HANDOFF.md`).

**Hypothesis.** High-resolution surface morphology (USGS 3DEP 1 m DEM), better fault supervision and independent
subsurface hypotheses reveal structures W2 misses; learned geological ranking + DTI-aware exact-budget selection turn
them into better predictions. **The first milestone decides whether the DEM helps at all** (DEM feasibility GO / NO_GO).
The 0.40 leaderboard DTI is a target, never inferred from offline proxies (W2: V1 0.282538, V3 0.130868, public 0.0942
- other protocols, not comparable).

| what | where |
|---|---|
| notebook | `notebooks/colab/GEMS_G07_MULTISCALE_FAULT_DISCOVERY.ipynb` (sections 00-30) |
| package | `gems_bench/g07/` (orchestrator `pipeline.G07Pipeline`) |
| configuration | `configs/g07/discovery.yaml` |
| tests (synthetic, run in Colab) | `tests/test_g07_*.py` (11) |
| outputs | `MyDrive/GEMS_2/outputs/g07_multiscale_fault_discovery/<run_id>/` |

Systems: **A** surface (DEM geomorphometry, scarps, lineaments -> classical models, optional U-Net), **B** subsurface
(G06 store reused; G06-P4 families with improved supervision), **C** candidate hypotheses (label-free pools with
provenance, geometric positive augmentation of TRAINING truth, calibrated relevance, utility), **D** selection / export
(exact-k with deterministic label-free tie-breaking, W2-preserving portfolio, strict gates, READY_FOR_MANUAL_UPLOAD at
best - nothing is uploaded).

Documents: `ARCHITECTURE.md`, `DEM_DATA_CONTRACT.md`, `FEATURE_DICTIONARY.md`, `TRAINING_DESIGN.md`,
`VALIDATION_PROTOCOL.md`, `EXPERIMENT_MATRIX.md`, `ARTIFACT_REGISTRY.md`, `COLAB_EXECUTION.md`, `SESSION_HANDOFF.md`.
