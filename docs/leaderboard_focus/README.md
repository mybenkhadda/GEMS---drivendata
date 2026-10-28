# Leaderboard focus

## What this phase is

A focused phase whose only goal is a real, trustworthy DrivenData submission. Milestone 1 (M1) recovers, validates
and exports the strongest **existing** W2-based solution.

Milestone 2 (GeoSelect: W2 + fallback fault-segment ranking) starts only after M1 is complete.

## Why nothing is retrained

The bottleneck is not a missing model. It is the absence of one champion whose exported file is provably the map
that was validated. Everything needed already exists as frozen artifacts:

* B0 / B1 CNN out-of-fold and discovery maps;
* W2 cross-fitted held-out maps;
* the boost-A grid of official components.

Retraining would only add variance and cost.

## What W2 is

W2 is a LightGBM ranker (`gems_bench/p6/stack.py`).

* **Inputs:**
  * the B0 / B1 CNN rank maps and their disagreement;
  * potential-field gradients;
  * the geometry of the permitted known-fault map;
  * G02 features when present.
* **Training:** cross-fitted leave-one-zone-out with a 2 km buffer.
* **Outputs:** held-out score maps per unit (3 geographic folds, 3 discovery splits).

The score is ranked inside each zone, blended with the B0+B1 fusion, and thinned to a 1 px representation at an area
budget.

## What M1 delivers

1. **A champion.** The p6g predeclared rule runs on the frozen boost-A grid, restricted to the W2 family plus the
   B0+B1 fusion fallback.
2. **Official validation.** Per-zone official components are recomputed, and recorded values must reproduce.
3. **A parity diagnosis** of the historical W2 export blocker, from evidence.
4. **One zone-stitched GeoTIFF**, built through the existing validated writer, with a `READY_FOR_UPLOAD` gate.
5. **A leaderboard log** for the manual upload.

## Execution order

`notebooks/leaderboard_focus/00 → 01 → 02 → 03`. Then upload by hand, then run `04`. See `M1_EXECUTION_GUIDE.md`.

## Required Drive directories (`MyDrive/GEMS_2/`)

* `outputs/p6_fast_track/{stack, boost/benchmark, recovery/export_parity, submissions}`
* `outputs/m9_geofault_production/baseline`
* `experiments/b0_reference__*/`, `experiments/b1_bg_weight__*/`
* `outputs/benchmark/folds/*`
* `data/raw/geodawn/example_submission.tif`

## Resume and inspect

* Re-run any notebook from the top. Stages whose fingerprint matches print `reloaded`.
* Status and reasons: `outputs/leaderboard_focus/m1_champion_recovery/manifests/<stage>.json`.
* Readable summaries: `reports/*.md`.
* Upload instructions: `submissions/README.md`.

## Documents

* `M1_AUDIT.md`: repository audit (written before the code);
* `ARCHITECTURE.md`;
* `M1_EXECUTION_GUIDE.md`;
* `ARTIFACT_CONTRACTS.md`;
* `VALIDATION_AND_PARITY.md`;
* `SUBMISSION_GUIDE.md`;
* `TROUBLESHOOTING.md`;
* `M1_HANDOFF.md`.
