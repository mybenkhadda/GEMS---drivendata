# Leaderboard focus

## What this phase is

A focused phase whose only goal is a real, trustworthy DrivenData submission. Milestone 1 (M1) recovers, validates
and exports the strongest **existing** W2-based solution.

Milestone 2 (GeoSelect: W2 + fallback fault-segment ranking) starts only after M1 is complete.

## One notebook

`notebooks/leaderboard_focus/GEMS_M1_CHAMPION_TO_SUBMISSION.ipynb` is the only M1 entry point (code rev `m1.2-r1`). It
replaces notebooks `00`–`04`, which are kept for history and marked superseded. One `CONFIG` cell, one `Runner`
(`gems_bench/leaderboard_focus/workflow.py`), one cell per stage, every stage resumable.

| stage | what | status that lets the next stage run |
|---|---|---|
| 0 | configuration, Drive, strict clone, version lock | – |
| 1 | artifact inventory + selective staging | COMPLETE |
| 2 | frozen champion selection (boost-A grid only) | COMPLETE (= selection complete, not promotion) |
| 3 | official validation | COMPLETE or WARNING |
| 4 | export parity, then promotion gates | parity COMPLETE; promotion COMPLETE / WARNING = READY_FOR_EXPORT |
| 5 | submission export | READY_FOR_UPLOAD |
| 6 | leaderboard review (manual entry) | – |
| 7 | final report | – |

Modes: `FULL`, `RECOVER`, `VALIDATE`, `SUBMIT` (`M1_EXECUTION_GUIDE.md`).

## Why nothing is retrained

The bottleneck is not a missing model. It is the absence of one champion whose exported file is provably the map
that was validated. Everything needed already exists as frozen artifacts:

* B0 / B1 CNN out-of-fold and discovery maps;
* W2 cross-fitted held-out maps;
* the boost-A grid of official components.

## What W2 is

W2 is a LightGBM ranker (`gems_bench/p6/stack.py`), cross-fitted leave-one-zone-out with a 2 km buffer, on the B0 / B1
CNN rank maps and their disagreement, potential-field gradients and permitted known-fault geometry. Its held-out
score is ranked inside each zone, blended with the B0+B1 fusion and thinned to a 1 px representation at an area
budget.

## Four separate questions

The workflow never conflates them:

1. Did a stage **compute**? (stage status)
2. Did the champion pass **promotion**? (`selection/promotion_gates.csv`: discovery eligibility, geographic floor
   0.1758, validation, parity → READY_FOR_EXPORT)
3. Did the prediction pass **parity**? (`parity/parity_report.csv`)
4. Did the file pass **export validation**? (`submissions/submission_manifest.json` → READY_FOR_UPLOAD)

The geographic floor is mandatory by default. The last recorded selection (`a_w2_rep_frac`, nested discovery DTI
0.242156, geographic DTI 0.105687) is below it, so with the default settings the workflow will report
`GEOGRAPHIC_GATE_PASS BLOCKED` and write no GeoTIFF. Exporting it anyway is an explicit, recorded user decision
(`VALIDATION_AND_PARITY.md`).

## Required Drive directories (`MyDrive/GEMS_2/`)

* `outputs/p6_fast_track/{stack, boost/benchmark, recovery/export_parity}` (+ `submissions/p6_stack_fusion*.json`)
* `outputs/m9_geofault_production/baseline`
* `experiments/b0_reference__d54e938a/`, `experiments/b1_bg_weight__7755635b/`
* `outputs/benchmark/folds/*`
* `data/raw/geodawn/example_submission.tif`

## Documents

* `M1_AUDIT.md`: repository audit (first M1 session);
* `ARCHITECTURE.md`: stage graph, gates, code map;
* `M1_EXECUTION_GUIDE.md`: how to run, modes, caching;
* `ARTIFACT_CONTRACTS.md`: inputs, staging, fingerprints, schemas;
* `VALIDATION_AND_PARITY.md`: protocols, gates, the geographic-floor question;
* `SUBMISSION_GUIDE.md`; `TROUBLESHOOTING.md`; `M1_HANDOFF.md`.
