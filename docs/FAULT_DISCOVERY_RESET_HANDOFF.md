# Fault-discovery reset: handoff (fdr.1-r1)

**State (2026-10-29): authored, NOTHING executed** (no local run, no tests run, no Colab output yet). Uncommitted.
Public reference: frozen submitted W2 = 0.0942 (external only, never a tuning target).

## Files

| file | role |
|---|---|
| `notebooks/colab/GEMS_FAULT_DISCOVERY_RESET.ipynb` | the one notebook (modes ATLAS / BENCHMARK / DEM_PILOT / CHALLENGER / FULL / RESUME) |
| `configs/reset/fault_discovery_reset.yaml` | every predeclared threshold |
| `gems_bench/reset/run.py` | `FaultDiscoveryReset(DiscoveryGeoMoE)`: stages, modes, reviews, export |
| `gems_bench/reset/audit.py` | source inventory, prior-results audit (+ Colab checks), validation contract, limitations |
| `gems_bench/reset/atlas.py` | honest stitched maps, categories A-H, panels (EVIDENCE_ONLY / DIAGNOSTIC), HTML, append-only reviews, widget |
| `gems_bench/reset/physics.py` | label-free layers, per-candidate modality evidence, contrast / disruption / continuity flags, localisation |
| `gems_bench/reset/benchmark.py` | A0-A6 table, W2 representation grid, mass-matched curves, fixed-policy selection, paired bootstrap |
| `gems_bench/reset/terrain.py` | opt-in DEM pilot via M7 (`dem_pilot.run/download`, `dem_morph.run`), stop rule |
| `gems_bench/reset/challenger.py` | case decision 1-4 from measured evidence; Case 1 export path; Case 2/3 verifier |
| `gems_bench/reset/report.py` | 10 figures (+ data CSV) and 8 reports |
| `gems_bench/reset/selftest.py`, `tests/test_fault_discovery_reset.py` | synthetic tests (enforced in stage [00]) |
| `gems_bench/geomoe/run.py`, `geomoe/decide.py` | **modified**: stage tables are class attributes; `dependency_status` takes a depends map |

## Reused (not rebuilt)

Submission factory attach (frozen B0 / B1 / W2), GeoMoE V1 / V2 / V3 scenario units, official scorer and eligible
domain, `frozen_policy` / `frozen_smap` (submitted W2 cell, factory construction), fixed-policy selection and paired
block bootstrap, validated stage cache + dependency enforcement + integrity, `p6.stack.pf_full` and
`m8.potential_fields`, M7 DEM pipeline, `p6.final_submission` build / recipe / write / validate +
`factory.readback_checks`.

## Assumptions under test (what each output answers)

1. Learned evidence vs mapping style: `benchmark/source_complementarity.csv`, A1 vs A4 per scenario.
2. Broad anomalies vs 300 m: `physics/localization_error.csv` (truth within 300 m vs 3 km, corridor index).
3. Skeleton discards evidence: `benchmark/area_response.csv` (equal realised mass), `selected_policy.json`.
4. Edges = contacts / intrusions: `physics/structural_evidence.parquet` (contrast without disruption) + reviews.
5. Independent observations: `audit/source_inventory.csv`, `atlas/reviewed_annotations.csv`.
6. Terrain localisation: `terrain/pilot_decision.json`.
7. Validation optimism: `audit/validation_contract.csv`; V1 0.242 vs public 0.0942 is the measured gap.

## Rules that must not be broken

Unreviewed = not negative; DIAGNOSTIC reviews are not blind labels; reviews are append-only; missing DEM / G02 =
UNAVAILABLE, never absent; oracle / scenario-selected numbers never select; soft / dilated representations are
measured but not exported (the factory validator enforces area <= frac: changing that is a user decision); no neural
training; no automatic upload.

## Exact next action

Push, then run the notebook with `mode='ATLAS'`. Inspect `audit/stage_integrity.csv`, `audit/prior_results_audit.csv`,
`atlas/atlas_overview.html`, `physics/localization_error.csv`. Review EVIDENCE_ONLY panels (widget or template),
then rerun with `mode='BENCHMARK'` and read `benchmark/selected_policy.json` and `reports/NEXT_ACTION.md`.

## Unresolved

Whether any representation beats the frozen W2 under the fixed policy; whether broad potential-field crests localise
within 300 m; whether reviewed candidates reveal non-W2 faults; whether the cached DEM mosaic exists on Drive.
