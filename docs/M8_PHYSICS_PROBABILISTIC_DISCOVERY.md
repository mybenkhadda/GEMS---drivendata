# M8: physics-guided probabilistic fault discovery (guide)

Notebook: `notebooks/colab/GEMS_M8_PHYSICS_PROBABILISTIC_DISCOVERY.ipynb` · code: `gems_bench/m8/` · config:
`configs/m8/physics_probabilistic_discovery.yaml` · outputs: `MyDrive/GEMS_2/outputs/m8_physics_probabilistic/` ·
tests: `tests/test_m8.py`.

**Status: code authored only. Nothing has been executed** (no cell, test, training, inference, download or raster
job). No M8 number exists yet. The M7 values quoted below are earlier offline records, not leaderboard results.

## Why

Recorded M7 offline results (official-style `disc_official` DTI): B0 raw 0.104, B1 raw 0.092, fusion 0.123, fusion +
skeleton 0.128, fusion + crest thinning 0.162, fusion + 300 m ring 0.198. The M7 candidate pool covered ~4.2 % of
withheld systems (5 / 120) and none of the 93 systems missed by both CNNs. M8 asks *why*: is the signal absent at
100 m, removed by preprocessing, invisible to the operators, extracted but ranked too low, only visible in 1 m DEM,
or only recoverable from the network geometry? Every section is built to separate these explanations, and the
CNN champion stays the benchmark and the fallback.

## Design

* **Composition, not duplication.** `M8Research` owns an `M7Research` engine (units, eligible budget domain,
  protocol keys, member maps, every M7 helper) and never calls an M7 stage that writes. Its GEMS3 engine is
  re-pointed to `outputs/m8_physics_probabilistic/gems3_engine`; M7 helpers that write receive `M7View` (the M7
  engine seen through the M8 workspace). M7 artifacts are read-only (`m7_output_dir`).
* **One protocol.** Every number is scored by `gems3.official.FastScorer` on the M7 units with the eligible budget
  domain; held-out maps carry protocol keys; `m7.comparison` refuses mismatched protocols.
* **Profiles = scope only.** `PILOT` (default, CPU, 16 realisations on one discovery unit), `VALIDATE` (64, one
  region per discovery unit), `FULL` (128, full grid). Profiles may only carry the keys in `config.SCOPE_KEYS`;
  units, masks, metric, budget domain, operator definitions and gate thresholds are identical in every profile.
  Gates decided under PILOT are labelled PROVISIONAL.
* **Label use.** Labels are read only by evaluation code (oracle, signal-recovery diagnosis, PU labels for training
  on other zones, comparisons). Physics layers, worms, agreement, Monte Carlo, completion and candidates are
  label-free.

## Integration inventory (also `manifests/integration_inventory.csv`)

| reused | file | M8 consumer |
|---|---|---|
| raw bands, grid, valid, template | `gems_bench/data/__init__.py` | layers, signal_recovery, profiles, montecarlo |
| units, eligible domain, protocol keys, ledgers, system hits, block bootstrap | `m7/protocol.py`, `gems3/official.py` | everywhere |
| baseline nested selection, held-out maps | `m7/baseline.py` | baseline, ensemble |
| failure analysis, system attributes | `m7/failure.py` | baseline, figures |
| response / NMS / gated crests / tracing / fuse / stable ids | `m7/generator.py` | candidates, worms, signal_recovery, montecarlo |
| coverage, density curves, oracle | `m7/coverage.py`, `gems3/oracle.py` | candidates, completion, dem |
| network elements, deterministic proposals, corruption benchmark | `m7/network.py`, `m7/corruption.py` | completion |
| DEM pilot / streaming / mosaic / DEM crests | `m7/dem_*.py` | dem |
| candidate descriptors, PU labels, LOZO ranking | `m7/cand_features.py`, `m7/ranking.py` | features, ranking |
| rasterisation helpers, held-out persistence | `m7/rasterize.py` | rasterize, ranking, completion, graph, ensemble |
| comparison, verdicts, exports, submission validation | `m7/comparison.py`, `m7/decision.py`, `m7/export.py`, `gems3/submission.py` | ensemble, submission |
| THDR / VDR / tilt recipes, prank, NaN-aware smoothing | `gems_eda/enhance.py`, `rs_features.py` | potential_fields, multimodal |

Change to existing code: one method, `Pipeline.m8()` in `gems_bench/pipeline/__init__.py`.

## Derivative provenance

`tmi_hg`, `tmi_vg`, `iso_grav_anom_hg`, `iso_grav_anom_vg`, `*_slope` are compared (label-free Spearman, fixed
sample) with the same derivative computed from their declared parent (`tmi`, `iso_grav_anom`, `det_elev`). A band is
`VERIFIED_METADATA` (user-confirmed in `bands.verified`), `CONSISTENT_EMPIRICAL` (|rho| >= 0.80) or `PROVISIONAL`.
`*_band` operators (analytic signal / tilt from the provided bands) are used only when both bands of a family are
verified or consistent and their scale factors agree within `unit_ratio_tolerance`. Units are recorded as unknown
unless confirmed in `bands.units`. Table: `geophysical_features/band_provenance.csv`.

## Sections and outputs

| section | module | main outputs (under `outputs/m8_physics_probabilistic/`) | gate |
|---|---|---|---|
| 00 setup | `run.py`, `inventory.py` | `manifests/setup.json`, `integration_inventory.csv`, `band_availability.csv`, `protocol_keys.csv` | stops on CRS / template mismatch |
| 01 baseline | `baseline.py` | `baseline/baseline_table__<fp>.csv` (immutable), `recorded_vs_this_run.csv`, `compatibility.csv`, `system_table.csv`, `baseline_manifest.json`, `validation/maps/s01_*` | — |
| 02 signal recovery | `signal_recovery.py` | `signal_recovery/stage_matrix`, `stage_long`, `reasons`, `stage_recall`, `missed_by_both_by_stage`, `stage_density`, `case_studies/*` (8 panels + NPZ) | G1 (after 06) |
| 03 potential fields | `potential_fields.py`, `layers.py` | `geophysical_features/band_provenance`, `effective_resolution`, `operator_inventory`, `<family>/<op>_<scale>m.tif` | — |
| 03b worms | `worms.py` | `geophysical_features/structural_worms/worms_<family>.tif`, `worm_levels`, `coverage_vs_fixed_scale` | — |
| 04 multimodal | `multimodal.py` | `multimodal/mag_grav_agreement.tif`, `crest_classes.tif`, `context/context_regional.tif` + tables | — |
| 05 profiles | `profiles.py` | `fault_normal_profiles/units/*`, `transects_*`, `gallery/gallery_profiles` + figure | — |
| 06 candidates | `candidates.py` | `candidate_generation/units/*.npz`, `candidates.gpkg`, `oracle/*`, `oracle/gate2.csv`, `signal_recovery/gate1.csv` | **G1, G2** |
| 07 Monte Carlo | `montecarlo.py` | `monte_carlo/realizations/{state_*.npz, manifest_*.csv, assumptions.json}`, `stability/{stability, stability_vs_truth, mc_only_coverage, gate3}`, `uncertainty_maps/occupancy.tif`, `extras/*` | G3 |
| 08 completion | `completion.py` | `fault_networks/units/*`, `coverage_audit`, `system_c_summary`, `benchmark/benchmark_selection`, `validation/maps/m8_completion` | — |
| 09 DEM | `dem.py` | `dem/pilot/*`, `dem/candidates/{results, systems, gate4, dem_volume_log}` | G4 |
| 10 features | `features.py` | `feature_tables/{feature_table, units/*, feature_groups, feature_provenance, missingness, label_balance}` | needs G2 GO |
| 11 ranking | `ranking.py` | `ranking/{ranking_summary, ranking_scores, training_audit, feature_importance, gate5}`, `validation/maps/m8_rank_*` | G5 |
| 12 graph (optional) | `graph.py` | `graphs/{edges, graph_summary, graph_metrics, graph_decision}` | needs G5 PASS |
| 13 rasterisation | `rasterize.py` | `rasterization/{rasterization_summary, rasterization_ablation, area_mass_ledger}`, `validation/maps/m8_physics` | — |
| 14 ensemble | `ensemble.py`, `gates.py`, `report.py` | `ensemble/{fusion_summary, gate6}`, `validation/metrics/*`, `reports/{gates, decision_verdicts, research_answers, M8_PHYSICS_PROBABILISTIC_REPORT.md}` | **G6** |
| 15 submission | `submission.py` | `submissions/*.tif` + `.json`, `submission_index.csv`, `leaderboard_log.csv` | never uploads |

## Monte Carlo methodology (section 07)

Per realisation r (seed = seed + 1000 + r), per survey family: correlated Gaussian noise (white noise smoothed at
`noise_corr_m` ∈ {200, 400, 800} m) with amplitude `noise_frac` ~ U(0, 0.10) × robust std of the high-pass field
(or the measured sigma of `measured_noise`), one rigid registration shift per family ~ N(0, 0.25 × effective
resolution), operator scale × exp(N(0, 0.2)), operator ~ uniform over the family's operators; extraction: top-frac
× U(0.7, 1.4), coherence gate ~ U(0.2, 0.4), linking gap ∈ {2, 3, 5} px, angle ∈ {20, 30, 40}°, persistence ∈ {1,
2}. Only base fields are perturbed (derived bands of one survey are never perturbed independently). These are
assumptions recorded in `monte_carlo/realizations/assumptions.json`; without `measured_noise` this is a heuristic
sensitivity analysis. Detection frequency is processing stability, never a calibrated fault probability.

## Ranking architecture (section 11)

Controls: `label_free` (rank mean of label-free physics / geometry / worm / agreement / profile evidence) and
`cnn_only` (B0 / B1 support along the line). LightGBM ablations: `physics_only`, `physics_mc`, `physics_dem`,
`physics_cnn`, `all` (+ optional CatBoost, PU negative-weight sensitivity). Training is M7's leave-one-zone-out
with a 2 km buffer and linked-network exclusion; negatives (far from truth) are down-weighted, unlabelled rows are
dropped. Scores are rankings, not probabilities.

## Colab run order

1. Setup cells (configuration, install, context), section 00. CPU only (`shared.submission.source = oof_stitched`).
2. Sections 01 → 02 → 03 → 03b → 04 (label-free layers are cached in `cache/m8/layers`).
3. Section 06 (it builds the section-08 hypotheses first). **Read `candidate_generation/oracle/gate2.csv` and
   `signal_recovery/gate1.csv`.**
4. Section 05 (profiles of the M8 pool), 07 (Monte Carlo, PILOT first), 08 (benchmark + System C), 09 (DEM: review
   `dem/pilot/pilot_tiles.csv`; streaming needs `dem_pilot.stream` and `shared.dem.download`).
5. GO: 10 → 11 → (12) → 13. NO_GO: read `signal_recovery/reasons.csv` and `oracle/pooled_groups.csv`, widen the
   generation families, re-run 06.
6. Section 14 (comparison, gates, report). Section 15 only after reading `reports/M8_PHYSICS_PROBABILISTIC_REPORT.md`.
7. Switch `profile` to `VALIDATE` and re-run 07 onwards (stages outside the profile's scope are reused).

## Risks (unverified at runtime)

* Runtime: full-grid operators (FFT VDR, upward continuation) and per-scale tracing; full-region Monte Carlo
  (minutes per realisation); per-candidate Python loops in features / profiles; the s14 fusion grid.
* Memory: EDTs with indices on the full grid (~0.3 GB each) in worms / multimodal / Monte Carlo.
* The derivative audit decides whether *_band operators run; if the bands are PROVISIONAL, only computed derivatives
  are used.
* Effective resolutions are configured assumptions (spectral estimates are reported next to them).
* The 'raw response' null is system-matched but still a heuristic; raw-absent ≠ invisible.
* MC covers only the profile's regions: rows outside get `has_mc = 0`.
* Pseudo-discovery under-represents new faults next to known traces; both ring variants are exported.
* The hidden new-fault density is unknown: the submission area fraction is the largest unverified choice.
