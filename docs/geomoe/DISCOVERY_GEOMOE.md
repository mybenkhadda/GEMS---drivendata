# GEMS discovery-first GeoMoE (dgm.1-r1)

One Colab experiment that asks why W2 (public DTI 0.0942) does not find new faults, whether independent geological
evidence recovers systems the learned maps miss, and whether a geology-conditioned gate beats a fixed fusion of the
same evidence, then writes a GeoTIFF only when a predeclared rule selects a model.

* Notebook: `notebooks/colab/GEMS_DISCOVERY_GEOMOE.ipynb` (CPU, High-RAM)
* Code: `gems_bench/geomoe/` (`run.DiscoveryGeoMoE`); config `configs/geomoe/discovery_geomoe.yaml`
* Tests: `tests/test_discovery_geomoe.py` + runtime `gems_bench/geomoe/selftest.py`
* Outputs: `MyDrive/GEMS_2/outputs/discovery_geomoe/`

**Status: authored, never executed.** No number in this document is a result.

## 1. Reuse (nothing rebuilt)

| component | available | executed before | reused here as | limitation | action |
|---|---|---|---|---|---|
| Submission factory (`leaderboard_focus/factory.py`) | yes | yes (fold-stitched files pass read-back) | frozen-engine loader (`selftest`, `preflight`, `attach`); `readback_checks`, `verdict` | its own diagnostics are refreshed in its root | attach only, no factory candidate exported |
| B0 / B1 honest maps (`gems3.official.unit_score`) | yes | yes | E1 sources | OOF on geo folds, discovery maps on disc splits only | none |
| W2 held-out maps (`stack/maps/*.npy`) | yes | yes | E1 source, Model A_pixel | full-region refit failed parity (P6.1 X1) | zone-stitched only |
| W2 unit features (`p6.stack.unit_raw` via `Boost.raw`) | yes | yes | CNN ranks, potential fields, permitted known geometry, G02 layers | one PF scale | second scale added (label-free) |
| G01 / G02 rasters (`g02_23686a63cb`) | partial | yes (FAST_RESEARCH) | E3 sources | zone 2 / split 2 rasters were missing | missing units -> E3 UNAVAILABLE there (never zero, never backfilled) |
| M7 DEM mosaic (100 m from 1 m tiles) | unknown | pilot | E4 `dem` layer | pilot coverage | `cached_only`: staged if on Drive, never downloaded |
| M7 protocol (`score_row`, `block_components`, `paired_block_bootstrap`, `eligible`, `protocol_key`) | yes | yes | every score and bootstrap | - | none |
| P6.1 X3 segment tools (`dedup_inventory`, `label_geometry`) | yes | yes | candidate inventory | skeleton approximations | none |
| P6.1 X2 definitions | yes | yes | reconciled in stage 05 | other pool / budget than M7 | same-definition table |
| `gems3.metric_opt.nested_select` | yes | yes | nested (rep, frac) per scenario | - | none |
| `p6.final_submission` + `export_parity` | yes | yes | stitch, write, validate, recipe, plan audit | zone-stitched | writer root redirected to this experiment |

## 2. Validation design (Experiment A)

| scenario | units | truth | known context / exclusion | sources |
|---|---|---|---|---|
| V1 withheld systems | `disc_official` splits | withheld systems | retained faults (pixel-exact, official convention) | split discovery CNN maps, W2 held-out |
| V2 network completion | each geo fold k (whole fold = evaluation zone) | real mapped fold-k pixels withheld as tips (0.5-2 km), junction branches (<= 3 km), short splays (< 2 km), interior sub-sections | faults outside fold k + retained fold-k faults; retained fold-k faults masked pixel-exactly | fold-k OOF CNN, W2 held-out (no model saw any fold-k label) |
| V3 geographic | geo folds | every mapped fault of fold k | faults outside fold k | fold-k OOF CNN, W2 held-out |

V2 never invents geometry (no lateral shift); the withheld pixels reach no input (leakage checks, stage 03). Every
model, gate weight, normaliser and (rep, frac) cell of held-out zone z is fitted on the other zones of the SAME
scenario, minus candidates within 2 km of zone z (`train_buffer_km`). Training, inner (training-zone) and outer
(held-out) numbers are separate columns (`insample_dti`, `dti_dev`, `dti_heldout`); oracle rows carry
`role = oracle` and are refused by every selection (`evaluate.guard_oracle`). V1 / V2 / V3 stay separate; the
decision counts scenarios (no weighted pseudo-DTI).

Distance bands (near <= 1 km, mid <= 5 km, far) split TP / FN by truth pixel and FP by predicted pixel; the bands sum
to the official components (`scenarios.banded_components`).

## 3. Candidates and the coverage ceiling (Experiment B)

Per unit: every available source layer -> 1-px `skeleton_1px` lines at 5 % of the eligible domain (low threshold);
straight tip continuations of the permitted known traces; dedup across sources (spatial + orientation, crossings and
short lines always kept); long branches cut to <= 3 km; provenance (`proposed_by`, `on_<source>`, `prop_<family>`).

Coverage: a system is covered by a pool when >= 50 % of its scored pixels lie within 300 m of it (the system recovery
rule of `m7.protocol.system_hits`). Tables: per pool / family, unique-family systems, missed systems (no pool),
coverage by subtype (V2 class) and length bin, and two curves inside the inventory under the same budgets: `oracle`
(truth-ordered, diagnostic) and `w2_order` (label-free). Gate (stage 06): GENERATION if the inventory covers < 60 % of
systems; RANKING if it covers >= 60 % and W2's deployed cell recovers >= 20 points less; NEITHER_DOMINANT otherwise.
The learned gate trains only on RANKING / NEITHER_DOMINANT (or when forced in CONFIG, which is recorded).

M7 vs X2: M7 measured LABEL-FREE crest coverage (~12.5 %); X2 attributed 82.9 % of missed LENGTH to ranking against a
pool of the LEARNED maps at up to 5 % area. Stage 05 re-measures `label_free`, `fam:E1_learned`, `inventory` and W2's
deployed recovery under one definition (`audit/protocol_comparison_measured.csv`).

## 4. Experts, fusion and gating (Experiments C, D)

Experts: E1 learned (B0, B1, W2), E2 potential field (mag / grav gradients, cross-modal strike agreement, two-scale
persistence), E3 structural (G02-C, G02-A, G01 persistence), E4 terrain / network (tip continuation x along-strike
cosine, cached DEM). Expert score = mean of the available member layers along the candidate; normalised by an ECDF
fitted on the training zones only. Missing = UNAVAILABLE: weights renormalise over the available experts; DEM absence
is `dem_observed = 0`, never evidence against a fault.

Models (same pool, same raster policies, same budget, same scorer):
* A_pixel frozen W2 pixel map (reference); A_seg W2 mean on the unified pool;
* B static fusion: simplex grid (step 0.25) chosen by length-weighted PU average precision on training candidates;
* C rule gate: B's weights modified by R1 terrain scarp, R2 structural displacement, R3 tip continuation, R4
  magnetic-edge-only down-weighting, R5 short multi-source preservation; C_all + one ablation per rule;
* D learned gate: softmax(V z + b) routing over available experts, logistic link, L2 + KL toward B's weights, L-BFGS
  with analytic gradient; collapse diagnostic (mean max weight and dominant share >= 0.9 -> not promotable).

PU targets: positive = >= 50 % of the candidate within 300 m of its training zone's scored truth; everything else
unlabelled with weight 0.3, never a confirmed negative.

Selection (`decide.select_model`): hard checks (no leakage, metric equivalence with the frozen boost-A grid, fair
budgets, W2 baseline present) -> the simplest model that beats A_pixel by >= 0.005 in >= 2 of 3 scenarios without a
scenario drop > 0.01 or worst-zone drop > 0.01 -> C_all replaces B only with a paired gain >= 0.005, D only with
>= 0.01 and bootstrap P(delta > 0) >= 0.8 in >= 2 scenarios and no collapse. Otherwise RETAIN_W2.

## 5. Deployment and export

Stage 11 compares on V3, at the same per-zone cells: `zone_stitched` (re-derived, must match the stage scores),
`fold_average` (one model per other zone, averaged) and `footprint_rank` (layers ranked over the stitched footprint:
the full-region refit transformation, on honest maps). Export uses `zone_stitched` only, and only when it is
self-consistent. Stage 12 rebuilds geo zone k from the V3 candidates and the model fitted without zone k, with ONE
all-zone cell (the model's in-sample choice on V1), writes through `p6.final_submission.write_file` under
`outputs/discovery_geomoe/submissions/`, validates (`final_submission.validate`, `audit_export`, plan audit), reads
the file back (`factory.readback_checks`) and records sha256. READY_FOR_UPLOAD only when every hard check passed on
the read-back file. RETAIN_W2 -> no file (`NO_NEW_SUBMISSION`), unless `export.exploratory` (labelled EXPLORATORY).

## 6. How to run

1. Commit and push; open the notebook in Colab from GitHub; Runtime CPU, High-RAM.
2. Edit only the CONFIG cell (pin `ref` / `EXPECT['commit']` to the pushed SHA).
3. Run all cells top to bottom (or `RUN_ALL()` after cell 2). Cell 3 runs the self-tests and pytest.
4. After a disconnect: restart, run cells 1-2, then the stage cells again (`SKIP_CACHED`, units `reused`).

Longest stage: [04] (per unit: W2 unit features, 13 pools, dedup, coverage). [08]-[10] evaluate every model on
2 reps x 5 fracs x 9 units; [11] rebuilds footprint-ranked layers once.

## 7. Outputs (Drive `outputs/discovery_geomoe/`)

`audit/` component_inventory, source_compatibility, label_leakage_audit, protocol_comparison(+_measured),
metric_equivalence, selftests, pytest_output · `validation/` scenarios, folds, v2_events, per_zone_metrics,
near_known_metrics, per_fault_type_metrics, system_recovery.parquet · `candidates/` candidate_inventory.parquet,
candidate_geometries.gpkg (optional), candidate_source_agreement, source_overlap, source_provenance, oracle_coverage,
oracle_curves (= oracle_curve), unique_systems, missed_systems, coverage_by_fault_type, coverage_decision.json,
units/<fp>/ (per-unit labels + parts) · `experts/` expert_features.parquet, expert_scores.parquet,
source_normalization.json, source_availability, pf_scale2.npz · `gating/` grid_*, scores_*.parquet,
static_fusion_results, rule_gate_results, rule_firing_rates, learned_gate_results, feature_importance,
gating_weights.parquet, model_summary, paired_comparison, selection.json, model_registry.json, deployment_* ·
`predictions/heldout/` chosen held-out maps (packed bits) · `predictions/full_region/` exported score + binary ·
`submissions/` *.tif + sidecars, submission_manifest.{csv,json}, checksums.txt, README_UPLOAD.md, export_gates ·
`figures/` 7 PNGs + `<name>_data.csv`, MISSING.txt · `reports/` DISCOVERY_DIAGNOSIS, GEOMOE_BENCHMARK,
SCIENTIFIC_DECISION, NEXT_ACTION · `manifests/` run.json, runtime_errors.json, stages/<stage>.json.

## 8. Unverified until Colab runs

* Every test and self-test (written, never run); every API call against the live engine (`Boost.raw`, `w2_map`,
  `m7.elig`, `G02Experiment` rasters, DEM mosaic bands).
* Metric equivalence of A_pixel vs the frozen boost-A grid (expected exact; a FAIL blocks selection).
* Runtime / memory of stage 04 and 11 on full crops; candidate counts per unit.
* Whether V1 eval zones overlap so much that the 2 km buffer starves training (recorded as `n_buffer_dropped`;
  models fall back to static / uniform weights, recorded).
* `OFFICIAL_EXCLUDE_SUPPORTS_TP` remains the repository's reading of the organiser statement.

## 9. Fresh-session handoff

```
GEMS discovery GeoMoE dgm.1-r1: authored 2026-10-29, NOTHING executed. Notebook
notebooks/colab/GEMS_DISCOVERY_GEOMOE.ipynb drives gems_bench/geomoe/run.DiscoveryGeoMoE through the submission
factory's frozen engine. Before claiming anything read, on Drive outputs/discovery_geomoe/: manifests/run.json,
manifests/stages/*.json, audit/selftests.csv, audit/metric_equivalence.csv, candidates/coverage_decision.json,
gating/selection.json, submissions/submission_manifest.json, reports/SCIENTIFIC_DECISION.md, reports/NEXT_ACTION.md.
W2 public 0.0942 is an external reference, never a tuning target. Do not start M9 full neural training.
```
