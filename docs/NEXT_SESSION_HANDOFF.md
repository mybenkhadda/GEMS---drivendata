# Next-session handoff

**Newest (M9, 2026-10-26):** `notebooks/colab/GEMS_M9_GEOFAULT_PRODUCTION.ipynb` + `gems_bench/m9/` + `configs/m9/
geofault_production.yaml`; guide `docs/M9_GEOFAULT_PRODUCTION.md` (reuse matrix, stages, gates, first experiments).
New multiscale reconstruction network + tracing + GBM benchmark + fusion; B0/B1 stay the fallback. Nothing executed;
`tests/test_m9.py` never run. Only existing-code change: `Pipeline.m9()` + optional extras `m9` / `rl` in pyproject.

**M8:** `notebooks/colab/GEMS_M8_PHYSICS_PROBABILISTIC_DISCOVERY.ipynb` + `gems_bench/m8/` (no saved outputs in the
local checkout; M9 section 00 records whatever M8 left on Drive).

**M7:** `notebooks/colab/GEMS_M7_DISCOVERY_FIRST.ipynb` + `gems_bench/m7/`; guide `docs/M7_DISCOVERY_FIRST.md`.
Shared submission fixes (template NaN mask, eligible-pixel budget, area ledger) apply to GEMS3 too. Nothing executed.

## GEMS 3.0 (2026-10-26: discovery-first research)

Code authored only. **Nothing was executed**: no notebook, test, training, inference or raster processing. No
GEMS3 or M6 number exists yet. Working tree on branch `m6-model-selection`, **uncommitted** (see "Git").

## Start here

1. `docs/GEMS_OFFICIAL_METRIC_AUDIT.md`: the verified competition contract and the mismatch it fixed.
2. This file: modules, run order, and open risks.
3. Then only the module you are changing (`gems_bench/gems3/*`).

## The key finding (verified from the organizers' forum answers, not from code)

The hidden truth is **new faults only**. Known faults are masked **pixel-exactly** (no buffer). A prediction near a
known trace but far from new truth is fully penalised. One Tversky index is pooled over all public pixels. The
historical pseudo-discovery policy `excl` excluded a **300 m dilation** of the retained faults, so it was
over-forgiving. New primary development task: `disc_official`. Scorer change: an `exclude_supports_tp` flag in
`metric.dti_components`, `metric.MaskedScorer` and `p3.curves.BinaryCurve`. The default `True` keeps every
historical number unchanged; `False` = official reading (`metric.OFFICIAL_EXCLUDE_SUPPORTS_TP`).

## GEMS3 = notebook section 18c (`g3 = pipe.gems3(GEMS3_CONFIG); stage calls`)

Config: `configs/gems3/research.yaml` (unknown keys rejected). Outputs: `reports/gems3/` (pushed to Drive after
every stage), label-free candidate cache: `cache/gems3/`. Maps are loaded through M6's read-only inventory into
`reports/gems3/inventory/` (M6's own `reports/m6` is never touched).

| stage (switch) | module | main outputs |
|---|---|---|
| official metric (`verify_official_metric`, not optional) | `gems3/official.py` | `audit/official_contract`, `masking_audit`, `official_metric_comparison`, `artifact_inventory` |
| metric optimisation (`run_metric_optimization`) | `gems3/metric_opt.py`, `gems3/representation.py` | `metric_optimization/{threshold_sweep, area_budget_curves, width_ablation, dilation_ablation, known_ring_ablation, ensemble_ablation, representation_ablation, selected_parameters[_per_zone], stage2_candidates, raw_scores}`, `ensembles/{paired_fold_deltas, b0_b1_comparison, calibration_summary}`, `manifests/selected_representation.json`, `predictions/metric_opt_<task>_heldout.tif` |
| submission (`prepare_submission_candidate`) | `gems3/submission.py` (+ `inference.full_raster.cnn5_full_region`) | `predictions/<member>__full_region_raw.tif` (+json fp), `predictions/<cand>__score.tif`, `submissions/<cand>.tif` + `.json` manifest, `submissions/submission_candidates.csv` |
| candidates (`run_candidate_generation`) | `gems3/candidates.py` | `proposals/candidate_inventory`, `proposals/units/<unit>.{npz,csv}`, `proposals/candidate_geometries.gpkg`, `label_free_sources` |
| oracle (`run_candidate_oracle`) | `gems3/oracle.py` | `proposals/{oracle_recall, oracle_curve, oracle_by_source, oracle_decision, candidate_truth_stats}` |
| ranker (`run_candidate_ranker`; refuses unless the oracle says CONTINUE or `ranker.force`) | `gems3/descriptors.py`, `gems3/ranker.py` | `proposals/candidate_descriptors`, `ranker/{ranker_curves, ranker_summary, training_audit, ranker_vs_pixel_models, feature_importance, full_region_candidates_ranked}`, `ranker/full_region_ranked.gpkg`, `predictions/ranker_full_region.tif` |
| DEM gate (`run_dem_gate`) | `gems3/dem_gate.py` | `dem/{coverage, pilot_results, decision, features}` |
| map completion (`run_map_completion`) | `gems3/map_completion.py` | `map_completion/{corruption_protocol, validation_results, selection}` |
| report (always) | `gems3/report.py` | `GEMS3_RESEARCH_REPORT.md`, `metric_optimization/metric_optimization_report.md` |

Orchestration: `gems3/run.py` (`GEMS3Research`): per-stage fingerprints (configuration section + M6 map
fingerprints + `GEMS3_VERSION`), reuse, failure isolation (`manifests/stage_log.json`). Writer:
`gems3/workspace.py` (M6 `Workspace` subclass; figures PNG + SVG with their data CSV; Parquet next to CSV; GPKG).
Every figure uses the fixed validated palette order in `gems3/figures.py`.

## Design decisions made this session

1. **Nested selection everywhere** (`metric_opt.nested_select`): for held-out zone z, choices are made on the
   other zones **and** restricted to candidates derived from z's own dev choice (the M6 rule). Stage 1 = (w, frac),
   stage 2 = (rep, w, frac), stage 3 = + known ring (disc only).
2. `FastScorer` (official.py) gathers TP at truth pixels only, with components identical to `MaskedScorer`
   (tested). It is needed because stage 2 scores thousands of maps.
3. Area semantics: `frac` = share of the zone's evaluation pixels. For `crest` it counts crest pixels, so at equal
   `frac` the core areas match. Post-dilation area is reported as `area_frac`.
4. The submission budget defaults to the **geo** all-zone fraction (`submission.area_frac_source: geo`), because a
   disc evaluation zone (10 km around withheld systems) is denser than the region. The hidden new-fault density is
   unknown: **the largest unverified assumption**.
5. Submission NaN policy: 0 wherever the template is finite, NaN only where it is NaN (`outside_valid: template`).
6. `known_fill` defaults to off. Masked pixels carry no penalty, so a fill/no-fill pair is a cheap check of the TP
   support convention, not a hidden-label probe.
7. `SubmissionRunner.train` now calls the shared `inference.full_raster.cnn5_full_region` (same behaviour,
   no duplicated code).
8. Map completion stops at the corruption protocol + deterministic endpoint extension (as ordered: no CNN).
   The DEM pipeline already had the descriptors (`dem/morphology.py`); GEMS3 adds only the complementarity gate.
9. Not implemented (deliberately, by priority): external fault compilations (Milestone F; licence questions in
   forum t/11563), specialist experts (G), learned map completer, DEM scaling.

## Run next (Colab; a GPU only for the submission stage)

1. Locally: `pytest tests/test_gems3.py`, then `tests/test_metric.py tests/test_phase3_scoring.py tests/test_m6.py`
   (the scorer change must keep them green). None has ever been run.
2. Colab: open the notebook from this branch. Run cells 1–4 and sections 7–12 (cached), then 18c top to bottom
   with the default switches. Without a GPU, set `'submission': {'source': 'oof_stitched'}`.
3. Read in this order: `audit/official_metric_comparison` (how much `excl` overstated B0/B1), `g3.summary()`,
   `ensembles/paired_fold_deltas` (worst zone), `proposals/oracle_decision`, `GEMS3_RESEARCH_REPORT.md` §9.
4. Next experiment, by observed outcome:
   * Representation beats the better member's binary map in every zone → upload the GEMS3 candidate (one slot).
     Otherwise upload the binary fallback (`'submission': {'representation': 'binary'}`).
   * Oracle `CONTINUE` → `run_candidate_ranker: True`. `STOP` → widen generation first (more scales, lower
     `top_frac`, longer tips), not the ranker.
   * The `known_ring` selection > 0 → be careful: the proxy under-represents corrections next to known traces.
     Consider one upload with ring 0 vs selected.
   * DEM gate `CONTINUE` → scale `dem.download` + `dem.morphology` to more tiles; `STOP` → do not.
   * Map completion shows clear endpoint recovery → a learned completer on damaged maps (systems split by
     `map_completion.systems_by_fold` before corruption).

## Unverified at runtime (everything). Specific risks

* Stage-2 runtime: (#weights ≤ 2–3) × (≤ 15 fracs) × 11 reps per unit, on 6 units (geo + disc_official). Expected
  well under an hour on CPU; `crest_map` / `orientation_field` / `skeleton` on full fold crops dominate. Not measured.
* `label_free`: 7 full-grid crest extractions (~tens of s each). The `unit_candidates` dedup loop is pure Python
  over candidates.
* `oracle.unit_oracle` rasterises every kd threshold and every source: O(candidates × thresholds).
* `orientation_field` behaviour at line ends (oriented dilation): tests relax this.
* `descriptors.describe` per-candidate Python loops (fine up to ~10^5 candidates).
* LightGBM feature names come from config source names (keep them alphanumeric + `_`).
* The DEM gate needs `dem:coverage` + `dem:*` in the Phase-5 store (section 9 mosaic).
* Drive staging of checkpoints (`pc.stage_in`) and CNN inference memory on the full grid (`cnn5_full_region`).

## M6 (section 18b, previous session; also never executed)

Unchanged. See `docs/M6_RESULTS_GUIDE.md` and `docs/M6_RESEARCH_PLAN.md`. Open question carried over: whether the
FULL `b1_bg_weight` run is COMPLETE on Drive (the GEMS3 audit fails with the same named error if it is not).

## Git

All GEMS3 work is uncommitted on `m6-model-selection`. Suggested: `git switch -c gems3-discovery-research`, commit,
push, open a PR to `main` after the first Colab run.
