# Corrected evaluation methodology

## Four different numbers (never interchanged)

| name | definition | where | use |
|---|---|---|---|
| training surrogate | differentiable soft DTI on patches (`g14.losses.dti_surrogate`, radius = 300 m / pixel) | `g14_1.losses` | loss term only |
| patch diagnostic | the same per validation batch, positive-enriched, averaged (G14's "inner surrogate", 0.85) | `training.inner_metrics` → `enriched_surrogate` | diagnostic only, never for selection |
| region DTI | official components (`gems_bench.metric` / `gems3.official`) **summed over the whole region**, then one ratio | `metric_audit.region_components`, `tiled_components`, `gems3` units | selection, calibration, comparison |
| leaderboard DTI | the competition's number | never computed | recorded by hand, never used for tuning |

Averaging per-batch ratios ("mean of ratios") is not a region DTI: `metric_audit.ratio_of_sums_vs_mean_of_ratios`
reports both.

## Protocols (F1)

`G141Config.validation.protocol` ∈ `geo_only | disc_only | geo_and_disc`, with `primary_task` ∈
`disc_official | geo`.

`validate()` refuses a primary task that the protocol does not produce. `discovery.check_protocol` cross-checks
protocol × tasks × primary task × available maps / checkpoints, with an actionable fix for each issue. A
geographic-only calibration produces a **PROVISIONAL_GEO** candidate.

## Discovery-first validation (F2)

Discovery units are G13 discovery splits: withheld systems, an ignore ring, an evaluation zone, and retained faults
masked pixel-exactly. Withheld systems are absent from:

* training targets;
* known context;
* derived channels (context recomputed per unit);
* label-dependent normalisation (none: per-unit stats are label-free, fitted on `unit.train`);
* candidate recovery inputs (the graph uses only visible labels).

Scoring uses the official-style discovery masks.

## Geographic generalisation (F3, secondary)

Spatial blocks with a 13 km embargo, whole-system crossing report, regime holdout (G13 regimes if stable).
Acquisition-region holdout uses coverage-boundary strata (`metric_audit.coverage_boundary`) as an atlas category and
an F07 evidence raster.

## Inner validation (F4)

`training.inner_metrics` reports four readings:

1. enriched patch diagnostic;
2. uniform / tiled patches;
3. deterministic non-overlapping tiles, aggregated by ratio of sums;
4. full-region inner prediction scored as one region DTI.

Checkpoint selection defaults to (4), via `training.selection_metric = 'inner_full_region_dti'`.

## Calibration per experiment (F5)

Each corrected model records:

* inner loss;
* inner patch diagnostic;
* inner region DTI;
* geographic and discovery cross-fitted DTI;
* worst zone;
* pooled components;
* per-zone variability.

`metric_audit.ranking_consistency` measures whether inner metrics rank models like held-out discovery DTI
(Spearman, Kendall; needs ≥ 3 models).

## Operating points

Every (representation, masking order, area, calibration) is chosen for zone z on the other zones of the task
(`g13.evaluation.crossfit`). The reports include:

* `optimism` (in-sample minus cross-fitted);
* requested vs scored area;
* the crest dilation factor;
* the grid-edge flag.

## Locked TEST

The G14 `evaluate_locked_test` is unchanged: single use, logged, and a second look is labelled `SECOND_LOOK`.
`train_final(include_test=True)` makes the official model use the TEST fold. After that the locked TEST is no longer
clean, and the docstring says so.
