# M7: discovery-first research (guide)

Notebook: `notebooks/colab/GEMS_M7_DISCOVERY_FIRST.ipynb` · code: `gems_bench/m7/` · config:
`configs/m7/discovery_first.yaml` · outputs: `MyDrive/GEMS_2/outputs/m7_discovery_first/`.

**Status: code authored only. Nothing has been executed** (no cell, test, training, inference or raster job).
No M7 number exists yet.

## Why

Recorded offline results suggest that ~78 % of withheld fault systems are missed by both B0 and B1, and that the
earlier candidate generator covered only ~12.5 % of them. If coverage is the bottleneck, a better ranker cannot
help. M7 therefore measures candidate coverage first (section 04) and only then ranks.

## Shared fixes (also used by GEMS3 and the legacy submission path)

| issue | file | fix |
|---|---|---|
| finite values written where the official template is NaN but the data footprint is valid | `gems3/submission.py::finalise`, `gems_bench/submission.py::finalise` | NaN wherever the template is NaN; 0 where it is finite outside the data |
| validator demanded finite values on template-NaN pixels inside the footprint | `gems3/submission.py::validate`, `submission.py::check_submission` | finite required on `valid & template_valid` only; finite on template NaN is a failure |
| area budget counted / selected on all valid pixels (known faults, ring, template nodata consumed it) | `gems3/representation.py::apply(eligible=)`, `gems3/submission.py::eligible_mask / transform` | budget domain `eligible` = valid & template finite & not known & not ring (GEMS3 config `submission.budget_domain`, default `eligible`; files get an `_elig` suffix so historical files are never overwritten) |
| no record of the area each mask removed | `gems3/submission.py::area_ledger` | positive area after every step, as shares of valid and eligible pixels, plus pixels on known / ring / template nodata |

The template convention was checked against `data.load_native` (template NaN = masked or non-finite pixels of
`example_submission.tif`) and the problem statement ("null / NaN outside the bounds"). Whether the scorer reads a
finite value inside the bounds but outside the data footprint is not documented: M7 keeps writing 0 there.

## Run order

1. Setup cells (configuration, install, context, section 00). CPU is enough for everything except fold-average
   inference in sections 15-16 (or set `submission.source = 'oof_stitched'`).
2. Sections 01-04 (CPU). **Read `candidate_generation/oracle/gate_decision.csv`.**
3. GO: sections 08-14. NO-GO: read `miss_diagnostics.csv` / `miss_summary.csv`, widen generation, re-run 03-04.
4. DEM: section 05 (no download) -> review `dem/pilot/pilot_tiles.csv` -> `dem.download = True`, enable 06-07.
5. Sections 15-16 only after reading the section-14 report.

## What to inspect first

1. `recovered_baselines/recorded_vs_recomputed.csv` and `baseline_table__*.csv` (immutable).
2. `failure_analysis/tables/summary.csv`, `recall_by_stratum.csv`, the case-study panels.
3. `candidate_generation/oracle/gate_decision.csv`, `pooled_curve.csv`, `incremental.csv`.
4. `validation/metrics/method_pooled.csv`, `paired_zone_deltas.csv`, `bootstrap.csv`.
5. `reports/M7_DISCOVERY_FIRST_REPORT.md`.

## Risks (unverified at runtime)

* Runtime of the full-grid generator (8 sources x up to 4 scales) and of the Python tracing / dedup loops;
  the label-free layer is cached in `cache/m7`.
* Full-region feature table (section 10) over all full-region candidates: per-candidate Python loops.
* `network.proposals` on the full region (all endpoints x hypotheses).
* DEM openness / multiscale filters per 1500 x 1500 block: memory ~ a few hundred MB per block; timings unknown.
* GDAL `q3` resampling availability depends on the rasterio / GDAL build.
* Pseudo-discovery under-represents new faults next to known traces: ring results may not transfer (probes A / B).
* The hidden new-fault density is unknown: the full-region area fraction is the largest unverified choice.
