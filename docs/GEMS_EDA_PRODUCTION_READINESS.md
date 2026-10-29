# GEMS exhaustive geological EDA — production readiness (code rev gde.1-r3)

**Status: authored, NOT executed.** Every statement below is from static source inspection plus the Drive inventory
given in the r3 brief (2026-10-29). No test, notebook cell or data job was run while writing r3.

## Implementation ledger (for a fresh session)

| rev | what | where |
|---|---|---|
| r1 | EDA package, 99 figures, notebook | `gems_eda/discovery/*`, notebook |
| r2 | e02 / e03 mandatory products, contract audit, safe review widget, RECORDED w/h fix | `products.py`, `run.py`, `reset/atlas.py` |
| r3 | geometry precision `geom.2`; artifact registry; preflight; diagnostics + history; e04 hardening; DEPENDS fixes; engine-keyed e11 cache; notebook production path; per-target patcher | see "r3 changes" |

Next step for a new session: read this file, then `docs/GEMS_EDA_COLAB_RUNBOOK.md`; Colab results land in
`outputs/geological_discovery_eda/audit/preflight_*` and `manifests/history/stage_events.jsonl`.

## r3 changes (file -> change)

* `gems_eda/morphology.py` — `GEOMETRY_REV = "geom.2"`, `axial_deg` (axial [0, 180); 180 -> 0), `pca_axis`
  (eigenvalue / linearity deviations up to 1e-9 x lambda_max corrected to the bound, larger ones raise
  `GeometryPrecisionError`; degenerate covariance -> NaN), used by `path_geometry`; `component_diagnostics` dtype fix.
* `gems_eda/profiles.py` — station strike via `axial_deg`; typed empty station frame.
* `gems_eda/discovery/data.py` — lineaments use `pca_axis`; `InsufficientData`; empty-station error;
  `station_metrics` chunked (4000 stations, float32); `validate_profiles` (unique ids, fault + control present, cube /
  table shapes, finite samples) before e04 writes anything.
* `gems_eda/discovery/products.py` — rules return row masks; violations report count + sample rows; geometry
  provenance sidecar `tables/fault_geometry.provenance.json` (geometry_rev, tolerance, conventions).
* `gems_eda/discovery/artifacts.py` (new) — registry over `ExhaustiveEDA.OUTPUTS` (single contract) + raw inputs,
  consumers, requirement class, domain rules, NPZ grid shapes, reload hints; `validate()` shared by cache reuse and
  preflight.
* `gems_eda/discovery/diagnostics.py` (new) — failure categories, file:line, downstream impact, recovery hint,
  append-only `manifests/history/stage_events.jsonl`, resolved-error marking, `manifests/latest_stage_state.csv`,
  `current_errors()`.
* `gems_eda/discovery/preflight.py` (new) — `E.preflight()`: environment, raw-raster metadata contract (bounded
  reads), cached-product validation, manifest fingerprints, geometry precision of the cached table, frozen-engine /
  DEM prerequisites, propagated stage readiness; writes `audit/preflight_*.csv`, `audit/preflight_report.json`,
  `reports/PREFLIGHT_READINESS.md`.
* `gems_eda/discovery/run.py` — `DEPENDS`: e03 needs e02 (reads `domains.npz`), e11 needs e07 (reads lineaments);
  e11 contract includes `atlas/cache/honest_maps.npz`; e11 fingerprint keyed on the 03_scenarios fingerprint and a
  maps cache from another engine state is archived to `atlas/cache/history/` (not deleted); e12 uses W2 maps only
  when e11 is valid in this run; `check_outputs` applies registry rules + grid shapes; `record` override (history,
  diagnosis, `eda_code_rev` in manifests); `preflight()`, `resume()`, run summary (core / conditional / optional);
  `verify_checkout`, `prepare_eda`; stage 00 runs all four EDA test files; schema bumps e03 "3", e04 "2", e07 "3".
* `gems_eda/discovery/synthesis.py` — modeling decision FINAL only when every core stage is valid, else PROVISIONAL
  with the stages it could not use.
* `gems_eda/discovery/core.py` — `load_colab_config`. `configs/eda/colab_full.yaml` (new) — reference FULL_EDA config.
* Notebook — config `EXPECT['code_rev'] = 'gde.1-r3'`; new cells 2b (revision contract + `diag`), 3-4 preflight, 5
  run, 6 health, 7 recovery, step-by-step heading; title / one-shot / atlas markdown. Cells with saved outputs were
  NOT replaced (outputs preserved); the old review cell is safe because `review_widget` gates itself.
* `scripts/eda/patch_gems_eda_notebook.py` — per-target idempotent (no early return), rev read from the checkout,
  never modifies outputs.
* Tests — `tests/test_eda_geometry_regression.py`, `tests/test_eda_artifact_contracts.py`,
  `tests/test_eda_preflight.py` (new); `tests/test_geological_eda.py` fixtures updated + notebook / patcher tests.

## Stage readiness (static)

| stage | reads | produces (contract) | cache reuse | remaining risk |
|---|---|---|---|---|
| 00 tests | test files | selftests.csv, pytest_output.txt | always runs | 4 test modules never executed |
| e01 | 3 rasters | inventory, band metadata, prior audit | fp + outputs (schema 2, unchanged since r2: Drive copy reusable) | — |
| e02 | stack, e01 metadata | regions, domains.npz (grid shape), band_characterization (rules) | reusable from Drive if fp matches | — |
| e03 | net, domains | segments, fault_geometry (+ provenance) | schema 3: recomputes once (Drive table was produced with an in-memory fix) | first run of geom.2 code |
| e04 | stack, net, domains | controls / profiles / structure parquet, cube npz | schema 2: first execution | never ran on real data; RAM (~1-3 GB transient) |
| e05 | e04 | matched_control_tests | first execution | bootstrap runtime (~75 effects x 500) |
| e06 | e04, stack | scale tables, station values, crests_1km.npz | first execution | Hessian at 7 scales x 5 modalities: CPU time |
| e07 | e04, e06 | lineaments parquet / npz, complementarity | schema 3 | lineament count unknown (python loop per component) |
| e08 / e09 | e07 / e05, e06 | label bias / regional (typed-empty allowed) | first execution | — |
| 01 / 03 | factory | engine, validation units | base GeoMoE cache | ~4 GB Drive staging |
| e11 | engine, e04, e07 | honest maps npz, taxonomy, error cases | keyed on 03 fp | first execution with real maps |
| e12 | e07 (+ e11 maps if valid) | catalog, index, reviews (append-only) | schema 2 | panel rendering time (48 x 2) |
| e13 | cached DEM | dem_pilot.csv | disabled by default -> OPTIONAL_UNAVAILABLE | — |
| e14 / e15 | manifests / interpretations | integrity, synthesis | always run | decision PROVISIONAL if a core stage failed |

## Artifact inventory (from the brief's Drive inspection; not re-verified here)

* Required and present: the three raw rasters (sizes recorded), `tnm_urls.json`; e01 / e02 / e03 products and
  F001-F022 files.
* Stale cache: `tables/fault_geometry.csv` + `datasets/fault_segments.parquet` (in-memory corrected run, no geom.2
  provenance) -> e03 recomputes (cheap). Historical manifests e04-e12 BLOCKED_UPSTREAM -> preflight shows
  NEEDS_RECOMPUTE, never BLOCKED, once e02 / e03 are valid.
* Missing (never produced): every e04-e15 product.
* Conditional, unverified: frozen B0 / B1 / W2 artifacts (factory verifies at 01_attach), validation units.
* Optional, unavailable by default: DEM (disabled).

## Remaining limitations

Static review cannot guarantee: real-data RAM / runtime of e04-e12 (never executed), the Colab package versions, the
factory / G02 staging behaviour (4 GB), lineament counts, and that all four new test modules pass on first run.
Failures that remain will stop at the failing stage with category, file:line, affected output, downstream impact and
recovery step (`manifests/history/stage_events.jsonl`).
