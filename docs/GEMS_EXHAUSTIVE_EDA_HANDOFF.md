# GEMS exhaustive geological discovery EDA — handoff (code rev gde.1-r2)

**State:** gde.1-r1 ran once in Colab (commit c1cfd90): tests and inventory passed; e02 / e03 FAILED on missing
contracted tables and blocked everything downstream. gde.1-r2 repairs the producers (section "r2 recovery" at the
end). r2 is authored only: nothing in it has been executed (no pytest, no lint, no notebook run).

## How to run

1. Commit and push `gems_eda/discovery/`, `tests/test_geological_eda.py`,
   `notebooks/colab/GEMS_EXHAUSTIVE_GEOLOGICAL_EDA.ipynb` and this file. Pin `CONFIG['ref']` (and optionally
   `EXPECT['commit']`) to the pushed SHA.
2. Open the notebook in Colab, set `mode`:
   * `QUICK_LOOK` (start here): 600 fault stations, 100 bootstrap draws, scales 0.5 / 2 / 10 km.
   * `FULL_EDA`: 5000 stations, 500 draws, all scales 0.2–20 km.
   * `RESUME`: FULL_EDA settings, validated caches always reused.
3. Run top to bottom. Stage 00 runs the synthetic self-tests and `pytest tests/test_geological_eda.py`; a failure
   (`TESTS_FAILED`) blocks every downstream stage. A CPU runtime is enough: no stage trains or runs a network
   (the notebook metadata requests a GPU; switch the runtime to CPU if you want to save quota).
4. Outputs: `/content/drive/MyDrive/GEMS_2/outputs/geological_discovery_eda/` (synced to Drive after each stage).

## Architecture

`gems_eda.discovery.run.ExhaustiveEDA` subclasses `gems_bench.reset.run.FaultDiscoveryReset` (itself the GeoMoE
`DiscoveryGeoMoE`), so validated stage caching, dependency blocking (BLOCKED_UPSTREAM / NOT_EVALUATED), stage
integrity, Drive sync, the frozen B0 / B1 / W2 engine, honest zone-stitched maps and append-only reviews are reused
unchanged. Data stages read the three competition rasters at native 100 m through `gems_eda`
(`load_feature_stack(max_dim=None)`), never the engine.

| module | content |
|---|---|
| `core.py` | config + modes, Cliff's delta / AUC, cluster bootstrap over fault systems, status rule, regional robustness, axial statistics, figure `Registry` (PNG + SVG + data CSV + interpretation JSON + `FIGURE_CATALOG.csv`) |
| `data.py` | data loading, inventory, label-free domains, fault stations, matched controls, profiles, station / structure metrics, crests, label-free lineaments |
| `viz.py` | map panels in UTM km, scale bars, ECDF, effect bars |
| `parts_ab.py` | A landscape F001–F012, B geometry F013–F022 |
| `parts_cd.py` | C signatures F023–F036, D scales F037–F047 |
| `parts_efg.py` | E cross-modal F048–F059, F label bias F060–F069, G regional F070–F078 |
| `parts_hi.py` | H frozen-model forensics F079–F090, I candidate atlas F091–F098, DEM pilot F099 |
| `synthesis.py` | dashboards D1–D10, findings, hypothesis register, modeling decision, reports, executive summary |
| `selftest.py` | 20 synthetic self-tests (also run by pytest) |

Stages: `00_tests → e01_inventory → e02_landscape (A) → e03_geometry (B) → e04_profiles → e05_signatures (C) →
e06_scales (D) → e07_crossmodal (E) → e08_label_bias (F) → e09_regional (G) → 01_attach → 03_scenarios →
e11_forensics (H) → e12_candidates (I) → e13_dem (opt-in) → e14_integrity → e15_synthesis`. A figure that raises is
recorded in `manifests/figure_failures/<stage>.json`; the stage fails only when more than half of its figures fail.

## Statistical contract

* Matched controls: same 3 x 3 region and physical setting (terrain-slope x magnetic-gradient terciles), 2–10 km
  from mapped traces (sensitivity 1–10 and 5–20 km), falling back to region then global (recorded as
  `match_level`). Controls are *unlabelled*, not confirmed non-faults.
* Effects: Cliff's delta with a cluster bootstrap over fault systems (mapped-fault connected components; controls
  move with their fault). Region-level tests (lineaments) cluster over regions and say so.
* Status: SUPPORTED when the 95 % CI excludes ±`min_effect` (0.10); NOT_SUPPORTED when the CI lies inside it;
  otherwise INCONCLUSIVE (including fewer than 5 groups). DESCRIPTIVE figures make no test claim; NOT_AVAILABLE
  figures write no image and no placeholder data.
* Every interpretation sentence is assembled from the figure's own `Result` (status + numbers); none is hardcoded.
* Circularity: lineaments and the candidate atlas are built from label-free crests; mapping status is attached
  afterwards for analysis only. F077 selects the signature on the other regions and tests it on the held-out one.

## Model forensics (part H)

Uses `reset.atlas.honest_maps` (held-out V3 zone ranks of B0 / B1 / W2 + the frozen submitted W2 binary map). Each
mapped branch inside the evaluated zones is classed `detected` (≥ 50 % within 300 m), `localisation_failure`
(≥ 50 % within 2 km), `missed_ranked_high` (mean held-out W2 rank ≥ 0.9) or `detection_failure`; frozen-W2
components > 300 m from mapped traces are `near_fp_0.3_2km`, `far_fp_with_lineament` or `far_fp_no_evidence`.
F087 decomposes the official components per V3 zone (`E.score`) and per region (`m7.protocol.block_components`).
Outputs: `tables/model_failure_taxonomy.csv`, `datasets/model_error_cases.parquet`.

## Candidate atlas and reviews (part I)

Categories A–G (unmapped multi-family lineaments, unmapped single-family, far W2 predictions with lineament support,
missed mapped branches, suspected non-faults (terrain-only), short fragments, random controls), seeded, region
round-robin, ≥ 5 km apart (`reset.atlas.select`). Per candidate: `atlas/panels/<id>_evidence_only.png` (no mapped
faults, no model output) and `<id>_diagnostic.png`; `atlas/candidate_catalog.parquet`, `atlas/atlas_index.csv`,
`atlas/candidate_geometries.gpkg`, `atlas/review_template.csv`, `atlas/atlas_index.html`. Reviews go through the
widget, `E.record_review` or `E.import_reviews` into `atlas/reviewed_annotations.csv`, which is append-only and never
overwritten. Only EVIDENCE_ONLY reviews citing no mapped-fault / model source count as INDEPENDENT_REVIEWED.

## Modeling decision

`reports/MODELING_DECISION.md` answers 15 questions from the cited figures and derives one recommendation by fixed
rules (`synthesis.decision`): REPRESENTATION_AND_BUDGET (localisation + high-rank misses exceed pure detection
failures), NEW_EVIDENCE_FEATURES (missed branches carry evidence comparable to detected ones), LABEL_REVIEW_AND_PU
(unmapped lineaments resemble mapped ones), KEEP_W2_AND_REVIEW_CANDIDATES, or INSUFFICIENT_EVIDENCE (forensics not
run). Each comes with an invalidation criterion. The public W2 score (0.0942) is an external reference only, never a
tuning target; nothing is uploaded to DrivenData.

## Known limitations / first things to check in the Colab run

* The stage-00 pytest has never run; expect to fix small API mismatches on first execution (the self-test details
  in `audit/selftests.csv` name the failing function).
* Part H / I W2-derived content needs `01_attach` and `03_scenarios`; without the factory they are
  BLOCKED_UPSTREAM and categories C / D of the atlas are recorded as unavailable.
* The honest-map grid must equal the EDA grid; a mismatch raises (never resampled silently).
* DEM: `dem.enabled=False` by default; when enabled only a cached M7 mosaic / cached tiles are used, never a full
  download. Missing coverage is NOT_AVAILABLE, never evidence of absence.
* Crest thresholds (top 3 %), the 300 m / 2 km / rank 0.9 forensic cut-offs and the 5 km domain blocks are analysis
  choices recorded in each interpretation's limitations.

## r2 recovery (gde.1-r2)

### Root causes

| Colab symptom | Root cause | Fix |
|---|---|---|
| e02 FAILED, 12/12 figures, `tables/band_characterization.csv: missing` | contract declared, no writer anywhere (not renamed, not cache-suppressed) | `products.band_characterization` (core, before the figures) + `enrich_band_characterization` (after F005–F008), schema-validated, atomic |
| e03 FAILED, 10/10 figures, `tables/fault_geometry.csv: missing` | same: contract without writer | `products.fault_geometry` (core, before the figures) |
| e04–e12 BLOCKED_UPSTREAM, their tables absent | consequence of e02 / e03 (dependency machinery correct) | none needed; they become eligible once e02 / e03 are COMPLETE in the session |
| widget `AttributeError: ... candidate_id` | `review_widget` read `atlas_index.csv` blindly (absent because e12 was blocked) | `reset.atlas.review_readiness` gate inside `review_widget` + notebook gate |
| inventory `mismatches ['check_shape']` | `data.RECORDED` had width / height swapped (repo evidence `outputs/benchmark/tables/native_grid_summary.csv`: rows 3730, cols 3292) | constant fixed; `shape_detail` per file; new `check_same_grid_as_features` (a raster off the features grid FAILS e01) |
| FutureWarning `reset/atlas.py:167` | `fillna("")` on float / bool columns of a reloaded review CSV | `review_keys`: object dtype, `where(notna, "")`, then `str` (same keys) |
| FutureWarning `morphology.py:296` | `fillna(False)` on an object column after `reindex` | nullable `boolean` -> `fillna(False)` -> `bool` (missing = no loop, as before) |

Latent defects found during the audit (would have surfaced after unblocking):
* `tests/test_geological_eda.py::test_notebook_structure` asserted that the notebook has no outputs: it would FAIL
  stage 00 (blocking everything) once the Colab-saved notebook is committed. Now only code is checked.
* `parts_cd.scale_data` kept 1 km crests only when 1 km is in the mode's scale list (not in QUICK_LOOK): part E would
  have run on empty crests. 1 km crests are now always computed.
* Mandatory tables of e07 / e08 / e09 / e11 / e12 were written inside figure functions (a failed plot = missing core
  product). They are now stage producers (`run.CORE`), run before (or, for the e07 summary, after) the figures.
* Downstream caches read by later stages (`fault_profile_cube.npz`, `crests_1km.npz`, `scale_regional.csv`,
  `station_scale_values.parquet`, `lineaments.npz`, `domains.npz`) were not in the contract: a REUSED stage could
  lack them. They are now contracted and `.npz` files must load with their arrays.

### Stage contract audit (stage -> output -> producer -> consumers)

| stage | contracted output (required columns / arrays) | producer | consumers |
|---|---|---|---|
| e01 | audit/raw_data_inventory.csv, band_metadata.csv, previous_eda_audit.csv | `run.s_inventory` | e02 (units), reports |
| e02 | datasets/physical_regions.parquet (domain); rasters/spatial_domains/domains.npz (domain); **tables/band_characterization.csv** | `run.s_landscape`, `products` | e04 (domains), part A/G figures, FE register |
| e03 | datasets/fault_segments.parquet; **tables/fault_geometry.csv** | `run.s_geometry`, `products` | parts B, H, reports |
| e04 | matched_controls / fault_profiles / station_structure .parquet; fault_profile_cube.npz | `run.s_profiles` | e05–e11 |
| e05 | tables/matched_control_tests.csv | `parts_cd.effects` (core) | part G, synthesis |
| e06 | scale_localization.csv, scale_regional.csv, station_scale_values.parquet, crests_1km.npz | `parts_cd.scale_data` (core) | e07, part G |
| e07 | structural_lineaments.parquet, lineaments.npz, crossmodal_complementarity.csv | `parts_efg.lin_data` (core), `crossmodal_table` (post) | e08, e11, e12 |
| e08 | label_selection_bias.csv | `parts_efg.label_bias_table` (core) | synthesis |
| e09 | regional_robustness.csv (typed header even with 0 regions) | `parts_efg.lro_table` (core) | synthesis |
| e11 | model_failure_taxonomy.csv, model_error_cases.parquet | `parts_hi.failure_taxonomy` (core) | e12, synthesis |
| e12 | candidate_catalog.parquet, atlas_index.csv, reviewed_annotations.csv (created once, never overwritten) | `parts_hi.write_atlas` (core) | review widget, F096/F097 |
| e13–e15 | unchanged | | |

`test_every_contracted_output_has_a_producer` checks statically that every contracted path is written somewhere.

### New / corrected artifacts

`tables/band_characterization.csv` (one row per band; `products.BAND_COLUMNS`): band, band_index, modality,
provenance_class, units (from e01 metadata, else "not available"), description, is_primary_of_modality, is_derived,
parent_band, n_footprint_px, n_valid_px, valid_frac_footprint, p01–p99, mean, std, robust_scale_mad, iqr,
dynamic_range_p01_p99, robust_cv, local_regional_std_ratio (median std 1 km / 10 km on a 200 k pixel sample),
n_sample_ratio, corr_length_km (F005), anisotropy_ratio_2km / anisotropy_max_dir_deg (F006, primary fields),
short_wavelength_power_share (F007, computed at 200 m), spearman_vs_recomputed_from_parent (F008),
n_unique_sample, measured_in, limitations (rules on the measured values only). Unmeasured = NaN. Rules: unique band,
fractions in [0, 1], ordered quantiles, non-negative spread.

`tables/fault_geometry.csv` (one row per traced branch; `products.GEOMETRY_COLUMNS`): seg_id, system_id (connected
component of the mapped raster), length_km, chord_km, sinuosity, strike_deg (axial, clockwise from north, [0, 180)),
linearity, curv_mean / p90 (deg/km), n_px, nodes and degrees, n_junction_ends, n_free_ends, is_loop, touches_edge,
is_short_lt_2km, aligned_gap_partners (≤ 5 km, ≤ 25 deg, other systems), system_n_segments / length_km / kind,
region, region_name, domain, trace_density_km_per_100km2 (10 km kernel), r/c_mid, x/y_mid_utm, in_footprint,
crs. Rules: unique seg_id, length > 0, chord ≤ length, strike in [0, 180), linearity in [0, 1], system_id > 0.

### Cache lifecycle

* Fingerprint = code version (`gde.1`), mode, sampling, statistics, scales, **the stage's output contract and its
  `STAGE_SCHEMA` version**, stage parts. Bumping one stage's schema invalidates only that stage. r2 bumps e01, e02,
  e03, e06, e07, e08, e09, e11, e12; e04 / e05 caches (if any) stay valid. e02 / e03 never had a COMPLETE manifest.
* Reuse needs: same fingerprint, manifest COMPLETE / REUSED / SKIPPED, every contracted output present, non-empty,
  readable, with its columns / arrays; otherwise the stage recomputes (the earlier files are overwritten atomically
  only by a successful producer; nothing is deleted).
* Session caches (`E.D.cache`) of a stage are dropped when it recomputes (`SESSION_CACHE`).
* Every atlas run is also archived at `atlas/catalog_history/<fingerprint>.parquet`; reviews stay append-only.

### Colab recovery sequence

0. Locally (or in any checkout): `git pull` (brings the Colab-saved notebook b43f49e), then
   `python scripts/eda/patch_gems_eda_notebook.py --check` and `python scripts/eda/patch_gems_eda_notebook.py`
   (idempotent, keeps all outputs), commit + push; note the SHA.
1. Notebook config cell: set `CONFIG['ref']` (and optionally `EXPECT['commit']`) to that SHA; keep
   `mode='FULL_EDA'`, `reuse_cached=True`. Runtime -> Restart session, run the config + environment cells
   (asserts `gde.1-r2`).
2. `E.s00_tests()` -> must be COMPLETE; else read `audit/pytest_output.txt` (all tests are new and unexecuted).
3. `E.s_inventory()` -> check the note: `check_shape` should no longer be listed; `audit/raw_data_inventory.csv`
   has `shape_detail` and `check_same_grid_as_features` (must be True for faults and submission).
4. `E.s_landscape(); E.s_geometry(); diag(['e02_landscape', 'e03_geometry'])` -> both COMPLETE, then
   `show('tables/band_characterization.csv')`, `show('tables/fault_geometry.csv')`.
5. Continue cell by cell (each ends with `diag(...)`): profiles -> signatures -> scales -> cross-modal -> label bias
   -> regional. These stages have never run: expect first-run errors there; `diag` names the file / error.
6. Engine only now: `E.s01_attach(); E.s03_scenarios(); E.s_forensics()`; then `E.s_candidates()`.
7. Review cell: prints READY / SKIPPED with the reason; the widget opens only when READY.
8. `E.s_dem(); E.s_integrity(); E.s_synthesis()`, then the health-check cell (`diag(all_rows=False)`).
9. Full integration: a fresh session with `E.run()`; every stage should be COMPLETE or REUSED_VALID_CACHE.
