# Artifact registry

Paths relative to `GEMS_GEOLOGICAL_DATA_ATLAS_V1/`. Fingerprints = `stable_hash` of inputs + config section + the
source code of the producing modules (`config.module_hash`). Stage records: `07_cache/manifests/<stage>.json`
(fingerprint, outputs with sizes, status COMPLETE). `AtlasPipeline.artifact_report()` re-validates every record
(MISSING / STALE / INCOMPLETE / COMPLETE) into `08_reports/stage_status.csv` and lists every file in
`08_reports/artifact_inventory.csv`.

| artifact | producer | consumer | fingerprint | completion / validation | recovery |
|---|---|---|---|---|---|
| `01_inventory/manifests/discovered_sources.csv` | `discover` | preflight, snapshot | rewritten | file | re-run 01 |
| `01_inventory/manifests/preflight.csv` | `preflight` | user | rewritten from runtime files | `PreflightError` if required raw is missing | fix sources, re-run 01 |
| `00_raw_snapshot/**` | `snapshot` | every later stage | SHA-256 per file | ledger VERIFIED + size | re-run 01 (resumes) |
| `01_inventory/{raw_inventory,provenance,source_checksums}.csv`, `manifests/snapshot_report.json` | `snapshot` | gates, `raw_fp` | - | `require_snapshot` | re-run 01 |
| `02_geospatial/grids/analysis_grid.json`, `masks/*.tif`, `03_features/raw_bands/raw_band_registry.*` | `geospatial` | all | raw_fp + raw/grid config + code | stage `geospatial` | re-run 02 |
| `01_inventory/dem_inventory.csv`, `02_geospatial/coverage/dem_{coverage_catalog_tiles,overlap_count,project_id,seam_mask}.tif`, `masks/multimodal_coverage.tif`, `footprints/*` | `dem_inventory` | DEM, figures, acquisition | raw_fp + dem_fp + dem config | stage `dem_inventory` | re-run 02 |
| `07_cache/download_state/{download_plan,download_results}.csv`, `downloads.jsonl` | `acquire` | dem_inventory | ledger | VERIFIED rows | re-run 02 (HTTP Range resume) |
| `03_features/raw_bands/*.tif` | `raw_bands` | registry, gravity aliases | raw_fp + code | stage `raw_bands` | re-run 03 |
| `03_features/magnetic/*.tif`, `04_eda/tables/magnetic_*` | `magnetic` | cross-modal, EDA | raw_fp + potential/evidence config + code | stage `magnetic` | re-run 05 |
| `03_features/gravity/*.tif`, `04_eda/tables/gravity_*` | `gravity` | cross-modal, EDA | same | stage `gravity` | re-run 06 |
| `03_features/terrain_100m/*.tif` | `terrain` | cross-modal, EDA | raw_fp + terrain config + code | stage `terrain` | re-run 07 |
| `07_cache/dem_windows/FAST_10M/tiles/*_10m.tif` | `dem_mosaic.build_tile_cache` | windows | deterministic, built once | file exists | delete the file, re-run 08 |
| `07_cache/dem_windows/<MODE>/<id>.{npz,json}` | `dem._process` | assembly | mode fp + window + source list | json written after npz, fingerprint | re-run 08 (resumes) |
| `03_features/dem_multiscale/<MODE>/dem_*.tif`, `02_geospatial/coverage/dem_{coverage,support,seam_fraction}_<MODE>.tif`, `dem_mosaic/*` | `dem.assemble` | cross-modal, EDA | dem_fp + mode fp | stage `dem_<MODE>` | re-run 08 |
| `07_cache/dem_windows/<MODE>/examples/*.tif` | `dem_examples` | figure 10 | window | file | re-run 08 |
| `03_features/cross_modal/xm_*.tif` | `cross_modal` | lineaments, EDA | upstream stage fps + evidence config | stage `cross_modal` | re-run 10 |
| `04_eda/tables/mapped_fault_*`, `geological_evidence/{mapped_fault_skeleton,distance_to_mapped_fault_m}.tif` | `faults` | proximity, profiles, lineaments | raw_fp + code | stage `faults` | re-run 04 |
| `04_eda/tables/feature_by_fault_distance_band.*`, `feature_response_vs_distance.*` | `proximity` | reports, figure 14 | feature ids + proximity config | stage `proximity` | re-run 11 |
| `04_eda/profiles/*` | `profiles` | figure 13 | faults fp + registry | stage `profiles` | re-run 11 |
| `04_eda/tables/{feature_quality,scale_consistency,feature_spearman_matrix,strike_by_modality}.*` | `quality`, `figures.strike_roses` | ML-ready, reports | registry fingerprints + analytics config | stage `quality` | re-run 12 / 09 |
| `04_eda/candidate_lineaments/*` | `lineaments` | user, reload API | cross_modal fp + rules | stage `lineaments` | re-run 13 |
| `06_ml_ready/{feature_stack,coverage_masks,normalization,schemas}/*` | `ml_ready` | later modelling | retained ids + registry | stage `ml_ready` | re-run 14 |
| `03_features/feature_registry/feature_registry.{parquet,csv}` | every feature stage | everything | per-row fingerprint | row status | re-run the row's stage |
| `05_figures/**` | `figures.*` | user | regenerated from saved products | files | re-run the figure cell |
| `08_reports/*` | `write_reports`, `artifact_report` | user | rewritten | files | re-run 15-16 |
| `04_eda/reports/results.json` | every stage | reports | rewritten | file | re-run the stages |
