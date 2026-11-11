# Troubleshooting

| message | meaning / fix |
|---|---|
| `AuditError: required source 'geodawn' not found` | Put the raw file in one of the `source_artifacts.geodawn_raster` candidate paths, or add its path. |
| `AuditError: … is not on the master grid` | The fault raster or template does not share the GeoDAWN transform. Do not resample the labels silently; fix the source. |
| `GridError: master grid mismatch` | The GeoDAWN raster differs from `geophysical_grid.expected_*`. Check that the file is the right one. |
| `DEMCatalogError: … without CRS` | A DEM tile cannot be located. Remove or repair it; it is never guessed. |
| Every DEM tile flagged `UNKNOWN_VERTICAL_UNIT` | There is no unit in the WKT or the XML sidecar. Add the sidecars, or set `dem.allow_unknown_vertical: true` (metres assumed, flagged). |
| `DEMWindowError: window … exceeds dem.max_window_mpix` | Use a coarser `resolution` for large windows (aggregation uses 2 m; E4 patches use `terrain_encoder.dem_res_m`). |
| `LeakageError` at splits | Read `04_splits/leakage_checks.parquet`. A label-derived predictor or an embargo violation stops training. |
| `StageGateError: no …-px patch lies entirely inside the … split` | Set `spatial_splits.block_km` to at least twice the patch footprint, or reduce `sampling.patch_px`. |
| `StageGateError: ranker needs positive and unlabeled candidates …` | Lower `candidate_generation` quantiles or check the split balance. The counts are in the message. |
| `StageGateError: test metrics require a prior validation-based selection` | Run `P.evaluate()` (validation) before `P.evaluate(final=True)`. |
| `OFFICIAL EVALUATOR UNAVAILABLE` | Expected: no official scorer is installed. The submission is still validated against the template; score it on the leaderboard manually. |
| `OSError: [Errno 107]` (Drive) | Remount (`drive.flush_and_unmount()`; `drive.mount(..., force_remount=True)`), re-run the setup cell, then the stage. |

## Recovery-first pipeline

| message | cause | fix |
|---|---|---|
| `feature dataset g11_features_v1 exists with another specification (...)` | A feature-relevant setting changed (DEM resolution, geophysics scales, sources). | Use a new `feature_version` with `feature_policy="RESUME_MISSING"`; only the affected class of layers is computed. |
| §04 shows many INCOMPATIBLE layers with `legacy feature config UNVERIFIABLE` | The legacy commit's config could not be read from git (offline or shallow-clone fetch failed). | Re-run §02/§04 with network access. Or set `recovery.accept_unverified_legacy_config: true` after checking that the config is unchanged. |
| INCOMPATIBLE `content changed: finite fraction ...` | The file differs from what the legacy run recorded. | Nothing to do: the layer is recomputed in §05. The file is kept for inspection. |
| INCOMPLETE `size ... != header-implied` / `.partial` | The write was interrupted. | §05 recomputes it. The partial file is kept, and you may delete it manually once §06 passes. |
| `feature processing incomplete: DEM chunks remain` | `feature_time_budget_min` was reached, or the session stopped. | Re-run §05. Committed chunks are skipped. |
| `DEM chunk ...: sha256 differs from its completion sidecar` | The chunk file was corrupted on Drive. | Move that `.npz` and its `.json` away and re-run §05 to recompute that chunk. |
| `split v2 rejected by its gates` | No candidate assignment passed every gate. | Read `05_splits/v2/split_gates_FAILED.parquet`. Then raise `spatial_splits.v2.n_regions` / `n_candidates` (or relax a gate deliberately) and use a new `split_version`. |
| `split v2 exists with other parameters` | Split settings changed after v2 was completed. | Use a new `split_version`. Split versions are never overwritten. |
| `archived previous checkpoints of E2 -> archive/...` | The configuration of that experiment changed. | Informational. The old checkpoints were moved, not deleted. |
