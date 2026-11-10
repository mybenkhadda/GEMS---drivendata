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
