# API (public functions; units in metres unless stated)

## Audit / grid / registry
- `audit.discover_artifacts(roots, patterns, max_depth)` → DataFrame with columns artifact_id, path, source_experiment, artifact_type, size_bytes, modification_time, extension, candidate_role. Unreadable directories are listed as INACCESSIBLE.
- `audit.inspect_raster(path)` → `RasterMetadata`: crs, epsg, bounds, transform, resolution, width, height, band_count, nodata, dtype, valid_fraction, compression, overviews, band_names.
- `audit.inspect_vector(path)` → `VectorMetadata`: geometry types, crs, feature count, bounds, invalid / empty / duplicate counts, schema.
- `audit.inspect_prediction(path, metadata)` → `PredictionMetadata`: shape, finite support, range, provenance, native_grid, split, and `alignment_confidence` (DOCUMENTED, TWIN_TIF or UNLOCATED).
- `audit.build_artifact_registry(cfg)` → (registry, summary). `audit.fail_fast(summary)` raises `AuditError`.
- `grid.load_master_grid(path, …)` → `GridSpec` (crs, transform, shape, bounds, res, valid).
- `grid.world_to_pixel`, `grid.pixel_to_world` (pixel centres; exact round trip), `grid.align_raster(source, target, kind)`. Kinds continuous / probability use bilinear; categorical / mask / label use nearest. Missing values become NaN.
- `grid.validate_alignment(a, b)` → `AlignmentReport`.
- `registry.RunContext`, `registry.StageStore` (fingerprinted stage records), `registry.atomic_write_*`.

## DEM
- `dem_catalog.build_dem_catalog(dem_root, master_crs, cfg)` → GeoDataFrame with footprint, resolution, valid fraction, year, source id, vertical unit and scale, quality flags, priority rank. `DEMIndex` is an STRtree over the footprints; `coverage_100m` gives overlap count, coverage and preferred source.
- `dem_reader.DEMReader.read_dem_window(bounds, target_crs, resolution, halo_m, resampling, snap)` → `DEMWindow`: elevation (m), transform, crs, valid mask, source_index, source_tiles, resolution, seam_mask, quality_flags, halo_px.
- `dem_reader.aggregate_dem_to_master(reader, grid, covered, cfg, store)`. Exact nested block statistics per 100 m cell (mean, median, min, max, std, p05, p95, relief, slope mean / p95, aspect circular mean, roughness, curvature, TRI, TPI, entropy, valid and seam fraction), plus context statistics at `dem.context_scales_m`.

## Faults / labels
- `fault_labels.load_fault_geometries`, `repair_fault_geometries` (adds length_m and azimuth_deg, axial 0–180), `build_fault_spatial_index`, `label_raster`, `rasterize_faults`, `segments_from_raster`.
- `distance_to_fault_m`, `orientation_raster` → axial (cos 2a, sin 2a), `label_states` (OUTSIDE / BACKGROUND / NEAR / FAULT / EXCLUDED), `make_targets` → Y_TRACE_100M, Y_BUFFER_<d>M, Y_DISTANCE, Y_ORIENTATION_*, and Y_CONFIDENCE only if the source has a confidence attribute.
- `segment_overlap_labels`.

## Splits / leakage
- `spatial_splits.make_splits(valid, faults, grid, cfg)` → `SplitResult` (code, blocks, embargo_m, groups). `embargo_width_m(cfg)` returns the width and its components.
- Checks: `check_disjoint`, `check_embargo`, `check_patch_leakage`, `check_segment_groups`, `check_predictor_names`, `check_oof_provenance`, `TrainOnlyNormalizer`, `run_leakage_checks`.

## Features
- `features.FeatureStore(root, grid)`: one float32 `.npy` memmap per feature plus a registry. `predictors()` excludes label-derived layers.
- `features.geophysics.compute_geophysical_features` / `compute_geophysical_to_store`. Per field and scale: gm, dx, dy, lap, hl1, hl2, ridge, valley, coh, sc2, ss2, aniso; plus persistence, wavelet energy and mag–grav axial agreement.
- `features.terrain.compute_terrain_features(z, valid, res, scales_m, cfg)`. Mask-aware first-order, second-order and structural morphology plus hillshade; edge-guarded.
- `features.structural.sample_cross_fault_profiles(segment, reader, offsets_m, spacing_m, crs, resolution)` → `ProfileSet`. `extract_cross_fault_features(profiles)` → elevation_step_m, slope_contrast, profile_asymmetry, scarp_width_m, scarp_strength, along_strike_consistency, cross_fault_variability, profile_valid_fraction.
- `features.multiscale.compute_multimodal_alignment`, `terrain_100m_summaries`, `structural_persistence`.
- `features.segment.segment_features(gdf, rasters, grid, cfg, dem_reader)`.

## Datasets / models / training
- `datasets.GeoPixelDataset.build(…)`, `GeoPatchDataset`, `MultiresolutionPatchDataset` (DEM branch with `modality[1]`), `FaultSegmentDataset.build(…)`.
- `models.xgb_baseline.train_pixel_model`, `predict_full_grid`; `models.segformer.SegFormer`, `load_compatible_weights`; `models.terrain_encoder.TerrainEncoder`; `models.gated_fusion.GatedFusionNet` (modes geo_only / dem_only / concat / gated / cross_attention); `models.segment_ranker.train_ranker` (binary / lambdarank / rank_xendcg).
- `training.losses`: weighted_bce, focal, soft_dice, boundary, positive_unlabeled, masked_multitask, build_loss. `training.trainer.Trainer`. `training.calibration.Calibrator` (stack split only). `training.checkpointing`. `training.sampling`.

## Inference / evaluation
- `inference.tiling.tiles`, `inference.blending.predict_tiled`, `inference.candidate_generation.generate_candidates` (threshold, hysteresis with 8-connectivity, ridge_nms, skeleton), `inference.topology.link_segments` / `deduplicate` / `stable_ids`, `inference.ranking.select_by_length_budget`, `inference.export.rasterize_segments`.
- `evaluation.pixel_metrics` (pr_auc, roc_auc, recall_at_area, ece, evaluate_region), `evaluation.segment_metrics` (candidate_metrics, budget_curve), `evaluation.spatial_bootstrap.block_bootstrap`.
- `evaluation.official_adapter`: `load_official_schema`, `validate_submission`, `export_submission`, `run_official_evaluator`, `proxy_dti`.
