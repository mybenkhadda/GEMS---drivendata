# G06 architecture

## Modules (`gems_bench/g06/`)

| module | role | main public API |
|---|---|---|
| `config` | YAML + notebook merge, validation, identities | `load_config`, `validate_config`, `canonical_config`, `compute_run_id`, `stage_config` |
| `contracts` | dataclasses + fail-loud validators | `GridSpec`, `BandSpec`, `FeatureSpec`, `FeatureGroup`, `RasterArtifact`, `SplitSpec`, `DatasetSpec`, `ModelSpec`, `ExperimentResult`, `validate_*`, `assert_label_free_columns` |
| `inventory` | source inspection, artifact discovery, frozen-engine adapter | `inspect_source_raster`, `discover_existing_artifacts`, `inspect_feature_coverage`, `validate_artifact_compatibility`, `build_dependency_report`, `attach_engine`, `Engine` |
| `raster_io` | exact bands, halo windows, tile cache, writers, point reads | `resolve_exact_band`, `iter_windows`, `read_aligned_window`, `write_feature_raster`, `GroupWriter`, `TileCache`, `verify_raster_alignment`, `compare_tile_seams`, `read_points` |
| `feature_raw` | physical vs model-input raw channels, band statistics | `build_raw_feature_specs`, `transform_raw_window`, `validate_raw_channels`, `band_stats`, `footprint_of` |
| `feature_physics` | masked multiscale potential-field operators, DoG energies, FFT group | `magnetic_features`, `gravity_features`, `multiscale_gradients`, `potential_field_boundaries`, `frequency_features`, `spectral_features` |
| `feature_texture` | masked window statistics | `local_statistics`, `local_entropy`, `orientation_histograms`, `edge_density`, `texture_feature_bank` |
| `feature_structure` | structure tensor, ridges, continuity, cross-modal, discontinuities | `structure_tensor_features`, `directional_coherence`, `ridge_features`, `cross_modal_orientation`, `discontinuity_features`, `structural_feature_bank` |
| `feature_context` | regional context, domains, permitted known-fault context | `regional_context_features`, `permitted_fault_context`, `geological_domain_features`, `context_feature_bank` |
| `feature_store` | build / open / read / validate the store | `build_feature_store`, `open_feature_store`, `load_features`, `validate_feature_store`, `FeatureStore` |
| `splits` | validation units, visibilities, isolation, inner folds | `load_geographic_folds`, `load_discovery_splits`, `build_validation_units`, `build_training_visibility`, `verify_fold_isolation`, `make_inner_folds` |
| `supervision` | PU classes, segment / connection / orientation targets | `build_pixel_supervision`, `build_segment_supervision`, `build_connection_supervision`, `build_orientation_targets`, `validate_label_visibility` |
| `sampling` | leakage-aware pool + weights | `sample_positive_pixels`, `sample_background_pixels`, `sample_hard_negatives`, `sample_spatial_blocks`, `build_pool`, `compute_sample_weights` |
| `datasets` | D1-D4 builders and loader | `build_pixel_pool`, `label_pool_for_fit`, `build_patch_index`, `build_segment_dataset`, `build_connection_dataset`, `load_dataset` |
| `models_pixel` | RF / LightGBM / XGBoost cross-fitting | `train_random_forest`, `train_lightgbm`, `train_xgboost`, `crossfit_pixel_model`, `predict_pixel_raster`, `PixelModel` |
| `models_segment` | segment classifiers / ranker | `train_segment_classifier`, `train_segment_ranker`, `crossfit_segment_model`, `segment_to_raster` |
| `models_patch` | compact U-Net | `create_patch_dataset`, `build_small_unet`, `train_patch_model`, `predict_patch_model`, `reconstruct_patch_predictions` |
| `models_meta` | complementarity with W2 | `build_oof_prediction_matrix`, `train_regularized_meta_model`, `fit_residual_ranker`, `crossfit_meta`, `evaluate_model_complementarity` |
| `evaluation` | canonical scoring and decisions | `evaluate_prediction_grid`, `aggregate_components`, `compare_with_w2`, `system_recovery_report`, `nested_policy_selection`, `promotion_decision` |
| `inference` | full-footprint research rasters, parity, gated export | `predict_full_raster`, `predict_full_segments`, `ensemble_full_predictions`, `write_research_geotiff`, `verify_inference_parity`, `submission_gate` |
| `visualization` | figures (PNG + SVG + source data) | `plot_feature_atlas`, `plot_feature_correlations`, `plot_fault_profiles`, `plot_class_distributions`, `plot_feature_importance`, `plot_model_comparison`, `plot_prediction_overlay`, `plot_failure_cases`, `plot_dti_frontier`, `plot_geographic_robustness` |
| `persistence` | manifests, cache validation, archiving, Drive workspace | `artifact_fingerprint`, `atomic_write`, `save_stage_manifest`, `validate_cached_stage`, `load_cached_stage`, `write_artifact_inventory`, `Workspace` |
| `pipeline` | orchestration + reports | `G06Pipeline` |

**Adjustments to the proposed interfaces** (documented, the notebook uses the real ones):
`G06Pipeline.__init__(config, run_flags, repository_commit, factory=None, progress=print)`; the brief's
`build_datasets` = `build_pixel_dataset` + `build_segment_dataset`; `prepare_features` = `raw_eda` +
`build_feature_store` + `feature_atlas`; `reconstruct_full_region` = `full_inference` + `export_submission`;
`build_pixel_dataset` is split into `build_pixel_pool` (features once) and `label_pool_for_fit` (labels per
visibility); `load_oof` / `oof_path` take the experiment fingerprint (predictions are never overwritten).
The pipeline adds a `tests` stage (pytest in Colab). Reports are written by `pipeline.write_reports`.

## Reused GEMS components

`gems3.official` (Unit, FastScorer, units), `m7.protocol` (eligible, protocol keys, score_row, assert_comparable),
`gems3.representation` (rank01, top_mask, apply), `geomoe.scenarios` (V1 / V2 / V3 units, system_recovery),
`geomoe.candidates` (split_long, connectivity, near_share), `p6.segment_recovery` (dedup_inventory,
label_geometry), `p6.stack` (permitted_known, known_features, training_masks, sample_weights), `p6.boost` via the
engine (honest W2 maps, CNN maps), `leaderboard_focus.factory.Factory` (frozen-engine attach),
`m8.potential_fields` (FFT operators), `g05_1.bands` (exact band resolution), `artifacts.atomic` / `artifacts.sync`,
`m7.ranking._zone_mask`, `postprocess.topology` (skeleton, disk).

## Stages and caching

See the docstring of `pipeline.py` for the stage graph. Each stage fingerprint = G06 version + stage + the config
section it reads (`config.stage_config`) + upstream fingerprints + sha256 of the source of the modules it runs.
A stage is reused only when its manifest is COMPLETE, the fingerprint matches and every required output is readable
with its columns. A stage whose RUN flag is off adopts a valid cache or is NOT_REQUESTED. A successful output that is
recomputed with another fingerprint is moved to `history/<stage>/<fp10>/` first; model and prediction directories
are keyed by their own fingerprint. The run directory is keyed by the data-engineering identity only, so training
changes never rebuild the feature store.
