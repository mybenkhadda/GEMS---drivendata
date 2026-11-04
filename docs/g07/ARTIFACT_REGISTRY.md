# G07 artifact registry

Root `outputs/g07_multiscale_fault_discovery/<run_id>/` (local `/content/gems_work/...`, mirrored to Drive).
`run_id = g07-<sha10(version, inputs, dem, g06, seed, source sha256)>`.

| path | producer | content |
|---|---|---|
| `manifests/run.json`, `runtime_errors.json`, `auxiliary_errors.json`, `execution_summary.csv`, `stages/<stage>.json`, `dependencies.json`, `artifact_inventory.csv` | all | identity, statuses, fingerprints |
| `audit/tests.csv`, `pytest_output.txt`, `gpu_runtime.json` | tests, gpu_preflight | synthetic tests in Colab, CUDA readiness |
| `dem_catalog/dem_invalid_sources.csv`, `dem_source_validation.json`, `training_gate.json` | DEM stages | excluded files with reasons, remedy, training decision |
| `tables/dem_seam_audit.csv`, `dem_features/dem_features_manifest.json`, `dem_features/coarse_parts/<fp>/` | dem_features | tile-boundary audit, fine / coarse progress, cached coarse parts |
| `datasets/surface/patch_readiness.csv`, `datasets/candidates/synthetic/<scenario>_<outer>/` | datasets, generate_candidates | READY / BLOCKED per U-Net fit, per-fit synthetic candidates |
| `models/<name>_registry.csv`, `<name>_validation.csv`, `<name>_experiment_status.csv`, `models/surface_unet/<fp>/<fit>.checkpoint.pt`, `<fit>.history.csv`, `training_history.csv` | training stages | per-fit status / device / strictness, completeness, resumable U-Net state |
| `evaluation/metrics/evaluation_status.csv`, `partial_unit_rows.csv`, `figures/training/unet_training_curves.*` | evaluate, U-Net | fully / partially evaluated, loss curves |
| `dependencies/existing_artifacts.csv`, `missing_dependencies.json`, `g06_store.json` | preflight, g06_restore | dependency report |
| `dem_catalog/dem_inventory.csv`, `source_licenses.csv`, `existing_products.csv`, `remote_headers.csv`, `download_log.csv`, `dem_coverage.geojson`, `feasibility_decision.json` | DEM stages | catalog / licences / decision |
| `tables/dem_coverage.csv`, `dem_fault_signal.csv`, `dem_feasibility_units.csv`, `dem_feasibility_cases.csv` | DEM coverage / feasibility | coverage, signal, cases |
| `dem_features/tiles/<fp>/fine_<tile>_<bytes>.tif (+json)`, `tiles/<fp>/coarse/<cfp>/coarse_*.tif`, `coarse/<cfp>/coarse_dem_10m.tif`, `pilot/dem_features_100m_pilot.tif`, `dem_features_100m.tif`, `dem_feature_registry.csv`, `dem_geo_agreement.tif` | feasibility / dem_features | resumable DEM products and the 100 m store |
| `splits/fits.csv`, `isolation_audit.csv` | datasets | visibilities + audit |
| `datasets/subsurface/pool_*.parquet`, `labels/<fit>.parquet`; `datasets/surface/...`, `surface/patches/{patch_index.parquet, surface_<h>.npy, surface_manifest.json}`; `datasets/candidates/<unit>(_labels).parquet`, `_lab.npz`, `nested_known/`; `datasets/meta/` | datasets / candidates / evaluate | reusable datasets |
| `models/<experiment>/<fp10>/<fit>/model.joblib, model.json, feature_importance.csv`; `models/surface_unet/<fp>/<fit>/`; `models/*_registry.csv`, `candidate_registry.json` | training stages | models |
| `predictions/oof/<experiment>/<fp10>/[nested/]<fit>__<unit>.npy (+json)`; `predictions/candidates/<exp>/<fp>/{scores, calibrated_scores, importance, portfolio/*.json, <map>/<unit>.npy}`; `predictions/full_region/{final_<policy>.tif, full_region_manifest.json, full_candidates.parquet}` | training / inference | held-out and research maps |
| `candidates/candidate_summary.csv`, `<unit>_provenance.csv` | generate_candidates | inventory + provenance |
| `evaluation/metrics/{metrics_rows, metrics, per_zone, exact_budget, system_recovery, model_comparison, complementarity, representation_selection}.csv`, `geographic/v3_summary.csv`, `recall/{candidate_recall, dem_candidate_recall}.csv`, `ablations/ablations.csv`, `promotion/{promotion_decisions.csv, decision.json, submission_eligibility.json}` | evaluate / export | results |
| `figures/{dem_atlas, dem_feasibility, structural_features, training, model_comparison, dti_frontiers, recovery_atlas, uncertainty}/` | several | PNG + SVG + source CSV / Parquet |
| `reports/{DEM_FEASIBILITY, TRAINING_RESULTS, COMPLEMENTARITY, DTI_EVALUATION, SUBMISSION_ELIGIBILITY, NEXT_EXPERIMENT}.md` | report | reports |
| `submissions/g07_<model>.tif (+json)` | export | only when every gate passes; never uploaded |
| `history/<stage>/<fp10>/`, `history/index.csv` | any | replaced successful outputs |
