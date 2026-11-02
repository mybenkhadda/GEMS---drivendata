# G05.1 artifact registry

Root: `outputs/g05_1_system_recovery/<run_id>/` (local `/content/gems_work/...`, synced to
`MyDrive/GEMS_2/...`). `run_id = g051-<sha1(version, inputs, target_systems, geomoe_overrides, ref)[:10]>`.

## Every stage

`manifests/stages/<stage>.json` (GeoMoE manifest, fingerprint) and `stages/<stage>/{status, manifest,
config_snapshot, input_fingerprints, decision}.json`. Replaced successful outputs: `history/<stage>/<fp10>/`,
indexed in `history/index.csv`.

## Per stage

| stage | files |
|---|---|
| 00 | `audit/selftests.csv`, `audit/pytest_output.txt` |
| A | `preflight/dependency_inventory.csv`, `preflight/band_resolution.csv`, `preflight/g05_run_selection.json`, `preflight/g05_schema.csv` |
| 02–05 | GeoMoE: `audit/*`, `validation/scenarios.csv`, `candidates/units/<fp>/*`, `candidates/*.parquet`, `gating/grid_A_W2.csv`, `validation/frozen_w2_baseline.csv` |
| B | `atlas/failure_atlas.csv`, `atlas/target_systems.csv`, `atlas/panels/*.png`, `atlas/case_browser.html`, `atlas/ATLAS_SUMMARY.md`, `atlas/panel_index.json` |
| C | `recall/recall_rows.csv`, `recall/recall_summary.csv`, `recall/new_lost_vs_w2.csv`, `recall/leave_one_expert_out.csv`, `recall/recovery_matrix.parquet`, `recall/target_system_recovery.csv`, `recall/oracle_ceiling.csv`, `recall/metric_definition.json`, `recall/recall_by_construction.png` |
| D | `registry/fault_hypothesis_registry.parquet`, `registry/units/<fp>/*.parquet`, `registry/feature_dictionary.csv`, `registry/feature_provenance.csv`, `registry/feature_availability.csv`, `registry/hypothesis_geometry.{gpkg,parquet}` |
| E | `displacement/displacement_evidence.parquet`, `displacement/control_summary.csv`, `displacement/OFFSET_PROXY_NOTE.md` |
| F | `graph/graph_features.parquet`, `graph/edges.parquet`, `graph/graph_summary.csv` |
| G | `rankers/scores_<model>.parquet`, `rankers/model_registry.json`, `rankers/label_summary.csv`, `rankers/feature_lists.json` |
| H | `nesting/nesting_audit.csv`, `nesting/strictness.json`, `nesting/retraining_plan.csv` |
| I | `selection/grid.csv`, `selection/exact_frozen_cell.csv`, `selection/model_summary.csv`, `selection/per_unit.csv`, `selection/deployable_policies.json`, `selection/promotion_decisions.csv`, `selection/challenger_vs_g0.csv`, `selection/decision.json` |
| J | `ablation/ablation_rows.csv`, `ablation/ablation_summary.csv`, `ablation/feature_importance.csv`, `ablation/family_importance.csv`, `ablation/target_system_contributions.csv` |
| K | `reconstruction/research_score_<model>.tif`, `reconstruction/reconstruction_manifest.json`, `submissions/submission_manifest.json`, `submissions/validated_submission_candidate_*.tif` (+ `.json`, `_validation.json`) only if promoted |
| Y / Z | `audit/stage_integrity.csv`, `reports/G051_REPORT.md`, `reports/EXECUTION_STATUS.md`, `reports/DELIVERABLES.csv` |

## The 12 deliverables (`reports/DELIVERABLES.csv`)

1 dependency inventory · 2 failure atlas + browser · 3 equal-budget recall · 4 recovery matrix · 5 hypothesis
registry · 6 feature dictionary + provenance · 7 displacement + controls · 8 candidate graph · 9 cross-fitted rankers
· 10 strict-nesting audit · 11 selection + promotion decision · 12 reconstruction / submission manifest.
