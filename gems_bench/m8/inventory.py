"""Integration inventory: every existing function M8 reuses, where it lives, which M8 module consumes it, whether it
had to change, and the compatibility assumption it carries. Written to ``manifests/integration_inventory.csv`` by
section 00 and mirrored in docs/M8_PHYSICS_PROBABILISTIC_DISCOVERY.md."""
from __future__ import annotations

import pandas as pd

INVENTORY = [
    # (existing function, original file, M8 consumer, change, compatibility assumption)
    ("NativeData.read_band / read_window / valid / labels / template_valid / grid", "gems_bench/data/__init__.py",
     "layers, signal_recovery, profiles, montecarlo", "none", "EPSG:32611, 100 m, NaN nodata (|v| > 1e30 masked)"),
    ("gems_eda.enhance.fill_nan / nan_gaussian", "gems_eda/enhance.py", "potential_fields, multimodal", "none",
     "nearest fill before derivative filters"),
    ("rs_features.prank", "gems_bench/rs_features.py", "multimodal, completion, graph", "none",
     "label-free percentile against a fixed valid sample"),
    ("M7Research (units, elig, pkey, protocol_keys, transfer, load_unit_set, completion_sets)", "gems_bench/m7/run.py",
     "run (composition, read-only)", "none; its GEMS3 engine is re-pointed to outputs/m8.../gems3_engine",
     "M7 stage methods that write are never called by M8"),
    ("gems3.official.Unit / FastScorer / build_units / unit_score / _unit", "gems_bench/gems3/official.py",
     "every evaluation", "none", "official kernel R = 300 m, alpha 0.2, beta 0.8, pixel-exact known mask"),
    ("m7.protocol.eligible / protocol_key / score_row / system_hits / system_ids / unit_ledger / restrict_unit",
     "gems_bench/m7/protocol.py", "baseline, signal_recovery, candidates, montecarlo, rasterize, dem", "none",
     "eligible budget domain = scored & template-finite & not known & not ring"),
    ("m7.baseline.run / heldout_map / MemberMaps", "gems_bench/m7/baseline.py", "baseline, ensemble", "none",
     "nested selection on the other zones; reused only when protocol keys match"),
    ("m7.failure.run / system_attributes / case_window", "gems_bench/m7/failure.py", "baseline, figures", "none",
     "system = label pixels linked within 1 km (ctx.splits.system_map)"),
    ("m7.generator.response / _nms / gated_crests / mask_to_paths / fuse / clip_to_unit / stable_id / geometry / "
     "structural_score / read_source", "gems_bench/m7/generator.py", "candidates, worms, signal_recovery, montecarlo",
     "none", "1-px crest lines, provenance kept in sources / member_uids"),
    ("gems3.candidates.CandidateSet / tip_paths / to_full", "gems_bench/gems3/candidates.py", "all candidate code", "none",
     "paths in unit crop coordinates"),
    ("m7.coverage.density_subset / density_order / pooled / _ocfg", "gems_bench/m7/coverage.py",
     "candidates, signal_recovery, montecarlo", "none", "label-free structural_score order"),
    ("gems3.oracle.coverage / candidate_truth_stats", "gems_bench/gems3/oracle.py", "candidates, completion, dem",
     "none", "oracle reads truth: diagnostic only"),
    ("m7.network.segments / endpoint_table / gap_table / proposals", "gems_bench/m7/network.py", "completion", "none",
     "endpoint headings CCW from east (gems convention)"),
    ("m7.corruption.corrupt_fold / _rasters / metrics, gems3.map_completion.systems_by_fold",
     "gems_bench/m7/corruption.py", "completion (benchmark)", "none",
     "systems assigned to folds BEFORE corruption; completer sees the damaged map only"),
    ("m7.dem_pilot.run / download, m7.dem_morph.run / DemMosaic / mosaic_path, m7.dem_value.dem_candidates / _clip / _cap",
     "gems_bench/m7/dem_*.py", "dem", "none (writes go through the M8 view: outputs/m8.../dem)",
     "pilot tiles and mosaic band names as produced by M7"),
    ("m7.cand_features.describe / groups / label_free_score / neural_score", "gems_bench/m7/cand_features.py",
     "features, ranking", "none", "PU labels: 1 near truth, 0 far, -1 unlabelled (never negative)"),
    ("m7.ranking.loo_scores / unit_rows / full_importance / _model", "gems_bench/m7/ranking.py", "ranking, submission",
     "none", "leave-one-zone-out with buffer + linked-network exclusion"),
    ("m7.rasterize.score_raster / select_core / shape_core / save_heldout / load_heldout / saved_methods / predict",
     "gems_bench/m7/rasterize.py", "rasterize, ranking, completion, graph, ensemble", "none",
     "held-out maps stamped with protocol keys"),
    ("gems3.metric_opt.nested_select", "gems_bench/gems3/metric_opt.py", "every selection", "none",
     "choices for zone z made on the other zones"),
    ("gems3.representation.rank01 / fuse / apply / crest_map", "gems_bench/gems3/representation.py",
     "ensemble, submission, rasterize", "none", "per-zone percentile ranks"),
    ("m7.comparison.evaluate, m7.decision.verdicts", "gems_bench/m7/comparison.py, decision.py", "ensemble", "none",
     "champion = s01_fusion_thin; refuses mismatched protocol keys"),
    ("m7.export.export_baseline / _write / _provenance, gems3.submission.finalise / validate / eligible_mask / area_ledger",
     "gems_bench/m7/export.py, gems3/submission.py", "submission", "none",
     "template is the authority; NaN where template NaN; never overwrite"),
    ("m7.figures.map_panels / bar_panels16 / line_panels16 / hillshade", "gems_bench/m7/figures.py", "figures", "none",
     "true geometry, equal aspect"),
    ("M7Workspace (atomic writes, immutable tables, section manifests)", "gems_bench/m7/workspace.py", "workspace",
     "subclassed (M8 folders, version)", "fingerprinted stage reuse"),
]


def table() -> pd.DataFrame:
    return pd.DataFrame(INVENTORY, columns=["existing_function", "original_file", "m8_consumer", "change",
                                            "compatibility_assumption"])
