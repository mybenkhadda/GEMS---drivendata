"""Typed stage results, documented table schemas and the explicit failure types.

Every public ``RecoveryRun`` stage returns a :class:`StageResult`. The attribute used by the notebook
(``.table``, ``.summary``, ``.unit_table``, ``.components``, ...) is an entry of ``tables`` and is listed in
:data:`STAGE_CONTRACTS`; ``stats`` carries the scalar findings used by the generated interpretations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


class RecoveryError(RuntimeError):
    pass


class PreflightError(RecoveryError):
    """A mandatory artifact is missing or inconsistent - stop before any analysis."""


class SplitProvenanceError(RecoveryError):
    """A reconstructed V1 training mask does not match the checkpoint's train_mask_sha."""


class StageBlocked(RecoveryError):
    """A prerequisite stage has not produced its verified outputs."""


class MetricConsistencyError(RecoveryError):
    """Pixel-level component maps disagree with gems_bench.metric."""


@dataclass
class StageResult:
    stage: str
    status: str
    tables: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    outputs: list = field(default_factory=list)
    limitations: list = field(default_factory=list)
    path: Path | None = None
    cached: bool = False

    def __getattr__(self, name):
        # only called when normal attribute lookup fails
        tables = self.__dict__.get("tables", {})
        if name in tables:
            return tables[name]
        stats = self.__dict__.get("stats", {})
        if name in stats:
            return stats[name]
        raise AttributeError(f"{self.__dict__.get('stage', '?')} result has no '{name}' "
                             f"(tables: {sorted(tables)}, stats: {sorted(stats)})")

    def summary_dict(self) -> dict:
        return {"stage": self.stage, "status": self.status, "cached": self.cached,
                "tables": {k: int(len(v)) for k, v in self.tables.items() if isinstance(v, pd.DataFrame)},
                "stats": self.stats, "outputs": [str(o) for o in self.outputs], "limitations": self.limitations}


# notebook attribute -> table key, per stage
STAGE_CONTRACTS = {
    "preflight": {"attr": "table", "files": ["00_inventory/source_inventory.csv", "00_inventory/source_fingerprints.json",
                                              "00_inventory/preflight.json"]},
    "baseline_diagnostics": {"attr": "summary", "files": ["08_tables/baseline_summary.csv",
                                                          "08_tables/baseline_thresholds.csv",
                                                          "08_tables/baseline_lorenz.csv"]},
    "known_fault_proximity_diagnostics": {"attr": "table", "files": ["08_tables/proximity_bands.csv",
                                                                     "08_tables/proximity_cumulative.csv",
                                                                     "08_tables/high_confidence_far_components.csv"]},
    "audit_original_splits": {"attr": "unit_table", "files": ["01_split_audit/original_masks_manifest.json",
                                                              "01_split_audit/checkpoint_split_validation.csv",
                                                              "01_split_audit/split_overlap.csv"]},
    "recompute_original_metrics": {"attr": "components", "files": ["03_validation/original_metrics.csv"]},
    "ensure_full_area_predictions": {"attr": "artifact_table", "files": ["02_full_inference/*_full_prediction.tif"]},
    "score_full_area_holdouts": {"attr": "components", "files": ["03_validation/full_area_metrics.csv",
                                                                 "03_validation/pooled_metrics.csv"]},
    "spatial_error_atlas": {"attr": "region_table", "files": ["03_validation/regional_metrics.csv",
                                                              "03_validation/tp_fp_fn_spatial/*.tif"]},
    "expert_diagnostics": {"attr": "summary", "files": ["04_expert_analysis/expert_similarity.csv",
                                                        "04_expert_analysis/expert_diversity.csv",
                                                        "04_expert_analysis/routing_statistics.csv"]},
    "geological_disagreement_diagnostics": {"attr": "summary", "files": ["05_geological_evidence/geological_disagreement.csv"]},
    "structural_stratification": {"attr": "table", "files": ["03_validation/fault_length_recall.csv"]},
    "candidate_lineament_review": {"attr": "top_candidates", "files": ["05_geological_evidence/candidate_model_agreement.csv"]},
    "evaluate_fixed_transform_shortlist": {"attr": "components_by_fold", "files": ["06_postprocessing/transform_components.csv"]},
    "crossfit_recovery_decision": {"attr": "decision_table", "files": ["06_postprocessing/crossfit_results.csv",
                                                                       "06_postprocessing/recovery_decision.json"]},
    "export_guarded_submission": {"attr": "validation_table", "files": ["10_submission/submission_validation.json"]},
    "build_final_report": {"attr": "summary", "files": ["09_reports/executive_summary.md", "09_reports/scientific_findings.md",
                                                       "09_reports/limitations.md", "09_reports/artifact_manifest.json"]},
}

_DTI = ["dti", "tp_w", "fp_w", "fn_w", "n_truth", "pred_mass", "recall_w", "precision_w"]

# Required columns (a subset; extra columns are allowed and documented in docs/georegime_recovery/ARTIFACT_REGISTRY.md)
TABLE_SCHEMAS = {
    "source_inventory": ["role", "key", "path", "required", "exists", "size", "sha256", "status", "detail"],
    "baseline_summary": ["quantity", "value", "unit", "note"],
    "baseline_thresholds": ["threshold", "eligible_fraction_above", "mass_fraction_above", "n_pixels_above"],
    "baseline_lorenz": ["top_pixel_fraction", "mass_fraction"],
    "proximity_bands": ["band", "d_min_m", "d_max_m", "area_share", "mass_share", "mean_confidence", "enrichment",
                        "high_conf_share"],
    "proximity_cumulative": ["distance_m", "cumulative_mass_share", "cumulative_area_share"],
    "high_confidence_far_components": ["component_id", "n_px", "length_km", "linearity", "strike_deg", "mean_p",
                                       "max_p", "min_distance_to_mapped_m", "centroid_e", "centroid_n"],
    "checkpoint_split_validation": ["unit", "protocol", "train_mask_sha_reconstructed", "train_mask_sha_checkpoint",
                                    "match", "train_px", "zone_px", "truth_px", "exclude_px"],
    "split_overlap": ["unit_a", "unit_b", "quantity", "pixels"],
    "holdout_fragmentation": ["unit", "component", "n_px", "min_dist_to_train_visible_m", "length_km"],
    "original_metrics": ["unit", "protocol", "source", *_DTI],
    "full_area_metrics": ["unit", "protocol", "exclusion", *_DTI],
    "pooled_metrics": ["protocol", "exclusion", "n_units", *_DTI],
    "inference_artifacts": ["unit", "status", "identity", "pixels", "path", "reproduction_max_abs_diff"],
    "regional_metrics": ["unit", "region", *_DTI],
    "block_metrics": ["unit", "block", "row0", "col0", "tp_w", "fp_w", "fn_w", "n_truth"],
    "fault_length_recall": ["unit", "component", "length_km", "length_class", "linearity", "strike_deg", "recall_w"],
    "expert_similarity": ["unit", "head_a", "head_b", "spearman", "pearson", "jaccard_top"],
    "expert_diversity": ["unit", "head", "mean", "median", "q99", "mass"],
    "routing_statistics": ["unit", "quantity", "value"],
    "geological_disagreement": ["unit", "region_id", "kind", "n_px", "mean_p", "flags"],
    "candidate_model_agreement": ["candidate_id", "length_km", "label_free_support", "model_mean_p", "model_pct_rank",
                                  "agreement_class"],
    "transform_components": ["policy", "unit", "protocol", *_DTI],
    "crossfit_results": ["fold", "chosen_policy", "dti_other_folds", "dti_heldout_chosen", "dti_heldout_identity",
                         "delta_dti"],
}


def check_schema(name: str, df: pd.DataFrame) -> None:
    need = TABLE_SCHEMAS.get(name)
    if need is None:
        return
    miss = [c for c in need if c not in df.columns]
    if miss and len(df):
        raise RecoveryError(f"table {name} lacks documented columns {miss}")
