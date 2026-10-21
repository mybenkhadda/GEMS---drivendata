"""Phase 3 — calibration, spatial generalisation and fault discovery.

Modules
-------
context      Phase3Paths (outputs/phase3), P3Context (data, folds, splits, segments, model zoo), manifest
curves       exact DTI of binary top-q masks for every threshold at once (+ generic cross-fit selection)
transforms   output-transform families (raw, linear, temperature, isotonic, power-floor, rank, soft/hard threshold)
audit        calibration / ensemble / mask integrity audit (static code findings + data checks)
budget       recall-area-DTI curves, matched-budget comparison, Pareto front, truth-density sensitivity
discovery3   corrected pseudo-discovery scoring (exact n_px, three known-fault policies, matched budgets)
lgbm3        LightGBM history diagnostics, controlled config grid, modality ablation
cnn3         compact U-Net benchmark: PU ignore band, DTI-surrogate loss, early stopping, discovery runs
lineaments   ridge/skeleton candidate lineaments, junction splitting, geometry, evidence, ranking, GPKG
ensemble3    nested ensemble validation and the decomposition of the Phase-2 rank-ensemble score
report3      figures, experiment manifests and PHASE3_FINDINGS.md with conditional decision rules

Theory note used throughout (see curves.py): for a fixed truth set the DTI is maximised by a
*binary* map, so a model is characterised by its ranking plus an area budget, not by its
probability calibration.
"""
from .context import P3Context, Phase3Paths, load_context

__all__ = ["P3Context", "Phase3Paths", "load_context"]
