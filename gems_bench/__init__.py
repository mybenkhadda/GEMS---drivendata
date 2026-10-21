"""Phase 2 fault-detection benchmark toolkit for the DrivenData GEMS challenge.

Modules
-------
config      paths, benchmark parameters, artifact manifest
persist     experiment runs: config/seed/runtime manifest, registry, skip-if-done
data        native 100 m grid, labels, valid footprint, submission template
features    native-resolution feature engineering, on-disk cache, feature registry, groups G1-G4
folds       geographic block folds with buffers, pseudo-discovery fault-system holdouts
metric      exact distance-weighted Tversky index (DTI) + diagnostics
sampling    PU-aware pixel sampling for tabular models
gbm         LightGBM spatial-CV training, OOF maps, importance, SHAP (pred_contrib)
cnn         patch dataset, U-Net (smp or built-in), losses, tiled inference
segments    native-grid segment-ID raster + Phase-1 segment attributes
geo_eval    per-segment recovery, stratified recall, candidates, error atlas
discovery   pseudo-discovery evaluation of withheld fault systems
ensemble    complementarity and OOF-tuned ensembles
submission  full-region inference export + sanity checks
report      benchmark summary tables, figures and BENCHMARK_FINDINGS.md

Phase 3: subpackage ``p3`` (calibration, matched budgets, discovery scoring, CNN-3, ensembles).
Phase 4:
rs_features          RS / geomorphometric / structural feature bank (configs/phase4/rs_features.yaml)
phase4_context       outputs/phase4 layout, Phase-2 + RS feature store, model-zoo lookup across phases
phase4_eval          scorecard (geo / discovery / thin structures), complementarity, promotion, figures
geological_encoders  single / dual (concat, gated) / context / aux-head U-Nets, masked autoencoder
phase4_training      RS LightGBM runs, DL00-DL02 CNN controls, train4 / run_cnn4_cv / discovery
geophysical_ssl      masked + cross-modal pretraining, strict vs transductive protocols
external_eo          1 m DEM / Sentinel-2 / foundation-model interfaces (no downloads)

Phase 5 (GEMS 2.0, installable as ``gems-fault-discovery``):
paths / artifacts    workspace + Drive layout, atomic writes, Drive sync, model registry
experiment           Experiment.from_yaml(...): restartable prepare / train / evaluate / export, one runner per kind
phase5_context       Phase-4 context + Phase-5 feature store (profiles, DEM) + Phase-5 model zoo
geology              axial strike algebra, label-free orientation fields, trace topology
features.profiles    fault-normal structural profiles (Experiment D)
losses               positive-unlabeled (B), geometry multi-task (C), mean-teacher consistency (B3)
models               GeoStructureNet (E), model factory, catalogue of baselines
training             P5Spec, resumable trainer, CV / discovery drivers, GPU auto-config, checkpoints
evaluation           protocol hash, metric parity (A2), spatial designs (A3), scoreboard + promotion (A4)
discovery            discovery-aware ensembling (G), fault-recovery atlas
dem                  1 m DEM inventory, pilot, download, morphology, controlled benchmark (F)
inference            multi-head tiled inference, full-region maps, submission export
calibration          area-budget export policy, known-fault policies
visualization        figures regenerable from saved tables
runtime              Colab / Drive session helpers
cli                  ``gems`` command
"""
from ._version import __version__
from .config import BenchConfig, BenchPaths
from .data import NativeData, load_native

__all__ = ["__version__", "BenchConfig", "BenchPaths", "NativeData", "load_native"]
