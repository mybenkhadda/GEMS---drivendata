"""Geological EDA toolkit for the DrivenData GEMS fault-discovery competition.

Modules
-------
config       paths, band/modality metadata, tunable parameters, output manifest
io           GeoTIFF / GeoPackage I/O on one working grid
enhance      multi-scale derivatives, tilt, Hessian ridges, curvature, wavelets, structure tensor
morphology   fault skeleton graph, segment geometry, label-suitability diagnostics
profiles     fault-perpendicular profiles vs matched background, calibrated evidence
crossmodal   shared orientations, alignment, edge coincidence, complementarity
archetypes   rule-based fault archetypes and example picking
candidates   unsupervised multi-modal lineament detector and candidate ranking
plotting     map primitives with equal ground aspect and shared zoom windows
context      shared state (data, network, signals, windows) + persistence helpers
sections     per-section analyses and figures used by the notebook
findings     EDA_FINDINGS.md writer
"""
from .config import EDAConfig, EDAPaths, Manifest
from .context import EDAContext, add_derived_signals, define_windows, load_context

__all__ = ["EDAConfig", "EDAPaths", "Manifest", "EDAContext", "load_context", "add_derived_signals", "define_windows"]
