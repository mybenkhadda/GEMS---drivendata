"""GEMS M9 research experiments (isolated from the M9 production stages; read-only on M9 caches and artifacts).

G01: physics-informed wavelet fault discovery (notebooks/colab/GEMS_G01_GEOLOGICAL_WAVELET_EXPERIMENT.ipynb,
docs/m9/G01_GEOLOGICAL_WAVELET.md, configs/m9/g01_wavelet_geology.yaml).
G02: displacement, truncation and structural continuity (notebooks/colab/GEMS_G02_STRUCTURAL_GEOLOGY_EXPERIMENT.ipynb,
docs/m9/G02_STRUCTURAL_GEOLOGY.md, configs/m9/g02_structural_geology.yaml).

    from gems_bench.m9.research import G01Experiment, G02Experiment
    g01 = G01Experiment(m9, G01_CONFIG)        # m9 = pipe.m9(M9_CONFIG): nothing trained, no M9 section re-run
    g01.run()                                  # prepare -> evaluate -> exploratory -> metrics -> figures -> report
    g02 = G02Experiment(m9, G02_CONFIG)
    g02.run()                                  # prepare (+ G01 review) -> evaluate -> metrics -> figures -> report

Importing this package computes nothing.

Modules
-------
``geology_wavelet``      G01 label-free components and candidate maps G01-A..F (+ ablations)
``geology_metrics``      equal-area budgets, distance-tolerant counts, block bootstrap, nested choice, leakage guards
``geology_figures``      G01 figures (shared 16:9 map helpers)
``geology_io``           G01 configuration, M9 cache access + audit, fingerprinted store
``experiment``           ``G01Experiment`` orchestration
``g01_review``           read-only review of a persisted G01 run (ten questions + decision table)
``geology_displacement`` G02-A apparent offset of fabric across a line (rotated-frame NCC, step vs oblique slope)
``geology_truncation``   G02-B ridges, endpoints, endpoint alignment along a line
``geology_continuity``   G02-C multiscale ridge continuity (narrow / broad bands)
``geology_network``      G02-D lineament graph, segment attributes, continuation without forced connectivity
``geology_hypotheses``   G02-E fault / contact / intrusive / terrain / ambiguous support maps
``g02_metrics``          nested virtual candidates, contrasts, complementarity, strata, lineament geometry, decisions
``g02_figures``          the fifteen G02 figures
``g02_io``               G02 configuration (FAST_RESEARCH / FULL_EVALUATION), store, vectors
``g02_experiment``       ``G02Experiment`` orchestration
"""
from .experiment import G01Experiment
from .g02_experiment import G02Experiment

__all__ = ["G01Experiment", "G02Experiment"]
