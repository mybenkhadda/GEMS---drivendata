"""GEMS M9 research experiments (isolated from the M9 production stages; read-only on M9 caches and artifacts).

G01: physics-informed wavelet fault discovery (notebooks/colab/GEMS_G01_GEOLOGICAL_WAVELET_EXPERIMENT.ipynb,
docs/m9/G01_GEOLOGICAL_WAVELET.md, configs/m9/g01_wavelet_geology.yaml).

    from gems_bench.m9.research import G01Experiment
    g01 = G01Experiment(m9, G01_CONFIG)        # m9 = pipe.m9(M9_CONFIG): nothing trained, no M9 section re-run
    g01.run()                                  # prepare -> evaluate -> exploratory -> metrics -> figures -> report

Modules
-------
``geology_wavelet``   label-free components (percentiles, tolerant co-location, axial orientation gate, persistence)
                      and the six candidate maps G01-A..F (+ ablations)
``geology_metrics``   equal-area budgets, distance-tolerant precision / recall, enrichment, block bootstrap, nested
                      choice, geographic leakage guards
``geology_figures``   the twelve 16:9 figures
``geology_io``        configuration, M9 cache access + audit, fingerprinted store (manifest, resume, Drive push)
``experiment``        ``G01Experiment`` orchestration
"""
from .experiment import G01Experiment

__all__ = ["G01Experiment"]
