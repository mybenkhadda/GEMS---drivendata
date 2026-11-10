"""G11 - multiresolution geological fault discovery (100 m geophysics x 1 m DEM x mapped faults x ML).

Entry points::

    from gems_bench.g11.config import load_config, Mode
    from gems_bench.g11.pipeline import G11Pipeline
    P = G11Pipeline(load_config(overrides={...}))
    P.run(Mode.AUDIT)

Docs: docs/g11/ (API, data paths, execution order, artifact registry, reproducibility).
"""
from .config import Mode, load_config

__all__ = ["Mode", "load_config"]
__version__ = "g11.1"
