"""E6 V2 - GeoStructureNet built from raw data only (clean-room pipeline).

One architecture, one configuration (``configs/e6_v2/e6_v2.yaml``), one Drive namespace
(``GEMS_2/e6_v2_from_raw/``). Nothing in this package imports another ``gems_bench`` / ``gems_eda``
module or reads an artifact produced outside its own namespace: the DTI metric, the spatial splits,
the feature extractors and the submission checks are re-implemented here.

Entry point::

    from gems_bench.e6_v2 import E6V2Pipeline, load_config
    pipe = E6V2Pipeline(load_config())
    pipe.preflight(); pipe.prepare_geophysics(); pipe.prepare_dem(); pipe.prepare_datasets()
    pipe.train(); pipe.evaluate(); pipe.infer_full_region(); pipe.export_submission()

Docs: ``docs/e6_v2/``.
"""
from .config import CODE_VERSION, load_config

__all__ = ["CODE_VERSION", "load_config", "E6V2Pipeline"]


def __getattr__(name):
    # the pipeline pulls in torch; keep ``import gems_bench.e6_v2`` light for the preflight
    if name == "E6V2Pipeline":
        from .pipeline import E6V2Pipeline

        return E6V2Pipeline
    raise AttributeError(name)
