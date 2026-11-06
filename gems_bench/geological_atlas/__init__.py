"""GEMS Geological Data Atlas V1 - raw-data snapshot, geological feature engineering and fault-discovery EDA.

No model training, no submissions. Everything is written to ``<drive_root>/GEMS_GEOLOGICAL_DATA_ATLAS_V1/``
and, after the verified raw snapshot, computed only from ``00_raw_snapshot/`` and this atlas' own products.

    from gems_bench.geological_atlas import AtlasPipeline, GeologicalAtlas
    P = AtlasPipeline(CONFIG)            # notebook section 00
    atlas = GeologicalAtlas.open(P.paths.root)

Docs: ``docs/geological_atlas/``. Notebook: ``notebooks/colab/GEMS_GEOLOGICAL_DATA_ATLAS.ipynb``.
"""
from .atlas import FeatureSet, GeologicalAtlas
from .config import ATLAS_VERSION, DEFAULTS, AtlasConfig

__all__ = ["ATLAS_VERSION", "DEFAULTS", "AtlasConfig", "AtlasPipeline", "FeatureSet", "GeologicalAtlas"]


def __getattr__(name):
    if name == "AtlasPipeline":
        from .pipeline import AtlasPipeline

        return AtlasPipeline
    raise AttributeError(name)
