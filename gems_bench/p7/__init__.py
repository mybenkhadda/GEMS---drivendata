"""P7 GeoFaultFormer: CNN + local Transformer + ConvGRU + structural heads + endpoint tracer (code-authoring-first).

Entry point ``P7GeoFaultFormer(p6, config)`` (``run.py``); docs ``docs/p7/P7_GEOFAULTFORMER_HANDOFF.md``. Importing
the package needs no torch: the network modules (``model``, ``attention``, ``convgru``, ``losses``, ``train``,
``infer``) import it, the configuration / data / evaluation helpers do not.
"""
from .config import P7_CODE_REV, P7_VERSION

__all__ = ["P7GeoFaultFormer", "P7_VERSION", "P7_CODE_REV"]


def __getattr__(name):
    if name == "P7GeoFaultFormer":
        from .run import P7GeoFaultFormer

        return P7GeoFaultFormer
    raise AttributeError(name)
