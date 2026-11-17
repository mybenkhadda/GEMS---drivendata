"""G13 GeoDiscovery research engine.

From the raw 19-band GeoDAWN raster to geological interpretation, physics-informed features, spatially honest
benchmarking, fault-network reconstruction and a validated DrivenData GeoTIFF.

Design rules (see docs/G13_GEODISCOVERY.md):

* Reuse what is correct and tested on ``main``: the DTI scorer (``gems_bench.metric``), the official-convention fast
  scorer and evaluation units (``gems_bench.gems3.official``), geographic folds and whole-fault-system discovery
  splits (``gems_bench.folds``), topology operations (``gems_bench.postprocess.topology``) and output
  representations (``gems_bench.gems3.representation``).
* Training labels never enter a feature. Learned transforms are fitted on training partitions only.
* Offline DTI on withheld *known* faults is a proxy. It is never reported as an official score.
* Every stage writes atomically and records a fingerprint; a completed, verified stage is reused, never recomputed.
"""

from __future__ import annotations

G13_VERSION = "g13.1.0"

__all__ = ["G13_VERSION"]
