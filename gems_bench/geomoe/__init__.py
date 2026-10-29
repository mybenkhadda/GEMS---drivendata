"""GEMS discovery-first GeoMoE (docs/geomoe/DISCOVERY_GEOMOE.md).

    from gems_bench.geomoe.run import DiscoveryGeoMoE
    G = DiscoveryGeoMoE(CONFIG, HEAD, factory=F)     # F = attached submission factory (frozen B0 / B1 / W2 engine)
    G.run()                                          # stages [01]-[13]; every stage cached and persisted to Drive

Candidate discovery -> geological experts -> static / rule / learned gating -> official DTI under three validation
scenarios (V1 withheld systems, V2 network completion, V3 geographic folds) -> a validated GeoTIFF only when the
predeclared selection rule says so. Nothing here trains a neural network or touches the frozen artifacts.
"""
GEOMOE_VERSION = "dgm.1"            # enters every stage fingerprint: bump only to invalidate every cache
GEOMOE_CODE_REV = "dgm.1-r1"        # checked by the notebook start-up cell (not fingerprinted)
