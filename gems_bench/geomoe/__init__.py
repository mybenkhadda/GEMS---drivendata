"""GEMS discovery-first GeoMoE (docs/geomoe/DISCOVERY_GEOMOE.md).

    from gems_bench.geomoe.run import DiscoveryGeoMoE
    G = DiscoveryGeoMoE(CONFIG, HEAD, factory=F)     # F = submission factory (frozen B0 / B1 / W2 engine)
    G.run()                                          # FAST_DIAGNOSTIC: [00]-[06]; FULL: every stage

Candidate discovery -> geological experts -> static / rule / learned gating -> official DTI under three validation
scenarios (V1 withheld systems, V2 network completion, V3 geographic folds), compared with the FROZEN submitted W2
policy under one fixed-policy rule -> a validated GeoTIFF only when the predeclared selection rule says so.
"""
GEOMOE_VERSION = "dgm.2"            # enters every stage fingerprint: bump only to invalidate every cache
GEOMOE_CODE_REV = "dgm.2-r1"        # checked by the notebook start-up cell (not fingerprinted)
