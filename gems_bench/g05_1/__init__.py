"""GEMS G05.1 fault-system recovery & intelligent geological ranking (docs/g05_1/README.md).

    from gems_bench.g05_1.runner import G051
    R = G051(CONFIG, RUN, HEAD, factory=F)
    R.run()                          # stages gated by RUN (preflight, atlas, recall, registry, ... export)

Built on the DiscoveryGeoMoE machinery (frozen B0 / B1 / W2 engine through the submission factory, V1 / V2 / V3 units,
multi-expert candidate inventory, canonical scorer, system recovery, frozen submitted W2 policy). Pure scientific
functions live in geometry / graph / displacement / recall / registry / rankers / selection; Colab orchestration in
runner. Nothing here uploads a submission.
"""
G051_VERSION = "g051.1"
G051_CODE_REV = "g051.1-r1"
