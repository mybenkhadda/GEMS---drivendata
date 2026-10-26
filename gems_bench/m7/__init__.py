"""GEMS M7: discovery-first research (notebooks/colab/GEMS_M7_DISCOVERY_FIRST.ipynb, docs/M7_DISCOVERY_FIRST.md).

    from gems_bench.m7.run import M7Research
    m7 = M7Research(pipe.pc, M7_CONFIG)                 # or pipe.m7(M7_CONFIG)
    m7.s00_setup(); m7.s01_baseline(); ...; m7.run()

Modules
-------
``config``        YAML defaults + validated notebook overrides (configs/m7/discovery_first.yaml)
``workspace``     outputs/m7_discovery_first writer (atomic, fingerprinted, immutable records, section manifests)
``protocol``      eligible budget domain, protocol keys + comparability guard, area ledgers, systems, block bootstrap
``baseline``      S01 B0 / B1 / fusion / thinning / ring under one protocol (nested selection)
``failure``       S02 fault-system diagnostics, stratified recall, regional density, case studies
``generator``     S03 multiscale geological candidate generator (label-free)
``coverage``      S04 oracle coverage audit and the GO / NO-GO gate
``dem_pilot``     S05 DEM inventory reuse and representative pilot selection
``dem_morph``     S06 tile-streaming 1 m morphology -> 100 m aggregates, cost estimator
``dem_value``     S07 geophysics vs DEM vs combined candidates (DEM-covered protocol)
``network``       S08 known-fault network elements and completion hypotheses (+ completion baseline)
``corruption``    S09 synthetic corruption benchmark, deterministic completion, gated learned completer
``cand_features`` S10 candidate-level features and ablation groups
``ranking``       S11 label-free control + LightGBM / CatBoost ablations (spatial + network-grouped LOZO)
``rasterize``     S12 candidate -> raster representations, hybrid, held-out map persistence
``comparison``    S13 every method under one guarded protocol, paired + bootstrap
``decision``      S14 PROMOTE / RETAIN / REJECT / INCONCLUSIVE + report
``export``        S15-S16 validated GeoTIFFs and leaderboard probe packages (never uploaded)
``figures``       16:9 analytical panels, true-geometry map panels
"""
