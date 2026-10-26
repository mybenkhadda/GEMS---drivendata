"""GEMS M8: physics-guided probabilistic fault discovery
(notebooks/colab/GEMS_M8_PHYSICS_PROBABILISTIC_DISCOVERY.ipynb, docs/M8_PHYSICS_PROBABILISTIC_DISCOVERY.md).

    from gems_bench.m8.run import M8Research
    m8 = M8Research(pipe.pc, M8_CONFIG)                 # or pipe.m8(M8_CONFIG)
    m8.s00_setup(); m8.s01_baseline(); ...; m8.run()

Modules
-------
``config``           YAML defaults + validated notebook overrides; PILOT / VALIDATE / FULL scope profiles
``workspace``        outputs/m8_physics_probabilistic writer (M7 workspace subclass)
``inventory``        integration inventory of every reused function
``baseline``         S01 baseline recovery (M7 artifacts, protocol-key guarded) + compatibility checks
``signal_recovery``  S02 system x extraction-stage matrix, matched-null raw responses, failure reasons, Gate 1
``potential_fields`` S03 operators: THDR, analytic signal, tilt, TDX, theta, LCN, upward continuation, spectra
``layers``           memory-mapped layer cache, derivative-band provenance audit
``worms``            S03b multiscale worms (upward-continuation scale-space)
``multimodal``       S04 tolerant magnetic-gravity agreement, surface-subsurface classes, context
``profiles``         S05 fault-normal transects, profile features, gallery
``candidates``       S06 high-recall proposal families, union + dedup, oracle gate (Gate 2)
``montecarlo``       S07 Monte Carlo signal-extraction stability, occupancy atlas, MC-only structures (Gate 3)
``completion``       S08 stochastic network completion, corruption benchmark, System C
``dem``              S09 DEM pilot reuse / streaming and incremental value (Gate 4)
``features``         S10 candidate feature table
``ranking``          S11 LightGBM ablations vs label-free / CNN-only controls (Gate 5)
``graph``            S12 optional fragment graph
``rasterize``        S13 uncertainty-aware rasterisation
``ensemble``         S14 Systems A / B / C, cross-fitted fusion, comparison, verdicts (Gate 6)
``submission``       S15 validated GeoTIFFs + leaderboard_log.csv (never uploaded)
``gates``, ``report``, ``figures``
"""
