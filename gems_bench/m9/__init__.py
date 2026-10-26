"""GEMS M9: GeoFault production system (notebooks/colab/GEMS_M9_GEOFAULT_PRODUCTION.ipynb,
docs/M9_GEOFAULT_PRODUCTION.md, configs/m9/geofault_production.yaml).

    from gems_bench.m9.run import M9Research
    m9 = M9Research(pipe.pc, M9_CONFIG)                 # or pipe.m9(M9_CONFIG)
    m9.s00_setup(); m9.run(stage="PREPARE"); m9.run(stage="TRAIN"); ...; m9.run()

Modules
-------
``config``       YAML defaults + validated overrides; stages PREPARE..SUBMIT; PILOT / FULL scope; training variants
``workspace``    outputs/m9_geofault_production writer (M7 workspace subclass)
``inventory``    static reuse matrix (existing module -> reuse / change -> M9 consumer)
``features``     S02 feature pyramid: store inventory, families, label-free redundancy, new tolerant cross-modal
                 evidence, channel groups, nested pixel screen (Gate 1 evidence)
``spectral``     S03 windowed directional FFT, steerable log-Gabor wavelets, upward continuation (+ optional STFT)
``dataset``      S04 training units, memory-mapped cube, fold-wise robust scaler, targets, patch batches, CV manifests
``corruption``   S05 conditional-reconstruction operators (training) + CV-C benchmark sets
``model``        S06 GeoFaultNet: modality stems, gated fusion, U-Net, low-resolution context branch, 4 heads
``losses``       S07 L1..L6 staged objective (L7 consistency in train)
``train``        S08 resumable trainer (AMP, accumulation, clipping, checkpoints, OOM back-off, collapse guard)
``infer``        tiled multi-head inference, halo tiles with seam blending, resumable full-region tiles
``validation``   S09 held-out maps, nested selection, CV-A..F diagnostics, Gates 2-3
``tracing``      S11 crest + minimum-cost baseline, batched orientation-aware beam search, branches, oracle (Gate 4)
``montecarlo``   S12 MC-A evidence perturbation, MC-B stochastic paths, convergence (Gate 5)
``rl``           S13 optional Gymnasium tracing environment, behavioural cloning, PPO / DQN, learned edge cost
``candidates``   S14 candidate-level geological features (Parquet + GeoPackage)
``gbm``          S15 XGBoost / LightGBM / CatBoost benchmark on identical inputs (+ optional LambdaMART)
``fusion``       S16 dense + structured (+ legacy) cross-fitted fusion, comparison, Gate 6
``submission``   S17 full-area inference products, S18 validated GeoTIFFs (champion + fallback; never uploaded)
``gates``, ``figures``, ``report``, ``run``
"""
