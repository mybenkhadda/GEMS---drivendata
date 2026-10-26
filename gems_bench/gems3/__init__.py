"""GEMS 3.0: discovery-first research on existing predictions (no CNN retraining).

    from gems_bench.gems3.run import GEMS3Research
    g3 = GEMS3Research(pipe.pc, GEMS3_CONFIG)          # or pipe.gems3(GEMS3_CONFIG)
    g3.run()                                            # every enabled stage; fingerprinted, resumable

Modules
-------
``config``          YAML defaults + validated notebook overrides (configs/gems3/research.yaml)
``workspace``       reports/gems3 writer (atomic tables / figures PNG+SVG / rasters, per-stage fingerprints)
``official``        evaluation units under the verified competition contract (geo, disc_official, disc_excl)
``representation``  output representations: top-area, crest lines, widths, oriented dilation, halos, known ring
``metric_opt``      Milestone A: thresholds, area budgets, widths, dilation, B0+B1 fusion (nested selection)
``submission``      Milestone B: full-region inference of the selected members + validated GeoTIFF export
``candidates``      Milestone E: label-free lineaments, model crest lines, known-fault tip extensions, dedup
``oracle``          candidate headroom: coverage, system / short-fault recall, oracle DTI
``descriptors``     per-candidate feature table (geometry, evidence, agreement, distance to known faults)
``ranker``          weighted evidence score + LightGBM ranker with spatially separated training
``map_completion``  Milestone D: corruption protocol + deterministic endpoint-extension baseline
``dem_gate``        Milestone C gate: held-out evidence gain of cached DEM features beyond B0 / B1
``report``          GEMS3_RESEARCH_REPORT.md
"""
