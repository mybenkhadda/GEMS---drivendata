# G01 review and reuse (input to G02)

**Reviewed 2026-10-27, read-only, nothing executed.**

## 1. What G01 actually established: nothing measured yet

| check | finding |
|---|---|
| G01 code | present: `gems_bench/m9/research/{experiment,geology_wavelet,geology_metrics,geology_io,geology_figures}.py`, `configs/m9/g01_wavelet_geology.yaml`, notebook `notebooks/colab/GEMS_G01_GEOLOGICAL_WAVELET_EXPERIMENT.ipynb`, 21 tests |
| G01 execution | **never run** (`docs/m9/G01_SESSION_HANDOFF.md`: "no project code executed"; no notebook outputs) |
| G01 outputs | **absent**: there is no `outputs/` directory in the repository and no `research/G01_wavelet_geology/g01_*` run folder, `LATEST.json`, table, raster, figure or report |
| G01 tests | authored, never run |

So every one of the ten review questions is **ABSENT** today, and every component in the decision table below is
**UNVERIFIED**. Nothing in this document claims that any G01 candidate works.

## 2. Ten review questions (current answers)

| # | question | answer now | how it will be answered |
|---|---|---|---|
| 1 | Which G01 experiments were completed? | ABSENT | `processed/done_*.json`, `tables/units` |
| 2 | Which wavelengths contributed most? | ABSENT | `tables/ablations` (G01-A/B/C/D@lambda), `tables/nested_choice` |
| 3 | Did gravity improve magnetic-only evidence? | ABSENT | contrast G01-C vs G01-A |
| 4 | Did strike agreement help? | ABSENT | G01-D vs G01-C |
| 5 | Did multiscale persistence help? | ABSENT | G01-E vs G01-D |
| 6 | Did spatial tolerance improve alignment? | ABSENT | G01-C\|tol=... ablation |
| 7 | Did G01 beat simple magnetic-edge detection? | ABSENT | G01-A vs edge; G01-E vs edge (bootstrap table) |
| 8 | Did G01 beat the M9 cross-modal features? | ABSENT | G01-E / G01-C vs G01-F |
| 9 | Consistent across geographic folds? | ABSENT | `zone_win_share` of every contrast |
| 10 | Plausible geology or merely strong gradients? | ABSENT | rank correlation of each G01 map with the edge on unit rasters + figures 9 / 12 |

## 3. Decision table (populated automatically from measured G01 results)

`gems_bench/m9/research/g01_review.py::review_g01(run_dir)` fills this table from a persisted run; it is also run by
`G02Experiment.prepare()` (`tables/g01_decision_table`, `reports/G01_REVIEW.md` of the G02 run). Mapping: SUPPORTED on
every task -> Keep; REFUTED and never SUPPORTED -> Reject; mixed / inconclusive -> Investigate; absent -> UNVERIFIED.

| Component | Evidence | Incremental value | Geological interpretation | Keep / Reject / Investigate |
|---|---|---|---|---|
| magnetic wavelet energy (G01-A vs edge) | ABSENT | ABSENT | not measured | UNVERIFIED |
| gravity wavelet energy (G01-B vs edge) | ABSENT | ABSENT | not measured | UNVERIFIED |
| tolerant magnetic-gravity co-location (G01-C vs G01-A) | ABSENT | ABSENT | not measured | UNVERIFIED |
| axial strike compatibility (G01-D vs G01-C) | ABSENT | ABSENT | not measured | UNVERIFIED |
| multiscale persistence (G01-E vs G01-D) | ABSENT | ABSENT | not measured | UNVERIFIED |
| G01 fusion vs M9 engineered cross-modal (G01-E vs G01-F) | ABSENT | ABSENT | not measured | UNVERIFIED |
| simple joint vs M9 engineered cross-modal (G01-C vs G01-F) | ABSENT | ABSENT | not measured | UNVERIFIED |
| wavelength / tolerance choice (nested) | ABSENT | ABSENT | not measured | UNVERIFIED |

Use in Colab (read-only on the G01 run):

```python
from gems_bench.m9.research.g01_review import review_g01, g01_run_dirs
from gems_bench.m9.research.geology_io import run_root, load_g01_config
runs = g01_run_dirs(run_root(m9, load_g01_config()))
rev = review_g01(runs[-1]) if runs else review_g01(None)
rev["questions"], rev["decision_table"], rev["edge_redundancy"]
```

## 4. What G02 reuses from G01 (code, not results)

Because no G01 result exists, G02 does **not** pick "the best G01 component" by inspection. It reuses G01 code and
decides B1 / B2 by nested (leave-one-zone-out) choice inside its own evaluation:

| G01 interface | used by G02 for |
|---|---|
| `G01Experiment(m9, _cfg=...)`: `fit_refs`, `read_window`, `channels`, `grid_meta`, `src` (`M9Source`) | per-unit training-geography percentile tables, cached-layer windows, input fingerprints (its store is never opened) |
| `geology_wavelet.build_evidence(..., keep_per_scale=True)` | B0 (`edge`), G01-A..F, per-wavelength variants, components M, G, J, P, orientation gate, S, per-scale percentiles (energy gate of G02-A, ridges of G02-B / C) |
| `tolerance_m`, `unit_doubled`, `doubled_agreement`, `axial_agreement`, `strike_from_doubled`, `masked_gaussian`, `combine`, `QuantileRef` | G02-C tolerances and orientation algebra, G02-E fabric, fusions |
| `geology_io.channel_audit`, `xmod_names`, `G01Store` (subclassed as `G02Store`), `_safe` | wavelength alignment, G01-F inputs, persistence |
| `geology_metrics`: `reference_mask`, `assert_disjoint`, `interior_mask`, `equal_area_mask`, `distance_m`, `block_index`, `unit_block_counts`, `summarize`, `versus_edge`, `paired_block_bootstrap`, `per_block_uncertainty`, `official_style_rows`, `pooled_official` | the entire held-out protocol (same domain, budgets, distances, blocks, bootstrap) |
| `geology_figures.map_grid`, `window`, `_extent_km`, `_outline` | G02 map figures |

## 5. Recommendation for G01 itself

Run G01 (its notebook, FULL configuration) before or alongside G02 FULL_EVALUATION: G02's B1 / B2 baselines are
G01 maps recomputed by G01 code, so G02 numbers are interpretable without G01's run, but G01's own ablations
(tolerance, per-wavelength nested choice) and its verdict table are only produced by G01. Then re-run
`review_g01` and replace section 3 with the generated table.
