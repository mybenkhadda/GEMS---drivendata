# G02 session handoff

**State (2026-10-27): code, config, notebook, tests and docs authored. NOTHING executed** (code-authoring-only mode:
no Python, shell, notebook, test, Ruff or import check was run; files were inspected with read-only tools). Branch
`gems3-discovery-research`, uncommitted. Every G02 number must be produced in Colab.

## Research objective

Does label-free structural evidence — apparent fabric offset across a line (G02-A), ridge truncation along a line
(G02-B), multiscale ridge continuity (G02-C), lineament-network geometry (G02-D), competing geological explanations
(G02-E) — add equal-area fault proximity beyond G01 (B1 / B2) and a simple magnetic gradient (B0), consistently across
held-out geographic folds and discovery splits? Decision criterion: incremental contrasts `F:<base>+<X>` vs `<base>`.

## Relevant previous experiments

* G01 (`docs/m9/G01_SESSION_HANDOFF.md`): authored, **never executed, no outputs anywhere**. Review:
  `docs/m9/G01_REVIEW_AND_REUSE.md` (all ten questions ABSENT, decision table UNVERIFIED).
* M9: only its context (`m9.ctx`, `m9.m7.units_of`, `m9.elig`, `m9.pkey`, `m9.layers`, `m9.stage_in` / `push`) and its
  layer cache are used. No M9 section, model or checkpoint is touched.

## Interfaces reused (read, not executed)

* `G01Experiment(m9, _cfg=...)`: `src` (`M9Source`), `fit_refs`, `read_window`, `channels`, `grid_meta`, `lams`, `xmod`
  (its store is never opened).
* `geology_wavelet`: `build_evidence(keep_per_scale=True)`, `tolerance_m`, `halo_m`, `combine`, `QuantileRef`,
  `unit_doubled`, `doubled_agreement`, `axial_agreement`, `strike_from_doubled`, `masked_gaussian`, `channel`,
  `is_candidate_key`, `Wavelength`.
* `geology_metrics`: `reference_mask`, `assert_disjoint`, `full_mask`, `interior_mask`, `equal_area_mask`,
  `distance_m`, `block_index`, `unit_block_counts`, `summarize`, `versus_edge`, `paired_block_bootstrap`,
  `per_block_uncertainty`, `official_style_rows`, `pooled_official`, `COUNTS`.
* `geology_io`: `channel_audit`, `xmod_names`, `G01Store` (subclassed), `_safe`, `_jsonable`, `run_root`, `latest_run`,
  `load_g01_config`, `fingerprint_config`, `LeakageError`, `G01ConfigError`.
* `geology_figures`: `map_grid`, `window`, `_extent_km`, `_outline`, `W16`, `H9`, colours.
* `pipeline.m6.fingerprint`, `m9.features.file_fp`, `m9.config.merge`, `artifacts.atomic.*`, `gems_eda.enhance.fill_nan`,
  `m7.protocol.score_row` (via G01), `gems3.official.Unit` fields (`sl`, `ev`, `known`, `eval_truth`, `key`).

## Files created

| file | role |
|---|---|
| `configs/m9/g02_structural_geology.yaml` | every G02 parameter; `modes.FAST_RESEARCH` / `FULL_EVALUATION` |
| `gems_bench/m9/research/g02_io.py` | config loader (mode resolution, validation), `G02Store` (folders, streamed NPZ, GeoJSON / GPKG, manifest), run listing |
| `gems_bench/m9/research/geology_displacement.py` | `RotFrame`, DoG band-pass, oriented NCC, step-vs-slope offset (G02-A), `side_means`, `noisy_or` |
| `gems_bench/m9/research/geology_truncation.py` | ridge skeletons, crossing-number nodes, tip direction, abruptness, aligned-endpoint z (G02-B), `across_step` |
| `gems_bench/m9/research/geology_continuity.py` | ridge persistence, strike stability, along-strike occupancy, straightness, bands (G02-C) |
| `gems_bench/m9/research/geology_network.py` | hysteresis skeleton, graph, segment attributes, continuation, raster, features (G02-D) |
| `gems_bench/m9/research/geology_hypotheses.py` | fault / contact / intrusive / terrain / ambiguous supports, G02-E |
| `gems_bench/m9/research/g02_metrics.py` | nested virtual candidates with overlap exclusion, contrasts, complementarity, strata, correlations, lineament metrics, `decide` |
| `gems_bench/m9/research/g01_review.py` | read-only G01 run review (ten questions, decision table, edge redundancy) |
| `gems_bench/m9/research/g02_figures.py` | figures 1-15 |
| `gems_bench/m9/research/g02_experiment.py` | `G02Experiment` (prepare / evaluate / metrics / figures / report / attach), `MapStore`, `LazyScores` |
| `notebooks/colab/GEMS_G02_STRUCTURAL_GEOLOGY_EXPERIMENT.ipynb` | Colab entry point |
| `tests/test_g02_structural_geology.py` | 26 synthetic tests (never run) |
| `docs/m9/G01_REVIEW_AND_REUSE.md`, `docs/m9/G02_STRUCTURAL_GEOLOGY.md`, this file | docs |

Modified: `gems_bench/m9/research/__init__.py` (exports `G02Experiment`; docstring). No other existing file changed.

## Known issues / risks (to check on the first run)

* Untested code: expect small runtime errors; fix the module the traceback names. Most likely spots:
  `ndimage.map_coordinates` with NaN inputs, pandas edge cases on empty tables, `skimage` API differences.
* Cost is unmeasured. Heaviest: G02-A (azimuths x 3 pairs x (2M + 1) shifts per wavelength and field). Levers:
  `orientation.n_azimuths`, `displacement.samples_per_wavelength`, `wavelengths_km`, `tiling.tile_px`.
* The halo is capped at 16 km: the 5 km regional fabric (3 sigma = 15 km) and the largest truncation background are
  close to the cap.
* `RotFrame` requires square pixels (2 %); the GEMS grid is believed to be 100 m.
* `fusion.bases` with three bases makes KEEP lenient (any base); see the guide, section 7.
* Network segments are cut at unit windows; skeleton spurs can create short segments.
* Truth-length strata derive from raster components (no segment attributes exist in the labels).

## Unverified assumptions

* `m9.m7.units_of`, `m9.m7.unit`, `m9.elig`, `m9.pkey`, `ctx.folds.train`, `ctx.splits.eval_zone`,
  `src.data.read_band`, `grid.rc_to_xy`, `grid.crs.to_epsg` behave as G01 reads them.
* Raw bands `rtp`, `iso_grav_anom`, `det_elev` exist (`g01` `raw_bands`).
* The synthetic tests' tolerances (offset within 100-500 m, contrast ratios) hold with the default constants.

## Remaining work

1. Colab: run the notebook in FAST_RESEARCH; fix failures; then FULL_EVALUATION.
2. Locally: `pytest tests/test_g02_structural_geology.py tests/test_g01_geology_wavelet.py` and Ruff.
3. Run G01 FULL as well, then refresh `docs/m9/G01_REVIEW_AND_REUSE.md` section 3 from `review_g01`.
4. Fill `docs/m9/G02_STRUCTURAL_GEOLOGY.md` section 10 from `reports/G02_REPORT.md`.

## Colab execution

1. Open `notebooks/colab/GEMS_G02_STRUCTURAL_GEOLOGY_EXPERIMENT.ipynb`, set `G02_CONFIG['mode']`.
2. Run section 0 (Drive, install, context), then sections 1-5 in order. Each unit / step is resumable; the run folder
   is pushed to Drive after every unit and step.
3. Reload later: section 0, then the "Resume" cell (`G02Experiment.attach(m9)`).

## Next-step instructions (fresh session)

Read this file, then `docs/m9/G02_STRUCTURAL_GEOLOGY.md`; open only the module a traceback names. Do not re-explore
M9 or G01.
