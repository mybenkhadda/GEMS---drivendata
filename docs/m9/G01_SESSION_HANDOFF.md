# G01 session handoff

**State (2026-10-27): code authored, no project code executed.** No notebook cell, test, import check, Ruff or pytest
has run (the only command run was a stray `python -c "print(1)"`, unrelated to the repository; files were read with
`cat` / `sed` / `grep`). Branch `gems3-discovery-research`, uncommitted. Every G01 number still has to be produced in
Colab.

## Objective

A falsifiable test of whether label-free fusion of magnetic + gravity wavelet evidence (tolerant co-location, axial
strike compatibility, multiscale persistence) puts more of an equal-area budget near held-out faults than either field
alone, a simple gradient edge, and M9's engineered cross-modal layers. Six candidates G01-A..F, no learned weights.

## Files inspected (read-only)

`gems_bench/m9/{spectral,features,dataset,config,run,figures,workspace,__init__}.py`, `configs/m9/geofault_production.yaml`,
`gems_bench/m7/protocol.py`, `gems_bench/gems3/official.py` (Unit), `gems_bench/folds.py` (roles), `gems_eda/io.py`
(Grid), `gems_bench/artifacts/atomic.py`, M9 notebook setup cells. Not inspected: `gems_bench/m9/model.py` (not needed:
G01 builds no model).

## Files created

| file | role |
|---|---|
| `configs/m9/g01_wavelet_geology.yaml` | all G01 parameters (unknown keys rejected) |
| `gems_bench/m9/research/__init__.py` | exports `G01Experiment` |
| `gems_bench/m9/research/geology_wavelet.py` | components, orientation gate, tolerant max, candidates, ablations |
| `gems_bench/m9/research/geology_metrics.py` | equal-area masks, distance P/R, enrichment, block bootstrap, nested choice, leakage guards |
| `gems_bench/m9/research/geology_figures.py` | figures 1-12 (16:9) |
| `gems_bench/m9/research/geology_io.py` | config loader, `M9Source`, cache audit, `G01Store` |
| `gems_bench/m9/research/experiment.py` | `G01Experiment` orchestration, hypothesis contrasts, report |
| `notebooks/colab/GEMS_G01_GEOLOGICAL_WAVELET_EXPERIMENT.ipynb` | Colab entry point |
| `tests/test_g01_geology_wavelet.py` | 21 synthetic tests (never run) |
| `docs/m9/G01_GEOLOGICAL_WAVELET.md` | guide + report template + next experiments |

No existing file was modified.

## Architecture decisions

* Inputs = M9 cached layers by NAME (`m9:spec:<fam>_wav_{energy,coherence,strike_c2,strike_s2}@<lam>km`,
  `m9:xmod:mag_grav_{tol,orient}@*`), discovered from `cache/m9/layers/*.npy` stems. `build_cube` metadata is never read
  (its `meta["pooled"]` is overwritten by a path).
* Energy is used only through percentiles (M9's standardisation is monotone, so irrelevant). Percentile tables are fit
  per unit on its TRAINING geography (`folds.train(k)` / outside `splits.eval_zone(d)`); `assert_disjoint` enforces it.
* M9 doubled-angle vectors already carry coherence: agreement A uses unit vectors, confidence kappa = sqrt(r_m r_g)
  enters once; gate = 1 - kappa (1 - A) (undefined orientation never penalises).
* Tolerance in metres -> pixels via `grid.pixel_km_{y,x}`; tol(lambda) = clip(0.25 lambda, 200, 1200) m; soft max
  with Gaussian distance discount on nested neighbourhoods (square = separable, disk = exact).
* Missing field / pixels: NaN-aware means; J undefined where either field is unobserved.
* Evaluation domain = `m9.elig(u, 0)` & interior (400 m from data edge), identical for all maps; exact-k selection.
* Store: `outputs/m9_geofault_production/research/G01_wavelet_geology/g01_<fp10>/`, write-once, manifest, done markers.

## Assumptions to verify on first run

* `m9.m7.units_of(task)`, `m9.m7.unit("task:zone")`, `m9.elig`, `m9.pkey`, `ctx.folds.train`, `ctx.splits.eval_zone`
  behave as read in the code (they were read, not executed).
* `cache/m9/layers` is either local or on Drive under the same relative path (M9 does not push it itself; G01 stages it
  and only pushes it after `build_missing`). If absent: run `m9.s03_spectral()` once or set `build_missing.wavelets`.
* Grid is 100 m (M9 wavelets hard-code PX_KM = 0.1); G01 warns otherwise.
* Exploratory full-region rasters (~18 float16 maps) fit in Colab RAM.

## Scientific risks

* Wavelet energy marks contrasts, not faults: contacts, intrusions and dykes score high (G03 is the follow-up).
* G01-C..E use fixed equal weights: a negative result refutes this fusion rule, not all fusion.
* Persistence floor 0.5 can penalise single-scale faults; per-wavelength ablations expose it.
* Held-out zones were seen in earlier GEMS work: not an untouched test set.
* Square tolerant neighbourhoods overweight diagonals by up to sqrt(2) in reach.

## Remaining tasks

1. Run the notebook top to bottom in Colab (PILOT M9 profile is fine: G01 uses all zones by default).
2. Run `pytest tests/test_g01_geology_wavelet.py` and Ruff locally; fix whatever fails.
3. Fill the report template (`docs/m9/G01_GEOLOGICAL_WAVELET.md`, section "Results") from `reports/G01_REPORT.md`.
4. Decide G02-G06 order from the verdict table (see the guide's "Next experiments").

## Next-step instructions (fresh session)

Read this file, then `docs/m9/G01_GEOLOGICAL_WAVELET.md`, then only the module that a failure points to. Do not
re-explore M9. Entry point: `notebooks/colab/GEMS_G01_GEOLOGICAL_WAVELET_EXPERIMENT.ipynb`.
