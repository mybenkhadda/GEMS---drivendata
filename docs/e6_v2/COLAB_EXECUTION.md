# Colab execution

Notebook: `notebooks/colab/GEMS_E6_V2_FULL_PIPELINE.ipynb` (clones `REF = "main"` of
`mybenkhadda/GEMS---drivendata`). Runtime: GPU (A100 / L4 preferred for bf16); high-RAM recommended (the datasets
are held in RAM, about 2.5 GB, plus about 0.3 GB of targets per unit).

## Before the first run
1. Put the raw files on Drive (RAW_DATA_CONTRACT.md): the three GeoDAWN files under `GEMS_2/data/raw/geodawn/`
   (or another search folder) and the USGS 1 m tiles **with their XML metadata** under `GEMS_2/data/raw/dem_1m/`.
2. Commit and push this code to `main` (the notebook clones it) - or set `REF` to the commit SHA you pushed.
3. Optional overrides in cell 00 (they enter the fingerprints; set them before the first run).

## Order (each section is restartable)
| section | call | runs on | notes |
|---|---|---|---|
| 00-01 | config, mount, clone, install | - | prints the commit and the namespace |
| 02 | `PIPE.inventory()` | CPU | first run hashes the GeoDAWN files |
| 03 | `PIPE.preflight()` | CPU | **stops** on any blocker with one list |
| 04 | `PIPE.prepare_geophysics()` | CPU | FFT on about 5600 x 4900 padded grids |
| 05 | `PIPE.prepare_dem(assemble=False)` | CPU, multi-process | the longest stage; budget + resume |
| 06 | `PIPE.prepare_dem(assemble=True)` | CPU | assembly + coverage figure |
| 07 | `PIPE.prepare_datasets(include_splits=False)` | CPU | |
| 08 | `PIPE.prepare_splits()` | CPU | |
| 09 | `PIPE.train_reference(GEO_UNITS)` | GPU | stages A, B |
| 10 | `PIPE.train_e6(GEO_UNITS)` | GPU | stages C, D, E |
| 11 | `PIPE.train(DISC_UNITS)` | GPU | stages A-E |
| 12 | `PIPE.evaluate()` | GPU | DTI tables + figures |
| 13 | `PIPE.infer_full_region()` | GPU | 3 fold maps + mean |
| 14 | `PIPE.export_submission()` | CPU | 3 GeoTIFFs + validation |
| 15 | `PIPE.validate_submission()` | CPU | re-check |
| 16 | download | - | manual upload |

Sections 02-08 can run on a CPU runtime (the preflight warns that training is blocked); switch to GPU for 09-13.
After any disconnect: run 00, 01, then the section you were in.

## Reading the first outputs (do this before leaving it unattended)
- 03: `reports/preflight_report.md` - DEM tile counts, vertical-unit sources, tiles intersecting the grid.
- 04: `cache/*/feature_registry.csv`; NaN shares in `normalization.json` (`finite_frac_sample`).
- 05: `cache/dem/manifests/window_log.csv` grows; `seconds` per window gives the stage runtime; the QA GeoTIFFs in
  `cache/dem/dem_2m/` and `features_2m/` should show terrain, not stripes or seams.
- 09: `checkpoints/geo0/history.csv` - loss falling, `inner_val_dti` rising above the first epoch.
- 10: `checkpoints/geo0/C_parity.json` (`gate_closed_max_abs_diff` near 0, `uncovered_exact` true).

## Not measured
Every runtime, memory figure and DTI is unknown until this notebook runs.
