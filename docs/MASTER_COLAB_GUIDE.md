# Master Colab guide

`notebooks/colab/GEMS_MASTER_PIPELINE.ipynb` runs the whole GEMS workflow from one notebook:

**Open the notebook → mount Drive → inspect status → run missing stages → reuse existing results → train selected models → evaluate → export a submission.**

The notebook orchestrates library code in `gems_bench.pipeline`. It contains no downloader, trainer, or
feature implementation of its own.

## 1. Open it in Colab

* From GitHub: `https://colab.research.google.com/github/mybenkhadda/GEMS---drivendata/blob/main/notebooks/colab/GEMS_MASTER_PIPELINE.ipynb`
* Or **File → Open notebook → GitHub**, then enter `mybenkhadda/GEMS---drivendata` and pick the notebook.
* For GPU stages: **Runtime → Change runtime type → GPU** (T4, L4, or A100; no particular GPU is assumed).
* For a private repository, add a Colab Secret `GITHUB_TOKEN` (key icon, enable notebook access). The token is
  never printed or stored.

## 2. Control panel

Section 1 holds two dicts, `CONFIG` and `DEM_CONFIG`. They override `configs/pipeline/master.yaml`, and
`gems_bench.pipeline.config.PipelineConfig` validates them. An unknown key raises an error that names the closest
valid key, so a typo cannot silently turn off a safety switch.

| key | default | effect |
|---|---|---|
| `run_mode` | `RESUME` | see the modes below |
| `download_missing` | `True` | download only files that are missing or invalid both on Drive and locally |
| `rebuild_missing` | `True` | in `RESUME`, build missing cheap stages (`False`: only continue stages that were started) |
| `run_eda` | `True` | minimal Phase 1 rebuild when `outputs/eda` is missing (CPU, once) |
| `run_baselines` | `True` | scan and verify baselines (cheap) |
| `train_baselines` | `[]` | names to retrain when missing: `B0_lineament`, `G1`, `G3`, `RS05_G3_noSS_RS` (CPU); `CNN_A_dti`, `DL00_CNN_B_dti`, `DL02_CNN_TMGB_dti`, `P4_B_gated` (GPU) |
| `run_advanced_features` | `True` | Phase 4 remote-sensing features used by the baselines |
| `run_advanced_models` | `False` | GPU research experiments listed in `advanced_experiments` |
| `run_dem_pilot` | `False` | 1 m DEM pilot (the download also needs `DEM_CONFIG['download']`) |
| `run_scoreboard`, `run_ensemble` | `True` | CPU evaluation of the available models |
| `run_full_inference`, `run_submission` | `False` | promoted model → full map → GeoTIFFs (never uploaded) |
| `use_gpu`, `max_stage_minutes` | `True`, `0` | GPU stages on or off; a per-stage time budget gives a clean, resumable stop |

Feature families, the validation protocol and the experiment lists are in `configs/pipeline/master.yaml`. Override
them with `Pipeline.from_notebook(CONFIG, DEM_CONFIG, overrides={...})`.

## 3. Execution modes

| mode | runs | builds missing prerequisites? |
|---|---|---|
| `SETUP_ONLY` | bootstrap and raw-data verification, no downloads | no (validate only) |
| `PREPARE_DATA` | raw data → grid → EDA repair → folds and splits → features (+ DEM inventory and pilot when enabled) | yes |
| `BASELINES` | + baselines (restore, or train the listed ones) + scoreboard + reports | yes |
| `ADVANCED_RESEARCH` | + PU, geometry, GeoStructureNet and multiscale experiments | yes |
| `FULL_PIPELINE` | every enabled stage, including ensemble, inference and submission | yes |
| `RESUME` (default) | every enabled stage, continuing from saved state | yes, when `rebuild_missing` |
| `SUBMISSION_ONLY` | full inference + export of the promoted model | no: prerequisites are validated and staged |

Expensive stages (GPU training, DEM tile download, full inference) run only when their switch is on, in every
mode. `FULL_PIPELINE` prints the expensive stages it will run before it starts.

Two ways to drive it:

* **Section by section.** Each section cell calls `pipe.step('<stage>')`. The step runs if the mode includes the
  stage, and otherwise only reports the validated state.
* **One shot.** `pipe.run()`, or `RUN_EVERYTHING_NOW = True` in section 6, executes the configured mode in
  dependency order.

`pipe.plan()` is read-only. It validates what is on Drive and predicts each stage's action.

## 4. Fresh download URLs

DrivenData links are personal and expire, and old signed links are never assumed to be valid. Provide new links
in one of these ways (first hit wins, per file):

1. the notebook's URL cell (`ASK_FOR_URLS = True`, hidden `getpass` input, held in memory only);
2. environment variables `GEMS_URL_FEATURES`, `GEMS_URL_FAULTS`, `GEMS_URL_SUBMISSION`, `GEMS_URL_DEM_JSON`;
3. Colab Secrets with the same names (key icon → add → enable notebook access).

URLs are never written to disk or Git. `data/raw/acquisition_manifest.json` records only the host and a hash. A
file that is valid on Drive is never downloaded again, even after a runtime restart. Per file the order is: Drive
(canonical path, then accepted alternates such as `data/raw/example_submission.tif` and
`data/raw/tnm_items.json`) → local disk → resumable download (`.partial` + HTTP Range + retries) → GeoTIFF or
JSON validation → verified atomic copy to Drive → manifest.

## 5. How cached artifacts are identified

A stage is skipped only when **all three** of these hold:

1. **Content validation passes.** Files are opened, not just checked for existence: GeoTIFF headers and grid
   alignment (CRS, transform, shape), band names, Parquet schemas and row counts, JSON keys, `.npy` shape and
   dtype, `run.json` status `done`, fold and split parameters. An empty Parquet file or a directory with partial
   contents never passes.
2. **The fingerprint matches.** The stage record (`environment/pipeline/stages/<stage>.json`) stores a hash of the
   configuration sections and input identities that determine the result. Code revisions are recorded but do not
   invalidate caches.
3. **Valid outputs without a record are adopted.** Artifacts uploaded earlier or produced by notebooks 00–07 are
   recorded with `origin: adopted_existing`.

Outcomes: `SKIP_CACHED`, `BUILD`, `RESUME` (partial or invalid; invalid local files are renamed `*.invalid-<ts>`
first), `INCOMPATIBLE` (never overwritten), `BLOCKED` (names the missing prerequisite), `DISABLED`, `FAILED` (error,
traceback tail, recovery action), `INTERRUPTED`.

## 6. Resume after a Colab disconnect

Run sections 1–4 again, then the same cells, or `pipe.run()`.

* Every stage pushes its outputs to Drive when it finishes. The feature store pushes each family as soon as it
  exists, baselines are pushed one by one, and experiments push after every CV fold or discovery split, with
  checkpoints at most every `ckpt_sync_min` minutes.
* A new VM has an empty `/content/gems_work`. Records are read from Drive, and outputs are staged back
  (unchanged files are skipped, using the sync ledger).
* An interrupted experiment restarts at its next unit and resumes training from the last checkpoint.
  Experiments are opened with `resume_policy="config"`: the configuration hash must match, while a newer code
  commit is only recorded as a warning.
* `pipe.status()`, `pipe.record('<stage>')`, `pipe.events()` and `reports/pipeline/PIPELINE_REPORT.md` show the
  state.

## 7. Where everything is saved (Drive: `MyDrive/GEMS_2/`)

| what | where |
|---|---|
| raw competition files | `data/raw/geodawn/`, `data/raw/faults/` (alternates accepted), `data/raw/acquisition_manifest.json` |
| DEM URL list, inventory, footprints | `data/raw/dem_1m/tnm_urls.json`, `data/raw/dem_1m/inventory/` (CSV, Parquet, GeoPackage, PNG, storage estimate) |
| DEM pilot tiles and products | `data/raw/dem_1m/<project>/USGS_1M_*.tif`, `data/processed/dem_1m_100m/{tiles/, dem_features_100m.tif}` |
| grid record | `environment/pipeline/native_grid.json` |
| Phase 1 EDA (minimal) | `outputs/eda/{datasets,features}/`, `outputs/eda/manifest.json`, `REBUILD_PROVENANCE.json`, `outputs/eda_rebuild/<version>/` |
| segment rasters | `outputs/benchmark/datasets/{native_segment_ids.tif, segment_attributes.parquet}` |
| folds and splits | `outputs/benchmark/folds/geo_k3_b20km_buf13km_s2026/`, `.../discovery_d3_link1km_s2026/` |
| feature caches | `outputs/benchmark/cache/features/` (Phase 2), `outputs/phase4/cache/features/` (RS), `data/feature_store/phase5/` (profiles, DEM); registry `data/feature_store/pipeline_feature_registry.csv` |
| baseline runs | `outputs/{benchmark,phase3,phase4}/experiments/<exp_id>/` (+ `PROVENANCE.json` when regenerated) |
| Phase 5 experiments | `experiments/<id>/{config.yaml, manifest.json, checkpoints/, metrics/, predictions/, figures/, logs/, reports/}` |
| model registry | `model_registry/` |
| submissions | `submissions/*.tif` + provenance JSON (never uploaded) |
| pipeline reports | `reports/pipeline/` (`PIPELINE_REPORT.md`, CSV tables, PNG + SVG figures in `figures/`), `reports/leaderboard_log.csv` (manual) |
| stage records and events | `environment/pipeline/stages/*.json`, `environment/pipeline/events.jsonl` |

## 8. Scientific safeguards built in

* Folds and splits are reused, never regenerated when an experiment starts. A changed protocol gets a new
  folder name.
* Unmapped pixels are not treated as verified non-faults: discovery splits give withheld systems weight 0, and
  the PU experiments treat unlabeled pixels explicitly.
* Geographic and pseudo-discovery scores are reported separately. `public_lb` comes only from your manual
  `reports/leaderboard_log.csv`.
* No model score is ever inferred from a report. A baseline without a verified run folder is `NOT_AVAILABLE`.
* Promotion is manual (`promote(...)` / `gems registry promote`). Submission files are written, never uploaded.
