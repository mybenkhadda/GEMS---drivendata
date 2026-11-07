# Colab execution and resume

1. Push `gems_bench/georegime_recovery/` and the notebook to `main`, or set `CFG["ref"]` to a commit that contains them. Cell 01 refuses a checkout that lacks the package.
2. Open `notebooks/colab/GEMS_GEOREGIME_RECOVERY_V2.ipynb` on a **GPU** runtime. CPU works but is slow.
3. Run the cells top to bottom. Section 02 must pass; it raises `PreflightError` with the list of failures otherwise.

| section | call | cost | cached? |
|---|---|---|---|
| 01 | bootstrap, `RecoveryRun(CFG)` | light | - |
| 02 | `RUN.preflight()` | hashes the checkpoints and V1 rasters once (hash cache) and loads the 7 checkpoints on CPU | hash cache |
| 03-04 | baseline, proximity | light (full-grid EDT, skeleton) | recomputed (seconds to minutes) |
| 05 | split audit | dilations + EDT | recomputed (needed in memory) |
| 06 | original metrics | 6 scorings | recomputed |
| 07 | full-area inference | **expensive**: 7 units × all windows, GPU | per-unit identity + snapshots |
| 08 | full-area scoring | moderate | stage record |
| 09 | error atlas | moderate (maps, EDTs, regions) | stage record |
| 10 | experts | moderate; stages the V1 bank (~0.8 GB) to scratch | stage record |
| 11 | geology | moderate (Atlas layers) | stage record |
| 12 | strata | moderate | stage record |
| 13 | lineaments | light | recomputed |
| 14 | transforms | moderate (4 policies × 6 units) | stage record |
| 15 | decision | light (bootstrap) | recomputed from the section 14 tables |
| 16 | export | only if enabled and promoted | - |
| 17 | report | light | - |

Each stage prints `CACHED`, `COMPUTING (reason)` or `SKIPPED (reason)`. A stage record is reused only if its fingerprint matches (source fingerprint, config hash, code hash, upstream inference identities) and every output still has its recorded size and sha256.

## After a disconnect (`OSError: [Errno 107] Transport endpoint is not connected`, or a runtime reset)
1. `from google.colab import drive; drive.flush_and_unmount()`, then `drive.mount('/content/drive', force_remount=True)`. If that keeps failing, use Runtime > Restart session.
2. Re-run cells 00, 01 and 02. Preflight is required, because it rebuilds the bank and checkpoint metadata in memory.
3. Re-run section 05. It is required before sections 06–09, but sections 06/07 run it automatically if it was skipped.
4. Resume at the section that failed:
   * **07**: completed units are verified and skipped. A unit in progress resumes from `cache/inference/<identity>/snapshot.npz`; only windows after the last snapshot are recomputed.
   * **08–14**: completed stages reload from `manifests/<stage>.json`.
   * **Figures**: re-run the `plot_group` line. They are rebuilt from the saved rasters and tables.
5. Sections 11 and 12 need `RUN.spatial_error_atlas()` (09) in the same session. It reloads from cache in seconds when complete.

## Forcing recomputation
* Set `CFG["reuse_valid_cache"] = False` to recompute stage records; full-area rasters are still reused while their identity is valid.
* Or delete one `manifests/<stage>.json`.
* To redo inference for one unit, delete its `02_full_inference/inference_manifests/<unit>_full_area.json`.

## Never
* Training. `train_models` must stay False.
* Writes into `GEMS_GEOREGIME_V1/` or the Atlas. `RecoveryPaths.check_writable` raises.
* Automatic upload.
