# G11 master Colab execution

**Notebook:** `notebooks/colab/GEMS_G11_MASTER_COLAB_EXECUTION.ipynb` (21 sections, 00–20; orchestration only; all logic is in `gems_bench/g11`).

## Launch
1. Push the G11 code to the branch in `CONFIG["ref"]` (default `g11-colab-production`). Section 02 checks that the branch exists on GitHub and contains `gems_bench/g11`; if not, it stops and prints the deployment commands.
2. Open the notebook in Colab, using `https://colab.research.google.com/github/mybenkhadda/GEMS---drivendata/blob/<ref>/notebooks/colab/GEMS_G11_MASTER_COLAB_EXECUTION.ipynb`. Choose a **GPU** runtime: T4 works, and L4 or A100 are faster. CPU works but is slow for E3/E4.
3. Run **`run_mode="SMOKE_TEST"`** first. This is LEVEL 2: a 30 km real-data window with mapped faults *and* DEM coverage, 2 epochs, written to `..._SMOKE/`.
4. Then set `run_mode="FULL"` (LEVEL 3) and run all cells. Section 03 always runs the synthetic suite first (LEVEL 1) and stops if it fails.

After a disconnect, re-run sections 00–03 and then the interrupted section. Completed stages are reused when their fingerprints and output hashes still match. Interrupted stages are left `RUNNING`/`FAILED` in `manifests/` and are recomputed.

## Values you may need to edit (section 00)
| key | when |
|---|---|
| `ref` | Branch, tag or commit that contains the G11 code. |
| `drive_root` | If your data is not at `/content/drive/MyDrive/GEMS_2`. |
| `run_mode` | `SMOKE_TEST` first, then `FULL`; the `*_ONLY` modes rerun one phase. |
| `models` | Drop `segformer` / `gated_fusion` on CPU-only runtimes. |
| `force_recompute` | `True` recomputes everything; `resume` must then be `False`. |
| `smoke_window_km` | Size of the LEVEL 2 subset. |
| `dem_training_resolution_m` | Resolution of the E4 DEM branch (25 m default). |

Advanced settings are YAML keys in `configs/g11_multiscale.yaml`:
- `dem.allow_unknown_vertical`;
- `training.loss_ablation`;
- `models.gated_fusion.mode_ablation`;
- `models.segformer.pretrained_path` (G07 `T1_segformer.pt`, used only if its tensors match);
- `official_submission.evaluator_command`.

## Resource requirements — ESTIMATES (not measured on the real data)
| item | estimate | basis |
|---|---|---|
| one float32 feature on the 3730 × 3292 grid | 49 MB | 12.3 M px × 4 B |
| feature store on Drive (`03_features/`) | about 12–17 GB (FULL) | about 250–350 layers × 49 MB; SMOKE uses about 0.2 GB |
| peak RAM, features | about 3–6 GB | about 15–18 float64 grid arrays per (field, scale) step |
| peak RAM, DEM aggregation | about 2–4 GB | 84 × 84-cell chunk at 2 m (30 layers) + output grids |
| local scratch | about 2–6 GB | staged deep-model channels + 25 m DEM mosaic |
| GPU memory, E3 / E4 | 3–8 GB | batch 8–32 at 128 px (T4: batch 8) |

## Stages expected to take longest (FULL; ESTIMATES)
1. **DEM aggregation** (section 09). It reads every DEM-covered 6.4 km chunk from roughly 21 GB of 1 m tiles over the Drive FUSE mount. Expect hours, depending on coverage and Drive throughput.
2. **SegFormer with the loss ablation** (section 13). That is 4 variants × up to 30 epochs (early stopping).
3. **Gated fusion with the mode ablation** (section 14). That is 4 variants, plus a one-time 25 m DEM mosaic in local scratch.
4. **EDA** (section 10). It covers about 300 features × block bootstrap, plus native-resolution DEM windows for the multiscale and atlas analyses.
5. **Feature computation** (section 09). Each feature is written to Drive.

## Known limitations
- Nothing has run on the real data yet. Runtimes, memory, coverage and all metrics are unknown until the SMOKE_TEST runs.
- The official DTI is unavailable: no official evaluator is installed. Every DTI is the repository **offline proxy**.
- DEM tiles with unknown or conflicting vertical units are excluded. The ingestion coverage report states how much fault coverage this costs.
- The Drive listing shows no `.xml` sidecars next to the DEM tiles, so units come from GeoTIFF metadata only (band units, tags, compound CRS).
- G07 arrays are placed only through their adjacent `manifest.json` (EPSG:32611, 200 m, origin 243350 / 4508550), after a check that this grid nests in the master grid.
- G07, M9 and P6 maps are EDA-only. Their original splits are not the G11 split, so they are never fusion members or training targets.
- The test split is reported once, after the validation-based selection. It must not be used to choose anything.
- The feature store on Drive is large (see the estimates above). Trim `feature_engineering.geophysics.scales_m` if Drive quota is short; this is a documented scientific change.
