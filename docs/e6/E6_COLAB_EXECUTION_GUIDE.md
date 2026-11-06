# E6 Colab execution guide

* **Notebook:** `notebooks/colab/GEMS_E6_SUBMISSION_MILESTONE.ipynb` (branch `e6-submission-fasttrack`)
* **Config:** `configs/experiments/e6_masked_threeway_full.yaml`
* **Code in Colab:** `/content/GEMS_E6` | **local work:** `/content/gems_work` | **Drive:** `/content/drive/MyDrive/GEMS_2`

Nothing in this guide has been executed. The notebook clones the branch from GitHub, so push the fixes before you run.

## 1. Before the first run

1. Commit and push the audit fixes to `e6-submission-fasttrack`. Without this the notebook runs the old code.
2. Colab: Runtime -> Change runtime type -> **GPU**. High RAM is recommended.
3. On Drive, the inputs of `E6_FINAL_ARTIFACT_MANIFEST.md` (Inputs table). The two that matter most:
   * `GEMS_2/outputs/g07_multiscale_fault_discovery/g07-*/dem_features/coarse_parts/*/*.tif`: the cached 10 m DEM parts;
   * `GEMS_2/experiments/e1_gsn_parity__b8b8c3db/checkpoints/cv_fold{0,1}_model.pt`: E1 warm start (fold 2 absent).
4. Decide `e6.usgs_dem_license_verified` (yaml). Export reads it as your statement.

## 2. Run

Runtime -> Run all. Cell order:

| cell | does | stops when |
|---|---|---|
| 00 | parameters | - |
| 01 | mount, clone, `pip install -e .[colab]`, stage raw data / folds / datasets / EDA metadata | Drive not mounted, clone fails |
| **01b** | **read-only preflight**: prints every artifact as AVAILABLE_VALID / DERIVABLE_MISSING / BLOCKED_MISSING | any BLOCKED item |
| 02 | `EXP.prepare()`: DEM bank from the cached parts, channel caches, spec, architecture record | missing parts, a failed part, unknown channel |
| 03 | `EXP.train()`: 3 geographic folds + 3 withheld-system splits | spec / warm-start mismatch, OOM after retries, collapse |
| 04 | `EXP.evaluate()`: `metrics/e6_dti.csv` | a missing fold / split raster |
| 05 | `EXP.export()`: full-region fold maps, average, exact-k selection, GeoTIFF, readback, Drive checksum | incomplete CV, stale models, any gate |
| 06 | checksum check and download of the new E6 file | no verified file |

Cache folders (all on Drive, reused):

* `outputs/e6_pipeline/cache/<fp>/`: DEM tiles, bank, manifest;
* `outputs/e6_pipeline/features/<fp>/`: DEM channel `.npy`;
* `experiments/<id>/checkpoints/`: resume and final models;
* `experiments/<id>/predictions/`: unit and full-region maps.

## 3. Resume after a disconnect

Reconnect a GPU runtime and Run all again with the same config:

* the preflight shows the experiment state (final models, resume files, submissions);
* finished stages are skipped;
* finished units are skipped when their final model matches the spec hash;
* an interrupted unit continues from its last epoch checkpoint (`ckpt_every: 1`, Drive sync every 2 min);
* the DEM bank and channel caches are reused when the fingerprint matches;
* full-region fold maps are reused only for the same checkpoint.

These stops are deliberate:

| message | cause | action |
|---|---|---|
| `written by a different spec … refuse to resume` | config or DEM bank changed while a unit was in progress | revert the change, or delete that unit's `*_ep*.pt` / `*_latest*` files |
| `resume state was warm-started but the E1 checkpoint is now absent` | E1 folder changed between sessions | restore it, or delete that unit's resume files |
| `E1 warm start refused` | E1 channel groups or order differ from E6 | fix the E1 path; never bypass it |
| `E6 DEM: n cached part(s) failed` | an unreadable part | see `outputs/e6_pipeline/cache/<fp>/tile_log.csv`; fix or remove the part |

## 4. Final E6 submission

```text
GEMS_2/experiments/e6_masked_threeway_full__<hash8>/submissions/submission_e6_geostructurenet.tif
GEMS_2/experiments/e6_masked_threeway_full__<hash8>/submissions/submission_e6_geostructurenet.json
GEMS_2/experiments/e6_masked_threeway_full__<hash8>/reports/e6_submission_manifest.json
```

A later export does not overwrite an existing file. It adds `__<timestamp>` to the name, and `OUTPUT["path"]` always names the new one.

The file is a binary exact-k selection: the top `e6.submission_fraction` (1 %) of the eligible domain, ranked by the fold-averaged E6 probability. That fraction is predeclared, with no threshold search. The continuous map is `predictions/e6_full_probability.tif`.

Upload it manually. Nothing uploads automatically, and offline DTI is not a leaderboard score.
