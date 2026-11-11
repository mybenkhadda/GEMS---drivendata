# G11 master Colab execution (recovery-first)

**Notebook:** `notebooks/colab/GEMS_G11_MASTER_COLAB_EXECUTION.ipynb` (sections 00–20). The design is in [RECOVERY_FIRST.md](RECOVERY_FIRST.md).

## Launch
1. Push the code to the branch in `CONFIG["ref"]` (`g11-colab-production`). §02 stops with the deployment commands if the branch lacks the recovery-first modules.
2. Open `https://colab.research.google.com/github/mybenkhadda/GEMS---drivendata/blob/g11-colab-production/notebooks/colab/GEMS_G11_MASTER_COLAB_EXECUTION.ipynb` on a **GPU** runtime (T4, L4 or A100).
3. Keep the defaults and run all cells:
   - `run_mode="FULL_RESUME"`
   - `feature_policy="REUSE_VERIFIED"`
   - `feature_version="g11_features_v1"`
   - `dem_statistics_resolution_m=10.0`
   - `split_version="v2"`
4. After a disconnect, run §00–§03, then the interrupted section. Completed stages, verified layers and committed DEM chunks are reused.

## Values you may need to edit (§00)
| key | when |
|---|---|
| `ref` | Branch or commit that contains the code. |
| `drive_root` | If the data is not at `/content/drive/MyDrive/GEMS_2`. |
| `run_mode` | `RECOVER_ONLY` to inspect only; `PROCESS_FEATURES_ONLY` / `FINALIZE_FEATURES_ONLY` for feature work; `SPLITS_ONLY`, `TRAIN_ONLY`, `INFER_ONLY` for model work (these never extract features). |
| `feature_time_budget_min` | To split the 10 m DEM processing over several sessions. |
| `dem_statistics_resolution_m` + `feature_version` | `2.0` + `g11_features_legacy_r2m` uses the verified 2 m legacy layers with no computation. |
| `feature_policy` | `RESUME_MISSING` for a full re-verification and repair. `FORCE_NEW_VERSION` only with a new `feature_version`. |
| `split_version` | A new name for a different split; the features stay reused. |
| `models` | Drop `segformer` / `gated_fusion` on a CPU runtime. |

Advanced YAML settings:
- `spatial_splits.v2` (regions, candidates, gates);
- `dem.engine_chunk_cells`;
- `dem.engine_use_overviews`;
- `recovery.*`;
- `training.loss_ablation`;
- `models.gated_fusion.mode_ablation`.

## Resource requirements: ESTIMATES (not measured on the real data)
| item | estimate | basis |
|---|---|---|
| §04 first recovery | 5–20 min | reads ~651 × 49 MB = 32 GB once from Drive at 30–100 MB/s |
| §04 later launches | < 2 min | `stat` of each layer + 8 sampled re-hashes |
| §05 10 m DEM | 20–60 min | 392 usable tiles (~12 GB) read once into the local cache, then ~250 chunks of 12.8 km computed locally |
| local disk | 2–5 GB | 10 m tile cache (~4 MB per tile) + staging + deep-model channels |
| new Drive data | ~3.0 GB | 59 layers × 49 MB + chunk files |
| peak RAM | 4–6 GB | dataset finalisation reads headers only; DEM assembly holds 42 grids × 49 MB ≈ 2 GB |
| GPU memory, E2 / E3 | 3–8 GB | batch 8–32 at 128 px |

## Longest stages (ESTIMATES)
1. §05 DEM tile cache + chunks (first launch only).
2. §13 SegFormer with the loss ablation (4 variants).
3. §14 gated fusion with the mode ablation (4 variants).
4. §04 first verification read (first launch only).
5. §08 EDA.

## Known limitations
- Nothing has run on the real data yet. Every estimate above is unmeasured.
- The official DTI is unavailable. Every DTI is the repository's offline proxy.
- **Legacy adoption** of the 2 m layers depends on:
  - the 2 m resolution recorded in the YAML and notebook default at `1618b0e` (it is not stored in the files);
  - a successful `git show` of that commit's config. If it fails, the layers are classified INCOMPATIBLE unless `recovery.accept_unverified_legacy_config` is set.
- 264 DEM tiles of unknown vertical unit stay excluded (12.3 % of the DEM-covered fault pixels).
- If no split candidate passes the gates, §09 stops with the failing gates (`split_gates_FAILED.parquet`). Adjust `spatial_splits.v2` (more regions or candidates) and choose a new `split_version`.
