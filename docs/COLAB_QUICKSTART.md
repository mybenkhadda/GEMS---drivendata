# Colab quickstart

## Once

1. Push this repository to GitHub (private is fine). In Colab: *Secrets* → add `GITHUB_TOKEN` (a fine-grained,
   read-only token for this repository) if it is private.
2. Upload data and the Phase 1–4 artifacts to Drive as in `docs/GOOGLE_DRIVE_LAYOUT.md` (Drive for desktop is easiest).
3. Optional: put the USGS URL JSON at `MyDrive/GEMS_2/data/raw/dem_1m/tnm_urls.json`.

## Every session

1. Runtime → Change runtime type → GPU.
2. Open a notebook from `notebooks/colab/`, set `REPO = 'owner/name'` and `COMMIT` (a SHA for reproducible runs),
   run the first cell (mount, clone/checkout, `pip install -e .[colab]`, env vars, staging).
3. Run the experiment cells. Interrupted? Reconnect and re-run the same cells: finished stages and units are skipped.

## Execution order

| notebook | what | GPU | rough cost |
|---|---|---|---|
| 00 | setup, environment report | – | 2 min |
| 01 | stage caches, build profile features | – | 30–60 min CPU |
| 02 | A2 parity · A1 + A4 gated-encoder discovery + scoreboard · A3 designs | A1 | ~40 min |
| 03 | B0 → B1 → B2 (×3 priors) → (B3) → C1 → C2 → C3 | yes | ~1 h per run |
| 04 | E1 → E2 → E3 → E4 → E5 | yes | ~1 h per run |
| 05 | F1 inventory + pilot → approve download → F3 → F4 | F4 CNN | tiles: 5–10 min each CPU |
| 06 | G ensembles, recovery atlas, promotion | – | 30–60 min CPU |
| 07 | champion full-raster inference + GeoTIFF export | yes | 10 min |

## Short sessions

Set `runtime.max_stage_minutes` (e.g. `200`) in `configs/colab/runtime.yaml` or in a copy of the experiment YAML.
`runtime` is not part of the config hash, so the experiment id does not change. Training stops at an epoch boundary,
writes a checkpoint, pushes it to Drive, and marks the stage `interrupted`; the next session resumes it.

## Variable GPUs

`training.gpu.autoconfig` keeps the effective batch fixed and adapts batch / accumulation / precision (bf16 on Ampere+,
fp16 + GradScaler on T4 / V100), and whether the stack lives on the GPU. A CUDA OOM halves the batch and doubles
accumulation (up to 2 times) instead of crashing. Override under `runtime.gpu` if needed.
