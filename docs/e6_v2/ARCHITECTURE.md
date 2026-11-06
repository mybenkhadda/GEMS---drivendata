# E6 V2 architecture

Status: authored 2026-11-06, **never executed** (no local run, no Colab run).

## Package layout (`gems_bench/e6_v2/`)

| module | responsibility |
|---|---|
| `config.py` | load + validate `configs/e6_v2/e6_v2.yaml`, section / module hashes |
| `paths.py` | the namespace `<drive_root>/e6_v2_from_raw/` and all its folders |
| `cache.py` | atomic writes (npy, npz, json, csv, GeoTIFF), checksums, stage manifests |
| `contracts.py` | raw band contract, channel lists of the three branches, schema hash |
| `raw_data.py` | GeoDAWN reader, footprint, labels, template |
| `inventory.py` | raw-data inventory (json + csv), raw fingerprint |
| `preflight.py` | header checks, single blocker report |
| `fft_features.py` | wavenumber-domain derivatives / continuation |
| `geophysics.py` | base masks, terrain branch, potential-field branch |
| `dem_sources.py` | DEM tile discovery, header, vertical unit, overlap order |
| `dem_mosaic.py` | window plan, windowed mosaic at 2 m with halo, seams |
| `dem_features.py` | 2 m operators |
| `dem_aggregation.py` | per-window 100 m statistics, coverage, resume, assembly |
| `normalization.py` | robust quantile normalisation |
| `datasets.py` | normalised stacks, per-unit targets, patch sampler |
| `spatial_splits.py` | geographic folds, discovery splits, unit masks |
| `augmentation.py` | dihedral augmentation with the axial sign rule |
| `encoders.py`, `fusion.py`, `context.py`, `decoder.py`, `heads.py`, `model.py` | the network |
| `losses.py` | nnPU, soft Tversky, proximity, orientation |
| `checkpoint.py` | resumable checkpoints, identity checks, immutable reference |
| `trainer.py` | stages A-E per unit |
| `metric.py` | canonical DTI |
| `inference.py` | tiled inference |
| `evaluator.py` | DTI tables, slices, short-fault recall, gate statistics |
| `submission.py` | finalise, area budget, template validation |
| `figures.py` | figures from saved tables |
| `artifacts.py` | artifact dependency graph + validation functions |
| `pipeline.py` | `E6V2Pipeline` |

No module imports another `gems_bench` / `gems_eda` module. One caveat: importing `gems_bench.e6_v2` executes
`gems_bench/__init__.py`, which imports the old `gems_bench.config` / `gems_bench.data` modules (nothing from them
is used). That is why the notebook installs the repository package (`pip install -e .[colab]`).

## Network (`model.py`)

Inputs per 256 x 256 px patch (25.6 km at 100 m):

| branch | channels | encoder |
|---|---|---|
| terrain | 14 (FEATURE_SCHEMA.md) | `Encoder(14, 32)`: widths 32 / 64 / 128 / 256 at 256 / 128 / 64 / 32 px |
| field | 44 = 17 magnetic + 17 gravity + 10 subsurface | `Encoder(44, 32)`: same widths |
| DEM | 22 statistics | `Encoder(22, 16)` (half width): 16 / 32 / 64 / 128 |
| coverage `m` | 1, measured, in [0, 1] | **not an encoder input**: only gates and fallback |
| context | 58 = terrain + field, 8 x 8 mean, 128 x 128 crop (102 km) | `ContextBranch` -> 32 ch at 32 x 32 |

Each encoder level = stride-2 conv + GroupNorm + SiLU + residual block (level 0: stem + residual block).

**Masked three-way gate (every level l)** - `fusion.MaskedThreeWayGate`:

```
s_T, s_F = conv1x1([t, f])                                  (from terrain + field only)
a        = conv1x1(d)                                       (DEM width -> level width)
s_D      = conv1x1([t, f, a]) + log(clamp(m_l, 1e-4));  s_D = -inf where m_l == 0
w        = softmax([s_T, s_F, s_D])                         (float32)
fused_l  = w_T P_T(t) + w_F P_F(f) + w_D P_D(a)             (P = 1x1 projections)
```
`m_l` = coverage average-pooled to the level. The DEM logit bias starts at `model.dem_gate_bias = -2`, its weights at 0.

Bottleneck: 4 residual blocks with dilations 1, 2, 4, 8 at 32 x 32, concatenated with the context features and
merged by a 1 x 1 conv. Decoder: U-Net, bilinear up-sampling, 1 x 1 reduction, concatenation with the fused skip,
residual block, up to 256 x 256 x 32. Heads: fault logit; proximity (sigmoid, min(d, 300 m) / 300 m); orientation
(tanh, doubled-angle strike) - proximity / orientation are used only in the loss.

**Reference and exact fallback.** `ReferenceNet` is the same network without the DEM encoder and with two-way
gates. It is trained from scratch per unit, frozen and stored immutable. `E6System` returns
`where(m > 0, E6 logit, reference logit)` at full resolution, so pixels without measured DEM coverage are the
reference prediction bit for bit (checked by `trainer.parity` after stage E: `E_parity.json`). Because the
three-way gate computes `s_T, s_F` from `(t, f)` only, an E6Net initialised from the reference with a closed DEM
gate reproduces the reference logits (checked in stage C: `C_parity.json`, tolerance 1e-3).

## Data flow

```
raw GeoDAWN tif ─┬─ base masks ─┬─ terrain features ───────────┐
                 │              ├─ FFT potential + subsurface ─┤
fault tif ───────┤              │                              ├─ datasets ─┬─ trainer (A-E, 6 units) ─ evaluator
template tif ────┘              │                              │            └─ full inference (3 folds) ─ submission
DEM 1 m tiles ── inventory ─────┴─ windows (2 m) ─ 100 m DEM ──┘
                                     splits (from base masks) ──┘
```
