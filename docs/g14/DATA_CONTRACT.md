# G14 data contract

Implemented in `gems_bench/g14/contract.py`. Schema version `g14.channels.v1`. **UNTESTED.**

## 1. Grid

The template is `example_submission.tif`: 3730 x 3292 px, 100 m, EPSG:32611 (checked by `g13.contracts.check_grid`).
The footprint is `data.valid`. Every stored array uses this grid at level factors 1, 2 and 8 (100 / 200 / 800 m),
with shape `ceil(H/f) x ceil(W/f)`.

## 2. Channel record

Each channel is a `Channel(name, modality, source, kind, resample, flip_safe, formula)`:

* `source`: `raw` (GeoDAWN band), `g13` (G13 feature store), `g14` (composite computed here), `known` (per-unit
  known-fault context, never stored globally).
* `kind`: `scalar | axial_cos | axial_sin | vec_x | vec_y`. Axial and vector channels always come in adjacent
  pairs, and augmentation transforms them correctly.
* `resample`: `mean` (block nanmean) for every stored channel.
* `flip_safe`: False for TMI, mag_anom, tmi_vg and tmi_hg. These are excluded by default, because D4 augmentation
  of an inclined-field anomaly is not physical. RTP is kept.

Modalities: magnetic, gravity, terrain, conductivity, geodesy, depth, seismicity, structural, known_context.

## 3. G14 composites (formulas in `G14_FORMULAS`)

| name | definition (on robust-scaled inputs) |
|---|---|
| aspect_x / aspect_y | unit DEM gradient vector (vector pair) |
| curvature, roughness | Laplacian of smoothed DEM; local std of detrended DEM |
| MSDI (magnetic structural discontinuity) | mean of magnetic edge / tilt / horizontal-gradient features |
| GSDI (gravity) | same for gravity |
| TLI (terrain lineament) | mean of terrain ridge / valley / edge responses |
| CSCI (cross-sensor coherence) | geometric mean of present-modality edge strengths |
| cond_depth_alignment | product of conductivity gradient and depth-to-basement gradient |

A composite is built only when all of its inputs exist. Skipped composites are listed in `channel_decisions.csv`.

## 4. Storage

`ChannelStack` directory:

* `stack_l{f}.npy` (float16 memmap, C x h x w; NaN outside coverage);
* `mask_l{f}.npy` (uint8 availability per modality);
* `manifest.json` (channels, modalities, factors, prescale, schema hash, sources);
* `channel_decisions.csv`.

The stored prescale is label-free (global median / IQR over valid pixels). It only puts values in float16 range.

## 5. Normalisation (train-only)

`UnitNormalizer` is fitted per unit on at most 200 000 pixels of `unit.train`: median / IQR for scalars, identity
for axial / vector channels, clip ±8. It is saved with every checkpoint and restored at inference. Values outside
coverage become 0 after normalisation, and the mask carries the absence.

## 6. Known-fault context (per unit)

Six channels are computed from the unit's **permitted** known-fault map: dist_km, density, tip_dist_km,
tip_along_cos, strike_cos2 and strike_sin2. The sources are `p6.stack.known_features` plus the nearest-trace strike.
The permitted map is given in SPATIAL_VALIDATION.md. Ablation `A_feat_no_known_context` removes them.

## 7. Targets (`multiscale.unit_targets`, from visible labels only)

* trace (binary);
* soft (Gaussian of distance);
* dist_px;
* cos2 / sin2 (axial strike of the nearest trace);
* endpoint;
* junction.

The loss mask is `unit.train & valid & ~ignore`. Negatives farther than `reliable_negative_px` are "reliable".
Nearer unlabeled pixels are down-weighted (PU-aware).

## 8. Contract checks (`check_contract`)

* grid;
* factor shapes;
* NaN outside footprint;
* finite inside coverage;
* paired kinds adjacent;
* no flip-unsafe channel when D4 is on;
* schema hash matches manifest;
* every modality has at least one channel.

Results are written as a table with PASS / WARN / FAIL. A FAIL stops the notebook.
