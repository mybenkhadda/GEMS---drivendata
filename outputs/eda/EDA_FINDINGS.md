# GEMS — Geological EDA findings

_Generated 2026-10-20 by `notebooks/01_geological_eda.ipynb`. Every number below is computed from
the data; reload paths for all artifacts are in `outputs/eda/manifest.json`._

## Data frame of reference
- Working grid 1865 x 1646 px at 0.200 km/px (native 3730 x 3292),
  CRS `EPSG:32611`, 19 geophysical bands in 7 modalities.
- Mapped fault pixels: 1.215 % of the grid; median label width 1.2 px.
- 2318 skeleton segments (5541 km of trace), 342 branch points,
  median segment 1.5 km (p90 4.9 km).
- Dominant strike families: 18°, 152°.

## 1. Most informative modalities
| modality | best_profile_signal | profile_auc | best_pixel_transform | pixel_auc | station_support | unique_support |
|---|---|---|---|---|---|---|
| terrain | terrain:curvature | 0.81 | <bound method Series.transform of modality       terrain
field         det_elev
transform    tilt_edge
scale_km           5.0
sigma_px          12.5
pixel_auc     0.631332
Name: 65, dtype: object>@5km | 0.63 | 0.77 | 0.04 |
| gravity | iso_grav_anom | 0.72 | <bound method Series.transform of modality           gravity
field        iso_grav_anom
transform        log_gabor
scale_km               2.0
sigma_px               5.0
pixel_auc         0.618221
Name: 52, dtype: object>@2km | 0.62 | 0.60 | 0.01 |
| magnetic | mag_anom | 0.72 | <bound method Series.transform of modality      magnetic
field              rtp
transform    log_gabor
scale_km           5.0
sigma_px          12.5
pixel_auc     0.573112
Name: 25, dtype: object>@5km | 0.57 | 0.47 | 0.03 |
| basement | basement:thdr | 0.71 | <bound method Series.transform of modality               basement
field        depth_to_base_surf
transform             tilt_edge
scale_km                   10.0
sigma_px                   25.0
pixel_auc              0.560427
Name: 94, dtype: object>@10km | 0.56 | 0.57 | 0.01 |
| conductivity | cond_surf | 0.69 | <bound method Series.transform of modality     conductivity
field           cond_surf
transform            thdr
scale_km             20.0
sigma_px             50.0
pixel_auc        0.546236
Name: 115, dtype: object>@20km | 0.55 | 0.38 | 0.01 |

- Ranking (mean of profile and pixel AUC): **terrain > gravity > magnetic > basement > conductivity**.
- 2 % of fault stations have no modality
  above the 10 %-false-alarm threshold: these are the label positives a model will struggle to justify from the data.
- Cross-modal edge coincidence: pixels where ≥3 modalities carry a top-10 % edge are
  135.7x enriched near mapped faults (best case).

## 2. Useful transformations and spatial scales
- Best transform per modality (pixel AUC, near vs far): `terrain:tilt_edge@5km`, `gravity:log_gabor@2km`, `magnetic:log_gabor@5km`, `basement:tilt_edge@10km`, `conductivity:thdr@20km`.
- Scale preference across all modality x transform pairs (count of pairs whose AUC peaks at that scale):
  2 km: 22, 5 km: 6, 10 km: 3, 20 km: 4.
- Fault response width (FWHM of the aligned profile response, top-8 signals, 75th percentile): 9.4 km.
- The multi-modal fused lineament score recovers 14 % of mapped trace within
  2 km; its detected traces fall near mapped faults 1.4x more often than random placement.

## 3. Fault patterns likely to be difficult
- Weak / ambiguous segments: 0 % of mapped trace length has no modality above 0.6 calibrated evidence.
- Median max-evidence by geometry: straight 1.00, curved 1.00,
  intersection 1.00, short 1.00 (0.5 = background level).
- Weakest strike sector: (-0.001, 30.0]
  (median max-evidence 1.00).
- Regional contrast in mapped density (km/100 km²): min 7.8,
  max 14.1 — part of this is mapping effort, not geology.

## 4. Limitations of the existing labels
- Raster components are not fault objects: by trace length blob 0 %, fragment 29 %, network 32 %, single_trace 39 %.
  Networks merge many faults through junctions; fragments split single faults.
- 872 collinear endpoint pairs across components lie within 5 km
  (50 % of fault tips) — mapped traces are interrupted, so pixel-exact
  negatives along strike are unreliable.
- 10 coherent lineaments ≥ 6 km have no mapped fault within
  6 km. Some are contacts or dykes, but their existence means the negatives contain
  unlabelled positives (positive-unlabelled setting).
- Labels are thin lines while geophysical responses are 9.4 km wide: evaluate and train with a
  distance tolerance, not pixel-exact overlap.

## 5. Recommended feature sets and patch sizes
- Raw bands (profile-ranked): `ieq_n100a15`, `det_elev`, `det_elev_slope`, `geod_shearrate`, `mag_anom`, `iso_grav_anom`, `depth_to_base_surf`, `geod_2ndinv`.
- Enhanced layers: per-modality best transforms above, tilt of RTP and isostatic gravity at 2 and 10 km
  (`features/enhanced_best.tif`), fused lineament score/support/coherence (`features/lineament_maps.tif`),
  structure-tensor coherence per modality (`features/crossmodal_orientation.tif`).
- Context only: strain and seismicity bands are regional (tens of km); use them as low-frequency context channels.
- Never feed `features/label_context.tif` (distance to mapped faults, trace density) to a model: it is derived from labels.
- Patch: **256 px (51 km)**, stride 128 px; needs ≥ 38 km
  of context (4 x response width, median fault length, profile span).
- Validation: spatial blocks = the 3x3 regions used here (hold out whole regions).

## 6. Three immediate baseline experiments
1. **Zero-training lineament floor** — threshold the fused lineament score (and the ≥3-modality edge-agreement map)
   into the submission grid; tune only the threshold on held-out regions with a 2 km tolerant F1.
   This is the reference every learned model must beat.
2. **Pixel GBM with ablation** — LightGBM on raw bands vs raw + enhanced layers at 2/5/10 km, target = labels dilated
   by 1 px, negatives sampled > 5 km from labels, spatial-block CV. The raw→enhanced gain
   measures the value of Section 4; SHAP by modality checks Section 1.
3. **Thin-structure U-Net** — 256 px patches of the recommended channels, loss = BCE + soft-Dice/clDice on a
   distance-tolerant target, down-weighted negatives on top-ranked unmapped lineaments (PU-aware), region hold-out.

## Top unmapped candidates (exploratory, not confirmed faults)
| rank_unmapped | length_km | azimuth | structural_score | evidence_score | n_supporting_modalities | x_mid | y_mid |
|---|---|---|---|---|---|---|---|
| 1.00 | 8.41 | 178.82 | 0.81 | 0.71 | 2.00 | 336250.00 | 4442250.00 |
| 2.00 | 7.01 | 3.78 | 0.57 | 0.58 | 2.00 | 432050.00 | 4498250.00 |
| 3.00 | 15.30 | 122.30 | 0.67 | 0.89 | 3.00 | 436450.00 | 4188050.00 |
| 4.00 | 20.21 | 177.59 | 0.52 | 0.73 | 2.00 | 391650.00 | 4501250.00 |
| 5.00 | 6.81 | 122.88 | 0.66 | 0.53 | 2.00 | 410450.00 | 4196050.00 |
