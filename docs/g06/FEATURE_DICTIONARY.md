# G06 feature dictionary

Every channel is label-free. Names are deterministic (`feature_store.group_names`); the authoritative list with
units, source, transform, scale, coverage and the `usable` flag is `features/feature_registry.parquet` of a run.
Scales are Gaussian smoothing widths (100 / 300 / 700 / 1500 m = 1 / 3 / 7 / 15 px) - **not source depths**.
Missing data: normalised convolution (gaps never read as zeros); a feature is NaN where the source pixel is missing
or the kernel support is below 25 %.

## raw (`raw/raw_model.tif`) and raw_physical (`raw/raw_physical.tif`)

| feature | meaning |
|---|---|
| `phys_<band>` | observed value (QC / plots / segment field steps; never a pixel-model input) |
| `raw_<band>` | fixed, declared transform of the observation: identity, asinh(x / s) for magnetics (large dynamic range, sign kept), log1p for event counts, signed log for conductivity |
| `miss_<band>` | 1 where the band is missing inside the footprint (only for bands whose gaps differ from the footprint) |
| `raw_n_missing` | number of missing bands at the pixel |

Bands (exact identifiers): magnetic `mag_anom, rtp, tmi, tmi_hg, tmi_vg, tc`; gravity `iso_grav_anom,
iso_grav_anom_hg, iso_grav_anom_vg, iso_grav_anom_slope`; terrain `det_elev, det_elev_slope`; basement
`depth_to_base_surf`; conductivity `cond_surf`; strain `geod_2ndinv, geod_shearrate, geod_dilaterate`; seismicity
`deq_n100a15, ieq_n100a15`. Units are declared in the YAML and must be checked against the band descriptions.

## physics (magnetic, gravity, multiscale, *_spectral)

| suffix | geophysical meaning |
|---|---|
| `hgm_s` | horizontal gradient magnitude: maxima over steep density / magnetisation contrasts (contacts, faults) |
| `ocos_s / osin_s` | axis of the gradient as a doubled angle (continuous over 0 / 180 deg) |
| `lap_s, hl1_s, hl2_s` | Laplacian and Hessian eigenvalues: curvature of the field (ridges, troughs, saddles) |
| `d2n_s / d2t_s` | second derivative across / along strike: sharpness of a contact vs bending along it |
| `curv_s` | compressed iso-line curvature (straight vs curved anomaly contours) |
| `edge_s` | thin gradient-maximum edge (Canny-type) |
| `persist` | share of scales at which the pixel is such an edge: a boundary that persists across scales is deeper-rooted or sharper |
| `ms_<field>_dog<a>_<b>` | log energy of the band-pass (difference of Gaussians) signal: local short- vs long-wavelength content |
| `ms_<field>_specslope` | finest minus coarsest band energy (roughness) |
| `ms_det_elev_*, ms_depth_to_base_surf_*` | terrain / basement gradient magnitude and edges (scarps, basement steps) |
| `*_tilt` | tilt derivative atan(VDR / THDR): normalised, amplitude-independent edge detector (zero contour over contacts) |
| `*_asig` | analytic-signal amplitude: maximal over source edges, little dependence on magnetisation direction |
| `*_tiltzero` | closeness to the tilt zero contour |
| `*_resid_uc<h>km` | field minus its upward continuation by h km: residual (shallow, short-wavelength) anomaly |

## texture

Masked window statistics of `rtp`, `iso_grav_anom`, `det_elev`: `std`, `range`, `mean`, `q10/q50/q90`
(local variability and contrast), `ent` (entropy of the value distribution: disordered vs homogeneous ground),
`edged` (density of thin edges), `gvar` (variability of the gradient), `aniso` (local anisotropy), `oh<j>`
(share of gradient energy per orientation bin).

## structure (`struct_*`)

| suffix | meaning |
|---|---|
| `coh` | structure-tensor coherence: 1 = one dominant orientation (lineament), 0 = isotropic |
| `ocos / osin` | dominant gradient axis (doubled angle); strike = axis + 90 deg |
| `eigratio` | l2 / l1 of the tensor |
| `ridge / valley` | scale-normalised bright / dark line strength (Hessian) |
| `along` | gradient continuity along strike relative to across: a lineament continues along itself |
| `cross` | field step across the local gradient direction over sigma (contrast across a contact) |
| `trunc` | asymmetry of edge strength ahead / behind along strike: an anomaly that stops abruptly (truncation proxy, not a measured offset) |
| `dircoh` | persistence of the orientation across scales |
| `x_<a>__<b>_agree / agreew` | axial agreement A = (1 + cos 2(theta_a - theta_b)) / 2 of two fields, and weighted by both coherences: co-aligned magnetic / gravity / terrain structure |
| `disc_count / disc_frac` | number / share of fields with a thin edge at the pixel: co-located discontinuities |

## context

Regional (stored): `ctx_<f>_regtex` (regional texture), `ctx_det_elev_rough` (terrain roughness), `ctx_seis_log`
(smoothed seismicity), `ctx_geod_*_asinh` (strain-rate indicators), `ctx_depth_to_base_surf_reg / _relief`
(basement depth / relief), `ctx_dist_nodata_km`, `ctx_domain` (unsupervised k-means regime).

Permitted known-fault context (per unit, never stored globally, never a pixel-model input): `known_dist_km`,
`known_density`, `tip_dist_km`, `tip_along_cos`, `known_orient_cos2 / sin2`, `known_net_density` - computed from
the faults VISIBLE to the unit (nested pairs: also without the outer unit's truth).

## Segment columns (D3)

`seg_*` geometry (length, orientation, linearity, straightness, degree), `on_<pool>` pool membership, `geo_mean_* /
geo_std_*` store means along the segment, `geo_jump_* / geo_jumpcons_*` field step across the segment and its sign
consistency, `ori_<field>_agree` agreement of the segment with the field's strike, `kn_*` permitted known context,
`ev_*` honest W2 / CNN / G06 evidence (S2 / geology-model columns), `unc_*` uncertainty.

## Label-free footprint statistics (`fit_scope = full_footprint_unsupervised`)

Texture-entropy bin edges, strain asinh scales (median |x|) and the k-means domains use statistics of the whole
footprint. They read no label but are transductive; models that must be strictly training-only fit their own
scaling inside the folds (`models_patch.FoldScaler`).
