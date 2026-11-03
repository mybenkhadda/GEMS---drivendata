# G07 feature dictionary (DEM / surface)

Subsurface features are the G06 store (`docs/g06/FEATURE_DICTIONARY.md`), reused unchanged. DEM features are computed
at the work resolution (fine tier, default 2 m) or on the merged 10 m DEM (coarse tier), then aggregated to the 100 m
grid (`mean`, `max`, orientation as doubled-angle `_cos2 / _sin2 / _resultant`). Scales are smoothing widths in metres
(fine 4 / 10 / 30 m, coarse 60 / 150 / 300 m); scales finer than 1.5 x the resolution are dropped. Every feature is
label-free; `dem_coverage` (share of the cell with valid DEM) accompanies every product.

| id | unit | meaning (geomorphology) |
|---|---|---|
| `dem_slope_s` / `dem_slope_max_s` | deg | surface gradient: scarps and fault-line escarpments are steep linear bands |
| `dem_hs<az>` | [0, 1] | Lambertian hillshade, sun altitude 45 deg (visual lineament detection depends on illumination) |
| `dem_hs_range` | [0, 1] | max - min hillshade over azimuths: strongly direction-dependent relief = linear structures |
| `dem_curvprof_s` / `dem_curvplan_s` / `dem_curvmean_s` | 1/m | profile (slope breaks: scarp crest / toe), plan (convergence: drainage), mean curvature |
| `dem_relief_w` / `dem_relief_max_w` | m | local relief in a window |
| `dem_tpi_s` | m | topographic position |z - G_s(z)| (ridges / valleys at scale s) |
| `dem_rough_s` | m | roughness of the detrended surface |
| `dem_openpos` / `dem_openneg` | deg | positive / negative openness (exposure of crests / incision of valleys) |
| `dem_ridge_s` / `dem_valley_s` (+ `_max`) | m | scale-normalised Hessian line strength |
| `dem_aniso_s`, `dem_orient_s` | 1, axial | structure-tensor coherence and doubled strike angle of the topographic grain |
| `dem_scarp_s` / `dem_scarp_max_s` | m | height step across the local gradient over +/- s, x coherence: scarp height proxy |
| `dem_scarpmask_s` | share | slope >= 12 deg & coherence >= 0.55 & step in [0.5, 30] m |
| `dem_lin_s`, `dem_lin_orient_s` | share, axial | thin ridge / valley lines (non-maximum suppression across strike) and their orientation |
| `dem_along_s` / `dem_trunc_s` | ratio / [0, 1] | continuity of the slope field along strike / abrupt termination (truncation proxy) |
| `dem_scarp_persist_<tier>` | share | share of scales at which the cell is a scarp |
| `dem_dircoh_<tier>` | [0, 1] | orientation persistence across scales |
| `dem_valleyline_<tier>`, `dem_valley_orient_<tier>` | share, axial | valley-line density / orientation (drainage alignment; an offset drainage is NOT inferred) |
| `x_dem_<field>_agree` | [0, 1] | axial agreement of the DEM strike with the G06 magnetic / gravity strike (300 m) |

Interpretation guard: these are observations. Erosional, anthropogenic (roads, field edges, mining) and lithological
contrasts also produce scarps and lineaments; continuity, persistence, cross-modal agreement and the learned models
weigh competing interpretations. Curvature / truncation are not displacement measurements.

Candidate columns (D1): `seg_*` geometry + expert count, `on_<pool>` provenance, `geo_mean_*` / `geo_std_*` means of
G06 and DEM channels along the candidate, `geo_jump_*` field step across it, `ori_*` agreement with the geophysical
strike, `kn_*` permitted known-fault context (nested for training units), `ev_<model>_rank_*` honest model evidence,
`unc_*` uncertainty.
