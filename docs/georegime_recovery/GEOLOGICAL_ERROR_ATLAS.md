# Geological error atlas (sections 09, 11, 12, 13)

## Section 09: spatial TP / FP / FN
Each discovery fold is scored with the full-area protocol, primary exclusion. `metrics.ComponentMaps` produces three pixel maps:
* `tp_support` = M(g) on held truth;
* `fp_mass` = p(1 - k) on scored pixels;
* `fn_support` = 1 - M(g).

Their sums are checked against `MaskedScorer`. The maps are saved as `03_validation/tp_fp_fn_spatial/<unit>_{tp_support,fp_mass,fn_support}.tif`.

Aggregates:
* `regional_metrics.csv`: per fold × geographic zone, plus "outside zones";
* `block_metrics.csv`: 10 km blocks;
* `fp_by_visible_distance.csv`: mean p and FP mass vs distance to training-visible traces. This is a diagnostic only;
* `fault_normal_profiles.csv`: mean p vs perpendicular distance from held and from visible traces, 0–3 km;
* `error_regions.csv`: see below.

Regions per fold:
* **FP regions**: 8-connected pixels with p ≥ the fold's q99 of scored pixels and more than 300 m from held truth (k = 0, so the full p counts as FP), at least 5 px. The 20 largest by FP mass are kept.
* **Missed systems**: held components with weighted recall < 0.2.
* **Correct systems**: held components with recall ≥ 0.5.

## Section 11: label-free evidence
Layers are read through `GeologicalAtlas.load_feature`, preferring the IDs listed in `geological_evidence.EVIDENCE`:
* magnetic and gravity boundary strength;
* edge persistence;
* magnetic THD after 0.5 / 1 / 2 / 5 km continuation;
* gravity THD after 5 km continuation and the long-wavelength field;
* terrain slope break (1 km);
* `xm_evidence_mean`, `xm_multiscale_persistence`, `xm_continuity`, `xm_orient_magn_grav`, `xm_evidence_count`, `xm_available_modalities`;
* DEM coverage.

Continuous layers become percentile ranks over the eligible footprint. A missing layer is listed in `evidence_availability.csv` and the limitations; it is never treated as weak evidence. DEM evidence is used only where a DEM coverage raster exists.

Per region the analysis records evidence means, distance to training-visible traces, distance to held truth (for FPs), distance to the footprint edge, DEM coverage, and descriptive flags:
* strong magnetic / weak gravity, or the reverse;
* terrain structure without potential-field support;
* multiscale continuity;
* orientation agreement;
* near a visible trace (possible extension or branch);
* close to held truth but outside the 300 m kernel (poor alignment);
* confidence present within 1 km of a missed trace;
* near the survey edge (possible artifact);
* little or no DEM coverage;
* fewer than two modalities.

Every flag means "consistent with". None of them confirms a fault.

Tables in `05_geological_evidence/`:
* `geological_disagreement.csv` (regions and flags);
* `flag_frequency.csv`;
* `evidence_class_profiles.csv` (supported truth / missed / high-FP / background);
* `confidence_vs_evidence.csv` (response curves with a 90% bootstrap band over 1 km block samples).

## Section 12: structural strata
* `fault_length_recall.csv`: one row per held component with length, class (< 3 km, 3–10 km, ≥ 10 km), linearity, strike and recall.
* `length_class_recall.csv`, `linearity_recall.csv`.
* `strike_agreement.csv`: structure-tensor strike of the smoothed held trace vs the smoothed prediction at supported truth (coherence > 0.2).
* `regional_recall.csv`: 10 km blocks; blocks with fewer than 20 truth pixels are not reported.
* Strike convention: azimuth clockwise from north, axial.

## Section 13: candidate lineaments
`candidate_model_agreement.csv` covers all Atlas proposals:
* the V1 label-free support formula (0.5 continuity + 0.3 orientation agreement + 0.2 persistence) and whether the V1 review filter passes;
* mean / q90 / max of the V1 full-model confidence along the rasterised trace, its percentile rank, and the mean of each discovery fold's full-area prediction;
* the agreement class;
* the Atlas limitations.

Proposals stay "UNVERIFIED geological structure". They are never used as labels or as submission inputs.
