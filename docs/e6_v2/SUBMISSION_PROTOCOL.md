# Submission protocol

## Prediction
`predictions/e6_full_probability.tif` = arithmetic mean of the three geographic-fold E6 systems
(`export.fold_combination: mean`), each predicting the whole grid with 256 px tiles, 50 % overlap, sin^2 blending.
Discovery models are for evaluation only.

## Two maps, one final file (`submission.py`)
Finalise (both): NaN where the template is not valid, 0 inside the template but outside the data footprint, float32
in [0, 1] elsewhere, written on the template's CRS / transform / shape.
- `submissions/e6_continuous.tif`: the probabilities.
- `submissions/e6_area_budget_binary.tif`: 1 on the top n pixels, 0 elsewhere, n = round(f x template-valid
  footprint pixels). f (`export.area_fraction_rule: training_label_density`) = mapped-fault pixels / footprint pixels
  x `density_factor` 1.0, clipped to [0.002, 0.05]. Ties: stable raster order. Mapped fault pixels are not ranked
  (`known_faults_in_budget: false`): the organisers mask them in the hidden test, so budget spent there cannot score.
- `submissions/submission_e6_final.tif` = the map named by `export.final` (**default `area_budget_binary`**).

The policy and f are fixed in the config before the run and are not tuned on validation DTI. Section 12 reports the
pooled DTI of both policies (the binary rule applied inside each scored region with that unit's training label
density); if you change `export.final` after reading those numbers, that is a selection on validation data - note
it.

**Unverified rule.** The DTI definition takes p in [0, 1], so continuous values are scorable by the metric. Whether
the competition accepts non-binary values was not checked against the official submission-format page in this
session. Read it before uploading and set `export.final` accordingly.

## Validation (section 14 and again in 15)
For each written file: 1 band, float32, shape / CRS / transform equal to the template, NaN pattern equal to the
template's invalid pixels, values in [0, 1], binary values for the binary map, read-back equal to the array that was
written, sha256 recorded. `submission_validation.json` holds every check; `submission_sha256.txt` the hashes.
Section 15 re-opens the final file, repeats the checks and compares the sha256 with the export record.

## Upload
Never automatic. Section 16 checks the sha256 once more and downloads the file; upload it manually on DrivenData.
External data: the USGS 3DEP 1 m DEM is public-domain US Government data; confirm the competition's external-data
rules yourself.
