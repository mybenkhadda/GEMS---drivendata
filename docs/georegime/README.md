# GeoRegime V1 — Atlas-informed shared GeoStructureNet

**Notebook:** [GEMS_GEOREGIME_V1_FULL_PIPELINE.ipynb](../../notebooks/colab/GEMS_GEOREGIME_V1_FULL_PIPELINE.ipynb)

**Status:** Implemented on main. Source-level review completed; real GPU training,
Colab runtime completion, DTI and leaderboard scores have **not** been measured.

## Why this model

The completed GEMS Geological Data Atlas shows relatively more high-wavenumber
energy in magnetics (30.1% at 0.2–10 km vs 2.3% for gravity), while 78.8% of
gravity energy is at wavelengths >30 km. Fault signatures are heterogeneous.
The prior two label-derived exploratory cluster logistic regressions had
internal system-held-out AUC ~0.579 and ~0.577. They are **not** reused.

## Architecture

1. **Input:** 33 available Atlas channels, selected by exact ID from
   \`06_ml_ready/schemas/feature_stack_schema.json\`. No raw \`tc\` (its physical
   definition and units were ambiguous). No high-resolution DEM because the
   Atlas DEM stage did not complete.
2. Three four-level residual encoders: magnetic (10 channels), gravity (8),
   terrain + cross-modal structure (15). Defaults widths 24,48,72,96.
3. Softmax-gated three-way fusion at each scale; dilated residual
   bottleneck with rates 1,2,4,8 and U-Net decoder.
4. **Regional gravity context:** masked average of the original isostatic
   anomaly onto 1 km coarse pixels, then whole-region Gaussian smoothing at
   sigma 5, 15 and 30 km, and reprojection to the 100 m analysis grid.
   Smoothing and normalisation are label-free; fold normalisation is
   fitted on training-region pixels only.
5. One global fault head and two lightweight specialist score heads share
   the backbone. A learned per-pixel soft router sees only physical input
   features. Its low-weight guidance uses label-free magnetic versus
   gravity edge persistence, **not mapped fault cluster IDs**.
6. Nonnegative PU surrogate, 300 m soft-Tversky with alpha 0.2, beta 0.8,
   proximity and axial orientation training heads. No invalid magnetic
   flips. A score head alone is used for submission.

## Required Atlas inputs — exact existing schema and grid

\`\`\`text
GEMS_2/GEMS_GEOLOGICAL_DATA_ATLAS_V1/
  06_ml_ready/schemas/feature_stack_schema.json
  06_ml_ready/feature_stack/magnetic_stack.tif
  06_ml_ready/feature_stack/gravity_stack.tif
  06_ml_ready/feature_stack/terrain_stack.tif
  06_ml_ready/feature_stack/raw_stack.tif
  06_ml_ready/feature_stack/cross_modal_stack.tif
  02_geospatial/masks/{valid_numerical,fault_labels,template_valid}.tif
  00_raw_snapshot/official_template/example_submission.tif
  04_eda/candidate_lineaments/candidate_lineaments.parquet
\`\`\`

The existing Atlas run's feature schema was inspected in Drive; the runner
will validate actual raster headers, transforms, EPSG:32611, shape 3730×3292,
band counts and masks **before** producing training artifacts. No raw-data
downloads or legacy model checkpoints are required.

## Run order

\`\`\`python
EXP = GeoRegimeExperiment(CONFIG)
EXP.preflight()                # real artifact checks and CUDA availability
EXP.prepare()                  # fingerprinted feature/context bank on Drive
EXP.run_geographic()           # three geographic holdout training runs
EXP.run_discovery()            # three withheld-system training runs
EXP.export()                   # final full-label training, inference and GeoTIFF
EXP.figures()
\`\`\`

Three contiguous geographic held-out regions use a 1.5 km train buffer.
Three discovery splits withhold whole connected mapped trace components and
mask an 800 m neighborhood during fitting, scoring their 3 km neighborhoods
with retained mapped-fault pixels excluded exactly. These are offline proxies
against historical mapped labels, not newly mapped truth. Pool DTI by summing
weighted TP/FP/FN, not by averaging fold scores.

The final model trains separately on all mapped labels; its output is a
**continuous** float32 GeoTIFF. Atlas lineaments are rasterized into a
separate review layer, not added to submission probabilities or used as
pseudo-ground truth.

## Cache and artifacts

\`\`\`text
GEMS_2/GEMS_GEOREGIME_V1/
  cache/bank_<feature_hash>.npy
  cache/context_<feature_hash>.npy
  cache/bank_<feature_hash>.json
  runs/<run_hash>/config.json
  runs/<run_hash>/checkpoints/{geo0,geo1,geo2,disc0,disc1,disc2,full}_latest.pt
  runs/<run_hash>/checkpoints/*_normalizer.json
  runs/<run_hash>/tables/*_history.csv
  runs/<run_hash>/tables/validation_components.csv
  runs/<run_hash>/tables/{geographic_dti,discovery_dti}.json
  runs/<run_hash>/tables/candidate_model_review.csv
  runs/<run_hash>/predictions/*_prediction.tif
  runs/<run_hash>/predictions/*_router_magnetic.tif
  runs/<run_hash>/predictions/atlas_candidate_support.tif
  runs/<run_hash>/figures/{training_curves,prediction_and_routing}.{png,svg}
  runs/<run_hash>/submissions/submission_georegime_v1.tif
  runs/<run_hash>/submissions/submission_georegime_v1.json
\`\`\`

All generated features are fingerprinted and only produced from Atlas outputs.
Model checkpoint files use atomic replacement after each epoch. A matching
checkpoint resumes at the next epoch; fully completed runs are reused.
Predictions are reused only when the checkpoint identity and input bank
match. Colab stages large Atlas raster stacks into
\`/content/gems_georegime_stage\` for rasterio performance, leaving Atlas
originals unchanged.

A mount failure may terminate the current training unit; previous completed
epochs and files remain on Drive and can be resumed after remount.

## Important qualifications

- Real Colab training, model convergence, GPU memory behavior and submission
  format validation are **unverified** until the notebook is executed.
- A global/specialist mixture can collapse or specialize poorly; inspecting
  routing maps is necessary and provides no substitute for discovery DTI.
- No rigorous comparison to a global-only model is included; this is a single
  model candidate, not an R&D ablation ladder.
- A false-positive cost is associated with including unverified lineaments
  in a submission. The lineaments are used only as a review layer by default.
- The competition's hidden DTI may differ from all offline proxies because
  previously mapped faults are incomplete.
