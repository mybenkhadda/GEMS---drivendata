# E6 Production — fully connected GPU training to NEW GEMS submission

**Implemented** on \`e6-submission-fasttrack\`; the old W2 champion fallback
was REMOVED. This milestone writes a NEW E6 prediction GeoTIFF from
trained three-way GeoStructureNet models.

**Run:** \`notebooks/colab/GEMS_E6_SUBMISSION_MILESTONE.ipynb\`

## Execution order, single pipeline

1. Mount Google Drive; clone pinned GitHub branch into /content/GEMS_E6.
2. Stage only required raw numerical features/labels, folds/splits and EDA
   metadata. Do NOT stage the whole historical 7GB feature cache.
3. \`Experiment.from_yaml(configs/experiments/e6_masked_threeway_full.yaml)\`;
   \`EXP.prepare()\`. This creates an immutable cache ID from existing
   10m coarse-part GeoTIFFs, builds seven compact 100m E6 feature channels
   and registers them with Phase5FeatureStore. A separate 0–1
   \`dem:dem_coverage\` mask stays unnormalised and never enters an encoder.
   Existing field and terrain features are lazily loaded from their caches.
4. \`EXP.train()\`: **actual** 3-fold geographic CV and three withheld-fault
   discovery fits, using the Phase5 CheckpointManager, nnPU + 300m
   soft-Tversky + auxiliary proximity/orientation losses, BF16/FP16 GPU,
   AMP/gradient accumulation, saved per-epoch checkpoint and per-unit
   prediction. Resuming skips completed folds and reopens interrupted units.
5. \`EXP.evaluate()\`: DTI for E6 only, geographic and withheld-system
   with pooled TP/FP/FN, saved as \`metrics/e6_dti.csv\`. No R&D comparisons.
6. \`EXP.export()\`: fold-average trained E6 predictions over the whole
   GeoDAWN valid footprint, saving each \`e6_full_fold<k>.tif\` to reuse
   after a disconnect. One predeclared top 1% eligible-cell selection
   (no threshold search) is applied, validated against the actual
   competition template, and atomically exported as:

\`\`\`text
GEMS_2/experiments/e6_masked_threeway_full__<config_hash>/
  submissions/submission_e6_geostructurenet.tif
  submissions/submission_e6_geostructurenet.json
  reports/e6_submission_manifest.json
\`\`\`

This is a **new E6 model output**, not the prior W2 score map.
No automatic DrivenData upload; only the user can upload the finished TIFF.

## Actual E6 architecture

Source: \`gems_bench/models/geostructurenet_e6.py\`.

- Uses the current \`GeoStructureNet\` class with two original
  terrain/subsurface Encoders, every E1 gated fuse and decoder; optional
  dilated/context branch remains supported. Full pretrained E1
  checkpoint loading uses the **matching CV fold only**. The original E1
  fold0 and fold1 checkpoints are already saved; E1 fold2 is unavailable.
  Fold2 and discovery runs consequently train the baseline from scratch:
  **no leak from another held-out fold or discovery split**.
- Adds a lightweight third Encoder (width=16), with one
  \`MaskedThreeWayGate\` at each of the five levels. Gating uses
  \`softmax([old terrain logit, old field logit, dem_logit + log(m)])\`,
  setting DEM logit to negative infinity when \`m == 0\`.
- The mask is not a learnable prediction input. It controls only
  availability, and at exactly missing DEM pixels an explicit final
  \`torch.where\` returns the original 2-way decoder's logits.
  This eliminates decoder cross-boundary contamination on uncovered
  pixels. DEM dropout=30% is applied per training patch only.
- E1 warm-start strict load allows *only* the new E6 auxiliary head
  parameters to initialize anew; other unexpected/missing backbone
  parameters fail. Warm stages freeze the base for two epochs, then
  fine-tune base at one-fifth the DEM adapter learning rate.
- The nnPU positive prior is currently 0.01. The Tversky term
  uses 0.2 FP / 0.8 FN and a training-only soft kernel from mapped
  *visible* faults within 300m. Proximity and orientation losses
  are training-time auxiliary heads only. At inference only seg
  probabilities are used.
- Exact baselines' two-branch architecture is retained outside DEM
  support, although base weights are allowed to fine-tune after warm-up
  (so they need not numerically equal the historical E1 weights).

## DEM resolution optimization and temporary caching

The already saved G07 feasibility pilot generated roughly 10
**10 m coarse-part** DEM rasters. Instead of opening hundreds of 1m
GeoTIFFs or processing 2m/100 windows for every tile, E6 computes
slope, slope-break, curvature, roughness, 60m directional coherence,
four-azimuth hillshade range and multiscale DoG at 10m on those
completed parts. It explicitly masks small-footprint edge support,
then averages into the 100m competition grid.

- \`outputs/e6_pipeline/cache/<input_hash>/tiles/\`: each completed
  100m feature tile, with JSON and retry-safe reuse.
- \`outputs/e6_pipeline/cache/<input_hash>/dem_features_100m.tif\`:
  cached, tile-mosaicked feature raster. Source originals are
  **never mirrored or overwritten**.
- \`outputs/e6_pipeline/features/<input_hash>/\`: one .npy channel
  cache; only the exact E6 channels are staged to Colab.
- \`experiments/<e6_id>/checkpoints/\`: actual fold/split resume
  checkpoints; \`predictions/\`: all fold, split and full-region scores.
- \`metrics/e6_dti.csv\`: DTI-only result, saved for continuation.

This approach uses only the coverage of cached 10m parts, **not**
the full 37.9% high-resolution DEM footprint. That limits the maximum
possible DEM contribution until more 10m caches exist, but it avoids
another multi-hour native-resolution DEM pilot. The experiment still
makes full-footprint predictions because its base terrain/field
branches cover the original numerical dataset everywhere.

## Boundary and claims

- No unit tests, synthetic experiments, ablation ladder or research
  comparisons are performed in the notebook.
- No original USGS raster is downloaded, modified or mirrored.
- No old W2 submission is copied or substituted.
- All training and features remain resumable and persisted to Drive.
- The implementation is authored and checked by GitHub readback only;
  it has **not yet been run in Colab**. Runtime completion and new
  leaderboard DTI are unmeasured. The code will fail clearly if the
  saved 10m coarse-part inputs or expected raw data are absent.
- External DEM sources are allowed by the GEMS competition provided
  their license permits use and sharing with the sponsor; preserve
  the original USGS source manifests and processing code.
