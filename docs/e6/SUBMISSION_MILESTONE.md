# E6 deadline milestone — production submission first

Status: code-authored only. No Colab training, unit tests, synthetic tests,
model inference, DEM extraction, or uploads were performed while preparing
these changes.

## 1. Today: a real submission candidate

Open the dedicated notebook:
\`notebooks/colab/GEMS_E6_SUBMISSION_MILESTONE.ipynb\`

Configuration defaults:

\`\`\`python
PREPARE_SUBMISSION_TODAY = True
TRAIN_E6 = False
RUN_TESTS = False
CREATE_MIRRORS = False
AUTO_UPLOAD = False
DOWNLOAD_TIF_TO_BROWSER = True
\`\`\`

The notebook mounts Drive, checks out only source code from the
\`e6-submission-fasttrack\` branch, then **reuses the frozen W2 champion**:

\`\`\`text
GEMS_2/
  outputs/leaderboard_focus/submission_factory/submissions/
    submission_w2_champion_0a8692b2ab.tif
    submission_w2_champion_0a8692b2ab.json
    submission_w2_champion_0a8692b2ab_validation.json
    submission_sha256.txt
\`\`\`

The independently recorded frozen SHA256 is
\`c634d321060f702720f37c82b5c692dcb84030d01af0f23ec130473273e3019b\`.

**No TIFF is copied on Drive.** Required integrity validation consists of:
matching the frozen SHA256, checking the previously recorded hard validation
checks (all must be true), a blockwise GeoTIFF readback for EPSG:32611, one
float32 100 m band, finite range [0,1], and 24,322 positive pixels.
These are mandatory artifact validity checks, NOT a model test suite.

One output folder is used:
\`GEMS_2/outputs/e6_submission_milestone/\`

Small persisted files:
- \`ready_for_manual_upload.json\`
- \`submission_inventory.csv\`
- \`e6_readiness.json\`

The notebook can download the original TIFF to the browser. Upload it
**manually** on DrivenData; no connector/code makes a competition submission.
The previously recorded discovery DTI of this W2 champion is a proxy metric,
not a claim about the current leaderboard.

## 2. E6 masked three-way architecture: implemented, not trained

File: \`gems_bench/models/geostructurenet_e6.py\`

Actual classes reused from the repository:
- \`GeoStructureNet\` from \`gems_bench/models/geostructurenet.py\`
- \`Encoder\` / \`GatedFuse\` blocks from \`gems_bench/geological_encoders.py\`
- existing baseline decoder, score/dist/orientation heads, context, and
  optional dilated long-range branch.

At every encoder level, E6 adds one smaller DEM encoder and one
\`MaskedThreeWayGate\`:
- Original terrain/field logits and projected features remain unchanged.
- DEM score uses the learned logit plus log(coverage), with exact minus
  infinity where coverage=0.
- Softmax is over terrain/potential/DEM.
- Model-wide DEM dropout is 30% *during training only*.
- The final output uses a hard uncovered-pixel fallback to the original
  two-branch decoder logits. This is necessary because convolution kernels
  can otherwise leak DEM information into adjacent uncovered pixels.
- Loading the E1 checkpoint uses a strict state_dict match; incorrect
  checkpoints are not silently tolerated.
- The first-stage trainable parameters are the DEM encoder and gates only.

**No evidence of E6 improvement is claimed.** The module is not yet
integrated into the old P5 trainer, whose \`_forward(model, x, ctx)\` does
not accept the separate DEM raster or coverage mask. The optional E6
training flow is consequently **not enabled** in this submission milestone.

Also, the saved E1 manifest examined on 2026-11-06 has only completed
cv_fold0 and cv_fold1. cv_fold2 is missing, so there is no complete
three-fold warm-start for E6. E6 may not be promoted until the missing
checkpoint, spatially isolated DEM datasets and full validation exist.

## 3. E6 inputs and efficient data choices

The supplied E6 design is retained as the *next* research route, but
does not block the current frozen W2 submission.

- Terrain branch: existing raw/100 m profile channels and pretrained
  E1 encoder unchanged.
- Potential field: **reuse** G06 magnetic/gravity derivatives when already
  present; avoid FFT duplication until the registry identifies missing
  families. Any new FFT derivatives must use padding/edge taper and robust
  quantile normalization.
- DEM branch: prefer the cached 10 m \`fast_coarse\` G07 features as the
  initial low-compute feature bank. True 2 m summaries are deferred because
  observed processing of a single 2 m tile took about 50 minutes. The
  existing G07 fast bank supplies slope, curvature, scarp proxy,
  multiazimuth hillshade, coherence, roughness and DoG.
- Input alignment: keep DEM data and coverage *separate* from the base
  terrain/field tensor. The coverage mask is a gate constraint, never a
  geologic evidence channel.
- Treat missing DEM as zero coverage, not zero elevation or a known-negative
  label.
- No training against withheld V1 fault systems or V3 geographic targets.
- Do not choose an E6 candidate using held-out DTI then claim that same
  score independent.

## 4. E6 training roadmap after today (not implemented as a runnable stage)

1. Complete the E1 third fold or choose another complete, strict
   two-branch gated reference checkpoint family.
2. Build the cached 100 m DEM feature tensor aligned to each baseline
   patch and register reproducible robust-normalization statistics.
3. Add \`dem\` and \`dem_mask\` support to the P5 batched trainer and
   inference loader. Use modality dropout and strict training-visible
   masks; check orientation-aware augmentation transforms.
4. Warm-start E6, freeze the base path, train only the DEM adapter/gates,
   and save resumable per-fold checkpoints (do not bulk-mirror datasets).
   Later unfreezing the base changes the original baseline and requires
   a separate frozen parity reference or explicit revalidation.
5. Score with full canonical 300 m DTI across V1/V3, covered and
   uncovered regions, and fault-system recall. Require full-fold
   coverage. Promote only with independently validated incremental gain.
6. Only after promotion, produce E6 full-region predictions and
   independently validated GeoTIFFs.

No naive masked 3-way concatenation, no DEM-coverage-as-evidence input,
no unverified claims from local AUC, no separate E6 submission until
all gates pass.

## 5. What is and is not meant by 'full pipeline' today

**Complete today:** cached model artifact -> checksum and strict format
validation -> durable readiness manifest -> browser download -> manual
competition upload.

**Not yet complete:** E6 custom model training, three-fold validation,
full-region E6 inference, or E6 leaderboard claim. Those are intentionally
blocked to prevent missing DEM / incomplete warm-start artifacts from
making the deadline submission fail.
