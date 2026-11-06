# Training protocol

## Units
- `geo0, geo1, geo2`: geographic folds (`spatial_splits.geo_folds`). Training pixels = role `train`; patches never
  touch val, buffer (13 km), inner_val or inner_buffer (5 km). Early stopping on the inner-val cluster DTI.
- `disc0, disc1, disc2`: withheld fault systems. Withheld systems + 1 km ring have weight 0 (not negatives);
  training everywhere else except the split's own inner-val cluster (chosen outside its evaluation zone).
Every unit trains its own reference from scratch: no weights move between units.

## Stages
| stage | trains | data | epochs (cfg) | lr |
|---|---|---|---|---|
| A | ReferenceNet (all) | all admissible patches, 50 % with visible faults | `epochs.baseline` = 20 | `lr` |
| B | - | best A -> `reference.pt` (immutable, sha256 in `reference.json`) | - | - |
| C | - | E6Net <- reference (all shared weights + gate T/F paths); parity check (closed DEM gate, max abs diff <= 1e-3; uncovered pixels exact) -> `C_parity.json` | - | - |
| D | DEM encoder + gate DEM paths (`adapt_d`, `proj_d`, `logit_d`) | 80 % patches with mean coverage >= 0.2 | `epochs.dem` = 8 | `lr` |
| E | all E6Net weights; shared ones at `lr x 0.2` | as D, plus 30 % DEM modality dropout | `epochs.joint` = 8 | `lr` / `lr x 0.2` |
After E: parity re-check on probe patches (uncovered pixels must equal the reference exactly) -> `E_parity.json`.

Per epoch: `steps_per_epoch` = 60 optimizer steps x `grad_accum` 2 x `batch_size` 8 patches of 256 px. AdamW
(weight decay 1e-4), 5 % linear warm-up then cosine, gradient clip 1.0, AMP bf16 if supported else fp16 with a
GradScaler. Early stopping: `patience` 5 after `min_epochs` 5; the best inner-val DTI checkpoint is finalised.

## Losses (`losses.py`)
- nnPU with logistic loss. Prior pi = (visible fault pixels / weighted training pixels) x 1.5, clipped to
  [0.002, 0.05], computed from the unit's training pixels only (`unit_info.json`). The x 1.5 encodes that mapped
  faults are a subset of the true faults; it is a stated assumption, not an estimate.
- soft Tversky (alpha 0.2, beta 0.8) against t = max(1 - d / 300 m, 0), d = distance to visible faults.
- proximity: Smooth L1 on min(d, 300) / 300 within 1 km of visible faults, weight 0.3.
- orientation: 1 - cos between predicted and target doubled-angle strike within 300 m where the mapped trace is
  coherent (>= 0.5), weight 0.1. Target strike = structure tensor of the visible labels, propagated from the nearest
  label pixel.
- weights: 1 on training pixels; 0 on unlabeled pixels within 100 m of a visible fault, on ignored pixels and
  outside the training region. In D and E the weights are additionally restricted to DEM-covered pixels (elsewhere
  the output is the frozen reference and has no gradient).

## DEM modality dropout (stage E)
Per patch with probability 0.3 the DEM inputs and the gate coverage are zeroed: the encoders run the two-way path,
while the output fallback still uses the measured coverage, so these patches train the shared path to stay good
without DEM inside covered areas. In stage D dropout is off: a dropped patch would carry no gradient to the DEM
parameters.

## Augmentation
90-degree rotations and horizontal flips only (AUGMENTATION contract in FEATURE_SCHEMA.md).

## What has not been measured
Runtime per epoch, GPU memory, convergence, every DTI value. The first Colab run is the first execution of this
code.
