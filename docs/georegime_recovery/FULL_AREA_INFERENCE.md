# Full-area inference (section 07)

## The V1 defect this replaces
`GeoRegimeExperiment.predict(unit, roi)` reuses `predictions/<unit>_prediction.tif` whenever the checkpoint size, mtime and bank match. The ROI is not part of that identity, so a prediction made for the narrow discovery zone would be returned for a full-area request. The recovery code never calls `predict`.

## Cache identity
`stable_hash` of the following:
* scope `FULL_AREA` and the unit;
* checkpoint sha256;
* bank fingerprint and fold normalizer hash;
* ROI = `bank.valid` and its mask hash;
* model-code hash (`georegime/model.py` + `data.py`);
* architecture arguments;
* patch / stride / AMP;
* output channels;
* schema version.

A raster set is reused only if all of these hold:
1. the manifest key equals the recomputed key;
2. the status is COMPLETE;
3. every output still has its recorded size and sha256.

Anything else is reported as MISSING, INCOMPLETE or STALE and recomputed. A narrow raster can never pass: different filename, different ROI hash, different scope.

## Computation
The computation replicates V1 `predict` with ROI = all eligible pixels:
* `_tile_positions(…, 256, 192)`;
* windows kept when they touch the ROI;
* Hann window floored at 0.05, times the ROI;
* weighted average;
* AMP float16 on CUDA, as in V1.

Windows are batched (`inference.batch_size`). Since GeoRegimeNet uses GroupNorm only, batching does not change the outputs.

Six channels per unit (compressed float32 GeoTIFF, NaN outside the footprint):

| channel | file | meaning |
|---|---|---|
| prob | `02_full_inference/<unit>_full_prediction.tif` | final mixture `prob` |
| global, expert0, expert1 | `02_full_inference/experts/<unit>_<ch>.tif` | sigmoid of the head logits |
| route0, specialist_weight | `02_full_inference/router/<unit>_<ch>.tif` | `routes[:,0]` (V1 "router_magnetic"), mixture strength |

Manifest: `02_full_inference/inference_manifests/<unit>_full_area.json`. It holds the identity, outputs with sha256, window count, device, seconds, and the reproduction check.

Each output is validated before it is accepted:
* finite values in [0, 1] inside the footprint;
* after readback, the finite footprint equals `bank.valid`;
* CRS and transform match.

## Resume
Accumulators (6 channels plus the weight sum) are held in RAM. Every `inference.snapshot_every_windows` windows they are written atomically, together with the boolean list of finished windows, to `cache/inference/<identity>/snapshot.npz`.

After a disconnect, the run reloads the snapshot and recomputes only the windows not yet in it. Accumulation is never applied twice, and the snapshot is deleted once the rasters are complete.

## Reproduction check
Inside a V1 ROI, the windows covering a pixel and their weights are identical to V1's. So the re-inferred `prob` must equal the V1 raster up to numerical noise. The manifest records:
* max, mean and p99.9 of the absolute difference;
* Pearson correlation;
* status: `REPRODUCED` (max diff ≤ `inference.reproduction_tol`), `DIFFERS`, or `V1_RASTER_ABSENT`.

`DIFFERS` blocks promotion through the provenance gate.

## Bank
The V1 bank and context (`GEMS_GEOREGIME_V1/cache/{bank,context}_<key>.npy`) are staged read-only into scratch. If they are missing, they are rebuilt from the Atlas into local scratch only (`Bank._make_bank` / `_make_context`); nothing is written to Drive and no raw data is copied.

## Runtime
On CPU the run falls back with a warning. GPU time, CPU time and memory are unmeasured.
