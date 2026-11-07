# Validation protocol

Each quantity has its own protocol name and is never merged with another. Every DTI shown is an offline proxy, not the leaderboard DTI.

| protocol | prediction | eval mask | truth | exclusion | where |
|---|---|---|---|---|---|
| `GEOGRAPHIC` (A) | V1 `geo*_prediction.tif` | zone & valid | labels & zone | none | section 06 |
| `WITHHELD_SYSTEM` (B) | V1 `disc*_prediction.tif` | dilate(held, 3 km) & valid | held | labels & ~held | section 06 |
| `WITHHELD_SYSTEM_NARROW_V1` | V1 disc raster | as B | held | as B | section 08 (side by side) |
| `DISCOVERY_NARROW_REINFERRED` | new full-area raster | as B | held | as B | section 08: separates the re-inference effect from the protocol effect |
| `DISCOVERY_FULL_AREA` (C) | new full-area raster | **all eligible pixels** | held | primary `training_visible` = labels & train; sensitivity `all_non_held` = labels & ~held | sections 08, 09, 14, 15 |
| `GEOGRAPHIC_ZONE_REINFERRED` / `GEOGRAPHIC_FULL_AREA` | new full-area geo rasters | zone / all eligible | labels & zone | none / as C | section 08; secondary gate in section 15 |

## Full-area pseudo-discovery (C), per discovery checkpoint
1. Reconstruct the masks and verify `train_mask_sha` (section 05).
2. Load the checkpoint and its embedded fold normalizer. The normalizer JSON must be equal to it if present.
3. Infer over `bank.valid` under a distinct `FULL_AREA` cache identity (section 07).
4. Score the held mapped faults over the whole eligible footprint:
   * training-visible mapped pixels are excluded pixel-exactly;
   * the 300 m kernel is used;
   * `exclude_supports_tp=False`.
5. Save TP_w, FP_w, FN_w, DTI, weighted recall and precision, prediction mass, and the FP share of the denominator.

Pooled values sum TP, FP and FN over units with `gems_bench.metric.pool`. A mean of fold DTIs is never reported as a pooled value.

## Why two exclusions
Labels that are neither held nor training-visible are the retained mapped pixels inside the 800 m training buffer. The checkpoint never saw them, but they are real faults. The two conventions treat them differently:
* **Primary** (`training_visible`, as the brief requires): predictions on those pixels count as false positives.
* **Sensitivity** (`all_non_held`, the V1 convention and closest to the leaderboard's "all known faults masked"): they are excluded.

Both are reported.

## Caveat
Full-area scoring treats every genuinely unmapped fault as a false positive. It is a pseudo-discovery proxy and is never described as an unbiased estimate of the private leaderboard score. Held labels never enter inference features or policy selection for their own fold.
