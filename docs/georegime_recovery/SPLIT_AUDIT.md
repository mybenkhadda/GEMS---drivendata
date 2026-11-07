# Split audit (section 05)

## Reconstruction
Masks are rebuilt with the checked-out V1 functions and the buffers stored in the V1 `config.json`:

* `geo_masks(bank.valid, bank.labels, geo_buffer_px=15)` gives three contiguous easting-quantile zones. Each zone is trained outside itself plus a 1.5 km buffer, with truth = labels & zone and no exclusion.
* `discovery_masks(bank.valid, bank.labels, discovery_buffer_px=8)` works on 8-connected mapped components of at least 10 px. These are assigned greedily, largest first, to the bucket with the fewest pixels. For each fold:
  * train = valid & ~dilate(held, 8 px) (800 m);
  * zone = dilate(held, 30 px) & valid (3 km);
  * exclude = labels & ~held.
* full: train = `bank.valid`.

## Verification (hard stop)
`sha256(train.tobytes())[:12]` must equal `contract["train_mask_sha"]` in every checkpoint. This is exactly the expression V1 wrote at training time.

On any mismatch the stage writes its tables and raises `SplitProvenanceError`. The masks are never replaced while the old checkpoints are still in use. A geologically defined split would need retraining, which belongs to a separate milestone.

## Outputs (`01_split_audit/`)
* **`checkpoint_split_validation.csv`** has one row per unit with these columns:
  * reconstructed and checkpoint sha, match;
  * pixel counts: train, zone, truth, exclude, training-visible;
  * `truth_inside_train_px` (must be 0);
  * `zone_inside_train_px`: the narrow zone overlaps the training region beyond 800 m, which is expected;
  * `unseen_non_held_labels_px`: retained labels inside the 800 m buffer, which are neither truth nor training-visible.
* **`holdout_fragmentation.csv`** has one row per held component: size, PCA length / linearity / strike, centroid, and `min_dist_to_train_visible_m`.
* **`split_overlap.csv`** gives pairwise zone and truth overlaps, truth of one unit inside the training mask of another, geo-zone coverage gaps, and mapped pixels never held.
* **`masks/<unit>_masks.tif`** is a uint8 bitmask: 1 train, 2 zone, 4 truth, 8 V1 exclusion, 16 training-visible.
* **`original_masks_manifest.json`** records the sha of each mask, the counts, and the sha256 of the source `data.py`.

## Interpretation caveat
Held groups are raster connectivity components, not verified geological systems. A held component lying within about 1 km of a training-visible trace is probably a continuation of a structure the checkpoint saw. The audit therefore reports the share of held components within 1 km and within 3 km of visible traces, and says this in the generated interpretation.
