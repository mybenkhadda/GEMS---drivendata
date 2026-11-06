# Raw snapshot

Destination: `GEMS_GEOLOGICAL_DATA_ATLAS_V1/00_raw_snapshot/` - `geodawn/`, `faults/`, `official_template/` (one
file each), `dem_1m/<original sub-path>` (project folders and file names preserved, XML next to its tile),
`source_metadata/` (catalogs and provenance documents, path flattened with `__`).

## Workflow (`raw_snapshot.snapshot`)
1. **Health check** (`reliability.drive_health_check`): stat and list the Drive root, write / fsync / read / delete
   a probe in `07_cache/manifests`.
2. **Capacity**: total source bytes, bytes already verified, bytes to copy, duplicate storage (= full snapshot size)
   and `shutil.disk_usage` free space on the mount (the Google Drive quota can differ; the report says so).
3. **Per file**:
   * the ledger `07_cache/copy_state/copy_ledger.jsonl` says VERIFIED for the same source size + mtime_ns and the
     destination has that size -> skipped, **no re-hash**;
   * destination present without a record -> both hashed (cached in `hash_cache.jsonl`); equal -> VERIFIED;
   * otherwise: copy in 16 MiB chunks to `<dest>.partial` (fsync) while hashing the source; re-read and hash the
     `.partial`; equal -> `os.replace` onto the destination and VERIFIED; different -> CHECKSUM_MISMATCH, `.partial`
     deleted, destination untouched. Transient I/O errors are retried (3x, back-off).
4. **Failures**: `Errno 107` -> `INTERRUPTED_DRIVE_DISCONNECTED` logged, report written, `DriveDisconnected` raised
   (remount, re-run: verified files are reused). ENOSPC / EDQUOT -> `QUOTA_EXCEEDED`, remaining files
   `NOT_ATTEMPTED`, report written, `DriveQuotaExceeded` raised. Unreadable source -> `SOURCE_INACCESSIBLE`.
5. **Reports**: `01_inventory/manifests/snapshot_status.csv`, `snapshot_report.json` (end state, capacity, counts,
   required files missing, DEM tiles planned / verified, `complete`), `01_inventory/source_checksums.csv`,
   `raw_inventory.csv`, `provenance.csv`.

Originals are opened read-only and never deleted. A verified destination is replaced only after a NEW copy has been
verified. `snapshot_raw=False` uses the existing snapshot without copying.

## Downstream gate (`AtlasPipeline.require_snapshot`)
Stops when the GeoDAWN raster, faults or template are not VERIFIED. DEM stages also stop when fewer DEM tiles are
verified than planned, unless `snapshot.allow_partial_dem=True` (explicit). Every stage after section 01 reads
`00_raw_snapshot/` only.
