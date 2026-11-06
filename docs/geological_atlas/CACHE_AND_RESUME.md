# Cache and resume

* **Atomic files**: every write goes to `<name>.partial` and is renamed with `os.replace`; readers refuse `.partial`.
* **Ledgers** (append-only JSONL, fsync per record, latest record wins, a torn last line is ignored):
  `07_cache/copy_state/copy_ledger.jsonl`, `07_cache/copy_state/hash_cache.jsonl`,
  `07_cache/download_state/downloads.jsonl`.
* **Stage records** `07_cache/manifests/<stage>.json`: with `reuse_valid_cache=True` a COMPLETE record whose
  fingerprint matches (and whose outputs exist with their recorded sizes) is reused; STALE / INCOMPLETE records are
  recomputed and the reason is logged (`08_reports/atlas_run.log`).
* **No repeated hashing**: verified snapshot files are skipped by (size, mtime_ns); hashes are cached per
  (path, size, mtime_ns).
* **DEM**: the 10 m tile caches are built once; each 5 km window is written as `<id>.npz`, then `<id>.json`; a
  restart loses only the windows in flight. `dem.budget_minutes > 0` stops cleanly (status INCOMPLETE) - re-run
  section 08.
* **Downloads**: `.partial` + HTTP Range resume; a tile is accepted only after header + data check + SHA-256.

## After a disconnect (`OSError: [Errno 107] Transport endpoint is not connected`)
1. `from google.colab import drive; drive.flush_and_unmount()`
2. `drive.mount('/content/drive', force_remount=True)`
3. If it keeps failing: Runtime > Restart session.
4. Re-run section 00, then the section that failed. Completed artifacts, verified copies and DEM windows are reused.

The snapshot never overwrites a verified destination with an incomplete copy, never deletes originals and never
treats an inaccessible source as missing.

## Forcing recomputation
Set `"reuse_valid_cache": False` (recomputes every stage you run) or delete one stage record in
`07_cache/manifests/`.
