"""Deadline-first GEMS submission preparation: reuse verified W2 champion as-is.

No model training, no unit tests, no data mirroring, no submission upload.
Only revalidate the frozen source artifact and persist a lightweight manifest.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# The artifact and checksum were verified in the existing GEMS_2
# outputs/leaderboard_focus/submission_factory/submissions directory.
CHAMPION_FILE = "submission_w2_champion_0a8692b2ab.tif"
FROZEN_SHA256 = "c634d321060f702720f37c82b5c692dcb84030d01af0f23ec130473273e3019b"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _check_source_raster(path: Path) -> dict:
    """Necessary format and data integrity checks, not a benchmarking test suite."""
    import rasterio

    positives, finite, has_inf = 0, 0, False
    low, high = float("inf"), float("-inf")
    with rasterio.open(path) as ds:
        info = dict(shape=[ds.height, ds.width], crs=str(ds.crs), epsg=ds.crs.to_epsg() if ds.crs else None,
                    dtype=ds.dtypes[0], bands=ds.count, resolution=list(map(float, ds.res)),
                    transform=list(ds.transform)[:6], nodata=ds.nodata)
        for _, window in ds.block_windows(1):
            a = ds.read(1, window=window, masked=False)
            has_inf |= bool(np.isinf(a).any())
            ok = np.isfinite(a)
            finite += int(ok.sum())
            positives += int((a[ok] > 0).sum())
            if ok.any():
                low, high = min(low, float(a[ok].min())), max(high, float(a[ok].max()))
    info.update(finite_px=finite, positive_px=positives, min_value=low, max_value=high, has_inf=has_inf)
    assert info["epsg"] == 32611, "Expected competition EPSG:32611"
    assert info["bands"] == 1 and info["dtype"] == "float32", "Expected single float32 band"
    assert all(abs(x - 100) < 1e-6 for x in info["resolution"]), "Expected 100 m GeoDAWN grid"
    assert not has_inf and 0 <= low <= high <= 1, "Invalid prediction values"
    assert finite > 0 and positives > 0, "Empty or all-zero source prediction"
    return info


def prepare_verified_fallback(drive_root: str | Path, *,
                              output_folder="outputs/e6_submission_milestone") -> dict:
    """One canonical source artifact; write only JSON/CSV status to Drive.

    The original TIFF is NOT copied or modified. Its SHA must exactly match
    the independently recorded SHA of the previously validated W2 champion.
    Previous hard validation checks are read, never bypassed.
    """
    drive = Path(drive_root)
    src_folder = drive / "outputs/leaderboard_focus/submission_factory/submissions"
    source = src_folder / CHAMPION_FILE
    checks_path = src_folder / "submission_w2_champion_0a8692b2ab_validation.json"
    source_meta = src_folder / "submission_w2_champion_0a8692b2ab.json"
    if not source.is_file():
        raise FileNotFoundError(f"Missing verified champion: {source}")
    if not checks_path.is_file() or not source_meta.is_file():
        raise FileNotFoundError("Missing frozen champion validation records; refusing unverifiable reuse")

    sha = _sha256(source)
    if sha != FROZEN_SHA256:
        raise RuntimeError(f"Original champion checksum changed: {sha} != {FROZEN_SHA256}")

    saved = json.loads(checks_path.read_text(encoding="utf-8"))
    hard = [c for c in saved.get("checks", []) if c.get("hard", False)]
    failing = [c for c in hard if c.get("ok") is not True]
    if not hard or failing:
        raise RuntimeError(f"Frozen hard validation is absent/failed: {failing[:3]}")
    prev = json.loads(source_meta.read_text(encoding="utf-8"))
    if prev.get("submission") != CHAMPION_FILE or prev.get("uploaded", False):
        raise RuntimeError("Champion manifest mismatch or previously uploaded status uncertain")

    info = _check_source_raster(source)
    if info["positive_px"] != 24322:
        raise RuntimeError("Champion positive-pixel count differs from frozen manifest")
    output = drive / output_folder
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "READY_FOR_MANUAL_UPLOAD",
        "candidate": "W2_CHAMPION_FROZEN",
        "submission_file": str(source),
        "original_sha256": sha,
        "previous_hard_checks_passed": len(hard),
        "new_raster_readback": info,
        "provenance": str(source_meta),
        "validation": str(checks_path),
        "mirror_created": False,
        "training_run": False,
        "test_suite_run": False,
        "uploaded": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Only the already validated W2 candidate is ready; E6 has not trained or proven a DTI gain.",
    }
    (output / "ready_for_manual_upload.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    pd.DataFrame([{
        "candidate": report["candidate"], "status": report["status"],
        "tif": report["submission_file"], "sha256": sha,
        "positive_pixels": info["positive_px"],
        "hard_checks_passed": len(hard),
    }]).to_csv(output / "submission_inventory.csv", index=False)
    return report
