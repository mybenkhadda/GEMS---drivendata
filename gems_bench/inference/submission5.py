"""Competition GeoTIFF export with provenance and validation (Milestone 5D).

Writes ``submissions/<name>__<mode>__<known_policy>.tif`` (single band, float32, template grid, values in
[0, 1], NaN outside the data footprint as in Phase 2/3), validates it (``evaluation.parity.submission_audit``)
and writes a JSON sidecar with model, experiment, config hash, git commit, protocol hash, area fraction and
policies. Leaderboard feedback is recorded separately in ``reports/leaderboard_log.csv`` by the user.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..artifacts.atomic import atomic_save, atomic_write_json
from ..calibration.area_policy import apply_known_policy, export_map
from ..submission import finalise, write_submission


def export_submission(ctx, prob_full: np.ndarray, out_dir: Path, name: str, *, mode: str = "binary_area",
                      frac: float | None = None, known_policies=("model", "suppress_known"), provenance: dict | None = None
                      ) -> list[dict]:
    from ..evaluation.parity import submission_audit

    data = ctx.data
    q = export_map(prob_full, data.valid, mode, frac)
    out = []
    for pol in known_policies:
        arr = finalise(apply_known_policy(q, data.labels, pol), data, None, "model")
        path = Path(out_dir) / f"{name}__{mode}__{pol}.tif"
        atomic_save(path, lambda t, a=arr: write_submission(a, data, Path(t)))
        audit = submission_audit(path, data)
        info = {"name": name, "file": path.name, "mode": mode, "area_frac": frac, "known_policy": pol,
                "written": time.strftime("%Y-%m-%dT%H:%M:%S"), **(provenance or {}), "audit": audit,
                "positive_frac_valid": float(np.nanmean(arr[data.valid] > 0.5))}
        atomic_write_json(path.with_suffix(".json"), json.loads(json.dumps(info, default=str)))
        out.append(info)
    return out
