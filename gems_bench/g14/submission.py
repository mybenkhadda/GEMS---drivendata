"""Explicit, guarded submission preparation (Phase 12). Nothing here runs on import.

``prepare_submission`` writes a GeoTIFF ONLY when every guard passes:

* ``cfg.run_mode == 'SUBMIT'`` and ``cfg.submission_enabled``;
* the caller passes ``confirm='WRITE_SUBMISSION'``;
* the map is a full-region map whose construction matches what was evaluated (``construction`` recorded by the
  pipeline: the final unit's model, permitted known context = every mapped fault);
* the operating point (representation, area fraction) comes from the cross-fitted selection, not from TEST.

The file goes through the G13 writer and validator (template geometry, float32, [0, 1], NaN outside the template,
0 on pixel-exact known faults, non-degenerate, checksum). The competition API is never called; upload is manual.

UNTESTED: written without execution.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

COMPETITION_URL = "https://www.drivendata.org/competitions/306/competition-doe-gems/"


def operating_point_map(full: np.ndarray, data, rep: str, frac: float) -> np.ndarray:
    """Apply the selected representation to the full-region map (eligible = valid, mapped faults excluded)."""
    from ..g13.dti_opt import render

    spec = {"rep": {"select": "top"} if rep == "top" else {"select": "crest", "post": [["dilate", {"radius_px": 1}]]},
            "frac": float(frac), "ring_px": 0}
    return render(spec, np.asarray(full, np.float32), data.valid, data.labels, None, data.valid)


def prepare_submission(cfg, data, full: np.ndarray, rep: str, frac: float, out_dir: str | Path, *, name: str,
                       provenance: dict, confirm: str = "") -> dict:
    from ..g13.cache import Workspace
    from ..g13.submission import export_candidate

    if cfg.run_mode != "SUBMIT" or not cfg.submission_enabled:
        raise PermissionError("submission requires run_mode='SUBMIT' and submission_enabled=True")
    if confirm != "WRITE_SUBMISSION":
        raise PermissionError("pass confirm='WRITE_SUBMISSION' to write a file")
    need = ("model", "construction", "selection", "cfg_hash")
    missing = [k for k in need if k not in provenance]
    if missing:
        raise ValueError(f"provenance incomplete: {missing}")
    if provenance["selection"] != "cross-fitted":
        raise ValueError("the operating point must come from cross-fitted selection, never from TEST")
    q = operating_point_map(full, data, rep, frac)
    ws = Workspace(Path(out_dir))
    rec = export_candidate(name, q, data, ws, {**provenance, "representation": f"{rep}|frac={frac}"},
                           "zero", extra_names=("predicted_faults.tif",))
    rec["upload"] = f"manual only: {COMPETITION_URL}"
    rec["written_unix"] = time.time()
    (Path(out_dir) / "submissions" / f"{name}_g14.json").write_text(json.dumps(rec, indent=1, default=str))
    return rec
