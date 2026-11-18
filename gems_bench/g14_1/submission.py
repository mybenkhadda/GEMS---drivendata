"""One reliable submission function (Phase R). Writes files only; never uploads, never calls the competition API.

``generate_submission`` resolves, in order, without retraining or re-running inference when artifacts exist:

1. the FULL-REGION map of the requested model (G14 ``maps/full.npy``, a G14.1 corrected final map, a G13
   ``full.npy`` or the P6 ``full_score.tif``), found by the artifact inventory (local or Drive, used in place);
2. the operating point (representation, area fraction, calibration) of THE SAME model, preferring:
   ``DISCOVERY_CALIBRATED``  G14.1 cross-fitted discovery selection   > G14 scorecard ``disc_*``
   ``PROVISIONAL_GEO``       G14.1 geographic selection                > G14 scorecard ``geo_*``  (labelled provisional)
   ``MANUAL``                explicit ``rep`` / ``frac`` passed by the user (labelled manual);
3. the representation on the official domain (known faults pixel-exact zero, template nodata NaN);
4. export through the G13 writer + validator, the G14.1 parity check, a provenance JSON, Drive sync.

Several independently named candidates can be produced for offline comparison. An operating point of another model
is refused unless ``allow_transfer=True`` (then labelled ``TRANSFERRED``). A missing artifact raises one actionable
error naming what is missing and which stage produces it.

UNTESTED: written without execution.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

COMPETITION_URL = "https://www.drivendata.org/competitions/306/competition-doe-gems/"


class SubmissionError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(2**20), b""):
            h.update(c)
    return h.hexdigest()[:16]


def find_full_map(model: str, inv, g141_root: Path) -> Path:
    """``model``: 'g14' (source experiment), 'g14_1:<spec>', 'g13:<model>', 'p6_w2' or a file path."""
    from .artifacts import best_copy

    if Path(model).suffix in (".npy", ".tif") and Path(model).exists():
        return Path(model)
    if model == "g14":
        p = best_copy(inv, "g14_full_map")
    elif model.startswith("g14_1:"):
        cand = sorted((g141_root / "corrected_models" / "maps").glob(f"{model.split(':', 1)[1]}__final_*.npy"))
        p = cand[-1] if cand else None
    elif model.startswith("g13:"):
        m = model.split(":", 1)[1]
        rows = inv[(inv.artifact == "g13_full_maps") & inv.rel.str.contains(f"/{m}/")]
        p = Path(rows.path.iloc[0]) if len(rows) else None
    elif model == "p6_w2":
        p = best_copy(inv, "p6_w2_full")
    else:
        raise SubmissionError(f"unknown model id {model!r} (use 'g14', 'g14_1:<spec>', 'g13:<model>', 'p6_w2' or a path)")
    if p is None or not Path(p).exists():
        how = {"g14": "G14 notebook predict_final() (needs the final_0 checkpoint) or restore maps/full.npy from Drive",
               "p6_w2": "P6 fast-track stack stage"}.get(model, "train / predict the final unit (stage R15/R16)")
        raise SubmissionError(f"no full-region map for {model!r}: produce it with {how}")
    return Path(p)


def find_operating_point(model: str, inv, g141_root: Path, primary_task: str = "disc_official") -> dict | None:
    """Best available operating point of ``model`` with its status label (or None)."""
    sel = g141_root / "calibration" / "selected_operating_points.json"
    if sel.exists():
        j = json.loads(sel.read_text())
        for task, status in ((primary_task, "DISCOVERY_CALIBRATED"), ("geo", "PROVISIONAL_GEO")):
            r = j.get(model, {}).get(task)
            if r:
                return {**r, "status": status if task.startswith("disc") else "PROVISIONAL_GEO",
                        "source": str(sel), "task": task}
    if model == "g14":
        cards = inv[inv.artifact == "g14_scorecards"]
        for _, row in cards.iterrows():
            c = json.loads(Path(row.path).read_text())
            if not str(c.get("model", "")).startswith("T"):
                continue
            for key, task, status in (("disc", "disc_official", "DISCOVERY_CALIBRATED"), ("geo", "geo", "PROVISIONAL_GEO")):
                if c.get(f"{key}_rep") is not None and c.get(f"{key}_frac") is not None:
                    return {"rep": c[f"{key}_rep"], "frac": float(c[f"{key}_frac"]), "calibration": "identity",
                            "order": "pixel_exact", "status": status, "task": task, "source": row.path,
                            "crossfit_dti": c.get(f"{key}_dti")}
    return None


def render_full(full: np.ndarray, data, rep: str, frac: float | None, order: str = "pixel_exact",
                calibration: dict | None = None) -> tuple[np.ndarray, dict]:
    """Representation on the official domain: eligible = valid minus pixel-exact known faults."""
    from .calibration import temperature
    from .representations import apply_rep

    s = np.nan_to_num(np.asarray(full, np.float32))
    cal = calibration or {}
    if cal.get("kind") == "temperature":
        s = temperature(s, float(cal["T"]))
    elif cal.get("kind") == "power":
        s = np.clip(s, 0, 1) ** float(cal["gamma"])
    elif cal.get("kind") not in (None, "identity"):
        raise SubmissionError(f"calibration {cal.get('kind')!r} cannot be re-applied from its parameters; refit it "
                              "on the out-of-fold maps (stage R03) or choose identity / temperature / power")
    scored = data.valid & ~data.labels
    return apply_rep(rep, s, data.valid, data.labels, scored, None if frac is None or frac < 0 else frac, order)


def generate_submission(cfg, data, inv, *, model: str = "g14", rep: str | None = None, frac: float | None = None,
                        order: str | None = None, name: str | None = None, allow_transfer: bool = False,
                        operating_point_of: str | None = None, sync: bool = True, log=print) -> dict:
    """Write one official-format candidate GeoTIFF + provenance; returns the record (path, validation, status)."""
    from ..g13.submission import validate, write
    from .artifacts import code_revision
    from .geometry import submission_parity

    if cfg.mode != "SUBMIT":
        raise SubmissionError("set mode='SUBMIT' (G141Config.for_mode('SUBMIT')) to write submission files")
    local = Path(cfg.local_root) / "g14_1"
    drive = Path(cfg.out_root)
    full_p = find_full_map(model, inv, drive if drive.exists() else local)
    if rep is not None:
        op = {"rep": rep, "frac": frac, "order": order or "pixel_exact", "calibration": "identity",
              "status": "MANUAL", "source": "user", "task": None}
    else:
        src = operating_point_of or model
        if src != model and not allow_transfer:
            raise SubmissionError(f"operating point of {src!r} requested for map {model!r}: pass allow_transfer=True "
                                  "to label it TRANSFERRED, or calibrate this model (stage R03)")
        op = find_operating_point(src, inv, drive if drive.exists() else local, cfg.validation.primary_task)
        if op is None:
            raise SubmissionError(f"no operating point for {src!r}: run the representation / calibration sweep "
                                  "(stage R03) on its held-out maps, or pass rep= and frac= explicitly (MANUAL)")
        if src != model:
            op["status"] = "TRANSFERRED_" + op["status"]
    cal = op.get("calibration_params") if isinstance(op.get("calibration_params"), dict) else None
    from .champion import read_map

    full = read_map(full_p)
    if tuple(full.shape[-2:]) != tuple(data.shape):
        raise SubmissionError(f"{full_p}: shape {full.shape} differs from the template grid {data.shape}")
    q, acc = render_full(np.asarray(full), data, op["rep"], op.get("frac"), op.get("order", "pixel_exact"), cal)
    tag = name or f"{model.replace(':', '-')}__{op['rep']}_{op.get('frac')}__{op['status']}"
    out = local / "submissions"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{tag}.tif"
    write(q, data, path, known_policy="zero", tags={"candidate": tag, "status": op["status"]})
    val = validate(path, data, "zero")
    par = submission_parity(path, data, expected=q)
    rec = {"candidate": tag, "path": str(path), "model": model, "full_map": str(full_p), "full_map_sha": _sha(full_p),
           "operating_point": op, "accounting": acc, "validation": val, "parity": par,
           "status": op["status"], "provisional": op["status"] != "DISCOVERY_CALIBRATED",
           "cfg_hash": cfg.semantic_hash(), "code_rev": code_revision("/content/GEMS"),
           "written_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
           "upload": f"manual only: {COMPETITION_URL}", "never_uploaded_automatically": True}
    (out / f"{tag}.json").write_text(json.dumps(rec, indent=1, default=str))
    if sync and Path(cfg.drive_root).exists():
        d = drive / "submissions"
        d.mkdir(parents=True, exist_ok=True)
        for f in (path, out / f"{tag}.json"):
            shutil.copy2(f, d / f.name)
        rec["drive_path"] = str(d / path.name)
    log(f"[g14.1] wrote {path} ({op['status']}); valid={val['valid']} {val['checks_failed']}")
    return rec
