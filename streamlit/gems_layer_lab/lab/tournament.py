"""Feature tournament: evaluate a formula template over a parameter grid.

* EXPLORE: every combination scored on the whole AOI (direction 'high', plus the exploratory best-direction AUROC
  flagged as optimistic) with regional stability (3 x 3 tiles) and a shifted-target null. Exploratory only.
* VALIDATE: spatial folds (an existing fold raster, or deterministic blocks) - for each held-out fold the combination
  AND direction are chosen on the other folds (minus a buffer) and only then scored on the held-out fold. The
  reported estimate is the mean held-out score of the dev-selected combinations; held-out scores of all other
  combinations are stored for transparency but never used for selection.

Results are written after every combination (``results.csv`` + ``manifest.json``) under a deterministic id, so an
interrupted tournament resumes where it stopped.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from . import association as A

MAX_COMBOS = 400


def parse_grid(text: str) -> dict:
    """``sigma: 2, 3, 5`` lines -> {"sigma": [2.0, 3.0, 5.0]} (numbers only)."""
    grid = {}
    for n, line in enumerate(text.splitlines(), 1):
        line = line.split("#")[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"grid line {n}: use 'name: v1, v2, ...'")
        k, vals = (s.strip() for s in line.split(":", 1))
        if not k.isidentifier():
            raise ValueError(f"grid line {n}: {k!r} is not a parameter name")
        try:
            grid[k] = [float(v) for v in vals.split(",") if v.strip()]
        except ValueError as e:
            raise ValueError(f"grid line {n}: values must be numbers ({e})") from e
        if not grid[k]:
            raise ValueError(f"grid line {n}: no value")
    n = int(np.prod([len(v) for v in grid.values()])) if grid else 0
    if n > MAX_COMBOS:
        raise ValueError(f"{n} combinations > {MAX_COMBOS}")
    return grid


def block_folds(shape, block_px=100, k=3, seed=2026) -> np.ndarray:
    """Deterministic spatial folds: blocks of block_px assigned to k folds (balanced random permutation)."""
    H, W = shape
    nb_r, nb_c = -(-H // block_px), -(-W // block_px)
    rng = np.random.default_rng(seed)
    ids = np.arange(nb_r * nb_c)
    fold_of = np.empty(len(ids), int)
    fold_of[rng.permutation(len(ids))] = np.arange(len(ids)) % k
    r, c = np.indices(shape)
    return fold_of[(r // block_px) * nb_c + (c // block_px)].astype(np.int16)


def load_fold_raster(path: str, window) -> np.ndarray:
    """Crop an aligned integer fold raster (.npy or single-band .tif; < 0 = excluded) to the window."""
    p = Path(path)
    r0, c0, h, w = int(window.row_off), int(window.col_off), int(window.height), int(window.width)
    if p.suffix == ".npy":
        return np.asarray(np.load(p, mmap_mode="r")[r0:r0 + h, c0:c0 + w], np.int16)
    import rasterio

    with rasterio.open(p) as s:
        return s.read(1, window=window).astype(np.int16)


def tournament_id(template, target, grid, window, mode, fold_spec, budget) -> str:
    key = json.dumps({"t": template, "y": target, "g": grid, "w": [int(window.col_off), int(window.row_off),
                                                                   int(window.width), int(window.height)],
                      "m": mode, "f": fold_spec, "b": budget}, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:14]


def run(engine, template: str, target: str, grid: dict, window, faults, mode="EXPLORE", folds=None, buffer_px=10,
        budget=0.01, out_dir: Path | None = None, progress=None, fold_spec="blocks") -> dict:
    """Run / resume. ``faults`` = mapped-fault mask of the window. Returns {results, summary, manifest, dir}."""
    if mode not in ("EXPLORE", "VALIDATE"):
        raise ValueError("mode must be EXPLORE or VALIDATE")
    base = engine.compile(template)
    if target not in base.names:
        raise ValueError(f"target {target!r} is not a variable of the template ({base.names})")
    for k, vals in grid.items():
        if k not in base.params:
            raise ValueError(f"grid parameter {k!r} is not declared in the template PARAMS ({list(base.params)})")
        p = base.params[k]
        bad = [v for v in vals if not p.min <= v <= p.max]
        if bad:
            raise ValueError(f"{k}: values {bad} outside the PARAMS range [{p.min}, {p.max}]")
    if mode == "VALIDATE" and folds is None:
        raise ValueError("VALIDATE needs a fold raster")
    tid = tournament_id(template, target, grid, window, mode, fold_spec, budget)
    d = Path(out_dir) / tid if out_dir else None
    res_path = d / "results.csv" if d else None
    done = pd.read_csv(res_path) if res_path and res_path.exists() else pd.DataFrame()
    done_ids = set(done.combo_id) if len(done) else set()
    keys = list(grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*[grid[k] for k in keys])] or [{}]
    manifest = {"id": tid, "template": template, "target": target, "grid": grid, "mode": mode, "budget": budget,
                "fold_spec": fold_spec, "buffer_px": buffer_px, "n_combos": len(combos),
                "window": [int(window.col_off), int(window.row_off), int(window.width), int(window.height)],
                "status": "RUNNING", "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if d:
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    f = np.asarray(faults, bool)
    rows = done.to_dict("records") if len(done) else []
    for i, combo in enumerate(combos):
        cid = hashlib.sha256(json.dumps(combo, sort_keys=True).encode()).hexdigest()[:10]
        if cid in done_ids:
            continue
        prog = engine.compile(template, combo)
        x = engine.evaluate(prog, window, targets=[target]).arrays[target]
        row = {"combo_id": cid, **{f"param_{k}": v for k, v in combo.items()}}
        if mode == "EXPLORE":
            m = A.metrics(x, f, None, "high", budgets=(budget,))
            row.update({k: m.get(k) for k in ("status", "auroc", "average_precision", "ap_lift", "point_biserial",
                                              f"recall_at_{budget * 100:g}pct")})
            a_hi = m.get("auroc_high", np.nan)
            row["auroc_best_direction_OPTIMISTIC"] = max(a_hi, 1 - a_hi) if np.isfinite(a_hi) else np.nan
            T = A.tile_table(x, f, "grid", 3, direction="high", budget=budget)
            ok = T[T.status == "OK"] if "status" in T else T
            row.update(regional_auroc_mean=float(ok.auroc.mean()) if len(ok) else np.nan,
                       regional_auroc_std=float(ok.auroc.std()) if len(ok) > 1 else np.nan,
                       regional_share_above_half=float((ok.auroc > 0.5).mean()) if len(ok) else np.nan,
                       regional_tiles_ok=int(len(ok)))
            nc = A.negative_control(x, f, None, "high", n=8)
            row.update(null_auroc_p95=nc["null_p95"], exceeds_null_p95=nc["exceeds_null_p95"])
        else:
            fz = np.asarray(folds)
            for k in sorted(set(np.unique(fz[fz >= 0]).tolist())):
                held = fz == k
                near = ndimage.binary_dilation(held, iterations=int(buffer_px)) if buffer_px else held
                dev = (fz >= 0) & ~near
                dirn = A.choose_direction(x, f, dev)
                md = A.metrics(x, f, dev, dirn, budgets=(budget,))
                mv = A.metrics(x, f, held, dirn, budgets=(budget,))
                row.update({f"fold{k}_direction": dirn, f"fold{k}_dev_auroc": md.get("auroc"),
                            f"fold{k}_val_auroc": mv.get("auroc"), f"fold{k}_val_ap_lift": mv.get("ap_lift"),
                            f"fold{k}_val_recall": mv.get(f"recall_at_{budget * 100:g}pct")})
        rows.append(row)
        if d:
            pd.DataFrame(rows).to_csv(res_path, index=False)
        if progress:
            progress(i + 1, len(combos))
    R = pd.DataFrame(rows)
    summary = summarize(R, mode, grid)
    manifest.update(status="COMPLETE", finished=time.strftime("%Y-%m-%dT%H:%M:%S"), summary=summary)
    if d:
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        try:
            R.to_parquet(d / "results.parquet", index=False)
        except Exception:
            pass
    return {"results": R, "summary": summary, "manifest": manifest, "dir": str(d) if d else None}


def summarize(R: pd.DataFrame, mode: str, grid: dict) -> dict:
    pcols = [c for c in R if c.startswith("param_")]
    if not len(R):
        return {}
    if mode == "EXPLORE":
        best = R.sort_values("auroc", ascending=False).iloc[0]
        return {"best_exploratory": {c: best[c] for c in pcols} | {"auroc": best.get("auroc")},
                "n_combos": len(R), "selection_bias_warning": f"best of {len(R)} combinations on the SAME data: "
                "optimistic; confirm in VALIDATE mode", "max_null_p95": float(R.null_auroc_p95.max())
                if "null_auroc_p95" in R else None}
    folds = sorted({int(c.split("_")[0][4:]) for c in R if c.startswith("fold") and c.endswith("_dev_auroc")})
    picks = []
    for k in folds:
        b = R.sort_values(f"fold{k}_dev_auroc", ascending=False).iloc[0]
        picks.append({"fold": k, **{c: b[c] for c in pcols}, "direction": b[f"fold{k}_direction"],
                      "dev_auroc": b[f"fold{k}_dev_auroc"], "val_auroc": b[f"fold{k}_val_auroc"],
                      "val_ap_lift": b[f"fold{k}_val_ap_lift"]})
    P = pd.DataFrame(picks)
    return {"selected_per_fold": picks, "validated_auroc_mean": float(P.val_auroc.mean()) if len(P) else None,
            "validated_ap_lift_mean": float(P.val_ap_lift.mean()) if len(P) else None, "n_combos": len(R),
            "stable_selection": bool(len(P) and P[pcols].drop_duplicates().shape[0] == 1),
            "note": "held-out scores of non-selected combinations are stored but were not used for selection"}
