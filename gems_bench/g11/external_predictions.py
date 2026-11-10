"""Locate, validate and align existing model predictions (G07 GeoFormer arrays, M9, P6, GeoRegime, ...).

A NumPy array is placed on the map ONLY from documented metadata, searched in this order:
  1. sidecar JSON next to the file (``<stem>.json`` / ``<stem>_meta.json``) with ``crs`` + ``transform``;
  2. an adjacent manifest / grid spec in the same folder (``manifest.json``, ``*manifest*.json``, ``grid*.json``) with
     ``crs`` + ``transform`` - this is how G07 recorded its 200 m grid (``outputs/G07_GEOFORMER/manifest.json``);
  3. a companion GeoTIFF in the same folder with exactly the same shape and a CRS.
The shape alone NEVER locates an array (a (1865, 1646) array is not assumed to be the 2x-decimated master grid).
A located grid must NEST in the master grid (same CRS, integer resolution ratio, integer pixel offset); the array is
then resampled by NEAREST block replication (coarse -> 100 m) so its finite support and values are preserved exactly.

Provenance: ``*_foldmasked*`` -> OOF_FOLD_MASKED (held-out under the ORIGINAL G07 split, named by its stack/test
suffix - NOT the G11 split); ``*_full*`` -> FULL_DOMAIN_INFERENCE (saw its own training region: never treated as
out-of-fold); GeoTIFF predictions without split information -> UNKNOWN_PROVENANCE (EDA only).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .grid import GridSpec

MAX_PER_FAMILY = 40


def _grid_from_json(p: Path, shape) -> dict | None:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(d, dict):
        return None
    for cand in (d, d.get("grid", {}) if isinstance(d.get("grid"), dict) else {}):
        t, c = cand.get("transform"), cand.get("crs")
        if t and c and len(t) >= 6:
            sh = cand.get("shape") or cand.get("native_shape")
            if sh and tuple(int(x) for x in sh[-2:]) != tuple(shape[-2:]):
                return None                                          # documented grid of a different array
            return {"crs": c, "transform": [float(x) for x in t[:6]], "shape": list(shape[-2:]), "evidence": str(p)}
    return None


def resolve_metadata(path, shape) -> tuple[dict | None, str]:
    """(grid dict or None, how). ``how`` in SIDECAR / ADJACENT_MANIFEST / COMPANION_TIF / UNLOCATED."""
    p = Path(path)
    for side in (p.with_suffix(".json"), p.with_name(p.stem + "_meta.json")):
        if side.is_file():
            g = _grid_from_json(side, shape)
            if g:
                return g, "SIDECAR"
    for m in sorted(p.parent.glob("*.json")):
        if re.search(r"manifest|grid|meta", m.name, re.I) and m.stat().st_size < 5_000_000:
            g = _grid_from_json(m, shape)
            if g:
                return g, "ADJACENT_MANIFEST"
    try:
        import rasterio

        for t in sorted(p.parent.glob("*.tif")):
            with rasterio.open(t) as ds:
                if ds.crs is not None and tuple(ds.shape) == tuple(shape[-2:]):
                    return {"crs": ds.crs.to_wkt(), "transform": list(ds.transform)[:6], "shape": list(ds.shape),
                            "evidence": str(t)}, "COMPANION_TIF"
    except ImportError:
        pass
    return None, "UNLOCATED"


def provenance(name: str) -> tuple[str, str | None]:
    n = name.lower()
    split = next((s for s in ("stack", "test", "oof", "val") if re.search(rf"(^|_){s}(_|$|\.)", n)), None)
    if "foldmasked" in n or split == "oof":
        return "OOF_FOLD_MASKED", split
    if re.search(r"(^|_)full(_|$|\.)", n):
        return "FULL_DOMAIN_INFERENCE", "full"
    return "UNKNOWN_PROVENANCE", split


def nesting(src: GridSpec, master: GridSpec) -> dict:
    from rasterio.crs import CRS

    same_crs = CRS.from_user_input(src.crs) == CRS.from_user_input(master.crs)
    a, e = src.transform[0], src.transform[4]
    A, E = master.transform[0], master.transform[4]
    rx, ry = a / A, e / E
    ox, oy = (src.transform[2] - master.transform[2]) / A, (src.transform[5] - master.transform[5]) / E
    ok = (same_crs and abs(rx - round(rx)) < 1e-6 and abs(ry - round(ry)) < 1e-6 and round(rx) >= 1
          and abs(ox - round(ox)) < 1e-6 and abs(oy - round(oy)) < 1e-6)
    return {"nested": bool(ok), "same_crs": bool(same_crs), "ratio": (rx, ry), "offset_px": (ox, oy)}


def to_master(arr: np.ndarray, src: GridSpec, master: GridSpec) -> np.ndarray:
    """Exact nearest block replication of a nested coarser (or equal) grid onto the master grid (NaN elsewhere)."""
    nst = nesting(src, master)
    if not nst["nested"]:
        raise ValueError(f"grid does not nest in the master grid: {nst}")
    f = int(round(nst["ratio"][0]))
    oc, orow = int(round(nst["offset_px"][0])), int(round(nst["offset_px"][1]))
    up = np.repeat(np.repeat(np.asarray(arr, np.float32), f, 0), f, 1)
    out = np.full(master.shape, np.nan, np.float32)
    r0, c0 = max(orow, 0), max(oc, 0)
    sr0, sc0 = r0 - orow, c0 - oc
    h = min(master.shape[0] - r0, up.shape[0] - sr0)
    w = min(master.shape[1] - c0, up.shape[1] - sc0)
    if h > 0 and w > 0:
        out[r0:r0 + h, c0:c0 + w] = up[sr0:sr0 + h, sc0:sc0 + w]
    return out


def load_any(path) -> tuple[np.ndarray, dict | None, str]:
    p = Path(path)
    if p.suffix.lower() in (".tif", ".tiff"):
        import rasterio

        with rasterio.open(p) as ds:
            a = ds.read(1, masked=True).astype(np.float32).filled(np.nan)
            if ds.crs is None:
                return a, None, "UNLOCATED"
            return a, {"crs": ds.crs.to_wkt(), "transform": list(ds.transform)[:6], "shape": list(ds.shape),
                       "evidence": "own GeoTIFF header"}, "GEOTIFF"
    a = np.load(p, mmap_mode="r")
    if a.ndim != 2:
        a = np.squeeze(a)
    if a.ndim != 2:
        return np.asarray(a), None, f"UNSUPPORTED_NDIM_{a.ndim}"
    g, how = resolve_metadata(p, a.shape)
    return np.asarray(a, np.float32), g, how


def register_external(paths_by_family: dict, master: GridSpec, out_dir, write_tif=True) -> pd.DataFrame:
    """Align every located prediction; returns the compatibility table (one row per file)."""
    from .features import write_geotiff

    out_dir = Path(out_dir)
    rows = []
    for fam, paths in paths_by_family.items():
        ranked = sorted(paths, key=lambda s: (0 if re.search(r"prob|pred|fold|oof|full|stack|test", Path(s).name, re.I)
                                              else 1, s))[:MAX_PER_FAMILY]
        for s in ranked:
            p = Path(s)
            prov, split = provenance(p.name)
            row = {"family": fam, "path": str(p), "name": p.stem, "provenance": prov, "source_split": split}
            try:
                a, g, how = load_any(p)
            except Exception as e:                                     # noqa: BLE001 - recorded
                rows.append({**row, "status": f"UNREADABLE: {type(e).__name__}: {e}"})
                continue
            row.update(native_shape=list(a.shape), metadata_source=how)
            if g is None:
                rows.append({**row, "status": "EXCLUDED: UNLOCATED (no documented CRS/transform)"})
                continue
            src = GridSpec(g["crs"], tuple(g["transform"]), tuple(a.shape))
            nst = nesting(src, master)
            row.update(native_res_m=abs(src.transform[0]), ratio=nst["ratio"][0], offset_px=list(nst["offset_px"]),
                       metadata_evidence=g.get("evidence"))
            if not nst["nested"]:
                rows.append({**row, "status": f"EXCLUDED: GRID_DOES_NOT_NEST ({nst})"})
                continue
            m = to_master(a, src, master)
            fin = np.isfinite(m) & (master.valid if master.valid is not None else True)
            v = m[fin]
            row.update(finite_support_px=int(fin.sum()), support_fraction=float(fin.mean()),
                       vmin=float(v.min()) if v.size else None, vmax=float(v.max()) if v.size else None,
                       probability_like=bool(v.size and v.min() >= -1e-6 and v.max() <= 1 + 1e-6),
                       resampling="nearest block replication (exact support)")
            if write_tif:
                t = write_geotiff(out_dir / f"{fam}__{p.stem}.tif", np.where(fin, m, np.nan), master,
                                  [f"{fam}:{p.stem} ({prov}) aligned to master grid"])
                row["aligned_path"] = str(t)
            row["status"] = "COMPATIBLE" if row["probability_like"] else "COMPATIBLE_NOT_PROBABILITY"
            rows.append(row)
    return pd.DataFrame(rows)
