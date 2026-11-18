"""Raster geometry and submission parity (Phase C, P0).

What this module establishes, from the files themselves (never from constants):

* the native grid of every official raster, of the G14 maps and of every exported submission: CRS, affine transform,
  shape, pixel size, origin, nodata, footprint;
* the metric kernel radius in pixels derived from the affine transform (300 m / pixel size, per axis);
* the exported file versus the map it was built from: identity, binarisation, clipping, NaN / footprint handling,
  known-fault zeroing, and displacement (integer and sub-pixel shifts, flips, transposition) by phase correlation;
* systematic offset between predicted ridges and the known-fault raster used for training (a mis-registration
  of the labels would appear as a consistent signed offset);
* an audit of hard-coded resolution assumptions in the G14 / G13 / P6 / metric code (static list, values checked
  against the actual transform at runtime).

UNTESTED: written without execution.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

RADIUS_M = 300.0

# Hard-coded resolution assumptions found by the static audit (docs/g14_1/STATIC_CODE_AUDIT.md, S13).
# (location, quantity, value, unit, assumed pixel metres)
HARD_CODED = [
    ("g14/config.py LossCfg.kernel_radius_px", "DTI kernel radius", 3.0, "px", 100.0),
    ("g14/config.py LossCfg.ignore_ring_px", "negative ignore ring", 3, "px", 100.0),
    ("g14/config.py LossCfg.reliable_negative_px", "reliable negative distance", 10, "px", 100.0),
    ("g14/multiscale.py orient_w", "orientation supervision band", 3, "px", 100.0),
    ("g14/evaluation.py diagnostics", "km per pixel", 0.1, "km/px", 100.0),
    ("g14/splits.py UnitSystem.pixel_m", "pixel size", 100.0, "m", 100.0),
    ("g14/config.py SplitCfg.embargo_m", "embargo", 13000.0, "m", None),
    ("g14/config.py DataCfg.local", "local view", 128, "px @100 m", 100.0),
    ("p6/stack.py known_features px_km", "km per pixel", 0.1, "km/px", 100.0),
    ("gems3/official.py FastScorer pixel_m", "pixel size", 100.0, "m", 100.0),
    ("gems3/official.py _dilate sampling", "pixel size", 100.0, "m", 100.0),
    ("metric.py dti_components pixel_m default", "pixel size", 100.0, "m", 100.0),
    ("g13/contracts.py PIXEL_M", "expected pixel size (asserted at load)", 100.0, "m", 100.0),
]


# =============================================================================== headers
def raster_header(path: str | Path) -> dict:
    import rasterio

    with rasterio.open(path) as s:
        t = s.transform
        return {"path": str(path), "count": s.count, "height": s.height, "width": s.width, "dtype": s.dtypes[0],
                "crs": str(s.crs), "epsg": s.crs.to_epsg() if s.crs else None, "transform": tuple(t)[:6],
                "pixel_x_m": abs(t.a), "pixel_y_m": abs(t.e), "origin_x": t.c, "origin_y": t.f,
                "rotation": (t.b, t.d), "north_up": t.e < 0 and t.b == 0 and t.d == 0, "nodata": s.nodata,
                "bounds": tuple(s.bounds)}


def pixel_size_m(transform) -> tuple[float, float]:
    """(row, col) pixel size in metres from an affine transform (rasterio Affine or 6-tuple a,b,c,d,e,f)."""
    t = tuple(transform)[:6]
    return abs(float(t[4])), abs(float(t[0]))


def kernel_radius_px(transform, radius_m: float = RADIUS_M) -> dict:
    py, px = pixel_size_m(transform)
    from ..metric import kernel_offsets

    offs = kernel_offsets(radius_m, (py, px))
    return {"radius_m": radius_m, "pixel_m_row": py, "pixel_m_col": px, "radius_px_row": radius_m / py,
            "radius_px_col": radius_m / px, "n_kernel_offsets": len(offs),
            "max_offset_px": max(max(abs(a), abs(b)) for a, b, _ in offs)}


def grid_table(headers: dict[str, dict], reference: str = "template") -> pd.DataFrame:
    """Every raster against the reference grid: CRS, shape, pixel size, origin offset (in reference pixels)."""
    ref = headers[reference]
    rows = []
    for name, h in headers.items():
        if "error" in h:
            rows.append({"raster": name, "status": "UNREADABLE", "detail": h["error"]})
            continue
        dx = (h["origin_x"] - ref["origin_x"]) / ref["pixel_x_m"]
        dy = (ref["origin_y"] - h["origin_y"]) / ref["pixel_y_m"]
        same = (h["epsg"] == ref["epsg"] and (h["height"], h["width"]) == (ref["height"], ref["width"])
                and np.allclose(h["transform"], ref["transform"]))
        rows.append({"raster": name, "status": "SAME_GRID" if same else "DIFFERENT_GRID", "epsg": h["epsg"],
                     "shape": (h["height"], h["width"]), "pixel_m": (h["pixel_y_m"], h["pixel_x_m"]),
                     "res_ratio_x": h["pixel_x_m"] / ref["pixel_x_m"], "res_ratio_y": h["pixel_y_m"] / ref["pixel_y_m"],
                     "origin_offset_px": (round(dy, 4), round(dx, 4)), "north_up": h["north_up"],
                     "nodata": h["nodata"], "dtype": h["dtype"], "count": h["count"]})
    return pd.DataFrame(rows)


def assumption_audit(transform, embargo_m: float = 13000.0, autocorr_km: float | None = 12.8) -> pd.DataFrame:
    """Each hard-coded assumption against the actual pixel size; an item is CONSISTENT when its assumed pixel size
    equals the real one (the metric radius in pixels is then 300 m / pixel)."""
    py, px = pixel_size_m(transform)
    rows = []
    for loc, qty, val, unit, assumed in HARD_CODED:
        ok = None if assumed is None else bool(abs(assumed - py) < 1e-6 and abs(assumed - px) < 1e-6)
        metres = val * assumed if unit == "px" and assumed else None
        rows.append({"location": loc, "quantity": qty, "value": val, "unit": unit, "assumed_pixel_m": assumed,
                     "actual_pixel_m": (py, px), "value_in_m_if_px": metres,
                     "status": "N/A" if ok is None else ("CONSISTENT" if ok else "INCONSISTENT")})
    rows.append({"location": "metric", "quantity": "DTI radius in pixels (derived)", "value": RADIUS_M / px,
                 "unit": "px", "assumed_pixel_m": None, "actual_pixel_m": (py, px), "value_in_m_if_px": RADIUS_M,
                 "status": "DERIVED"})
    if autocorr_km:
        rows.append({"location": "splits", "quantity": "embargo vs G13 variogram length", "value": embargo_m,
                     "unit": "m", "assumed_pixel_m": None, "actual_pixel_m": (py, px),
                     "value_in_m_if_px": autocorr_km * 1000,
                     "status": "SHORTER_THAN_AUTOCORRELATION" if embargo_m < autocorr_km * 1000 else "OK"})
    return pd.DataFrame(rows)


# =============================================================================== displacement
def phase_correlation(a: np.ndarray, b: np.ndarray, upsample: bool = True) -> dict:
    """Translation (dy, dx) that best aligns ``b`` onto ``a`` (positive = b must move down / right), with a
    parabolic sub-pixel refinement; NaN -> 0. Peak strength near 1 for identical images."""
    A = np.nan_to_num(np.asarray(a, np.float64))
    B = np.nan_to_num(np.asarray(b, np.float64))
    A, B = A - A.mean(), B - B.mean()
    Fa, Fb = np.fft.rfft2(A), np.fft.rfft2(B, s=A.shape)
    R = Fa * np.conj(Fb)
    R /= np.maximum(np.abs(R), 1e-12)
    r = np.fft.irfft2(R, s=A.shape)
    i, j = np.unravel_index(int(np.argmax(r)), r.shape)
    H, W = r.shape

    def sub(c, m, n):
        if not upsample:
            return 0.0
        l, cc, rr = c[(m - 1) % n], c[m], c[(m + 1) % n]
        den = l - 2 * cc + rr
        return 0.0 if abs(den) < 1e-12 else 0.5 * (l - rr) / den

    di = sub(r[:, j], i, H)
    dj = sub(r[i, :], j, W)
    dy = (i if i <= H // 2 else i - H) + di
    dx = (j if j <= W // 2 else j - W) + dj
    return {"dy_px": float(dy), "dx_px": float(dx), "peak": float(r[i, j])}


def d4_match(expected: np.ndarray, actual: np.ndarray) -> pd.DataFrame:
    """Correlation of ``actual`` with each D4 transform of ``expected`` (flips / transposition detection)."""
    e = np.nan_to_num(np.asarray(expected, np.float64))
    a = np.nan_to_num(np.asarray(actual, np.float64))
    cands = {"identity": e, "flip_ud": e[::-1], "flip_lr": e[:, ::-1], "rot180": e[::-1, ::-1]}
    if e.shape[0] == e.shape[1]:
        cands.update({"transpose": e.T, "rot90": np.rot90(e), "rot270": np.rot90(e, 3), "anti_transpose": np.rot90(e).T[::-1]})
    rows = []
    for k, c in cands.items():
        if c.shape != a.shape:
            rows.append({"transform": k, "corr": np.nan, "note": "shape differs"})
            continue
        sa, sc = a.std(), c.std()
        corr = float(((a - a.mean()) * (c - c.mean())).mean() / (sa * sc)) if sa > 0 and sc > 0 else np.nan
        rows.append({"transform": k, "corr": corr, "note": ""})
    return pd.DataFrame(rows).sort_values("corr", ascending=False, na_position="last")


def ridge_offset_to_known(score: np.ndarray, known: np.ndarray, valid: np.ndarray, max_px: int = 3,
                          top_frac: float = 0.02) -> dict:
    """Mean signed vector from high-score ridge pixels to their nearest known-fault pixel (within ``max_px``).

    A registration error between the label raster and the inputs would appear as a consistent non-zero mean
    vector; a well-aligned map gives a mean near (0, 0) with symmetric spread."""
    from ..gems3.representation import crest_map, top_mask

    s = np.where(valid, np.nan_to_num(score), 0.0).astype(np.float32)
    crest = crest_map(s, valid) & top_mask(s, valid, top_frac)
    if not known.any() or not crest.any():
        return {"n": 0}
    d, (ir, ic) = ndimage.distance_transform_edt(~known, return_indices=True)
    rr, cc = np.nonzero(crest & (d <= max_px) & (d > 0))
    if not len(rr):
        n_on = int((crest & known).sum())
        return {"n": 0, "crest_on_known_px": n_on}
    vy, vx = ir[rr, cc] - rr, ic[rr, cc] - cc
    return {"n": int(len(rr)), "mean_dy_px": float(vy.mean()), "mean_dx_px": float(vx.mean()),
            "sem_dy": float(vy.std() / np.sqrt(len(vy))), "sem_dx": float(vx.std() / np.sqrt(len(vx))),
            "crest_on_known_px": int((crest & known).sum()), "crest_px": int(crest.sum())}


# =============================================================================== submission parity
def read_band(path: str | Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as s:
        return s.read(1, masked=True).astype(np.float32).filled(np.nan)


def raw_known(data) -> np.ndarray:
    """The official known-fault raster as read from disk (value >= 1), NOT restricted to the data footprint."""
    a = read_band(data.known_path)
    return np.nan_to_num(a) >= 1


def submission_parity(sub_path: str | Path, data, expected: np.ndarray | None = None) -> dict:
    """The exported file against the template and (optionally) the array it was supposed to contain.

    ``expected``: the map handed to the writer (before ``finalise``). Every check is descriptive; FAIL items are the
    ones that change the official score."""
    from ..g13.submission import finalise

    h_sub, h_tpl = raster_header(sub_path), raster_header(data.template_path)
    a = read_band(sub_path)
    tv = data.template_valid
    known_raw = raw_known(data)
    fin = np.isfinite(a)
    inside = np.nan_to_num(a[tv])
    uniq = np.unique(inside[: 5_000_000])
    res = {"file": str(sub_path), "same_grid": grid_table({"template": h_tpl, "sub": h_sub}).status.iloc[-1],
           "nan_inside_template": int((~fin & tv).sum()), "finite_outside_template": int((fin & ~tv).sum()),
           "min": float(inside.min()) if inside.size else np.nan, "max": float(inside.max()) if inside.size else np.nan,
           "n_unique_sampled": int(len(uniq)), "binary": bool(len(uniq) <= 2),
           "share_exact_0": float((inside == 0).mean()), "share_exact_1": float((inside == 1).mean()),
           "positive_frac_of_template": float((inside > 0).mean()) if inside.size else np.nan,
           "positive_on_known_labels": int(((np.nan_to_num(a) > 0) & data.labels).sum()),
           "positive_on_raw_known_raster": int(((np.nan_to_num(a) > 0) & known_raw).sum()),
           "raw_known_outside_labels_px": int((known_raw & ~data.labels).sum()),
           "pred_mass": float(inside.sum())}
    if expected is not None:
        e = finalise(np.asarray(expected, np.float32), data, "zero")
        both = np.isfinite(e) & fin
        diff = np.abs(np.nan_to_num(e) - np.nan_to_num(a))[both]
        res.update(identical=bool(both.sum() == fin.sum() and (diff.max() if diff.size else 0) < 1e-6),
                   max_abs_diff=float(diff.max()) if diff.size else 0.0,
                   share_different=float((diff > 1e-6).mean()) if diff.size else 0.0)
        pc = phase_correlation(e, a)
        res.update({f"shift_{k}": v for k, v in pc.items()})
        dm = d4_match(e, a)
        res["best_d4_transform"] = str(dm.transform.iloc[0])
        res["best_d4_corr"] = float(dm["corr"].iloc[0])
    return res


def round_trip(q: np.ndarray, data, tmp_dir: str | Path) -> dict:
    """Write ``q`` with the official writer, read it back and compare (export-only parity)."""
    from ..g13.submission import write

    p = Path(tmp_dir) / "parity_round_trip.tif"
    write(q, data, p, known_policy="zero")
    return submission_parity(p, data, expected=q)


def geometry_report(data, extra: dict[str, str] | None = None) -> dict:
    """Headers of the three official rasters (+ ``extra`` named GeoTIFFs), grid table, kernel and assumptions."""
    heads = {"template": raster_header(data.template_path), "features": raster_header(data.raw_path),
             "known_faults": raster_header(data.known_path)}
    for k, p in (extra or {}).items():
        try:
            heads[k] = raster_header(p)
        except Exception as e:  # noqa: BLE001
            heads[k] = {"error": str(e)}
    known_raw = raw_known(data)
    return {"grid": grid_table(heads), "kernel": kernel_radius_px(data.transform),
            "assumptions": assumption_audit(data.transform),
            "footprint": {"template_valid_px": int(data.template_valid.sum()), "data_valid_px": int(data.valid.sum()),
                          "template_valid_no_data_px": int((data.template_valid & ~data.valid).sum()),
                          "known_raw_px": int(known_raw.sum()), "known_in_labels_px": int(data.labels.sum()),
                          "known_outside_valid_px": int((known_raw & ~data.valid).sum())},
            "array_axes": {"rows": data.shape[0], "cols": data.shape[1],
                           "note": "numpy arrays are (row = north->south, col = west->east) as read by rasterio"}}
