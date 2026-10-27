"""G01 figures: 16:9 canvases, one geographic window per figure (common extent, equal aspect, UTM kilometres),
shared colour scales for panels of the same quantity (percentile evidence is always [0, 1]) and readable colourbars.
Fault labels appear only on evaluation / inspection figures (9, 12) and are drawn as outlines (display dilation of
one pixel: 100 m traces vanish when a 50 km window is rasterised). Every function returns (figure or None, plotting
data as a DataFrame); ``G01Store.save_fig`` writes PNG + PDF + CSV."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from ...gems3.figures import INK, SERIES

W16, H9 = 16.0, 9.0
INK2 = "#52514e"
TRUTH_C = "#e34948"
KNOWN_C = "#0b0b0b"
SEL_C = SERIES[0]
CAND_COLORS = {"G01-A": SERIES[0], "G01-B": SERIES[1], "G01-C": SERIES[2], "G01-D": SERIES[3], "G01-E": SERIES[6],
               "G01-F": SERIES[4], "edge": "#8a8986", "random": "#c9c8c4"}
EVIDENCE = {"vmin": 0.0, "vmax": 1.0, "cmap": "magma"}


def window(shape, center, half: int) -> tuple[int, int, int, int]:
    r, c = int(center[0]), int(center[1])
    r0, c0 = max(r - half, 0), max(c - half, 0)
    return r0, min(r0 + 2 * half, shape[0]), c0, min(c0 + 2 * half, shape[1])


def unit_center(u) -> tuple[int, int]:
    t = u.eval_truth()
    rr, cc = np.nonzero(t if t.any() else u.ev)
    return (int(np.median(rr)) + u.sl[0].start, int(np.median(cc)) + u.sl[1].start) if len(rr) else (0, 0)


def _extent_km(grid, w):
    left, right, bottom, top = grid.window_extent(*w)
    return left / 1000.0, right / 1000.0, bottom / 1000.0, top / 1000.0


def _outline(ax, mask, ext, color, label=None, lw=0.8):
    if mask is None or not np.asarray(mask).any():
        return
    m = ndimage.binary_dilation(np.asarray(mask, bool), iterations=1)
    ax.contour(m.astype(np.float32), levels=[0.5], colors=[color], linewidths=lw, extent=ext, origin="upper")
    if label:
        ax.plot([], [], color=color, lw=1.5, label=label)


def _stats(name, a) -> dict:
    v = np.asarray(a, np.float32)
    v = v[np.isfinite(v)]
    return {"panel": name, "n": int(v.size), "p05": float(np.percentile(v, 5)) if v.size else np.nan,
            "p50": float(np.median(v)) if v.size else np.nan, "p95": float(np.percentile(v, 95)) if v.size else np.nan}


def map_grid(grid, w, panels: list[dict], *, title: str, ncols: int, shared: dict | None = None,
             cbar_label: str = "", note: str = ""):
    """Aligned map panels on one window. ``panels`` items: {title, raster, cmap, vmin, vmax, alpha, truth, known,
    sel, legend}. ``shared`` (vmin, vmax, cmap) forces one colour scale + one colourbar for every panel."""
    import matplotlib.pyplot as plt

    ext = _extent_km(grid, w)
    n = len(panels)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(W16, H9), squeeze=False, constrained_layout=True)
    im_last, stats = None, []
    for ax, P in zip(axes.ravel(), panels):
        a = P.get("raster")
        if a is not None:
            v = np.asarray(a, np.float32)
            fin = np.isfinite(v).any()
            sc = shared or {}
            lo = sc.get("vmin", P.get("vmin", np.nanpercentile(v, 2) if fin else 0.0))
            hi = sc.get("vmax", P.get("vmax", np.nanpercentile(v, 98) if fin else 1.0))
            im = ax.imshow(v, extent=ext, cmap=sc.get("cmap", P.get("cmap", "viridis")), vmin=lo, vmax=hi,
                           interpolation="nearest", origin="upper", alpha=P.get("alpha"))
            im_last = im
            stats.append(_stats(P.get("title", ""), v))
            if not shared:
                cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
                cb.ax.tick_params(labelsize=7)
                if P.get("cbar"):
                    cb.set_label(P["cbar"], fontsize=7)
        if P.get("sel") is not None:
            from matplotlib.colors import ListedColormap

            ax.imshow(np.where(np.asarray(P["sel"], bool), 1.0, np.nan), extent=ext, cmap=ListedColormap([SEL_C]),
                      alpha=0.85, interpolation="nearest", origin="upper")
            if P.get("legend"):
                ax.plot([], [], color=SEL_C, lw=6, label="selected (equal area)")
        _outline(ax, P.get("known"), ext, KNOWN_C, "known faults (masked)" if P.get("legend") else None)
        _outline(ax, P.get("truth"), ext, TRUTH_C, "scored truth (evaluation only)" if P.get("legend") else None)
        ax.set_aspect("equal")
        ax.set_title(P.get("title", ""), fontsize=9, loc="left", color=INK)
        ax.tick_params(labelsize=7, colors=INK2)
        ax.set_xlabel("Easting (km, UTM 11N)", fontsize=7, color=INK2)
        ax.set_ylabel("Northing (km)", fontsize=7, color=INK2)
        if P.get("legend"):
            ax.legend(fontsize=7, frameon=False, loc="lower left")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    if shared and im_last is not None:
        cb = fig.colorbar(im_last, ax=axes.ravel().tolist(), shrink=0.7, pad=0.01)
        cb.set_label(cbar_label, fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle(title + (f"\n{note}" if note else ""), x=0.01, ha="left", fontsize=11, color=INK)
    return fig, pd.DataFrame(stats)


def _crop(a, w):
    """Window of a full-grid array; arrays already of the window's shape are returned unchanged."""
    if a is None:
        return None
    a = np.asarray(a)
    if a.shape == (w[1] - w[0], w[3] - w[2]):
        return a
    return a[w[0]:w[1], w[2]:w[3]]


# 1 ----------------------------------------------------------------------------------------------- raw fields
def raw_signals(src, w):
    rb = src.cfg["raw_bands"]
    panels = []
    for role, label in (("mag", "RTP magnetic anomaly"), ("grav", "isostatic gravity anomaly"),
                        ("terrain", "detrended elevation")):
        b = rb.get(role)
        if b in src.data.band_names:
            a = np.asarray(src.data.read_band(b), np.float32)[w[0]:w[1], w[2]:w[3]]
            panels.append({"title": f"{label} ({b})", "raster": a, "cmap": "cividis" if role != "terrain" else "gist_earth",
                           "cbar": "band value (2-98 % stretch)"})
    if not panels:
        return None, pd.DataFrame()
    return map_grid(src.data.grid, w, panels, title="1. Raw potential fields and terrain (independent scales: "
                    "different physical quantities)", ncols=len(panels))


# 2-3 --------------------------------------------------------------------------------------------- energy by scale
def energy_scales(grid, w, ev: dict, lams: list, role: str):
    key = "M" if role == "mag" else "G"
    panels = [{"title": f"{'magnetic' if role == 'mag' else 'gravity'} log-Gabor energy, lambda = {x.label}",
               "raster": _crop(ev.get(f"scale:{key}@{x.label}"), w)} for x in lams]
    panels = [p for p in panels if p["raster"] is not None]
    agg = ev.get(f"comp:{key}")
    if agg is not None:
        panels.append({"title": f"multiscale mean ({'G01-A' if role == 'mag' else 'G01-B'})", "raster": _crop(agg, w)})
    if not panels:
        return None, pd.DataFrame()
    n = 2 if role == "mag" else 3
    return map_grid(grid, w, panels, title=f"{n}. {'Magnetic' if role == 'mag' else 'Gravity'} wavelet energy across "
                    "spatial scales", ncols=min(len(panels), 3), shared=EVIDENCE,
                    cbar_label="energy percentile vs reference geography")


# 4 ----------------------------------------------------------------------------------------------- strike / coherence
def strike_coherence(src, w, fields: dict, lam):
    from .geology_wavelet import channel, strike_from_doubled

    panels = []
    for role in ("mag", "grav"):
        t = lam.tag(role)
        names = [channel(fields[role], k, t) for k in ("strike_c2", "strike_s2", "coherence")]
        if not all(src.has(n) for n in names):
            continue
        c2, s2, coh = (np.asarray(src.get(n), np.float32)[w[0]:w[1], w[2]:w[3]] for n in names)
        st = strike_from_doubled(c2, s2)
        nm = "magnetic" if role == "mag" else "gravity"
        panels.append({"title": f"{nm} strike (cyclic hue; opacity = coherence), {lam.label}", "raster": st,
                       "cmap": "twilight", "vmin": 0.0, "vmax": 180.0, "alpha": np.clip(np.nan_to_num(coh), 0, 1),
                       "cbar": "strike (deg from north, axial)"})
        panels.append({"title": f"{nm} directional coherence, {lam.label}", "raster": coh, "cmap": "viridis",
                       "vmin": 0.0, "vmax": 1.0, "cbar": "coherence [0, 1]"})
    if not panels:
        return None, pd.DataFrame()
    return map_grid(src.data.grid, w, panels, title="4. Strike orientation and directional coherence (M9 cached "
                    "doubled-angle products)", ncols=2)


# 5 ----------------------------------------------------------------------------------------------- orientation
def orientation_agreement(grid, w, ev: dict, lams: list):
    panels = []
    for x in lams:
        for k, t in (("A", "axial agreement A"), ("gate", "gate 1 - kappa (1 - A)")):
            a = ev.get(f"scale:{k}@{x.label}")
            if a is not None:
                panels.append({"title": f"{t}, {x.label}", "raster": _crop(a, w)})
    if not panels:
        return None, pd.DataFrame()
    return map_grid(grid, w, panels, title="5. Magnetic-gravity orientation agreement (axial, 180 deg periodic; "
                    "coherence enters once, through kappa)", ncols=min(len(panels), 4), shared={"vmin": 0, "vmax": 1,
                    "cmap": "RdYlBu_r"}, cbar_label="agreement / gate [0, 1]")


# 6 / 7 / 8 ------------------------------------------------------------------------------------------ evidence maps
def evidence_panels(grid, w, ev: dict, keys: list[tuple[str, str]], title: str, ncols: int = 3, note: str = ""):
    panels = [{"title": t, "raster": _crop(ev.get(k), w)} for k, t in keys if ev.get(k) is not None]
    if not panels:
        return None, pd.DataFrame()
    return map_grid(grid, w, panels, title=title, ncols=min(ncols, len(panels)), shared=EVIDENCE,
                    cbar_label="evidence score [0, 1] (not a probability)", note=note)


def single_vs_joint(grid, w, ev):
    return evidence_panels(grid, w, ev, [("comp:M", "M: magnetic (G01-A)"), ("comp:G", "G: gravity (G01-B)"),
                                         ("comp:J", "J: tolerant co-location"), ("G01-C", "G01-C = (M + G + J) / 3")],
                           "6. Single-field versus joint magnetic-gravity evidence", ncols=2)


def persistence_maps(grid, w, ev):
    return evidence_panels(grid, w, ev, [("comp:P_mag", "P_mag: magnetic persistence"),
                                         ("comp:P_grav", "P_grav: gravity persistence"),
                                         ("comp:P_cross", "P_cross: tolerant joint persistence"),
                                         ("comp:P", "P = max(P_mag, P_grav)")],
                           "7. Multiscale persistence (share of scales with percentile >= q)", ncols=2)


def six_way(grid, w, ev, note: str = ""):
    from .geology_wavelet import CANDIDATES

    return evidence_panels(grid, w, ev, [(k, f"{k}: {v}") for k, v in CANDIDATES.items()],
                           "8. Six-way G01 candidate comparison (identical scale)", ncols=3, note=note)


# 9 ----------------------------------------------------------------------------------------------- label overlays
def label_overlays(grid, w, scores: dict, sels: dict, truth, known, frac: float):
    panels = []
    for k in list(scores):
        if k not in sels:
            continue
        panels.append({"title": f"{k}: top {100 * frac:g} % of domain", "raster": _crop(scores[k], w),
                       "sel": _crop(sels[k], w), "truth": _crop(truth, w), "known": _crop(known, w),
                       "legend": not panels})
    if not panels:
        return None, pd.DataFrame()
    return map_grid(grid, w, panels, title="9. Equal-area selections vs held-out fault labels (EVALUATION ONLY: "
                    "labels never enter a score)", ncols=min(4, len(panels)), shared={"vmin": 0, "vmax": 1,
                    "cmap": "Greys"}, cbar_label="evidence score")


# 10 / 11 ---------------------------------------------------------------------------------------- curves
def _curves(df: pd.DataFrame, x: str, metrics: list[str], title: str, xlabel: str, logx: bool = False):
    import matplotlib.pyplot as plt

    if not len(df):
        return None, df
    tasks = sorted(df.task.unique())
    fig, axes = plt.subplots(len(tasks), len(metrics), figsize=(W16, H9), squeeze=False, constrained_layout=True)
    for i, t in enumerate(tasks):
        g = df[df.task == t]
        for j, m in enumerate(metrics):
            ax = axes[i, j]
            for c, gc in g.groupby("candidate"):
                gc = gc.sort_values(x)
                ax.plot(gc[x], gc[m], marker="o", ms=3, lw=1.4 if c.startswith("G01") else 1.0,
                        ls="-" if c.startswith("G01") else "--", color=CAND_COLORS.get(c, INK2), label=c)
                if f"{m}_lo" in gc and f"{m}_hi" in gc:
                    ax.fill_between(gc[x], gc[f"{m}_lo"], gc[f"{m}_hi"], color=CAND_COLORS.get(c, INK2), alpha=0.12)
            if m == "enrichment":
                ax.axhline(1.0, color=INK2, lw=0.8, ls=":")
            if logx:
                ax.set_xscale("log")
            ax.set_title(f"{t}: {m}", fontsize=9, loc="left", color=INK)
            ax.set_xlabel(xlabel, fontsize=8, color=INK2)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25)
    axes[0, -1].legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    return fig, df


def perf_vs_area(summary: pd.DataFrame, tol: float, cands: list[str]):
    d = summary[(summary.tol_m == float(tol)) & summary.candidate.isin(cands)]
    return _curves(d, "area_frac", ["precision", "recall", "enrichment"],
                   f"10. Performance vs prediction area (distance tolerance {tol:g} m; pooled held-out zones)",
                   "area fraction of the domain", logx=True)


def perf_vs_distance(summary: pd.DataFrame, frac: float, cands: list[str]):
    d = summary[(summary.area_frac == float(frac)) & summary.candidate.isin(cands)]
    return _curves(d, "tol_m", ["precision", "recall", "enrichment"],
                   f"11. Performance vs distance tolerance (budget {100 * frac:g} % of the domain)", "distance (m)")


# 12 ---------------------------------------------------------------------------------------------- cases
def cases(grid, items: list[dict], cand: str):
    """``items``: {label, window, score, sel, truth, known, M, G} (windows already cropped)."""
    import matplotlib.pyplot as plt

    if not items:
        return None, pd.DataFrame()
    n = len(items)
    fig, axes = plt.subplots(n, 4, figsize=(W16, H9), squeeze=False, constrained_layout=True)
    rows = []
    for i, it in enumerate(items):
        ext = _extent_km(grid, it["window"])
        for j, (k, t) in enumerate((("score", cand), ("sel", "selection + truth"), ("M", "M (magnetic)"),
                                    ("G", "G (gravity)"))):
            ax = axes[i, j]
            a = it.get(k)
            if k == "sel":
                ax.imshow(np.where(np.asarray(a, bool), 1.0, np.nan), extent=ext, cmap="Blues", vmin=0, vmax=1.2,
                          interpolation="nearest", origin="upper")
            elif a is not None:
                ax.imshow(a, extent=ext, vmin=0, vmax=1, cmap="magma", interpolation="nearest", origin="upper")
            _outline(ax, it.get("truth"), ext, TRUTH_C)
            _outline(ax, it.get("known"), ext, KNOWN_C)
            ax.set_aspect("equal")
            ax.set_title(f"{it['label']} | {t}" if j == 0 else t, fontsize=8, loc="left", color=INK)
            ax.tick_params(labelsize=6)
        rows.append({"case": it["label"], "r0": it["window"][0], "c0": it["window"][2], **(it.get("stats") or {})})
    fig.suptitle(f"12. Representative successful and unsuccessful cases ({cand}; red = held-out truth, "
                 "black = known faults)", x=0.01, ha="left", fontsize=11, color=INK)
    return fig, pd.DataFrame(rows)
