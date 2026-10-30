"""Shared UI state and helpers of the Feature Discovery Studio (Streamlit only; no science here)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from rasterio.windows import Window

DARK = "plotly_dark"
CMAPS = ["Viridis", "Cividis", "Magma", "Inferno", "RdBu_r", "Spectral_r", "Greys", "Turbo", "Twilight"]


# ------------------------------------------------------------------ state
def ctx():
    """The shared objects set by app.py: engine, store, library, window."""
    return st.session_state["_ctx"]


def window() -> Window:
    r, c, h, w = st.session_state["aoi"]
    return Window(c, r, w, h)


def program_result():
    """Last Feature Lab evaluation if it belongs to the current AOI, else None."""
    res = st.session_state.get("lab_result")
    if res is None:
        return None
    w = window()
    rw = res.window
    same = (int(rw.row_off), int(rw.col_off), int(rw.height), int(rw.width)) == (int(w.row_off), int(w.col_off),
                                                                                  int(w.height), int(w.width))
    return res if same else None


def layer_catalog() -> dict:
    """{display group: [names]} of everything that can be shown / evaluated on the AOI."""
    E = ctx()["engine"]
    res = program_result()
    lib = E.library.features if E.library else {}
    return {"raw inputs": E.inputs, "program variables": list(res.arrays) if res else [],
            "saved features": list(lib), "predictions (comparison only)": E.predictions}


def all_layer_names(include_predictions=False) -> list:
    cat = layer_catalog()
    out = cat["raw inputs"] + cat["program variables"] + [n for n in cat["saved features"]
                                                          if n not in cat["program variables"]]
    return out + (cat["predictions (comparison only)"] if include_predictions else [])


def get_layers(names) -> dict:
    """Arrays of the current AOI for raw / prediction layers, program variables and saved features."""
    E = ctx()["engine"]
    w = window()
    res = program_result()
    out, lib = {}, []
    raw = [n for n in names if n in E.sources]
    if raw:
        out.update(E.raw(raw, w))
    for n in names:
        if n in out:
            continue
        if res is not None and n in res.arrays:
            out[n] = res.arrays[n]
        elif E.library and n in E.library.features:
            lib.append(n)
    if lib:
        out.update(E.evaluate(None, w, library_targets=lib).arrays)
    return {n: out[n] for n in names if n in out}


def faults(halo: int = 0):
    E = ctx()["engine"]
    return E.faults(window(), halo) if halo else E.faults(window())


# ------------------------------------------------------------------ plotting
def axes_km(arr_shape, w: Window, step: int):
    a = ctx()["engine"].details["transform"]
    cc = w.col_off + np.arange(0, arr_shape[1], step) + 0.5
    rr = w.row_off + np.arange(0, arr_shape[0], step) + 0.5
    return (a[2] + cc * a[0]) / 1000, (a[5] + rr * a[4]) / 1000


def stretch(v, pct):
    g = v[np.isfinite(v)]
    if not g.size:
        return 0.0, 1.0
    lo, hi = np.percentile(g, pct)
    return float(lo), float(hi if hi > lo else lo + 1e-9)


def heat(z, w: Window, cmap="Viridis", pct=(2, 98), name="", max_side=520, zmin=None, zmax=None):
    a = np.asarray(z, np.float32)
    step = max(1, int(np.ceil(max(a.shape) / max_side)))
    v = a[::step, ::step]
    xs, ys = axes_km(a.shape, w, step)
    lo, hi = (zmin, zmax) if zmin is not None else stretch(v, pct)
    return go.Heatmap(x=xs, y=ys, z=v, colorscale=cmap, zmin=lo, zmax=hi, colorbar=dict(title=name[:18], len=0.8),
                      hovertemplate="E %{x:.2f} km<br>N %{y:.2f} km<br>" + name + " %{z:.5g}<extra></extra>"), step


def fault_trace(fmask, w: Window, max_points=8000):
    if fmask is None or not np.any(fmask):
        return None
    rr, cc = np.nonzero(fmask)
    if len(rr) > max_points:
        i = np.random.default_rng(2026).choice(len(rr), max_points, replace=False)
        rr, cc = rr[i], cc[i]
    a = ctx()["engine"].details["transform"]
    return go.Scattergl(x=(a[2] + (w.col_off + cc + 0.5) * a[0]) / 1000,
                        y=(a[5] + (w.row_off + rr + 0.5) * a[4]) / 1000, mode="markers", name="mapped faults",
                        marker=dict(size=2.5, color="crimson", opacity=0.75), hoverinfo="skip", showlegend=False)


def map_fig(z, name, w: Window, fmask=None, cmap="Viridis", pct=(2, 98), height=480, zmin=None, zmax=None):
    tr, _ = heat(z, w, cmap, pct, name, zmin=zmin, zmax=zmax)
    fig = go.Figure(tr)
    ft = fault_trace(fmask, w)
    if ft is not None:
        fig.add_trace(ft)
    fig.update_layout(template=DARK, title=name, height=height, margin=dict(l=5, r=5, t=40, b=5),
                      xaxis_title="UTM E (km)", yaxis_title="UTM N (km)")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def multi_map(arrays: dict, w: Window, fmask=None, cmap="Viridis", pct=(2, 98), height=440):
    """Side-by-side maps with SHARED axes (zoom / pan synchronised)."""
    n = len(arrays)
    fig = make_subplots(rows=1, cols=n, shared_xaxes=True, shared_yaxes=True, subplot_titles=list(arrays),
                        horizontal_spacing=0.03)
    for i, (k, a) in enumerate(arrays.items(), start=1):
        tr, _ = heat(a, w, cmap, pct, k, max_side=360)
        tr.update(colorbar=dict(x=i / n - 0.01, len=0.7, thickness=8))
        fig.add_trace(tr, 1, i)
        ft = fault_trace(fmask, w, 4000)
        if ft is not None:
            fig.add_trace(ft, 1, i)
    fig.update_layout(template=DARK, height=height, margin=dict(l=5, r=5, t=40, b=5))
    for i in range(1, n + 1):
        fig.update_yaxes(scaleanchor=f"x{i if i > 1 else ''}", scaleratio=1, row=1, col=i)
    return fig


def stats_row(name, a) -> dict:
    v = np.asarray(a, np.float32)
    g = v[np.isfinite(v)]
    r = {"layer": name, "valid_px": int(g.size), "valid_share": float(g.size / v.size) if v.size else 0.0}
    if g.size:
        r.update(min=float(g.min()), p02=float(np.percentile(g, 2)), median=float(np.median(g)),
                 p98=float(np.percentile(g, 98)), max=float(g.max()), mean=float(g.mean()), std=float(g.std()))
    return r


def units_table() -> dict:
    """Band units from the EDA inventory (audit/band_metadata.csv) when present."""
    root = ctx()["root"]
    p = Path(root) / "outputs/geological_discovery_eda/audit/band_metadata.csv"
    if p.exists():
        try:
            T = pd.read_csv(p)
            return dict(zip(T.band, T.units.fillna("not tagged")))
        except (OSError, ValueError, AttributeError):
            pass
    return {}


def save_buttons(label: str, tables: dict | None = None, figures: list | None = None, key: str = ""):
    """'Save to Drive' button: tables {name: DataFrame} -> CSV + Parquet, figures [(callable(store, run_dir))]."""
    store = ctx()["store"]
    if st.button(f"💾 Save {label} to outputs", key=f"save_{key or label}"):
        d = store.run_dir("tables" if not figures else "figures", label)
        written = []
        for n, T in (tables or {}).items():
            written += store.table(T, d / n)
        for f in figures or []:
            written += f(store, d)
        store.json({"label": label, "aoi": list(st.session_state["aoi"]), "files": [str(p) for p in written]},
                   d / "manifest.json")
        st.success(f"Saved {len(written)} files to {d}")


def download_csv(T: pd.DataFrame, name: str, key: str = ""):
    st.download_button(f"⬇ {name}.csv", T.to_csv(index=False).encode(), file_name=f"{name}.csv",
                       key=f"dl_{key or name}")


def warn_box(msgs):
    for m in msgs:
        st.warning(m)


def json_block(d):
    st.code(json.dumps(d, indent=2, default=str), language="json")
