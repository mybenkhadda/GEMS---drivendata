"""Page 1 - Geological Map Explorer: overview + box AOI, synchronised maps, stretch / colour map, statistics,
raw vs transformed, cross-sections."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lab import figures as FG
from lab import transects as TR
from lab.rasters import overview

from . import common as C

OV_SIDE = 360


def _overview(E):
    st.markdown("#### Overview — drag a box to set the AOI")
    band = st.selectbox("overview band", E.inputs, index=E.inputs.index("rtp") if "rtp" in E.inputs else 0,
                        key="ov_band")
    src = E.sources[band]
    if src.kind != "tif":
        st.info("overview available for TIFF bands only")
        return
    ov = overview(src.path, src.band, OV_SIDE)
    r0, c0, h, w = st.session_state["aoi"]
    sy, sx = OV_SIDE / E.H, OV_SIDE / E.W
    lo, hi = C.stretch(ov, (2, 98))
    fig = go.Figure(go.Heatmap(z=ov, colorscale="Viridis", zmin=lo, zmax=hi, showscale=False,
                               hovertemplate="row %{y:.0f} col %{x:.0f}<extra></extra>"))
    fig.add_shape(type="rect", x0=c0 * sx, x1=(c0 + w) * sx, y0=r0 * sy, y1=(r0 + h) * sy, line=dict(color="lime"))
    fig.update_layout(template=C.DARK, height=420, dragmode="select", margin=dict(l=5, r=5, t=10, b=5))
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=E.H / E.W)     # true ground aspect
    ev = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="box", key="ov_sel")
    try:
        box = (ev or {}).get("selection", {}).get("box", [])
    except AttributeError:
        box = []
    if box:
        b = box[0]
        xs, ys = sorted(b["x"]), sorted(b["y"])
        nr0, nr1 = int(max(0, ys[0] / sy)), int(min(E.H, ys[1] / sy))
        nc0, nc1 = int(max(0, xs[0] / sx)), int(min(E.W, xs[1] / sx))
        side = max(nr1 - nr0, nc1 - nc0)
        side = int(min(max(side, 10), 1000))
        new = (max(0, min(E.H - side, nr0)), max(0, min(E.W - side, nc0)), min(side, E.H), min(side, E.W))
        if new != st.session_state["aoi"]:
            st.session_state["aoi"] = new
            st.rerun()


def _transects(E, names):
    st.markdown("#### Cross-section explorer")
    r0, c0, h, w = st.session_state["aoi"]
    a, b, c, d, e = st.columns(5)
    sr = a.number_input("start row (AOI px)", 0, h - 1, h // 2, key="tr_sr")
    sc = b.number_input("start col", 0, w - 1, 0, key="tr_sc")
    er = c.number_input("end row", 0, h - 1, h // 2, key="tr_er")
    ec = d.number_input("end col", 0, w - 1, w - 1, key="tr_ec")
    spacing = e.select_slider("spacing (px)", [0.5, 1.0, 2.0, 5.0], 1.0, key="tr_sp")
    sel = st.multiselect("signals", names, default=names[:3], key="tr_sig")
    norm = st.checkbox("normalise for display (robust z; stored values unchanged)", True, key="tr_norm")
    if not sel:
        return
    L = C.get_layers(sel)
    T, marks = TR.transect(L, (sr, sc), (er, ec), spacing, E.pixel_m, C.faults())
    D = TR.display_normalize(T, sel) if norm else T
    fig = go.Figure([go.Scatter(x=D.distance_km, y=D[k], name=k, mode="lines") for k in sel])
    for m in marks:
        fig.add_vline(x=m, line=dict(color="crimson", dash="dash", width=1))
    fig.update_layout(template=C.DARK, height=360, xaxis_title="distance along transect (km)",
                      yaxis_title="robust z (display)" if norm else "value", title="red dashes: mapped-fault crossings "
                      "(diagnostic)")
    st.plotly_chart(fig, use_container_width=True)
    C.download_csv(T, "transect_samples", "tr")
    C.save_buttons("transect", {"transect_samples": T}, [lambda s, d: FG.lines_png(
        s, d, "transect", D, "distance_km", sel, xlabel="km", markers=marks)], key="tr")


def render():
    E = C.ctx()["engine"]
    st.header("🗺 Geological Map Explorer")
    left, right = st.columns([1, 2.2])
    with left:
        _overview(E)
        st.caption("A drag-selected box becomes a square AOI (max 100 km). Fine adjustments: sidebar.")
    names = C.all_layer_names(include_predictions=True)
    with right:
        sel = st.multiselect("layers (side by side, zoom synchronised)", names, default=names[:2], max_selections=4,
                             key="map_sel")
        a, b, c = st.columns(3)
        cmap = a.selectbox("colour map", C.CMAPS, key="map_cmap")
        pct = b.slider("percentile stretch", 0.0, 10.0, 2.0, 0.5, key="map_pct")
        show_f = c.checkbox("overlay mapped faults", True, key="map_f")
        if sel:
            L = C.get_layers(sel)
            f = C.faults() if show_f else None
            st.plotly_chart(C.multi_map(L, C.window(), f, cmap, (pct, 100 - pct)), use_container_width=True)
            units = C.units_table()
            S = pd.DataFrame([{**C.stats_row(k, v), "units": units.get(k, "derived / not tagged")} for k, v in
                              L.items()])
            st.dataframe(S, use_container_width=True, hide_index=True)
            C.save_buttons("map_view", {"layer_statistics": S}, [
                (lambda s, d, k=k, v=v: FG.map_png(s, d, k, v, C.window(), E.details["transform"], f,
                                                   pct=(pct, 100 - pct))) for k, v in L.items()], key="maps")
    st.divider()
    st.markdown("#### Raw vs transformed")
    a, b, c = st.columns(3)
    base = a.selectbox("layer", E.inputs, key="rt_layer")
    op = b.selectbox("transform", ["highpass", "gaussian", "gradmag", "local_std", "laplacian", "ridge_strength",
                                   "coherence", "orientation", "local_zscore"], key="rt_op")
    sig = c.slider("sigma (px; 10 px = 1 km)", 0.5, 40.0, 5.0, 0.5, key="rt_sig")
    prog = E.compile(f"transformed = {op}({base}, sigma={sig})")
    res = E.evaluate(prog, C.window())
    raw = C.get_layers([base])[base]
    st.plotly_chart(C.multi_map({base: raw, f"{op}({base}, {sig})": res.arrays["transformed"]}, C.window(),
                                C.faults(), "Viridis"), use_container_width=True)
    C.warn_box(res.diagnostics["transformed"]["warnings"])
    st.divider()
    _transects(E, names)
