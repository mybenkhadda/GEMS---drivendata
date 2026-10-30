"""Page 4 - Local & Regional Analysis: tile association heatmaps (n x n, fixed km tiles, label-free domains), tile
inspection vs global, moving-window association map."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from rasterio.windows import Window

from lab import association as A
from lab import figures as FG

from . import common as C

METRICS = {"local point-biserial": "point_biserial", "local AUROC": "auroc", "local AP lift": "ap_lift",
           "local recall @ budget": "recall_at_budget", "local matched-control effect": "matched_delta"}


def render():
    E = C.ctx()["engine"]
    st.header("🧩 Local & Regional Analysis")
    f = C.faults()
    if f is None:
        st.error("no aligned mapped-fault raster")
        return
    names = C.all_layer_names()
    a, b, c, d = st.columns(4)
    one = a.selectbox("feature", names, key="rg_feat")
    dirn = b.selectbox("direction (fixed beforehand)", ["high", "low"], key="rg_dir")
    part = c.selectbox("partition", ["2 x 2", "3 x 3", "5 x 5", "10 x 10", "1 km tiles", "2 km tiles", "5 km tiles",
                                      "10 km tiles", "20 km tiles", "label-free domains"], index=2, key="rg_part")
    metric = d.selectbox("heatmap metric", list(METRICS), index=1, key="rg_metric")
    a2, b2, c2 = st.columns(3)
    budget = a2.select_slider("pixel budget", [0.005, 0.01, 0.02, 0.05, 0.1], 0.01, key="rg_budget")
    min_pos = b2.number_input("min fault px per tile", 3, 500, 20, key="rg_minpos")
    min_neg = c2.number_input("min unlabelled px per tile", 10, 5000, 200, key="rg_minneg")
    x = C.get_layers([one])[one]
    dom = E.domains(C.window())
    if part.endswith("domains"):
        mode, n, tile = "domain", 0, 0
        if dom is None:
            st.warning("no label-free domain raster (outputs/geological_discovery_eda/rasters/spatial_domains/"
                       "domains.npz)")
            return
    elif "x" in part:
        mode, n, tile = "grid", int(part.split(" x ")[0]), 0
    else:
        mode, n, tile = "fixed", 0, int(float(part.split(" ")[0]) * 1000 / E.pixel_m)
    T = A.tile_table(x, f, mode, n, tile, dom, None, dirn, budget, int(min_pos), int(min_neg))
    g = A.metrics(x, f, None, dirn, budgets=(budget,))
    st.caption(f"global (this AOI): AUROC {g.get('auroc', np.nan):.3f} • AP lift {g.get('ap_lift', np.nan):.2f} • "
               f"{(T.status == A.INSUFFICIENT).sum()} of {len(T)} tiles INSUFFICIENT_DATA")
    col = METRICS[metric]
    if mode != "domain":
        nr, nc = int(T.tile_row.max()) + 1, int(T.tile_col.max()) + 1
        Z = np.full((nr, nc), np.nan)
        txt = np.full((nr, nc), "", object)
        for r in T.itertuples():
            v = getattr(r, col, np.nan) if r.status == "OK" else np.nan
            Z[r.tile_row, r.tile_col] = v
            txt[r.tile_row, r.tile_col] = "INSUFFICIENT" if r.status != "OK" else f"{v:.2f}<br>n+={r.n_fault_px}"
        zmid = 0.5 if col == "auroc" else (1.0 if col == "ap_lift" else 0.0)
        fig = go.Figure(go.Heatmap(z=Z, text=txt, texttemplate="%{text}", colorscale="RdBu_r", zmid=zmid,
                                   hovertemplate="tile %{y},%{x}<br>%{z:.3f}<extra></extra>"))
        fig.update_layout(template=C.DARK, height=460, title=f"{metric}: {one}", xaxis_title="tile column",
                          yaxis_title="tile row")
        fig.update_yaxes(autorange="reversed")
        ev = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points", key="rg_hm")
        pts = []
        try:
            pts = (ev or {}).get("selection", {}).get("points", [])
        except AttributeError:
            pass
        if pts:
            st.session_state["rg_tile"] = (int(pts[0]["y"]), int(pts[0]["x"]))
    else:
        fig = go.Figure(go.Bar(x=T.label, y=T[col] if col in T else None,
                               text=np.where(T.status == "OK", "", "INSUFFICIENT")))
        fig.update_layout(template=C.DARK, height=380, title=f"{metric} by label-free domain")
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(T, hide_index=True, use_container_width=True)
    C.save_buttons("regional_association", {f"regional_{one}_{mode}": T}, [lambda s, d: FG.heatmap_png(
        s, d, f"regional_{one}_{col}", Z, f"{metric}: {one}", data=T)] if mode != "domain" else None, key="rg")
    if mode != "domain":
        st.markdown("#### Tile inspection (click a heatmap cell or choose)")
        tr, tc = st.session_state.get("rg_tile", (0, 0))
        a, b = st.columns(2)
        tr = a.number_input("tile row", 0, int(T.tile_row.max()), int(min(tr, T.tile_row.max())), key="rg_tr")
        tc = b.number_input("tile col", 0, int(T.tile_col.max()), int(min(tc, T.tile_col.max())), key="rg_tc")
        row = T[(T.tile_row == tr) & (T.tile_col == tc)].iloc[0]
        r0, r1, c0, c1 = int(row.r0), int(row.r1), int(row.c0), int(row.c1)
        w = C.window()
        sub = Window(w.col_off + c0, w.row_off + r0, c1 - c0, r1 - r0)
        left, right = st.columns([1.2, 1])
        left.plotly_chart(C.map_fig(x[r0:r1, c0:c1], f"{one} tile {tr},{tc}", sub, f[r0:r1, c0:c1]),
                          use_container_width=True)
        with right:
            cmp_ = pd.DataFrame([{"scope": "tile", **{k: row.get(k) for k in ("status", "auroc", "ap_lift",
                                                                                "point_biserial", "recall_at_budget",
                                                                                "matched_delta", "n_fault_px")}},
                                 {"scope": "AOI (global)", "status": g.get("status"), "auroc": g.get("auroc"),
                                  "ap_lift": g.get("ap_lift"), "point_biserial": g.get("point_biserial"),
                                  "recall_at_budget": g.get(f"recall_at_{budget * 100:g}pct"),
                                  "n_fault_px": g.get("n_fault_px")}])
            st.dataframe(cmp_, hide_index=True)
            xs, fs = x[r0:r1, c0:c1], f[r0:r1, c0:c1]
            ok = np.isfinite(xs)
            fig = go.Figure([go.Histogram(x=xs[ok & fs], name="fault", histnorm="probability density", opacity=.6),
                             go.Histogram(x=xs[ok & ~fs], name="unlabelled", histnorm="probability density",
                                          opacity=.6)])
            fig.update_layout(template=C.DARK, barmode="overlay", height=300)
            st.plotly_chart(fig, use_container_width=True)
    st.divider()
    st.markdown("#### Moving-window association (point-biserial)")
    win_km = st.select_slider("window (km)", [1.1, 2.1, 3.1, 5.1, 9.1], 3.1, key="rg_win")
    wpx = max(3, int(round(win_km * 1000 / E.pixel_m)))
    wpx += (wpx + 1) % 2
    with st.spinner("moving window ..."):
        mv, _ = A.moving_association(x, f, wpx)
    st.plotly_chart(C.map_fig(mv, f"moving point-biserial ({wpx} px window)", C.window(), f, "RdBu_r", zmin=-0.5,
                              zmax=0.5), use_container_width=True)
    st.caption("Windows with < 3 mapped-fault pixels are blank (insufficient support), not zero association.")
