"""Page 6 - W2 / M9 complementarity at identical realised area budgets (ranking diagnostic, not DTI)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lab import complementarity as CP
from lab import figures as FG

from . import common as C

COLORS = ["#202020", "#1b9e77", "#7570b3", "#e6ab02", "#d95f02", "#4c4c8c", "#a6761d", "#666666"]


def render():
    E = C.ctx()["engine"]
    st.header("🤝 W2 / M9 Complementarity")
    st.caption("Register baseline maps in the sidebar as `pred:w2=/path/w2_rank.tif` (aligned single band). A held-out "
               "claim needs `<raster>.provenance.json` with held_out=true and its folds / zones; otherwise the map is "
               "treated as IN-SAMPLE. This is NOT the official DTI - use the existing scorer for scores.")
    if not E.predictions:
        st.info("No prediction layer registered (sidebar → Extra aligned layers → `pred:name=/path.tif`).")
        return
    f = C.faults()
    if f is None:
        st.error("no aligned mapped-fault raster")
        return
    a, b, c, d = st.columns(4)
    base = a.selectbox("baseline prediction", E.predictions, key="cp_base")
    feat = b.selectbox("feature", C.all_layer_names(), key="cp_feat")
    dirn = c.selectbox("feature direction (fixed beforehand)", ["high", "low"], key="cp_dir")
    budget = d.select_slider("area budget (share of valid AOI px)", [0.005, 0.01, 0.02, 0.05, 0.1], 0.01,
                             key="cp_budget")
    radius = st.slider("recovery radius (px; 3 = 300 m)", 1, 10, 3, key="cp_r")
    prov = CP.provenance(E.sources[base].path)
    (st.success if prov["status"] == "HELD_OUT_VERIFIED" else st.warning)(
        f"baseline provenance: {prov['status']} {prov.get('reason', '')}")
    L = C.get_layers([base, feat])
    fp = E.footprint(C.window())
    T, cat, comp = CP.compare(L[base], L[feat], f, fp, budget, radius, feature_direction=dirn)
    st.dataframe(T, hide_index=True, use_container_width=True)
    st.json({k: v for k, v in T.attrs.items()})
    fig = go.Figure(go.Heatmap(z=cat.astype(float), colorscale=[[i / 7, COLORS[i]] for i in range(8)], zmin=0,
                               zmax=7, colorbar=dict(tickvals=list(CP.CATEGORY), ticktext=list(CP.CATEGORY.values()))))
    fig.update_layout(template=C.DARK, height=520, title="recovery categories (fault pixels) and unlabelled selections")
    fig.update_yaxes(autorange="reversed", scaleanchor="x")
    st.plotly_chart(fig, use_container_width=True)
    if len(comp):
        st.dataframe(comp.category.value_counts().rename_axis("component category").reset_index(), hide_index=True)
    C.save_buttons("complementarity", {f"complementarity_{base}_{feat}": T.assign(provenance=prov["status"]),
                                       f"components_{base}_{feat}": comp if len(comp) else pd.DataFrame()},
                   [lambda s, dd: FG.map_png(s, dd, f"categories_{base}_{feat}", cat.astype(float), C.window(),
                                             E.details["transform"], cmap="tab10", categorical=True)], key="cp")
