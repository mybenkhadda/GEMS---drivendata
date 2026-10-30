"""Page 5 - Feature Comparison & Tournament (EXPLORE / VALIDATE, resumable)."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from lab import featurelang as FL
from lab import tournament as TO

from . import common as C

TEMPLATE = '''PARAMS = {"sigma": {"min": 1, "max": 20, "step": 1, "default": 3},
          "eps": {"min": 0.001, "max": 0.2, "step": 0.001, "default": 0.01}}
feature = ratio(gradmag(rtp, sigma=sigma), gradmag(iso_grav_anom, sigma=sigma), eps=eps)
'''


def render():
    ctx = C.ctx()
    E, store = ctx["engine"], ctx["store"]
    st.header("🏁 Feature Comparison & Tournament")
    st.caption("EXPLORE ranks every combination on the AOI (optimistic). VALIDATE selects parameters and direction on "
               "development folds only and scores the held-out fold - validation pixels never influence selection.")
    f = C.faults()
    if f is None:
        st.error("no aligned mapped-fault raster")
        return
    left, right = st.columns([1.4, 1])
    with left:
        tpl = st.text_area("template (declare tuned parameters in PARAMS)", TEMPLATE, height=170, key="to_tpl")
        try:
            prog = E.compile(tpl)
        except FL.FeatureError as e:
            st.error(e)
            return
        target = st.selectbox("feature to score", prog.names, index=len(prog.names) - 1, key="to_target")
        default_grid = "\n".join(f"{k}: " + ", ".join(f"{v:g}" for v in sorted({p.min, p.default, p.max}))
                                 for k, p in prog.params.items())
        grid_txt = st.text_area("parameter grid (name: v1, v2, ...)", default_grid, height=90, key="to_grid")
    with right:
        mode = st.radio("mode", ["EXPLORE", "VALIDATE"], horizontal=True, key="to_mode")
        budget = st.select_slider("pixel budget", [0.005, 0.01, 0.02, 0.05, 0.1], 0.01, key="to_budget")
        fold_src = st.selectbox("folds (VALIDATE)", ["deterministic blocks", "existing fold raster"], key="to_fsrc",
                                disabled=mode == "EXPLORE")
        k = st.slider("number of folds", 2, 6, 3, key="to_k", disabled=mode == "EXPLORE")
        block_km = st.slider("fold block (km)", 2, 50, 10, key="to_block", disabled=mode == "EXPLORE")
        buf_km = st.slider("buffer between dev and held-out (km)", 0.0, 5.0, 1.0, 0.5, key="to_buf",
                           disabled=mode == "EXPLORE")
        fold_path = st.text_input("fold raster (.npy / .tif, aligned, int; < 0 excluded)", "", key="to_fpath",
                                  disabled=fold_src != "existing fold raster" or mode == "EXPLORE")
    try:
        grid = TO.parse_grid(grid_txt)
    except ValueError as e:
        st.error(e)
        return
    n = 1
    for v in grid.values():
        n *= len(v)
    st.caption(f"{n} combinations • results saved under {store.root / 'tournaments'} (same configuration = resumed)")
    if st.button("▶ Run / resume tournament", type="primary"):
        folds, spec = None, "none"
        if mode == "VALIDATE":
            if fold_src == "existing fold raster":
                folds, spec = TO.load_fold_raster(fold_path, C.window()), f"raster:{fold_path}"
            else:
                bp = int(block_km * 1000 / E.pixel_m)
                folds, spec = TO.block_folds(f.shape, bp, k), f"blocks:{bp}px:k{k}"
        bar = st.progress(0.0)
        try:
            out = TO.run(E, tpl, target, grid, C.window(), f, mode, folds, int(buf_km * 1000 / E.pixel_m), budget,
                         store.root / "tournaments", lambda i, m: bar.progress(i / m, text=f"{i}/{m}"), spec)
        except (ValueError, FL.FeatureError) as e:
            st.error(e)
            return
        st.session_state["to_out"] = out
    out = st.session_state.get("to_out")
    if not out:
        return
    R = out["results"]
    st.success(f"{len(R)} combinations • {out['dir']}")
    st.dataframe(R, hide_index=True, use_container_width=True)
    C.json_block(out["summary"])
    pcols = [c for c in R if c.startswith("param_")]
    if out["manifest"]["mode"] == "EXPLORE" and pcols and "auroc" in R:
        fig = px.scatter(R, x=pcols[0], y="auroc", color=pcols[1] if len(pcols) > 1 else None,
                         error_y=R.regional_auroc_std if "regional_auroc_std" in R else None,
                         title="AUROC by parameters (error bar: std across 3 x 3 tiles)")
        fig.update_layout(template=C.DARK, height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.warning(out["summary"].get("selection_bias_warning", ""))
    elif pcols:
        P = pd.DataFrame(out["summary"].get("selected_per_fold", []))
        st.markdown("**Dev-selected combination per held-out fold** (the only honest estimate)")
        st.dataframe(P, hide_index=True)
