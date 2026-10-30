"""Page 3 - Fault Association Explorer: leaderboard (frozen-direction option), distributions, proximity profile,
fault-normal profiles, matched controls, block-bootstrap CI and shifted-target negative controls."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lab import association as A
from lab import figures as FG

from . import common as C

LEADER_COLS = ["feature", "status", "direction", "direction_source", "point_biserial", "spearman", "auroc", "auroc_high",
               "average_precision", "ap_lift", "recall_at_0.5pct", "recall_at_1pct", "recall_at_2pct",
               "recall_at_5pct", "recall_at_10pct", "median_difference", "cliffs_delta", "robust_effect",
               "near_trace_enrichment", "association_sign", "n_valid", "n_fault_px", "prevalence"]


def render():
    E = C.ctx()["engine"]
    st.header("🎯 Fault Association Explorer")
    st.caption("Target: data/raw/faults/existing_faults.tif. Unlabelled pixels are NOT confirmed non-faults; AUROC / AP "
               "here rank mapped faults above unlabelled pixels (not DTI).")
    f_ext, roi = C.faults(halo=60)
    if f_ext is None:
        st.error("no aligned mapped-fault raster")
        return
    f = f_ext[roi]
    names = C.all_layer_names()
    a, b, c = st.columns([2, 1, 1])
    sel = a.multiselect("features", names, default=[n for n in names if n in (C.program_result().arrays
                        if C.program_result() else {})][:6] or names[:4], key="as_sel")
    mode = b.selectbox("direction", ["high", "low", "frozen (development split)", "best_exploratory"], key="as_dir")
    use_fp = c.checkbox("restrict to submission footprint", True, key="as_fp")
    split = st.selectbox("development / evaluation split (frozen mode)", ["halves_ew", "halves_ns", "checkerboard"],
                         key="as_split", disabled=not mode.startswith("frozen"))
    if not sel:
        return
    L = C.get_layers(sel)
    fp = E.footprint(C.window()) if use_fp else None
    dev = ev = None
    if mode.startswith("frozen"):
        dev, ev = A.split_masks(f.shape, split, block_px=50, buffer_px=5)
        if fp is not None:
            dev, ev = dev & fp, ev & fp
    T = A.leaderboard(L, f, fp if not mode.startswith("frozen") else None, mode.split(" ")[0], dev, ev,
                      faults_ext=f_ext, roi=roi)
    st.dataframe(T[[c for c in LEADER_COLS if c in T]], hide_index=True, use_container_width=True)
    if mode == "best_exploratory":
        st.warning("best_exploratory picks the better direction on the SAME pixels: optimistic, exploratory only.")
    if mode.startswith("frozen"):
        st.info(f"direction chosen on the development part ({split}), metrics on the evaluation part only.")
    C.save_buttons("fault_association", {"fault_association": T}, [lambda s, d: FG.bar_png(
        s, d, "auroc_leaderboard", T, "feature", "auroc")], key="as")
    lib = C.ctx()["library"]
    saved = [n for n in T.feature if n in lib.features]
    if saved and st.button(f"record these results in the library for {saved}"):
        for r in T[T.feature.isin(saved)].to_dict("records"):
            lib.record_evaluation(r["feature"], {"aoi": list(st.session_state["aoi"]), "mode": mode, "split": split
                                                 if mode.startswith("frozen") else None,
                                                 **{k: r.get(k) for k in ("direction", "auroc", "ap_lift",
                                                                          "recall_at_1pct", "n_valid", "n_fault_px",
                                                                          "status")}})
        st.success("recorded (registry evaluations list; AOI-level, not a held-out claim unless frozen mode)")
    st.divider()
    one = st.selectbox("inspect feature", sel, key="as_one")
    x = L[one]
    dirn = T.set_index("feature").direction.get(one, "high") if "direction" in T else "high"
    t1, t2, t3, t4 = st.tabs(["distributions", "distance to faults", "fault-normal profile", "formal comparison"])
    with t1:
        rng = np.random.default_rng(2026)
        ok = np.isfinite(x)
        fa, un = x[ok & f], x[ok & ~f]
        un = un[rng.choice(len(un), min(len(un), 20000), replace=False)] if len(un) else un
        fig = go.Figure([go.Histogram(x=fa, name="mapped fault", histnorm="probability density", opacity=0.6,
                                      nbinsx=70),
                         go.Histogram(x=un, name="unlabelled (sample)", histnorm="probability density", opacity=0.6,
                                      nbinsx=70)])
        fig.update_layout(template=C.DARK, barmode="overlay", height=380, title=one)
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        P = A.proximity_profile(x, f_ext, roi, E.pixel_m)
        fig = go.Figure([go.Scatter(x=P.band, y=P.p75, line=dict(width=0), showlegend=False),
                         go.Scatter(x=P.band, y=P.p25, fill="tonexty", line=dict(width=0), name="p25-p75"),
                         go.Scatter(x=P.band, y=P["median"], name="median", mode="lines+markers"),
                         go.Scatter(x=P.band, y=P.p05, name="p05", line=dict(dash="dot")),
                         go.Scatter(x=P.band, y=P.p95, name="p95", line=dict(dash="dot"))])
        fig.update_layout(template=C.DARK, height=380, title=f"{one} by distance to mapped faults (retrospective)",
                          xaxis_title="distance band")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(P, hide_index=True)
        st.caption("Distance-to-known-fault is a DIAGNOSTIC: it must never become a predictor of unseen faults.")
        C.save_buttons("proximity_profile", {f"proximity_{one}": P}, [lambda s, d: FG.lines_png(
            s, d, f"proximity_{one}", P.assign(i=range(len(P))), "i", ["median", "p05", "p95"], ("p25", "p75"),
            xlabel="distance band index (see CSV)")], key="prox")
    with t3:
        half = st.slider("half length (px)", 10, 100, 30, 5, key="fn_half")
        N = A.fault_normal_profiles(x, f_ext, roi, half)
        fig = go.Figure([go.Scatter(x=N.offset_px * E.pixel_m / 1000, y=N.p75, line=dict(width=0), showlegend=False),
                         go.Scatter(x=N.offset_px * E.pixel_m / 1000, y=N.p25, fill="tonexty", line=dict(width=0),
                                    name="p25-p75"),
                         go.Scatter(x=N.offset_px * E.pixel_m / 1000, y=N["median"], name="median")])
        fig.update_layout(template=C.DARK, height=360, xaxis_title="offset from mapped trace along its normal (km)",
                          title="Fault-normal profile (retrospective; normal polarity not aligned)")
        st.plotly_chart(fig, use_container_width=True)
        C.download_csv(N, f"fault_normal_{one}", "fn")
    with t4:
        st.caption("Spatial block bootstrap (2 km blocks), shifted-target negative controls and physically matched "
                   "controls (same 5 km tile and covariate tercile, controls >= 1 km from mapped traces).")
        cov = st.selectbox("matching covariate (label-free)", [n for n in E.inputs], index=E.inputs.index(
            "det_elev_slope") if "det_elev_slope" in E.inputs else 0, key="mc_cov")
        if st.button("run formal comparison", key="formal"):
            with st.spinner("bootstrap + surrogates + matched controls ..."):
                bb = A.block_bootstrap(x, f, fp, dirn)
                nc = A.negative_control(x, f, fp, dirn)
                mc = A.matched_control_effect(x, f, C.get_layers([cov])[cov], fp, tile_px=50, min_dist_px=10)
            R = pd.DataFrame([{"feature": one, "direction": dirn, **{k: v for k, v in bb.items()},
                               **{f"negctl_{k}": v for k, v in nc.items()},
                               **{f"matched_{k}": v for k, v in mc.items() if k != "strata"}}])
            st.session_state["formal"] = (one, R, mc["strata"], nc["exceeds_null_p95"], st.session_state["aoi"])
        fm = st.session_state.get("formal")
        if fm and fm[0] == one and fm[4] == st.session_state["aoi"]:
            _, R, strata, beats, _ = fm
            st.dataframe(R.astype(str).T.rename(columns={0: "value"}), use_container_width=True)
            if not beats:
                st.warning("the feature does not beat the 95th percentile of shifted-target surrogates: no evidence "
                           "beyond spatial structure")
            C.save_buttons("formal_comparison", {f"formal_{one}": R, f"matched_strata_{one}": strata},
                           key="formal_save")
