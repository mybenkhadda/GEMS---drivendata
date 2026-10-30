"""Page 2 - Python Feature Laboratory: restricted multi-line formula editor, PARAMS sliders, Run Features, maps,
diagnostics, comparison with sources, parameter diff, quick fault evaluation, freezing, saving to the library,
trusted plugins and the function reference."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from lab import association as A
from lab import featurelang as FL
from lab import ops as O
from lab.plugins import load_plugins

from . import common as C

DEFAULT_PROGRAM = '''# sigma / size in PIXELS (1 px = 100 m). Every variable becomes a feature.
PARAMS = {"sigma": {"min": 1, "max": 20, "step": 1, "default": 3},
          "eps": {"min": 0.001, "max": 0.2, "step": 0.001, "default": 0.01}}
mag = gradmag(rtp, sigma=sigma)
grav = gradmag(iso_grav_anom, sigma=sigma)
ratio_mg = ratio(mag, grav, eps=eps)
terrain = highpass(det_elev, sigma=8)
feature_A = abs(terrain) * ratio_mg
mag_fabric = orientation(rtp, sigma=5)
terrain_fabric = orientation(det_elev, sigma=5)
fabric_agreement = axial_agreement(mag_fabric, terrain_fabric)
'''


def _editor(store):
    if "program_text" not in st.session_state:
        last = store.root / "registry" / "last_program.py"
        st.session_state["program_text"] = last.read_text(encoding="utf-8") if last.exists() else DEFAULT_PROGRAM
    try:
        from streamlit_ace import st_ace                # optional nicer editor

        txt = st_ace(value=st.session_state["program_text"], language="python", theme="tomorrow_night", height=300,
                     key=f"ace_program_{st.session_state.get('editor_ver', 0)}", auto_update=True, font_size=13)
        if txt is not None:
            st.session_state["program_text"] = txt
    except ImportError:
        st.text_area("Formula code", key="program_text", height=300,
                     help="one assignment per line; PARAMS (optional) first")
    return st.session_state["program_text"]


def _set_program(text: str):
    """Callback (runs before widgets are created, so the editor can be updated)."""
    st.session_state["program_text"] = text
    st.session_state["editor_ver"] = st.session_state.get("editor_ver", 0) + 1


def _reset():
    _set_program(DEFAULT_PROGRAM)
    st.session_state.pop("param_values", None)


def _freeze(text, normalization):
    new, done = FL.freeze_normalization(text, normalization)
    _set_program(new)
    st.session_state["freeze_msg"] = f"frozen: {done}"


def _compile(E, text):
    try:
        return E.compile(text, st.session_state.get("param_values")), None
    except FL.FeatureError as e:
        return None, str(e)


def _sliders(prog):
    vals = dict(st.session_state.get("param_values") or {})
    if not prog.params:
        return vals
    st.markdown("**Parameters** (declared in PARAMS)")
    cols = st.columns(min(4, len(prog.params)))
    for i, (k, p) in enumerate(prog.params.items()):
        vals[k] = cols[i % len(cols)].slider(k, float(p.min), float(p.max), float(vals.get(k, p.default)),
                                             float(p.step), key=f"param_{k}")
    st.session_state["param_values"] = vals
    return vals


def _save_block(E, prog, res, lib):
    st.markdown("#### Save features to the library")
    names = st.multiselect("variables to save (their dependencies are saved too)", prog.names, key="save_names")
    a, b, c = st.columns(3)
    status = a.selectbox("status", ["experimental", "candidate", "validated", "deprecated"], key="save_status")
    units = b.text_input("units (optional)", key="save_units")
    desc = c.text_input("description (optional)", key="save_desc")
    if st.button("💾 Save selected", disabled=not names):
        need, todo = [], list(names)
        while todo:
            n = todo.pop()
            if n in need:
                continue
            need.append(n)
            todo += list(prog.statement(n).locals)
        order = [s.name for s in prog.statements if s.name in need]
        saved = []
        for n in order:
            st_ = prog.statement(n)
            used = {k: prog.params[k] for k in st_.params}
            raw = set(st_.raw)
            for d in st_.library:
                raw |= set(lib.get(d).get("raw_inputs", []))
            for d in st_.locals:
                raw |= set(prog.statement(d).raw)
            lib.upsert(n, st_.source, raw_inputs=sorted(raw), depends_on=sorted(st_.locals | st_.library),
                       param_specs={k: {"min": p.min, "max": p.max, "step": p.step, "default": p.default}
                                    for k, p in used.items()},
                       parameters={k: prog.values[k] for k in used}, description=desc if n in names else
                       f"dependency of {', '.join(names)}", status=status, units=units if n in names else "",
                       aoi_dependent=st_.aoi_dependent, halo_px=st_.halo, source_program=prog.source)
            saved.append(n)
        st.success(f"saved {saved} (v{[lib.get(n)['version'] for n in saved]}) to {lib.path}")


def render():
    ctx = C.ctx()
    E, store, lib = ctx["engine"], ctx["store"], ctx["library"]
    st.header("🧪 Python Feature Laboratory")
    left, right = st.columns([1.55, 1])
    with left:
        text = _editor(store)
        prog, err = _compile(E, text)
        if err:
            st.error(err)
        else:
            vals = _sliders(prog)
            prog = E.compile(text, vals)
            C.warn_box(prog.warnings)
            st.caption(f"{len(prog.names)} variables • inputs {sorted(prog.raw_inputs)} • saved features "
                       f"{sorted(prog.library_refs) or 'none'} • halo {prog.halo} px")
        a, b, c = st.columns(3)
        run = a.button("▶ Run Features", type="primary", disabled=prog is None)
        live = b.toggle("live update on slider change", False, key="live")
        c.button("↺ reset example", on_click=_reset)
        if st.session_state.get("freeze_msg"):
            st.success(st.session_state.pop("freeze_msg"))
    with right:
        with st.expander("Function reference", expanded=False):
            st.dataframe(pd.DataFrame(O.reference_table()), hide_index=True, use_container_width=True, height=330)
            st.caption("Angles: deg clockwise from north, axial [0, 180). Gradients per pixel. AOI-dependent "
                       "normalisations (zscore without center/scale, percentile_rank, ...) must be frozen before "
                       "full-grid export.")
        with st.expander(f"Trusted plugins ({len(E.plugins)})", expanded=False):
            st.caption("Loaded from local trusted directories (full Python, NOT sandboxed). Call them like "
                       "functions: one array per declared input, then keyword parameters.")
            for p in E.plugins.values():
                st.code(p.describe()["call"] + f"   # {p.description}", language="python")
                C.warn_box(p.warnings)
            if st.button("reload plugins"):
                E.plugins, E.plugin_errors = load_plugins()
                E._meta_key = None
                st.rerun()
        with st.expander("Inputs available", expanded=False):
            st.code("\n".join(E.inputs))
            st.caption(f"saved features: {', '.join(lib.features) or 'none'}; predictions / faults are NOT formula "
                       "inputs")
    if prog is None:
        return
    key = (text, tuple(sorted((st.session_state.get("param_values") or {}).items())), st.session_state["aoi"])
    if run or (live and st.session_state.get("lab_key") != key and st.session_state.get("lab_result") is not None):
        try:
            with st.spinner("evaluating on the AOI (+ halo) ..."):
                prev = st.session_state.get("lab_result")
                res = E.evaluate(prog, C.window())
            if prev is not None:
                st.session_state["lab_prev"] = {"arrays": prev.arrays, "params": st.session_state.get(
                    "lab_params_prev"), "window": prev.window}
            st.session_state.update(lab_result=res, lab_program_text=text, lab_key=key,
                                    lab_params_prev=dict(st.session_state.get("param_values") or {}))
            store.text(text, store.root / "registry" / "last_program.py")
        except (FL.FeatureError, O.OpError, ValueError, MemoryError) as e:
            st.error(f"evaluation failed: {e}")
            return
    res = C.program_result()
    if res is None:
        st.info("Press ▶ Run Features to compute the variables on the current AOI.")
        return
    st.success(f"{len(res.arrays)} features on {res.window.height} × {res.window.width} px (halo {res.halo} px) in "
               f"{res.seconds:.2f} s • recomputed {res.computed or 'none'} • cache hits {res.cache_hits}")
    rows = []
    for n, d in res.diagnostics.items():
        rows.append({"feature": n, "aoi_dependent": res.aoi_dependent.get(n), "valid_share": d.get("valid_share"),
                     "median": d.get("median"), "p02": d.get("p02"), "p98": d.get("p98"), "n_inf": d.get("n_inf"),
                     "warnings": len(d["warnings"]), "hash": res.hashes.get(n)})
    D = pd.DataFrame(rows)
    st.dataframe(D, hide_index=True, use_container_width=True)
    show = st.selectbox("show feature", list(res.arrays), index=len(res.arrays) - 1, key="lab_show")
    C.warn_box(res.diagnostics[show]["warnings"])
    f = C.faults()
    st_ = prog.statement(show) if show in prog.names else None
    srcs = sorted(st_.raw | st_.locals | st_.library)[:3] if st_ else []
    comp = {show: res.arrays[show], **{k: v for k, v in C.get_layers(srcs).items()}}
    st.plotly_chart(C.multi_map(comp, C.window(), f, "Viridis"), use_container_width=True)
    prev = st.session_state.get("lab_prev")
    if prev and show in prev["arrays"] and prev["arrays"][show].shape == res.arrays[show].shape:
        with st.expander("Difference from the previous run (parameters "
                         f"{prev.get('params')} -> {st.session_state.get('param_values')})"):
            diff = res.arrays[show] - prev["arrays"][show]
            st.plotly_chart(C.map_fig(diff, f"{show}: new - previous", C.window(), None, "RdBu_r"),
                            use_container_width=True)
            if f is not None:
                m_new = A.metrics(res.arrays[show], f)
                m_old = A.metrics(prev["arrays"][show], f)
                st.dataframe(pd.DataFrame([{"run": "previous", **{k: m_old.get(k) for k in ("auroc", "ap_lift",
                                                                                            "recall_at_1pct")}},
                                           {"run": "current", **{k: m_new.get(k) for k in ("auroc", "ap_lift",
                                                                                           "recall_at_1pct")}}]),
                             hide_index=True)
    if st.button("🎯 Evaluate against faults (quick, this AOI)"):
        if f is None:
            st.warning("no mapped-fault raster")
        else:
            L = {**res.arrays, **C.get_layers(sorted(prog.raw_inputs))}
            T = A.leaderboard(L, f, direction_mode="high")
            st.session_state["lab_quick"] = T
    if st.session_state.get("lab_quick") is not None:
        T = st.session_state["lab_quick"]
        st.dataframe(T[[c for c in ("feature", "status", "auroc", "auroc_high", "ap_lift", "recall_at_1pct",
                                     "point_biserial", "cliffs_delta", "n_valid", "n_fault_px") if c in T]],
                     hide_index=True, use_container_width=True)
        st.caption("Direction 'high' for all rows (auroc_high < 0.5 means low values rank faults). Full analysis: "
                   "page 3. Unlabelled pixels are not confirmed non-faults.")
    if res.normalization:
        with st.expander(f"Freeze AOI-dependent statistics ({len(res.normalization)} call sites)"):
            st.warning("Freeze ONLY on a development AOI: the frozen center / scale become constants of the formula "
                       "(required for full-grid export).")
            C.json_block(res.normalization)
            st.button("write frozen statistics into the formula", on_click=_freeze,
                      args=(st.session_state.get("lab_program_text", text), res.normalization))
    _save_block(E, prog, res, lib)
