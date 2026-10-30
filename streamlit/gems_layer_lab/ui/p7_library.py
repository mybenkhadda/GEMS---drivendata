"""Page 7 - Feature Library & Export: registry, versions, dependency graph, import / export, duplicate / rename /
delete, AOI GeoTIFF download, full-footprint native export, P6 / M9 integration snippet."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from lab.library import STATUSES, LibraryError
from lab.rasters import export_tiff

from . import common as C


def render():
    ctx = C.ctx()
    E, store, lib = ctx["engine"], ctx["store"], ctx["library"]
    st.header("📚 Feature Library & Export")
    st.caption(f"Registry: {lib.path}")
    feats = lib.features
    if feats:
        T = pd.DataFrame([{k: e.get(k) for k in ("name", "version", "status", "formula", "raw_inputs", "depends_on",
                                                 "aoi_dependent", "halo_px", "formula_hash", "updated")}
                          for e in feats.values()])
        st.dataframe(T, hide_index=True, use_container_width=True)
        bad = E.validate_library()
        if bad:
            st.error(f"features that no longer compile: {bad}")
        with st.expander("Dependency graph", expanded=True):
            try:
                st.graphviz_chart(lib.dot())
            except Exception as e:                       # graphviz rendering is optional
                st.code(lib.dot(), language="dot")
                st.caption(f"(graph rendering unavailable: {e})")
            if st.button("save graph snapshot"):
                store.text(lib.dot(), store.root / "registry" / "dependency_graph.dot")
                store.json({n: sorted(d) for n, d in lib.edges().items()}, store.root / "registry" /
                           "dependency_graph.json")
                st.success("saved registry/dependency_graph.dot|json")
    else:
        st.info("No saved feature yet: write formulas on page 2 and press 💾 Save selected.")
    up = st.file_uploader("import registry JSON (gll-registry.1)", type=["json"], key="lib_import")
    if up is not None and st.button("import"):
        try:
            st.success(lib.import_json(up.getvalue().decode("utf-8")))
        except (LibraryError, ValueError, KeyError) as e:
            st.error(e)
    if not feats:
        return
    st.download_button("⬇ export whole registry JSON", lib.export_json(), file_name="feature_registry.json")
    st.divider()
    name = st.selectbox("feature", list(feats), key="lib_name")
    e = lib.get(name)
    a, b = st.columns([1.2, 1])
    with a:
        st.code(lib.program_source(name), language="python")
        st.caption(f"v{e['version']} • {e['status']} • hash {e['formula_hash']} • halo {e.get('halo_px')} px • "
                   f"AOI-dependent: {e.get('aoi_dependent')} • {e.get('spatial_units')}")
        if e.get("description"):
            st.write(e["description"])
        with st.expander(f"history ({len(e.get('history', []))}), evaluations ({len(e.get('evaluations', []))}), "
                         f"exports ({len(e.get('exports', []))})"):
            C.json_block({"history": e.get("history"), "evaluations": e.get("evaluations"),
                          "exports": e.get("exports")})
        st.download_button("⬇ export this feature (+ dependencies)", lib.export_json([name]),
                           file_name=f"{name}_registry.json")
    with b:
        new = st.text_input("new name (duplicate / rename)", key="lib_new")
        c1, c2, c3 = st.columns(3)
        try:
            if c1.button("duplicate", disabled=not new):
                lib.duplicate(name, new)
                st.rerun()
            if c2.button("rename", disabled=not new):
                st.info(f"dependents rewritten: {lib.rename(name, new)}")
                st.rerun()
            if c3.button("delete"):
                lib.delete(name)
                st.rerun()
            stt = st.selectbox("status", STATUSES, index=STATUSES.index(e["status"]), key="lib_status")
            if stt != e["status"] and st.button("set status"):
                lib.set_status(name, stt)
                st.rerun()
        except LibraryError as err:
            st.error(err)
        st.caption(f"dependents: {lib.dependents(name) or 'none'}")
    st.divider()
    st.markdown("#### Export")
    sel = st.multiselect("features to export", list(feats), default=[name], key="lib_exp")
    if sel:
        w = C.window()
        if st.button("prepare AOI GeoTIFF downloads"):
            res = E.evaluate(None, w, library_targets=sel)
            st.session_state["lib_aoi_tifs"] = {n: export_tiff(res.arrays[n], w, ctx["feature_tif"], n) for n in sel}
        for n, data in (st.session_state.get("lib_aoi_tifs") or {}).items():
            st.download_button(f"⬇ {n} (AOI) .tif", data, file_name=f"{n}_r{int(w.row_off)}_c{int(w.col_off)}.tif",
                               mime="image/tiff", key=f"aoi_tif_{n}")
        aoi = [n for n in sel if E.library_meta().get(n, (0, False))[1]]
        if aoi:
            st.warning(f"AOI-dependent (unfrozen normalisation): {aoi} - cannot be exported full-footprint")
        fpm = st.checkbox("mask outside the submission footprint (NaN)", False, key="lib_fpm")
        st.caption(f"Full-footprint export: native {E.H} × {E.W} grid, tiled with halos, written to "
                   f"{store.root / 'exports'} (never overwrites; several minutes per feature on the full raster).")
        if st.button("🚀 export full-footprint native GeoTIFF(s)", disabled=bool(aoi)):
            from lab.export_native import export_features

            bar = st.progress(0.0)
            try:
                man = export_features(E, sel, store.root / "exports", 512, fpm,
                                      progress=lambda i, n: bar.progress(i / n, text=f"tile {i}/{n}"))
                st.success(f"exported: {man['layers']}")
                C.json_block(man["alignment_checks"])
            except (ValueError, FileExistsError, RuntimeError) as err:
                st.error(err)
    st.divider()
    st.markdown("#### Reuse in P6 / M9 (no Streamlit)")
    st.code(f'''import sys; sys.path.insert(0, "/content/GEMS/streamlit/gems_layer_lab")
from lab.integration import FeatureStore
fs = FeatureStore("{ctx['root']}")
arr = fs.compute("{name}")                 # native grid, float32, NaN = invalid (same engine as this app)
path = fs.exported_path("{name}")          # GeoTIFF of the current version, if exported''', language="python")
