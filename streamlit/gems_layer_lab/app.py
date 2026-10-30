"""GEMS Geological Feature Discovery Studio (upgrade of the Geological Layer Lab).

Seven pages: Map Explorer | Feature Laboratory | Fault Association | Local & Regional | Tournament |
W2 / M9 Complementarity | Library & Export. Science lives in ``lab/`` (importable without Streamlit); this file only
wires the sidebar, the shared engine and the page modules in ``ui/``.

    streamlit run app.py            (Windows: start_windows.bat; Colab: Launch_GEMS_Layer_Lab_Colab.ipynb)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from lab.engine import Engine  # noqa: E402
from lab.library import FeatureLibrary  # noqa: E402
from lab.rasters import aoi_window, parse_extra_entries  # noqa: E402
from lab.store import OutputStore  # noqa: E402

st.set_page_config(page_title="GEMS | Feature Discovery Studio", page_icon="🧭", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""
<style>
.stApp {background:#0c121a;}
[data-testid="stSidebar"] {background:#101b25;}
[data-testid="stMetric"] {border:1px solid #314453; border-radius:10px; padding:10px;}
.block-container {padding-top:1.2rem; max-width:1650px;}
textarea {font-family: ui-monospace, Consolas, monospace !important; font-size: 13px !important;}
</style>""", unsafe_allow_html=True)

ROOT = Path(os.getenv("GEMS_ROOT", "/content/drive/MyDrive/GEMS_2"))
DEMO_ROOT = HERE / "demo" / "data"
FEAT_REL = "data/raw/geodawn/gems-geodawn-numerical-features.tif"
PAGES = ["1 · Map Explorer", "2 · Feature Laboratory", "3 · Fault Association", "4 · Local & Regional",
         "5 · Feature Tournament", "6 · W2 / M9 Complementarity", "7 · Library & Export"]


def parse_extras(text: str) -> list:
    """Lines ``name=path`` (input layer) or ``pred:name=path`` (prediction: comparison only, never a formula input).
    NPZ arrays: ``name=/path/a.npz::key``."""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        role = "input"
        if s.lower().startswith("pred:"):
            role, s = "prediction", s[5:].strip()
        (name, path, key), = parse_extra_entries(s)
        out.append((name, path, key, role))
    return out


@st.cache_resource(show_spinner="Opening rasters, plugins and the feature library ...")
def get_engine(feature_tif, fault_tif, extras, registry, plugin_dir, footprint, domains, stamp):
    lib = FeatureLibrary(registry)
    dirs = None
    if plugin_dir:
        from lab.plugins import plugin_dirs

        dirs = plugin_dirs([Path(plugin_dir)])
    return Engine(feature_tif, fault_tif, list(extras), lib, dirs, footprint, domains)


LOCAL_FEATURES = HERE / "gems-geodawn-numerical-features.tif"      # shipped next to app.py
LOCAL_FAULTS = HERE / "existing_faults.tif"
CUSTOM, NONE = "📁 Custom path…", "— none —"


def tif_candidates(*dirs) -> list[str]:
    """Existing .tif / .tiff files in the given folders (non-recursive), de-duplicated, local app files first."""
    out = []
    for d in dirs:
        d = Path(d)
        if d.is_dir():
            out += [str(p) for p in sorted(d.glob("*.tif*")) if p.is_file()]
    return list(dict.fromkeys(out))


def file_selector(label: str, candidates: list[str], default: str | None, placeholder: str, key: str,
                  optional: bool = False, help: str = "") -> str | None:
    """Dropdown of discovered rasters + 'Custom path…' (text box with a placeholder) [+ 'none' when optional]."""
    opts = ([NONE] if optional else []) + candidates + [CUSTOM]
    idx = opts.index(default) if default in opts else (0 if optional or candidates else len(opts) - 1)
    choice = st.selectbox(label, opts, index=idx, key=f"{key}_sel", help=help,
                          format_func=lambda p: p if p in (CUSTOM, NONE) else f"{Path(p).name}  ·  {Path(p).parent}")
    if choice == NONE:
        return None
    if choice == CUSTOM:
        p = st.text_input(f"{label} path", value="", placeholder=placeholder, key=f"{key}_path").strip().strip('"')
        if not p:
            return None
        if not Path(p).is_file():
            st.error(f"not found: {p}")
            return None
        return p
    return choice


with st.sidebar:
    st.markdown("## 🧭 Feature Discovery Studio")
    page = st.radio("Page", PAGES, label_visibility="collapsed")
    st.markdown("### Data")
    root = Path(st.text_input("GEMS_2 root (outputs, domains)", value=str(ROOT) if ROOT.is_dir() else "",
                              placeholder="e.g. /content/drive/MyDrive/GEMS_2  or  G:/My Drive/GEMS_2",
                              help="Saved features, tables, figures and exports go to <root>/outputs/"
                                   "geological_layer_lab. Empty = the app folder.") or HERE)
    use_demo = st.toggle("Synthetic demo data", value=False, help="400 × 420 px synthetic rasters (not evidence)")
    if use_demo:
        root = DEMO_ROOT
        if not (root / FEAT_REL).exists():
            st.info("Demo rasters not generated yet.")
            if st.button("Generate demo rasters"):
                from demo.create_demo import generate_demo

                generate_demo(DEMO_ROOT)
                st.rerun()
            st.stop()
        feature_tif = str(root / FEAT_REL)
        fault_tif = str(root / "data/raw/faults/existing_faults.tif")
        footprint = None
    else:
        cands = tif_candidates(HERE, root, root / "data/raw/geodawn", root / "data/raw/faults", root / "data/raw")
        feat_default = str(LOCAL_FEATURES) if LOCAL_FEATURES.is_file() else str(root / FEAT_REL)
        fault_default = str(LOCAL_FAULTS) if LOCAL_FAULTS.is_file() else str(root / "data/raw/faults/existing_faults.tif")
        feature_tif = file_selector("Features GeoTIFF (19 bands)", [c for c in cands if "fault" not in Path(c).name
                                    and "submission" not in Path(c).name], feat_default,
                                    "e.g. C:/GEMS/gems-geodawn-numerical-features.tif", "feat",
                                    help="Multiband GeoDAWN numerical features (EPSG:32611, 100 m).")
        fault_tif = file_selector("Known faults GeoTIFF (target)", [c for c in cands if "fault" in Path(c).name],
                                  fault_default, "e.g. C:/GEMS/existing_faults.tif", "faults", optional=True,
                                  help="Binary mapped-fault raster on the same grid. Used only for evaluation.")
        footprint = file_selector("Submission template (optional footprint)",
                                  [c for c in cands if "submission" in Path(c).name],
                                  str(root / "data/raw/example_submission.tif"), "e.g. C:/GEMS/example_submission.tif",
                                  "tmpl", optional=True, help="Finite pixels define the valid submission footprint.")
    if not feature_tif:
        st.warning("Select a features GeoTIFF (dropdown or custom path).")
    dom = root / "outputs/geological_discovery_eda/rasters/spatial_domains/domains.npz"
    with st.expander("Extra aligned layers (inputs / W2-M9 predictions)"):
        extras_text = st.text_area("one per line", value=st.session_state.get("extras_text", ""), height=110,
                                   help="name=/path/layer.tif (formula input) | pred:w2=/path/w2_rank.tif (prediction: "
                                        "comparison only). Rasters must match the feature grid exactly.")
        st.session_state["extras_text"] = extras_text
        plugin_dir = st.text_input("Extra TRUSTED plugin directory (optional)", value="",
                                   help="Python files here are imported with full rights (not sandboxed).")

if not feature_tif or not Path(feature_tif).is_file():
    st.info("👈 Choose the features GeoTIFF in the sidebar. The app folder's `gems-geodawn-numerical-features.tif` and "
            "`existing_faults.tif` are selected automatically when present.")
    st.stop()

try:
    extras = tuple(parse_extras(extras_text))
    store = OutputStore(root)
    stamp = tuple((p, Path(p).stat().st_mtime_ns) for p in [feature_tif] + [e[1] for e in extras]
                  if Path(p).exists())
    engine = get_engine(feature_tif, fault_tif if fault_tif and Path(fault_tif).is_file() else None, extras,
                        str(store.registry_path), plugin_dir,
                        footprint if footprint and Path(footprint).is_file() else None,
                        str(dom) if dom.is_file() else None, stamp)
except Exception as e:                                   # shown, never swallowed
    st.error(f"Source validation failed: {type(e).__name__}: {e}")
    st.stop()

H, W, px_m = engine.H, engine.W, engine.pixel_m
with st.sidebar:
    st.markdown("### Area of interest")
    st.caption(f"Grid {H:,} × {W:,} px • {engine.details['crs']} • {px_m:g} m/px")
    if "aoi" not in st.session_state:
        w0 = aoi_window(H, W, H // 2, W // 2, 12, px_m)
        st.session_state["aoi"] = (int(w0.row_off), int(w0.col_off), int(w0.height), int(w0.width))
    r0, c0, h0, w0_ = st.session_state["aoi"]
    with st.form("aoi_form"):
        cr = st.number_input("centre row", 0, H - 1, int(r0 + h0 // 2), step=20)
        cc = st.number_input("centre column", 0, W - 1, int(c0 + w0_ // 2), step=20)
        side = st.slider("side (km)", 1, int(min(100, min(H, W) * px_m / 1000)), max(1, int(round(h0 * px_m / 1000))))
        if st.form_submit_button("Apply AOI"):
            ww = aoi_window(H, W, cr, cc, side, px_m)
            st.session_state["aoi"] = (int(ww.row_off), int(ww.col_off), int(ww.height), int(ww.width))
            st.rerun()
    r0, c0, h0, w0_ = st.session_state["aoi"]
    st.caption(f"AOI rows {r0}–{r0 + h0}, cols {c0}–{c0 + w0_} ({h0 * w0_:,} px). Draw a box on the overview "
               "(page 1) to set it interactively.")
    if engine.plugin_errors:
        st.warning(f"plugin errors: {engine.plugin_errors}")
    st.caption(f"Outputs: {store.root}")

st.session_state["_ctx"] = {"engine": engine, "store": store, "root": str(root), "library": engine.library,
                            "feature_tif": feature_tif}

from ui import (p1_maps, p2_lab, p3_association, p4_regional, p5_tournament, p6_complementarity,  # noqa: E402
                p7_library)

{PAGES[0]: p1_maps, PAGES[1]: p2_lab, PAGES[2]: p3_association, PAGES[3]: p4_regional, PAGES[4]: p5_tournament,
 PAGES[5]: p6_complementarity, PAGES[6]: p7_library}[page].render()

st.divider()
st.caption("Research tool • 1 px = 100 m • unmapped ≠ non-fault • associations are not DTI • select formulas, "
           "directions and parameters on development data only.")
