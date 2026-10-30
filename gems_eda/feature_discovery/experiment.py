"""FeatureDiscoveryExperiment: orchestration of the geological feature discovery study (sections 01-13).

Every section writes its tables (CSV + Parquet), figures (PNG + SVG + plotted data CSV) and a checkpoint
``checkpoints/<section>.json`` (fingerprint = code rev + config + formula hashes). A section whose checkpoint matches
and whose outputs exist is reloaded instead of recomputed (``force=True`` recomputes).

Modes: FULL = whole native grid (3730 x 3292, final measurements); FAST = one configurable window (exploration).
Labels (mapped faults) enter ONLY evaluation; they are never formula inputs (Layer Lab leakage guard).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rasterio.windows import Window
from scipy import ndimage

from . import FD_CODE_REV
from . import catalog as CAT
from . import metrics as M

REPO = Path(__file__).resolve().parents[2]
LAB_DIR = REPO / "streamlit" / "gems_layer_lab"
OUT_REL = "outputs/geological_feature_discovery"
SUBDIRS = ("configs", "formulas", "layers", "datasets", "tables", "figures", "figures/data", "comparisons",
           "checkpoints", "manifests", "reports")
EXPECTED = {"height": 3730, "width": 3292, "epsg": 32611, "res_m": 100.0, "bands": 19}
DEFAULT_CONFIG = {
    "gems_root": "/content/drive/MyDrive/GEMS_2",
    "features": "data/raw/geodawn/gems-geodawn-numerical-features.tif",
    "faults": "data/raw/faults/existing_faults.tif",
    "template": "data/raw/example_submission.tif",
    "folds": "outputs/benchmark/folds/geo_k3_b20km_buf13km_s2026/fold_map.tif",
    "domains": "outputs/geological_discovery_eda/rasters/spatial_domains/domains.npz",
    "w2_heldout": None,                 # aligned single-band raster of genuinely held-out W2 scores (+ .provenance.json)
    "mode": "FULL",                     # FULL | FAST
    "fast_window": {"row": 1600, "col": 1400, "height": 600, "width": 600},
    "budgets": [0.005, 0.01, 0.02, 0.05],
    "tiles_km": [1, 2, 5, 10, 20],
    "min_pos": 20, "min_neg": 200,
    "complementarity_top_k": 10,
    "unique_budget": 0.01,
    "n_boot": 200, "block_px": 20,
    "seed": 2026,
    "canonical_dti": False,             # canonical scorer on thresholded maps per fold (diagnostic only)
    "thresholds": dict(M.DEFAULT_THRESHOLDS),
    "covariate": "det_elev_slope",      # label-free covariate for matched controls
    "custom_formulas": "",
}


def _lab():
    """Import the Layer Lab package (streamlit/gems_layer_lab/lab) from the repository checkout."""
    if str(LAB_DIR) not in sys.path:
        sys.path.insert(0, str(LAB_DIR))
    import lab.rasters as R  # noqa: F401
    from lab import association, complementarity, featurelang, ops
    from lab.engine import Engine
    from lab.library import FeatureLibrary

    return {"Engine": Engine, "FeatureLibrary": FeatureLibrary, "FL": featurelang, "ops": ops, "A": association,
            "CP": complementarity, "R": R}


def _js(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class Store:
    """Atomic writes under the output namespace (also the duck-typed store of ``lab.figures``)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        for s in SUBDIRS:
            (self.root / s).mkdir(parents=True, exist_ok=True)

    def path(self, rel):
        return self.root / rel

    @staticmethod
    def _atomic(path: Path, writer):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        writer(tmp)
        os.replace(tmp, path)
        return path

    def table(self, df: pd.DataFrame, path, parquet=True):
        path = Path(path)
        out = [self._atomic(path.with_suffix(".csv"), lambda t: df.to_csv(t, index=False))]
        if parquet and len(df):
            try:
                d = df.copy()
                for c in d.columns:
                    if d[c].dtype == object:
                        d[c] = d[c].map(lambda v: v if v is None or isinstance(v, str) else json.dumps(v, default=_js))
                out.append(self._atomic(path.with_suffix(".parquet"), lambda t: d.to_parquet(t, index=False)))
            except Exception:
                pass
        return out

    def json(self, obj, rel):
        return self._atomic(self.path(rel), lambda t: Path(t).write_text(json.dumps(obj, indent=2, default=_js),
                                                                          encoding="utf-8"))

    def read_json(self, rel):
        p = self.path(rel)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def text(self, s, rel):
        return self._atomic(self.path(rel), lambda t: Path(t).write_text(s, encoding="utf-8"))


class FeatureDiscoveryExperiment:
    def __init__(self, config: dict | None = None):
        self.c = {**DEFAULT_CONFIG, **(config or {})}
        self.c["thresholds"] = {**M.DEFAULT_THRESHOLDS, **(self.c.get("thresholds") or {})}
        if self.c["mode"] not in ("FULL", "FAST"):
            raise ValueError("mode must be FULL or FAST")
        self.root = Path(self.c["gems_root"])
        self.store = Store(self.root / OUT_REL)
        self.L = _lab()
        self.custom = self.c.get("custom_formulas") or ""
        self.E = None
        self._cache = {}
        self.tables: dict[str, pd.DataFrame] = {}

    # ================================================================== paths / fingerprints
    def p(self, key):
        v = self.c[key]
        return None if v is None else (Path(v) if Path(v).is_absolute() else self.root / v)

    @property
    def program_text(self) -> str:
        return CAT.CATALOG + ("\n# ---- custom formulas ----\n" + self.custom if self.custom.strip() else "")

    def fingerprint(self, section: str, extra=None) -> str:
        cfg = {k: v for k, v in self.c.items() if k not in ("custom_formulas",)}
        payload = {"rev": FD_CODE_REV, "section": section, "config": cfg, "program": self.program_text,
                   "extra": extra}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def _ckpt(self, section, outputs, info=None):
        self.store.json({"section": section, "fingerprint": self.fingerprint(section), "outputs": outputs,
                         "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "code_rev": FD_CODE_REV, **(info or {})},
                        f"checkpoints/{section}.json")

    def _reuse(self, section, outputs, force) -> bool:
        ck = self.store.read_json(f"checkpoints/{section}.json")
        ok = (not force and ck and ck.get("fingerprint") == self.fingerprint(section) and
              all(self.store.path(o).exists() for o in outputs))
        if ok:
            print(f"[{section}] reused checkpoint ({ck['time']})")
        return bool(ok)

    def _table(self, name, df, sub="tables"):
        self.tables[name] = df
        self.store.table(df, self.store.path(f"{sub}/{name}"))
        return df

    def load_table(self, name, sub="tables"):
        p = self.store.path(f"{sub}/{name}.parquet")
        if p.exists():
            return pd.read_parquet(p)
        q = self.store.path(f"{sub}/{name}.csv")
        return pd.read_csv(q) if q.exists() else pd.DataFrame()

    # ================================================================== 01 load
    def load(self):
        """Open rasters (metadata + footprint + labels), validate the grid, build the Layer Lab engine."""
        L = self.L
        feats, faults, tmpl = self.p("features"), self.p("faults"), self.p("template")
        for k, f in (("features", feats), ("faults", faults)):
            if not f or not f.is_file():
                raise FileNotFoundError(f"{k} raster not found: {f}")
        extras = []
        w2 = self.p("w2_heldout")
        if w2 is not None:
            if not w2.is_file():
                raise FileNotFoundError(f"w2_heldout raster not found: {w2}")
            extras.append(("w2", str(w2), None, "prediction"))
        lib = L["FeatureLibrary"](self.store.path("formulas/feature_registry.json"))
        dom = self.p("domains")
        self.E = L["Engine"](str(feats), str(faults), extras, lib, None, str(tmpl) if tmpl and tmpl.is_file() else
                             None, str(dom) if dom and dom.is_file() else None)
        E = self.E
        L["R"].MAX_AOI_PIXELS = max(L["R"].MAX_AOI_PIXELS, E.H * E.W)       # whole-grid reads are intended here
        rep = E.grid_report()
        rep["checks"] = {"shape": (E.H, E.W) == (EXPECTED["height"], EXPECTED["width"]),
                         "epsg_32611": "32611" in str(rep["crs"]), "res_100m": abs(E.pixel_m - EXPECTED["res_m"]) < 1e-6,
                         "bands_19": int(rep["bands"]) == EXPECTED["bands"],
                         "faults_aligned": rep.get("faults_aligned"), "template_aligned": rep.get("footprint_aligned")}
        if rep.get("faults_aligned") is False:
            raise ValueError("fault raster is not aligned with the features raster (never resampled silently)")
        if self.c["mode"] == "FULL":
            self.window = Window(0, 0, E.W, E.H)
        else:
            w = self.c["fast_window"]
            self.window = Window(int(w["col"]), int(w["row"]), int(w["width"]), int(w["height"]))
        self.f_ext, self.roi = E.faults(self.window, halo=60)
        self.f = self.f_ext[self.roi]
        fp = E.footprint(self.window)
        self.valid = fp if fp is not None else np.ones(self.f.shape, bool)
        self.dist_px = ndimage.distance_transform_edt(~self.f_ext)[self.roi].astype(np.float32)
        self.dist_km = self.dist_px * E.pixel_m / 1000.0
        self.near300 = self.dist_px <= 3
        self.comp_lab, self.n_comp = M.fault_components(self.f & self.valid)
        rep.update(mode=self.c["mode"], window=[int(self.window.row_off), int(self.window.col_off),
                                                int(self.window.height), int(self.window.width)],
                   fault_px=int(self.f.sum()), valid_px=int(self.valid.sum()), components=int(self.n_comp),
                   inputs=E.inputs, predictions=E.predictions, plugin_errors=E.plugin_errors)
        self.store.json(rep, "manifests/data_report.json")
        self.store.json(self.c, "configs/config.json")
        units = self._band_units()
        self._table("band_metadata", pd.DataFrame([{"band": b, "units": units.get(b, "not tagged"),
                                                    "family": CAT.family(b)} for b in E.inputs]), "datasets")
        print(f"[01] grid {E.H}x{E.W} {rep['crs']} {E.pixel_m:g} m | checks {rep['checks']} | window "
              f"{rep['window']} | fault px {rep['fault_px']:,} | components {self.n_comp:,}")
        return rep

    def _band_units(self) -> dict:
        p = self.root / "outputs/geological_discovery_eda/audit/band_metadata.csv"
        if p.exists():
            try:
                T = pd.read_csv(p)
                return dict(zip(T.band, T.units))
            except (OSError, ValueError, AttributeError):
                pass
        return {}

    # ================================================================== 03 / 04 formulas + layers
    def register_formulas(self, text: str, replace: bool = False):
        """Add custom formulas (restricted language; may reference catalog features). Validated immediately."""
        new = text if replace else (self.custom + "\n" + text)
        self.L["FL"].compile_program(CAT.CATALOG + "\n" + new, self.E.env()) if self.E else None
        self.custom = new
        stamp = time.strftime("%Y%m%d_%H%M%S")
        h = hashlib.sha256(new.encode()).hexdigest()[:12]
        self.store.json({"custom_formulas": new, "hash": h, "time": stamp}, f"formulas/custom_{stamp}_{h}.json")
        self.store.text(new, "formulas/custom_latest.py")
        print(f"[04] custom formulas registered (hash {h}); names: {self.names(custom_only=True)}")

    def program(self, values=None):
        return self.E.compile(self.program_text, values)

    def names(self, custom_only=False) -> list[str]:
        prog = self.program()
        if not custom_only:
            return prog.names
        base = set(self.E.compile(CAT.CATALOG).names)
        return [n for n in prog.names if n not in base]

    def _hashes(self, prog) -> dict:
        FL = self.L["FL"]
        h = dict(self.E.stamps)
        h.update({k: s.source_hash for k, s in self.E.plugins.items()})
        for st in prog.statements:
            h[st.name] = FL.expr_hash(st, prog, h)
        return h

    def _layer_path(self, name):
        tag = "full" if self.c["mode"] == "FULL" else "fast_{row}_{col}_{height}_{width}".format(**self.c["fast_window"])
        return self.store.path(f"layers/{tag}/{name}.npy")

    def compute_features(self, names=None, batch: int = 10, force: bool = False):
        """Evaluate catalog + custom formulas on the analysis window and persist each feature as NPY + sidecar JSON
        (formula, hash, parameters, source bands, CRS, transform, window, units, nodata, AOI dependence). Features
        whose hash is unchanged are reused."""
        prog = self.program()
        hashes = self._hashes(prog)
        todo = [n for n in (names or prog.names) if force or not self._layer_ok(n, hashes[n])]
        print(f"[03] {len(prog.names)} formula features; {len(todo)} to compute, {len(prog.names) - len(todo)} cached")
        norm = {}
        for i in range(0, len(todo), batch):
            part = todo[i:i + batch]
            res = self.E.evaluate(prog, self.window, targets=part)
            norm.update(res.normalization)
            for n in part:
                self._save_layer(n, res.arrays[n], prog, hashes[n], res.diagnostics.get(n, {}), res.aoi_dependent.get(n))
            self.E.clear_cache()
            print(f"      {min(i + batch, len(todo))}/{len(todo)} ({res.seconds:.0f} s)")
        if norm:                                            # frozen formula for tiled export / P6 / M9
            frozen, done = self.L["FL"].freeze_normalization(self.program_text, norm)
            self.store.text(frozen, "formulas/frozen_catalog.py")
            self.store.json({"frozen_sites": done, "statistics": norm, "scope": f"{self.c['mode']} window (label-free)"},
                            "formulas/frozen_statistics.json")
        self.store.text(self.program_text, "formulas/catalog_used.py")
        self.store.json({n: {"hash": hashes[n], "path": str(self._layer_path(n))} for n in prog.names},
                        "formulas/layer_index.json")
        return todo

    def _layer_ok(self, name, h) -> bool:
        p = self._layer_path(name)
        side = p.with_suffix(".json")
        if not (p.exists() and side.exists()):
            return False
        return json.loads(side.read_text(encoding="utf-8")).get("hash") == h

    def _save_layer(self, name, arr, prog, h, diag, aoi):
        p = self._layer_path(name)
        p.parent.mkdir(parents=True, exist_ok=True)

        def write(t):                                     # file handle: np.save would append ".npy" to a str path
            with open(t, "wb") as fh:
                np.save(fh, np.asarray(arr, np.float32))
        self.store._atomic(p, write)
        st = prog.statement(name)
        meta = {"name": name, "formula": st.source, "hash": h, "parameters": {k: prog.values[k] for k in st.params},
                "source_bands": sorted(st.raw), "depends_on": sorted(st.locals | st.library), "halo_px": st.halo,
                "aoi_dependent": bool(aoi), "crs": self.E.details["crs"], "transform": self.E.details["transform"],
                "window": [int(self.window.row_off), int(self.window.col_off), int(self.window.height),
                           int(self.window.width)], "units": CAT.META.get(name, {}).get("units", "derived"),
                "nodata": "NaN", "diagnostics": diag, "family": CAT.family(name),
                "hypothesis": CAT.META.get(name, {}).get("hypothesis", ""), "code_rev": FD_CODE_REV,
                "reload": f"np.load(r'{p}', mmap_mode='r')"}
        self.store._atomic(p.with_suffix(".json"), lambda t: Path(t).write_text(json.dumps(meta, indent=2,
                                                                                           default=_js)))

    def get(self, name) -> np.ndarray:
        """Feature (formula NPY, memory-mapped) or raw band of the analysis window."""
        if name in self._cache:
            return self._cache[name]
        p = self._layer_path(name)
        if p.exists():
            return np.load(p, mmap_mode="r")
        if name in self.E.sources:
            a = self.E.raw([name], self.window)[name]
            if len(self._cache) < 24:
                self._cache[name] = a
            return a
        raise KeyError(f"{name}: not computed (run compute_features) and not a raw band")

    # ================================================================== plots (PNG + SVG + data)
    def _fig_save(self, fig, name, data: pd.DataFrame | None):
        import matplotlib.pyplot as plt

        for fmt in ("png", "svg"):
            self.store._atomic(self.store.path(f"figures/{name}.{fmt}"),
                               lambda t, f=fmt: fig.savefig(t, dpi=150, bbox_inches="tight", format=f))
        plt.close(fig)
        if data is not None:
            self.store.table(data, self.store.path(f"figures/data/{name}"), parquet=False)
        return self.store.path(f"figures/{name}.png")

    def _extent(self):
        a = self.E.details["transform"]
        w = self.window
        x0, y0 = a[2] + w.col_off * a[0], a[5] + w.row_off * a[4]
        return [x0 / 1e3, (x0 + w.width * a[0]) / 1e3, (y0 + w.height * a[4]) / 1e3, y0 / 1e3]

    def plot_feature(self, name, overlay=True, step=None, cmap="viridis", pct=(2, 98), show=True):
        """Feature map with mapped-fault overlay (display subsampled for the full grid; data unchanged)."""
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        a = np.asarray(self.get(name), np.float32)
        s = step or max(1, int(math.ceil(max(a.shape) / 1200)))
        v = a[::s, ::s]
        g = v[np.isfinite(v)]
        lo, hi = np.percentile(g, pct) if g.size else (0, 1)
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(np.where(self.valid[::s, ::s], v, np.nan), extent=self._extent(), cmap=cmap, vmin=lo, vmax=hi,
                       interpolation="nearest")
        if overlay:
            fr, fc = np.nonzero(self.f[::s, ::s])
            ex = self._extent()
            ax.plot(ex[0] + (fc + .5) * (ex[1] - ex[0]) / v.shape[1], ex[3] - (fr + .5) * (ex[3] - ex[2]) / v.shape[0],
                    ",", color="crimson")
        fig.colorbar(im, ax=ax, fraction=0.035, label=name)
        ax.set(title=f"{name} (mapped faults red; display step {s})", xlabel="UTM E (km)", ylabel="UTM N (km)")
        out = self._fig_save(fig, f"map_{name}", pd.DataFrame([{"feature": name, "vmin": lo, "vmax": hi,
                                                                "display_step": s, "valid_px": int(g.size)}]))
        return out

    # ================================================================== 02 / 05 global association
    def evaluate_against_faults(self, name) -> dict:
        r = M.association_both(self.get(name), self.f, self.valid, tuple(self.c["budgets"]), self.near300)
        return {"feature": name, "family": CAT.family(name), **r}

    def global_association(self, names=None, section="05_global", force=False) -> pd.DataFrame:
        names = names or (list(self.E.inputs) + self.names() + ["neg_rtp"])
        names = list(dict.fromkeys(names))
        out = "tables/global_fault_association.csv" if section == "05_global" else f"tables/{section}.csv"
        if self._reuse(section, [out], force):
            return self.load_table(Path(out).stem)
        rows = []
        for n in names:
            rows.append(self.evaluate_against_faults(n))
        T = pd.DataFrame(rows)
        T["best_auroc_EXPLORATORY"] = T[["auroc_high", "auroc_low"]].max(axis=1)
        T = T.sort_values("best_auroc_EXPLORATORY", ascending=False, ignore_index=True)
        self._table(Path(out).stem, T)
        if section == "05_global":
            P = T[["feature", "family", "auroc_high", "auroc_low", "ap_lift_high", "ap_lift_low",
                   "recall_1pct_high", "recall_1pct_low", "better_direction_EXPLORATORY"]].copy()
            P["note"] = np.where(P.feature == "neg_rtp", "direction twin of rtp: no new information", "")
            self._table("feature_polarity_comparison", P)
            self._plot_auroc(T)
            self._plot_recall_curves(T)
        self._ckpt(section, [out])
        return T

    def compare_features(self, names, show=True) -> pd.DataFrame:
        T = pd.DataFrame([self.evaluate_against_faults(n) for n in names])
        self._plot_auroc(T, name="compare_" + "_".join(n[:8] for n in names)[:80])
        return T

    def _plot_auroc(self, T, name="auroc_ap_polarity"):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        d = T[T.status == "OK"].head(40)
        fig, ax = plt.subplots(1, 2, figsize=(14, 0.28 * len(d) + 2))
        y = np.arange(len(d))
        ax[0].barh(y - 0.2, d.auroc_high, 0.4, label="high values rank first")
        ax[0].barh(y + 0.2, d.auroc_low, 0.4, label="low values rank first")
        ax[0].axvline(0.5, color="k", lw=0.6)
        ax[0].set_yticks(y)
        ax[0].set_yticklabels(d.feature, fontsize=7)
        ax[0].invert_yaxis()
        ax[0].set(xlabel="AUROC (mapped vs unlabelled)", title="Polarity comparison (exploratory)")
        ax[0].legend(fontsize=7)
        ax[1].barh(y - 0.2, d.ap_lift_high, 0.4)
        ax[1].barh(y + 0.2, d.ap_lift_low, 0.4)
        ax[1].axvline(1, color="k", lw=0.6)
        ax[1].set_yticks(y)
        ax[1].set_yticklabels([])
        ax[1].invert_yaxis()
        ax[1].set(xlabel="AP lift vs prevalence")
        self._fig_save(fig, name, d)

    def _plot_recall_curves(self, T, top=12):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        d = T[T.status == "OK"].head(top)
        bs = self.c["budgets"]
        fig, ax = plt.subplots(figsize=(8, 5))
        rows = []
        for r in d.itertuples():
            dirn = r.better_direction_EXPLORATORY
            ys = [T.loc[T.feature == r.feature, f"recall_{b * 100:g}pct_{dirn}"].iloc[0] for b in bs]
            ax.plot([b * 100 for b in bs], ys, marker="o", label=f"{r.feature} ({dirn})")
            rows += [{"feature": r.feature, "direction": dirn, "budget": b, "recall_exact": yv} for b, yv in zip(bs, ys)]
        ax.plot([b * 100 for b in bs], bs, "k--", lw=0.8, label="random")
        ax.set(xlabel="area budget (% of valid pixels)", ylabel="exact mapped-pixel recall",
               title="Recall vs area budget (exploratory direction)")
        ax.legend(fontsize=6)
        self._fig_save(fig, "recall_vs_budget", pd.DataFrame(rows))

    # ================================================================== 06 distance / profiles
    def fault_distance_responses(self, names, force=False) -> pd.DataFrame:
        out = "tables/fault_distance_responses.csv"
        if self._reuse("06_distance", [out], force):
            return self.load_table("fault_distance_responses")
        A = self.L["A"]
        rows, cls, prof = [], [], []
        cov = self.get(self.c["covariate"]) if self.c["covariate"] in self.E.sources else None
        for n in names:
            x = self.get(n)
            D = M.distance_response(x, self.dist_km, self.valid, self.f, seed=self.c["seed"])
            rows.append(D.assign(feature=n))
            c = M.response_class(D)
            if cov is not None:
                c.update(M.matched_controls(x, self.f, cov, self.valid, dist_px=self.dist_px))
            cls.append({"feature": n, **c})
            N = A.fault_normal_profiles(np.asarray(x, np.float32), self.f_ext, self.roi, half_px=30,
                                        max_stations=3000, seed=self.c["seed"])
            prof.append(N.assign(feature=n, offset_km=N.offset_px * self.E.pixel_m / 1000))
        T = self._table("fault_distance_responses", pd.concat(rows, ignore_index=True))
        self._table("fault_response_classes", pd.DataFrame(cls))
        P = self._table("fault_normal_profiles", pd.concat(prof, ignore_index=True), "datasets")
        self._plot_distance(T)
        self._plot_profiles(P)
        self._ckpt("06_distance", [out])
        return T

    def _plot_distance(self, T):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        for n, g in T.groupby("feature", sort=False):
            ax.plot(range(len(g)), g.normalized_response, marker="o", label=n)
        ax.set_xticks(range(len(M.DIST_LABELS)))
        ax.set_xticklabels(M.DIST_LABELS, rotation=30, fontsize=7)
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(3.5, color="crimson", lw=0.8, ls="--")
        ax.set(ylabel="(median - median beyond 5 km) / IQR beyond 5 km",
               title="Fault-distance response (red: 300 m official radius) - retrospective diagnostic")
        ax.legend(fontsize=6, ncol=2)
        self._fig_save(fig, "fault_distance_response", T)

    def _plot_profiles(self, P):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        for n, g in P.groupby("feature", sort=False):
            med = g["median"]
            s = (med - med.median()) / (g.p75 - g.p25).median() if (g.p75 - g.p25).median() else med * np.nan
            ax.plot(g.offset_km, s, label=n)
        ax.axvspan(-0.3, 0.3, color="0.85", zorder=0)
        ax.set(xlabel="offset along the mapped-trace normal (km)", ylabel="robust-normalised median",
               title="Fault-normal cross-sections (mapped geometry: retrospective only)")
        ax.legend(fontsize=6, ncol=2)
        self._fig_save(fig, "fault_normal_profiles", P)

    # ================================================================== 07 regional
    def regional_association(self, names, global_table=None, force=False) -> pd.DataFrame:
        out = "tables/regional_fault_association.csv"
        if self._reuse("07_regional", [out], force):
            return self.load_table("regional_fault_association")
        G = (global_table if global_table is not None else self.load_table("global_fault_association")).set_index(
            "feature")
        H, W = self.f.shape
        parts = {"region_3x3": M.block_index((H, W), -(-H // 3), -(-W // 3))}
        for km in self.c["tiles_km"]:
            t = max(1, int(round(km * 1000 / self.E.pixel_m)))
            parts[f"tile_{km:g}km"] = M.block_index((H, W), t, t)
        dom = self.E.domains(self.window)
        rows, maps = [], {}
        for n in names:
            x = self.get(n)
            gdir = G.better_direction_EXPLORATORY.get(n, "high") if n in G.index else "high"
            for pname, (bid, nbr, nbc) in parts.items():
                T = M.regional(x, self.f, self.valid, bid, self.c["unique_budget"], self.c["min_pos"],
                               self.c["min_neg"], gdir)
                rows.append(T.assign(feature=n, partition=pname, global_direction=gdir))
                if pname in ("region_3x3", "tile_10km", "tile_20km"):
                    for col in ("auroc_global_direction", "ap_lift_global_direction", "recall_at_budget"):
                        maps[f"{n}|{pname}|{col}"] = M.block_map(T, col, nbr, nbc)
                    if "polarity" in T:
                        maps[f"{n}|{pname}|polarity"] = M.block_map(
                            T.assign(pol=np.where(T.polarity == "high", 1.0, -1.0)), "pol", nbr, nbc)
            if dom is not None:
                T = M.regional(x, self.f, self.valid, np.where(dom >= 0, dom, -1).astype(np.int64),
                               self.c["unique_budget"], self.c["min_pos"], self.c["min_neg"], gdir)
                rows.append(T.assign(feature=n, partition="label_free_domain", global_direction=gdir))
        R = self._table("regional_fault_association", pd.concat(rows, ignore_index=True))
        np.savez_compressed(self.store.path("datasets/regional_maps.npz"), **{k.replace("|", "__"): v for k, v in
                                                                              maps.items()})
        S = self._stability(R)
        self._plot_regional(R)
        self._ckpt("07_regional", [out, "tables/feature_scale_comparison.csv"], {"stability_rows": len(S)})
        return R

    def _stability(self, R) -> pd.DataFrame:
        ok = R[R.status == "OK"]
        rows = []
        for (n, part), g in ok.groupby(["feature", "partition"]):
            same = (g.polarity == g.global_direction).mean() if len(g) else np.nan
            rows.append({"feature": n, "partition": part, "supported_cells": len(g),
                         "insufficient_cells": int(((R.feature == n) & (R.partition == part) &
                                                    (R.status != "OK")).sum()),
                         "same_polarity_share": float(same), "median_local_auroc": float(
                             g.auroc_global_direction.median()), "iqr_local_auroc": float(
                             g.auroc_global_direction.quantile(.75) - g.auroc_global_direction.quantile(.25)),
                         "median_recall_at_budget": float(g.recall_at_budget.median())})
        S = self._table("feature_scale_comparison", pd.DataFrame(rows))
        reg = ok[ok.partition == "region_3x3"]
        if len(reg):
            piv = reg.pivot_table(index="block", columns="feature", values="auroc_global_direction")
            rk = piv.rank(axis=1, ascending=False)
            self._table("feature_rank_by_region", rk.reset_index())
            if "neg_rtp" in piv:
                self._table("region_vs_neg_rtp", (piv.sub(piv["neg_rtp"], axis=0)).reset_index())
        return S

    def _plot_regional(self, R):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        reg = R[(R.partition == "region_3x3")]
        if not len(reg) or "auroc_global_direction" not in reg:
            print("[07] no 3x3 region with sufficient support: heatmap skipped")
            return
        piv = reg.pivot_table(index="feature", columns="block", values="auroc_global_direction")
        fig, ax = plt.subplots(figsize=(8, 0.3 * len(piv) + 2))
        im = ax.imshow(piv.to_numpy(), cmap="RdBu_r", vmin=0.3, vmax=0.7, aspect="auto")
        for (i, j), v in np.ndenumerate(piv.to_numpy()):
            ax.text(j, i, "INSUFF." if not np.isfinite(v) else f"{v:.2f}", ha="center", va="center", fontsize=6)
        ax.set_yticks(range(len(piv)))
        ax.set_yticklabels(piv.index, fontsize=7)
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels([f"R{c}" for c in piv.columns])
        fig.colorbar(im, ax=ax, label="local AUROC (global direction)")
        ax.set_title("Feature x 3x3 region association (INSUFF. = insufficient support)")
        self._fig_save(fig, "region_feature_heatmap", piv.reset_index())

    # ================================================================== 08 / 09 recovery + complementarity
    def _branches(self):
        if "branches" not in self._cache:
            from gems_eda.morphology import build_fault_network

            class _G:                                       # minimal grid for build_fault_network
                pass
            g = _G()
            g.height, g.width = self.f.shape
            g.pixel_km_x = g.pixel_km_y = self.E.pixel_m / 1000
            g.rc_to_km = lambda r, c: (np.asarray(c, float) * g.pixel_km_x, -np.asarray(r, float) * g.pixel_km_y)
            a = self.E.details["transform"]
            g.rc_to_xy = lambda r, c: (a[2] + (np.asarray(c, float) + .5) * a[0], a[5] + (np.asarray(r, float) + .5)
                                       * a[4])
            try:
                net = build_fault_network(self.f & self.valid, g, (3, 3))
                self._cache["branches"] = [p for p in net.paths if len(p) >= 3]
            except Exception as e:                          # branch metrics optional; reported
                print(f"[08] branch tracing skipped: {type(e).__name__}: {e}")
                self._cache["branches"] = None
        return self._cache["branches"]

    def fault_recovery(self, names, directions=("high", "low"), force=False) -> pd.DataFrame:
        out = "tables/fault_recovery_by_budget.csv"
        if self._reuse("08_recovery", [out], force):
            return self.load_table("fault_recovery_by_budget")
        br = self._branches()
        rows = []
        for n in names:
            x = np.asarray(self.get(n), np.float32)
            for d in directions:
                s = M.oriented(x, d)
                for b in self.c["budgets"]:
                    r = M.recovery(s, self.f, self.valid, b, self.comp_lab, self.n_comp, br, 3, self.dist_km)
                    r.pop("_comp_recovered", None)
                    rows.append({"feature": n, "direction": d, **r})
        T = self._table("fault_recovery_by_budget", pd.DataFrame(rows))
        self._ckpt("08_recovery", [out])
        return T

    def unique_recovery(self, names, references=None, force=False) -> pd.DataFrame:
        out = "tables/unique_fault_recovery.csv"
        if self._reuse("09_unique", [out], force):
            return self.load_table("unique_fault_recovery")
        G = self.load_table("global_fault_association").set_index("feature")
        refs = [r for r in (references or CAT.REFERENCES) if r in self.names() or r in self.E.sources]
        dirn = lambda n: G.better_direction_EXPLORATORY.get(n, "high") if n in G.index else "high"  # noqa: E731
        b = self.c["unique_budget"]
        rows, uniq_mask = [], np.zeros(self.f.shape, np.int16)
        for ref in refs:
            R = M.oriented(self.get(ref), dirn(ref))
            for n in names:
                if n == ref:
                    continue
                u = M.unique_components(M.oriented(self.get(n), dirn(n)), R, self.f, self.valid, b, self.comp_lab,
                                        self.n_comp)
                ids = u.pop("_feature_only_ids")
                if ref == refs[0] and ids:
                    uniq_mask[np.isin(self.comp_lab, ids)] += 1
                rows.append({"feature": n, "reference": ref, "budget": b, "feature_direction": dirn(n),
                             "reference_direction": dirn(ref), **u})
        T = self._table("unique_fault_recovery", pd.DataFrame(rows))
        np.save(self.store.path("datasets/unique_recovery_count_vs_first_reference.npy"), uniq_mask)
        self._plot_unique_map(uniq_mask, refs[0] if refs else "")
        top = list(G[G.status == "OK"].head(self.c["complementarity_top_k"]).index) if len(G) else names[:10]
        self.complementarity(top)
        self._ckpt("09_unique", [out, "tables/feature_complementarity.csv"])
        return T

    def complementarity(self, names) -> pd.DataFrame:
        """Pairwise unique-component share (row feature beyond column feature) + Spearman redundancy on a sample."""
        G = self.load_table("global_fault_association").set_index("feature")
        dirn = lambda n: G.better_direction_EXPLORATORY.get(n, "high") if n in G.index else "high"  # noqa: E731
        b = self.c["unique_budget"]
        rec = {}
        for n in names:
            r = M.recovery(M.oriented(self.get(n), dirn(n)), self.f, self.valid, b, self.comp_lab, self.n_comp)
            rec[n] = set(r.get("_comp_recovered", np.array([], int)).tolist())
        ncomp = max(self.n_comp, 1)
        rows = []
        rng = np.random.default_rng(self.c["seed"])
        rr, cc = np.nonzero(self.valid)
        i = rng.choice(len(rr), min(200_000, len(rr)), replace=False)
        S = pd.DataFrame({n: np.asarray(self.get(n), np.float32)[rr[i], cc[i]] for n in names}).corr("spearman")
        for a in names:
            for c in names:
                rows.append({"feature": a, "versus": c, "unique_component_share": len(rec[a] - rec[c]) / ncomp,
                             "spearman": float(S.loc[a, c])})
        T = self._table("feature_complementarity", pd.DataFrame(rows))
        self._plot_matrix(T, "unique_component_share", "complementarity_matrix", "share of components recovered by "
                          "ROW but not COLUMN (1 % budget)")
        self._plot_matrix(T, "spearman", "redundancy_matrix", "Spearman correlation (200 k valid px sample)",
                          vmin=-1, vmax=1, cmap="RdBu_r")
        return T

    def _plot_matrix(self, T, col, name, title, vmin=None, vmax=None, cmap="viridis"):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        P = T.pivot(index="feature", columns="versus", values=col)
        fig, ax = plt.subplots(figsize=(0.5 * len(P) + 3, 0.45 * len(P) + 2))
        im = ax.imshow(P.to_numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(P)))
        ax.set_xticklabels(P.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(len(P)))
        ax.set_yticklabels(P.index, fontsize=6)
        fig.colorbar(im, ax=ax)
        ax.set_title(title, fontsize=9)
        self._fig_save(fig, name, P.reset_index())

    def _plot_unique_map(self, cnt, ref):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        s = max(1, int(math.ceil(max(cnt.shape) / 1200)))
        fig, ax = plt.subplots(figsize=(8, 8))
        base = np.where(self.f, 0.3, np.nan)[::s, ::s]
        ax.imshow(np.where(self.valid[::s, ::s], 0, np.nan), extent=self._extent(), cmap="Greys", vmin=0, vmax=1)
        ax.imshow(base, extent=self._extent(), cmap="Greys", vmin=0, vmax=1)
        ax.imshow(np.where(cnt[::s, ::s] > 0, cnt[::s, ::s], np.nan), extent=self._extent(), cmap="autumn_r")
        ax.set(title=f"Mapped components recovered by >= 1 feature but not by {ref} (count of features)",
               xlabel="UTM E (km)", ylabel="UTM N (km)")
        self._fig_save(fig, "unique_fault_recovery_map", pd.DataFrame([{"reference": ref, "components_marked": int(
            len(np.unique(self.comp_lab[cnt > 0])) - (0 in np.unique(self.comp_lab[cnt > 0])))}]))

    # ================================================================== 10 W2 (optional)
    def w2_failure_analysis(self, names, budget=0.01, force=False) -> pd.DataFrame | None:
        if "w2" not in self.E.predictions:
            print("[10] no held-out W2 raster configured (w2_heldout): skipped")
            return None
        prov = self.L["CP"].provenance(str(self.p("w2_heldout")))
        w2 = self.E.raw(["w2"], self.window)["w2"]
        sel = M.top_mask(w2, self.valid, budget)
        near = ndimage.binary_dilation(sel, M.S8, iterations=3)
        missed = self.f & self.valid & ~near
        detected = self.f & self.valid & near
        rows = []
        for n in names:
            x = np.asarray(self.get(n), np.float32)
            ok = np.isfinite(x)
            a, b = x[missed & ok], x[detected & ok]
            un = x[self.valid & ok & ~self.f]
            rr = np.random.default_rng(self.c["seed"]).choice(len(un), min(len(un), 200_000), replace=False)
            u = un[rr]
            from scipy.stats import mannwhitneyu

            dm = (2 * mannwhitneyu(a, u).statistic / (len(a) * len(u)) - 1) if len(a) > 10 else np.nan
            dd = (2 * mannwhitneyu(b, u).statistic / (len(b) * len(u)) - 1) if len(b) > 10 else np.nan
            G = self.load_table("global_fault_association").set_index("feature")
            d = G.better_direction_EXPLORATORY.get(n, "high") if n in G.index else "high"
            fsel = M.top_mask(M.oriented(x, d), self.valid, budget)
            fnear = ndimage.binary_dilation(fsel, M.S8, iterations=3)
            rows.append({"feature": n, "w2_provenance": prov["status"], "missed_px": int(missed.sum()),
                         "delta_on_w2_missed_vs_unlabelled": dm, "delta_on_w2_detected_vs_unlabelled": dd,
                         "feature_recall_300m_of_w2_missed": float((fnear & missed).sum() / max(missed.sum(), 1)),
                         "extra_selected_px_beyond_w2": int((fsel & ~near).sum())})
        T = self._table("w2_missed_fault_signatures", pd.DataFrame(rows))
        self._ckpt("10_w2", ["tables/w2_missed_fault_signatures.csv"], {"provenance": prov["status"]})
        return T

    # ================================================================== 11 held-out
    def heldout_validation(self, names, force=False) -> pd.DataFrame:
        out = "tables/heldout_validation.csv"
        if self._reuse("11_heldout", [out], force):
            return self.load_table("heldout_validation")
        fp = self.p("folds")
        if not fp or not fp.is_file():
            raise FileNotFoundError(f"GEMS geographic folds not found: {fp} (outputs/benchmark/folds/...)")
        import rasterio

        with rasterio.open(fp) as s:
            if (s.height, s.width) != (self.E.H, self.E.W):
                raise ValueError(f"fold raster {s.height}x{s.width} != grid")
            roles = s.read(window=self.window)[1:]                       # role_fold{k} bands
        roles = np.where(np.isfinite(roles), roles, -1).astype(np.int8)
        A = self.L["A"]
        rows = []
        for k in range(roles.shape[0]):
            dev, ev = M.fold_masks(roles[k], self.valid)
            if not (dev & self.f).any() or not (ev & self.f).any():
                continue
            for n in names:
                x = np.asarray(self.get(n), np.float32)
                r = M.heldout_feature(x, self.f, dev, ev, tuple(self.c["budgets"]), self.near300)
                if r["status"] == "OK":
                    bb = A.block_bootstrap(x, self.f, ev, r["direction"], self.c["block_px"], self.c["n_boot"],
                                           seed=self.c["seed"])
                    r.update(eval_auroc_ci_lo=bb["auroc_ci"][0], eval_auroc_ci_hi=bb["auroc_ci"][1],
                             eval_ap_lift_ci_lo=bb["ap_lift_ci"][0], n_blocks=bb["n_blocks"])
                    rc = M.recovery(M.oriented(x, r["direction"]), self.f, ev, 0.01, *M.fault_components(self.f & ev))
                    r.update(eval_recall_300m_1pct=rc["recall_300m"], eval_component_recall_1pct=rc.get(
                        "component_recall"))
                rows.append({"fold": k, "feature": n, **r})
        T = self._table("heldout_validation", pd.DataFrame(rows))
        sel = []
        for k, g in T[T.status == "OK"].groupby("fold"):
            b = g.sort_values("dev_auroc", ascending=False).iloc[0]
            sel.append({"fold": k, "selected_on_dev": b.feature, "dev_auroc": b.dev_auroc, "eval_auroc": b.eval_auroc,
                        "eval_recall_1pct": b.get("eval_recall_1pct")})
        self._table("heldout_selected_on_dev", pd.DataFrame(sel))
        self._combinations(T, roles)
        self._ckpt("11_heldout", [out])
        return T

    def _combinations(self, T, roles, top=3):
        """Rank-average pairs of the dev-best non-redundant features, pair chosen on dev, scored on eval."""
        from itertools import combinations

        rows = []
        for k, g in T[T.status == "OK"].groupby("fold"):
            dev, ev = M.fold_masks(roles[k], self.valid)
            cand = list(g.sort_values("dev_auroc", ascending=False).feature.head(top))
            best = None
            for a, b in combinations(cand, 2):
                da = g.set_index("feature").direction
                xa, xb = M.oriented(self.get(a), da[a]), M.oriented(self.get(b), da[b])
                z = self._rank_average(xa, xb)
                d = M.association_both(z, self.f, dev)
                if d.get("status") == "OK" and (best is None or d["auroc_high"] > best[1]):
                    best = ((a, b), d["auroc_high"], z)
            if best:
                e = M.association_both(best[2], self.f, ev, (0.01,))
                rows.append({"fold": k, "pair": " + ".join(best[0]), "dev_auroc": best[1], "eval_auroc":
                             e.get("auroc_high"), "eval_recall_1pct": e.get("recall_1pct_high"),
                             "note": "pair chosen on dev only; combination tested after single features"})
        self._table("heldout_combinations", pd.DataFrame(rows))

    def _rank_average(self, a, b):
        from scipy.stats import rankdata

        a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
        ok = np.isfinite(a) & np.isfinite(b) & self.valid
        out = np.full(a.shape, np.nan, np.float32)
        out[ok] = ((rankdata(a[ok]) + rankdata(b[ok])) / (2 * ok.sum())).astype(np.float32)
        return out

    def canonical_dti(self, names, budget=0.01) -> pd.DataFrame:
        """Canonical repository scorer (gems_bench.metric.dti_components) on each feature's top-``budget`` binary
        map inside each held-out fold (direction from the held-out table). DIAGNOSTIC: not a submission."""
        if REPO not in map(Path, sys.path):
            sys.path.insert(0, str(REPO))
        from gems_bench.metric import dti_components

        import rasterio

        with rasterio.open(self.p("folds")) as s:
            roles = s.read(window=self.window)[1:]
        roles = np.where(np.isfinite(roles), roles, -1).astype(np.int8)
        H = self.load_table("heldout_validation")
        rows = []
        for k in range(roles.shape[0]):
            _, ev = M.fold_masks(roles[k], self.valid)
            for n in names:
                h = H[(H.fold == k) & (H.feature == n)]
                if not len(h) or h.status.iloc[0] != "OK":
                    continue
                pred = M.top_mask(M.oriented(self.get(n), h.direction.iloc[0]), ev, budget).astype(np.float32)
                c = dti_components(pred, self.f, eval_mask=ev, pixel_m=(self.E.pixel_m, self.E.pixel_m))
                rows.append({"fold": k, "feature": n, "budget": budget, **c.as_dict(),
                             "note": "canonical scorer on a thresholded feature map; diagnostic only"})
        return self._table("canonical_dti_diagnostic", pd.DataFrame(rows))

    # ================================================================== 12 ranking / export
    def rank_and_classify(self, force=False) -> pd.DataFrame:
        G = self.load_table("global_fault_association")
        Hd = self.load_table("heldout_validation")
        U = self.load_table("unique_fault_recovery")
        S = self.load_table("feature_scale_comparison")
        Rc = self.load_table("fault_recovery_by_budget")
        Cm = self.load_table("feature_complementarity")
        th = self.c["thresholds"]
        rows = []
        order = list(G.feature)
        for n in order:
            g = G[G.feature == n].iloc[0].to_dict()
            row = {"feature": n, "family": CAT.family(n), "exploratory_auroc": g.get("best_auroc_EXPLORATORY")}
            side = self._layer_path(n).with_suffix(".json")
            diag = json.loads(side.read_text()).get("diagnostics", {}) if side.exists() else {}
            rel = g.get("status") == "OK" and diag.get("n_inf", 0) == 0 and diag.get("valid_share", 1) >= 0.5
            row["numerically_reliable"] = bool(rel)
            row["reliability_note"] = "; ".join(diag.get("warnings", [])[:2]) if diag else ""
            h = Hd[(Hd.feature == n) & (Hd.status == "OK")] if len(Hd) else Hd
            if len(h):
                row.update(heldout_folds=int(len(h)), heldout_folds_ci_above_half=int((h.eval_auroc_ci_lo > 0.5).sum()),
                           heldout_direction_consistent=bool(h.direction.nunique() == 1),
                           heldout_auroc_mean=float(h.eval_auroc.mean()),
                           heldout_recall_1pct_mean=float(h.get("eval_recall_1pct", pd.Series(np.nan)).mean()),
                           heldout_recall_300m_mean=float(h.get("eval_recall_300m_1pct", pd.Series(np.nan)).mean()))
            u = U[(U.feature == n)] if len(U) else U
            if len(u):
                row["unique_component_share_vs_best_reference"] = float(u.groupby("reference")
                                                                        .unique_share_of_components.first().min())
            s = S[(S.feature == n) & (S.partition == "tile_20km")] if len(S) else S
            if len(s):
                row["regional_same_polarity_share"] = float(s.same_polarity_share.iloc[0])
            rc = Rc[(Rc.feature == n) & (Rc.budget == 0.01)] if len(Rc) else Rc
            if len(rc):
                d = g.get("better_direction_EXPLORATORY", "high")
                r1 = rc[rc.direction == d]
                if len(r1):
                    row.update(localization_index=float(r1.localization_index.iloc[0]),
                               selected_unlabelled_px=int(r1.selected_unlabelled_beyond_300m_px.iloc[0]))
            earlier = order[:order.index(n)]
            c = Cm[(Cm.feature == n) & (Cm.versus.isin(earlier))] if len(Cm) else Cm
            if len(c):
                j = c.spearman.abs().idxmax()
                row.update(max_abs_spearman_vs_earlier=float(abs(c.spearman[j])), most_similar=c.versus[j])
            row["class"], row["reason"] = M.classify(row, th)
            rows.append(row)
        T = pd.DataFrame(rows)
        z = lambda s: (s - s.mean()) / (s.std() or 1)  # noqa: E731
        T["score"] = (z(T.get("heldout_recall_1pct_mean", pd.Series(0, index=T.index)).fillna(0))
                      + z(T.get("unique_component_share_vs_best_reference", pd.Series(0, index=T.index)).fillna(0))
                      + 0.5 * z(T.get("localization_index", pd.Series(0, index=T.index)).fillna(0))
                      + 0.5 * z(T.get("regional_same_polarity_share", pd.Series(0, index=T.index)).fillna(0))
                      - 0.25 * z(T.get("selected_unlabelled_px", pd.Series(0, index=T.index)).fillna(0)))
        cls_order = {"PROMOTE": 0, "EXPERIMENTAL": 1, "REDUNDANT": 2, "REJECT": 3}
        T = T.sort_values(["class", "score"], key=lambda s: s.map(cls_order) if s.name == "class" else -s,
                          ignore_index=True)
        return self._table("feature_recommendations", T)

    def export_features(self, names) -> list[str]:
        """Native GeoTIFFs of computed FULL-mode features (exact grid of the features raster) + provenance JSON."""
        import rasterio

        if self.c["mode"] != "FULL":
            raise ValueError("export only from FULL mode (native full grid)")
        out = []
        with rasterio.open(self.p("features")) as src:
            prof = src.profile.copy()
        prof.update(count=1, dtype="float32", nodata=np.nan, compress="deflate", predictor=3, tiled=True,
                    blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
        for n in names:
            meta = json.loads(self._layer_path(n).with_suffix(".json").read_text())
            p = self.store.path(f"layers/geotiff/{n}_{meta['hash'][:10]}.tif")
            if p.exists():
                out.append(str(p))
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(p, "w", **prof) as d:
                d.write(np.asarray(self.get(n), np.float32), 1)
                d.set_band_description(1, n)
            self.store.json({**meta, "geotiff": str(p), "labels_used": False}, f"layers/geotiff/{p.stem}.json")
            out.append(str(p))
        return out

    def publish_to_layer_lab(self, names) -> list[str]:
        """Save features (with dependencies, frozen statistics) into the Layer Lab registry so P6 / M9 can call
        ``lab.integration.FeatureStore(root).compute(name)`` with the identical formula."""
        frozen = self.store.path("formulas/frozen_catalog.py")
        src = frozen.read_text() if frozen.exists() else self.program_text
        prog = self.E.compile(src)
        reg = self.L["FeatureLibrary"](self.root / "outputs/geological_layer_lab/registry/feature_registry.json")
        need, todo = [], list(names)
        while todo:
            n = todo.pop()
            if n not in need:
                need.append(n)
                todo += list(prog.statement(n).locals)
        done = []
        for st in [s for s in prog.statements if s.name in need]:
            if st.aoi_dependent:
                raise ValueError(f"{st.name} still AOI-dependent after freezing: compute in FULL mode first")
            reg.upsert(st.name, st.source, raw_inputs=sorted(st.raw), depends_on=sorted(st.locals | st.library),
                       description=f"geological feature discovery {FD_CODE_REV}", status="candidate",
                       halo_px=st.halo, source_program="gems_eda.feature_discovery catalog")
            done.append(st.name)
        return done

    # ================================================================== 13 synthesis
    def synthesis(self) -> str:
        from .report import synthesis_markdown

        md = synthesis_markdown(self)
        self.store.text(md, "reports/FEATURE_DISCOVERY_SYNTHESIS.md")
        self.store.json({"code_rev": FD_CODE_REV, "config": self.c, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                         "checkpoints": sorted(p.name for p in self.store.path("checkpoints").glob("*.json"))},
                        "manifests/run_manifest.json")
        return md
