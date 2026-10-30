"""Feature evaluation engine (shared by the Streamlit app, the native exporter and ``lab.integration``).

* Catalog + roles: every raster layer has a role - ``input`` (usable in formulas), ``prediction`` (W2 / M9 maps:
  comparison only) - and the mapped-fault raster is the ``target`` (evaluation only). Formulas can reference inputs
  and saved features only.
* Windowed evaluation: a request for an AOI reads ONLY the needed bands for the AOI padded by the formula halo, so
  neighbourhood operations on the AOI equal the full-grid result (exact for halo >= kernel support; documented
  tolerance 1e-5 in tests).
* Content-hash cache: every statement / saved feature is cached by (content hash, padded window); the hash includes
  the formula AST, parameter values, the hashes of its inputs (raw band file stamps, upstream features, plugin source)
  and the ops version - changing one parameter recomputes only the dependent statements.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rasterio.windows import Window

from . import featurelang as FL
from . import ops as O
from .library import FeatureLibrary
from .plugins import load_plugins
from .rasters import aligned, catalog, padded, read_faults, read_layers

TARGET_ALIASES = ("faults", "existing_faults", "mapped_faults", "fault", "label", "labels", "target", "y")
CACHE_BYTES = 1_500_000_000


@dataclass
class EvalResult:
    arrays: dict                       # name -> float32 array cropped to the AOI window
    diagnostics: dict                  # name -> diagnose() dict
    hashes: dict                       # name -> content hash
    halo: int
    window: Window
    extended: Window
    normalization: dict                # call site -> recorded AOI statistics (for freezing)
    aoi_dependent: dict                # name -> bool
    seconds: float
    cache_hits: int = 0
    computed: list = field(default_factory=list)


class Engine:
    def __init__(self, feature_tif: str, fault_tif: str | None = None, extras: list | None = None,
                 library: FeatureLibrary | None = None, plugin_dirs: list | None = None,
                 footprint_tif: str | None = None, domains_path: str | None = None):
        """``extras``: [(name, path, npz_key_or_None, role)] with role 'input' | 'prediction'."""
        ex = extras or []
        self.sources, self.details = catalog(feature_tif, [(n, p, k) for n, p, k, _ in ex])
        self.roles = {n: "input" for n in self.sources}
        for n, _, _, role in ex:
            if role not in ("input", "prediction"):
                raise ValueError(f"extra layer {n}: role must be 'input' or 'prediction'")
            self.roles[n] = role
        self.feature_tif, self.fault_tif, self.footprint_tif = feature_tif, fault_tif, footprint_tif
        self.domains_path = domains_path
        self.library = library
        self.plugins, self.plugin_errors = load_plugins(plugin_dirs)
        self.stamps = {}
        for s in self.sources.values():
            p = Path(s.path)
            self.stamps[s.name] = hashlib.sha256(f"{p.resolve()}|{s.band}|{s.key}|{p.stat().st_size}|"
                                                 f"{p.stat().st_mtime_ns}".encode()).hexdigest()[:16]
        self._cache: OrderedDict = OrderedDict()
        self._bytes = 0
        self._meta = None
        self._meta_key = None
        self._domains = None
        self.H, self.W = int(self.details["height"]), int(self.details["width"])
        self.pixel_m = float(self.details["resolution"][0])

    # ------------------------------------------------------------------ names / env
    @property
    def inputs(self) -> list[str]:
        return [n for n, r in self.roles.items() if r == "input"]

    @property
    def predictions(self) -> list[str]:
        return [n for n, r in self.roles.items() if r == "prediction"]

    def env(self) -> FL.Env:
        forbidden = {n: "prediction" for n in self.predictions}
        forbidden.update({a: "target" for a in TARGET_ALIASES if a not in self.roles})
        return FL.Env(set(self.inputs), forbidden, self.library_meta(), self.plugins)

    def library_meta(self) -> dict:
        """{saved feature: (halo_px, aoi_dependent)} compiled in dependency order (cached per registry state)."""
        if self.library is None:
            return {}
        key = json.dumps({n: e.get("formula_hash") for n, e in self.library.features.items()}, sort_keys=True)
        if self._meta_key == key:
            return self._meta
        meta = {}
        for n in self.library.check_graph():
            if n not in self.library.features:
                continue
            env = FL.Env(set(self.inputs), {n_: "prediction" for n_ in self.predictions}, dict(meta), self.plugins)
            try:
                prog = FL.compile_program(self.library.program_source(n), env)
                st = prog.statement(n)
                meta[n] = (st.halo, st.aoi_dependent)
            except FL.FeatureError:
                continue                                  # reported by validate_library()
        self._meta, self._meta_key = meta, key
        return meta

    def validate_library(self) -> dict:
        """{name: error} for saved features that no longer compile (missing band, removed plugin, ...)."""
        out = {}
        meta = {}
        for n in (self.library.check_graph() if self.library else []):
            if n not in self.library.features:
                continue
            try:
                env = FL.Env(set(self.inputs), {}, dict(meta), self.plugins)
                prog = FL.compile_program(self.library.program_source(n), env)
                st = prog.statement(n)
                meta[n] = (st.halo, st.aoi_dependent)
            except FL.FeatureError as e:
                out[n] = str(e)
        return out

    def compile(self, source: str, values: dict | None = None) -> FL.Program:
        return FL.compile_program(source, self.env(), values)

    # ------------------------------------------------------------------ cache
    def _get(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _put(self, key, arr):
        self._cache[key] = arr
        self._bytes += arr.nbytes
        while self._bytes > CACHE_BYTES and len(self._cache) > 1:
            _, old = self._cache.popitem(last=False)
            self._bytes -= old.nbytes

    def clear_cache(self):
        self._cache.clear()
        self._bytes = 0

    # ------------------------------------------------------------------ evaluation
    def _library_programs(self, names) -> list:
        out = []
        meta = self.library_meta()
        for n in self.library.closure(names) if self.library and names else []:
            env = FL.Env(set(self.inputs), {}, {k: v for k, v in meta.items() if k != n}, self.plugins)
            out.append((n, FL.compile_program(self.library.program_source(n), env)))
        return out

    def evaluate(self, program: FL.Program | None, window: Window, targets=None, library_targets=(),
                 min_halo: int = 0) -> EvalResult:
        """Evaluate ``targets`` (default: every statement) of ``program`` and the saved features ``library_targets``
        on ``window``; returns arrays cropped to the window."""
        t0 = time.time()
        stmts = list(program.statements) if program else []
        if targets is not None and program is not None:
            need, todo = set(), list(targets)
            while todo:
                n = todo.pop()
                if n in need:
                    continue
                need.add(n)
                todo += list(program.statement(n).locals)
            stmts = [s for s in stmts if s.name in need]
        local_names = {s.name for s in stmts}
        lib_refs = set(library_targets) | (set().union(*[s.library for s in stmts]) if stmts else set())
        lib_progs = [(n, p) for n, p in self._library_programs(lib_refs - local_names)]
        meta = self.library_meta()
        halo = max([s.halo for s in stmts] + [meta.get(n, (0, False))[0] for n in library_targets] + [min_halo, 0])
        ext, roi = padded(window, self.H, self.W, halo)
        ekey = (int(ext.col_off), int(ext.row_off), int(ext.width), int(ext.height))
        raw = set().union(*[s.raw for s in stmts]) if stmts else set()
        for _, p in lib_progs:
            raw |= p.raw_inputs
        values, hashes = {}, {}
        hits = 0
        miss = [r for r in raw if self._get(("raw", r, self.stamps[r], ekey)) is None]
        if miss:
            for r, a in read_layers(self.sources, miss, ext).items():
                self._put(("raw", r, self.stamps[r], ekey), np.asarray(a, np.float32))
        for r in raw:
            values[r] = self._get(("raw", r, self.stamps[r], ekey))
            hashes[r] = self.stamps[r]
        for name, spec in self.plugins.items():
            hashes[name] = spec.source_hash
        shape = (int(ext.height), int(ext.width))
        rec = O.set_recorder(O.Recorder())
        computed = []

        def run(st, prog):
            nonlocal hits
            h = FL.expr_hash(st, prog, hashes)
            hashes[st.name] = h
            cached = self._get(("expr", h, ekey))
            if cached is not None:
                hits += 1
                values[st.name] = cached
                return
            arr = FL.run_statement(st, prog, lambda k: values[k], self.env_for(prog), shape)
            arr = np.where(np.isfinite(arr), arr, np.nan).astype(np.float32)
            if arr.nbytes > 512 * 1024 * 1024:
                raise FL.FeatureError(f"{st.name}: result exceeds 512 MiB; select a smaller AOI")
            self._put(("expr", h, ekey), arr)
            values[st.name] = arr
            computed.append(st.name)

        divisions = {}
        for n, p in lib_progs:
            k = len(rec.division)
            run(p.statement(n), p)
            divisions[n] = rec.division[k:]
        for st in stmts:
            k = len(rec.division)
            run(st, program)
            divisions[st.name] = rec.division[k:]
        crop = lambda a: np.asarray(a[roi], np.float32)                        # noqa: E731
        out_names = [s.name for s in stmts] + [n for n in library_targets if n not in local_names]
        arrays = {n: crop(values[n]) for n in out_names}
        diags = {}
        for st in stmts:
            vin = np.logical_and.reduce([np.isfinite(values[r])[roi] for r in st.raw]) if st.raw else None
            diags[st.name] = FL.diagnose(arrays[st.name], vin, divisions.get(st.name))
        for n in library_targets:
            if n not in local_names:
                diags[n] = FL.diagnose(arrays[n], None, divisions.get(n))
        aoi = {s.name: s.aoi_dependent for s in stmts}
        aoi.update({n: meta.get(n, (0, False))[1] for n in library_targets if n not in local_names})
        O.set_recorder(None)
        return EvalResult(arrays, diags, {n: hashes.get(n) for n in out_names}, halo, window, ext,
                          dict(rec.normalization), aoi, round(time.time() - t0, 3), hits, computed)

    def env_for(self, prog: FL.Program) -> FL.Env:
        return FL.Env(set(self.inputs), {}, {}, self.plugins)

    def raw(self, names, window: Window) -> dict:
        """Input or prediction layers on the window (no halo)."""
        return {k: np.asarray(v, np.float32) for k, v in read_layers(self.sources, list(names), window).items()}

    # ------------------------------------------------------------------ targets / masks (evaluation only)
    def faults(self, window: Window, halo: int = 0):
        """Mapped-fault mask of the window (TARGET: evaluation and diagnostics only). With ``halo`` > 0 returns
        (padded mask, roi slices) so distance-based diagnostics see faults just outside the AOI."""
        if not self.fault_tif or not Path(self.fault_tif).is_file():
            return None if not halo else (None, None)
        if not halo:
            return read_faults(self.fault_tif, self.feature_tif, window)
        ext, roi = padded(window, self.H, self.W, halo)
        return read_faults(self.fault_tif, self.feature_tif, ext), roi

    def footprint(self, window: Window):
        """Valid submission footprint (finite template pixels) of the window, or None."""
        if not self.footprint_tif or not Path(self.footprint_tif).is_file():
            return None
        import rasterio

        with rasterio.open(self.footprint_tif) as s, rasterio.open(self.feature_tif) as ref:
            if not aligned(s, ref):
                raise ValueError("submission template is not aligned with the feature raster")
            a = s.read(1, window=window, masked=True)
        return ~np.ma.getmaskarray(a) & np.isfinite(np.ma.filled(a.astype(np.float32), np.nan))

    def domains(self, window: Window):
        """Label-free domain ids (EDA ``rasters/spatial_domains/domains.npz``) of the window, or None."""
        if not self.domains_path or not Path(self.domains_path).is_file():
            return None
        if self._domains is None:
            with np.load(self.domains_path) as z:
                d = z["domain"]
            if d.shape != (self.H, self.W):
                raise ValueError(f"domains raster {d.shape} != grid {(self.H, self.W)}")
            self._domains = d
        r0, c0 = int(window.row_off), int(window.col_off)
        return self._domains[r0:r0 + int(window.height), c0:c0 + int(window.width)]

    def grid_report(self) -> dict:
        """CRS / transform / shape of the feature raster and alignment of the fault and template rasters."""
        import rasterio

        rep = {"crs": self.details["crs"], "transform": self.details["transform"], "height": self.H,
               "width": self.W, "pixel_m": self.pixel_m, "bands": self.details["bands"]}
        with rasterio.open(self.feature_tif) as ref:
            rep["feature_nodata"] = ref.nodata
            for k, p in (("faults", self.fault_tif), ("footprint", self.footprint_tif)):
                if p and Path(p).is_file():
                    with rasterio.open(p) as s:
                        rep[f"{k}_aligned"] = bool(aligned(s, ref))
                        rep[f"{k}_nodata"] = s.nodata
                else:
                    rep[f"{k}_aligned"] = None
        return rep
