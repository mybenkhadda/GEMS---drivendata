"""G01 inputs and persistence: configuration, read-only access to the M9 layer cache, channel audit, versioned store.

Inputs
    ``M9Source(m9)`` reads cached layers through ``m9.features.read`` (memory-mapped ``cache/m9/layers/*.npy``) and
    serves two label-free virtual channels computed once per session from raw bands: ``g01:edge`` (THDR of the raw
    magnetic band: the simple-edge baseline) and ``g01:surface`` (THDR of detrended elevation: surface expression).
    G01 never reads ``build_cube`` metadata (``meta["pooled"]`` is overwritten by a path there): only named layers.

Channel audit (``channel_audit``)
    Parses the cached layer names, reports per (field, wavelength, kind) present / missing / duplicate (two tags that
    parse to the same wavelength, e.g. ``0.4`` and ``0.40``), and returns the wavelengths cached for BOTH fields
    (aligned within ``wavelength_rel_tol``). Unaligned scales are recorded and not used.

Store (``G01Store``)
    ``<m9 workspace>/research/G01_wavelet_geology/g01_<fingerprint>/{config,figures,tables,rasters,processed,reports}``.
    The fingerprint covers the resolved configuration, every input layer fingerprint, the grid and the G01 code version:
    other inputs => another folder; nothing is overwritten silently (an existing file is kept and reused unless
    ``overwrite``, which is recorded in the manifest). ``manifest.json`` maps every artifact to its sources and
    parameters; ``processed/done_<step>.json`` markers make every step resumable. ``LATEST.json`` (one level up)
    points to the newest run.
"""
from __future__ import annotations

import copy
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .geology_wavelet import KINDS, ROLES, Wavelength, channel

G01_VERSION = "g01.1"
CONFIG_PARTS = ("m9", "g01_wavelet_geology.yaml")
NORMALIZATIONS = ("training_geography", "all_valid")


class G01ConfigError(ValueError):
    pass


class LeakageError(RuntimeError):
    """A label-free calibration / reference region overlaps the geography it is evaluated on."""


# =============================================================================== configuration
def load_g01_config(overrides: dict | None = None, path: Path | None = None) -> dict:
    import yaml

    from ...paths import config_path
    from ..config import M9ConfigError, merge

    p = Path(path) if path else config_path(*CONFIG_PARTS)
    base = yaml.safe_load(p.read_text(encoding="utf-8"))
    try:
        cfg = merge(base, overrides or {})
    except M9ConfigError as e:
        raise G01ConfigError(str(e).replace("M9", "G01")) from None
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    from .geology_wavelet import CANDIDATES

    bad = [c for c in cfg["candidates"] if c not in CANDIDATES]
    if bad:
        raise G01ConfigError(f"candidates: unknown {bad}; known {list(CANDIDATES)}")
    if set(cfg["fields"]) != set(ROLES):
        raise G01ConfigError(f"fields must map exactly {ROLES} to M9 spectral families")
    if cfg["normalization"]["reference"] not in NORMALIZATIONS:
        raise G01ConfigError(f"normalization.reference must be one of {NORMALIZATIONS}")
    ev = cfg["evaluation"]
    for k in ("area_fracs", "ablation_area_fracs"):
        if not ev[k] or any(not 0 < float(f) < 1 for f in ev[k]):
            raise G01ConfigError(f"evaluation.{k}: fractions in (0, 1)")
    if float(ev["primary_area_frac"]) not in [float(f) for f in ev["area_fracs"]]:
        raise G01ConfigError("evaluation.primary_area_frac must be one of evaluation.area_fracs")
    if float(ev["primary_distance_m"]) not in [float(d) for d in ev["distance_tol_m"]]:
        raise G01ConfigError("evaluation.primary_distance_m must be one of evaluation.distance_tol_m")
    if any(float(d) < 0 for d in ev["distance_tol_m"]):
        raise G01ConfigError("evaluation.distance_tol_m must be >= 0")
    if ev["zones"] not in ("all", "profile"):
        raise G01ConfigError("evaluation.zones must be all | profile")
    bad = [t for t in ev["tasks"] if t not in ("geo", "disc_official", "disc_retained_fp")]
    if bad:
        raise G01ConfigError(f"evaluation.tasks: unknown {bad}")
    tc = cfg["tolerance"]
    if not 0 <= float(tc["min_m"]) <= float(tc["max_m"]) or float(tc["per_wavelength"]) < 0:
        raise G01ConfigError("tolerance: 0 <= min_m <= max_m and per_wavelength >= 0")
    if int(tc["n_rings"]) < 1 or float(tc["reach"]) <= 0 or tc["footprint"] not in ("square", "disk"):
        raise G01ConfigError("tolerance: n_rings >= 1, reach > 0, footprint square | disk")
    if any(float(t) < 0 for t in tc.get("ablation_m") or []):
        raise G01ConfigError("tolerance.ablation_m must be >= 0")
    pc = cfg["persistence"]
    if not 0 < float(pc["q"]) < 1 or not 0 <= float(pc["floor"]) <= 1:
        raise G01ConfigError("persistence: q in (0, 1), floor in [0, 1]")
    if float(cfg["boundary_margin_m"]) < 0:
        raise G01ConfigError("boundary_margin_m must be >= 0")
    if cfg["save"]["raster_dtype"] not in ("float16", "float32"):
        raise G01ConfigError("save.raster_dtype must be float16 | float32")
    if cfg["baselines"]["edge_band"] not in cfg["raw_bands"]:
        raise G01ConfigError("baselines.edge_band must be a key of raw_bands")
    dc = cfg["decision"]
    if dc["metric"] not in ("f1", "precision", "recall", "enrichment"):
        raise G01ConfigError("decision.metric must be f1 | precision | recall | enrichment")
    if not 0.5 <= float(dc["min_p_positive"]) < 1 or not 0 < float(dc["min_zone_win_share"]) <= 1:
        raise G01ConfigError("decision: min_p_positive in [0.5, 1), min_zone_win_share in (0, 1]")
    lam = cfg["wavelengths_km"]
    if lam != "auto" and (not isinstance(lam, list) or not lam or any(float(x) <= 0 for x in lam)):
        raise G01ConfigError("wavelengths_km must be 'auto' or a non-empty list of positive wavelengths")


def fingerprint_config(cfg: dict) -> dict:
    """The part of the configuration that changes results (display / persistence switches excluded)."""
    c = copy.deepcopy(cfg)
    for k in ("overwrite", "figures", "save", "build_missing"):
        c.pop(k, None)
    return c


# =============================================================================== channel names / audit
_SPEC = re.compile(r"^m9_spec_(?P<fam>[A-Za-z0-9]+)_wav_(?P<kind>energy|coherence|strike_c2|strike_s2)@(?P<tag>[0-9.]+)km$")
_XMOD = re.compile(r"^m9_xmod_(?P<rest>.+)$")


def names_from_stems(stems: list[str]) -> list[str]:
    """Layer-cache file stems (``LayerCache.path`` sanitises ':' to '_') -> canonical M9 layer names (spec / xmod)."""
    out = []
    for s in stems:
        m = _SPEC.match(s)
        if m:
            out.append(channel(m["fam"], m["kind"], m["tag"]))
            continue
        x = _XMOD.match(s)
        if x:
            out.append(f"m9:xmod:{x['rest']}")
    return sorted(set(out))


def parse_wavelet_names(names: list[str]) -> pd.DataFrame:
    rx = re.compile(r"^m9:spec:(?P<fam>[A-Za-z0-9]+)_wav_(?P<kind>energy|coherence|strike_c2|strike_s2)@(?P<tag>[0-9.]+)km$")
    rows = []
    for n in names:
        m = rx.match(n)
        if m:
            rows.append({"name": n, "family": m["fam"], "kind": m["kind"], "tag": m["tag"], "lam_km": float(m["tag"])})
    return pd.DataFrame(rows, columns=["name", "family", "kind", "tag", "lam_km"])


def channel_audit(names: list[str], fields: dict, requested="auto", rel_tol: float = 0.02) -> tuple[pd.DataFrame, list]:
    """(audit table, aligned ``Wavelength`` list). Status per (role, wavelength, kind): present | missing |
    duplicate (extra tags for the same wavelength: the lexicographically first tag is used) | unaligned."""
    P = parse_wavelet_names(names)
    rows, per_role = [], {}
    for role in ROLES:
        fam = fields[role]
        sub = P[P.family == fam]
        lams = {}
        for lam, g in sub.groupby(sub.lam_km.round(6)):
            tags = sorted(set(g.tag))
            lams[float(lam)] = tags
            for t in tags[1:]:
                rows.append({"role": role, "family": fam, "lam_km": float(lam), "kind": "*", "tag": t,
                             "status": "duplicate", "name": ""})
        per_role[role] = lams
    aligned = []
    for lam_m, tags_m in sorted(per_role["mag"].items()):
        match = [lg for lg in per_role["grav"] if abs(lg - lam_m) <= rel_tol * lam_m]
        if not match:
            continue
        lam_g = min(match, key=lambda x: abs(x - lam_m))
        aligned.append(Wavelength(round(lam_m, 6), (("mag", tags_m[0]), ("grav", per_role["grav"][lam_g][0]))))
    if requested != "auto":
        want = [float(x) for x in requested]
        keep = [w for w in aligned if any(abs(w.lam_km - x) <= rel_tol * x for x in want)]
        for x in want:
            if not any(abs(w.lam_km - x) <= rel_tol * x for w in keep):
                rows.append({"role": "both", "family": "", "lam_km": x, "kind": "energy", "tag": "", "status":
                             "requested_missing", "name": ""})
        aligned = keep
    have = set(names)
    used = {(r, w.tag(r)) for w in aligned for r in ROLES}
    for role in ROLES:
        fam = fields[role]
        for lam, tags in per_role[role].items():
            if (role, tags[0]) not in used:
                rows.append({"role": role, "family": fam, "lam_km": lam, "kind": "*", "tag": tags[0],
                             "status": "unaligned", "name": ""})
    for w in aligned:
        for role in ROLES:
            for k in KINDS:
                n = channel(fields[role], k, w.tag(role))
                rows.append({"role": role, "family": fields[role], "lam_km": w.lam_km, "kind": k, "tag": w.tag(role),
                             "status": "present" if n in have else "missing", "name": n})
    T = pd.DataFrame(rows, columns=["role", "family", "lam_km", "kind", "tag", "status", "name"])
    usable = [w for w in aligned if all(channel(fields[r], "energy", w.tag(r)) in have for r in ROLES)]
    return T, usable


def xmod_names(names: list[str], patterns: list[str]) -> list[str]:
    pats = [re.compile(p) for p in patterns]
    return sorted(n for n in names if any(p.search(n) for p in pats))


# =============================================================================== M9 source
class M9Source:
    """Read-only channel access for G01 (M9 layer cache + label-free virtual channels)."""

    def __init__(self, m9, cfg: dict):
        self.m9, self.cfg = m9, cfg
        self._virt: dict = {}

    @property
    def data(self):
        return self.m9.ctx.data

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.data.shape)

    @property
    def valid(self) -> np.ndarray:
        return np.asarray(self.data.valid, bool)

    @property
    def pixel_m(self) -> tuple[float, float]:
        g = self.data.grid
        return float(g.pixel_km_y) * 1000.0, float(g.pixel_km_x) * 1000.0

    def names(self) -> list[str]:
        d = self.m9.layers.dir
        return names_from_stems([p.stem for p in Path(d).glob("*.npy")]) if Path(d).exists() else []

    def has(self, name: str) -> bool:
        if name in ("g01:edge", "g01:surface"):
            return self._band(name) in self.data.band_names
        from ..features import available

        return available(self.m9, name)

    def get(self, name: str) -> np.ndarray:
        if name in ("g01:edge", "g01:surface"):
            if name not in self._virt:
                self._virt[name] = self._thdr(self._band(name))
            return self._virt[name]
        from ..features import read

        return read(self.m9, name)

    def fingerprint(self, name: str) -> str:
        if name in ("g01:edge", "g01:surface"):
            return f"virtual:thdr:{self._band(name)}@{float(self.cfg['baselines']['edge_sigma_m']):g}m"
        from ..features import file_fp

        return file_fp(self.m9.layers.path(name)) if name.startswith("m9:") else "store"

    def release(self) -> None:
        self.m9.layers.release()

    def _band(self, name: str) -> str:
        rb = self.cfg["raw_bands"]
        return rb[self.cfg["baselines"]["edge_band"]] if name == "g01:edge" else rb["terrain"]

    def _thdr(self, band: str) -> np.ndarray:
        """Total horizontal derivative (per metre) of a raw band at ``edge_sigma_m`` (label-free)."""
        from scipy import ndimage

        from ...m8.potential_fields import fill

        a = self.data.read_band(band)
        f, m = fill(a)
        py, px = self.pixel_m
        s = max(float(self.cfg["baselines"]["edge_sigma_m"]), 1.0)
        sig = (s / py, s / px)
        dr = ndimage.gaussian_filter(f.astype(np.float32), sig, order=(1, 0), mode="nearest") / py
        dc = ndimage.gaussian_filter(f.astype(np.float32), sig, order=(0, 1), mode="nearest") / px
        return np.where(m & self.valid, np.hypot(dr, dc), np.nan).astype(np.float32)


def build_missing_layers(m9, cfg: dict, audit: pd.DataFrame, say=print) -> list[str]:
    """Opt-in: compute ONLY what the M9 cache lacks with M9's own code and naming (never S01-S09 as a whole)."""
    built = []
    bm = cfg["build_missing"]
    miss = audit[audit.status == "missing"] if len(audit) else audit
    if bm.get("wavelets") and (len(miss) or not len(audit)):
        from gems_eda.enhance import standardize_signal

        from .. import spectral as SP

        sc = m9.cfg["spectral"]
        valid = np.asarray(m9.ctx.data.valid, bool)
        for role, fam in cfg["fields"].items():
            band = sc["fields"].get(fam)
            if band is None or band not in m9.ctx.data.band_names:
                say(f"G01 build_missing: spectral family {fam} has no band -> skipped")
                continue
            need = [channel(fam, "energy", f"{lam:g}") for lam in sc["wavelet"]["wavelengths_km"]]
            if all(m9.layers.has(n) for n in need):
                continue
            say(f"G01 build_missing: log-Gabor layers of {fam} ({band}) with M9 spectral.wavelet_layers")
            for k, v in SP.wavelet_layers(m9.ctx.data.read_band(band), valid, sc["wavelet"], fam).items():
                full = f"m9:spec:{k}"
                if m9.layers.has(full):
                    continue
                v = np.where(valid, v, np.nan).astype(np.float32)
                z = v if any(t in k for t in SP.BOUNDED) else standardize_signal(v)[0]   # exactly spectral.build's rule
                m9.layers.put(full, z)
                built.append(full)
            m9.layers.release()
    if bm.get("crossmodal"):
        from ..features import crossmodal_layers

        say("G01 build_missing: M9 section-02 cross-modal layers (features.crossmodal_layers; existing ones kept)")
        reg = crossmodal_layers(m9)
        built += list(reg.name) if len(reg) else []
    return built


# =============================================================================== store
def _jsonable(o):
    return json.loads(json.dumps(o, default=str))


class G01Store:
    SUBDIRS = ("config", "figures", "tables", "rasters", "rasters/units", "processed", "reports")

    def __init__(self, root: Path, *, overwrite: bool = False, fingerprint: str = "", push=None):
        self.root = Path(root)
        self.overwrite = bool(overwrite)
        self.fp = fingerprint
        self._push = push
        for s in self.SUBDIRS:
            (self.root / s).mkdir(parents=True, exist_ok=True)
        self.entries: dict = {}

    # -- paths / markers -------------------------------------------------------------
    def path(self, rel: str) -> Path:
        return self.root / rel

    def exists(self, rel: str) -> bool:
        return self.path(rel).exists()

    def _writable(self, rel: str) -> bool:
        """False when the file exists and overwrite is off (kept and reused: never replaced silently)."""
        return self.overwrite or not self.path(rel).exists()

    def is_done(self, step: str) -> bool:
        return self.exists(f"processed/done_{_safe(step)}.json")

    def mark_done(self, step: str, info: dict | None = None) -> None:
        from ...artifacts.atomic import atomic_write_json

        atomic_write_json(self.path(f"processed/done_{_safe(step)}.json"),
                          _jsonable({"step": step, "fingerprint": self.fp, "g01_version": G01_VERSION,
                                     "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "info": info or {}}))

    def record(self, rel: str, kind: str, sources=None, params=None) -> None:
        from ..features import file_fp

        self.entries[rel] = {"path": rel, "kind": kind, "sources": sources or [], "params": params or {},
                             "content_fp": file_fp(self.path(rel)), "recorded": time.strftime("%Y-%m-%dT%H:%M:%S"),
                             "overwrite_mode": self.overwrite}

    def flush_manifest(self, extra: dict | None = None) -> Path:
        from ...artifacts.atomic import atomic_write_json

        p = self.path("manifest.json")
        old = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        arts = {**(old.get("artifacts") or {}), **self.entries}
        body = {**old, **(extra or {}), "fingerprint": self.fp, "g01_version": G01_VERSION,
                "updated": time.strftime("%Y-%m-%dT%H:%M:%S"), "artifacts": arts}
        return atomic_write_json(p, _jsonable(body))

    # -- writers -----------------------------------------------------------------------
    def save_json(self, obj, rel: str, sources=None, params=None) -> Path:
        from ...artifacts.atomic import atomic_write_json

        if self._writable(rel):
            atomic_write_json(self.path(rel), _jsonable(obj))
        self.record(rel, "json", sources, params)
        return self.path(rel)

    def load_json(self, rel: str) -> dict | None:
        p = self.path(rel)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def save_table(self, df: pd.DataFrame, name: str, sub: str = "tables", sources=None, params=None,
                   parquet: bool = True) -> Path:
        """CSV always + Parquet (reusable intermediate) when possible."""
        from ...artifacts.atomic import atomic_save
        from ...pipeline.m6 import _parquet_safe

        df = df if df is not None else pd.DataFrame()
        rel = f"{sub}/{name}.csv"
        if self._writable(rel):
            atomic_save(self.path(rel), lambda t: df.to_csv(t, index=False))
        self.record(rel, "table_csv", sources, params)
        if parquet and len(df):
            rq = f"{sub}/{name}.parquet"
            if self._writable(rq):
                try:
                    atomic_save(self.path(rq), lambda t: _parquet_safe(df).to_parquet(t, index=False))
                    self.record(rq, "table_parquet", sources, params)
                except Exception as e:                               # pyarrow missing: CSV stays the record
                    print(f"[g01] parquet skipped for {name}: {type(e).__name__}: {e}")
            else:
                self.record(rq, "table_parquet", sources, params)
        return self.path(rel)

    def load_table(self, name: str, sub: str = "tables") -> pd.DataFrame:
        for ext, reader in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
            p = self.path(f"{sub}/{name}{ext}")
            if p.exists():
                try:
                    return reader(p)
                except Exception:
                    continue
        return pd.DataFrame()

    def save_raster(self, arrays: dict, rel: str, meta: dict, dtype: str = "float16", sources=None) -> Path:
        """Compressed NPZ; float arrays cast to ``dtype``, bool / int arrays kept; ``meta`` stored as JSON text."""
        from ...artifacts.atomic import atomic_save

        if self._writable(rel):
            pack = {}
            for k, v in arrays.items():
                v = np.asarray(v)
                pack[_safe(k)] = v.astype(dtype) if np.issubdtype(v.dtype, np.floating) else v
            keys = {_safe(k): k for k in arrays}
            pack["__meta__"] = np.array(json.dumps(_jsonable({**meta, "keys": keys, "dtype": dtype})))

            def w(t):
                with open(t, "wb") as f:
                    np.savez_compressed(f, **pack)
            atomic_save(self.path(rel), w)
        self.record(rel, "raster_npz", sources, {k: v for k, v in meta.items() if k != "refs"})
        return self.path(rel)

    def load_raster(self, rel: str) -> tuple[dict, dict]:
        with np.load(self.path(rel), allow_pickle=False) as z:
            meta = json.loads(str(z["__meta__"]))
            keys = meta.get("keys") or {}
            arrs = {keys.get(k, k): np.asarray(z[k]) for k in z.files if k != "__meta__"}
        return arrs, meta

    def save_geotiff(self, a: np.ndarray, rel: str, grid, window: tuple[int, int, int, int]) -> Path | None:
        """Float32 GeoTIFF of one window (CRS and transform of the working grid)."""
        try:
            import rasterio
            from rasterio.transform import Affine
        except ImportError:
            return None
        from ...artifacts.atomic import atomic_save

        r0, r1, c0, c1 = window
        tr = grid.transform * Affine.translation(c0, r0)
        prof = {"driver": "GTiff", "height": r1 - r0, "width": c1 - c0, "count": 1, "dtype": "float32",
                "crs": grid.crs, "transform": tr, "nodata": np.nan, "compress": "deflate"}
        if self._writable(rel):
            def w(t):
                with rasterio.open(t, "w", **prof) as dst:
                    dst.write(np.asarray(a, np.float32), 1)
            atomic_save(self.path(rel), w)
        self.record(rel, "geotiff", [], {"window": list(window)})
        return self.path(rel)

    def save_fig(self, fig, name: str, data: pd.DataFrame | None = None, dpi: int = 170, pdf: bool = True,
                 params=None) -> Path | None:
        import matplotlib.pyplot as plt

        from ...artifacts.atomic import atomic_save

        if fig is None:
            return None
        for ext in ("png", "pdf") if pdf else ("png",):
            rel = f"figures/{name}.{ext}"
            if self._writable(rel):
                atomic_save(self.path(rel), lambda t, e=ext: fig.savefig(t, dpi=dpi, bbox_inches="tight", format=e))
            self.record(rel, f"figure_{ext}", [], params)
        if data is not None and len(data):
            self.save_table(data, f"{name}_data", "figures", parquet=False)
        plt.close(fig)
        return self.path(f"figures/{name}.png")

    def push(self) -> None:
        if self._push is not None:
            self._push()


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_@.=|-]+", "_", str(s))


def run_root(m9, cfg: dict) -> Path:
    return Path(m9.ws.root) / cfg["output_subdir"]


def latest_run(parent: Path) -> str | None:
    p = Path(parent) / "LATEST.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("run_id")
    except (OSError, json.JSONDecodeError):
        return None


def set_latest(parent: Path, run_id: str, fp: str) -> None:
    from ...artifacts.atomic import atomic_write_json

    atomic_write_json(Path(parent) / "LATEST.json", {"run_id": run_id, "fingerprint": fp, "g01_version": G01_VERSION,
                                                      "updated": time.strftime("%Y-%m-%dT%H:%M:%S")})


def list_runs(parent: Path) -> pd.DataFrame:
    rows = []
    for d in sorted(Path(parent).glob("g01_*")):
        man = d / "manifest.json"
        info = json.loads(man.read_text(encoding="utf-8")) if man.exists() else {}
        rows.append({"run_id": d.name, "fingerprint": info.get("fingerprint"), "updated": info.get("updated"),
                     "n_artifacts": len(info.get("artifacts") or {}),
                     "steps_done": sorted(p.stem.replace("done_", "") for p in (d / "processed").glob("done_*.json"))})
    return pd.DataFrame(rows)
