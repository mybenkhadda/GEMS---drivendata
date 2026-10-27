"""G02 configuration and persistence (built on the G01 store; read-only on M9 and on G01 runs).

Configuration
    ``load_g02_config(overrides)``: ``configs/m9/g02_structural_geology.yaml`` <- ``modes[<mode>]`` <- ``overrides``
    (the notebook's G02_CONFIG). Unknown keys are rejected at every step (M9 ``merge``). The mode is read from the
    overrides first, so ``{"mode": "FULL_EVALUATION"}`` alone switches every mode-dependent default.

Store (``G02Store``)
    ``<m9 workspace>/research/G02_structural_geology/g02_<fingerprint>/``
        config/  processed/  rasters/units/  lineaments/  tables/  figures/  reports/  manifests/
    Write-once like G01 (an existing file is kept unless ``overwrite``), ``manifests/manifest.json`` maps every artifact
    to its sources and parameters, ``processed/done_<step>.json`` markers make every step resumable, ``LATEST.json``
    one level up names the newest run. Rasters are written array by array into the NPZ (bounded memory); vectors as
    GeoJSON (always) and GeoPackage (when geopandas is installed).
"""
from __future__ import annotations

import copy
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .geology_io import G01ConfigError, G01Store, _jsonable, _safe

G02_VERSION = "g02.1"
CONFIG_PARTS = ("m9", "g02_structural_geology.yaml")
MODES = ("FAST_RESEARCH", "FULL_EVALUATION")
G02_FAMILIES = ("G02-A", "G02-B", "G02-C", "G02-D", "G02-E")


class G02ConfigError(ValueError):
    pass


# =============================================================================== configuration
def _merge(base: dict, over: dict) -> dict:
    from ..config import M9ConfigError, merge

    try:
        return merge(base, over or {})
    except M9ConfigError as e:
        raise G02ConfigError(str(e).replace("M9", "G02")) from None


def load_g02_config(overrides: dict | None = None, path: Path | None = None) -> dict:
    import yaml

    from ...paths import config_path

    p = Path(path) if path else config_path(*CONFIG_PARTS)
    base = yaml.safe_load(p.read_text(encoding="utf-8"))
    over = copy.deepcopy(overrides or {})
    mode = over.get("mode", base.get("mode"))
    if mode not in MODES:
        raise G02ConfigError(f"mode must be one of {MODES}, got {mode!r}")
    cfg = _merge(_merge(base, (base.get("modes") or {}).get(mode) or {}), over)
    cfg["mode"] = mode
    validate(cfg)
    return cfg


def g01_config(cfg: dict) -> dict:
    """The G01 configuration G02 runs G01 code with (G01 validation applies)."""
    from .geology_io import load_g01_config

    try:
        return load_g01_config(cfg["g01"].get("overrides") or {})
    except G01ConfigError as e:
        raise G02ConfigError(f"g01.overrides: {e}") from None


def validate(cfg: dict) -> None:
    ev = cfg["evaluation"]
    for k in ("area_fracs", "ablation_area_fracs"):
        if not ev[k] or any(not 0 < float(f) < 1 for f in ev[k]):
            raise G02ConfigError(f"evaluation.{k}: fractions in (0, 1)")
    if float(ev["primary_area_frac"]) not in [float(f) for f in ev["area_fracs"]]:
        raise G02ConfigError("evaluation.primary_area_frac must be one of evaluation.area_fracs")
    if float(ev["primary_area_frac"]) not in [float(f) for f in ev["ablation_area_fracs"]]:
        raise G02ConfigError("evaluation.primary_area_frac must be one of evaluation.ablation_area_fracs (nested "
                             "choices and contrasts are made there)")
    if float(ev["primary_distance_m"]) not in [float(d) for d in ev["distance_tol_m"]]:
        raise G02ConfigError("evaluation.primary_distance_m must be one of evaluation.distance_tol_m")
    if any(float(d) < 0 for d in ev["distance_tol_m"]):
        raise G02ConfigError("evaluation.distance_tol_m must be >= 0")
    if ev["zones"] not in ("all", "profile"):
        raise G02ConfigError("evaluation.zones must be all | profile")
    bad = [t for t in ev["tasks"] if t not in ("geo", "disc_official", "disc_retained_fp")]
    if bad:
        raise G02ConfigError(f"evaluation.tasks: unknown {bad}")
    st = [float(x) for x in ev["length_strata_km"]]
    if len(st) < 2 or any(b <= a for a, b in zip(st, st[1:])):
        raise G02ConfigError("evaluation.length_strata_km must be strictly increasing (>= 2 edges)")
    if int(cfg["orientation"]["n_azimuths"]) < 2:
        raise G02ConfigError("orientation.n_azimuths >= 2")
    if float(cfg["bandpass"]["sigma_ratio"]) <= 1.0:
        raise G02ConfigError("bandpass.sigma_ratio must be > 1")
    dc = cfg["displacement"]
    if not 0 < float(dc["h1_frac"]) < float(dc["h2_frac"]):
        raise G02ConfigError("displacement: 0 < h1_frac < h2_frac")
    lo, hi = (float(x) for x in dc["energy_gate"])
    if not 0 <= lo < hi <= 1:
        raise G02ConfigError("displacement.energy_gate: 0 <= lo < hi <= 1")
    for sect in ("displacement", "truncation"):
        bad = [f for f in cfg[sect]["fields"] if f not in ("mag", "grav")]
        if bad or not cfg[sect]["fields"]:
            raise G02ConfigError(f"{sect}.fields: non-empty subset of [mag, grav]")
    tc = cfg["truncation"]
    if not 0 < float(tc["ridge_q"]) < 1:
        raise G02ConfigError("truncation.ridge_q in (0, 1)")
    nc = cfg["network"]
    if not nc["sources"]:
        raise G02ConfigError("network.sources: at least one evidence map")
    if not 0 < float(nc["hi_area_frac"]) < float(nc["lo_area_frac"]) < 1:
        raise G02ConfigError("network: 0 < hi_area_frac < lo_area_frac < 1")
    if not 0 <= float(nc["continuation_weight"]) < 1:
        raise G02ConfigError("network.continuation_weight in [0, 1): an isolated segment must keep a positive score")
    fc = cfg["fusion"]
    bad = [f for f in fc["families"] if f not in G02_FAMILIES]
    if bad:
        raise G02ConfigError(f"fusion.families: unknown {bad}")
    if any(not 0 < float(w) < 1 for w in fc["weights"]):
        raise G02ConfigError("fusion.weights in (0, 1)")
    if fc["budgets"] not in ("primary", "all"):
        raise G02ConfigError("fusion.budgets must be primary | all")
    dd = cfg["decision"]
    if dd["metric"] not in ("f1", "precision", "recall", "enrichment"):
        raise G02ConfigError("decision.metric must be f1 | precision | recall | enrichment")
    if not 0.5 <= float(dd["min_p_positive"]) < 1 or not 0 < float(dd["min_zone_win_share"]) <= 1:
        raise G02ConfigError("decision: min_p_positive in [0.5, 1), min_zone_win_share in (0, 1]")
    if dd["primary_task"] not in ev["tasks"]:
        raise G02ConfigError("decision.primary_task must be one of evaluation.tasks")
    if cfg["save"]["raster_dtype"] not in ("float16", "float32"):
        raise G02ConfigError("save.raster_dtype must be float16 | float32")
    lam = cfg["wavelengths_km"]
    if lam != "auto" and (not isinstance(lam, list) or not lam or any(float(x) <= 0 for x in lam)):
        raise G02ConfigError("wavelengths_km must be 'auto' or a non-empty list of positive wavelengths")


def fingerprint_config(cfg: dict) -> dict:
    """The part of the configuration that changes results (display / persistence switches excluded)."""
    c = copy.deepcopy(cfg)
    for k in ("overwrite", "figures", "save", "modes"):
        c.pop(k, None)
    c["g01"] = {"overrides": c.get("g01", {}).get("overrides")}
    return c


# =============================================================================== store
class G02Store(G01Store):
    SUBDIRS = ("config", "processed", "rasters", "rasters/units", "lineaments", "tables", "figures", "reports",
               "manifests")

    def mark_done(self, step: str, info: dict | None = None) -> None:
        from ...artifacts.atomic import atomic_write_json

        atomic_write_json(self.path(f"processed/done_{_safe(step)}.json"),
                          _jsonable({"step": step, "fingerprint": self.fp, "g02_version": G02_VERSION,
                                     "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "info": info or {}}))

    def flush_manifest(self, extra: dict | None = None) -> Path:
        from ...artifacts.atomic import atomic_write_json

        p = self.path("manifests/manifest.json")
        old = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        arts = {**(old.get("artifacts") or {}), **self.entries}
        body = {**old, **(extra or {}), "fingerprint": self.fp, "g02_version": G02_VERSION,
                "updated": time.strftime("%Y-%m-%dT%H:%M:%S"), "artifacts": arts}
        return atomic_write_json(p, _jsonable(body))

    def save_raster_stream(self, keys: list[str], get, rel: str, meta: dict, dtype: str = "float16",
                           sources=None) -> Path:
        """NPZ written one array at a time (``get(key)`` loads it): never every map in memory at once. Readable by
        ``np.load`` / ``load_raster`` exactly like ``save_raster``."""
        from ...artifacts.atomic import atomic_save

        if self._writable(rel):
            names = {_safe(k): k for k in keys}

            def w(t):
                with zipfile.ZipFile(t, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
                    for sk, k in names.items():
                        v = np.asarray(get(k))
                        v = v.astype(dtype) if np.issubdtype(v.dtype, np.floating) else v
                        with zf.open(f"{sk}.npy", "w", force_zip64=True) as f:
                            np.lib.format.write_array(f, np.ascontiguousarray(v), allow_pickle=False)
                        del v
                    m = np.array(json.dumps(_jsonable({**meta, "keys": names, "dtype": dtype})))
                    with zf.open("__meta__.npy", "w") as f:
                        np.lib.format.write_array(f, m, allow_pickle=False)
            atomic_save(self.path(rel), w)
        self.record(rel, "raster_npz", sources, {k: v for k, v in meta.items() if k != "refs"})
        return self.path(rel)

    def load_raster_keys(self, rel: str, keys: list[str] | None = None) -> tuple[dict, dict]:
        """Selected arrays of an NPZ (all when ``keys`` is None) without loading the others."""
        with np.load(self.path(rel), allow_pickle=False) as z:
            meta = json.loads(str(z["__meta__"]))
            names = meta.get("keys") or {}
            inv = {v: k for k, v in names.items()}
            want = keys if keys is not None else list(names.values())
            arrs = {k: np.asarray(z[inv.get(k, _safe(k))]) for k in want if inv.get(k, _safe(k)) in z.files}
        return arrs, meta

    def raster_keys(self, rel: str) -> list[str]:
        with np.load(self.path(rel), allow_pickle=False) as z:
            meta = json.loads(str(z["__meta__"]))
        return list((meta.get("keys") or {}).values())

    def save_vectors(self, features: list[dict], rel_stem: str, crs_epsg, props_table: pd.DataFrame | None = None,
                     sources=None, params=None) -> list[Path]:
        """``features``: [{"coords": [(x, y), ...], "properties": {...}}] in CRS units. GeoJSON always (with a
        ``crs`` member), GeoPackage when geopandas is available; ``props_table`` additionally as Parquet / CSV."""
        from ...artifacts.atomic import atomic_write_text

        out = []
        rel = f"{rel_stem}.geojson"
        fc = {"type": "FeatureCollection",
              "crs": {"type": "name", "properties": {"name": f"EPSG:{crs_epsg}"}} if crs_epsg else None,
              "features": [{"type": "Feature", "geometry": {"type": "LineString",
                                                             "coordinates": [[float(x), float(y)] for x, y in f["coords"]]},
                            "properties": _jsonable(f.get("properties") or {})} for f in features
                           if len(f["coords"]) >= 2]}
        if self._writable(rel):
            atomic_write_text(self.path(rel), json.dumps(fc))
        self.record(rel, "vector_geojson", sources, params)
        out.append(self.path(rel))
        try:
            import geopandas as gpd
            from shapely.geometry import LineString

            from ...artifacts.atomic import atomic_save

            rq = f"{rel_stem}.gpkg"
            if fc["features"] and self._writable(rq):
                g = gpd.GeoDataFrame([ft["properties"] for ft in fc["features"]],
                                     geometry=[LineString(ft["geometry"]["coordinates"]) for ft in fc["features"]],
                                     crs=f"EPSG:{crs_epsg}" if crs_epsg else None)
                atomic_save(self.path(rq), lambda t: g.to_file(t, driver="GPKG"))
                self.record(rq, "vector_gpkg", sources, params)
                out.append(self.path(rq))
        except ImportError:
            pass
        except Exception as e:                                     # GPKG is a convenience copy: GeoJSON is the record
            print(f"[g02] gpkg skipped for {rel_stem}: {type(e).__name__}: {e}")
        if props_table is not None:
            parent, name = rel_stem.rsplit("/", 1) if "/" in rel_stem else ("lineaments", rel_stem)
            self.save_table(props_table, name, parent, sources, params)
        return out


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


def set_latest(parent: Path, run_id: str, fp: str, mode: str) -> None:
    from ...artifacts.atomic import atomic_write_json

    atomic_write_json(Path(parent) / "LATEST.json", {"run_id": run_id, "fingerprint": fp, "g02_version": G02_VERSION,
                                                      "mode": mode, "updated": time.strftime("%Y-%m-%dT%H:%M:%S")})


def resolve_run(parent: Path, run_id: str | None = None) -> tuple[str, str]:
    """(run id, how it was resolved) of an attachable run: ``run_id`` when given, else ``LATEST.json``, else the newest
    complete run folder (``config/g02_config.json`` + ``config/inputs.json``) by manifest time.

    ``LATEST.json`` lives one level above the run folders and was never pushed to Drive by runs before g02.1-p6b, so
    a fresh runtime staged the run folder without it and ``attach`` refused an existing run."""
    parent = Path(parent)

    def complete(rid: str) -> bool:
        d = parent / rid
        return (d / "config" / "g02_config.json").exists() and (d / "config" / "inputs.json").exists()

    if run_id:
        if not complete(run_id):
            raise FileNotFoundError(f"G02 run {run_id} under {parent} is absent or incomplete (config/g02_config.json "
                                    "+ config/inputs.json required)")
        return run_id, "explicit run_id"
    rid = latest_run(parent)
    if rid and complete(rid):
        return rid, "LATEST.json"
    runs = list_runs(parent) if parent.exists() else pd.DataFrame()
    ok = runs[[complete(r) for r in runs.run_id]] if len(runs) else runs
    if not len(ok):
        raise FileNotFoundError(f"no complete G02 run under {parent} (known: {list(runs.run_id) if len(runs) else []}; "
                                f"LATEST.json {'names ' + repr(rid) if rid else 'absent'})")
    ok = ok.assign(_t=ok.updated.fillna("")).sort_values(["_t", "run_id"])
    return str(ok.run_id.iloc[-1]), f"newest complete run folder (LATEST.json {'stale' if rid else 'absent'})"


def list_runs(parent: Path) -> pd.DataFrame:
    rows = []
    for d in sorted(Path(parent).glob("g02_*")):
        man = d / "manifests" / "manifest.json"
        info = json.loads(man.read_text(encoding="utf-8")) if man.exists() else {}
        cfgp = d / "config" / "g02_config.json"
        mode = json.loads(cfgp.read_text(encoding="utf-8")).get("mode") if cfgp.exists() else None
        rows.append({"run_id": d.name, "mode": mode, "fingerprint": info.get("fingerprint"),
                     "updated": info.get("updated"), "n_artifacts": len(info.get("artifacts") or {}),
                     "steps_done": sorted(p.stem.replace("done_", "") for p in (d / "processed").glob("done_*.json"))})
    return pd.DataFrame(rows)
