"""The master-pipeline stages: each one wraps existing library functions (no re-implementation).

Graph (hard dependencies; ``~`` = soft, used when available)::

    bootstrap -> raw_data -> native_grid -+-> dem_inventory -> dem_pilot (needs feature_store)
                                          +-> geological_eda -> segment_attributes ~~+
                                          +-> spatial_folds -------------------------+-> feature_store
                                          +-> discovery_splits ----------------------+
    spatial_folds + discovery_splits (~feature_store, ~segment_attributes) -> baselines
    baselines -> discovery_validation (~advanced_models) -> ensemble
    feature_store + baselines -> advanced_models
    feature_store (~ensemble, ~advanced_models) -> full_inference -> submission
    bootstrap (~everything) -> reports

Why ``feature_store`` hard-needs folds and splits: every model-producing stage works on the Phase-5 research
context (``load_context5``), whose loader requires them. The Phase-1 EDA is needed only by the ``eda:*``
feature family (``FAMILY_EDA_GROUPS``) and by segment-stratified evaluation, so it is a soft dependency:
ordered first, but a failed EDA does not stop the raw / Phase-2 families or a G1 LightGBM baseline.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import validation as V
from ..artifacts.atomic import atomic_write_json, sha256_file
from . import baselines as B
from .context import PipelineStop, discovery_params, fold_params
from .registry import (ALWAYS_RUN, CPU, CPU_HEAVY, DELEGATED, GPU, LIGHT, NETWORK, NETWORK_HEAVY, OPTIONAL,
                       RAW_REQUIRED, RETRAIN, Stage, StageRegistry)

NATIVE_GRID_JSON = "environment/pipeline/native_grid.json"
EDA_STAGE_IN = ["outputs/eda/manifest.json", "outputs/eda/datasets", "outputs/eda/features/lineament_maps.tif",
                "outputs/eda/features/crossmodal_orientation.tif", "outputs/eda/REBUILD_PROVENANCE.json",
                "outputs/eda/INSTALL_STATE.json"]
SEG_RELS = ["outputs/benchmark/datasets/native_segment_ids.tif", "outputs/benchmark/datasets/segment_attributes.parquet",
            "outputs/benchmark/datasets/native_segments_provenance.json"]
# feature family -> Phase-1 EDA groups it is computed from (other families need only raw data + the grid)
FAMILY_EDA_GROUPS = {"phase2_eda": ("lineaments", "crossmodal")}
FEATURE_DIRS = {"phase2": "outputs/benchmark/cache/features", "phase4": "outputs/phase4/cache/features",
                "phase5": "data/feature_store/phase5"}
FEATURE_REGISTRY = "data/feature_store/pipeline_feature_registry.csv"
FEATURE_PROVENANCE = "data/feature_store/feature_provenance.json"
DEM_INV_DIR = "data/raw/dem_1m/inventory"


def _report(name: str, *checks) -> V.Report:
    r = V.Report(name)
    for c in checks:
        r.add(c)
    return r


def _sha(pc, rel: str) -> str | None:
    p = pc.local / rel
    if not p.exists():
        return None
    key = f"sha:{rel}:{p.stat().st_size}:{p.stat().st_mtime_ns}"
    if key not in pc._cache:
        pc._cache[key] = sha256_file(p)
    return pc._cache[key]


def folds_sha(pc) -> str | None:
    return _sha(pc, f"{pc.fold_rel}/fold_map.tif")


def splits_sha(pc) -> str | None:
    return _sha(pc, f"{pc.discovery_rel}/discovery_map.tif")


def _save_table(pc, df: pd.DataFrame, name: str) -> Path:
    p = pc.reports / f"{name}.csv"
    df.to_csv(p, index=False)
    return p


# =============================================================================== bootstrap
def bootstrap_run(pc) -> dict:
    from .._version import provenance
    from ..paths import WORKSPACE_DIRS

    dirs = list(WORKSPACE_DIRS) + ["outputs/eda", "outputs/benchmark", "outputs/phase3", "outputs/phase4",
                                   "reports/pipeline/figures"]
    for d in dirs:
        (pc.local / d).mkdir(parents=True, exist_ok=True)
        if pc.drive is not None:
            (pc.drive / d).mkdir(parents=True, exist_ok=True)
    atomic_write_json(pc.local / "environment/pipeline/config_snapshot.json", pc.cfg.public())
    from ..runtime.colab import environment_report

    env = environment_report()
    _save_table(pc, env, "environment_report")
    g = pc.gpu()
    return {**provenance(), "gpu": g.get("name"), "cuda": g.get("cuda"), "vram_gb": g.get("vram_gb"),
            "drive_mounted": pc.drive is not None}


def bootstrap_validate(pc, root) -> V.Report:
    r = V.Report("bootstrap")
    r.add(V.Check(str(pc.local), V.OK if pc.local.exists() else V.MISSING, "local workspace"))
    want = str(pc.cfg.get("paths.drive_root") or "")
    if want:
        ok = pc.drive is not None
        r.add(V.Check(want, V.OK if ok else V.INVALID,
                      "" if ok else "Google Drive is not mounted: nothing would persist after the session"))
    try:
        code = pc.code_root
        r.add(V.Check(str(code), V.OK if (code / "configs").is_dir() else V.MISSING, "code checkout with configs/"))
    except FileNotFoundError as e:
        r.add(V.Check("code_root", V.MISSING, str(e)))
    return r


# =============================================================================== raw data
def raw_run(pc) -> dict:
    from ..data.acquisition import Acquirer, canonicalize_dem_json

    urls = {**(pc.cfg.get("data.urls") or {}), **(pc.urls or {})}
    acq = Acquirer(pc.local, pc.drive, urls=urls, download_missing=bool(pc.cfg.get("data.download_missing")) and
                   pc.allow_download, use_secrets=bool(pc.cfg.get("data.use_colab_secrets")),
                   verify=pc.cfg.get("data.verify", "sha256"))
    df = acq.acquire_all()
    _save_table(pc, df, "raw_data_acquisition")
    dem = canonicalize_dem_json(pc.local, pc.drive)
    miss = df[(df.required) & (df.status != "ok")]
    if len(miss):
        lines = "; ".join(f"{r.key}: {r.detail}" for r in miss.itertuples())
        raise PipelineStop(f"required raw files unavailable - {lines}",
                           "Paste fresh DrivenData links (they expire) into Colab Secrets GEMS_URL_FEATURES / "
                           "GEMS_URL_FAULTS / GEMS_URL_SUBMISSION (key icon, enable notebook access) or the "
                           "notebook's URL cell, or upload the files to MyDrive/GEMS_2/data/raw/ - then re-run.")
    pc.forget("data", "ctx5")
    for k in list(pc._cache):
        if k.startswith("grid:"):
            pc._cache.pop(k)
    return {"files": {r.key: r.action for r in df.itertuples()}, "dem_json": str(dem) if dem else None}


def raw_validate(pc, root) -> V.Report:
    from ..data.acquisition import CORE_FILES, Acquirer

    acq = Acquirer(pc.local, pc.drive, download_missing=False, use_secrets=False)
    r = V.Report("raw_data")
    for f in CORE_FILES:
        if not f.required:
            continue
        hit = acq.find(f, root)
        r.add(hit[1] if hit else V.Check(str(root / f.rel), V.MISSING, "not found (canonical or alternate path)"))
    return r


# =============================================================================== native grid
def native_run(pc) -> dict:
    d = pc.data
    s = dict(zip(d.summary().key, d.summary().value))
    if abs(d.grid.pixel_km - 0.1) > 1e-3:
        raise PipelineStop(f"unexpected pixel size {d.grid.pixel_km * 1000:.1f} m (expected 100 m)",
                           "check that data/raw/geodawn holds the competition GeoDAWN release")
    if not d.template_matches():
        raise PipelineStop("example_submission.tif is not on the features grid",
                           "re-acquire the submission template from the competition page")
    if int(d.labels.sum()) == 0:
        raise PipelineStop("existing_faults.tif has no pixel >= 1 on the features grid", "re-acquire the labels file")
    rec = {"crs": str(d.grid.crs), "transform": list(d.grid.transform)[:6], "height": d.grid.height,
           "width": d.grid.width, "pixel_km": [d.grid.pixel_km_x, d.grid.pixel_km_y], "bands": d.band_names,
           "label_px": int(d.labels.sum()), "valid_frac": float(d.valid.mean()),
           "labels_outside_valid": int((d.labels & ~d.valid).sum()), "template_same_grid": True,
           "raw": {k: V.fingerprint(pc.raw_path(k), sha=(k != "features")) for k in ("features", "faults", "submission")}}
    rec["raw"]["features"]["sha256"] = _sha(pc, Path(pc.raw_path("features")).relative_to(pc.local).as_posix())
    atomic_write_json(pc.local / NATIVE_GRID_JSON, rec)
    _save_table(pc, d.band_table(), "native_band_table")
    return {k: s.get(k) for k in ("crs", "height", "width", "label_px", "valid_frac", "template_same_grid")}


def native_validate(pc, root) -> V.Report:
    p = root / NATIVE_GRID_JSON
    c = V.check_json(p, required_keys=("crs", "height", "width", "raw"))
    if c.ok:
        rec = V.read_json(p, {})
        for k in ("features", "faults", "submission"):
            cur = V.fingerprint(pc.raw_path(k, root))
            old = (rec.get("raw") or {}).get(k) or {}
            if cur is None or cur.get("bytes") != old.get("bytes"):
                c.status, c.reason = V.INCOMPATIBLE, f"raw file {k} changed since the grid was recorded (new release?)"
                break
    return _report("native_grid", c)


def native_fp(pc) -> dict:
    """Raw file sizes; read from Drive when not yet staged locally (a read-only plan on a fresh VM)."""
    out = {}
    for k in ("features", "faults", "submission"):
        p = pc.raw_path(k)
        if not p.exists() and pc.drive is not None:
            p = pc.raw_path(k, pc.drive)
        out[k] = (V.fingerprint(p) or {}).get("bytes")
    return out


# =============================================================================== Phase-1 EDA (minimal)
def eda_run(pc) -> dict:
    from .eda_rebuild import adopt_manifest, rebuild_minimal, rebuild_sources

    max_dim, seed = pc.cfg.get("eda.max_dim"), int(pc.cfg.get("eda.seed"))
    if pc.drive is not None:                  # a finished rebuild / interrupted install from an earlier session
        pc.stage_in(rebuild_sources(pc.drive, max_dim, seed))
    rep = rebuild_minimal(pc.local, max_dim=max_dim, seed=seed)
    adopt_manifest(pc.local)
    if rep["blocked"]:
        raise PipelineStop(f"historical EDA groups are incomplete / invalid: {rep['blocked']}"
                           + (f" ({rep['blocked_detail']})" if rep.get("blocked_detail") else ""),
                           "the pipeline never mixes or overwrites historical files: restore the complete group from "
                           "the local project (docs/ARTIFACT_RECOVERY.md, EDA) or move the partial files out of "
                           "MyDrive/GEMS_2/outputs/eda/ yourself, then re-run")
    return {k: v for k, v in rep.items() if k != "states_before"}


def eda_validate(pc, root) -> V.Report:
    return V.validate_eda_minimal(root / "outputs" / "eda")


def eda_outputs(pc) -> list[str]:
    out = list(EDA_STAGE_IN)
    if pc.cfg.get("reports.save_intermediate"):
        out.append("outputs/eda_rebuild")
    return out


def eda_fp(pc) -> dict:
    return {"eda": pc.cfg.get("eda"), "raw": native_fp(pc)}


# =============================================================================== native segments
def seg_run(pc) -> dict:
    from .eda_rebuild import build_native_segments

    written = build_native_segments(pc.data, pc.local)
    pc.forget("ctx5")                          # a cached research context was loaded without these segments
    return {"written": written}


def seg_validate(pc, root) -> V.Report:
    return V.validate_native_segments(root / "outputs" / "benchmark", grid=pc.grid(root),
                                      eda_dir=root / "outputs" / "eda")


# =============================================================================== folds / splits
def folds_run(pc) -> dict:
    from ..folds import fold_summary, make_geo_folds

    d = pc.local / pc.fold_rel
    present = [f for f in V.FOLD_FILES if (d / f).exists()]
    if present:
        raise PipelineStop(f"fold folder {pc.fold_rel} is incomplete or inconsistent ({present} present)",
                           "historical folds are never overwritten: restore the missing files from the local project, "
                           "or move the folder aside on Drive to regenerate; a changed protocol gets a new folder name")
    v, data = pc.cfg.get("validation"), pc.data
    f = make_geo_folds(data.valid, data.labels, data.grid, n_folds=int(v["n_folds"]), block_km=float(v["block_km"]),
                       clusters_per_fold=int(v["clusters_per_fold"]), buffer_km=float(v["buffer_km"]),
                       inner_buffer_km=float(v["inner_buffer_km"]), seed=pc.seed)
    f.save(d)
    fold_summary(f, data.labels, data.valid).to_csv(d / "fold_summary.csv", index=False)
    pc.forget("ctx5")
    return {"dir": pc.fold_rel, "n_folds": f.n_folds}


def folds_validate(pc, root) -> V.Report:
    return V.validate_folds_dir(root / pc.fold_rel, grid=pc.grid(root),
                                expected=fold_params(pc.cfg.get("validation"), pc.seed))


def splits_run(pc) -> dict:
    from ..folds import discovery_summary, make_discovery_splits

    d = pc.local / pc.discovery_rel
    present = [f for f in V.DISCOVERY_FILES if (d / f).exists()]
    if present:
        raise PipelineStop(f"discovery folder {pc.discovery_rel} is incomplete or inconsistent ({present} present)",
                           "restore the missing files from the local project, or move the folder aside on Drive")
    v, data = pc.cfg.get("validation"), pc.data
    s = make_discovery_splits(data.labels, data.valid, data.grid, n_splits=int(v["n_discovery"]),
                              link_km=float(v["link_km"]), frac_km=float(v["frac_km"]),
                              min_system_km=float(v["min_system_km"]), max_system_frac=float(v["max_system_frac"]),
                              ignore_km=float(v["ignore_km"]), eval_km=float(v["eval_km"]), seed=pc.seed)
    s.save(d)
    discovery_summary(s, data.labels).to_csv(d / "discovery_summary.csv", index=False)
    pc.forget("ctx5")
    return {"dir": pc.discovery_rel, "n_splits": s.n_splits}


def splits_validate(pc, root) -> V.Report:
    return V.validate_discovery_dir(root / pc.discovery_rel, grid=pc.grid(root),
                                    expected=discovery_params(pc.cfg.get("validation"), pc.seed))


# =============================================================================== feature store
def _family(name: str) -> str:
    if name.startswith("raw:"):
        return "raw"
    if name.startswith("eda:"):
        return "phase2_eda"
    if name.startswith("rs:"):
        return "phase4_rs"
    if name.startswith("p5:"):
        return "phase5_profiles"
    if name.startswith("dem:"):
        return "dem"
    return "phase2"


def spec_hash(spec) -> str:
    from dataclasses import asdict

    return hashlib.sha256(json.dumps(asdict(spec), sort_keys=True, default=str).encode()).hexdigest()[:12]


def feature_catalogue(pc, root: Path) -> pd.DataFrame:
    """Every declarable feature (no data read): name, family, cache path, spec metadata. Header reads only."""
    import rasterio

    from gems_eda.io import _band_names

    from .. import rs_features
    from ..features import _fname, build_specs
    from ..features.phase5_store import dem_specs, load_profile_specs

    with rasterio.open(pc.raw_path("features", root)) as src:
        bands = _band_names(src)[0]
    rs_cfg = rs_features.load_rs_config(pc.config_file("phase4", "rs_features.yaml"))
    specs = (build_specs(bands, include_eda=True) + rs_features.build_rs_specs(bands, rs_cfg)
             + load_profile_specs(bands) + dem_specs(root / "data" / "processed" / "dem_1m_100m"))
    rows = []
    for s in specs:
        fam = _family(s.name)
        base = FEATURE_DIRS["phase5"] if fam in ("phase5_profiles", "dem") else \
            FEATURE_DIRS["phase4"] if fam == "phase4_rs" else FEATURE_DIRS["phase2"]
        rows.append({"name": s.name, "family": fam, "rel": f"{base}/{_fname(s.name)}", "modality": s.modality,
                     "source": s.source, "transform": s.transform, "scale_km": s.scale_km,
                     "support_km": s.support_km, "kind": s.kind, "origin": s.origin, "units": s.units,
                     "standardize": bool(getattr(s, "standardize", True)), "note": s.note, "spec_hash": spec_hash(s)})
    return pd.DataFrame(rows).drop_duplicates("name")


def _snapshot_names() -> set[str]:
    out = set()
    for n in B.RECIPES:
        s = B.legacy_spec(n) or {}
        out |= set(s.get("features") or []) | set(s.get("channels") or [])
        for v in (s.get("groups") or {}).values():
            out |= set(v)
    return out


def required_features(pc, cat: pd.DataFrame) -> list[str]:
    fams = set(pc.cfg.get("features.families"))
    if not pc.cfg.get("stages.run_advanced_features"):
        fams -= {"phase4_rs_used", "phase4_rs_all", "phase5_profiles"}
    want = set(pc.cfg.get("features.extra") or [])
    used = _snapshot_names()
    for r in cat.itertuples():
        f = r.family
        if f in fams or (f == "phase4_rs" and ("phase4_rs_all" in fams or ("phase4_rs_used" in fams and r.name in used))):
            want.add(r.name)
    return [n for n in cat.name if n in want]


def features_validate(pc, root) -> V.Report:
    g = pc.grid(root)
    rep = V.Report("feature_store")
    if g is None:
        rep.add(V.Check("features.tif", V.MISSING, "raw features unavailable"))
        return rep
    cat = feature_catalogue(pc, root)
    names = set(required_features(pc, cat))
    prov = V.read_json(root / FEATURE_PROVENANCE, {}) or {}
    lazy = _lazy_features(pc)
    for r in cat[cat.name.isin(names)].itertuples():
        f = root / r.rel
        if lazy and not f.exists() and (pc.drive / r.rel).exists():
            f = pc.drive / r.rel                     # header-only check of the Drive copy (staged on first read)
        c = rep.add(V.check_npy(f, shape=(g["height"], g["width"])))
        old = (prov.get(r.name) or {}).get("spec_hash")
        if c.ok and old and old != r.spec_hash:
            c.status, c.reason = V.INCOMPATIBLE, "cached under another feature definition (spec hash differs)"
    return rep


def _lazy_features(pc) -> bool:
    return pc.drive is not None and pc.cfg.get("features.stage", "all") == "lazy"


def features_stage_in(pc) -> list[str]:
    root = pc.drive or pc.local
    if _lazy_features(pc):                           # registries only; arrays staged when a run reads them
        return [f"{FEATURE_DIRS['phase2']}/feature_registry.csv", f"{FEATURE_DIRS['phase4']}/rs_feature_registry.csv",
                f"{FEATURE_DIRS['phase5']}/p5_feature_registry.csv", FEATURE_REGISTRY, FEATURE_PROVENANCE]
    try:
        cat = feature_catalogue(pc, root if pc.raw_path("features", root).exists() else pc.local)
    except Exception:
        return []
    rels = list(cat[cat.name.isin(required_features(pc, cat))].rel)
    regs = [f"{FEATURE_DIRS['phase2']}/feature_registry.csv", f"{FEATURE_DIRS['phase4']}/rs_feature_registry.csv",
            f"{FEATURE_DIRS['phase5']}/p5_feature_registry.csv", FEATURE_REGISTRY, FEATURE_PROVENANCE]
    return rels + regs


def eda_groups_ready(root: Path, groups) -> list[str]:
    """EDA groups (of ``groups``) that are NOT valid under ``root/outputs/eda``."""
    return [g for g in groups if not V.validate_eda_group(Path(root) / "outputs" / "eda", g).ok]


def features_run(pc) -> dict:
    ctx = pc.ctx5()
    cat = feature_catalogue(pc, pc.local)
    names = required_features(pc, cat)
    built, deferred = {}, {}
    for fam in ("raw", "phase2", "phase2_eda", "phase4_rs", "phase5_profiles", "dem"):
        todo = [n for n in names if _family(n) == fam and not ctx.store.has(n)]
        if not todo:
            continue
        missing = eda_groups_ready(pc.local, FAMILY_EDA_GROUPS.get(fam, ()))
        if missing:                            # build every other family first; report this one precisely
            deferred[fam] = {"n": len(todo), "missing_eda_groups": missing}
            print(f"[features] {fam}: {len(todo)} feature(s) deferred - Phase-1 EDA groups {missing} not available")
            continue
        t0 = time.perf_counter()
        print(f"[features] {fam}: building {len(todo)} feature(s)")
        ctx.store.build(todo)
        built[fam] = {"n": len(todo), "minutes": round((time.perf_counter() - t0) / 60, 1)}
        pc.push([r for r in cat[cat.name.isin(todo)].rel])          # persist each family as soon as it exists
    write_feature_registry(pc, cat, names)
    if deferred:
        raise PipelineStop(f"feature families {sorted(deferred)} need Phase-1 EDA rasters that are missing: {deferred}; "
                           f"built and kept: {sorted(built) or 'nothing new'}",
                           "run geological_eda (pipe.step('geological_eda')), then re-run feature_store (only the "
                           "deferred features are computed). LightGBM baselines without eda:* features can already train.")
    return {"required": len(names), "built": built}


def write_feature_registry(pc, cat: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Declared metadata of every required feature + provenance (spec hash, raw input identity)."""
    g = pc.grid(pc.local) or {}
    raw = (V.read_json(pc.local / NATIVE_GRID_JSON, {}) or {}).get("raw", {}).get("features", {})
    prov = V.read_json(pc.local / FEATURE_PROVENANCE, {}) or {}
    rows = []
    for r in cat[cat.name.isin(names)].itertuples():
        p = pc.local / r.rel
        cached = p.exists()
        if cached and r.name not in prov:
            prov[r.name] = {"spec_hash": r.spec_hash, "raw_features_sha256": raw.get("sha256"),
                            "recorded": time.strftime("%Y-%m-%dT%H:%M:%S"), "path": r.rel}
        rows.append({**r._asdict(), "cached": cached, "bytes": p.stat().st_size if cached else None,
                     "crs": str(g.get("crs")), "grid": f"{g.get('height')}x{g.get('width')}",
                     "nodata": "NaN outside the valid footprint",
                     "standardization": "robust z (median / 1.4826 MAD, signed-log for skewed fields)"
                     if r.standardize else "none (bounded / count / fraction)",
                     "storage": "float32 .npy (memory-mapped reads)"})
    df = pd.DataFrame(rows).drop(columns=["Index"], errors="ignore")
    p = pc.local / FEATURE_REGISTRY
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    atomic_write_json(pc.local / FEATURE_PROVENANCE, prov)
    return df


def features_outputs(pc) -> list[str]:
    """Registries + the required feature files (pushed even after an interruption; unchanged files are skipped)."""
    try:
        cat = feature_catalogue(pc, pc.local)
        files = list(cat[cat.name.isin(required_features(pc, cat))].rel)
    except Exception:                                  # raw data not staged yet
        files = []
    return files + [FEATURE_REGISTRY, FEATURE_PROVENANCE, f"{FEATURE_DIRS['phase2']}/feature_registry.csv",
            f"{FEATURE_DIRS['phase2']}/feature_build_log.csv", f"{FEATURE_DIRS['phase4']}/rs_feature_registry.csv",
            f"{FEATURE_DIRS['phase4']}/rs_feature_build_log.csv", f"{FEATURE_DIRS['phase5']}/p5_feature_registry.csv",
            f"{FEATURE_DIRS['phase5']}/p5_feature_build_log.csv"]


def features_fp(pc) -> dict:
    return {"families": sorted(pc.cfg.get("features.families")), "extra": sorted(pc.cfg.get("features.extra") or []),
            "advanced": bool(pc.cfg.get("stages.run_advanced_features"))}


# =============================================================================== DEM inventory / pilot
def _dem_json(pc, root: Path) -> Path | None:
    from ..data.acquisition import BY_KEY

    f = BY_KEY["dem_json"]
    for rel in (f.rel, *f.alternates):
        if (root / rel).exists():
            return root / rel
    return None


def dem_inv_enabled(pc) -> tuple[bool, str]:
    if not pc.cfg.get("dem.inventory"):
        return False, "dem.inventory is false"
    if _dem_json(pc, pc.local) is None and (pc.drive is None or _dem_json(pc, pc.drive) is None):
        return False, "no USGS DEM URL list (data/raw/dem_1m/tnm_urls.json or data/raw/tnm_items.json)"
    return True, ""


def dem_inv_run(pc) -> dict:
    from types import SimpleNamespace

    from ..data.acquisition import canonicalize_dem_json
    from ..dem import inventory as I

    js = canonicalize_dem_json(pc.local, pc.drive) or _dem_json(pc, pc.local)
    if js is None:
        raise PipelineStop("DEM URL list not found locally", "place tnm_urls.json under MyDrive/GEMS_2/data/raw/dem_1m/")
    inv = I.load_inventory(js)
    inv = I.study_area_intersection(inv, SimpleNamespace(data=pc.data),
                                    float(pc.cfg.get("dem.min_valid_frac", 0.05) or 0.05))
    out = pc.local / DEM_INV_DIR
    out.mkdir(parents=True, exist_ok=True)
    inv.to_csv(out / "dem_inventory.csv", index=False)
    try:
        inv.to_parquet(out / "dem_inventory.parquet", index=False)
    except Exception as e:                                           # mixed-type columns: CSV stays authoritative
        print(f"[dem] parquet export skipped: {e}")
    est = I.storage_estimate(inv)
    sel = I.choose_per_tile(inv[inv.in_study_area], "latest") if "in_study_area" in inv else inv.iloc[:0]
    est.update(n_urls=int(len(inv)), n_duplicate_urls=int(inv.get("duplicate_url", pd.Series(dtype=bool)).sum()),
               n_unparsed=int((~inv.parse_ok.fillna(False).astype(bool)).sum()), n_selected_one_per_tile=int(len(sel)),
               pilot_tiles=int(pc.cfg.get("dem.pilot_tiles")))
    atomic_write_json(out / "dem_storage_estimate.json", est)
    if "n_projects_for_tile" in inv:                   # same tile in several acquisition projects = overlap
        ov = inv[inv.n_projects_for_tile.fillna(1) > 1]
        ov.sort_values(["tile_id", "year"]).to_csv(out / "dem_overlaps.csv", index=False)
        est["n_tiles_with_overlapping_projects"] = int(ov.tile_id.nunique())
        atomic_write_json(out / "dem_storage_estimate.json", est)
    _footprints(inv, out / "dem_footprints.gpkg")
    _footprint_figure(pc, inv, out / "dem_footprints.png")
    return {"urls": len(inv), "in_study_area": int(inv.get("in_study_area", pd.Series(dtype=bool)).sum()),
            "gib_estimate_all_intersecting": round(est["gib_estimate"], 1)}


def _footprints(inv: pd.DataFrame, path: Path) -> None:
    """Nominal tile footprints (from tile names) as a GeoPackage, one layer per UTM zone (NAD83 / UTM)."""
    try:
        import geopandas as gpd
        from shapely.geometry import box
    except ImportError:
        print("[dem] geopandas missing: footprint GeoPackage skipped")
        return
    ok = inv[inv.parse_ok.fillna(False).astype(bool)] if "parse_ok" in inv else inv.iloc[:0]
    for zone, g in ok.groupby("zone"):
        gdf = gpd.GeoDataFrame(g[["tile_id", "project", "file", "year", "in_study_area", "valid_frac", "label_px"]
                                 if "in_study_area" in g else ["tile_id", "project", "file", "year"]].copy(),
                               geometry=[box(r.west, r.south, r.east, r.north) for r in g.itertuples()],
                               crs=f"EPSG:{26900 + int(zone)}")
        gdf.to_file(path, layer=f"utm{int(zone)}", driver="GPKG")


def _footprint_figure(pc, inv: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    d = pc.data
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(np.where(d.valid, 1.0, np.nan)[::8, ::8], extent=d.grid.extent, cmap="Greys", vmin=0, vmax=2)
    sub = inv[(inv.get("zone") == 11) & inv.parse_ok.fillna(False).astype(bool)] if "zone" in inv else inv.iloc[:0]
    for r in sub.drop_duplicates("tile_id").itertuples():
        ax.add_patch(Rectangle((r.west, r.south), 10_000, 10_000, fill=False, lw=0.4,
                               ec="#2a78d6" if getattr(r, "in_study_area", False) else "#bbbbbb"))
    ax.set_title(f"USGS 1 m DEM tiles (blue = intersects data footprint): {sub.tile_id.nunique()} zone-11 tiles")
    ax.set_aspect("equal")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def dem_inv_validate(pc, root) -> V.Report:
    d = root / DEM_INV_DIR
    return _report("dem_inventory", V.check_csv(d / "dem_inventory.csv", columns=("url",)),
                   V.check_json(d / "dem_storage_estimate.json", required_keys=("n_tiles", "gib_estimate")))


def dem_inv_fp(pc) -> dict:
    js = _dem_json(pc, pc.local) or (_dem_json(pc, pc.drive) if pc.drive else None)
    return {"json_bytes": (V.fingerprint(js) or {}).get("bytes") if js else None}


def _dem_cfg(pc):
    from .experiments import experiment_config

    d = pc.cfg.get("dem")
    over = {}
    base = experiment_config(pc, d["experiment"])
    want = {"n_tiles": int(d["pilot_tiles"]), "max_tiles": int(d["max_tiles"]), "max_gib": float(d["max_gib"]),
            "inspect_headers": int(d["inspect_headers"])}
    for k, v in want.items():
        if base.get(f"dem.{k}") != v:
            over.setdefault("dem", {})[k] = v              # semantic: a changed pilot is a new experiment id
    over["runtime"] = {"dem_download": bool(d["download"])}   # non-semantic: approving keeps the id
    return experiment_config(pc, d["experiment"], over)


def dem_pilot_enabled(pc) -> tuple[bool, str]:
    d = pc.cfg.get("dem")
    if not (pc.cfg.get("stages.run_dem_pilot") and d.get("enabled")):
        return False, "DEM pilot disabled (stages.run_dem_pilot and dem.enabled)"
    if d.get("mode") != "pilot":
        return False, "dem.mode is inventory_only"
    return True, ""


def dem_pilot_run(pc) -> dict:
    from .experiments import open_experiment

    d = pc.cfg.get("dem")
    exp = open_experiment(pc, _dem_cfg(pc))
    exp.prepare()
    est = V.read_json(exp.rd.reports / "dem_storage_estimate.json", {})
    sel = exp.rd.load_table("dem_pilot_tiles")
    print(f"[dem] pilot: {len(sel)} tiles selected (cap {d['max_tiles']} tiles / {d['max_gib']} GiB); all intersecting "
          f"tiles would be {est.get('gib_estimate', float('nan')):.1f} GiB - only the pilot is ever downloaded")
    if not d.get("download"):
        return {"experiment": exp.id, "selected": len(sel), "downloaded": False,
                "next": "review dem_pilot_tiles.csv, then set dem.download=True to approve the download"}
    if d.get("compute_morphology") and d.get("aggregate_to_100m"):
        exp.train()
        exp.evaluate()
        exp.export()
    else:
        from ..dem.download import download_selected

        log = download_selected(sel, pc.local, max_tiles=int(d["max_tiles"]), max_gib=float(d["max_gib"]), sync=pc.sync)
        exp.rd.save_table(log, "dem_download_log")
        exp.persist()
    pc.forget("ctx5")                                          # dem:* specs are read when the store is created
    return {"experiment": exp.id, "selected": len(sel), "downloaded": True}


def dem_pilot_validate(pc, root) -> V.Report:
    from .experiments import experiment_state

    d = pc.cfg.get("dem")
    st = experiment_state(root, _dem_cfg(pc))
    r = V.Report("dem_pilot")
    need = ("prepare", "train", "evaluate") if (d.get("download") and d.get("compute_morphology")) else ("prepare",)
    for s in need:
        r.add(V.Check(f"experiments/{st['experiment_id']}:{s}", V.OK if st[s] == "done" else
                      (V.MISSING if st[s] == "pending" else V.INVALID), st[s]))
    if d.get("download") and d.get("compute_morphology"):
        r.add(V.check_geotiff(root / "data/processed/dem_1m_100m/dem_features_100m.tif", grid=pc.grid(root),
                              bands=("coverage",)))
    return r


def dem_pilot_rels(pc) -> list[str]:
    try:
        return [f"experiments/{_dem_cfg(pc).experiment_id()}", "data/processed/dem_1m_100m"]
    except Exception:
        return []


# =============================================================================== baselines
def baselines_run(pc) -> dict:
    fs, ss = folds_sha(pc), splits_sha(pc)
    n = pc.bench_cfg.n_discovery_folds
    before = B.status_table(pc.local, grid=pc.grid(), n_splits=n, folds_sha=fs, splits_sha=ss)
    req = list(pc.cfg.get("stages.train_baselines") or [])
    gpu_blocked = gpu_blocked_baselines(pc, req)
    if gpu_blocked:
        print(f"[baselines] not attempted on this runtime: {gpu_blocked} ({pc.cuda_ready()[1]}); "
              "CPU baselines continue")
    done, failed = [], {}
    for name in req:
        row = before.set_index("model").loc[name]
        if row.status == B.AVAILABLE or name in gpu_blocked:
            continue
        try:
            res = B.train_baseline(pc, name, folds_sha=fs, splits_sha=ss)
            pc.push(res.get("rels", []))
            done.append(name)
        except PipelineStop as e:                      # e.g. a CNN without GPU: keep going with the others
            failed[name] = f"{e} -> {e.recovery}"
            print(f"[baselines] {name}: {e}")
    after = B.status_table(pc.local, grid=pc.grid(), n_splits=n, folds_sha=fs, splits_sha=ss)
    after["requested"] = after.model.isin(req)
    _save_table(pc, after, "baseline_status")
    if failed:
        raise PipelineStop(f"baselines not trained: {failed}", "see each message; other baselines were kept")
    return {"trained": done, "gpu_blocked": gpu_blocked, "status": dict(zip(after.model, after.status))}


def gpu_blocked_baselines(pc, names) -> list[str]:
    """Requested GPU baselines that cannot run here (no CUDA): reported, never attempted, never faked."""
    if pc.cuda_ready()[0]:
        return []
    return [n for n in names if n in B.RECIPES and B.RECIPES[n].gpu]


def baselines_validate(pc, root) -> V.Report:
    n = int(pc.cfg.get("validation.n_discovery"))
    r = V.Report("baselines")
    req = set(pc.cfg.get("stages.train_baselines") or [])
    blocked = set(gpu_blocked_baselines(pc, req))
    tab = B.status_table(root, grid=pc.grid(root), n_splits=n)
    for row in tab.itertuples():
        if row.model in blocked and row.status != B.AVAILABLE:
            # an environment limit, not a missing CPU result: noted, excluded from this runtime's checks;
            # the stage re-runs (always_run) in a GPU session and trains it then
            r.notes.append(f"{row.model}: requested, needs a CUDA GPU - not trained on this runtime")
        elif row.model in req:
            r.add(V.Check(row.cv_dir, V.OK if row.status == B.AVAILABLE else V.MISSING, f"{row.model}: {row.status}"))
    r.add(V.check_csv(root / "reports/pipeline/baseline_status.csv", columns=("model", "status")))
    return r


def baselines_rels(pc) -> list[str]:
    return B.all_rels(int(pc.cfg.get("validation.n_discovery"))) + ["reports/pipeline/baseline_status.csv"]


# =============================================================================== advanced research
def _advanced(pc) -> list[str]:
    return list(pc.cfg.get("experiments.advanced") or [])


def advanced_enabled(pc) -> tuple[bool, str]:
    if not pc.cfg.get("stages.run_advanced_models"):
        return False, "run_advanced_models is false (GPU training is opt-in)"
    if not _advanced(pc):
        return False, "experiments.advanced is empty"
    if not pc.cfg.get("gpu.use_gpu"):
        return False, "use_gpu is false"
    return pc.cuda_ready()


def advanced_run(pc) -> dict:
    """Reuse-first: inventory -> plan -> train only RUN / RESUME items at the session profile -> screening report.
    A completed compatible FULL run is never retrained for a cheaper profile; a failure in one experiment never
    re-runs another's finished units."""
    from ..training.trainer import TrainingInterrupted
    from .experiments import plan_experiments, run_experiment
    from .screening import export_screening_report

    plan = plan_experiments(pc)
    _save_table(pc, plan, "experiment_plan")
    print("[advanced] plan (" + str(pc.cfg.get("experiments.profile")) + "): " +
          ", ".join(f"{r.experiment}={r.action}" for r in plan.itertuples()))
    todo = list(plan[plan.action.isin(["RUN", "RESUME"])].experiment)
    rows = [{"experiment": r.experiment, "experiment_id": r.experiment_id if isinstance(r.experiment_id, str) else "",
             "profile": r.profile, "action": r.action, "reason": r.reason}
            for r in plan.itertuples() if r.experiment not in todo]
    failed = {r.experiment: r.reason for r in plan.itertuples() if r.action == "CONFIG_ERROR"}
    for stem in todo:
        try:
            rows.append(run_experiment(pc, stem))
        except TrainingInterrupted:
            raise
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            failed[stem] = msg + (f" -> {e.recovery}" if isinstance(e, PipelineStop) else "")
            rows.append({"experiment": stem, "error": msg})
            print(f"[advanced] {stem} failed: {msg}")
            if pc.cfg.get("experiments.stop_on_failure"):
                break
        finally:
            _save_table(pc, pd.DataFrame(rows), "experiment_status")
    try:
        export_screening_report(pc)                  # read-only consolidation of every saved result
    except Exception as e:                           # reporting never fails the training stage
        print(f"[advanced] screening report skipped: {type(e).__name__}: {e}")
    if failed:
        raise PipelineStop(f"experiments not completed: {list(failed)}", "; ".join(f"{k}: {v}" for k, v in failed.items()))
    return {"completed": [r["experiment"] for r in rows], "trained": todo}


def advanced_validate(pc, root) -> V.Report:
    from .experiments import experiment_config, experiment_state, plan_experiments

    r = V.Report("advanced_models")
    try:
        reused = {p.experiment: p for p in plan_experiments(pc).itertuples()        # Drive inventory
                  if p.action in ("REUSE_FULL", "REUSE_HIGHER", "NOT_NEEDED")}
    except Exception:
        reused = {}
    for stem in _advanced(pc):
        if stem in reused:
            p = reused[stem]
            rid = getattr(p, "reuse_id", None)
            rid = rid if isinstance(rid, str) and rid else p.experiment_id
            r.add(V.Check(f"experiments/{rid}", V.OK, f"{p.action}: {p.reason}"))
            continue
        try:
            st = experiment_state(root, experiment_config(pc, stem))
        except FileNotFoundError as e:
            r.add(V.Check(stem, V.INVALID, str(e)))
            continue
        started = any(st[s] != "pending" for s in ("prepare", "train", "evaluate", "export"))
        r.add(V.Check(f"experiments/{st['experiment_id']}", V.OK if st["complete"] else
                      (V.INVALID if started and not st["complete"] and "failed" in st.values() else V.MISSING),
                      ", ".join(f"{s}={st[s]}" for s in ("prepare", "train", "evaluate", "export"))))
    return r


def advanced_rels(pc) -> list[str]:
    from .experiments import experiment_config

    out = []
    for stem in _advanced(pc):
        try:
            out.append(f"experiments/{experiment_config(pc, stem).experiment_id()}")
        except FileNotFoundError:
            pass
    return out


def advanced_fp(pc) -> dict:
    p = str(pc.cfg.get("experiments.profile") or "FULL").upper()
    # FULL keeps the historical fingerprint (records made before fidelity existed stay consistent)
    return {"experiments": _advanced(pc)} if p == "FULL" else {"experiments": _advanced(pc), "profile": p}


# =============================================================================== scoreboard / ensemble
def available_models(pc, root: Path, need_discovery: bool = False) -> list[str]:
    """Models with maps under ``root`` (ctx-free): baselines by status, Phase-5 models from experiment manifests."""
    from ..phase5_context import scan_experiment_maps

    n = int(pc.cfg.get("validation.n_discovery"))
    tab = B.status_table(root, grid=pc.grid(root), n_splits=n)
    ok = {B.AVAILABLE} if need_discovery else {B.AVAILABLE, B.CV_ONLY}
    names = [m for m in pc.cfg.get("experiments.scoreboard_models") if m in set(tab.model[tab.status.isin(ok)])]
    for name, z in scan_experiment_maps(root / "experiments").items():
        if z.get("oof") and Path(z["oof"]).exists() and (not need_discovery or len(z.get("disc", {})) == n):
            names.append(name)
    return list(dict.fromkeys(names))


def _scoreboard_cfg(pc, root: Path):
    from .experiments import experiment_config

    models = available_models(pc, root)
    if not models:
        return None
    ctrl = "DL00_CNN_B_dti" if "DL00_CNN_B_dti" in models else None
    cands = [m for m in models if m == "P4_B_gated"] if ctrl else []
    return experiment_config(pc, "a1_zoo_discovery", {"name": "pipeline_scoreboard", "complete_discovery": [],
                                                      "models": models,
                                                      "evaluation": {"control": ctrl, "candidates": cands}})


def scoreboard_run(pc) -> dict:
    from .experiments import open_experiment

    cfg = _scoreboard_cfg(pc, pc.local)
    if cfg is None:
        raise PipelineStop("no model maps available to score",
                           "restore legacy run folders (docs/ARTIFACT_RECOVERY.md) or list baselines in "
                           "train_baselines, e.g. ['B0_lineament', 'G1', 'G3']")
    exp = open_experiment(pc, cfg)
    exp.run_all()
    board = exp.rd.load_table("scoreboard")
    _save_table(pc, board.assign(experiment_id=exp.id), "scoreboard_latest")
    return {"experiment": exp.id, "models": list(board.model)}


def scoreboard_validate(pc, root) -> V.Report:
    from .experiments import experiment_state

    cfg = _scoreboard_cfg(pc, root)
    if cfg is None:
        return _report("discovery_validation", V.Check("models", V.MISSING, "no model maps available"))
    st = experiment_state(root, cfg)
    return _report("discovery_validation", V.Check(f"experiments/{st['experiment_id']}",
                                                   V.OK if st["complete"] else V.MISSING, f"models={cfg.get('models')}"))


def scoreboard_rels(pc) -> list[str]:
    root = pc.drive if pc.drive is not None and pc.raw_path("features", pc.drive).exists() else pc.local
    try:
        cfg = _scoreboard_cfg(pc, root)
    except Exception:
        return []
    return [f"experiments/{cfg.experiment_id()}", "reports/pipeline/scoreboard_latest.csv"] if cfg else []


def _ensemble_cfg(pc, root: Path):
    from .experiments import experiment_config

    stem = pc.cfg.get("experiments.ensemble")
    base = experiment_config(pc, stem)
    have = set(available_models(pc, root, need_discovery=True))
    members = [m for m in base.get("members") if m in have]
    members += [m for m in sorted(have) if m.startswith("P5_") and m not in members]
    if len(members) < 2:
        return None, members
    return experiment_config(pc, stem, {"members": members} if members != base.get("members") else None), members


def ensemble_run(pc) -> dict:
    from .experiments import open_experiment

    cfg, members = _ensemble_cfg(pc, pc.local)
    if cfg is None:
        raise PipelineStop(f"fewer than 2 members with OOF + every discovery map ({members})",
                           "complete pseudo-discovery maps of at least two models (baselines / experiments) first")
    exp = open_experiment(pc, cfg)
    exp.run_all()
    return {"experiment": exp.id, "members": members}


def ensemble_validate(pc, root) -> V.Report:
    from .experiments import experiment_state

    cfg, members = _ensemble_cfg(pc, root)
    if cfg is None:
        return _report("ensemble", V.Check("members", V.MISSING, f"only {members} have complete maps"))
    st = experiment_state(root, cfg)
    return _report("ensemble", V.Check(f"experiments/{st['experiment_id']}", V.OK if st["complete"] else V.MISSING,
                                       f"members={members}"))


def ensemble_rels(pc) -> list[str]:
    try:
        cfg, _ = _ensemble_cfg(pc, pc.local)
    except Exception:
        return []
    return [f"experiments/{cfg.experiment_id()}"] if cfg else []


# =============================================================================== inference / submission
def _submission_cfg(pc):
    from .experiments import experiment_config

    return experiment_config(pc, pc.cfg.get("experiments.submission"))


def _champion(pc):
    from ..artifacts.registry import ModelRegistry

    pc.stage_in(["model_registry"])
    return ModelRegistry(pc.local / "model_registry").champion()


def inference_run(pc) -> dict:
    from .experiments import open_experiment

    ch = _champion(pc)
    cfg = _submission_cfg(pc)
    if ch is None and not cfg.get("model.name"):
        raise PipelineStop("no promoted model in the registry",
                           "promotion is a manual decision: review promotion_decisions.csv of the candidate's "
                           "experiment, then `from gems_bench.experiment.promote import promote; "
                           "promote('<model>', '<experiment_id>')` (or `gems registry promote`)")
    if ch is not None and ch.get("kind") == "cnn5":
        pc.require_gpu(f"full-raster inference of {ch['name']}")
    if ch is not None and ch.get("experiment_id"):
        pc.stage_in([f"experiments/{ch['experiment_id']}"])
    exp = open_experiment(pc, cfg)
    exp.prepare()
    exp.train()
    return {"experiment": exp.id, "model": (ch or {}).get("name") or cfg.get("model.name")}


def inference_validate(pc, root) -> V.Report:
    from .experiments import experiment_state

    st = experiment_state(root, _submission_cfg(pc))
    return _report("full_inference", V.Check(f"experiments/{st['experiment_id']}:train",
                                             V.OK if st["train"] == "done" else V.MISSING, st["train"]),
                   V.check_geotiff(root / "experiments" / st["experiment_id"] / "predictions" / "full_region.tif",
                                   grid=pc.grid(root)))


def submission_run(pc) -> dict:
    from .experiments import open_experiment

    exp = open_experiment(pc, _submission_cfg(pc))
    exp.export()
    files = exp.rd.load_table("submission_files")
    tmpl = V.grid_of(pc.raw_path("submission"))
    bad = [f for f in files.file if not V.check_geotiff(pc.local / "submissions" / Path(str(f)).name, grid=tmpl).ok]
    if bad:
        raise PipelineStop(f"exported files fail the template-grid check: {bad}", "inspect the submission export audit")
    return {"files": list(files.file), "note": "files are written, never uploaded: upload manually and log the score "
                                               "in reports/leaderboard_log.csv"}


def submission_validate(pc, root) -> V.Report:
    from .experiments import experiment_state

    st = experiment_state(root, _submission_cfg(pc))
    r = _report("submission", V.Check(f"experiments/{st['experiment_id']}:export",
                                      V.OK if st["export"] == "done" else V.MISSING, st["export"]))
    t = root / "experiments" / st["experiment_id"] / "metrics" / "submission_files.csv"
    if t.exists():
        tmpl = V.grid_of(pc.raw_path("submission", root)) if pc.raw_path("submission", root).exists() else None
        for f in pd.read_csv(t).file:
            r.add(V.check_geotiff(root / "submissions" / Path(str(f)).name, grid=tmpl))
    return r


def submission_rels(pc) -> list[str]:
    try:
        return [f"experiments/{_submission_cfg(pc).experiment_id()}", "submissions", "model_registry"]
    except Exception:
        return ["submissions", "model_registry"]


# =============================================================================== reports
def reports_run(pc) -> dict:
    from .status import write_report

    return write_report(pc)


def reports_validate(pc, root) -> V.Report:
    return _report("reports", V.check_csv(root / "reports/pipeline/stage_status.csv", columns=("stage",)))


# =============================================================================== registry
def _flag(key: str, why: str):
    def f(pc):
        return (True, "") if pc.cfg.get(key) else (False, why)

    return f


def _gpu_flag(key: str, why: str):
    def f(pc):
        if not pc.cfg.get(key):
            return False, why
        if not pc.cfg.get("gpu.use_gpu"):
            return False, "use_gpu is false"
        return pc.cuda_ready()                 # checked before any model is allocated

    return f


def build_registry() -> StageRegistry:
    """Declaration order = notebook order (ties of the topological sort follow it):

    bootstrap, raw_data, native_grid, dem_inventory, geological_eda, segment_attributes, spatial_folds,
    discovery_splits, feature_store, dem_pilot, baselines, advanced_models, discovery_validation, ensemble,
    full_inference, submission, reports.

    ``critical`` marks the data-preparation backbone (failure_policy "interactive" stops on its failure)."""
    R = StageRegistry()
    R.add(Stage("bootstrap", "Environment, Drive, workspace folders", run=bootstrap_run, validate=bootstrap_validate,
                reuse=ALWAYS_RUN, resource=LIGHT, section="5", est_minutes="<1", critical=True,
                outputs=lambda pc: ["environment/pipeline/config_snapshot.json", "reports/pipeline/environment_report.csv"],
                recovery="mount Drive (drive.mount('/content/drive')) and re-run the bootstrap cell"))
    R.add(Stage("raw_data", "Competition files: locate, validate, download once", deps=("bootstrap",), run=raw_run,
                validate=raw_validate, resource=NETWORK, requirement=RAW_REQUIRED, section="7", est_minutes="1-10",
                critical=True,
                outputs=lambda pc: ["data/raw/acquisition_manifest.json", "reports/pipeline/raw_data_acquisition.csv"],
                recovery="configure fresh download URLs (Colab Secrets GEMS_URL_*) or upload the files to Drive"))
    R.add(Stage("native_grid", "Native 100 m grid, labels, template checks", deps=("raw_data",), run=native_run,
                validate=native_validate, fingerprint=native_fp, resource=CPU, section="8", est_minutes="1-2",
                critical=True, outputs=lambda pc: [NATIVE_GRID_JSON, "reports/pipeline/native_band_table.csv"],
                stage_in=lambda pc: [NATIVE_GRID_JSON]))
    R.add(Stage("dem_inventory", "USGS 1 m DEM inventory (metadata only)", deps=("native_grid",), run=dem_inv_run,
                validate=dem_inv_validate, fingerprint=dem_inv_fp, enabled=dem_inv_enabled, requirement=OPTIONAL,
                outputs=lambda pc: [DEM_INV_DIR, "data/raw/dem_1m/tnm_urls.json"], stage_in=lambda pc: [DEM_INV_DIR],
                resource=LIGHT, section="9", est_minutes="1"))
    R.add(Stage("geological_eda", "Phase-1 EDA dependency repair (minimal rebuild)", deps=("native_grid",),
                run=eda_run, validate=eda_validate, fingerprint=eda_fp, enabled=_flag("stages.run_eda", "run_eda is false"),
                outputs=eda_outputs, stage_in=lambda pc: list(EDA_STAGE_IN), resource=CPU_HEAVY, section="10",
                est_minutes="15-45", critical=True))
    R.add(Stage("segment_attributes", "Native-grid segment ids + attributes", deps=("geological_eda",), run=seg_run,
                validate=seg_validate, outputs=lambda pc: list(SEG_RELS), stage_in=lambda pc: list(SEG_RELS),
                resource=CPU, section="10", est_minutes="2-5", critical=True))
    R.add(Stage("spatial_folds", "Geographic folds (Phase-2 protocol)", deps=("native_grid",), run=folds_run,
                validate=folds_validate, fingerprint=lambda pc: fold_params(pc.cfg.get("validation"), pc.seed),
                outputs=lambda pc: [pc.fold_rel], stage_in=lambda pc: [pc.fold_rel], versioned=True, resource=CPU,
                section="11", est_minutes="1-3", critical=True))
    R.add(Stage("discovery_splits", "Pseudo-discovery fault-system splits", deps=("native_grid",), run=splits_run,
                validate=splits_validate, fingerprint=lambda pc: discovery_params(pc.cfg.get("validation"), pc.seed),
                outputs=lambda pc: [pc.discovery_rel], stage_in=lambda pc: [pc.discovery_rel], versioned=True,
                resource=CPU, section="11", est_minutes="1-3", critical=True))
    # feature_store needs the grid, folds and splits (research-context loader). The EDA is a *family-level*
    # prerequisite (eda:* features, FAMILY_EDA_GROUPS), so it is a soft dependency: ordered first, never blocking
    # the families that do not use it.
    R.add(Stage("feature_store", "Feature store: compute once, reuse", deps=("spatial_folds", "discovery_splits"),
                soft_deps=("geological_eda", "segment_attributes"),
                run=features_run, validate=features_validate, fingerprint=features_fp, outputs=features_outputs,
                stage_in=features_stage_in, reuse=DELEGATED, resource=CPU_HEAVY, section="12", est_minutes="10-90",
                critical=True))
    R.add(Stage("dem_pilot", "1 m DEM pilot: selection, download, morphology, 100 m aggregation",
                deps=("dem_inventory", "feature_store"), run=dem_pilot_run, validate=dem_pilot_validate,
                enabled=dem_pilot_enabled, outputs=dem_pilot_rels, stage_in=dem_pilot_rels, reuse=DELEGATED,
                resource=NETWORK_HEAVY, requirement=OPTIONAL, section="12", est_minutes="30-180"))
    # baselines build the features each recipe needs (_ensure_features), so a G1 LightGBM can train while an
    # unrelated feature family (e.g. eda:*) is still missing; recipes that need the EDA say so explicitly.
    R.add(Stage("baselines", "Baselines: restore, verify, retrain on request", deps=("spatial_folds", "discovery_splits"),
                soft_deps=("feature_store", "segment_attributes"),
                run=baselines_run, validate=baselines_validate, enabled=_flag("stages.run_baselines", "run_baselines is false"),
                outputs=baselines_rels, stage_in=baselines_rels, reuse=ALWAYS_RUN, resource=CPU, requirement=RETRAIN,
                fingerprint=lambda pc: {"train": sorted(pc.cfg.get("stages.train_baselines") or [])},
                section="13", est_minutes="scan <1; CPU baselines 20-60 each; CNNs 30-60 GPU each"))
    R.add(Stage("advanced_models", "PU / geometry / GeoStructureNet / multiscale experiments",
                deps=("feature_store", "baselines"), run=advanced_run, validate=advanced_validate,
                fingerprint=advanced_fp, enabled=advanced_enabled, outputs=advanced_rels, stage_in=advanced_rels,
                # always run when enabled: advanced_run plans from the inventory and trains only missing work
                reuse=ALWAYS_RUN, resource=GPU, requirement=RETRAIN, section="14-15",
                est_minutes="per experiment (estimates): SCREEN ~5, VALIDATE ~30, FULL ~60 GPU"))
    R.add(Stage("discovery_validation", "Unified geographic + pseudo-discovery scoreboard", deps=("baselines",),
                soft_deps=("advanced_models", "segment_attributes"), run=scoreboard_run, validate=scoreboard_validate,
                enabled=_flag("stages.run_scoreboard", "run_scoreboard is false"), outputs=scoreboard_rels,
                stage_in=scoreboard_rels, reuse=DELEGATED, resource=CPU, section="16-17", est_minutes="5-20"))
    R.add(Stage("ensemble", "Discovery-aware ensembling + promotion gate", deps=("discovery_validation",),
                run=ensemble_run, validate=ensemble_validate, enabled=_flag("stages.run_ensemble", "run_ensemble is false"),
                outputs=ensemble_rels, stage_in=ensemble_rels, reuse=DELEGATED, resource=CPU, section="18",
                est_minutes="10-30"))
    R.add(Stage("full_inference", "Full-area inference of the promoted model", deps=("feature_store",),
                soft_deps=("ensemble", "advanced_models", "baselines"), run=inference_run, validate=inference_validate,
                enabled=_gpu_flag("stages.run_full_inference", "run_full_inference is false"), outputs=submission_rels,
                stage_in=submission_rels, reuse=DELEGATED, resource=GPU, requirement=RETRAIN, section="19",
                est_minutes="5-30"))
    R.add(Stage("submission", "Competition GeoTIFF export (never uploaded)", deps=("full_inference",),
                run=submission_run, validate=submission_validate,
                enabled=_flag("stages.run_submission", "run_submission is false"), outputs=submission_rels,
                stage_in=submission_rels, reuse=DELEGATED, resource=CPU, section="20", est_minutes="2-5"))
    R.add(Stage("reports", "Figures, tables and the pipeline report", deps=("bootstrap",),
                soft_deps=("raw_data", "native_grid", "feature_store", "baselines", "advanced_models",
                           "discovery_validation", "ensemble", "submission"),
                run=reports_run, validate=reports_validate, enabled=_flag("stages.run_reports", "run_reports is false"),
                outputs=lambda pc: ["reports/pipeline"], reuse=ALWAYS_RUN, resource=LIGHT, section="21",
                est_minutes="1-3"))
    R.check()
    return R
