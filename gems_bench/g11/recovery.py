"""Recovery of an existing (possibly interrupted) G11 feature stage - inspect, verify, classify; never delete.

:func:`recover_feature_stage` discovers every feature artifact under ``03_features`` (layers ``*.npy``, interrupted
``*.partial*`` writes, DEM spatial chunks), reads each file's real schema (npy header: shape, dtype, order, data
offset), checks size / truncation, content (finite fraction over the valid mask against the value recorded when the
layer was written, nodata outside the valid mask), source provenance (stage manifest, source hashes, registry hash,
write time) and configuration compatibility (layer name produced by the current configuration; feature-relevant
config sections equal to those of the code commit that wrote the layer), and classifies it:

==================== ==========================================================================================
VERIFIED_COMPLETE    has a completion manifest row; stat (size, mtime) matches; sha256 re-checked when sampled
RECOVERABLE          no completion manifest, but provenance AND content verified -> manifest reconstructed
INCOMPLETE           interrupted write (``.partial``), truncated / unreadable file, wrong shape or dtype
INCOMPATIBLE         readable but not usable: other spec / resolution / sources, unverifiable provenance, duplicate
MISSING              required by the requested feature version but absent
==================== ==========================================================================================

A file is never "complete" merely because it exists. Nothing is moved, overwritten or deleted: the only writes are
the recovery outputs (``recovery/``) and reconstructed completion-manifest rows (``03_features/layer_manifest.parquet``).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import stable_hash
from .feature_versions import (LayerManifest, dep_class, layer_id, res_tag, spec_hashes)
from .registry import atomic_write_json, atomic_write_parquet, read_json

STATUSES = ("VERIFIED_COMPLETE", "RECOVERABLE", "INCOMPLETE", "INCOMPATIBLE", "MISSING")
SKIP_DIRS = {"datasets", "dem_chunks", "versions"}
FEATURE_SECTIONS = {
    "geophysical": [("feature_engineering", "geophysics"), ("geophysical_grid", "valid_min_band_share"),
                    ("geophysical_grid", "nodata_abs")],
    "dem": [("dem", "context_scales_m"), ("dem", "allow_unknown_vertical"), ("dem", "seam_tolerance_m"),
            ("dem", "priority")],
    "label": [("faults",), ("labels", "buffers_m"), ("labels", "near_band_m")],
}


class RecoveryError(RuntimeError):
    pass


@dataclass
class RecoveryReport:
    table: pd.DataFrame                    # one row per discovered / expected layer
    missing: pd.DataFrame                  # work list for the requested version (layers + DEM chunks)
    counts: dict
    request: dict
    legacy: dict
    notes: list = field(default_factory=list)
    paths: dict = field(default_factory=dict)

    @property
    def reusable(self) -> list[str]:
        t = self.table
        return t[t.reusable_for_request.astype(bool)].layer_id.tolist() if len(t) else []

    @property
    def to_compute(self) -> pd.DataFrame:
        return self.missing

    def as_dict(self) -> dict:
        return {"counts": self.counts, "request": self.request, "legacy": self.legacy, "notes": self.notes,
                "n_reusable_for_request": len(self.reusable),
                "n_layers_to_compute": int((self.missing.kind == "layer").sum()) if len(self.missing) else 0,
                "n_dem_chunks_to_compute": int((self.missing.kind == "dem_chunk").sum()) if len(self.missing) else 0,
                "paths": {k: str(v) for k, v in self.paths.items()}}


# ------------------------------------------------------------------------------------------ file inspection
def inspect_npy(path, shape=None) -> dict:
    """Header of a ``.npy`` file (no data read): shape, dtype, fortran order, data offset, expected vs actual size."""
    p = Path(path)
    out = {"readable": False, "header_ok": False, "size": p.stat().st_size if p.is_file() else None}
    try:
        with open(p, "rb") as f:
            major, _minor = np.lib.format.read_magic(f)
            reader = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
            shp, fortran, dtype = reader(f)
            off = f.tell()
    except Exception as e:                                   # noqa: BLE001 - classified, never raised
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    exp = off + int(np.prod(shp)) * np.dtype(dtype).itemsize
    out.update(readable=True, shape=tuple(int(x) for x in shp), dtype=str(np.dtype(dtype)), fortran=bool(fortran),
               offset=int(off), expected_size=int(exp))
    probs = []
    if out["size"] != exp:
        probs.append(f"size {out['size']} != header-implied {exp} (truncated or padded write)")
    if shape is not None and tuple(shp) != tuple(shape):
        probs.append(f"shape {tuple(shp)} != grid {tuple(shape)}")
    if np.dtype(dtype) != np.float32:
        probs.append(f"dtype {dtype} != float32")
    if fortran:
        probs.append("fortran order")
    out["problems"] = probs
    out["header_ok"] = not probs
    return out


def verify_npy_content(path, info: dict, valid: np.ndarray | None) -> dict:
    """One read of the file: sha256 + finite fraction over ``valid`` + finite values outside ``valid`` + inf count."""
    data = Path(path).read_bytes()
    h = hashlib.sha256(data).hexdigest()
    a = np.frombuffer(data, dtype=np.dtype(info["dtype"]), offset=info["offset"]).reshape(info["shape"])
    fin = np.isfinite(a)
    v = valid if valid is not None and valid.shape == a.shape else np.ones(a.shape, bool)
    out = {"sha256": h, "finite_fraction": float(fin[v].mean()) if v.any() else 0.0,
           "finite_outside_valid": int((fin & ~v).sum()), "n_inf": int(np.isinf(a).sum())}
    del a, data
    return out


def sha256_of(path) -> str:
    from .registry import sha256_file

    return sha256_file(path)


# ------------------------------------------------------------------------------------------ legacy provenance
def _utc(s) -> float | None:
    if not s:
        return None
    try:
        import calendar

        return float(calendar.timegm(time.strptime(str(s), "%Y-%m-%dT%H:%M:%SZ")))
    except ValueError:
        return None


def _pick(cfg: dict, path: tuple):
    v = cfg
    for k in path:
        if not isinstance(v, dict) or k not in v:
            return "<absent>"
        v = v[k]
    return v


def legacy_config_check(cfg: dict, commit: str | None, repo_dir=None, config_rel="configs/g11_multiscale.yaml") -> dict:
    """Compare the feature-relevant config sections of the CURRENT config with the YAML of the code commit that
    wrote the legacy layers (``git show <commit>:<config>``; the commit is fetched shallowly when absent).

    Returns ``{class: "MATCH" | "MISMATCH: ..." | "UNVERIFIABLE: ..."}`` per dependency class.
    """
    import yaml

    if not commit:
        return {k: "UNVERIFIABLE: no code commit in the stage manifest" for k in FEATURE_SECTIONS}
    d = Path(repo_dir) if repo_dir else Path(__file__).resolve().parents[2]

    def git(*a):
        return subprocess.run(["git", "-C", str(d), *a], capture_output=True, text=True, timeout=120)

    try:
        if git("cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
            git("fetch", "--quiet", "--depth", "1", "origin", commit)
        r = git("show", f"{commit}:{config_rel}")
    except (OSError, subprocess.SubprocessError) as e:
        return {k: f"UNVERIFIABLE: git unavailable ({type(e).__name__})" for k in FEATURE_SECTIONS}
    if r.returncode != 0:
        return {k: f"UNVERIFIABLE: {commit[:10]}:{config_rel} not readable ({r.stderr.strip()[:120]})"
                for k in FEATURE_SECTIONS}
    old = yaml.safe_load(r.stdout) or {}
    out = {}
    for cls, keys in FEATURE_SECTIONS.items():
        diffs = [f"{'.'.join(k)}: {_pick(old, k)!r} -> {_pick(cfg, k)!r}" for k in keys
                 if json.dumps(_pick(old, k), sort_keys=True, default=str) !=
                 json.dumps(_pick(cfg, k), sort_keys=True, default=str)]
        out[cls] = "MATCH" if not diffs else "MISMATCH: " + "; ".join(diffs)
    return out


# ------------------------------------------------------------------------------------------ main entry
def recover_feature_stage(ctx, config: dict, *, grid, source_hashes: dict | None = None, expected=None,
                          verify: str = "AUTO", dem_catalog_sha: str | None = None, g06_sources=None,
                          dem_covered: np.ndarray | None = None, legacy_check: dict | None = None,
                          sample_rehash: int = 8, log=print, write: bool = True) -> RecoveryReport:
    """Inspect ``<output_root>/03_features`` and classify every layer (see module docstring).

    ``expected``   - :func:`feature_versions.expected_layers` table for the REQUESTED version (MISSING rows)
    ``verify``     - "AUTO": layers with a completion manifest are stat-checked (+ ``sample_rehash`` re-hashed),
                     layers without one are fully read once; "FULL": every layer is fully re-read and re-hashed
    ``dem_covered``- covered-cell mask (100 m) of the usable DEM: DEM chunk work for the requested resolution
    ``legacy_check`` - result of :func:`legacy_config_check` (computed from the legacy manifest commit if None)
    """
    t0 = time.time()
    cfg = config
    froot = ctx.root / "03_features"
    rec_dir = ctx.root / "recovery"
    sh = dict(source_hashes or ctx.source_hashes or {})
    req_res = float(cfg["dem"]["stats_resolution_m"])
    rc = cfg.get("recovery", {}) or {}
    legacy_res = float(rc.get("legacy_dem_statistics_resolution_m", 2.0))
    accept_unverified = bool(rc.get("accept_unverified_legacy_config", False))
    spec_req = spec_hashes(cfg, sh, grid, dem_catalog_sha, g06_sources)
    spec_leg = spec_hashes({**cfg, "dem": {**cfg["dem"], "stats_resolution_m": legacy_res}}, sh, grid,
                           dem_catalog_sha, g06_sources)
    pool = LayerManifest(froot)
    man_f = read_json(ctx.root / "manifests" / "features.json") or {}
    man_i = read_json(ctx.root / "manifests" / "ingest.json") or {}
    reg_p = froot / "feature_registry.parquet"
    legacy_reg = pd.read_parquet(reg_p) if reg_p.is_file() else pd.DataFrame(columns=["name", "relpath"])
    reg_sha = ctx.hashes.sha(reg_p) if reg_p.is_file() else None
    reg_rec = {Path(o["path"]).name: o.get("sha256") for o in man_f.get("outputs", [])}
    registry_verified = bool(reg_sha and reg_rec.get("feature_registry.parquet") == reg_sha)
    if legacy_check is None:
        legacy_check = legacy_config_check(cfg, man_f.get("code_commit") or man_i.get("code_commit"))
    notes = []
    legacy = {"features_manifest_status": man_f.get("status", "ABSENT"), "features_manifest_completed": man_f.get("completed_utc"),
              "code_commit": man_f.get("code_commit"), "config_hash": man_f.get("config_hash"),
              "registry_rows": int(len(legacy_reg)), "registry_verified_against_manifest": registry_verified,
              "legacy_dem_statistics_resolution_m": legacy_res,
              "legacy_dem_resolution_source": rc.get("legacy_dem_resolution_source",
                                                     "YAML dem.stats_resolution_m and notebook default "
                                                     "dem_statistics_resolution_m at the legacy commit (both 2.0); "
                                                     "not written into the legacy artifacts"),
              "config_check": legacy_check, "accept_unverified_legacy_config": accept_unverified}
    by_rel = {str(r).replace("\\", "/"): i for i, r in enumerate(legacy_reg.get("relpath", pd.Series(dtype=str)))
              if isinstance(r, str) and r}
    by_name = {n: i for i, n in enumerate(legacy_reg.name)}
    valid = grid.valid
    exp_by_name = {} if expected is None else {r.name: r._asdict() for r in expected.itertuples(index=False)}

    # ---- discovery (layers, interrupted writes)
    files = sorted(p for p in froot.rglob("*.npy") if not (set(p.relative_to(froot).parts[:-1]) & SKIP_DIRS))
    partials = sorted(p for p in froot.rglob("*.partial*") if not (set(p.relative_to(froot).parts[:-1]) & SKIP_DIRS))
    rows = []
    pool_by_rel = {str(r): r2 for r, r2 in zip(pool.df.relpath, pool.df.layer_id)} if len(pool) else {}
    seen_primary: dict = {}
    rng = np.random.default_rng(int(cfg.get("reproducibility", {}).get("seed", 0)))
    manifest_rows = [p for p in files if str(p.relative_to(froot)).replace("\\", "/") in pool_by_rel]
    rehash = set(rng.choice(len(manifest_rows), min(sample_rehash, len(manifest_rows)), replace=False).tolist()) \
        if manifest_rows else set()
    rehash_paths = {manifest_rows[i] for i in rehash}
    n_read = 0
    for p in partials:
        rel = str(p.relative_to(froot)).replace("\\", "/")
        nm = p.name.split(".partial")[0].removesuffix(".npy")
        rows.append({"layer_id": nm, "name": nm, "relpath": rel, "status": "INCOMPLETE",
                     "reason": "interrupted write (.partial file); final file never committed", "size": p.stat().st_size})
    for i, p in enumerate(files):
        rel = str(p.relative_to(froot)).replace("\\", "/")
        name = p.stem
        row = {"name": name, "relpath": rel, "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
        info = inspect_npy(p, grid.shape)
        row.update(shape=str(info.get("shape")), dtype=info.get("dtype"), header_ok=info["header_ok"])
        # ---- completion manifest (pool) fast path
        lid_pool = pool_by_rel.get(rel)
        if lid_pool is not None:
            pr = pool.row(lid_pool)
            row.update(layer_id=lid_pool, family=pr.get("family"), dep_class=pr.get("dep_class"),
                       processing_resolution_m=pr.get("processing_resolution_m"), spec_hash=pr.get("spec_hash"),
                       provenance=pr.get("provenance"), sha256=pr.get("sha256"), finite_fraction=pr.get("finite_fraction"))
            ok, why = pool.stat_ok(lid_pool)
            if ok and info["header_ok"] and (verify == "FULL" or p in rehash_paths):
                n_read += 1
                c = verify_npy_content(p, info, valid)
                if c["sha256"] != pr.get("sha256"):
                    ok, why = False, "sha256 differs from the completion manifest"
                else:
                    why += "; sha256 re-verified"
            row.update(status="VERIFIED_COMPLETE" if ok and info["header_ok"] else "INCOMPLETE",
                       reason=why if info["header_ok"] else "; ".join(info.get("problems") or [info.get("error", "")]))
            if row["status"] == "INCOMPLETE":
                pool.upsert([{**pr, "status": "INCOMPLETE", "verified_utc": None}])
            rows.append(row)
            seen_primary.setdefault(name, []).append(rel)
            continue
        # ---- legacy / orphan layer: full verification
        ri = by_rel.get(rel)
        if ri is None and name in by_name:
            r_rel = str(legacy_reg.iloc[by_name[name]].get("relpath") or "")
            ri = by_name[name] if (not r_rel or r_rel == rel) else None
        reg = legacy_reg.iloc[ri].to_dict() if ri is not None else None
        exp = exp_by_name.get(name)
        family = (reg or {}).get("family") or (exp or {}).get("family")
        dcls = dep_class(family or "", name)
        res = legacy_res if dcls == "dem" else None
        lid = layer_id(name, dcls, res)
        row.update(layer_id=lid, family=family, dep_class=dcls, processing_resolution_m=res,
                   provenance="legacy run" if reg is not None else "orphan (not in the legacy registry)")
        if not info["header_ok"]:
            row.update(status="INCOMPLETE", reason="; ".join(info.get("problems") or [info.get("error", "unreadable")]))
            rows.append(row)
            continue
        if name in seen_primary and reg is None:
            row.update(status="INCOMPATIBLE", reason=f"DUPLICATE of {seen_primary[name][0]} (not referenced; kept, never used)")
            rows.append(row)
            continue
        n_read += 1
        c = verify_npy_content(p, info, valid)
        row.update(sha256=c["sha256"], finite_fraction=c["finite_fraction"], nodata_outside_valid=c["finite_outside_valid"],
                   n_inf=c["n_inf"])
        probs = []
        man = man_i if dcls == "label" else man_f
        src_keys = ("fault_raster",) if dcls == "label" else ("geodawn",)
        if not man:
            probs.append("no stage manifest for the run that wrote it")
        else:
            bad = [k for k in src_keys if (man.get("source_hashes") or {}).get(k) != sh.get(k)]
            if bad:
                probs.append(f"source changed since it was written ({bad})")
            done = _utc(man.get("completed_utc"))
            if man.get("status") == "COMPLETE" and done and p.stat().st_mtime > done + 300:
                probs.append("written after its stage completed (unknown writer)")
        if reg is not None:
            if not registry_verified:
                probs.append("legacy registry does not match the stage manifest hash")
            ff = reg.get("finite_fraction")
            if ff is not None and np.isfinite(float(ff)) and abs(float(ff) - c["finite_fraction"]) > 1e-6:
                probs.append(f"content changed: finite fraction {c['finite_fraction']:.6f} != recorded {float(ff):.6f}")
        else:
            st_ok = man.get("status") in ("RUNNING", "FAILED") and (
                _utc(man.get("started_utc")) is None or p.stat().st_mtime >= _utc(man.get("started_utc")) - 300)
            if not st_ok:
                probs.append("orphan file outside any interrupted stage window (provenance unverifiable)")
            if exp is None:
                probs.append("orphan without metadata in the current feature plan")
        if c["n_inf"]:
            probs.append(f"{c['n_inf']} infinite values")
        chk = legacy_check.get(dcls, "UNVERIFIABLE")
        if chk.startswith("MISMATCH"):
            probs.append(f"feature config changed since the legacy run ({chk})")
        elif chk.startswith("UNVERIFIABLE") and not accept_unverified:
            probs.append(f"legacy feature config {chk} (set recovery.accept_unverified_legacy_config to accept)")
        if expected is not None and exp is None:
            probs.append("not produced by the current feature configuration")
        if probs:
            row.update(status="INCOMPATIBLE", reason="; ".join(probs))
        else:
            meta = reg or exp or {}
            spec = (spec_leg if dcls == "dem" else spec_req)[dcls]
            prow = {"layer_id": lid, "name": name, "family": family, "dep_class": dcls, "processing_resolution_m": res,
                    "relpath": rel, "size": int(p.stat().st_size), "mtime_ns": int(p.stat().st_mtime_ns),
                    "sha256": c["sha256"], "finite_fraction": c["finite_fraction"],
                    "nodata_outside_valid": c["finite_outside_valid"], "spec_hash": spec, "status": "VERIFIED_COMPLETE",
                    "provenance": f"reconstructed from {row['provenance']} (commit {legacy.get('code_commit')}, "
                                  f"config {legacy.get('config_hash')})",
                    "created_by": "recovery", "unit": meta.get("unit"), "scale_m": meta.get("scale_m"),
                    "source": meta.get("source"), "label_derived": bool(meta.get("label_derived", dcls == "label"))}
            pool.upsert([prow])
            row.update(status="RECOVERABLE", spec_hash=spec,
                       reason="provenance + content verified; completion manifest reconstructed")
        rows.append(row)
        seen_primary.setdefault(name, []).append(rel)
        if log and n_read and n_read % 50 == 0:
            log(f"[recovery] verified {n_read} layer files ({time.time() - t0:.0f} s)")
    # ---- duplicates with identical names in several folders
    t = pd.DataFrame(rows)
    if len(t):
        dup = t[t.duplicated("layer_id", keep=False) & (t.status != "INCOMPLETE")]
        for lid, g in dup.groupby("layer_id"):
            keep = g[g.status.isin(["VERIFIED_COMPLETE", "RECOVERABLE"])].head(1).index
            for ix in g.index.difference(keep):
                if t.at[ix, "status"] in ("VERIFIED_COMPLETE", "RECOVERABLE"):
                    t.at[ix, "status"] = "INCOMPATIBLE"
                    t.at[ix, "reason"] = f"DUPLICATE of {t.loc[keep[0], 'relpath'] if len(keep) else '?'}"
    # ---- requested version: reusable / missing
    want = expected if expected is not None else pd.DataFrame(columns=["name", "layer_id", "dep_class", "family"])
    reuse = []
    for r in (t.itertuples() if len(t) else []):
        ok = False
        if r.status in ("VERIFIED_COMPLETE", "RECOVERABLE") and isinstance(r.layer_id, str):
            ok, _ = pool.reusable(r.layer_id, spec_req.get(getattr(r, "dep_class", "geophysical") or "geophysical", ""))
        reuse.append(bool(ok))
    if len(t):
        t["reusable_for_request"] = reuse
    have = set(t[t.reusable_for_request].layer_id) if len(t) else set()
    miss_rows = []
    for r in want.itertuples():
        if r.layer_id in have:
            continue
        cur = t[t.layer_id == r.layer_id] if len(t) else t
        why = "absent" if cur is None or cur.empty else f"present but {cur.status.iloc[0]}: {cur.reason.iloc[0]}"
        miss_rows.append({"kind": "layer", "layer_id": r.layer_id, "name": r.name, "family": r.family,
                          "dep_class": r.dep_class, "reason": why})
        if cur is None or cur.empty:
            t = pd.concat([t, pd.DataFrame([{"layer_id": r.layer_id, "name": r.name, "family": r.family,
                                             "dep_class": r.dep_class, "status": "MISSING", "reason": "absent",
                                             "reusable_for_request": False}])], ignore_index=True)
    missing = pd.DataFrame(miss_rows, columns=["kind", "layer_id", "name", "family", "dep_class", "reason"])
    # ---- DEM spatial chunks for the requested resolution
    dem_cnt = {}
    if dem_covered is not None and (missing.dep_class == "dem").any():
        from .dem_engine import chunk_plan, committed_chunks

        plan = chunk_plan(grid, dem_covered, int(cfg["dem"].get("engine_chunk_cells", 128)))
        done = committed_chunks(froot / "dem_chunks" / res_tag(req_res), spec_req["dem"])
        todo = plan[~plan.chunk_id.isin(done)]
        dem_cnt = {"dem_chunks_total": int(len(plan)), "dem_chunks_committed": int(len(plan) - len(todo)),
                   "dem_chunks_to_compute": int(len(todo))}
        missing = pd.concat([missing, todo.assign(kind="dem_chunk", layer_id=todo.chunk_id, name=todo.chunk_id,
                                                  family="dem_100m", dep_class="dem",
                                                  reason=f"DEM chunk at {res_tag(req_res)} not committed")
                             [missing.columns.tolist() + ["r0", "r1", "q0", "q1", "covered_cells"]]],
                            ignore_index=True)
    pool.flush()
    counts = {s: int((t.status == s).sum()) if len(t) else 0 for s in STATUSES}
    counts.update(files_inspected=len(files), partial_files=len(partials), files_fully_read=n_read, **dem_cnt)
    request = {"dem_statistics_resolution_m": req_res, "spec": spec_req, "legacy_spec": spec_leg,
               "expected_layers": int(len(want)), "reusable_expected": int(len(set(want.layer_id) & have)),
               "verify": verify, "seconds": round(time.time() - t0, 1)}
    if legacy["features_manifest_status"] == "COMPLETE":
        notes.append(f"legacy feature stage manifest is COMPLETE ({legacy['features_manifest_completed']}); "
                     f"{counts['RECOVERABLE'] + counts['VERIFIED_COMPLETE']} layer(s) verified")
    if req_res != legacy_res:
        notes.append(f"requested DEM statistics resolution {req_res:g} m != legacy {legacy_res:g} m: legacy DEM layers "
                     f"are kept as their own version ({res_tag(legacy_res)}) and the requested DEM layers are computed "
                     f"separately ({res_tag(req_res)}); 2 m and 10 m layers never share an id or a file")
    rep = RecoveryReport(t, missing, counts, request, legacy, notes)
    if write:
        rep.paths = save_recovery(rep, rec_dir, grid, dem_covered, cfg)
    if log:
        log(f"[recovery] {counts} in {time.time() - t0:.0f} s; reusable for request: {request['reusable_expected']}/"
            f"{request['expected_layers']}; to compute: {rep.as_dict()['n_layers_to_compute']} layer(s), "
            f"{rep.as_dict()['n_dem_chunks_to_compute']} DEM chunk(s)")
    return rep


def save_recovery(rep: RecoveryReport, d: Path, grid, dem_covered, cfg) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    paths = {"registry": atomic_write_parquet(d / "feature_recovery_registry.parquet", rep.table),
             "missing": atomic_write_parquet(d / "missing_chunks.parquet", rep.missing)}
    try:
        paths["coverage_map"] = coverage_map(rep, grid, dem_covered, cfg, d / "feature_coverage_map.png")
    except Exception as e:                                          # noqa: BLE001 - a figure never blocks recovery
        rep.notes.append(f"coverage map not drawn: {type(e).__name__}: {e}")
    paths["report"] = atomic_write_json(d / "feature_recovery_report.json", {**rep.as_dict(), "paths": {
        k: str(v) for k, v in paths.items()}, "report_hash": stable_hash(rep.counts)})
    return paths


def coverage_map(rep: RecoveryReport, grid, dem_covered, cfg, out) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.3, 1]})
    img = np.where(grid.valid, 1.0, np.nan)
    ax[0].imshow(img, cmap="Greys", vmin=0, vmax=3, interpolation="nearest")
    if dem_covered is not None:
        ax[0].imshow(np.where(dem_covered, 1.0, np.nan), cmap="Blues", vmin=0, vmax=1.5, alpha=0.6, interpolation="nearest")
    ch = rep.missing[rep.missing.kind == "dem_chunk"] if len(rep.missing) else rep.missing
    for r in ch.itertuples():
        ax[0].add_patch(plt.Rectangle((r.q0, r.r0), r.q1 - r.q0, r.r1 - r.r0, fill=False, ec="#d95f02", lw=0.8))
    ax[0].set_title(f"valid geophysics (grey), usable DEM (blue)\nDEM chunks still to compute at "
                    f"{rep.request['dem_statistics_resolution_m']:g} m (orange): {len(ch)}")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    t = rep.table
    if len(t) and "family" in t:
        cnt = t.assign(family=t.family.fillna("?")).groupby(["family", "status"]).size().unstack(fill_value=0)
        cnt.plot.barh(stacked=True, ax=ax[1], width=0.8)
        ax[1].set_xlabel("layers")
        ax[1].set_title("layer status by family")
    fig.tight_layout()
    out = Path(out)
    tmp = out.with_name(out.stem + ".partial.png")
    fig.savefig(tmp, dpi=110)
    plt.close(fig)
    tmp.replace(out)
    return out
