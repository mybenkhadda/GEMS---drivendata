"""Stage ``preflight`` (M1.3): read-only Drive inspection BEFORE the P6 engine is attached.

    PF.run(CONFIG, overrides)          # notebook cell 'Drive preflight'; needs only the mounted Drive + the clone

Reads ``<drive_root>`` directly (never stages, never writes there except ``<output_dir>/diagnostics/`` and
``manifests/preflight.json`` through the normal M1 push) and records, per file: absolute path, existence, size, mtime,
extension, readability, schema / metadata (CSV header + rows, JSON / YAML keys, Parquet schema, NPY shape / dtype /
sampled value statistics, GeoTIFF grid / CRS / nodata / sampled finite coverage) and a sha256 for small files.
Lightweight: CSVs above ``preflight.full_read_max_mb`` are only header-read, arrays and rasters are sampled.

Every finding carries an evidence label: OBSERVED (read from the file in this call), INFERRED (derived from observed
facts by a stated rule), UNVERIFIED (needs the engine / data; the stage that checks it is named).

Outputs (``diagnostics/``): drive_inventory.csv/.json, schema_checks.csv, artifact_compatibility.csv,
artifact_dependency_graph.json, budget_root_cause_preflight.csv (from the PERSISTED tables of the last run),
DRIVE_READINESS_REPORT.md; figure figures/per_zone_area_budget_preflight.png (+ data). Status BLOCKED = a required
frozen input is missing / unreadable / malformed or the rasters are misaligned: the Runner then refuses every engine
stage (INCOMPLETE, reason quoted) instead of failing after an expensive attach.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BLOCKED, COMPLETE, WARNING, artifact_path

OBSERVED, INFERRED, UNVERIFIED = "OBSERVED", "INFERRED", "UNVERIFIED"
OK, MISSING, EMPTY, UNREADABLE, SCHEMA = "OK", "MISSING", "EMPTY", "UNREADABLE", "SCHEMA_MISMATCH"
OUT = ("diagnostics/drive_inventory.csv", "diagnostics/drive_inventory.json", "diagnostics/schema_checks.csv",
       "diagnostics/artifact_compatibility.csv", "diagnostics/artifact_dependency_graph.json",
       "diagnostics/DRIVE_READINESS_REPORT.md")
# M1 outputs of the previous run (relative to output_dir): (path, producing stage, required columns / keys)
M1_FILES = (
    ("artifact_inventory.json", "artifacts", ("summary",)),
    ("staging_manifest.json", "artifacts", ("files",)),
    ("selection/champion_candidates.csv", "selection", ("candidate_id", "deploy", "deploy_cell", "disc_dti", "geo_dti")),
    ("selection/champion_selection.json", "selection", ("champion", "recipe_id")),
    ("selection/champion_recipe.yaml", "selection", ("recipe_id", "policy", "deploy_kind")),
    ("selection/champion_provenance.json", "selection", ()),
    ("selection/candidate_per_zone.csv", "selection", ("name", "task", "zone", "rep", "frac")),
    ("selection/promotion_gates.csv", "promotion", ("gate", "status", "evidence")),
    ("validation/official_scores.csv", "validation", ("candidate",)),
    ("validation/per_zone_metrics.csv", "validation", ("task", "zone", "rep", "frac", "status", "area_px",
                                                       "area_frac_eligible")),
    ("validation/zone_budget.csv", "validation", ("zone", "eligible_px", "positive_px", "frac_of_eligible",
                                                  "positive_on_known_px")),
    ("validation/weighted_components.csv", "validation", ("task",)),
    ("validation/validation_status.json", "validation", ("status", "reasons", "recipe_id")),
    ("validation/protocol_comparison.csv", "validation", ("value_id", "protocol", "metric", "value")),
    ("parity/zone_stitched_parity.csv", "validation", ("unit", "check", "status")),
    ("parity/parity_report.csv", "parity", ("scope", "check", "unit", "status")),
    ("reports/M1_EXECUTIVE_SUMMARY.md", "report", ()),
    ("reports/M1_FAILURE_DIAGNOSTICS.md", "report", ()),
    ("reports/M1_NEXT_ACTION.md", "report", ()),
    *((f"manifests/{s}.json", s, ("status", "fingerprint")) for s in ("artifacts", "selection", "validation", "parity",
                                                                      "promotion")),
)
# required columns of frozen CSV sources (artifact id -> columns)
SOURCE_COLUMNS = {"boost_a_grid": ("task", "zone", "candidate", "blend", "rep", "frac", "tp_w", "fp_w", "fn_w"),
                  "w2_stack_grid": ("task", "zone", "tp_w", "fp_w", "fn_w")}
# GeoTIFFs whose values are sampled (a decimated read still decodes every block); others: metadata only
RASTER_STATS = ("oof.tif", "example_submission.tif")


# =============================================================================== file facts
def _sample(a: np.ndarray, n: int) -> np.ndarray:
    flat = a.reshape(-1)
    step = max(1, int(np.ceil(flat.size / max(int(n), 1))))
    return np.asarray(flat[::step], np.float64)


def value_stats(v: np.ndarray) -> dict:
    """Finite share, min / max and quantiles of a sample (pure)."""
    fin = np.isfinite(v)
    out = {"sampled_px": int(v.size), "finite_share": float(fin.mean()) if v.size else None,
           "nonfinite_sampled": int((~fin).sum())}
    if fin.any():
        f = v[fin]
        out.update({"min": float(f.min()), "max": float(f.max()),
                    **{f"q{int(q * 100):02d}": float(np.quantile(f, q)) for q in (0.01, 0.5, 0.99)}})
    return out


def read_meta(p: Path, pc: dict) -> dict:
    """Schema / metadata of one file (raises on an unreadable file; the caller records UNREADABLE)."""
    ext = p.suffix.lower()
    size_mb = p.stat().st_size / 2**20
    if ext == ".csv":
        cols = list(pd.read_csv(p, nrows=0).columns)    # EmptyDataError: no header
        m = {"columns": cols}
        if size_mb <= float(pc["full_read_max_mb"]):
            T = pd.read_csv(p, low_memory=False)
            num = T.select_dtypes("number")
            m.update({"rows": int(len(T)), "duplicate_rows": int(T.duplicated().sum()),
                      "null_cells": {c: int(n) for c, n in T.isna().sum().items() if n},
                      "nonfinite_numeric": int((~np.isfinite(num.to_numpy(np.float64))).sum() - num.isna().sum().sum())
                      if len(num.columns) else 0})
        else:
            with open(p, "rb") as f:
                m["rows"] = max(sum(1 for _ in f) - 1, 0)
        return m
    if ext == ".json":
        d = json.loads(p.read_text(encoding="utf-8"))
        return {"keys": sorted(d)[:40] if isinstance(d, dict) else f"<{type(d).__name__}>",
                **{k: d.get(k) for k in ("status", "recipe_id", "champion", "fingerprint", "impl")
                   if isinstance(d, dict) and k in d}}
    if ext in (".yaml", ".yml"):
        import yaml

        d = yaml.safe_load(p.read_text(encoding="utf-8"))
        return {"keys": sorted(d)[:40] if isinstance(d, dict) else f"<{type(d).__name__}>",
                **({"recipe_id": d.get("recipe_id")} if isinstance(d, dict) else {})}
    if ext == ".parquet":
        import pyarrow.parquet as pq

        f = pq.ParquetFile(p)
        return {"rows": int(f.metadata.num_rows), "columns": list(f.schema_arrow.names)}
    if ext == ".npy":
        a = np.load(p, mmap_mode="r", allow_pickle=False)
        return {"shape": list(a.shape), "dtype": str(a.dtype), **value_stats(_sample(a, pc["sample_px"]))}
    if ext in (".tif", ".tiff"):
        import rasterio

        with rasterio.open(p) as s:
            nod = s.nodata
            m = {"width": s.width, "height": s.height, "count": s.count, "dtype": s.dtypes[0],
                 "epsg": s.crs.to_epsg() if s.crs else None, "transform": [float(x) for x in list(s.transform)[:6]],
                 "bounds": [float(x) for x in s.bounds], "res": [float(abs(r)) for r in s.res],
                 "nodata": None if nod is None else ("nan" if np.isnan(nod) else float(nod))}
            if p.name not in RASTER_STATS:
                return m
            f = max(1, int(np.ceil(np.sqrt(s.width * s.height / max(int(pc["sample_px"]), 1)))))
            a = s.read(1, out_shape=(max(1, s.height // f), max(1, s.width // f)), masked=False).astype(np.float64)
            if nod is not None and np.isfinite(nod):
                a[a == nod] = np.nan
            return {**m, "decimation": f, **value_stats(a.reshape(-1))}
    if ext in (".md", ".txt"):
        return {"lines": len(p.read_text(encoding="utf-8").splitlines())}
    return {}


def file_row(p: Path, root: Path, fid: str, scope: str, stage: str, required: bool, pc: dict) -> dict:
    from .artifacts import file_digest

    r = {"id": fid, "scope": scope, "stage": stage, "required": bool(required), "path": str(p),
         "rel": p.relative_to(root).as_posix() if p.is_relative_to(root) else str(p), "ext": p.suffix.lower(),
         "exists": p.exists(), "evidence": OBSERVED, "size_bytes": None, "modified": None, "digest": None}
    if not p.exists():
        return {**r, "status": MISSING, "detail": "absent on Drive", "meta": {}}
    if p.is_dir():
        n = sum(1 for x in p.rglob("*") if x.is_file())
        return {**r, "status": OK if n else EMPTY, "size_bytes": None, "detail": f"directory, {n} files",
                "meta": {"n_files": n}}
    st = p.stat()
    r.update({"size_bytes": int(st.st_size), "modified": time.strftime("%Y-%m-%dT%H:%M:%S",
                                                                      time.gmtime(st.st_mtime)) + "Z"})
    if not st.st_size:
        return {**r, "status": EMPTY, "detail": "0 bytes", "meta": {}}
    try:
        meta = read_meta(p, pc)
    except Exception as e:                               # an unreadable artifact is a finding, not a crash
        return {**r, "status": UNREADABLE, "detail": f"{type(e).__name__}: {str(e)[:300]}", "meta": {}}
    dig = file_digest(p, float(pc["hash_max_mb"]))
    return {**r, "status": OK, "detail": "", "meta": meta, "digest": dig}


# =============================================================================== inventory
def inventory(drive_root: Path, cfg: dict) -> pd.DataFrame:
    """Facts of every frozen source (``cfg['artifacts']``, glob-expanded) and every M1 output of the last run."""
    from .artifacts import _matches

    root, pc = Path(drive_root), cfg["preflight"]
    rows = []
    for a in cfg["artifacts"]:
        pats = [artifact_path(a, cfg), *(a.get("alternates") or [])]
        files = []
        for pat in pats:
            files = _matches(root, pat)
            if files:
                break
        if not files:
            rows.append(file_row(root / pats[0], root, a["id"], "frozen_source", "artifacts", a.get("required"), pc))
            continue
        for f in files:
            rows.append(file_row(f, root, a["id"], "frozen_source", "artifacts", a.get("required"), pc))
        for n in a.get("required_files") or []:
            if n not in {f.name for f in files}:
                rows.append({**file_row(files[0].parent / n, root, a["id"], "frozen_source", "artifacts", True, pc),
                             "detail": f"required file {n} absent"})
    out = Path(root) / cfg["output_dir"]
    for rel, stage, _ in M1_FILES:
        rows.append(file_row(out / rel, root, rel, "m1_output", stage, False, pc))
    T = pd.DataFrame(rows)
    T["meta"] = T.meta.map(lambda m: json.dumps(m, default=str))
    return T


def schema_checks(T: pd.DataFrame) -> pd.DataFrame:
    """Required columns / keys of every readable file that has a contract (pure)."""
    req = {rel: cols for rel, _, cols in M1_FILES if cols}
    rows = []
    for r in T.to_dict("records"):
        cols = SOURCE_COLUMNS.get(r["id"]) if r["scope"] == "frozen_source" else req.get(r["id"])
        if not cols:
            continue
        if r["status"] != OK:
            rows.append({"id": r["id"], "rel": r["rel"], "status": r["status"], "missing": list(cols),
                         "evidence": OBSERVED, "detail": r.get("detail", "")})
            continue
        m = json.loads(r["meta"])
        have = set(m.get("columns") or m.get("keys") or [])
        miss = [c for c in cols if c not in have]
        dup = int(m.get("duplicate_rows") or 0)
        rows.append({"id": r["id"], "rel": r["rel"], "status": SCHEMA if miss else OK, "missing": miss,
                     "rows": m.get("rows"), "duplicate_rows": dup, "null_cells": json.dumps(m.get("null_cells") or {}),
                     "evidence": OBSERVED, "detail": "" if not dup else f"{dup} duplicated rows"})
    return pd.DataFrame(rows)


# =============================================================================== compatibility
def _meta(T: pd.DataFrame, fid: str, name: str | None = None) -> dict | None:
    d = T[(T.id == fid) & (T.status == OK)]
    if name is not None:
        d = d[d.rel.str.endswith("/" + name)]
    return json.loads(d.meta.iloc[0]) if len(d) else None


def _chk(name, ok, evidence, detail, critical=False) -> dict:
    st = "PASS" if ok is True else "FAIL" if ok is False else "UNVERIFIED"
    return {"check": name, "status": st, "evidence": evidence if ok is not None else UNVERIFIED, "critical": critical,
            "detail": detail}


def compatibility(T: pd.DataFrame, root: Path, cfg: dict, impl_now: dict) -> pd.DataFrame:
    """Cross-file consistency (recipe ids, experiment ids, raster alignment, grid content, stage code) - observed
    facts only; anything that needs the engine is UNVERIFIED with the stage that checks it."""
    out = Path(root) / cfg["output_dir"]
    rows = []
    req = T[(T.scope == "frozen_source") & T.required]
    bad = req[req.status != OK]
    rows.append(_chk("required frozen sources readable", not len(bad), OBSERVED,
                     "; ".join(f"{r.id}:{r.rel} {r.status} {r.detail}" for r in bad.itertuples()) or
                     f"{len(req)} files OK", critical=True))
    ids = {}
    for fid in ("champion_selection", "recipe_yaml", "selection_manifest", "validation_status"):
        rel = {"champion_selection": "selection/champion_selection.json", "recipe_yaml": "selection/champion_recipe.yaml",
               "selection_manifest": "manifests/selection.json",
               "validation_status": "validation/validation_status.json"}[fid]
        m = _meta(T, rel)
        if m is not None and m.get("recipe_id") is not None:
            ids[fid] = m["recipe_id"]
    rows.append(_chk("recipe id identical across selection / recipe / manifests / validation",
                     (len(set(ids.values())) == 1) if ids else None, OBSERVED, json.dumps(ids)))
    for a in cfg["artifacts"]:
        if not a.get("expected_id"):
            continue
        seen = sorted({Path(p).relative_to(root).parts[1] for p in T[(T.id == a["id"]) & T.exists].path
                       if len(Path(p).relative_to(root).parts) > 1})
        rows.append(_chk(f"{a['id']} experiment id == {a['expected_id']}", (seen == [a["expected_id"]]) if seen
                         else None, OBSERVED, f"seen {seen}", critical=True))
    tpl = _meta(T, "template")
    for fid in ("b0_maps", "b1_maps"):
        for name in ("oof.tif", "disc0.tif"):
            m = _meta(T, fid, name)
            if tpl is None or m is None:
                rows.append(_chk(f"{fid}/{name} grid == template", None, UNVERIFIED, "file or template unreadable"))
                continue
            same = all(m[k] == tpl[k] for k in ("width", "height", "epsg")) and np.allclose(m["transform"],
                                                                                          tpl["transform"])
            rows.append(_chk(f"{fid}/{name} grid == template (RASTER_ALIGNMENT)", bool(same), OBSERVED,
                             f"{m['width']}x{m['height']} EPSG {m['epsg']} vs template {tpl['width']}x{tpl['height']} "
                             f"EPSG {tpl['epsg']}", critical=True))
    if tpl is not None:
        e = cfg["submission"]["expected"]
        rows.append(_chk("template EPSG / resolution / band / dtype == submission.expected",
                         tpl["epsg"] == int(e["epsg"]) and all(abs(r - float(e["res_m"])) < 1e-6 for r in tpl["res"])
                         and tpl["count"] == int(e["count"]) and tpl["dtype"] == e["dtype"], OBSERVED,
                         f"EPSG {tpl['epsg']} res {tpl['res']} count {tpl['count']} dtype {tpl['dtype']} nodata "
                         f"{tpl['nodata']}", critical=True))
    for k in range(3):
        m = _meta(T, "w2_maps", f"geo_{k}.npy")
        rows.append(_chk(f"W2 geo_{k}.npy readable with finite values", None if m is None else
                         bool(m.get("finite_share")), OBSERVED,
                         json.dumps({x: m.get(x) for x in ("shape", "dtype", "finite_share", "min", "max")}) +
                         " (NaN outside the zone's evaluation mask may be by design: INFERRED)" if m else "unreadable"))
    rows.append(_chk("W2 geo_k.npy shape == geo zone crop", None, UNVERIFIED,
                     "needs the fold crops: checked by final_submission.zone_sources during validation (raises)"))
    rp = out / "selection" / "champion_recipe.yaml"
    gp = req[(req.id == "boost_a_grid") & (req.status == OK)]
    if rp.exists() and len(gp):
        import yaml

        try:
            pol = (yaml.safe_load(rp.read_text(encoding="utf-8")) or {}).get("policy") or {}
            G = pd.read_csv(gp.path.iloc[0], usecols=["candidate", "rep", "frac"])
            orig = str(pol.get("grid_candidate", "")).partition(":")[2]
            hit = G[(G.candidate.astype(str) == orig) & (G.rep == pol.get("rep")) &
                    np.isclose(pd.to_numeric(G.frac, errors="coerce"), float(pol.get("frac", np.nan)))]
            rows.append(_chk("champion recipe cell present in the boost-A grid", bool(len(hit)), OBSERVED,
                             f"{orig} {pol.get('rep')} @ {pol.get('frac')}: {len(hit)} rows"))
        except Exception as e:
            rows.append(_chk("champion recipe cell present in the boost-A grid", False, OBSERVED,
                             f"{type(e).__name__}: {e}"))
    for s, now in impl_now.items():
        m = _meta(T, f"manifests/{s}.json")
        if m is None or now is None:
            continue
        rows.append(_chk(f"manifests/{s}.json produced by the current stage code", m.get("impl") == now, INFERRED,
                         f"recorded {m.get('impl')} vs current {now}: a mismatch means the stage re-runs"))
    return pd.DataFrame(rows)


def dependency_graph(cfg: dict) -> dict:
    from . import champion, gates, parity, submission, validation

    srcs = [a["id"] for a in cfg["artifacts"]]
    return {"preflight": {"inputs": ["Drive (read-only)"], "outputs": list(OUT), "engine": False},
            "artifacts": {"inputs": srcs, "outputs": ["artifact_inventory.json", "staging_manifest.json"],
                          "engine": False},
            "selection": {"inputs": ["artifacts", "boost_a_grid", "w2_stack_grid", "m9_selection"],
                          "outputs": list(champion.OUT), "engine": True},
            "validation": {"inputs": ["selection", "w2_maps", "b0_maps", "b1_maps", "geo_folds", "template"],
                           "outputs": list(validation.OUT), "engine": True},
            "parity": {"inputs": ["validation", "recovery_parity", "hist_submission_sidecar"],
                       "outputs": list(parity.OUT), "engine": True},
            "promotion": {"inputs": ["selection", "validation", "parity"], "outputs": list(gates.OUT), "engine": False},
            "submissions": {"inputs": ["promotion (READY_FOR_EXPORT)", "validation"], "outputs": list(submission.OUT),
                            "engine": True},
            "exploratory": {"inputs": ["promotion (only GEOGRAPHIC_GATE_PASS failing)", "exploratory.enabled + reason"],
                            "outputs": list(submission.EXPLORATORY_OUT), "engine": True}}


def historical_budget(root: Path, cfg: dict) -> pd.DataFrame | None:
    """Budget forensics on the PERSISTED zone_budget / per_zone_metrics of the last run (no engine)."""
    from . import forensics as FO
    from .session import Workspace

    ws = Workspace(Path(root) / cfg["output_dir"])
    Z = ws.load_table("validation/zone_budget.csv")
    if not len(Z) or not {"zone", "frac_of_eligible", "positive_on_known_px"} <= set(Z):
        return None
    from ..p6.full_gpu import load_full_config

    frac = float(Z.target_frac.iloc[0]) if "target_frac" in Z else np.nan
    tol = float(load_full_config()["submission"]["budget_tolerance"])
    keep = [c for c in Z.columns if c in ("zone", "zone_px", "eligible_px", "positive_px", "frac_of_eligible",
                                          "positive_on_known_px", "eval_eligible_px", "known_in_zone_px")]
    return FO.budget_table(Z[keep], frac, tol, FO.evaluated_area(ws.load_table("validation/per_zone_metrics.csv")))


# =============================================================================== report + stage
def readiness_report(T: pd.DataFrame, S: pd.DataFrame, C: pd.DataFrame, B: pd.DataFrame | None, status: str,
                     blocking: list[str]) -> str:
    from .reporting import _md

    miss = T[(T.scope == "m1_output") & (T.status != OK)]
    lines = ["# Drive readiness report (M1.3 preflight)", "",
             f"Status: **{status}**" + (f" - blocking: {blocking}" if blocking else ""), "",
             "Evidence labels: OBSERVED = read from Drive in this call; INFERRED = derived from observed facts; "
             "UNVERIFIED = needs the engine (the checking stage is named).", "",
             "## Frozen sources", "", _md(T[T.scope == "frozen_source"][["id", "rel", "required", "status",
                                                                             "size_bytes", "detail"]]), "",
             "## Previous M1 outputs", "",
             _md(T[T.scope == "m1_output"][["id", "stage", "status", "size_bytes", "modified"]]), "",
             f"Absent / unreadable M1 outputs ({len(miss)}) are downstream products, documented here; they are not "
             "evidence that an earlier stage failed.", "",
             "## Schema checks", "", _md(S), "", "## Compatibility", "", _md(C), ""]
    if B is not None:
        cols = [c for c in ("zone", "eligible_px", "positive_px", "frac_of_eligible", "target_frac", "pool_px",
                            "pool_yield", "eval_positive_px", "eval_area_frac_eligible", "positive_on_known_px",
                            "m1_1_two_sided_ok", "ok", "cause") if c in B]
        lines += ["## Area budget of the last run, re-read under the upper-bound contract (OBSERVED tables)", "",
                  _md(B[cols]), ""]
    return "\n".join(lines)


def run(CONFIG: dict, m1_config: dict | None = None) -> dict:
    """The preflight stage (no pipeline, no engine). Returns the manifest."""
    from ..artifacts.sync import DriveSync
    from . import forensics as FO
    from .config import load_m1_config
    from .session import M1Session, Workspace, impl_fp

    cfg = load_m1_config(m1_config)
    drive, local = Path(CONFIG["drive_root"]), Path(CONFIG["local_root"])
    out = cfg.get("outputs") or {}
    ws = Workspace(local / cfg["output_dir"], DriveSync(local, drive) if out.get("sync_to_drive", True) else None,
                   local, save_figures=bool(out.get("save_figures", True)), save_parquet=bool(out.get("save_parquet",
                                                                                                    True)))
    t0 = time.perf_counter()
    T = inventory(drive, cfg)
    S = schema_checks(T)
    C = compatibility(T, drive, cfg, {s: impl_fp(s) for s in ("selection", "validation", "parity", "submissions")})
    B = historical_budget(drive, cfg)
    ws.save_table(T, "diagnostics/drive_inventory.csv", parquet=True)
    ws.save_json(T.to_dict("records"), "diagnostics/drive_inventory.json")
    ws.save_table(S, "diagnostics/schema_checks.csv")
    ws.save_table(C, "diagnostics/artifact_compatibility.csv")
    ws.save_json(dependency_graph(cfg), "diagnostics/artifact_dependency_graph.json")
    if B is not None:
        ws.save_table(B, "diagnostics/budget_root_cause_preflight.csv")
        ws.save_json({**FO.root_cause_summary(B), "source": "persisted validation/zone_budget.csv of the last run",
                      "evidence": OBSERVED}, "diagnostics/root_cause_summary_preflight.json")
        if "target_frac" in B and B.target_frac.notna().all():
            FO.budget_figure(ws, B, "figures/per_zone_area_budget_preflight.png")
    src_schema = S[(S.status != OK) & S.id.isin(list(SOURCE_COLUMNS))] if len(S) else S
    blocking = [f"{r.check}: {r.detail}" for r in C.itertuples() if r.critical and r.status == "FAIL"]
    blocking += [f"schema {r.id}: {r.status} missing {r.missing}" for r in src_schema.itertuples()]
    warns = [f"{r.check}: {r.detail}" for r in C.itertuples() if not r.critical and r.status == "FAIL"]
    warns += [f"schema {r.id}: {r.status} missing {r.missing}" for r in S.itertuples()
              if r.status not in (OK, MISSING) and r.id not in SOURCE_COLUMNS] if len(S) else []
    status = BLOCKED if blocking else WARNING if warns else COMPLETE
    ws.save_text(readiness_report(T, S, C, B, status, blocking), "diagnostics/DRIVE_READINESS_REPORT.md")
    from types import SimpleNamespace

    sess = M1Session(SimpleNamespace(bst=None, p6=None, cfg=None), cfg, ws)
    return sess.record("preflight", status, "", reasons=blocking, warnings=warns, outputs=list(OUT),
                       seconds=round(time.perf_counter() - t0, 1),
                       counts={"files": int(len(T)), "frozen_not_ok": int(((T.scope == "frozen_source") &
                                                                         (T.status != OK)).sum()),
                               "m1_outputs_absent": int(((T.scope == "m1_output") & (T.status != OK)).sum())})
