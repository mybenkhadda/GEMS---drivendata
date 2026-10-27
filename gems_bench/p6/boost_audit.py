"""P6 boost preflight, contracts, provenance and reports (docs/p6/P6_BOOST_HANDOFF.md).

    pre = bst.preflight()    # boost/preflight/*: G02 attachment + backfill plan, feature contract, raster alignment,
                             #   DEM coverage, training / inference parity, evaluation contract, submission provenance
    rep = bst.report()       # boost/reports/*: summary, per zone, systems, stage status, readiness, artifact registry

Read-only on W0-W2, the G02 run and the submissions: nothing here trains, scores a candidate, backfills or exports.
Every answer comes from a saved artifact; an absent one is reported PENDING / BLOCKED, never estimated. Status words:
PASS / WARNING / BLOCKED for checks; PENDING / BLOCKED / RUNNING / COMPLETE / FAILED for stages.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import stack as ST

PASS, WARNING, BLOCKED = "PASS", "WARNING", "BLOCKED"
_ORDER = {PASS: 0, WARNING: 1, BLOCKED: 2}
PREFLIGHT_VERSION = "p6b-pre.1"

_RANK = "rank in [0, 1] inside the unit evaluation mask (training) / the valid footprint (full region)"
_G02 = ("G02 run (label-free maps of the M9 GeoDAWN potential-field layers); percentile references on the unit's "
        "training geography (assert_disjoint with the evaluation zone)")
# Static construction contract of every W2 / boost feature. Availability is never inferred from these names: it is
# read from the artifacts by ``feature_contract``.
CATALOG = {
    "cnn_b0": {"family": "cnn", "source": "P5_B0_gated CNN", "construction": "gems3.official.unit_score",
               "units": "probability -> rank", "labels": "trained on labels; held out per unit (OOF geo fold / "
               "discovery split)", "train": "unit_score: OOF (geo) / discovery_pred:<split> (disc)",
               "inference": "p6.view.full_maps()[0] (full-region model)"},
    "cnn_b1": {"family": "cnn", "source": "P5_B1_bgw CNN", "construction": "gems3.official.unit_score",
               "units": "probability -> rank", "labels": "as cnn_b0", "train": "as cnn_b0",
               "inference": "p6.view.full_maps()[1] (full-region model)"},
    "cnn_disagree": {"family": "cnn", "source": "B0 / B1", "construction": "|rank(B0) - rank(B1)|", "units": "[0, 1]",
                     "labels": "as the CNN maps", "train": "ranks inside the unit evaluation mask",
                     "inference": "ranks inside the valid footprint"},
    "g02c": {"family": "g02", "source": _G02, "construction": "G02-C structural continuity", "units": "score -> rank",
             "labels": "none", "train": "rasters/units/<task>_<zone>.npz",
             "inference": "stack.g02_full (whole footprint as reference; resumable tiles)"},
    "g02a": {"family": "g02", "source": _G02, "construction": "G02-A apparent displacement", "units": "score -> rank",
             "labels": "none", "train": "rasters/units/<task>_<zone>.npz", "inference": "stack.g02_full"},
    "g01_persistence": {"family": "g02", "source": _G02, "construction": "G01 wavelet persistence (comp:P)",
                        "units": "score -> rank", "labels": "none", "train": "rasters/units/<task>_<zone>.npz",
                        "inference": "stack.g02_full"},
    "mag_hgrad": {"family": "potential_field", "source": "GeoDAWN magnetic band (stack.bands.mag)",
                  "construction": "total horizontal derivative (m8.potential_fields, sigma 1 px)",
                  "units": "nT / px -> rank", "labels": "none", "train": "stack/features/pf_full.npz (unit crop)",
                  "inference": "stack/features/pf_full.npz"},
    "grav_hgrad": {"family": "potential_field", "source": "GeoDAWN gravity band (stack.bands.grav)",
                   "construction": "total horizontal derivative", "units": "mGal / px -> rank", "labels": "none",
                   "train": "stack/features/pf_full.npz (unit crop)", "inference": "stack/features/pf_full.npz"},
    "mag_tilt": {"family": "potential_field", "source": "GeoDAWN magnetic band", "construction": "tilt angle",
                 "units": "rad -> rank", "labels": "none", "train": "stack/features/pf_full.npz (unit crop)",
                 "inference": "stack/features/pf_full.npz"},
    "mg_orient_agree": {"family": "potential_field", "source": "GeoDAWN magnetic + gravity bands",
                        "construction": "0.5 + 0.5 cos(2 x strike difference)", "units": "[0, 1]", "labels": "none",
                        "train": "stack/features/pf_full.npz (unit crop)", "inference": "stack/features/pf_full.npz"},
    "known_dist_km": {"family": "known", "source": "mapped fault labels (permitted set)",
                      "construction": "distance to permitted known faults (capped)", "units": "km",
                      "labels": "permitted known map only (geo: outside the fold; disc: retained faults); never the "
                      "unit's own truth", "train": "stack.permitted_known(ctx, u)",
                      "inference": "data.labels (every mapped fault)"},
    "known_density": {"family": "known", "source": "mapped fault labels (permitted set)",
                      "construction": "uniform-filter density of permitted known pixels", "units": "share of pixels",
                      "labels": "as known_dist_km", "train": "stack.permitted_known(ctx, u)",
                      "inference": "data.labels"},
    "tip_dist_km": {"family": "known", "source": "mapped fault labels (permitted set)",
                    "construction": "distance to the nearest permitted trace tip", "units": "km",
                    "labels": "as known_dist_km", "train": "stack.permitted_known(ctx, u)", "inference": "data.labels"},
    "tip_along_cos": {"family": "known", "source": "mapped fault labels (permitted set)",
                      "construction": "cos(tip -> pixel, trace direction) within endpoint_max_km", "units": "[-1, 1]",
                      "labels": "as known_dist_km", "train": "stack.permitted_known(ctx, u)",
                      "inference": "data.labels"},
}
DEM_SOURCE = ("M7 DEM mosaic outputs/m7_discovery_first/dem/features/m7_dem_features_100m.tif (1 m USGS tiles from "
              "the tnm JSON index -> m7.dem_morph, aggregated exactly to the 100 m grid)")
DEM_INTEGRATION = {
    "reads_json_metadata": "yes: m7.dem_pilot.load_inventory (tnm_urls.json / data/raw/tnm_items.json)",
    "resolves_tiff_files": "yes: m7.dem_pilot.tile_path (download only after approval: DEM_CONFIG['download'])",
    "uses_pixel_values": "yes: m7.dem_morph tile-streamed 1 m morphology -> 100 m mosaic bands",
    "aligned_to_geodawn_grid": "aggregated to the 100 m grid by M7; W4 checks the residual shift (best_shift)",
    "in_training": "no: no W2 / boost candidate has a dem_* feature",
    "in_inference": "no",
    "used_by": "W4 pilot gate only (p6/dem_pilot.py); coverage is a mask there, never a score input",
    "adapter": "boost_audit.dem_features (optional, label-free, NaN outside covered cells, in no candidate)",
    "not_available_without_new_processing": ["elevation", "local relief", "ridge / valley response"],
}


# =============================================================================== pure helpers
def worst(statuses) -> str:
    s = [x for x in statuses if x in _ORDER]
    return max(s, key=_ORDER.get) if s else PASS


def file_hash(p: Path, max_mb: float = 1024) -> str | None:
    """sha256 of a file up to ``max_mb``; above it ``size:<bytes>:mtime:<s>`` (recorded as such, never mixed up)."""
    p = Path(p)
    if not p.exists():
        return None
    st = p.stat()
    if st.st_size > max_mb * 1e6:
        return f"size:{st.st_size}:mtime:{int(st.st_mtime)}"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def npz_shapes(path: Path) -> dict:
    """{member: (shape, dtype)} of an ``.npz`` from the array headers only (nothing is decompressed in full)."""
    out = {}
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if not n.endswith(".npy"):
                continue
            with z.open(n) as f:
                v = np.lib.format.read_magic(f)
                hdr = np.lib.format.read_array_header_1_0 if v == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _, dtype = hdr(f)
            out[n[:-4]] = (tuple(int(x) for x in shape), str(dtype))
    return out


def grid_meta(data) -> dict:
    g = data.grid
    crs = g.crs.to_epsg() if hasattr(g.crs, "to_epsg") else str(g.crs)
    return {"shape": [int(x) for x in data.shape], "crs_epsg": crs,
            "transform": [float(x) for x in tuple(g.transform)[:6]]}


def align_status(expected: dict, got: dict, tol: float = 1e-6) -> tuple[str, list[str]]:
    """Compare shape / CRS / affine transform: a recorded difference BLOCKs, an unrecorded field is a WARNING."""
    why, st = [], PASS
    for k in ("shape", "crs_epsg", "transform"):
        if k not in expected:
            continue
        if got.get(k) is None:
            why.append(f"{k} not recorded")
            st = worst([st, WARNING])
            continue
        a, b = expected[k], got[k]
        if k == "transform":
            same = len(a) == len(b) and np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol, rtol=0)
        elif k == "crs_epsg":
            same = str(a) == str(b)
        else:
            same = [int(x) for x in a] == [int(x) for x in b]
        if not same:
            why.append(f"{k} {b} != expected {a}")
            st = BLOCKED
    return st, why


def unit_window(u) -> list[int]:
    return [int(u.sl[0].start), int(u.sl[0].stop), int(u.sl[1].start), int(u.sl[1].stop)]


def unit_shape(u) -> tuple[int, int]:
    r0, r1, c0, c1 = unit_window(u)
    return r1 - r0, c1 - c0


def feature_order_check(features: list[str], row_columns: list[str] | None, model_n: int | None) -> tuple[str, list]:
    """Candidate feature list vs the training rows and the saved model: missing columns / count mismatch BLOCK."""
    why = []
    if row_columns is not None:
        miss = [f for f in features if f not in row_columns]
        if miss:
            why.append(f"rows.parquet lacks {miss}")
    if model_n is not None and int(model_n) != len(features):
        why.append(f"model expects {model_n} features, candidate lists {len(features)}")
    if len(set(features)) != len(features):
        why.append("duplicate feature names")
    return (BLOCKED if why else PASS), why


def rank_agreement(a: np.ndarray, b: np.ndarray, mask: np.ndarray, top_frac: float = 0.01, n: int = 200_000,
                   seed: int = 0) -> dict:
    """Spearman correlation (sample of ``n`` pixels) and Jaccard of the top ``top_frac`` pixels inside ``mask``."""
    from scipy.stats import spearmanr

    m = np.asarray(mask, bool) & np.isfinite(a) & np.isfinite(b)
    va, vb = np.asarray(a, np.float64)[m], np.asarray(b, np.float64)[m]
    if va.size < 10:
        return {"n_px": int(va.size), "spearman": np.nan, "top_jaccard": np.nan}
    idx = np.random.default_rng(seed).choice(va.size, min(n, va.size), replace=False)
    rho = float(spearmanr(va[idx], vb[idx])[0])
    k = max(int(round(top_frac * va.size)), 1)
    ta, tb = set(np.argpartition(-va, k - 1)[:k]), set(np.argpartition(-vb, k - 1)[:k])
    return {"n_px": int(va.size), "spearman": rho, "top_jaccard": len(ta & tb) / max(len(ta | tb), 1)}


def classify_agreement(r: dict, cfg: dict) -> str:
    rho, jac = r.get("spearman"), r.get("top_jaccard")
    if rho is None or not np.isfinite(rho):
        return WARNING
    if rho < float(cfg["block_rank_corr"]):
        return BLOCKED
    if rho < float(cfg["warn_rank_corr"]) or (jac is not None and np.isfinite(jac) and jac < 0.2):
        return WARNING
    return PASS


def parity_by_feature(parity: dict | None) -> dict:
    """{feature: worst status} from ``preflight/train_inference_parity.json`` ("*" = checks of every feature)."""
    out: dict = {}
    for r in (parity or {}).get("rows", []):
        f = str(r.get("feature", "*"))
        out[f] = worst([out.get(f, PASS), r.get("status", PASS)])
    return out


def parity_ok(by_feature: dict, names: list[str]) -> tuple[bool, str]:
    """Export / promotion needs a parity audit with no BLOCKED check on the candidate's features or globally."""
    if not by_feature:
        return False, "train/inference parity not audited: run bst.preflight() first"
    bad = [f for f in ["*"] + list(names) if by_feature.get(f) == BLOCKED]
    return (not bad), (f"BLOCKED parity: {bad}" if bad else "parity: no BLOCKED check")


def protocol_consistency(G: pd.DataFrame) -> pd.DataFrame:
    """Per (task, zone): number of distinct protocol keys / eligible pixel counts over every candidate (must be 1)."""
    if not len(G) or "protocol_key" not in G:
        return pd.DataFrame(columns=["task", "zone", "n_protocol_keys", "n_eligible_px", "status"])
    g = G.groupby(["task", "zone"]).agg(n_protocol_keys=("protocol_key", "nunique"),
                                        n_eligible_px=("eligible_px", "nunique")).reset_index()
    g["status"] = np.where((g.n_protocol_keys == 1) & (g.n_eligible_px == 1), PASS, BLOCKED)
    return g


def budget_violations(G: pd.DataFrame, tol: float = 0.02, hybrid_base: str | None = None,
                      hybrids: tuple = ("D1", "D2")) -> pd.DataFrame:
    """Fixed cells whose selected eligible area exceeds the nominal ``frac`` (+ relative ``tol``), and hybrid cells
    larger than their base map at the same (task, zone, blend, rep, frac)."""
    if not len(G) or "frac" not in G:
        return pd.DataFrame()
    g = G[np.isfinite(pd.to_numeric(G.frac, errors="coerce"))].copy()
    over = g[g.area_frac_eligible > g.frac.astype(float) * (1.0 + tol) + 1e-9]
    out = [over.assign(violation="area_frac_eligible > frac")[["task", "zone", "candidate", "rep", "frac",
                                                                 "area_frac_eligible", "violation"]]]
    if hybrid_base is not None and "blend" in g:
        key = ["task", "zone", "blend", "rep", "frac"]
        b = g[g.candidate == hybrid_base][key + ["area_px"]].rename(columns={"area_px": "base_area_px"})
        h = g[g.candidate.isin(hybrids)].merge(b, on=key, how="left")
        bad = h[h.area_px > h.base_area_px]
        out.append(bad.assign(violation="hybrid area > base area")[["task", "zone", "candidate", "rep", "frac",
                                                                     "area_frac_eligible", "violation"]])
    return pd.concat([o for o in out if len(o)], ignore_index=True) if any(len(o) for o in out) else pd.DataFrame()


def provenance_verdict(sidecar: dict | None, disc_choice: dict, geo_choice: dict, exists: bool,
                       tol: float = 0.02) -> dict:
    """Reconstruct an exported file's (rep, frac, blend) from its writer sidecar and compare with the nested
    choices. VERIFIED = file + complete parameters + area consistent with them; the separate ``selected_cell``
    flag says whether any nested evaluation chose that cell."""
    why = []
    p = (sidecar or {}).get("parameters") or {}
    rep, frac, blend = p.get("rep"), p.get("frac"), p.get("blend")
    if not exists:
        why.append("submission file absent")
    if not sidecar:
        why.append("writer sidecar submissions/<stem>.json absent")
    if rep is None or frac is None or blend is None:
        why.append(f"parameters incomplete: {p}")
    ff = (sidecar or {}).get("final_frac_of_eligible")
    if frac is not None and ff is not None and float(ff) > float(frac) * (1.0 + tol) + 1e-9:
        why.append(f"final area {float(ff):.4f} of eligible exceeds frac {float(frac):.4f}")

    def same(c):
        return bool(c) and c.get("rep") == rep and frac is not None and c.get("frac") is not None and \
            abs(float(c["frac"]) - float(frac)) < 1e-9

    return {"status": "UNVERIFIED" if why else "VERIFIED", "reasons": why, "rep": rep, "frac": frac, "blend": blend,
            "frac_source": (sidecar or {}).get("frac_source"), "final_frac_of_eligible": ff,
            "matches_disc_choice": same(disc_choice), "matches_geo_choice": same(geo_choice),
            "selected_cell": same(disc_choice) or same(geo_choice)}


def dem_features(mosaic, bands: list[str], min_cov: float, sl=None) -> dict:
    """Optional label-free DEM layers from the M7 mosaic (no tile processing): ``dem_<band>``, NaN outside covered
    cells; orientation stays as the doubled-angle cos / sin pair (cyclic). In no trained candidate."""
    cov = mosaic.coverage(float(min_cov))
    out = {}
    for b in bands:
        if mosaic.has(b):
            a = np.where(cov, np.asarray(mosaic.band(b), np.float32), np.nan).astype(np.float32)
            out[f"dem_{b}"] = a[sl] if sl is not None else a
    return out


def md_table(df: pd.DataFrame, cols: list[str] | None = None, digits: int = 4) -> str:
    """Markdown table without optional dependencies."""
    if df is None or not len(df):
        return "_(none)_\n"
    d = df[[c for c in (cols or list(df.columns)) if c in df]]

    def f(v):
        if isinstance(v, float):
            return "" if not np.isfinite(v) else f"{v:.{digits}f}"
        return str(v).replace("|", "/").replace("\n", " ")

    lines = ["| " + " | ".join(d.columns) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(f(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def _is_sha(ref: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{7,40}", str(ref or "")))


def startup_check(requested_ref: str, head: str, expected: dict, boost_config: dict | None = None,
                  installed_commit: str | None = None) -> dict:
    """Notebook start-up check: requested ref, resolved commit, loaded boost module, boost config fingerprint,
    notebook version and artifact root. Raises on any mismatch (never falls back to another ref)."""
    from .._version import git_commit
    from ..pipeline.m6 import fingerprint
    from . import boost as B

    mod = Path(B.__file__).resolve()
    cfg = B.load_boost_config(boost_config)
    inst = installed_commit if installed_commit is not None else git_commit()
    rep = {"requested_ref": requested_ref, "resolved_commit": head, "installed_commit": inst,
           "boost_module": str(mod), "boost_code_rev": B.BOOST_CODE_REV, "boost_version": B.BOOST_VERSION,
           "boost_config_fp": fingerprint(B.BOOST_VERSION, cfg)[:12],
           "notebook_version": expected.get("notebook_version"), "artifact_root": expected.get("artifact_root")}
    bad = []
    if not _is_sha(head):
        bad.append(f"resolved commit {head!r} is not a SHA")
    if _is_sha(requested_ref) and not str(head).startswith(requested_ref):
        bad.append(f"HEAD {head} is not the requested commit {requested_ref}")
    if expected.get("commit") and not str(head).startswith(str(expected["commit"])):
        bad.append(f"HEAD {head} != expected commit {expected['commit']}")
    if str(inst).endswith("+dirty"):
        bad.append(f"checkout has uncommitted changes ({inst})")
    elif _is_sha(str(inst)) and _is_sha(head) and not str(inst).startswith(str(head)[:12]):
        bad.append(f"imported code is at {inst}, not at HEAD {head} (restart the runtime)")
    if expected.get("code_root") and not str(mod).startswith(str(Path(expected["code_root"]).resolve())):
        bad.append(f"gems_bench.p6.boost loaded from {mod}, not from {expected['code_root']}")
    if expected.get("code_rev") and B.BOOST_CODE_REV != expected["code_rev"]:
        bad.append(f"boost code rev {B.BOOST_CODE_REV} != expected {expected['code_rev']}")
    rep["problems"] = bad
    if bad:
        raise RuntimeError("start-up check failed:\n  " + "\n  ".join(bad) + f"\n{json.dumps(rep, indent=1)}")
    return rep


# =============================================================================== preflight (Colab side)
def g02_attachment(bst) -> dict:
    """``preflight/g02_attachment_audit.json``: resolution, manifest hash, per-unit rasters (path, keys, shapes,
    window, grid), missing files, provenance and the compatibility decision. Attach computes nothing."""
    from ..m9.research.g02_io import list_runs

    gcf, hmb = bst.cfg["geology"], float(bst.cfg["preflight"]["hash_max_mb"])
    g = bst.g02()
    out = {"requested_run_id": gcf["g02_run_id"], "keys": gcf["keys"], "g02f": gcf.get("g02f")}
    if g is None:
        return {**out, "decision": "UNAVAILABLE", "status": BLOCKED, "error": getattr(bst, "g02_error", "")}
    a = dict(getattr(g, "attach_info", {}) or {})
    st = g.store
    runs = list_runs(Path(a["parent"])) if a.get("parent") else pd.DataFrame()
    exp = grid_meta(bst.p6.ctx.data)
    man = (st.load_json("manifests/manifest.json") or {}).get("artifacts") or {}
    units, schemas, problems = [], {}, []
    for u in bst.units():
        key = f"{u.task}_{u.zone}"
        rel = f"rasters/units/{key}.npz"
        p = st.path(rel)
        row = {"unit": u.key, "raster": str(p), "exists": p.exists()}
        if not p.exists():
            units.append({**row, "status": BLOCKED, "missing": [rel], "features_present": []})
            continue
        _, meta = st.load_raster_keys(rel, [])
        names = meta.get("keys") or {}
        shapes = {names.get(k, k): v for k, v in npz_shapes(p).items() if k != "__meta__"}
        why = []
        if (meta.get("task"), int(meta.get("zone", -1))) != (u.task, int(u.zone)):
            why.append("task / zone metadata differ from the unit")
        if meta.get("labels_stored"):
            why.append("labels stored in the raster")
        if [int(x) for x in meta.get("window", [])] != unit_window(u):
            why.append(f"window {meta.get('window')} != unit crop {unit_window(u)}")
        bad = sorted(k for k, (s, _) in shapes.items() if tuple(s) != unit_shape(u))
        if bad:
            why.append(f"array shape != unit crop {unit_shape(u)}: {bad[:5]}")
        gs, gw = align_status(exp, meta.get("grid") or {})
        if gs == BLOCKED:
            why += gw
        refp = (man.get(f"processed/refs_{key}.json") or {}).get("params") or {}
        present = [f for f, rk in gcf["keys"].items() if rk in shapes]
        schemas[u.key] = sorted(shapes)
        units.append({**row, "status": BLOCKED if why else PASS, "problems": why, "grid_check": gs, "grid_notes": gw,
                      "sha256": file_hash(p, hmb), "keys": sorted(shapes), "shapes": {k: list(s) for k, (s, _) in
                                                                                       shapes.items()},
                      "features_present": present,
                      "missing": [f"{rel}:{rk}" for f, rk in gcf["keys"].items() if rk not in shapes],
                      "backfill": bool(meta.get("backfill", False)), "mode": meta.get("mode"),
                      "labels_stored": meta.get("labels_stored"),
                      "normalization_reference": refp.get("reference", "unknown"), "reference_px": refp.get("ref_px"),
                      "window": meta.get("window"), "grid": meta.get("grid")})
        problems += [f"{u.key}: {w}" for w in why]
    if len({tuple(v) for v in schemas.values()}) > 1:
        problems.append("raster key sets differ between units (feature schema not uniform)")
    have = [x for x in units if x["exists"]]
    full = all(len(x["features_present"]) == len(gcf["keys"]) for x in units)
    decision = "INCOMPATIBLE" if problems else ("COMPATIBLE" if have and full else "PARTIAL")
    return {**out, "resolved_run_id": a.get("run_id"), "resolved_by": a.get("resolved_by"),
            "source_path": str(st.root), "mode": a.get("mode"), "fingerprint": a.get("fingerprint"),
            "manifest_sha256": a.get("manifest_sha256"), "runs_on_disk": runs.to_dict("records") if len(runs) else [],
            "zones_available": sorted({x["unit"] for x in have}), "feature_families": sorted(set().union(
                *[set(x.get("keys") or []) for x in have])) if have else [],
            "units": units, "problems": problems, "decision": decision,
            "status": BLOCKED if decision != "COMPATIBLE" else PASS,
            "consequence": {"COMPATIBLE": "S1-S3 can run", "PARTIAL": "S1-S3 BLOCKED until the missing units are "
                            "backfilled (geology.backfill_missing)", "INCOMPATIBLE": "G02 features unusable: fix the "
                            "listed problems; no backfill can repair a mismatched run"}[decision],
            "g02f_reason": "EXCLUDED: label-selected fusion weight, never persisted as a raster"}


def g02_backfill_plan(bst, att: dict) -> dict:
    """``preflight/g02_backfill_plan.json``: what an opt-in backfill would compute, its dependencies, measured unit
    times and output sizes. Computes nothing (the input-drift check fingerprints the inputs only)."""
    gcf = bst.cfg["geology"]
    plan = {"switch": "BOOST_CONFIG['geology']['backfill_missing']", "enabled": bool(gcf.get("backfill_missing")),
            "max_units_per_call": int(gcf["backfill_max_units"]), "executes_in_preflight": False,
            "label_independent": True, "rewrites_existing_rasters": False,
            "resume": "per-tile MapStore under processed/tmp_<unit>; finished rasters are skipped"}
    g = bst.g02()
    if g is None or att.get("decision") == "INCOMPATIBLE":
        return {**plan, "status": BLOCKED, "reason": att.get("error") or att.get("problems")}
    have = set(g.raster_units())
    todo = [u for u in bst.units() if f"{u.task}_{u.zone}" not in have]
    sec = [float((g.store.load_json(f"processed/unit_{k}.json") or {}).get("seconds", np.nan)) for k in sorted(have)]
    nk = max((len(x.get("keys") or []) for x in att.get("units", [])), default=0)
    nbytes = 2 if "16" in str(g.cfg["save"]["raster_dtype"]) else 4
    try:
        drift = g.input_drift()
    except Exception as e:
        drift = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    try:                                                     # P6.1 fix: the run's RECORDED channels vs src.has();
        rec = (g.store.load_json("config/inputs.json") or {}).get("input_fingerprints") or {}
        cached = set(g.src.names())                          # names() lists cached .npy stems only, so the virtual
        chans = [c for c in rec if not c.startswith("band:") and c != "m9pred"]   # g01:edge / g01:surface (derived
        miss_ch = [c for c in chans if not g.src.has(c)]     # from raw bands on demand) were falsely "missing"
        registry_only = [c for c in chans if c not in cached and g.src.has(c)]
    except Exception as e:
        miss_ch, registry_only = [f"channel listing failed: {type(e).__name__}: {e}"], []
    est_h = len(todo) * float(np.nanmax(sec)) / 3600 if sec and np.isfinite(np.nanmax(sec)) else np.nan
    left = bst.p6.budget.left_h("wb", "cpu")
    units = [{"unit": u.key, "output": f"{att.get('source_path')}/rasters/units/{u.task}_{u.zone}.npz",
              "refs": f"processed/refs_{u.task}_{u.zone}.json", "done_marker": f"processed/done_raster_{u.task}_{u.zone}",
              "px": int(np.prod(unit_shape(u))), "est_output_mb": round(nk * float(np.prod(unit_shape(u))) * nbytes / 1e6,
                                                                       1)} for u in todo]
    blocked = (not drift.get("ok")) or bool(miss_ch)
    status = BLOCKED if blocked else ("NOTHING_TO_DO" if not todo else ("READY" if plan["enabled"] else "PENDING"))
    return {**plan, "status": status, "units": units, "input_drift": drift, "missing_channels": miss_ch,
            "registry_only_channels": registry_only,
            "registry_note": "virtual channels (computed from raw bands) are absent from the cached-layer listing by "
                             "construction; available through M9Source.has(): not a blocker",
            "measured_unit_seconds": [round(s, 0) for s in sec],
            "estimate_cpu_h_upper": est_h, "estimate_basis": "max measured FULL unit evaluation (metrics included); a "
            "raster-only backfill is cheaper but unmeasured", "wb_cpu_h_left": left,
            "fits_budget": bool(np.isfinite(est_h) and est_h <= left),
            "ram_note": "same order as one G02 unit evaluation (unmeasured for raster-only); high-RAM CPU runtime",
            "next": "set BOOST_CONFIG['geology']['backfill_missing'] = True and run the 3B-B1 cell"
            if status in ("PENDING", "READY") else ""}


def dem_coverage(bst) -> pd.DataFrame:
    """``preflight/dem_coverage.csv``: is the M7 100 m mosaic local, its bands, covered share per unit / footprint,
    and whether any candidate uses it (none does)."""
    from ..m7.dem_morph import DemMosaic, mosaic_path

    dc = bst.cfg["preflight"]["dem"]
    m7 = bst.p6.m9.m7
    p = mosaic_path(m7)
    if not p.exists() and dc.get("stage_mosaic"):
        m7.stage_in([p.relative_to(m7.local).as_posix()])
    base = {"mosaic": str(p), "source": DEM_SOURCE, "used_in_training": False, "used_in_inference": False}
    if not p.exists():
        return pd.DataFrame([{**base, "scope": "*", "status": "ABSENT", "note": "mosaic not local (preflight.dem."
                              "stage_mosaic = true copies it once; W4 / M7 sections 05-06 create it)"}])
    mos = DemMosaic(p)
    cov = mos.coverage(float(dc["min_coverage"])) & np.asarray(bst.p6.ctx.data.valid, bool)
    bands = [b for b in dc["candidate_bands"] if mos.has(b)]
    rows = [{**base, "scope": "full_region", "covered_px": int(cov.sum()),
             "covered_share_of_valid": float(cov.sum() / max(int(np.asarray(bst.p6.ctx.data.valid).sum()), 1)),
             "bands_available": ";".join(bands), "bands_missing": ";".join(b for b in dc["candidate_bands"]
                                                                            if b not in bands), "status": PASS}]
    valid = np.asarray(bst.p6.ctx.data.valid, bool)
    for u in bst.units():
        v = valid[u.sl]
        rows.append({**base, "scope": u.key, "covered_px": int(cov[u.sl].sum()),
                     "covered_share_of_valid": float(cov[u.sl].sum() / max(int(v.sum()), 1)),
                     "bands_available": ";".join(bands), "status": PASS if cov[u.sl].any() else WARNING})
    return pd.DataFrame(rows)


def feature_contract(bst, att: dict, dem: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(``feature_inventory``, ``feature_compatibility``, ``feature_provenance``): every W2 / boost feature with its
    construction contract and its availability read from the artifacts, per unit and at full region."""
    p6, hmb = bst.p6, float(bst.cfg["preflight"]["hash_max_mb"])
    root = p6.ws.root
    data = p6.ctx.data
    gm = grid_meta(data)
    w2f = list((bst.w2_inputs().get("candidate") or {}).get("features") or [])
    prov = p6.ws.load_table("provenance", "stack")
    pfp = root / "stack" / "features" / "pf_full.npz"
    pf_keys = set(npz_shapes(pfp)) if pfp.exists() else set()
    g02_have = {x["unit"]: set(x.get("features_present") or []) for x in att.get("units", [])}
    full_score = root / "stack" / "maps" / "full_score.tif"
    g02_full = root / "stack" / "features" / "g02_full"
    dem_cov = {r.scope: r.covered_px for r in dem.itertuples()} if len(dem) and "covered_px" in dem else {}
    units = bst.units()
    labels_hash = hashlib.sha256(np.packbits(np.asarray(data.labels, bool)).tobytes()).hexdigest()

    def avail(f, u):
        fam = CATALOG.get(f, {}).get("family", "dem" if f.startswith("dem_") else "")
        if fam == "cnn":
            if not len(prov) or "unit" not in prov:
                return None
            r = prov[prov.unit == u.key]
            cols = ["cnn_b0", "cnn_b1"] if f == "cnn_disagree" else [f]
            return bool(len(r)) and all(c in r and str(r[c].iloc[0]) not in ("absent", "nan") for c in cols)
        if fam == "g02":
            return f in g02_have.get(u.key, set())
        if fam == "potential_field":
            return f in pf_keys
        if fam == "known":
            return True
        if fam == "dem":
            return bool(dem_cov.get(u.key, 0))
        return None

    def full(f):
        fam = CATALOG.get(f, {}).get("family", "dem" if f.startswith("dem_") else "")
        return {"cnn": "available" if full_score.exists() else "unverified (no stack/maps/full_score.tif yet)",
                "g02": ("tiles present (stack/features/g02_full)" if g02_full.exists() else
                        "computable, not computed (stack.g02_full: heaviest export step)"),
                "potential_field": "available" if f in pf_keys else "absent",
                "known": "available (data.labels)",
                "dem": "available (mosaic)" if dem_cov else "absent"}.get(fam, "unknown")

    def source_hash(f):
        fam = CATALOG.get(f, {}).get("family", "dem" if f.startswith("dem_") else "")
        if fam == "potential_field":
            return file_hash(pfp, hmb)
        if fam == "known":
            return f"labels:{labels_hash[:16]}"
        if fam == "g02":
            return att.get("manifest_sha256")
        if fam == "cnn":
            return "managed by the M9 / gems3 map inventory (provenance asserted per unit)"
        return None

    dem_names = [f"dem_{b}" for b in bst.cfg["preflight"]["dem"]["candidate_bands"]]
    rows = []
    for f in list(CATALOG) + dem_names:
        c = CATALOG.get(f, {"family": "dem", "source": DEM_SOURCE, "construction": f"M7 mosaic band {f[4:]}",
                            "units": "band units (slope deg, curvature 1/m, densities)", "labels": "none",
                            "train": "optional adapter boost_audit.dem_features", "inference": "same mosaic"})
        av = {u.key: avail(f, u) for u in units}
        rows.append({"feature_id": f, **c, "ranked_in_domain": f in ST.RANKED,
                     "preprocessing": _RANK if f in ST.RANKED else "none (raw value)",
                     "missing_policy": "never zero-filled: KeyError / BLOCKED (DEM: NaN outside coverage)",
                     "grid_shape": json.dumps(gm["shape"]), "crs_epsg": gm["crs_epsg"],
                     "transform": json.dumps(gm["transform"]), "resolution_m": abs(gm["transform"][0]),
                     "in_w2": f in w2f, "units_available": sum(bool(v) for v in av.values()),
                     "units_unverified": sum(v is None for v in av.values()), "n_units": len(units),
                     "per_unit": json.dumps(av), "full_region": full(f), "content_hash": source_hash(f),
                     "fold_specific": c["family"] in ("cnn", "known", "g02")})
    bands = list(getattr(data, "band_names", []) or [])
    for b in bands:
        rows.append({"feature_id": f"band:{b}", "family": "geodawn_raw", "source": "GeoDAWN cube",
                     "construction": "raw channel", "in_w2": False, "units_available": len(units),
                     "n_units": len(units), "full_region": "available",
                     "note": "present in the data cube; not a W2 / boost feature (mag / grav enter via potential_field)"})
    INV = pd.DataFrame(rows)
    gk = list(bst.cfg["geology"]["keys"])
    specs = {f"{n}": w2f + [a for a in add if a not in w2f] for n, add in bst.cfg["geology"]["variants"].items()}
    specs.update({n: w2f for n in bst.cfg["robustness"]["variants"]})
    comp = []
    for name, need in specs.items():
        miss = [f"{u.key}:{f}" for f in need for u in units if avail(f, u) is False]
        unv = [f"{u.key}:{f}" for f in need for u in units if avail(f, u) is None]
        inf = {f: full(f) for f in need}
        inf_bad = [f for f, v in inf.items() if v in ("absent", "unknown")]
        inf_warn = [f for f, v in inf.items() if v.startswith(("computable", "unverified"))]
        st = BLOCKED if (miss or inf_bad or not need) else (WARNING if (unv or inf_warn) else PASS)
        comp.append({"candidate": name, "features": ";".join(need), "status": st, "missing_training": ";".join(miss),
                     "unverified": ";".join(unv), "inference_absent": ";".join(inf_bad),
                     "inference_to_compute": ";".join(inf_warn),
                     "note": "uses G02 rasters" if set(need) & set(gk) else ""})
    comp.append({"candidate": "D1/D2 (on the stage-C base)", "features": "base features + optional g02c layer",
                 "status": PASS, "note": "D2 drops its geology term when g02c is absent (never zero-filled)"})
    comp.append({"candidate": "dem_* (no candidate)", "status": "NOT_IN_ANY_CANDIDATE",
                 "note": "optional adapter only; a DEM variant needs its own predeclared stage"})
    COMP = pd.DataFrame(comp)
    PROV = {"version": PREFLIGHT_VERSION, "written": time.strftime("%Y-%m-%dT%H:%M:%S"), "grid": gm,
            "w2_features": w2f, "w2_inputs": bst.w2_inputs(), "g02": {k: att.get(k) for k in (
                "resolved_run_id", "resolved_by", "fingerprint", "manifest_sha256", "decision", "mode")},
            "pf_full": {"path": str(pfp), "hash": file_hash(pfp, hmb), "keys": sorted(pf_keys)},
            "labels_sha256": labels_hash, "dem_integration": DEM_INTEGRATION,
            "rule": "an optional feature unavailable at inference is excluded from its candidate, never zero-filled"}
    return INV, COMP, PROV


def raster_alignment(bst, att: dict) -> pd.DataFrame:
    """``preflight/raster_alignment_audit.csv``: every map a stage reads or an export writes against the grid."""
    p6 = bst.p6
    root = p6.ws.root
    exp = grid_meta(p6.ctx.data)
    rows = []
    for u in bst.units():
        p = root / "stack" / "maps" / f"{u.task}_{u.zone}.npy"
        if not p.exists():
            rows.append({"artifact": "w2_heldout_map", "unit": u.key, "path": str(p), "status": BLOCKED,
                         "notes": "absent"})
            continue
        a = np.load(p, mmap_mode="r")
        ok = tuple(a.shape) == unit_shape(u)
        rows.append({"artifact": "w2_heldout_map", "unit": u.key, "path": str(p), "shape": list(a.shape),
                     "expected_shape": list(unit_shape(u)), "dtype": str(a.dtype), "status": PASS if ok else BLOCKED,
                     "notes": "" if ok else "shape != unit crop"})
    for x in att.get("units", []):
        rows.append({"artifact": "g02_unit_raster", "unit": x["unit"], "path": x["raster"],
                     "status": x["status"] if x["exists"] else BLOCKED,
                     "notes": "; ".join((x.get("problems") or []) + (x.get("grid_notes") or [])) or
                     ("" if x["exists"] else "absent (backfill required)")})
    pfp = root / "stack" / "features" / "pf_full.npz"
    if pfp.exists():
        for k, (s, dt) in npz_shapes(pfp).items():
            ok = list(s) == exp["shape"]
            rows.append({"artifact": "pf_full", "unit": "*", "path": f"{pfp}:{k}", "shape": list(s), "dtype": dt,
                         "expected_shape": exp["shape"], "status": PASS if ok else BLOCKED})
    tifs = [("w2_full_score", root / "stack" / "maps" / "full_score.tif"),
            ("fallback_submission", p6.m9_path(f"submissions/{p6.cfg['probes']['fallback_file']}"))]
    sf = (bst.w2_inputs().get("candidate") or {}).get("submission_file")
    if sf:
        tifs.append(("w2_submission", root / "submissions" / sf))
    from ..m7.dem_morph import mosaic_path

    tifs.append(("dem_mosaic", mosaic_path(p6.m9.m7)))
    for name, p in tifs:
        if not Path(p).exists():                             # not read by stages A-D: absence is not a blocker here
            rows.append({"artifact": name, "unit": "*", "path": str(p), "status": WARNING,
                         "notes": "absent locally (not staged or never written)"})
            continue
        import rasterio

        with rasterio.open(p) as s:
            got = {"shape": [int(s.height), int(s.width)], "crs_epsg": s.crs.to_epsg() if s.crs else None,
                   "transform": [float(x) for x in tuple(s.transform)[:6]]}
        st, why = align_status(exp, got)
        rows.append({"artifact": name, "unit": "*", "path": str(p), "shape": got["shape"], "crs": got["crs_epsg"],
                     "transform": json.dumps(got["transform"]), "status": st, "notes": "; ".join(why)})
    return pd.DataFrame(rows)


def parity_metadata(bst) -> list[dict]:
    """Training vs inference contract of the W2 feature schema and of the export path (metadata only)."""
    p6 = bst.p6
    root = p6.ws.root
    feats = list((bst.w2_inputs().get("candidate") or {}).get("features") or [])
    rp = root / "stack" / "features" / "rows.parquet"
    cols = None
    if rp.exists():
        import pyarrow.parquet as pq

        cols = list(pq.read_schema(rp).names)
    mp = root / "stack" / "models" / "stack_full.joblib"
    n_model = None
    if mp.exists():
        import joblib

        n_model = getattr(joblib.load(mp), "n_features_in_", None)
    rows = []
    st, why = feature_order_check(feats, cols, n_model)
    rows.append({"check": "feature_schema", "feature": "*", "train": f"rows.parquet ({len(cols or [])} columns)",
                 "inference": f"stack_full.joblib n_features_in_={n_model}", "status": st if feats else BLOCKED,
                 "note": "; ".join(why) or ("candidate.json lists no features" if not feats else "same order: "
                                            "stack.assemble(raw, mask, names) at training and prediction")})
    note = {"cnn": (WARNING, "training: held-out maps of models that never saw the unit's labels; inference: "
                             "full-region maps (every label). Rank-normalised in both; distribution shift expected "
                             "(numeric check: preflight.parity_numeric)"),
            "g02": (WARNING, "training references: the unit's training geography; inference: whole footprint, "
                             "resumable tiles (stack.g02_full)"),
            "potential_field": (PASS, "identical source array (pf_full.npz); ranked inside the domain in both"),
            "known": (WARNING, "by design: training sees the permitted known map, inference every mapped fault; "
                               "same units (km) and caps")}
    for f in feats:
        c = CATALOG.get(f)
        if c is None:
            rows.append({"check": "construction", "feature": f, "status": BLOCKED,
                         "note": "no inference construction in the feature contract"})
            continue
        s, n = note[c["family"]]
        if f == "cnn_disagree":
            n = "ranks inside the unit evaluation mask (training) vs the valid footprint (inference); not re-ranked"
        if f == "mg_orient_agree":
            s, n = PASS, "identical source array; not ranked (fixed [0, 1] scale)"
        rows.append({"check": "construction", "feature": f, "train": c["train"], "inference": c["inference"],
                     "normalization": _RANK if f in ST.RANKED else "none", "units": c["units"], "status": s, "note": n})
    ev = p6.m9.m7.cfg["evaluation"]
    dom = ev.get("budget_domain")
    rows.append({"check": "budget_domain", "feature": "*", "train": f"m7.elig: {dom}, template "
                 f"{bool(ev.get('include_template_mask'))}", "inference": "gems3.submission.eligible_mask(data, 0, "
                 "'eligible') (template included)", "status": PASS if dom == "eligible" and
                 ev.get("include_template_mask") else BLOCKED,
                 "note": "evaluation and export must share the area-budget domain"})
    rows.append({"check": "ring", "feature": "*", "train": "ring 0", "inference": "ring 0", "status": PASS})
    rows.append({"check": "known_masking", "feature": "*", "train": "R.apply(known=u.known)",
                 "inference": "R.apply(known=data.labels)", "status": PASS,
                 "note": "every mapped fault visible at full region; the scorer masks known pixels"})
    src = p6.cfg["stack"]["full_region"]["area_frac_source"]
    sf = (bst.w2_inputs().get("candidate") or {}).get("submission_file")
    side = root / "submissions" / Path(sf).with_suffix(".json").name if sf else None
    recorded = (json.loads(side.read_text(encoding="utf-8")).get("frac_source") if side is not None and side.exists()
                else None)
    mixed = src == "geo" or str(recorded) == "geo"
    rows.append({"check": "w2_export_cell", "feature": "w2", "train": "nested choice (rep, frac) per task",
                 "inference": f"W2 full_region: rep from disc, frac from {recorded or src} (config now {src})",
                 "status": WARNING if mixed else PASS,
                 "note": "the historical W2 file uses a (rep, frac) no nested evaluation selected "
                         "(submission_provenance.json); P6.1 exports go through export_parity.build_recipe"
                 if mixed else ""})
    rows.append({"check": "boost_export_cell", "feature": "*", "train": "nested (blend, rep, frac) of the primary task",
                 "inference": f"area_frac_source = {bst.cfg['selection']['area_frac_source']}",
                 "status": PASS if bst.cfg["selection"]["area_frac_source"] == "matched" else BLOCKED})
    rows.append({"check": "fusion_weight", "feature": "*", "train": "champion per-zone w (fusion term of each blend)",
                 "inference": "champion params_all w (one weight for the full region)", "status": WARNING,
                 "note": "same rule as the published fallback export"})
    rows.append({"check": "tiling", "feature": "*", "train": "per-pixel LightGBM on unit crops",
                 "inference": "per-pixel LightGBM on the full grid (no tiles, no overlap, no stitching)",
                 "status": PASS, "note": "CNN full-region maps come from upstream M9 / P5 inference (not re-audited); "
                 "G02 full maps are tiled with the run's halo"})
    rows.append({"check": "checkpoint", "feature": "w2", "train": "stack/models/stack_full.joblib",
                 "inference": "refit on every primary-task row (boost export refits its own model)",
                 "status": PASS if mp.exists() else WARNING, "note": "" if mp.exists() else "W2 full model absent locally"})
    return rows


def parity_numeric(bst) -> list[dict]:
    """Colab only (``preflight.parity_numeric``): training vs inference values of the CNN features on the discovery
    units. Loads the full-region maps once; G02 needs ``stack/features/g02_full`` (never computed here)."""
    pc = bst.cfg["preflight"]["parity"]
    maps, _ = bst.p6.view.full_maps()
    valid = np.asarray(bst.p6.ctx.data.valid, bool)
    r_full = [ST.rank_in(maps[0], valid), ST.rank_in(maps[1], valid)]
    rows = []
    for u in bst.disc_units():
        b0, b1 = bst.cnn(u)
        ev = u.ev
        tr = {"cnn_b0": ST.rank_in(b0, ev), "cnn_b1": ST.rank_in(b1, ev)}
        tr["cnn_disagree"] = np.abs(tr["cnn_b0"] - tr["cnn_b1"])
        inf = {"cnn_b0": r_full[0][u.sl], "cnn_b1": r_full[1][u.sl]}
        inf["cnn_disagree"] = np.abs(inf["cnn_b0"] - inf["cnn_b1"])
        for f in tr:
            r = rank_agreement(tr[f], inf[f], ev, float(pc["top_frac"]), seed=int(bst.p6.cfg["seed"]))
            rows.append({"check": "numeric", "feature": f, "unit": u.key, **r, "status": classify_agreement(r, pc),
                         "note": "training (held-out unit map) vs inference (full-region map crop), both ranked"})
        u.release()
    return rows


def evaluation_contract(bst) -> dict:
    """``preflight/evaluation_contract.json``: one metric, one budget domain, one protocol key per unit, separate
    discovery / geographic objectives, nested selection; checks on every saved stage grid."""
    from ..gems3.metric_opt import nested_select
    from . import boost as B

    G = pd.DataFrame({"zone": [0, 1, 2], "frac": [0.01] * 3, "tp_w": [3.0, 1.0, 2.0], "fp_w": [5.0, 2.0, 1.0],
                      "fn_w": [1.0, 4.0, 2.0], "n_truth": [10, 10, 10]})
    _, s = nested_select(G, ["frac"])
    ref = float(B.dti(G.tp_w.sum(), G.fp_w.sum(), G.fn_w.sum()))
    metric_ok = abs(float(s.get("dti_crossfit", np.nan)) - ref) < 1e-6
    ev = bst.p6.m9.m7.cfg["evaluation"]
    tol = float(bst.cfg["preflight"]["budget_tolerance"])
    checks, grids = [], {}
    for stage in ("benchmark", "geology", "robustness", "hybrid"):
        p = bst.root / stage / "grid.csv"
        if p.exists():
            grids[stage] = pd.read_csv(p)
    allg = pd.concat([g.assign(stage=k) for k, g in grids.items()], ignore_index=True) if grids else pd.DataFrame()
    pc = protocol_consistency(allg[allg.candidate != "fallback_published"] if len(allg) else allg)
    checks.append({"check": "one protocol key + eligible count per unit across all stages and candidates",
                   "status": (PASS if (pc.status == PASS).all() else BLOCKED) if len(pc) else "PENDING",
                   "detail": pc.to_dict("records")})
    viol = []
    for k, g in grids.items():
        base = None
        if k == "hybrid":
            dec = bst.read_json("hybrid", "decision") or {}
            base = dec.get("base")
        v = budget_violations(g[g.candidate != "fallback_published"], tol, base)
        if len(v):
            viol += v.assign(stage=k).to_dict("records")
    checks.append({"check": f"selected eligible area <= nominal frac (+{tol:.0%}); hybrids <= base area",
                   "status": (BLOCKED if viol else PASS) if grids else "PENDING", "detail": viol[:50]})
    geo = []
    for k in ("geology", "robustness", "hybrid"):
        p = bst.root / k / "ablation.csv"
        if p.exists():
            A = pd.read_csv(p)
            run = A[A.get("status", pd.Series("RUN", index=A.index)).fillna("RUN") != "BLOCKED"] if len(A) else A
            geo += [{"stage": k, "candidate": r.candidate} for r in run.itertuples()
                    if not np.isfinite(pd.to_numeric(getattr(r, "geo_dti", np.nan), errors="coerce"))]
    checks.append({"check": "geographic DTI computed (not a proxy) for every evaluated candidate",
                   "status": BLOCKED if geo else PASS, "detail": geo})
    return {"version": PREFLIGHT_VERSION, "metric": "DTI = TP_w / (TP_w + 0.2 FP_w + 0.8 FN_w) of the official "
            "distance-weighted components (m7.protocol.score_row -> u.components); implementation unchanged",
            "metric_check": {"status": PASS if metric_ok else BLOCKED, "nested_select": s.get("dti_crossfit"),
                             "boost_dti": ref},
            "pooled": "DTI of the components summed over zones (never a mean of per-zone DTI)",
            "per_zone": "reported separately; worst zone = min over held-out zones",
            "representations": bst.cfg["grid"]["reps"], "area_fracs": bst.cfg["grid"]["area_fracs"],
            "blends": bst.cfg["grid"]["blends"],
            "budget_domain": {"evaluation": f"m7.elig(u, 0): {ev.get('budget_domain')}, template "
                              f"{bool(ev.get('include_template_mask'))}",
                              "export": "gems3.submission.eligible_mask(data, 0, 'eligible')"},
            "known_mask": "permitted known map per unit (training / evaluation); every mapped fault at full region",
            "objectives": {"discovery": "disc_official units (withheld systems)", "geographic": "geo folds",
                           "gate": "discovery gain; geographic DTI only as a floor (>= selection.geo_floor)"},
            "nested": "gems3.metric_opt.nested_select: each zone's (blend, rep, frac) chosen on the other zones, "
                      "scored on the held-out zone (tests/test_p6_boost.py::test_nested_never_selects_on_the_"
                      "heldout_zone); no zone-specific rule or threshold",
            "systems": "m7.protocol.system_hits per truth system (a system counts once however many segments hit it)",
            "checks": checks, "status": worst([PASS if metric_ok else BLOCKED] + [c["status"] for c in checks])}


def submission_provenance(bst) -> dict:
    """``preflight/submission_provenance.json``: what the uploaded W2 file (and the fallback) actually is."""
    from . import probes

    p6 = bst.p6
    root = p6.ws.root
    cand = bst.w2_inputs().get("candidate") or {}
    pa = cand.get("params_all") or {}
    sf = cand.get("submission_file")
    path = root / "submissions" / sf if sf else None
    side = None
    if path is not None and path.with_suffix(".json").exists():
        try:
            side = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            side = None
    v = provenance_verdict(side, pa.get(bst.prim) or {}, pa.get("geo") or {}, bool(path is not None and path.exists()),
                           float(bst.cfg["preflight"]["budget_tolerance"]))
    sha = file_hash(path, 1e9) if path is not None else None
    log = probes.read_log(p6)
    logged = log[log.sha256.astype(str) == str(sha)].to_dict("records") if len(log) and sha else []
    M = bst.load_table("matched_comparison", "benchmark")
    cell = {}
    if len(M) and v["rep"] is not None:
        lam = float(v["blend"]) if v["blend"] is not None else 1.0
        cw = "w2" if lam == 1.0 else ("hybrid" if lam == float(bst.cfg["benchmark"]["hybrid_blend"]) else None)
        for t in (bst.prim, "geo"):
            m = M[(M.candidate == cw) & (M.rep == v["rep"]) & np.isclose(M.frac, float(v["frac"])) & (M.task == t)]
            cell[t] = float(m.dti.iloc[0]) if len(m) else None
    fb = p6.m9_path(f"submissions/{p6.cfg['probes']['fallback_file']}")
    return {"w2": {"file": sf, "path": str(path) if path else None, "sha256": sha, **v,
                   "nested_disc_choice": pa.get(bst.prim), "nested_geo_choice": pa.get("geo"),
                   "matched_cell_dti": cell or "PENDING (stage A not run)", "leaderboard_log_rows": logged,
                   "recorded": {k: cand.get(k) for k in ("disc_dti", "disc_worst", "geo_dti")},
                   "interpretation": ("the recorded W2 numbers are nested choices; the uploaded file is the cell "
                                      f"{v['rep']} @ {v['frac']} (blend {v['blend']}), which "
                                      + ("matches a nested choice" if v["selected_cell"] else
                                         "NO nested evaluation selected: its offline score is the matched cell above"))
                   if v["status"] == "VERIFIED" else "UNVERIFIED: the file's construction cannot be reconstructed"},
            "fallback": {"file": fb.name, "exists": fb.exists(), "sha256": file_hash(fb, 1e9) if fb.exists() else None},
            "status": PASS if v["status"] == "VERIFIED" and v["selected_cell"] else WARNING if v["status"] == "VERIFIED"
            else BLOCKED}


def run_preflight(bst) -> dict:
    """All preflight audits (each isolated: one failure is recorded and the others still run)."""
    from .._version import provenance

    t0 = time.perf_counter()
    bst.set_status("preflight", "RUNNING")
    res, att, dem = {}, {}, pd.DataFrame()

    def step(name, fn):
        try:
            r = fn()
            res[name] = r if isinstance(r, str) else PASS
            return r
        except Exception as e:
            res[name] = f"FAILED: {type(e).__name__}: {e}"
            bst.say(f"preflight {name}: {res[name]}")
            return None

    att = step("g02_attachment", lambda: g02_attachment(bst)) or {}
    if att:
        bst.save_json(att, "preflight", "g02_attachment_audit")
        res["g02_attachment"] = att["status"]
    plan = step("g02_backfill_plan", lambda: g02_backfill_plan(bst, att))
    if plan:
        bst.save_json(plan, "preflight", "g02_backfill_plan")
        res["g02_backfill_plan"] = BLOCKED if plan["status"] == BLOCKED else (WARNING if plan["status"] in
                                                                             ("PENDING", "READY") else PASS)
    d = step("dem_coverage", lambda: dem_coverage(bst))
    if d is not None:
        dem = d
        bst.save_table(dem, "dem_coverage", "preflight")
        res["dem_coverage"] = WARNING if (dem.status == "ABSENT").any() else PASS
    fc = step("feature_contract", lambda: feature_contract(bst, att, dem))
    if fc:
        bst.save_table(fc[0], "feature_inventory", "preflight")
        bst.save_table(fc[1], "feature_compatibility", "preflight")
        bst.save_json(fc[2], "preflight", "feature_provenance")
        res["feature_contract"] = worst(fc[1].status)
    ra = step("raster_alignment", lambda: raster_alignment(bst, att))
    if ra is not None:
        bst.save_table(ra, "raster_alignment_audit", "preflight")
        res["raster_alignment"] = worst(ra.status)
    pr = step("train_inference_parity", lambda: parity_metadata(bst))
    if pr is not None:
        if bst.cfg["preflight"].get("parity_numeric"):
            pr += step("parity_numeric", lambda: parity_numeric(bst)) or []
        st = worst(r["status"] for r in pr)
        bst.save_json({"version": PREFLIGHT_VERSION, "status": st, "rows": pr,
                       "numeric": bool(bst.cfg["preflight"].get("parity_numeric")),
                       "rule": "a BLOCKED row on a candidate's feature (or '*') refuses its promotion and export"},
                      "preflight", "train_inference_parity")
        bst.save_table(pd.DataFrame(pr), "train_inference_parity", "preflight")
        res["train_inference_parity"] = st
    ec = step("evaluation_contract", lambda: evaluation_contract(bst))
    if ec:
        bst.save_json(ec, "preflight", "evaluation_contract")
        res["evaluation_contract"] = ec["status"]
    sp = step("submission_provenance", lambda: submission_provenance(bst))
    if sp:
        bst.save_json(sp, "preflight", "submission_provenance")
        res["submission_provenance"] = sp["status"]
    failed = [k for k, v in res.items() if str(v).startswith("FAILED")]
    overall = BLOCKED if failed else worst(res.values())
    out = {"version": PREFLIGHT_VERSION, "status": overall, "areas": res, "failed": failed,
           "seconds": round(time.perf_counter() - t0, 1), "code": provenance(),
           "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
    bst.save_json(out, "preflight", "manifest")
    bst.set_status("preflight", "FAILED" if failed else "COMPLETE", overall=overall, areas=res)
    bst.p6.push()
    return out


# =============================================================================== reports
_KIND = {".csv": "table", ".parquet": "table", ".json": "manifest/decision", ".png": "figure", ".svg": "figure",
         ".npy": "array", ".npz": "array", ".tif": "raster", ".joblib": "model", ".md": "report"}


def artifact_registry(bst, hash_max_mb: float = 64) -> pd.DataFrame:
    """Every file under ``boost/`` plus the frozen W2 inputs and the submissions, with kind, size and hash."""
    rows = []
    roots = [("boost", bst.root)]
    st = bst.p6.ws.root / "stack"
    extra = [st / "candidate.json", st / "features" / "rows.parquet", *sorted((st / "maps").glob("*.npy"))]
    sub = bst.p6.ws.root / "submissions"
    extra += sorted(sub.glob("*.tif")) + sorted(sub.glob("*.json")) if sub.exists() else []
    files = [(n, p) for n, r in roots if r.exists() for p in sorted(r.rglob("*")) if p.is_file()]
    files += [("w2_input" if "stack" in p.parts else "submission", p) for p in extra if p.exists()]
    for origin, p in files:
        rel = p.relative_to(bst.p6.ws.root).as_posix()
        stage = rel.split("/")[1] if rel.startswith("boost/") and rel.count("/") >= 2 else origin
        s = p.stat()
        rows.append({"path": rel, "origin": origin, "stage": stage, "kind": _KIND.get(p.suffix, "other"),
                     "bytes": s.st_size, "modified": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(s.st_mtime)),
                     "hash": file_hash(p, hash_max_mb)})
    return pd.DataFrame(rows)


def _ablation(bst, stage: str) -> pd.DataFrame:
    p = bst.root / stage / "ablation.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _js_dict(v) -> dict:
    try:
        return {str(k): x for k, x in json.loads(v).items()} if isinstance(v, str) else (v or {})
    except (TypeError, ValueError):
        return {}


def stage_status(bst) -> pd.DataFrame:
    from .boost import STAGES

    rows = []
    for st in ("preflight",) + tuple(STAGES):
        s = bst.read_json(st, "status") or {}
        m = bst.read_json(st, "manifest") or {}
        rows.append({"stage": st, "status": s.get("status", "COMPLETE" if m.get("status") == "done" else "PENDING"),
                     "fingerprint": (s.get("fingerprint") or m.get("fingerprint") or "")[:12],
                     "reason": s.get("reason", ""), "updated": s.get("updated", m.get("written", "")),
                     "seconds": m.get("seconds"), "source_commit": (m.get("code") or {}).get("git_commit")})
    return pd.DataFrame(rows)


def readiness(bst) -> dict:
    """Answers to the eleven report questions from saved outputs; PENDING / BLOCKED where a stage has not run."""
    from . import boost as B

    v = bst.read_json("benchmark", "verdict") or {}
    S = bst.load_table("nested_selection", "benchmark")
    rec = bst.cfg["champion"]["recorded"]
    w2m = S[S.space == "w2_space|blend+rep+frac"] if len(S) else S
    A = {k: _ablation(bst, k) for k in ("geology", "robustness", "hybrid")}
    stat = stage_status(bst).set_index("stage").status.to_dict()

    def row(stage, cand):
        a = A[stage]
        r = a[a.candidate == cand] if len(a) else a
        return r.iloc[0] if len(r) else None

    def val(r, k):
        try:
            x = float(r.get(k)) if r is not None else np.nan
        except (TypeError, ValueError):
            return None
        return x if np.isfinite(x) else None

    def geo_q(a, b):
        ra, rb = row("geology", a), row("geology", b)
        if stat.get("geology") == "PENDING" or rb is None:
            return {"status": "PENDING", "answer": None}
        if str(rb.get("decision")) == "BLOCKED":
            return {"status": BLOCKED, "answer": None, "evidence": rb.get("reasons")}
        va, vb = val(ra, "disc_dti"), val(rb, "disc_dti")
        if va is None or vb is None:
            return {"status": "PENDING", "answer": None, "evidence": f"no discovery DTI for {a} or {b}"}
        return {"status": "ANSWERED", "answer": bool(vb - va >= float(bst.cfg["geology"]["accept"]["min_gain"])),
                "delta_disc": vb - va, "decision": rb.get("decision"), "evidence": f"geology/ablation.csv {b} vs {a}"}

    sel = bst.read_json("selection", "decision") or {}
    chosen = sel.get("chosen")
    cands = bst.load_table("candidates", "selection")
    c = cands[cands.name == chosen] if len(cands) and chosen else pd.DataFrame()
    cr = c.iloc[0] if len(c) else None
    par = parity_by_feature(bst.read_json("preflight", "train_inference_parity"))
    q = {}
    q["q1_w2_better_than_fallback_matched"] = ({"status": "ANSWERED", "answer": v.get("w2_gain_is_ranking"),
                                                "ranking_gain": v.get("ranking_gain"),
                                                "zones_improved": v.get("zones_improved"),
                                                "cell_win_share": v.get("cell_win_share"),
                                                "evidence": "benchmark/verdict.json"} if v else {"status": "PENDING"})
    q["q2_ranking_vs_representation_budget"] = ({"status": "ANSWERED", "components": v.get("components"),
                                                 "largest": v.get("largest_component"),
                                                 "evidence": "benchmark/ranking_gain.csv"} if v else
                                                {"status": "PENDING"})
    q["q3_g02c_improves_discovery"] = geo_q("S0", "S1")
    q["q4_g02a_incremental"] = geo_q("S1", "S2")
    q["q5_g01_persistence_independent"] = geo_q("S2", "S3")
    z2 = []
    for k, a in A.items():
        for r in a.itertuples() if len(a) else []:
            pz = _js_dict(getattr(r, "disc_per_zone", None))
            if "2" in pz:
                z2.append({"stage": k, "candidate": r.candidate, "zone2_dti": float(pz["2"]),
                           "vs_w2_recorded": float(pz["2"]) - float(bst.cfg["champion"]["zone_disc"][2])})
    q["q6_zone2_improving"] = {"status": "ANSWERED" if z2 else "PENDING", "rows": z2,
                               "w2_zone2_recorded": bst.cfg["champion"]["zone_disc"][2]}
    sysr = [{"stage": k, "candidate": r.candidate, "gained": getattr(r, "systems_gained", None),
             "lost": getattr(r, "systems_lost", None)} for k, a in A.items() for r in (a.itertuples() if len(a) else [])]
    if len(w2m):
        sysr.insert(0, {"stage": "benchmark", "candidate": "w2_space", "gained": w2m.iloc[0].get("systems_gained"),
                        "lost": w2m.iloc[0].get("systems_lost")})
    q["q7_additional_systems"] = {"status": "ANSWERED" if sysr else "PENDING", "rows": sysr,
                                  "reference": "fallback held-out maps (systems_vs)"}
    fp = []
    for k in ("geology", "robustness", "hybrid"):
        dec = bst.read_json(k, "decision") or {}
        p, g = bst.root / k / "per_zone.csv", bst.root / k / "grid.csv"
        if not (p.exists() and g.exists() and dec.get("base")):
            continue
        P, G = pd.read_csv(p), pd.read_csv(g)
        for name in sorted(set(G.candidate) - {dec["base"]}):
            per = P[(P.space == f"{name}|blend+rep+frac") & (P.task == bst.prim)]
            if len(per):
                fp.append({"stage": k, "candidate": name, "fp_weight_change_vs_base": B._fp_increase(
                    G, per, name, dec["base"])})
    q["q8_false_positives"] = {"status": "ANSWERED" if fp else "PENDING", "rows": fp,
                               "rule": "relative FP weight at the candidate's held-out choices vs its base"}
    floor = float(bst.cfg["selection"]["geo_floor"])
    if not sel:
        q["q9_geographic_preserved"] = {"status": "PENDING", "answer": None}
    elif cr is None or chosen == "p6_stack (W2)":
        q["q9_geographic_preserved"] = {"status": "ANSWERED", "answer": "no boosted candidate passed; W2 stays",
                                        "w2_geo_recorded": rec.get("geo_dti"), "floor": floor}
    else:
        g = val(cr, "geo_dti")
        q["q9_geographic_preserved"] = {"status": "ANSWERED", "answer": bool(g is not None and g >= floor),
                                        "geo_dti": g, "floor": floor}
    spec = _js_dict(cr.get("spec")) if cr is not None else {}
    names = list((spec.get("base", spec) if spec.get("kind") == "hybrid" else spec).get("features") or [])
    ok, why = parity_ok(par, names) if names else (False, "no boosted candidate selected")
    q["q10_full_region_valid"] = {"status": "ANSWERED" if par and names else "PENDING", "answer": ok if names else
                                  None, "why": why}
    q["q11_leaderboard_submission_justified"] = {
        "status": "ANSWERED" if sel else "PENDING", "answer": bool(sel.get("replace_champion") and
                                                                   sel.get("slot_worthy") and sel.get("submission")
                                                                   and not (sel.get("validation") or {})
                                                                   .get("checks_failed")),
        "replace_champion": sel.get("replace_champion"), "slot_worthy": sel.get("slot_worthy"),
        "submission": sel.get("submission"), "rule": "gate PASS + slot gain >= selection.slot_min_gain + validated file"}
    return {"written": time.strftime("%Y-%m-%dT%H:%M:%S"), "recorded_baselines": {
        "fallback": {"disc_dti": 0.161576, "geo_dti": 0.182832}, "w2": rec}, "stage_status": stat, "questions": q,
        "note": "measured values only; PENDING = stage not run, BLOCKED = stage ran but the variant could not"}


def build_report(bst) -> dict:
    """``boost/reports/*`` from saved outputs only (cheap; safe to rerun after any stage)."""
    S = bst.load_table("nested_selection", "benchmark")
    rows = []
    rec = bst.cfg["champion"]["recorded"]
    rows.append({"stage": "recorded", "candidate": "P5 B0+B1 fallback", "disc_dti": 0.161576, "geo_dti": 0.182832,
                 "decision": "fallback"})
    rows.append({"stage": "recorded", "candidate": "P6 W2 LightGBM", **rec, "decision": "champion"})
    for r in S.itertuples() if len(S) else []:
        rows.append({"stage": "benchmark", "candidate": r.space, "disc_dti": getattr(r, "disc_dti", np.nan),
                     "disc_worst": getattr(r, "disc_worst", np.nan), "geo_dti": getattr(r, "geo_dti", np.nan),
                     "systems_gained": getattr(r, "systems_gained", np.nan),
                     "systems_lost": getattr(r, "systems_lost", np.nan), "decision": "diagnostic"})
    for k in ("geology", "robustness", "hybrid"):
        a = _ablation(bst, k)
        for r in a.itertuples() if len(a) else []:
            rows.append({"stage": k, "candidate": r.candidate, "disc_dti": getattr(r, "disc_dti", np.nan),
                         "disc_worst": getattr(r, "disc_worst", np.nan), "geo_dti": getattr(r, "geo_dti", np.nan),
                         "systems_gained": getattr(r, "systems_gained", np.nan),
                         "systems_lost": getattr(r, "systems_lost", np.nan), "decision": getattr(r, "decision", ""),
                         "reasons": getattr(r, "reasons", "")})
    SUM = pd.DataFrame(rows)
    pz = []
    for k in ("benchmark", "geology", "robustness", "hybrid"):
        p = bst.root / k / ("nested_per_zone.csv" if k == "benchmark" else "per_zone.csv")
        if p.exists():
            P = pd.read_csv(p)
            keep = [c for c in ("task", "space", "zone", "blend", "rep", "frac", "dti_heldout", "systems") if c in P]
            pz.append(P[keep].assign(stage=k))
    PZ = pd.concat(pz, ignore_index=True) if pz else pd.DataFrame()
    sy = []
    cov = bst.root / "benchmark" / "system_coverage.csv"
    if cov.exists():
        C = pd.read_csv(cov)
        sy.append(C[C.kind == "nested_choice"].assign(stage="benchmark"))
    for k in ("geology", "robustness", "hybrid"):
        p = bst.root / k / "systems.csv"
        if p.exists() and p.stat().st_size:
            sy.append(pd.read_csv(p).assign(stage=k))
    SY = pd.concat(sy, ignore_index=True) if sy else pd.DataFrame()
    ST_ = stage_status(bst)
    RD = readiness(bst)
    for df, n in ((SUM, "boost_summary"), (PZ, "per_zone_comparison"), (SY, "system_recovery"),
                  (ST_, "experiment_status")):
        bst.save_table(df, n, "reports")
    bst.save_json(RD, "reports", "leaderboard_readiness")
    M = bst.load_table("matched_comparison", "benchmark")
    if len(M):
        bst._fig_area(M, "reports")
    REG = artifact_registry(bst)
    bst.save_table(REG, "artifact_registry", "reports")
    q = RD["questions"]
    md = [f"# P6 boost report ({RD['written']})", "",
          "Measured values only. PENDING = stage not run; BLOCKED = stage ran, variant could not.", "",
          "## Stage status", "", md_table(ST_), "## Candidates", "",
          md_table(SUM, ["stage", "candidate", "disc_dti", "disc_worst", "geo_dti", "systems_gained", "systems_lost",
                         "decision"]), "## Questions", ""]
    for k, x in q.items():
        ans = x.get("answer", x.get("components", x.get("rows")))
        md.append(f"* **{k}**: {x.get('status')} — {json.dumps(ans, default=str)[:400]}")
    pre = bst.read_json("preflight", "manifest") or {}
    md += ["", "## Preflight", "", f"status: {pre.get('status', 'PENDING')}", "",
           "```", json.dumps(pre.get("areas", {}), indent=1), "```", "",
           f"Figure: `reports/dti_by_area.png` ({'written' if len(M) else 'PENDING: stage A'}). "
           f"Registry: `reports/artifact_registry.csv` ({len(REG)} files)."]
    bst.p6.ws.save_text("\n".join(md) + "\n", f"{bst.sub('reports')}/boost_report.md")
    bst.p6.push()
    return {"summary": SUM, "status": ST_, "readiness": RD, "registry": REG}
