"""P6 final submission: zone-stitched full-resolution map of the selected policy, official GeoTIFF, validation gate.

Official format (DrivenData problem description, ``docs/GEMS_OFFICIAL_METRIC_AUDIT.md``): one float32 band, EPSG:32611,
100 m, the training bounds of ``example_submission.tif``, values in [0, 1], NaN outside the bounds. The template is
inspected at run time (``inspect_spec``); a template that differs from the feature grid stops the stage.

Construction ``zone_stitched`` (the validated one). The P6.1 recovery run measured that the historical full-region W2
export does NOT reproduce its validated construction (``recovery/export_parity/train_inference_parity.csv``: w2_score
and cnn_disagree BLOCKED on geo units; features ranked inside each unit during training, inside the whole footprint at
inference). Here every geographic fold zone k (``valid & fold_map == k``, disjoint by construction) is predicted from
the maps that were evaluated on that zone: B0 / B1 out-of-fold maps, the cross-fitted held-out stacker map (W2 or a
boost variant), or the W1 seed families, all ranked inside the zone exactly as in evaluation. The policy cell (rep,
frac, blend from ONE all-zone selection) is applied per zone on the full-region eligible domain (valid & template
finite & not a mapped fault), so the area fraction holds zone by zone and overall. No held-out label enters inference:
the maps are predictions; mapped faults are used only as the official mask. Parity is re-checked per zone (the
stitched score crop vs the evaluation construction, Spearman >= ``parity_min_spearman``).

    out = run(fg)      # winner + strongest baseline: GeoTIFF, manifest, validation, preview, diagnostics, sha256
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import export_parity as EP

READY, NOT_READY = "READY_FOR_UPLOAD", "NOT_READY"


# =============================================================================== pure
def zone_partition(fold_map: np.ndarray, valid: np.ndarray, ids) -> tuple[np.ndarray, dict]:
    """Zone id per pixel (-1 outside every listed zone or outside ``valid``) + coverage facts."""
    fm = np.asarray(fold_map)
    v = np.asarray(valid, bool)
    z = np.full(fm.shape, -1, np.int16)
    for k in ids:
        z[v & (fm == int(k))] = int(k)
    n = int(v.sum())
    unc = int((v & (z < 0)).sum())
    return z, {"valid_px": n, "uncovered_px": unc, "uncovered_frac": unc / max(n, 1),
               "zone_px": {int(k): int((z == int(k)).sum()) for k in ids}}


def stitch(shape, parts) -> np.ndarray:
    """Full grid from (slices, crop mask, crop array) of disjoint zones; 0 elsewhere."""
    out = np.zeros(shape, np.float32)
    for sl, m, a in parts:
        view = out[sl]
        view[m] = np.asarray(a, np.float32)[m]
    return out


def split_cell(cell: dict) -> tuple[str, str]:
    """(grid source, grid candidate) of a benchmark deploy cell (``<source>:<candidate>``)."""
    src, _, orig = str(cell["candidate"]).partition(":")
    if not orig:
        raise ValueError(f"deploy cell candidate {cell['candidate']!r} lacks its grid prefix")
    return src, orig


def seeds_weight(orig: str) -> tuple[str, float]:
    """``seeds|w0.5`` -> ("seeds", 0.5)."""
    m, _, w = orig.partition("|w")
    if not w:
        raise ValueError(f"seeds cell {orig!r}: no fusion weight")
    return m, float(w)


def zone_score(kind: str, cell: dict, src: dict) -> np.ndarray:
    """Ranking score of one zone from its rank maps (``src``: fusion, cnn_b0, cnn_b1, stack, seeds_b0, seeds_b1),
    the same expressions as the evaluation (boost ``eval_variant`` / ``eval_score``, seeds ``evaluate``)."""
    from ..gems3 import representation as R

    _, orig = split_cell(cell)
    if kind == "fusion":
        return src["fusion"]
    if kind == "cnn":
        return src[orig]
    if kind == "stack":
        lam = float(cell["blend"])
        return ((1.0 - lam) * src["fusion"] + lam * src["stack"]).astype(np.float32)
    if kind == "seeds":
        return R.fuse(src["seeds_b0"], src["seeds_b1"], seeds_weight(orig)[1])
    raise ValueError(f"deploy kind {kind!r} has no zone-stitched builder")


def submission_path(sub_dir: Path, name: str, recipe_fp: str) -> tuple[Path, bool]:
    """(path, reuse). ``<name>.tif`` when free; the same file when its sidecar carries this recipe; otherwise
    ``<name>__<fp8>.tif``: an existing submission is never overwritten."""
    p = Path(sub_dir) / f"{name}.tif"
    for cand in (p, p.with_name(f"{name}__{recipe_fp[:8]}.tif")):
        if not cand.exists():
            return cand, False
        side = cand.with_suffix(".json")
        try:
            if side.exists() and json.loads(side.read_text(encoding="utf-8")).get("recipe_fp") == recipe_fp:
                return cand, True
        except (OSError, json.JSONDecodeError):
            pass
    raise FileExistsError(f"{p.name} and its fingerprinted variant exist for other recipes: refusing to overwrite")


def validation_status(groups: dict) -> dict:
    """PASS only when every hard check of every group passed; the failed checks are listed by group."""
    failed = [f"{g}: {c['check']}" for g, cs in groups.items() for c in cs if c.get("hard", True) and not c["ok"]]
    return {"status": "PASS" if not failed else "FAIL", "failed": failed, "n_checks": sum(len(c) for c in
                                                                                       groups.values())}


def readiness(validation: dict, selection: dict, name_role: str) -> dict:
    """READY_FOR_UPLOAD requires a PASS validation and a selection decision; anything else blocks the artifact."""
    why = []
    if validation.get("status") != "PASS":
        why.append(f"validation {validation.get('status')}: {validation.get('failed')}")
    if not selection.get("decision"):
        why.append("no selection decision")
    return {"status": READY if not why else NOT_READY, "role": name_role, "reasons": why, "uploaded": False,
            "upload": "manual only (docs/p6/P6_FULL_GPU.md section 5)"}


# =============================================================================== Colab: specification
def inspect_spec(fg) -> dict:
    """Official template facts vs the configured expectation and the feature grid (raises on a mismatch)."""
    data = fg.p6.ctx.data
    t = data.template_profile
    tr = t["transform"]
    exp = fg.cfg["submission"]["expected"]
    crs = t["crs"]
    epsg = crs.to_epsg() if hasattr(crs, "to_epsg") else None
    h, w = int(t["height"]), int(t["width"])
    bounds = [tr.c, tr.f + tr.e * h, tr.c + tr.a * w, tr.f]
    nod = t.get("nodata")
    spec = {"template": str(data.paths.raw("submission")) if hasattr(data.paths, "raw") else "example_submission.tif",
            "epsg": epsg, "res_m": [abs(tr.a), abs(tr.e)], "shape": [h, w], "bounds": bounds,
            "dtype": t.get("dtype"), "count": t.get("count"), "nodata": None if nod is None else
            ("nan" if isinstance(nod, float) and np.isnan(nod) else nod),
            "template_valid_frac": float(np.asarray(data.template_valid, bool).mean()),
            "data_valid_frac": float(np.asarray(data.valid, bool).mean()), "template_matches_features":
            bool(data.template_matches()), "expected": exp,
            "sources": {"problem_description": "drivendata.org/competitions/306 page/967 (format)",
                        "metric_audit": "docs/GEMS_OFFICIAL_METRIC_AUDIT.md",
                        "reference_solution": "github.com/drivendataorg/gems-prize-reference-solution (no scorer)",
                        "local_scorer": "gems_bench/gems3/official.py (FastScorer; parity tests tests/test_gems3.py)"}}
    bad = []
    if epsg != int(exp["epsg"]):
        bad.append(f"template EPSG {epsg} != {exp['epsg']}")
    if not all(abs(r - float(exp["res_m"])) < 1e-6 for r in spec["res_m"]):
        bad.append(f"template resolution {spec['res_m']} != {exp['res_m']} m")
    if not spec["template_matches_features"]:
        bad.append("example_submission.tif grid differs from the feature grid (resample before export)")
    spec["problems"] = bad
    fg.save_json(spec, "submission", "official_spec")
    if bad:
        raise ValueError(f"official specification check failed: {bad}")
    return spec


# =============================================================================== Colab: map construction
def stack_dir(fg, src: str, orig: str) -> Path:
    """Directory of the cross-fitted held-out stacker maps of a grid candidate (W2: stack/maps; a boost variant: its
    unique ``<stage>/maps/<name>_<fp8>`` cache, the base variant reusing W2)."""
    root, bst = fg.p6.ws.root, fg.bst
    if src == "boost_a" and orig in ("w2", "hybrid"):
        return root / "stack" / "maps"
    stage = {"boost_b": "geology", "boost_c": "robustness"}.get(src)
    if stage is None:
        raise ValueError(f"{src}:{orig}: no stacker maps for this grid")
    dec = bst.read_json(stage, "decision") or {}
    if orig == dec.get("base"):
        return root / "stack" / "maps"
    ds = [d for d in sorted((bst.root / stage / "maps").glob(f"{orig}_*")) if (d / "geo_0.npy").exists()]
    if len(ds) != 1:
        raise FileNotFoundError(f"{src}:{orig}: expected one cross-fitted map directory, found {[d.name for d in ds]}")
    return ds[0]


def variant_features(fg, src: str, orig: str) -> list[str]:
    """Feature list of a boost variant from its stage ablation table (``run_variants`` summary rows)."""
    stage = {"boost_b": "geology", "boost_c": "robustness"}[src]
    A = fg.bst.load_table("ablation", stage)
    r = A[A.candidate.astype(str) == orig] if len(A) and "candidate" in A else A
    if not len(r) or "features" not in r or not isinstance(r.features.iloc[0], str):
        raise FileNotFoundError(f"{src}:{orig}: feature list not found in boost/{stage}/ablation.csv")
    return list(json.loads(r.features.iloc[0]))


def zone_sources(fg, u, kind: str, cell: dict, w_all: float, stack_d: Path | None, seeds_cand: dict | None) -> dict:
    """Rank maps of geo unit ``u`` ranked inside ``u.ev`` (the validated construction)."""
    from ..gems3 import representation as R
    from . import seeds as SE

    b0, b1 = fg.bst.cnn(u)
    r0, r1 = R.rank01(b0, u.ev), R.rank01(b1, u.ev)
    out = {"cnn_b0": r0, "cnn_b1": r1, "fusion": R.fuse(r0, r1, float(w_all))}
    if kind == "stack":
        m = np.load(Path(stack_d) / f"{u.task}_{u.zone}.npy")
        if m.shape != u.ev.shape:
            raise ValueError(f"stacker map {u.key} shape {m.shape} != unit crop {u.ev.shape}")
        out["stack"] = fg.bst.stack_rank(u, m)
    if kind == "seeds":
        meth, _ = seeds_weight(split_cell(cell)[1])
        fam = SE.family_maps(fg.p6, u, seeds_cand["members"], meth.endswith("_tta"))
        if fam is None:
            raise FileNotFoundError(f"{u.key}: seed family maps missing")
        out["seeds_b0"], out["seeds_b1"] = fam["b0"], fam["b1"]
    return out


def eval_score(fg, u, kind: str, cell: dict, src: dict, seeds_cand: dict | None) -> np.ndarray:
    """The score exactly as the benchmark evaluated it on this unit (per-zone champion w for the fusion)."""
    if kind in ("fusion", "stack"):
        e = dict(src, fusion=fg.bst.fusion(u))
        return zone_score(kind, cell, e)
    return zone_score(kind, cell, src)


def build_map(fg, row: dict) -> dict:
    """Zone-stitched binary map + score of one benchmark row (its all-zone deploy cell), with per-zone parity."""
    from ..gems3 import representation as R
    from ..gems3 import submission as SUB

    p6, bst = fg.p6, fg.bst
    data = p6.ctx.data
    cell = row["deploy_cell"] if isinstance(row["deploy_cell"], dict) else json.loads(row["deploy_cell"])
    kind = row["deploy"]
    src_name, orig = split_cell(cell)
    sh = p6.m9.sh
    sigma = float(sh["evaluation"]["ridge_sigma_px"])
    rep = sh["baseline"]["representations"][cell["rep"]]
    w_all = float(bst.sel[bst.prim][p6.cfg["probes"]["champion"]]["params_all"]["w"])
    sd = stack_dir(fg, src_name, orig) if kind == "stack" else None
    sc = json.loads((p6.ws.root / "seeds" / "candidate.json").read_text(encoding="utf-8")) if kind == "seeds" else None
    geo = [u for u in bst.units() if u.task == "geo"]
    zid, cov = zone_partition(p6.ctx.folds.fold_map, data.valid, [u.zone for u in geo])
    el_full = SUB.eligible_mask(data, 0, "eligible")
    labels = np.asarray(data.labels, bool)
    pc = fg.cfg["submission"]
    qs, ss, par, zrows = [], [], [], []
    for u in geo:
        m = np.asarray(zid[u.sl] == u.zone) & u.ev
        src = zone_sources(fg, u, kind, cell, w_all, sd, sc)
        s = zone_score(kind, cell, src)
        el = el_full[u.sl] & m
        q = R.apply(rep, s, m, float(cell["frac"]), known=labels[u.sl], sigma_px=sigma, eligible=el)
        se = eval_score(fg, u, kind, cell, src, sc)
        ag = EP.agreement(s, se, m, (0.01, 0.05), 200_000, int(p6.cfg["seed"]))
        qe = bst.binary(u, se, cell["rep"], float(cell["frac"]))
        ma = EP.map_agreement(qe, np.asarray(q) > 0)
        ok = bool(np.isfinite(ag.get("spearman", np.nan)) and ag["spearman"] >= float(pc["parity_min_spearman"]))
        par.append({"unit": u.key, "check": "zone score vs evaluation construction", **ag, "status": "PASS" if ok else
                    ("WARNING" if kind in ("fusion", "stack") and float(cell.get("blend", 1)) < 1.0 else "BLOCKED"),
                    "note": "fusion w: all-zone (deploy) vs per-zone champion (evaluation)" if kind in ("fusion",
                                                                                                       "stack") else ""})
        par.append({"unit": u.key, "check": "binary map vs evaluation map", **ma, "status": "PASS" if (ma["jaccard"] or
                                                                                                       0) >= 0.5
                    else "WARNING", "note": "eligibility differs by design: full-region eligible excludes mapped "
                                            "faults (official mask); geo-unit evaluation counts them as truth"})
        pos = np.asarray(q) > 0
        zrows.append({"zone": int(u.zone), "zone_px": int(m.sum()), "eligible_px": int(el.sum()),
                      "positive_px": int(pos.sum()), "frac_of_eligible": float(pos.sum() / max(int(el.sum()), 1)),
                      "positive_on_known_px": int((pos & labels[u.sl]).sum())})
        qs.append((u.sl, m, q))
        ss.append((u.sl, m, s))
        u.release()
    Q, S_ = stitch(data.shape, qs), stitch(data.shape, ss)
    return {"q": Q, "score": S_, "zone_id": zid, "coverage": cov, "parity": pd.DataFrame(par),
            "zones": pd.DataFrame(zrows), "cell": cell, "kind": kind, "w_all": w_all,
            "stack_dir": str(sd.relative_to(p6.ws.root)) if sd is not None else None, "seeds": sc}


def recipe(fg, row: dict, built: dict, objective: str) -> dict:
    """Export recipe (export_parity contract): rep, frac and blend of ONE all-zone selection."""
    from ..artifacts.atomic import sha256_file
    from .boost_audit import file_hash

    p6, bst = fg.p6, fg.bst
    cell, kind = built["cell"], built["kind"]
    root = p6.ws.root
    if kind == "stack":
        d = root / built["stack_dir"]
        schema = bst.w2_features() if built["stack_dir"] == "stack/maps" else variant_features(fg, *split_cell(cell))
        maps = {p.name: sha256_file(p) for p in sorted(d.glob("geo_*.npy"))}
    elif kind == "seeds":
        schema = [m for f in sorted(built["seeds"]["members"]) for m in built["seeds"]["members"][f]]
        maps = {"seeds_candidate": file_hash(root / "seeds" / "candidate.json")}
    else:
        schema = ["cnn_b0", "cnn_b1"]
        maps = {}
    sh = p6.m9.sh
    sid = f"{row['name']}|{objective}|all_zones"
    blend = float(cell["blend"]) if kind == "stack" else 0.0
    return EP.build_recipe(
        candidate=row["name"], kind=f"zone_stitched_{kind}",
        policy={"selection_id": sid, "space": row.get("space") or "", "objective": objective, "rep": cell["rep"],
                "frac": float(cell["frac"]), "blend": blend, "grid_candidate": cell["candidate"],
                "grid_blend": float(cell["blend"])},
        model={"kind": kind, "features": list(schema), "refit": False, "maps": built["stack_dir"] or kind,
               "note": "held-out maps per geographic zone; no model is refitted or re-run"},
        sources={"zone_maps": maps, "benchmark_grid": file_hash(root / fg.cfg["grids"][split_cell(cell)[0]]["path"]),
                 "oof_maps": {m: str(p6.ctx.zoo5.get(m, {}).get("oof")) if hasattr(p6.ctx, "zoo5") else m
                              for m in list(p6.m9.models)[:2]}},
        feature_schema=list(schema),
        preprocessing={"construction": "zone_stitched", "ranking": "gems3.representation.rank01 inside each geo fold "
                       "zone (valid & fold_map == k): the validated construction", "fusion_w": built["w_all"],
                       "sigma_px": float(sh["evaluation"]["ridge_sigma_px"]),
                       "representation": sh["baseline"]["representations"][cell["rep"]],
                       "eligibility": "gems3.submission.eligible_mask(data, 0, 'eligible') per zone",
                       "known": "data.labels (official pixel-exact mask) passed to R.apply", "ring_px": 0,
                       "uncovered": "valid pixels outside every geo zone are written 0"})


# =============================================================================== Colab: write + validate
def write_file(fg, name: str, q: np.ndarray, rec: dict, ledger: list) -> dict:
    """Finalise (template NaN pattern), write the GeoTIFF and its sidecar (never overwriting another recipe)."""
    from ..artifacts.atomic import atomic_save, atomic_write_json
    from ..gems3 import submission as SUB
    from ..submission import write_submission

    data = fg.p6.ctx.data
    ov = fg.cfg["submission"]["outside_valid"]
    sub = fg.root / "submissions"
    sub.mkdir(parents=True, exist_ok=True)
    path, reuse = submission_path(sub, name, rec["fingerprint"])
    if not reuse:
        arr = SUB.finalise(q, data, ov)
        atomic_save(path, lambda t: write_submission(arr, data, Path(t)))
        pol = rec["policy"]
        atomic_write_json(path.with_suffix(".json"), json.loads(json.dumps(
            {"submission": path.name, "recipe_fp": rec["fingerprint"], "features": rec["feature_schema"],
             "parameters": {"rep": pol["rep"], "frac": pol["frac"], "blend": pol["blend"], "ring": 0},
             "recipe": rec, "area_ledger": ledger, "outside_valid": ov, "uploaded": False,
             "written": time.strftime("%Y-%m-%dT%H:%M:%S")}, default=str)))
    return {"path": path, "reused": reuse}


def validate(fg, path: Path, rec: dict, built: dict) -> dict:
    """Official checks (``gems3.submission.validate``), the export audit (``export_parity.audit_export``) and the
    zone-stitched checks (coverage, parity, finite values). Any hard failure = FAIL."""
    import rasterio

    from ..gems3 import submission as SUB

    data = fg.p6.ctx.data
    pc = fg.cfg["submission"]
    off = []
    try:
        facts = SUB.validate(path, data, pc["outside_valid"])
        off.append({"check": "gems3.submission.validate (format, CRS, grid, range, NaN pattern)", "ok": True})
    except AssertionError as e:
        facts = {"error": str(e)}
        off.append({"check": "gems3.submission.validate (format, CRS, grid, range, NaN pattern)", "ok": False,
                    "detail": str(e)[:2000]})
    el = SUB.eligible_mask(data, 0, "eligible")
    aud = EP.audit_export(path, rec, data, el, float(pc["budget_tolerance"]), int(pc["expected"]["epsg"]))
    t = data.template_profile
    with rasterio.open(path) as src:
        a = src.read(1)
        nod = src.nodata
        ext = [{"check": "nodata tag is NaN", "ok": nod is not None and np.isnan(nod), "detail": str(nod)},
               {"check": "bounds equal template", "ok": src.transform == t["transform"] and src.width == t["width"]
                and src.height == t["height"]},
               {"check": "resolution", "ok": all(abs(abs(r) - float(pc["expected"]["res_m"])) < 1e-6 for r in
                                                  src.res), "detail": str(src.res)}]
    fin = np.isfinite(a)
    ext += [{"check": "no infinite value", "ok": not bool(np.isinf(a).any())},
            {"check": "finite everywhere the template is finite", "ok": bool(fin[np.asarray(data.template_valid,
                                                                                            bool)].all())},
            {"check": f"uncovered valid pixels <= {pc['max_uncovered_frac']:g}",
             "ok": built["coverage"]["uncovered_frac"] <= float(pc["max_uncovered_frac"]),
             "detail": json.dumps(built["coverage"], default=str)},
            {"check": "zone-stitched parity (no BLOCKED zone)", "ok": not (built["parity"].status == "BLOCKED").any(),
             "detail": json.dumps(built["parity"][["unit", "check", "status"]].to_dict("records"))},
            {"check": "positive pixels non-empty", "ok": bool((np.nan_to_num(a) > 0).any())}]
    groups = {"official": off, "audit": [{**c, "hard": c.get("hard", True)} for c in aud["checks"]],
              "zone_stitched": ext}
    st = validation_status(groups)
    return {**st, "groups": groups, "official_facts": facts, "audit": {k: aud.get(k) for k in
                                                                       ("status", "failed", "sha256", "raster",
                                                                        "area_frac_eligible", "recipe_fp")},
            "file": path.name, "validated": time.strftime("%Y-%m-%dT%H:%M:%S")}


def preview(fg, name: str, q: np.ndarray, title: str) -> Path:
    """Down-sampled preview: prediction (red), mapped faults (black), outside the template (white)."""
    import matplotlib.pyplot as plt

    data = fg.p6.ctx.data
    step = max(1, int(np.ceil(max(data.shape) / int(fg.cfg["submission"]["preview_max_px"]))))
    tv = np.asarray(data.template_valid, bool)[::step, ::step]
    img = np.ones(tv.shape + (3,), np.float32)
    img[tv] = 0.93
    from scipy import ndimage

    kn = ndimage.maximum_filter(np.asarray(data.labels, np.uint8), size=step)[::step, ::step] > 0
    pr = ndimage.maximum_filter((np.nan_to_num(q) > 0).astype(np.uint8), size=step)[::step, ::step] > 0
    img[kn] = (0.15, 0.15, 0.15)
    img[pr] = (0.85, 0.1, 0.1)
    fig, ax = plt.subplots(figsize=(9, 9 * tv.shape[0] / max(tv.shape[1], 1)))
    ax.imshow(img, interpolation="nearest")
    ax.set(title=title, xticks=[], yticks=[])
    data_ = pd.DataFrame([{"step_px": step, "positive_px": int((np.nan_to_num(q) > 0).sum()),
                           "known_px": int(np.asarray(data.labels, bool).sum())}])
    p = fg.save_fig(fig, f"{name}_preview", data_, "submissions")
    return p


def export_one(fg, row: dict, name: str, role: str, objective: str, selection: dict) -> dict:
    """Build, write, validate one submission; manifest / validation / diagnostics / sha256 / preview."""
    from ..gems3 import submission as SUB

    data = fg.p6.ctx.data
    t0 = time.perf_counter()
    built = build_map(fg, row)
    rec = recipe(fg, row, built, objective)
    plan = EP.plan_audit(rec)
    fg.save_json({"recipe": rec, "plan_audit": plan}, "submissions", f"{name}_plan")
    if plan["status"] != "PASS":
        raise RuntimeError(f"{name}: recipe plan audit {plan}")
    ledger = [SUB.area_ledger(built["q"], data, 0, f"{row['name']} zone-stitched {rec['policy']['rep']} @ "
                                                  f"{rec['policy']['frac']:g} blend {rec['policy']['blend']:g}")]
    w = write_file(fg, name, built["q"], rec, ledger)
    path = w["path"]
    v = validate(fg, path, rec, built)
    ready = readiness(v, selection, role)
    from ..artifacts.atomic import atomic_write_text

    sha = v["audit"].get("sha256")
    stem = path.stem
    atomic_write_text(fg.root / "submissions" / f"{stem}_sha256.txt", f"{sha}  {path.name}\n")
    rasters = fg.root / "submissions" / "rasters"
    fg.save_raster({"zone_stitched_score": built["score"], "binary": built["q"],
                    "zone_id": built["zone_id"].astype(np.float32)}, f"submissions/rasters/{stem}_maps.tif")
    D = pd.concat([built["zones"].assign(kind="zone"), pd.DataFrame(ledger).assign(kind="ledger"),
                   built["parity"].assign(kind="parity")], ignore_index=True)
    fg.save_table(D, f"{stem}_diagnostics", "submissions")
    preview(fg, stem, built["q"], f"{name}: {row['name']} ({rec['policy']['rep']} @ {rec['policy']['frac']:g}) - "
                                  f"validation {v['status']}")
    from .._version import provenance

    man = {"submission": path.name, "path": str(path.relative_to(fg.root)), "role": role, "candidate": row["name"],
           "label": row.get("label"), "recipe_fp": rec["fingerprint"], "recipe": rec, "readiness": ready,
           "validation_status": v["status"], "sha256": sha, "reused": w["reused"],
           "offline": {k: row.get(k) for k in ("disc_dti", "disc_worst", "disc_worst_zone", "geo_dti", "objective")},
           "coverage": built["coverage"], "zones": built["zones"].to_dict("records"), "code": provenance(),
           "data_fingerprint": fg.data_fingerprint(), "seconds": round(time.perf_counter() - t0, 1),
           "rasters": str(rasters.relative_to(fg.root)), "uploaded": False,
           "note": "offline numbers are nested held-out estimates, not leaderboard scores"}
    fg.save_json(man, "submissions", f"{stem}_manifest")
    fg.save_json(v, "submissions", f"{stem}_validation")
    return man


def run(fg) -> dict:
    """Submission stage: official spec, winner + strongest baseline (when different), validation gate."""
    stage = "submissions"
    ms = fg.read_json("benchmark", "model_selection")
    if not ms or not ms.get("winner"):
        raise RuntimeError("submission refused: benchmark/model_selection.json has no winner (run the benchmark)")
    M = fg.load_table("candidate_metrics", "benchmark")
    rows = {r["name"]: r for r in M.to_dict("records")}
    fp = fg.fp(stage, fg.cfg["submission"], ms.get("winner"), ms.get("baseline"),
               (fg.read_json("benchmark", "manifest") or {}).get("fingerprint"))
    if fg.done(stage, fp):
        return fg.load(stage)
    t0 = fg.begin(stage, fp)
    spec = inspect_spec(fg)
    pc = fg.cfg["submission"]
    out = {"official_spec": spec, "decision": ms["decision"], "files": []}
    jobs = [(ms["winner"], pc["name"], "winner")]
    if ms.get("baseline") and ms["baseline"] != ms["winner"]:
        jobs.append((ms["baseline"], pc["baseline_name"], "baseline"))
    for cand, name, role in jobs:
        r = rows[cand]
        for k in ("deploy_cell", "disc_per_zone"):
            if isinstance(r.get(k), str) and r[k]:
                r[k] = json.loads(r[k])
        out["files"].append(export_one(fg, r, name, role, ms["objective"], ms))
    summ = [{"role": f["role"], "candidate": f["candidate"], "submission": f["submission"],
             "validation": f["validation_status"], "readiness": f["readiness"]["status"], "sha256": f["sha256"],
             "offline_disc_dti": f["offline"].get("disc_dti")} for f in out["files"]]
    fg.save_table(pd.DataFrame(summ), "submission_index", stage)
    w = out["files"][0]
    top = {"winner_file": w["submission"], "winner_status": w["readiness"]["status"], "files": summ,
           "decision": ms["decision"], "slot_worthy": ms.get("slot_worthy"), "uploaded": False,
           "next": "upload manually only when winner_status == READY_FOR_UPLOAD; then lb.record(...)"}
    fg.save_json(top, stage, "submission_manifest")
    fg.save_json({f["submission"]: {"status": f["validation_status"], "readiness": f["readiness"]} for f in
                  out["files"]}, stage, "submission_validation")
    from ..artifacts.atomic import atomic_write_text

    atomic_write_text(fg.root / stage / "submission_sha256.txt", "".join(f"{f['sha256']}  {f['submission']}\n" for f in
                                                                         out["files"]))
    stem = Path(w["submission"]).stem
    fg.save_table(fg.load_table(f"{stem}_diagnostics", stage), "submission_diagnostics", stage)
    fg.copy_canonical_preview(stem)
    fg.finish(stage, fp, t0, {"spec": spec}, top)
    return fg.load(stage)
