"""Sections 17-18: full-area inference products and validated competition GeoTIFFs (never uploaded).

Section 17 (``full_region``): resumable tiled inference of the ``core`` dense model over the whole grid
(``inference.source``: fold average of the trained CV checkpoints, or the ``full`` unit), with the production map
context (all known faults, flag 1) -> ``predictions/full/core_heads.tif`` (seg, cos 2t, sin 2t, endpoint, junction);
optional full-region traces (profile ``full_region_traces``) scored by the chosen GBM refitted on every labelled
primary-task row -> ``tracing/full_region.gpkg``.

Section 18 (``run``): always the B0/B1 champion fallback (``m7.export.export_baseline``, ring 0); the M9 champion only
when Gate 6 PROMOTES a method (``submission.export_m9_if``), rebuilt on the full region with its all-zone parameters
(area fraction from ``shared.submission.area_frac_source``, counted on ELIGIBLE pixels: valid & template-finite &
not a known fault). Writing reuses M7 / GEMS3: template = authority for shape / CRS / transform / bounds / nodata;
NaN only where the template is NaN; float32 in [0, 1]; area ledger after every step; ``gems3.submission.validate``
raises on any failed check; existing files are never overwritten. M9 adds: positive-area fraction within
[min, max] (no all-zero / all-one maps), no positive known-fault pixel, every model checkpoint sha256 re-checked
against the registry, configuration fingerprint recorded. Files failing an M9 check stay on disk but are marked
REJECTED in ``submission_index.csv``.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd


# =============================================================================== section 17
def full_dense(m9, vname: str = "core") -> dict | None:
    from .infer import device_of, full_region
    from .train import load_unit

    vcfg = m9.variant_cfg(vname)
    meta, cube, pooled = m9.cube_for(vcfg)
    ic, dc, tc = m9.cfg["inference"], vcfg["data"], vcfg["training"]
    dev = device_of(ic["device"])
    if ic["source"] == "full_model":
        units = ["full"]
    else:
        units = [f"geo_f{k}" for k in m9.train_folds()] or [f"disc_s{d}" for d in m9.train_splits()]
    models = []
    for un in units:
        lu = load_unit(m9, vname, un, dev)
        if lu is not None:
            models.append(lu[:3])
    if not models:
        return None
    lab = np.asarray(m9.ctx.data.labels, bool).astype(np.float32)
    prod = m9.cfg["validation"]["production_context"]
    P = int(dc["patch_px"])
    tile_dir = m9.ws.root / "inference" / "tiles" / f"{vname}_{prod}_{'-'.join(units)}"
    out = full_region(models, cube, pooled, np.asarray(m9.ctx.data.valid, bool), tile_dir,
                      ctx_map=lab if prod == "known" else None, flag=1.0 if prod == "known" else 0.0, P=P,
                      f=int(dc["context_factor"]), stride=int(P * float(tc["stride_frac"])), tile_px=int(ic["tile_px"]),
                      halo_px=int(ic["halo_px"]), device=dev, amp=bool(tc["amp"]), batch=int(tc["infer_batch"]),
                      log=m9.say)
    if m9.cfg["inference"].get("save_heads", True):
        m9.ws.save_raster({k: out[k] for k in ("seg", "ocos", "osin", "ep", "jn")}, f"predictions/full/{vname}_heads.tif",
                          m9.ctx.data.grid)
    else:
        m9.ws.save_raster({"seg": out["seg"]}, f"predictions/full/{vname}_heads.tif", m9.ctx.data.grid)
    return {"heads": out, "units": units, "context": prod}


def load_full_dense(m9, vname: str = "core") -> dict | None:
    p = m9.ws.root / "predictions" / "full" / f"{vname}_heads.tif"
    if not p.exists():
        return None
    import rasterio

    with rasterio.open(p) as src:
        a = src.read().astype(np.float32)
    keys = ("seg", "ocos", "osin", "ep", "jn")[:a.shape[0]]
    return {k: a[i] for i, k in enumerate(keys)}


def full_unit_for_tracing(m9):
    """Whole footprint; known faults masked pixel-exactly (traces never spend budget on them)."""
    from ..gems3.official import Unit

    H, W = m9.ctx.data.shape
    lab = np.asarray(m9.ctx.data.labels, bool)
    return Unit("full", 0, (slice(0, H), slice(0, W)), np.asarray(m9.ctx.data.valid, bool), lab, lab, False, lab)


# =============================================================================== section 18
def _frac(m9, summary: dict, method: str) -> tuple[float, str]:
    src = m9.sh["submission"]["area_frac_source"]
    if isinstance(src, (int, float)):
        return float(src), "explicit"
    task = "geo" if src == "geo" else m9.prim
    s = (summary.get(method) or {}).get(task) or (summary.get(method) or {}).get(m9.prim)
    if not s or not s.get("params_all"):
        raise KeyError(f"no all-zone parameters for {method} on {task}")
    pa = s["params_all"] if isinstance(s["params_all"], dict) else json.loads(s["params_all"])
    return float(pa["frac"]), f"{task} all-zone choice of {method}"


def full_score(m9, method: str, summary: dict) -> tuple[np.ndarray, dict]:
    """Full-region score + representation parameters of a promoted method."""
    from ..gems3 import representation as R
    from ..m8.ensemble import _paint_rank

    valid = np.asarray(m9.ctx.data.valid, bool)
    dense = load_full_dense(m9, "core")
    if dense is None:
        raise FileNotFoundError("predictions/full/core_heads.tif missing: run section 17 first")
    D = R.rank01(dense["seg"], valid)
    pa = (summary.get(method) or {}).get(m9.prim, {}).get("params_all") or {}
    pa = pa if isinstance(pa, dict) else json.loads(pa)
    if method.startswith("m9_dense_"):
        return D, {"rep": pa.get("rep", "binary")}
    S = None
    cs = m9.full_traces()
    if cs is not None and len(cs) and "score_full" in cs.table:
        S = _paint_rank(cs, cs.table.score_full.fillna(-1).to_numpy(), valid)
    if method == "m9_structured":
        if S is None:
            raise FileNotFoundError("full-region traces / scores missing (profile full_region_traces, section 17)")
        return S, {"rep": "binary"}
    w = json.loads(pa["w"]) if isinstance(pa.get("w"), str) else pa.get("w")
    ch = {"D": D}
    if S is not None:
        ch["S"] = S
    if w and w[2] > 0:
        maps, _ = m9.m7view.full_maps()
        sel = (m9.baseline_selection() or {})[m9.prim][m9.cfg["baseline"]["champion"]]
        ch["L"] = R.fuse(R.rank01(maps[0], valid), R.rank01(maps[1], valid), float(sel["params_all"]["w"]))
    s = sum(wi * ch[k] for wi, k in zip(w, "DSL") if wi > 0 and k in ch)
    return np.asarray(s, np.float32), {"rep": pa.get("rep", "binary"), "w": w}


def m9_checks(m9, info: dict) -> list[str]:
    sc = m9.cfg["submission"]
    v = info["validation"]
    fails = []
    pf = float(v.get("positive_frac_valid", 0.0))
    if pf < float(sc["min_positive_frac"]):
        fails.append(f"positive fraction {pf:.2e} < {sc['min_positive_frac']} (near-empty map)")
    if pf > float(sc["max_positive_frac"]):
        fails.append(f"positive fraction {pf:.3f} > {sc['max_positive_frac']} (near-full map)")
    if int(v.get("positive_px_on_known", 0)) > 0:
        fails.append("positive pixels on known faults (they consume no score but signal a masking error)")
    reg = m9.ws.load_table("model_registry", "reports/model_registry")
    from ..artifacts.atomic import sha256_file
    from pathlib import Path

    for r in reg[reg.variant == "core"].itertuples() if len(reg) else []:
        p = Path(str(r.checkpoint))
        if p.exists() and isinstance(getattr(r, "sha256", None), str) and sha256_file(p) != r.sha256:
            fails.append(f"checkpoint {p.name} differs from its registered sha256")
    return fails


def run(m9) -> dict:
    from ..gems3 import representation as R
    from ..gems3 import submission as S
    from ..m7 import export as E

    rows = []
    champ = m9.cfg["baseline"]["champion"]
    fb = E.export_baseline(m9.m7view_all, f"m9_fallback_{champ}", champ, ring_px=0)
    rows.append({"file": fb["submission"], "role": "fallback", "method": f"s01_{champ}",
                 "frac_of_eligible": fb["final_frac_of_eligible"], "checks_failed": ";".join(fb["validation"]["checks_failed"]),
                 "m9_checks_failed": ";".join(m9_checks(m9, fb)), "status": "VALID"})
    g6 = m9.ws.load_table("gate6", "fusion")
    if m9.cfg["submission"]["export_m9_if"] == "PROMOTE" and len(g6) and g6.decision.iloc[0] == "PROMOTE":
        method = str(g6.method.iloc[0])
        summary = m9.read_json("fusion/method_parameters.json") or {}
        try:
            score, rp = full_score(m9, method, summary)
            frac, fsrc = _frac(m9, summary, method)
            data = m9.ctx.data
            el = S.eligible_mask(data, 0, "eligible")
            rep = m9.cfg["validation"]["representations"][rp["rep"]]
            sigma = float(m9.sh["evaluation"]["ridge_sigma_px"])
            q1 = R.apply({"select": rep.get("select", "top")}, score, data.valid, frac, known=data.labels, sigma_px=sigma,
                         eligible=el)
            led = [S.area_ledger(q1, data, 0, "selection (top-frac of eligible)")]
            q = R.apply(rep, score, data.valid, frac, known=data.labels, sigma_px=sigma, eligible=el)
            led.append(S.area_ledger(q, data, 0, "representation"))
            man = {"model_id": method, "method": method, "parameters": {**rp, "frac": frac}, "frac_source": fsrc,
                   "gate6": g6.iloc[0].to_dict(), "registry": m9.ws.load_table("model_registry", "reports/model_registry")
                   .to_dict("records"), "m9_config_fp": m9.fp("config"), "cube": m9.cube_keys(),
                   "written_by": "m9.submission"}
            info = E._write(m9.m7view_all, f"m9_champion_{method}", q, led, man, 0)
            fails = m9_checks(m9, info)
            rows.append({"file": info["submission"], "role": "champion", "method": method,
                         "frac_of_eligible": info["final_frac_of_eligible"],
                         "checks_failed": ";".join(info["validation"]["checks_failed"]), "m9_checks_failed": ";".join(fails),
                         "status": "REJECTED" if fails else "VALID"})
        except (KeyError, FileNotFoundError) as e:
            m9.say(f"s18 promoted {method} not exportable: {type(e).__name__}: {e}")
            rows.append({"file": None, "role": "champion", "method": method, "status": f"NOT_EXPORTED: {e}"})
    else:
        m9.say("s18 no M9 method promoted (Gate 6): the B0/B1 fallback is the only submission file")
    idx = pd.DataFrame(rows)
    return {"index": idx, "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
