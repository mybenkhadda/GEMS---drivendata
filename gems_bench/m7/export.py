"""Sections 15-16: competition GeoTIFFs and leaderboard probe packages (nothing is ever uploaded).

Full-region rule = the development rule applied to one zone (the whole footprint): per-zone rank of each member,
the all-zone choice of (rep, w[, ring]) on the primary task, the area fraction from ``submission.area_frac_source``
counted on the ELIGIBLE full-region pixels (valid & template finite & not a known fault & not in the ring),
then ``gems3.submission.finalise`` (NaN wherever the template is NaN, 0 where it is finite outside the data) and
``validate`` (single float32 band, EPSG:32611, template grid, [0, 1], no NaN inside the scored footprint, no
finite value where the template is NaN). An area ledger records the positive area after every step.

Probes (at most three; never optimise dozens of variants against the public leaderboard):

    A  champion fusion + 1-px thinning, known-fault ring 300 m ON
    B  identical (same rep, weight and number of selected pixels), ring OFF   -> isolates the ring hypothesis
    C  the best method that passed its section-14 gate, else the conservative fallback (``probes.conservative_fallback``)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts.atomic import atomic_save
from ..gems3 import representation as R
from ..gems3 import submission as S

LEADERBOARD_COLUMNS = ["submission_id", "date_utc", "experiment_id", "model", "configuration_fingerprint",
                       "file", "candidate_area_frac_eligible", "ring_px", "public_lb_score", "offline_disc_official_dti",
                       "offline_geo_dti", "offline_retained_fp_dti", "notes", "interpretation"]


def frac_for(m7, method: str, sel: dict) -> tuple[float, str]:
    sub = m7.cfg["submission"]
    src = sub["area_frac_source"]
    prim = m7.cfg["evaluation"]["primary_task"]
    if isinstance(src, (int, float)):
        return float(src), "explicit"
    if src == "geo":
        geo_m = method.replace("_ring300", "").replace("_ring_sel", "")
        g = sel.get("geo", {}).get(geo_m) or sel.get("geo", {}).get("fusion_thin")
        if g is None or g["params_all"].get("frac") is None:
            raise KeyError(f"no geo all-zone fraction for {geo_m}: run section 01 with 'geo' in evaluation.tasks")
        return float(g["params_all"]["frac"]), f"geo all-zone choice of {geo_m} (share of eligible pixels)"
    return float(sel[prim][method]["params_all"]["frac"]), f"{prim} all-zone choice (share of its eligible pixels)"


def _write(m7, name: str, q: np.ndarray, ledger: list, manifest: dict, ring_px: int) -> dict:
    from ..submission import write_submission

    data = m7.ctx.data
    ov = m7.cfg["submission"]["outside_valid"]
    arr = S.finalise(q, data, ov)
    ledger = ledger + [S.area_ledger(arr, data, ring_px, f"finalised (outside_valid={ov})")]
    p = m7.ws.root / "submissions" / f"{name}.tif"
    if p.exists() and not m7.cfg.get("overwrite_existing"):
        p = p.with_name(f"{name}__{time.strftime('%Y%m%dT%H%M%S')}.tif")      # never overwrite an earlier file
    atomic_save(p, lambda t: write_submission(arr, data, Path(t)))
    m7.ws._track(p)
    chk = S.validate(p, data, ov)
    info = {**manifest, "submission": p.name, "area_ledger": ledger, "validation": chk, "uploaded": False,
            "final_frac_of_eligible": ledger[-1]["frac_of_eligible"], "final_positive_px": ledger[-1]["positive_px"],
            "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
    m7.ws.save_json(json.loads(json.dumps(info, default=str)), f"submissions/{p.stem}.json")
    return info


def _provenance(m7, extra: dict) -> dict:
    from .._version import provenance
    from ..evaluation.comparative import file_fp

    feats = sorted({s["feature"] for s in m7.cfg["candidates"]["sources"] if s.get("feature")} |
                   set(m7.cfg["map_completion"]["evidence_layers"]))
    fman = {}
    for f in feats:
        try:
            p = m7.ctx.store.path(f)
            fman[f] = file_fp(p) if Path(p).exists() else "absent"
        except Exception:
            fman[f] = "unknown"
    try:
        members = m7.g3.members()
    except Exception as e:                                          # provenance never blocks an export
        members = [{"error": f"{type(e).__name__}: {e}"}]
    return {"code": provenance(), "members": members, "input_feature_manifest": fman,
            "candidate_generator_fp": m7.fp_of("s03_candidates"), "protocol": m7.cfg["evaluation"]["budget_domain"],
            "m7_config_fp": m7.fp(), **extra}


def export_baseline(m7, name: str, method: str, *, ring_px: int | None = None, k_px: int | None = None) -> dict:
    """Full-region GeoTIFF of a section-01 method. ``k_px`` forces the number of selected pixels (probe pairs)."""
    sel = m7.baseline_selection()
    prim = m7.cfg["evaluation"]["primary_task"]
    s = sel[prim][method]
    pa = dict(s["params_all"])
    ring = int(pa.get("ring", 0) or 0) if ring_px is None else int(ring_px)
    frac, fsrc = frac_for(m7, method, sel)
    maps, metas = m7.full_maps()
    data = m7.ctx.data
    elig = S.eligible_mask(data, ring, "eligible")
    if k_px is not None:
        frac, fsrc = float(k_px) / max(int(elig.sum()), 1), f"matched pixel count ({k_px} px)"
    reps = m7.cfg["baseline"]["representations"]
    rep = reps[pa["rep"]]
    score = R.fuse(R.rank01(maps[0], data.valid), R.rank01(maps[1], data.valid), float(pa["w"]))
    sigma = float(m7.cfg["evaluation"]["ridge_sigma_px"])
    q1 = R.apply({"select": rep.get("select", "top")}, score, data.valid, frac, known=data.labels, sigma_px=sigma,
                 eligible=elig)
    led = [S.area_ledger(q1, data, ring, "selection (top-frac of eligible)")]
    q2 = R.apply(rep, score, data.valid, frac, known=data.labels, sigma_px=sigma, eligible=elig)
    led.append(S.area_ledger(q2, data, ring, "post operations"))
    q = R.apply(R.with_ring(rep, ring), score, data.valid, frac, known=data.labels, sigma_px=sigma, eligible=elig)
    led.append(S.area_ledger(q, data, ring, f"known-fault ring {ring} px"))
    man = _provenance(m7, {"model_id": f"s01_{method}", "method": method, "parameters": {**pa, "ring": ring, "frac": frac},
                           "frac_source": fsrc, "selected_on": prim, "dti_crossfit": s["dti_crossfit"],
                           "member_maps": metas, "postprocessing": rep})
    return {**_write(m7, name, q, led, man, ring), "k_px": int((q1 > 0).sum())}


def export_candidate_model(m7, name: str, method: str) -> dict:
    """Full-region GeoTIFF of a section-11/12 candidate method (ranked full-region candidates, chosen rep / blend /
    frac / ring; the hybrid adds the section-01 fusion)."""
    from .ranking import fit_full
    from .rasterize import blend, hybrid, predict

    cfg = m7.cfg
    prim = cfg["evaluation"]["primary_task"]
    rz = m7.ws.load_table("rasterization_summary", "candidate_ranking/predictions")
    row = rz[(rz.task == prim) & (rz.method == method)]
    if not len(row):
        raise KeyError(f"{method}: not in the section-12 summary")
    pa = json.loads(row.params_all.iloc[0])
    T = m7.ws.load_table("ranking_scores", "candidate_ranking/features")
    best = m7.read_json("candidate_ranking/predictions/best_ranker.json").get("best_ranker", "label_free")
    full_u = m7.full_unit()
    cs = m7.load_unit_set("full:0")
    F = m7.ws.load_table("full_0", "candidate_ranking/features")
    if cs is None or not len(F):
        raise FileNotFoundError("full-region candidates / features missing: run sections 03 and 10")
    from .cand_features import label_free_score, neural_score

    F = F.sort_values("cand_id").assign(unit="full:0")
    sc = {"label_free": label_free_score(F), "neural": neural_score(F)}
    if best not in ("label_free", "neural_support"):
        model, feats = fit_full(T, best, cfg["ranking"], prim)
        for c in feats:
            if c not in F:
                F[c] = np.nan
        sc["ranker"] = model.predict_proba(F[feats].astype(float))[:, 1]
    else:
        sc["ranker"] = sc["label_free"] if best == "label_free" else sc["neural"]
    fw = json.loads(pa["fw"]) if isinstance(pa.get("fw"), str) else (pa.get("fw") or cfg["rasterization"]["fusion_weights"][0])
    s = blend(sc, fw)
    src = cfg["submission"]["area_frac_source"]
    geo_row = rz[(rz.task == "geo") & (rz.method == method)]
    if isinstance(src, (int, float)):
        frac = float(src)
    elif src == "geo" and len(geo_row):
        frac = float(json.loads(geo_row.params_all.iloc[0])["frac"])
    else:
        frac = float(pa["frac"])
    data = m7.ctx.data
    tmpl = np.asarray(data.template_valid, bool)
    elig_of = (lambda u, r: S.eligible_mask(data, r, "eligible") & tmpl)
    ring = int(pa.get("ring", 0) or 0)
    if method.startswith("hybrid"):
        base = export_components_fusion(m7, (1.0 - float(pa["share"])) * frac)
        q = hybrid(full_u, base, cs, s, frac, float(pa["share"]), ring, elig_of)
        led = [S.area_ledger(base, data, ring, "fusion share"), S.area_ledger(q, data, ring, "fusion + candidate lines")]
    else:
        q, led_u = predict(full_u, cs, s, cfg["rasterization"]["representations"][pa["rep"]], frac, ring, elig_of)
        led = [S.area_ledger(q, data, ring, f"{pa['rep']} ring {ring}")]
    man = _provenance(m7, {"model_id": method, "method": method, "ranker": best, "parameters": pa, "frac": frac,
                           "frac_source": cfg["submission"]["area_frac_source"],
                           "candidate_generator": cfg["candidates"], "representation": pa.get("rep")})
    return _write(m7, name, q, led, man, ring)


def export_components_fusion(m7, frac: float) -> np.ndarray:
    """The section-01 fusion_thin full-region map at ``frac`` of the eligible pixels (hybrid base)."""
    sel = m7.baseline_selection()
    prim = m7.cfg["evaluation"]["primary_task"]
    pa = sel[prim]["fusion_thin"]["params_all"]
    maps, _ = m7.full_maps()
    data = m7.ctx.data
    ring = int(pa.get("ring", 0) or 0)
    score = R.fuse(R.rank01(maps[0], data.valid), R.rank01(maps[1], data.valid), float(pa["w"]))
    return R.apply(R.with_ring(m7.cfg["baseline"]["representations"][pa["rep"]], ring), score, data.valid, frac,
                   known=data.labels, sigma_px=float(m7.cfg["evaluation"]["ridge_sigma_px"]),
                   eligible=S.eligible_mask(data, ring, "eligible"))


def promoted_method(m7) -> str | None:
    v = m7.ws.load_table("decision_verdicts", "reports")
    if not len(v):
        return None
    p = v[v.verdict == "PROMOTE"].sort_values("dti_pooled", ascending=False)
    return str(p.method.iloc[0]) if len(p) else None


def probes(m7) -> pd.DataFrame:
    """Probe packages A / B / C + leaderboard_log.csv template (created once, never overwritten)."""
    cfg = m7.cfg
    rows = []
    sel = m7.baseline_selection() or {}
    prim = cfg["evaluation"]["primary_task"]
    champ = "fusion_thin_ring300" if "fusion_thin_ring300" in sel.get(prim, {}) else "fusion_thin"
    a = export_baseline(m7, "probe_A_fusion_thin_ring300", champ, ring_px=3)
    rows.append({"probe": "A", "file": a["submission"], "method": champ, "ring_px": 3,
                 "hypothesis": "the known-fault ring removes FP mass that the hidden test also penalises",
                 "frac_of_eligible": a["final_frac_of_eligible"]})
    b = export_baseline(m7, "probe_B_fusion_thin_no_ring", champ, ring_px=0, k_px=a["k_px"])
    rows.append({"probe": "B", "file": b["submission"], "method": champ, "ring_px": 0,
                 "hypothesis": "same map without the ring: B > A means hidden new faults lie next to known traces",
                 "frac_of_eligible": b["final_frac_of_eligible"]})
    pm = promoted_method(m7)
    c = None
    if pm and int(cfg["probes"]["max_probes"]) >= 3:
        try:
            if pm.startswith("s01_"):
                c = export_baseline(m7, f"probe_C_{pm[4:]}", pm[4:])
            elif pm.startswith("s12_"):
                c = export_candidate_model(m7, f"probe_C_{pm[4:]}", pm[4:])
            elif pm.startswith("rank_"):
                c = export_candidate_model(m7, f"probe_C_cand_{pm[5:]}", f"cand_{pm[5:]}")
            else:
                raise KeyError(f"{pm} has no full-region exporter")
            rows.append({"probe": "C", "file": c["submission"], "method": pm, "ring_px": np.nan,
                         "hypothesis": "the promoted discovery-first method transfers to the hidden test",
                         "frac_of_eligible": c["final_frac_of_eligible"]})
        except (KeyError, FileNotFoundError) as e:
            print(f"[m7] probe C: promoted {pm} not exportable ({e}); conservative fallback used")
            c = None
    if c is None and int(cfg["probes"]["max_probes"]) >= 3:
        fb = cfg["probes"]["conservative_fallback"]
        c = export_baseline(m7, f"probe_C_{fb}", fb)
        rows.append({"probe": "C", "file": c["submission"], "method": f"s01_{fb}", "ring_px": 0,
                     "hypothesis": "conservative fallback (no new method passed its promotion gate)",
                     "frac_of_eligible": c["final_frac_of_eligible"]})
    t = pd.DataFrame(rows)
    for r in rows:
        m7.ws.save_json({**{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()},
                         "uploaded": False, "instructions": "upload manually; record the score in leaderboard_log.csv"},
                        f"submissions/probe_{r['probe']}.json")
    log = m7.ws.root / "submissions" / "leaderboard_log.csv"
    if not log.exists():
        pd.DataFrame(columns=LEADERBOARD_COLUMNS).to_csv(log, index=False)
    m7.ws._track(log)
    return t
