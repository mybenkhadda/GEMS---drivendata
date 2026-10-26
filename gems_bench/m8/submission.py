"""Section 15: competition GeoTIFFs (validated, never uploaded).

Always: the CNN champion (section-01 ``fusion_thin`` rule on full-region member maps) WITHOUT the known-fault ring and
WITH the optional 300 m ring at the SAME number of selected pixels (the pair isolates the ring hypothesis). Only when
Gate 6 PROMOTES an M8 method (``submission.export_physics_if``): the physics champion on the full region:

    full-region pool (section 06 'full' unit + MC-only structures) -> label-free features (section 10 code, no labels)
    -> ranker refitted on every labelled row of the primary task (the promoted method's feature set) -> section-13
    rasterisation with the all-zone choice (rep, regional beta, ring) -> optional fusion with System A / C at the
    all-zone fusion weights -> eligible budget (frac from ``shared.submission.area_frac_source``)

Writing reuses M7 / GEMS3 (``m7.export._write``): template = authority for shape / CRS / transform / bounds / nodata,
NaN wherever the template is NaN, 0 where it is finite outside the data, float32 in [0, 1], area ledger after every
mask, ``gems3.submission.validate`` (raises on any failed check), manifest with provenance. Existing files are never
overwritten. ``leaderboard_log.csv`` separates public-leaderboard scores (filled manually), offline official-style
scores and diagnostic oracles.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

LOG_COLUMNS = ["submission_id", "date_utc", "file", "experiment_id", "method", "profile", "configuration_fingerprint",
               "ring_px", "frac_of_eligible", "score_type", "public_lb_score", "offline_disc_official_dti",
               "offline_geo_dti", "offline_retained_fp_dti", "oracle_dti_diagnostic", "notes"]


def cnn_baselines(m8) -> list[dict]:
    from ..m7 import export as E

    champ = m8.cfg["baseline"]["champion"]
    rings = sorted(int(r) for r in m8.cfg["submission"]["ring_variants"])
    out = []
    k_px = None
    for r in sorted(rings, reverse=True):                       # largest ring first: its pixel count is matched
        info = E.export_baseline(m8.view, f"m8_cnn_{champ}_ring{r}", champ, ring_px=r, k_px=k_px)
        k_px = info["k_px"] if k_px is None else k_px
        out.append({"file": info["submission"], "method": f"s01_{champ}", "ring_px": r,
                    "frac_of_eligible": info["final_frac_of_eligible"],
                    "checks_failed": ";".join(info["validation"]["checks_failed"])})
    return out


def _frac(m8, summ: pd.DataFrame, method: str) -> tuple[float, str]:
    src = m8.sh["submission"]["area_frac_source"]
    if isinstance(src, (int, float)):
        return float(src), "explicit"
    task = "geo" if src == "geo" else m8.prim
    row = summ[(summ.task == task) & (summ.method == method)]
    if not len(row):
        row = summ[(summ.task == m8.prim) & (summ.method == method)]
    pa = json.loads(row.params_all.iloc[0])
    return float(pa["frac"]), f"{task} all-zone choice of {method}"


def full_region_physics(m8, method: str) -> dict:
    """Full-region map of the promoted M8 method (m8_physics or an ens_* fusion)."""
    from ..gems3 import representation as R
    from ..gems3 import submission as S
    from ..m7.export import _provenance, _write
    from ..m7.ranking import _model
    from . import features as FT
    from . import rasterize as RZ
    from .ranking import score_column

    data = m8.ctx.data
    fu = m8.full_unit()
    cs = m8.pool("full:0")
    if cs is None or not len(cs):
        cs = m8.build_full_pool()
    T = m8.ws.load_table("ranking_scores", "ranking")
    rsum = m8.ws.load_table("ranking_summary", "ranking")
    col = score_column(m8, rsum)
    method_r = col.replace("score_", "")
    fm = m8.full_member_maps()
    stems = list(m8.sh["models"])
    F, _ = FT.describe(m8, fu, cs, with_labels=False, scores={"b0": fm[0], "b1": fm[1]},
                       profiles=m8.unit_profiles("full:0"), mc=m8.mc_unit("full:0"), mos=m8.dem_mosaic())
    F = F.sort_values("cand_id").reset_index(drop=True)
    if method_r in ("label_free", "cnn_only"):
        from ..m7.cand_features import neural_score
        from .ranking import label_free_score

        sc = label_free_score(F) if method_r == "label_free" else neural_score(F)
    else:
        feats = [c for c in FT.ablation_features(T).get(method_r, []) if c in T]
        lab = T[(T.task == m8.prim) & (T.label >= 0)]
        mdl = _model(m8.sh["ranking"], False)
        y = lab.label.to_numpy()
        mdl.fit(lab[feats].astype(float), y, sample_weight=np.where(y == 1, 1.0, float(m8.sh["ranking"]["negative_weight"])))
        for c in feats:
            if c not in F:
                F[c] = np.nan
        sc = mdl.predict_proba(F[feats].astype(float))[:, 1]
    rz_sum = m8.ws.load_table("rasterization_summary", "rasterization")
    pa = json.loads(rz_sum[(rz_sum.task == m8.prim)].params_all.iloc[0])
    frac, fsrc = _frac(m8, rz_sum, "m8_physics")
    tmpl = np.asarray(data.template_valid, bool)
    elig_of = (lambda u, r: S.eligible_mask(data, r, "eligible") & tmpl)
    occ = m8.mc_occupancy()
    stab = m8.ws.load_table("stability", "monte_carlo/stability")
    pos = None
    if len(stab):
        s = stab[stab.unit == "full:0"].set_index("cand_id").mc_pos_std_m
        pos = s.reindex(F.cand_id.to_numpy()).to_numpy() if len(s) else None
    rep = m8.cfg["rasterization"]["representations"][pa["rep"]]
    block = max(int(round(float(m8.cfg["rasterization"]["regional_block_km"]) / 0.1)), 10)
    ring = int(pa.get("ring", 0) or 0)
    led = []
    if method == "m8_physics":
        q, led = RZ.predict(fu, cs, sc, rep, frac, ring, float(pa["beta"]), elig_of, occupancy=occ, pos_std_m=pos,
                            block_px=block)
        params = {"ranker": method_r, **pa, "frac": frac}
    else:
        es = m8.ws.load_table("fusion_summary", "ensemble")
        pe = json.loads(es[(es.task == m8.prim) & (es.method == method)].params_all.iloc[0])
        frac, fsrc = _frac(m8, es, method)
        w = json.loads(pe["w"]) if isinstance(pe["w"], str) else pe["w"]
        from .ensemble import _paint_rank

        sel = (m8.baseline_selection() or {})[m8.prim][m8.cfg["baseline"]["champion"]]["params_all"]
        parts = {"A": np.where(data.valid, R.fuse(R.rank01(fm[0], data.valid), R.rank01(fm[1], data.valid),
                                                  float(sel["w"])), 0.0)}
        parts["B"] = _paint_rank(cs, sc, data.valid)
        if w[2] > 0:
            parts["C"] = m8.full_completion_score()
        s = sum(wi * parts[k] for wi, k in zip(w, "ABC") if wi > 0)
        rep2 = m8.sh["baseline"]["representations"][pe["rep"]]
        sigma = float(m8.sh["evaluation"]["ridge_sigma_px"])
        crest = R.crest_map(s, data.valid, sigma) if rep2.get("select") == "crest" else None
        q = R.apply(rep2, s, data.valid, frac, known=data.labels, crest=crest, sigma_px=sigma,
                    eligible=S.eligible_mask(data, 0, "eligible") & tmpl)
        params = {"weights": w, "rep": pe["rep"], "frac": frac, "ranker": method_r}
        ring = 0
    led = [S.area_ledger(q, data, ring, f"{method} full region")] + [
        {k: v for k, v in x.items() if k in ("step", "positive_px", "frac_of_eligible")} for x in led]
    man = _provenance(m8.view, {"model_id": method, "method": method, "parameters": params, "frac_source": fsrc,
                                "m8_profile": m8.profile["name"], "m8_fingerprints": {s: m8.fp_of(s) for s in (
                                    "s06_candidates", "s07_monte_carlo_signal", "s11_ranking", "s13_rasterization",
                                    "s14_ensemble")},
                                "member_stems": stems})
    return _write(m8.view, f"m8_{method.replace('+', '')}", q, led, man, ring)


def leaderboard_log(m8, rows: list[dict]) -> pd.DataFrame:
    p = m8.ws.root / "submissions" / "leaderboard_log.csv"
    m8.stage_in([f"{m8.cfg['output_dir']}/submissions/leaderboard_log.csv"])
    old = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=LOG_COLUMNS)
    pooled = m8.ws.load_table("method_pooled", "validation/metrics")

    def off(method, task):
        g = pooled[(pooled.task == task) & (pooled.method == method)] if len(pooled) else pooled
        return float(g.dti_pooled.iloc[0]) if len(g) else np.nan

    def off_alt(method):
        g = pooled[(pooled.task == m8.prim) & (pooled.method == method)] if len(pooled) else pooled
        return float(g.dti_retained_fp.iloc[0]) if len(g) and "dti_retained_fp" in g else np.nan

    new = []
    for r in rows:
        if len(old) and r["file"] in set(old.file.astype(str)):
            continue
        new.append({"submission_id": "", "date_utc": "", "file": r["file"], "experiment_id": m8.ws.experiment_id(
            "s15_submission", m8.fp_of("s14_ensemble") or "none"), "method": r["method"], "profile": m8.profile["name"],
            "configuration_fingerprint": m8.fp(), "ring_px": r.get("ring_px"), "frac_of_eligible": r.get("frac_of_eligible"),
            "score_type": "public_lb_score is filled MANUALLY after upload; offline_* = nested cross-fitted proxies; "
                          "oracle = diagnostic upper bound, never a score",
            "public_lb_score": np.nan, "offline_disc_official_dti": off(r["method"], m8.prim),
            "offline_geo_dti": off(r["method"], "geo"), "offline_retained_fp_dti": off_alt(r["method"]),
            "oracle_dti_diagnostic": np.nan, "notes": "not uploaded by code"})
    log = pd.concat([old, pd.DataFrame(new, columns=LOG_COLUMNS)], ignore_index=True)
    from ..artifacts.atomic import atomic_save

    atomic_save(p, lambda t: log.to_csv(t, index=False))
    m8.ws._track(p)
    return log


def run(m8) -> dict:
    rows = cnn_baselines(m8)
    g6 = m8.ws.load_table("gate6", "ensemble")
    rule = m8.cfg["submission"]["export_physics_if"]
    if rule == "PROMOTE" and len(g6) and g6.decision.iloc[0] == "PROMOTE":
        method = str(g6.method.iloc[0])
        if method == "m8_physics" or method.startswith("ens_"):
            try:
                info = full_region_physics(m8, method)
                rows.append({"file": info["submission"], "method": method, "ring_px": np.nan,
                             "frac_of_eligible": info["final_frac_of_eligible"],
                             "checks_failed": ";".join(info["validation"]["checks_failed"])})
            except (KeyError, FileNotFoundError, IndexError) as e:
                m8.say(f"s15 promoted {method} not exportable: {type(e).__name__}: {e}")
        else:
            m8.say(f"s15 promoted {method}: no full-region exporter (only m8_physics and ens_*)")
    else:
        m8.say("s15 no M8 method promoted: CNN champion files only")
    idx = pd.DataFrame(rows)
    log = leaderboard_log(m8, rows)
    return {"index": idx, "log": log, "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
