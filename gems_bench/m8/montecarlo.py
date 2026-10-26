"""Section 07: Monte Carlo geophysical uncertainty of structural extraction.

What is perturbed (per realisation; distributions in ``monte_carlo.distributions``, every draw recorded)::

    measurement noise      spatially CORRELATED Gaussian noise (white noise convolved with a Gaussian of
                           ``noise_corr_m``), amplitude = noise_frac x robust std of the family's high-pass field
                           (heuristic sensitivity) or the measured sigma of ``measured_noise`` (measured uncertainty)
    registration           one rigid sub-pixel shift per SURVEY FAMILY, sigma = registration_factor x effective
                           resolution (all derived products of one survey move together)
    operator scale         multiplicative log-normal jitter of every operator scale
    edge operator          uniform choice among the family's operators (THDR / tilt edge / TDX ...)
    extraction choices     top-frac factor, coherence gate, linking gap / angle, persistence requirement

Physical constraints: only the BASE field of each survey family is perturbed and every operator is recomputed from it,
so derived bands of one measurement (e.g. tmi_hg / tmi_vg) are never perturbed as if they were independent sensors
(band-derived operators are refused by the config validator); noise is spatially correlated at >= one pixel and no
perturbation is finer than the configured effective resolution allows. Unless ``measured_noise`` is supplied, this is
a HEURISTIC SENSITIVITY ANALYSIS of the extraction chain, not measured source uncertainty.

Per realisation: perturbed fields -> operator layers -> local-threshold crests -> persistence -> linked paths (the
section-06 pipeline) -> one line mask, a strike raster and a path-id raster over the processing region. Matching is
spatially tolerant: a section-06 candidate is DETECTED when >= ``detect_frac`` of its pixels lie within
``match_radius_px`` of the realisation's lines; fragments of one structure therefore count once (fragmentation is a
separate statistic). Aggregates per candidate::

    mc_freq              detection frequency (structural-processing STABILITY, never a calibrated fault probability)
    mc_pos_mean_m / _std positional offset of the matched realisation lines (mean / dispersion across realisations)
    mc_orient_disp       1 - resultant length of the matched strikes (orientation variability)
    mc_orient_diff       mean axial difference to the candidate strike
    mc_frag_mean         mean number of distinct realisation paths matching the candidate (fragmentation)
    mc_cov_mean / _cv    covered length share and its coefficient of variation (length variability)
    mc_xmod_freq         share of realisations where >= 2 survey families detect it (cross-modal stability)
    mc_freq_<family>     per-family detection frequency

The occupancy atlas (share of realisations with a line within 1 px) and MC-only structures (occupancy crests >=
``occupancy_extra_threshold`` outside the section-06 pool) are saved. Resumable: aggregates are checkpointed every
``checkpoint_every`` realisations; realisation r always uses seed + r.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy import ndimage

from ..postprocess.topology import _disk
from . import potential_fields as PF


# =============================================================================== sampling
def draw(spec: dict, rng, choices=None):
    d = spec["dist"]
    if d == "uniform":
        return float(rng.uniform(float(spec["lo"]), float(spec["hi"])))
    if d == "normal":
        return float(rng.normal(float(spec.get("mu", 0.0)), float(spec["sigma"])))
    vals = list(choices if choices is not None else spec["values"])
    return vals[int(rng.integers(len(vals)))]


def draw_params(m8, r: int) -> dict:
    mc = m8.cfg["monte_carlo"]
    D = mc["distributions"]
    rng = np.random.default_rng(int(m8.cfg["seed"]) + 1000 + int(r))
    fams = {}
    for f in mc["families"]:
        fam = f["family"]
        eff = m8.layers.eff_res_m(fam)
        meas = (mc.get("measured_noise") or {}).get(fam)
        fams[fam] = {"operator": draw(D["operator"], rng, f["operators"]),
                     "noise_frac": draw(D["noise_frac"], rng), "noise_corr_m": float(draw(D["noise_corr_m"], rng)),
                     "shift_r_m": draw(D["registration_factor"], rng) * eff,
                     "shift_c_m": draw(D["registration_factor"], rng) * eff,
                     "sigma_factor": float(np.exp(draw(D["sigma_log_jitter"], rng))),
                     "measured_sigma": None if meas is None else float(meas),
                     "uncertainty_type": "measured" if meas is not None else "heuristic_sensitivity"}
    return {"realization": int(r), "seed": int(m8.cfg["seed"]) + 1000 + int(r), "families": fams,
            "top_frac_factor": draw(D["top_frac_factor"], rng), "min_coherence": draw(D["min_coherence"], rng),
            "link_gap_px": int(draw(D["link_gap_px"], rng)), "link_angle_deg": float(draw(D["link_angle_deg"], rng)),
            "min_persistence": int(draw(D["min_persistence"], rng)), "noise_seed": int(rng.integers(1 << 31))}


# =============================================================================== regions
def regions(m8) -> list[dict]:
    """Processing regions (full-grid bboxes) of the active profile; stats are computed per region."""
    w = str(m8.profile["mc_window"])
    H, W = m8.ctx.data.shape
    if w == "full":
        return [{"name": "full", "r0": 0, "r1": H, "c0": 0, "c1": W}]
    if w == "eval_units":
        return [{"name": u.key, "r0": u.sl[0].start, "r1": u.sl[0].stop, "c0": u.sl[1].start, "c1": u.sl[1].stop}
                for u in m8.units_of(m8.prim)]
    if w.startswith("unit:"):
        u = m8.unit(w[5:])
        if u is None:
            raise KeyError(f"monte_carlo window {w}: unknown unit")
        return [{"name": u.key, "r0": u.sl[0].start, "r1": u.sl[0].stop, "c0": u.sl[1].start, "c1": u.sl[1].stop}]
    r0, r1, c0, c1 = (int(x) for x in w[5:].split(","))
    return [{"name": w, "r0": r0, "r1": r1, "c0": c0, "c1": c1}]


def tiles(reg: dict, tile: int, halo: int, shape) -> list[tuple]:
    out = []
    for r in range(reg["r0"], reg["r1"], tile):
        for c in range(reg["c0"], reg["c1"], tile):
            core = (r, min(r + tile, reg["r1"]), c, min(c + tile, reg["c1"]))
            ext = (max(core[0] - halo, 0), min(core[1] + halo, shape[0]), max(core[2] - halo, 0), min(core[3] + halo, shape[1]))
            out.append((core, ext))
    return out


# =============================================================================== one realisation on one tile
def _hp_std(F: np.ndarray, m: np.ndarray) -> float:
    hp = F - ndimage.gaussian_filter(F, 10.0, mode="nearest")
    v = hp[m]
    if not v.size:
        return 0.0
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def perturb(F: np.ndarray, m: np.ndarray, fp: dict, rng, pixel_m: float, hp_std: float) -> tuple[np.ndarray, np.ndarray]:
    corr = max(fp["noise_corr_m"] / pixel_m, 1.0)
    noise = ndimage.gaussian_filter(rng.standard_normal(F.shape), corr, mode="reflect")
    noise /= max(float(noise.std()), 1e-12)
    sig = fp["measured_sigma"] if fp["measured_sigma"] is not None else fp["noise_frac"] * hp_std
    G = F + sig * noise
    sh = (fp["shift_r_m"] / pixel_m, fp["shift_c_m"] / pixel_m)
    G = ndimage.shift(G, sh, order=1, mode="nearest")
    mm = ndimage.shift(m.astype(np.float32), sh, order=0, mode="constant", cval=0.0) > 0.5
    return G, mm


def path_strike(p: np.ndarray) -> np.ndarray:
    q = p.astype(float)
    d = np.gradient(q, axis=0) if len(q) > 1 else np.zeros_like(q)
    return (np.degrees(np.arctan2(d[:, 1], -d[:, 0])) % 180.0).astype(np.float32)


def realize_tile(m8, prm: dict, ext: tuple, rng, hp: dict) -> dict:
    """{family: (paths in ext coords)} for one tile."""
    from ..m7.generator import response

    from .candidates import gated_local, persistence, trace_family

    r0, r1, c0, c1 = ext
    mc, gc = m8.cfg["monte_carlo"], m8.cfg["generation"]
    L = m8.layers
    ph = m8.cfg["physics"]
    block = int(round(float(gc["local_threshold_block_km"]) * 1000.0 / L.pixel_m))
    g2 = {**gc, "link_variants": [[prm["link_gap_px"], prm["link_angle_deg"]]]}
    out = {}
    for f in mc["families"]:
        fam = f["family"]
        fp = prm["families"][fam]
        band = L.families[fam]["field"]
        if not L.has_band(band):
            continue
        Fraw = m8.ctx.data.read_window(band, r0, r1, c0, c1)
        Ff, m = PF.fill(Fraw)
        m &= np.asarray(m8.ctx.data.valid, bool)[r0:r1, c0:c1]
        G, mm = perturb(Ff, m, fp, rng, L.pixel_m, hp[fam])
        per, scales, sig = [], [], []
        for s in L.scales(fam, f["scales_m"]):
            s2 = float(s) * fp["sigma_factor"]
            a, _ = PF.operator(fp["operator"], np.where(mm, G, np.nan).astype(np.float32), PF.sigma_px_of(s2, L.pixel_m),
                               px_km=L.px_km, amp_floor_q=float(ph["amplitude_floor_quantile"]),
                               lcn_sigma_px=float(ph["lcn_window_m"]) / 2.0 / L.pixel_m)
            v = mm & np.isfinite(a)
            rr = response(a, v, 0.7, "ridge")
            per.append(gated_local(rr["strength"], rr["nms"], rr["coherence"], v, gc, block,
                                   top_frac=min(float(gc["top_frac_per_scale"]) * prm["top_frac_factor"], 0.5),
                                   min_coh=prm["min_coherence"]))
            scales.append(s2)
            sig.append(PF.sigma_px_of(s2, L.pixel_m))
        P = persistence(per, sig)
        if P is None:
            out[fam] = []
            continue
        ps, _ = trace_family(per, scales, P, g2, {"name": f"mc_{fam}", "modality": fam}, min_persistence=prm["min_persistence"])
        out[fam] = ps
    return out


def realize(m8, prm: dict, reg: dict, hp: dict) -> dict:
    """Stitched realisation over one region: line mask, per-family masks, strike and path-id rasters (region coords)."""
    mc = m8.cfg["monte_carlo"]
    H, W = m8.ctx.data.shape
    h, w = reg["r1"] - reg["r0"], reg["c1"] - reg["c0"]
    rng = np.random.default_rng(prm["noise_seed"])
    line = np.zeros((h, w), bool)
    fam_m = {f["family"]: np.zeros((h, w), bool) for f in mc["families"]}
    strike = np.full((h, w), np.nan, np.float32)
    pid = np.zeros((h, w), np.int32)
    k = 0
    n_paths = 0
    for core, ext in tiles(reg, int(mc["tile_px"]), int(mc["halo_px"]), (H, W)):
        res = realize_tile(m8, prm, ext, rng, hp)
        for fam, ps in res.items():
            for p in ps:
                q = p + np.array([ext[0], ext[2]])
                ok = (q[:, 0] >= core[0]) & (q[:, 0] < core[1]) & (q[:, 1] >= core[2]) & (q[:, 1] < core[3])
                if not ok.any():
                    continue
                k += 1
                n_paths += 1
                st = path_strike(p)[ok]
                rr, cc = q[ok, 0] - reg["r0"], q[ok, 1] - reg["c0"]
                line[rr, cc] = True
                fam_m[fam][rr, cc] = True
                strike[rr, cc] = st
                pid[rr, cc] = k
    return {"line": line, "fam": fam_m, "strike": strike, "pid": pid, "n_paths": n_paths}


# =============================================================================== aggregation
class Accum:
    """Per-candidate accumulators of one unit's pool (only candidates assigned to this region)."""

    FIELDS = ("det", "sd", "sd2", "c2", "s2", "adiff", "frag", "cov", "cov2", "xmod")

    def __init__(self, n: int, fams: list[str]):
        self.a = {k: np.zeros(n, np.float64) for k in self.FIELDS}
        self.fam = {f: np.zeros(n, np.float64) for f in fams}

    def to_npz(self, prefix: str) -> dict:
        out = {f"{prefix}|{k}": v for k, v in self.a.items()}
        out.update({f"{prefix}|fam:{k}": v for k, v in self.fam.items()})
        return out

    def load(self, z, prefix: str) -> None:
        for k in self.a:
            if f"{prefix}|{k}" in z:
                self.a[k] = np.asarray(z[f"{prefix}|{k}"], np.float64)
        for k in self.fam:
            if f"{prefix}|fam:{k}" in z:
                self.fam[k] = np.asarray(z[f"{prefix}|fam:{k}"], np.float64)


def update(acc: Accum, paths_reg: list, strikes: np.ndarray, real: dict, edt, mc: dict) -> None:
    d, ir, ic = edt
    rad = float(mc["match_radius_px"])
    dil = {f: ndimage.binary_dilation(m, structure=_disk(int(rad))) for f, m in real["fam"].items()}
    for i, p in enumerate(paths_reg):
        dd = d[p[:, 0], p[:, 1]]
        near = dd <= rad
        cov = float(near.mean())
        acc.a["cov"][i] += cov
        acc.a["cov2"][i] += cov * cov
        nf = 0
        for f, dm in dil.items():
            hit = float(dm[p[:, 0], p[:, 1]].mean()) >= float(mc["detect_frac"])
            acc.fam[f][i] += hit
            nf += hit
        acc.a["xmod"][i] += nf >= 2
        if cov < float(mc["detect_frac"]):
            continue
        acc.a["det"][i] += 1
        m = float(dd[near].mean()) * 100.0
        acc.a["sd"][i] += m
        acc.a["sd2"][i] += m * m
        jr, jc = ir[p[near, 0], p[near, 1]], ic[p[near, 0], p[near, 1]]
        st = real["strike"][jr, jc]
        ok = np.isfinite(st)
        if ok.any():
            acc.a["c2"][i] += float(np.cos(np.radians(2 * st[ok])).mean())
            acc.a["s2"][i] += float(np.sin(np.radians(2 * st[ok])).mean())
            acc.a["adiff"][i] += float(PF.axial_diff(st[ok], strikes[i]).mean())
        acc.a["frag"][i] += len(np.unique(real["pid"][jr, jc]))


def finalize(acc: Accum, n_real: int) -> pd.DataFrame:
    a = acc.a
    R = max(n_real, 1)
    det = np.maximum(a["det"], 1e-9)
    mean_d = np.where(a["det"] > 0, a["sd"] / det, np.nan)
    std_d = np.where(a["det"] > 1, np.sqrt(np.maximum(a["sd2"] / det - mean_d ** 2, 0)), np.nan)
    Rbar = np.where(a["det"] > 0, np.hypot(a["c2"], a["s2"]) / det, np.nan)
    cm = a["cov"] / R
    cv = np.sqrt(np.maximum(a["cov2"] / R - cm ** 2, 0)) / np.maximum(cm, 1e-9)
    out = pd.DataFrame({"mc_freq": a["det"] / R, "mc_pos_mean_m": mean_d, "mc_pos_std_m": std_d,
                        "mc_orient_disp": 1.0 - Rbar, "mc_orient_diff": np.where(a["det"] > 0, a["adiff"] / det, np.nan),
                        "mc_frag_mean": np.where(a["det"] > 0, a["frag"] / det, np.nan), "mc_cov_mean": cm,
                        "mc_len_cv": cv, "mc_xmod_freq": a["xmod"] / R})
    for f, v in acc.fam.items():
        out[f"mc_freq_{f}"] = v / R
    return out


# =============================================================================== run
def assignment(m8, regs: list[dict], pools: dict) -> dict:
    """{region: {unit: (candidate indices, region-coordinate paths, strikes)}}; a candidate belongs to the first
    region that contains all its pixels (never counted twice)."""
    out = {r["name"]: {} for r in regs}
    for key, cs in pools.items():
        u = m8.unit(key)
        if u is None or cs is None or not len(cs):
            continue
        taken = np.zeros(len(cs), bool)
        strikes = cs.table.strike_deg.to_numpy(np.float64) if "strike_deg" in cs.table else np.zeros(len(cs))
        for reg in regs:
            idx, ps, st = [], [], []
            for i, p in enumerate(cs.paths):
                if taken[i]:
                    continue
                q = p + np.array([u.sl[0].start - reg["r0"], u.sl[1].start - reg["c0"]])
                if (q[:, 0] >= 0).all() and (q[:, 0] < reg["r1"] - reg["r0"]).all() and (q[:, 1] >= 0).all() and \
                        (q[:, 1] < reg["c1"] - reg["c0"]).all():
                    idx.append(i)
                    ps.append(q)
                    st.append(strikes[i])
                    taken[i] = True
            if idx:
                out[reg["name"]][key] = (np.array(idx), ps, np.array(st))
    return out


def run(m8, pools: dict) -> dict:
    from ..artifacts.atomic import atomic_save

    mc = m8.cfg["monte_carlo"]
    R = int(m8.profile["mc_realizations"])
    regs = regions(m8)
    fams = [f["family"] for f in mc["families"] if m8.layers.has_band(m8.layers.families[f["family"]]["field"])]
    key = m8.fp("mc", mc, regs, {k: len(v) for k, v in pools.items() if v is not None}, m8.fp_of("s06_candidates"))
    state_rel = f"monte_carlo/realizations/state_{key}.npz"
    man_rel = f"monte_carlo/realizations/manifest_{key}.csv"
    asg = assignment(m8, regs, pools)
    acc = {(rg, k): Accum(len(v[0]), fams) for rg, d in asg.items() for k, v in d.items()}
    H, W = m8.ctx.data.shape
    occ = np.zeros((H, W), np.uint16)
    occ_f = {f: np.zeros((H, W), np.uint16) for f in fams}
    done, manifest = 0, []
    sp = m8.ws.root / state_rel
    m8.stage_in([f"{m8.cfg['output_dir']}/{state_rel}", f"{m8.cfg['output_dir']}/{man_rel}"])
    if sp.exists() and not m8.cfg.get("overwrite_existing"):
        z = np.load(sp)
        done = int(z["done"])
        occ, occ_f = z["occ"].astype(np.uint16), {f: z[f"occ:{f}"].astype(np.uint16) for f in fams if f"occ:{f}" in z}
        for (rg, k), a in acc.items():
            a.load(z, f"{rg}#{k}")
        mp = m8.ws.root / man_rel
        manifest = pd.read_csv(mp).to_dict("records") if mp.exists() else []
        m8.say(f"s07 resuming at realisation {done} / {R}")
    hp = {}
    for f in fams:
        F, m = PF.fill(m8.layers.field(f))
        hp[f] = _hp_std(F, m & np.asarray(m8.ctx.data.valid, bool))
        del F
    m8.layers.release()
    every = max(int(mc.get("checkpoint_every", 4)), 1)
    for r in range(done, R):
        t0 = time.perf_counter()
        prm = draw_params(m8, r)
        n_paths = 0
        for reg in regs:
            real = realize(m8, prm, reg, hp)
            n_paths += real["n_paths"]
            sl = (slice(reg["r0"], reg["r1"]), slice(reg["c0"], reg["c1"]))
            occ[sl] += ndimage.binary_dilation(real["line"], structure=np.ones((3, 3), bool)).astype(np.uint16)
            for f in fams:
                occ_f[f][sl] += real["fam"][f].astype(np.uint16)
            if not asg[reg["name"]]:
                continue
            d, (ir, ic) = ndimage.distance_transform_edt(~real["line"], return_indices=True) if real["line"].any() else \
                (np.full(real["line"].shape, np.inf), (np.zeros(real["line"].shape, int), np.zeros(real["line"].shape, int)))
            for k, (idx, ps, st) in asg[reg["name"]].items():
                update(acc[(reg["name"], k)], ps, st, real, (d, ir, ic), mc)
            del real, d, ir, ic
        manifest.append({"realization": r, "seed": prm["seed"], "n_paths": n_paths,
                         "seconds": round(time.perf_counter() - t0, 1), "params": json.dumps(prm, default=float)})
        m8.say(f"s07 realisation {r + 1}/{R}: {n_paths} paths, {time.perf_counter() - t0:.0f}s")
        if (r + 1) % every == 0 or r + 1 == R:
            arrs = {"done": np.array(r + 1), "occ": occ, **{f"occ:{f}": v for f, v in occ_f.items()}}
            for (rg, k), a in acc.items():
                arrs.update(a.to_npz(f"{rg}#{k}"))
            atomic_save(sp, lambda t: np.savez_compressed(t, **arrs))
            atomic_save(m8.ws.root / man_rel, lambda t: pd.DataFrame(manifest).to_csv(t, index=False))
            m8.ws._track(sp)
            m8.ws._track(m8.ws.root / man_rel)
            m8.push([f"{m8.cfg['output_dir']}/monte_carlo/realizations"])
    rows = []
    for (rg, k), a in acc.items():
        idx = asg[rg][k][0]
        cs = pools[k]
        f = finalize(a, R)
        f.insert(0, "cand_id", cs.table.cand_id.to_numpy()[idx] if "cand_id" in cs.table else idx)
        f.insert(0, "uid", cs.table.uid.astype(str).to_numpy()[idx] if "uid" in cs.table else "")
        f.insert(0, "unit", k)
        f["mc_region"] = rg
        rows.append(f)
    stab = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    covered = np.zeros((H, W), bool)
    for reg in regs:
        covered[reg["r0"]:reg["r1"], reg["c0"]:reg["c1"]] = True
    return {"stability": stab, "occupancy": occ, "occupancy_family": occ_f, "covered": covered, "n_real": R,
            "manifest": pd.DataFrame(manifest), "regions": pd.DataFrame(regs), "key": key,
            "distributions": mc["distributions"],
            "uncertainty_type": {f["family"]: ("measured" if (mc.get("measured_noise") or {}).get(f["family"]) is not None
                                               else "heuristic_sensitivity") for f in mc["families"]}}


# =============================================================================== MC-only structures + evaluation
def extras(m8, out: dict, pools: dict) -> dict:
    """Occupancy crests >= threshold that no section-06 candidate already explains -> {unit: CandidateSet}."""
    from ..gems3.candidates import CandidateSet
    from ..m7.generator import mask_to_paths, stable_id

    mc, gc = m8.cfg["monte_carlo"], m8.cfg["generation"]
    R = max(int(out["n_real"]), 1)
    occf = np.where(out["covered"], out["occupancy"] / R, 0.0).astype(np.float32)
    sm = ndimage.gaussian_filter(occf, 0.7)
    mask = (sm >= float(mc["occupancy_extra_threshold"])) & out["covered"]
    paths, _ = mask_to_paths(mask, {**gc, "min_len_km": mc["min_extra_len_km"], "link_gap_px": 0})
    res = {}
    for key, cs in pools.items():
        u = m8.unit(key)
        if u is None:
            continue
        own = np.zeros(u.ev.shape, bool)
        if cs is not None and len(cs):
            own = ndimage.binary_dilation(cs.raster(u.ev.shape) > 0, structure=_disk(int(gc["dedup_radius_px"])))
        rows, ps = [], []
        off = np.array([u.sl[0].start, u.sl[1].start])
        for p in paths:
            q = p - off
            ok = (q[:, 0] >= 0) & (q[:, 0] < u.ev.shape[0]) & (q[:, 1] >= 0) & (q[:, 1] < u.ev.shape[1])
            q = q[ok]
            if len(q) < 3:
                continue
            q = q[u.ev[q[:, 0], q[:, 1]]]
            if len(q) < 3 or own[q[:, 0], q[:, 1]].mean() >= float(gc["dedup_overlap"]):
                continue
            rows.append({"uid": stable_id("mc_occupancy", 0, q + off), "source": "mc_occupancy", "modality": "montecarlo",
                         "kind": "mc_occupancy", "mode": "occupancy", "scale_m": np.nan,
                         "network_id": f"{key}/mc:{len(ps)}", "conf_lf": float(occf[(q + off)[:, 0], (q + off)[:, 1]].mean()),
                         "strength_mean": np.nan, "coherence_mean": np.nan, "persistence_mean": np.nan,
                         "junction_ends": 0, "linked_px": 0})
            ps.append(q)
        res[key] = CandidateSet(pd.DataFrame(rows), ps)
    return res


def evaluate(m8, pools: dict, stab: pd.DataFrame, ext: dict) -> dict:
    """EVALUATION (reads truth): stability vs truth, and unique coverage of MC-only structures at matched density."""
    from ..gems3.oracle import candidate_truth_stats
    from ..m7.coverage import density_subset
    from ..m7.protocol import system_hits

    rows, uniq = [], []
    cap = float(m8.cfg["oracle"]["gate_density_km_per_100km2"])
    for key, cs in pools.items():
        u = m8.unit(key)
        if u is None or cs is None or not len(cs) or not u.eval_truth().any():
            continue
        st = candidate_truth_stats(cs, u)
        s = stab[stab.unit == key].set_index("cand_id") if len(stab) else pd.DataFrame()
        if len(s):
            f = s.mc_freq.reindex(cs.table.cand_id.to_numpy()).to_numpy()
            ok = np.isfinite(f)
            if ok.sum() >= 9:
                q = np.quantile(f[ok], [1 / 3, 2 / 3])
                b = np.where(~ok, -1, np.digitize(f, q))
                for k, name in ((0, "unstable"), (1, "intermediate"), (2, "stable"), (-1, "outside_mc_region")):
                    m = b == k
                    if not m.any():
                        continue
                    hs = system_hits(m8.ctx, u, cs.subset(np.flatnonzero(m).tolist()).raster(u.ev.shape), m8.sh)
                    rows.append({"task": u.task, "zone": u.zone, "stability_bin": name, "n_candidates": int(m.sum()),
                                 "share_near_truth_kd05": float((st.kd_mean.to_numpy()[m] >= 0.5).mean()),
                                 "share_far_1500m": float((st.dist_median_km.to_numpy()[m] >= 1.5).mean()),
                                 "systems_covered": int(hs.recovered.sum()) if len(hs) else 0,
                                 "km": float(sum(len(cs.paths[i]) for i in np.flatnonzero(m)) * 0.112)})
        e = ext.get(key)
        if e is not None and len(e):
            from ..gems3.candidates import CandidateSet

            both = CandidateSet(pd.concat([cs.table, e.table.assign(structural_score=e.table.conf_lf)], ignore_index=True),
                                cs.paths + e.paths)
            h0 = system_hits(m8.ctx, u, density_subset(cs, u, cap).raster(u.ev.shape), m8.sh)
            h1 = system_hits(m8.ctx, u, density_subset(both, u, cap).raster(u.ev.shape), m8.sh)
            ha = system_hits(m8.ctx, u, e.raster(u.ev.shape), m8.sh)
            hp = system_hits(m8.ctx, u, cs.raster(u.ev.shape), m8.sh)
            m = ha.merge(hp, on="system_id", suffixes=("_mc", "_pool"))
            uniq.append({"task": u.task, "zone": u.zone, "n_mc_only": int(len(e)),
                         "systems_only_mc": int((m.recovered_mc & ~m.recovered_pool).sum()) if len(m) else 0,
                         "systems_at_gate_pool": int(h0.recovered.sum()) if len(h0) else 0,
                         "systems_at_gate_pool_plus_mc": int(h1.recovered.sum()) if len(h1) else 0})
    return {"stability_vs_truth": pd.DataFrame(rows), "unique": pd.DataFrame(uniq)}


def gate3(m8, unique: pd.DataFrame, ranking_summary: pd.DataFrame | None = None) -> pd.DataFrame:
    g = m8.cfg["monte_carlo"]["gate"]
    prim = m8.prim
    u = unique[unique.task == prim] if len(unique) else unique
    n_only = int(u.systems_only_mc.sum()) if len(u) else 0
    at_gain = int((u.systems_at_gate_pool_plus_mc - u.systems_at_gate_pool).sum()) if len(u) else 0
    gain = np.nan
    if ranking_summary is not None and len(ranking_summary):
        r = ranking_summary[ranking_summary.task == prim].set_index("method").dti_crossfit
        if {"physics_mc", "physics_only"} <= set(r.index):
            gain = float(r["physics_mc"] - r["physics_only"])
    ok_cov = at_gain >= int(g["min_unique_systems"])
    ok_rank = np.isfinite(gain) and gain >= float(g["min_ranking_gain"])
    dec = "USE_IN_MODEL" if (ok_cov or ok_rank) else ("UNCERTAINTY_REPORTING_ONLY" if np.isfinite(gain) or len(u)
                                                     else "INCONCLUSIVE")
    return pd.DataFrame([{"gate": "G3_monte_carlo", "task": prim, "systems_only_mc_all": n_only,
                          "additional_systems_at_gate_density": at_gain, "ranking_gain_physics_mc": gain,
                          "pass_coverage": ok_cov, "pass_ranking": ok_rank, "decision": dec,
                          "profile": m8.profile["name"], "provisional": bool(m8.profile["provisional"]),
                          "rule": (f"additional systems at matched density >= {g['min_unique_systems']} OR ranking gain "
                                   f">= {g['min_ranking_gain']}"),
                          "note": "MC detection frequency = processing stability, not a calibrated fault probability"}])
