"""Experiment 2: return to the geophysical signals (label-free measurements; interpretation is kept separate).

Layers (native 100 m, full grid, cached): raw magnetic (RTP), gravity, detrended elevation; horizontal gradient
magnitude (THDR) at several smoothing scales; tilt derivative; structure-tensor coherence; magnetic / gravity strike
agreement (``p6.stack.pf_full``); a label-free lineament score (scale persistence x coherence, averaged over the
potential fields). Everything comes from ``m8.potential_fields`` / ``p6.stack`` (reused, not re-derived).

Per candidate and modality (``modality_evidence``): signal strength, local contrast to background, orientation,
coherence, scale persistence, cross-modal agreement, step contrast across the trace, texture change, along-strike
correlation of the two side profiles (low = disruption / truncation) and its best lag (non-zero with high correlation
= apparent lateral offset). ``evidence_flags`` turns them into CONTRAST / DISRUPTION / CONTINUITY / MULTI_SUPPORT
observations and an alternative-explanation flag. These are OBSERVATIONS, never fault labels.

Status vocabulary: PRESENT (observed, above threshold), ABSENT (observed, below threshold), NOT_OBSERVED (no valid
pixel in the window), UNAVAILABLE (band / source missing). "Absent" is never inferred from "unavailable".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

PRESENT, ABSENT, NOT_OBSERVED, UNAVAILABLE = "PRESENT", "ABSENT", "NOT_OBSERVED", "UNAVAILABLE"
PX_KM = 0.1


# =============================================================================== pure: layers
def hillshade(z: np.ndarray, azimuth: float = 315.0, altitude: float = 45.0, px_m: float = 100.0) -> np.ndarray:
    """Standard hillshade in [0, 1] (NaN where ``z`` is NaN)."""
    zz = np.where(np.isfinite(z), z, np.nanmean(z) if np.isfinite(z).any() else 0.0)
    gy, gx = np.gradient(zz, px_m)
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(360.0 - azimuth + 90.0), np.radians(altitude)
    hs = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    return np.where(np.isfinite(z), np.clip(hs, 0, 1), np.nan).astype(np.float32)


def multi_hillshade(z: np.ndarray, px_m: float = 100.0) -> np.ndarray:
    """Mean of four illumination azimuths (reduces the azimuth bias of single-direction hillshade)."""
    return np.nanmean(np.stack([hillshade(z, a, 45.0, px_m) for a in (315, 45, 135, 225)]), 0).astype(np.float32)


def field_parts(a: np.ndarray, sigma_px: float) -> dict:
    """THDR, strike, coherence of one field at one scale (``m8.potential_fields.gradient_parts``)."""
    from ..m8 import potential_fields as PFm

    g = PFm.gradient_parts(np.asarray(a, np.float32), float(sigma_px))
    return {"thdr": np.asarray(g["thdr"], np.float32), "strike": np.asarray(g["strike"], np.float32),
            "coherence": np.asarray(g["coherence"], np.float32)}


def tilt(a: np.ndarray, sigma_px: float = 1.0) -> np.ndarray:
    """Tilt derivative atan2(VDR, THDR) (rad), as ``p6.stack.potential_features`` computes it for magnetics."""
    from ..m8 import potential_fields as PFm

    f, m = PFm.fill(np.asarray(a, np.float32))
    vz = PFm.vdr_fft(PFm.smooth(f, sigma_px))
    ge, gn = PFm.hgrad(f, sigma_px, (0.1, 0.1))
    return np.asarray(PFm.restore(np.arctan2(vz, np.hypot(ge, gn) + 1e-12), m), np.float32)


def rank_valid(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    from ..p6.stack import rank_in

    return rank_in(np.asarray(a, np.float32), np.asarray(valid, bool))


def lineament_score(ranks_by_field: dict, coherence_by_field: dict) -> np.ndarray:
    """Label-free lineament score: per field, scale persistence (min rank over the scales) x structure-tensor
    coherence; averaged over the available fields. No label enters it."""
    parts = []
    for f, rs in ranks_by_field.items():
        if not rs:
            continue
        pers = np.fmin.reduce(rs) if len(rs) > 1 else rs[0]
        coh = coherence_by_field.get(f)
        parts.append(pers * (np.clip(coh, 0, 1) if coh is not None else 1.0))
    if not parts:
        raise ValueError("no potential field available for the lineament score")
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.stack(parts), 0).astype(np.float32)


# =============================================================================== pure: geometry / profiles
def orientation(rr: np.ndarray, cc: np.ndarray) -> float:
    """Axial orientation (rad, from +column toward +row) of a pixel set (second moments); NaN for < 2 px."""
    if len(rr) < 2:
        return float("nan")
    r, c = rr - rr.mean(), cc - cc.mean()
    return float(np.mod(0.5 * np.arctan2(2 * (r * c).mean(), (c * c).mean() - (r * r).mean()), np.pi))


def probe_line(r0: float, c0: float, theta: float, length_px: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(-length_px // 2, length_px // 2 + 1, dtype=float)
    return r0 + t * np.sin(theta), c0 + t * np.cos(theta)


def sample(a: np.ndarray, rr: np.ndarray, cc: np.ndarray) -> np.ndarray:
    """Bilinear samples (NaN outside / where the field is NaN)."""
    a = np.asarray(a, np.float64)
    ok = np.isfinite(a)
    v = ndimage.map_coordinates(np.where(ok, a, 0.0), [rr, cc], order=1, mode="constant", cval=np.nan)
    m = ndimage.map_coordinates(ok.astype(float), [rr, cc], order=1, mode="constant", cval=0.0)
    return np.where(m > 0.99, v, np.nan)


def perpendicular_profile(a: np.ndarray, r0: float, c0: float, theta: float, half_px: int) -> pd.DataFrame:
    n = np.arange(-half_px, half_px + 1, dtype=float)
    rr, cc = r0 + n * np.cos(theta), c0 - n * np.sin(theta)
    return pd.DataFrame({"distance_km": n * PX_KM, "value": sample(a, rr, cc)})


def side_correlation(a: np.ndarray, rr: np.ndarray, cc: np.ndarray, theta: float, offset_px: float,
                     max_lag: int) -> dict:
    """Along-strike profiles on both sides of a trace: max normalised cross-correlation and its lag. A pattern that
    continues across the trace gives high correlation at lag 0; an offset gives high correlation at a non-zero lag;
    a truncation / disruption gives low correlation at every lag."""
    nr, nc = np.cos(theta), -np.sin(theta)
    a1 = sample(a, rr + offset_px * nr, cc + offset_px * nc)
    a2 = sample(a, rr - offset_px * nr, cc - offset_px * nc)
    best, lag = -np.inf, 0
    for k in range(-max_lag, max_lag + 1):
        x, y = (a1[k:], a2[:len(a2) - k]) if k >= 0 else (a1[:k], a2[-k:])
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 8 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
            continue
        r = float(np.corrcoef(x[ok], y[ok])[0, 1])
        if r > best:
            best, lag = r, k
    return {"side_corr": best if np.isfinite(best) else np.nan, "side_lag_px": lag if np.isfinite(best) else np.nan}


def step_and_texture(a: np.ndarray, rr: np.ndarray, cc: np.ndarray, theta: float, width_px: int) -> dict:
    """Mean step across the trace and texture (std) change, both in units of the window std."""
    nr, nc = np.cos(theta), -np.sin(theta)
    s = np.nanstd(a) if np.isfinite(a).any() else np.nan
    left = np.concatenate([sample(a, rr + k * nr, cc + k * nc) for k in range(1, width_px + 1)])
    right = np.concatenate([sample(a, rr - k * nr, cc - k * nc) for k in range(1, width_px + 1)])
    if not np.isfinite(s) or s == 0 or not np.isfinite(left).any() or not np.isfinite(right).any():
        return {"step_z": np.nan, "texture_change_z": np.nan}
    return {"step_z": float(abs(np.nanmean(left) - np.nanmean(right)) / s),
            "texture_change_z": float(abs(np.nanstd(left) - np.nanstd(right)) / s)}


def modality_evidence(win: dict, rr: np.ndarray, cc: np.ndarray, theta: float, pc: dict,
                      probe: tuple | None = None) -> dict:
    """Measurements of ONE modality on a candidate trace inside one window. ``win``: field, thdr_<sigma>,
    coherence, strike (window crops; ``field`` None = UNAVAILABLE). ``rr, cc``: trace pixels (any order: strength,
    coherence); ``probe``: ORDERED straight samples along the trace orientation (step, texture, side correlation;
    defaults to ``rr, cc`` when they are already ordered)."""
    pr_, pc_ = probe if probe is not None else (rr, cc)
    f = win.get("field")
    if f is None:
        return {"status": UNAVAILABLE}
    if not np.isfinite(f).any() or not np.isfinite(sample(f, rr, cc)).any():
        return {"status": NOT_OBSERVED}
    th = pc["thresholds"]
    out = {}
    pers = []
    for s in pc["sigmas_px"]:
        g = win.get(f"thdr_{s}")
        if g is None:
            continue
        v = sample(g, rr, cc)
        bg = np.nanmedian(g)
        # percentile of the trace's mean gradient among the window's pixels = scale-wise strength rank
        pr = float((g[np.isfinite(g)] < np.nanmean(v)).mean()) if np.isfinite(v).any() else np.nan
        out[f"strength_rank_s{s}"] = pr
        if s == pc["sigmas_px"][0]:
            out["signal_strength"] = float(np.nanmean(v))
            out["local_contrast"] = float(np.nanmean(v) / bg) if bg and np.isfinite(bg) else np.nan
        pers.append(pr)
    out["scale_persistence"] = float(np.nanmin(pers[:3])) if pers else np.nan
    coh = win.get("coherence")
    out["coherence"] = float(np.nanmean(sample(coh, rr, cc))) if coh is not None else np.nan
    st = win.get("strike")
    if st is not None:
        sv = sample(st, rr, cc)
        out["field_orientation_rad"] = float(np.angle(np.nanmean(np.exp(2j * sv[np.isfinite(sv)]))) / 2) \
            if np.isfinite(sv).any() else np.nan
    out.update(step_and_texture(f, pr_, pc_, theta, max(int(round(1.0 / PX_KM)), 1)))
    out.update(side_correlation(f, pr_, pc_, theta, pc["parallel_offset_km"] / PX_KM, int(pc["max_lag_px"])))
    present = np.isfinite(out.get("scale_persistence", np.nan)) and out["scale_persistence"] >= th["present_rank"]
    out["status"] = PRESENT if present else ABSENT
    return out


def axial_agreement(a: float, b: float) -> float:
    """cos(2 delta) mapped to [0, 1] (1 = parallel strikes)."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return np.nan
    return float(0.5 + 0.5 * np.cos(2 * (a - b)))


def evidence_flags(ev: dict, th: dict) -> dict:
    """Observations from the per-modality measurements (``ev``: modality -> dict). NOT a fault label."""
    obs = {m: e for m, e in ev.items() if e.get("status") in (PRESENT, ABSENT)}
    present = [m for m, e in obs.items() if e["status"] == PRESENT]
    contrast = [m for m, e in obs.items() if (e.get("step_z") or 0) >= th["step_z"]]
    disrupt = [m for m, e in obs.items() if np.isfinite(e.get("side_corr", np.nan)) and (
        e["side_corr"] < th["disruption_corr"] or (abs(e.get("side_lag_px") or 0) >= th["offset_lag_px"] and
                                                   e["side_corr"] >= th["offset_corr"]))]
    contin = [m for m, e in obs.items() if e["status"] == PRESENT and (e.get("coherence") or 0) >= th["coherence"]]
    agree = axial_agreement(ev.get("magnetic", {}).get("field_orientation_rad", np.nan),
                            ev.get("gravity", {}).get("field_orientation_rad", np.nan))
    multi = len(present) >= 2 and (not np.isfinite(agree) or agree >= 0.75)
    return {"n_observed_modalities": len(obs), "present_modalities": ";".join(present),
            "contrast_modalities": ";".join(contrast), "disruption_modalities": ";".join(disrupt),
            "continuity_modalities": ";".join(contin), "mag_grav_strike_agreement": agree,
            "GEOLOGICAL_CONTRAST": bool(contrast), "GEOLOGICAL_DISRUPTION": bool(disrupt),
            "STRUCTURAL_CONTINUITY": bool(contin), "MULTI_SUPPORT": bool(multi),
            "alternative_explanation_flag": bool(contrast and not disrupt and not multi),
            "alternative_explanation": ("step contrast without disruption or multi-modal support: a lithological "
                                        "contact / intrusion boundary is as plausible as a fault")
            if contrast and not disrupt and not multi else "",
            "unavailable_modalities": ";".join(m for m, e in ev.items() if e.get("status") == UNAVAILABLE),
            "not_observed_modalities": ";".join(m for m, e in ev.items() if e.get("status") == NOT_OBSERVED)}


# =============================================================================== pure: localization
def localization(pred: np.ndarray, truth: np.ndarray, ev: np.ndarray, tols_km) -> dict:
    """Distances (km) predicted -> nearest truth pixel and truth -> nearest predicted pixel inside ``ev``; shares
    within each tolerance and the corridor index share(<= 3 km) / share(<= 300 m) of the truth side."""
    p, t = np.asarray(pred, bool) & ev, np.asarray(truth, bool) & ev
    out = {"pred_px": int(p.sum()), "truth_px": int(t.sum())}
    if not p.any() or not t.any():
        return out
    dpt = ndimage.distance_transform_edt(~t)[p] * PX_KM
    dtp = ndimage.distance_transform_edt(~p)[t] * PX_KM
    out.update(pred_to_truth_median_km=float(np.median(dpt)), truth_to_pred_median_km=float(np.median(dtp)))
    for k in tols_km:
        out[f"pred_within_{k:g}km"] = float((dpt <= k).mean())
        out[f"truth_within_{k:g}km"] = float((dtp <= k).mean())
    lo, hi = min(tols_km), max(tols_km)
    out["corridor_index"] = float(out[f"truth_within_{hi:g}km"] / max(out[f"truth_within_{lo:g}km"], 1e-9))
    return out


# =============================================================================== Colab side: full-grid layers
LAYER_KEYS = ("mag", "grav", "elev", "slope", "mag_thdr_1", "mag_thdr_3", "mag_thdr_8", "mag_thdr_20",
              "grav_thdr_1", "grav_thdr_3", "grav_thdr_8", "grav_thdr_20", "mag_coh", "grav_coh", "mag_strike",
              "grav_strike", "mag_tilt", "grav_tilt", "mg_agree", "lineament")


def full_layers(R) -> dict:
    """Label-free full-grid layers (cached ``physics/cache/layers.npz``, float16)."""
    rel = "physics/cache/layers.npz"
    p = R.ws.path(rel)
    if p.exists():
        with np.load(p) as z:
            return {k: z[k].astype(np.float32) for k in z.files}
    d = R.ctx.data
    valid = np.asarray(d.valid, bool)
    b = R.rc["bands"]
    out = {}

    def band(name):
        return np.where(valid, np.asarray(d.read_band(name), np.float32), np.nan) if name in d.band_names else None

    sig = R.rc["physics"]["sigmas_px"]
    ranks, coh = {}, {}
    for key, nm in (("mag", b["magnetic"]), ("grav", b["gravity"])):
        a = band(nm)
        if a is None:
            continue
        out[key] = a
        ranks[key] = []
        for s in sig:
            fp = field_parts(a, s)
            out[f"{key}_thdr_{s}"] = fp["thdr"]
            if s == sig[0]:
                out[f"{key}_coh"], out[f"{key}_strike"] = fp["coherence"], fp["strike"]
                coh[key] = fp["coherence"]
            if s in sig[:2]:
                ranks[key].append(rank_valid(fp["thdr"], valid))
        out[f"{key}_tilt"] = tilt(a, 1.0)
    for key, nm in (("elev", b["terrain"]), ("slope", b["terrain_slope"])):
        a = band(nm)
        if a is not None:
            out[key] = a
    from ..p6.stack import pf_full

    pf = pf_full(R.p6)
    if "mg_orient_agree" in pf:
        out["mg_agree"] = np.asarray(pf["mg_orient_agree"], np.float32)
    out["lineament"] = np.where(valid, lineament_score(ranks, coh), np.nan).astype(np.float32)
    R.save_npz(rel, **{k: v.astype(np.float16) for k, v in out.items()})
    return out


# =============================================================================== Colab side: candidate evidence
MODALITIES = {"magnetic": "mag", "gravity": "grav", "terrain": "elev"}


def _filled(a: np.ndarray) -> np.ndarray:
    return np.nan_to_num(a, nan=float(np.nanmean(a)) if np.isfinite(a).any() else 0.0)


def window_of(L: dict, key: str, sl, sigmas=(1, 3, 8, 20)) -> dict:
    """Window crops of one modality: field, gradient magnitude per scale, coherence, strike."""
    f = L.get(key)
    if f is None:
        return {"field": None}
    w = {"field": f[sl]}
    for k, v in L.items():
        if k.startswith(f"{key}_thdr_"):
            w[f"thdr_{k.rsplit('_', 1)[1]}"] = v[sl]
    if f"{key}_coh" in L:
        w["coherence"], w["strike"] = L[f"{key}_coh"][sl], L[f"{key}_strike"][sl]
    if key == "elev":                                           # terrain: gradient of the elevation at each scale
        for s in sigmas:
            g = ndimage.gaussian_gradient_magnitude(_filled(f[sl]), s).astype(np.float32)
            w[f"thdr_{s}"] = np.where(np.isfinite(f[sl]), g, np.nan)
    return w


def candidate_evidence(R, cat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(evidence_by_modality long table, structural flags per candidate, perpendicular profiles)."""
    L = full_layers(R)
    pc = R.rc["physics"]
    half = int(R.rc["atlas"]["patch_size_px"]) // 2
    H, W = R.ctx.data.shape
    rows, flags, prof = [], [], []
    for c in cat.itertuples():
        r0, c0 = int(c.row), int(c.col)
        sl = (slice(max(r0 - half, 0), min(r0 + half, H)), slice(max(c0 - half, 0), min(c0 + half, W)))
        geom = R.candidate_pixels(c.candidate_id)
        n_probe = int(pc["probe_km"] / PX_KM)
        if geom is not None and len(geom[0]) >= 3:
            rr, cc = (geom[0] - sl[0].start).astype(float), (geom[1] - sl[1].start).astype(float)
            theta = orientation(rr, cc)
            ext = int(np.hypot(np.ptp(rr), np.ptp(cc)))
            line = probe_line(float(rr.mean()), float(cc.mean()), theta, max(n_probe, ext))
            probe = "candidate_trace"
        else:
            g = L.get("mag_strike")
            theta = float(g[r0, c0]) if g is not None and np.isfinite(g[r0, c0]) else 0.0
            rr, cc = probe_line(r0 - sl[0].start, c0 - sl[1].start, theta, n_probe)
            line = (rr, cc)
            probe = "straight_probe (point candidate, local magnetic strike)"
        ev = {}
        for mod, key in MODALITIES.items():
            win = window_of(L, key, sl, pc["sigmas_px"])
            e = modality_evidence(win, np.asarray(rr, float), np.asarray(cc, float), theta, pc, line)
            ev[mod] = e
            rows.append({"candidate_id": c.candidate_id, "modality": mod, "probe": probe, **e})
            if e.get("status") in (PRESENT, ABSENT):
                f = win["field"]
                for s in (0, pc["sigmas_px"][2]):                       # raw field and one smoothed scale
                    a = f if s == 0 else np.where(np.isfinite(f), ndimage.gaussian_filter(_filled(f), s), np.nan)
                    pf = perpendicular_profile(a, float(np.mean(rr)), float(np.mean(cc)), theta,
                                               int(pc["profile_half_km"] / PX_KM))
                    prof.append(pf.assign(candidate_id=c.candidate_id, modality=mod, smoothing_px=s))
        fl = evidence_flags(ev, pc["thresholds"])
        flags.append({"candidate_id": c.candidate_id, "category": c.category, "probe": probe,
                      "trace_orientation_deg": float(np.degrees(theta) % 180) if np.isfinite(theta) else np.nan, **fl})
    return pd.DataFrame(rows), pd.DataFrame(flags), (pd.concat(prof, ignore_index=True) if prof else pd.DataFrame())


def scale_localization(R) -> pd.DataFrame:
    """Do broad responses LOCATE faults or only indicate corridors? On each held-out geo fold (V3): label-free crest
    lines of the magnetic / gravity gradient at every smoothing scale, the label-free lineament crests and the frozen
    submitted W2 map, vs that fold's mapped faults (diagnostic: mapped faults are not hidden new faults)."""
    from ..gems3 import representation as R_

    L = full_layers(R)
    pc = R.rc["physics"]
    rows = []
    for su in [s for s in R.units if s.scenario == "V3"]:
        u = su.unit
        truth = u.eval_truth()
        srcs = {}
        for f in ("mag", "grav"):
            for s in pc["sigmas_px"]:
                k = f"{f}_thdr_{s}"
                if k in L:
                    srcs[(f"{f}_gradient_crest", s)] = L[k][u.sl]
        srcs[("label_free_lineament", 0)] = L["lineament"][u.sl]
        for (name, s), a in srcs.items():
            q = R_.apply({"select": "crest"}, np.where(u.ev, np.nan_to_num(a, nan=0.0), 0.0), u.ev, 0.02,
                         sigma_px=1.0, eligible=R.gm.elig(u)) > 0
            rows.append({"unit": su.key, "zone": su.zone, "source": name, "scale_px": s, "scale_km": s * PX_KM,
                         **localization(q, truth, u.ev, pc["localization_km"])})
        sm = R.gm.frozen_smap(su)
        if sm is not None:
            fz = R.gm.frozen_policy()["cell"]
            from ..geomoe.evaluate import policy_q

            q = policy_q(R.gm, u, sm, str(fz["rep"]), float(fz["frac"])) > 0
            rows.append({"unit": su.key, "zone": su.zone, "source": "frozen_submitted_W2", "scale_px": np.nan,
                         **localization(q, truth, u.ev, pc["localization_km"])})
        u.release()
    return pd.DataFrame(rows)
