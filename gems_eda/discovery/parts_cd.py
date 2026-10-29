"""Part C (F023-F036): fault-centred physical signatures vs matched controls. Part D (F037-F047): spatial scale and
localisation. Effects: Cliff's delta fault vs matched control, cluster bootstrap over fault systems; regional
robustness = same-sign share across the 3 x 3 regions with enough stations."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import viz as V
from .core import (DESCRIPTIVE, INCONCLUSIVE, NOT_SUPPORTED, PALETTE, SUPPORTED, Result, cliffs_delta, cluster_ci,
                   cluster_effect, effect_result, robustness, status_of)
from .data import LOCAL, PRIMARY, Sampler, along_normal, crest_mask, scale_row
from .parts_ab import TEST_OUT, domain_windows, sp

METRICS = ("abs_step", "abs_peak", "local_contrast", "flank_asymmetry", "abs_grad_peak_offset_km")
STRUCT = ("along_persistence", "side_corr", "abs_side_lag_km", "strike_change_deg", "coherence")


# =============================================================================== effect table
def effects(E) -> pd.DataFrame:
    """Every (signal, metric) and (field, structure metric) effect, with regional effects; cached + saved."""
    if "effects" in E.D.cache:
        return E.D.cache["effects"]
    p = E.ws.path("tables/matched_control_tests.csv")
    if p.exists() and E.current_stage != "e05_signatures":
        E.D.cache["effects"] = pd.read_csv(p)
        return E.D.cache["effects"]
    S, M, _, _, _, ST = E.stations_data()
    rows = []
    minr = int(E.c["statistics"]["min_region_stations"])

    def one(sig, metric, f, c, fg, cg, fr, cr):
        e = cluster_effect(f, fg, c, cg, E.n_boot, E.seed)
        reg = []
        for k in np.unique(fr):
            a, b = f[fr == k], c[cr == k]
            if np.isfinite(a).sum() >= minr and np.isfinite(b).sum() >= minr:
                reg.append({"region": int(k), "effect": cliffs_delta(a, b)})
        rob = robustness(pd.DataFrame(reg), e["effect"], E.min_effect)
        rows.append({"signal": sig, "metric": metric, **e, **rob,
                     "status": status_of(e["effect"], e["ci_lo"], e["ci_hi"], E.min_effect),
                     "regional": ";".join(f"{r['region']}:{r['effect']:+.2f}" for r in reg)})

    for sig, g in M.groupby("signal"):
        F, C = g[g.kind == "fault"], g[g.kind == "control"]
        for m in METRICS:
            one(sig, m, F[m].to_numpy(), C[m].to_numpy(), F.system.to_numpy(), C.system.to_numpy(),
                F.region.to_numpy(), C.region.to_numpy())
    X = ST.merge(S[["sample_id", "kind", "system", "region"]], on="sample_id")
    X["abs_side_lag_km"] = X.side_lag_km.abs()
    for fld, g in X.groupby("field"):
        F, C = g[g.kind == "fault"], g[g.kind == "control"]
        for m in STRUCT:
            one(f"{fld}:structure", m, F[m].to_numpy(), C[m].to_numpy(), F.system.to_numpy(), C.system.to_numpy(),
                F.region.to_numpy(), C.region.to_numpy())
    T = pd.DataFrame(rows)
    T["modality"] = [E.D.modality(s.split(":")[0]) if ":" not in s or not s.endswith("structure") else
                     s.split(":")[0] for s in T.signal]
    E.ws.save_table(T, "tables/matched_control_tests.csv")
    E.D.cache["effects"] = T
    return T


def aligned(cube: np.ndarray, off: np.ndarray, core: float = 0.5) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        step = np.nanmean(cube[..., off > core], -1) - np.nanmean(cube[..., off < -core], -1)
    return np.where((step < 0)[..., None], cube[..., ::-1], cube)


def agg(E, sig: str, mask: np.ndarray, aligned_: bool = True) -> pd.DataFrame:
    S, M, cube, off, sigs, _ = E.stations_data()
    i = sigs.index(sig)
    c = cube[mask, i, :]
    c = aligned(c[:, None, :], off)[:, 0, :] if aligned_ else c
    q = np.nanpercentile(c, [25, 50, 75], 0)
    return pd.DataFrame({"offset_km": off, "q25": q[0], "q50": q[1], "q75": q[2], "n": int(mask.sum())})


def profile_figure(E, fid, name, title, signals, question, why, implication_sup, pr=1, context=False):
    S, M, cube, off, sigs, _ = E.stations_data()
    signals = [s for s in signals if s in sigs]
    s = sp(fid, name, title, "C", question, "Polarity-aligned profiles across mapped faults vs matched controls "
           "(same region, physical setting and normal direction)", why, "|step| and |peak| across +-0.5 km core; "
           "Cliff's delta with cluster CI; regional same-sign share", TEST_OUT,
           {SUPPORTED: implication_sup, NOT_SUPPORTED: "Do not promote this signal as fault evidence at this scale.",
            INCONCLUSIVE: "Undecided: needs more systems or a different scale (see part D)."},
           "Matched controls are not confirmed non-faults; labels incomplete; profiles mirror the higher flank to the "
           "right (polarity alignment)." + (" Regional context bands: localisation metrics are not meaningful." if
                                             context else ""), pr)
    if not signals:
        return E.reg.skip(s, "no band of this modality present")
    T = effects(E)
    fig, ax = V.grid(2 * len(signals), 2, 5.0, 3.2)
    fk, ck = (S.kind == "fault").to_numpy(), (S.kind == "control").to_numpy()
    rows, best = [], None
    rng = np.random.default_rng(E.seed)
    for j, sig in enumerate(signals):
        i = sigs.index(sig)
        a0, a1 = ax[2 * j], ax[2 * j + 1]
        for k in rng.choice(np.flatnonzero(fk), min(25, int(fk.sum())), replace=False):
            a0.plot(off, aligned(cube[k:k + 1, i:i + 1, :], off)[0, 0], color=PALETTE["faults"], lw=0.3, alpha=0.4)
        a0.set(title=f"{sig}: 25 individual fault profiles", xlabel="offset across fault (km)", ylabel="robust z")
        for mask, lab, col in ((fk, "mapped faults", PALETTE.get(E.D.modality(sig))), (ck, "matched controls",
                                                                                      PALETTE["control"])):
            A = agg(E, sig, mask)
            a1.fill_between(A.offset_km, A.q25, A.q75, color=col, alpha=0.25)
            a1.plot(A.offset_km, A.q50, color=col, label=f"{lab} (n={A.n.iloc[0]})")
            rows.append(A.assign(signal=sig, population=lab))
        a1.axvspan(-0.3, 0.3, color="r", alpha=0.08)
        r = T[(T.signal == sig) & (T.metric == "abs_step")]
        a1.set(title=f"{sig}: |step| delta {r.effect.iloc[0]:+.2f} [{r.ci_lo.iloc[0]:+.2f}, {r.ci_hi.iloc[0]:+.2f}]"
               if len(r) else sig, xlabel="offset (km)")
        a1.legend()
        for m in ("abs_step", "abs_peak"):
            x = T[(T.signal == sig) & (T.metric == m)]
            if len(x) and (best is None or abs(x.effect.iloc[0]) > abs(best["effect"])):
                best = x.iloc[0].to_dict()
    res = effect_result(best, E.min_effect, {k: best.get(k) for k in ("same_sign_share", "stability")},
                        f"strongest {best['signal']} {best['metric']}:") if best else Result(INCONCLUSIVE, "no effect")
    E.reg.save(s, fig, pd.concat(rows, ignore_index=True), res)


def f023(E):
    profile_figure(E, "F023", "magnetic_profiles", "Magnetic profiles across mapped faults", ("rtp", "tmi_hg",
                   "rtp:thdr1km"), "Do faults align with magnetic gradients, steps or peaks?", "Faults juxtapose "
                   "rocks of different magnetisation or demagnetise along damage zones", "Magnetic contrast features "
                   "are candidates (check localisation in part D).")


def f024(E):
    profile_figure(E, "F024", "gravity_profiles", "Gravity profiles across mapped faults", ("iso_grav_anom",
                   "iso_grav_anom_hg", "iso_grav_anom:thdr1km"), "Do density contrasts localise faults?",
                   "Basin-bounding normal faults juxtapose dense basement against basin fill", "Gravity-gradient "
                   "features are candidates (expected broad: check part D).")


def f025(E):
    profile_figure(E, "F025", "terrain_profiles", "Terrain profiles across mapped faults", ("det_elev",
                   "det_elev_slope"), "Which faults have a detectable surface expression?", "Active / young faults "
                   "form scarps and range fronts", "Terrain morphology is a candidate (DEM detail could add more).")


def f026(E):
    profile_figure(E, "F026", "basement_profiles", "Basement-depth profiles", ("depth_to_base_surf",),
                   "Do basement changes coincide with mapped faults?", "Basin geometry is fault-controlled",
                   "Basement-step context is a candidate at corridor scale.", 2)


def f027(E):
    profile_figure(E, "F027", "conductivity_profiles", "Conductivity profiles", ("cond_surf",), "Are conductivity "
                   "variations associated with fault locations?", "Fault zones can host fluids / clay alteration",
                   "Conductivity is a candidate secondary evidence.", 2)


def f028(E):
    profile_figure(E, "F028", "regional_context_profiles", "Strain and seismicity as regional context",
                   ("geod_shearrate", "ieq_n100a15"), "Do strain / seismicity bands provide context rather than "
                   "trace evidence?", "Geodetic strain and seismicity are smooth fields at tens of km",
                   "Use as low-frequency context channels, not trace evidence.", 3, context=True)


def f029(E):
    T = effects(E)
    A = E.table("datasets/control_sensitivity.parquet")
    S, M, *_ = E.stations_data()
    s = sp("F029", "fault_vs_control_effects", "Fault vs matched-control effects across all signals", "C",
           "Which observations distinguish mapped faults from similarly situated locations?", "Cliff's delta of "
           "|step| and |peak|, 95 % cluster CI; sensitivity to the control distance band", "Matched controls remove "
           "region / setting / orientation confounding that random negatives would carry", "Effect sizes, CI, "
           "regional stability, alternative control bands", TEST_OUT,
           {SUPPORTED: "Signals with stable supported effects enter the feature-hypothesis register.",
            NOT_SUPPORTED: "No signal separates faults from controls: label-independent features alone will not "
                           "discover faults.", INCONCLUSIVE: "Weak separation; prioritise sensitivity checks."},
           "Controls 2-10 km from mapped traces may include unmapped faults (PU): effects are lower bounds.", 1)
    d = T[T.metric.isin(["abs_step", "abs_peak"]) & ~T.signal.str.endswith("structure")].copy()
    d["label"] = d.signal + " " + d.metric
    d = d.sort_values("effect")
    sens = []
    if len(A):
        F = M[M.kind == "fault"]
        for kind, g in A.groupby("kind"):
            for sig in g.signal.unique():
                fv = F[F.signal == sig].abs_step.to_numpy()
                cv = g[g.signal == sig].abs_step.to_numpy()
                sens.append({"control_set": kind, "signal": sig, "metric": "abs_step", "effect": cliffs_delta(fv, cv)})
    fig, ax = V.grid(2, 2, 6.5, max(3.5, 0.18 * len(d)))
    V.effect_bars(ax[0], d, title="|step| / |peak| effects (main controls)", min_effect=E.min_effect,
                  color_col="modality")
    Sd = pd.DataFrame(sens)
    if len(Sd):
        main = T[T.metric == "abs_step"][["signal", "effect"]].assign(control_set="control_2_10km")
        P = pd.concat([main, Sd[["signal", "effect", "control_set"]]]).pivot_table(index="signal",
                                                                                   columns="control_set",
                                                                                   values="effect")
        for col in P.columns:
            ax[1].plot(P[col].to_numpy(), np.arange(len(P)), "o", label=col)
        ax[1].set_yticks(np.arange(len(P)))
        ax[1].set_yticklabels(P.index, fontsize=6)
        ax[1].axvline(0, color="k", lw=0.6)
        ax[1].legend()
        ax[1].set(title="|step| delta by control distance band", xlabel="Cliff's delta")
    n_sup = int((d.status == SUPPORTED).sum())
    stable = d[(d.status == SUPPORTED) & (d.stability == "STABLE")]
    st = SUPPORTED if len(stable) else (NOT_SUPPORTED if (d.status == NOT_SUPPORTED).all() else INCONCLUSIVE)
    E.reg.save(s, fig, pd.concat([d.assign(table="effects"), Sd.assign(table="sensitivity")], ignore_index=True),
               Result(st, f"{n_sup} of {len(d)} signal-metrics SUPPORTED, {len(stable)} also regionally STABLE: "
                          f"{list(stable.label)[:6]}", {"n_supported": n_sup, "n_stable": len(stable)}, len(d)))


def f030(E):
    T = effects(E)
    s = sp("F030", "gradient_vs_disruption", "Gradient strength vs structural disruption", "C", "Is a "
           "structural-disruption signal more specific than a strong edge?", "Per primary field: |peak| of the 1 km "
           "gradient vs side-profile correlation, apparent offset lag, strike change, along-strike coherence",
           "A lithological contact gives a strong edge with continuous side patterns; a fault can interrupt / "
           "offset them", "Cliff's delta (fault vs control); a NEGATIVE delta of side correlation = more disruption "
           "at faults", TEST_OUT,
           {SUPPORTED: "Disruption metrics separate faults: develop truncation / offset features.",
            NOT_SUPPORTED: "Disruption metrics do not separate faults at 100 m: do not prioritise them.",
            INCONCLUSIVE: "Undecided."}, "Side profiles at +-1 km along a 4 km tangent; 100 m sampling.", 1)
    rows = []
    for fld in ("magnetic", "gravity", "terrain"):
        g = T[T.signal == f"{fld}:structure"]
        rows.append(g)
        sig = {"magnetic": "rtp:thdr1km", "gravity": "iso_grav_anom:thdr1km", "terrain": "det_elev_slope"}[fld]
        rows.append(T[(T.signal == sig) & (T.metric == "abs_peak")])
    d = pd.concat(rows, ignore_index=True)
    if not len(d):
        return E.reg.skip(s, "no structure metrics")
    d["label"] = d.signal + " " + d.metric
    fig, ax = V.grid(1, 1, 8, max(3.5, 0.25 * len(d)))
    V.effect_bars(ax[0], d, title="edge strength vs disruption metrics", min_effect=E.min_effect)
    dis = d[d.metric.isin(["side_corr", "abs_side_lag_km", "strike_change_deg"])]
    best = dis.iloc[dis.effect.abs().argmax()].to_dict() if len(dis) else None
    edge = d[d.metric == "abs_peak"]
    res = effect_result(best, E.min_effect, None, f"strongest disruption {best['signal']} {best['metric']}:") \
        if best else Result(INCONCLUSIVE, "no disruption metric")
    res.summary += f"; strongest edge |delta| {edge.effect.abs().max():.3f}" if len(edge) else ""
    E.reg.save(s, fig, d, res)


def simple_effect_figure(E, fid, name, title, metric, question, why, meas, sup, notsup, lim, pr=2, sign=1):
    T = effects(E)
    d = T[T.metric == metric].copy()
    s = sp(fid, name, title, "C", question, f"Cliff's delta of {metric}, fault vs matched control", why, meas,
           TEST_OUT, {SUPPORTED: sup, NOT_SUPPORTED: notsup, INCONCLUSIVE: "Undecided."}, lim, pr)
    if not len(d):
        return E.reg.skip(s, f"no {metric}")
    d["label"] = d.signal
    fig, ax = V.grid(1, 1, 7, max(3.0, 0.22 * len(d)))
    V.effect_bars(ax[0], d.sort_values("effect"), title=title, min_effect=E.min_effect, color_col="modality")
    best = d.iloc[(sign * d.effect).argmax()].to_dict()
    E.reg.save(s, fig, d, effect_result(best, E.min_effect, {k: best.get(k) for k in ("same_sign_share",
                                                                                     "stability")},
                                        f"best {best['signal']}:"))


def f031(E):
    simple_effect_figure(E, "F031", "signal_asymmetry", "Flank asymmetry of gradients", "flank_asymmetry",
                         "Do fault-related contrasts show consistent asymmetry?", "Dipping faults / juxtaposition "
                         "produce asymmetric gradients", "Normalised |left - right| mean gradient within 3 km",
                         "Asymmetry is a candidate dip-sensitive feature.", "Asymmetry adds nothing.",
                         "Polarity alignment uses the step sign; asymmetry is unsigned.")


def f032(E):
    S, M, *_ = E.stations_data()
    s = sp("F032", "peak_offset", "Distance from mapped trace to the nearest gradient maximum", "C", "Which "
           "signals can plausibly localise faults within 300 m?", "|offset| of the profile's maximum |gradient| "
           "within +-5 km: fault stations vs matched controls, per signal", "A localising signal has its gradient "
           "peak on the trace; a corridor signal peaks km away", "Share within 300 m (cluster CI) vs controls; chance "
           "share for a uniform offset in +-5 km = 0.06", TEST_OUT,
           {SUPPORTED: "At least one signal places its gradient maximum on mapped traces more often than chance: a "
                       "trace-placement feature is justified.", NOT_SUPPORTED: "No signal localises better than "
                                                                              "controls: 100 m inputs give corridors, "
                                                                              "not traces.",
            INCONCLUSIVE: "Undecided."}, "Labels have their own positional error; offsets measured at 100 m.", 1)
    rows = []
    for sig, g in M.groupby("signal"):
        F, C = g[g.kind == "fault"], g[g.kind == "control"]
        a = cluster_ci((F.abs_grad_peak_offset_km <= 0.3).astype(float), F.system, E.n_boot, E.seed, np.nanmean)
        b = float((C.abs_grad_peak_offset_km <= 0.3).mean())
        e = cluster_effect(-F.abs_grad_peak_offset_km, F.system, -C.abs_grad_peak_offset_km, C.system, E.n_boot,
                           E.seed)
        rows.append({"signal": sig, "modality": E.D.modality(sig.split(":")[0]), "fault_within_300m": a["estimate"],
                     "fault_ci_lo": a["ci_lo"], "fault_ci_hi": a["ci_hi"], "control_within_300m": b,
                     "median_offset_fault_km": float(F.abs_grad_peak_offset_km.median()), **e})
    R = pd.DataFrame(rows).sort_values("fault_within_300m")
    p = V.plt()
    fig, ax = p.subplots(figsize=(7, max(3.5, 0.3 * len(R))))
    y = np.arange(len(R))
    ax.errorbar(R.fault_within_300m, y, xerr=[R.fault_within_300m - R.fault_ci_lo, R.fault_ci_hi -
                                              R.fault_within_300m], fmt="o", label="mapped faults")
    ax.plot(R.control_within_300m, y, "s", color=PALETTE["control"], label="matched controls")
    ax.axvline(0.06, color="r", lw=0.6, label="chance (uniform +-5 km)")
    ax.set_yticks(y)
    ax.set_yticklabels(R.signal, fontsize=6)
    ax.set(xlabel="share of stations with gradient maximum within 300 m", title="Peak-to-trace offset")
    ax.legend()
    best = R.iloc[R.effect.argmax()].to_dict()
    E.reg.save(s, fig, R, effect_result(best, E.min_effect, None, f"best localiser {best['signal']} (within 300 m "
                                                                  f"{best['fault_within_300m']:.2f} vs control "
                                                                  f"{best['control_within_300m']:.2f}):"))


def fwhm(off: np.ndarray, resp: np.ndarray) -> float:
    r = np.nan_to_num(np.abs(resp))
    if r.max() <= 0:
        return np.nan
    k = int(np.argmax(r))
    lo, hi = k, k
    while lo > 0 and r[lo - 1] >= 0.5 * r[k]:
        lo -= 1
    while hi < len(r) - 1 and r[hi + 1] >= 0.5 * r[k]:
        hi += 1
    return float(off[hi] - off[lo] + (off[1] - off[0]))


def width_of(E, sig: str, fmask: np.ndarray, cmask: np.ndarray, half: float = 10.0, detrend: bool = False,
             align: bool = True) -> float:
    S, M, cube, off, sigs, _ = E.stations_data()
    i = sigs.index(sig)
    w = np.abs(off) <= half

    def med(mask):
        c = cube[mask, i, :][:, w]
        if detrend:
            o = off[w]
            A = np.vstack([o, np.ones_like(o)]).T
            ok = np.isfinite(c)
            c = np.where(ok, c, 0.0)
            coef = np.linalg.lstsq(A, c.T, rcond=None)[0]
            c = c - (A @ coef).T
        if align:
            c = aligned(c[:, None, :], off[w])[:, 0, :]
        return np.nanmedian(c, 0)

    resp = np.abs(np.gradient(med(fmask) - med(cmask), off[w]))
    return fwhm(off[w], resp)


def f033(E):
    S, M, cube, off, sigs, _ = E.stations_data()
    fk, ck = (S.kind == "fault").to_numpy(), (S.kind == "control").to_numpy()
    s = sp("F033", "response_width", "Cross-fault response width and its sensitivity", "C", "Is the previously "
           "reported ~9.4 km response width robust?", "FWHM of the gradient of the median fault-minus-control profile "
           "under profile half-width, background-removal and alignment choices", "A width far above 300 m means the "
           "evidence marks corridors", "FWHM (km); cluster-bootstrap CI for the main setting",
           {DESCRIPTIVE: "Response widths per signal."},
           {DESCRIPTIVE: "Widths >> 0.6 km: use these signals as corridor context; trace placement needs another "
                         "source."}, "FWHM of a median profile; sensitive to profile length (reported).", 1)
    rows = []
    rng = np.random.default_rng(E.seed)
    systems = S.system.to_numpy()
    gs = np.unique(systems[fk])
    for sig in sigs:
        main = width_of(E, sig, fk, ck)
        draws = []
        for _ in range(min(E.n_boot, 100)):
            pick = set(rng.choice(gs, len(gs)))
            m = np.isin(systems, list(pick))
            draws.append(width_of(E, sig, fk & m, ck & m))
        lo, hi = np.nanquantile(draws, [0.025, 0.975]) if np.isfinite(draws).any() else (np.nan, np.nan)
        rows.append({"signal": sig, "fwhm_km": main, "ci_lo": lo, "ci_hi": hi,
                     "fwhm_half5km": width_of(E, sig, fk, ck, half=5.0),
                     "fwhm_detrended": width_of(E, sig, fk, ck, detrend=True),
                     "fwhm_unaligned": width_of(E, sig, fk, ck, align=False)})
    R = pd.DataFrame(rows)
    p = V.plt()
    fig, ax = p.subplots(figsize=(8, max(3.5, 0.3 * len(R))))
    y = np.arange(len(R))
    ax.errorbar(R.fwhm_km, y, xerr=[R.fwhm_km - R.ci_lo, R.ci_hi - R.fwhm_km], fmt="o", label="main (+-10 km)")
    for col, mk in (("fwhm_half5km", "x"), ("fwhm_detrended", "^"), ("fwhm_unaligned", "s")):
        ax.plot(R[col], y, mk, label=col, alpha=0.7)
    ax.axvline(0.6, color="r", lw=0.6, label="0.6 km")
    ax.axvline(9.4, color="0.5", lw=0.6, ls=":", label="historical 9.4 km (200 m EDA)")
    ax.set_yticks(y)
    ax.set_yticklabels(R.signal, fontsize=6)
    ax.set_xscale("log")
    ax.set(xlabel="FWHM (km)", title="Response width and sensitivity")
    ax.legend(fontsize=6)
    E.reg.save(s, fig, R, Result(DESCRIPTIVE, f"median FWHM {R.fwhm_km.median():.2f} km (range {R.fwhm_km.min():.2f}"
                                              f"-{R.fwhm_km.max():.2f}); narrowest {R.signal[R.fwhm_km.idxmin()]}",
                                 {}, len(R)))


def f034(E):
    simple_effect_figure(E, "F034", "along_strike_coherence", "Along-strike persistence of the 1 km gradient",
                         "along_persistence", "Does structural persistence add information beyond contrast?",
                         "Faults are continuous along strike over km; isolated noise is not", "mean / std of the 1 km "
                         "gradient along a 4 km tangent at the trace", "Persistence is a candidate feature.",
                         "Persistence does not separate faults at 100 m.", "Controls use the fault's tangent "
                         "direction (orientation-matched).")


def station_classes(E) -> pd.DataFrame:
    """Per fault station: background-calibrated |step| percentile per family and a signature class."""
    S, M, *_ = E.stations_data()
    fam = {"magnetic": "rtp", "gravity": "iso_grav_anom", "terrain": "det_elev"}
    out = S[S.kind == "fault"][["sample_id", "system", "r", "c", "x_utm", "y_utm", "region"]].copy()
    for f, sig in fam.items():
        g = M[M.signal == sig]
        if not len(g):
            continue
        ref = np.sort(g[g.kind == "control"].abs_step.dropna().to_numpy())
        v = g[g.kind == "fault"].set_index("sample_id").abs_step
        out[f"p_{f}"] = out.sample_id.map(v).map(lambda x: np.searchsorted(ref, x) / max(len(ref), 1)
                                                 if np.isfinite(x) else np.nan)
    P = out[[c for c in out if c.startswith("p_")]]
    out["class"] = np.select([(P >= 0.9).sum(axis=1) >= 2,                        # >= 2 families in the top 10 %
                              (P >= 0.9).any(axis=1) & (P <= 0.2).any(axis=1),    # one strong, another very weak
                              (P < 0.5).all(axis=1),                              # nothing above the control median
                              (P < 0.7).all(axis=1)],
                             ["strong_multimodal", "contradictory", "absent", "weak"], "single_family")
    return out


def f035(E):
    D = E.D
    C = station_classes(E)
    s = sp("F035", "signature_examples", "Strong, weak, contradictory and absent fault signatures", "C",
           "What heterogeneity do average curves hide?", "Fault stations classified by control-calibrated |step| "
           "percentiles of magnetic, gravity and terrain; one example window per class", "Averages mix faults with "
           "and without expression", "Class counts (by station and system)", {DESCRIPTIVE: "Class shares."},
           {DESCRIPTIVE: "A large 'absent' share: those faults are undetectable from these inputs (label noise or "
                         "missing evidence)."}, "Percentiles relative to matched controls; one station per example.",
           1)
    counts = C["class"].value_counts().to_dict()
    classes = [k for k in ("strong_multimodal", "single_family", "contradictory", "weak", "absent") if k in counts]
    fig, ax = V.grid(len(classes) * 3, 3, 3.6, 3.2)
    rng = np.random.default_rng(E.seed)
    i = 0
    for k in classes:
        x = C[C["class"] == k].iloc[int(rng.integers(0, (C["class"] == k).sum()))]
        sl = V.win(D, int(x.r), int(x.c), 60)
        for b in ("rtp", "iso_grav_anom", "det_elev"):
            if D.has(b):
                V.show(ax[i], D, D.band(b), f"{k}: {b}", sl=sl, band=b, faults=True, colorbar=False)
            i += 1
    sysc = C.groupby("class").system.nunique().to_dict()
    E.reg.save(s, fig, C, Result(DESCRIPTIVE, f"station classes {counts}; systems per class {sysc}",
                                 {"counts": counts}, len(C)))


def f036(E):
    T = effects(E)
    s = sp("F036", "signature_matrix", "Fault-signature summary matrix", "C", "Which measurements deserve further "
           "investigation?", "All (signal, metric) effects with status and regional stability", "One place to read "
           "effect, uncertainty and stability together", "Cliff's delta heatmap; markers: S supported, n not "
           "supported, ? inconclusive", {DESCRIPTIVE: "Matrix of evidence."},
           {DESCRIPTIVE: "Only SUPPORTED + STABLE cells become feature hypotheses."}, "Multiple comparisons: read "
                                                                                     "stability, not single cells.", 1)
    P = T.pivot_table(index="signal", columns="metric", values="effect")
    St = T.pivot_table(index="signal", columns="metric", values="status", aggfunc="first")
    p = V.plt()
    fig, ax = p.subplots(figsize=(1.2 * P.shape[1] + 4, 0.35 * P.shape[0] + 2))
    im = ax.imshow(P.to_numpy(), cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            st = St.iloc[i, j] if j < St.shape[1] else None
            ax.text(j, i, {"SUPPORTED": "S", "NOT_SUPPORTED": "n", "INCONCLUSIVE": "?"}.get(st, ""), ha="center",
                    va="center", fontsize=7)
    ax.set_xticks(range(P.shape[1]))
    ax.set_xticklabels(P.columns, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(P.shape[0]))
    ax.set_yticklabels(P.index, fontsize=7)
    fig.colorbar(im, ax=ax, label="Cliff's delta")
    n = T.groupby("status").size().to_dict()
    E.reg.save(s, fig, T, Result(DESCRIPTIVE, f"status counts {n}; stable supported "
                                              f"{int(((T.status == SUPPORTED) & (T.stability == 'STABLE')).sum())}",
                                 {"status_counts": n}, len(T)))


PART_C = [f023, f024, f025, f026, f027, f028, f029, f030, f031, f032, f033, f034, f035, f036]


# =============================================================================== PART D
def scale_data(E) -> pd.DataFrame:
    """Scale table (detectability, localisation, crest precision) per modality and scale, regional rows, station
    values, crest-distance per station, persistence counts and 1 km crests (saved for part E)."""
    if "scale" in E.D.cache:
        return E.D.cache["scale"]
    p = E.ws.path("tables/scale_localization.csv")
    if p.exists() and E.current_stage != "e06_scales":
        E.D.cache["scale"] = pd.read_csv(p)
        return E.D.cache["scale"]
    from scipy import ndimage

    D = E.D
    S, *_ = E.stations_data()
    rows, sv, reg, crests1, pers = [], [], [], {}, {}
    if "region_map" not in D.cache:
        D.cache["region_map"] = D.region(*np.indices(D.grid.shape)).astype(np.int8)
    regmap = D.cache["region_map"]
    within = D.net.distance_km <= 0.3
    for mod in LOCAL:
        b = PRIMARY[mod]
        if not D.has(b):
            continue
        pers[mod] = np.zeros(D.grid.shape, np.int8)
        for sc in E.c["scales_used_km"]:
            g = D.enh.thdr(D.band(b), sc)
            cr = crest_mask(D, g, sc)
            pers[mod] += cr.astype(np.int8)
            r = scale_row(D, S, mod, sc, g, cr, E.c)
            det, loc = r.pop("_det"), r.pop("_loc")
            dcr = ndimage.distance_transform_edt(~cr) * D.grid.pixel_km if cr.any() else np.full(D.grid.shape, 99.0)
            near = Sampler(dcr)(S.r.to_numpy(float), S.c.to_numpy(float))
            r["crest_recall_300m"] = float(np.nanmean(near[(S.kind == "fault").to_numpy()] <= 0.3))
            rows.append(r)
            sv.append(pd.DataFrame({"sample_id": S.sample_id, "modality": mod, "scale_km": sc, "det": det, "loc": loc,
                                    "crest_dist_km": near}))
            creg = regmap[cr]
            cwithin = within[cr]
            for k in np.unique(S.region):
                m = (S.region == k).to_numpy()
                f, c = m & (S.kind == "fault").to_numpy(), m & (S.kind == "control").to_numpy()
                if f.sum() >= int(E.c["statistics"]["min_region_stations"]):
                    rk = creg == k
                    reg.append({"modality": mod, "scale_km": sc, "region": int(k), "detect_delta":
                                cliffs_delta(det[f], det[c]), "loc_within_300m": float(np.nanmean(loc[f] <= 0.3)),
                                "crest_recall_300m": float(np.nanmean(near[f] <= 0.3)),
                                "crest_precision_300m": float(cwithin[rk].mean()) if rk.any() else np.nan,
                                "footprint_share_within_300m": float(within[D.valid & (regmap == k)].mean())})
            if abs(sc - 1.0) < 1e-9:
                crests1[mod] = cr
            del g
        if mod not in crests1:                           # part E needs 1 km crests whatever the mode's scale list
            crests1[mod] = crest_mask(D, D.enh.thdr(D.band(b), 1.0), 1.0)
    T = pd.DataFrame(rows)
    E.ws.save_table(T, "tables/scale_localization.csv")
    E.ws.save_table(pd.DataFrame(reg), "tables/scale_regional.csv")
    E.save_parquet(pd.concat(sv, ignore_index=True), "datasets/station_scale_values.parquet")
    E.save_npz("rasters/evidence_maps/crests_1km.npz", **{k: np.packbits(v) for k, v in crests1.items()},
               shape=np.asarray(D.grid.shape))
    E.save_npz("rasters/scale_space/crest_persistence.npz", **pers)
    E.D.cache.update(scale=T, crests1=crests1, persistence=pers)
    return T


def crests1(E) -> dict:
    if "crests1" not in E.D.cache:
        with np.load(E.ws.path("rasters/evidence_maps/crests_1km.npz")) as z:
            sh = tuple(int(x) for x in z["shape"])
            E.D.cache["crests1"] = {k: np.unpackbits(z[k])[:np.prod(sh)].reshape(sh).astype(bool) for k in z.files
                                    if k != "shape"}
    return E.D.cache["crests1"]


def f037(E):
    D = E.D
    d, r, c = domain_windows(E, 1, 250)[0]
    sl = V.win(D, r, c, 250)
    sc = E.c["scales_used_km"]
    mods = [m for m in ("magnetic", "gravity") if D.has(PRIMARY[m])]
    s = sp("F037", "scale_space", "Scale-space of horizontal gradients", "D", "When do small structures disappear and "
           "broad patterns dominate?", "|grad| at every analysis scale, one window, magnetic and gravity",
           "Fault-scale features must survive at the scales used for detection", "Visual + pixel share of the window "
           "above the window's 95th percentile at each scale", {DESCRIPTIVE: "Scale-space panels."},
           {DESCRIPTIVE: "Structure visible only at >= 5 km: corridor evidence."}, "One window.", 2)
    fig, ax = V.grid(len(mods) * len(sc), len(sc), 2.8, 2.8)
    i = 0
    for m in mods:
        for x in sc:
            g = D.enh.thdr(D.band(PRIMARY[m])[sl], x)              # window-only: edge effects within ~2 sigma
            ax[i].imshow(g, extent=V.extent_km(D, sl), cmap="magma", vmin=V.stretch(g)[0], vmax=V.stretch(g)[1])
            ax[i].set_title(f"{m} {x:g} km", fontsize=7)
            ax[i].set_aspect("equal")
            ax[i].tick_params(labelsize=5)
            i += 1
    E.reg.save(s, fig, pd.DataFrame({"modality": np.repeat(mods, len(sc)), "scale_km": np.tile(sc, len(mods)),
                                     "window_row": r, "window_col": c}),
               Result(DESCRIPTIVE, f"window around domain {d} centre ({r}, {c})", {}, len(sc) * len(mods)))


def f038(E):
    T = scale_data(E)
    s = sp("F038", "detectability_vs_scale", "Fault vs control separation across scales", "D", "At which scale "
           "does each measurement discriminate best?", "Cliff's delta of the max |grad| within +-300 m of the trace vs "
           "controls, per scale", "Corridor-scale signals can separate faults without localising them", "delta with "
           "cluster CI per (modality, scale)", TEST_OUT,
           {SUPPORTED: "Best detection scale identified per modality.", NOT_SUPPORTED: "No scale separates faults.",
            INCONCLUSIVE: "Undecided."}, "Controls matched at 2-10 km.", 1)
    p = V.plt()
    fig, ax = p.subplots(figsize=(7, 4.2))
    for m, g in T.groupby("modality"):
        ax.errorbar(g.scale_km, g.detect_delta, yerr=[g.detect_delta - g.detect_lo, g.detect_hi - g.detect_delta],
                    fmt="o-", label=m, color=PALETTE.get(m), capsize=2)
    ax.set_xscale("log")
    ax.axhspan(-E.min_effect, E.min_effect, color="0.9")
    ax.set(xlabel="scale (km)", ylabel="Cliff's delta", title="Detectability vs scale")
    ax.legend()
    b = T.loc[T.detect_delta.idxmax()]
    E.reg.save(s, fig, T, effect_result({"effect": b.detect_delta, "ci_lo": b.detect_lo, "ci_hi": b.detect_hi,
                                         "n_groups": b.n_systems}, E.min_effect, None,
                                        f"best {b.modality} at {b.scale_km:g} km:"))


def f039(E):
    T = scale_data(E)
    SV = E.table("datasets/station_scale_values.parquet")
    S, *_ = E.stations_data()
    s = sp("F039", "localization_vs_scale", "Localisation error of the gradient maximum vs scale", "D", "Is the most "
           "detectable scale also the best for localisation?", "|offset| of the max |grad| within +-5 km of fault "
           "stations, per scale", "Smoothing moves and widens maxima", "Median |offset| (cluster CI) and share within "
           "300 m (chance 0.06)", TEST_OUT,
           {SUPPORTED: "Some modality / scale localises within 300 m above chance: a trace-placement scale exists.",
            NOT_SUPPORTED: "No modality / scale localises above chance.", INCONCLUSIVE: "Undecided."},
           "Labels have positional error; offsets quantised at 100 m.", 1)
    p = V.plt()
    fig, ax = p.subplots(1, 2, figsize=(12, 4.2))
    for m, g in T.groupby("modality"):
        ax[0].errorbar(g.scale_km, g.loc_median_km, yerr=[g.loc_median_km - g.loc_lo, g.loc_hi - g.loc_median_km],
                       fmt="o-", label=m, color=PALETTE.get(m), capsize=2)
        ax[1].plot(g.scale_km, g.loc_within_300m, "o-", color=PALETTE.get(m), label=m)
    for a_ in ax:
        a_.set_xscale("log")
        a_.set_xlabel("scale (km)")
    ax[0].axhline(0.3, color="r", lw=0.6)
    ax[1].axhline(0.06, color="r", lw=0.6, label="chance")
    ax[0].set(ylabel="median |offset| (km)", title="localisation error")
    ax[1].set(ylabel="share within 300 m")
    ax[1].legend()
    F = SV.merge(S[["sample_id", "kind", "system"]], on="sample_id")
    F = F[F.kind == "fault"]
    best = T.loc[T.loc_within_300m.idxmax()]
    g = F[(F.modality == best.modality) & (F.scale_km == best.scale_km)]
    ci = cluster_ci((g["loc"] <= 0.3).astype(float), g.system, E.n_boot, E.seed, np.nanmean)
    eff = {"effect": ci["estimate"] - 0.06, "ci_lo": ci["ci_lo"] - 0.06, "ci_hi": ci["ci_hi"] - 0.06,
           "n_groups": ci["n_groups"], "n_fault": ci["n"]}
    res = effect_result(eff, 0.05, None, f"best {best.modality} {best.scale_km:g} km share within 300 m minus "
                                         "chance:")
    E.reg.save(s, fig, T, res)


def f040(E):
    R = E.table("tables/scale_regional.csv")
    T = scale_data(E)
    s = sp("F040", "scale_precision_recall", "Crest recall / precision by scale (leave-region-out selection)", "D",
           "Which scales produce too many unrelated lineaments?", "Recall = fault stations with a crest within 300 m; "
           "precision proxy = crest pixels within 300 m of mapped traces; scale chosen on the other regions",
           "A scale that recalls everything by marking everything is useless", "F = 2PR / (P + R) of the "
           "selected scale on each held-out region", {DESCRIPTIVE: "Held-out scale performance."},
           {DESCRIPTIVE: "Use the region-stable scale for each modality; precision near the footprint baseline "
                         "means no trace information."}, "Precision counts unmapped true faults as errors (PU).", 2)
    if not len(R):
        return E.reg.skip(s, "no regional scale rows")
    R = R.copy()
    R["F"] = 2 * R.crest_precision_300m * R.crest_recall_300m / (R.crest_precision_300m + R.crest_recall_300m + 1e-9)
    rows = []
    for m, g in R.groupby("modality"):
        for k in g.region.unique():
            dev = g[g.region != k].groupby("scale_km").F.mean()
            if not len(dev):
                continue
            sc = float(dev.idxmax())
            h = g[(g.region == k) & (g.scale_km == sc)]
            rows.append({"modality": m, "held_out_region": int(k), "selected_scale_km": sc,
                         "heldout_F": float(h.F.iloc[0]) if len(h) else np.nan})
    H = pd.DataFrame(rows)
    p = V.plt()
    fig, ax = p.subplots(1, 2, figsize=(12, 4.2))
    for m, g in T.groupby("modality"):
        ax[0].plot(g.crest_recall_300m, g.crest_share_within_300m, "o-", color=PALETTE.get(m), label=m)
        for x in g.itertuples():
            ax[0].annotate(f"{x.scale_km:g}", (x.crest_recall_300m, x.crest_share_within_300m), fontsize=6)
    ax[0].axhline(T.footprint_share_within_300m.iloc[0], color="r", lw=0.6, label="footprint baseline")
    ax[0].set(xlabel="crest recall (300 m)", ylabel="crest precision proxy", title="all regions")
    ax[0].legend()
    if len(H):
        for m, g in H.groupby("modality"):
            ax[1].plot(g.held_out_region, g.heldout_F, "o", color=PALETTE.get(m), label=m)
    ax[1].set(xlabel="held-out region", ylabel="F of the scale selected elsewhere", title="leave-region-out")
    E.reg.save(s, fig, H, Result(DESCRIPTIVE, "held-out F by modality: " + ", ".join(
        f"{m} {g.heldout_F.mean():.3f}" for m, g in H.groupby("modality")) if len(H) else "none", {}, len(H)))


def width_groups(E, groups: dict, sig: str) -> pd.DataFrame:
    S, *_ = E.stations_data()
    fk, ck = (S.kind == "fault").to_numpy(), (S.kind == "control").to_numpy()
    rows = []
    for name, sysmask in groups.items():
        m = np.isin(S.system.to_numpy(), list(sysmask))
        if (fk & m).sum() >= 30:
            rows.append({"group": name, "fwhm_km": width_of(E, sig, fk & m, ck & m), "n_fault": int((fk & m).sum())})
    return pd.DataFrame(rows)


def f041(E):
    S, *_ = E.stations_data()
    F = S[S.kind == "fault"]
    sys_len = F.groupby("system").seg_length_km.max()
    groups = {f"{a}-{b} km": set(sys_len[(sys_len >= a) & (sys_len < b)].index) for a, b in
              ((0, 2), (2, 5), (5, 15), (15, 1e9))}
    s = sp("F041", "width_vs_length", "Response width vs fault length", "D", "Do short and long faults need different "
           "spatial contexts?", "FWHM per fault-length class, magnetic and gravity 1 km gradients", "Long master "
           "faults may carry wider signatures", "FWHM of the median fault-minus-control gradient profile",
           {DESCRIPTIVE: "Width by length class."}, {DESCRIPTIVE: "Length-dependent widths: multi-context inputs."},
           "Segment length of the station's branch (max per system).", 2)
    rows = []
    for sig in ("rtp:thdr1km", "iso_grav_anom:thdr1km", "det_elev_slope"):
        if sig in E.stations_data()[4]:
            rows.append(width_groups(E, groups, sig).assign(signal=sig))
    R = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    p = V.plt()
    fig, ax = p.subplots(figsize=(7, 4))
    for sig, g in R.groupby("signal") if len(R) else []:
        ax.plot(g.group, g.fwhm_km, "o-", label=sig)
    ax.set(ylabel="FWHM (km)", title="Response width by fault length")
    ax.legend()
    E.reg.save(s, fig, R, Result(DESCRIPTIVE, "; ".join(f"{r.signal} {r.group}: {r.fwhm_km:.2f}" for r in
                                                       R.itertuples()) if len(R) else "insufficient", {}, len(R)))


def f042(E):
    S, *_ = E.stations_data()
    F = S[S.kind == "fault"]
    groups = {f"domain {d}": set(F[F.domain == d].system) for d in sorted(F.domain.unique()) if d >= 0}
    s = sp("F042", "width_vs_domain", "Response width vs geological domain", "D", "Is localisation uncertainty "
           "region dependent?", "FWHM per label-free domain", "Burial depth and lithology change response widths",
           "FWHM per domain", {DESCRIPTIVE: "Width by domain."},
           {DESCRIPTIVE: "Domain-dependent widths: domain-aware scales."}, "Domains with < 30 stations omitted.", 2)
    rows = [width_groups(E, groups, sig).assign(signal=sig) for sig in ("rtp:thdr1km", "iso_grav_anom:thdr1km")
            if sig in E.stations_data()[4]]
    R = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    p = V.plt()
    fig, ax = p.subplots(figsize=(7, 4))
    for sig, g in R.groupby("signal") if len(R) else []:
        ax.plot(g.group, g.fwhm_km, "o-", label=sig)
    ax.set(ylabel="FWHM (km)", title="Response width by domain")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    E.reg.save(s, fig, R, Result(DESCRIPTIVE, f"width range {R.fwhm_km.min():.2f}-{R.fwhm_km.max():.2f} km" if len(R)
                                 else "insufficient", {}, len(R)))


def f043(E):
    D = E.D
    scale_data(E)
    P = E.D.cache.get("persistence")
    if P is None:
        with np.load(E.ws.path("rasters/scale_space/crest_persistence.npz")) as z:
            P = {k: z[k] for k in z.files}
    S, *_ = E.stations_data()
    s = sp("F043", "multiscale_persistence", "Crests persistent across scales vs transient edges", "D", "Does "
           "persistence distinguish structure from noise / artifacts?", "Number of scales at which a pixel is a crest "
           "(within 300 m of the station), faults vs controls", "Real structures persist over a range of filter "
           "scales; noise and survey seams do not", "Cliff's delta of the persistence count", TEST_OUT,
           {SUPPORTED: "Persistence is a candidate feature.", NOT_SUPPORTED: "Persistence does not separate faults.",
            INCONCLUSIVE: "Undecided."}, "Crest = top 3 % line strength per scale.", 2)
    rows = []
    fig, ax = V.grid(len(P) + 1, 3, 4, 3.4)
    fk, ck = (S.kind == "fault").to_numpy(), (S.kind == "control").to_numpy()
    for i, (m, cnt) in enumerate(P.items()):
        from scipy import ndimage

        mx = ndimage.maximum_filter(cnt, size=7)
        v = Sampler(mx.astype(np.float32), order=0)(S.r.to_numpy(float), S.c.to_numpy(float))
        e = cluster_effect(v[fk], S.system[fk], v[ck], S.system[ck], E.n_boot, E.seed)
        rows.append({"modality": m, **e})
        d, r, c = domain_windows(E, 1, 250)[0]
        V.show(ax[i], D, cnt.astype(float), f"{m}: crest persistence (delta {e['effect']:+.2f})",
               sl=V.win(D, r, c, 250), cmap="viridis", faults=True, vmin=0, vmax=len(E.c["scales_used_km"]))
    R = pd.DataFrame(rows)
    best = R.iloc[R.effect.argmax()].to_dict() if len(R) else None
    E.reg.save(s, fig, R, effect_result(best, E.min_effect, None, f"best {best['modality']}:") if best else
               Result(INCONCLUSIVE, "no modality"))


def f044(E):
    from scipy.stats import spearmanr

    D = E.D
    SV = E.table("datasets/station_scale_values.parquet")
    S, *_ = E.stations_data()
    s = sp("F044", "localization_uncertainty", "Label-free localisation sharpness vs localisation error", "D",
           "Should fault-location uncertainty be modelled explicitly?", "Sharpness = 0.5 km / 5 km gradient ratio at "
           "the station vs |offset| of the 1 km gradient maximum", "Sharp gradients localise; diffuse ones do not",
           "Spearman rho (cluster bootstrap) at fault stations", TEST_OUT,
           {SUPPORTED: "A label-free sharpness map predicts localisation error: model location uncertainty "
                       "(e.g. width-aware rasterisation).", NOT_SUPPORTED: "Sharpness does not predict error.",
            INCONCLUSIVE: "Undecided."}, "Requires the 0.5 and 5 km scales (FULL_EDA).", 2)
    rows = []
    fig, ax = V.grid(2, 2, 5, 4)
    for i, (m, b) in enumerate((("magnetic", "rtp"), ("gravity", "iso_grav_anom"))):
        if not D.has(b):
            continue
        sh = D.enh.thdr(D.band(b), 0.5) / (D.enh.thdr(D.band(b), 5.0) + 1e-12)
        v = Sampler(sh)(S.r.to_numpy(float), S.c.to_numpy(float))
        loc = SV[(SV.modality == m) & (np.isclose(SV.scale_km, 1.0))].set_index("sample_id")["loc"]
        F = S[S.kind == "fault"].assign(sharp=v[(S.kind == "fault").to_numpy()])
        F["loc"] = F.sample_id.map(loc)
        ok = F[["sharp", "loc"]].notna().all(axis=1)
        rho = spearmanr(F.sharp[ok], F["loc"][ok]).correlation if ok.sum() > 30 else np.nan
        rng = np.random.default_rng(E.seed)
        gs = F.system[ok].unique()
        bs = []
        for _ in range(min(E.n_boot, 300)):
            pick = np.isin(F.system[ok], rng.choice(gs, len(gs)))
            bs.append(spearmanr(F.sharp[ok][pick], F["loc"][ok][pick]).correlation)
        lo, hi = np.nanquantile(bs, [0.025, 0.975]) if bs else (np.nan, np.nan)
        rows.append({"modality": m, "effect": -rho, "ci_lo": -hi, "ci_hi": -lo, "n_groups": len(gs),
                     "n_fault": int(ok.sum()), "spearman_sharp_vs_error": rho})
        ax[i].hexbin(np.log10(F.sharp[ok] + 1e-6), F["loc"][ok], gridsize=30, bins="log", cmap="Greys")
        ax[i].set(title=f"{m}: rho {rho:.2f} [{lo:.2f}, {hi:.2f}]", xlabel="log10 sharpness",
                  ylabel="|offset| of 1 km max (km)")
    R = pd.DataFrame(rows)
    best = R.iloc[R.effect.argmax()].to_dict() if len(R) else None
    E.reg.save(s, fig, R, effect_result(best, E.min_effect, None, f"{best['modality']} (sign: sharper -> smaller "
                                                                  "error):") if best else Result(INCONCLUSIVE, "none"))


def f045(E):
    T = scale_data(E)
    s = sp("F045", "detection_localization_tradeoff", "Detection vs localisation vs false-positive proxy", "D",
           "Which measurements find corridors and which place traces?", "Per (modality, scale): detect delta vs share "
           "within 300 m, colour = crest enrichment", "Separates corridor finders from trace placers",
           "Best detection scale and best localisation scale per modality", {DESCRIPTIVE: "Trade-off map."},
           {DESCRIPTIVE: "Different best scales: two-stage design (corridor then trace placement)."},
           "Enrichment counts unmapped faults as false positives.", 1)
    p = V.plt()
    fig, ax = p.subplots(figsize=(7, 5))
    sc_ = ax.scatter(T.detect_delta, T.loc_within_300m, c=T.crest_enrichment, cmap="viridis", s=40)
    for x in T.itertuples():
        ax.annotate(f"{x.modality[:3]} {x.scale_km:g}", (x.detect_delta, x.loc_within_300m), fontsize=6)
    fig.colorbar(sc_, ax=ax, label="crest enrichment within 300 m")
    ax.set(xlabel="detect delta", ylabel="share localised within 300 m", title="Detection vs localisation")
    rows = []
    for m, g in T.groupby("modality"):
        rows.append({"modality": m, "best_detect_scale_km": float(g.loc[g.detect_delta.idxmax(), "scale_km"]),
                     "best_localise_scale_km": float(g.loc[g.loc_within_300m.idxmax(), "scale_km"])})
    R = pd.DataFrame(rows)
    E.reg.save(s, fig, T.merge(R, on="modality"), Result(DESCRIPTIVE, "; ".join(
        f"{r.modality}: detect {r.best_detect_scale_km:g} km, localise {r.best_localise_scale_km:g} km"
        for r in R.itertuples()), {}, len(T)))


def f046(E):
    from gems_eda.enhance import Enhancer
    from gems_eda.io import Grid

    D = E.D
    S, *_ = E.stations_data()
    g0 = D.grid
    g2 = Grid(g0.transform * g0.transform.scale(2, 2), g0.crs, g0.height // 2, g0.width // 2, 2 * g0.pixel_km_x,
              2 * g0.pixel_km_y, g0.is_geographic)
    e2 = Enhancer(g2)
    s = sp("F046", "resolution_sensitivity", "Native 100 m vs block-averaged 200 m", "D", "What did the 200 m EDA "
           "lose?", "Detectability and localisation at 1, 2, 5 km scales computed at 100 m and 200 m at the same "
           "stations", "The historical EDA ran at 200 m", "Delta and share within 300 m at both resolutions",
           {DESCRIPTIVE: "Resolution effect."}, {DESCRIPTIVE: "A loss at 200 m: keep native resolution for trace "
                                                              "features."}, "Block averaging only (no anti-alias).", 2)
    off_n, off_w = np.round(np.arange(-0.3, 0.31, 0.1), 2), np.round(np.arange(-5, 5.01, 0.1), 2)
    S2 = S.assign(r=S.r / 2.0, c=S.c / 2.0, dr=S.dr / 2.0, dc=S.dc / 2.0)
    fk, ck = (S.kind == "fault").to_numpy(), (S.kind == "control").to_numpy()
    rows = []
    for m in ("magnetic", "gravity", "terrain"):
        b = PRIMARY[m]
        if not D.has(b):
            continue
        a = D.band(b)
        H, W = (a.shape[0] // 2) * 2, (a.shape[1] // 2) * 2
        a2 = np.nanmean(a[:H, :W].reshape(H // 2, 2, W // 2, 2), axis=(1, 3))
        for sc in (1.0, 2.0, 5.0):
            for res, st, g in (("100m", S, D.enh.thdr(a, sc)), ("200m", S2, e2.thdr(a2, sc))):
                near, wide = along_normal(g, st, off_n), along_normal(g, st, off_w)
                det = np.nanmax(near, 1)
                loc = np.abs(off_w[np.nanargmax(np.where(np.isfinite(wide), wide, -np.inf), 1)])
                rows.append({"modality": m, "scale_km": sc, "resolution": res, "detect_delta": cliffs_delta(
                    det[fk], det[ck]), "loc_within_300m": float(np.nanmean(loc[fk] <= 0.3))})
    R = pd.DataFrame(rows)
    p = V.plt()
    fig, ax = p.subplots(1, 2, figsize=(12, 4))
    for (m, res), g in R.groupby(["modality", "resolution"]):
        ls = "-" if res == "100m" else "--"
        ax[0].plot(g.scale_km, g.detect_delta, ls, marker="o", color=PALETTE.get(m), label=f"{m} {res}")
        ax[1].plot(g.scale_km, g.loc_within_300m, ls, marker="o", color=PALETTE.get(m), label=f"{m} {res}")
    ax[0].set(xlabel="scale (km)", ylabel="detect delta")
    ax[1].set(xlabel="scale (km)", ylabel="share within 300 m")
    ax[1].legend(fontsize=6)
    P = R.pivot_table(index=["modality", "scale_km"], columns="resolution", values="loc_within_300m")
    loss = (P["100m"] - P["200m"]).mean() if {"100m", "200m"} <= set(P.columns) else np.nan
    E.reg.save(s, fig, R, Result(DESCRIPTIVE, f"mean loss of share within 300 m at 200 m: {loss:+.3f}", {}, len(R)))


def f047(E):
    T = scale_data(E)
    sup = E.table("figures/data/F009_effective_support.csv")
    s = sp("F047", "scale_recommendation", "Spatial-scale recommendation matrix", "D", "Which scales support which "
           "task per modality?", "Per modality: corridor scale, trace scale, effective support, localisation above "
           "chance", "Not a winner-takes-all ranking: tasks differ", "Rules: TRACE if share within 300 m >= 0.15 at "
           "some scale, CORRIDOR if detection supported but no trace scale, WEAK otherwise",
           {DESCRIPTIVE: "Recommendation per modality."},
           {DESCRIPTIVE: "TRACE modalities may place traces; CORRIDOR ones are context."},
           "Thresholds predeclared here; see F039 for the tested trace-placement claim.", 1)
    rows = []
    for m, g in T.groupby("modality"):
        det_ok = (g.detect_lo > E.min_effect).any()
        tr = g[g.loc_within_300m >= 0.15]
        rows.append({"modality": m, "corridor_scale_km": float(g.loc[g.detect_delta.idxmax(), "scale_km"]),
                     "trace_scale_km": float(tr.loc[tr.loc_within_300m.idxmax(), "scale_km"]) if len(tr) else np.nan,
                     "max_share_within_300m": float(g.loc_within_300m.max()),
                     "recommendation": "TRACE" if len(tr) else "CORRIDOR" if det_ok else "WEAK"})
    R = pd.DataFrame(rows)
    if len(sup):
        R = R.merge(sup.rename(columns={sup.columns[-1]: "effective_support_km"}), on="modality", how="left")
    p = V.plt()
    fig, ax = p.subplots(figsize=(8, 0.5 * len(R) + 1.5))
    ax.axis("off")
    tb = ax.table(cellText=R.round(3).astype(str).values, colLabels=R.columns, loc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(7)
    E.reg.save(s, fig, R, Result(DESCRIPTIVE, "; ".join(f"{r.modality}: {r.recommendation}" for r in R.itertuples()),
                                 {}, len(R)))


PART_D = [f037, f038, f039, f040, f041, f042, f043, f044, f045, f046, f047]
