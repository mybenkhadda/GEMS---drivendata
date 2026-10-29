"""Part A (F001-F012): data and geological landscape. Part B (F013-F022): mapped-fault geometry and networks.
Every function computes its statistics first, then saves the figure with an interpretation built from them."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from . import viz as V
from .core import DESCRIPTIVE, INCONCLUSIVE, NOT_SUPPORTED, PALETTE, SUPPORTED, Result, Spec, cluster_effect, \
    effect_result
from .data import LOCAL, PRIMARY


def sp(fid, name, title, part, q, comp, why, meas, outcomes, impl, lim, pr=2) -> Spec:
    return Spec(fid, name, title, part, q, comp, why, meas, outcomes, impl, lim, pr)


TEST_OUT = {SUPPORTED: "The difference is supported (CI outside the equivalence band).",
            NOT_SUPPORTED: "No meaningful difference: the CI lies inside the equivalence band.",
            INCONCLUSIVE: "Inconclusive: the CI overlaps the equivalence band boundary."}


def bands_meta(E) -> pd.DataFrame:
    return E.table("audit/band_metadata.csv")


def units_of(E, band: str) -> str:
    B = bands_meta(E)
    r = B[B.band == band] if len(B) else B
    return str(r.units.iloc[0]) if len(r) else ""


def domain_windows(E, n: int = 4, half_px: int = 300) -> list[tuple[int, int, int]]:
    """(domain, row, col): the centre of the largest connected block area of the n largest domains."""
    dom = E.domains()
    out = []
    ids, cnt = np.unique(dom[dom >= 0], return_counts=True)
    for d in ids[np.argsort(-cnt)][:n]:
        lab, k = ndimage.label(dom == d)
        if not k:
            continue
        sizes = ndimage.sum(np.ones_like(lab), lab, np.arange(1, k + 1))
        big = int(np.argmax(sizes)) + 1
        r, c = ndimage.center_of_mass(lab == big)
        H, W = dom.shape
        out.append((int(d), int(np.clip(r, half_px, H - half_px)), int(np.clip(c, half_px, W - half_px))))
    return out


# =============================================================================== PART A
def f001(E):
    D = E.D
    names = list(D.stack.names)
    for fid, faults in (("F001", False), ("F001b", True)):
        s = sp(fid, "band_atlas" + ("_faults" if faults else ""), "Complete 19-band atlas" + (
            " with mapped faults" if faults else " (no labels)"), "A", "Which observations carry local structure, "
            "broad trends or artifacts?", "All bands, grouped by modality, identical extent", "A common footprint "
            "view shows which surveys resolve km-scale structure", "Per-band 2-98 % stretch (symmetric for signed "
            "anomalies), physical units from band tags", {DESCRIPTIVE: "Atlas of every band."},
            {DESCRIPTIVE: "Bands without local structure in this view are context, not trace evidence."},
            "Visual only; stretch differs per band (recorded).", 2)
        fig, ax = V.grid(len(names), 5, 3.6, 3.4)
        st = {}
        for a_, n in zip(ax, names):
            st[n] = V.show(a_, D, D.band(n), f"{n} [{D.modality(n)}]", band=n, units=units_of(E, n), step=3,
                           faults=faults)
        miss = [] if len(names) == 19 else [f"{len(names)} bands (recorded 19)"]
        E.reg.save(s, fig, V.map_data(fid, {n: D.band(n)[::3, ::3] for n in names}, D, {"display_step": 3}),
                   Result(DESCRIPTIVE, f"{len(names)} bands shown; {miss or 'band count matches the record'}",
                          {"stretch": st}, len(names)))


def f002(E):
    D = E.D
    wins = domain_windows(E, 3, 300)
    mods = [m for m in ("magnetic", "gravity", "terrain", "basement") if D.has(PRIMARY[m])]
    s = sp("F002", "modality_comparison", "Same windows, four independent measurements", "A",
           "Do the same structures appear in independent measurements?", "Magnetic, gravity, detrended elevation, "
           "basement depth over identical 60 km windows", "Independent physics seeing the same edge argues for a "
           "real geological boundary", "Mean pairwise Spearman correlation of 1 km gradient maps within each window",
           {DESCRIPTIVE: "Cross-modal coincidence of gradients by window."},
           {DESCRIPTIVE: "Low coincidence = modalities are complementary (do not average them blindly)."},
           "Windows chosen from label-free domains; correlation is not causation.", 2)
    fig, ax = V.grid(len(wins) * len(mods), len(mods), 3.6, 3.4)
    rows, i = [], 0
    from scipy.stats import spearmanr

    for d, r, c in wins:
        sl = V.win(D, r, c, 300)
        grads = {}
        for m in mods:
            b = PRIMARY[m]
            V.show(ax[i], D, D.band(b), f"domain {d}: {m} ({b})", sl=sl, band=b, units=units_of(E, b))
            grads[m] = D.enh.thdr(D.band(b)[sl], 1.0)
            i += 1
        for a in mods:
            for b in mods:
                if a < b:
                    x, y = grads[a].ravel(), grads[b].ravel()
                    ok = np.isfinite(x) & np.isfinite(y)
                    rho = spearmanr(x[ok][::7], y[ok][::7]).correlation if ok.sum() > 50 else np.nan
                    rows.append({"domain": d, "row": r, "col": c, "a": a, "b": b, "spearman_grad": rho})
    R = pd.DataFrame(rows)
    E.reg.save(s, fig, R, Result(DESCRIPTIVE, f"median pairwise gradient Spearman {R.spearman_grad.median():.2f} "
                                              f"(range {R.spearman_grad.min():.2f}..{R.spearman_grad.max():.2f})",
                                 {"median_rho": float(R.spearman_grad.median())}, len(R)))


def f003(E):
    D = E.D
    dom = E.domains()
    B = E.table("datasets/physical_regions.parquet")
    s = sp("F003", "structural_domains", "Label-free structural domains", "A", "Does one global strategy face very "
           "different geological environments?", "k-means domains of 5 km block descriptors (no fault label)",
           "Different magnetic texture / gravity / relief regimes imply different fault expressions",
           "Domain map + standardised descriptor medians per domain", {DESCRIPTIVE: "Domains and their signatures."},
           {DESCRIPTIVE: "Strong descriptor contrasts between domains: test regional (domain-aware) features."},
           "k = 6 is a modelling choice; domains are statistical, not mapped geology.", 1)
    p = V.plt()
    fig, ax = p.subplots(1, 2, figsize=(13, 5.5))
    V.show(ax[0], D, np.where(dom >= 0, dom, np.nan).astype(float), "domains (label-free)", cmap="tab10", step=2,
           vmin=-0.5, vmax=9.5)
    cols = [c for c in B.columns if c not in ("bi", "bj", "domain", "x_center", "y_center")]
    Z = (B[cols] - B[cols].median()) / (B[cols].quantile(0.75) - B[cols].quantile(0.25)).replace(0, 1)
    M = Z.assign(domain=B.domain).groupby("domain").median()
    im = ax[1].imshow(M.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax[1].set_xticks(range(len(cols)))
    ax[1].set_xticklabels(cols, rotation=60, ha="right", fontsize=6)
    ax[1].set_yticks(range(len(M)))
    ax[1].set_yticklabels([f"domain {d} (n={int((B.domain == d).sum())})" for d in M.index])
    fig.colorbar(im, ax=ax[1], fraction=0.04)
    ax[1].set_title("standardised descriptor medians")
    sizes = B.domain.value_counts().to_dict()
    E.reg.save(s, fig, M.reset_index(), Result(DESCRIPTIVE, f"{len(M)} domains; blocks per domain {sizes}",
                                               {"blocks_per_domain": sizes}, len(B)))


def f004(E):
    D = E.D
    s = sp("F004", "signal_variability", "Local vs regional variability by modality", "A", "Which measurements "
           "contain localised structure vs only broad context?", "Std in 1 km vs 10 km moving windows per band",
           "A fault-scale signal needs variance at km scale, not only at tens of km", "Ratio std(1 km) / std(10 km) "
           "on a 5 % pixel sample", {DESCRIPTIVE: "Short-scale variance share per band."},
           {DESCRIPTIVE: "Bands with a ratio near 0 carry no trace-scale information at 100 m."},
           "Moving-window std mixes noise and signal; compare with F007 spectra.", 2)
    rows = []
    rng = np.random.default_rng(E.seed)
    rr, cc = np.nonzero(D.valid)
    pick = rng.choice(len(rr), min(200_000, len(rr)), replace=False)
    for n in D.stack.names:
        a = np.nan_to_num(D.band(n), nan=float(np.nanmedian(D.band(n))))

        def lstd(w):
            m1 = ndimage.uniform_filter(a, w)
            return np.sqrt(np.maximum(ndimage.uniform_filter(a * a, w) - m1 * m1, 0))
        r = lstd(11)[rr[pick], cc[pick]] / (lstd(101)[rr[pick], cc[pick]] + 1e-12)
        rows.append(pd.DataFrame({"band": n, "modality": D.modality(n), "ratio": r}))
    R = pd.concat(rows, ignore_index=True)
    order = R.groupby("band").ratio.median().sort_values().index
    p = V.plt()
    fig, ax = p.subplots(figsize=(12, 4.5))
    ax.boxplot([R[R.band == b].ratio.clip(0, 2) for b in order], labels=order, showfliers=False)
    ax.tick_params(axis="x", rotation=70)
    ax.set(ylabel="std(1 km) / std(10 km)", title="Short-scale variability share (5 % stratified pixel sample)")
    med = R.groupby("band").ratio.median()
    E.reg.save(s, fig, med.reset_index(), Result(DESCRIPTIVE, f"highest short-scale share: "
                                                              f"{med.sort_values().index[-1]} {med.max():.2f}; lowest "
                                                              f"{med.sort_values().index[0]} {med.min():.2f}",
                                                 {"median_ratio": med.to_dict()}, len(pick)))


def autocorr(a: np.ndarray, valid: np.ndarray, lags_px, direction, n=150_000, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rr, cc = np.nonzero(valid)
    H, W = a.shape
    i = rng.choice(len(rr), min(n, len(rr)), replace=False)
    r0, c0 = rr[i], cc[i]
    out = []
    for L in lags_px:
        r1, c1 = r0 + int(round(L * direction[0])), c0 + int(round(L * direction[1]))
        ok = (r1 >= 0) & (r1 < H) & (c1 >= 0) & (c1 < W)
        x, y = a[r0[ok], c0[ok]], a[r1[ok], c1[ok]]
        m = np.isfinite(x) & np.isfinite(y)
        out.append(np.corrcoef(x[m], y[m])[0, 1] if m.sum() > 100 else np.nan)
    return np.asarray(out)


def corr_length(lags_km, r) -> float:
    below = np.flatnonzero(np.asarray(r) < 1 / np.e)
    return float(lags_km[below[0]]) if len(below) else np.nan


def f005(E):
    D = E.D
    lags = np.unique(np.round(np.geomspace(1, 300, 20)).astype(int))
    dirs = {"E-W": (0, 1), "N-S": (1, 0), "NE-SW": (-0.7071, 0.7071), "NW-SE": (0.7071, 0.7071)}
    s = sp("F005", "autocorrelation", "Empirical spatial autocorrelation by direction", "A", "What are the "
           "characteristic length scales, and how much of the 100 m resolution is meaningful?", "Correlation vs lag in "
           "four directions for primary fields", "Correlation length bounds the smallest resolvable structure",
           "Pearson r at lags 0.1-30 km on a pixel sample; correlation length = first lag with r < 1/e",
           {DESCRIPTIVE: "Correlation lengths per field and direction."},
           {DESCRIPTIVE: "A correlation length >> 300 m means the field cannot localise traces by itself."},
           "Non-stationary fields; r mixes regional trends and local structure.", 1)
    mods = [m for m in PRIMARY if D.has(PRIMARY[m])]
    fig, ax = V.grid(len(mods), 4, 3.6, 3.0)
    rows = []
    for a_, m in zip(ax, mods):
        for k, (dn, d) in enumerate(dirs.items()):
            r = autocorr(D.band(PRIMARY[m]), D.valid, lags, d, seed=E.seed + k)
            lk = lags * D.grid.pixel_km
            a_.semilogx(lk, r, label=dn)
            rows.append({"modality": m, "band": PRIMARY[m], "direction": dn, "corr_length_km": corr_length(lk, r),
                         **{f"r_{x:.2f}km": y for x, y in zip(lk, r)}})
        a_.axvline(0.3, color="r", lw=0.6)
        a_.axhline(1 / np.e, color="0.5", lw=0.5, ls=":")
        a_.set(title=f"{m} ({PRIMARY[m]})", xlabel="lag (km)", ylabel="r")
    ax[0].legend()
    R = pd.DataFrame(rows)
    cl = R.groupby("modality").corr_length_km.median()
    E.reg.save(s, fig, R, Result(DESCRIPTIVE, "median correlation length (km): " + ", ".join(
        f"{k} {v:.2f}" for k, v in cl.items()), {"corr_length_km": cl.to_dict()}, len(R)))


def f006(E):
    D = E.D
    ang = np.arange(0, 180, 22.5)
    s = sp("F006", "anisotropy", "Directional variogram at 2 km lag", "A", "Preferred signal orientations / "
           "acquisition artifacts?", "Semivariance at 2 km in 8 directions, primary fields", "Survey line "
           "artifacts appear as sharp, grid-aligned anisotropy; tectonic fabric as broad NNE / NNW anisotropy",
           "gamma(2 km, theta); anisotropy ratio max / min", {DESCRIPTIVE: "Directional anisotropy per field."},
           {DESCRIPTIVE: "Grid-aligned (0 / 90 deg) maxima suggest survey artifacts to filter before edge features."},
           "Single lag; anisotropy also reflects regional trends.", 2)
    mods = [m for m in LOCAL if D.has(PRIMARY[m])]
    p = V.plt()
    fig, ax = p.subplots(1, len(mods), subplot_kw={"projection": "polar"}, figsize=(3.2 * len(mods), 3.4))
    ax = np.atleast_1d(ax)
    rows = []
    for a_, m in zip(ax, mods):
        a = D.band(PRIMARY[m])
        g = []
        for t in ang:
            d = (np.sin(np.radians(t)) * -1, np.cos(np.radians(t)))
            r = autocorr(a, D.valid, [20], d, n=80_000, seed=E.seed)[0]
            g.append(1 - r)
        g = np.asarray(g)
        th = np.radians(np.r_[ang, ang + 180, ang[:1]])
        a_.plot(th, np.r_[g, g, g[:1]], color=PALETTE.get(m))
        a_.set_theta_zero_location("N")
        a_.set_theta_direction(-1)
        a_.set_title(m)
        rows.append({"modality": m, **{f"g_{int(t)}": v for t, v in zip(ang, g)},
                     "anisotropy_ratio": float(np.nanmax(g) / max(np.nanmin(g), 1e-9)),
                     "max_dir_deg": float(ang[int(np.nanargmax(g))])})
    R = pd.DataFrame(rows)
    E.reg.save(s, fig, R, Result(DESCRIPTIVE, "; ".join(f"{r.modality}: ratio {r.anisotropy_ratio:.2f}, max at "
                                                       f"{r.max_dir_deg:.0f} deg" for r in R.itertuples()),
                                 {}, len(R)))


def psd(a: np.ndarray, valid: np.ndarray, px_km: float) -> tuple[np.ndarray, np.ndarray]:
    f = np.where(valid & np.isfinite(a), a, np.nan)
    f = np.where(np.isfinite(f), f - np.nanmean(f), 0.0)
    w = np.outer(np.hanning(f.shape[0]), np.hanning(f.shape[1]))
    P = np.abs(np.fft.fftshift(np.fft.fft2(f * w))) ** 2
    ky = np.fft.fftshift(np.fft.fftfreq(f.shape[0], px_km))
    kx = np.fft.fftshift(np.fft.fftfreq(f.shape[1], px_km))
    K = np.hypot(*np.meshgrid(ky, kx, indexing="ij"))
    bins = np.geomspace(1 / 200, 1 / (2 * px_km), 40)
    idx = np.digitize(K, bins)
    p_ = np.array([P[idx == i].mean() if (idx == i).any() else np.nan for i in range(1, len(bins))])
    return 1 / (0.5 * (bins[1:] + bins[:-1])), p_


def f007(E):
    D = E.D
    s = sp("F007", "power_spectra", "Radially averaged power spectra", "A", "At which wavelengths does each "
           "modality carry variation?", "PSD of primary fields vs wavelength", "Trace localisation within 300 m "
           "needs power at wavelengths below ~1-2 km", "Share of power at wavelengths < 2 km",
           {DESCRIPTIVE: "Spectral content per modality."},
           {DESCRIPTIVE: "Little short-wavelength power: filtering / high-pass features cannot create trace detail."},
           "Windowed FFT of a masked field with zero fill; edges add leakage.", 1)
    mods = [m for m in PRIMARY if D.has(PRIMARY[m])]
    p = V.plt()
    fig, ax = p.subplots(figsize=(7, 4.5))
    rows = []
    for m in mods:
        wl, pw = psd(D.band(PRIMARY[m])[::2, ::2], D.valid[::2, ::2], 2 * D.grid.pixel_km)
        ax.loglog(wl, pw / np.nanmax(pw), label=m, color=PALETTE.get(m))
        sh = float(np.nansum(pw[wl < 2]) / np.nansum(pw))
        rows.append({"modality": m, "short_wavelength_power_share": sh,
                     **{f"P_{w:.2f}km": v for w, v in zip(wl, pw / np.nanmax(pw))}})
    ax.axvline(0.6, color="r", lw=0.6)
    ax.set(xlabel="wavelength (km) [computed at 200 m to bound memory]", ylabel="normalised power",
           title="Power spectra (red: 0.6 km = 2 x 300 m)")
    ax.invert_xaxis()
    ax.legend()
    R = pd.DataFrame(rows)
    E.reg.save(s, fig, R, Result(DESCRIPTIVE, "power share < 2 km: " + ", ".join(
        f"{r.modality} {r.short_wavelength_power_share:.3f}" for r in R.itertuples()), {}, len(R)))


def f008(E):
    D = E.D
    pairs = [("tmi", "tmi_hg", "thdr"), ("iso_grav_anom", "iso_grav_anom_hg", "thdr"),
             ("iso_grav_anom", "iso_grav_anom_slope", "thdr"), ("det_elev", "det_elev_slope", "thdr"),
             ("tmi", "tmi_vg", "vdr")]
    pairs = [x for x in pairs if D.has(x[0]) and D.has(x[1])]
    s = sp("F008", "raw_vs_derived", "Do derived bands add information beyond their parent field?", "A",
           "Which bands are genuinely incremental?", "Provided derivative vs our recomputation from the parent band",
           "A derivative fully predictable from its parent is redundant input, not new evidence",
           "Spearman rho and R^2 of a monotone fit (rank space) on a pixel sample",
           {DESCRIPTIVE: "Redundancy of each provided derivative."},
           {DESCRIPTIVE: "rho >= 0.9: drop or treat as one evidence family with the parent."},
           "Different filters / scales in the provider's processing lower rho without new information.", 2)
    from scipy.stats import spearmanr

    fig, ax = V.grid(max(len(pairs), 1), 3, 4, 3.6)
    rows = []
    rng = np.random.default_rng(E.seed)
    rr, cc = np.nonzero(D.valid)
    i = rng.choice(len(rr), min(100_000, len(rr)), replace=False)
    for a_, (p_, d_, op) in zip(ax, pairs):
        ours = D.enh.thdr(D.band(p_), 0.2) if op == "thdr" else D.enh.vertical_derivative(D.band(p_))
        x, y = ours[rr[i], cc[i]], D.band(d_)[rr[i], cc[i]]
        m = np.isfinite(x) & np.isfinite(y)
        rho = float(spearmanr(x[m], y[m]).correlation) if m.sum() > 100 else np.nan
        a_.hexbin(np.argsort(np.argsort(x[m])) / m.sum(), np.argsort(np.argsort(y[m])) / m.sum(), gridsize=40,
                  bins="log", cmap="Greys")
        a_.set(title=f"{d_} vs {op}({p_}) rho={rho:.2f}", xlabel=f"rank {op}({p_})", ylabel=f"rank {d_}")
        rows.append({"parent": p_, "derived": d_, "recomputed_as": op, "spearman": rho, "n": int(m.sum()),
                     "redundant": bool(rho >= 0.9)})
    R = pd.DataFrame(rows)
    E.reg.save(s, fig, R, Result(DESCRIPTIVE, "; ".join(f"{r.derived}: rho {r.spearman:.2f}" for r in R.itertuples())
                                 or "no parent / derivative pair present", {}, len(R)))


def f009(E):
    D = E.D
    R5 = E.table("figures/data/F005_autocorrelation.csv")
    s = sp("F009", "effective_support", "Nominal 100 m pixel vs effective spatial support", "A", "Which channels "
           "can realistically contribute to 300 m localisation?", "Correlation length (F005) and half-power "
           "wavelength vs the 100 m pixel and the 300 m tolerance", "A band interpolated from sparse data has "
           "a 100 m pixel but km-scale support", "Effective support = median directional correlation length",
           {DESCRIPTIVE: "Effective support per band family."},
           {DESCRIPTIVE: "Support > 1 km: corridor context only; <= 300 m: candidate trace evidence."},
           "Correlation length also grows with real geological smoothness (not only interpolation).", 1)
    if not len(R5):
        return E.reg.skip(s, "F005 data absent")
    cl = R5.groupby("modality").corr_length_km.median().sort_values()
    p = V.plt()
    fig, ax = p.subplots(figsize=(7, 3.8))
    ax.barh(cl.index, cl.values, color=[PALETTE.get(m) for m in cl.index])
    ax.axvline(0.1, color="k", lw=0.6, label="100 m pixel")
    ax.axvline(0.3, color="r", lw=0.8, label="300 m tolerance")
    ax.set_xscale("log")
    ax.set(xlabel="effective support (km)", title="Effective spatial support")
    ax.legend()
    E.reg.save(s, fig, cl.reset_index(), Result(DESCRIPTIVE, "families with support <= 0.3 km: "
                                                             f"{list(cl[cl <= 0.3].index) or 'none'}",
                                                {"support_km": cl.to_dict()}, len(cl)))


def f010(E):
    D = E.D
    dom = E.domains()
    s = sp("F010", "missingness", "Data coverage by band and domain", "A", "Is missing coverage confounded with "
           "geological environments?", "Finite share inside the footprint, per band and label-free domain",
           "Coverage gaps aligned with domains bias any domain-level conclusion", "Share of footprint pixels with "
           "finite values", {DESCRIPTIVE: "Coverage table."},
           {DESCRIPTIVE: "Bands with domain-dependent gaps need explicit availability masks, never zero fill."},
           "The footprint itself is defined by band availability (>= half of the bands).", 2)
    rows = []
    for n in D.stack.names:
        a = np.isfinite(D.band(n))
        r = {"band": n, "modality": D.modality(n), "footprint_share": float(a[D.valid].mean())}
        for d in np.unique(dom[dom >= 0]):
            m = dom == d
            r[f"domain_{d}"] = float(a[m].mean())
        rows.append(r)
    R = pd.DataFrame(rows)
    p = V.plt()
    fig, ax = p.subplots(figsize=(8, 6))
    cols = [c for c in R if c.startswith("domain_")]
    im = ax.imshow(R[["footprint_share"] + cols].to_numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(R)))
    ax.set_yticklabels(R.band, fontsize=6)
    ax.set_xticks(range(len(cols) + 1))
    ax.set_xticklabels(["footprint"] + cols, rotation=45, fontsize=6)
    fig.colorbar(im, ax=ax)
    ax.set_title("Finite-value share")
    rng_ = (R[cols].max(axis=1) - R[cols].min(axis=1)) if cols else pd.Series(0, index=R.index)
    E.reg.save(s, fig, R, Result(DESCRIPTIVE, f"bands with < 99 % footprint coverage: "
                                              f"{list(R.band[R.footprint_share < 0.99]) or 'none'}; max between-domain "
                                              f"coverage range {rng_.max():.3f}", {}, len(R)))


def f011(E):
    D = E.D
    dom = E.domains()
    s = sp("F011", "regional_distributions", "Primary-field distributions by domain", "A", "Could global "
           "normalisation obscure region-specific contrasts?", "Robust-z distributions per domain",
           "If domains differ by several MADs, a global threshold means different things in different regions",
           "Between-domain variance share (eta^2) on a pixel sample", {DESCRIPTIVE: "Domain shift per field."},
           {DESCRIPTIVE: "High eta^2: use within-domain ranks / normalisation for edge features."},
           "Domains were built from some of these fields (circular for those fields; reported).", 2)
    mods = [m for m in PRIMARY if D.has(PRIMARY[m])]
    fig, ax = V.grid(len(mods), 4, 3.8, 3.0)
    rng = np.random.default_rng(E.seed)
    rr, cc = np.nonzero(dom >= 0)
    i = rng.choice(len(rr), min(150_000, len(rr)), replace=False)
    rows = []
    for a_, m in zip(ax, mods):
        from .data import zscore

        z, _, _ = zscore(D.band(PRIMARY[m]), D.valid)
        v, d = z[rr[i], cc[i]], dom[rr[i], cc[i]]
        ok = np.isfinite(v)
        ds = sorted(np.unique(d[ok]))
        a_.boxplot([v[ok & (d == k)] for k in ds], labels=[str(k) for k in ds], showfliers=False)
        tot = np.var(v[ok])
        between = np.sum([((d[ok] == k).mean()) * (np.mean(v[ok & (d == k)]) - np.mean(v[ok])) ** 2 for k in ds])
        eta = float(between / tot) if tot > 0 else np.nan
        a_.set(title=f"{m}: eta^2 = {eta:.2f}", xlabel="domain", ylabel="robust z")
        rows.append({"modality": m, "eta2_between_domains": eta})
    R = pd.DataFrame(rows)
    E.reg.save(s, fig, R, Result(DESCRIPTIVE, "eta^2: " + ", ".join(f"{r.modality} {r.eta2_between_domains:.2f}"
                                                                    for r in R.itertuples()), {}, len(i)))


def f012(E):
    D = E.D
    wins = domain_windows(E, 4, 250)
    mods = [m for m in ("magnetic", "gravity", "terrain", "basement") if D.has(PRIMARY[m])]
    s = sp("F012", "regional_signatures", "Representative domains, four-way comparison", "A", "How strongly does the "
           "underlying geophysics vary across the study area?", "Four domains x four modalities, identical window size "
           "and per-modality shared colour scale", "Shared scales make cross-domain contrasts visible",
           "Per-modality 2-98 % stretch over the union of the four windows", {DESCRIPTIVE: "Regional contrast panel."},
           {DESCRIPTIVE: "Visibly different textures: domain context is a candidate input."},
           "Windows centred on the largest block cluster of each domain.", 2)
    fig, ax = V.grid(len(wins) * len(mods), len(mods), 3.6, 3.4)
    i = 0
    for m in mods:
        b = PRIMARY[m]
        allv = np.concatenate([D.band(b)[V.win(D, r, c, 250)].ravel() for _, r, c in wins])
        lo, hi = V.stretch(allv, (2, 98), b in V.DIVERGING)
        E.D.cache[f"_st_{b}"] = (lo, hi)
    for d, r, c in wins:
        for m in mods:
            b = PRIMARY[m]
            lo, hi = E.D.cache[f"_st_{b}"]
            V.show(ax[i], D, D.band(b), f"domain {d}: {m}", sl=V.win(D, r, c, 250), band=b, vmin=lo, vmax=hi,
                   units=units_of(E, b))
            i += 1
    E.reg.save(s, fig, pd.DataFrame([{"domain": d, "row": r, "col": c} for d, r, c in wins]),
               Result(DESCRIPTIVE, f"{len(wins)} domains compared", {}, len(wins)))


# =============================================================================== PART B
def segs(E) -> pd.DataFrame:
    return E.D.net.segments


def f013(E):
    D = E.D
    s = sp("F013", "fault_network", "Mapped-fault network over terrain and domains", "B", "What does the training "
           "label set look like at regional and local scale?", "Mapped traces over detrended elevation; zoom on the "
           "densest network", "Context for every later comparison", "Counts and total length",
           {DESCRIPTIVE: "Network overview."}, {DESCRIPTIVE: "Context only."}, "Labels are incomplete.", 2)
    p = V.plt()
    fig, ax = p.subplots(1, 2, figsize=(13, 6))
    V.show(ax[0], D, D.band("det_elev") if D.has("det_elev") else np.zeros(D.grid.shape), "mapped faults over "
           "detrended elevation", cmap="gray", step=3, faults=True)
    from gems_eda.morphology import trace_density

    dens = trace_density(D.net, D.grid, 10.0)
    r, c = np.unravel_index(np.nanargmax(np.where(D.valid, dens, np.nan)), dens.shape)
    sl = V.win(D, int(r), int(c), 250)
    V.show(ax[1], D, D.band("det_elev") if D.has("det_elev") else dens, "densest network (50 km)", cmap="gray", sl=sl,
           faults=True)
    S = segs(E)
    E.reg.save(s, fig, pd.DataFrame([{"n_segments": len(S), "total_km": S.length_km.sum(),
                                      "n_components": int(D.net.components.max()),
                                      "fault_px": int(D.labels.sum())}]),
               Result(DESCRIPTIVE, f"{len(S)} traced segments, {S.length_km.sum():.0f} km, "
                                   f"{int(D.net.components.max())} raster components, {int(D.labels.sum())} fault px",
                      {}, len(S)))


def f014(E):
    S = segs(E)
    comp = S.groupby("component").length_km.sum()
    s = sp("F014", "fault_lengths", "Fault-length distributions (branches vs components)", "B", "Do labels "
           "represent short structures, long systems or fragmented networks?", "ECDF of traced branch lengths and "
           "connected-component lengths; length-weighted histogram", "Branch vs component lengths separate "
           "fragmentation from genuine fault size", "Median, p90, share of total length in branches < 2 km",
           {DESCRIPTIVE: "Length statistics."}, {DESCRIPTIVE: "A large short-branch share: candidate generators must "
                                                              "keep short segments."},
           "Skeleton branches split at junctions; components merge networks.", 1)
    p = V.plt()
    fig, ax = p.subplots(1, 2, figsize=(11, 4))
    V.ecdf(ax[0], S.length_km, "branches")
    V.ecdf(ax[0], comp, "components")
    ax[0].set_xscale("log")
    ax[0].set(xlabel="length (km)", ylabel="ECDF")
    ax[0].legend()
    ax[1].hist(S.length_km, bins=np.geomspace(0.1, 100, 40), weights=S.length_km, color="0.4")
    ax[1].set_xscale("log")
    ax[1].set(xlabel="branch length (km)", ylabel="total length (km)", title="length-weighted")
    short = float(S.length_km[S.length_km < 2].sum() / S.length_km.sum())
    E.reg.save(s, fig, pd.DataFrame({"length_km": S.length_km, "component": S.component}),
               Result(DESCRIPTIVE, f"branch median {S.length_km.median():.2f} km (p90 {S.length_km.quantile(0.9):.2f}); "
                                   f"component median {comp.median():.2f} km; {short:.1%} of trace length in branches "
                                   "< 2 km", {"short_share": short}, len(S)))


def rose(ax, az, w, color, label):
    from gems_eda.stats import axial_hist

    c, h = axial_hist(az, w, 36)
    th = np.radians(np.r_[c, c + 180])
    ax.bar(th, np.r_[h, h], width=np.radians(5), color=color, alpha=0.7, label=label)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_yticklabels([])


def f015(E):
    from .core import axial_diff_deg, axial_mean_deg

    S = segs(E)
    s = sp("F015", "orientation_families", "Orientation families, global vs regional", "B", "Is strike spatially "
           "dependent? Could a model learn a biased global direction prior?", "Length-weighted roses: all faults and "
           "each 3 x 3 region", "Basin-and-Range fault sets rotate across the region", "Axial mean and resultant "
           "length R; regional deviation from the global mean", {DESCRIPTIVE: "Regional strike variation."},
           {DESCRIPTIVE: "Regional deviations > 20 deg: avoid a single global orientation prior."},
           "Regions are a grid, not tectonic provinces.", 2)
    p = V.plt()
    fig, ax = p.subplots(2, 5, subplot_kw={"projection": "polar"}, figsize=(16, 6.5))
    ax = ax.ravel()
    g, R0 = axial_mean_deg(S.azimuth, S.length_km)
    rose(ax[0], S.azimuth, S.length_km, "k", "all")
    ax[0].set_title(f"all: mean {g:.0f} deg, R {R0:.2f}")
    rows = [{"region": "all", "mean_az": g, "R": R0, "n": len(S), "dev_from_global": 0.0}]
    for k in range(9):
        x = S[S.region == k]
        if len(x) < 10:
            ax[k + 1].set_axis_off()
            continue
        m, R_ = axial_mean_deg(x.azimuth, x.length_km)
        rose(ax[k + 1], x.azimuth, x.length_km, "0.3", str(k))
        ax[k + 1].set_title(f"region {k}: {m:.0f} deg (n={len(x)})", fontsize=7)
        rows.append({"region": k, "mean_az": m, "R": R_, "n": len(x), "dev_from_global": float(axial_diff_deg(m, g))})
    T = pd.DataFrame(rows)
    E.reg.save(s, fig, T, Result(DESCRIPTIVE, f"global mean {g:.0f} deg (R {R0:.2f}); max regional deviation "
                                              f"{T.dev_from_global.max():.0f} deg", {}, len(S)))


def f016(E):
    from .core import axial_mean_deg

    S = segs(E)
    bins = [(0, 1), (1, 3), (3, 10), (10, 1e9)]
    s = sp("F016", "orientation_vs_length", "Strike by segment length", "B", "Do short splays follow different "
           "geometries from regional faults?", "Roses per length class", "Splays / relay structures can be oblique "
           "to master faults", "Axial mean and R per class", {DESCRIPTIVE: "Strike by length class."},
           {DESCRIPTIVE: "Different short-fault strikes: orientation priors must not be learned from long faults only."},
           "Length classes from skeleton branches.", 2)
    p = V.plt()
    fig, ax = p.subplots(1, 4, subplot_kw={"projection": "polar"}, figsize=(13, 3.6))
    rows = []
    for a_, (lo, hi) in zip(ax, bins):
        x = S[(S.length_km >= lo) & (S.length_km < hi)]
        if not len(x):
            a_.set_axis_off()
            continue
        m, R_ = axial_mean_deg(x.azimuth, x.length_km)
        rose(a_, x.azimuth, x.length_km, "0.3", "")
        a_.set_title(f"{lo}-{hi if hi < 1e9 else 'inf'} km: {m:.0f} deg R {R_:.2f} (n={len(x)})", fontsize=7)
        rows.append({"length_class": f"{lo}-{hi}", "mean_az": m, "R": R_, "n": len(x)})
    T = pd.DataFrame(rows)
    E.reg.save(s, fig, T, Result(DESCRIPTIVE, "; ".join(f"{r.length_class}: {r.mean_az:.0f} deg (R {r.R:.2f})"
                                                       for r in T.itertuples()), {}, len(S)))


def f017(E):
    S = segs(E).copy()
    dom = E.domains()
    S["domain"] = dom[S.r_mid, S.c_mid]
    S["length_class"] = pd.cut(S.length_km, [0, 1, 3, 10, 1e9], labels=["<1", "1-3", "3-10", ">10"])
    s = sp("F017", "curvature_straightness", "Sinuosity, linearity and curvature", "B", "Are straight-line "
           "candidate algorithms unsuitable for some structures?", "Distributions by length class and domain",
           "Curved / sinuous traces defeat straight-probe and Hough-type detectors", "Medians per class / domain",
           {DESCRIPTIVE: "Geometry statistics."}, {DESCRIPTIVE: "High curvature classes need curve-following "
                                                                "detectors (not straight probes)."},
           "Pixel-path curvature is noisy for short segments (NaN below 4 steps).", 2)
    fig, ax = V.grid(3, 3, 4.2, 3.4)
    for a_, col in zip(ax, ("sinuosity", "linearity", "curv_mean_deg_km")):
        g = [S[S.length_class == k][col].dropna() for k in ["<1", "1-3", "3-10", ">10"]]
        a_.boxplot(g, labels=["<1", "1-3", "3-10", ">10"], showfliers=False)
        a_.set(title=col, xlabel="length class (km)")
    T = S.groupby(["length_class", "domain"], observed=True)[["sinuosity", "linearity", "curv_mean_deg_km"]].median() \
        .reset_index()
    E.reg.save(s, fig, T, Result(DESCRIPTIVE, f"median sinuosity {S.sinuosity.median():.3f}, linearity "
                                              f"{S.linearity.median():.3f}, curvature {S.curv_mean_deg_km.median():.2f} "
                                              "deg/km", {}, len(S)))


def f018(E):
    from gems_eda.morphology import component_diagnostics

    D = E.D
    C = component_diagnostics(D.net, D.grid, 3.0)
    nodes = D.net.nodes
    s = sp("F018", "network_topology", "Endpoints, junctions, branches, isolated components", "B", "How diverse is "
           "fault geometry; what is lost when components are treated as single faults?", "Node and component "
           "classes; three representative networks", "Networks merge many faults through junctions",
           "Counts by class", {DESCRIPTIVE: "Topology statistics."},
           {DESCRIPTIVE: "Many junction-rich networks: evaluate per branch, not per raster component."},
           "Component classes depend on the 3 km short-fault threshold and width heuristics.", 2)
    p = V.plt()
    fig = p.figure(figsize=(14, 4.5))
    a0 = fig.add_subplot(1, 4, 1)
    kc = C.kind.value_counts()
    a0.bar(kc.index, kc.values, color="0.4")
    a0.set(title="component classes", ylabel="count")
    a0.tick_params(axis="x", rotation=30)
    big = C.sort_values("n_segments", ascending=False).head(3)
    for k, comp in enumerate(big.component, start=2):
        a = fig.add_subplot(1, 4, k)
        rr, cc = np.nonzero(D.net.components == comp)
        sl = V.win(D, int(np.median(rr)), int(np.median(cc)), 120)
        V.show(a, D, D.labels.astype(float), f"network {comp}", cmap="Greys", sl=sl, colorbar=False, vmin=0, vmax=1)
    nk = nodes.kind.value_counts().to_dict() if len(nodes) else {}
    E.reg.save(s, fig, C, Result(DESCRIPTIVE, f"component classes {kc.to_dict()}; nodes {nk}", {}, len(C)))


def f019(E):
    from gems_eda.morphology import endpoint_gaps

    D = E.D
    G = endpoint_gaps(D.net, D.grid, 5.0, 25.0)
    s = sp("F019", "fragmentation", "Collinear endpoints across components", "B", "How often are mapped traces "
           "interrupted; is network completion justified?", "Endpoint pairs of different components within 5 km, "
           "aligned within 25 deg", "Collinear gaps suggest unmapped continuations (or mapping breaks)",
           "Gap-length ECDF; share of endpoints with a collinear partner", {DESCRIPTIVE: "Fragmentation statistics."},
           {DESCRIPTIVE: "Many short aligned gaps: tip-continuation candidates are geologically motivated."},
           "Alignment is geometric only; gaps can be real fault terminations.", 2)
    p = V.plt()
    fig, ax = p.subplots(figsize=(6, 4))
    if len(G):
        V.ecdf(ax, G.gap_km, "aligned endpoint gaps")
    ax.set(xlabel="gap (km)", ylabel="ECDF", title="Aligned endpoint gaps")
    nodes = D.net.nodes
    n_end = int((nodes.kind == "endpoint").sum()) if len(nodes) and "kind" in nodes else 0
    share = 2 * len(G) / max(n_end, 1)
    E.reg.save(s, fig, G, Result(DESCRIPTIVE, f"{len(G)} aligned pairs; ~{share:.1%} of {n_end} endpoints have a "
                                              f"partner; median gap {G.gap_km.median() if len(G) else np.nan:.2f} km",
                                 {"aligned_pairs": len(G)}, len(G)))


def f020(E):
    from gems_eda.morphology import trace_density
    from scipy.stats import spearmanr

    D = E.D
    B = E.table("datasets/physical_regions.parquet")
    dens = trace_density(D.net, D.grid, 5.0)
    s = sp("F020", "density_vs_evidence", "Mapped-fault density vs label-free physical evidence (5 km blocks)", "B",
           "Does mapped density vary with geology, mapping effort, or both?", "Block trace density vs block magnetic / "
           "gravity gradient and terrain relief", "A strong dependence on easily visible terrain could indicate "
           "mapping-effort bias; a dependence on buried fields could indicate geology", "Spearman rho with a "
           "block-bootstrap CI", {DESCRIPTIVE: "Association only (no causal attribution)."},
           {DESCRIPTIVE: "A strong terrain-only association warns that labels over-represent surface-visible faults."},
           "Ecological correlation; density is smoothed.", 1)
    bp = 50
    B = B.copy()
    B["density"] = [float(np.nanmean(dens[i * bp:(i + 1) * bp, j * bp:(j + 1) * bp])) for i, j in zip(B.bi, B.bj)]
    cols = [c for c in ("mag_grad_2km", "grav_grad_2km", "terrain_relief", "terrain_slope") if c in B]
    fig, ax = V.grid(len(cols), 4, 3.6, 3.2)
    rows = []
    rng = np.random.default_rng(E.seed)
    for a_, c_ in zip(ax, cols):
        x, y = B[c_].to_numpy(), B.density.to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        rho = spearmanr(x[ok], y[ok]).correlation
        bs = [spearmanr(x[ok][i], y[ok][i]).correlation for i in
              (rng.integers(0, ok.sum(), ok.sum()) for _ in range(min(E.n_boot, 300)))]
        lo, hi = np.nanquantile(bs, [0.025, 0.975])
        a_.hexbin(x[ok], y[ok], gridsize=30, bins="log", cmap="Greys")
        a_.set(title=f"{c_}: rho {rho:.2f} [{lo:.2f}, {hi:.2f}]", xlabel=c_, ylabel="trace km / 100 km2")
        rows.append({"variable": c_, "spearman": rho, "ci_lo": lo, "ci_hi": hi, "n_blocks": int(ok.sum())})
    T = pd.DataFrame(rows)
    E.reg.save(s, fig, T, Result(DESCRIPTIVE, "; ".join(f"{r.variable} rho {r.spearman:.2f} [{r.ci_lo:.2f}, "
                                                       f"{r.ci_hi:.2f}]" for r in T.itertuples()), {}, len(B)))


def f021(E):
    D = E.D
    S = segs(E)
    s = sp("F021", "short_vs_long_context", "Physical context of short (< 2 km) vs long (> 10 km) faults", "B",
           "Do different fault scales appear in different physical contexts?", "Context at segment midpoints, "
           "short vs long", "Short splays may live in different settings than master faults", "Cliff's delta, "
           "cluster bootstrap over components", TEST_OUT,
           {SUPPORTED: "Different contexts: stratify validation / features by fault scale.",
            NOT_SUPPORTED: "Same context: one feature set can serve both scales.", INCONCLUSIVE: "Undecided."},
           "Midpoint context only; long faults sample more domains.", 2)
    ctx = {"slope": D.band("det_elev_slope"), "mag_grad_2km": D.enh.thdr(D.band("rtp"), 2.0) if D.has("rtp") else None,
           "grav_grad_2km": D.enh.thdr(D.band("iso_grav_anom"), 2.0) if D.has("iso_grav_anom") else None}
    sh, lg = S[S.length_km < 2], S[S.length_km > 10]
    rows = []
    fig, ax = V.grid(len(ctx), 3, 4, 3.2)
    for a_, (k, a) in zip(ax, ctx.items()):
        if a is None:
            a_.set_axis_off()
            continue
        x, y = a[sh.r_mid, sh.c_mid], a[lg.r_mid, lg.c_mid]
        e = cluster_effect(x, sh.component, y, lg.component, E.n_boot, E.seed)
        V.ecdf(a_, x, "short < 2 km")
        V.ecdf(a_, y, "long > 10 km")
        a_.set(title=f"{k}: delta {e['effect']:+.2f}", xlabel=k)
        a_.legend()
        rows.append({"variable": k, **e})
    T = pd.DataFrame(rows)
    best = T.iloc[T.effect.abs().argmax()] if len(T) else None
    res = effect_result(best.to_dict(), E.min_effect, None, f"largest contrast {best.variable}:") if best is not None \
        else Result(INCONCLUSIVE, "no context band")
    E.reg.save(s, fig, T, res)


def f022(E):
    D = E.D
    S = segs(E).copy()
    from gems_eda.morphology import endpoint_gaps

    G = endpoint_gaps(D.net, D.grid, 5.0, 25.0)
    picks = {}
    a = S[(S.length_km.between(3, 10)) & (S.linearity > 0.98)]
    if len(a):
        picks["simple_linear"] = int(a.sort_values("linearity").seg_id.iloc[-1])
    if S.curv_mean_deg_km.notna().any():
        picks["curved"] = int(S.loc[S.curv_mean_deg_km.idxmax(), "seg_id"])
    if (S.n_junction_ends > 0).any():
        picks["branching"] = int(S[S.n_junction_ends == 2].sort_values("length_km").seg_id.iloc[-1]) \
            if (S.n_junction_ends == 2).any() else int(S[S.n_junction_ends > 0].seg_id.iloc[0])
    nodes = D.net.nodes
    if len(nodes) and (nodes.degree >= 4).any():
        n0 = nodes[nodes.degree >= 4].iloc[0]
        picks["intersection"] = int(S.iloc[np.argmin(np.hypot(S.r_mid - n0.r, S.c_mid - n0.c))].seg_id)
    if len(G):
        picks["fragmented"] = int(G.seg_a.iloc[0])
    iso = S[(S.start_deg <= 1) & (S.end_deg <= 1) & (S.length_km < 2)]
    if len(iso):
        picks["isolated_short"] = int(iso.seg_id.iloc[0])
    s = sp("F022", "complexity_atlas", "Fault complexity atlas", "B", "Which structural cases must later models "
           "handle?", "Six representative geometries, same window size", "Each case breaks a different model "
           "assumption", "Selected by explicit geometric rules (recorded)", {DESCRIPTIVE: "Example gallery."},
           {DESCRIPTIVE: "Each class should appear in validation reporting."}, "One example per class.", 3)
    fig, ax = V.grid(max(len(picks), 1), 3, 4.2, 3.8)
    rows = []
    for a_, (k, sid) in zip(ax, picks.items()):
        sg = S[S.seg_id == sid].iloc[0]
        sl = V.win(D, int(sg.r_mid), int(sg.c_mid), 60)
        V.show(a_, D, D.band("det_elev") if D.has("det_elev") else D.labels.astype(float), f"{k} (segment {sid})",
               cmap="gray", sl=sl, faults=True, colorbar=False)
        rows.append({"class": k, "seg_id": sid, "x_mid": sg.x_mid, "y_mid": sg.y_mid, "length_km": sg.length_km})
    E.reg.save(s, fig, pd.DataFrame(rows), Result(DESCRIPTIVE, f"classes found: {list(picks)}", {}, len(picks)))


PART_A = [f001, f002, f003, f004, f005, f006, f007, f008, f009, f010, f011, f012]
PART_B = [f013, f014, f015, f016, f017, f018, f019, f020, f021, f022]
