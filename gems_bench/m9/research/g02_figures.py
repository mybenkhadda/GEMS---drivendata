"""G02 figures (16:9, one geographic window per map figure, UTM kilometres, physical units on raw fields).

Every function returns (figure or None, plotting data as a DataFrame); ``G02Store.save_fig`` writes PNG + PDF + CSV.
Map figures use the evaluated unit with the most truth (figures 1-11, 15 overlay held-out truth: EVALUATION figures;
labels never enter a map). Examples are chosen by fixed, seeded rules that include STRONG and WEAK evidence (and strong
gradients with no G02 evidence): never only the visually best cases. Each panel title states which physical
interpretation the panel supports or contradicts.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import ndimage

from ...gems3.figures import INK, SERIES
from .geology_figures import H9, INK2, KNOWN_C, TRUTH_C, W16, _extent_km, _outline, map_grid

COLORS = {"B0": "#8a8986", "B1": SERIES[0], "B2": SERIES[1], "G02-A": SERIES[2], "G02-B": SERIES[3],
          "G02-C": SERIES[4], "G02-D": SERIES[5], "G02-E": SERIES[6], "G02-F": "#0b0b0b", "random": "#c9c8c4"}
HYP_COLORS = ["#d1495b", "#edae49", "#00798c", "#9a9a9a"]          # fault, contact, intrusive, ambiguous
UNITS = {"mag": "RTP magnetic anomaly (nT)", "grav": "isostatic gravity anomaly (mGal)",
         "terrain": "detrended elevation (m)"}


def _f(a):
    return None if a is None else np.asarray(a, np.float32)


def _rank(a, dom=None):
    """Window percentile rank (display only: makes maps with different score scales comparable)."""
    a = _f(a)
    if a is None:
        return None
    m = np.isfinite(a) & (np.ones(a.shape, bool) if dom is None else dom)
    out = np.full(a.shape, np.nan, np.float32)
    if m.any():
        out[m] = pd.Series(a[m]).rank(pct=True).to_numpy(np.float32)
    return out


def _sel(score, dom, frac):
    from .geology_metrics import equal_area_mask

    return equal_area_mask(np.where(dom, _f(score), np.nan), dom, frac)


def _fused(A, name):
    from .g02_experiment import parse_fusion
    from .geology_wavelet import combine

    if name in A:
        return _f(A[name])
    p = parse_fusion(name) if name else None
    if p and p[0] in A and p[1] in A:
        return combine([_f(A[p[0]]), _f(A[p[1]])], [1.0 - p[2], p[2]])
    return None


# 1 ------------------------------------------------------------------------------------------------- raw fields
def raw_fields(grid, w, raw: dict, truth, known):
    panels = [{"title": f"{UNITS[r]}", "raster": raw[r], "cmap": "cividis" if r != "terrain" else "gist_earth",
               "cbar": UNITS[r].split("(")[-1].rstrip(")"), "truth": truth, "known": known, "legend": i == 0}
              for i, r in enumerate(("mag", "grav", "terrain")) if raw.get(r) is not None]
    if not panels:
        return None, pd.DataFrame()
    return map_grid(grid, w, panels, ncols=len(panels),
                    title="1. Raw potential fields and terrain with reference faults (red = held-out truth, black = "
                          "known faults). A fault predicts offsets / truncations of anomaly bands; a contact predicts a "
                          "level change along the grain")


# 2 ------------------------------------------------------------------------------------------------- energy
def energy(grid, w, A: dict, lam_labels: list[str]):
    panels = []
    for key, lab in (("M", "magnetic"), ("G", "gravity")):
        for lam in lam_labels:
            a = A.get(f"scale:{key}@{lam}")
            if a is not None:
                panels.append({"title": f"{lab} wavelet energy percentile at {lam}", "raster": a})
    if not panels:
        return None, pd.DataFrame()
    return map_grid(grid, w, panels, ncols=max(len(lam_labels), 1), shared={"vmin": 0, "vmax": 1, "cmap": "magma"},
                    cbar_label="energy percentile (training-geography reference)",
                    title="2. Wavelet energy at the G02 wavelengths: contrasts, NOT faults (contacts, dykes and "
                          "intrusion margins are equally bright)")


# 3 ------------------------------------------------------------------------------------------------- orientation
def orientation(grid, w, A: dict):
    spec = [("meas:theta_line", "lineament strike (deg from N, axial)", "twilight", 0, 180),
            ("cont:strike_stability", "strike stability across scales", "viridis", 0, 1),
            ("comp:orient_gate", "G01 magnetic-gravity orientation gate", "viridis", 0, 1),
            ("meas:discord", "discordance with the 5 km regional grain", "magma", 0, 1)]
    panels = [{"title": t, "raster": A[k], "cmap": c, "vmin": lo, "vmax": hi} for k, t, c, lo, hi in spec if k in A]
    if not panels:
        return None, pd.DataFrame()
    return map_grid(grid, w, panels, ncols=len(panels),
                    title="3. Orientation and coherence. High discordance = the lineament cuts the regional grain "
                          "(fault-compatible); parallel to the grain = contact-compatible")


# 4 / 5 ---------------------------------------------------------------------------------------------- profiles
def pick_points(A: dict, key: str, n: int, seed: int) -> list[dict]:
    """Seeded examples: half STRONG (top 1 % of ``key``), half WEAK = strong G01 energy (M >= 0.9) but ``key`` in
    its bottom half (strong gradient, no G02 evidence)."""
    rng = np.random.default_rng(seed)
    s = _f(A.get(key))
    dom = np.asarray(A.get("domain", np.ones_like(s)), bool) if s is not None else None
    if s is None or not np.isfinite(s[dom]).any():
        return []
    v = s[dom & np.isfinite(s)]
    hi, med = np.quantile(v, 0.99), np.quantile(v, 0.5)
    out = []
    strong = np.argwhere(dom & (s >= hi))
    M = _f(A.get("comp:M"))
    weak = np.argwhere(dom & (s <= med) & ((M >= 0.9) if M is not None else True))
    for lab, pool, k in (("strong", strong, (n + 1) // 2), ("weak (strong gradient)", weak, n // 2)):
        if len(pool):
            for i in rng.choice(len(pool), min(k, len(pool)), replace=False):
                out.append({"r": int(pool[i][0]), "c": int(pool[i][1]), "label": lab})
    return out


def _profile(a, r, c, az_deg, dist_m, pixel_m):
    from .geology_displacement import offset_px

    dr, dc = offset_px(dist_m, az_deg, pixel_m)
    return ndimage.map_coordinates(np.asarray(a, np.float32), [r + dr, c + dc], order=1, mode="constant",
                                   cval=np.nan, prefilter=False)


def _pt_params(A, p):
    az = A.get("disp[mag]:azimuth")
    lam = A.get("disp[mag]:lambda_km")
    a = float(az[p["r"], p["c"]]) if az is not None and np.isfinite(az[p["r"], p["c"]]) else 0.0
    lm = float(lam[p["r"], p["c"]]) if lam is not None and np.isfinite(lam[p["r"], p["c"]]) else 1.6
    return a, lm


def cross_profiles(grid, w, raw: dict, A: dict, pts: list[dict], pixel_m, truth):
    import matplotlib.pyplot as plt

    if not pts or raw.get("mag") is None:
        return None, pd.DataFrame()
    n = len(pts)
    fig, axes = plt.subplots(1, n, figsize=(W16, H9 * 0.6), squeeze=False, constrained_layout=True)
    rows = []
    for ax, p in zip(axes[0], pts):
        az, lam = _pt_params(A, p)
        v = np.arange(-3.0 * lam * 1000, 3.0 * lam * 1000 + 1, 100.0)
        pm = _profile(raw["mag"], p["r"], p["c"], az + 90.0, v, pixel_m)
        ax.plot(v / 1000, pm, color=SERIES[0], lw=1.4, label="RTP (nT)")
        if raw.get("grav") is not None:
            ax2 = ax.twinx()
            ax2.plot(v / 1000, _profile(raw["grav"], p["r"], p["c"], az + 90.0, v, pixel_m), color=SERIES[1], lw=1.2,
                     ls="--", label="gravity (mGal)")
            ax2.tick_params(labelsize=7, colors=SERIES[1])
        if truth is not None:
            tv = _profile(np.asarray(truth, np.float32), p["r"], p["c"], az + 90.0, v, pixel_m)
            for x in v[np.nan_to_num(tv) > 0.5]:
                ax.axvline(x / 1000, color=TRUTH_C, lw=0.6, alpha=0.6)
        ax.axvline(0, color=INK, lw=0.8, ls=":")
        sc = A.get("G02-A")
        s = float(sc[p["r"], p["c"]]) if sc is not None else np.nan
        off = A.get("disp[mag]:offset_m")
        o = float(off[p["r"], p["c"]]) if off is not None else np.nan
        verdict = "supports offset fabric" if s > 0.05 else "no displacement evidence (fault NOT excluded)"
        ax.set_title(f"{p['label']}: G02-A={s:.2f}, offset {o:+.0f} m\nprofile normal to {az:.0f} deg; {verdict}",
                     fontsize=8, loc="left", color=INK)
        ax.set_xlabel("distance across candidate line (km)", fontsize=7, color=INK2)
        ax.tick_params(labelsize=7)
        rows.append({"label": p["label"], "row": p["r"], "col": p["c"], "azimuth": az, "lambda_km": lam, "G02_A": s,
                     "offset_m": o})
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("4. Cross-fault magnetic (solid) and gravity (dashed) profiles through STRONG and WEAK G02-A points "
                 "(red ticks = held-out truth crossings; dotted = candidate line)", x=0.01, ha="left", fontsize=11,
                 color=INK)
    return fig, pd.DataFrame(rows)


def displacement_examples(grid, w, raw: dict, A: dict, pts: list[dict], pixel_m, cfg: dict, truth):
    import matplotlib.pyplot as plt

    from .geology_displacement import bandpass, disp_scale, frame_vectors

    if not pts or raw.get("mag") is None:
        return None, pd.DataFrame()
    n = min(len(pts), 4)
    fig, axes = plt.subplots(n, 2, figsize=(W16, H9), squeeze=False, constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 2]})
    rows = []
    valid = np.isfinite(raw["mag"])
    cache = {}
    for i, p in enumerate(pts[:n]):
        az, lam = _pt_params(A, p)
        if lam not in cache:
            cache[lam] = bandpass(raw["mag"], valid, lam, pixel_m, float(cfg["bandpass"]["sigma_ratio"]))
        f = cache[lam]
        sc = disp_scale(lam, pixel_m, cfg["displacement"])
        h1, L = sc.k1 * sc.step_m, sc.half * sc.step_m
        u = np.arange(-L, L + 1, 50.0)
        (tE, tN), (nE, nN) = frame_vectors(az)

        def prof(vn):
            E, N = u * tE + vn * nE, u * tN + vn * nN
            return ndimage.map_coordinates(f, [p["r"] - N / pixel_m[0], p["c"] + E / pixel_m[1]], order=1,
                                           mode="constant", cval=np.nan, prefilter=False)
        Lp, Rp = prof(-h1), prof(h1)
        off = A.get("disp[mag]:offset_m")
        o = float(off[p["r"], p["c"]]) if off is not None and np.isfinite(off[p["r"], p["c"]]) else 0.0
        amb = A.get("disp[mag]:ambiguity")
        am = float(amb[p["r"], p["c"]]) if amb is not None else np.nan
        ax = axes[i, 0]
        hw = int(3 * lam * 1000 / pixel_m[0])
        r0, r1, c0, c1 = max(p["r"] - hw, 0), p["r"] + hw, max(p["c"] - hw, 0), p["c"] + hw
        sub = f[r0:r1, c0:c1]
        ext = _extent_km(grid, (w[0] + r0, w[0] + min(r1, f.shape[0]), w[2] + c0, w[2] + min(c1, f.shape[1])))
        if sub.size:
            lo, hi = np.nanpercentile(sub, [2, 98]) if np.isfinite(sub).any() else (0, 1)
            ax.imshow(sub, extent=ext, cmap="RdBu_r", vmin=lo, vmax=hi, origin="upper", interpolation="nearest")
            _outline(ax, None if truth is None else np.asarray(truth)[r0:r1, c0:c1], ext, TRUTH_C)
            x0, y0 = grid.rc_to_xy(w[0] + p["r"], w[2] + p["c"])
            for vn, ls in ((0.0, "-"), (-h1, ":"), (h1, ":")):
                xs = (x0 + u * tE + vn * nE) / 1000
                ys = (y0 + u * tN + vn * nN) / 1000
                ax.plot(xs, ys, color=INK if vn == 0 else INK2, lw=0.9, ls=ls)
            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])
        ax.set_title(f"{p['label']}: band-pass {lam:g} km, line {az:.0f} deg", fontsize=8, loc="left", color=INK)
        ax.tick_params(labelsize=6)
        ax = axes[i, 1]
        ax.plot(u / 1000, Lp, color=SERIES[0], lw=1.3, label="-n side profile")
        ax.plot(u / 1000, Rp, color=SERIES[1], lw=1.0, ls="--", label="+n side profile")
        ax.plot((u - o) / 1000, Rp, color=SERIES[1], lw=1.4, label=f"+n side shifted by {-o:+.0f} m")
        sense = "apparent sinistral" if o > 0 else "apparent dextral" if o < 0 else "none"
        ax.set_title(f"apparent offset {o:+.0f} m ({sense}), periodicity ambiguity {am:.2f}. An APPARENT pattern "
                     "offset: dip slip, source depth or a pre-existing bend can mimic it", fontsize=8, loc="left",
                     color=INK)
        ax.set_xlabel("distance along the candidate line (km)", fontsize=7, color=INK2)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, frameon=False)
        rows.append({"label": p["label"], "row": p["r"], "col": p["c"], "azimuth": az, "lambda_km": lam,
                     "offset_m": o, "ambiguity": am})
    fig.suptitle("5. Displacement examples: side profiles at +-h1 parallel to the candidate line; agreement after the "
                 "shift = displaced continuation of fabric", x=0.01, ha="left", fontsize=11, color=INK)
    return fig, pd.DataFrame(rows)


# 6 -------------------------------------------------------------------------------------------------- truncation
def truncation_examples(grid, w, A: dict, lam_labels: list[str], truth, seed: int):
    import matplotlib.pyplot as plt

    from .geology_displacement import frame_vectors
    from .geology_truncation import skeleton_nodes

    pts = pick_points(A, "G02-B", 4, seed + 11)
    if not pts:
        return None, pd.DataFrame()
    fig, axes = plt.subplots(1, len(pts), figsize=(W16, H9 * 0.6), squeeze=False, constrained_layout=True)
    rows = []
    for ax, p in zip(axes[0], pts):
        lk = A.get("trunc[mag]:lambda_km")
        lam = float(lk[p["r"], p["c"]]) if lk is not None and np.isfinite(lk[p["r"], p["c"]]) else None
        lab = f"{lam:g}km" if lam is not None else (lam_labels[0] if lam_labels else "")
        E = A.get(f"scale:M@{lab}")
        R = A.get(f"ridge[mag]@{lab}")
        hw = int(60)
        r0, r1, c0, c1 = max(p["r"] - hw, 0), p["r"] + hw, max(p["c"] - hw, 0), p["c"] + hw
        if E is None or R is None:
            ax.axis("off")
            continue
        sub, rsub = np.asarray(E)[r0:r1, c0:c1], np.asarray(R)[r0:r1, c0:c1] > 0
        ext = _extent_km(grid, (w[0] + r0, w[0] + r0 + sub.shape[0], w[2] + c0, w[2] + c0 + sub.shape[1]))
        ax.imshow(sub, extent=ext, cmap="magma", vmin=0, vmax=1, origin="upper", interpolation="nearest")
        ax.imshow(np.where(rsub, 1.0, np.nan), extent=ext, cmap="Greys_r", vmin=0, vmax=1, origin="upper",
                  interpolation="nearest")
        ep, _ = skeleton_nodes(rsub)
        er, ec = np.nonzero(ep)
        if er.size:
            xs, ys = grid.rc_to_xy(w[0] + r0 + er, w[2] + c0 + ec)
            ax.scatter(np.asarray(xs) / 1000, np.asarray(ys) / 1000, s=8, c=SERIES[2], label="ridge endpoints")
        az = A.get("trunc[mag]:azimuth")
        a = float(az[p["r"], p["c"]]) if az is not None and np.isfinite(az[p["r"], p["c"]]) else 0.0
        (tE, tN), _ = frame_vectors(a)
        x0, y0 = grid.rc_to_xy(w[0] + p["r"], w[2] + p["c"])
        L = 2.0 * (lam or 1.0) * 1000
        ax.plot([(x0 - L * tE) / 1000, (x0 + L * tE) / 1000], [(y0 - L * tN) / 1000, (y0 + L * tN) / 1000],
                color="#33a6ff", lw=1.2, label="candidate line")
        _outline(ax, None if truth is None else np.asarray(truth)[r0:r1, c0:c1], ext, TRUTH_C)
        s = float(np.asarray(A["G02-B"])[p["r"], p["c"]])
        es = A.get("trunc[mag]:energy_step")
        e = float(es[p["r"], p["c"]]) if es is not None else np.nan
        verdict = "ridges END along the line (truncation-compatible)" if s > 0.3 else \
            "endpoints not aligned: no truncation evidence"
        ax.set_title(f"{p['label']}: G02-B={s:.2f}, energy step {e:.2f}\n{verdict}", fontsize=8, loc="left", color=INK)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.tick_params(labelsize=6)
        rows.append({"label": p["label"], "row": p["r"], "col": p["c"], "G02_B": s, "energy_step": e, "azimuth": a,
                     "lambda": lab})
    axes[0, 0].legend(fontsize=6, frameon=False, loc="lower left")
    fig.suptitle("6. Truncation examples: magnetic energy percentile, ridge skeleton (white), endpoints (dots), "
                 "candidate line (blue), truth (red). Also produced by intrusive / unconformable contacts", x=0.01,
                 ha="left", fontsize=11, color=INK)
    return fig, pd.DataFrame(rows)


# 7 / 8 -------------------------------------------------------------------------------------------- continuity
def continuity_maps(grid, w, A: dict, lam_labels: list[str], truth):
    panels = []
    for lam in lam_labels:
        e, r = A.get(f"scale:M@{lam}"), A.get(f"ridge[mag]@{lam}")
        if e is not None and r is not None:
            panels.append({"title": f"magnetic ridges at {lam}", "raster": e, "cmap": "Greys", "vmin": 0, "vmax": 1,
                           "sel": np.asarray(r) > 0})
    for k, t in (("cont:along", "along-strike ridge occupancy"), ("cont:straightness", "straightness"),
                 ("G02-C[narrow]", "G02-C narrow band"), ("G02-C[broad]", "G02-C broad band")):
        if k in A:
            panels.append({"title": t, "raster": A[k], "cmap": "viridis", "vmin": 0, "vmax": 1, "truth": truth})
    if not panels:
        return None, pd.DataFrame()
    return map_grid(grid, w, panels, ncols=min(4, len(panels)),
                    title="7. Structural-ridge continuity. Narrow and broad bands are scored separately so short "
                          "fine-scale faults are not averaged away")


def persistence_maps(grid, w, A: dict, truth):
    spec = [("cont:P_ridge", "ridge POSITION persistence across scales (G02)"),
            ("comp:P", "energy persistence (G01)"), ("cont:strike_stability", "strike stability across scales"),
            ("G02-C", "G02-C structural continuity")]
    panels = [{"title": t, "raster": A[k], "truth": truth} for k, t in spec if k in A]
    if not panels:
        return None, pd.DataFrame()
    return map_grid(grid, w, panels, ncols=len(panels), shared={"vmin": 0, "vmax": 1, "cmap": "viridis"},
                    cbar_label="[0, 1]", title="8. Multiscale lineament persistence. Deep-rooted structures persist; "
                                               "a single-scale fault is NOT contradicted by low persistence")


# 9 ------------------------------------------------------------------------------------------------- network
def network_map(grid, w, A: dict, store, key: str, source: str, truth, known):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    from .geology_io import _safe

    p = store.path(f"lineaments/{key}_{_safe(source)}.geojson")
    ext = _extent_km(grid, w)
    fig, ax = plt.subplots(1, 1, figsize=(W16, H9), constrained_layout=True)
    if "G02-D" in A:
        im = ax.imshow(A["G02-D"], extent=ext, cmap="Greys", vmin=0, vmax=max(float(np.nanmax(A["G02-D"])), 1e-6)
                       if np.isfinite(A["G02-D"]).any() else 1, origin="upper", interpolation="nearest", alpha=0.6)
        fig.colorbar(im, ax=ax, shrink=0.6, pad=0.01).set_label("G02-D score", fontsize=8)
    rows = []
    if p.exists():
        fc = json.loads(p.read_text(encoding="utf-8"))
        segs, cols, styles = [], [], []
        for ft in fc.get("features", []):
            xy = np.asarray(ft["geometry"]["coordinates"], float) / 1000.0
            if not ((xy[:, 0] >= ext[0]) & (xy[:, 0] <= ext[1]) & (xy[:, 1] >= ext[2]) & (xy[:, 1] <= ext[3])).any():
                continue
            pr = ft["properties"]
            segs.append(xy)
            cols.append(pr.get("score") or 0.0)
            styles.append("solid" if (pr.get("continuation") or 0) > 0 else "dashed")
            ax.scatter(xy[[0, -1], 0], xy[[0, -1], 1], s=5, c=INK2)
            rows.append({k: pr.get(k) for k in ("segment", "length_m", "straightness", "azimuth", "continuation",
                                                "continuation_kind", "regional_alignment", "score")})
        if segs:
            lc = LineCollection(segs, array=np.asarray(cols, float), cmap="plasma", linewidths=1.4,
                                linestyles=styles)
            ax.add_collection(lc)
            fig.colorbar(lc, ax=ax, shrink=0.6, pad=0.01).set_label("segment score", fontsize=8)
    _outline(ax, truth, ext, TRUTH_C, "held-out truth")
    _outline(ax, known, ext, KNOWN_C, "known faults")
    ax.plot([], [], color=INK2, ls="--", label="isolated segment (kept: connectivity is not forced)")
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (km, UTM 11N)", fontsize=8, color=INK2)
    ax.set_ylabel("Northing (km)", fontsize=8, color=INK2)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.set_title(f"9. Candidate geological network from {source} (solid = continuation support: colinear or stepover; "
                 "dots = segment ends)", fontsize=11, loc="left", color=INK)
    return fig, pd.DataFrame(rows)


# 10 ------------------------------------------------------------------------------------------------ G01 vs G02
def g01_vs_g02(grid, w, A: dict, b2: str, gf: str | None, truth, known, frac: float):
    dom = np.asarray(A.get("domain", np.ones(next(iter(A.values())).shape)), bool)
    spec = [("edge", "B0 simple magnetic gradient"), ("G01-A", "G01-A magnetic wavelet"), (b2, f"B2 ({b2})"),
            ("G02-A", "G02-A displacement"), ("G02-B", "G02-B truncation"), ("G02-C", "G02-C continuity"),
            ("G02-D", "G02-D network"), ("G02-E", "G02-E fault hypothesis"), (gf, f"G02-F ({gf})")]
    panels = []
    for k, t in spec:
        s = _fused(A, k) if k else None
        if s is None:
            continue
        panels.append({"title": t, "raster": _rank(s, dom), "sel": _sel(s, dom, frac), "truth": truth,
                       "known": known, "legend": not panels})
    if not panels:
        return None, pd.DataFrame()
    return map_grid(grid, w, panels, ncols=3, shared={"vmin": 0, "vmax": 1, "cmap": "Greys"},
                    cbar_label="window percentile rank (display only)",
                    title=f"10. G01 versus G02 maps, top {100 * frac:g} % of the window domain selected (blue). "
                          "Selections here are per window, for display; metrics use the whole unit")


# 11 ------------------------------------------------------------------------------------------------ case studies
def case_studies(grid, w, A: dict, b2: str, gf: str | None, truth, frac: float, tol: float, pixel_m, seed: int,
                 n: int = 3):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    from .geology_metrics import distance_m

    if truth is None or gf is None:
        return None, pd.DataFrame()
    dom = np.asarray(A.get("domain"), bool)
    sb, sf = _fused(A, b2), _fused(A, gf)
    if sb is None or sf is None:
        return None, pd.DataFrame()
    T = np.asarray(truth, bool)
    selb, self_ = _sel(sb, dom, frac), _sel(sf, dom, frac)
    dT, dB, dF = distance_m(T, pixel_m), distance_m(selb, pixel_m), distance_m(self_, pixel_m)
    eE = _f(A.get("G02-E"))
    eE = np.full(T.shape, np.nan, np.float32) if eE is None else eE
    cats = {"G02 adds a hit": self_ & (dT <= tol) & (dB > tol), "G02 adds a false positive": self_ & (dT > tol) & ~selb,
            "missed by both": T & (dB > tol) & (dF > tol),
            "strong G02-E evidence on truth": T & (np.nan_to_num(eE, nan=0.0) >= 0.5),
            "weak G02-E evidence on truth": T & (np.nan_to_num(eE, nan=1.0) < 0.1)}
    tile = 64
    rng = np.random.default_rng(seed + 23)
    picks = []
    for lab, m in cats.items():
        cnt = ndimage.uniform_filter(m.astype(np.float32), tile)[tile // 2::tile, tile // 2::tile] * tile * tile
        cand = np.argwhere(cnt >= 5)
        if len(cand):
            i = cand[rng.integers(len(cand))]
            picks.append((lab, int(i[0]) * tile, int(i[1]) * tile))
    picks = picks[:max(n, 1) + 2]
    if not picks:
        return None, pd.DataFrame()
    fig, axes = plt.subplots(len(picks), 3, figsize=(W16, H9), squeeze=False, constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1, 1.4]})
    rows = []
    outcome = np.zeros(T.shape, np.float32)
    outcome[self_ & (dT <= tol)] = 1          # hit
    outcome[self_ & (dT > tol)] = 2           # false positive
    outcome[T & (dF > tol)] = 3               # missed truth
    dmn = np.asarray(A.get("hyp:dominant", np.full(T.shape, -1)), np.float32)
    for i, (lab, r0, c0) in enumerate(picks):
        r1, c1 = min(r0 + tile, T.shape[0]), min(c0 + tile, T.shape[1])
        ext = _extent_km(grid, (w[0] + r0, w[0] + r1, w[2] + c0, w[2] + c1))
        ax = axes[i, 0]
        ax.imshow(_rank(sf, dom)[r0:r1, c0:c1], extent=ext, cmap="Greys", vmin=0, vmax=1, origin="upper")
        o = outcome[r0:r1, c0:c1]
        ax.imshow(np.where(o > 0, o, np.nan), extent=ext, cmap=ListedColormap(["#2ca02c", "#ff7f0e", TRUTH_C]),
                  vmin=0.5, vmax=3.5, origin="upper", interpolation="nearest")
        ax.set_title(f"{lab}: G02-F outcome (green hit, orange FP, red missed)", fontsize=7, loc="left", color=INK)
        ax = axes[i, 1]
        d = dmn[r0:r1, c0:c1]
        ax.imshow(np.where(d >= 0, d, np.nan), extent=ext, cmap=ListedColormap(HYP_COLORS), vmin=-0.5, vmax=3.5,
                  origin="upper", interpolation="nearest")
        _outline(ax, T[r0:r1, c0:c1], ext, TRUTH_C)
        ax.set_title("dominant hypothesis (red fault, yellow contact, teal intrusive, grey ambiguous)", fontsize=7,
                     loc="left", color=INK)
        stats = {k: float(np.nanmean(np.asarray(A[k], np.float32)[r0:r1, c0:c1])) if k in A else np.nan
                 for k in ("G02-A", "G02-B", "G02-C", "hyp:fault", "hyp:contact", "hyp:intrusive", "hyp:ambiguous",
                           "hyp:uncertainty")}
        share = {h: float(np.mean(d[d >= 0] == j)) if (d >= 0).any() else np.nan
                 for j, h in enumerate(("fault", "contact", "intrusive", "ambiguous"))}
        top = max(share, key=lambda k: -1 if not np.isfinite(share[k]) else share[k])
        s_ = stats
        text = [f"hits {int((o == 1).sum())}, false positives {int((o == 2).sum())}, missed {int((o == 3).sum())} px",
                f"mean displacement {s_['G02-A']:.2f}, truncation {s_['G02-B']:.2f}, continuity {s_['G02-C']:.2f}",
                f"supports: fault {s_['hyp:fault']:.2f}, contact {s_['hyp:contact']:.2f}, intrusive "
                f"{s_['hyp:intrusive']:.2f}, ambiguous {s_['hyp:ambiguous']:.2f}",
                f"dominant: {top} ({share[top]:.0%} of lineament pixels); uncertainty {stats['hyp:uncertainty']:.2f}",
                ("SUPPORTS a fault interpretation" if top == "fault" else
                 "CONTRADICTS a fault interpretation (better explained as " + top + ")" if top in ("contact", "intrusive")
                 else "evidence cannot tell (ambiguous)")]
        ax = axes[i, 2]
        ax.axis("off")
        ax.text(0.0, 0.95, "\n".join(text), va="top", ha="left", fontsize=8, color=INK, transform=ax.transAxes)
        for a in axes[i, :2]:
            a.set_aspect("equal")
            a.tick_params(labelsize=6)
        rows.append({"case": lab, "r0": r0 + w[0], "c0": c0 + w[2], **stats, **{f"share_{k}": v for k, v in share.items()},
                     "hits": int((o == 1).sum()), "false_pos": int((o == 2).sum()), "missed": int((o == 3).sum())})
    fig.suptitle(f"11. Case studies (seeded picks per category, NOT the best cases): {gf} vs {b2}", x=0.01, ha="left",
                 fontsize=11, color=INK)
    return fig, pd.DataFrame(rows)


# 12 / 13 -------------------------------------------------------------------------------------------- curves
def perf_curves(S: pd.DataFrame, x: str, fixed: float, heads: list[str], title: str):
    import matplotlib.pyplot as plt

    other = "area_frac" if x == "tol_m" else "tol_m"
    d = S[(S[other] == float(fixed)) & S.candidate.isin(heads)]
    if not len(d):
        return None, d
    tasks = sorted(d.task.unique())
    metrics = ["precision", "recall", "enrichment"]
    fig, axes = plt.subplots(len(tasks), 3, figsize=(W16, H9), squeeze=False, constrained_layout=True)
    for i, t in enumerate(tasks):
        g = d[d.task == t]
        for j, m in enumerate(metrics):
            ax = axes[i, j]
            for c, gc in g.groupby("candidate"):
                gc = gc.sort_values(x)
                ax.plot(gc[x], gc[m], marker="o", ms=3, lw=1.6 if c.startswith("G02") else 1.0,
                        ls="-" if c.startswith("G02") else "--", color=COLORS.get(c, INK2), label=c)
                if f"{m}_lo" in gc and gc[f"{m}_lo"].notna().any():
                    ax.fill_between(gc[x], gc[f"{m}_lo"], gc[f"{m}_hi"], color=COLORS.get(c, INK2), alpha=0.1)
            if m == "enrichment":
                ax.axhline(1.0, color=INK2, lw=0.8, ls=":")
            if x == "area_frac":
                ax.set_xscale("log")
            ax.set_title(f"{t}: {m}", fontsize=9, loc="left", color=INK)
            ax.set_xlabel("distance tolerance (m)" if x == "tol_m" else "area fraction of the domain", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25)
    axes[0, -1].legend(fontsize=7, frameon=False)
    fig.suptitle(title + " -- pooled held-out zones, 90 % block-bootstrap bands", x=0.01, ha="left", fontsize=11,
                 color=INK)
    return fig, d


# 14 ------------------------------------------------------------------------------------------------ per fold
def per_fold(D: pd.DataFrame, H: pd.DataFrame):
    import matplotlib.pyplot as plt

    keep = D[D.kind.isin(["incremental", "fusion", "baseline"])]
    if not len(keep):
        return None, keep
    tasks = sorted(keep.task.unique())
    fig, axes = plt.subplots(len(tasks), 1, figsize=(W16, H9), squeeze=False, constrained_layout=True)
    for i, t in enumerate(tasks):
        ax = axes[i, 0]
        g = keep[keep.task == t]
        labs = sorted(set(zip(g.candidate, g.reference)))
        for x, (c, r) in enumerate(labs):
            gg = g[(g.candidate == c) & (g.reference == r)]
            jit = np.linspace(-0.15, 0.15, max(len(gg), 1))
            ax.scatter(x + jit, gg.delta, s=14, color=COLORS.get(c.split("+")[-1] if "+" in c else c, INK2))
            if len(H):
                h = H[(H.task == t) & (H.candidate == c) & (H.reference == r)]
                if len(h) and "delta" in h and np.isfinite(h.delta.iloc[0]):
                    ax.errorbar(x, h.delta.iloc[0], yerr=[[h.delta.iloc[0] - h.ci_lo.iloc[0]],
                                                          [h.ci_hi.iloc[0] - h.delta.iloc[0]]],
                                fmt="_", color=INK, capsize=4, ms=14)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels([f"{c}\nvs {r}" for c, r in labs], fontsize=6, rotation=0)
        ax.set_ylabel(f"{keep.metric.iloc[0]} delta per held-out zone", fontsize=8)
        ax.set_title(f"{t}: dots = zones, bar = pooled delta with 90 % block-bootstrap interval", fontsize=9, loc="left",
                     color=INK)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2, axis="y")
    fig.suptitle("14. Per-geographic-fold improvements (a consistent gain has most dots above zero)", x=0.01, ha="left",
                 fontsize=11, color=INK)
    return fig, keep


# 15 ------------------------------------------------------------------------------------------------ hypotheses
def hypothesis_maps(grid, w, A: dict, truth):
    from matplotlib.colors import ListedColormap

    spec = [("hyp:fault", "fault support", "magma", 0, 1), ("hyp:contact", "lithological contact support", "magma", 0, 1),
            ("hyp:intrusive", "intrusive boundary support", "magma", 0, 1),
            ("hyp:terrain", "structurally controlled terrain", "magma", 0, 1),
            ("hyp:ambiguous", "ambiguous lineament", "magma", 0, 1),
            ("meas:artifact", "artefact flags (data edge, resolution seam)", "Greys", 0, 1),
            ("hyp:uncertainty", "uncertainty 1 - margin / top", "viridis", 0, 1)]
    panels = [{"title": t, "raster": A[k], "cmap": c, "vmin": lo, "vmax": hi, "truth": truth}
              for k, t, c, lo, hi in spec if k in A]
    if "hyp:dominant" in A:
        d = np.asarray(A["hyp:dominant"], np.float32)
        panels.insert(5, {"title": "dominant: fault / contact / intrusive / ambiguous", "raster": np.where(d >= 0, d, np.nan),
                          "cmap": ListedColormap(HYP_COLORS), "vmin": -0.5, "vmax": 3.5, "truth": truth})
    if not panels:
        return None, pd.DataFrame()
    return map_grid(grid, w, panels, ncols=4,
                    title="15. Competing geological explanations (hypothesis SCORES, not probabilities). Without "
                          "displacement or truncation a real fault is 'ambiguous': cannot tell, not 'no fault'")
