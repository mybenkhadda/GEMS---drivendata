"""G01 geological wavelet evidence: bounded, label-free components and the six candidate maps.

Inputs are the M9 section-03 steerable log-Gabor layers (``m9:spec:<family>_wav_<kind>@<lam>km``) of the magnetic
(RTP) and gravity (isostatic residual) fields, read from the M9 layer cache, never recomputed here:

    energy      robust-standardised log1p(E / median E): a MONOTONE transform of the quadrature energy, so its
                percentile (below) does not depend on the M9 standardisation
    coherence   directional coherence |sum_o A_o e^{2i theta_o}| / sum_o A_o in [0, 1]
    strike_c2   cos(2 strike) * coherence      (doubled angle: 0 / 180 deg continuous; coherence ALREADY applied)
    strike_s2   sin(2 strike) * coherence

Components (float32, [0, 1], NaN = no evidence at that pixel):

    M_lam, G_lam   percentile of the wavelet energy against a REFERENCE distribution (the unit's training geography
                   for held-out evaluation; every valid pixel for exploratory maps). The percentile table is the only
                   fitted quantity in G01 and it never sees labels.
    J_lam          spatially tolerant co-location max(sqrt(M T[G]), sqrt(G T[M])): T[x] = max_y x(y) w(|x - y|), the
                   distance-discounted max with w(r) = exp(-r^2 / (2 tol^2)), tol = clip(c * lambda, min, max) metres.
                   A quarter wavelength is the lateral uncertainty of an edge seen at wavelength lambda; gravity
                   sources are deeper and the survey coarser, so the gravity maximum may sit beside the magnetic one.
                   Defined only where BOTH fields are observed (NaN otherwise: never a penalty for a missing field).
    gate_lam       orientation compatibility 1 - kappa (1 - A). A = (1 + cos 2(theta_m - theta_g)) / 2 is computed from
                   UNIT doubled-angle vectors (axial, 180 deg periodic); kappa = sqrt(r_m r_g) is the coherence of the
                   tolerance-smoothed doubled-angle fields. The cached vectors already carry the coherence, so it
                   enters exactly once (through kappa); A itself is coherence-free. kappa = 0 (no orientation) gives
                   gate = 1: undefined orientation never penalises.
    P_mag, P_grav  persistence: share of scales whose percentile >= q (per field)
    P_cross        share of scales with J_lam >= q (both fields, tolerant)
    P              max(P_mag, P_grav): a structure persistent in ONE field is not penalised for the other.

Candidates (fixed weights, nothing learned; each one adds ONE ingredient to the previous):

    G01-A   M = mean_lam M_lam                               magnetic structure alone
    G01-B   G = mean_lam G_lam                               gravity structure alone
    G01-C   (M + G + J) / 3,  J = mean_lam J_lam            joint, spatially tolerant evidence
    G01-D   (M + G + J_D) / 3,  J_D = mean_lam J_lam gate   + axial strike compatibility (J_D <= J)
    G01-E   D * (floor + (1 - floor) P)                      + multiscale persistence
    G01-F   mean of percentiles of M9's engineered layers (m9:xmod:mag_grav_tol@*, mag_grav_orient@*)

Means are NaN-aware (a missing component is dropped and the others renormalised). High wavelet energy is NOT proof
of faulting: lithological contacts, intrusion margins and dykes give the same signal. The maps are evidence scores,
never calibrated fault probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

KINDS = ("energy", "coherence", "strike_c2", "strike_s2")
ROLES = ("mag", "grav")
CANDIDATES = {
    "G01-A": "magnetic wavelet energy (multiscale percentile)",
    "G01-B": "gravity wavelet energy (multiscale percentile)",
    "G01-C": "magnetic + gravity + tolerant co-location",
    "G01-D": "C with axial strike compatibility gating the co-location term",
    "G01-E": "D modulated by multiscale persistence",
    "G01-F": "M9 engineered cross-modal layers (mag_grav_tol / mag_grav_orient percentiles)",
}
BASELINES = {"edge": "simple gradient edge (THDR percentile of the raw band)",
             "random": "seeded uniform random score (enrichment 1 by construction)"}


# =============================================================================== naming
@dataclass(frozen=True)
class Wavelength:
    """One scale shared by both fields; ``tags`` = the wavelength string of each field's cached layer names."""
    lam_km: float
    tags: tuple = field(default=())        # ((role, tag), ...) hashable

    def tag(self, role: str) -> str | None:
        return dict(self.tags).get(role)

    @property
    def label(self) -> str:
        return f"{self.lam_km:g}km"


def channel(family: str, kind: str, tag: str) -> str:
    """M9 layer name of one cached wavelet product (``spectral.build``: ``m9:spec:<fam>_wav_<kind>@<lam:g>km``)."""
    return f"m9:spec:{family}_wav_{kind}@{tag}km"


# =============================================================================== geometry helpers
def meters_to_px(dist_m: float, pixel_m) -> tuple[int, int]:
    """(rows, cols) half-widths covering ``dist_m`` metres on a grid of (row, col) pixel sizes ``pixel_m`` (metres).
    Floor: a pixel is inside when its centre offset is <= dist_m (no hard-coded pixel size)."""
    py, px = float(pixel_m[0]), float(pixel_m[1])
    if py <= 0 or px <= 0:
        raise ValueError(f"pixel size must be positive, got {pixel_m}")
    return int(np.floor(float(dist_m) / py + 1e-6)), int(np.floor(float(dist_m) / px + 1e-6))


def sigma_px(dist_m: float, pixel_m) -> tuple[float, float]:
    return max(float(dist_m) / float(pixel_m[0]), 0.0), max(float(dist_m) / float(pixel_m[1]), 0.0)


def tolerance_m(lam_km: float, tc: dict) -> float:
    """Lateral tolerance of one scale: clip(per_wavelength * lambda, min_m, max_m) metres."""
    return float(np.clip(float(tc["per_wavelength"]) * float(lam_km) * 1000.0, float(tc["min_m"]), float(tc["max_m"])))


def _ellipse(ry: int, rx: int, pixel_m) -> np.ndarray:
    y, x = np.ogrid[-ry:ry + 1, -rx:rx + 1]
    d = np.hypot(y * float(pixel_m[0]), x * float(pixel_m[1]))
    return d <= max(ry * float(pixel_m[0]), rx * float(pixel_m[1])) + 1e-6


def soft_tolerant_max(a: np.ndarray, tol_m: float, pixel_m, n_rings: int = 4, reach: float = 2.0,
                      footprint: str = "square") -> np.ndarray:
    """T[a](x) = max_y a(y) w(|x - y|), w(r) = exp(-r^2 / (2 tol^2)), searched to ``reach * tol``.

    Stepwise on ``n_rings`` nested neighbourhoods: a source at distance d in (r_{k-1}, r_k] counts with w(r_k) (never
    more than its true weight). ``square`` uses separable max filters (Chebyshev neighbourhoods, fast on the full
    grid); ``disk`` uses exact Euclidean footprints. NaN counts as no evidence (0). ``a`` must be >= 0."""
    a0 = np.where(np.isfinite(a), a, 0.0).astype(np.float32)
    if tol_m <= 0:
        return a0
    out = a0.copy()
    for k in range(1, int(n_rings) + 1):
        r = float(reach) * float(tol_m) * k / int(n_rings)
        ry, rx = meters_to_px(r, pixel_m)
        if ry == 0 and rx == 0:
            continue
        w = np.float32(np.exp(-0.5 * (r / float(tol_m)) ** 2))
        if footprint == "disk":
            mx = ndimage.maximum_filter(a0, footprint=_ellipse(ry, rx, pixel_m), mode="constant", cval=0.0)
        else:
            mx = ndimage.maximum_filter(a0, size=(2 * ry + 1, 2 * rx + 1), mode="constant", cval=0.0)
        np.maximum(out, w * mx, out=out)
    return out


def masked_gaussian(a: np.ndarray, sig_px, valid: np.ndarray | None = None) -> np.ndarray:
    """Normalised (NaN-aware) Gaussian smoothing: boundaries do not shrink the mean."""
    m = np.isfinite(a) if valid is None else (np.asarray(valid, bool) & np.isfinite(a))
    if not np.any(np.asarray(sig_px, float) > 0):
        return np.where(m, a, np.nan).astype(np.float32)
    num = ndimage.gaussian_filter(np.where(m, a, 0.0).astype(np.float32), sig_px, mode="constant")
    den = ndimage.gaussian_filter(m.astype(np.float32), sig_px, mode="constant")
    return np.where(den > 1e-3, num / np.maximum(den, 1e-6), np.nan).astype(np.float32)


# =============================================================================== orientation
def axial_agreement(theta_a_deg, theta_b_deg) -> np.ndarray:
    """A = (1 + cos 2(theta_a - theta_b)) / 2: 1 parallel, 0 perpendicular, invariant to +-180 deg on either angle."""
    d = np.radians(np.asarray(theta_a_deg, np.float64) - np.asarray(theta_b_deg, np.float64))
    return (0.5 * (1.0 + np.cos(2.0 * d))).astype(np.float32)


def doubled_from_strike(theta_deg, weight=1.0) -> tuple[np.ndarray, np.ndarray]:
    t = np.radians(np.asarray(theta_deg, np.float64))
    w = np.asarray(weight, np.float64)
    return (np.cos(2 * t) * w).astype(np.float32), (np.sin(2 * t) * w).astype(np.float32)


def strike_from_doubled(c2, s2) -> np.ndarray:
    """Axial strike in [0, 180) from (possibly weighted) doubled-angle components."""
    return (np.degrees(0.5 * np.arctan2(np.asarray(s2, np.float64), np.asarray(c2, np.float64))) % 180.0).astype(np.float32)


def unit_doubled(c2, s2, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(cos 2t, sin 2t, r): unit vector + its length r = the coherence carried by the (weighted) vector."""
    c2 = np.asarray(c2, np.float32)
    s2 = np.asarray(s2, np.float32)
    r = np.hypot(c2, s2)
    ok = np.isfinite(r) & (r > eps)
    safe = np.where(ok, r, 1.0)
    cu = np.where(ok, c2 / safe, np.nan).astype(np.float32)
    su = np.where(ok, s2 / safe, np.nan).astype(np.float32)
    return cu, su, np.where(np.isfinite(r), np.minimum(r, 1.0), np.nan).astype(np.float32)


def doubled_agreement(cu_a, su_a, cu_b, su_b) -> np.ndarray:
    """Axial agreement of two UNIT doubled-angle vectors: (1 + cos 2(t_a - t_b)) / 2 = (1 + c_a c_b + s_a s_b) / 2."""
    return np.clip(0.5 * (1.0 + cu_a * cu_b + su_a * su_b), 0.0, 1.0).astype(np.float32)


def orientation_gate(c2m, s2m, c2g, s2g, sig_px=(0.0, 0.0)) -> dict:
    """Orientation compatibility of magnetic and gravity strike (coherence-weighted M9 vectors).

    Both weighted vector fields are smoothed at the tolerance scale (a gravity lineament displaced beside the
    magnetic one still contributes its orientation); unit vectors give the coherence-free agreement A; the smoothed
    lengths r_m, r_g (<= local coherence) give the confidence kappa = sqrt(r_m r_g). gate = 1 - kappa (1 - A)."""
    vec = []
    for c, s in ((c2m, s2m), (c2g, s2g)):
        c = masked_gaussian(np.asarray(c, np.float32), sig_px)
        s = masked_gaussian(np.asarray(s, np.float32), sig_px)
        vec.append(unit_doubled(c, s))
    (cm, sm, rm), (cg, sg, rg) = vec
    A = doubled_agreement(cm, sm, cg, sg)
    kappa = np.sqrt(np.nan_to_num(rm) * np.nan_to_num(rg)).astype(np.float32)
    gate = (1.0 - kappa * (1.0 - np.where(np.isfinite(A), A, 1.0))).astype(np.float32)
    return {"A": A, "kappa": kappa, "gate": gate}


# =============================================================================== percentile reference
@dataclass
class QuantileRef:
    """Empirical CDF (quantile table) of one channel on reference pixels; maps values to percentiles in [0, 1]."""
    q: np.ndarray
    p: np.ndarray
    n: int

    @classmethod
    def fit(cls, values: np.ndarray, n_q: int = 1001) -> "QuantileRef":
        v = np.asarray(values, np.float64).ravel()
        v = v[np.isfinite(v)]
        if v.size < 10:
            return cls(np.empty(0), np.empty(0), 0)
        probs = np.linspace(0.0, 1.0, int(n_q))
        qs = np.quantile(v, probs)
        u, inv = np.unique(qs, return_inverse=True)            # ties (flat / discrete channels) -> mean probability
        p = np.bincount(inv.ravel(), weights=probs) / np.bincount(inv.ravel())
        return cls(u.astype(np.float64), p.astype(np.float64), int(v.size))

    def __call__(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, np.float32)
        out = np.full(a.shape, np.nan, np.float32)
        if not self.n:
            return out
        m = np.isfinite(a)
        out[m] = np.interp(a[m].astype(np.float64), self.q, self.p).astype(np.float32)
        return out

    def to_dict(self) -> dict:
        return {"q": self.q.tolist(), "p": self.p.tolist(), "n": int(self.n)}

    @classmethod
    def from_dict(cls, d: dict) -> "QuantileRef":
        return cls(np.asarray(d["q"], np.float64), np.asarray(d["p"], np.float64), int(d["n"]))


# =============================================================================== aggregation helpers
def nanmean_stack(arrs: list) -> np.ndarray | None:
    """Pixel-wise mean over the finite members (NaN where none is finite); None when the list is empty."""
    arrs = [np.asarray(a, np.float32) for a in arrs if a is not None]
    if not arrs:
        return None
    s = np.zeros(arrs[0].shape, np.float32)
    n = np.zeros(arrs[0].shape, np.int16)
    for a in arrs:
        f = np.isfinite(a)
        s += np.where(f, a, 0.0)
        n += f
    return np.where(n > 0, s / np.maximum(n, 1), np.nan).astype(np.float32)


def combine(parts: list, weights: list | None = None) -> np.ndarray | None:
    """Weighted NaN-aware mean of bounded components (missing components dropped, weights renormalised)."""
    pw = [(np.asarray(a, np.float32), float(w)) for a, w in zip(parts, weights or [1.0] * len(parts)) if a is not None]
    if not pw:
        return None
    s = np.zeros(pw[0][0].shape, np.float32)
    n = np.zeros(pw[0][0].shape, np.float32)
    for a, w in pw:
        f = np.isfinite(a)
        s += np.where(f, a * w, 0.0)
        n += f * w
    return np.where(n > 0, s / np.maximum(n, 1e-9), np.nan).astype(np.float32)


def persistence(stack: list, q: float) -> np.ndarray | None:
    """Share of the finite scales whose value is >= q (NaN where no scale is finite)."""
    stack = [np.asarray(a, np.float32) for a in stack if a is not None]
    if not stack:
        return None
    hits = np.zeros(stack[0].shape, np.int16)
    n = np.zeros(stack[0].shape, np.int16)
    for a in stack:
        f = np.isfinite(a)
        hits += f & (np.where(f, a, -np.inf) >= q)
        n += f
    return np.where(n > 0, hits / np.maximum(n, 1), np.nan).astype(np.float32)


def tolerant_agreement(M: np.ndarray, G: np.ndarray, tol_m: float, pixel_m, tc: dict) -> np.ndarray:
    """J = max(sqrt(M T[G]), sqrt(G T[M])): high where both fields carry structure within the tolerance; anchored on
    whichever field has the ridge locally. NaN where either field is unobserved (never a penalty)."""
    kw = {"n_rings": int(tc.get("n_rings", 4)), "reach": float(tc.get("reach", 2.0)),
          "footprint": str(tc.get("footprint", "square"))}
    tg = soft_tolerant_max(G, tol_m, pixel_m, **kw)
    tm = soft_tolerant_max(M, tol_m, pixel_m, **kw)
    both = np.isfinite(M) & np.isfinite(G)
    j = np.maximum(np.sqrt(np.nan_to_num(M) * tg), np.sqrt(np.nan_to_num(G) * tm))
    return np.where(both, j, np.nan).astype(np.float32)


# =============================================================================== evidence builder
def build_evidence(win: dict, refs: dict, lams: list, cfg: dict, pixel_m, *, keep_per_scale: bool = False,
                   ablations: bool = True) -> dict:
    """Every G01 component and candidate on one window (label-free; no label argument exists).

    ``win``: {channel name: (h, w) array} of the AVAILABLE channels (missing ones simply absent), plus optional
    ``"g01:edge"`` / ``"g01:surface"`` arrays and the ``m9:xmod:*`` layers of G01-F. ``refs``: {channel name:
    QuantileRef} fitted on the reference geography. ``lams``: aligned ``Wavelength`` list."""
    fam = cfg["fields"]
    tc, oc, pc = cfg["tolerance"], cfg["orientation"], cfg["persistence"]
    shape = next(iter(win.values())).shape if win else (0, 0)
    nan = np.full(shape, np.nan, np.float32)

    def pct(name):
        if name not in win or name not in refs:
            return None
        return refs[name](win[name])

    def raw(name):
        return np.asarray(win[name], np.float32) if name in win else None

    out, per = {}, []
    for w in lams:
        tags = {r: w.tag(r) for r in ROLES}
        M = pct(channel(fam["mag"], "energy", tags["mag"])) if tags["mag"] else None
        G = pct(channel(fam["grav"], "energy", tags["grav"])) if tags["grav"] else None
        tol = tolerance_m(w.lam_km, tc)
        J = tolerant_agreement(M, G, tol, pixel_m, tc) if (M is not None and G is not None) else None
        sv = [raw(channel(fam[r], k, tags[r])) if tags[r] else None for r in ROLES for k in ("strike_c2", "strike_s2")]
        if all(s is not None for s in sv):
            o = orientation_gate(sv[0], sv[1], sv[2], sv[3], sigma_px(float(oc["smooth_frac_of_tol"]) * tol, pixel_m))
        else:
            o = {"A": None, "kappa": None, "gate": None}
        JD = J * o["gate"] if (J is not None and o["gate"] is not None) else J
        per.append({"w": w, "tol_m": tol, "M": M, "G": G, "J": J, "JD": JD, **o})
        if ablations and cfg["ablations"].get("per_wavelength", True):
            lab = w.label
            if M is not None:
                out[f"G01-A@{lab}"] = M
            if G is not None:
                out[f"G01-B@{lab}"] = G
            c = combine([M, G, J])
            if c is not None:
                out[f"G01-C@{lab}"] = c
            d = combine([M, G, JD])
            if d is not None and o["gate"] is not None:
                out[f"G01-D@{lab}"] = d

    M = nanmean_stack([p["M"] for p in per])
    G = nanmean_stack([p["G"] for p in per])
    J = nanmean_stack([p["J"] for p in per])
    JD = nanmean_stack([p["JD"] for p in per])
    q = float(pc["q"])
    Pm = persistence([p["M"] for p in per], q)
    Pg = persistence([p["G"] for p in per], q)
    Px = persistence([p["J"] for p in per], q)
    P = np.fmax(Pm, Pg) if (Pm is not None and Pg is not None) else (Pm if Pm is not None else Pg)
    comps = {"M": M, "G": G, "J": J, "J_D": JD, "P_mag": Pm, "P_grav": Pg, "P_cross": Px, "P": P,
             "orient_agreement": nanmean_stack([p["A"] for p in per]),
             "orient_confidence": nanmean_stack([p["kappa"] for p in per]),
             "orient_gate": nanmean_stack([p["gate"] for p in per])}
    if "g01:surface" in win and "g01:surface" in refs:
        comps["S"] = refs["g01:surface"](win["g01:surface"])
    out.update({f"comp:{k}": v for k, v in comps.items() if v is not None})

    cand = {"G01-A": M, "G01-B": G, "G01-C": combine([M, G, J]), "G01-D": combine([M, G, JD])}
    D = cand["G01-D"]
    if D is not None:
        fl = float(pc["floor"])
        cand["G01-E"] = (D * (fl + (1.0 - fl) * np.nan_to_num(P if P is not None else nan, nan=0.0))).astype(np.float32)
    xm = [refs[n](win[n]) for n in sorted(win) if n.startswith("m9:xmod:") and n in refs]
    cand["G01-F"] = nanmean_stack(xm)
    if "g01:edge" in win and "g01:edge" in refs:
        cand["edge"] = refs["g01:edge"](win["g01:edge"])
    want = set(cfg["candidates"]) | {"edge"}
    out.update({k: v for k, v in cand.items() if v is not None and k in want})

    if ablations and cfg["ablations"].get("tolerance", True) and per:
        for t in tc.get("ablation_m") or []:
            js = [tolerant_agreement(p["M"], p["G"], float(t), pixel_m, tc) for p in per
                  if p["M"] is not None and p["G"] is not None]
            if js:
                out[f"G01-C|tol={float(t):g}m"] = combine([M, G, nanmean_stack(js)])
    if keep_per_scale:
        for p in per:
            lab = p["w"].label
            for k in ("M", "G", "J", "gate", "A", "kappa"):
                if p[k] is not None:
                    out[f"scale:{k}@{lab}"] = p[k]
    return out


def is_candidate_key(k: str) -> bool:
    return not k.startswith(("comp:", "scale:"))


def tolerance_table(lams: list, cfg: dict, pixel_m) -> list[dict]:
    """Per scale: tolerance and orientation smoothing in metres AND in pixels of the actual grid (provenance)."""
    tc, oc = cfg["tolerance"], cfg["orientation"]
    rows = []
    for w in lams:
        t = tolerance_m(w.lam_km, tc)
        ry, rx = meters_to_px(float(tc.get("reach", 2.0)) * t, pixel_m)
        sy, sx = sigma_px(float(oc["smooth_frac_of_tol"]) * t, pixel_m)
        rows.append({"wavelength_km": w.lam_km, "tags": dict(w.tags), "tol_m": t,
                     "search_m": float(tc.get("reach", 2.0)) * t,
                     "search_px_rows": ry, "search_px_cols": rx, "orient_sigma_m": float(oc["smooth_frac_of_tol"]) * t,
                     "orient_sigma_px_rows": sy, "orient_sigma_px_cols": sx,
                     "pixel_m_rows": float(pixel_m[0]), "pixel_m_cols": float(pixel_m[1])})
    return rows


def halo_m(lams: list, cfg: dict) -> float:
    """Context (metres) a window needs around its core so tolerant filters and smoothing see real neighbours."""
    tc, oc = cfg["tolerance"], cfg["orientation"]
    tols = [tolerance_m(w.lam_km, tc) for w in lams] + [float(t) for t in tc.get("ablation_m") or []] + [0.0]
    t = max(tols)
    return max(float(tc.get("reach", 2.0)) * t, 3.0 * float(oc["smooth_frac_of_tol"]) * t,
               float(cfg.get("boundary_margin_m", 0.0))) + 200.0
