"""Lineament, orientation and discontinuity evidence (label-free).

Orientation is AXIAL (period 180 deg). It is never fed to a model as a raw angle: the doubled-angle pair
(cos 2 theta, sin 2 theta) is continuous across 0 / 180 deg. Agreement between two axial orientations uses

    A(theta_1, theta_2) = (1 + cos(2 (theta_1 - theta_2))) / 2        (1 = parallel, 0 = perpendicular)

Per field ``<f>`` and scale ``s<m>``::

    coh, ocos, osin, eigratio   structure tensor J = G_rho(grad S grad S^T): coherence ((l1-l2)/(l1+l2))^2, dominant
                                gradient axis (doubled angle), eigenvalue ratio l2 / l1 (rho = integration scale)
    ridge, valley               scale-normalised Hessian line strength: max(0, -h2) sigma^2 / max(0, h1) sigma^2
    along                       mean |grad| sampled ALONG strike / mean |grad| sampled ACROSS (> 1: the edge
                                continues along its own strike - a lineament, not a blob)
    cross                       |S(x + d n) - S(x - d n)|: field step across the local gradient direction (d = sigma)
    trunc                       |E+ - E-| / (E+ + E-), E+- = mean |grad| ahead / behind along strike: an edge that ends
                                abruptly (anomaly truncation PROXY - not a measured displacement)
    dircoh                      multiscale directional persistence: |sum_s coh_s exp(2 i theta_s)| / sum_s coh_s
Cross-modal::

    x_<a>__<b>_agree_s<m>       A(theta_a, theta_b);  ..._agreew_s<m> = A x sqrt(coh_a coh_b)
    disc_count_s<m>, disc_frac  number / share of fields with a thin edge within 1 px (co-located discontinuities)

Curvature operators are NOT displacement measurements; shifted-marker matching lives in ``g05_1.displacement`` and is
added only as an explicitly separate segment feature.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import feature_physics as FP

field_at_scale, gaussian_radius, hessian_eigen = FP.field_at_scale, FP.gaussian_radius, FP.hessian_eigen
masked_smooth, nms_edges, sigma_px = FP.masked_smooth, FP.nms_edges, FP.sigma_px


def axial_agreement(t1, t2) -> np.ndarray:
    """A(theta_1, theta_2) = (1 + cos(2 (theta_1 - theta_2))) / 2 for angles in radians."""
    return 0.5 * (1.0 + np.cos(2.0 * (np.asarray(t1, np.float64) - np.asarray(t2, np.float64))))


def structure_tensor(sr: np.ndarray, sc: np.ndarray, rho_px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(J_ee, J_en, J_nn): masked Gaussian integration of gradient products (east = +col, north = -row)."""
    ge, gn = sc, -sr
    jee, _ = masked_smooth(ge * ge, rho_px)
    jen, _ = masked_smooth(ge * gn, rho_px)
    jnn, _ = masked_smooth(gn * gn, rho_px)
    return jee, jen, jnn


def tensor_eigen(jee, jen, jnn) -> dict:
    tr = jee + jnn
    disc = np.sqrt(np.maximum((jee - jnn) ** 2 + 4.0 * jen * jen, 0.0))
    l1, l2 = 0.5 * (tr + disc), 0.5 * (tr - disc)
    with np.errstate(invalid="ignore", divide="ignore"):
        coh = np.where(tr > 0, (disc / tr) ** 2, 0.0)
        ratio = np.where(l1 > 0, l2 / l1, 1.0)
    theta = 0.5 * np.arctan2(2.0 * jen, jee - jnn)                # dominant gradient axis, from east, CCW
    return {"l1": l1, "l2": l2, "coh": coh, "eigratio": ratio, "theta": theta}


def structure_tensor_features(d: dict, rho_px: float) -> dict:
    """Structure-tensor descriptors from a ``field_at_scale`` dict."""
    jee, jen, jnn = structure_tensor(d["sr"], d["sc"], rho_px)
    e = tensor_eigen(jee, jen, jnn)
    return {"coh": e["coh"], "ocos": np.cos(2 * e["theta"]), "osin": np.sin(2 * e["theta"]),
            "eigratio": e["eigratio"], "theta": e["theta"]}


def directional_coherence(thetas, weights) -> np.ndarray:
    """Weighted mean resultant length of doubled angles across scales, in [0, 1]."""
    num_c = sum(w * np.cos(2 * t) for t, w in zip(thetas, weights))
    num_s = sum(w * np.sin(2 * t) for t, w in zip(thetas, weights))
    den = sum(weights)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, np.hypot(num_c, num_s) / den, np.nan)


def ridge_features(d: dict) -> dict:
    """Scale-normalised ridge (bright line) and valley (dark line) strength of the smoothed field."""
    h1, h2 = hessian_eigen(d)
    s2 = d["sigma_km"] ** 2
    return {"ridge": np.maximum(0.0, -h2) * s2, "valley": np.maximum(0.0, h1) * s2}


def _sample(a: np.ndarray, rr, cc, dr, dc) -> np.ndarray:
    return ndimage.map_coordinates(a, [rr + dr, cc + dc], order=1, mode="nearest")


def along_cross_features(d: dict, theta: np.ndarray, n_steps: int = 4) -> dict:
    """Along-strike continuity, cross-strike field step and truncation proxy at the scale of ``d``."""
    hgm, S = d["hgm"], d["S"]
    rr, cc = np.indices(hgm.shape, dtype=np.float64)
    nc, nr = np.cos(theta), -np.sin(theta)                       # gradient (across) direction, array coordinates
    tc, tr = -nr, nc                                             # strike (along) direction
    step = max(1.0, d["sigma_px"] / 2.0)
    fwd_t, bwd_t, acr = [], [], []
    for k in range(1, int(n_steps) + 1):
        s = k * step
        fwd_t.append(_sample(hgm, rr, cc, s * tr, s * tc))
        bwd_t.append(_sample(hgm, rr, cc, -s * tr, -s * tc))
        acr.append(_sample(hgm, rr, cc, s * nr, s * nc))
        acr.append(_sample(hgm, rr, cc, -s * nr, -s * nc))
    import warnings

    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                  # all-NaN samples near the footprint edge
        Ep, Em = np.nanmean(np.stack(fwd_t), 0), np.nanmean(np.stack(bwd_t), 0)
        Ea = np.nanmean(np.stack(acr), 0)
        along = (0.5 * (Ep + Em) + 1e-9) / (Ea + 1e-9)
        trunc = np.abs(Ep - Em) / (Ep + Em + 1e-9)
    dd = max(1.0, float(round(d["sigma_px"])))
    cross = np.abs(_sample(S, rr, cc, dd * nr, dd * nc) - _sample(S, rr, cc, -dd * nr, -dd * nc))
    return {"along": along, "cross": cross, "trunc": trunc}


def cross_modal_orientation(theta_a, coh_a, theta_b, coh_b) -> dict:
    agree = axial_agreement(theta_a, theta_b)
    return {"agree": agree, "agreew": agree * np.sqrt(np.maximum(coh_a, 0) * np.maximum(coh_b, 0))}


def discontinuity_features(edge_maps: list) -> dict:
    """Count / share of fields with a thin edge within 1 px (NaN edges = field missing there)."""
    cnt = np.zeros(edge_maps[0].shape)
    nval = np.zeros(edge_maps[0].shape)
    for e in edge_maps:
        ok = np.isfinite(e)
        near = ndimage.maximum_filter(np.where(ok, e, 0.0), 3, mode="constant") > 0
        cnt += (near & ok)
        nval += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"count": np.where(nval > 0, cnt, np.nan), "frac": np.where(nval > 0, cnt / np.maximum(nval, 1), np.nan)}


# =============================================================================== names + bank
def rho_m(scale_m: float, cfg: dict) -> float:
    return min(float(cfg.get("integration_factor", 1.5)) * float(scale_m), float(cfg.get("max_integration_m", 1500)))


def structure_names(cfg: dict) -> list[str]:
    out = []
    for f in cfg["fields"]:
        for s in cfg["scales_m"]:
            out += [f"struct_{f}_{k}_s{int(s)}" for k in ("coh", "ocos", "osin", "eigratio", "ridge", "valley")]
        if len(cfg["scales_m"]) > 1:
            out.append(f"struct_{f}_dircoh")
    for f in cfg.get("continuity_fields", []):
        for s in cfg["scales_m"]:
            out += [f"struct_{f}_{k}_s{int(s)}" for k in ("along", "cross", "trunc")]
    for a, b in cfg.get("cross_modal_pairs", []):
        for s in cfg["scales_m"]:
            out += [f"struct_x_{a}__{b}_agree_s{int(s)}", f"struct_x_{a}__{b}_agreew_s{int(s)}"]
    if cfg.get("discontinuity_fields"):
        s = int(cfg.get("discontinuity_scale_m", 300))
        out += [f"struct_disc_count_s{s}", f"struct_disc_frac_s{s}"]
    return out


def required_halo_px(cfg: dict) -> int:
    h = 0
    for s in cfg["scales_m"]:
        base = gaussian_radius(sigma_px(s)) + 2
        h = max(h, base + gaussian_radius(sigma_px(rho_m(s, cfg))),
                base + int(np.ceil(4 * max(1.0, sigma_px(s) / 2.0))) + 2)
    return h + 2


def structural_feature_bank(fields: dict, cfg: dict, cache: dict | None = None, px_km: float = 0.1) -> dict:
    """Every feature of ``structure_names(cfg)``. ``cache`` {(field, scale_m): field_at_scale dict} is reused when
    the physics bank already computed that field / scale (and filled for the scales computed here)."""
    cache = {} if cache is None else cache

    def at(f, s):
        k = (f, float(s))
        if k not in cache:
            cache[k] = field_at_scale(fields[f], s, px_km)
        return cache[k]

    out, theta, coh = {}, {}, {}
    all_f = list(dict.fromkeys(list(cfg["fields"]) + list(cfg.get("continuity_fields", [])) +
                               [x for p in cfg.get("cross_modal_pairs", []) for x in p]))
    for f in all_f:
        valid = np.isfinite(fields[f])
        for s in cfg["scales_m"]:
            d = at(f, s)
            st = structure_tensor_features(d, sigma_px(rho_m(s, cfg)))
            theta[(f, s)], coh[(f, s)] = st["theta"], st["coh"]
            if f in cfg["fields"]:
                for k in ("coh", "ocos", "osin", "eigratio"):
                    out[f"struct_{f}_{k}_s{int(s)}"] = np.where(valid, st[k], np.nan)
                rd = ridge_features(d)
                out[f"struct_{f}_ridge_s{int(s)}"] = np.where(valid, rd["ridge"], np.nan)
                out[f"struct_{f}_valley_s{int(s)}"] = np.where(valid, rd["valley"], np.nan)
            if f in cfg.get("continuity_fields", []):
                ac = along_cross_features(d, st["theta"], int(cfg.get("continuity_steps", 4)))
                for k in ("along", "cross", "trunc"):
                    out[f"struct_{f}_{k}_s{int(s)}"] = np.where(valid, ac[k], np.nan)
        if f in cfg["fields"] and len(cfg["scales_m"]) > 1:
            out[f"struct_{f}_dircoh"] = np.where(valid, directional_coherence(
                [theta[(f, s)] for s in cfg["scales_m"]], [coh[(f, s)] for s in cfg["scales_m"]]), np.nan)
    for a, b in cfg.get("cross_modal_pairs", []):
        valid = np.isfinite(fields[a]) & np.isfinite(fields[b])
        for s in cfg["scales_m"]:
            x = cross_modal_orientation(theta[(a, s)], coh[(a, s)], theta[(b, s)], coh[(b, s)])
            out[f"struct_x_{a}__{b}_agree_s{int(s)}"] = np.where(valid, x["agree"], np.nan)
            out[f"struct_x_{a}__{b}_agreew_s{int(s)}"] = np.where(valid, x["agreew"], np.nan)
    if cfg.get("discontinuity_fields"):
        s = float(cfg.get("discontinuity_scale_m", 300))
        edges = []
        for f in cfg["discontinuity_fields"]:
            d = at(f, s)
            edges.append(nms_edges(d["hgm"], d["sr"], d["sc"]))
        dc = discontinuity_features(edges)
        any_valid = np.any(np.stack([np.isfinite(fields[f]) for f in cfg["discontinuity_fields"]]), 0)
        out[f"struct_disc_count_s{int(s)}"] = np.where(any_valid, dc["count"], np.nan)
        out[f"struct_disc_frac_s{int(s)}"] = np.where(any_valid, dc["frac"], np.nan)
    return {k: np.asarray(v, np.float32) for k, v in out.items()}
