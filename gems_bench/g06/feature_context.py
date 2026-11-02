"""Geological context evidence.

Two kinds, kept strictly apart:

* REGIONAL, label-free (stored in the feature store): regional magnetic / gravity texture, terrain roughness,
  seismicity density, geodetic strain-rate indicators (asinh-compressed), basement depth and relief, distance to the
  data footprint edge, and an unsupervised geological-domain partition (k-means on regional features, fitted on a
  footprint sample - label-free but transductive; flagged ``fit_scope=full_footprint_unsupervised``).

* PERMITTED KNOWN-FAULT context (never stored globally): distance to the nearest visible fault, local fault density,
  distance to / alignment with the nearest trace tip, orientation of the nearest visible trace and network density.
  It is reconstructed PER VALIDATION UNIT from the faults visible to that unit only (``p6.stack.permitted_known`` for
  geo folds / discovery splits, or an explicit visibility mask for nested pairs). A full historical fault map is never
  an input of a unit that withholds some of those faults. These columns are permitted for segment / connection /
  meta models (cross-unit training on other units' truth); they are FORBIDDEN for pixel models trained on the visible
  faults themselves (the distance to a visible fault would be the target).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .feature_physics import field_at_scale, masked_smooth, sigma_px

KNOWN_COLUMNS = ("known_dist_km", "known_density", "tip_dist_km", "tip_along_cos", "known_orient_cos2",
                 "known_orient_sin2", "known_net_density")


# =============================================================================== regional (tiled)
def context_names(cfg: dict) -> list[str]:
    out = [f"ctx_{f}_regtex" for f in cfg.get("texture_fields", [])]
    if cfg.get("roughness_field"):
        out.append(f"ctx_{cfg['roughness_field']}_rough")
    if cfg.get("seismicity"):
        out.append("ctx_seis_log")
    out += [f"ctx_{s}_asinh" for s in cfg.get("strain", [])]
    if cfg.get("basement"):
        out += [f"ctx_{cfg['basement']}_reg", f"ctx_{cfg['basement']}_relief"]
    return out


def required_halo_px(cfg: dict) -> int:
    from .feature_physics import gaussian_radius

    ss = [float(cfg.get("regional_sigma_m", 1500)), float(cfg.get("roughness_sigma_m", 700)),
          float(cfg.get("seismicity_sigma_m", 700))]
    return max(gaussian_radius(sigma_px(s)) for s in ss) + 4


def masked_std(a: np.ndarray, sigma: float) -> np.ndarray:
    m1, _ = masked_smooth(a, sigma)
    m2, _ = masked_smooth(np.asarray(a, np.float64) ** 2, sigma)
    return np.sqrt(np.maximum(m2 - m1 * m1, 0.0))


def regional_context_features(fields: dict, cfg: dict, scales: dict, px_km: float = 0.1) -> dict:
    """Tile-level regional context; ``scales`` {band: footprint median |x|} for the asinh strain compression."""
    out = {}
    sr = sigma_px(float(cfg.get("regional_sigma_m", 1500)))
    for f in cfg.get("texture_fields", []):
        out[f"ctx_{f}_regtex"] = np.where(np.isfinite(fields[f]), masked_std(fields[f], sr), np.nan)
    rf = cfg.get("roughness_field")
    if rf:
        out[f"ctx_{rf}_rough"] = np.where(np.isfinite(fields[rf]),
                                          masked_std(fields[rf], sigma_px(float(cfg.get("roughness_sigma_m", 700)))),
                                          np.nan)
    if cfg.get("seismicity"):
        st = np.stack([np.maximum(np.asarray(fields[s], np.float64), 0.0) for s in cfg["seismicity"]])
        ok = np.all(np.isfinite(st), 0)
        tot = np.where(ok, np.log1p(np.nansum(st, 0)), np.nan)
        sm, _ = masked_smooth(tot, sigma_px(float(cfg.get("seismicity_sigma_m", 700))))
        out["ctx_seis_log"] = np.where(ok, sm, np.nan)
    for s in cfg.get("strain", []):
        v = scales.get(s)
        sc = float(v) if v is not None and np.isfinite(v) and float(v) > 0 else 1.0
        out[f"ctx_{s}_asinh"] = np.arcsinh(np.asarray(fields[s], np.float64) / sc)
    b = cfg.get("basement")
    if b:
        d = field_at_scale(fields[b], float(cfg.get("regional_sigma_m", 1500)), px_km)
        ok = np.isfinite(fields[b])
        out[f"ctx_{b}_reg"] = np.where(ok, d["S"], np.nan)
        out[f"ctx_{b}_relief"] = np.where(ok, d["hgm"], np.nan)
    return {k: np.asarray(v, np.float32) for k, v in out.items()}


def context_feature_bank(fields: dict, cfg: dict, scales: dict, px_km: float = 0.1) -> dict:
    return regional_context_features(fields, cfg, scales, px_km)


# =============================================================================== regional (global)
def global_context_names(cfg: dict) -> list[str]:
    out = ["ctx_dist_nodata_km"]
    if (cfg.get("domains") or {}).get("enabled", True):
        out.append("ctx_domain")
    return out


def distance_to_nodata_km(valid: np.ndarray, cap_km: float = 20.0, px_km: float = 0.1) -> np.ndarray:
    v = np.asarray(valid, bool)
    if v.all():
        return np.full(v.shape, cap_km, np.float32)
    d = ndimage.distance_transform_edt(v) * px_km
    return np.where(v, np.minimum(d, cap_km), np.nan).astype(np.float32)


def fit_domains(X: np.ndarray, k: int, seed: int):
    """Standardise (footprint sample statistics, label-free) + k-means; returns (scaler, model)."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    ok = np.all(np.isfinite(X), 1)
    sc = StandardScaler().fit(X[ok])
    km = KMeans(n_clusters=int(k), n_init=10, random_state=int(seed)).fit(sc.transform(X[ok]))
    return sc, km


def predict_domains(sc, km, X: np.ndarray) -> np.ndarray:
    ok = np.all(np.isfinite(X), 1)
    out = np.full(X.shape[0], np.nan, np.float32)
    if ok.any():
        out[ok] = km.predict(sc.transform(X[ok])).astype(np.float32)
    return out


def geological_domain_features(sample_X: np.ndarray, k: int, seed: int):
    """Unsupervised geological-regime model from a footprint sample of regional features (label-free)."""
    return fit_domains(sample_X, k, seed)


# =============================================================================== permitted known-fault context
def known_orientation(known: np.ndarray, sigma_px_: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Doubled-angle orientation of the visible traces at their own pixels (structure tensor of the binary map)."""
    k = np.asarray(known, np.float64)
    gr = ndimage.gaussian_filter(k, sigma_px_, order=(1, 0))
    gc = ndimage.gaussian_filter(k, sigma_px_, order=(0, 1))
    ge, gn = gc, -gr
    jee = ndimage.gaussian_filter(ge * ge, sigma_px_)
    jen = ndimage.gaussian_filter(ge * gn, sigma_px_)
    jnn = ndimage.gaussian_filter(gn * gn, sigma_px_)
    theta_grad = 0.5 * np.arctan2(2 * jen, jee - jnn)
    strike = theta_grad + np.pi / 2
    return np.cos(2 * strike), np.sin(2 * strike)


def permitted_fault_context(known: np.ndarray, valid: np.ndarray, cfg: dict, px_km: float = 0.1) -> dict:
    """Known-fault geometry of the VISIBLE faults of one unit (crop arrays). ``known`` must already exclude every
    fault the unit withholds; this function cannot check that - ``splits.verify_fold_isolation`` does."""
    from ..p6.stack import known_features
    from ..postprocess.topology import skeleton

    known = np.asarray(known, bool)
    valid = np.asarray(valid, bool)
    cap = float(cfg.get("cap_km", 20.0))
    out = known_features(known, valid, px_km, cap, float(cfg.get("density_radius_km", 2.0)),
                         float(cfg.get("tip_max_km", 5.0)))
    shape = known.shape
    if known.any():
        c2, s2 = known_orientation(known, sigma_px(float(cfg.get("orientation_sigma_m", 300))))
        _, (ir, ic) = ndimage.distance_transform_edt(~known, return_indices=True)
        out["known_orient_cos2"] = c2[ir, ic].astype(np.float32)
        out["known_orient_sin2"] = s2[ir, ic].astype(np.float32)
        sk = skeleton(known).astype(np.float64)
        r = max(int(round(float(cfg.get("network_radius_km", 5.0)) / px_km)), 1)
        out["known_net_density"] = (ndimage.uniform_filter(sk, 2 * r + 1, mode="constant") / px_km).astype(np.float32)
    else:
        out["known_orient_cos2"] = np.zeros(shape, np.float32)
        out["known_orient_sin2"] = np.zeros(shape, np.float32)
        out["known_net_density"] = np.zeros(shape, np.float32)
    return {k: np.where(valid, np.asarray(out[k], np.float32), np.nan) for k in KNOWN_COLUMNS}


def known_context_provenance(fit_or_unit: str, known: np.ndarray) -> dict:
    from ..m7.workspace import mask_hash

    return {"visibility": fit_or_unit, "known_hash": mask_hash(np.asarray(known, bool)),
            "known_px": int(np.asarray(known, bool).sum())}
