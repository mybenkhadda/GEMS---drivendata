"""Physics-informed feature engineering (sections 09-13) and the on-disk feature store.

Every builder is a generator of ``(name, array, meta)`` and receives the raw cube and the valid mask only: training
labels are never an input (``test_features_are_label_free``). Arrays are written one at a time to a per-feature
memmap after a *label-free* robust scaling (median / IQR over all valid pixels), so peak RAM stays at a few rasters.

Families (``G13Config.families``)::

    magnetic       spectral potential-field products of RTP (THD, VD, tilt, tilt gradient, analytic signal, upward
                   continuation THD/tilt and residuals, normalised source strength), multiscale gradient / coherence /
                   Hessian, monogenic local phase and phase congruency, texture entropy
    gravity        the same on isostatic gravity
    conductivity   multiscale gradient, coherence, Hessian, entropy of surface conductivity
    depth          multiscale discontinuity proxies of depth to basement
    terrain        slope, ridge/valley Hessian strength, slope break, relief anisotropy, TPI and local relief
    directional    oriented log-Gabor energy / anisotropy / directional coherence / strike of the primary fields
    structural     cross-contact asymmetry, multiscale orientation persistence, characteristic edge scale
    cross_modal    axial alignment, joint edge response, discordance, relative strength, shifted cross-correlation,
                   modal agreement fraction, family disagreement (common-source pairs are flagged)
    regional_norm  local z-scores of the primary fields at 2 and 8 km (vs the global robust scaling)
    validity       coverage flags, distance to nodata boundaries, local validity fraction (artefact auditing)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from . import signal as S
from .cache import atomic_save_table, atomic_write_json, combine
from .contracts import COMMON_SOURCE, MODALITY, PRIMARY_FIELD, GeoData

FEATURE_VERSION = "g13_features_v1"
FALLBACK_FIELD = {
    "magnetic": ("rtp", "mag_anom", "tmi"),
    "gravity": ("iso_grav_anom",),
    "terrain": ("det_elev",),
    "depth": ("depth_to_base_surf",),
    "conductivity": ("cond_surf",),
}


def primary(data: GeoData, modality: str) -> str | None:
    for n in FALLBACK_FIELD.get(modality, (PRIMARY_FIELD.get(modality),)):
        if n and data.has(n) and data.coverage.get(n, 0) > 0.05:
            return n
    return None


def _px(scale_m: float, pixel_m: float = 100.0) -> float:
    return max(float(scale_m) / pixel_m, 1.0)


def _meta(family, modality, source, transform, scale_m=None, formula="", deps=(), common=False):
    return {
        "family": family,
        "modality": modality,
        "source": source,
        "transform": transform,
        "scale_m": scale_m,
        "formula": formula,
        "dependencies": list(deps),
        "common_source": bool(common),
    }


def _rank01(a: np.ndarray, valid: np.ndarray, n_ref: int = 400_000, seed: int = 0) -> np.ndarray:
    """Global percentile (label-free) via a fixed random reference sample; NaN outside valid."""
    v = valid & np.isfinite(a)
    out = np.full(a.shape, np.nan, np.float32)
    if not v.any():
        return out
    vals = a[v]
    rng = np.random.default_rng(seed)
    ref = np.sort(rng.choice(vals, min(n_ref, vals.size), replace=False))
    out[v] = np.searchsorted(ref, vals, side="right") / len(ref)
    return out


# =============================================================================== family builders
def _field_family(data: GeoData, modality: str, cfg) -> tuple:
    name = primary(data, modality)
    if name is None:
        return
    x = data.band(name)
    v = data.valid & np.isfinite(x)
    fam = modality
    prods = S.potential_field_products(x, v, heights_m=cfg.upward_continuation_m)
    docs = {
        "thd": "sqrt(fx^2+fy^2) (FFT derivatives)",
        "vd": "F^-1[|k| F]",
        "tilt": "atan(vd/thd)",
        "tdx": "|grad tilt|",
        "asig": "sqrt(fx^2+fy^2+vd^2)",
        "nss": "sqrt(-l2^2 - l1 l3) of gradient tensor",
    }
    for k, a in prods.items():
        if k == "theta":
            c, s = S.axial_encode(a)
            yield (
                f"{fam}:{name}:pf_strike_cos2",
                S.remask(c, v),
                _meta(fam, modality, name, "pf_strike", None, "cos 2theta of THD strike"),
            )
            yield (
                f"{fam}:{name}:pf_strike_sin2",
                S.remask(s, v),
                _meta(fam, modality, name, "pf_strike", None, "sin 2theta of THD strike"),
            )
            continue
        if k in ("thd", "asig", "tdx", "nss") or k.endswith("_thd"):
            a = np.log1p(np.abs(a) / (np.nanmedian(np.abs(a[v])) + 1e-12))
        scale = None
        if k.startswith("uc") or k.startswith("res"):
            scale = float(k[2:].split("_")[0]) if k.startswith("uc") else float(k[3:])
        yield (
            f"{fam}:{name}:{k}",
            S.remask(a, v),
            _meta(fam, modality, name, k.split("_")[-1] if "_" in k else k, scale, docs.get(k, k)),
        )
    f = S.fill_nodata(x, v)
    for sm in cfg.scales_m:
        s = _px(sm) / 2
        gx, gy = S.gaussian_gradient(f, s)
        g = np.hypot(gx, gy)
        yield (
            f"{fam}:{name}:grad_s{sm}",
            S.remask(np.log1p(g / (np.nanmedian(g[v]) + 1e-12)), v),
            _meta(fam, modality, name, "gaussian_gradient", sm, "log1p(|grad G_s * f| / median)"),
        )
        coh, th, _ = S.structure_tensor(f, s, max(s, 1.5))
        yield (
            f"{fam}:{name}:coh_s{sm}",
            S.remask(coh, v),
            _meta(fam, modality, name, "structure_coherence", sm, "((l1-l2)/(l1+l2)) of J_s"),
        )
        l1, l2, _ = S.hessian_eigen(f, s)
        sc = np.nanstd(l1[v]) + 1e-12
        yield (
            f"{fam}:{name}:hess_l1_s{sm}",
            S.remask(l1 / sc, v),
            _meta(fam, modality, name, "hessian_l1", sm, "valley / trough strength (s^2 H)"),
        )
        yield (
            f"{fam}:{name}:hess_l2_s{sm}",
            S.remask(l2 / sc, v),
            _meta(fam, modality, name, "hessian_l2", sm, "ridge / high strength (s^2 H)"),
        )
    if modality in ("magnetic", "gravity", "conductivity"):
        yield (
            f"{fam}:{name}:entropy_w9",
            S.local_entropy(f, v, 9),
            _meta(fam, modality, name, "texture_entropy", 900, "local Shannon entropy, 16 bins"),
        )
    if modality in ("magnetic", "gravity"):
        mono = S.monogenic(x, v, [s * 2 for s in cfg.scales_m[1:]])
        for k, a in mono.items():
            yield (
                f"{fam}:{name}:{k}",
                a,
                _meta(fam, modality, name, "monogenic", None, "Riesz monogenic signal (Felsberg & Sommer 2001)"),
            )
        # cross-scale anomaly consistency: residuals agree in sign at short and long wavelength
        r1, r2 = prods.get(f"res{cfg.upward_continuation_m[0]}"), prods.get(f"res{cfg.upward_continuation_m[-1]}")
        if r1 is not None and r2 is not None:
            z1 = r1 / (np.nanstd(r1[v]) + 1e-12)
            z2 = r2 / (np.nanstd(r2[v]) + 1e-12)
            yield (
                f"{fam}:{name}:xscale_consistency",
                S.remask(np.sign(z1 * z2) * np.minimum(np.abs(z1), np.abs(z2)), v),
                _meta(fam, modality, name, "cross_scale_consistency", None, "sign(z1 z2) min(|z1|,|z2|) of residuals"),
            )


def build_magnetic(data, cfg):
    yield from _field_family(data, "magnetic", cfg)


def build_gravity(data, cfg):
    yield from _field_family(data, "gravity", cfg)


def _simple_family(data, modality, cfg):
    name = primary(data, modality)
    if name is None:
        return
    x = data.band(name)
    v = data.valid & np.isfinite(x)
    f = S.fill_nodata(x, v)
    for sm in cfg.scales_m:
        s = _px(sm) / 2
        gx, gy = S.gaussian_gradient(f, s)
        g = np.hypot(gx, gy)
        yield (
            f"{modality}:{name}:grad_s{sm}",
            S.remask(np.log1p(g / (np.nanmedian(g[v]) + 1e-12)), v),
            _meta(modality, modality, name, "gaussian_gradient", sm, "log1p(|grad G_s * f| / median)"),
        )
        coh, _, _ = S.structure_tensor(f, s, max(s, 1.5))
        yield f"{modality}:{name}:coh_s{sm}", S.remask(coh, v), _meta(modality, modality, name, "structure_coherence", sm)
        l1, l2, _ = S.hessian_eigen(f, s)
        sc = np.nanstd(l1[v]) + 1e-12
        yield f"{modality}:{name}:hess_l1_s{sm}", S.remask(l1 / sc, v), _meta(modality, modality, name, "hessian_l1", sm)
        yield f"{modality}:{name}:hess_l2_s{sm}", S.remask(l2 / sc, v), _meta(modality, modality, name, "hessian_l2", sm)
    if modality == "conductivity":
        yield (
            f"{modality}:{name}:entropy_w9",
            S.local_entropy(f, v, 9),
            _meta(modality, modality, name, "texture_entropy", 900),
        )
    if modality == "depth":
        for sm in (1000, 4000):
            res = x - S.masked_smooth(f, v, _px(sm) / 2)
            yield (
                f"depth:{name}:residual_s{sm}",
                S.remask(res, v),
                _meta("depth", "depth", name, "residual", sm, "f - G_s * f (local discontinuity proxy)"),
            )


def build_conductivity(data, cfg):
    yield from _simple_family(data, "conductivity", cfg)


def build_depth(data, cfg):
    yield from _simple_family(data, "depth", cfg)


def build_terrain(data, cfg):
    name = primary(data, "terrain")
    if name is None:
        return
    x = data.band(name)
    v = data.valid & np.isfinite(x)
    f = S.fill_nodata(x, v)
    for sm in cfg.scales_m:
        s = _px(sm) / 2
        gx, gy = S.gaussian_gradient(f, s)
        slope = np.hypot(gx, gy) / 100.0
        yield (
            f"terrain:{name}:slope_s{sm}",
            S.remask(np.log1p(slope / (np.nanmedian(slope[v]) + 1e-12)), v),
            _meta("terrain", "terrain", name, "slope", sm, "|grad G_s * z| (m/m)"),
        )
        sb = np.hypot(*S.gaussian_gradient(slope, s))
        yield (
            f"terrain:{name}:slope_break_s{sm}",
            S.remask(np.log1p(sb / (np.nanmedian(sb[v]) + 1e-12)), v),
            _meta("terrain", "terrain", name, "slope_break", sm, "|grad slope| (range-front slope breaks)"),
        )
        l1, l2, _ = S.hessian_eigen(f, s)
        sc = np.nanstd(l1[v]) + 1e-12
        yield (
            f"terrain:{name}:valley_s{sm}",
            S.remask(np.maximum(l1, 0) / sc, v),
            _meta("terrain", "terrain", name, "valley_strength", sm, "max(l1,0): drainage / linear valley"),
        )
        yield (
            f"terrain:{name}:ridge_s{sm}",
            S.remask(np.maximum(-l2, 0) / sc, v),
            _meta("terrain", "terrain", name, "ridge_strength", sm, "max(-l2,0): ridgeline"),
        )
        coh, _, _ = S.structure_tensor(f, s, max(s, 1.5))
        yield (
            f"terrain:{name}:relief_aniso_s{sm}",
            S.remask(coh, v),
            _meta("terrain", "terrain", name, "relief_anisotropy", sm),
        )
    rel = S.multiscale_relief(f, v, (5, 20, 60))
    for k, a in rel.items():
        yield f"terrain:{name}:{k}", a, _meta("terrain", "terrain", name, k.split("_")[0], int(k.split("_s")[1]) * 100)
    # optional DEM-derived 100 m aggregates (G07 dem_1m_100m), with an explicit coverage mask
    dem_dir = getattr(cfg, "dem_dir", "")
    if dem_dir and Path(dem_dir).exists():
        yield from _dem_aggregates(data, Path(dem_dir))


def _dem_aggregates(data: GeoData, dem_dir: Path):
    import rasterio

    from .contracts import check_grid

    with rasterio.open(data.template_path) as t:
        for p in sorted(dem_dir.glob("*.tif")):
            with rasterio.open(p) as src:
                try:
                    check_grid(t, src)
                except ValueError:
                    continue
                for b in range(1, src.count + 1):
                    a = src.read(b, masked=True).filled(np.nan).astype(np.float32)
                    nm = (src.descriptions[b - 1] or f"{p.stem}_{b}").lower()
                    yield (
                        f"terrain:dem1m:{nm}",
                        S.remask(a, data.valid & np.isfinite(a)),
                        _meta("terrain", "terrain", f"dem1m:{p.name}", "dem_aggregate", None, "1 m DEM product at 100 m"),
                    )
                cov = np.isfinite(src.read(1, masked=True).filled(np.nan)) & data.valid
                yield (
                    f"validity:dem1m:{p.stem}_coverage",
                    cov.astype(np.float32),
                    _meta("validity", "terrain", p.name, "coverage", None, "1 m DEM coverage mask"),
                )
                break


def build_directional(data, cfg):
    lams = (800, 3200)
    for modality in ("magnetic", "gravity", "terrain"):
        name = primary(data, modality)
        if name is None:
            continue
        x = data.band(name)
        v = data.valid & np.isfinite(x)
        g = S.gabor_bank(x, v, lams, cfg.gabor_orientations)
        for k, a in g.items():
            if k.endswith("_theta"):
                c, s = S.axial_encode(a)
                yield f"directional:{name}:{k}_cos2", S.remask(c, v), _meta("directional", modality, name, "gabor_strike")
                yield f"directional:{name}:{k}_sin2", S.remask(s, v), _meta("directional", modality, name, "gabor_strike")
            else:
                yield (
                    f"directional:{name}:{k}",
                    a,
                    _meta(
                        "directional",
                        modality,
                        name,
                        k.split("_")[-1],
                        float(k[5:].split("_")[0]),
                        "oriented log-Gabor bank",
                    ),
                )


def _edge_and_strike(data, modality, sm=800):
    name = primary(data, modality)
    if name is None:
        return None
    x = data.band(name)
    v = data.valid & np.isfinite(x)
    f = S.fill_nodata(x, v)
    s = _px(sm) / 2
    gx, gy = S.gaussian_gradient(f, s)
    coh, th, _ = S.structure_tensor(f, s, max(s, 1.5))
    return {"name": name, "valid": v, "f": f, "edge": np.hypot(gx, gy), "gx": gx, "gy": gy, "theta": th, "coh": coh}


def build_structural(data, cfg):
    for modality in ("magnetic", "gravity", "terrain", "depth", "conductivity"):
        name = primary(data, modality)
        if name is None:
            continue
        x = data.band(name)
        v = data.valid & np.isfinite(x)
        f = S.fill_nodata(x, v)
        thetas, weights, resp = [], [], []
        for sm in cfg.scales_m:
            s = _px(sm) / 2
            coh, th, _ = S.structure_tensor(f, s, max(s, 1.5))
            thetas.append(th)
            weights.append(coh)
            gx, gy = S.gaussian_gradient(f, s)
            resp.append(s * np.hypot(gx, gy))  # scale-normalised edge response (Lindeberg)
        yield (
            f"structural:{name}:orient_persistence",
            S.remask(S.orientation_persistence(thetas, weights), v),
            _meta(
                "structural",
                modality,
                name,
                "orientation_persistence",
                None,
                "resultant of coherence-weighted axial strikes over scales",
            ),
        )
        R = np.stack(resp)
        yield (
            f"structural:{name}:char_scale",
            S.remask(np.asarray(cfg.scales_m, np.float32)[np.argmax(R, 0)] / 1000.0, v),
            _meta(
                "structural",
                modality,
                name,
                "characteristic_scale",
                None,
                "argmax_s s|grad G_s f| (km): lineament width proxy",
            ),
        )
        # cross-contact asymmetry: values on both sides of the local gradient normal at 2 offsets
        mid = cfg.scales_m[len(cfg.scales_m) // 2]
        s = _px(mid) / 2
        a = S.masked_smooth(x, v, s)
        gx, gy = S.gaussian_gradient(a, s)
        amp = np.hypot(gx, gy) + 1e-12
        nx, ny = gx / amp, gy / amp
        rr, cc = np.indices(x.shape, dtype=np.float32)
        loc = np.sqrt(np.maximum(S.masked_smooth((a - S.masked_smooth(a, v, 2 * s)) ** 2, v, s), 1e-12))
        for d_m in (300, 1000):
            d = d_m / 100.0
            right = ndi.map_coordinates(a, [rr - d * ny, cc + d * nx], order=1, mode="nearest")
            left = ndi.map_coordinates(a, [rr + d * ny, cc - d * nx], order=1, mode="nearest")
            yield (
                f"structural:{name}:across_step_d{d_m}",
                S.remask(np.arcsinh((right - left) / loc), v),
                _meta("structural", modality, name, "across_contact_step", d_m, "asinh((f+ - f-)/local std) along normal"),
            )
            mid_v = a
            asym = np.abs((right - mid_v) + (left - mid_v)) / (np.abs(right - left) + loc)
            yield (
                f"structural:{name}:across_asym_d{d_m}",
                S.remask(asym, v),
                _meta(
                    "structural",
                    modality,
                    name,
                    "across_contact_asymmetry",
                    d_m,
                    "|(f+ - f0) + (f- - f0)| / (|f+ - f-| + std): 0 = step, high = line",
                ),
            )


def build_cross_modal(data, cfg):
    mods = [m for m in ("magnetic", "gravity", "terrain", "depth", "conductivity") if primary(data, m)]
    E = {m: _edge_and_strike(data, m) for m in mods}
    ranks = {m: _rank01(E[m]["edge"], E[m]["valid"]) for m in mods}
    pairs = [
        ("magnetic", "gravity"),
        ("magnetic", "conductivity"),
        ("gravity", "depth"),
        ("terrain", "magnetic"),
        ("terrain", "gravity"),
        ("gravity", "conductivity"),
        ("terrain", "depth"),
    ]
    for a, b in pairs:
        if a not in E or b not in E:
            continue
        A, B = E[a], E[b]
        v = A["valid"] & B["valid"]
        common = (
            COMMON_SOURCE.get(B["name"]) == A["name"]
            or COMMON_SOURCE.get(A["name"]) == B["name"]
            or {a, b} == {"gravity", "depth"}
        )
        tag = f"{a[:4]}_{b[:4]}"
        align = np.cos(S.axial_diff(A["theta"], B["theta"])) ** 2 * np.minimum(A["coh"], B["coh"])
        yield (
            f"cross_modal:{tag}:axial_alignment",
            S.remask(align, v),
            _meta(
                "cross_modal",
                f"{a}+{b}",
                f"{A['name']},{B['name']}",
                "axial_alignment",
                800,
                "cos^2(dtheta) min(cohA, cohB)",
                common=common,
            ),
        )
        joint = (
            np.sqrt(np.nan_to_num(ranks[a]) * np.nan_to_num(ranks[b])) * np.cos(S.axial_diff(A["theta"], B["theta"])) ** 2
        )
        yield (
            f"cross_modal:{tag}:joint_edge",
            S.remask(joint, v),
            _meta(
                "cross_modal",
                f"{a}+{b}",
                f"{A['name']},{B['name']}",
                "joint_edge",
                800,
                "sqrt(rankA rankB) cos^2(dtheta)",
                common=common,
            ),
        )
        yield (
            f"cross_modal:{tag}:discordance",
            S.remask(np.abs(ranks[a] - ranks[b]), v),
            _meta(
                "cross_modal", f"{a}+{b}", f"{A['name']},{B['name']}", "discordance", 800, "|rankA - rankB|", common=common
            ),
        )
        rel = np.log(
            (A["edge"] / (np.nanmedian(A["edge"][v]) + 1e-12) + 1e-3)
            / (B["edge"] / (np.nanmedian(B["edge"][v]) + 1e-12) + 1e-3)
        )
        yield (
            f"cross_modal:{tag}:relative_strength",
            S.remask(rel, v),
            _meta(
                "cross_modal",
                f"{a}+{b}",
                f"{A['name']},{B['name']}",
                "relative_strength",
                800,
                "log(edgeA/edgeB)",
                common=common,
            ),
        )
        lag, corr = shifted_edge_correlation(ranks[a], ranks[b], A["theta"], v)
        yield (
            f"cross_modal:{tag}:shift_corr",
            S.remask(corr, v),
            _meta(
                "cross_modal",
                f"{a}+{b}",
                f"{A['name']},{B['name']}",
                "shifted_correlation",
                800,
                "max local corr of edge ranks over normal shifts <= 5 px",
                common=common,
            ),
        )
        yield (
            f"cross_modal:{tag}:shift_lag_km",
            S.remask(lag, v),
            _meta(
                "cross_modal",
                f"{a}+{b}",
                f"{A['name']},{B['name']}",
                "shift_lag",
                800,
                "signed normal offset (km) of best correlation",
                common=common,
            ),
        )
    if len(mods) >= 2:
        R = np.stack([np.nan_to_num(ranks[m]) for m in mods])
        av = np.stack([E[m]["valid"] for m in mods])
        v = av.any(0)
        near_top = np.stack([ndi.maximum_filter(r, size=7) >= 0.9 for r in R]) & av
        yield (
            "cross_modal:all:agreement_frac",
            S.remask(near_top.sum(0) / np.maximum(av.sum(0), 1), v),
            _meta(
                "cross_modal",
                "all",
                ",".join(mods),
                "modal_agreement",
                300,
                "share of available modalities with a top-10% edge within 300 m",
            ),
        )
        indep = [m for m in mods if m != "depth"]
        Ri = np.stack([np.nan_to_num(ranks[m]) for m in indep])
        yield (
            "cross_modal:all:family_disagreement",
            S.remask(Ri.std(0), v),
            _meta("cross_modal", "all", ",".join(indep), "family_disagreement", 800, "std of edge ranks across modalities"),
        )
        yield (
            "cross_modal:all:independent_max_edge",
            S.remask(Ri.max(0), v),
            _meta("cross_modal", "all", ",".join(indep), "max_edge", 800, "max edge rank over independent modalities"),
        )


def shifted_edge_correlation(
    a: np.ndarray, b: np.ndarray, theta: np.ndarray, valid: np.ndarray, max_shift: int = 5, window: float = 4.0
) -> tuple[np.ndarray, np.ndarray]:
    """Local correlation of two edge maps after shifting ``b`` along the local normal of ``a`` by -max..+max px.

    Returns (signed lag in km of the best correlation, measured along n = (-sin theta, cos theta) of ``a``'s strike,
    and the best correlation). A consistent non-zero lag is a geologically
    interpretable cross-modal offset (e.g. a dipping contact seen at depth by gravity, at surface by terrain)."""
    a = np.nan_to_num(a).astype(np.float32)
    b = np.nan_to_num(b).astype(np.float32)
    nx, ny = -np.sin(theta), np.cos(theta)
    rr, cc = np.indices(a.shape, dtype=np.float32)
    ma = ndi.gaussian_filter(a, window)
    sa = np.sqrt(np.maximum(ndi.gaussian_filter(a * a, window) - ma * ma, 1e-8))
    best, lag = np.full(a.shape, -1.0, np.float32), np.zeros(a.shape, np.float32)
    for d in range(-max_shift, max_shift + 1):
        bs = ndi.map_coordinates(b, [rr - d * ny, cc + d * nx], order=1, mode="nearest")
        mb = ndi.gaussian_filter(bs, window)
        sb = np.sqrt(np.maximum(ndi.gaussian_filter(bs * bs, window) - mb * mb, 1e-8))
        corr = (ndi.gaussian_filter(a * bs, window) - ma * mb) / (sa * sb)
        upd = corr > best
        best[upd] = corr[upd]
        lag[upd] = d * 0.1
    return lag, np.clip(best, -1, 1)


def build_regional_norm(data, cfg):
    for modality in ("magnetic", "gravity", "terrain", "conductivity", "depth"):
        name = primary(data, modality)
        if name is None:
            continue
        x = data.band(name)
        v = data.valid & np.isfinite(x)
        for km in (2, 8):
            s = km * 10 / 2
            m = S.masked_smooth(x, v, s)
            sd = np.sqrt(np.maximum(S.masked_smooth((x - m) ** 2, v, s), 1e-12))
            yield (
                f"regional_norm:{name}:localz_{km}km",
                S.remask(np.clip((x - m) / sd, -8, 8), v),
                _meta("regional_norm", modality, name, "local_z", km * 1000, "(f - G_s f) / local std (label-free)"),
            )


def build_validity(data, cfg):
    core = [n for n in data.names if MODALITY.get(n) in ("magnetic", "gravity", "terrain", "conductivity", "depth")]
    allv = np.ones(data.shape, bool)
    for n in core:
        allv &= np.isfinite(data.band(n))
    bd = S.boundary_distance(allv & data.valid)
    yield (
        "validity:all:boundary_dist_km",
        S.remask(np.minimum(bd, 200) / 10.0, data.valid),
        _meta("validity", "all", ",".join(core), "boundary_distance", None, "distance to nearest nodata/edge (km)"),
    )
    for mod in ("magnetic", "gravity", "terrain", "conductivity", "depth"):
        n = primary(data, mod)
        if n is None:
            continue
        ok = np.isfinite(data.band(n)) & data.valid
        frac = ndi.uniform_filter(ok.astype(np.float32), 25)
        yield (
            f"validity:{n}:local_valid_frac",
            S.remask(frac, data.valid),
            _meta("validity", mod, n, "local_validity", 2500, "share of valid cells in 2.5 km window"),
        )


BUILDERS = {
    "magnetic": build_magnetic,
    "gravity": build_gravity,
    "conductivity": build_conductivity,
    "depth": build_depth,
    "terrain": build_terrain,
    "directional": build_directional,
    "structural": build_structural,
    "cross_modal": build_cross_modal,
    "regional_norm": build_regional_norm,
    "validity": build_validity,
}


# =============================================================================== feature store
class FeatureStore:
    """Directory of per-feature ``.npy`` memmaps + ``registry.csv``. Raw bands are addressed as ``raw:<band>``."""

    def __init__(self, root: str | Path, data: GeoData | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.data = data
        self.reg_path = self.root / "registry.csv"
        self.registry = pd.read_csv(self.reg_path) if self.reg_path.exists() else pd.DataFrame()
        self._mm: dict[str, np.ndarray] = {}

    # ---------------------------------------------------------------- naming
    @staticmethod
    def fname(name: str) -> str:
        return name.replace(":", "__").replace("/", "_") + ".npy"

    def names(self, families=None, include_raw: bool = True) -> list[str]:
        out = []
        if include_raw and self.data is not None:
            out += [f"raw:{n}" for n in self.data.names]
        if len(self.registry):
            r = self.registry if families is None else self.registry[self.registry.family.isin(families)]
            out += r.name.tolist()
        return out

    def family_of(self, name: str) -> str:
        if name.startswith("raw:"):
            return "raw"
        r = self.registry.loc[self.registry.name == name, "family"]
        return str(r.iloc[0]) if len(r) else "unknown"

    def has(self, name: str) -> bool:
        if name.startswith("raw:"):
            return self.data is not None and name[4:] in self.data.names
        return (self.root / self.fname(name)).exists() and (len(self.registry) and name in set(self.registry.name))

    # ---------------------------------------------------------------- io
    def get(self, name: str) -> np.ndarray:
        if name.startswith("raw:"):
            return self.data.band(name[4:])
        if name not in self._mm:
            self._mm[name] = np.load(self.root / self.fname(name), mmap_mode="r")
        return np.asarray(self._mm[name], np.float32)

    def physical(self, name: str) -> np.ndarray:
        """Feature in its original (unscaled) units: undo the label-free robust scaling."""
        a = self.get(name)
        if name.startswith("raw:"):
            return a
        r = self.registry.loc[self.registry.name == name].iloc[0]
        return a * float(r.scale_iqr) + float(r.scale_median)

    def column(self, name: str, ids: np.ndarray) -> np.ndarray:
        if name.startswith("raw:"):
            return np.asarray(self.data.raw[self.data.names.index(name[4:])].reshape(-1)[ids], np.float32)
        if name not in self._mm:
            self._mm[name] = np.load(self.root / self.fname(name), mmap_mode="r")
        return np.asarray(self._mm[name].reshape(-1)[ids], np.float32)

    def matrix(self, ids: np.ndarray, names: list[str]) -> np.ndarray:
        order = np.argsort(ids, kind="stable")
        sid = ids[order]
        X = np.empty((len(ids), len(names)), np.float32)
        for j, n in enumerate(names):
            X[order, j] = self.column(n, sid)
        X[~np.isfinite(X)] = np.nan
        return X

    def write(self, name: str, arr: np.ndarray, meta: dict, dtype: str, valid: np.ndarray, seconds: float) -> dict:
        a = np.asarray(arr, np.float32)
        fin = np.isfinite(a) & valid
        vals = a[fin]
        if vals.size:
            sub = vals[:: max(vals.size // 500_000, 1)]
            med = float(np.median(sub))
            iqr = float(np.subtract(*np.percentile(sub, [75, 25]))) or float(np.std(sub)) or 1.0
        else:
            med, iqr = 0.0, 1.0
        scaled = (a - med) / iqr
        if dtype == "float16":
            scaled = np.clip(scaled, -6e4, 6e4)
        scaled = scaled.astype(dtype)
        scaled[~np.isfinite(a)] = np.nan
        p = self.root / self.fname(name)
        tmp = p.with_name(p.stem + ".partial.npy")
        np.save(tmp, scaled)
        tmp.replace(p)
        self._mm.pop(name, None)
        return {
            "name": name,
            **{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in meta.items()},
            "scale_median": med,
            "scale_iqr": iqr,
            "dtype": dtype,
            "coverage_frac": float(fin.sum() / max(valid.sum(), 1)),
            "seconds": round(seconds, 3),
            "bytes": p.stat().st_size,
            "file": p.name,
            "version": FEATURE_VERSION,
        }

    def save_registry(self, rows: list[dict]) -> None:
        new = pd.DataFrame(rows)
        if len(self.registry):
            keep = self.registry[~self.registry.name.isin(new.name)] if len(new) else self.registry
            new = pd.concat([keep, new], ignore_index=True)
        self.registry = new
        atomic_save_table(self.reg_path, new)


def build_store(data: GeoData, cfg, root: Path, log=print) -> FeatureStore:
    """Build every requested family not already present with the same family fingerprint."""
    store = FeatureStore(root, data)
    stamps_p = Path(root) / "family_stamps.json"
    stamps = json.loads(stamps_p.read_text()) if stamps_p.exists() else {}
    for fam in cfg.families:
        stamp = combine(
            FEATURE_VERSION,
            fam,
            cfg.scales_m,
            cfg.upward_continuation_m,
            cfg.gabor_orientations,
            cfg.store_dtype,
            data.names,
            getattr(cfg, "dem_dir", ""),
        )
        if stamps.get(fam) == stamp and len(store.registry) and (store.registry.family == fam).any():
            files_ok = all((Path(root) / f).exists() for f in store.registry.loc[store.registry.family == fam, "file"])
            if files_ok:
                log(f"[features] {fam}: reused ({int((store.registry.family == fam).sum())} features)")
                continue
        rows, t0 = [], time.perf_counter()
        t_last = t0
        for name, arr, meta in BUILDERS[fam](data, cfg):
            now = time.perf_counter()
            rows.append(store.write(name, arr, meta, cfg.store_dtype, data.valid, now - t_last))
            t_last = time.perf_counter()
            del arr
        if rows:
            store.save_registry(rows)
        stamps[fam] = stamp
        atomic_write_json(stamps_p, stamps)
        log(f"[features] {fam}: {len(rows)} features in {time.perf_counter() - t0:.1f}s")
    return store


def family_feature_sets(store: FeatureStore) -> dict[str, list[str]]:
    """Feature lists for the ablation design A0-A9 (A10/A11 are built from selection results)."""
    raw = store.names(families=[], include_raw=True)
    fam = lambda *f: store.names(families=list(f), include_raw=False)  # noqa: E731
    return {
        "A0_raw": raw,
        "A1_magnetic": raw + fam("magnetic"),
        "A2_gravity": raw + fam("gravity"),
        "A3_conductivity": raw + fam("conductivity"),
        "A4_terrain": raw + fam("terrain"),
        "A5_depth": raw + fam("depth"),
        "A6_structural": raw + fam("structural"),
        "A7_directional": raw + fam("directional"),
        "A8_cross_modal": raw + fam("cross_modal"),
        "A9_all": store.names(),
    }
