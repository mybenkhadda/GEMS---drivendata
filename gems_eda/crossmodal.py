"""Cross-modal structural evidence: shared orientations, alignment with faults, complementarity."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import ndimage

from .enhance import Enhancer
from .io import Grid
from .morphology import FaultNetwork
from .stats import auc_score, axial_hist, rank_normalize, subsample


def orientation_fields(fields: dict[str, np.ndarray], enh: Enhancer, scale_km: float) -> dict[str, dict]:
    out = {}
    for name, arr in fields.items():
        strike, coh, energy = enh.structure_tensor(arr, scale_km)
        out[name] = {"strike": strike, "coherence": coh, "energy_rank": rank_normalize(energy)}
    return out


def fabric_roses(orient: dict[str, dict], valid: np.ndarray, rng: np.random.Generator,
                 bins: int = 36, n: int = 300_000, min_coherence: float = 0.3) -> pd.DataFrame:
    """Orientation histograms weighted by coherence x energy rank (strong, organised fabric)."""
    idx = subsample(np.flatnonzero(valid.ravel()), n, rng)
    rows = []
    for name, o in orient.items():
        s, c, e = (o[k].ravel()[idx] for k in ("strike", "coherence", "energy_rank"))
        w = np.where(c >= min_coherence, c * e, 0)
        centers, h = axial_hist(s, w, bins)
        rows.append(pd.DataFrame({"field": name, "strike_bin": centers, "density": h}))
    return pd.concat(rows, ignore_index=True)


def rose_correlation(roses: pd.DataFrame) -> pd.DataFrame:
    return roses.pivot(index="strike_bin", columns="field", values="density").corr()


def alignment_at_stations(orient: dict[str, dict], stations: pd.DataFrame) -> pd.DataFrame:
    """cos(2 * delta strike) between modality fabric and local fault/background strike.

    +1 = parallel, -1 = perpendicular, 0 expected for unrelated orientations.
    """
    r, c = stations.r.to_numpy(), stations.c.to_numpy()
    rows = []
    for name, o in orient.items():
        d = np.radians(o["strike"][r, c] - stations.strike.to_numpy())
        rows.append(pd.DataFrame({
            "sample_id": stations.sample_id.to_numpy(), "kind": stations.kind.to_numpy(), "field": name,
            "coherence": o["coherence"][r, c], "cos2": np.cos(2 * d),
        }))
    return pd.concat(rows, ignore_index=True)


def near_far_masks(net: FaultNetwork, grid: Grid, valid: np.ndarray, tol_px: int, far_km: float):
    near = ndimage.binary_dilation(net.skeleton, iterations=tol_px) & valid if tol_px else net.skeleton & valid
    far = (net.distance_km > far_km) & valid
    return near, far


def pixel_auc(response: np.ndarray, near: np.ndarray, far: np.ndarray, rng: np.random.Generator,
              n: int = 100_000) -> float:
    pos = subsample(np.flatnonzero(near.ravel()), n, rng)
    neg = subsample(np.flatnonzero(far.ravel()), n, rng)
    flat = response.ravel()
    return auc_score(flat[pos], flat[neg])


def edge_agreement(edges: dict[str, np.ndarray], valid: np.ndarray, q: float = 0.90) -> tuple[dict, np.ndarray]:
    """Binary top-(1-q) edge maps per modality and the per-pixel agreement count."""
    binary = {}
    for name, e in edges.items():
        thr = np.nanquantile(e[valid], q)
        binary[name] = np.nan_to_num(e) >= thr
    count = np.sum(np.stack(list(binary.values())), axis=0).astype(np.int8)
    count[~valid] = -1
    return binary, count


def evidence_combinations(evidence: pd.DataFrame, groups, strong: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Which modality groups support each fault station (UpSet-style counts, Jaccard, unique share)."""
    f = evidence[evidence.kind == "fault"]
    flags = pd.DataFrame({g: f[f"ev_{g}"] >= strong for g in groups if f"ev_{g}" in f})
    key = flags.apply(lambda r: "+".join([g for g in flags.columns if r[g]]) or "none", axis=1)
    combos = key.value_counts().rename_axis("combination").reset_index(name="n_stations")
    combos["share"] = combos.n_stations / len(f)
    jac = pd.DataFrame(index=flags.columns, columns=flags.columns, dtype=float)
    for a, b in combinations(flags.columns, 2):
        inter = (flags[a] & flags[b]).sum()
        union = (flags[a] | flags[b]).sum()
        jac.loc[a, b] = jac.loc[b, a] = inter / union if union else np.nan
    for a in flags.columns:
        jac.loc[a, a] = 1.0
    others = flags.sum(axis=1)
    uniq = pd.DataFrame({
        "group": flags.columns,
        "support_share": [flags[g].mean() for g in flags.columns],
        "unique_share": [(flags[g] & (others == 1)).mean() for g in flags.columns],
    })
    uniq.loc[len(uniq)] = {"group": "none", "support_share": (others == 0).mean(), "unique_share": np.nan}
    return combos, jac, uniq
