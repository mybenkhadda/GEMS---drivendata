"""Unsupervised, multi-modal lineament detection and ranking of candidate unmapped structures.

Detector: for each modality, ridges of the total horizontal derivative (source
edges / scarps) are computed at several scales, scale-normalised and rank-mapped;
modalities are averaged into a fused lineament map and weighted by the local
orientation coherence. Hysteresis thresholding + skeletonisation yields traces.

Candidates are *exploratory*: they are coherent geophysical lineaments, not
confirmed faults (lithological contacts, dykes, cultural noise and survey
artefacts produce the same signatures).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import remove_small_objects, skeletonize

from .enhance import Enhancer
from .io import Grid
from .morphology import FaultNetwork, path_geometry, trace_skeleton
from .stats import axial_diff, percentile_in, rank_normalize


@dataclass
class LineamentDetector:
    enh: Enhancer
    scales_km: tuple[float, ...] = (2.0, 5.0, 10.0)
    high_q: float = 0.95
    low_q: float = 0.85
    min_len_km: float = 6.0
    support_q: float = 0.90
    responses: dict[str, np.ndarray] = field(default_factory=dict)

    def modality_response(self, field_: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Max over scales of the normalised Hessian ridge of THDR, rank-mapped to (0, 1]."""
        best = None
        for s in self.scales_km:
            ridge = self.enh.ridge(self.enh.thdr(field_, s), s, "bright")[0]
            q = np.nanquantile(ridge[valid], 0.99)
            ridge = ridge / q if q > 0 else ridge
            best = ridge if best is None else np.fmax(best, ridge)
        return rank_normalize(best, valid)

    def fit(self, fields: dict[str, np.ndarray], valid: np.ndarray) -> "LineamentDetector":
        self.responses = {k: self.modality_response(v, valid) for k, v in fields.items()}
        return self

    def fused(self, valid: np.ndarray) -> dict[str, np.ndarray]:
        stack = np.stack(list(self.responses.values()))
        fused = np.nanmean(stack, axis=0)
        tol = np.stack([ndimage.maximum_filter(np.nan_to_num(r), size=3) for r in stack])
        support = (tol >= self.support_q).sum(0).astype(np.float32)
        _, coherence, _ = self.enh.structure_tensor(fused, self.scales_km[0], integration=2.0)
        score = fused * np.sqrt(np.clip(coherence, 0, 1))
        for a in (fused, support, coherence, score):
            a[~valid] = np.nan
        return {"fused": fused.astype(np.float32), "support": support, "coherence": coherence.astype(np.float32),
                "score": score.astype(np.float32)}

    def detect(self, maps: dict[str, np.ndarray], valid: np.ndarray, grid: Grid):
        score = maps["score"]
        hi, lo = np.nanquantile(score[valid], [self.high_q, self.low_q])
        binary = apply_hysteresis_threshold(np.nan_to_num(score), lo, hi) & valid
        binary = remove_small_objects(binary, min_size=max(int(self.min_len_km / grid.pixel_km), 3))
        skel = skeletonize(binary)
        return skel, trace_skeleton(skel, spur_px=max(3.0, 1.0 / grid.pixel_km))


def candidate_table(graph, grid: Grid, maps: dict, responses: dict, net: FaultNetwork,
                    fault_rose: tuple[np.ndarray, np.ndarray], min_len_km: float,
                    match_km: float, support_q: float = 0.90) -> tuple[pd.DataFrame, list[np.ndarray]]:
    centers, density = fault_rose
    rows, keep = [], []
    for path in graph.paths:
        if len(path) < 3:
            continue
        geo = path_geometry(path, grid)
        if geo["length_km"] < min_len_km:
            continue
        r, c = path[:, 0], path[:, 1]
        d = net.distance_km[r, c]
        rec = {
            **geo,
            "fused_mean": float(np.nanmean(maps["fused"][r, c])),
            "coherence_mean": float(np.nanmean(maps["coherence"][r, c])),
            "support_mean": float(np.nanmean(maps["support"][r, c])),
            "dist_known_median_km": float(np.median(d)),
            "frac_near_known": float(np.mean(d <= match_km)),
            "r_mid": int(path[len(path) // 2, 0]), "c_mid": int(path[len(path) // 2, 1]),
        }
        for name, resp in responses.items():
            rec[f"resp_{name}"] = float(np.nanmean(resp[r, c]))
            rec[f"supp_{name}"] = float(np.mean(np.nan_to_num(resp[r, c]) >= support_q))
        b = int(np.argmin(axial_diff(centers, geo["azimuth"])))
        rec["fabric_density"] = float(density[b] * len(density))    # 1 = uniform expectation
        rows.append(rec)
        keep.append(path)
    df = pd.DataFrame(rows)
    if df.empty:
        return df, keep
    df["status"] = np.select([df.frac_near_known >= 0.5, df.dist_known_median_km <= 3 * match_km],
                             ["mapped", "near_mapped"], "unmapped")
    supp_cols = [c for c in df.columns if c.startswith("supp_")]
    df["n_supporting_modalities"] = (df[supp_cols] >= 0.5).sum(axis=1)
    pct = lambda s: percentile_in(s, s)                                       # noqa: E731
    df["structural_score"] = np.nanmean(np.vstack([
        pct(np.log(df.length_km)), pct(df.linearity), pct(df.coherence_mean), pct(-df.sinuosity.fillna(1))]), axis=0)
    df["evidence_score"] = np.nanmean(np.vstack([
        pct(df.fused_mean), pct(df.support_mean), pct(df.n_supporting_modalities)]), axis=0)
    df["fabric_score"] = pct(df.fabric_density)
    df["rank_score"] = 0.4 * df.structural_score + 0.4 * df.evidence_score + 0.2 * df.fabric_score
    x, y = grid.rc_to_xy(df.r_mid, df.c_mid)
    df["x_mid"], df["y_mid"] = x, y
    df.insert(0, "cand_id", np.arange(len(df)))
    return df, keep


def detector_validation(cands: pd.DataFrame, paths: list[np.ndarray], net: FaultNetwork, grid: Grid,
                        valid: np.ndarray, match_km: float) -> dict:
    """Recall of known traces by the detector, precision proxy, and lift over random placement."""
    cand_mask = np.zeros(grid.shape, bool)
    for p in paths:
        cand_mask[p[:, 0], p[:, 1]] = True
    d_cand = ndimage.distance_transform_edt(~cand_mask, sampling=(grid.pixel_km_y, grid.pixel_km_x))
    known = net.skeleton & valid
    recall = float((d_cand[known] <= match_km).mean()) if known.any() else np.nan
    on_cand = cand_mask & valid
    precision = float((net.distance_km[on_cand] <= match_km).mean()) if on_cand.any() else np.nan
    area_frac = float((net.distance_km[valid] <= match_km).mean())
    return {"recall_known_within_km": recall, "candidate_precision_proxy": precision,
            "random_expectation": area_frac, "lift": precision / area_frac if area_frac else np.nan,
            "match_km": match_km, "n_candidates": int(len(cands)),
            "n_unmapped": int((cands.status == "unmapped").sum()) if len(cands) else 0}
