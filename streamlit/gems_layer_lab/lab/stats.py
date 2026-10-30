"""Exploratory spatial statistics: pixels are NOT statistically independent."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.ndimage import uniform_filter, distance_transform_edt


def correlation_matrices(layers: dict, names: list[str], mask=None, max_pixels=25000,
                         seed=2026, block_px=1):
    """Return Pearson / Spearman / pairwise-N and sampled pixel table.

    For block_px>1, choose at most one valid raster location per block to reduce
    adjacency overrepresentation. No independence or significance claim.
    """
    if not names:
        raise ValueError("Choose at least one layer")
    shape = layers[names[0]].shape
    valid = np.ones(shape, bool) if mask is None else np.asarray(mask, bool)
    valid &= np.logical_and.reduce([np.isfinite(layers[k]) for k in names])
    rng = np.random.default_rng(seed)
    if block_px > 1:
        pts = []
        for row in range(0, shape[0], block_px):
            for col in range(0, shape[1], block_px):
                rr, cc = np.where(valid[row:row+block_px, col:col+block_px])
                if len(rr):
                    j = rng.integers(len(rr))
                    pts.append((row + rr[j], col + cc[j]))
        locs = np.array(pts, dtype=int).reshape((-1, 2))
    else:
        locs = np.column_stack(np.where(valid))
    if len(locs) > max_pixels:
        locs = locs[rng.choice(len(locs), max_pixels, replace=False)]
    if not len(locs):
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(columns=["row", "col"]), 0)
    table = pd.DataFrame({k: layers[k][locs[:, 0], locs[:, 1]] for k in names})
    table.insert(0, "col", locs[:, 1])
    table.insert(0, "row", locs[:, 0])
    df = table[names]
    return df.corr(method="pearson"), df.corr(method="spearman"), pd.DataFrame(len(table), index=names, columns=names, dtype=int), table, len(locs)


def local_corr(x, y, window_px=21, min_valid_share=0.6):
    """NaN-aware moving-window Pearson correlation and valid pixel count."""
    if window_px < 3 or window_px % 2 == 0:
        raise ValueError("Moving window must be odd and at least three pixels")
    x = np.asarray(x, np.float32)
    y = np.asarray(y, np.float32)
    good = np.isfinite(x) & np.isfinite(y)
    fx = np.where(good, x, 0).astype(np.float64)
    fy = np.where(good, y, 0).astype(np.float64)
    f = lambda a: uniform_filter(a, size=window_px, mode="constant", cval=0) * window_px**2
    n = f(good.astype(np.float64))
    sx, sy = f(fx), f(fy)
    xx, yy, xy = f(fx * fx), f(fy * fy), f(fx * fy)
    vx = np.maximum(xx - sx * sx / np.maximum(n, 1), 0)
    vy = np.maximum(yy - sy * sy / np.maximum(n, 1), 0)
    cov = xy - sx * sy / np.maximum(n, 1)
    result = np.divide(cov, np.sqrt(vx * vy), out=np.full_like(cov, np.nan), where=(vx > 1e-14) & (vy > 1e-14))
    result[n < min_valid_share * window_px**2] = np.nan
    return np.clip(result, -1, 1).astype(np.float32), n.astype(np.int32)


def grid_correlations(x, y, grid=5, method="pearson", max_per_cell=5000, seed=2026):
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    h,w = x.shape
    out = []
    corr_map = np.full((grid, grid), np.nan, np.float32)
    for ri in range(grid):
        for ci in range(grid):
            r0,r1 = h*ri//grid, h*(ri+1)//grid
            c0,c1 = w*ci//grid, w*(ci+1)//grid
            a,b = x[r0:r1,c0:c1],y[r0:r1,c0:c1]
            good = np.isfinite(a)&np.isfinite(b)
            aa,bb = a[good],b[good]
            total_n = len(aa)
            if len(aa)>max_per_cell:
                idx = rng.choice(len(aa), max_per_cell,replace=False)
                aa,bb = aa[idx],bb[idx]
            corr = np.nan
            if len(aa)>=8 and np.std(aa)>1e-12 and np.std(bb)>1e-12:
                corr = float(np.corrcoef(aa,bb)[0,1]) if method=="pearson" else float(spearmanr(aa,bb).statistic)
            corr_map[ri,ci] = corr
            out.append({"row_tile":ri,"col_tile":ci,"r0":r0,"r1":r1,"c0":c0,"c1":c1,
                        "n_pixels":total_n,"sample_n":len(aa),"correlation":corr})
    return corr_map, pd.DataFrame(out)


def fault_exploration(layer, faults, pixel_m=100, min_background_km=2.0,
                      sample_size=20000, seed=2026):
    """Descriptive fault-vs-far-unlabelled distribution; NOT a supervised metric."""
    from scipy.stats import mannwhitneyu
    if faults is None:
        return {"error":"No labels loaded"},pd.DataFrame()
    x = np.asarray(layer, np.float32)
    finite=np.isfinite(x)
    fault=faults&finite
    distance=distance_transform_edt(~faults)*pixel_m/1000.0
    background=finite & (distance>=float(min_background_km))
    rng=np.random.default_rng(seed)
    a=x[fault];b=x[background]
    if len(a)>sample_size: a=a[rng.choice(len(a),sample_size,replace=False)]
    if len(b)>sample_size: b=b[rng.choice(len(b),sample_size,replace=False)]
    if len(a)<8 or len(b)<8:
        return {"error":"Insufficient mapped-fault or distant-unlabeled pixels in this AOI",
                "fault_n":len(a),"unlabelled_n":len(b)},pd.DataFrame()
    # U/(n1*n2) is rank separation probability, not confirmed nonfault AUC.
    stat=mannwhitneyu(a,b,alternative="two-sided",method="asymptotic").statistic
    p=float(stat/(len(a)*len(b)))
    desc={"fault_n":len(a),"unlabelled_n":len(b),"fault_median":float(np.median(a)),
          "unlabelled_median":float(np.median(b)),"rank_separation":p,
          "abs_separation_from_random":abs(p-0.5),"background_min_km":min_background_km}
    points=pd.concat([pd.DataFrame({"value":a,"category":"Mapped fault"}),
                      pd.DataFrame({"value":b,"category":"Distant unlabeled"})],ignore_index=True)
    return desc,points
