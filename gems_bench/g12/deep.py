"""G12 deep geological features, spatial ablations, error atlas, and submission.

Raw inputs are never overwritten. Training labels are NEVER inputs to signatures.
The offline DTI on withheld *known* faults is a proxy, not an official score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage as ndi

# Exact names used by GeoDAWN numerical-feature GeoTIFF. Optional fields are skipped.
ALIASES = {
    "mag": ("rtp", "mag_anom", "tmi"),
    "grav": ("iso_grav_anom", "iso_grav_anom_hg", "iso_grav_anom_vg"),
    "cond": ("cond_surf",),
    "terrain": ("det_elev", "det_elev_slope"),
    "depth": ("depth_to_base_surf",),
}
FEATURE_VERSION = "g12_deep_v2"
SCALES = (2, 6, 12)  # 200, 600, 1200m on 100m cells
HALO = 96


def channel_indices(descriptions):
    names = [str(x or "").strip().lower() for x in descriptions]
    return {family: next(((i, n) for n in options for i, x in enumerate(names) if x == n), None)
            for family, options in ALIASES.items()}


def ensure_grid(*sources):
    a = sources[0]
    for b in sources[1:]:
        if (a.width, a.height) != (b.width, b.height) or a.transform != b.transform or a.crs != b.crs:
            raise ValueError(f"Raster mismatch: {a.name} and {b.name}; no silent reprojection")
    if a.crs is None or a.crs.to_epsg() != 32611 or not (99.9 <= abs(a.transform.a) <= 100.1):
        raise ValueError("GEMS GeoDAWN must use EPSG:32611 at 100m spacing")
    if not (99.9 <= abs(a.transform.e) <= 100.1):
        raise ValueError("GEMS GeoDAWN requires 100m pixel height")


def smooth(x, valid, sigma):
    """Mask-normalized smoothing: avoid nodata seeping into geophysical gradients."""
    support = ndi.gaussian_filter(valid.astype(np.float32), sigma, mode="reflect", truncate=3)
    numerator = ndi.gaussian_filter(np.where(valid, x, 0).astype(np.float32), sigma,
                                    mode="reflect", truncate=3)
    return numerator / np.maximum(support, 1.e-5), support


def frame_features(x, valid, scale, pixel_m):
    """Local geology: contrast, gradient, tilt, ridge, structural tensor, support."""
    a, coverage = smooth(x, valid, scale)
    broad, _ = smooth(x, valid, scale * 2)
    fluct = a - broad
    variance = ndi.gaussian_filter(fluct ** 2, scale, mode="reflect", truncate=3)
    z = np.clip(fluct / np.maximum(np.sqrt(variance), 1.e-4), -6, 6)
    gy, gx = np.gradient(a, pixel_m)
    amp = np.hypot(gx, gy)
    gy2, gx2 = np.gradient(gy, pixel_m)[0], np.gradient(gx, pixel_m)[1]
    lap = gy2 + gx2
    jxx = ndi.gaussian_filter(gx * gx, max(2, scale // 2), mode="reflect")
    jxy = ndi.gaussian_filter(gx * gy, max(2, scale // 2), mode="reflect")
    jyy = ndi.gaussian_filter(gy * gy, max(2, scale // 2), mode="reflect")
    coh = np.hypot(jxx - jyy, 2 * jxy) / np.maximum(jxx + jyy, 1.e-12)
    theta = np.arctan2(gy, gx)
    # Tilt angle highlights edges over different anomaly amplitudes.
    tilt = np.arctan2(np.hypot(gx, gy), np.maximum(np.abs(lap) * pixel_m, 1.e-6))
    # Cross-contact asymmetry: compare values on both sides of the local normal.
    rr, cc = np.indices(x.shape, dtype=np.float32)
    normal_x, normal_y = gx / np.maximum(amp, 1.e-6), gy / np.maximum(amp, 1.e-6)
    distance = float(max(2, scale))
    left = ndi.map_coordinates(a, [rr-distance*normal_y, cc-distance*normal_x],
                               order=1, mode="nearest")
    right = ndi.map_coordinates(a, [rr+distance*normal_y, cc+distance*normal_x],
                                order=1, mode="nearest")
    across = np.arcsinh((right-left)/np.maximum(np.sqrt(variance), 1.e-4))
    return {"z": z, "edge": np.log1p(100 * amp), "coh": np.clip(coh, 0, 1),
            "lap": np.arcsinh(1.e4 * lap), "tilt": tilt, "across": across,
            "theta": theta, "support": coverage, "amp": amp}


def derive(patch, descriptions, pixel_m=100., scales=SCALES):
    """Build a deterministic dict of physically motivated raster features.

    Input: multiband float [bands, rows, cols], with np.nan for missing cells.
    Output: feature_names -> float32 arrays and feature_family per feature.
    """
    idx = channel_indices(descriptions)
    if idx["mag"] is None and idx["grav"] is None:
        raise ValueError("Expected at least magnetic or gravity bands in source descriptions")
    out, groups, stats = {}, {}, {}
    for fam, selected in idx.items():
        if selected is None:
            continue
        band, label = selected
        x = np.asarray(patch[band], dtype=np.float32)
        ok = np.isfinite(x) & (np.abs(x) < 1.e20)
        if not ok.any():
            # Schema remains identical on fully nodata windows; mask normalized
            # convolutions receive zeros and downstream outputs stay NaN.
            x = np.zeros_like(x)
        s = {}
        for radius in scales:
            f = frame_features(x, ok, radius, pixel_m)
            s[radius] = f
            for key in ("z", "edge", "coh", "lap", "tilt", "across"):
                name = f"{fam}_{label}_{key}_s{radius}"
                value = f[key].astype(np.float32)
                value[~ok] = np.nan
                out[name], groups[name] = value, fam
        # An axial orientation persistence score, invariant to +/-180 degree direction.
        if len(scales) >= 2:
            left, right = s[scales[0]], s[scales[-1]]
            d = left["theta"] - right["theta"]
            stability = (0.5 + 0.5 * np.cos(2 * d)) * np.minimum(left["coh"], right["coh"])
            name = f"{fam}_{label}_orientation_persistence"
            out[name], groups[name] = np.where(ok, stability, np.nan).astype(np.float32), fam
        name = f"{fam}_{label}_coverage"
        out[name], groups[name] = ok.astype(np.float32), fam
        # Local support fraction distinguishes genuine geological edges from DEM tile boundaries.
        name = f"{fam}_{label}_local_validity"
        out[name], groups[name] = s[scales[-1]]["support"].astype(np.float32), fam
        stats[fam] = {"ok": ok, "small": s[scales[0]], "large": s[scales[-1]]}
    for a, b in (("mag", "grav"), ("mag", "cond"), ("grav", "cond"), ("mag", "terrain")):
        if a not in stats or b not in stats:
            continue
        valid = stats[a]["ok"] & stats[b]["ok"]
        A, B = stats[a]["small"], stats[b]["small"]
        alignment = np.cos(A["theta"] - B["theta"]) ** 2
        edge_joint = np.sqrt(np.maximum(A["edge"] * B["edge"], 0)) * alignment
        name = f"cross_{a}_{b}_axial_alignment"
        out[name], groups[name] = np.where(valid, alignment, np.nan).astype(np.float32), "cross"
        name = f"cross_{a}_{b}_edge_coincidence"
        out[name], groups[name] = np.where(valid, edge_joint, np.nan).astype(np.float32), "cross"
        # Sign disagreement helps identify contacts masked by single-sensor strength.
        name = f"cross_{a}_{b}_local_discordance"
        out[name], groups[name] = np.where(valid, np.abs(A["z"] - B["z"]), np.nan).astype(np.float32), "cross"
    return out, groups


def path_stamp(path):
    p = Path(path)
    s = p.stat()
    return {"name": p.name, "size": s.st_size, "mtime_ns": s.st_mtime_ns}


def build(raw_path, out_path, *, block=256, halo=HALO, force=False):
    """Atomic, cache-verified, haloed feature raster creation (no labels involved)."""
    raw_path, out_path = Path(raw_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path.with_suffix(".manifest.json")
    stamp = {"version": FEATURE_VERSION, "source": path_stamp(raw_path), "scales": SCALES,
             "halo": halo, "block": block}
    if out_path.is_file() and manifest_path.exists() and not force:
        old = json.loads(manifest_path.read_text())
        if old.get("stamp") == stamp:
            with rasterio.open(raw_path) as r, rasterio.open(out_path) as o:
                ensure_grid(r, o)
                if o.count == len(old["names"]) and list(o.descriptions) == old["names"]:
                    return old
    if halo < 6 * max(SCALES):
        raise ValueError("Increase halo: multiscale smoothing and derivatives need padding")
    with rasterio.open(raw_path) as src:
        ensure_grid(src)
        descriptions = list(src.descriptions)
        probe = src.read(window=Window(0, 0, min(src.width, 128), min(src.height, 128)),
                         masked=True).filled(np.nan).astype(np.float32)
        features, groups = derive(probe, descriptions)
        names = list(features)
        profile = src.profile.copy()
        profile.update(driver="GTiff", count=len(names), dtype="float32", nodata=np.nan,
                       compress="deflate", predictor=3, tiled=True, blockxsize=256,
                       blockysize=256, BIGTIFF="IF_SAFER")
        tmp = out_path.with_suffix(".partial.tif")
        if tmp.exists():
            tmp.unlink()
        with rasterio.open(tmp, "w", **profile) as dst:
            for i, name in enumerate(names, 1):
                dst.set_band_description(i, name)
            for row in range(0, src.height, block):
                for col in range(0, src.width, block):
                    height, width = min(block, src.height-row), min(block, src.width-col)
                    r0, c0 = max(0, row-halo), max(0, col-halo)
                    r1, c1 = min(src.height, row+height+halo), min(src.width, col+width+halo)
                    window = Window(c0, r0, c1-c0, r1-r0)
                    patch = src.read(window=window, masked=True).filled(np.nan).astype(np.float32)
                    generated, _ = derive(patch, descriptions)
                    if list(generated) != names:
                        raise RuntimeError("Feature schema changed across tiles; examine missing bands")
                    rr, cc = row-r0, col-c0
                    for i, name in enumerate(names, 1):
                        dst.write(generated[name][rr:rr+height, cc:cc+width], i,
                                  window=Window(col, row, width, height))
        os.replace(tmp, out_path)
    manifest = {"stamp": stamp, "names": names, "groups": groups,
                "source": str(raw_path), "derived": str(out_path), "created_unix": time.time()}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def masks(raw_path, known_path, template_path, seed=2026, block=256, embargo=64):
    """Spatial blocks with 6.4km embargo; no random pixel validation leakage."""
    with rasterio.open(raw_path) as raw, rasterio.open(known_path) as known, rasterio.open(template_path) as template:
        ensure_grid(raw, known, template)
        valid = template.read_masks(1).astype(bool) & raw.read_masks(1).astype(bool)
        faults = (known.read(1, masked=True).filled(0) > 0) & valid
    h, w = valid.shape
    rng = np.random.default_rng(seed)
    nrow, ncol = math.ceil(h/block), math.ceil(w/block)
    region = rng.choice(3, size=(nrow, ncol), p=(.65, .2, .15))
    region = np.repeat(np.repeat(region, block, axis=0), block, axis=1)[:h, :w]
    val = (region == 1) & valid
    test = (region == 2) & valid
    if not (faults & val).any() or not (faults & test).any():
        raise RuntimeError("Spatial holdouts have no known faults; try a different seed")
    # A hole around all held-out blocks. Withheld labels never enter feature engineering.
    protected = (region != 0)
    train = (region == 0) & valid & (ndi.distance_transform_edt(~protected) > embargo)
    if (faults & train).sum() < 20:
        raise RuntimeError("Insufficient positive examples after embargo")
    return {"train": train, "val": val, "test": test, "valid": valid, "faults": faults}


def pick_samples(train, faults, rng, cap=80000, negative_ratio=5):
    """Sample training positives/negatives; unlabeled != guaranteed negative."""
    target = ndi.binary_dilation(faults, iterations=2) & train
    pi = np.flatnonzero(target)
    ni = np.flatnonzero(train & ~target)
    if pi.size == 0 or ni.size == 0:
        raise ValueError("Training split has insufficient positives or negatives")
    p = rng.choice(pi, size=min(cap, pi.size), replace=False)
    n = rng.choice(ni, size=min(negative_ratio*len(p), len(ni)), replace=False)
    ids = np.concatenate((p, n)); rng.shuffle(ids)
    return ids, target.ravel()[ids].astype(np.uint8)


def pixel_matrix(raw_path, feature_path, ids, extra_names=(), block=256):
    """Read only pixel windows touched by sample indices; no full cube in RAM."""
    with rasterio.open(raw_path) as raw, rasterio.open(feature_path) as feat:
        ensure_grid(raw, feat)
        names = list(feat.descriptions)
        bands = [names.index(n)+1 for n in extra_names]
        n_features = raw.count + len(bands)
        matrix = np.empty((len(ids), n_features), dtype=np.float32)
        r, c = ids // raw.width, ids % raw.width
        tiles_per_row = math.ceil(raw.width / block)
        tiles = (r//block)*tiles_per_row + (c//block)
        order = np.argsort(tiles, kind="stable")
        boundaries = np.r_[0, np.flatnonzero(np.diff(tiles[order]))+1, len(order)]
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            ix = order[begin:end]
            row, col = int((r[ix[0]]//block)*block), int((c[ix[0]]//block)*block)
            win = Window(col, row, min(block, raw.width-col), min(block, raw.height-row))
            y, x = r[ix]-row, c[ix]-col
            a = raw.read(window=win, masked=True).filled(np.nan)
            matrix[ix, :raw.count] = a[:, y, x].T
            if bands:
                b = feat.read(bands, window=win, masked=True).filled(np.nan)
                matrix[ix, raw.count:] = b[:, y, x].T
    return np.where(np.isfinite(matrix), matrix, np.nan)


def predict(model, raw_path, feature_path, extra_names, dest, *, block=256):
    """Memory-bounded whole-map inference. Output is a probability GeoTIFF."""
    from xgboost import XGBClassifier
    if not isinstance(model, XGBClassifier):
        raise TypeError("Expected fitted XGBClassifier")
    with rasterio.open(raw_path) as raw, rasterio.open(feature_path) as features:
        ensure_grid(raw, features)
        bands = [list(features.descriptions).index(n)+1 for n in extra_names]
        profile = raw.profile.copy()
        profile.update(count=1, dtype="float32", nodata=np.nan, compress="deflate", predictor=3)
        dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".partial.tif")
        with rasterio.open(tmp, "w", **profile) as dst:
            for row in range(0, raw.height, block):
                for col in range(0, raw.width, block):
                    h, w = min(block, raw.height-row), min(block, raw.width-col)
                    win = Window(col, row, w, h)
                    x = raw.read(window=win, masked=True).filled(np.nan).reshape(raw.count,-1).T
                    if bands:
                        z = features.read(bands, window=win, masked=True).filled(np.nan).reshape(len(bands),-1).T
                        x = np.concatenate((x,z),axis=1)
                    x = np.where(np.isfinite(x), x, np.nan).astype(np.float32)
                    p = model.predict_proba(x)[:,1].reshape(h,w).astype(np.float32)
                    p[raw.read_masks(1, window=win)==0] = np.nan
                    dst.write(p,1,window=win)
        os.replace(tmp,dest)
    return str(dest)


def ablations(raw_path, feature_path, known_path, template_path, out_dir,
              *, seed=2026, groups=("raw", "mag", "grav", "cross", "all"), cap=80000,
              split_block=256, embargo=64, n_estimators=280):
    """Controlled spatially disjoint ablations + full-map predictions and proxy DTI."""
    from xgboost import XGBClassifier
    from sklearn.metrics import average_precision_score, brier_score_loss
    from gems_bench.metric import dti_components, OFFICIAL_EXCLUDE_SUPPORTS_TP
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    split = masks(raw_path, known_path, template_path, seed=seed,
                  block=split_block, embargo=embargo)
    rng=np.random.default_rng(seed)
    train_ids, y=pick_samples(split["train"],split["faults"],rng,cap=cap)
    val_ids=np.flatnonzero(split["val"])
    if len(val_ids) > 250000:
        val_ids=rng.choice(val_ids,size=250000,replace=False)
    # Evaluate buffered-label PR-AUC as a diagnostic only; validation DTI uses centerlines.
    val_target = ndi.binary_dilation(split["faults"], iterations=2).ravel()[val_ids].astype(np.uint8)
    with rasterio.open(feature_path) as f:
        all_names=list(f.descriptions)
    manifest=json.loads(Path(feature_path).with_suffix(".manifest.json").read_text())
    fam=manifest["groups"]
    experiments={"raw": []}
    for family in ("mag", "grav", "cond", "terrain", "depth", "cross"):
        experiments[family]=[n for n in all_names if fam[n]==family]
    experiments["all"]=all_names
    rows=[]
    for group in groups:
        if group not in experiments: raise ValueError(f"Unknown family {group}")
        names=experiments[group]
        if group != "raw" and not names: continue
        X=pixel_matrix(raw_path,feature_path,train_ids,names)
        V=pixel_matrix(raw_path,feature_path,val_ids,names)
        clf=XGBClassifier(n_estimators=n_estimators,max_depth=5,learning_rate=.065,subsample=.8,
                          colsample_bytree=.8,tree_method="hist",n_jobs=4,
                          random_state=seed,eval_metric="logloss",missing=np.nan)
        started=time.monotonic();clf.fit(X,y)
        pp=clf.predict_proba(V)[:,1]
        model_dir=out_dir/"models";model_dir.mkdir(exist_ok=True)
        model_file=model_dir/f"{group}.json";clf.save_model(model_file)
        raster=out_dir/"predictions"/f"{group}.tif"
        predict(clf,raw_path,feature_path,names,raster)
        with rasterio.open(raster) as p:
            pred=p.read(1)
        dti=dti_components(pred,split["faults"],eval_mask=split["val"],
                           exclude_supports_tp=OFFICIAL_EXCLUDE_SUPPORTS_TP)
        row={"model":group,"features":len(names)+X.shape[1]-len(names),
             "extra_features":len(names),"train_n":len(y),"val_n":len(val_ids),
             "val_pr_auc_buffered_sample":float(average_precision_score(val_target,pp)),
             "val_brier_buffered_sample":float(brier_score_loss(val_target,pp)),
             "val_dti_known_fault_proxy":dti.dti,"val_tp":dti.tp,"val_fp":dti.fp,
             "val_fn":dti.fn,"seconds":round(time.monotonic()-started,1),
             "prediction_file":str(raster),"model_file":str(model_file)}
        rows.append(row)
        (out_dir/"ablation_results.json").write_text(json.dumps(rows,indent=2))
    return rows


def error_atlas(raw_path, feature_path, known_path, template_path, pred_path, out_dir,
                *, seed=2026, downsample=6, split_block=256, embargo=64):
    """Generate evidence maps and failure archetypes from a held-out geographic region."""
    import matplotlib.pyplot as plt
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    m=masks(raw_path,known_path,template_path,seed=seed,
            block=split_block,embargo=embargo)
    with rasterio.open(pred_path) as src, rasterio.open(feature_path) as ff:
        ensure_grid(src,ff)
        p=src.read(1)
        names=list(ff.descriptions)
        contrast=[]
        near=ndi.binary_dilation(m["faults"],iterations=2)&m["val"]
        far=(ndi.distance_transform_edt(~m["faults"])>10)&m["val"]
        rng=np.random.default_rng(seed)
        ai=np.flatnonzero(near);bi=np.flatnonzero(far)
        ai=rng.choice(ai,min(12000,len(ai)),replace=False)
        bi=rng.choice(bi,min(30000,len(bi)),replace=False)
        if not len(ai) or not len(bi):raise ValueError("Insufficient faults/background in validation region")
        for i,name in enumerate(names,1):
            arr=ff.read(i)
            pos=arr.ravel()[ai];neg=arr.ravel()[bi]
            pos=pos[np.isfinite(pos)];neg=neg[np.isfinite(neg)]
            if len(pos)<20 or len(neg)<20:continue
            med_pos=float(np.median(pos));med_neg=float(np.median(neg))
            pooled=float(np.median(np.abs(neg-med_neg)))
            contrast.append({"feature":name,"median_fault":med_pos,"median_background":med_neg,
                             "robust_contrast":round((med_pos-med_neg)/max(pooled*1.4826,1.e-6),4),
                             "n_fault":len(pos),"n_background":len(neg)})
    contrast.sort(key=lambda x:abs(x["robust_contrast"]),reverse=True)
    (out_dir/"feature_contrast.json").write_text(json.dumps(contrast,indent=2))
    # Distinguish false-positive lineaments from actual truth support; absence of a known fault is NOT proof of error.
    fig,axs=plt.subplots(2,2,figsize=(12,11),layout="constrained")
    maps=[("Validation probability",np.where(m["val"],p,np.nan)),
          ("Mapped fault traces",np.where(m["val"],m["faults"].astype(float),np.nan)),
          ("Strong unmapped candidates (unverified)",np.where(m["val"]&~ndi.binary_dilation(m["faults"],iterations=4),p,np.nan)),
          ("Known-fault misses",np.where(m["val"]&m["faults"],1.-p,np.nan))]
    for ax,(title,arr) in zip(axs.flat,maps):
        im=ax.imshow(arr[::downsample,::downsample],vmin=0,vmax=1,cmap="magma",interpolation="nearest")
        ax.set_title(title);ax.set_axis_off();fig.colorbar(im,ax=ax,shrink=.65)
    path=out_dir/"fault_evidence_atlas.png";fig.savefig(path,dpi=140);plt.close(fig)
    return {"contrast_file":str(out_dir/"feature_contrast.json"),"map_file":str(path)}


def submit(pred_path,known_path,template_path,out_path,*,min_conf=0.,scale=1.):
    """Strict GeoTIFF: 100m EPSG:32611, float32 [0,1], known faults exact-zero, outside NaN."""
    out_path=Path(out_path);out_path.parent.mkdir(parents=True,exist_ok=True)
    with rasterio.open(pred_path) as p,rasterio.open(known_path) as k,rasterio.open(template_path) as t:
        ensure_grid(p,k,t)
        profile=t.profile.copy();profile.update(driver="GTiff",count=1,dtype="float32",nodata=np.nan,
                                                 compress="deflate",predictor=3)
        tmp=out_path.with_suffix(".partial.tif")
        with rasterio.open(tmp,"w",**profile) as dst:
            for _,win in t.block_windows(1):
                valid=t.read_masks(1,window=win)>0
                known=(k.read(1,window=win,masked=True).filled(0)>0)
                a=p.read(1,window=win)
                a=np.clip(np.nan_to_num(a,nan=0.,posinf=0.,neginf=0.)*scale,0,1)
                a[a<min_conf]=0.
                a[known]=0.
                a[~valid]=np.nan
                dst.write(a.astype("float32"),1,window=win)
        os.replace(tmp,out_path)
    with rasterio.open(out_path) as o,rasterio.open(template_path) as t,rasterio.open(known_path) as k:
        ensure_grid(o,t,k)
        assert o.count==1 and o.dtypes[0]=="float32"
        positive=0;mass=0.
        for _,win in o.block_windows(1):
            a=o.read(1,window=win);valid=t.read_masks(1,window=win)>0
            kk=(k.read(1,window=win,masked=True).filled(0)>0)
            if np.any(np.isfinite(a[~valid])) or np.any(~np.isfinite(a[valid])):
                raise AssertionError("Invalid nodata geometry in submission")
            if np.any(a[valid]<0) or np.any(a[valid]>1) or np.any(a[valid&kk]!=0):
                raise AssertionError("Submission out of bounds or known-fault mask violated")
            positive+=int(np.count_nonzero(a[valid]>0));mass+=float(a[valid].sum())
    return {"path":str(out_path),"positive_pixels":positive,"prediction_mass":mass}


def run(raw,known,template,out,*,seed=2026,groups=("raw","mag","grav","cross","all"),
        cap=80000,split_block=256,embargo=64,n_estimators=280):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with rasterio.open(raw) as r,rasterio.open(known) as k,rasterio.open(template) as t:
        ensure_grid(r,k,t)
    man=build(raw,out/"physics_g12.tif")
    rows=ablations(raw,out/"physics_g12.tif",known,template,out,seed=seed,groups=groups,cap=cap,
                   split_block=split_block,embargo=embargo,n_estimators=n_estimators)
    if not rows:raise RuntimeError("No models trained")
    champion=max(rows,key=lambda r:r["val_dti_known_fault_proxy"])
    atlas=error_atlas(raw,out/"physics_g12.tif",known,template,champion["prediction_file"],out/"atlas",
                      seed=seed,split_block=split_block,embargo=embargo)
    # Evaluate ONCE on a geographically held-out test partition AFTER model selection.
    # This label set is public known faults, so the resulting DTI is ONLY a proxy.
    from gems_bench.metric import dti_components, OFFICIAL_EXCLUDE_SUPPORTS_TP
    split=masks(raw,known,template,seed=seed,block=split_block,embargo=embargo)
    with rasterio.open(champion["prediction_file"]) as pp:
        test_dti=dti_components(pp.read(1),split["faults"],eval_mask=split["test"],
                                exclude_supports_tp=OFFICIAL_EXCLUDE_SUPPORTS_TP).as_dict()
    sub=submit(champion["prediction_file"],known,template,out/"submission_g12_deep.tif")
    report={"schema":FEATURE_VERSION,"feature_count":len(man["names"]),"ablation":rows,
            "champion_by_proxy":champion["model"],"heldout_test_dti_known_fault_proxy":test_dti,
            "atlas":atlas,"submission":sub,
            "warning":"Offline validation uses withheld KNOWN faults. It is not the private test/official DTI; do not infer true novelty."}
    (out/"report.json").write_text(json.dumps(report,indent=2))
    return report


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw",required=True);ap.add_argument("--known",required=True)
    ap.add_argument("--template",required=True);ap.add_argument("--out",required=True)
    ap.add_argument("--seed",type=int,default=2026)
    ap.add_argument("--groups",default="raw,mag,grav,cross,all")
    ap.add_argument("--cap",type=int,default=80000)
    ap.add_argument("--split-block",type=int,default=256)
    ap.add_argument("--embargo",type=int,default=64)
    ap.add_argument("--n-estimators",type=int,default=280)
    args=ap.parse_args(argv)
    result=run(args.raw,args.known,args.template,args.out,seed=args.seed,
               groups=tuple(s.strip() for s in args.groups.split(",")),cap=args.cap,
               split_block=args.split_block,embargo=args.embargo,n_estimators=args.n_estimators)
    print(json.dumps({"champion":result["champion_by_proxy"],"submission":result["submission"]},indent=2))


if __name__=="__main__":
    main()