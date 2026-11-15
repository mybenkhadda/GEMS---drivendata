import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from gems_bench.g12.deep import (
    build, channel_indices, derive, ensure_grid, masks, submit, smooth,
)


def raster(path, array, descriptions=(), nodata=np.nan):
    array=np.asarray(array, dtype="float32")
    if array.ndim==2:array=array[None]
    profile=dict(driver="GTiff",width=array.shape[-1],height=array.shape[-2],
                 count=array.shape[0],dtype="float32",crs="EPSG:32611",
                 transform=from_origin(500000,4200000,100,100),nodata=nodata)
    with rasterio.open(path,"w",**profile) as dst:
        dst.write(array)
        for i,n in enumerate(descriptions,1):dst.set_band_description(i,n)
    return path


def test_signatures_detect_modalities_and_orientations():
    yy,xx=np.indices((180,180))
    raw=np.stack([np.tanh((xx-yy)/5),np.tanh((xx-yy-5)/7),
                  500+0.25*np.sin(xx/3)]).astype("float32")
    raw[2,20:60,20:40]=np.nan
    sig,groups=derive(raw,["rtp","iso_grav_anom","det_elev"])
    assert len(sig)>=45
    assert any(k.startswith("cross_mag_grav_") for k in sig)
    assert set(groups.values()) >= {"mag","grav","terrain","cross"}
    for v in sig.values():
        assert v.shape==(180,180)
        assert not np.isinf(v).any()
    key="cross_mag_grav_axial_alignment"
    assert np.nanmedian(sig[key][65:125,65:125])>.9


def test_smoothing_ignores_nodata():
    x=np.ones((96,96),dtype="float32")
    ok=np.ones((96,96),bool);ok[:,45:48]=False
    x[~ok]=9999999
    sm,support=smooth(x,ok,6)
    assert np.max(np.abs(sm-1))<1.e-3
    assert support.min()<1 and support.max()<=1.00001


def test_build_tile_schema_cache_and_submit(tmp_path):
    h=w=120;yy,xx=np.indices((h,w))
    base=np.stack([np.sin(xx/7)+np.cos(yy/5),np.sin((xx+yy)/9)]).astype("float32")
    raw=raster(tmp_path/"raw.tif",base,["rtp","iso_grav_anom"])
    labels=np.zeros((h,w),dtype="float32");labels[30:90,60]=1
    known=raster(tmp_path/"known.tif",labels,nodata=None)
    template=np.zeros((h,w),dtype="float32");template[:7]=np.nan
    temp=raster(tmp_path/"template.tif",template)
    manifest=build(raw,tmp_path/"derived.tif",block=64,halo=96)
    assert len(manifest["names"])>30
    assert build(raw,tmp_path/"derived.tif",block=64,halo=96)["stamp"]==manifest["stamp"]
    with rasterio.open(tmp_path/"derived.tif") as r:
        assert r.count==len(manifest["names"])
        assert np.isfinite(r.read(1)[40:100,40:100]).all()
    pred=raster(tmp_path/"pred.tif",np.full((h,w),.75,dtype="float32"))
    result=submit(pred,known,temp,tmp_path/"submission.tif",min_conf=.5)
    with rasterio.open(result["path"]) as f:
        a=f.read(1)
        assert np.isnan(a[:7]).all()
        assert (a[30:90,60]==0).all()
        assert np.isclose(a[10,10],.75)


def test_halo_rejects_seams(tmp_path):
    x=np.ones((100,100),dtype="float32")
    r=raster(tmp_path/"raw.tif",np.stack([x,x]),["rtp","iso_grav_anom"])
    with pytest.raises(ValueError,match="halo"):
        build(r,tmp_path/"bad.tif",halo=12)


def test_projection_assertion(tmp_path):
    r=raster(tmp_path/"a.tif",np.ones((12,12)))
    with rasterio.open(r) as f:ensure_grid(f)
    with rasterio.open(r,"r+") as f:f.crs="EPSG:4326"
    with rasterio.open(r) as f:
        with pytest.raises(ValueError,match="EPSG"):
            ensure_grid(f)


def test_spatial_split_no_overlap_and_embargo(tmp_path):
    from scipy.ndimage import distance_transform_edt
    h=w=384
    yy,xx=np.indices((h,w))
    r=raster(tmp_path/"r.tif",np.stack([np.sin(xx/6),np.cos(yy/7)]),["rtp","iso_grav_anom"])
    known=np.zeros((h,w),np.float32)
    known[20:360:3,20:360:3]=1
    k=raster(tmp_path/"k.tif",known,nodata=None)
    t=raster(tmp_path/"t.tif",np.zeros((h,w),np.float32),nodata=None)
    split=masks(r,k,t,seed=1,block=64,embargo=16)
    for a,b in (("train","val"),("train","test"),("val","test")):
        assert not (split[a]&split[b]).any()
    assert np.min(distance_transform_edt(~(split["val"]|split["test"]))[split["train"]])>16