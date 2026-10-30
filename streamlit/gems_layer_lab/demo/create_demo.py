"""Create a small synthetic GEMS-shaped geospatial demonstration (not real geology)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from rasterio.transform import from_origin
import rasterio

BANDS = [
    "mag_anom", "rtp", "tmi", "tmi_hg", "tmi_vg", "tc",
    "iso_grav_anom", "iso_grav_anom_hg", "iso_grav_anom_vg", "iso_grav_anom_slope",
    "det_elev", "det_elev_slope", "depth_to_base_surf", "cond_surf",
    "geod_2ndinv", "geod_shearrate", "geod_dilaterate", "deq_n100a15", "ieq_n100a15"
]


def generate_demo(root: Path):
    root=Path(root)
    features=root/"data/raw/geodawn/gems-geodawn-numerical-features.tif"
    faults=root/"data/raw/faults/existing_faults.tif"
    features.parent.mkdir(parents=True,exist_ok=True)
    faults.parent.mkdir(parents=True,exist_ok=True)
    h,w=400,420
    rng=np.random.default_rng(2026)
    r,c=np.mgrid[0:h,0:w]
    # Two synthetic fault-like traces and smooth geophysical context.
    f1 = abs(c-(80+0.42*r+12*np.sin(r/48)))<1.6
    f2 = abs(r-(295-0.34*c+8*np.sin(c/39)))<1.4
    fault=(f1|f2).astype(np.uint8)
    trend = gaussian_filter(rng.normal(size=(h,w)).astype(np.float32),16)
    ridge = np.tanh((c-(80+0.42*r+12*np.sin(r/48)))/9)
    ridge2=np.tanh((r-(295-0.34*c+8*np.sin(c/39)))/9)
    mag=1.8*ridge+1.2*trend+rng.normal(0,.35,(h,w))
    grav=.7*ridge2+.3*ridge + gaussian_filter(rng.normal(size=(h,w)),30)
    elev=800 + .3*r + .15*c + 12*ridge2 + 15*ridge + 10*trend
    dy,dx=np.gradient(elev)
    mag_y,mag_x=np.gradient(mag)
    gy,gx=np.gradient(grav)
    dat={
        "mag_anom":mag+.3*trend,"rtp":mag,"tmi":mag*.85,"tmi_hg":np.hypot(mag_y,mag_x),
        "tmi_vg":mag_y,"tc":gaussian_filter(mag,7),
        "iso_grav_anom":grav,"iso_grav_anom_hg":np.hypot(gy,gx),"iso_grav_anom_vg":gy,
        "iso_grav_anom_slope":np.hypot(gy,gx),
        "det_elev":elev,"det_elev_slope":np.hypot(dy,dx),
        "depth_to_base_surf":1000+45*ridge,"cond_surf":gaussian_filter(mag,24),
        "geod_2ndinv":trend,"geod_shearrate":abs(trend),"geod_dilaterate":grad(elev),
        "deq_n100a15":gaussian_filter(fault.astype(float),20),"ieq_n100a15":gaussian_filter(fault.astype(float),12)
    }
    profile=dict(driver="GTiff",width=w,height=h,count=len(BANDS),dtype="float32",nodata=-9999.0,
                 transform=from_origin(243350,4508550,100,100),crs="EPSG:32611",compress="deflate")
    with rasterio.open(features,"w",**profile) as dst:
        for i,name in enumerate(BANDS,start=1):
            dst.write(np.asarray(dat[name],np.float32),i)
            dst.set_band_description(i,name)
            dst.update_tags(i,band_name=name)
    profile.update(count=1,dtype="int8",nodata=-1)
    with rasterio.open(faults,"w",**profile) as dst:dst.write(fault.astype("int8"),1)
    return features, faults


def grad(a):
    dy,dx=np.gradient(a)
    return np.hypot(dy,dx)


if __name__=="__main__":
    a,b=generate_demo(Path(__file__).resolve().parent/"data")
    print("Created synthetic demo:",a,b)
