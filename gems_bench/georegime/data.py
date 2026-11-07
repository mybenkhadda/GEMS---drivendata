"""Label-free Atlas feature bank, source validation and spatial CV/discovery masks."""
from __future__ import annotations
import hashlib, json, os, shutil
from pathlib import Path
import numpy as np
import rasterio
from scipy import ndimage

FEATURES={
 "magnetic":["mag_analytic_signal","mag_thd_uc1000","mag_thd_uc2000","mag_thd_uc5000",
             "mag_residual1000","mag_edge_persistence","mag_coherence_1000m",
             "mag_strike_c2_1000m","mag_strike_s2_1000m"],
 "gravity":["grav_analytic_signal","grav_thd_uc5000","grav_residual1000","grav_edge_persistence",
            "grav_coherence_3000m","grav_strike_c2_3000m","grav_strike_s2_3000m"],
 "terrain":["terrain_profile_curvature_1000m","terrain_valley_1000m",
            "terrain_slope_break_1000m","terrain_coherence_1000m","terrain_tpi_2000m",
            "terrain_ridge_2000m","terrain_dog_300_1000m",
            "terrain_strike_c2_1000m","terrain_strike_s2_1000m"],
 "raw":["rtp","iso_grav_anom","det_elev","det_elev_slope"],
 "cross_modal":["xm_edge_magn_grav","xm_orient_magn_grav",
                "xm_multiscale_persistence","xm_continuity"],
}
GROUPS={
 "mag":FEATURES["magnetic"]+["rtp"],
 "grav":FEATURES["gravity"]+["iso_grav_anom"],
 "terrain":FEATURES["terrain"]+["det_elev","det_elev_slope"]+FEATURES["cross_modal"]
}

def atomic_json(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_name(path.name+".partial")
 temp.write_text(json.dumps(obj,indent=2,default=str),encoding="utf-8")
 os.replace(temp,path)

def file_sha256(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  while True:
   part=f.read(8<<20)
   if not part:break
   h.update(part)
 return h.hexdigest()

def stage_file(src,folder):
 src,folder=Path(src),Path(folder)
 if not src.is_file():raise FileNotFoundError(f"Missing Atlas artifact: {src}")
 folder.mkdir(parents=True,exist_ok=True)
 dest=folder/src.name
 if dest.is_file() and dest.stat().st_size==src.stat().st_size:return dest
 tmp=dest.with_name(dest.name+".partial")
 shutil.copyfile(src,tmp)
 if tmp.stat().st_size!=src.stat().st_size:raise IOError(f"Short file transfer: {src}")
 os.replace(tmp,dest)
 return dest

class Bank:
 """One immutable selected-feature bank; mapped-fault labels stored separately."""
 def __init__(self,atlas_root,cache,scratch):
  self.atlas_root=Path(atlas_root);self.cache=Path(cache);self.scratch=Path(scratch)
  self.cache.mkdir(parents=True,exist_ok=True);self.scratch.mkdir(parents=True,exist_ok=True)
  schema_file=self.atlas_root/"06_ml_ready/schemas/feature_stack_schema.json"
  if not schema_file.is_file():raise FileNotFoundError(f"Run Atlas ML-ready stage first: {schema_file}")
  self.schema=json.loads(schema_file.read_text())
  self.shape=tuple(self.schema["grid"][k] for k in ("height","width"))
  self.names=[];self.group_indices={}
  for group,names in GROUPS.items():
   self.group_indices[group]=np.arange(len(self.names),len(self.names)+len(names))
   self.names.extend(names)
  if len(set(self.names))!=len(self.names):raise ValueError("Duplicate feature channels")
  self.assets={}
  for family,names in FEATURES.items():
   spec=self.schema["families"].get(family)
   if spec is None:raise ValueError(f"Atlas family missing: {family}")
   if not set(names).issubset(spec["bands"]):
    raise ValueError(f"Atlas {family} missing: {sorted(set(names)-set(spec['bands']))}")
   path=self.atlas_root/spec["file"]
   if not path.is_file():raise FileNotFoundError(f"Required Atlas feature stack: {path}")
   self.assets[family]=(path,spec)
  maskdir=self.atlas_root/"02_geospatial/masks"
  self.mask_sources={k:maskdir/f"{k}.tif" for k in ("valid_numerical","fault_labels","template_valid")}
  for k,p in self.mask_sources.items():
   if not p.is_file():raise FileNotFoundError(f"Required Atlas mask {k}: {p}")
  self.template_path=self.atlas_root/"00_raw_snapshot/official_template/example_submission.tif"
  if not self.template_path.is_file():raise FileNotFoundError(f"Official template missing: {self.template_path}")
  fp={"schema":self.schema,"names":self.names,
      "sources":[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p,_ in self.assets.values()],
      "version":"georegime-bank-v1"}
  self.key=hashlib.sha256(json.dumps(fp,sort_keys=True).encode()).hexdigest()[:16]
  self.bank_drive=self.cache/f"bank_{self.key}.npy"
  self.bank_local=self.scratch/self.bank_drive.name
  self.aux_drive=self.cache/f"context_{self.key}.npy"
  self.aux_local=self.scratch/self.aux_drive.name
  self.manifest=self.cache/f"bank_{self.key}.json"
  self.array=None;self.context=None
  self.preflight()

 def preflight(self):
  from rasterio.crs import CRS
  with rasterio.open(self.template_path) as ds:
   if ds.shape!=self.shape or ds.crs!=CRS.from_epsg(32611) or ds.count!=1:
    raise ValueError("Official template grid/CRS mismatch")
   self.transform=ds.transform;self.profile=ds.profile.copy()
   self.template_valid=np.isfinite(ds.read(1,masked=True).astype(np.float32).filled(np.nan))
  for fam,(path,spec) in self.assets.items():
   with rasterio.open(path) as ds:
    if ds.shape!=self.shape or ds.transform!=self.transform or ds.crs!=CRS.from_epsg(32611):
     raise ValueError(f"Atlas stack grid mismatch: {fam}")
    if ds.count!=len(spec["bands"]):raise ValueError(f"Atlas band count mismatch: {fam}")
  masks={}
  for k,path in self.mask_sources.items():
   with rasterio.open(path) as ds:
    if ds.shape!=self.shape or ds.transform!=self.transform or ds.crs!=CRS.from_epsg(32611):
     raise ValueError(f"Misaligned Atlas mask: {k}")
    masks[k]=ds.read(1)>0
  self.valid=masks["valid_numerical"]&self.template_valid
  self.labels=masks["fault_labels"]&self.valid
  if self.labels.sum()<1000 or self.valid.sum()<100000:
   raise RuntimeError("Fault or valid mask unexpectedly empty")
  return {"feature_channels":len(self.names),"shape":self.shape,
          "eligible_pixels":int(self.valid.sum()),"mapped_pixels":int(self.labels.sum()),
          "bank_fingerprint":self.key}

 def _make_bank(self):
  target=self.bank_local.with_name(self.bank_local.name+".partial")
  a=np.lib.format.open_memmap(target,mode="w+",dtype="float16",shape=(len(self.names),*self.shape))
  indices={name:i for i,name in enumerate(self.names)}
  for family,(path,spec) in self.assets.items():
   staged=stage_file(path,self.scratch/"staged")
   with rasterio.open(staged) as ds:
    for b,name in enumerate(spec["bands"],1):
     if name not in indices:continue
     z=ds.read(b,masked=True).astype(np.float32).filled(np.nan)
     z[~self.valid]=np.nan
     z[~np.isfinite(z)|(np.abs(z)>=1e30)]=np.nan
     a[indices[name]]=z.astype(np.float16)
     a.flush()
     print(f"[bank] {indices[name]+1}/{len(self.names)} {name}")
  del a;os.replace(target,self.bank_local)

 def _make_context(self):
  # Label-free, whole-footprint gravity context. 5/15/30 km smoothing sigma
  # supplies very long range context to the 25.6 km local encoder.
  raw=np.asarray(self.array[self.names.index("iso_grav_anom")],np.float32)
  ok=np.isfinite(raw)&self.valid
  h,w=self.shape
  # Reduce to 1 km BEFORE wide smoothing: sigma 5/15/30 pixels rather
  # than 50/150/300 at 100 m. Weighted block aggregation preserves nodata.
  fac=10
  nh,nw=(h+fac-1)//fac,(w+fac-1)//fac
  pad=((0,nh*fac-h),(0,nw*fac-w))
  numerator=np.pad(np.where(ok,raw,0.0),pad,constant_values=0)
  denominator=np.pad(ok.astype(np.float32),pad,constant_values=0)
  coarse_n=numerator.reshape(nh,fac,nw,fac).sum(axis=(1,3))
  coarse_d=denominator.reshape(nh,fac,nw,fac).sum(axis=(1,3))
  target=self.aux_local.with_name(self.aux_local.name+".partial")
  out=np.lib.format.open_memmap(target,mode="w+",dtype="float16",shape=(3,*self.shape))
  for i,sigma in enumerate((5.,15.,30.)):
   smooth_n=ndimage.gaussian_filter(coarse_n,sigma,mode="nearest")
   smooth_d=ndimage.gaussian_filter(coarse_d,sigma,mode="nearest")
   small=np.divide(smooth_n,smooth_d,out=np.zeros_like(smooth_n),where=smooth_d>1)
   sm=ndimage.zoom(small,(h/nh,w/nw),order=1)
   if sm.shape!=self.shape:
    raise RuntimeError(f"Context upsampling produced {sm.shape}, expected {self.shape}")
   sm[~self.valid]=np.nan
   out[i]=sm.astype(np.float16);out.flush()
   print(f"[gravity context] sigma={sigma:g} km, coarsened at 1 km")
  del out;os.replace(target,self.aux_local)

 def _ensure(self,drive,local,producer):
  if local.is_file() and drive.is_file() and local.stat().st_size==drive.stat().st_size:return
  if drive.is_file():stage_file(drive,local.parent)
  else:
   producer()
   tmp=drive.with_name(drive.name+".partial")
   shutil.copyfile(local,tmp)
   os.replace(tmp,drive)

 def prepare(self):
  self._ensure(self.bank_drive,self.bank_local,self._make_bank)
  self.array=np.load(self.bank_local,mmap_mode="r")
  if self.array.shape!=(len(self.names),*self.shape):raise RuntimeError("Corrupt bank shape")
  self._ensure(self.aux_drive,self.aux_local,self._make_context)
  self.context=np.load(self.aux_local,mmap_mode="r")
  if self.context.shape!=(3,*self.shape):raise RuntimeError("Corrupt context shape")
  atomic_json(self.manifest,{"fingerprint":self.key,"names":self.names,
    "groups":{k:v.tolist() for k,v in self.group_indices.items()},
    "shape":self.shape,"source":"label-free Geological Atlas", "context_sigma_m":[5000,15000,30000],
    "training_labels_in_input":False})
  return self

 def fit_normalizer(self,mask,seed=2026,n=28000):
  rng=np.random.default_rng(seed)
  rr,cc=np.nonzero(mask&self.valid)
  if len(rr)<500:raise RuntimeError("Insufficient training-only normalization pixels")
  if len(rr)>n:
   ix=rng.choice(len(rr),n,replace=False);rr,cc=rr[ix],cc[ix]
  arr=np.concatenate([np.asarray(self.array[:,rr,cc],np.float32),
                      np.asarray(self.context[:,rr,cc],np.float32)],axis=0)
  med=np.nanmedian(arr,axis=1)
  q05=np.nanpercentile(arr,5,axis=1)
  q95=np.nanpercentile(arr,95,axis=1)
  scale=np.maximum((q95-q05)/2,1e-6)
  med=np.where(np.isfinite(med),med,0)
  scale=np.where(np.isfinite(scale),scale,1)
  return {"median":med.astype(np.float32).tolist(),"scale":scale.astype(np.float32).tolist(),
          "n_pixels":int(len(rr)),"train_only":True}

 def crop(self,row,col,patch,norm):
  sl=np.s_[row:row+patch,col:col+patch]
  a=np.asarray(self.array[:,sl[0],sl[1]],np.float32)
  ctx=np.asarray(self.context[:,sl[0],sl[1]],np.float32)
  v=np.concatenate((a,ctx),axis=0)
  med=np.asarray(norm["median"],np.float32)[:,None,None]
  scl=np.asarray(norm["scale"],np.float32)[:,None,None]
  v=np.clip(np.nan_to_num((v-med)/scl,nan=0,posinf=0,neginf=0),-5,5)
  out={k:v[idx] for k,idx in self.group_indices.items()}
  out["context"]=v[len(self.names):]
  return out

 def regime_prior(self,row,col,patch):
  # Intentionally no mapped-fault labels, distance or old cluster IDs.
  mi=self.names.index("mag_edge_persistence")
  gi=self.names.index("grav_edge_persistence")
  m=np.asarray(self.array[mi,row:row+patch,col:col+patch],np.float32)
  g=np.asarray(self.array[gi,row:row+patch,col:col+patch],np.float32)
  return np.clip(0.5+1.5*(np.nan_to_num(m,nan=0)-np.nan_to_num(g,nan=0)),0.05,0.95)

def dilate(mask,px):
 return ndimage.binary_dilation(mask,iterations=px) if px else mask.copy()

def geo_masks(valid,labels,buffer_px=15):
 h,w=valid.shape
 cols=np.flatnonzero(np.any(valid,axis=0))
 if len(cols)<100:raise RuntimeError("Invalid footprint for geographic CV")
 boundaries=np.quantile(np.repeat(np.arange(w),valid.sum(axis=0).astype(int)),[1/3,2/3]).astype(int)
 for i in range(3):
  c0=0 if i==0 else boundaries[i-1]
  c1=w if i==2 else boundaries[i]
  zone=valid&(np.arange(w)[None,:]>=c0)&(np.arange(w)[None,:]<c1)
  train=valid&~dilate(zone,buffer_px)
  if (train&labels).sum()<100 or (zone&labels).sum()<100:
   raise RuntimeError(f"Empty geographic fault split {i}")
  yield f"geo{i}",train,zone,labels&zone,np.zeros_like(labels)

def discovery_masks(valid,labels,buffer_px=8):
 components,n=ndimage.label(labels,structure=np.ones((3,3),np.uint8))
 if n<9:raise RuntimeError("Fewer than nine connected mapped faults")
 counts=np.bincount(components[labels]);counts[0]=0
 eligible=np.where(counts>=10)[0]
 if len(eligible)<9:raise RuntimeError("Insufficient >=1 km independent mapped fault fragments")
 buckets=[[],[],[]];totals=[0,0,0]
 for comp in sorted(eligible,key=lambda i:-counts[i]):
  k=int(np.argmin(totals))
  buckets[k].append(comp);totals[k]+=counts[comp]
 for i,ids in enumerate(buckets):
  held=np.isin(components,np.asarray(ids))&labels
  train=valid&~dilate(held,buffer_px)
  zone=dilate(held,max(30,3*buffer_px))&valid
  retained=labels&~held
  if (train&retained).sum()<100 or held.sum()<100:
   raise RuntimeError(f"Discovery split {i} empty")
  yield f"disc{i}",train,zone,held,retained
