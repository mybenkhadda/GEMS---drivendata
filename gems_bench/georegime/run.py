"""Resumable Atlas-driven GeoRegime GPU training, validation, inference and export.

Only label-free Atlas rasters enter model/router. Fold labels construct targets
and holdouts, never input features. Requires real GPU Colab execution.
"""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from scipy import ndimage
from .data import Bank, atomic_json, file_sha256, geo_masks, discovery_masks
from .model import GeoRegimeNet

DEFAULT={
 "drive_root":"/content/drive/MyDrive/GEMS_2",
 "atlas_root":"GEMS_GEOLOGICAL_DATA_ATLAS_V1",
 "output_root":"GEMS_GEOREGIME_V1",
 "scratch_root":"/content/gems_georegime_stage",
 "seed":2026,"patch":256,"width":24,"epochs":8,
 "steps_per_epoch":90,"batch_size":4,"lr":0.00035,"pu_prior":0.015,
 "geo_buffer_px":15,"discovery_buffer_px":8,
 "predict_stride":192,"use_amp":True
}

def _amp(device,enabled):
 import torch
 return torch.autocast(device_type="cuda",dtype=torch.float16,
                       enabled=(device.type=="cuda" and enabled))

def _write_tif(path,array,profile,valid):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 p=profile.copy()
 p.update(driver="GTiff",count=1,dtype="float32",nodata=np.nan,
          tiled=True,blockxsize=256,blockysize=256,compress="deflate",predictor=3)
 x=np.where(valid,np.asarray(array,np.float32),np.nan)
 tmp=path.with_name(path.stem+".partial.tif")
 with rasterio.open(tmp,"w",**p) as ds:
  ds.write(x,1);ds.set_band_description(1,"georegime_fault_confidence")
 with rasterio.open(tmp) as ds:
  if ds.shape!=x.shape or ds.crs!=p["crs"] or ds.transform!=p["transform"] or ds.count!=1:
   raise RuntimeError(f"GeoTIFF readback profile mismatch: {tmp}")
 os.replace(tmp,path)
 return path

def _write_submission_masked(path,prob,profile,template_mask):
 """Competition TIFF: float32 [0,1] stored EVERYWHERE, explicit internal valid-data mask.

 The historical exporter stored NaNs outside the eligible pixels. Some
 upload validators evaluate raw array range before honoring GDAL_NODATA;
 an internal validity mask retains the official footprint without raw NaNs.
 """
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 clean=np.clip(np.nan_to_num(np.asarray(prob,np.float32),
                              nan=0.,posinf=1.,neginf=0.),0.,1.)
 mask=np.asarray(template_mask,bool)
 if clean.shape!=mask.shape:raise ValueError("Submission value/mask shape mismatch")
 clean[~mask]=0.
 prof=profile.copy()
 prof.update(driver="GTiff",count=1,dtype="float32",nodata=None,
             tiled=True,blockxsize=256,blockysize=256,
             compress="deflate",predictor=3)
 tmp=path.with_name(path.stem+".partial.tif")
 with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
  with rasterio.open(tmp,"w",**prof) as dst:
   dst.write(clean,1)
   dst.write_mask(mask.astype(np.uint8)*255)
   dst.set_band_description(1,"georegime_fault_confidence")
 with rasterio.open(tmp) as ds:
  v=ds.read(1)
  if (ds.count!=1 or ds.dtypes[0]!="float32"
      or ds.transform!=prof["transform"] or ds.crs!=prof["crs"]
      or not np.isfinite(v).all()
      or not ((v>=0)&(v<=1)).all()
      or not np.array_equal(ds.dataset_mask()>0,mask)):
   raise RuntimeError(f"Submission readback/mask/range validation failed: {tmp}")
 os.replace(tmp,path)
 return path


def _orient(positive):
 """Axial strike targets constructed from TRAIN-VISIBLE mapped labels only."""
 from skimage.morphology import skeletonize
 sk=skeletonize(positive)
 sm=ndimage.gaussian_filter(sk.astype(np.float32),1.3)
 gy,gx=np.gradient(sm)
 # Rows point south; the tangent vector is (dE=gy, dN=gx).
 strike=np.arctan2(gy,gx)
 c=np.cos(2*strike).astype(np.float32)
 s=np.sin(2*strike).astype(np.float32)
 xx=ndimage.gaussian_filter(gx*gx,2)
 yy=ndimage.gaussian_filter(gy*gy,2)
 xy=ndimage.gaussian_filter(gx*gy,2)
 coh=np.hypot(xx-yy,2*xy)/(xx+yy+1e-9)
 _,idx=ndimage.distance_transform_edt(~sk,return_indices=True)
 return c[tuple(idx)].astype(np.float16),s[tuple(idx)].astype(np.float16),coh[tuple(idx)]>0.2

class Tiles:
 def __init__(self,bank,train_mask,norm,patch,length,seed):
  self.bank=bank;self.mask=np.asarray(train_mask,bool)
  self.patch=int(patch);self.length=int(length);self.seed=int(seed);self.epoch=0
  self.pos=bank.labels&self.mask
  if self.pos.sum()<100:raise ValueError("No mapped faults in training region")
  self.pr,self.pc=np.nonzero(self.pos)
  self.dist=ndimage.distance_transform_edt(~self.pos).astype(np.float32)*100
  self.c,self.s,self.ok=_orient(self.pos)
  self.norm=norm;self.h,self.w=self.mask.shape
 def __len__(self):return self.length
 def set_epoch(self,e):self.epoch=e
 def __getitem__(self,i):
  import torch
  rng=np.random.default_rng(self.seed+self.epoch*1000007+i*104729)
  if rng.random()<0.75:
   k=int(rng.integers(len(self.pr)))
   r,col=int(self.pr[k]),int(self.pc[k])
  else:
   for _ in range(500):
    r,col=int(rng.integers(self.h)),int(rng.integers(self.w))
    if self.mask[r,col]:break
   else:raise RuntimeError("Cannot sample a valid training patch")
  r0=int(np.clip(r-self.patch//2,0,self.h-self.patch))
  c0=int(np.clip(col-self.patch//2,0,self.w-self.patch))
  sl=np.s_[r0:r0+self.patch,c0:c0+self.patch]
  feats=self.bank.crop(r0,c0,self.patch,self.norm)
  batch={k:torch.from_numpy(v.copy()).float()
         for k,v in feats.items() if k in ("mag","grav","terrain","context")}
  batch.update({
   "truth":torch.from_numpy(self.pos[sl].astype(np.float32)),
   "supervise":torch.from_numpy(self.mask[sl].copy()),
   "proximity":torch.from_numpy(self.dist[sl].copy()),
   "orientation":torch.from_numpy(np.stack([self.c[sl],self.s[sl]]).astype(np.float32)),
   "orientation_ok":torch.from_numpy(self.ok[sl].copy()),
   "regime":torch.from_numpy(self.bank.regime_prior(r0,c0,self.patch).astype(np.float32))})
  return batch

def _tile_positions(length,patch,stride):
 arr=list(range(0,max(1,length-patch+1),stride))
 last=length-patch
 if last>=0 and (not arr or arr[-1]!=last):arr.append(last)
 return sorted(set(arr))

class GeoRegimeExperiment:
 def __init__(self,config=None):
  self.cfg={**DEFAULT,**(config or {})}
  c=self.cfg
  self.drive=Path(c["drive_root"])
  self.atlas_root=self.drive/c["atlas_root"]
  self.root=self.drive/c["output_root"]
  self.scratch=Path(c["scratch_root"])
  self.root.mkdir(parents=True,exist_ok=True)
  self.scratch.mkdir(parents=True,exist_ok=True)
  self.bank=Bank(self.atlas_root,self.root/"cache",self.scratch)
  self.device=None;self.run_root=None;self.results=[]

 def preflight(self):
  import torch
  result=self.bank.preflight()
  result.update({"cuda_available":bool(torch.cuda.is_available()),
                 "dem_used":False,"old_label_derived_cluster_models_used":False,
                 "schema":str(self.atlas_root/"06_ml_ready/schemas/feature_stack_schema.json")})
  if not result["cuda_available"]:
   raise RuntimeError("Select a GPU Colab runtime before GeoRegime training.")
  print("[preflight]",result)
  return result

 def prepare(self):
  import torch
  self.bank.prepare()
  semantic={k:v for k,v in self.cfg.items() if k not in ("drive_root","scratch_root")}
  key=hashlib.sha256(json.dumps({"config":semantic,"bank":self.bank.key},
                                sort_keys=True).encode()).hexdigest()[:12]
  self.run_root=self.root/"runs"/key
  for d in ("checkpoints","predictions","tables","figures","submissions"):
   (self.run_root/d).mkdir(parents=True,exist_ok=True)
  atomic_json(self.run_root/"config.json",
   {"config":self.cfg,"bank_fingerprint":self.bank.key,"feature_ids":self.bank.names,
    "gravity_context_sigma_m":[5000,15000,30000],
    "router_prior":"magnetic minus gravity edge persistence, label-free",
    "splits":"3 contiguous geographic regions and 3 withheld connected fault-system groups"})
  self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
  return {"run_root":str(self.run_root),"features":str(self.bank.bank_drive),
          "context":str(self.bank.aux_drive),"device":str(self.device)}

 def _model(self):
  g=self.bank.group_indices
  return GeoRegimeNet(len(g["mag"]),len(g["grav"]),len(g["terrain"]),self.cfg["width"])

 def _ckpt(self,unit):
  return self.run_root/"checkpoints"/f"{unit}_latest.pt"

 def train_unit(self,unit,train_mask):
  import torch
  from torch.utils.data import DataLoader
  from .loss import loss_all
  if self.run_root is None:raise RuntimeError("Call prepare() before train.")
  normpath=self.run_root/"checkpoints"/f"{unit}_normalizer.json"
  if normpath.is_file():norm=json.loads(normpath.read_text())
  else:
   norm=self.bank.fit_normalizer(train_mask,seed=self.cfg["seed"]+len(unit))
   atomic_json(normpath,norm)
  unit_seed=self.cfg["seed"]+int(hashlib.sha256(unit.encode()).hexdigest()[:8],16)%100000
  torch.manual_seed(unit_seed)
  if self.device.type=="cuda":torch.cuda.manual_seed_all(unit_seed)
  model=self._model().to(self.device)
  optimizer=torch.optim.AdamW(model.parameters(),lr=self.cfg["lr"],weight_decay=1e-4)
  amp=bool(self.cfg["use_amp"] and self.device.type=="cuda")
  scaler=torch.amp.GradScaler("cuda",enabled=amp)
  ckpt=self._ckpt(unit)
  contract={"features":self.bank.key,"unit":unit,
            "train_mask_sha":hashlib.sha256(train_mask.tobytes()).hexdigest()[:12],
            "config_hash":hashlib.sha256(json.dumps(self.cfg,sort_keys=True).encode()).hexdigest()[:12]}
  first=0
  if ckpt.is_file():
   saved=torch.load(ckpt,map_location="cpu",weights_only=False)
   if saved["contract"]!=contract:raise RuntimeError(f"Checkpoint belongs to different unit/config: {ckpt}")
   if saved["normalizer"]!=norm:raise RuntimeError(f"Normalizer changed after {unit} training")
   model.load_state_dict(saved["model"],strict=True)
   optimizer.load_state_dict(saved["optimizer"])
   scaler.load_state_dict(saved["scaler"])
   first=int(saved["epoch"])+1
   if "rng_torch" in saved:torch.set_rng_state(saved["rng_torch"])
   if "rng_numpy" in saved:np.random.set_state(saved["rng_numpy"])
   print(f"[{unit}] resumed at epoch {first+1}")
  if first>=self.cfg["epochs"]:
   return {"unit":unit,"status":"CACHED_COMPLETE","checkpoint":str(ckpt)}
  ds=Tiles(self.bank,train_mask,norm,self.cfg["patch"],
           self.cfg["steps_per_epoch"]*self.cfg["batch_size"],unit_seed)
  loader=DataLoader(ds,batch_size=self.cfg["batch_size"],shuffle=False,num_workers=0,
                    pin_memory=self.device.type=="cuda")
  table=self.run_root/"tables"/f"{unit}_history.csv"
  hist=pd.read_csv(table).to_dict("records") if table.is_file() else []
  for epoch in range(first,self.cfg["epochs"]):
   ds.set_epoch(epoch);model.train()
   parts_epoch=[]
   for batch in loader:
    b={k:v.to(self.device,non_blocking=True) for k,v in batch.items()}
    optimizer.zero_grad(set_to_none=True)
    with _amp(self.device,amp):
     pred=model(b["mag"],b["grav"],b["terrain"],b["context"])
     loss,parts=loss_all(pred,b["truth"],b["supervise"],b["proximity"],
                         b["orientation"],b["orientation_ok"],b["regime"],
                         pu_prior=self.cfg["pu_prior"])
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    scaler.step(optimizer);scaler.update()
    parts_epoch.append(parts)
   mean={k:float(np.mean([p[k] for p in parts_epoch])) for k in parts_epoch[0]}
   hist.append({"unit":unit,"epoch":epoch+1,**mean})
   pd.DataFrame(hist).to_csv(table,index=False)
   state={"epoch":epoch,"model":model.state_dict(),"optimizer":optimizer.state_dict(),
          "scaler":scaler.state_dict(),"normalizer":norm,"contract":contract,
          "rng_torch":torch.get_rng_state(),"rng_numpy":np.random.get_state(),
          "feature_schema":self.bank.names,"model_type":"GeoRegimeNet_v1"}
   tmp=ckpt.with_name(ckpt.name+".partial")
   torch.save(state,tmp);os.replace(tmp,ckpt)
   print(f"[{unit}] epoch {epoch+1}/{self.cfg['epochs']} | loss={mean['loss']:.5f} | checkpoint saved")
  del ds,loader,model,optimizer
  if self.device.type=="cuda":torch.cuda.empty_cache()
  return {"unit":unit,"status":"TRAINED","checkpoint":str(ckpt)}

 def _load_model(self,unit):
  import torch
  p=self._ckpt(unit)
  if not p.is_file():raise FileNotFoundError(f"Trained {unit} checkpoint not found: {p}")
  state=torch.load(p,map_location="cpu",weights_only=False)
  if state["epoch"]+1!=self.cfg["epochs"]:
   raise RuntimeError(f"{unit} incomplete: epoch {state['epoch']+1} of {self.cfg['epochs']}")
  if state["contract"]["features"]!=self.bank.key or state["feature_schema"]!=self.bank.names:
   raise RuntimeError(f"{unit} checkpoint feature schema mismatch")
  model=self._model().to(self.device)
  model.load_state_dict(state["model"],strict=True);model.eval()
  return model,state["normalizer"]

 def predict(self,unit,roi=None):
  import torch
  roi=self.bank.valid if roi is None else (roi&self.bank.valid)
  path=self.run_root/"predictions"/f"{unit}_prediction.tif"
  gatepath=self.run_root/"predictions"/f"{unit}_router_magnetic.tif"
  meta=path.with_suffix(".json")
  ckpt=self._ckpt(unit)
  if ckpt.is_file() and path.is_file() and gatepath.is_file() and meta.is_file():
   state=json.loads(meta.read_text())
   with rasterio.open(path) as ds:
    ok=ds.shape==self.bank.shape and ds.transform==self.bank.transform and ds.crs.to_epsg()==32611
   if (ok and state.get("checkpoint_size")==ckpt.stat().st_size
       and state.get("checkpoint_mtime_ns")==ckpt.stat().st_mtime_ns
       and state.get("bank")==self.bank.key):
    with rasterio.open(path) as ds:return ds.read(1)
  model,norm=self._load_model(unit)
  patch,stride=self.cfg["patch"],self.cfg["predict_stride"]
  h,w=self.bank.shape
  acc=np.zeros((h,w),np.float32);route=np.zeros((h,w),np.float32);wsum=np.zeros((h,w),np.float32)
  win=np.maximum(np.outer(np.hanning(patch),np.hanning(patch)).astype(np.float32),0.05)
  positions=[(r,c) for r in _tile_positions(h,patch,stride)
             for c in _tile_positions(w,patch,stride) if roi[r:r+patch,c:c+patch].any()]
  print(f"[predict {unit}] windows: {len(positions)}")
  with torch.no_grad():
   for j,(r,c) in enumerate(positions,1):
    inp=self.bank.crop(r,c,patch,norm)
    x={k:torch.from_numpy(inp[k][None].copy()).float().to(self.device)
       for k in ("mag","grav","terrain","context")}
    with _amp(self.device,bool(self.cfg["use_amp"])):
     pred=model(x["mag"],x["grav"],x["terrain"],x["context"])
    p=pred["prob"][0,0].float().cpu().numpy()
    g=pred["routes"][0,0].float().cpu().numpy()
    wt=win*(roi[r:r+patch,c:c+patch])
    acc[r:r+patch,c:c+patch]+=p*wt
    route[r:r+patch,c:c+patch]+=g*wt
    wsum[r:r+patch,c:c+patch]+=wt
    if j%80==0:print(f"[predict {unit}] {j}/{len(positions)}")
  ok=wsum>0
  if (roi&~ok).any():raise RuntimeError(f"Model did not predict all {unit} evaluation pixels")
  out=np.divide(acc,wsum,out=np.zeros_like(acc),where=ok)
  routed=np.divide(route,wsum,out=np.zeros_like(route),where=ok)
  _write_tif(path,out,self.bank.profile,ok)
  _write_tif(gatepath,routed,self.bank.profile,ok)
  atomic_json(meta,{"unit":unit,"bank":self.bank.key,"checkpoint_size":ckpt.stat().st_size,
                    "checkpoint_mtime_ns":ckpt.stat().st_mtime_ns,
                    "pixels":int(ok.sum()),"source_model":"GeoRegimeNet_v1"})
  del model
  if self.device.type=="cuda":torch.cuda.empty_cache()
  return np.where(ok,out,np.nan)

 def _record(self,unit,pred,truth,zone,exclude,protocol):
  from gems_bench.metric import dti_components,OFFICIAL_EXCLUDE_SUPPORTS_TP
  stats=dti_components(pred,truth,eval_mask=zone&self.bank.valid,
    exclude_mask=exclude,exclude_supports_tp=OFFICIAL_EXCLUDE_SUPPORTS_TP)
  self.results.append({"unit":unit,"protocol":protocol,"representation":"continuous",**stats.as_dict()})
  pd.DataFrame(self.results).to_csv(self.run_root/"tables"/"validation_components.csv",index=False)
  print(f"[DTI offline proxy] {unit} = {stats.dti:.5f}")
  return stats

 def run_geographic(self):
  from gems_bench.metric import pool
  stats=[]
  for unit,train,zone,truth,exclude in geo_masks(self.bank.valid,self.bank.labels,self.cfg["geo_buffer_px"]):
   self.train_unit(unit,train)
   p=self.predict(unit,zone)
   stats.append(self._record(unit,p,truth,zone,exclude,"GEOGRAPHIC"))
  pooled=pool(stats)
  atomic_json(self.run_root/"tables"/"geographic_dti.json",pooled.as_dict())
  return pooled.as_dict()

 def run_discovery(self):
  from gems_bench.metric import pool
  stats=[]
  for unit,train,zone,truth,exclude in discovery_masks(
        self.bank.valid,self.bank.labels,self.cfg["discovery_buffer_px"]):
   self.train_unit(unit,train)
   p=self.predict(unit,zone)
   stats.append(self._record(unit,p,truth,zone,exclude,"WITHHELD_SYSTEM"))
  pooled=pool(stats)
  atomic_json(self.run_root/"tables"/"discovery_dti.json",pooled.as_dict())
  return pooled.as_dict()

 def candidate_proposals(self,prob):
  """Unverified label-free proposal layer, kept separate from submission."""
  import geopandas as gpd
  from rasterio.features import rasterize
  f=self.atlas_root/"04_eda/candidate_lineaments/candidate_lineaments.parquet"
  if not f.is_file():
   print(f"[candidates] no Atlas lineaments: {f}")
   return {"status":"UNAVAILABLE","path":str(f)}
  g=gpd.read_parquet(f)
  required={"evidence_count","orientation_agreement","continuity",
            "multiscale_persistence","geometry"}
  if not required.issubset(g.columns):
   raise ValueError(f"Atlas lineament data lacks {sorted(required-set(g.columns))}")
  good=g[(g.evidence_count>=2)&(g.orientation_agreement>=0.5)&(g.continuity>=0.65)].copy()
  if len(good):
   good["label_free_support"]=(0.5*good.continuity+
      0.3*good.orientation_agreement.clip(0,1)+
      0.2*good.multiscale_persistence.clip(0,1)).clip(0,1)
  pairs=[(geom,float(s)) for geom,s in zip(good.geometry,good.get("label_free_support",[]))
         if geom is not None and not geom.is_empty]
  support=rasterize(pairs,out_shape=self.bank.shape,transform=self.bank.transform,
                   fill=0,dtype="float32",all_touched=True) if pairs else np.zeros(self.bank.shape,np.float32)
  raster=self.run_root/"predictions"/"atlas_candidate_support.tif"
  _write_tif(raster,support,self.bank.profile,self.bank.valid)
  if len(good):
   rr=np.clip(np.floor((good.centroid_n.to_numpy()-self.bank.transform.f)/
                       self.bank.transform.e).astype(int),0,self.bank.shape[0]-1)
   cc=np.clip(np.floor((good.centroid_e.to_numpy()-self.bank.transform.c)/
                       self.bank.transform.a).astype(int),0,self.bank.shape[1]-1)
   good["model_confidence_at_centroid"]=prob[rr,cc]
  columns=[v for v in ("candidate_id","length_km","strike_deg","evidence_count",
                        "continuity","orientation_agreement","multiscale_persistence",
                        "label_free_support","model_confidence_at_centroid") if v in good.columns]
  table=self.run_root/"tables"/"candidate_model_review.csv"
  good[columns].to_csv(table,index=False)
  return {"status":"COMPLETE","n_proposals":len(good),"raster":str(raster),"table":str(table)}

 def export(self):
  for file in ("geographic_dti.json","discovery_dti.json"):
   if not (self.run_root/"tables"/file).is_file():
    raise RuntimeError(f"Missing finished validation prerequisite: {file}")
  self.train_unit("full",self.bank.valid)
  p=self.predict("full")
  proposals=self.candidate_proposals(p)
  dest=self.run_root/"submissions"/"submission_georegime_v1.tif"
  eligible=self.bank.valid&self.bank.template_valid
  _write_submission_masked(dest,p,self.bank.profile,self.bank.template_valid)
  with rasterio.open(dest) as ds:
   output=ds.read(1)
   checks={"one_band":ds.count==1,"dtype":ds.dtypes[0]=="float32",
           "crs":ds.crs.to_epsg()==32611,
           "shape":ds.shape==self.bank.shape,
           "transform":ds.transform==self.bank.transform,
           "finite_all":bool(np.isfinite(output).all()),
           "mask_matches_template":bool(np.array_equal(ds.dataset_mask()>0,self.bank.template_valid)),
           "range_all":bool(((output>=0)&(output<=1)).all()),
           "full_checkpoint":self._ckpt("full").is_file()}
  if not all(checks.values()):raise RuntimeError(f"Submission validation failed: {checks}")
  manifest={"status":"READY_FOR_MANUAL_UPLOAD","file":str(dest),"sha256":file_sha256(dest),
            "validation":checks,"bank":self.bank.key,
            "model":"GeoRegimeNet_v1 trained on full mapped labels",
            "policy":"continuous confidence, no top-k mask",
            "geographic":json.loads((self.run_root/"tables"/"geographic_dti.json").read_text()),
            "discovery":json.loads((self.run_root/"tables"/"discovery_dti.json").read_text()),
            "candidate_proposals":proposals,
            "no_legacy_model":True,"no_label_derived_router_inputs":True}
  atomic_json(self.run_root/"submissions"/"submission_georegime_v1.json",manifest)
  print(f"[NEW GeoRegime submission] {dest}")
  return manifest

 def figures(self):
  import matplotlib.pyplot as plt
  frames=[pd.read_csv(p) for p in sorted((self.run_root/"tables").glob("*_history.csv"))]
  if frames:
   hist=pd.concat(frames,ignore_index=True)
   fig,ax=plt.subplots(figsize=(10,5))
   for unit,g in hist.groupby("unit"):ax.plot(g.epoch,g.loss,label=unit)
   ax.set(xlabel="Epoch",ylabel="Training objective",title="GeoRegime per-unit training")
   ax.legend(ncol=4,fontsize=8);fig.tight_layout()
   fig.savefig(self.run_root/"figures"/"training_curves.png",dpi=180)
   fig.savefig(self.run_root/"figures"/"training_curves.svg")
   plt.show();plt.close(fig)
  a=self.run_root/"predictions"/"full_prediction.tif"
  b=self.run_root/"predictions"/"full_router_magnetic.tif"
  if a.is_file() and b.is_file():
   with rasterio.open(a) as ds:
    p=ds.read(1,out_shape=(750,660),resampling=Resampling.average)
   with rasterio.open(b) as ds:
    route=ds.read(1,out_shape=(750,660),resampling=Resampling.average)
   fig,axs=plt.subplots(1,2,figsize=(12,7))
   v=float(np.nanquantile(p,0.995))
   im=axs[0].imshow(p,cmap="magma",vmin=0,vmax=max(v,1e-5))
   fig.colorbar(im,ax=axs[0],shrink=.7)
   im=axs[1].imshow(route,cmap="RdBu_r",vmin=0,vmax=1)
   fig.colorbar(im,ax=axs[1],shrink=.7)
   axs[0].set_title("GeoRegime full fault confidence")
   axs[1].set_title("Magnetic specialist routing weight")
   for ax in axs:ax.axis("off")
   fig.tight_layout()
   fig.savefig(self.run_root/"figures"/"prediction_and_routing.png",dpi=180)
   fig.savefig(self.run_root/"figures"/"prediction_and_routing.svg")
   plt.show();plt.close(fig)
