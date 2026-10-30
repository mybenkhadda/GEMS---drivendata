"""Streaming exporter: formula-derived native-grid GeoTIFFs, with bounded memory."""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import rasterio
from rasterio.windows import Window
from .rasters import catalog, parse_extra_entries, read_layers, padded
from .formulas import compile_formulas, evaluate_formulas, suggested_halo, FormulaError


def main(argv=None):
    p = argparse.ArgumentParser(description="Export geological formula layers in georeferenced tiles")
    p.add_argument("--features",required=True,help="Full-resolution GEMS numerical-feature TIFF")
    p.add_argument("--formulas",required=True,help="JSON library with 'formulas' key or plain .txt assignments")
    p.add_argument("--extras",help="Optional text file with name=/path/layer.tif or .npy")
    p.add_argument("--out",required=True,help="Empty/unique output directory (existing outputs not overwritten)")
    p.add_argument("--layers",nargs="+",required=True,help="Derived layer names to export")
    p.add_argument("--tile",type=int,default=256,choices=[128,256,512])
    args=p.parse_args(argv)
    if not Path(args.features).is_file(): p.error("Feature raster does not exist")
    source=Path(args.formulas).read_text(encoding="utf-8")
    if args.formulas.lower().endswith(".json"):
        source=json.loads(source)["formulas"]
    extras=parse_extra_entries(Path(args.extras).read_text(encoding="utf-8")) if args.extras else []
    sources,details=catalog(args.features,extras)
    f=compile_formulas(source,list(sources))
    known={formula.name for formula in f}
    unknown=set(args.layers)-known
    if unknown: p.error(f"Not defined by formulas: {sorted(unknown)}")
    # zscore and robust_z are AOI-dependent and would create artificial tile seams.
    forbidden=[]
    for formula in f:
        if any(isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and
               node.func.id in ("zscore","robust_z") for node in ast.walk(formula.expression)):
            forbidden.append(formula.name)
    if forbidden:
        p.error(f"Full-footprint export cannot use ROI-dependent zscore/robust_z: {forbidden}")
    if any(name in sources for name in args.layers): p.error("Exports must be derived formula layers")
    halo=suggested_halo(f)
    out=Path(args.out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    paths={name:out/f"{name}.tif" for name in args.layers}
    existing=[str(path) for path in paths.values() if path.exists()]
    if existing: p.error(f"Refusing to overwrite existing outputs: {existing}")
    with rasterio.open(args.features) as src:
        profile=src.profile.copy()
        profile.update(driver="GTiff",count=1,dtype="float32",nodata=np.nan,
                       tiled=True,blockxsize=256,blockysize=256,compress="deflate",predictor=3,
                       BIGTIFF="IF_SAFER")
        dsts={name:rasterio.open(path,"w",**profile) for name,path in paths.items()}
        try:
            for name,dst in dsts.items():dst.set_band_description(1,name)
            for row in range(0,src.height,args.tile):
                for col in range(0,src.width,args.tile):
                    w=Window(col,row,min(args.tile,src.width-col),min(args.tile,src.height-row))
                    ext,crop=padded(w,src.height,src.width,halo)
                    raw=read_layers(sources,list(sources),ext)
                    context=evaluate_formulas(f,raw)
                    for name,dst in dsts.items():
                        dst.write(np.asarray(context[name][crop],np.float32),1,window=w)
                print(f"Completed rows {row}–{min(row+args.tile,src.height)} / {src.height}")
        except BaseException:
            # Don't delete failed files: preserve forensics, but mark incomplete and refuse to claim success.
            (out/"EXPORT_INCOMPLETE.json").write_text(json.dumps({"status":"FAILED","layers":list(paths),
                                                 "utc":datetime.now(timezone.utc).isoformat()},indent=2),encoding="utf-8")
            raise
        finally:
            for dst in dsts.values():dst.close()
    manifest={"status":"COMPLETE","source":str(Path(args.features).resolve()),
              "source_size":Path(args.features).stat().st_size,
              "source_mtime_ns":Path(args.features).stat().st_mtime_ns,
              "created_utc":datetime.now(timezone.utc).isoformat(),
              "crs":details["crs"],"grid_shape":[details["height"],details["width"]],
              "transform":details["transform"],"tile":args.tile,"halo":halo,"layers":{name:str(path) for name,path in paths.items()},
              "formulas":source,"extras":extras,"note":"No labels used. Gaussian halo. For model training use zone-safe validation."}
    (out/"export_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("COMPLETED:",out)


if __name__=="__main__":
    main()
