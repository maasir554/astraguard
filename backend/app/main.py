from datetime import datetime,timezone
import json,logging
from typing import Literal
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel,ConfigDict,ValidationError,Field
from starlette.concurrency import run_in_threadpool
from . import inference as engine
from . import sentinel

app=FastAPI(title='AstraGuard local inference',version='1.0')
MAX_UPLOAD=64*1024*1024
class Options(BaseModel):
 model_config=ConfigDict(extra='forbid')
 classification:bool=True
 coverage:bool=True
 buildings:bool=True
 change:bool=False
 buildingModel:Literal['building_unet','building_deeplab']='building_deeplab'
 landModel:Literal['land_deeplab','land_segformer']='land_segformer'
 changeModel:Literal['change_siamese','change_former']='change_siamese'
 aligned:bool=False
 sentinelAuto:bool=False
 sentinelStart:str=""
 sentinelEnd:str=""
 sentinelCloud:float=Field(default=20,ge=0,le=100)

async def read(file):
 if file is None:return None
 try:
  data=await file.read(MAX_UPLOAD+1)
  if len(data)>MAX_UPLOAD:raise HTTPException(413,'Each upload must be under 64 MiB')
  if not data:raise HTTPException(400,'Uploaded file is empty')
  return data
 finally:await file.close()

def status(state,message,model=None):return {'status':state,'message':message,'model':model}

def run(image,land,before,after,points,opts):
 result={'createdAt':datetime.now(timezone.utc).isoformat(),'classification':None,'coverage':None,'buildings':None,'change':None,'landcover':None,'statuses':{},'demo':False}
 statuses=result['statuses']
 if opts.classification:
  result['classification']=[82.6,5.1,3.8,.4,.2,2.4,3.2,.7,1,.6]
  statuses['classification']=status('fallback','EuroSAT scene-classification weights were not supplied; showing sample values.')
 if opts.coverage:
  try:
   metadata=None
   if land is None and opts.sentinelAuto:
    land,metadata=sentinel.fetch_crop(points,opts.sentinelStart,opts.sentinelEnd,opts.sentinelCloud)
   if land is None:raise ValueError('Upload a six-band Sentinel-2 GeoTIFF or enable automatic retrieval.')
   result['landcover']=engine.landcover(land,opts.landModel,points)
   if metadata:result['landcover']['imagery']=metadata
   result['coverage']=result['landcover']['coverage']
   statuses['coverage']=status('real','Sentinel-2 inference' if metadata else 'Local six-band inference',opts.landModel)
  except Exception as exc:
   logging.exception('Land-cover inference failed')
   statuses['coverage']=status('fallback',str(exc),opts.landModel);result['coverage']=[64.2,8.3,5.6,17.4,1.1,3.4]
 if opts.buildings:
  try:result['buildings']=engine.binary(image,opts.buildingModel);statuses['buildings']=status('real','Local RGB inference',opts.buildingModel)
  except Exception as exc:
   logging.exception('Building inference failed');statuses['buildings']=status('fallback',str(exc),opts.buildingModel)
   result['buildings']={'status':'fallback','coveragePercent':12.4,'scope':'Sample values; no mask generated'}
 if opts.change:
  if not before or not after or not opts.aligned:
   statuses['change']=status('needs_input','Upload aligned before/after RGB images and confirm they show the same geographic extent.')
  else:
   try:result['change']=engine.binary(after,opts.changeModel,before);statuses['change']=status('real','Local paired-image inference',opts.changeModel)
   except Exception as exc:
    logging.exception('Change inference failed');statuses['change']=status('fallback',str(exc),opts.changeModel)
    result['change']={'status':'fallback','coveragePercent':3.2,'scope':'Sample values; no mask generated'}
 result['demo']=any(s['status']=='fallback' for s in statuses.values())
 return result

@app.get('/api/health')
def health():return {'status':'ok','device':str(engine.DEVICE),'weights':{k:p.exists() for k,p in engine.PATHS.items()},'classificationAvailable':False}

@app.post('/api/analyze')
async def analyze(image:UploadFile=File(...),options:str=Form('{}'),polygon:str=Form('[]'),landcover:UploadFile|None=File(None),before:UploadFile|None=File(None),after:UploadFile|None=File(None)):
 try:
  opts=Options.model_validate_json(options);points=json.loads(polygon)
  if not isinstance(points,list) or (points and len(points)<3) or len(points)>1000:raise ValueError('Invalid polygon')
  for p in points:
   if len(p)!=2 or not all(isinstance(v,(float,int)) and np.isfinite(v) for v in p) or abs(p[0])>180 or abs(p[1])>85:raise ValueError('Invalid coordinates')
 except (ValueError,TypeError,ValidationError) as e:raise HTTPException(400,str(e))
 data=await read(image);land=await read(landcover);b=await read(before);a=await read(after)
 try:engine.read_rgb(data)
 except Exception as e:raise HTTPException(400,f'Invalid RGB input: {e}')
 # Reject competing work instead of retaining an unbounded inference queue.
 if not engine.LOCK.acquire(blocking=False):raise HTTPException(503,'Inference is busy; retry after the current request')
 try:return await run_in_threadpool(run,data,land,b,a,points,opts)
 finally:engine.LOCK.release()
