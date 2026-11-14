"""Exercise every checkpoint on synthetic inputs; not an accuracy evaluation."""
from pathlib import Path
import sys,time,io,json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from backend.app.inference import PATHS,binary,landcover
rng=np.random.default_rng(42)
def rgb():
 out=io.BytesIO();Image.fromarray(rng.integers(0,255,(256,256,3),dtype='uint8')).save(out,format='PNG');return out.getvalue()
a,b=rgb(),rgb()
with MemoryFile() as mem:
 with mem.open(driver='GTiff',height=256,width=256,count=6,dtype='uint16',crs='EPSG:4326',transform=from_origin(77.44,23.26,.00002,.00002)) as ds:ds.write(rng.integers(100,4000,(6,256,256),dtype='uint16'))
 raster=mem.read()
results=[]
for key in PATHS:
 start=time.monotonic()
 try:
  r=landcover(raster,key,[]) if key.startswith('land') else binary(a,key,b if key.startswith('change') else None)
  assert r['status']=='real' and r['mask'].startswith('data:image/png;base64,')
  if key.startswith('land'):assert abs(sum(r['coverage'])-100)<.01
  row={'model':key,'status':'passed','seconds':round(time.monotonic()-start,2),'percentage':r.get('coveragePercent'),'coverage':r.get('coverage')}
 except Exception as exc:row={'model':key,'status':'failed','error':str(exc)}
 results.append(row);print(json.dumps(row),flush=True)
Path(__file__).with_name('smoke-results.json').write_text(json.dumps({'input':'synthetic smoke test; not accuracy validation','models':results},indent=2))
if any(r['status']=='failed' for r in results):raise SystemExit(1)
