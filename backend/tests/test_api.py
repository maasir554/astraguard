import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app import inference
client=TestClient(app)
def image():
 b=io.BytesIO();Image.new('RGB',(32,32),(50,100,150)).save(b,format='PNG');return b.getvalue()
def test_health():assert client.get('/api/health').json()['status']=='ok'
def test_missing_multispectral_and_temporal_inputs_are_explicit():
 r=client.post('/api/analyze',files={'image':('image.png',image(),'image/png')},data={'options':'{"buildings":false,"change":true}'})
 assert r.status_code==200
 assert r.json()['statuses']['coverage']['status']=='fallback'
 assert r.json()['statuses']['change']['status']=='needs_input'
 assert r.json()['change'] is None
 assert r.json()['landcover'] is None

def test_invalid_image_is_rejected():
 assert client.post('/api/analyze',files={'image':('bad.png',b'invalid','image/png')}).status_code==400

def test_invalid_architecture_is_rejected():
 assert client.post('/api/analyze',files={'image':('image.png',image(),'image/png')},data={'options':'{"buildingModel":"unknown"}'}).status_code==400

def test_busy_server_does_not_queue_work():
 inference.LOCK.acquire()
 try:assert client.post('/api/analyze',files={'image':('image.png',image(),'image/png')}).status_code==503
 finally:inference.LOCK.release()

def test_partial_failure_keeps_other_results(monkeypatch):
 def broken(*a,**kw):raise RuntimeError('missing checkpoint')
 monkeypatch.setattr(inference,'binary',broken)
 r=client.post('/api/analyze',files={'image':('image.png',image(),'image/png')}).json()
 assert r['statuses']['buildings']['status']=='fallback'
 assert r['buildings']['coveragePercent']==12.4
 assert r['coverage'] is not None

def test_rgb_rejects_transparent_input():
 b=io.BytesIO();Image.new('RGBA',(4,4),(0,0,0,0)).save(b,format='PNG')
 import pytest
 with pytest.raises(ValueError):inference.read_rgb(b.getvalue())

def test_landcover_clips_polygon_and_preserves_georeference(monkeypatch):
 import base64
 import torch
 from rasterio.io import MemoryFile
 from rasterio.transform import from_origin
 class Agriculture(torch.nn.Module):
  def forward(self,x):
   scores=torch.zeros((1,6,256,256),device=x.device)
   scores[:,1]=1
   return scores
 monkeypatch.setattr(inference,'load_model',lambda key:(Agriculture(),{'band_mean':[0]*6,'band_std':[1]*6,'scale_divisor':10000}))
 transform=from_origin(77,24,.01,.01)
 with MemoryFile() as source:
  with source.open(driver='GTiff',height=10,width=10,count=6,dtype='uint16',crs='EPSG:4326',transform=transform) as ds:ds.write(np.full((6,10,10),1000,dtype='uint16'))
  data=source.read()
 result=inference.landcover(data,'land_segformer',[[77,24],[77.05,24],[77.05,23.9],[77,23.9]])
 assert result['validPixels']==50
 assert result['coverage']==[100,0,0,0,0,0]
 with MemoryFile(base64.b64decode(result['geotiff'].split(',')[1])) as output,output.open() as ds:
  assert ds.transform==transform and ds.crs.to_epsg()==4326
  assert ds.nodata==255
  assert (ds.read(1)==1).sum()==50
  assert (ds.read(1)==255).sum()==50

def test_rgb_geotiff_cannot_substitute_for_six_bands():
 import pytest
 from rasterio.io import MemoryFile
 from rasterio.transform import from_origin
 with MemoryFile() as source:
  with source.open(driver='GTiff',height=4,width=4,count=3,dtype='uint8',crs='EPSG:4326',transform=from_origin(77,24,.01,.01)) as ds:ds.write(np.ones((3,4,4),dtype='uint8'))
  with pytest.raises(ValueError,match='six bands'):inference.landcover(source.read(),'land_segformer',[])
