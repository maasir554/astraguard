"""Local, bounded inference. Models load lazily; at most one stays resident."""
from pathlib import Path
import base64, io, os, sys, threading
import numpy as np
import torch
from PIL import Image
import segmentation_models_pytorch as smp
from .architectures import SiameseUNet, ChangeFormer
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'backend/vendor/AstraGuard-LandClassifier/src'))
from astraguard_landcover.models import build_model
from astraguard_landcover.classes import CLASS_NAMES, BAND_NAMES
WEIGHTS = Path(os.getenv('ASTRAGUARD_WEIGHTS', ROOT/'weights'))
PATHS = {
 'building_unet': WEIGHTS/'UNet_best.pth',
 'building_deeplab': WEIGHTS/'DeepLabV3Plus_best.pth',
 'change_siamese': WEIGHTS/'SiameseUNet_best.pth',
 'change_former': WEIGHTS/'ChangeFormer_best.pth',
 'land_deeplab': WEIGHTS/'AstraGuard Landcover Segmentation Models/DeepLabV3+'/ 'best.pt',
 'land_segformer': WEIGHTS/'AstraGuard Landcover Segmentation Models/SegFormer'/'best.pt',
}
# CPU is deterministic and avoids platform-specific unsupported MPS kernels.
DEVICE = torch.device(os.getenv('ASTRAGUARD_DEVICE','cpu'))
torch.set_num_threads(min(4, os.cpu_count() or 1))
LOCK = threading.Lock()
_cached = None
ORDER = [1,5,2,3,0,4]  # training IDs -> the UI's six-class order
PALETTE = np.array([[158,158,174],[183,238,128],[201,149,121],[86,184,154],[120,187,223],[220,184,135]],dtype=np.uint8)

def load_model(key):
 global _cached
 if _cached and _cached[0]==key: return _cached[1:]
 _cached=None
 checkpoint=torch.load(PATHS[key],map_location='cpu',weights_only=True)
 stats=None
 if key.startswith('land_'):
  if tuple(checkpoint['class_names'])!=CLASS_NAMES: raise ValueError('Checkpoint class mapping differs from the expected contract')
  model=build_model(checkpoint['config']['model'],initialize_pretrained=False)
  stats=checkpoint['stats'];state=checkpoint['model_state']
 elif key=='building_unet':
  model=smp.Unet(encoder_name='resnet34',encoder_weights=None,in_channels=3,classes=1);state=checkpoint
 elif key=='building_deeplab':
  model=smp.DeepLabV3Plus(encoder_name='resnet50',encoder_weights=None,in_channels=3,classes=1);state=checkpoint
 elif key=='change_siamese': model=SiameseUNet();state=checkpoint
 else: model=ChangeFormer();state=checkpoint
 model.load_state_dict(state,strict=True)
 model.to(DEVICE).eval()
 _cached=(key,model,stats)
 return model,stats

def png(array):
 out=io.BytesIO();Image.fromarray(array).save(out,format='PNG')
 return 'data:image/png;base64,'+base64.b64encode(out.getvalue()).decode()

def read_rgb(data):
 im=Image.open(io.BytesIO(data))
 if im.width*im.height>16_000_000: raise ValueError('RGB images must be at most 16 megapixels')
 im=im.convert('RGBA');arr=np.array(im)
 if not (arr[:,:,3]>0).any():raise ValueError('Image has no visible pixels')
 return arr

def rgb_tensor(arr,size):
 # Reproduce the notebooks exactly: channel min/max to [0,1], followed by
 # Albumentations Normalize's default max_pixel_value=255. This double scaling
 # is unusual but changing it would not match the published training pipeline.
 rgb=arr[:,:,:3].astype(np.float32)
 valid=arr[:,:,3]>0
 for c in range(3):
  lo,hi=rgb[:,:,c][valid].min(),rgb[:,:,c][valid].max()
  if hi>lo:rgb[:,:,c]=(rgb[:,:,c]-lo)/(hi-lo)
  rgb[:,:,c][~valid]=0
 x=torch.from_numpy(rgb.transpose(2,0,1)).unsqueeze(0)
 x=torch.nn.functional.interpolate(x,size=(size,size),mode='bilinear',align_corners=False)
 mean=torch.tensor([.485,.456,.406])[None,:,None,None]
 std=torch.tensor([.229,.224,.225])[None,:,None,None]
 return ((x/255.-mean)/std).to(DEVICE)

@torch.inference_mode()
def binary(image,key,before=None):
 arr=read_rgb(image);size=256 if key.startswith('change') else 512
 valid=np.array(Image.fromarray(arr[:,:,3]).resize((size,size),Image.Resampling.NEAREST))>0
 model,_=load_model(key);x=rgb_tensor(arr,size)
 if before is not None:
  previous=read_rgb(before)
  if previous.shape!=arr.shape:raise ValueError('Before and after images must have identical dimensions and aligned geography')
  valid &= np.array(Image.fromarray(previous[:,:,3]).resize((size,size),Image.Resampling.NEAREST))>0
  logits=model(rgb_tensor(previous,size),x)
 else: logits=model(x)
 probabilities=logits.sigmoid()[0,0].cpu().numpy()
 if not np.isfinite(probabilities).all(): raise ValueError('Model produced non-finite values')
 mask=(probabilities>=.5)&valid
 if not valid.any():raise ValueError('Images have no common valid pixels')
 overlay=np.zeros((size,size,4),dtype=np.uint8)
 overlay[mask]=[245,140,70,170] if before is not None else [110,224,240,170]
 preview=np.array(Image.fromarray(arr).resize((size,size)))
 return {'status':'real','model':key,'coveragePercent':round(float(mask.sum()/valid.sum()*100),3),'mask':png((mask*255).astype(np.uint8)),'overlay':png(overlay),'preview':png(preview),'width':size,'height':size,'validPixels':int(valid.sum()),'scope':'Uploaded aligned image pair' if before is not None else 'Selected polygon','note':'Threshold 0.5. RGB imagery may differ from the SpaceNet training distribution.'}

@torch.inference_mode()
def landcover(data,key,polygon):
 import rasterio
 from rasterio.io import MemoryFile
 from rasterio.features import geometry_mask
 from rasterio.warp import transform_geom
 with MemoryFile(data) as mem,mem.open() as src:
  if src.count!=6:raise ValueError('Upload six bands in order B02, B03, B04, B08, B11, B12; an RGB map screenshot cannot replace them')
  if src.width*src.height>1_048_576:raise ValueError('For local inference, crop the GeoTIFF to at most 1,048,576 pixels')
  if not src.crs:raise ValueError('GeoTIFF must include its coordinate reference system')
  if any(src.descriptions) and tuple(src.descriptions)!=BAND_NAMES:raise ValueError('GeoTIFF band descriptions must match B02 B03 B04 B08 B11 B12')
  raw=src.read(out_dtype='float32');valid=np.all(src.read_masks()>0,axis=0)&np.isfinite(raw).all(axis=0)
  if polygon:
   geom={'type':'Polygon','coordinates':[polygon+[polygon[0]]]}
   geom=transform_geom('EPSG:4326',src.crs,geom)
   valid &= geometry_mask([geom],raw.shape[1:],src.transform,invert=True)
  if not valid.any():raise ValueError('The GeoTIFF has no valid pixels inside the selected polygon')
  model,stats=load_model(key)
  mean=np.array(stats['band_mean'],dtype='float32')[:,None,None];std=np.array(stats['band_std'],dtype='float32')[:,None,None]
  normalized=(np.nan_to_num(raw)/stats['scale_divisor']-mean)/std
  h,w=valid.shape;pred=np.zeros((h,w),dtype=np.uint8)
  # Non-overlapping bounded tiles use the checkpoint's exact band normalization.
  for y in range(0,h,256):
   for x in range(0,w,256):
    tile=normalized[:,y:y+256,x:x+256];th,tw=tile.shape[1:]
    if not valid[y:y+th,x:x+tw].any():continue
    padded=np.zeros((6,256,256),dtype='float32');padded[:,:th,:tw]=tile
    logits=model(torch.from_numpy(padded[None]).to(DEVICE))
    if not torch.isfinite(logits).all():raise ValueError('Model produced non-finite values')
    pred[y:y+th,x:x+tw]=logits.argmax(1)[0,:th,:tw].cpu().numpy()
  counts=np.bincount(pred[valid],minlength=6);coverage=(counts/counts.sum()*100)[ORDER].round(3).tolist()
  overlay=np.zeros((h,w,4),dtype=np.uint8);overlay[:,:,:3]=PALETTE[pred];overlay[:,:,3]=valid*160
  rgb=np.moveaxis(raw[[2,1,0]],0,2);preview=np.zeros((h,w,4),dtype=np.uint8)
  for c in range(3):
   lo,hi=np.percentile(rgb[:,:,c][valid],[2,98]);preview[:,:,c]=np.clip((rgb[:,:,c]-lo)/max(hi-lo,1)*255,0,255).astype('uint8')
  preview[:,:,3]=valid*255
  mask=pred.copy();mask[~valid]=255
  profile=src.profile.copy();profile.update(count=1,dtype='uint8',nodata=255,compress='deflate')
  with MemoryFile() as output:
   with output.open(**profile) as target:target.write(mask,1)
   geotiff=base64.b64encode(output.read()).decode()
  return {'status':'real','model':key,'coverage':coverage,'mask':png(mask),'overlay':png(overlay),'preview':png(preview),'geotiff':'data:image/tiff;base64,'+geotiff,'width':w,'height':h,'validPixels':int(valid.sum()),'scope':'GeoTIFF intersected with selected polygon','classNames':list(CLASS_NAMES),'note':'Six-band inference; output mask preserves the uploaded CRS and transform.'}
