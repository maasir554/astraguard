"""Bounded Sentinel-2 C1 L2A retrieval from Element 84's public STAC/COGs.

Only remote windows are read. A byte-bounded, one-hour memory cache stores crops;
no scenes, archives or temporary rasters are written to disk.
"""
from collections import OrderedDict
from datetime import date, timedelta
import json
import math
import time
from urllib.parse import urlparse

import httpx
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds, transform_geom

CATALOG = 'https://earth-search.aws.element84.com/v1'
COLLECTION = 'sentinel-2-c1-l2a'
BANDS = [('blue','B02'),('green','B03'),('red','B04'),('nir','B08'),('swir16','B11'),('swir22','B12')]
CACHE_BYTES = 32 * 1024 * 1024
CACHE_TTL = 3600
_cache = OrderedDict()


def date_range(start='', end=''):
    last = date.fromisoformat(end) if end else date.today()
    first = date.fromisoformat(start) if start else last - timedelta(days=90)
    if first > last or (last-first).days > 366:
        raise ValueError('Choose a date range of at most one year, with start before end.')
    if last > date.today():
        raise ValueError('The imagery end date cannot be in the future.')
    return first.isoformat(), last.isoformat()


def asset_url(asset):
    url = asset['href']
    parsed = urlparse(url)
    if parsed.scheme != 'https' or parsed.hostname != 'e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com':
        raise ValueError('Unexpected Sentinel-2 asset host')
    return url


def scaled_band(raw, asset):
    band = asset['raster:bands'][0]
    # C1 stores unharmonized DN. Apply its declared physical scale/offset,
    # then express reflectance * 10000 for the checkpoint's own normalization.
    return (raw.astype('float32') * float(band['scale']) + float(band['offset'])) * 10000


def remember(key, data, metadata):
    now = time.monotonic()
    for old in list(_cache):
        if now - _cache[old][0] > CACHE_TTL:
            del _cache[old]
    _cache[key] = (now, data, metadata)
    _cache.move_to_end(key)
    while sum(len(value[1]) for value in _cache.values()) > CACHE_BYTES:
        _cache.popitem(last=False)


def crop_scene(item, points):
    assets = item['assets']
    crs = rasterio.crs.CRS.from_epsg(item['properties']['proj:epsg'])
    bbox = [min(p[0] for p in points),min(p[1] for p in points),max(p[0] for p in points),max(p[1] for p in points)]
    left,bottom,right,top = transform_bounds('EPSG:4326',crs,*bbox)
    left,bottom,right,top = math.floor(left/10)*10,math.floor(bottom/10)*10,math.ceil(right/10)*10,math.ceil(top/10)*10
    width,height = round((right-left)/10),round((top-bottom)/10)
    if width <= 0 or height <= 0 or width*height > 1_048_576:
        raise ValueError('Select a smaller region: the 10 m crop must be at most 1 megapixel.')
    transform = from_origin(left,top,10,10)
    geom = transform_geom('EPSG:4326',crs,{'type':'Polygon','coordinates':[points+[points[0]]]})
    inside = geometry_mask([geom],(height,width),transform,invert=True)
    if inside.sum() < 4:
        raise ValueError('Select an area containing at least four Sentinel-2 pixels (10 m grid).')

    def read(asset, method):
        with rasterio.open(asset_url(asset)) as source:
            with WarpedVRT(source, crs=crs, transform=transform,width=width,height=height,
                           resampling=method,src_nodata=0,nodata=0) as vrt:
                return vrt.read(1)

    # GDAL uses HTTP range requests, bounded in-memory caches and finite waits.
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',
                      GDAL_HTTP_TIMEOUT='12',GDAL_HTTP_CONNECTTIMEOUT='6',GDAL_HTTP_MAX_RETRY='0',
                      GDAL_CACHEMAX=16*1024*1024,VSI_CACHE=False):
        scl = read(assets['scl'],Resampling.nearest)
        if ((scl>0)&inside).sum()/inside.sum() < .98:
            raise ValueError('Scene does not fully cover this polygon; try another date or a smaller polygon.')
        usable = inside & np.isin(scl,[2,4,5,6,7])
        if usable.sum()/inside.sum() < .2:
            raise ValueError('Too few clear pixels in this scene; widen the date range.')
        stack = []
        for key,_ in BANDS:
            raw = read(assets[key],Resampling.bilinear)
            usable &= raw != 0
            stack.append(scaled_band(raw,assets[key]))
    if usable.sum()/inside.sum() < .2:
        raise ValueError('Insufficient valid six-band pixels in this scene.')
    array = np.stack(stack)
    array[:,~usable] = -9999
    with MemoryFile() as memory:
        with memory.open(driver='GTiff',width=width,height=height,count=6,dtype='float32',
                         crs=crs,transform=transform,nodata=-9999,compress='deflate') as target:
            target.write(array)
            target.descriptions = tuple(name for _,name in BANDS)
        data = memory.read()
    props = item['properties']
    metadata = {'sceneId':item['id'],'acquiredAt':props['datetime'],'sceneCloudPercent':props['eo:cloud_cover'],
                'usablePixelPercent':round(float(usable.sum()/inside.sum()*100),1),'resolutionMeters':10,
                'collection':COLLECTION,'source':'Copernicus Sentinel-2 / Element 84 Earth Search',
                'sourceUrl':f"{CATALOG}/collections/{COLLECTION}/items/{item['id']}",
                'note':'Scene cloud percentage covers the entire tile. Clouds, shadows, snow and invalid pixels are excluded from polygon results. 20 m SWIR bands are resampled to 10 m; Esri imagery may have a different date.'}
    return data,metadata


def fetch_crop(points,start='',end='',max_cloud=20):
    if len(points)<3:
        raise ValueError('Draw a polygon to retrieve Sentinel-2 imagery.')
    first,last = date_range(start,end)
    bbox = [min(p[0] for p in points),min(p[1] for p in points),max(p[0] for p in points),max(p[1] for p in points)]
    if bbox[2]-bbox[0]>1 or bbox[3]-bbox[1]>1:
        raise ValueError('Select a smaller polygon for local Sentinel-2 retrieval.')
    key = json.dumps([points,first,last,max_cloud])
    cached = _cache.get(key)
    if cached and time.monotonic()-cached[0]<CACHE_TTL:
        _cache.move_to_end(key)
        return cached[1],dict(cached[2],cached=True)
    params = {'collections':COLLECTION,'bbox':','.join(map(str,bbox)),
              'datetime':f'{first}T00:00:00Z/{last}T23:59:59Z','limit':12,'sortby':'-properties.datetime',
              'query':json.dumps({'eo:cloud_cover':{'lte':max_cloud}})}
    try:
        response = httpx.get(f'{CATALOG}/search',params=params,timeout=20)
        response.raise_for_status()
        items = response.json()['features']
    except (httpx.HTTPError,KeyError,ValueError) as exc:
        raise ValueError('Sentinel-2 catalog unavailable. Retry or upload a six-band GeoTIFF.') from exc
    candidates = [i for i in items if i['properties'].get('eo:cloud_cover',101)<=max_cloud]
    if not candidates:
        raise ValueError('No Sentinel-2 scenes match these dates and cloud limit. Widen the dates or increase the cloud limit.')
    errors = []
    started = time.monotonic()
    for item in candidates[:3]:
        try:
            data,metadata = crop_scene(item,points)
            remember(key,data,metadata)
            return data,dict(metadata,cached=False)
        except Exception as exc:
            errors.append(str(exc))
            if time.monotonic()-started>80:
                break
    raise ValueError('Could not obtain a usable Sentinel-2 crop. '+errors[-1]+' Upload a GeoTIFF or change the search settings.')
