from datetime import date,timedelta
import numpy as np
import pytest
from backend.app import sentinel


def test_dates_and_reflectance_offset():
    start,end=sentinel.date_range()
    assert date.fromisoformat(end)-date.fromisoformat(start)==timedelta(days=90)
    with pytest.raises(ValueError):sentinel.date_range('2025-03-01','2025-02-01')
    with pytest.raises(ValueError):sentinel.date_range('2020-01-01','2025-01-01')
    asset={'raster:bands':[{'scale':.0001,'offset':-.1}]}
    np.testing.assert_allclose(sentinel.scaled_band(np.array([1000,2000]),asset),[0,1000],atol=.001)


def test_memory_cache_stays_bounded(monkeypatch):
    monkeypatch.setattr(sentinel,'CACHE_BYTES',10)
    monkeypatch.setattr(sentinel,'_cache',sentinel.OrderedDict())
    sentinel.remember('first',b'123456',{})
    sentinel.remember('second',b'abcdef',{})
    assert list(sentinel._cache)==['second']


def test_no_matching_scenes_is_actionable(monkeypatch):
    class Response:
        def raise_for_status(self):pass
        def json(self):return {'features':[]}
    monkeypatch.setattr(sentinel.httpx,'get',lambda *a,**k:Response())
    with pytest.raises(ValueError,match='No Sentinel-2 scenes'):
        sentinel.fetch_crop([[77,23],[77.01,23],[77.01,23.01]])


def test_untrusted_asset_host_is_rejected():
    with pytest.raises(ValueError):sentinel.asset_url({'href':'http://127.0.0.1/private.tif'})


def test_auto_retrieval_failure_preserves_building_inference(monkeypatch):
    from backend.app.main import run,Options
    from backend.app import inference
    def unavailable(*args):raise ValueError('Catalog unavailable')
    monkeypatch.setattr(sentinel,'fetch_crop',unavailable)
    monkeypatch.setattr(inference,'binary',lambda *args:{'status':'real','coveragePercent':10})
    result=run(b'image',None,None,None,[[77,23],[77.01,23],[77.01,23.01]],Options(sentinelAuto=True,classification=False))
    assert result['statuses']['coverage']['status']=='fallback'
    assert result['statuses']['buildings']['status']=='real'
