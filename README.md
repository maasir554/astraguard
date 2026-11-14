# AstraGuard

Local satellite analysis workspace: React + TypeScript + Vite + Tailwind frontend, FastAPI + PyTorch inference backend.

## Start on this machine

```sh
cd /Users/maasir/Projects/sem7project
npm run dev
```

Open http://127.0.0.1:5174. The command starts the UI and inference API (port 8000); Ctrl+C stops both. The Python environment and supplied weights are already installed. For a new environment, install Node and Python 3.11, run `npm ci`, then `npm run setup:backend`. Setup uses pip without a download cache. `ASTRAGUARD_PYTHON` can select the Python executable.

Use `npm run dev:ui` or `npm run dev:api` to run either service separately. The UI automatically uses sample fallback values when inference is unavailable, with a status on the affected result. Refresh analysis retries. The static hosted site has not been updated by this local integration.

## Model inputs

| Task | Architectures | Required input |
| --- | --- | --- |
| Building footprints | U-Net / ResNet34, DeepLabV3+ / ResNet50 | Current polygon satellite capture; runs automatically |
| Land-cover segmentation | DeepLabV3+, SegFormer | Georeferenced six-band Sentinel-2 GeoTIFF uploaded in the sidebar |
| Temporal building changes | Siamese U-Net, ChangeFormer | Uploaded, aligned before/after RGB PNG/JPEG images |
| Ten-class scene classification | Not supplied | Sample fallback only; disabled by default |

The land-cover GeoTIFF must contain **B02, B03, B04, B08, B11, B12**, in that order, with reflectance scaled by 10000 as used in training. If band descriptions are present they must match these names. A normal RGB Esri screenshot cannot substitute for the missing infrared bands. Include a CRS and overlap the selected polygon; no-data and pixels outside the polygon are excluded. Local upload limits: 64 MiB per file, 1,048,576 pixels for GeoTIFF and 16 megapixels for RGB. Land-cover processing uses bounded 256-pixel tiles.

Temporal images must have identical dimensions, geographic extent and pixel alignment, from different dates. Confirm alignment in the UI. Temporal results apply to the uploaded pair, not the drawn polygon. The app does not invent historical Esri images or register the pair automatically.

Polygon, architecture and upload changes trigger analysis automatically. Result cards show model status, class percentages and masks. Land-cover masks can also be downloaded as GeoTIFF with the original CRS and transform. Class IDs in masks follow training order: 0 other, 1 agriculture, 2 built_up, 3 natural_vegetation, 4 water, 5 bare_or_sparse; 255 is no-data. Display charts reorder these labels for readability. Building/change masks are 0/255 with a 0.5 probability threshold. These percentages are pixel coverage, not confidence or validated disaster severity.

## Structure and API

- `src/components`: map, floating panels, inference inputs, result visualization.
- `src/services/analysis.ts`: multipart API adapter, cancellation and fallback.
- `backend/app/main.py`: input validation, per-model failure handling and HTTP endpoints.
- `backend/app/inference.py`: preprocessing, lazy checkpoint loading, inference and mask export.
- `backend/app/architectures.py`: inference-only custom temporal architectures.
- `backend/vendor`: upstream source and pinned provenance in each `SOURCE.json`.
- `weights`: local checkpoints and SHA-256 manifest; excluded from version control.

`GET /api/health` reports device and checkpoint presence. `POST /api/analyze` accepts multipart `image`, JSON `polygon` (longitude/latitude pairs), JSON `options`, and optional `landcover`, `before`, `after`. Open http://127.0.0.1:8000/docs for the schema. Results include per-task `real`, `fallback` or `needs_input` statuses; failures in one model do not discard successful outputs from another.

Inference binds to localhost, defaults to CPU, serializes work and caches only one model. Set `ASTRAGUARD_WEIGHTS` to change the checkpoint directory; `ASTRAGUARD_DEVICE` can select another supported PyTorch device, though CPU is the tested default. Uploaded images and outputs are processed in memory and not persisted. Browser imagery requests still go to Esri.

## Verification and model limitations

```sh
npm run build
npm test
npm run test:api
npm run test:models
```

All six supplied checkpoints loaded strictly and passed forward-pass smoke tests on this Mac. `backend/scripts/smoke-results.json` records the synthetic-input checks; these establish execution, not predictive accuracy. Real-world accuracy needs evaluation against labeled imagery.

The Nikhil notebooks normalize each RGB band to 0–1 and then apply Albumentations normalization with its default 255 divisor. The adapter preserves that unusual double scaling to match the published pipeline. Confirm this preprocessing with the model author before relying on predictions. Esri imagery also differs from the SpaceNet training distribution. No EuroSAT ten-class checkpoint was supplied, so that task remains fallback.

Sources: https://github.com/NikhilR5577/AstraGuard-Models and https://github.com/AntimonyTrisulfide/AstraGuard-LandClassifier . Consult upstream terms before redistribution.

## Disk usage

The supplied land-cover archive was CRC-verified, extracted and deleted. The four `.pth.zip` files were native PyTorch containers: they were verified and renamed to `.pth` without extracting or duplicating their internal tensors. `weights/manifest.json` records the cleanup and hashes. Setup preserves this history on subsequent runs. Weights total about 720 MiB; the Python environment about 1.1 GiB. No training datasets or pretrained backbone downloads are needed.

## Automatic Sentinel-2 imagery

Land-cover analysis now fetches Sentinel-2 automatically when no GeoTIFF is uploaded. Select a polygon, optionally set the date range and maximum scene cloud percentage, and let analysis update. Blank dates search the last 90 days; ranges are capped at one year. An uploaded GeoTIFF overrides automatic retrieval; removing it restores retrieval.

The backend searches Element 84 Earth Search's `sentinel-2-c1-l2a` collection, newest first, and tries up to three recent matching scenes. It reads only the crop's COG windows, aligns B02/B03/B04/B08/B11/B12 to a 10 m projected grid (bilinear for reflectance), and applies each C1 asset's declared scale and offset before converting to the model's reflectance-times-10000 input convention. SWIR remains natively 20 m information despite resampling. Scene classification (nearest-neighbor) excludes clouds, cirrus, shadows, snow, defective and missing pixels. At least 98% scene footprint coverage and 20% usable polygon pixels are required; mosaicking across tiles is not implemented.

Result cards and JSON exports include the acquisition date, scene ID, scene cloud percentage, usable polygon percentage and source metadata URL. Coverage percentages describe **usable pixels only**, not the full polygon when masked pixels exist. Scene cloud cover is a whole-tile statistic, not a guarantee that the polygon is clear. The background Esri image may have a different date. This is not a live satellite feed.

No API key is needed. Polygon bounds and search dates are sent to the public Earth Search catalog; image windows are fetched from its AWS data bucket. No imagery files are written to disk: crops use a 32 MiB LRU memory cache with one-hour expiry. Native GDAL caches are also bounded. Crop bounds are limited to 1,048,576 pixels at 10 m. If a scene is unavailable, cloudy, incomplete or too large, the result explains the failure and uses the existing sample fallback; manual GeoTIFF upload remains available.

Provider documentation: https://github.com/Element84/earth-search and https://earth-search.aws.element84.com/v1/collections/sentinel-2-c1-l2a .
