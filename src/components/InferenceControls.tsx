import type { ModelOptions } from "@/services/analysis";
interface Props {
  models: ModelOptions;
  onChange: (m: ModelOptions) => void;
  landcover: File | null;
  before: File | null;
  after: File | null;
  onFile: (key: "landcover" | "before" | "after", file: File | null) => void;
  onRetry: () => void;
  busy: boolean;
}
export function InferenceControls(p: Props) {
  const choices = [
    ["coverage", "Land-cover segmentation", "6 classes · Sentinel-2"],
    [
      "classification",
      "Scene classification",
      "10 classes · weights not supplied",
    ],
    [
      "buildings",
      "Building footprints",
      "Binary mask · current satellite image",
    ],
    ["change", "Temporal change", "Building changes · aligned image pair"],
  ] as const;
  return (
    <div className="inference-controls">
      <div className="section-label">Models</div>
      {choices.map(([key, title, description]) => (
        <label className="model-card" key={key}>
          <div>
            <b>{title}</b>
            <span>{description}</span>
          </div>
          <input
            type="checkbox"
            checked={p.models[key]}
            onChange={(e) =>
              p.onChange({ ...p.models, [key]: e.target.checked })
            }
          />
        </label>
      ))}
      {p.models.coverage && (
        <div className="input-group">
          <label>
            Land-cover architecture
            <select
              value={p.models.landModel}
              onChange={(e) =>
                p.onChange({ ...p.models, landModel: e.target.value })
              }
            >
              <option value="land_segformer">SegFormer</option>
              <option value="land_deeplab">DeepLabV3+</option>
            </select>
          </label>
          <label className="alignment-check">
            <input type="checkbox" checked={p.models.sentinelAuto} onChange={e => p.onChange({...p.models, sentinelAuto:e.target.checked})} />
            Fetch Sentinel-2 automatically
          </label>
          {p.models.sentinelAuto && <>
            <label>From<input type="date" value={p.models.sentinelStart} max={p.models.sentinelEnd || new Date().toISOString().slice(0,10)} onChange={e => p.onChange({...p.models,sentinelStart:e.target.value})} /></label>
            <label>Through<input type="date" value={p.models.sentinelEnd} min={p.models.sentinelStart} max={new Date().toISOString().slice(0,10)} onChange={e => p.onChange({...p.models,sentinelEnd:e.target.value})} /></label>
            <label>Maximum scene cloud cover<select value={p.models.sentinelCloud} onChange={e => p.onChange({...p.models,sentinelCloud:Number(e.target.value)})}>
              {[10,20,40,60,100].map(v => <option key={v} value={v}>{v}%</option>)}
            </select></label>
            <p>Blank dates use the last 90 days. Retrieves a recent clear scene for your polygon. First retrieval may take a minute. No files saved to disk.</p>
          </>}
          <label>
            Upload GeoTIFF instead (overrides automatic retrieval)
            <input
              type="file"
              accept=".tif,.tiff"
              onChange={(e) =>
                p.onFile("landcover", e.target.files?.[0] ?? null)
              }
            />
          </label>
          <p>
            6 bands: B02, B03, B04, B08, B11, B12. Must overlap the polygon.
            Maximum 1 megapixel / 64 MiB.
          </p>
          {p.landcover && (
            <button
              className="text-link"
              onClick={() => p.onFile("landcover", null)}
            >
              Remove {p.landcover.name}
            </button>
          )}
        </div>
      )}
      {p.models.buildings && (
        <label className="input-group">
          Building architecture
          <select
            value={p.models.buildingModel}
            onChange={(e) =>
              p.onChange({ ...p.models, buildingModel: e.target.value })
            }
          >
            <option value="building_deeplab">DeepLabV3+ · ResNet50</option>
            <option value="building_unet">U-Net · ResNet34</option>
          </select>
        </label>
      )}
      {p.models.change && (
        <div className="input-group">
          <label>
            Change architecture
            <select
              value={p.models.changeModel}
              onChange={(e) =>
                p.onChange({ ...p.models, changeModel: e.target.value })
              }
            >
              <option value="change_siamese">Siamese U-Net</option>
              <option value="change_former">ChangeFormer</option>
            </select>
          </label>
          {(["before", "after"] as const).map((key) => (
            <label key={key}>
              {key === "before" ? "Before image" : "After image"}
              <input
                type="file"
                accept="image/png,image/jpeg"
                onChange={(e) => p.onFile(key, e.target.files?.[0] ?? null)}
              />
            </label>
          ))}
          <label className="alignment-check">
            <input
              type="checkbox"
              checked={p.models.aligned}
              onChange={(e) =>
                p.onChange({ ...p.models, aligned: e.target.checked })
              }
            />
            These images show the same extent, pixel alignment and dimensions.
          </label>
          <p>
            PNG or JPEG, captured on different dates. Change results cover this
            uploaded pair, not the map polygon.
          </p>
        </div>
      )}
      <button className="text-link" disabled={p.busy} onClick={p.onRetry}>
        Refresh analysis
      </button>
    </div>
  );
}
