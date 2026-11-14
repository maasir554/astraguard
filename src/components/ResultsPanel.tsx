import { ScanLine, Download } from "lucide-react";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "./ui/tabs";
import { Button } from "./ui/button";
import {
  type AnalysisResult,
  type Segmentation,
  type ModelStatus,
  LAND_CLASSES,
  COVER_LABELS,
  COLORS,
  download,
  downloadImage,
} from "@/services/analysis";
import { polygon, type Coordinate } from "@/lib/geometry";
interface Props {
  result: AnalysisResult | null;
  image: string | null;
  points: Coordinate[];
  busy: boolean;
  area: number;
}
function Status({ value }: { value?: ModelStatus }) {
  return value ? (
    <div className={`inference-status ${value.status}`}>
      <b>
        {value.status === "real"
          ? "Local model"
          : value.status === "fallback"
            ? "Sample fallback"
            : "Input required"}
      </b>
      <span>{value.message}</span>
      {value.model && <small>{value.model.replaceAll("_", " ")}</small>}
    </div>
  ) : null;
}
function MaskView({ data, name }: { data: Segmentation; name: string }) {
  return (
    <>
      {data.preview && (
        <div className="mask-preview">
          <img src={data.preview} alt={`${name} input`} />
          {data.overlay && (
            <img
              className="mask-overlay"
              src={data.overlay}
              alt={`${name} prediction overlay`}
            />
          )}
        </div>
      )}
      {data.imagery && <div className="inference-status real">
        <b>Sentinel-2 · {new Date(data.imagery.acquiredAt).toLocaleDateString()}</b>
        <span>{data.imagery.sceneCloudPercent.toFixed(1)}% scene clouds · {data.imagery.usablePixelPercent}% usable polygon pixels · {data.imagery.resolutionMeters} m grid</span>
        <small>{data.imagery.note}</small>
        <a href={data.imagery.sourceUrl} target="_blank" rel="noreferrer">Scene metadata ↗</a>
      </div>}
      <p className="muted">{data.scope}</p>
      {data.note && <p className="muted">{data.note}</p>}
      {data.mask && (
        <Button
          variant="outline"
          size="sm"
          className="mt-3"
          onClick={() =>
            downloadImage(data.mask!, `astraguard-${name}-mask.png`)
          }
        >
          <Download />
          Mask PNG
        </Button>
      )}
      {data.geotiff && (
        <Button
          variant="outline"
          size="sm"
          className="mt-3 ml-2"
          onClick={() => downloadImage(data.geotiff!, `astraguard-${name}.tif`)}
        >
          GeoTIFF
        </Button>
      )}
    </>
  );
}
export function ResultsPanel({ result, image, points, busy, area }: Props) {
  const dominant = result?.classification
    ? result.classification.indexOf(Math.max(...result.classification))
    : -1;
  const coverTop = result?.coverage
    ? result.coverage.indexOf(Math.max(...result.coverage))
    : -1;
  return (
    <aside className="results-panel">
      <Tabs defaultValue="overview">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="input">Input image</TabsTrigger>
        </TabsList>
        <TabsContent value="overview">
          {!result ? (
            <div className="results-empty">
              <ScanLine size={32} />
              <h3>{busy ? "Running local inference…" : "Select an area"}</h3>
              <p>
                {busy
                  ? "Retrieving imagery and preparing masks. The first run may take a minute."
                  : "Analysis updates when your boundary, models or inputs change."}
              </p>
            </div>
          ) : (
            <>
              {result.classification && (
                <section className="result-section">
                  <div className="section-label">Scene classification</div>
                  <Status value={result.statuses.classification} />
                  <div className="prediction">
                    <div>
                      <small>Predicted class</small>
                      <h3>
                        {LAND_CLASSES[dominant].replace(
                          /([a-z])([A-Z])/g,
                          "$1 $2",
                        )}
                      </h3>
                    </div>
                  </div>
                  <div className="confidence">
                    <span>Confidence</span>
                    <strong>
                      {result.classification[dominant].toFixed(1)}%
                    </strong>
                  </div>
                  <details>
                    <summary>All classes</summary>
                    <div className="class-list">
                      {result.classification.map((v, i) => (
                        <div key={i}>
                          <span>{LAND_CLASSES[i]}</span>
                          <b>{v.toFixed(1)}%</b>
                        </div>
                      ))}
                    </div>
                  </details>
                </section>
              )}
              {result.statuses.coverage && (
                <section className="result-section">
                  <div className="section-label">Land-cover segmentation</div>
                  <Status value={result.statuses.coverage} />
                  {result.coverage && (
                    <>
                      <div className="coverage-head">
                        <h3>{COVER_LABELS[coverTop]}</h3>
                        <small>Largest class</small>
                      </div>
                      <div className="coverage-bar">
                        {result.coverage.map((v, i) => (
                          <span
                            key={i}
                            style={{ width: `${v}%`, background: COLORS[i] }}
                          />
                        ))}
                      </div>
                      <div className="coverage-list">
                        {result.coverage.map((v, i) => (
                          <div key={i}>
                            <span>
                              <i style={{ background: COLORS[i] }} />
                              {COVER_LABELS[i]}
                            </span>
                            <b>{v.toFixed(1)}%</b>
                          </div>
                        ))}
                      </div>
                    </>
                  )}
                  {result.landcover && (
                    <MaskView data={result.landcover} name="landcover" />
                  )}
                </section>
              )}
              {(["buildings", "change"] as const).map(
                (key) =>
                  result.statuses[key] && (
                    <section key={key} className="result-section">
                      <div className="section-label">
                        {key === "buildings"
                          ? "Building footprints"
                          : "Temporal building change"}
                      </div>
                      <Status value={result.statuses[key]} />
                      {result[key] && (
                        <>
                          <div className="confidence">
                            <span>
                              {key === "buildings"
                                ? "Building pixels"
                                : "Changed pixels"}
                            </span>
                            <strong>
                              {result[key]!.coveragePercent?.toFixed(2)}%
                            </strong>
                          </div>
                          <MaskView data={result[key]!} name={key} />
                        </>
                      )}
                    </section>
                  ),
              )}
              <Button
                variant="outline"
                className="w-full"
                onClick={() =>
                  download(
                    new Blob(
                      [
                        JSON.stringify(
                          {
                            ...result,
                            areaHectares: area,
                            region: polygon(points),
                            attribution:
                              "Tiles © Esri, Maxar, Earthstar Geographics, GIS User Community",
                          },
                          null,
                          2,
                        ),
                      ],
                      { type: "application/json" },
                    ),
                    "astraguard-report.json",
                  )
                }
              >
                <Download />
                Export report
              </Button>
            </>
          )}
        </TabsContent>
        <TabsContent value="input">
          {image ? (
            <>
              <div className="image-preview">
                <img
                  src={image}
                  alt="Satellite image masked to selected polygon"
                />
              </div>
              <p className="muted mt-3">
                RGB map image used for building segmentation. Land-cover uses
                retrieved Sentinel-2 bands or your GeoTIFF; temporal change uses the uploaded pair.
              </p>
              <Button
                variant="outline"
                className="w-full mt-4"
                onClick={() => downloadImage(image, "astraguard-region.png")}
              >
                <Download />
                Download PNG
              </Button>
              <p className="image-credit">
                © Esri, Maxar, Earthstar Geographics, GIS User Community
              </p>
            </>
          ) : (
            <p className="muted">
              Select an area to capture its satellite image.
            </p>
          )}
        </TabsContent>
      </Tabs>
    </aside>
  );
}
