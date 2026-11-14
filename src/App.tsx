import { InferenceControls } from "./components/InferenceControls";
import { FloatingPanel } from "./components/FloatingPanel";
import { useRef, useState, useEffect } from "react";
import {
  Mountain,
  ArrowUpRight,
  Pentagon,
  MapPin,
  ChevronRight,
  ScanLine,
  RotateCcw,
  Trash2,
  ArrowRight,
  LoaderCircle,
  Check,
  Code2,
  Leaf,
  Grid2X2,
  HelpCircle,
  X,
} from "lucide-react";
import { Button } from "./components/ui/button";
import { MapWorkspace, type MapHandle } from "./components/MapWorkspace";
import { ResultsPanel } from "./components/ResultsPanel";
import {
  SAMPLE,
  hectares,
  validatePolygon,
  parseCoordinates,
  type Coordinate,
} from "./lib/geometry";
import {
  analyze,
  DEFAULT_MODELS,
  type AnalysisResult,
} from "./services/analysis";
export default function App() {
  const map = useRef<MapHandle>(null);
  const [points, setPoints] = useState<Coordinate[]>(SAMPLE),
    [drawing, setDrawing] = useState(false),
    [editing, setEditing] = useState(false),
    [mapReady, setMapReady] = useState(false),
    [previous, setPrevious] = useState<Coordinate[]>(SAMPLE),
    [result, setResult] = useState<AnalysisResult | null>(null),
    [image, setImage] = useState<string | null>(null),
    [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [models, setModels] = useState(DEFAULT_MODELS),
    [query, setQuery] = useState(""),
    [coordinateMode, setCoordinateMode] = useState(false),
    [coordinates, setCoordinates] = useState(""),
    [help, setHelp] = useState(false);
  const [uploads, setUploads] = useState<{
    landcover: File | null;
    before: File | null;
    after: File | null;
  }>({ landcover: null, before: null, after: null });
  const [retry, setRetry] = useState(0);
  const area = hectares(points);
  const valid = points.length >= 3 && !drawing;
  const analysisQueue = useRef<Promise<void>>(Promise.resolve());
  useEffect(
    () => () => {
      if (image) URL.revokeObjectURL(image);
    },
    [image],
  );
  function invalidate() {
    setResult(null);
    setImage(null);
    setError("");
  }
  function draw() {
    if (busy) return;
    if (points.length >= 3 && !drawing) {
      setEditing(true);
      return;
    }
    startNew();
  }
  function startNew() {
    setEditing(false);
    setPrevious(points);
    setPoints([]);
    setDrawing(true);
    invalidate();
  }
  function cancel() {
    setEditing(false);
    if (drawing) {
      setPoints(previous);
      setDrawing(false);
      invalidate();
    }
  }
  function finish() {
    const issue = validatePolygon(points);
    if (issue) {
      setError(issue);
      return;
    }
    setDrawing(false);
    setError("");
  }
  function loadSample() {
    setPoints(SAMPLE);
    setDrawing(false);
    invalidate();
    map.current?.flyTo([77.4424, 23.252]);
  }
  useEffect(() => {
    let stale = false;
    const controller = new AbortController();
    setResult(null);
    setImage(null);
    if (
      !mapReady ||
      !valid ||
      (!models.classification &&
        !models.coverage &&
        !models.buildings &&
        !models.change)
    ) {
      setBusy(false);
      return;
    }
    const issue = validatePolygon(points);
    if (issue) {
      setError(issue);
      setBusy(false);
      return;
    }
    setBusy(true);
    const timer = window.setTimeout(() => {
      analysisQueue.current = analysisQueue.current
        .catch(() => {})
        .then(async () => {
          if (stale) return;
          try {
            setError("");
            const blob = await map.current!.capture();
            if (stale) return;
            const next = await analyze({
              image: blob,
              polygon: points,
              models,
              ...uploads,
              signal: controller.signal,
            });
            if (stale) return;
            setImage(URL.createObjectURL(blob));
            setResult(next);
          } catch (error) {
            if (!stale)
              setError(
                error instanceof Error
                  ? error.message
                  : "Analysis failed. Adjust the boundary to retry.",
              );
          } finally {
            if (!stale) setBusy(false);
          }
        });
    }, 600);
    return () => {
      stale = true;
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [points, drawing, mapReady, models, uploads, retry]);
  function locate(e: React.FormEvent) {
    e.preventDefault();
    if (busy) return;
    const locations: Record<string, Coordinate> = {
      bhopal: [77.4424, 23.252],
      delhi: [77.209, 28.614],
      pune: [73.8567, 18.5204],
      bengaluru: [77.5946, 12.9716],
    };
    let center = locations[query.trim().toLowerCase()];
    if (!center) {
      const p = query.split(",").map(Number);
      if (
        p.length === 2 &&
        query.trim() &&
        p.every(Number.isFinite) &&
        Math.abs(p[0]) <= 180 &&
        Math.abs(p[1]) <= 85
      )
        center = p as Coordinate;
    }
    if (center) {
      map.current?.flyTo(center);
      setError("");
    } else
      setError(
        "Enter longitude, latitude, or try Bhopal, Delhi, Pune or Bengaluru.",
      );
  }
  return (
    <div className="app">
      <header className="header">
        <a className="brand" href="/" aria-label="AstraGuard home">
          <div className="brand-icon">
            <Mountain size={24} />
          </div>
          <span>AstraGuard</span>
        </a>
        <div className="header-middle">
          <span>Workspace</span>
          <ChevronRight size={14} />
          <b>Untitled assessment</b>
        </div>
        <div className="header-actions">
          <Button
            variant="ghost"
            size="icon"
            aria-label="How to use AstraGuard"
            onClick={() => setHelp(!help)}
          >
            <HelpCircle />
          </Button>
          <span className="avatar">A</span>
        </div>
      </header>
      {help && (
        <div className="help-banner">
          <div>
            <strong>Your field guide</strong>
            <p>
              Find a location using longitude, latitude or a supported city.
              Draw at least three vertices, then choose Finish. Select your
              models. Analysis updates automatically when the boundary changes.
              Inspect or download the masked image in “Input image”.
            </p>
          </div>
          <Button
            variant="ghost"
            size="icon"
            aria-label="Close help"
            onClick={() => setHelp(false)}
          >
            <X />
          </Button>
        </div>
      )}
      <main className="workspace">
        <FloatingPanel title="Area of interest" side="left">
          <aside className="setup-panel">
            <div className="panel-heading">
              <p>Select an area for your satellite assessment.</p>
            </div>
            <form className="search-form" onSubmit={locate}>
              <MapPin size={16} />
              <input
                aria-label="Search city or coordinates"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="City or longitude, latitude"
              />
              <button aria-label="Go to location">
                <ArrowRight size={16} />
              </button>
            </form>
            <p className="search-help">
              Bhopal, Delhi, Pune, Bengaluru or coordinates
            </p>
            <div className="section-label">
              <span className="step-title">Select your region</span>
            </div>
            <Button
              disabled={busy}
              onClick={draw}
              variant="outline"
              className={`draw-button ${drawing ? "active" : ""}`}
            >
              <Pentagon />
              <span>
                {drawing
                  ? "Drawing boundary…"
                  : points.length >= 3
                    ? "Edit polygon"
                    : "Draw a polygon"}
              </span>
              <span className="shortcut">＋</span>
            </Button>
            {points.length >= 3 && !drawing && (
              <button className="text-link" disabled={busy} onClick={startNew}>
                Draw a new polygon
              </button>
            )}
            <button
              className="text-link"
              disabled={busy}
              onClick={() => {
                setCoordinateMode(!coordinateMode);
                setCoordinates(points.map((p) => p.join(", ")).join("\n"));
              }}
            >
              <Code2 size={14} />
              {coordinateMode
                ? "Hide coordinates"
                : "Enter coordinates manually"}
              <ArrowUpRight size={13} />
            </button>
            {coordinateMode && (
              <div className="coordinate-editor">
                <label htmlFor="coords">
                  Longitude, latitude · one pair per line
                </label>
                <textarea
                  id="coords"
                  rows={5}
                  value={coordinates}
                  onChange={(e) => setCoordinates(e.target.value)}
                />
                <Button
                  size="sm"
                  disabled={busy}
                  onClick={() => {
                    try {
                      setPoints(parseCoordinates(coordinates));
                      setDrawing(false);
                      invalidate();
                      setCoordinateMode(false);
                    } catch (e) {
                      setError((e as Error).message);
                    }
                  }}
                >
                  Apply boundary
                </Button>
              </div>
            )}
            <div className={`region-card ${valid ? "selected" : ""}`}>
              <div className="region-card-heading">
                <span>
                  <span className={valid ? "live-dot" : "inactive-dot"} />
                  {drawing
                    ? "Boundary in progress"
                    : valid
                      ? "Region selected"
                      : "No region selected"}
                </span>
                <Button
                  variant="ghost"
                  size="icon"
                  disabled={busy || !points.length}
                  aria-label="Clear region"
                  onClick={() => {
                    setPoints([]);
                    setDrawing(false);
                    invalidate();
                  }}
                >
                  <Trash2 size={15} />
                </Button>
              </div>
              <div className="region-metrics">
                <div>
                  <strong>
                    {area.toFixed(2)}
                    <small> ha</small>
                  </strong>
                  <span>Total area</span>
                </div>
                <div>
                  <strong>{points.length.toString().padStart(2, "0")}</strong>
                  <span>Vertices</span>
                </div>
              </div>
              <div className="region-card-footer">
                <Pentagon size={13} />
                <span>WGS 84 · EPSG:4326</span>
                {valid && <Check size={13} />}
              </div>
            </div>
            <button
              className="text-link sample-link"
              disabled={busy}
              onClick={loadSample}
            >
              <RotateCcw size={13} />
              Try the Bhopal sample region
            </button>
            <div className="panel-divider" />
            <InferenceControls
              models={models}
              onChange={setModels}
              {...uploads}
              busy={busy}
              onRetry={() => setRetry((v) => v + 1)}
              onFile={(key, file) => {
                if (file && file.size > 64 * 1024 * 1024) {
                  setError("Each file must be under 64 MiB");
                  return;
                }
                setUploads((u) => ({ ...u, [key]: file }));
                if (key !== "landcover")
                  setModels((m) => ({ ...m, aligned: false }));
              }}
            />
            <div className="analysis-action">
              {error && (
                <div role="alert" className="error-message">
                  {error}
                </div>
              )}
              {busy && (
                <p role="status">
                  <LoaderCircle className="animate-spin" size={14} /> Updating
                  analysis…
                </p>
              )}
            </div>
            <div className="source-note">
              <span>Imagery source</span>
              <b>
                Esri World Imagery <ArrowUpRight size={13} />
              </b>
              <p>
                Satellite and aerial imagery.
                <br />
                Resolution varies by location.
              </p>
            </div>
          </aside>
        </FloatingPanel>
        <div className={busy ? "map-container processing" : "map-container"}>
          <MapWorkspace
            ref={map}
            points={points}
            drawing={drawing}
            editing={editing}
            onReady={() => setMapReady(true)}
            onEditFinish={() => setEditing(false)}
            onVertex={(index, point) => {
              const next = points.map((p, i) => (i === index ? point : p));
              const issue = validatePolygon(next);
              if (issue) {
                setError(issue);
                setPoints([...points]);
              } else {
                setPoints(next);
                setError("");
              }
            }}
            onPoint={(p) => {
              setPoints((prev) => [...prev, p]);
              setError("");
            }}
            onFinish={finish}
            onDraw={draw}
            onUndo={() => setPoints((p) => p.slice(0, -1))}
            onCancel={cancel}
          />
          {busy && (
            <div className="capture-shield">
              <LoaderCircle className="animate-spin" />
              Preparing satellite input
            </div>
          )}
        </div>
        <FloatingPanel title="Assessment report" side="right">
          <ResultsPanel
            result={result}
            image={image}
            points={points}
            busy={busy}
            area={area}
          />
        </FloatingPanel>
      </main>
    </div>
  );
}
