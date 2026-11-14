import {
  useEffect,
  useRef,
  useState,
  forwardRef,
  useImperativeHandle,
} from "react";
import * as maplibregl from "maplibre-gl";
import { type GeoJSONSource, type Map as GLMap } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
maplibregl.setWorkerUrl(workerUrl);
import {
  Plus,
  Minus,
  LocateFixed,
  Layers,
  MousePointer2,
  Pentagon,
  Undo2,
  Check,
  X,
} from "lucide-react";
import { Button } from "./ui/button";
import { bounds, polygon, type Coordinate } from "@/lib/geometry";
export interface MapHandle {
  capture: () => Promise<Blob>;
  flyTo: (center: Coordinate) => void;
  fit: () => void;
}
interface Props {
  points: Coordinate[];
  drawing: boolean;
  editing: boolean;
  onReady: () => void;
  onEditFinish: () => void;
  onVertex: (index: number, point: Coordinate) => void;
  onPoint: (p: Coordinate) => void;
  onFinish: () => void;
  onDraw: () => void;
  onUndo: () => void;
  onCancel: () => void;
}
const empty: GeoJSON.FeatureCollection = {
  type: "FeatureCollection",
  features: [],
};
function waitIdle(map: GLMap): Promise<void> {
  return new Promise((resolve, reject) => {
    if (map.loaded() && map.areTilesLoaded()) return resolve();
    const timeout = setTimeout(() => {
      map.off("idle", done);
      reject(
        new Error("Imagery is still loading. Wait a moment and try again."),
      );
    }, 15000);
    function done() {
      clearTimeout(timeout);
      resolve();
    }
    map.once("idle", done);
  });
}
export const MapWorkspace = forwardRef<MapHandle, Props>(
  function MapWorkspace(props, ref) {
    const scaleHost = useRef<HTMLDivElement>(null);
    const attributionHost = useRef<HTMLDivElement>(null);
    const el = useRef<HTMLDivElement>(null),
      map = useRef<GLMap | null>(null),
      latest = useRef(props);
    latest.current = props;
    const [ready, setReady] = useState(false),
      [error, setError] = useState(""),
      [layer, setLayer] = useState("satellite"),
      [position, setPosition] = useState("77.4424° E   23.2520° N"),
      [zoom, setZoom] = useState("15.0");
    useEffect(() => {
      const m = new maplibregl.Map({
        container: el.current!,
        center: [77.4424, 23.252],
        zoom: 15,
        maxZoom: 19,
        minZoom: 2,
        canvasContextAttributes: { preserveDrawingBuffer: true },
        style: {
          version: 8,
          sources: {
            satellite: {
              type: "raster",
              tiles: [
                "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
              ],
              tileSize: 256,
              maxzoom: 19,
              attribution:
                "Tiles © Esri, Maxar, Earthstar Geographics, and the GIS User Community",
            },
            streets: {
              type: "raster",
              tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
              tileSize: 256,
              maxzoom: 19,
              attribution: "© OpenStreetMap contributors",
            },
          },
          layers: [
            { id: "satellite", type: "raster", source: "satellite" },
            {
              id: "streets",
              type: "raster",
              source: "streets",
              layout: { visibility: "none" },
            },
          ],
        },
        attributionControl: { compact: false },
      });
      map.current = m;
      m.addControl(
        new maplibregl.ScaleControl({ maxWidth: 100 }),
        "bottom-left",
      );
      // Keep native controls (including attribution links) in dedicated footer slots.
      const scaleElement = m
        .getContainer()
        .querySelector(".maplibregl-ctrl-scale");
      const attributionElement = m
        .getContainer()
        .querySelector(".maplibregl-ctrl-attrib");
      if (scaleElement) scaleHost.current?.append(scaleElement);
      if (attributionElement)
        attributionHost.current?.append(attributionElement);
      m.on("load", () => {
        m.addSource("region", { type: "geojson", data: empty });
        m.addSource("vertices", { type: "geojson", data: empty });
        m.addLayer({
          id: "region-fill",
          type: "fill",
          source: "region",
          filter: ["==", "$type", "Polygon"],
          paint: { "fill-color": "#b7ee80", "fill-opacity": 0.16 },
        });
        m.addLayer({
          id: "region-line",
          type: "line",
          source: "region",
          paint: {
            "line-color": "#c6f699",
            "line-width": 2,
            "line-dasharray": [3, 2],
          },
        });
        m.addLayer({
          id: "vertices",
          type: "circle",
          source: "vertices",
          paint: {
            "circle-radius": 5,
            "circle-color": "#172219",
            "circle-stroke-color": "#d5ffa7",
            "circle-stroke-width": 2,
          },
        });
        setReady(true);
        latest.current.onReady();
      });
      m.on("click", (e) => {
        if (latest.current.drawing)
          latest.current.onPoint([
            Number(e.lngLat.lng.toFixed(6)),
            Number(e.lngLat.lat.toFixed(6)),
          ]);
      });
      m.on("move", () => {
        const p = m.getCenter();
        setPosition(
          `${Math.abs(p.lng).toFixed(4)}° ${p.lng < 0 ? "W" : "E"}   ${Math.abs(p.lat).toFixed(4)}° ${p.lat < 0 ? "S" : "N"}`,
        );
        setZoom(m.getZoom().toFixed(1));
      });
      m.on("error", () =>
        setError(
          "Some map tiles could not load. Check your connection or zoom out.",
        ),
      );
      m.on("idle", () => {
        if (m.isSourceLoaded("satellite")) setError("");
      });
      return () => {
        m.remove();
        map.current = null;
      };
    }, []);
    useEffect(() => {
      const m = map.current;
      if (!ready || !m) return;
      const p = props.points;
      (m.getSource("region") as GeoJSONSource).setData(
        p.length > 2
          ? polygon(p)
          : p.length > 1
            ? {
                type: "Feature",
                properties: {},
                geometry: { type: "LineString", coordinates: p },
              }
            : empty,
      );
      (m.getSource("vertices") as GeoJSONSource).setData({
        type: "FeatureCollection",
        features: p.map((coordinates) => ({
          type: "Feature",
          properties: {},
          geometry: { type: "Point", coordinates },
        })),
      });
      m.getCanvas().style.cursor = props.drawing ? "crosshair" : "";
      if (!props.drawing && !props.editing && p.length > 2)
        m.fitBounds(bounds(p), { padding: 70, maxZoom: 17, duration: 0 });
    }, [props.points, props.drawing, ready]);
    useEffect(() => {
      if (!ready || !props.editing || !map.current) return;
      const markers = props.points.map((point, index) => {
        const el = document.createElement("button");
        el.className = "vertex-handle";
        el.setAttribute(
          "aria-label",
          `Vertex ${index + 1}; use arrow keys to move`,
        );
        const marker = new maplibregl.Marker({ element: el, draggable: true })
          .setLngLat(point)
          .addTo(map.current!);
        marker.on("dragend", () => {
          const p = marker.getLngLat();
          latest.current.onVertex(index, [p.lng, p.lat]);
        });
        el.addEventListener("keydown", (e) => {
          if (
            !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key)
          )
            return;
          e.preventDefault();
          const pixel = map.current!.project(marker.getLngLat());
          const step = e.shiftKey ? 10 : 2;
          const p = map.current!.unproject([
            pixel.x +
              (e.key === "ArrowRight"
                ? step
                : e.key === "ArrowLeft"
                  ? -step
                  : 0),
            pixel.y +
              (e.key === "ArrowDown" ? step : e.key === "ArrowUp" ? -step : 0),
          ]);
          latest.current.onVertex(index, [p.lng, p.lat]);
        });
        return marker;
      });
      return () => markers.forEach((marker) => marker.remove());
    }, [ready, props.editing, props.points]);
    useImperativeHandle(
      ref,
      () => ({
        flyTo: (center) => map.current?.flyTo({ center, zoom: 15 }),
        fit: () => {
          if (props.points.length > 2)
            map.current?.fitBounds(bounds(props.points), {
              padding: 90,
              maxZoom: 17,
            });
        },
        capture: async () => {
          const m = map.current;
          if (!m || !ready) throw new Error("The map is not ready yet.");
          if (!m.isSourceLoaded("satellite"))
            throw new Error("Satellite imagery has not loaded.");
          m.setLayoutProperty("satellite", "visibility", "visible");
          m.setLayoutProperty("streets", "visibility", "none");
          setLayer("satellite");
          m.fitBounds(bounds(props.points), {
            padding: 70,
            duration: 0,
            maxZoom: 18,
          });
          await waitIdle(m);
          const ids = ["region-fill", "region-line", "vertices"];
          ids.forEach((id) => m.setLayoutProperty(id, "visibility", "none"));
          try {
            await new Promise<void>((r) => {
              m.once("render", () => r());
              m.triggerRepaint();
            });
            const source = m.getCanvas(),
              ratio = source.width / m.getContainer().clientWidth;
            const projected = props.points.map((p) => m.project(p));
            const minX = Math.floor(
                Math.min(...projected.map((p) => p.x)) * ratio,
              ),
              minY = Math.floor(Math.min(...projected.map((p) => p.y)) * ratio),
              w =
                Math.ceil(Math.max(...projected.map((p) => p.x)) * ratio) -
                minX,
              h =
                Math.ceil(Math.max(...projected.map((p) => p.y)) * ratio) -
                minY;
            const canvas = document.createElement("canvas");
            canvas.width = w;
            canvas.height = h;
            const ctx = canvas.getContext("2d")!;
            ctx.beginPath();
            projected.forEach((p, i) => {
              const x = p.x * ratio - minX,
                y = p.y * ratio - minY;
              if (i === 0) ctx.moveTo(x, y);
              else ctx.lineTo(x, y);
            });
            ctx.closePath();
            ctx.clip();
            ctx.drawImage(source, minX, minY, w, h, 0, 0, w, h);
            return await new Promise<Blob>((resolve, reject) =>
              canvas.toBlob(
                (b) =>
                  b
                    ? resolve(b)
                    : reject(new Error("Could not capture imagery.")),
                "image/png",
              ),
            );
          } finally {
            ids.forEach((id) =>
              m.setLayoutProperty(id, "visibility", "visible"),
            );
          }
        },
      }),
      [props.points, ready],
    );
    return (
      <div className="map-shell">
        <div
          ref={el}
          className="map-canvas"
          aria-label="Interactive satellite map"
        />
        <div className="map-dock">
          <div className="map-tools">
            <Button
              size="icon"
              variant={props.drawing ? "outline" : "default"}
              title="Pan map"
              aria-label="Pan map"
              onClick={props.onCancel}
            >
              <MousePointer2 />
            </Button>
            <Button
              size="icon"
              variant={props.drawing ? "default" : "outline"}
              title="Draw or edit polygon"
              aria-label="Draw or edit polygon"
              onClick={props.onDraw}
            >
              <Pentagon />
            </Button>
            <div className="tool-divider" />
            <Button
              size="icon"
              variant="outline"
              aria-label="Zoom in"
              onClick={() => map.current?.zoomIn()}
            >
              <Plus />
            </Button>
            <Button
              size="icon"
              variant="outline"
              aria-label="Zoom out"
              onClick={() => map.current?.zoomOut()}
            >
              <Minus />
            </Button>
            <Button
              size="icon"
              variant="outline"
              aria-label="Fit selected area"
              onClick={() =>
                props.points.length > 2 &&
                map.current?.fitBounds(bounds(props.points), { padding: 90 })
              }
            >
              <LocateFixed />
            </Button>
          </div>
          <button
            className="layer-switch"
            onClick={() => {
              const next = layer === "satellite" ? "streets" : "satellite";
              setLayer(next);
              map.current?.setLayoutProperty(
                "satellite",
                "visibility",
                next === "satellite" ? "visible" : "none",
              );
              map.current?.setLayoutProperty(
                "streets",
                "visibility",
                next === "streets" ? "visible" : "none",
              );
            }}
          >
            <Layers size={17} />
            <span>
              {layer === "satellite" ? "World Imagery" : "Street map"}
              <small>
                {layer === "satellite" ? "Esri satellite" : "OpenStreetMap"}
              </small>
            </span>
          </button>
        </div>
        <div className="north">
          <span> N </span>
          <div>↑</div>
        </div>
        {error && (
          <div className="map-error" role="alert">
            {error}
          </div>
        )}
        {props.editing && (
          <div className="draw-hint">
            <span>Drag a vertex to adjust the boundary.</span>
            <Button size="sm" onClick={props.onEditFinish}>
              <Check />
              Done
            </Button>
          </div>
        )}
        {props.drawing ? (
          <div className="draw-hint">
            <span>
              <b>{props.points.length} vertices</b> · Click the map to trace a
              boundary
            </span>
            <Button
              size="icon"
              variant="ghost"
              aria-label="Undo last vertex"
              onClick={props.onUndo}
            >
              <Undo2 />
            </Button>
            <Button
              size="sm"
              disabled={props.points.length < 3}
              onClick={props.onFinish}
            >
              <Check />
              Finish
            </Button>
            <Button
              size="icon"
              variant="ghost"
              aria-label="Cancel drawing"
              onClick={props.onCancel}
            >
              <X />
            </Button>
          </div>
        ) : (
          <div className="map-caption">
            <span className="corner-mark">⌜</span>
            <div>
              <b>Assess your area of interest.</b>
              <span>Draw a boundary to review satellite imagery.</span>
            </div>
          </div>
        )}
        <footer className="map-status" aria-label="Map information">
          <div
            ref={scaleHost}
            className="footer-scale"
            aria-label="Map scale"
          />
          <span className="footer-coordinates">{position}</span>
          <span className="footer-zoom">Zoom {zoom}</span>
          <span className="footer-projection">WGS 84</span>
          <div ref={attributionHost} className="footer-attribution" />
        </footer>
      </div>
    );
  },
);
