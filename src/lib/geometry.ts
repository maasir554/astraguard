import area from "@turf/area";
import kinks from "@turf/kinks";
export type Coordinate = [number, number];
export const SAMPLE: Coordinate[] = [
  [77.4388, 23.2552],
  [77.4455, 23.256],
  [77.4473, 23.2513],
  [77.4434, 23.2487],
  [77.4375, 23.2509],
];
export function polygon(
  points: Coordinate[],
): GeoJSON.Feature<GeoJSON.Polygon> {
  return {
    type: "Feature",
    properties: {},
    geometry: { type: "Polygon", coordinates: [[...points, points[0]]] },
  };
}
export function validatePolygon(points: Coordinate[]): string | null {
  if (points.length < 3)
    return "Add at least 3 vertices to complete your boundary.";
  if (
    points.some(
      ([x, y]) =>
        !Number.isFinite(x) ||
        !Number.isFinite(y) ||
        Math.abs(x) > 180 ||
        Math.abs(y) > 85,
    )
  )
    return "Use valid longitude (−180 to 180) and latitude (−85 to 85).";
  if (new Set(points.map((p) => p.join(","))).size !== points.length)
    return "Each vertex must be unique. Do not repeat the closing vertex.";
  if (
    Math.max(...points.map((p) => p[0])) -
      Math.min(...points.map((p) => p[0])) >
    180
  )
    return "Boundaries crossing the antimeridian are not supported.";
  if (kinks(polygon(points)).features.length)
    return "Boundary edges cannot cross. Adjust your vertices.";
  if (area(polygon(points)) < 1)
    return "Select a region larger than one square metre.";
  return null;
}
export function hectares(points: Coordinate[]) {
  return points.length > 2 ? area(polygon(points)) / 10000 : 0;
}
export function bounds(points: Coordinate[]): [Coordinate, Coordinate] {
  return [
    [
      Math.min(...points.map((p) => p[0])),
      Math.min(...points.map((p) => p[1])),
    ],
    [
      Math.max(...points.map((p) => p[0])),
      Math.max(...points.map((p) => p[1])),
    ],
  ];
}
export function parseCoordinates(value: string): Coordinate[] {
  const lines = value.trim().split("\n").filter(Boolean);
  if (lines.some((line) => line.split(",").some((v) => !v.trim())))
    throw new Error("Coordinate values cannot be empty.");
  const points = lines.map((line) =>
    line.split(",").map((v) => Number(v.trim())),
  );
  if (points.some((p) => p.length !== 2))
    throw new Error("Enter one longitude, latitude pair per line.");
  const error = validatePolygon(points as Coordinate[]);
  if (error) throw new Error(error);
  return points as Coordinate[];
}
