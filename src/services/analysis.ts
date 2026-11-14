import type { Coordinate } from "@/lib/geometry";
export const LAND_CLASSES = [
  "AnnualCrop",
  "Forest",
  "HerbaceousVegetation",
  "Highway",
  "Industrial",
  "Pasture",
  "PermanentCrop",
  "Residential",
  "River",
  "SeaLake",
] as const;
export const COVER_LABELS = [
  "Agriculture",
  "Bare / sparse land",
  "Built-up",
  "Natural vegetation",
  "Other",
  "Water",
];
export const COLORS = [
  "#b7ee80",
  "#dcb887",
  "#c99579",
  "#56b89a",
  "#9e9eae",
  "#78bbdf",
];
export interface ModelOptions {
  classification: boolean;
  coverage: boolean;
  buildings: boolean;
  change: boolean;
  buildingModel: string;
  landModel: string;
  changeModel: string;
  aligned: boolean;
  sentinelAuto: boolean;
  sentinelStart: string;
  sentinelEnd: string;
  sentinelCloud: number;
}
export const DEFAULT_MODELS: ModelOptions = {
  classification: false,
  coverage: true,
  buildings: true,
  change: false,
  buildingModel: "building_deeplab",
  landModel: "land_segformer",
  changeModel: "change_siamese",
  aligned: false,
  sentinelAuto: true,
  sentinelStart: "",
  sentinelEnd: "",
  sentinelCloud: 20,
};
export interface ModelStatus {
  status: "real" | "fallback" | "needs_input";
  message: string;
  model?: string;
}
export interface Segmentation {
  status: "real" | "fallback";
  model?: string;
  coveragePercent?: number;
  mask?: string;
  overlay?: string;
  preview?: string;
  geotiff?: string;
  imagery?: {sceneId: string; acquiredAt: string; sceneCloudPercent: number; usablePixelPercent: number; resolutionMeters: number; source: string; sourceUrl: string; note: string; cached: boolean};
  scope: string;
  note?: string;
  width?: number;
  height?: number;
}
export interface AnalysisResult {
  demo: boolean;
  createdAt: string;
  classification: number[] | null;
  coverage: number[] | null;
  landcover: Segmentation | null;
  buildings: Segmentation | null;
  change: Segmentation | null;
  statuses: Record<string, ModelStatus>;
}
export interface AnalysisInput {
  image: Blob;
  polygon: Coordinate[];
  models: ModelOptions;
  landcover?: File | null;
  before?: File | null;
  after?: File | null;
  signal?: AbortSignal;
}
function fallback(input: AnalysisInput, reason: string): AnalysisResult {
  const statuses: Record<string, ModelStatus> = {};
  for (const key of [
    "classification",
    "coverage",
    "buildings",
    "change",
  ] as const)
    if (input.models[key])
      statuses[key] = { status: "fallback", message: reason };
  return {
    demo: true,
    createdAt: new Date().toISOString(),
    statuses,
    classification: input.models.classification
      ? [82.6, 5.1, 3.8, 0.4, 0.2, 2.4, 3.2, 0.7, 1, 0.6]
      : null,
    coverage: input.models.coverage ? [64.2, 8.3, 5.6, 17.4, 1.1, 3.4] : null,
    landcover: null,
    buildings: input.models.buildings
      ? {
          status: "fallback",
          coveragePercent: 12.4,
          scope: "Sample values; no mask generated",
        }
      : null,
    change: input.models.change
      ? {
          status: "fallback",
          coveragePercent: 3.2,
          scope: "Sample values; no mask generated",
        }
      : null,
  };
}
export async function analyze(input: AnalysisInput): Promise<AnalysisResult> {
  if (!input.image.size) throw new Error("The satellite image is empty.");
  const form = new FormData();
  form.append("image", input.image, "region.png");
  form.append("polygon", JSON.stringify(input.polygon));
  form.append("options", JSON.stringify(input.models));
  for (const key of ["landcover", "before", "after"] as const)
    if (input[key]) form.append(key, input[key]);
  const controller = new AbortController();
  const abort = () => controller.abort();
  input.signal?.addEventListener("abort", abort, { once: true });
  const timer = setTimeout(() => controller.abort(), 180000);
  try {
    if (input.signal?.aborted)
      throw new DOMException("Cancelled", "AbortError");
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: form,
      signal: controller.signal,
    });
    if (!response.ok) {
      let message = `Inference server returned ${response.status}`;
      try {
        message = (await response.json()).detail || message;
      } catch {}
      throw new Error(message);
    }
    const data = await response.json();
    if (!data.statuses || !data.createdAt)
      throw new Error("Invalid inference response");
    return data;
  } catch (error) {
    if (input.signal?.aborted) throw error;
    return fallback(
      input,
      error instanceof Error && error.name === "AbortError"
        ? "Inference timed out; showing sample values."
        : `Local inference unavailable: ${error instanceof Error ? error.message : "connection failed"}. Showing sample values.`,
    );
  } finally {
    clearTimeout(timer);
    input.signal?.removeEventListener("abort", abort);
  }
}
export function download(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export async function downloadImage(url: string, name: string) {
  download(await (await fetch(url)).blob(), name);
}
