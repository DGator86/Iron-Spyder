/**
 * Rasterize a (time x price) scalar field to an RGBA bitmap.
 *
 * The alternative — one ECharts rect per cell — puts ~14k shapes on the canvas
 * per field and collapses under two or three stacked layers. Rasterizing gives
 * one draw call per layer regardless of lattice size, and the browser's own
 * bilinear filter smooths the nine-point quantile lattice into the continuous
 * wash the brief asks for.
 *
 * Orientation: field is indexed [timeIndex][priceIndex] with price ascending,
 * but canvas y grows downward, so rows are emitted bottom-up.
 */

import { buildLut, type Ramp } from "./colormap";

export interface FieldImageOptions {
  ramp: Ramp;
  /** Multiplies the ramp's own alpha. */
  opacity?: number;
  /**
   * "unit"      – values already in [0,1]
   * "max"       – scale by a high percentile of |value| (robust to one hot cell)
   * "diverging" – map [-m, +m] onto [0,1] with zero at 0.5
   * "column"    – normalize each time column against a blend of its own peak
   *               and the global peak
   */
  scale?: "unit" | "max" | "diverging" | "column";
  /** Percentile used by "max"/"diverging" scaling. */
  percentile?: number;
  /**
   * "column" scaling only. 0 = pure global scaling, 1 = pure per-column.
   *
   * A forecast field is a spike at t=0 that spreads with sqrt(t), so purely
   * global scaling leaves everything past the first few minutes near-black,
   * while purely per-column scaling implies the distant future is as certain as
   * the near. Blending in log space keeps the plume legible across the whole
   * horizon without claiming uniform confidence.
   */
  columnMix?: number;
}

export interface FieldImage {
  dataUrl: string;
  width: number;
  height: number;
  /** The value that mapped to the top of the ramp — for legend labelling. */
  peak: number;
}

function createCanvas(width: number, height: number): HTMLCanvasElement | null {
  if (typeof document === "undefined") return null;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

/** Robust upper bound: the p-th percentile of |value| across the field. */
function percentileAbs(field: number[][], p: number): number {
  const values: number[] = [];
  for (const row of field) {
    for (const v of row) {
      if (Number.isFinite(v) && v !== 0) values.push(Math.abs(v));
    }
  }
  if (values.length === 0) return 1;
  values.sort((a, b) => a - b);
  const idx = Math.min(
    values.length - 1,
    Math.max(0, Math.floor(values.length * p)),
  );
  return values[idx] || 1;
}

export function renderFieldImage(
  field: number[][],
  options: FieldImageOptions,
): FieldImage | null {
  const nT = field.length;
  if (nT === 0) return null;
  const nP = field[0].length;
  if (nP === 0) return null;

  const canvas = createCanvas(nT, nP);
  if (!canvas) return null;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  const scale = options.scale ?? "max";
  const opacity = options.opacity ?? 1;
  const lut = buildLut(options.ramp);

  let peak = 1;
  if (scale === "max" || scale === "diverging" || scale === "column") {
    peak = percentileAbs(field, options.percentile ?? 0.995);
  }

  // Per-column divisors for "column" scaling, blended against the global peak.
  const mix = Math.min(1, Math.max(0, options.columnMix ?? 0.75));
  const divisors: number[] =
    scale === "column"
      ? field.map((column) => {
          const columnPeak = Math.max(...column.filter(Number.isFinite), 0);
          if (columnPeak <= 0) return peak;
          return Math.exp(
            mix * Math.log(columnPeak) + (1 - mix) * Math.log(peak),
          );
        })
      : [];

  const image = ctx.createImageData(nT, nP);
  const data = image.data;

  for (let t = 0; t < nT; t += 1) {
    const column = field[t];
    for (let p = 0; p < nP; p += 1) {
      const raw = column[p];
      let norm: number;
      if (!Number.isFinite(raw)) {
        norm = 0;
      } else if (scale === "unit") {
        norm = raw;
      } else if (scale === "diverging") {
        norm = 0.5 + raw / (2 * peak);
      } else if (scale === "column") {
        norm = raw / (divisors[t] || peak);
      } else {
        norm = raw / peak;
      }
      norm = Math.min(1, Math.max(0, norm));

      const lutIndex = Math.round(norm * 255) * 4;
      // Flip vertically: price ascends up the screen.
      const y = nP - 1 - p;
      const offset = (y * nT + t) * 4;
      data[offset] = lut[lutIndex];
      data[offset + 1] = lut[lutIndex + 1];
      data[offset + 2] = lut[lutIndex + 2];
      data[offset + 3] = Math.round(lut[lutIndex + 3] * opacity);
    }
  }

  ctx.putImageData(image, 0, 0);
  return {
    dataUrl: canvas.toDataURL("image/png"),
    width: nT,
    height: nP,
    peak,
  };
}

/**
 * Rasterize only the forecast half of a field.
 *
 * The density surface is zero to the left of `startIndex`; cropping rather than
 * painting zeros keeps the image narrow and lets it be positioned precisely
 * against the forecast boundary instead of relying on transparent padding.
 */
export function renderForecastField(
  field: number[][],
  startIndex: number,
  options: FieldImageOptions,
): FieldImage | null {
  const clipped = field.slice(Math.max(0, startIndex));
  if (clipped.length === 0) return null;
  return renderFieldImage(clipped, options);
}
