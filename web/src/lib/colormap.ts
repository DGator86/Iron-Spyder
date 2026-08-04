/**
 * Colour ramps for the canvas fields.
 *
 * Stops are [position, r, g, b, a]. Alpha is part of the ramp, not a uniform
 * multiplier, because low-magnitude regions must fall away so stacked layers
 * stay readable.
 */

export type Stop = [number, number, number, number, number];
export type Ramp = Stop[];

/**
 * Forecast probability density.
 *
 * Turbo-like progression (navy -> blue -> cyan -> green -> yellow -> red) so
 * ordering is legible without relying on hue discrimination alone; luminance
 * also rises monotonically, which keeps it readable in greyscale.
 */
export const DENSITY_RAMP: Ramp = [
  [0.0, 8, 16, 34, 0],
  [0.04, 18, 40, 92, 40],
  [0.14, 30, 78, 170, 120],
  [0.3, 34, 168, 208, 180],
  [0.46, 60, 200, 150, 210],
  [0.62, 168, 220, 78, 232],
  [0.76, 250, 200, 50, 244],
  [0.88, 246, 132, 38, 250],
  [1.0, 232, 54, 44, 255],
];

/**
 * Net gamma exposure — pressure field that moves price.
 *
 * Diverging, zero-centred, matching the desk reference:
 *   blue  = negative GEX (dealers amplify — unstable trough)
 *   red   = positive GEX (dealers dampen — resistance / pin mass)
 *
 * Near-zero stays a dark navy wash (not a hole) so the field reads as a
 * continuous fluid pressure map instead of striped bands with gaps.
 */
export const GEX_RAMP: Ramp = [
  [0.0, 18, 56, 210, 235],
  [0.12, 28, 110, 230, 220],
  [0.26, 40, 170, 220, 195],
  [0.38, 55, 150, 170, 110],
  [0.5, 18, 28, 48, 55],
  [0.62, 180, 90, 40, 120],
  [0.74, 220, 100, 36, 195],
  [0.88, 236, 64, 38, 225],
  [1.0, 248, 42, 32, 245],
];

/** Implied volatility — single-hue violet, alpha carries magnitude. */
export const IV_RAMP: Ramp = [
  [0.0, 30, 20, 50, 0],
  [0.3, 120, 60, 180, 90],
  [0.65, 190, 110, 240, 160],
  [1.0, 240, 190, 255, 210],
];

/** Model disagreement — desaturated grey, per the "uncertainty" encoding. */
export const DISAGREEMENT_RAMP: Ramp = [
  [0.0, 90, 110, 140, 0],
  [0.5, 120, 138, 165, 90],
  [1.0, 190, 205, 225, 160],
];

/** Sample a ramp at t in [0,1], returning premultiplied-safe RGBA bytes. */
export function sampleRamp(
  ramp: Ramp,
  t: number,
): [number, number, number, number] {
  const x = Math.min(1, Math.max(0, t));
  if (x <= ramp[0][0]) return [ramp[0][1], ramp[0][2], ramp[0][3], ramp[0][4]];
  const last = ramp[ramp.length - 1];
  if (x >= last[0]) return [last[1], last[2], last[3], last[4]];

  for (let i = 0; i < ramp.length - 1; i += 1) {
    const a = ramp[i];
    const b = ramp[i + 1];
    if (x >= a[0] && x <= b[0]) {
      const span = b[0] - a[0];
      const w = span === 0 ? 0 : (x - a[0]) / span;
      return [
        Math.round(a[1] + w * (b[1] - a[1])),
        Math.round(a[2] + w * (b[2] - a[2])),
        Math.round(a[3] + w * (b[3] - a[3])),
        Math.round(a[4] + w * (b[4] - a[4])),
      ];
    }
  }
  return [last[1], last[2], last[3], last[4]];
}

/** Build a 256-entry lookup table so per-pixel work is a single index. */
export function buildLut(ramp: Ramp): Uint8ClampedArray {
  const lut = new Uint8ClampedArray(256 * 4);
  for (let i = 0; i < 256; i += 1) {
    const [r, g, b, a] = sampleRamp(ramp, i / 255);
    lut[i * 4] = r;
    lut[i * 4 + 1] = g;
    lut[i * 4 + 2] = b;
    lut[i * 4 + 3] = a;
  }
  return lut;
}

/** CSS gradient string for legends and swatches. */
export function rampToCss(ramp: Ramp, direction = "to right"): string {
  const stops = ramp
    .map(
      ([pos, r, g, b, a]) =>
        `rgba(${r},${g},${b},${(a / 255).toFixed(3)}) ${(pos * 100).toFixed(1)}%`,
    )
    .join(", ");
  return `linear-gradient(${direction}, ${stops})`;
}
