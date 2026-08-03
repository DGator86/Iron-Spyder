/**
 * Reconstruction of a continuous forecast field from discrete quantile grids.
 *
 * The engine emits, per horizon, a quantile function tau -> price on a nine
 * point grid (0.01 .. 0.99). The radar needs P(price, time) over a dense
 * (time x price) lattice. Two interpolations get us there:
 *
 *  1. ACROSS TIME, in quantile space. We interpolate each tau's price between
 *     bracketing horizons rather than blending their densities. Blending two
 *     densities would produce a spurious bimodal smear whenever the horizons
 *     disagree; interpolating the quantile function is displacement (Wasserstein)
 *     interpolation, which slides the distribution instead of superposing it.
 *     The interpolation runs in sqrt(minutes) because diffusive price spread
 *     grows with the square root of time — linear-in-t would open the cone far
 *     too fast near the current timestamp.
 *
 *  2. ACROSS PRICE, by differencing the CDF. For each price bin we invert the
 *     quantile function at both bin edges and take the difference. That yields
 *     exact bin probabilities and avoids the numerical noise of differentiating
 *     an interpolated density.
 *
 * The result is mass-preserving: every time column sums to 1 before display
 * scaling, so colour intensity is comparable across the whole field.
 */

/** A quantile grid: ascending [tau, price] pairs. */
export type QuantileGrid = Array<[number, number]>;

export interface HorizonQuantiles {
  /** Minutes from now to this horizon. */
  minutes: number;
  grid: QuantileGrid;
}

// ---------------------------------------------------------------------------
// Quantile function primitives (mirrors spy_der/domain/forecast.py)
// ---------------------------------------------------------------------------

/** Piecewise-linear evaluation of the quantile function at `tau`. */
export function interpolateQuantile(grid: QuantileGrid, tau: number): number {
  if (grid.length === 0) throw new Error("empty quantile grid");
  if (tau <= grid[0][0]) return grid[0][1];
  if (tau >= grid[grid.length - 1][0]) return grid[grid.length - 1][1];
  for (let i = 0; i < grid.length - 1; i += 1) {
    const [t0, v0] = grid[i];
    const [t1, v1] = grid[i + 1];
    if (t0 <= tau && tau <= t1) {
      if (t1 === t0) return v0;
      return v0 + ((tau - t0) / (t1 - t0)) * (v1 - v0);
    }
  }
  return grid[grid.length - 1][1];
}

/**
 * Invert the quantile function to a CDF value at `price`.
 *
 * Saturates at the tabulated extremes rather than extrapolating, so a far
 * out-of-range strike returns 0 or 1 instead of a nonsense probability.
 */
export function invertQuantile(grid: QuantileGrid, price: number): number {
  if (grid.length === 0) throw new Error("empty quantile grid");
  const byPrice = [...grid].sort((a, b) => a[1] - b[1]);
  if (price <= byPrice[0][1]) return byPrice[0][0];
  if (price >= byPrice[byPrice.length - 1][1]) return byPrice[byPrice.length - 1][0];
  for (let i = 0; i < byPrice.length - 1; i += 1) {
    const [t0, v0] = byPrice[i];
    const [t1, v1] = byPrice[i + 1];
    if (v0 <= price && price <= v1) {
      if (v1 === v0) return t0;
      return t0 + ((price - v0) / (v1 - v0)) * (t1 - t0);
    }
  }
  return byPrice[byPrice.length - 1][0];
}

// ---------------------------------------------------------------------------
// Time interpolation
// ---------------------------------------------------------------------------

/**
 * Quantile grid at an arbitrary `minutes` ahead, interpolated between the
 * bracketing horizons in sqrt-time.
 *
 * Below the first horizon we interpolate against a degenerate grid collapsed to
 * `spot`, which makes the cone emerge from a point at t=0 instead of springing
 * open at full width.
 */
export function quantilesAtMinutes(
  horizons: HorizonQuantiles[],
  minutes: number,
  spot: number,
): QuantileGrid {
  if (horizons.length === 0) throw new Error("no horizons supplied");
  const sorted = [...horizons].sort((a, b) => a.minutes - b.minutes);
  const taus = sorted[0].grid.map(([tau]) => tau);

  const clamped = Math.max(0, minutes);
  if (clamped === 0) return taus.map((tau) => [tau, spot] as [number, number]);

  const last = sorted[sorted.length - 1];
  if (clamped >= last.minutes) return last.grid;

  // Degenerate point mass at t=0 anchors the near field.
  const anchor: HorizonQuantiles = {
    minutes: 0,
    grid: taus.map((tau) => [tau, spot] as [number, number]),
  };
  const track = [anchor, ...sorted];

  for (let i = 0; i < track.length - 1; i += 1) {
    const lo = track[i];
    const hi = track[i + 1];
    if (clamped >= lo.minutes && clamped <= hi.minutes) {
      const rLo = Math.sqrt(lo.minutes);
      const rHi = Math.sqrt(hi.minutes);
      const r = Math.sqrt(clamped);
      const w = rHi === rLo ? 0 : (r - rLo) / (rHi - rLo);
      return taus.map((tau, k) => {
        const a = lo.grid[k]?.[1] ?? spot;
        const b = hi.grid[k]?.[1] ?? spot;
        return [tau, a + w * (b - a)] as [number, number];
      });
    }
  }
  return last.grid;
}

// ---------------------------------------------------------------------------
// Surface assembly
// ---------------------------------------------------------------------------

export interface DensityResult {
  /** density[timeIndex][priceIndex], each time column summing to 1. */
  density: number[][];
  /** Per-column argmax price (the mode / highest-density path). */
  modePath: number[];
  /** Per-column median (tau = 0.5). */
  medianPath: number[];
  /** Per-column quantile tracks keyed by tau. */
  bands: Record<string, number[]>;
}

const BAND_TAUS: Array<[string, number]> = [
  ["p05", 0.05],
  ["p10", 0.1],
  ["p25", 0.25],
  ["p50", 0.5],
  ["p75", 0.75],
  ["p90", 0.9],
  ["p95", 0.95],
];

/**
 * Build the (time x price) probability field.
 *
 * `minutesAxis` must align 1:1 with the chart's time axis; entries <= 0 are
 * treated as realized time and produce an empty column, so the field only ever
 * paints to the right of the forecast boundary.
 */
export function buildDensitySurface(
  horizons: HorizonQuantiles[],
  minutesAxis: number[],
  priceAxis: number[],
  spot: number,
): DensityResult {
  const nT = minutesAxis.length;
  const nP = priceAxis.length;
  const density: number[][] = new Array(nT);
  const modePath: number[] = new Array(nT);
  const medianPath: number[] = new Array(nT);
  const bands: Record<string, number[]> = {};
  for (const [name] of BAND_TAUS) bands[name] = new Array(nT);

  // Bin edges midway between adjacent price samples.
  const edges = priceEdges(priceAxis);

  for (let t = 0; t < nT; t += 1) {
    const minutes = minutesAxis[t];
    const column = new Array<number>(nP).fill(0);

    if (minutes <= 0) {
      density[t] = column;
      modePath[t] = NaN;
      medianPath[t] = NaN;
      for (const [name] of BAND_TAUS) bands[name][t] = NaN;
      continue;
    }

    const grid = quantilesAtMinutes(horizons, minutes, spot);

    let total = 0;
    for (let p = 0; p < nP; p += 1) {
      const lo = invertQuantile(grid, edges[p]);
      const hi = invertQuantile(grid, edges[p + 1]);
      const mass = Math.max(0, hi - lo);
      column[p] = mass;
      total += mass;
    }
    if (total > 0) {
      for (let p = 0; p < nP; p += 1) column[p] /= total;
    }

    density[t] = column;

    let bestIdx = 0;
    for (let p = 1; p < nP; p += 1) if (column[p] > column[bestIdx]) bestIdx = p;
    modePath[t] = priceAxis[bestIdx];
    medianPath[t] = interpolateQuantile(grid, 0.5);
    for (const [name, tau] of BAND_TAUS) bands[name][t] = interpolateQuantile(grid, tau);
  }

  return { density, modePath, medianPath, bands };
}

/** Bin edges for a (possibly uneven) price axis: n samples -> n+1 edges. */
function priceEdges(priceAxis: number[]): number[] {
  const n = priceAxis.length;
  if (n === 0) return [];
  if (n === 1) return [priceAxis[0] - 0.5, priceAxis[0] + 0.5];
  const edges = new Array<number>(n + 1);
  for (let i = 1; i < n; i += 1) edges[i] = (priceAxis[i - 1] + priceAxis[i]) / 2;
  edges[0] = priceAxis[0] - (edges[1] - priceAxis[0]);
  edges[n] = priceAxis[n - 1] + (priceAxis[n - 1] - edges[n - 1]);
  return edges;
}

/**
 * Separable 3x3 box blur over the field.
 *
 * Purely cosmetic: it softens the banding that a nine-point quantile grid
 * leaves behind when stretched over a dense lattice. Applied after
 * normalization and never used for any reported probability.
 */
export function smoothField(field: number[][], passes = 1): number[][] {
  let out = field.map((row) => [...row]);
  for (let pass = 0; pass < passes; pass += 1) {
    const next = out.map((row) => [...row]);
    for (let t = 0; t < out.length; t += 1) {
      for (let p = 0; p < out[t].length; p += 1) {
        let sum = 0;
        let count = 0;
        for (let dt = -1; dt <= 1; dt += 1) {
          for (let dp = -1; dp <= 1; dp += 1) {
            const tt = t + dt;
            const pp = p + dp;
            if (tt < 0 || tt >= out.length) continue;
            if (pp < 0 || pp >= out[tt].length) continue;
            sum += out[tt][pp];
            count += 1;
          }
        }
        next[t][p] = count > 0 ? sum / count : out[t][p];
      }
    }
    out = next;
  }
  return out;
}

/**
 * Probability that price finishes above `level` at the given horizon.
 * Complement of the CDF, so it composes with the strike ladder directly.
 */
export function finishAbove(grid: QuantileGrid, level: number): number {
  return 1 - invertQuantile(grid, level);
}

/**
 * Probability of touching `level` before the horizon.
 *
 * Uses the reflection principle for driftless Brownian motion:
 * P(touch) ~= 2 * P(finish beyond), capped at 1. The engine reports exact
 * touch probabilities per strike where it has them; this is the fallback used
 * to fill the ladder between reported strikes.
 */
export function touchProbability(grid: QuantileGrid, level: number, spot: number): number {
  const beyond = level >= spot ? finishAbove(grid, level) : invertQuantile(grid, level);
  return Math.min(1, Math.max(0, 2 * beyond));
}
