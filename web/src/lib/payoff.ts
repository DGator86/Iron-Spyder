import type { StrategyCandidate } from "@/lib/types";

/**
 * Expiry payoff geometry, derived from the legs.
 *
 * Zones and breakevens are computed from the leg set rather than read off the
 * candidate's reported fields, so the overlay can never draw a profit region
 * that disagrees with the strikes drawn beside it. The reported max profit /
 * max loss are returned alongside the computed ones so a mismatch is visible
 * instead of silently reconciled.
 */

export interface StrategyGeometry {
  profitZones: Array<[number, number]>;
  lossZones: Array<[number, number]>;
  breakevens: number[];
  shortStrikes: number[];
  longStrikes: number[];
  /** Computed from the payoff curve, in dollars per spread. */
  computedMaxProfit: number;
  computedMaxLoss: number;
  /** Payoff sampled across the evaluated range, for the payoff sparkline. */
  curve: Array<{ price: number; payoff: number }>;
}

/** Expiry payoff in dollars per one spread, at underlying price `s`. */
export function payoffAt(candidate: StrategyCandidate, s: number): number {
  let intrinsic = 0;
  for (const leg of candidate.legs) {
    const value =
      leg.right === "call" ? Math.max(0, s - leg.strike) : Math.max(0, leg.strike - s);
    intrinsic += leg.quantity * value;
  }
  const net = candidate.isCredit ? candidate.netPrice : -candidate.netPrice;
  return intrinsic * 100 + net;
}

export function strategyGeometry(candidate: StrategyCandidate): StrategyGeometry {
  const strikes = candidate.legs.map((l) => l.strike);
  if (strikes.length === 0) {
    return {
      profitZones: [],
      lossZones: [],
      breakevens: candidate.breakevens,
      shortStrikes: [],
      longStrikes: [],
      computedMaxProfit: candidate.maxProfit ?? 0,
      computedMaxLoss: candidate.maxLoss,
      curve: [],
    };
  }

  const lo = Math.min(...strikes) - 12;
  const hi = Math.max(...strikes) + 12;
  const steps = 480;

  const curve: Array<{ price: number; payoff: number }> = [];
  for (let i = 0; i <= steps; i += 1) {
    const price = lo + ((hi - lo) * i) / steps;
    curve.push({ price, payoff: payoffAt(candidate, price) });
  }

  // Zero crossings -> breakevens, with linear interpolation between samples.
  const breakevens: number[] = [];
  for (let i = 1; i < curve.length; i += 1) {
    const a = curve[i - 1];
    const b = curve[i];
    if ((a.payoff <= 0 && b.payoff > 0) || (a.payoff >= 0 && b.payoff < 0)) {
      const span = b.payoff - a.payoff;
      const w = span === 0 ? 0 : -a.payoff / span;
      breakevens.push(round2(a.price + w * (b.price - a.price)));
    }
  }

  const profitZones = extractZones(curve, (p) => p > 0);
  const lossZones = extractZones(curve, (p) => p < 0);

  const payoffs = curve.map((c) => c.payoff);

  return {
    profitZones,
    lossZones,
    breakevens: breakevens.length > 0 ? breakevens : candidate.breakevens,
    shortStrikes: unique(candidate.legs.filter((l) => l.quantity < 0).map((l) => l.strike)),
    longStrikes: unique(candidate.legs.filter((l) => l.quantity > 0).map((l) => l.strike)),
    computedMaxProfit: Math.round(Math.max(...payoffs)),
    computedMaxLoss: Math.round(Math.abs(Math.min(...payoffs))),
    curve,
  };
}

function extractZones(
  curve: Array<{ price: number; payoff: number }>,
  predicate: (payoff: number) => boolean,
): Array<[number, number]> {
  const zones: Array<[number, number]> = [];
  let start: number | null = null;

  for (let i = 0; i < curve.length; i += 1) {
    const inside = predicate(curve[i].payoff);
    if (inside && start === null) start = curve[i].price;
    if (!inside && start !== null) {
      zones.push([round2(start), round2(curve[i].price)]);
      start = null;
    }
  }
  if (start !== null) zones.push([round2(start), round2(curve[curve.length - 1].price)]);
  return zones;
}

function unique(values: number[]): number[] {
  return Array.from(new Set(values)).sort((a, b) => a - b);
}

function round2(v: number): number {
  return Math.round(v * 100) / 100;
}
