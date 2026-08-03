/**
 * Deterministic synthetic radar data.
 *
 * The engine's status API binds loopback on the VPS, so a Vercel deployment has
 * nothing to talk to until a reverse proxy is configured. Rather than render an
 * empty shell, the BFF falls back to this generator. It is seeded, so the field
 * is stable within a minute and evolves smoothly across minutes — a reviewer
 * sees a plausible session, not noise.
 *
 * Every number here is fabricated. `RadarSnapshot.source` is set to "synthetic"
 * and the UI must surface that; nothing in this file may be presented as a real
 * market observation.
 */

import { buildDensitySurface, smoothField, type HorizonQuantiles } from "./density";
import { strategyGeometry } from "./payoff";
import type {
  ForecastChartPayload,
  Horizon,
  InterpretationPayload,
  RadarSnapshot,
  StrategyCandidate,
  StrikeRow,
  SystemPayload,
} from "./types";

// ---------------------------------------------------------------------------
// Seeded RNG
// ---------------------------------------------------------------------------

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller, so the walk has Gaussian rather than uniform increments. */
function gauss(rand: () => number): number {
  const u = Math.max(1e-9, rand());
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ---------------------------------------------------------------------------
// Session geometry
// ---------------------------------------------------------------------------

const SPOT = 531.26;
const CALL_WALL = 538;
const PUT_WALL = 525;
const GAMMA_FLIP = 532.4;
const VOL_TRIGGER = 534.1;
const PIN_STRIKE = 532;

const HORIZON_MINUTES: Record<Horizon, number> = {
  "5m": 5,
  "15m": 15,
  "30m": 30,
  "60m": 60,
  eod: 390,
  "1d": 780,
  expiry: 390,
};

const TIME_COLUMNS = 150;
const PRICE_ROWS = 92;
const HISTORY_MINUTES = 120;

/**
 * Minutes in a trading year: 252 sessions of 390 minutes.
 *
 * Not 252 * 24 * 60 — implied vol is quoted per trading year, and scaling by
 * calendar minutes understates sigma by sqrt(1440/390) ~= 1.9x, which collapses
 * the forecast cone to roughly half its true width.
 */
const MINUTES_PER_TRADING_YEAR = 252 * 390;

/** Session open in ET, expressed as minutes from midnight. */
const SESSION_OPEN = 9 * 60 + 30;
const SESSION_CLOSE = 16 * 60;

function sessionMinuteNow(date: Date): number {
  // Anchor the synthetic session to a plausible mid-morning point so the
  // canvas always has both realized and forecast territory to draw.
  const elapsed = (date.getUTCHours() * 60 + date.getUTCMinutes()) % 330;
  return SESSION_OPEN + 45 + elapsed;
}

function clockLabel(minuteOfDay: number): string {
  const h = Math.floor(minuteOfDay / 60);
  const m = Math.floor(minuteOfDay % 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}

// ---------------------------------------------------------------------------
// Quantile construction
// ---------------------------------------------------------------------------

const TAUS = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99];

/** Inverse standard normal (Acklam). Used to shape the synthetic quantiles. */
function probit(p: number): number {
  const a = [-39.6968302866538, 220.946098424521, -275.928510446969, 138.357751867269, -30.6647980661472, 2.50662827745924];
  const b = [-54.4760987982241, 161.585836858041, -155.698979859887, 66.8013118877197, -13.2806815528857];
  const c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184, -2.54973253934373, 4.37466414146497, 2.93816398269878];
  const d = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742];
  const pl = 0.02425;
  if (p < pl) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > 1 - pl) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  const q = p - 0.5;
  const r = q * q;
  return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

/**
 * Quantile grid at a horizon.
 *
 * Base is lognormal-ish diffusion from `spot`, then bent two ways so the field
 * looks like a real 0DTE session rather than a symmetric cone:
 *  - mean reversion pulls the median toward the pin strike as time grows
 *  - the wall between spot and each tail compresses that tail (dealers defend)
 */
function horizonQuantiles(minutes: number, spot: number, atmIv: number, seed: number): HorizonQuantiles {
  const rand = mulberry32(seed + Math.round(minutes));
  const years = minutes / MINUTES_PER_TRADING_YEAR;
  const sigma = atmIv * Math.sqrt(Math.max(years, 1e-6));

  // Pin pull strengthens with time-to-close but saturates.
  const pull = Math.min(0.55, minutes / 700);
  const center = spot + (PIN_STRIKE - spot) * pull + gauss(rand) * 0.05;

  const grid = TAUS.map((tau) => {
    const z = probit(tau);
    let price = center * Math.exp(sigma * z - 0.5 * sigma * sigma);

    // Wall compression: the further past a wall, the more the tail is squeezed.
    if (price > CALL_WALL) price = CALL_WALL + (price - CALL_WALL) * 0.55;
    if (price < PUT_WALL) price = PUT_WALL - (PUT_WALL - price) * 0.6;

    return [tau, price] as [number, number];
  });

  return { minutes, grid };
}

// ---------------------------------------------------------------------------
// GEX profile
// ---------------------------------------------------------------------------

/**
 * Net gamma exposure as a function of spot.
 *
 * Modelled as a sum of Gaussian bumps at the major strikes: positive above the
 * flip, negative below it, with the wall strikes carrying the most mass. The
 * zero crossing is placed at GAMMA_FLIP by construction.
 */
function gexAt(price: number): number {
  const bump = (center: number, weight: number, width: number) =>
    weight * Math.exp(-((price - center) ** 2) / (2 * width * width));

  return (
    bump(CALL_WALL, 2.4, 3.2) +
    bump(PIN_STRIKE, 1.3, 1.8) +
    bump(GAMMA_FLIP + 2, 0.9, 2.4) -
    bump(PUT_WALL, 2.1, 3.0) -
    bump(GAMMA_FLIP - 4, 1.5, 2.6) -
    bump(SPOT - 8, 0.7, 3.5)
  );
}

// ---------------------------------------------------------------------------
// Snapshot
// ---------------------------------------------------------------------------

export interface MockOptions {
  horizon: Horizon;
  expiration: string;
  /** Minutes back from live; 0 = now. Drives replay. */
  replayOffset?: number;
  now?: Date;
}

export function generateSnapshot(opts: MockOptions): RadarSnapshot {
  const now = opts.now ?? new Date();
  const replayOffset = opts.replayOffset ?? 0;
  const minuteSeed = Math.floor(now.getTime() / 60000) + replayOffset;
  const rand = mulberry32(minuteSeed);

  const nowMinute = sessionMinuteNow(now) + replayOffset;
  const horizonMinutes = HORIZON_MINUTES[opts.horizon];
  const atmIv = 0.178;

  // ---- Realized price path -------------------------------------------
  const historyStart = Math.max(SESSION_OPEN, nowMinute - HISTORY_MINUTES);
  const historyLen = Math.max(2, nowMinute - historyStart);
  const historicalPrice: Array<{ time: string; value: number }> = [];
  const historicalVwap: Array<{ time: string; value: number }> = [];

  // Walk backwards from spot so the path terminates exactly at SPOT.
  const walk: number[] = new Array(historyLen + 1);
  walk[historyLen] = SPOT;
  const walkRand = mulberry32(minuteSeed ^ 0x5f3759df);
  for (let i = historyLen - 1; i >= 0; i -= 1) {
    walk[i] = walk[i + 1] - gauss(walkRand) * 0.16 - 0.004;
  }

  let vwapNum = 0;
  let vwapDen = 0;
  for (let i = 0; i <= historyLen; i += 1) {
    const minute = historyStart + i;
    const value = walk[i];
    const volume = 800 + Math.abs(gauss(walkRand)) * 400;
    vwapNum += value * volume;
    vwapDen += volume;
    historicalPrice.push({ time: clockLabel(minute), value: round2(value) });
    historicalVwap.push({ time: clockLabel(minute), value: round2(vwapNum / vwapDen) });
  }
  const vwap = round2(vwapNum / vwapDen);

  // ---- Axes ------------------------------------------------------------
  // The axis must always contain the structural levels — a canvas that crops the
  // call wall out of frame cannot answer "where is resistance", which is the
  // whole point of the surface.
  const diffusionSpan = atmIv * SPOT * Math.sqrt(horizonMinutes / MINUTES_PER_TRADING_YEAR) * 3.2;
  const halfSpan = Math.max(
    6,
    diffusionSpan,
    Math.abs(CALL_WALL - SPOT) + 2.5,
    Math.abs(SPOT - PUT_WALL) + 2.5,
  );
  const priceLow = SPOT - halfSpan;
  const priceHigh = SPOT + halfSpan;
  const priceAxis = Array.from(
    { length: PRICE_ROWS },
    (_, i) => priceLow + ((priceHigh - priceLow) * i) / (PRICE_ROWS - 1),
  );

  const windowStart = historyStart;
  const windowEnd = nowMinute + horizonMinutes;
  const timeAxis: string[] = [];
  const minutesAxis: number[] = [];
  for (let i = 0; i < TIME_COLUMNS; i += 1) {
    const minute = windowStart + ((windowEnd - windowStart) * i) / (TIME_COLUMNS - 1);
    timeAxis.push(clockLabel(minute));
    minutesAxis.push(minute - nowMinute);
  }
  const forecastStartIndex = minutesAxis.findIndex((m) => m > 0);

  // ---- Forecast field ---------------------------------------------------
  const horizons: HorizonQuantiles[] = [5, 15, 30, 60, 120, 240, 390]
    .filter((m) => m <= Math.max(horizonMinutes, 60) * 2.5)
    .map((m) => horizonQuantiles(m, SPOT, atmIv, minuteSeed));
  if (horizons.length === 0) horizons.push(horizonQuantiles(horizonMinutes, SPOT, atmIv, minuteSeed));

  const built = buildDensitySurface(horizons, minutesAxis, priceAxis, SPOT);
  const density = smoothField(built.density, 2);

  // ---- GEX surface (price-dependent, replicated across time) -----------
  // GEX is a function of spot at this snapshot, not of time. Replicating the
  // profile across the time axis is the honest projection onto the canvas.
  const gexColumn = priceAxis.map((p) => gexAt(p));
  const gexSurface = minutesAxis.map(() => gexColumn);

  // ---- Flow arrows ------------------------------------------------------
  // Arrows point toward the price dealers are pushed to hedge into: in positive
  // gamma they lean back toward the flip, in negative gamma they lean away.
  const vectors = [];
  const arrowTimeStride = Math.max(1, Math.floor(TIME_COLUMNS / 22));
  const arrowPriceStride = Math.max(1, Math.floor(PRICE_ROWS / 14));
  for (let t = 0; t < TIME_COLUMNS; t += arrowTimeStride) {
    for (let p = 0; p < PRICE_ROWS; p += arrowPriceStride) {
      const price = priceAxis[p];
      const g = gexAt(price);
      const toward = g >= 0 ? Math.sign(GAMMA_FLIP - price) : Math.sign(price - GAMMA_FLIP);
      const strength = Math.min(1, Math.abs(g) / 2.4);
      vectors.push({
        timeIndex: t,
        priceIndex: p,
        dx: 1,
        dy: toward * strength,
        strength,
      });
    }
  }

  // ---- Strike ladder ----------------------------------------------------
  const strikes: StrikeRow[] = [];
  const lastGrid = horizons[horizons.length - 1].grid;
  for (let strike = Math.ceil(priceLow); strike <= Math.floor(priceHigh); strike += 1) {
    if (strike % 2 !== 0 && Math.abs(strike - SPOT) > 6) continue;
    const distance = Math.abs(strike - SPOT);
    const callBias = strike > SPOT ? 1.6 : 0.6;
    const putBias = strike < SPOT ? 1.6 : 0.6;
    const base = 90000 * Math.exp(-(distance ** 2) / 40) + 4000 * rand();
    const wallBoost = strike === CALL_WALL || strike === PUT_WALL ? 2.1 : 1;
    strikes.push({
      strike,
      callOi: Math.round(base * callBias * wallBoost),
      putOi: Math.round(base * putBias * wallBoost),
      callVolume: Math.round(base * 0.32 * callBias),
      putVolume: Math.round(base * 0.32 * putBias),
      netGex: round2(gexAt(strike) * 1e3) / 1e3,
      touchProbability: clamp01(2 * tailMass(lastGrid, strike, SPOT)),
      finishAbove: clamp01(1 - cdf(lastGrid, strike)),
      signedPremium: Math.round((rand() - 0.45) * base * 0.04),
    });
  }

  // ---- Simulated paths ---------------------------------------------------
  const simulatedPaths: number[][] = [];
  for (let s = 0; s < 24; s += 1) {
    const pathRand = mulberry32(minuteSeed + s * 7919);
    const path: number[] = [];
    let value = SPOT;
    for (let t = 0; t < TIME_COLUMNS; t += 1) {
      if (minutesAxis[t] <= 0) {
        path.push(NaN);
        continue;
      }
      const stepMinutes = (windowEnd - windowStart) / (TIME_COLUMNS - 1);
      const sigma = atmIv * Math.sqrt(stepMinutes / MINUTES_PER_TRADING_YEAR);
      const pinDrift = (PIN_STRIKE - value) * 0.006;
      value = value * Math.exp(sigma * gauss(pathRand) - 0.5 * sigma * sigma) + pinDrift;
      path.push(round2(value));
    }
    simulatedPaths.push(path);
  }

  const sessionHigh = round2(Math.max(...historicalPrice.map((p) => p.value)) + 0.4);
  const sessionLow = round2(Math.min(...historicalPrice.map((p) => p.value)) - 0.3);
  const expectedMove = round2(SPOT * atmIv * Math.sqrt(horizonMinutes / MINUTES_PER_TRADING_YEAR));

  const chart: ForecastChartPayload = {
    timestamp: now.toISOString(),
    spot: SPOT,
    horizon: opts.horizon,
    expiration: opts.expiration,
    timeAxis,
    priceAxis: priceAxis.map(round2),
    forecastStartIndex: forecastStartIndex < 0 ? TIME_COLUMNS - 1 : forecastStartIndex,
    historicalPrice,
    historicalVwap,
    forecastDensity: density,
    gexSurface,
    medianPath: built.medianPath.map(safeRound),
    modePath: built.modePath.map(safeRound),
    simulatedPaths,
    quantiles: {
      p05: built.bands.p05.map(safeRound),
      p10: built.bands.p10.map(safeRound),
      p25: built.bands.p25.map(safeRound),
      p50: built.bands.p50.map(safeRound),
      p75: built.bands.p75.map(safeRound),
      p90: built.bands.p90.map(safeRound),
      p95: built.bands.p95.map(safeRound),
    },
    levels: {
      gammaFlip: GAMMA_FLIP,
      callWall: CALL_WALL,
      putWall: PUT_WALL,
      volatilityTrigger: VOL_TRIGGER,
      expectedMoveUpper: round2(SPOT + expectedMove),
      expectedMoveLower: round2(SPOT - expectedMove),
      pinStrike: PIN_STRIKE,
      sessionHigh,
      sessionLow,
      vwap,
    },
    vectors,
    contours: [],
    strikes,
    gexProfile: priceAxis.map((p) => ({ price: round2(p), gex: round2(gexAt(p)) })),
    confidence: 0.67,
    modelAgreement: 0.76,
    dataQuality: 0.99,
    markers: [
      { time: clockLabel(nowMinute), label: "NOW", kind: "now" },
      { time: clockLabel(SESSION_CLOSE), label: "CLOSE", kind: "close" },
      ...(opts.expiration === "0DTE"
        ? [{ time: clockLabel(SESSION_CLOSE), label: "EXPIRY", kind: "expiry" as const }]
        : []),
    ],
  };

  const interpretation: InterpretationPayload = {
    primaryState: "BroadRange",
    stateConfidence: 0.67,
    stateProbabilities: [
      { state: "BroadRange", probability: 0.67 },
      { state: "BullGrind", probability: 0.08 },
      { state: "BearGrind", probability: 0.04 },
      { state: "BullBreakout", probability: 0.03 },
      { state: "VolExpansion", probability: 0.03 },
      { state: "BearBreakdown", probability: 0.02 },
      { state: "VolContraction", probability: 0.02 },
      { state: "DirectionalRange", probability: 0.01 },
      { state: "StrongPin", probability: 0.1 },
      { state: "Transition", probability: 0.0 },
      { state: "Unstable", probability: 0.0 },
    ],
    mostLikelyPath: `Reversion toward ${PIN_STRIKE.toFixed(0)}–${(PIN_STRIKE + 2).toFixed(0)}`,
    expectedRangeLow: round2(SPOT - expectedMove * 1.15),
    expectedRangeHigh: round2(SPOT + expectedMove * 1.12),
    highestDensityLow: round2(PIN_STRIKE - 0.5),
    highestDensityHigh: round2(PIN_STRIKE + 1),
    upsideBreakoutProbability: 0.18,
    downsideBreakdownProbability: 0.12,
    pinProbability: 0.34,
    pinStrike: PIN_STRIKE,
    modelAgreement: 0.76,
    structuralVeto: null,
    stateStability: 0.72,
    dominantDrivers: [
      "Positive gamma above flip",
      "Call wall compression at 538",
      "Charm pinning into 532",
      "IV above realized",
    ],
  };

  const strategies = buildStrategies(expectedMove);

  const system: SystemPayload = {
    mode: "paper",
    connected: true,
    killSwitch: false,
    killSwitchReasons: [],
    spot: SPOT,
    change: 1.48,
    changePercent: 0.28,
    vwap,
    ivRank: 42.6,
    atmIv: atmIv * 100,
    realizedVol: 15.3,
    dte: opts.expiration,
    serverTime: clockLabel(nowMinute),
    equity: 100250,
    dailyPnl: 312.5,
    dailyPnlPercent: 0.31,
    openRisk: 0,
    buyingPower: 98740,
    dailyLossLimit: 1000,
    maxLossPerTrade: 314,
    openPositions: 1,
    maxOpenPositions: 1,
    dataQuality: 0.99,
    modelConfidence: 0.67,
    netGex: 1.28,
    gexSlope: -0.28,
    dexTrend: -219,
    vannaExposure: -0.42,
    charmExposure: 0.31,
    expectedMove,
    expectedMovePercent: round2((expectedMove / SPOT) * 100),
    emUtilization: 0.48,
  };

  return { chart, interpretation, strategies, system, source: "synthetic" };
}

// ---------------------------------------------------------------------------
// Strategy candidates
// ---------------------------------------------------------------------------

interface StrategySpec {
  family: string;
  label: string;
  legs: Array<[number, "call" | "put", number]>;
  /** Dollars per spread; positive = credit received. */
  net: number;
  pop: number;
  ev: number;
  ror: number;
  utility: number;
  fill: number;
  assignment: "low" | "medium" | "high";
}

/**
 * Candidate specs carry only what cannot be derived: the legs, the net price,
 * and the model's own scores. Max profit, max loss, and breakevens are computed
 * from the payoff curve so the table can never contradict the overlay drawn
 * from the same legs.
 */
const STRATEGY_SPECS: StrategySpec[] = [
  {
    family: "IronCondor",
    label: "Iron Condor",
    legs: [[520, "put", 1], [525, "put", -1], [538, "call", -1], [543, "call", 1]],
    net: 186,
    pop: 0.64,
    ev: 58,
    ror: 0.18,
    utility: 0.72,
    fill: 0.91,
    assignment: "low",
  },
  {
    family: "BullPutCreditSpread",
    label: "Put Credit Spread",
    legs: [[522, "put", 1], [524, "put", -1]],
    net: 92,
    pop: 0.78,
    ev: 41,
    ror: 0.38,
    utility: 0.68,
    fill: 0.94,
    assignment: "low",
  },
  {
    family: "BrokenWingPutButterfly",
    label: "Broken Wing Butterfly",
    legs: [[522, "put", 1], [530, "put", -2], [534, "put", 1]],
    net: 164,
    pop: 0.58,
    ev: 45,
    ror: 0.13,
    utility: 0.61,
    fill: 0.86,
    assignment: "low",
  },
  {
    family: "IronButterfly",
    label: "Iron Butterfly",
    legs: [[527, "put", 1], [532, "put", -1], [532, "call", -1], [537, "call", 1]],
    net: 142,
    pop: 0.62,
    ev: 32,
    ror: 0.09,
    utility: 0.54,
    fill: 0.88,
    assignment: "medium",
  },
  {
    family: "BearCallCreditSpread",
    label: "Call Credit Spread",
    legs: [[538, "call", -1], [540, "call", 1]],
    net: 88,
    pop: 0.74,
    ev: 29,
    ror: 0.26,
    utility: 0.48,
    fill: 0.92,
    assignment: "low",
  },
  {
    family: "LongStrangle",
    label: "Long Strangle",
    legs: [[527, "put", 1], [536, "call", 1]],
    net: -312,
    pop: 0.36,
    ev: 14,
    ror: 0.04,
    utility: 0.22,
    fill: 0.83,
    assignment: "low",
  },
  {
    family: "BullCallDebitSpread",
    label: "Bull Call Spread",
    legs: [[531, "call", 1], [534, "call", -1]],
    net: -210,
    pop: 0.55,
    ev: 11,
    ror: 0.04,
    utility: 0.19,
    fill: 0.9,
    assignment: "low",
  },
];

function buildStrategies(_expectedMove: number): StrategyCandidate[] {
  return STRATEGY_SPECS.map((spec, index) => {
    const base: StrategyCandidate = {
      rank: index + 1,
      strategyId: `${spec.family}-${index + 1}`,
      family: spec.family,
      label: spec.label,
      expiration: "0DTE",
      legs: spec.legs.map(([strike, right, quantity]) => ({
        strike,
        right,
        quantity,
        expiry: "0DTE",
        entryPrice: round2(Math.abs(spec.net) / 100 / spec.legs.length),
      })),
      netPrice: Math.abs(spec.net),
      isCredit: spec.net > 0,
      maxProfit: null,
      maxLoss: 0,
      breakevens: [],
      probabilityOfProfit: spec.pop,
      expectedValue: spec.ev,
      expectedReturnOnRisk: spec.ror,
      utility: spec.utility,
      fillProbability: spec.fill,
      assignmentRisk: spec.assignment,
      rejectionReason: null,
      profitTarget: 0.5,
      stopLevel: 2.0,
      maxHoldingMinutes: 240,
    };

    const geo = strategyGeometry(base);
    return {
      ...base,
      // A long strangle has unbounded upside; report it as such rather than
      // as the largest value the sampled window happened to reach.
      maxProfit: spec.family === "LongStrangle" ? null : geo.computedMaxProfit,
      maxLoss: geo.computedMaxLoss,
      breakevens: geo.breakevens,
    };
  });
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function round2(v: number): number {
  return Math.round(v * 100) / 100;
}

function safeRound(v: number): number {
  return Number.isFinite(v) ? round2(v) : NaN;
}

function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}

function cdf(grid: Array<[number, number]>, price: number): number {
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
  return 1;
}

function tailMass(grid: Array<[number, number]>, level: number, spot: number): number {
  const c = cdf(grid, level);
  return level >= spot ? 1 - c : c;
}
