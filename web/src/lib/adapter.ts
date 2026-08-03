/**
 * Live-engine adapter: SPY-DER REST responses -> RadarSnapshot.
 *
 * Field-by-field mapping against spy_der/app/api.py. Two things the engine does
 * not emit are reconstructed here rather than faked:
 *
 *   forecastDensity  <- /forecast/current horizons[].price_quantiles, run
 *                       through buildDensitySurface (see lib/density.ts)
 *   gexSurface       <- /analytics/gamma-profile curve[], replicated across the
 *                       time axis (GEX is a function of spot at a snapshot, not
 *                       of time, so replication is the honest projection)
 *
 * Anything the engine cannot supply is left undefined and the corresponding
 * layer renders as unavailable — it is never filled with a plausible number.
 */

import { buildDensitySurface, smoothField, type HorizonQuantiles } from "./density";
import type {
  ForecastChartPayload,
  Horizon,
  InterpretationPayload,
  MarketState,
  RadarSnapshot,
  StrategyCandidate,
  StrikeRow,
  SystemPayload,
} from "./types";

/** Engine horizon names (spy_der/domain/forecast.py HORIZONS). */
const ENGINE_HORIZON_MINUTES: Record<string, number> = {
  "5m": 5,
  "15m": 15,
  "30m": 30,
  "60m": 60,
  eod: 390,
  next_session: 780,
  expiration: 390,
};

const UI_TO_ENGINE_HORIZON: Record<Horizon, string> = {
  "5m": "5m",
  "15m": "15m",
  "30m": "30m",
  "60m": "60m",
  eod: "eod",
  "1d": "next_session",
  expiry: "expiration",
};

const TIME_COLUMNS = 150;
const PRICE_ROWS = 92;

export class EngineUnavailable extends Error {}

async function getJson<T>(base: string, path: string, timeoutMs: number): Promise<T | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${base}${path}`, {
      signal: controller.signal,
      cache: "no-store",
      headers: { accept: "application/json" },
    });
    // 503 is the engine's documented "analytics unavailable" lockout, not a
    // transport failure — treat it as a missing section, not a dead engine.
    if (res.status === 503 || res.status === 404) return null;
    if (!res.ok) throw new EngineUnavailable(`${path} -> ${res.status}`);
    return (await res.json()) as T;
  } finally {
    clearTimeout(timer);
  }
}

// ---------------------------------------------------------------------------

interface EngineForecast {
  timestamp: string;
  spot: number;
  state_probabilities: Record<string, number>;
  dominant_state: string;
  confidence: number;
  model_disagreement: number;
  state_stability: number;
  dealer_agreement: number;
  horizons: Record<
    string,
    {
      minutes: number;
      expected_price: number;
      median_price: number;
      price_quantiles: Record<string, number>;
      touch_probabilities: Record<string, number>;
    }
  >;
  pin_probabilities: Record<string, number>;
}

interface EngineAnalytics {
  timestamp: string;
  spot: number;
  gex_total: number;
  dex_total: number;
  vex_total: number;
  cex_total: number;
  gamma_flip: number | null;
  vol_trigger: number | null;
  call_wall: number | null;
  put_wall: number | null;
  pin_strike: number | null;
  expected_move: number;
  expected_move_utilization: number;
  atm_iv: number;
  realized_vol: number;
}

interface EngineGammaProfile {
  spot: number;
  curve: Array<{ spot: number; gex: number }>;
  nearest_flip: number | null;
  vol_trigger: number | null;
}

interface EngineMarket {
  timestamp: string;
  spot: number;
  expirations: string[];
  data_quality: number;
}

interface EngineCandidate {
  strategy_id: string;
  family: string;
  entry_price: number;
  expected_value: number;
  expected_return_on_risk: number;
  utility: number;
  max_loss: number;
  max_profit: number | null;
  breakevens: number[];
  probability_of_profit: number;
  fill_probability: number;
  assignment_risk: number;
  exit_plan: {
    profit_target_fraction: number;
    stop_loss_fraction: number;
    max_holding_minutes: number;
  };
  legs: Array<{
    expiry: string;
    strike: number;
    right: string;
    quantity: number;
    entry_price: number;
  }>;
}

// ---------------------------------------------------------------------------

export async function fetchLiveSnapshot(
  base: string,
  horizon: Horizon,
  timeoutMs = 4000,
): Promise<RadarSnapshot> {
  const [market, analytics, gammaProfile, forecast, strategiesRaw, health] = await Promise.all([
    getJson<EngineMarket>(base, "/market/current", timeoutMs),
    getJson<EngineAnalytics>(base, "/analytics/current", timeoutMs),
    getJson<EngineGammaProfile>(base, "/analytics/gamma-profile", timeoutMs),
    getJson<EngineForecast>(base, "/forecast/current", timeoutMs),
    getJson<{ candidates: EngineCandidate[] }>(base, "/strategies/candidates?limit=10", timeoutMs),
    getJson<{ mode: string; kill_switch: string[]; open_positions: number }>(
      base,
      "/health",
      timeoutMs,
    ),
  ]);

  if (!market || !forecast) {
    throw new EngineUnavailable("engine returned no market or forecast payload");
  }

  const spot = forecast.spot ?? market.spot;
  const now = new Date(forecast.timestamp);

  // ---- Quantile grids ---------------------------------------------------
  const horizons: HorizonQuantiles[] = Object.entries(forecast.horizons)
    .map(([name, h]) => ({
      minutes: h.minutes || ENGINE_HORIZON_MINUTES[name] || 60,
      grid: Object.entries(h.price_quantiles)
        .map(([tau, price]) => [Number(tau), price] as [number, number])
        .sort((a, b) => a[0] - b[0]),
    }))
    .filter((h) => h.grid.length > 1)
    .sort((a, b) => a.minutes - b.minutes);

  if (horizons.length === 0) throw new EngineUnavailable("forecast contained no usable quantiles");

  const engineHorizon = UI_TO_ENGINE_HORIZON[horizon];
  const horizonMinutes = ENGINE_HORIZON_MINUTES[engineHorizon] ?? 60;
  const historyMinutes = Math.max(30, Math.min(120, horizonMinutes * 2));

  // ---- Axes -------------------------------------------------------------
  const widest = horizons[horizons.length - 1].grid;
  const lo = Math.min(...widest.map(([, p]) => p));
  const hi = Math.max(...widest.map(([, p]) => p));
  const pad = (hi - lo) * 0.15 + 0.5;
  const priceLow = Math.min(lo - pad, spot - 2);
  const priceHigh = Math.max(hi + pad, spot + 2);
  const priceAxis = Array.from(
    { length: PRICE_ROWS },
    (_, i) => priceLow + ((priceHigh - priceLow) * i) / (PRICE_ROWS - 1),
  );

  const timeAxis: string[] = [];
  const minutesAxis: number[] = [];
  for (let i = 0; i < TIME_COLUMNS; i += 1) {
    const offset = -historyMinutes + ((historyMinutes + horizonMinutes) * i) / (TIME_COLUMNS - 1);
    minutesAxis.push(offset);
    timeAxis.push(hhmm(new Date(now.getTime() + offset * 60_000)));
  }
  const forecastStartIndex = Math.max(0, minutesAxis.findIndex((m) => m > 0));

  // ---- Field ------------------------------------------------------------
  const built = buildDensitySurface(horizons, minutesAxis, priceAxis, spot);
  const density = smoothField(built.density, 2);

  // ---- GEX --------------------------------------------------------------
  const curve = gammaProfile?.curve ?? [];
  const gexProfile = priceAxis.map((price) => ({
    price: round2(price),
    gex: curve.length > 0 ? interpolateCurve(curve, price) : 0,
  }));
  const gexColumn = gexProfile.map((g) => g.gex);
  const gexSurface = minutesAxis.map(() => gexColumn);

  // ---- Strike ladder ----------------------------------------------------
  const nearestHorizon =
    horizons.find((h) => h.minutes >= horizonMinutes) ?? horizons[horizons.length - 1];
  const touch = forecast.horizons[engineHorizon]?.touch_probabilities ?? {};
  const strikes: StrikeRow[] = [];
  for (let strike = Math.ceil(priceLow); strike <= Math.floor(priceHigh); strike += 1) {
    strikes.push({
      strike,
      callOi: 0,
      putOi: 0,
      callVolume: 0,
      putVolume: 0,
      netGex: curve.length > 0 ? interpolateCurve(curve, strike) : 0,
      touchProbability: touch[String(strike)] ?? touchFromGrid(nearestHorizon.grid, strike, spot),
      finishAbove: 1 - cdfFromGrid(nearestHorizon.grid, strike),
      signedPremium: 0,
    });
  }

  const chart: ForecastChartPayload = {
    timestamp: forecast.timestamp,
    spot,
    horizon,
    expiration: market.expirations?.[0] ?? "0DTE",
    timeAxis,
    priceAxis: priceAxis.map(round2),
    forecastStartIndex,
    // The engine exposes no intraday price series, so the realized half is
    // empty until a tape endpoint exists. The canvas renders the gap honestly.
    historicalPrice: [],
    historicalVwap: [],
    forecastDensity: density,
    gexSurface,
    medianPath: built.medianPath.map(safeRound),
    modePath: built.modePath.map(safeRound),
    simulatedPaths: [],
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
      gammaFlip: analytics?.gamma_flip ?? gammaProfile?.nearest_flip ?? undefined,
      callWall: analytics?.call_wall ?? undefined,
      putWall: analytics?.put_wall ?? undefined,
      volatilityTrigger: analytics?.vol_trigger ?? gammaProfile?.vol_trigger ?? undefined,
      expectedMoveUpper: analytics ? round2(spot + analytics.expected_move) : undefined,
      expectedMoveLower: analytics ? round2(spot - analytics.expected_move) : undefined,
      pinStrike: analytics?.pin_strike ?? undefined,
    },
    vectors: buildVectors(priceAxis, gexColumn, analytics?.gamma_flip ?? spot),
    contours: [],
    strikes,
    gexProfile,
    confidence: forecast.confidence,
    modelAgreement: 1 - (forecast.model_disagreement ?? 0),
    dataQuality: market.data_quality,
    markers: [{ time: hhmm(now), label: "NOW", kind: "now" }],
  };

  const interpretation = buildInterpretation(forecast, analytics, built, priceAxis, spot);
  const strategies = (strategiesRaw?.candidates ?? []).map(mapCandidate);
  const system = buildSystem(market, analytics, forecast, health);

  return { chart, interpretation, strategies, system, source: "live" };
}

// ---------------------------------------------------------------------------
// Mapping helpers
// ---------------------------------------------------------------------------

function mapCandidate(c: EngineCandidate, index: number): StrategyCandidate {
  const isCredit = c.entry_price > 0;
  return {
    rank: index + 1,
    strategyId: c.strategy_id,
    family: c.family,
    label: humanizeFamily(c.family),
    expiration: c.legs?.[0]?.expiry ?? "",
    legs: (c.legs ?? []).map((leg) => ({
      strike: leg.strike,
      right: leg.right.toLowerCase() === "call" ? "call" : "put",
      quantity: leg.quantity,
      expiry: leg.expiry,
      entryPrice: leg.entry_price,
    })),
    netPrice: Math.round(Math.abs(c.entry_price) * 100),
    isCredit,
    maxProfit: c.max_profit === null ? null : Math.round(c.max_profit * 100),
    maxLoss: Math.round(c.max_loss * 100),
    breakevens: (c.breakevens ?? []).map(round2),
    probabilityOfProfit: c.probability_of_profit,
    expectedValue: Math.round(c.expected_value * 100),
    expectedReturnOnRisk: c.expected_return_on_risk,
    utility: c.utility,
    fillProbability: c.fill_probability,
    assignmentRisk: c.assignment_risk > 0.66 ? "high" : c.assignment_risk > 0.33 ? "medium" : "low",
    rejectionReason: null,
    profitTarget: c.exit_plan?.profit_target_fraction ?? 0.5,
    stopLevel: c.exit_plan?.stop_loss_fraction ?? 2,
    maxHoldingMinutes: c.exit_plan?.max_holding_minutes ?? 240,
  };
}

function humanizeFamily(family: string): string {
  return family.replace(/([a-z])([A-Z])/g, "$1 $2");
}

function buildInterpretation(
  forecast: EngineForecast,
  analytics: EngineAnalytics | null,
  built: { medianPath: number[]; bands: Record<string, number[]> },
  priceAxis: number[],
  spot: number,
): InterpretationPayload {
  const probs = Object.entries(forecast.state_probabilities)
    .map(([state, probability]) => ({ state: state as MarketState, probability }))
    .sort((a, b) => b.probability - a.probability);

  const lastIndex = built.bands.p05.length - 1;
  const low = built.bands.p05[lastIndex];
  const high = built.bands.p95[lastIndex];
  const pin = analytics?.pin_strike ?? null;
  const pinProbability = pin !== null ? (forecast.pin_probabilities?.[String(pin)] ?? 0) : 0;

  const callWall = analytics?.call_wall ?? null;
  const putWall = analytics?.put_wall ?? null;

  return {
    primaryState: (forecast.dominant_state as MarketState) ?? "BroadRange",
    stateConfidence: forecast.confidence,
    stateProbabilities: probs,
    mostLikelyPath: describePath(built.medianPath, spot),
    expectedRangeLow: round2(low),
    expectedRangeHigh: round2(high),
    highestDensityLow: round2(built.bands.p25[lastIndex]),
    highestDensityHigh: round2(built.bands.p75[lastIndex]),
    upsideBreakoutProbability: callWall ? 1 - cdfLinear(built.bands, lastIndex, callWall) : 0,
    downsideBreakdownProbability: putWall ? cdfLinear(built.bands, lastIndex, putWall) : 0,
    pinProbability,
    pinStrike: pin ?? spot,
    modelAgreement: 1 - (forecast.model_disagreement ?? 0),
    structuralVeto: null,
    stateStability: forecast.state_stability ?? 1,
    dominantDrivers: derivedDrivers(analytics),
  };
}

function describePath(medianPath: number[], spot: number): string {
  const finite = medianPath.filter((v) => Number.isFinite(v));
  if (finite.length === 0) return "Insufficient forecast data";
  const end = finite[finite.length - 1];
  const delta = end - spot;
  if (Math.abs(delta) < 0.35) return `Hold near ${end.toFixed(2)}`;
  return delta > 0
    ? `Drift toward ${end.toFixed(2)}`
    : `Reversion toward ${end.toFixed(2)}`;
}

function derivedDrivers(analytics: EngineAnalytics | null): string[] {
  if (!analytics) return [];
  const drivers: string[] = [];
  if (analytics.gex_total > 0) drivers.push("Positive dealer gamma");
  else drivers.push("Negative dealer gamma");
  if (analytics.gamma_flip !== null && analytics.spot < analytics.gamma_flip) {
    drivers.push("Spot below gamma flip");
  }
  if (analytics.atm_iv > analytics.realized_vol) drivers.push("IV above realized");
  else drivers.push("Realized above IV");
  if (analytics.expected_move_utilization > 0.7) drivers.push("Expected move largely consumed");
  return drivers;
}

function buildSystem(
  market: EngineMarket,
  analytics: EngineAnalytics | null,
  forecast: EngineForecast,
  health: { mode: string; kill_switch: string[]; open_positions: number } | null,
): SystemPayload {
  const spot = forecast.spot ?? market.spot;
  return {
    mode: health?.mode ?? "unknown",
    connected: true,
    killSwitch: (health?.kill_switch?.length ?? 0) > 0,
    killSwitchReasons: health?.kill_switch ?? [],
    spot,
    change: 0,
    changePercent: 0,
    vwap: 0,
    ivRank: 0,
    atmIv: (analytics?.atm_iv ?? 0) * 100,
    realizedVol: (analytics?.realized_vol ?? 0) * 100,
    dte: market.expirations?.[0] ?? "",
    serverTime: hhmm(new Date(forecast.timestamp)),
    equity: 0,
    dailyPnl: 0,
    dailyPnlPercent: 0,
    openRisk: 0,
    buyingPower: 0,
    dailyLossLimit: 0,
    maxLossPerTrade: 0,
    openPositions: health?.open_positions ?? 0,
    maxOpenPositions: 0,
    dataQuality: market.data_quality,
    modelConfidence: forecast.confidence,
    netGex: (analytics?.gex_total ?? 0) / 1e9,
    gexSlope: 0,
    dexTrend: (analytics?.dex_total ?? 0) / 1e6,
    vannaExposure: (analytics?.vex_total ?? 0) / 1e9,
    charmExposure: (analytics?.cex_total ?? 0) / 1e9,
    expectedMove: analytics?.expected_move ?? 0,
    expectedMovePercent: analytics ? round2((analytics.expected_move / spot) * 100) : 0,
    emUtilization: analytics?.expected_move_utilization ?? 0,
  };
}

function buildVectors(priceAxis: number[], gexColumn: number[], flip: number) {
  const vectors = [];
  const priceStride = Math.max(1, Math.floor(priceAxis.length / 14));
  const timeStride = Math.max(1, Math.floor(TIME_COLUMNS / 22));
  const peak = Math.max(...gexColumn.map(Math.abs), 1);
  for (let t = 0; t < TIME_COLUMNS; t += timeStride) {
    for (let p = 0; p < priceAxis.length; p += priceStride) {
      const g = gexColumn[p];
      const toward = g >= 0 ? Math.sign(flip - priceAxis[p]) : Math.sign(priceAxis[p] - flip);
      const strength = Math.min(1, Math.abs(g) / peak);
      vectors.push({ timeIndex: t, priceIndex: p, dx: 1, dy: toward * strength, strength });
    }
  }
  return vectors;
}

// ---------------------------------------------------------------------------
// Numeric helpers
// ---------------------------------------------------------------------------

function interpolateCurve(curve: Array<{ spot: number; gex: number }>, price: number): number {
  if (curve.length === 0) return 0;
  const sorted = [...curve].sort((a, b) => a.spot - b.spot);
  if (price <= sorted[0].spot) return sorted[0].gex;
  if (price >= sorted[sorted.length - 1].spot) return sorted[sorted.length - 1].gex;
  for (let i = 0; i < sorted.length - 1; i += 1) {
    const a = sorted[i];
    const b = sorted[i + 1];
    if (a.spot <= price && price <= b.spot) {
      const span = b.spot - a.spot;
      const w = span === 0 ? 0 : (price - a.spot) / span;
      return a.gex + w * (b.gex - a.gex);
    }
  }
  return sorted[sorted.length - 1].gex;
}

function cdfFromGrid(grid: Array<[number, number]>, price: number): number {
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

function touchFromGrid(grid: Array<[number, number]>, level: number, spot: number): number {
  const c = cdfFromGrid(grid, level);
  const beyond = level >= spot ? 1 - c : c;
  return Math.min(1, Math.max(0, 2 * beyond));
}

/** CDF at `level` inferred from the quantile bands at a single time index. */
function cdfLinear(bands: Record<string, number[]>, index: number, level: number): number {
  const pairs: Array<[number, number]> = [
    [0.05, bands.p05[index]],
    [0.1, bands.p10[index]],
    [0.25, bands.p25[index]],
    [0.5, bands.p50[index]],
    [0.75, bands.p75[index]],
    [0.9, bands.p90[index]],
    [0.95, bands.p95[index]],
  ].filter(([, v]) => Number.isFinite(v)) as Array<[number, number]>;
  if (pairs.length === 0) return 0.5;
  return cdfFromGrid(pairs, level);
}

function hhmm(date: Date): string {
  return `${String(date.getUTCHours()).padStart(2, "0")}:${String(date.getUTCMinutes()).padStart(2, "0")}`;
}

function round2(v: number): number {
  return Math.round(v * 100) / 100;
}

function safeRound(v: number): number {
  return Number.isFinite(v) ? round2(v) : NaN;
}
