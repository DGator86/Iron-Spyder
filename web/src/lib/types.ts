/**
 * Wire contract between the SPY-DER engine and the radar frontend.
 *
 * The engine (spy_der/app/api.py) does NOT emit a 2-D density surface — it
 * emits a quantile grid at seven discrete horizons. `forecastDensity` and
 * `gexSurface` are therefore reconstructed (see lib/density.ts) rather than
 * transported. Everything else maps to a real endpoint; see lib/adapter.ts for
 * the field-by-field mapping.
 */

// ---------------------------------------------------------------------------
// Layers
// ---------------------------------------------------------------------------

export type LayerId =
  | "spy-price"
  | "vwap"
  | "forecast-density"
  | "forecast-median"
  | "forecast-mode"
  | "forecast-quantiles"
  | "simulated-paths"
  | "gex-heatmap"
  | "gex-arrows"
  | "gex-profile"
  | "gamma-flip"
  | "call-wall"
  | "put-wall"
  | "vol-trigger"
  | "call-oi"
  | "put-oi"
  | "oi-contours"
  | "volume-profile"
  | "dex"
  | "vanna"
  | "charm"
  | "iv-surface"
  | "skew"
  | "expected-move"
  | "session-range"
  | "strategy-payoff"
  | "model-disagreement"
  | "structural-veto";

export type LayerGroup =
  | "Price & Forecast"
  | "Gamma & Dealer Positioning"
  | "Open Interest & Volume"
  | "Greeks"
  | "Volatility"
  | "Market Structure"
  | "Strategy"
  | "Model Diagnostics";

export interface LayerState {
  id: LayerId;
  enabled: boolean;
  opacity: number;
  order: number;
  settings: Record<string, string | number | boolean>;
}

export interface LayerMeta {
  id: LayerId;
  label: string;
  group: LayerGroup;
  /** Shown in the info tooltip — what the layer means, not how it's drawn. */
  description: string;
  /** Legend swatch: the encoding this layer uses on the canvas. */
  encoding: {
    kind: "line" | "band" | "field" | "contour" | "arrows" | "marker" | "hatch";
    color: string;
    dash?: boolean;
  };
  /** Layers that are expensive enough to warn about when stacked. */
  heavy?: boolean;
  defaultOpacity: number;
}

// ---------------------------------------------------------------------------
// View state
// ---------------------------------------------------------------------------

export type Horizon = "5m" | "15m" | "30m" | "60m" | "eod" | "1d" | "expiry";

export interface ChartViewState {
  horizon: Horizon;
  expiration: string;
  selectedStrike?: number;
  selectedTime?: string;
  selectedStrategyId?: string;
  preset: string;
  live: boolean;
  playbackSpeed: number;
}

// ---------------------------------------------------------------------------
// Chart payload
// ---------------------------------------------------------------------------

export interface Quantiles {
  p05: number[];
  p10: number[];
  p25: number[];
  p50: number[];
  p75: number[];
  p90: number[];
  p95: number[];
}

export interface ForecastLevels {
  gammaFlip?: number;
  callWall?: number;
  putWall?: number;
  volatilityTrigger?: number;
  expectedMoveUpper?: number;
  expectedMoveLower?: number;
  pinStrike?: number;
  sessionHigh?: number;
  sessionLow?: number;
  vwap?: number;
}

export interface FlowVector {
  timeIndex: number;
  priceIndex: number;
  dx: number;
  dy: number;
  strength: number;
}

export interface Contour {
  value: number;
  points: Array<[number, number]>;
}

/** Per-strike ladder used by the strike inspector and OI layers. */
export interface StrikeRow {
  strike: number;
  callOi: number;
  putOi: number;
  callVolume: number;
  putVolume: number;
  netGex: number;
  touchProbability: number;
  finishAbove: number;
  signedPremium: number;
}

export interface ForecastChartPayload {
  timestamp: string;
  spot: number;
  horizon: string;
  expiration: string;
  timeAxis: string[];
  priceAxis: number[];
  /** Index into timeAxis where forecast begins; everything left of it is realized. */
  forecastStartIndex: number;
  historicalPrice: Array<{ time: string; value: number }>;
  historicalVwap: Array<{ time: string; value: number }>;
  forecastDensity: number[][];
  gexSurface: number[][];
  openInterestSurface?: number[][];
  impliedVolatilitySurface?: number[][];
  medianPath: number[];
  modePath: number[];
  simulatedPaths: number[][];
  quantiles: Quantiles;
  levels: ForecastLevels;
  vectors: FlowVector[];
  contours: Contour[];
  strikes: StrikeRow[];
  gexProfile: Array<{ price: number; gex: number }>;
  confidence: number;
  modelAgreement: number;
  dataQuality: number;
  /** Vertical rules the canvas must draw: close, expiry, forecast boundary. */
  markers: Array<{ time: string; label: string; kind: "now" | "close" | "expiry" }>;
}

// ---------------------------------------------------------------------------
// Interpretation / model state
// ---------------------------------------------------------------------------

export type MarketState =
  | "StrongPin"
  | "BroadRange"
  | "BullGrind"
  | "BearGrind"
  | "BullBreakout"
  | "BearBreakdown"
  | "VolExpansion"
  | "VolContraction"
  | "DirectionalRange"
  | "Transition"
  | "Unstable";

export interface InterpretationPayload {
  primaryState: MarketState;
  stateConfidence: number;
  stateProbabilities: Array<{ state: MarketState; probability: number }>;
  mostLikelyPath: string;
  expectedRangeLow: number;
  expectedRangeHigh: number;
  highestDensityLow: number;
  highestDensityHigh: number;
  upsideBreakoutProbability: number;
  downsideBreakdownProbability: number;
  pinProbability: number;
  pinStrike: number;
  modelAgreement: number;
  structuralVeto: string | null;
  stateStability: number;
  dominantDrivers: string[];
}

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

export type OptionRight = "call" | "put";

export interface StrategyLeg {
  strike: number;
  right: OptionRight;
  quantity: number;
  expiry: string;
  entryPrice: number;
}

export interface StrategyCandidate {
  rank: number;
  strategyId: string;
  family: string;
  label: string;
  expiration: string;
  legs: StrategyLeg[];
  netPrice: number;
  isCredit: boolean;
  maxProfit: number | null;
  maxLoss: number;
  breakevens: number[];
  probabilityOfProfit: number;
  expectedValue: number;
  expectedReturnOnRisk: number;
  utility: number;
  fillProbability: number;
  assignmentRisk: "low" | "medium" | "high";
  rejectionReason: string | null;
  profitTarget: number;
  stopLevel: number;
  maxHoldingMinutes: number;
}

// ---------------------------------------------------------------------------
// System / health
// ---------------------------------------------------------------------------

export interface SystemPayload {
  mode: string;
  connected: boolean;
  killSwitch: boolean;
  killSwitchReasons: string[];
  spot: number;
  change: number;
  changePercent: number;
  vwap: number;
  ivRank: number;
  atmIv: number;
  realizedVol: number;
  dte: string;
  serverTime: string;
  equity: number;
  dailyPnl: number;
  dailyPnlPercent: number;
  openRisk: number;
  buyingPower: number;
  dailyLossLimit: number;
  maxLossPerTrade: number;
  openPositions: number;
  maxOpenPositions: number;
  dataQuality: number;
  modelConfidence: number;
  netGex: number;
  gexSlope: number;
  dexTrend: number;
  vannaExposure: number;
  charmExposure: number;
  expectedMove: number;
  expectedMovePercent: number;
  emUtilization: number;
}

/** Everything one poll of the BFF returns. */
export interface RadarSnapshot {
  chart: ForecastChartPayload;
  interpretation: InterpretationPayload;
  strategies: StrategyCandidate[];
  system: SystemPayload;
  /** "live" when proxied from the engine, "synthetic" when generated locally. */
  source: "live" | "synthetic";
}
