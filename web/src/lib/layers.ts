import type { LayerGroup, LayerId, LayerMeta } from "./types";

/**
 * Layer catalogue.
 *
 * `order` is assigned from this array's position and defines the canvas z-order:
 * fields paint first, then contours, then arrows, then lines, then markers.
 * Reordering here reorders the draw stack, so keep fields at the top.
 */
export const LAYER_CATALOGUE: LayerMeta[] = [
  // ---- Fields (painted first, lowest z) --------------------------------
  {
    id: "forecast-density",
    label: "Forecast Probability Field",
    group: "Price & Forecast",
    description:
      "Reconstructed probability of SPY trading at each price through time. Warm = high density. Derived from the hybrid model's quantile grid at every horizon.",
    encoding: { kind: "field", color: "#22D3EE" },
    heavy: true,
    defaultOpacity: 0.85,
  },
  {
    id: "gex-heatmap",
    label: "GEX Heatmap",
    group: "Gamma & Dealer Positioning",
    description:
      "Dealer gamma pressure by price. Blue = negative GEX (amplifies moves). Red = positive GEX (dampens / pins). This is the field that moves price.",
    encoding: { kind: "field", color: "#3B82F6" },
    heavy: true,
    defaultOpacity: 0.95,
  },
  {
    id: "iv-surface",
    label: "IV Surface",
    group: "Volatility",
    description:
      "Implied volatility across strike and tenor, projected onto the price axis.",
    encoding: { kind: "field", color: "#E879F9" },
    heavy: true,
    defaultOpacity: 0.5,
  },
  {
    id: "model-disagreement",
    label: "Model Disagreement",
    group: "Model Diagnostics",
    description:
      "Where the ensemble members diverge. Desaturated regions mean the forecast there rests on conflicting models.",
    encoding: { kind: "field", color: "#5A6E8C" },
    defaultOpacity: 0.45,
  },
  {
    id: "structural-veto",
    label: "Structural Veto",
    group: "Model Diagnostics",
    description:
      "Hatched zones the structural layer has vetoed. No strategy will be proposed with its profit region inside a veto.",
    encoding: { kind: "hatch", color: "#F87171" },
    defaultOpacity: 0.6,
  },

  // ---- Contours & profiles ---------------------------------------------
  {
    id: "oi-contours",
    label: "OI Contours",
    group: "Open Interest & Volume",
    description: "Iso-lines of total open interest across strikes.",
    encoding: { kind: "contour", color: "#94A6C0" },
    defaultOpacity: 0.7,
  },
  {
    id: "call-oi",
    label: "Call Open Interest",
    group: "Open Interest & Volume",
    description:
      "Call open interest by strike, drawn as a right-margin profile.",
    encoding: { kind: "contour", color: "#34D399" },
    defaultOpacity: 0.8,
  },
  {
    id: "put-oi",
    label: "Put Open Interest",
    group: "Open Interest & Volume",
    description:
      "Put open interest by strike, drawn as a right-margin profile.",
    encoding: { kind: "contour", color: "#F87171" },
    defaultOpacity: 0.8,
  },
  {
    id: "volume-profile",
    label: "Volume Profile",
    group: "Open Interest & Volume",
    description: "Traded contract volume by price level for the session.",
    encoding: { kind: "contour", color: "#64748B" },
    defaultOpacity: 0.7,
  },
  {
    id: "gex-profile",
    label: "GEX Profile",
    group: "Gamma & Dealer Positioning",
    description:
      "Net gamma exposure as a function of spot, drawn in the right margin.",
    encoding: { kind: "contour", color: "#FBBF24" },
    defaultOpacity: 0.9,
  },

  // ---- Vectors ----------------------------------------------------------
  {
    id: "gex-arrows",
    label: "Gamma Flow Arrows",
    group: "Gamma & Dealer Positioning",
    description:
      "Pressure gradients that move price. Arrows point the way dealer hedging pushes spot — toward the flip in +GEX, away from it in −GEX.",
    encoding: { kind: "arrows", color: "#0F172A" },
    defaultOpacity: 0.9,
  },

  // ---- Bands ------------------------------------------------------------
  {
    id: "forecast-quantiles",
    label: "Quantile Bands",
    group: "Price & Forecast",
    description:
      "5/10/25 – 75/90/95 percentile envelopes, progressively more transparent.",
    encoding: { kind: "band", color: "#22D3EE" },
    defaultOpacity: 0.55,
  },
  {
    id: "expected-move",
    label: "Expected Move",
    group: "Volatility",
    description: "Option-implied one-sigma move for the selected expiration.",
    encoding: { kind: "band", color: "#E879F9", dash: true },
    defaultOpacity: 0.8,
  },
  {
    id: "session-range",
    label: "Session / Overnight Range",
    group: "Price & Forecast",
    description: "Session high-low and the overnight range that preceded it.",
    encoding: { kind: "band", color: "#5A6E8C", dash: true },
    defaultOpacity: 0.6,
  },
  {
    id: "strategy-payoff",
    label: "Strategy Payoff",
    group: "Strategy",
    description:
      "Payoff geometry of the selected strategy: profit region, loss region, breakevens, and short/long strikes.",
    encoding: { kind: "band", color: "#34D399" },
    defaultOpacity: 0.7,
  },

  // ---- Paths ------------------------------------------------------------
  {
    id: "simulated-paths",
    label: "Simulated Paths",
    group: "Price & Forecast",
    description:
      "A thin sample of Monte Carlo paths drawn from the forecast distribution.",
    encoding: { kind: "line", color: "#22D3EE" },
    heavy: true,
    defaultOpacity: 0.35,
  },
  {
    id: "forecast-median",
    label: "Median Path",
    group: "Price & Forecast",
    description:
      "The 50th percentile of the forecast distribution through time.",
    encoding: { kind: "line", color: "#22D3EE" },
    defaultOpacity: 1,
  },
  {
    id: "forecast-mode",
    label: "Mode Path",
    group: "Price & Forecast",
    description:
      "The highest-density price at each forecast step — often pinned to a strike.",
    encoding: { kind: "line", color: "#A5F3FC", dash: true },
    defaultOpacity: 0.9,
  },
  {
    id: "vwap",
    label: "VWAP",
    group: "Price & Forecast",
    description: "Session volume-weighted average price.",
    encoding: { kind: "line", color: "#FBBF24" },
    defaultOpacity: 1,
  },
  {
    id: "spy-price",
    label: "SPY Price",
    group: "Price & Forecast",
    description: "Realized SPY price for the visible session window.",
    encoding: { kind: "line", color: "#E6EDF7" },
    defaultOpacity: 1,
  },

  // ---- Greeks -----------------------------------------------------------
  {
    id: "dex",
    label: "Delta Exposure",
    group: "Greeks",
    description: "Aggregate dealer delta exposure by strike.",
    encoding: { kind: "contour", color: "#38BDF8" },
    defaultOpacity: 0.7,
  },
  {
    id: "vanna",
    label: "Vanna Exposure",
    group: "Greeks",
    description:
      "Sensitivity of delta to implied volatility — drives flows when IV moves.",
    encoding: { kind: "contour", color: "#E879F9" },
    defaultOpacity: 0.7,
  },
  {
    id: "charm",
    label: "Charm Exposure",
    group: "Greeks",
    description:
      "Delta decay through time — the source of pinning pressure into expiration.",
    encoding: { kind: "contour", color: "#FB923C" },
    defaultOpacity: 0.7,
  },
  {
    id: "skew",
    label: "Skew",
    group: "Volatility",
    description:
      "Put-versus-call implied volatility skew and the resulting risk reversal.",
    encoding: { kind: "contour", color: "#C084FC" },
    defaultOpacity: 0.7,
  },

  // ---- Levels (highest z) -----------------------------------------------
  {
    id: "gamma-flip",
    label: "Gamma Flip",
    group: "Market Structure",
    description:
      "Price where aggregate dealer gamma changes sign. Above it dealers dampen moves; below it they amplify.",
    encoding: { kind: "marker", color: "#67E8F9", dash: true },
    defaultOpacity: 1,
  },
  {
    id: "call-wall",
    label: "Call Wall",
    group: "Market Structure",
    description:
      "Strike with the largest call gamma concentration — typically resistance.",
    encoding: { kind: "marker", color: "#34D399", dash: true },
    defaultOpacity: 1,
  },
  {
    id: "put-wall",
    label: "Put Wall",
    group: "Market Structure",
    description:
      "Strike with the largest put gamma concentration — typically support.",
    encoding: { kind: "marker", color: "#F87171", dash: true },
    defaultOpacity: 1,
  },
  {
    id: "vol-trigger",
    label: "Volatility Trigger",
    group: "Market Structure",
    description: "Level below which realized volatility historically expands.",
    encoding: { kind: "marker", color: "#FB923C", dash: true },
    defaultOpacity: 1,
  },
];

export const LAYER_BY_ID: Record<LayerId, LayerMeta> = Object.fromEntries(
  LAYER_CATALOGUE.map((l) => [l.id, l]),
) as Record<LayerId, LayerMeta>;

export const LAYER_GROUPS: LayerGroup[] = [
  "Price & Forecast",
  "Gamma & Dealer Positioning",
  "Open Interest & Volume",
  "Greeks",
  "Volatility",
  "Market Structure",
  "Strategy",
  "Model Diagnostics",
];

/**
 * First paint = pressure desk from the reference:
 * GEX field + flow arrows + walls/flip + price/VWAP + right-edge profile.
 * Forecast plume is opt-in (Full Model / Volatility presets).
 */
export const DEFAULT_ACTIVE: LayerId[] = [
  "gex-heatmap",
  "gex-arrows",
  "gex-profile",
  "gamma-flip",
  "call-wall",
  "put-wall",
  "spy-price",
  "vwap",
];

// ---------------------------------------------------------------------------
// Presets
// ---------------------------------------------------------------------------

export interface Preset {
  id: string;
  label: string;
  hint: string;
  layers: LayerId[];
}

export const PRESETS: Preset[] = [
  {
    id: "gamma-map",
    label: "Pressure",
    hint: "GEX gradients that move price — the desk default",
    layers: [
      "gex-heatmap",
      "gex-profile",
      "gamma-flip",
      "gex-arrows",
      "call-wall",
      "put-wall",
      "spy-price",
      "vwap",
    ],
  },
  {
    id: "full-model",
    label: "Full Model",
    hint: "Everything the model currently believes, in one frame",
    layers: [
      "forecast-density",
      "gex-heatmap",
      "gex-arrows",
      "oi-contours",
      "spy-price",
      "vwap",
      "call-wall",
      "put-wall",
      "gamma-flip",
      "forecast-median",
    ],
  },
  {
    id: "open-interest",
    label: "Open Interest",
    hint: "Where the contracts actually sit",
    layers: [
      "call-oi",
      "put-oi",
      "oi-contours",
      "volume-profile",
      "spy-price",
      "call-wall",
      "put-wall",
    ],
  },
  {
    id: "volatility",
    label: "Volatility",
    hint: "IV structure against the forecast",
    layers: [
      "iv-surface",
      "skew",
      "expected-move",
      "forecast-density",
      "spy-price",
    ],
  },
  {
    id: "dealer-flow",
    label: "Dealer Flow",
    hint: "Greeks that move dealers, not price",
    layers: ["dex", "vanna", "charm", "gex-arrows", "spy-price"],
  },
  {
    id: "price-forecast",
    label: "Price Forecast",
    hint: "Distribution first, structure second",
    layers: [
      "forecast-density",
      "forecast-median",
      "forecast-mode",
      "forecast-quantiles",
      "simulated-paths",
      "expected-move",
      "spy-price",
    ],
  },
  {
    id: "pinning",
    label: "Pinning",
    hint: "Charm, concentration, and the magnet strike",
    layers: [
      "gex-heatmap",
      "call-oi",
      "put-oi",
      "charm",
      "forecast-mode",
      "spy-price",
      "strategy-payoff",
    ],
  },
  {
    id: "range",
    label: "Range",
    hint: "Walls, expected range, condor geometry",
    layers: [
      "call-wall",
      "put-wall",
      "forecast-density",
      "expected-move",
      "strategy-payoff",
      "spy-price",
    ],
  },
  {
    id: "breakout",
    label: "Breakout",
    hint: "Negative gamma and the tails",
    layers: [
      "gex-heatmap",
      "vol-trigger",
      "gex-arrows",
      "forecast-quantiles",
      "forecast-density",
      "strategy-payoff",
      "spy-price",
    ],
  },
  {
    id: "clean",
    label: "Clean View",
    hint: "Price, field, median, walls — nothing else",
    layers: [
      "spy-price",
      "forecast-density",
      "forecast-median",
      "call-wall",
      "put-wall",
    ],
  },
];

export const PRESET_BY_ID: Record<string, Preset> = Object.fromEntries(
  PRESETS.map((p) => [p.id, p]),
);
