"use client";

import * as echarts from "echarts/core";
import { CustomChart, LineChart } from "echarts/charts";
import {
  GraphicComponent,
  GridComponent,
  MarkAreaComponent,
  MarkLineComponent,
  TooltipComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import * as React from "react";

import {
  DENSITY_RAMP,
  DISAGREEMENT_RAMP,
  GEX_RAMP,
  IV_RAMP,
} from "@/lib/colormap";
import { renderFieldImage, renderForecastField } from "@/lib/fieldImage";
import { strategyGeometry } from "@/lib/payoff";
import type {
  ForecastChartPayload,
  LayerId,
  StrategyCandidate,
} from "@/lib/types";
import { alpha } from "@/lib/utils";
import { CanvasReadout, type ReadoutState } from "./CanvasReadout";

echarts.use([
  LineChart,
  CustomChart,
  GridComponent,
  GraphicComponent,
  MarkLineComponent,
  MarkAreaComponent,
  TooltipComponent,
  CanvasRenderer,
]);

/**
 * Pixel insets for the plot rect.
 *
 * The field layers are bitmaps positioned in absolute pixels, so the grid
 * geometry has to be known before ECharts lays out. Computing the insets
 * ourselves makes the mapping exact and side-steps a convertToPixel round trip
 * on every frame.
 *
 * The right margin is reserved only when a profile layer is actually on, and
 * never on a narrow viewport — holding it open unconditionally cost a third of
 * the canvas on a phone.
 */
const PROFILE_WIDTH = 108;
/** Keep the GEX histogram visible on tablet; phone uses the Layers tab profile. */
const PROFILE_MIN_VIEWPORT = 520;

const PROFILE_LAYERS: LayerId[] = [
  "gex-profile",
  "call-oi",
  "put-oi",
  "volume-profile",
];

function computeInsets(width: number, showProfile: boolean) {
  const compact = width < 520;
  return {
    left: compact ? 38 : 54,
    right: showProfile ? PROFILE_WIDTH + 20 : compact ? 12 : 20,
    top: 26,
    bottom: compact ? 34 : 40,
  };
}

export interface ForecastCanvasProps {
  payload: ForecastChartPayload;
  activeLayers: LayerId[];
  opacityFor: (id: LayerId) => number;
  selectedStrategy?: StrategyCandidate;
  onSelectStrike?: (strike: number) => void;
  onSelectTime?: (time: string) => void;
  className?: string;
}

export function ForecastCanvas({
  payload,
  activeLayers,
  opacityFor,
  selectedStrategy,
  onSelectStrike,
  onSelectTime,
  className,
}: ForecastCanvasProps) {
  const hostRef = React.useRef<HTMLDivElement | null>(null);
  const chartRef = React.useRef<echarts.ECharts | null>(null);
  const [size, setSize] = React.useState({ width: 0, height: 0 });
  const [readout, setReadout] = React.useState<ReadoutState | null>(null);

  const active = React.useMemo(() => new Set(activeLayers), [activeLayers]);
  const isOn = React.useCallback((id: LayerId) => active.has(id), [active]);

  // ---- init / teardown --------------------------------------------------
  React.useEffect(() => {
    if (!hostRef.current) return;
    const chart = echarts.init(hostRef.current, undefined, {
      renderer: "canvas",
      useDirtyRect: true,
    });
    chartRef.current = chart;

    const observer = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (!rect) return;
      chart.resize({ width: rect.width, height: rect.height });
      setSize({ width: rect.width, height: rect.height });
    });
    observer.observe(hostRef.current);
    setSize({
      width: hostRef.current.clientWidth,
      height: hostRef.current.clientHeight,
    });

    return () => {
      observer.disconnect();
      chart.dispose();
      chartRef.current = null;
    };
  }, []);

  // ---- geometry ---------------------------------------------------------
  const showProfile = React.useMemo(
    () =>
      size.width >= PROFILE_MIN_VIEWPORT &&
      PROFILE_LAYERS.some((id) => active.has(id)),
    [size.width, active],
  );

  const insets = React.useMemo(
    () => computeInsets(size.width, showProfile),
    [size.width, showProfile],
  );

  const geometry = React.useMemo(() => {
    const plotLeft = insets.left;
    const plotTop = insets.top;
    const plotWidth = Math.max(0, size.width - insets.left - insets.right);
    const plotHeight = Math.max(0, size.height - insets.top - insets.bottom);
    const n = payload.timeAxis.length;
    const priceLow = payload.priceAxis[0];
    const priceHigh = payload.priceAxis[payload.priceAxis.length - 1];

    return {
      plotLeft,
      plotTop,
      plotWidth,
      plotHeight,
      priceLow,
      priceHigh,
      xOf: (index: number) =>
        plotLeft + (plotWidth * index) / Math.max(1, n - 1),
      yOf: (price: number) =>
        plotTop +
        plotHeight *
          (1 - (price - priceLow) / Math.max(1e-9, priceHigh - priceLow)),
      indexOfX: (px: number) =>
        Math.round(
          ((px - plotLeft) / Math.max(1, plotWidth)) * Math.max(1, n - 1),
        ),
      priceOfY: (py: number) =>
        priceHigh -
        ((py - plotTop) / Math.max(1, plotHeight)) * (priceHigh - priceLow),
    };
  }, [size, insets, payload.timeAxis.length, payload.priceAxis]);

  // ---- field bitmaps ----------------------------------------------------
  const densityImage = React.useMemo(() => {
    if (!isOn("forecast-density")) return null;
    return renderForecastField(
      payload.forecastDensity,
      payload.forecastStartIndex,
      {
        ramp: DENSITY_RAMP,
        scale: "column",
        columnMix: 0.72,
        opacity: opacityFor("forecast-density"),
      },
    );
  }, [isOn, payload.forecastDensity, payload.forecastStartIndex, opacityFor]);

  const gexImage = React.useMemo(() => {
    if (!isOn("gex-heatmap")) return null;
    return renderFieldImage(payload.gexSurface, {
      ramp: GEX_RAMP,
      scale: "diverging",
      opacity: opacityFor("gex-heatmap"),
    });
  }, [isOn, payload.gexSurface, opacityFor]);

  const ivImage = React.useMemo(() => {
    if (!isOn("iv-surface") || !payload.impliedVolatilitySurface) return null;
    return renderFieldImage(payload.impliedVolatilitySurface, {
      ramp: IV_RAMP,
      scale: "max",
      opacity: opacityFor("iv-surface"),
    });
  }, [isOn, payload.impliedVolatilitySurface, opacityFor]);

  const disagreementImage = React.useMemo(() => {
    if (!isOn("model-disagreement")) return null;
    // Disagreement is highest where the field is broad — use normalized spread
    // of the density column as a stand-in until the engine exposes per-model
    // dispersion on the lattice.
    const field = payload.forecastDensity.map((column) => {
      const peak = Math.max(...column, 1e-9);
      return column.map((v) => (v > 0 ? 1 - v / peak : 0));
    });
    return renderForecastField(field, payload.forecastStartIndex, {
      ramp: DISAGREEMENT_RAMP,
      scale: "unit",
      opacity: opacityFor("model-disagreement") * 0.6,
    });
  }, [isOn, payload.forecastDensity, payload.forecastStartIndex, opacityFor]);

  // ---- option -----------------------------------------------------------
  const option = React.useMemo(
    () =>
      buildOption({
        payload,
        isOn,
        opacityFor,
        geometry,
        size,
        insets,
        showProfile,
        densityImage,
        gexImage,
        ivImage,
        disagreementImage,
        selectedStrategy,
      }),
    [
      payload,
      isOn,
      opacityFor,
      geometry,
      size,
      insets,
      showProfile,
      densityImage,
      gexImage,
      ivImage,
      disagreementImage,
      selectedStrategy,
    ],
  );

  React.useEffect(() => {
    const chart = chartRef.current;
    if (!chart || size.width === 0) return;
    chart.setOption(option, { notMerge: true, lazyUpdate: false });
  }, [option, size.width]);

  // ---- hover readout ----------------------------------------------------
  React.useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    const zr = chart.getZr();

    const onMove = (event: { offsetX: number; offsetY: number }) => {
      const { offsetX, offsetY } = event;
      const inPlot =
        offsetX >= geometry.plotLeft &&
        offsetX <= geometry.plotLeft + geometry.plotWidth &&
        offsetY >= geometry.plotTop &&
        offsetY <= geometry.plotTop + geometry.plotHeight;

      if (!inPlot) {
        setReadout(null);
        return;
      }

      const timeIndex = Math.min(
        payload.timeAxis.length - 1,
        Math.max(0, geometry.indexOfX(offsetX)),
      );
      const price = geometry.priceOfY(offsetY);
      const priceIndex = nearestIndex(payload.priceAxis, price);

      setReadout({
        x: offsetX,
        y: offsetY,
        timeIndex,
        priceIndex,
        time: payload.timeAxis[timeIndex],
        price,
        isForecast: timeIndex >= payload.forecastStartIndex,
      });
    };

    const onLeave = () => setReadout(null);

    const onClick = (event: { offsetX: number; offsetY: number }) => {
      const price = geometry.priceOfY(event.offsetY);
      const timeIndex = geometry.indexOfX(event.offsetX);
      onSelectStrike?.(Math.round(price));
      const label =
        payload.timeAxis[
          Math.min(payload.timeAxis.length - 1, Math.max(0, timeIndex))
        ];
      if (label) onSelectTime?.(label);
    };

    zr.on("mousemove", onMove);
    zr.on("globalout", onLeave);
    zr.on("click", onClick);
    return () => {
      zr.off("mousemove", onMove);
      zr.off("globalout", onLeave);
      zr.off("click", onClick);
    };
  }, [geometry, payload, onSelectStrike, onSelectTime]);

  return (
    <div className={className}>
      <div ref={hostRef} className="h-full w-full" />
      {readout ? (
        <CanvasReadout
          state={readout}
          payload={payload}
          activeLayers={activeLayers}
          containerWidth={size.width}
        />
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Option builder
// ---------------------------------------------------------------------------

interface BuildArgs {
  payload: ForecastChartPayload;
  isOn: (id: LayerId) => boolean;
  opacityFor: (id: LayerId) => number;
  geometry: ReturnType<typeof Object> & {
    plotLeft: number;
    plotTop: number;
    plotWidth: number;
    plotHeight: number;
    xOf: (index: number) => number;
    yOf: (price: number) => number;
  };
  size: { width: number; height: number };
  insets: { left: number; right: number; top: number; bottom: number };
  showProfile: boolean;
  densityImage: { dataUrl: string } | null;
  gexImage: { dataUrl: string } | null;
  ivImage: { dataUrl: string } | null;
  disagreementImage: { dataUrl: string } | null;
  selectedStrategy?: StrategyCandidate;
}

function buildOption(args: BuildArgs): echarts.EChartsCoreOption {
  const {
    payload,
    isOn,
    opacityFor,
    geometry,
    insets,
    showProfile,
    selectedStrategy,
  } = args;
  const n = payload.timeAxis.length;
  const compact = geometry.plotWidth < 460;
  const priceLow = payload.priceAxis[0];
  const priceHigh = payload.priceAxis[payload.priceAxis.length - 1];

  const graphics: Record<string, unknown>[] = [];

  const pushField = (image: { dataUrl: string } | null, fromIndex: number) => {
    if (!image || geometry.plotWidth <= 0) return;
    const x = geometry.xOf(fromIndex);
    graphics.push({
      type: "image",
      z: -20,
      silent: true,
      style: {
        image: image.dataUrl,
        x,
        y: geometry.plotTop,
        width: geometry.plotLeft + geometry.plotWidth - x,
        height: geometry.plotHeight,
      },
    });
  };

  // Draw order: GEX beneath, then density, then diagnostics.
  pushField(args.gexImage, 0);
  pushField(args.ivImage, 0);
  pushField(args.densityImage, payload.forecastStartIndex);
  pushField(args.disagreementImage, payload.forecastStartIndex);

  // Realized window tint — keep it light so the pressure field stays continuous.
  if (geometry.plotWidth > 0 && args.densityImage) {
    graphics.push({
      type: "rect",
      z: -30,
      silent: true,
      shape: {
        x: geometry.plotLeft,
        y: geometry.plotTop,
        width: Math.max(
          0,
          geometry.xOf(payload.forecastStartIndex) - geometry.plotLeft,
        ),
        height: geometry.plotHeight,
      },
      style: { fill: "rgba(3, 7, 14, 0.35)" },
    });
  }

  const series: Record<string, unknown>[] = [];

  // ---- Quantile bands ---------------------------------------------------
  if (isOn("forecast-quantiles")) {
    const o = opacityFor("forecast-quantiles");
    const pairs: Array<
      [keyof typeof payload.quantiles, keyof typeof payload.quantiles, number]
    > = [
      ["p05", "p95", 0.1],
      ["p10", "p90", 0.14],
      ["p25", "p75", 0.2],
    ];
    for (const [lower, upper, strength] of pairs) {
      series.push(
        bandSeries(`${lower}-base`, payload.quantiles[lower], n),
        bandSeries(
          `${lower}-fill`,
          payload.quantiles[upper].map((v, i) =>
            Number.isFinite(v) && Number.isFinite(payload.quantiles[lower][i])
              ? v - payload.quantiles[lower][i]
              : NaN,
          ),
          n,
          alpha("#22D3EE", strength * o),
        ),
      );
    }
  }

  // ---- Simulated paths --------------------------------------------------
  if (isOn("simulated-paths")) {
    const o = opacityFor("simulated-paths");
    payload.simulatedPaths.slice(0, 24).forEach((path, i) => {
      series.push({
        type: "line",
        name: `sim-${i}`,
        data: path,
        showSymbol: false,
        silent: true,
        z: 3,
        lineStyle: { width: 0.7, color: alpha("#22D3EE", 0.28 * o) },
        animation: false,
      });
    });
  }

  // ---- Strategy payoff --------------------------------------------------
  if (isOn("strategy-payoff") && selectedStrategy) {
    const geo = strategyGeometry(selectedStrategy);
    const o = opacityFor("strategy-payoff");
    const markAreas = [
      ...geo.profitZones.map((zone) => [
        { yAxis: zone[0], itemStyle: { color: alpha("#34D399", 0.16 * o) } },
        { yAxis: zone[1] },
      ]),
      ...geo.lossZones.map((zone) => [
        { yAxis: zone[0], itemStyle: { color: alpha("#F87171", 0.14 * o) } },
        { yAxis: zone[1] },
      ]),
    ];

    series.push({
      type: "line",
      name: "payoff-zones",
      data: [],
      silent: true,
      z: 2,
      markArea: { silent: true, data: markAreas },
      markLine: {
        silent: true,
        symbol: "none",
        data: [
          ...geo.breakevens.map((be) => ({
            yAxis: be,
            lineStyle: {
              color: alpha("#FBBF24", o),
              width: 1,
              type: "dashed" as const,
            },
            label: {
              formatter: `BE ${be.toFixed(2)}`,
              position: "insideEndTop" as const,
              color: "#FBBF24",
              fontSize: 9,
              fontFamily: "var(--font-mono)",
            },
          })),
          ...geo.shortStrikes.map((strike) => ({
            yAxis: strike,
            lineStyle: {
              color: alpha("#F87171", 0.9 * o),
              width: 1.5,
              type: "solid" as const,
            },
            label: {
              formatter: `SHORT ${strike}`,
              position: "insideStartTop" as const,
              color: "#F87171",
              fontSize: 9,
            },
          })),
          ...geo.longStrikes.map((strike) => ({
            yAxis: strike,
            lineStyle: {
              color: alpha("#34D399", 0.9 * o),
              width: 1.5,
              type: "solid" as const,
            },
            label: {
              formatter: `LONG ${strike}`,
              position: "insideStartTop" as const,
              color: "#34D399",
              fontSize: 9,
            },
          })),
        ],
      },
    });
  }

  // ---- Expected move / session range ------------------------------------
  const levelLines: Record<string, unknown>[] = [];
  const pushLevel = (
    id: LayerId,
    value: number | undefined,
    color: string,
    label: string,
    dashed = true,
  ) => {
    if (!isOn(id) || value === undefined || !Number.isFinite(value)) return;
    levelLines.push({
      yAxis: value,
      lineStyle: {
        color: alpha(color, opacityFor(id)),
        width: 1.2,
        type: dashed ? ("dashed" as const) : ("solid" as const),
      },
      label: {
        formatter: `${label} ${value.toFixed(2)}`,
        position: "insideEndTop" as const,
        color,
        fontSize: 10,
        fontWeight: "bold" as const,
        fontFamily: "var(--font-mono)",
        backgroundColor: "rgba(5,9,15,0.75)",
        padding: [2, 4, 2, 4],
        borderRadius: 3,
      },
    });
  };

  pushLevel("call-wall", payload.levels.callWall, "#34D399", "CALL WALL");
  pushLevel("put-wall", payload.levels.putWall, "#F87171", "PUT WALL");
  pushLevel("gamma-flip", payload.levels.gammaFlip, "#67E8F9", "GEX FLIP");
  pushLevel(
    "vol-trigger",
    payload.levels.volatilityTrigger,
    "#FB923C",
    "VOL TRIG",
  );
  pushLevel(
    "expected-move",
    payload.levels.expectedMoveUpper,
    "#E879F9",
    "EM+",
  );
  pushLevel(
    "expected-move",
    payload.levels.expectedMoveLower,
    "#E879F9",
    "EM−",
  );
  pushLevel("session-range", payload.levels.sessionHigh, "#5A6E8C", "HIGH");
  pushLevel("session-range", payload.levels.sessionLow, "#5A6E8C", "LOW");

  if (levelLines.length > 0) {
    series.push({
      type: "line",
      name: "levels",
      data: [],
      silent: true,
      z: 12,
      markLine: { silent: true, symbol: "none", data: levelLines },
    });
  }

  // ---- Flow arrows — pressure gradients that move price -----------------
  if (isOn("gex-arrows")) {
    const o = opacityFor("gex-arrows");
    // Thin against pixel spacing so the field stays a vector lattice, not hatch.
    const times = [...new Set(payload.vectors.map((v) => v.timeIndex))].sort(
      (a, b) => a - b,
    );
    const prices = [...new Set(payload.vectors.map((v) => v.priceIndex))].sort(
      (a, b) => a - b,
    );
    const xSpacing = geometry.plotWidth / Math.max(1, times.length);
    const ySpacing = geometry.plotHeight / Math.max(1, prices.length);
    const kx = Math.max(1, Math.ceil(40 / Math.max(1, xSpacing)));
    const ky = Math.max(1, Math.ceil(32 / Math.max(1, ySpacing)));
    const keepTimes = new Set(times.filter((_, i) => i % kx === 0));
    const keepPrices = new Set(prices.filter((_, i) => i % ky === 0));

    series.push({
      type: "custom",
      name: "gamma-arrows",
      silent: true,
      z: 6,
      data: payload.vectors
        .filter(
          (v) =>
            keepTimes.has(v.timeIndex) &&
            keepPrices.has(v.priceIndex) &&
            v.strength > 0.08,
        )
        .map((v) => [
          v.timeIndex,
          payload.priceAxis[v.priceIndex],
          v.dx,
          v.dy,
          v.strength,
        ]),
      renderItem: (_params: unknown, api: ArrowApi) => {
        const x = api.value(0);
        const price = api.value(1);
        const dx = api.value(2);
        const dy = api.value(3);
        const strength = api.value(4);
        const point = api.coord([x, price]);

        // Primary motion is vertical (price). Mild +x lean keeps the CFD look.
        const len = 7 + strength * 11;
        const mag = Math.hypot(dx * 0.35, dy) || 1;
        const ux = (dx * 0.35) / mag;
        const uy = -dy / mag; // screen y grows downward
        const x1 = point[0] - ux * len * 0.45;
        const y1 = point[1] - uy * len * 0.45;
        const x2 = point[0] + ux * len * 0.55;
        const y2 = point[1] + uy * len * 0.55;
        const opacity = 0.45 + strength * 0.5;
        // Dark ink on warm cells, deep teal on cool cells — readable on both.
        const color = dy >= 0 ? "#042F2E" : "#020617";
        const head = 3.4 + strength * 1.6;
        const px = -uy;
        const py = ux;

        return {
          type: "group",
          silent: true,
          children: [
            {
              type: "line",
              shape: { x1, y1, x2, y2 },
              style: {
                stroke: alpha(color, opacity * o),
                lineWidth: 1 + strength * 0.6,
              },
            },
            {
              type: "polygon",
              shape: {
                points: [
                  [x2, y2],
                  [x2 - ux * head + px * head * 0.55, y2 - uy * head + py * head * 0.55],
                  [x2 - ux * head - px * head * 0.55, y2 - uy * head - py * head * 0.55],
                ],
              },
              style: { fill: alpha(color, opacity * o) },
            },
          ],
        };
      },
    });
  }

  // ---- Paths ------------------------------------------------------------
  if (isOn("forecast-quantiles") || isOn("forecast-median")) {
    // p50 already carried by quantiles; median drawn separately for weight.
  }

  if (isOn("forecast-mode")) {
    series.push({
      type: "line",
      name: "Mode Path",
      data: payload.modePath,
      showSymbol: false,
      silent: true,
      z: 9,
      lineStyle: {
        width: 1.4,
        type: "dashed",
        color: alpha("#A5F3FC", opacityFor("forecast-mode")),
      },
      animation: false,
    });
  }

  if (isOn("forecast-median")) {
    series.push({
      type: "line",
      name: "Median Path",
      data: payload.medianPath,
      showSymbol: false,
      silent: true,
      z: 10,
      lineStyle: {
        width: 2,
        color: alpha("#22D3EE", opacityFor("forecast-median")),
        shadowBlur: 8,
        shadowColor: "rgba(34,211,238,0.5)",
      },
      animation: false,
    });
  }

  if (isOn("vwap")) {
    series.push({
      type: "line",
      name: "VWAP",
      data: alignToAxis(payload.historicalVwap, payload.timeAxis),
      showSymbol: false,
      silent: true,
      z: 8,
      lineStyle: {
        width: 1.8,
        color: alpha("#FACC15", opacityFor("vwap")),
      },
      animation: false,
    });
  }

  if (isOn("spy-price")) {
    series.push({
      type: "line",
      name: "SPY Price",
      data: alignToAxis(payload.historicalPrice, payload.timeAxis),
      showSymbol: false,
      silent: true,
      z: 11,
      lineStyle: {
        width: 2.2,
        color: alpha("#0A0A0A", opacityFor("spy-price")),
        shadowBlur: 0,
      },
      animation: false,
    });
    // Hairline highlight so the path stays readable on dark red/blue cells.
    series.push({
      type: "line",
      name: "SPY Price Edge",
      data: alignToAxis(payload.historicalPrice, payload.timeAxis),
      showSymbol: false,
      silent: true,
      z: 11.5,
      lineStyle: {
        width: 1.1,
        color: alpha("#F8FAFC", 0.92 * opacityFor("spy-price")),
      },
      animation: false,
    });
  }

  // ---- Right-margin profile --------------------------------------------
  const profileSeries: Record<string, unknown>[] = [];
  if (isOn("gex-profile")) {
    const o = opacityFor("gex-profile");
    const peak = Math.max(
      ...payload.gexProfile.map((g) => Math.abs(g.gex)),
      1e-6,
    );
    // Diverging histogram on the right edge (blue − / red +).
    profileSeries.push({
      type: "custom",
      name: "GEX Profile",
      xAxisIndex: 1,
      yAxisIndex: 1,
      silent: true,
      data: payload.gexProfile.map((g) => [g.gex, g.price]),
      renderItem: (_params: unknown, api: ArrowApi) => {
        const gex = api.value(0);
        const price = api.value(1);
        const start = api.coord([0, price]);
        const end = api.coord([gex, price]);
        const half = Math.max(
          1.1,
          (geometry.plotHeight / Math.max(payload.gexProfile.length, 1)) * 0.45,
        );
        const x = Math.min(start[0], end[0]);
        const width = Math.max(1.5, Math.abs(end[0] - start[0]));
        return {
          type: "rect",
          silent: true,
          shape: {
            x,
            y: start[1] - half,
            width,
            height: half * 2,
          },
          style: {
            fill: alpha(
              gex >= 0 ? "#EF4444" : "#3B82F6",
              (0.35 + 0.55 * Math.min(1, Math.abs(gex) / peak)) * o,
            ),
          },
        };
      },
      animation: false,
    });
  }
  if (isOn("call-oi")) {
    profileSeries.push(
      oiProfileSeries(payload, "callOi", "#34D399", opacityFor("call-oi")),
    );
  }
  if (isOn("put-oi")) {
    profileSeries.push(
      oiProfileSeries(payload, "putOi", "#F87171", opacityFor("put-oi")),
    );
  }
  if (isOn("volume-profile")) {
    profileSeries.push(
      oiProfileSeries(
        payload,
        "callVolume",
        "#64748B",
        opacityFor("volume-profile"),
      ),
    );
  }

  // Profile series are dropped entirely when the strip is not reserved; drawing
  // them into a zero-width grid produces a smear against the right edge.
  const hasProfile = showProfile && profileSeries.length > 0;

  // ---- Vertical markers -------------------------------------------------
  // A marker outside the visible window is dropped, not clamped. Clamping stacked
  // CLOSE and EXPIRY onto the NOW rule and made three different instants look
  // like one.
  const markerLines = payload.markers
    .map((marker) => ({ marker, index: payload.timeAxis.indexOf(marker.time) }))
    .filter(({ index }) => index >= 0)
    .map(({ marker, index }) => {
      const color =
        marker.kind === "now"
          ? "#E6EDF7"
          : marker.kind === "expiry"
            ? "#E879F9"
            : "#94A6C0";
      return {
        xAxis: index,
        lineStyle: {
          color: alpha(color, 0.75),
          width: 1,
          type: "dashed" as const,
        },
        label: {
          formatter: marker.label,
          position: "insideEndBottom" as const,
          // ECharts rotates labels on vertical mark lines by default.
          rotate: 0,
          color,
          fontSize: 9,
          fontWeight: "bold" as const,
          backgroundColor: "rgba(5,9,15,0.8)",
          padding: [2, 3, 2, 3],
          borderRadius: 2,
        },
      };
    });

  series.push({
    type: "line",
    name: "markers",
    data: [],
    silent: true,
    z: 13,
    markLine: { silent: true, symbol: "none", data: markerLines },
  });

  return {
    animation: false,
    backgroundColor: "transparent",
    grid: [
      {
        left: insets.left,
        right: insets.right,
        top: insets.top,
        bottom: insets.bottom,
        containLabel: false,
      },
      {
        right: 8,
        width: showProfile ? PROFILE_WIDTH : 0,
        top: insets.top,
        bottom: insets.bottom,
        containLabel: false,
      },
    ],
    xAxis: [
      {
        type: "category",
        gridIndex: 0,
        data: payload.timeAxis,
        boundaryGap: false,
        axisLine: { lineStyle: { color: "#1B2740" } },
        axisTick: { show: false },
        splitLine: {
          show: true,
          interval: (index: number) => index % (compact ? 32 : 20) === 0,
          lineStyle: { color: "rgba(27,39,64,0.55)", type: "solid" },
        },
        axisLabel: {
          color: "#5A6E8C",
          fontSize: compact ? 9 : 10,
          fontFamily: "var(--font-mono)",
          interval: (index: number) => index % (compact ? 32 : 20) === 0,
          margin: 10,
        },
      },
      {
        type: "value",
        gridIndex: 1,
        show: false,
        scale: true,
      },
    ],
    yAxis: [
      {
        type: "value",
        gridIndex: 0,
        min: priceLow,
        max: priceHigh,
        interval: niceInterval(priceHigh - priceLow),
        axisLine: { show: false },
        axisTick: { show: false },
        splitLine: { lineStyle: { color: "rgba(27,39,64,0.5)" } },
        axisLabel: {
          color: "#5A6E8C",
          fontSize: compact ? 9 : 10,
          fontFamily: "var(--font-mono)",
          formatter: (value: number) => value.toFixed(0),
          margin: compact ? 5 : 8,
        },
      },
      {
        type: "value",
        gridIndex: 1,
        min: priceLow,
        max: priceHigh,
        show: false,
      },
    ],
    series: [...series, ...(hasProfile ? profileSeries : [])],
    graphic: graphics,
  } as echarts.EChartsCoreOption;
}

// ---------------------------------------------------------------------------
// Series helpers
// ---------------------------------------------------------------------------

interface ArrowApi {
  value: (index: number) => number;
  coord: (data: [number, number]) => [number, number];
}

/** Transparent base for a stacked band fill. */
function bandSeries(name: string, data: number[], _n: number, fill?: string) {
  return {
    type: "line",
    name,
    stack: name.endsWith("-fill")
      ? name.replace("-fill", "")
      : name.replace("-base", ""),
    data,
    showSymbol: false,
    silent: true,
    z: 4,
    lineStyle: { width: 0, opacity: 0 },
    areaStyle: fill
      ? { color: fill, origin: "start" }
      : { opacity: 0, color: "transparent" },
    animation: false,
  };
}

function oiProfileSeries(
  payload: ForecastChartPayload,
  key: "callOi" | "putOi" | "callVolume",
  color: string,
  opacity: number,
) {
  const peak = Math.max(...payload.strikes.map((s) => s[key]), 1);
  return {
    type: "bar",
    name: key,
    xAxisIndex: 1,
    yAxisIndex: 1,
    data: payload.strikes.map((s) => [(s[key] / peak) * 100, s.strike]),
    barWidth: 2,
    silent: true,
    itemStyle: { color: alpha(color, 0.7 * opacity) },
    animation: false,
  };
}

/**
 * Project a sparse {time,value} series onto the dense chart axis.
 *
 * The realized price series is sampled per minute while the canvas axis is a
 * fixed 150-column lattice; without this the line would compress into the left
 * edge instead of aligning with the forecast boundary.
 */
function alignToAxis(
  points: Array<{ time: string; value: number }>,
  timeAxis: string[],
): number[] {
  if (points.length === 0) return timeAxis.map(() => NaN);
  const byTime = new Map(points.map((p) => [p.time, p.value]));
  let last = NaN;
  return timeAxis.map((label) => {
    const direct = byTime.get(label);
    if (direct !== undefined) {
      last = direct;
      return direct;
    }
    // Only carry forward inside the realized window; never into the forecast.
    return byTime.size > 0 && label <= points[points.length - 1].time
      ? last
      : NaN;
  });
}

function nearestIndex(axis: number[], value: number): number {
  let best = 0;
  let bestDistance = Infinity;
  for (let i = 0; i < axis.length; i += 1) {
    const distance = Math.abs(axis[i] - value);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = i;
    }
  }
  return best;
}

function niceInterval(span: number): number {
  const target = span / 8;
  const candidates = [0.5, 1, 2, 2.5, 5, 10, 20, 25, 50];
  return candidates.find((c) => c >= target) ?? 100;
}
