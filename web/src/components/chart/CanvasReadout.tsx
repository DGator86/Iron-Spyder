"use client";

import * as React from "react";
import type { ForecastChartPayload, LayerId } from "@/lib/types";
import { cn, compactUsd, pct, price as fmtPrice } from "@/lib/utils";

export interface ReadoutState {
  x: number;
  y: number;
  timeIndex: number;
  priceIndex: number;
  time: string;
  price: number;
  isForecast: boolean;
}

/**
 * Cursor readout.
 *
 * Reports the value of every *active* layer at the cursor — a layer that is
 * switched off never contributes a line, so the card is a readout of the view
 * the user actually built rather than a fixed metric dump.
 */
export function CanvasReadout({
  state,
  payload,
  activeLayers,
  containerWidth,
}: {
  state: ReadoutState;
  payload: ForecastChartPayload;
  activeLayers: LayerId[];
  containerWidth: number;
}) {
  const active = React.useMemo(() => new Set(activeLayers), [activeLayers]);

  const rows = React.useMemo(() => {
    const out: Array<{ label: string; value: string; tone?: string }> = [];

    out.push({ label: "Time", value: state.time });
    out.push({ label: "Cursor Price", value: fmtPrice(state.price) });

    if (active.has("forecast-density") && state.isForecast) {
      const column = payload.forecastDensity[state.timeIndex];
      const value = column?.[state.priceIndex] ?? 0;
      // Column sums to 1 across the price lattice, so this is the probability
      // of settling inside this price bin at this time.
      out.push({
        label: "Probability Density",
        value: pct(value, 2),
        tone: "text-signal",
      });
    }

    if (active.has("forecast-median")) {
      const median = payload.medianPath[state.timeIndex];
      if (Number.isFinite(median)) {
        out.push({
          label: "Forecast Median",
          value: fmtPrice(median),
          tone: "text-signal",
        });
      }
    }

    if (active.has("forecast-mode")) {
      const mode = payload.modePath[state.timeIndex];
      if (Number.isFinite(mode))
        out.push({ label: "Mode Path", value: fmtPrice(mode) });
    }

    if (active.has("spy-price") && !state.isForecast) {
      const point =
        payload.historicalPrice[
          Math.min(payload.historicalPrice.length - 1, state.timeIndex)
        ];
      if (point) out.push({ label: "SPY Price", value: fmtPrice(point.value) });
    }

    if (active.has("vwap")) {
      const vwap = payload.levels.vwap;
      if (vwap !== undefined)
        out.push({ label: "VWAP", value: fmtPrice(vwap), tone: "text-warn" });
    }

    const gex = payload.gexSurface[state.timeIndex]?.[state.priceIndex];
    if (active.has("gex-heatmap") && gex !== undefined) {
      out.push({
        label: "Net GEX",
        value: compactUsd(gex * 1e9),
        tone: gex >= 0 ? "text-bull" : "text-bear",
      });
      out.push({
        label: "Gamma State",
        value: gex >= 0 ? "Positive" : "Negative",
        tone: gex >= 0 ? "text-bull" : "text-bear",
      });
    }

    const strike = nearestStrike(payload, state.price);
    if (strike) {
      if (active.has("call-oi")) {
        out.push({
          label: `Call OI @${strike.strike}`,
          value: strike.callOi.toLocaleString(),
        });
      }
      if (active.has("put-oi")) {
        out.push({
          label: `Put OI @${strike.strike}`,
          value: strike.putOi.toLocaleString(),
        });
      }
      if (state.isForecast) {
        out.push({
          label: "Touch Probability",
          value: pct(strike.touchProbability, 0),
        });
        out.push({ label: "Finish Above", value: pct(strike.finishAbove, 0) });
      }
    }

    return out;
  }, [state, payload, active]);

  // Flip the card to the cursor's left near the right edge so it never clips.
  const flip = state.x > containerWidth - 260;
  const style: React.CSSProperties = {
    left: flip ? state.x - 248 : state.x + 18,
    top: Math.max(8, state.y - 12),
  };

  return (
    <div
      className={cn(
        "pointer-events-none absolute z-30 w-[230px] rounded-md border border-line-bright",
        "bg-void/95 px-2.5 py-2 shadow-panel backdrop-blur",
      )}
      style={style}
    >
      <div className="mb-1.5 flex items-center justify-between border-b border-line pb-1">
        <span className="text-micro font-semibold uppercase tracking-wider text-ink-dim">
          {state.isForecast ? "Forecast" : "Realized"}
        </span>
        <span className="tnum text-micro text-ink-mute">{state.time}</span>
      </div>
      <dl className="space-y-0.5">
        {rows.map((row) => (
          <div
            key={row.label}
            className="flex items-baseline justify-between gap-3"
          >
            <dt className="truncate text-[10px] text-ink-mute">{row.label}</dt>
            <dd
              className={cn(
                "tnum shrink-0 text-[11px] font-medium",
                row.tone ?? "text-ink",
              )}
            >
              {row.value}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function nearestStrike(payload: ForecastChartPayload, price: number) {
  if (payload.strikes.length === 0) return null;
  let best = payload.strikes[0];
  let bestDistance = Math.abs(best.strike - price);
  for (const row of payload.strikes) {
    const distance = Math.abs(row.strike - price);
    if (distance < bestDistance) {
      best = row;
      bestDistance = distance;
    }
  }
  // Only report a strike the cursor is actually near.
  return bestDistance <= 1.5 ? best : null;
}
