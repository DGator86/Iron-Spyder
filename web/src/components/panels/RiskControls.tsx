"use client";

import { Meter } from "@/components/ui/primitives";
import type { SystemPayload } from "@/lib/types";
import { cn, NA, orNA, pct, usd } from "@/lib/utils";

/** Compact risk rail matching the desk screenshot's right column. */
export function RiskControls({
  system,
  className,
}: {
  system: SystemPayload;
  className?: string;
}) {
  const lossUsed =
    system.dailyLossLimit !== null &&
    system.dailyLossLimit > 0 &&
    system.dailyPnl !== null
      ? Math.abs(Math.min(0, system.dailyPnl)) / system.dailyLossLimit
      : null;
  const positionsUsed =
    system.maxOpenPositions !== null && system.maxOpenPositions > 0
      ? system.openPositions / system.maxOpenPositions
      : null;

  return (
    <section className={cn("panel flex min-h-0 flex-col", className)}>
      <div className="panel-header">
        <span className="panel-title">Risk &amp; Controls</span>
      </div>
      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto px-3 py-2.5">
        <Stat
          label="Account Equity"
          value={orNA(system.equity, (v) => usd(v, 2))}
          big
        />
        <Stat
          label="Daily P&L"
          value={orNA(
            system.dailyPnl,
            (v) =>
              `${v >= 0 ? "+" : "−"}${usd(Math.abs(v), 2)}${
                system.dailyPnlPercent !== null
                  ? ` (${v >= 0 ? "+" : ""}${system.dailyPnlPercent.toFixed(2)}%)`
                  : ""
              }`,
          )}
          tone={
            system.dailyPnl === null
              ? undefined
              : system.dailyPnl >= 0
                ? "text-bull"
                : "text-bear"
          }
        />
        <Stat
          label="Buying Power"
          value={orNA(system.buyingPower, (v) => usd(v))}
        />
        <Stat label="Open Risk" value={orNA(system.openRisk, (v) => usd(v))} />

        <div>
          <div className="mb-1 flex items-baseline justify-between">
            <span className="text-[9px] font-semibold uppercase tracking-[0.12em] text-ink-mute">
              Daily Loss Limit
            </span>
            <span className="tnum text-[10px] text-ink-dim">
              {lossUsed === null ? NA : pct(lossUsed, 0)}
            </span>
          </div>
          {lossUsed === null ? (
            <p className="text-[10px] text-ink-mute">
              Limit not reported by the engine
            </p>
          ) : (
            <Meter value={lossUsed} tone={lossUsed > 0.7 ? "bear" : "bull"} />
          )}
        </div>

        <div>
          <div className="mb-1 flex items-baseline justify-between">
            <span className="text-[9px] font-semibold uppercase tracking-[0.12em] text-ink-mute">
              Open Positions
            </span>
            <span className="tnum text-[10px] text-ink-dim">
              {system.openPositions}
              {system.maxOpenPositions !== null
                ? ` / ${system.maxOpenPositions}`
                : ""}
            </span>
          </div>
          {positionsUsed === null ? (
            <p className="text-[10px] text-ink-mute">No position cap reported</p>
          ) : (
            <Meter value={positionsUsed} />
          )}
        </div>

        <div className="border-t border-line pt-2">
          <div className="mb-1 flex items-baseline justify-between">
            <span className="text-[9px] font-semibold uppercase tracking-[0.12em] text-ink-mute">
              Model Confidence
            </span>
            <span className="tnum text-[11px] font-semibold text-signal">
              {pct(system.modelConfidence, 0)}
            </span>
          </div>
          <Meter
            value={system.modelConfidence}
            tone={
              system.modelConfidence > 0.7
                ? "bull"
                : system.modelConfidence > 0.4
                  ? "warn"
                  : "bear"
            }
          />
          <div className="mt-2 flex items-baseline justify-between text-[10px] text-ink-mute">
            <span>Data quality</span>
            <span className="tnum text-ink-dim">
              {pct(system.dataQuality, 0)}
            </span>
          </div>
        </div>
      </div>
    </section>
  );
}

function Stat({
  label,
  value,
  big,
  tone,
}: {
  label: string;
  value: string;
  big?: boolean;
  tone?: string;
}) {
  return (
    <div>
      <div className="text-[9px] font-semibold uppercase tracking-[0.12em] text-ink-mute">
        {label}
      </div>
      <div
        className={cn(
          "tnum font-semibold text-ink",
          big ? "text-lg" : "text-sm",
          tone,
        )}
      >
        {value}
      </div>
    </div>
  );
}
