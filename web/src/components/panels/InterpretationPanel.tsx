"use client";

import { AlertTriangle, ShieldCheck } from "lucide-react";
import { Badge, Meter } from "@/components/ui/primitives";
import { InfoDot } from "@/components/ui/tooltip";
import type { InterpretationPayload } from "@/lib/types";
import { cn, pct, price } from "@/lib/utils";

/**
 * What the model currently believes, in words and one number each.
 *
 * Deliberately narrow: this panel exists to name the state and its confidence.
 * Anything spatial — where the mass sits, which levels bound it — belongs on
 * the canvas, not repeated here as text.
 */
export function InterpretationPanel({
  data,
  className,
}: {
  data: InterpretationPayload;
  className?: string;
}) {
  const top = data.stateProbabilities.slice(0, 8).filter((s) => s.probability > 0.001);

  return (
    <section className={cn("panel flex min-h-0 flex-col", className)}>
      <div className="panel-header">
        <span className="panel-title">Market State</span>
        <InfoDot text="Hidden-state posterior from the ensemble. Confidence is the dominant state's probability; stability is its persistence across recent cycles." />
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-3 py-2.5">
        <div className="mb-3">
          <div className="flex items-baseline justify-between gap-2">
            <span className="text-lg font-semibold leading-tight text-signal">
              {humanizeState(data.primaryState)}
            </span>
            <span className="tnum text-lg font-semibold text-ink">
              {pct(data.stateConfidence, 0)}
            </span>
          </div>
          <Meter value={data.stateConfidence} className="mt-1.5" />
          <div className="mt-1.5 flex items-center justify-between text-[10px] text-ink-mute">
            <span>
              Stability <span className="tnum text-ink-dim">{data.stateStability.toFixed(2)}</span>
            </span>
            <span>
              Agreement{" "}
              <span className="tnum text-ink-dim">{data.modelAgreement.toFixed(2)}</span>
            </span>
          </div>
        </div>

        <Row label="Most Likely Path" value={data.mostLikelyPath} />
        <Row
          label="Expected Range"
          value={`${price(data.expectedRangeLow)} – ${price(data.expectedRangeHigh)}`}
        />
        <Row
          label="Highest Density"
          value={`${price(data.highestDensityLow)} – ${price(data.highestDensityHigh)}`}
          tone="text-signal"
        />
        <Row
          label="Upside Breakout"
          value={pct(data.upsideBreakoutProbability, 0)}
          tone="text-bull"
        />
        <Row
          label="Downside Breakdown"
          value={pct(data.downsideBreakdownProbability, 0)}
          tone="text-bear"
        />
        <Row
          label={`Pin @ ${data.pinStrike.toFixed(0)}`}
          value={pct(data.pinProbability, 0)}
          tone="text-warn"
        />

        <div className="mt-3 border-t border-line pt-2">
          <div className="mb-1.5 flex items-center gap-1.5">
            {data.structuralVeto ? (
              <>
                <AlertTriangle className="h-3 w-3 text-bear" />
                <span className="text-[10px] font-semibold uppercase tracking-wider text-bear">
                  Structural Veto
                </span>
              </>
            ) : (
              <>
                <ShieldCheck className="h-3 w-3 text-bull" />
                <span className="text-[10px] font-semibold uppercase tracking-wider text-ink-mute">
                  Structural Veto
                </span>
              </>
            )}
          </div>
          <p className={cn("text-[11px]", data.structuralVeto ? "text-bear" : "text-ink-mute")}>
            {data.structuralVeto ?? "None"}
          </p>
        </div>

        {data.dominantDrivers.length > 0 ? (
          <div className="mt-3 border-t border-line pt-2">
            <div className="panel-title mb-1.5">Dominant Drivers</div>
            <ul className="space-y-1">
              {data.dominantDrivers.map((driver) => (
                <li key={driver} className="flex items-start gap-1.5 text-[11px] text-ink-dim">
                  <span className="mt-1 h-1 w-1 shrink-0 rounded-full bg-signal" />
                  {driver}
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        <div className="mt-3 border-t border-line pt-2">
          <div className="panel-title mb-1.5">State Probabilities</div>
          <div className="space-y-1">
            {top.map((entry) => (
              <div key={entry.state} className="flex items-center gap-2">
                <span
                  className="h-2 w-2 shrink-0 rounded-sm"
                  style={{ backgroundColor: stateColor(entry.state) }}
                  aria-hidden
                />
                <span className="min-w-0 flex-1 truncate text-[11px] text-ink-dim">
                  {humanizeState(entry.state)}
                </span>
                <div className="h-1 w-14 overflow-hidden rounded-full bg-line">
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${entry.probability * 100}%`,
                      backgroundColor: stateColor(entry.state),
                    }}
                  />
                </div>
                <span className="tnum w-8 shrink-0 text-right text-[10px] text-ink-dim">
                  {pct(entry.probability, 0)}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

function Row({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: string;
}) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-t border-line/60 py-1.5 first:border-t-0">
      <span className="text-[11px] text-ink-mute">{label}</span>
      <span className={cn("tnum text-[12px] font-medium text-ink", tone)}>{value}</span>
    </div>
  );
}

export function humanizeState(state: string): string {
  return state.replace(/([a-z])([A-Z])/g, "$1 $2");
}

/** Stable colour per state so the bar chart and any canvas tint agree. */
export function stateColor(state: string): string {
  const map: Record<string, string> = {
    StrongPin: "#FBBF24",
    BroadRange: "#38BDF8",
    BullGrind: "#34D399",
    BearGrind: "#FB923C",
    BullBreakout: "#22C55E",
    BearBreakdown: "#EF4444",
    VolExpansion: "#E879F9",
    VolContraction: "#22D3EE",
    DirectionalRange: "#A78BFA",
    Transition: "#64748B",
    Unstable: "#94A6C0",
  };
  return map[state] ?? "#64748B";
}
