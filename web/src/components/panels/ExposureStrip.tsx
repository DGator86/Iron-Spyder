"use client";

import type { SystemPayload } from "@/lib/types";
import { cn, NA, orNA, price } from "@/lib/utils";

/**
 * Dealer-positioning strip under the radar — the five numbers the desk keeps
 * visible without opening a tab.
 */
export function ExposureStrip({
  system,
  className,
}: {
  system: SystemPayload;
  className?: string;
}) {
  const tiles: Array<{
    label: string;
    value: string;
    tone?: string;
    sub?: string;
  }> = [
    {
      label: "Net GEX",
      value: signedUnit(system.netGex, "B"),
      tone: system.netGex >= 0 ? "text-bull" : "text-bear",
      sub: "γ exposure",
    },
    {
      label: "GEX Slope",
      value: orNA(system.gexSlope, (v) => v.toFixed(2)),
      tone:
        system.gexSlope === null
          ? undefined
          : system.gexSlope >= 0
            ? "text-bull"
            : "text-bear",
      sub: "at spot",
    },
    {
      label: "DEX Trend",
      value: signedUnit(system.dexTrend, "M"),
      tone: system.dexTrend >= 0 ? "text-bull" : "text-bear",
      sub: "δ flow",
    },
    {
      label: "Vanna",
      value: signedUnit(system.vannaExposure, "B"),
      tone: system.vannaExposure >= 0 ? "text-magenta" : "text-bear",
      sub: "ν exposure",
    },
    {
      label: "Charm",
      value: signedUnit(system.charmExposure, "B"),
      tone: system.charmExposure >= 0 ? "text-bull" : "text-bear",
      sub: "θ·δ",
    },
  ];

  const emLow = system.spot - system.expectedMove;
  const emHigh = system.spot + system.expectedMove;

  return (
    <section
      className={cn(
        "grid shrink-0 grid-cols-2 gap-1.5 sm:grid-cols-3 lg:grid-cols-6",
        className,
      )}
    >
      {tiles.map((tile) => (
        <div
          key={tile.label}
          className="border border-line bg-deep/70 px-2.5 py-2"
        >
          <div className="flex items-baseline justify-between gap-1">
            <span className="text-[9px] font-semibold uppercase tracking-[0.12em] text-ink-mute">
              {tile.label}
            </span>
            {tile.sub ? (
              <span className="text-[9px] text-ink-mute/70">{tile.sub}</span>
            ) : null}
          </div>
          <div className={cn("tnum mt-0.5 text-base font-semibold", tile.tone)}>
            {tile.value}
          </div>
        </div>
      ))}

      <div className="col-span-2 border border-line bg-deep/70 px-2.5 py-2 sm:col-span-3 lg:col-span-1">
        <div className="text-[9px] font-semibold uppercase tracking-[0.12em] text-ink-mute">
          Expected Move
        </div>
        <div className="tnum mt-0.5 text-sm font-semibold text-ink">
          ±{system.expectedMove.toFixed(2)}{" "}
          <span className="text-[11px] font-medium text-ink-dim">
            ({system.expectedMovePercent.toFixed(2)}%)
          </span>
        </div>
        <div className="mt-1.5 h-1.5 overflow-hidden bg-raised">
          <div
            className="h-full bg-gradient-to-r from-bear/70 via-signal/50 to-bull/70"
            style={{
              width: `${Math.min(100, Math.max(8, system.emUtilization * 100))}%`,
            }}
          />
        </div>
        <div className="mt-1 flex justify-between text-[9px] text-ink-mute">
          <span className="tnum">{price(emLow)}</span>
          <span className="tnum">{price(emHigh)}</span>
        </div>
      </div>
    </section>
  );
}

function signedUnit(value: number, unit: "B" | "M"): string {
  if (!Number.isFinite(value)) return NA;
  const sign = value >= 0 ? "+" : "−";
  return `${sign}${Math.abs(value).toFixed(2)}${unit}`;
}
