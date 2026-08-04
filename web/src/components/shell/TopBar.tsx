"use client";

import { AlertOctagon, Settings, Wifi, WifiOff } from "lucide-react";
import { Badge, Button } from "@/components/ui/primitives";
import type { SystemPayload } from "@/lib/types";
import { cn, NA, orNA, price, signedPct } from "@/lib/utils";

/**
 * Instrument header. Everything here is session-constant or slow-moving —
 * fast-moving analytics belong on the canvas where they can be compared
 * against price, not stranded in a header strip.
 */
export function TopBar({
  system,
  source,
  degradedReason,
  className,
}: {
  system: SystemPayload;
  source: "live" | "synthetic";
  degradedReason?: string;
  className?: string;
}) {
  // No change feed means no direction to colour by; stay neutral rather than
  // painting the tape green because null failed a `< 0` test.
  const up = system.change === null ? null : system.change >= 0;

  const sessionStats = [
    { label: "VWAP", value: orNA(system.vwap, price) },
    {
      label: "IVR",
      value: orNA(system.ivRank, (v) => `${v.toFixed(1)}%`),
    },
    { label: "IV ATM", value: system.atmIv.toFixed(1) },
    { label: "HV", value: system.realizedVol.toFixed(1) },
    { label: "DTE", value: system.dte },
    { label: "Time", value: system.serverTime },
  ];

  return (
    <header
      className={cn(
        "shrink-0 border-b border-line bg-panel/70 backdrop-blur",
        className,
      )}
    >
      <div className="flex items-center gap-3 overflow-x-auto px-3 py-2 sm:gap-4">
        <div className="flex shrink-0 items-center gap-2">
          <div className="grid h-7 w-7 place-items-center rounded-md border border-signal/40 bg-signal/10">
            <svg viewBox="0 0 24 24" className="h-4 w-4" aria-hidden>
              <path
                d="M12 2 L20 7 L20 17 L12 22 L4 17 L4 7 Z"
                fill="none"
                stroke="#22D3EE"
                strokeWidth="1.6"
              />
              <circle cx="12" cy="12" r="2.5" fill="#22D3EE" />
            </svg>
          </div>
          <div className="leading-none">
            <div className="text-sm font-bold tracking-wide text-ink">
              SPY-DER
            </div>
            <div className="hidden text-[9px] uppercase tracking-[0.14em] text-ink-mute sm:block">
              Defined Risk Option Intelligence
            </div>
          </div>
        </div>

        <div className="hidden shrink-0 items-center gap-1.5 sm:flex">
          <span
            className={cn(
              "h-1.5 w-1.5 rounded-full",
              source === "live" ? "bg-live animate-pulse-live" : "bg-warn",
            )}
          />
          <span className="text-[10px] font-semibold uppercase tracking-wider text-ink-dim">
            {system.mode} trading
          </span>
        </div>

        {source === "synthetic" ? (
          <Badge tone="warn" className="shrink-0">
            Synthetic data
          </Badge>
        ) : null}

        {degradedReason ? (
          <span
            className="hidden max-w-[220px] truncate text-[10px] text-warn lg:inline"
            title={degradedReason}
          >
            {degradedReason}
          </span>
        ) : null}

        <div className="flex shrink-0 items-baseline gap-2 border-l border-line pl-3 sm:pl-4">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-ink-dim">
            SPY
          </span>
          <span
            className={cn(
              "tnum text-lg font-bold sm:text-xl",
              up === null ? "text-ink" : up ? "text-bull" : "text-bear",
            )}
          >
            {price(system.spot)}
          </span>
          <span
            className={cn(
              "tnum text-[11px]",
              up === null ? "text-ink-mute" : up ? "text-bull" : "text-bear",
            )}
          >
            {system.change === null
              ? NA
              : `${system.change >= 0 ? "+" : ""}${system.change.toFixed(2)} (${orNA(
                  system.changePercent,
                  (v) => signedPct(v),
                )})`}
          </span>
        </div>

        <div className="hidden min-w-0 flex-1 items-center gap-3 overflow-x-auto md:flex lg:gap-4">
          {sessionStats.map((stat) => (
            <HeaderStat key={stat.label} label={stat.label} value={stat.value} />
          ))}
        </div>

        <div className="ml-auto flex shrink-0 items-center gap-2">
          <div
            className={cn(
              "flex items-center gap-1.5 border px-2 py-1",
              system.killSwitch
                ? "border-bear bg-bear/15"
                : "border-bear/50 bg-bear/10",
            )}
            title={
              system.killSwitch
                ? `Kill switch tripped: ${system.killSwitchReasons.join(", ")}`
                : "Kill switch armed and clear"
            }
          >
            <AlertOctagon
              className={cn(
                "h-3 w-3",
                system.killSwitch ? "text-bear" : "text-bear/80",
              )}
            />
            <span
              className={cn(
                "text-[10px] font-bold uppercase tracking-wider",
                system.killSwitch ? "text-bear" : "text-bear/90",
              )}
            >
              <span className="sm:hidden">KILL</span>
              <span className="hidden sm:inline">Kill Switch</span>
            </span>
          </div>

          <div
            className="flex items-center gap-1"
            title={system.connected ? "Connected" : "Disconnected"}
          >
            {system.connected ? (
              <Wifi className="h-3.5 w-3.5 text-live" />
            ) : (
              <WifiOff className="h-3.5 w-3.5 text-dead" />
            )}
          </div>

          <Button size="icon" variant="ghost" aria-label="Settings">
            <Settings className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      {/* Phone ticker — same session stats as desktop, never dropped */}
      <div className="flex gap-3 overflow-x-auto border-t border-line/70 px-3 py-1.5 md:hidden">
        <div className="flex shrink-0 items-center gap-1.5">
          <span
            className={cn(
              "h-1.5 w-1.5 rounded-full",
              source === "live" ? "bg-live animate-pulse-live" : "bg-warn",
            )}
          />
          <span className="text-[9px] font-semibold uppercase tracking-wider text-ink-dim">
            {system.mode}
          </span>
        </div>
        {sessionStats.map((stat) => (
          <HeaderStat key={stat.label} label={stat.label} value={stat.value} />
        ))}
      </div>
    </header>
  );
}

function HeaderStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex shrink-0 items-baseline gap-1.5">
      <span className="text-[10px] uppercase tracking-wider text-ink-mute">
        {label}
      </span>
      <span className="tnum text-[12px] font-medium text-ink">{value}</span>
    </div>
  );
}
