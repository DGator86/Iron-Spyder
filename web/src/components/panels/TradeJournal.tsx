"use client";

import { Loader2, RefreshCw } from "lucide-react";
import * as React from "react";

import { Badge, Button } from "@/components/ui/primitives";
import { useJournal } from "@/hooks/useJournal";
import type { JournalEntry } from "@/lib/deskTypes";
import { cn, signedUsd, usd } from "@/lib/utils";

function fmtWhen(value: string | null): string {
  if (!value) return "—";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function PnlCell({ value }: { value: number | null }) {
  if (value === null || !Number.isFinite(value)) {
    return <span className="text-ink-mute">—</span>;
  }
  return (
    <span className={value >= 0 ? "text-bull" : "text-bear"}>
      {signedUsd(value)}
    </span>
  );
}

function EntryRow({ entry }: { entry: JournalEntry }) {
  return (
    <tr className="border-b border-line/70 last:border-0">
      <td className="px-2 py-2">
        <Badge tone={entry.status === "open" ? "signal" : "neutral"}>
          {entry.status}
        </Badge>
      </td>
      <td className="px-2 py-2">
        <div className="text-[11px] font-semibold text-ink">{entry.strategy}</div>
        <div className="text-[9px] text-ink-mute">{entry.id || "—"}</div>
      </td>
      <td className="px-2 py-2 text-[10px] text-ink-dim">
        <div>{fmtWhen(entry.openedAt)}</div>
        {entry.closedAt ? (
          <div className="text-ink-mute">→ {fmtWhen(entry.closedAt)}</div>
        ) : null}
      </td>
      <td className="px-2 py-2 text-right text-[11px] tabular-nums text-ink-dim">
        {entry.contracts}
      </td>
      <td className="px-2 py-2 text-right text-[11px] tabular-nums text-ink-dim">
        {usd(entry.cost, 0)}
      </td>
      <td className="px-2 py-2 text-right text-[11px] tabular-nums">
        {entry.status === "open" ? (
          <div>
            <PnlCell value={entry.unrealizedPnl} />
            <div className="text-[9px] text-ink-mute">
              mark {entry.currentPrice?.toFixed(2) ?? "—"}
            </div>
          </div>
        ) : (
          <div>
            <PnlCell value={entry.realizedPnl} />
            <div className="text-[9px] uppercase tracking-wider text-ink-mute">
              {entry.exitReason?.replaceAll("_", " ") ?? "closed"}
            </div>
          </div>
        )}
      </td>
    </tr>
  );
}

export function TradeJournal({ className }: { className?: string }) {
  const { data, isLoading, isFetching, refetch, isError, error } = useJournal();
  const [filter, setFilter] = React.useState<"all" | "open" | "closed">("all");

  const entries = React.useMemo(() => {
    const rows = data?.entries ?? [];
    if (filter === "all") return rows;
    return rows.filter((e) => e.status === filter);
  }, [data?.entries, filter]);

  return (
    <section className={cn("panel flex min-h-0 flex-col", className)}>
      <header className="flex shrink-0 items-center justify-between gap-2 border-b border-line px-3 py-2">
        <div>
          <h2 className="text-[12px] font-semibold uppercase tracking-[0.14em] text-ink">
            Trade Journal
          </h2>
          <p className="text-[10px] text-ink-mute">
            Open marks and closed results from the live paper book
          </p>
        </div>
        <Button
          variant="outline"
          size="xs"
          onClick={() => void refetch()}
          disabled={isFetching}
        >
          {isFetching ? (
            <Loader2 className="h-3 w-3 animate-spin" />
          ) : (
            <RefreshCw className="h-3 w-3" />
          )}
          Refresh
        </Button>
      </header>

      {data?.degradedReason ? (
        <div className="border-b border-warn/40 bg-warn/10 px-3 py-1.5 text-[10px] text-warn">
          {data.degradedReason}
        </div>
      ) : null}

      <div className="grid shrink-0 grid-cols-2 gap-2 border-b border-line p-2 sm:grid-cols-4">
        <Stat label="Open" value={String(data?.openCount ?? 0)} />
        <Stat label="Closed" value={String(data?.closedCount ?? 0)} />
        <Stat
          label="Unrealized"
          value={signedUsd(data?.unrealizedPnl ?? 0)}
          tone={(data?.unrealizedPnl ?? 0) >= 0 ? "bull" : "bear"}
        />
        <Stat
          label="Realized"
          value={signedUsd(data?.realizedPnl ?? 0)}
          tone={(data?.realizedPnl ?? 0) >= 0 ? "bull" : "bear"}
        />
      </div>

      <div className="flex shrink-0 gap-1 border-b border-line px-2 py-1.5">
        {(["all", "open", "closed"] as const).map((id) => (
          <button
            key={id}
            type="button"
            onClick={() => setFilter(id)}
            className={cn(
              "rounded px-2 py-1 text-[10px] font-semibold uppercase tracking-wider",
              filter === id
                ? "bg-signal/15 text-signal"
                : "text-ink-mute hover:text-ink-dim",
            )}
          >
            {id}
          </button>
        ))}
      </div>

      <div className="min-h-0 flex-1 overflow-auto">
        {isLoading ? (
          <div className="grid h-full place-items-center text-ink-mute">
            <Loader2 className="h-4 w-4 animate-spin" />
          </div>
        ) : isError ? (
          <div className="p-4 text-[11px] text-bear">
            {error instanceof Error ? error.message : "Journal unavailable"}
          </div>
        ) : entries.length === 0 ? (
          <div className="grid h-full place-items-center px-6 text-center">
            <div>
              <p className="text-[12px] font-semibold text-ink-dim">No trades yet</p>
              <p className="mt-1 text-[10px] text-ink-mute">
                Open positions and closed fills from the supervisor will land here.
              </p>
            </div>
          </div>
        ) : (
          <table className="w-full min-w-[640px] border-collapse text-left">
            <thead className="sticky top-0 bg-panel/95 text-[9px] uppercase tracking-wider text-ink-mute backdrop-blur">
              <tr className="border-b border-line">
                <th className="px-2 py-2 font-semibold">Status</th>
                <th className="px-2 py-2 font-semibold">Strategy</th>
                <th className="px-2 py-2 font-semibold">Timeline</th>
                <th className="px-2 py-2 text-right font-semibold">Qty</th>
                <th className="px-2 py-2 text-right font-semibold">Cost</th>
                <th className="px-2 py-2 text-right font-semibold">
                  Result / Value
                </th>
              </tr>
            </thead>
            <tbody>
              {entries.map((entry) => (
                <EntryRow key={`${entry.status}-${entry.id}-${entry.openedAt}`} entry={entry} />
              ))}
            </tbody>
          </table>
        )}
      </div>
    </section>
  );
}

function Stat({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "bull" | "bear";
}) {
  return (
    <div className="rounded border border-line bg-deep/50 px-2 py-1.5">
      <div className="text-[9px] uppercase tracking-wider text-ink-mute">{label}</div>
      <div
        className={cn(
          "text-[13px] font-semibold tabular-nums",
          tone === "bull" && "text-bull",
          tone === "bear" && "text-bear",
          !tone && "text-ink",
        )}
      >
        {value}
      </div>
    </div>
  );
}
