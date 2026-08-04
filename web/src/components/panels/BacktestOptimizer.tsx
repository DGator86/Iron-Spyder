"use client";

import { Loader2, Play, RefreshCw } from "lucide-react";
import * as React from "react";

import { Badge, Button } from "@/components/ui/primitives";
import { useOptimize } from "@/hooks/useOptimize";
import type { MetricDelta, OptimizeJob } from "@/lib/deskTypes";
import { cn, pct, signedUsd, usd } from "@/lib/utils";

const METRIC_LABELS: Record<string, string> = {
  expectancy: "Expectancy",
  win_rate: "Win rate",
  profit_factor: "Profit factor",
  total_pnl: "Total P&L",
  max_drawdown_fraction: "Max DD",
  sharpe_like: "Sharpe-like",
  no_trade_rate: "No-trade",
  trades: "Trades",
};

/** Lower is better for these keys when coloring deltas. */
const LOWER_BETTER = new Set(["max_drawdown_fraction", "no_trade_rate"]);

function formatMetric(key: string, value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  if (key === "win_rate" || key === "no_trade_rate" || key === "max_drawdown_fraction") {
    return pct(value, 1);
  }
  if (key === "total_pnl" || key === "expectancy" || key === "average_win" || key === "average_loss") {
    return key === "total_pnl" ? usd(value, 0) : signedUsd(value);
  }
  if (key === "trades") return String(Math.round(value));
  return value.toFixed(2);
}

function DeltaCell({
  metricKey,
  delta,
}: {
  metricKey: string;
  delta: MetricDelta | undefined;
}) {
  if (!delta || delta.delta === null || delta.delta === undefined) {
    return <span className="text-ink-mute">—</span>;
  }
  const improved = LOWER_BETTER.has(metricKey)
    ? delta.delta < 0
    : delta.delta > 0;
  const unchanged = Math.abs(delta.delta) < 1e-9;
  return (
    <span
      className={cn(
        "tabular-nums",
        unchanged && "text-ink-mute",
        !unchanged && improved && "text-bull",
        !unchanged && !improved && "text-bear",
      )}
    >
      {unchanged
        ? "0"
        : `${delta.delta > 0 ? "+" : ""}${formatMetric(metricKey, delta.delta)}`}
    </span>
  );
}

export function BacktestOptimizer({ className }: { className?: string }) {
  const { data, isLoading, isFetching, refetch, run, schedule } = useOptimize();
  const [cadence, setCadence] = React.useState<"daily" | "weekly">("daily");
  const [hour, setHour] = React.useState(6);

  React.useEffect(() => {
    if (!data?.schedule) return;
    setCadence(data.schedule.cadence === "weekly" ? "weekly" : "daily");
    setHour(data.schedule.hour_utc ?? 6);
  }, [data?.schedule]);

  const job = data?.activeJob;
  const busy =
    run.isPending ||
    Boolean(job && (job.status === "queued" || job.status === "running"));

  const metricKeys = Object.keys(METRIC_LABELS);

  return (
    <section className={cn("panel flex min-h-0 flex-col", className)}>
      <header className="flex shrink-0 flex-wrap items-center justify-between gap-2 border-b border-line px-3 py-2">
        <div>
          <h2 className="text-[12px] font-semibold uppercase tracking-[0.14em] text-ink">
            Backtest Optimizer
          </h2>
          <p className="text-[10px] text-ink-mute">
            Replay stored SPY tapes, sweep configs, keep what improves expectancy
          </p>
        </div>
        <div className="flex items-center gap-1.5">
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
          <Button
            variant="solid"
            size="sm"
            disabled={busy || !data?.data.available}
            onClick={() =>
              run.mutate({
                session_count: data?.schedule.session_count,
                snapshot_limit: data?.schedule.snapshot_limit,
              })
            }
          >
            {busy ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Play className="h-3.5 w-3.5" />
            )}
            Run backtest
          </Button>
        </div>
      </header>

      {data?.degradedReason ? (
        <div className="border-b border-warn/40 bg-warn/10 px-3 py-1.5 text-[10px] text-warn">
          {data.degradedReason}
        </div>
      ) : null}

      <div className="min-h-0 flex-1 overflow-auto">
        {isLoading || !data ? (
          <div className="grid h-40 place-items-center text-ink-mute">
            <Loader2 className="h-4 w-4 animate-spin" />
          </div>
        ) : (
          <div className="space-y-3 p-3">
            <div className="grid gap-2 sm:grid-cols-3">
              <InfoCard
                label="Stored sessions"
                value={String(data.data.sessions)}
                sub={data.data.available ? "Ready to train" : "No tapes found"}
              />
              <InfoCard
                label="Active job"
                value={job?.status ?? "idle"}
                sub={job?.id ?? "Queue empty"}
              />
              <InfoCard
                label="Last result"
                value={
                  data.latestRun?.improved
                    ? "Improved"
                    : data.latestRun
                      ? "No lift"
                      : "—"
                }
                sub={
                  data.latestRun?.finished_at
                    ? new Date(data.latestRun.finished_at).toLocaleString()
                    : "No completed runs"
                }
              />
            </div>

            {busy ? <RunProgress job={job} pending={run.isPending} /> : null}

            <div className="rounded border border-line bg-deep/40 p-3">
              <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                <div>
                  <h3 className="text-[11px] font-semibold uppercase tracking-wider text-ink">
                    Training schedule
                  </h3>
                  <p className="text-[10px] text-ink-mute">
                    Worker polls every minute; scheduled runs enqueue automatically
                  </p>
                </div>
                <Badge tone={data.schedule.enabled ? "bull" : "neutral"}>
                  {data.schedule.enabled ? "Enabled" : "Off"}
                </Badge>
              </div>
              <div className="flex flex-wrap items-end gap-2">
                <label className="text-[10px] text-ink-mute">
                  Cadence
                  <select
                    className="mt-1 block rounded border border-line bg-panel px-2 py-1.5 text-[11px] text-ink"
                    value={cadence}
                    onChange={(e) =>
                      setCadence(e.target.value === "weekly" ? "weekly" : "daily")
                    }
                  >
                    <option value="daily">Daily</option>
                    <option value="weekly">Weekly (Mon)</option>
                  </select>
                </label>
                <label className="text-[10px] text-ink-mute">
                  Hour (UTC)
                  <input
                    type="number"
                    min={0}
                    max={23}
                    value={hour}
                    onChange={(e) => setHour(Number(e.target.value))}
                    className="mt-1 block w-20 rounded border border-line bg-panel px-2 py-1.5 text-[11px] text-ink"
                  />
                </label>
                <Button
                  variant="solid"
                  size="sm"
                  disabled={schedule.isPending}
                  onClick={() =>
                    schedule.mutate({
                      enabled: true,
                      cadence,
                      hour_utc: hour,
                      session_count: data.schedule.session_count,
                      snapshot_limit: data.schedule.snapshot_limit,
                    })
                  }
                >
                  Save schedule
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={schedule.isPending || !data.schedule.enabled}
                  onClick={() => schedule.mutate({ enabled: false })}
                >
                  Disable
                </Button>
              </div>
              <div className="mt-2 flex flex-wrap gap-3 text-[10px] text-ink-mute">
                <span>
                  Next:{" "}
                  {data.schedule.next_run_at
                    ? new Date(data.schedule.next_run_at).toLocaleString()
                    : "—"}
                </span>
                <span>
                  Last:{" "}
                  {data.schedule.last_run_at
                    ? new Date(data.schedule.last_run_at).toLocaleString()
                    : "—"}
                </span>
              </div>
            </div>

            <div className="rounded border border-line">
              <div className="border-b border-line px-3 py-2">
                <h3 className="text-[11px] font-semibold uppercase tracking-wider text-ink">
                  Metrics vs prior configuration
                </h3>
                <p className="text-[10px] text-ink-mute">
                  Green = better than the baseline config that started this run
                </p>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full min-w-[560px] border-collapse text-left">
                  <thead className="text-[9px] uppercase tracking-wider text-ink-mute">
                    <tr className="border-b border-line">
                      <th className="px-3 py-2 font-semibold">Metric</th>
                      <th className="px-3 py-2 text-right font-semibold">Prior</th>
                      <th className="px-3 py-2 text-right font-semibold">Current</th>
                      <th className="px-3 py-2 text-right font-semibold">Δ</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metricKeys.map((key) => {
                      const row = data.deltas[key];
                      return (
                        <tr key={key} className="border-b border-line/70 last:border-0">
                          <td className="px-3 py-2 text-[11px] text-ink-dim">
                            {METRIC_LABELS[key] ?? key}
                          </td>
                          <td className="px-3 py-2 text-right text-[11px] tabular-nums text-ink-mute">
                            {formatMetric(key, row?.prior)}
                          </td>
                          <td className="px-3 py-2 text-right text-[11px] tabular-nums text-ink">
                            {formatMetric(key, row?.current)}
                          </td>
                          <td className="px-3 py-2 text-right text-[11px]">
                            <DeltaCell metricKey={key} delta={row} />
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {data.latestRun?.config_changes &&
            Object.keys(data.latestRun.config_changes).length > 0 ? (
              <div className="rounded border border-line p-3">
                <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-ink">
                  Config changes promoted
                </h3>
                <ul className="space-y-1 text-[11px] text-ink-dim">
                  {Object.entries(data.latestRun.config_changes).map(
                    ([key, change]) => (
                      <li key={key} className="flex justify-between gap-3">
                        <span className="text-ink-mute">{key}</span>
                        <span className="tabular-nums">
                          {String(change.prior)} → {String(change.current)}
                        </span>
                      </li>
                    ),
                  )}
                </ul>
              </div>
            ) : null}

            {data.runs.length > 0 ? (
              <div className="rounded border border-line">
                <div className="border-b border-line px-3 py-2 text-[11px] font-semibold uppercase tracking-wider text-ink">
                  Recent runs
                </div>
                <ul className="divide-y divide-line/70">
                  {data.runs.slice(0, 6).map((item) => (
                    <li
                      key={item.id}
                      className="flex flex-wrap items-center justify-between gap-2 px-3 py-2 text-[11px]"
                    >
                      <div>
                        <div className="font-semibold text-ink">{item.id}</div>
                        <div className="text-[10px] text-ink-mute">
                          {(item.sessions || []).slice(-3).join(", ") || "—"}
                        </div>
                      </div>
                      <div className="text-right">
                        <Badge tone={item.improved ? "bull" : "neutral"}>
                          {item.improved ? "Improved" : item.status}
                        </Badge>
                        <div className="mt-1 tabular-nums text-ink-dim">
                          E[PnL]{" "}
                          {formatMetric(
                            "expectancy",
                            typeof item.metrics?.expectancy === "number"
                              ? item.metrics.expectancy
                              : null,
                          )}
                        </div>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
          </div>
        )}
      </div>
    </section>
  );
}

function InfoCard({
  label,
  value,
  sub,
}: {
  label: string;
  value: string;
  sub: string;
}) {
  return (
    <div className="rounded border border-line bg-deep/50 px-3 py-2">
      <div className="text-[9px] uppercase tracking-wider text-ink-mute">{label}</div>
      <div className="text-[14px] font-semibold capitalize text-ink">{value}</div>
      <div className="truncate text-[10px] text-ink-mute">{sub}</div>
    </div>
  );
}

function RunProgress({
  job,
  pending,
}: {
  job: OptimizeJob | null | undefined;
  pending: boolean;
}) {
  const progress = job?.progress ?? null;
  const status = job?.status ?? null;
  const percent = clampPercent(
    progress?.percent ??
      (status === "running" ? 8 : pending || status === "queued" ? 2 : 0),
  );
  const message =
    progress?.message ||
    (pending
      ? "Queueing backtest"
      : status === "queued"
        ? "Waiting for worker"
        : status === "running"
          ? "Running backtest"
          : "Working");
  const stepLabel =
    progress && progress.total > 0
      ? `${Math.min(progress.current, progress.total)} / ${progress.total}`
      : null;

  return (
    <div
      className="rounded border border-signal/35 bg-signal/5 px-3 py-3"
      role="status"
      aria-live="polite"
      aria-busy="true"
    >
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-[11px] font-semibold text-ink">
            <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-signal" />
            <span className="truncate">{message}</span>
          </div>
          {progress?.detail ? (
            <p className="mt-0.5 truncate text-[10px] text-ink-mute">
              {progress.detail}
            </p>
          ) : null}
        </div>
        <div className="text-right text-[11px] tabular-nums text-ink">
          <div className="font-semibold">{percent.toFixed(0)}%</div>
          {stepLabel ? (
            <div className="text-[10px] text-ink-mute">Step {stepLabel}</div>
          ) : null}
        </div>
      </div>
      <div
        className="h-2 overflow-hidden rounded-sm bg-line/70"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(percent)}
        aria-label="Backtest progress"
      >
        <div
          className="h-full rounded-sm bg-signal transition-[width] duration-500 ease-out"
          style={{ width: `${percent}%` }}
        />
      </div>
      {progress?.phase ? (
        <div className="mt-1.5 text-[9px] uppercase tracking-[0.14em] text-ink-mute">
          {progress.phase}
        </div>
      ) : null}
    </div>
  );
}

function clampPercent(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, value));
}
