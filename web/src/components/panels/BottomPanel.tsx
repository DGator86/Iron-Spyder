"use client";

import * as React from "react";
import { Badge, Meter } from "@/components/ui/primitives";
import { strategyGeometry } from "@/lib/payoff";
import type {
  ForecastChartPayload,
  InterpretationPayload,
  StrategyCandidate,
  SystemPayload,
} from "@/lib/types";
import { cn, compactUsd, pct, price, usd } from "@/lib/utils";
import { useViewStore } from "@/store/viewStore";
import { PayoffSpark } from "./PayoffSpark";

const TABS = [
  "Strategy Candidates",
  "Terminal Probabilities",
  "Strike Inspector",
  "Options Flow",
  "Volatility Surface",
  "Risk",
  "Model Health",
  "Backtest Replay",
] as const;

type Tab = (typeof TABS)[number];

export function BottomPanel({
  chart,
  strategies,
  interpretation,
  system,
  className,
}: {
  chart: ForecastChartPayload;
  strategies: StrategyCandidate[];
  interpretation: InterpretationPayload;
  system: SystemPayload;
  className?: string;
}) {
  const [tab, setTab] = React.useState<Tab>("Strategy Candidates");
  const selectedStrategyId = useViewStore((s) => s.selectedStrategyId);
  const selectStrategy = useViewStore((s) => s.selectStrategy);
  const selectedStrike = useViewStore((s) => s.selectedStrike);

  return (
    <section className={cn("panel flex min-h-0 flex-col", className)}>
      <div
        role="tablist"
        aria-label="Analytical panels"
        className="flex shrink-0 items-center gap-0.5 overflow-x-auto border-b border-line px-1.5"
      >
        {TABS.map((t) => (
          <button
            key={t}
            role="tab"
            aria-selected={t === tab}
            onClick={() => setTab(t)}
            className={cn(
              "shrink-0 whitespace-nowrap px-2.5 py-2 text-[10px] font-semibold uppercase tracking-wider",
              "border-b-2 transition-colors duration-100",
              t === tab
                ? "border-signal text-signal"
                : "border-transparent text-ink-mute hover:text-ink-dim",
            )}
          >
            {t}
          </button>
        ))}
      </div>

      <div className="min-h-0 flex-1 overflow-auto">
        {tab === "Strategy Candidates" ? (
          <StrategyTable
            strategies={strategies}
            selectedId={selectedStrategyId}
            onSelect={(id) => selectStrategy(id === selectedStrategyId ? undefined : id)}
          />
        ) : null}
        {tab === "Terminal Probabilities" ? <TerminalProbabilities chart={chart} /> : null}
        {tab === "Strike Inspector" ? (
          <StrikeInspector chart={chart} strike={selectedStrike} />
        ) : null}
        {tab === "Options Flow" ? <OptionsFlow chart={chart} /> : null}
        {tab === "Volatility Surface" ? <VolatilitySurface system={system} /> : null}
        {tab === "Risk" ? <RiskPanel system={system} /> : null}
        {tab === "Model Health" ? (
          <ModelHealth chart={chart} interpretation={interpretation} system={system} />
        ) : null}
        {tab === "Backtest Replay" ? <BacktestReplay /> : null}
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------

function StrategyTable({
  strategies,
  selectedId,
  onSelect,
}: {
  strategies: StrategyCandidate[];
  selectedId?: string;
  onSelect: (id: string) => void;
}) {
  if (strategies.length === 0) {
    return <Empty>No candidate cleared the no-trade utility floor this cycle.</Empty>;
  }

  return (
    <table className="w-full border-collapse text-[11px]">
      <thead className="sticky top-0 z-10 bg-panel">
        <tr className="text-[9px] uppercase tracking-wider text-ink-mute">
          <Th className="w-8 text-center">#</Th>
          <Th className="text-left">Strategy</Th>
          <Th className="text-left">Exp</Th>
          <Th className="text-left">Strikes</Th>
          <Th>Net</Th>
          <Th>Max Profit</Th>
          <Th>Max Loss</Th>
          <Th>POP</Th>
          <Th>EV</Th>
          <Th>R/R</Th>
          <Th>Utility</Th>
          <Th>Fill</Th>
          <Th>Assign</Th>
          <Th className="w-16 text-center">Payoff</Th>
        </tr>
      </thead>
      <tbody>
        {strategies.map((s) => {
          const selected = s.strategyId === selectedId;
          const geo = strategyGeometry(s);
          return (
            <tr
              key={s.strategyId}
              onClick={() => onSelect(s.strategyId)}
              className={cn(
                "cursor-pointer border-t border-line/60 transition-colors",
                selected ? "bg-signal/10" : "hover:bg-raised/50",
              )}
            >
              <Td className="text-center">
                <span
                  className={cn(
                    "inline-grid h-4 w-4 place-items-center rounded text-[9px] font-bold",
                    s.rank === 1 ? "bg-warn text-void" : "bg-raised text-ink-dim",
                  )}
                >
                  {s.rank}
                </span>
              </Td>
              <Td className="text-left font-medium text-ink">{s.label}</Td>
              <Td className="text-left text-ink-mute">{s.expiration}</Td>
              <Td className="text-left">
                <span className="tnum text-ink-dim">
                  {s.legs
                    .map((l) => `${l.quantity > 0 ? "+" : ""}${l.quantity}×${l.strike}${l.right[0].toUpperCase()}`)
                    .join(" ")}
                </span>
              </Td>
              <Td className={s.isCredit ? "text-bull" : "text-bear"}>
                {s.isCredit ? "+" : "−"}
                {usd(s.netPrice)}
              </Td>
              <Td className="text-bull">
                {s.maxProfit === null ? "Unlimited" : usd(s.maxProfit)}
              </Td>
              <Td className="text-bear">{usd(s.maxLoss)}</Td>
              <Td>{pct(s.probabilityOfProfit, 0)}</Td>
              <Td className={s.expectedValue >= 0 ? "text-bull" : "text-bear"}>
                {usd(s.expectedValue)}
              </Td>
              <Td>{s.expectedReturnOnRisk.toFixed(2)}</Td>
              <Td className="text-signal">{s.utility.toFixed(2)}</Td>
              <Td>{pct(s.fillProbability, 0)}</Td>
              <Td>
                <Badge
                  tone={
                    s.assignmentRisk === "high"
                      ? "bear"
                      : s.assignmentRisk === "medium"
                        ? "warn"
                        : "neutral"
                  }
                >
                  {s.assignmentRisk}
                </Badge>
              </Td>
              <Td className="text-center">
                <PayoffSpark curve={geo.curve} className="mx-auto h-6 w-14" />
              </Td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

// ---------------------------------------------------------------------------

function TerminalProbabilities({ chart }: { chart: ForecastChartPayload }) {
  const rows = chart.strikes.filter((s) => Math.abs(s.strike - chart.spot) <= 10);

  return (
    <table className="w-full border-collapse text-[11px]">
      <thead className="sticky top-0 bg-panel">
        <tr className="text-[9px] uppercase tracking-wider text-ink-mute">
          <Th className="text-left">Strike</Th>
          <Th>Touch %</Th>
          <Th>Finish Above %</Th>
          <Th>Finish Below %</Th>
          <Th className="w-32">Distribution</Th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => {
          const atSpot = Math.abs(row.strike - chart.spot) < 0.75;
          return (
            <tr
              key={row.strike}
              className={cn(
                "border-t border-line/60",
                atSpot ? "bg-signal/10" : "hover:bg-raised/40",
              )}
            >
              <Td className="text-left font-medium text-ink">{row.strike}</Td>
              <Td>{pct(row.touchProbability, 0)}</Td>
              <Td className="text-bull">{pct(row.finishAbove, 0)}</Td>
              <Td className="text-bear">{pct(1 - row.finishAbove, 0)}</Td>
              <Td>
                <div className="flex h-2 w-full overflow-hidden rounded-full bg-line">
                  <div
                    className="h-full bg-bear/70"
                    style={{ width: `${(1 - row.finishAbove) * 100}%` }}
                  />
                  <div className="h-full bg-bull/70" style={{ width: `${row.finishAbove * 100}%` }} />
                </div>
              </Td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

// ---------------------------------------------------------------------------

function StrikeInspector({
  chart,
  strike,
}: {
  chart: ForecastChartPayload;
  strike?: number;
}) {
  const row = strike ? chart.strikes.find((s) => s.strike === strike) : undefined;

  if (!row) {
    return <Empty>Click a strike on the canvas to inspect it.</Empty>;
  }

  const total = row.callOi + row.putOi;

  return (
    <div className="grid grid-cols-2 gap-3 p-3 md:grid-cols-4">
      <Field label="Strike" value={String(row.strike)} big />
      <Field label="Distance from spot" value={`${(row.strike - chart.spot).toFixed(2)}`} />
      <Field label="Net GEX" value={compactUsd(row.netGex * 1e9)} />
      <Field label="Signed premium" value={compactUsd(row.signedPremium)} />
      <Field label="Call OI" value={row.callOi.toLocaleString()} />
      <Field label="Put OI" value={row.putOi.toLocaleString()} />
      <Field label="Call volume" value={row.callVolume.toLocaleString()} />
      <Field label="Put volume" value={row.putVolume.toLocaleString()} />
      <Field label="Touch probability" value={pct(row.touchProbability, 0)} />
      <Field label="Finish above" value={pct(row.finishAbove, 0)} />
      <div className="col-span-2 md:col-span-4">
        <div className="panel-title mb-1.5">Call / Put OI split</div>
        <div className="flex h-3 overflow-hidden rounded bg-line">
          <div
            className="h-full bg-bull/70"
            style={{ width: total > 0 ? `${(row.callOi / total) * 100}%` : "50%" }}
          />
          <div
            className="h-full bg-bear/70"
            style={{ width: total > 0 ? `${(row.putOi / total) * 100}%` : "50%" }}
          />
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------

function OptionsFlow({ chart }: { chart: ForecastChartPayload }) {
  const rows = [...chart.strikes]
    .filter((s) => s.signedPremium !== 0)
    .sort((a, b) => Math.abs(b.signedPremium) - Math.abs(a.signedPremium))
    .slice(0, 14);

  if (rows.length === 0) {
    return <Empty>The engine does not expose strike-level premium flow yet.</Empty>;
  }

  const peak = Math.max(...rows.map((r) => Math.abs(r.signedPremium)), 1);

  return (
    <div className="space-y-1 p-3">
      {rows.map((row) => (
        <div key={row.strike} className="flex items-center gap-3">
          <span className="tnum w-12 shrink-0 text-[11px] text-ink-dim">{row.strike}</span>
          <div className="relative h-3 flex-1 rounded bg-line/40">
            <div
              className={cn(
                "absolute top-0 h-full rounded",
                row.signedPremium >= 0 ? "left-1/2 bg-bull/70" : "right-1/2 bg-bear/70",
              )}
              style={{ width: `${(Math.abs(row.signedPremium) / peak) * 50}%` }}
            />
            <div className="absolute left-1/2 top-0 h-full w-px bg-line-bright" />
          </div>
          <span
            className={cn(
              "tnum w-16 shrink-0 text-right text-[11px]",
              row.signedPremium >= 0 ? "text-bull" : "text-bear",
            )}
          >
            {compactUsd(row.signedPremium)}
          </span>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------

function VolatilitySurface({ system }: { system: SystemPayload }) {
  return (
    <div className="grid grid-cols-2 gap-3 p-3 md:grid-cols-4">
      <Field label="ATM IV" value={`${system.atmIv.toFixed(2)}%`} big />
      <Field label="Realized vol" value={`${system.realizedVol.toFixed(2)}%`} />
      <Field
        label="IV − RV spread"
        value={`${(system.atmIv - system.realizedVol).toFixed(2)}%`}
      />
      <Field label="IV rank" value={`${system.ivRank.toFixed(1)}%`} />
      <Field label="Expected move" value={`±${system.expectedMove.toFixed(2)}`} />
      <Field label="EM %" value={`${system.expectedMovePercent.toFixed(2)}%`} />
      <Field label="EM utilization" value={pct(system.emUtilization, 0)} />
      <Field label="Vanna exposure" value={`${system.vannaExposure.toFixed(2)}B`} />
    </div>
  );
}

// ---------------------------------------------------------------------------

function RiskPanel({ system }: { system: SystemPayload }) {
  const lossUsed = system.dailyLossLimit > 0 ? Math.abs(Math.min(0, system.dailyPnl)) / system.dailyLossLimit : 0;

  return (
    <div className="grid grid-cols-2 gap-3 p-3 md:grid-cols-4">
      <Field label="Account equity" value={usd(system.equity, 2)} big />
      <Field
        label="Daily P&L"
        value={`${system.dailyPnl >= 0 ? "+" : "−"}${usd(Math.abs(system.dailyPnl), 2)}`}
        tone={system.dailyPnl >= 0 ? "text-bull" : "text-bear"}
      />
      <Field label="Open risk" value={usd(system.openRisk)} />
      <Field label="Buying power" value={usd(system.buyingPower)} />
      <div className="col-span-2">
        <div className="panel-title mb-1">Daily loss limit</div>
        <Meter value={lossUsed} tone={lossUsed > 0.7 ? "bear" : "bull"} />
        <div className="mt-1 flex justify-between text-[10px] text-ink-mute">
          <span>{usd(Math.abs(Math.min(0, system.dailyPnl)))} used</span>
          <span>{usd(system.dailyLossLimit)} limit</span>
        </div>
      </div>
      <div className="col-span-2">
        <div className="panel-title mb-1">Open positions</div>
        <Meter
          value={system.maxOpenPositions > 0 ? system.openPositions / system.maxOpenPositions : 0}
        />
        <div className="mt-1 text-[10px] text-ink-mute">
          {system.openPositions} / {system.maxOpenPositions || "—"}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------

function ModelHealth({
  chart,
  interpretation,
  system,
}: {
  chart: ForecastChartPayload;
  interpretation: InterpretationPayload;
  system: SystemPayload;
}) {
  const items = [
    { label: "Model confidence", value: chart.confidence },
    { label: "Model agreement", value: chart.modelAgreement },
    { label: "Data quality", value: chart.dataQuality },
    { label: "State stability", value: interpretation.stateStability },
    { label: "EM utilization", value: system.emUtilization },
  ];

  return (
    <div className="grid grid-cols-1 gap-3 p-3 sm:grid-cols-2 lg:grid-cols-3">
      {items.map((item) => (
        <div key={item.label} className="rounded border border-line bg-deep/50 px-3 py-2.5">
          <div className="mb-1 flex items-baseline justify-between">
            <span className="text-[10px] uppercase tracking-wider text-ink-mute">
              {item.label}
            </span>
            <span className="tnum text-sm font-semibold text-ink">{pct(item.value, 0)}</span>
          </div>
          <Meter
            value={item.value}
            tone={item.value > 0.7 ? "bull" : item.value > 0.4 ? "warn" : "bear"}
          />
        </div>
      ))}
    </div>
  );
}

function BacktestReplay() {
  return (
    <Empty>
      Load a recorded session with the timeline scrubber, then press play to watch the layers
      evolve. Historical tapes are served by the engine&apos;s import pipeline.
    </Empty>
  );
}

// ---------------------------------------------------------------------------

function Th({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <th className={cn("px-2 py-1.5 text-right font-semibold", className)}>{children}</th>
  );
}

function Td({ children, className }: { children: React.ReactNode; className?: string }) {
  return <td className={cn("tnum px-2 py-1.5 text-right", className)}>{children}</td>;
}

function Field({
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
      <div className="panel-title mb-0.5">{label}</div>
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

function Empty({ children }: { children: React.ReactNode }) {
  return (
    <div className="grid h-full min-h-[120px] place-items-center px-6 text-center">
      <p className="max-w-md text-[11px] leading-relaxed text-ink-mute">{children}</p>
    </div>
  );
}
