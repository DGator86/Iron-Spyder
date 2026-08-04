"use client";

import { Loader2 } from "lucide-react";
import * as React from "react";

import { ForecastCanvas } from "@/components/chart/ForecastCanvas";
import { TimelineControls } from "@/components/chart/TimelineControls";
import { LayerPanel } from "@/components/layers/LayerPanel";
import { PresetChips } from "@/components/layers/PresetChips";
import { BottomPanel } from "@/components/panels/BottomPanel";
import { ExposureStrip } from "@/components/panels/ExposureStrip";
import { InterpretationPanel } from "@/components/panels/InterpretationPanel";
import { RiskControls } from "@/components/panels/RiskControls";
import { LeftNav } from "@/components/shell/LeftNav";
import { MobileTabBar, type MobileTab } from "@/components/shell/MobileTabBar";
import { TopBar } from "@/components/shell/TopBar";
import { Segmented } from "@/components/ui/primitives";
import { useRadar } from "@/hooks/useRadar";
import type { Horizon } from "@/lib/types";
import { cn } from "@/lib/utils";
import { useLayerStore } from "@/store/layerStore";
import { HORIZONS, useViewStore } from "@/store/viewStore";

const NAV_TO_TAB: Record<string, MobileTab> = {
  dashboard: "radar",
  forecast: "radar",
  analytics: "radar",
  "market-state": "state",
  strategies: "desk",
  positions: "desk",
  orders: "desk",
  backtest: "desk",
  "model-health": "desk",
  logs: "desk",
  risk: "risk",
};

export function RadarShell() {
  const { data, isLoading, isError, error } = useRadar();
  const [mobileTab, setMobileTab] = React.useState<MobileTab>("radar");
  const [navActive, setNavActive] = React.useState("dashboard");

  const horizon = useViewStore((s) => s.horizon);
  const setHorizon = useViewStore((s) => s.setHorizon);
  const expiration = useViewStore((s) => s.expiration);
  const setExpiration = useViewStore((s) => s.setExpiration);
  const selectedStrategyId = useViewStore((s) => s.selectedStrategyId);
  const selectStrike = useViewStore((s) => s.selectStrike);
  const selectTime = useViewStore((s) => s.selectTime);

  const layers = useLayerStore((s) => s.layers);
  const globalOpacity = useLayerStore((s) => s.globalOpacity);

  const activeLayers = React.useMemo(
    () =>
      Object.values(layers)
        .filter((l) => l.enabled)
        .sort((a, b) => a.order - b.order)
        .map((l) => l.id),
    [layers],
  );

  const opacityFor = React.useCallback(
    (id: (typeof activeLayers)[number]) =>
      (layers[id]?.opacity ?? 1) * globalOpacity,
    [layers, globalOpacity],
  );

  const selectedStrategy = React.useMemo(
    () => data?.strategies.find((s) => s.strategyId === selectedStrategyId),
    [data?.strategies, selectedStrategyId],
  );

  const onNavSelect = React.useCallback((id: string) => {
    setNavActive(id);
    setMobileTab(NAV_TO_TAB[id] ?? "radar");
  }, []);

  if (isError) {
    return (
      <div className="grid h-full place-items-center p-8">
        <div className="panel max-w-md px-5 py-4 text-center">
          <p className="mb-1 text-sm font-semibold text-bear">
            Radar feed unavailable
          </p>
          <p className="text-[11px] leading-relaxed text-ink-mute">
            {error instanceof Error
              ? error.message
              : "The chart endpoint did not respond."}
          </p>
        </div>
      </div>
    );
  }

  if (isLoading || !data) {
    return (
      <div className="grid h-full place-items-center">
        <div className="flex items-center gap-2 text-ink-mute">
          <Loader2 className="h-4 w-4 animate-spin" />
          <span className="text-[11px] uppercase tracking-wider">
            Acquiring forecast field
          </span>
        </div>
      </div>
    );
  }

  const { chart, interpretation, strategies, system, source } = data;

  const canvas = (
    <div className="panel relative min-h-0 flex-1 overflow-hidden">
      <ForecastCanvas
        payload={chart}
        activeLayers={activeLayers}
        opacityFor={opacityFor}
        selectedStrategy={selectedStrategy}
        onSelectStrike={selectStrike}
        onSelectTime={selectTime}
        className="relative h-full w-full"
      />
      <CanvasLegend className="left-2 top-1.5 flex sm:left-[58px]" />
    </div>
  );

  const toolbar = (
    <div className="flex shrink-0 items-center gap-2 overflow-x-auto border-b border-line bg-deep/80 px-2 py-1.5 sm:gap-3 sm:px-3">
      <Segmented<Horizon>
        ariaLabel="Forecast horizon"
        options={HORIZONS.map((h) => ({ value: h.id, label: h.label }))}
        value={horizon}
        onChange={setHorizon}
      />
      <Segmented<string>
        size="xs"
        ariaLabel="Expiration"
        options={[
          { value: "0DTE", label: "0DTE" },
          { value: "1DTE", label: "1DTE" },
          { value: "WEEKLY", label: "Wk" },
        ]}
        value={expiration}
        onChange={setExpiration}
      />
      <PresetChips className="min-w-0 flex-1" />
    </div>
  );

  return (
    <div className="flex h-full min-h-0">
      <LeftNav
        className="hidden lg:flex"
        active={navActive}
        onSelect={onNavSelect}
        paper={system.mode === "paper"}
        connected={system.connected}
      />

      <main className="flex min-w-0 flex-1 flex-col">
        <TopBar
          system={system}
          source={source}
          degradedReason={data.degradedReason}
        />

        {/* —— Desktop / tablet desk —— */}
        <div className="hidden min-h-0 flex-1 flex-col md:flex">
          {toolbar}

          <div className="flex min-h-0 flex-1 gap-1.5 p-1.5 lg:gap-2 lg:p-2">
            <InterpretationPanel
              data={interpretation}
              className="hidden w-[200px] shrink-0 md:flex xl:w-[240px]"
            />

            <div className="flex min-w-0 flex-1 flex-col gap-1.5 lg:gap-2">
              {canvas}
              <TimelineControls className="shrink-0" />
              <ExposureStrip system={system} />
              <BottomPanel
                chart={chart}
                strategies={strategies}
                interpretation={interpretation}
                system={system}
                className="h-[200px] shrink-0 lg:h-[220px]"
              />
            </div>

            <div className="hidden w-[240px] shrink-0 flex-col gap-1.5 lg:flex xl:w-[280px]">
              <LayerPanel className="min-h-0 flex-[1.1]" />
              <RiskControls system={system} className="min-h-0 flex-1" />
            </div>
          </div>
        </div>

        {/* —— Mobile: one job per tab, nothing dropped —— */}
        <div className="flex min-h-0 flex-1 flex-col md:hidden">
          {mobileTab === "radar" ? (
            <>
              {toolbar}
              <div className="flex min-h-0 flex-1 flex-col gap-1.5 p-1.5">
                {canvas}
                <TimelineControls className="shrink-0" />
                <ExposureStrip system={system} />
              </div>
            </>
          ) : null}

          {mobileTab === "state" ? (
            <InterpretationPanel
              data={interpretation}
              className="min-h-0 flex-1 rounded-none border-0"
            />
          ) : null}

          {mobileTab === "desk" ? (
            <BottomPanel
              chart={chart}
              strategies={strategies}
              interpretation={interpretation}
              system={system}
              className="min-h-0 flex-1 rounded-none border-0"
            />
          ) : null}

          {mobileTab === "risk" ? (
            <RiskControls
              system={system}
              className="min-h-0 flex-1 rounded-none border-0"
            />
          ) : null}

          {mobileTab === "layers" ? (
            <LayerPanel className="min-h-0 flex-1 rounded-none border-0" />
          ) : null}

          <MobileTabBar active={mobileTab} onChange={setMobileTab} />
        </div>
      </main>
    </div>
  );
}

function CanvasLegend({ className }: { className?: string }) {
  const items = [
    { label: "SPY", color: "#F8FAFC", dash: false },
    { label: "VWAP", color: "#FACC15", dash: false },
    { label: "−GEX", color: "#3B82F6", dash: false },
    { label: "+GEX", color: "#EF4444", dash: false },
    { label: "GEX Flip", color: "#67E8F9", dash: true },
    { label: "Call Wall", color: "#34D399", dash: true },
    { label: "Put Wall", color: "#F87171", dash: true },
  ];

  return (
    <div
      className={cn(
        "pointer-events-none absolute flex max-w-[calc(100%-1rem)] flex-wrap items-center gap-x-2.5 gap-y-1",
        "border border-line/60 bg-void/75 px-2 py-1 backdrop-blur-sm",
        className,
      )}
    >
      {items.map((item) => (
        <span key={item.label} className="flex items-center gap-1.5">
          <svg width="14" height="6" aria-hidden>
            <line
              x1="0"
              y1="3"
              x2="14"
              y2="3"
              stroke={item.color}
              strokeWidth="1.6"
              strokeDasharray={item.dash ? "3 2" : undefined}
            />
          </svg>
          <span className="text-[9px] uppercase tracking-wider text-ink-mute">
            {item.label}
          </span>
        </span>
      ))}
    </div>
  );
}
