"use client";

import { LayoutPanelTop, Layers as LayersIcon, Loader2 } from "lucide-react";
import * as React from "react";

import { ForecastCanvas } from "@/components/chart/ForecastCanvas";
import { TimelineControls } from "@/components/chart/TimelineControls";
import { LayerPanel } from "@/components/layers/LayerPanel";
import { PresetChips } from "@/components/layers/PresetChips";
import { BottomPanel } from "@/components/panels/BottomPanel";
import { InterpretationPanel } from "@/components/panels/InterpretationPanel";
import { LeftNav } from "@/components/shell/LeftNav";
import { TopBar } from "@/components/shell/TopBar";
import { Button, Segmented } from "@/components/ui/primitives";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { useRadar } from "@/hooks/useRadar";
import type { Horizon } from "@/lib/types";
import { cn } from "@/lib/utils";
import { useLayerStore } from "@/store/layerStore";
import { HORIZONS, useViewStore } from "@/store/viewStore";

export function RadarShell() {
  const { data, isLoading, isError, error } = useRadar();

  const horizon = useViewStore((s) => s.horizon);
  const setHorizon = useViewStore((s) => s.setHorizon);
  const expiration = useViewStore((s) => s.expiration);
  const setExpiration = useViewStore((s) => s.setExpiration);
  const selectedStrategyId = useViewStore((s) => s.selectedStrategyId);
  const selectStrike = useViewStore((s) => s.selectStrike);
  const selectTime = useViewStore((s) => s.selectTime);

  const layers = useLayerStore((s) => s.layers);
  const globalOpacity = useLayerStore((s) => s.globalOpacity);

  // Subscribing to the maps rather than calling the store's selectors keeps the
  // canvas memo keyed on real state — a getter identity never changes and the
  // chart would not repaint on toggle.
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

  return (
    <div className="flex h-full min-h-0">
      <LeftNav className="hidden lg:flex" />

      <main className="flex min-w-0 flex-1 flex-col">
        <TopBar
          system={system}
          source={source}
          degradedReason={data.degradedReason}
        />

        {/* Horizon + presets */}
        <div className="flex shrink-0 items-center gap-3 overflow-x-auto border-b border-line bg-panel/40 px-3 py-1.5">
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

          {/* Layers move into a drawer below xl, where the rail would crowd the canvas. */}
          <Sheet>
            <SheetTrigger asChild>
              <Button
                size="sm"
                variant="outline"
                className="shrink-0 xl:hidden"
              >
                <LayersIcon className="h-3 w-3" />
                Layers
              </Button>
            </SheetTrigger>
            <SheetContent title="Layers" side="right" className="p-0">
              <LayerPanel className="h-full rounded-none border-0 shadow-none" />
            </SheetContent>
          </Sheet>
        </div>

        <div className="flex min-h-0 flex-1 gap-2 p-2">
          {/* Centre column: canvas dominates, analytics sit beneath it. */}
          <div className="flex min-w-0 flex-1 flex-col gap-2">
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
              <CanvasLegend />
            </div>

            <TimelineControls className="shrink-0" />

            {/* Secondary analytics: present on desktop, a sheet on small screens. */}
            <BottomPanel
              chart={chart}
              strategies={strategies}
              interpretation={interpretation}
              system={system}
              className="hidden h-[210px] shrink-0 md:flex"
            />

            <Sheet>
              <SheetTrigger asChild>
                <Button
                  size="sm"
                  variant="outline"
                  className="shrink-0 md:hidden"
                >
                  <LayoutPanelTop className="h-3 w-3" />
                  Analysis
                </Button>
              </SheetTrigger>
              <SheetContent title="Analysis" side="bottom" className="p-0">
                <BottomPanel
                  chart={chart}
                  strategies={strategies}
                  interpretation={interpretation}
                  system={system}
                  className="h-[60vh] rounded-none border-0 shadow-none"
                />
              </SheetContent>
            </Sheet>
          </div>

          {/* Right rail */}
          <div className="hidden w-[290px] shrink-0 flex-col gap-2 xl:flex">
            <InterpretationPanel
              data={interpretation}
              className="max-h-[46%] shrink-0"
            />
            <LayerPanel className="min-h-0 flex-1" />
          </div>
        </div>
      </main>
    </div>
  );
}

/**
 * Canvas legend. Encodings are shown with the same dash pattern and weight the
 * chart uses, so it is a key rather than a colour swatch list.
 */
function CanvasLegend() {
  const items = [
    { label: "SPY", color: "#E6EDF7", dash: false },
    { label: "VWAP", color: "#FBBF24", dash: false },
    { label: "Median", color: "#22D3EE", dash: false },
    { label: "GEX Flip", color: "#67E8F9", dash: true },
    { label: "Call Wall", color: "#34D399", dash: true },
    { label: "Put Wall", color: "#F87171", dash: true },
  ];

  return (
    <div
      className={cn(
        // Top-left of the plot rect: the bottom strip belongs to the time axis,
        // and the legend was landing on top of the tick labels. Hidden on phones,
        // where it wrapped to two rows and ate a fifth of the canvas.
        "pointer-events-none absolute left-[58px] top-1.5 hidden flex-wrap items-center gap-x-3 gap-y-1 sm:flex",
        "rounded border border-line/60 bg-void/70 px-2 py-1 backdrop-blur-sm",
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
