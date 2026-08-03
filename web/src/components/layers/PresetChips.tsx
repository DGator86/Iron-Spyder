"use client";

import { Layers } from "lucide-react";
import { PRESETS } from "@/lib/layers";
import { cn } from "@/lib/utils";
import { useLayerStore } from "@/store/layerStore";
import { useViewStore } from "@/store/viewStore";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

/**
 * Weather-map presets. Horizontally scrollable so the row never wraps and never
 * pushes the canvas down — on mobile this is the primary layer control.
 */
export function PresetChips({ className }: { className?: string }) {
  const preset = useViewStore((s) => s.preset);
  const setPreset = useViewStore((s) => s.setPreset);
  const applyPreset = useLayerStore((s) => s.applyPreset);

  return (
    <div className={cn("flex items-center gap-1.5 overflow-x-auto", className)}>
      <Layers className="h-3 w-3 shrink-0 text-ink-mute" aria-hidden />
      {PRESETS.map((p) => {
        const active = p.id === preset;
        return (
          <Tooltip key={p.id}>
            <TooltipTrigger asChild>
              <button
                onClick={() => {
                  setPreset(p.id);
                  applyPreset(p.id);
                }}
                className={cn(
                  "shrink-0 whitespace-nowrap rounded-full border px-2.5 py-1",
                  "text-[10px] font-semibold uppercase tracking-wider transition-colors duration-100",
                  active
                    ? "border-signal/60 bg-signal/15 text-signal"
                    : "border-line text-ink-mute hover:border-line-bright hover:text-ink-dim",
                )}
              >
                {p.label}
              </button>
            </TooltipTrigger>
            <TooltipContent>{p.hint}</TooltipContent>
          </Tooltip>
        );
      })}
    </div>
  );
}
