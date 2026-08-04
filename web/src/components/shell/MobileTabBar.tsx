"use client";

import {
  BarChart3,
  Layers,
  LayoutDashboard,
  NotebookTabs,
  Radar,
  Shield,
  Target,
} from "lucide-react";
import { cn } from "@/lib/utils";

export type MobileTab =
  | "radar"
  | "state"
  | "desk"
  | "journal"
  | "optimize"
  | "risk"
  | "layers";

const TABS: Array<{
  id: MobileTab;
  label: string;
  icon: typeof Radar;
}> = [
  { id: "radar", label: "Radar", icon: Radar },
  { id: "state", label: "State", icon: LayoutDashboard },
  { id: "desk", label: "Desk", icon: Target },
  { id: "journal", label: "Journal", icon: NotebookTabs },
  { id: "optimize", label: "Train", icon: BarChart3 },
  { id: "risk", label: "Risk", icon: Shield },
  { id: "layers", label: "Layers", icon: Layers },
];

/** Fixed bottom dock — every desk panel is one tap away on phone. */
export function MobileTabBar({
  active,
  onChange,
  className,
}: {
  active: MobileTab;
  onChange: (tab: MobileTab) => void;
  className?: string;
}) {
  return (
    <nav
      className={cn(
        "flex shrink-0 items-stretch overflow-x-auto border-t border-line bg-panel/95 backdrop-blur-md",
        "pb-[env(safe-area-inset-bottom)]",
        className,
      )}
      aria-label="Mobile desk sections"
    >
      {TABS.map((tab) => {
        const Icon = tab.icon;
        const isActive = tab.id === active;
        return (
          <button
            key={tab.id}
            type="button"
            onClick={() => onChange(tab.id)}
            aria-current={isActive ? "page" : undefined}
            className={cn(
              "flex min-h-[52px] min-w-[4.25rem] flex-1 flex-col items-center justify-center gap-0.5 px-1",
              "text-[9px] font-semibold uppercase tracking-[0.1em] transition-colors",
              isActive ? "text-signal" : "text-ink-mute",
            )}
          >
            <Icon
              className={cn("h-4 w-4", isActive ? "text-signal" : "text-ink-mute")}
            />
            {tab.label}
          </button>
        );
      })}
    </nav>
  );
}
