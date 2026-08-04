"use client";

import {
  Activity,
  BarChart3,
  ChevronsLeft,
  CircleGauge,
  FileText,
  Heart,
  LayoutDashboard,
  LineChart,
  Radar,
  Shield,
  Target,
  Wallet,
} from "lucide-react";
import * as React from "react";
import { cn } from "@/lib/utils";

const NAV = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "market-state", label: "Market State", icon: CircleGauge },
  { id: "analytics", label: "Analytics", icon: Activity },
  { id: "forecast", label: "Forecast", icon: Radar },
  { id: "strategies", label: "Strategies", icon: Target },
  { id: "positions", label: "Journal", icon: Wallet },
  { id: "orders", label: "Orders", icon: LineChart },
  { id: "risk", label: "Risk", icon: Shield },
  { id: "backtest", label: "Optimizer", icon: BarChart3 },
  { id: "model-health", label: "Model Health", icon: Heart },
  { id: "logs", label: "Logs", icon: FileText },
] as const;

export function LeftNav({
  active = "dashboard",
  onSelect,
  className,
  paper = true,
  connected = true,
}: {
  active?: string;
  onSelect?: (id: string) => void;
  className?: string;
  paper?: boolean;
  connected?: boolean;
}) {
  const [collapsed, setCollapsed] = React.useState(false);

  return (
    <nav
      className={cn(
        "flex flex-col border-r border-line bg-panel/60 transition-[width] duration-200",
        collapsed ? "w-12" : "w-40",
        className,
      )}
      aria-label="Primary"
    >
      <ul className="flex-1 space-y-0.5 p-1.5">
        {NAV.map((item) => {
          const Icon = item.icon;
          const isActive = item.id === active;
          return (
            <li key={item.id}>
              <button
                onClick={() => onSelect?.(item.id)}
                title={collapsed ? item.label : undefined}
                aria-current={isActive ? "page" : undefined}
                className={cn(
                  "flex w-full items-center gap-2.5 rounded-md px-2 py-2 transition-colors duration-100",
                  "text-[11px] font-medium",
                  isActive
                    ? "bg-signal/12 text-signal shadow-[inset_2px_0_0_0_#22D3EE]"
                    : "text-ink-mute hover:bg-raised hover:text-ink-dim",
                )}
              >
                <Icon className="h-4 w-4 shrink-0" />
                {!collapsed ? (
                  <span className="truncate uppercase tracking-wider">
                    {item.label}
                  </span>
                ) : null}
              </button>
            </li>
          );
        })}
      </ul>

      <div className="space-y-1 border-t border-line p-1.5">
        {!collapsed ? (
          <div className="space-y-1 px-1.5 py-1 text-[9px] uppercase tracking-wider text-ink-mute">
            <div className="flex items-center gap-1.5">
              <span
                className={cn(
                  "h-1.5 w-1.5 rounded-full",
                  paper ? "bg-live" : "bg-warn",
                )}
              />
              {paper ? "Paper Mode" : "Live Mode"}
            </div>
            <div className="flex items-center gap-1.5">
              <span
                className={cn(
                  "h-1.5 w-1.5 rounded-full",
                  connected ? "bg-signal" : "bg-dead",
                )}
              />
              {connected ? "Connected" : "Disconnected"}
            </div>
          </div>
        ) : null}
        <button
          onClick={() => setCollapsed((c) => !c)}
          className={cn(
            "flex w-full items-center justify-center rounded-md py-1.5",
            "text-ink-mute transition-colors hover:bg-raised hover:text-ink-dim",
          )}
          aria-label={collapsed ? "Expand navigation" : "Collapse navigation"}
        >
          <ChevronsLeft
            className={cn(
              "h-4 w-4 transition-transform duration-200",
              collapsed && "rotate-180",
            )}
          />
        </button>
      </div>
    </nav>
  );
}
