"use client";

import { ChevronDown, ChevronUp, RotateCcw, Settings2 } from "lucide-react";
import * as React from "react";

import { Button } from "@/components/ui/primitives";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { InfoDot } from "@/components/ui/tooltip";
import { LAYER_CATALOGUE, LAYER_GROUPS } from "@/lib/layers";
import type { LayerGroup, LayerMeta } from "@/lib/types";
import { cn } from "@/lib/utils";
import { useLayerStore } from "@/store/layerStore";

export function LayerPanel({ className }: { className?: string }) {
  const layers = useLayerStore((s) => s.layers);
  const globalOpacity = useLayerStore((s) => s.globalOpacity);
  const toggle = useLayerStore((s) => s.toggle);
  const setOpacity = useLayerStore((s) => s.setOpacity);
  const setGlobalOpacity = useLayerStore((s) => s.setGlobalOpacity);
  const move = useLayerStore((s) => s.move);
  const resetAll = useLayerStore((s) => s.resetAll);

  const [expanded, setExpanded] = React.useState<string | null>(null);
  const [collapsedGroups, setCollapsedGroups] = React.useState<Set<LayerGroup>>(
    new Set(),
  );

  const grouped = React.useMemo(() => {
    const map = new Map<LayerGroup, LayerMeta[]>();
    for (const group of LAYER_GROUPS) map.set(group, []);
    // Order within a group follows the store, so arrow reordering is visible.
    const ordered = [...LAYER_CATALOGUE].sort(
      (a, b) => (layers[a.id]?.order ?? 0) - (layers[b.id]?.order ?? 0),
    );
    for (const meta of ordered) map.get(meta.group)?.push(meta);
    return map;
  }, [layers]);

  const activeCount = Object.values(layers).filter((l) => l.enabled).length;

  return (
    <aside className={cn("panel flex min-h-0 flex-col", className)}>
      <div className="panel-header">
        <div className="flex items-center gap-2">
          <span className="panel-title">Layers</span>
          <span className="tnum rounded bg-raised px-1.5 py-px text-[10px] text-ink-mute">
            {activeCount}
          </span>
        </div>
        <Button
          size="xs"
          variant="ghost"
          onClick={resetAll}
          title="Reset all layers"
        >
          <RotateCcw className="h-3 w-3" />
          Reset
        </Button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-2 py-1.5">
        {LAYER_GROUPS.map((group) => {
          const metas = grouped.get(group) ?? [];
          if (metas.length === 0) return null;
          const collapsed = collapsedGroups.has(group);
          const groupActive = metas.filter((m) => layers[m.id]?.enabled).length;

          return (
            <section key={group} className="mb-1.5">
              <button
                onClick={() =>
                  setCollapsedGroups((prev) => {
                    const next = new Set(prev);
                    if (next.has(group)) next.delete(group);
                    else next.add(group);
                    return next;
                  })
                }
                className={cn(
                  "flex w-full items-center justify-between rounded px-1 py-1",
                  "text-[10px] font-semibold uppercase tracking-[0.12em] text-ink-mute",
                  "transition-colors hover:bg-raised hover:text-ink-dim",
                )}
              >
                <span className="flex items-center gap-1.5">
                  <ChevronDown
                    className={cn(
                      "h-3 w-3 transition-transform duration-150",
                      collapsed && "-rotate-90",
                    )}
                  />
                  {group}
                </span>
                {groupActive > 0 ? (
                  <span className="tnum text-signal">{groupActive}</span>
                ) : null}
              </button>

              {!collapsed ? (
                <div className="mt-0.5 space-y-px">
                  {metas.map((meta) => {
                    const state = layers[meta.id];
                    const isExpanded = expanded === meta.id;
                    return (
                      <div key={meta.id} className="rounded">
                        <div
                          className={cn(
                            "group flex items-center gap-2 rounded px-1.5 py-1 transition-colors",
                            state.enabled
                              ? "bg-raised/50"
                              : "hover:bg-raised/30",
                          )}
                        >
                          <Switch
                            checked={state.enabled}
                            onCheckedChange={() => toggle(meta.id)}
                            accent={meta.encoding.color}
                            aria-label={meta.label}
                          />

                          <LayerGlyph meta={meta} dimmed={!state.enabled} />

                          <span
                            className={cn(
                              "min-w-0 flex-1 truncate text-[11px] transition-colors",
                              state.enabled ? "text-ink" : "text-ink-mute",
                            )}
                            title={meta.label}
                          >
                            {meta.label}
                          </span>

                          {meta.heavy && state.enabled ? (
                            <span
                              className="text-[9px] text-warn/70"
                              title="Render-heavy layer"
                            >
                              ●
                            </span>
                          ) : null}

                          <InfoDot text={meta.description} />

                          <button
                            onClick={() =>
                              setExpanded(isExpanded ? null : meta.id)
                            }
                            className={cn(
                              "opacity-0 transition-opacity group-hover:opacity-100",
                              isExpanded && "opacity-100",
                              "text-ink-mute hover:text-signal",
                            )}
                            aria-label={`${meta.label} settings`}
                          >
                            <Settings2 className="h-3 w-3" />
                          </button>
                        </div>

                        {isExpanded ? (
                          <div className="mb-1 ml-2 mr-1 rounded border border-line bg-deep/60 px-2 py-2">
                            <div className="mb-1 flex items-center justify-between">
                              <span className="text-[10px] uppercase tracking-wider text-ink-mute">
                                Opacity
                              </span>
                              <span className="tnum text-[10px] text-ink-dim">
                                {Math.round(state.opacity * 100)}%
                              </span>
                            </div>
                            <Slider
                              value={[state.opacity * 100]}
                              onValueChange={([v]) =>
                                setOpacity(meta.id, v / 100)
                              }
                              min={0}
                              max={100}
                              step={1}
                              accent={meta.encoding.color}
                              aria-label={`${meta.label} opacity`}
                            />
                            <div className="mt-2 flex items-center justify-between">
                              <span className="text-[10px] uppercase tracking-wider text-ink-mute">
                                Draw order
                              </span>
                              <div className="flex gap-1">
                                <Button
                                  size="icon"
                                  variant="outline"
                                  onClick={() => move(meta.id, "up")}
                                  aria-label="Move layer up"
                                >
                                  <ChevronUp className="h-3 w-3" />
                                </Button>
                                <Button
                                  size="icon"
                                  variant="outline"
                                  onClick={() => move(meta.id, "down")}
                                  aria-label="Move layer down"
                                >
                                  <ChevronDown className="h-3 w-3" />
                                </Button>
                              </div>
                            </div>
                          </div>
                        ) : null}
                      </div>
                    );
                  })}
                </div>
              ) : null}
            </section>
          );
        })}
      </div>

      <div className="border-t border-line px-3 py-2.5">
        <div className="mb-1.5 flex items-center justify-between">
          <span className="text-micro font-semibold uppercase tracking-[0.14em] text-ink-dim">
            Layer Opacity
          </span>
          <span className="tnum text-[11px] text-ink">
            {Math.round(globalOpacity * 100)}%
          </span>
        </div>
        <Slider
          value={[globalOpacity * 100]}
          onValueChange={([v]) => setGlobalOpacity(v / 100)}
          min={0}
          max={100}
          step={1}
          aria-label="Global layer opacity"
        />
        <div className="mt-1 flex justify-between text-[9px] text-ink-mute">
          <span>0%</span>
          <span>50%</span>
          <span>100%</span>
        </div>
      </div>
    </aside>
  );
}

/**
 * Legend glyph. Encoding is carried by shape as well as colour — a dashed rule,
 * a filled field, a contour, an arrow — so the panel stays readable without
 * relying on hue discrimination.
 */
function LayerGlyph({ meta, dimmed }: { meta: LayerMeta; dimmed: boolean }) {
  const color = meta.encoding.color;
  const opacity = dimmed ? 0.3 : 1;

  return (
    <span
      className="grid h-3.5 w-3.5 shrink-0 place-items-center"
      style={{ opacity }}
    >
      <svg viewBox="0 0 14 14" className="h-3.5 w-3.5" aria-hidden>
        {meta.encoding.kind === "field" ? (
          <>
            <defs>
              <linearGradient id={`g-${meta.id}`} x1="0" x2="1">
                <stop offset="0%" stopColor={color} stopOpacity="0.15" />
                <stop offset="100%" stopColor={color} stopOpacity="0.9" />
              </linearGradient>
            </defs>
            <rect
              x="1"
              y="3"
              width="12"
              height="8"
              rx="1.5"
              fill={`url(#g-${meta.id})`}
            />
          </>
        ) : null}

        {meta.encoding.kind === "band" ? (
          <>
            <rect
              x="1"
              y="4"
              width="12"
              height="6"
              rx="1"
              fill={color}
              fillOpacity="0.25"
            />
            <line x1="1" y1="4" x2="13" y2="4" stroke={color} strokeWidth="1" />
            <line
              x1="1"
              y1="10"
              x2="13"
              y2="10"
              stroke={color}
              strokeWidth="1"
            />
          </>
        ) : null}

        {meta.encoding.kind === "line" ? (
          <line
            x1="1"
            y1="7"
            x2="13"
            y2="7"
            stroke={color}
            strokeWidth="1.6"
            strokeDasharray={meta.encoding.dash ? "3 2" : undefined}
          />
        ) : null}

        {meta.encoding.kind === "marker" ? (
          <line
            x1="1"
            y1="7"
            x2="13"
            y2="7"
            stroke={color}
            strokeWidth="1.4"
            strokeDasharray="3 2"
          />
        ) : null}

        {meta.encoding.kind === "contour" ? (
          <>
            <path
              d="M1 10 Q4 4 7 7 T13 5"
              fill="none"
              stroke={color}
              strokeWidth="1.2"
            />
            <path
              d="M1 12 Q4 7 7 10 T13 8"
              fill="none"
              stroke={color}
              strokeWidth="0.8"
              opacity="0.6"
            />
          </>
        ) : null}

        {meta.encoding.kind === "arrows" ? (
          <>
            <line
              x1="1"
              y1="7"
              x2="10"
              y2="7"
              stroke={color}
              strokeWidth="1.2"
            />
            <polygon points="13,7 9,5 9,9" fill={color} />
          </>
        ) : null}

        {meta.encoding.kind === "hatch" ? (
          <>
            <rect
              x="1"
              y="3"
              width="12"
              height="8"
              rx="1"
              fill="none"
              stroke={color}
              strokeWidth="1"
            />
            <line
              x1="2"
              y1="10"
              x2="6"
              y2="4"
              stroke={color}
              strokeWidth="0.9"
            />
            <line
              x1="6"
              y1="10"
              x2="10"
              y2="4"
              stroke={color}
              strokeWidth="0.9"
            />
          </>
        ) : null}
      </svg>
    </span>
  );
}
