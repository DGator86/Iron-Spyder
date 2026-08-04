"use client";

import { create } from "zustand";
import { DEFAULT_ACTIVE, LAYER_CATALOGUE, PRESET_BY_ID } from "@/lib/layers";
import type { LayerId, LayerState } from "@/lib/types";

/** Bump when default layers change so stale client stores re-init. */
const LAYER_DEFAULTS_VERSION = 2;

function initialLayers(): Record<LayerId, LayerState> {
  const entries = LAYER_CATALOGUE.map((meta, index) => [
    meta.id,
    {
      id: meta.id,
      enabled: DEFAULT_ACTIVE.includes(meta.id),
      opacity: meta.defaultOpacity,
      order: index,
      settings: {},
    } satisfies LayerState,
  ]);
  return Object.fromEntries(entries) as Record<LayerId, LayerState>;
}

interface LayerStore {
  defaultsVersion: number;
  layers: Record<LayerId, LayerState>;
  /** Master multiplier applied on top of each layer's own opacity. */
  globalOpacity: number;

  toggle: (id: LayerId) => void;
  setEnabled: (id: LayerId, enabled: boolean) => void;
  setOpacity: (id: LayerId, opacity: number) => void;
  setGlobalOpacity: (opacity: number) => void;
  setSetting: (
    id: LayerId,
    key: string,
    value: string | number | boolean,
  ) => void;
  move: (id: LayerId, direction: "up" | "down") => void;
  applyPreset: (presetId: string) => void;
  resetLayer: (id: LayerId) => void;
  resetAll: () => void;

  activeIds: () => LayerId[];
  isActive: (id: LayerId) => boolean;
  /** Layer opacity already multiplied by the global control. */
  effectiveOpacity: (id: LayerId) => number;
}

export const useLayerStore = create<LayerStore>((set, get) => ({
  defaultsVersion: LAYER_DEFAULTS_VERSION,
  layers: initialLayers(),
  globalOpacity: 1,

  toggle: (id) =>
    set((s) => ({
      layers: {
        ...s.layers,
        [id]: { ...s.layers[id], enabled: !s.layers[id].enabled },
      },
    })),

  setEnabled: (id, enabled) =>
    set((s) => ({
      layers: { ...s.layers, [id]: { ...s.layers[id], enabled } },
    })),

  setOpacity: (id, opacity) =>
    set((s) => ({
      layers: {
        ...s.layers,
        [id]: { ...s.layers[id], opacity: Math.min(1, Math.max(0, opacity)) },
      },
    })),

  setGlobalOpacity: (opacity) =>
    set({ globalOpacity: Math.min(1, Math.max(0, opacity)) }),

  setSetting: (id, key, value) =>
    set((s) => ({
      layers: {
        ...s.layers,
        [id]: {
          ...s.layers[id],
          settings: { ...s.layers[id].settings, [key]: value },
        },
      },
    })),

  move: (id, direction) =>
    set((s) => {
      const ordered = Object.values(s.layers).sort((a, b) => a.order - b.order);
      const index = ordered.findIndex((l) => l.id === id);
      const target = direction === "up" ? index - 1 : index + 1;
      if (index < 0 || target < 0 || target >= ordered.length) return s;
      const swapped = [...ordered];
      [swapped[index], swapped[target]] = [swapped[target], swapped[index]];
      const layers = { ...s.layers };
      swapped.forEach((layer, i) => {
        layers[layer.id] = { ...layers[layer.id], order: i };
      });
      return { layers };
    }),

  applyPreset: (presetId) =>
    set((s) => {
      const preset = PRESET_BY_ID[presetId];
      if (!preset) return s;
      const layers = { ...s.layers };
      for (const meta of LAYER_CATALOGUE) {
        layers[meta.id] = {
          ...layers[meta.id],
          enabled: preset.layers.includes(meta.id),
        };
      }
      return { layers };
    }),

  resetLayer: (id) =>
    set((s) => {
      const meta = LAYER_CATALOGUE.find((m) => m.id === id);
      if (!meta) return s;
      return {
        layers: {
          ...s.layers,
          [id]: {
            ...s.layers[id],
            opacity: meta.defaultOpacity,
            enabled: DEFAULT_ACTIVE.includes(id),
            settings: {},
          },
        },
      };
    }),

  resetAll: () => set({ layers: initialLayers(), globalOpacity: 1 }),

  activeIds: () =>
    Object.values(get().layers)
      .filter((l) => l.enabled)
      .sort((a, b) => a.order - b.order)
      .map((l) => l.id),

  isActive: (id) => get().layers[id]?.enabled ?? false,

  effectiveOpacity: (id) => {
    const s = get();
    return (s.layers[id]?.opacity ?? 1) * s.globalOpacity;
  },
}));
