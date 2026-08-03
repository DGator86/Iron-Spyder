"use client";

import { create } from "zustand";
import type { Horizon } from "@/lib/types";

export type TimeWindow =
  | "prev30"
  | "session"
  | "next15"
  | "next30"
  | "next60"
  | "eod"
  | "next-session"
  | "expiry";

export const HORIZONS: Array<{ id: Horizon; label: string; minutes: number }> =
  [
    { id: "5m", label: "5M", minutes: 5 },
    { id: "15m", label: "15M", minutes: 15 },
    { id: "30m", label: "30M", minutes: 30 },
    { id: "60m", label: "60M", minutes: 60 },
    { id: "eod", label: "EOD", minutes: 390 },
    { id: "1d", label: "1D", minutes: 780 },
    { id: "expiry", label: "EXPIRY", minutes: 390 },
  ];

export const PLAYBACK_SPEEDS = [1, 2, 5, 10] as const;
export type StepMode = "tick" | "1m" | "5m";

interface ViewStore {
  horizon: Horizon;
  expiration: string;
  window: TimeWindow;
  selectedStrike?: number;
  selectedTime?: string;
  selectedStrategyId?: string;
  preset: string;
  live: boolean;
  playing: boolean;
  playbackSpeed: number;
  stepMode: StepMode;
  /** Offset in minutes from live; 0 = now, negative = replaying the past. */
  replayOffset: number;
  replayDate: string | null;

  setHorizon: (h: Horizon) => void;
  setExpiration: (e: string) => void;
  setWindow: (w: TimeWindow) => void;
  selectStrike: (strike?: number) => void;
  selectTime: (time?: string) => void;
  selectStrategy: (id?: string) => void;
  setPreset: (id: string) => void;
  setLive: (live: boolean) => void;
  togglePlay: () => void;
  setSpeed: (speed: number) => void;
  setStepMode: (mode: StepMode) => void;
  step: (direction: -1 | 1) => void;
  setReplayOffset: (offset: number) => void;
  setReplayDate: (date: string | null) => void;
  returnToLive: () => void;
}

const STEP_MINUTES: Record<StepMode, number> = { tick: 1, "1m": 1, "5m": 5 };

export const useViewStore = create<ViewStore>((set) => ({
  horizon: "60m",
  expiration: "0DTE",
  window: "session",
  preset: "full-model",
  live: true,
  playing: false,
  playbackSpeed: 1,
  stepMode: "1m",
  replayOffset: 0,
  replayDate: null,

  setHorizon: (horizon) => set({ horizon }),
  setExpiration: (expiration) => set({ expiration }),
  setWindow: (window) => set({ window }),
  selectStrike: (selectedStrike) => set({ selectedStrike }),
  selectTime: (selectedTime) => set({ selectedTime }),
  selectStrategy: (selectedStrategyId) => set({ selectedStrategyId }),
  setPreset: (preset) => set({ preset }),
  setLive: (live) =>
    set({ live, playing: false, ...(live ? { replayOffset: 0 } : {}) }),
  // Starting playback leaves live mode; stopping it does not silently rejoin.
  togglePlay: () =>
    set((s) => ({ playing: !s.playing, live: s.playing ? s.live : false })),
  setSpeed: (playbackSpeed) => set({ playbackSpeed }),
  setStepMode: (stepMode) => set({ stepMode }),

  step: (direction) =>
    set((s) => {
      const delta = STEP_MINUTES[s.stepMode] * direction;
      const next = Math.min(0, s.replayOffset + delta);
      return { replayOffset: next, live: next === 0 };
    }),

  setReplayOffset: (replayOffset) =>
    set({ replayOffset: Math.min(0, replayOffset), live: replayOffset === 0 }),

  setReplayDate: (replayDate) => set({ replayDate, live: replayDate === null }),

  returnToLive: () =>
    set({ replayOffset: 0, live: true, playing: false, replayDate: null }),
}));
