"use client";

import { Pause, Play, Radio, SkipBack, SkipForward } from "lucide-react";
import * as React from "react";

import { Button, Segmented } from "@/components/ui/primitives";
import { cn } from "@/lib/utils";
import { PLAYBACK_SPEEDS, useViewStore, type StepMode } from "@/store/viewStore";

/**
 * Radar-loop transport.
 *
 * The scrubber addresses replay offset in minutes back from live. Playback
 * advances that offset on an interval scaled by the speed control, so the field
 * animates exactly as a weather loop does.
 */
export function TimelineControls({ className }: { className?: string }) {
  const {
    playing,
    playbackSpeed,
    stepMode,
    replayOffset,
    live,
    togglePlay,
    setSpeed,
    setStepMode,
    step,
    setReplayOffset,
    returnToLive,
  } = useViewStore();

  // Drive playback from a single interval; speed changes retarget it.
  React.useEffect(() => {
    if (!playing) return;
    const period = Math.max(120, 1000 / playbackSpeed);
    const timer = setInterval(() => {
      const state = useViewStore.getState();
      const next = state.replayOffset + (stepMode === "5m" ? 5 : 1);
      if (next >= 0) {
        useViewStore.setState({ replayOffset: 0, playing: false, live: true });
      } else {
        useViewStore.setState({ replayOffset: next });
      }
    }, period);
    return () => clearInterval(timer);
  }, [playing, playbackSpeed, stepMode]);

  const REPLAY_SPAN = 240; // minutes of scrub range

  return (
    <div
      className={cn(
        "panel flex items-center gap-3 px-3 py-2",
        className,
      )}
    >
      <div className="flex items-center gap-1">
        <Button
          size="icon"
          variant="outline"
          onClick={() => step(-1)}
          aria-label="Step backward"
        >
          <SkipBack className="h-3 w-3" />
        </Button>
        <Button
          size="icon"
          variant={playing ? "solid" : "outline"}
          onClick={togglePlay}
          aria-label={playing ? "Pause" : "Play"}
        >
          {playing ? <Pause className="h-3 w-3" /> : <Play className="h-3 w-3" />}
        </Button>
        <Button
          size="icon"
          variant="outline"
          onClick={() => step(1)}
          disabled={replayOffset >= 0}
          aria-label="Step forward"
        >
          <SkipForward className="h-3 w-3" />
        </Button>
      </div>

      <Button
        size="sm"
        variant={live ? "ghost" : "solid"}
        onClick={returnToLive}
        disabled={live}
        className={cn(live && "text-live")}
      >
        <Radio className={cn("h-3 w-3", live && "animate-pulse-live")} />
        {live ? "Live" : "Return to live"}
      </Button>

      <div className="flex min-w-0 flex-1 items-center gap-2">
        <span className="tnum shrink-0 text-[10px] text-ink-mute">
          −{Math.abs(replayOffset)}m
        </span>
        <input
          type="range"
          min={-REPLAY_SPAN}
          max={0}
          step={1}
          value={replayOffset}
          onChange={(e) => setReplayOffset(Number(e.target.value))}
          aria-label="Replay position"
          className={cn(
            "h-1 min-w-0 flex-1 cursor-pointer appearance-none rounded-full bg-line",
            "[&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:w-3",
            "[&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full",
            "[&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-signal",
            "[&::-webkit-slider-thumb]:bg-void",
            "[&::-moz-range-thumb]:h-3 [&::-moz-range-thumb]:w-3 [&::-moz-range-thumb]:rounded-full",
            "[&::-moz-range-thumb]:border-2 [&::-moz-range-thumb]:border-signal",
            "[&::-moz-range-thumb]:bg-void",
          )}
        />
        <span className="tnum shrink-0 text-[10px] text-ink-mute">now</span>
      </div>

      {/* Step size and speed drop below md: on a phone the scrubber and the
          live button are the controls that matter, and keeping all four in the
          row forced a horizontal scroll on the transport itself. */}
      <Segmented<StepMode>
        size="xs"
        ariaLabel="Step size"
        className="hidden md:inline-flex"
        options={[
          { value: "tick", label: "Tick" },
          { value: "1m", label: "1M" },
          { value: "5m", label: "5M" },
        ]}
        value={stepMode}
        onChange={setStepMode}
      />

      <Segmented<string>
        size="xs"
        ariaLabel="Playback speed"
        className="hidden md:inline-flex"
        options={PLAYBACK_SPEEDS.map((s) => ({ value: String(s), label: `${s}×` }))}
        value={String(playbackSpeed)}
        onChange={(v) => setSpeed(Number(v))}
      />
    </div>
  );
}
