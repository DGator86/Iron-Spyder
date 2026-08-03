"use client";

import { useQuery } from "@tanstack/react-query";
import { useViewStore } from "@/store/viewStore";
import type { RadarSnapshot } from "@/lib/types";

export interface RadarResponse extends RadarSnapshot {
  degraded?: boolean;
  degradedReason?: string;
}

/**
 * Poll the BFF for a full snapshot.
 *
 * Polling rather than SSE: the engine's decision cycle runs on a multi-second
 * interval, so a socket would spend most of its life idle while adding a
 * reconnect path to get wrong. When the cycle time drops, swap the queryFn for
 * an EventSource — nothing else in the tree needs to change.
 */
export function useRadar() {
  const horizon = useViewStore((s) => s.horizon);
  const expiration = useViewStore((s) => s.expiration);
  const replayOffset = useViewStore((s) => s.replayOffset);
  const live = useViewStore((s) => s.live);

  return useQuery<RadarResponse>({
    queryKey: ["radar", horizon, expiration, replayOffset],
    queryFn: async ({ signal }) => {
      const params = new URLSearchParams({
        horizon,
        expiration,
        replayOffset: String(replayOffset),
      });
      const res = await fetch(`/api/chart?${params}`, {
        signal,
        cache: "no-store",
      });
      if (!res.ok) throw new Error(`radar request failed: ${res.status}`);
      return (await res.json()) as RadarResponse;
    },
    // Only chase the tape while live; a paused replay frame must stay put.
    refetchInterval: live ? 5_000 : false,
    placeholderData: (previous) => previous,
  });
}
