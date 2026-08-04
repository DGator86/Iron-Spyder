"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { OptimizeSnapshot } from "@/lib/deskTypes";

async function loadOptimize(): Promise<OptimizeSnapshot> {
  const res = await fetch("/api/optimize", { cache: "no-store" });
  if (!res.ok) throw new Error(`optimize -> ${res.status}`);
  return (await res.json()) as OptimizeSnapshot;
}

export function useOptimize() {
  const client = useQueryClient();
  const query = useQuery({
    queryKey: ["optimize"],
    queryFn: loadOptimize,
    refetchInterval: (q) => {
      const job = q.state.data?.activeJob;
      if (job && (job.status === "queued" || job.status === "running")) {
        // Poll faster while a bar is live so percent/phase stay fresh.
        return 2_000;
      }
      return 20_000;
    },
  });

  const run = useMutation({
    mutationFn: async (body?: {
      session_count?: number;
      snapshot_limit?: number;
    }) => {
      const res = await fetch("/api/optimize", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ action: "run", ...body }),
      });
      if (!res.ok) throw new Error(`optimize run -> ${res.status}`);
      return (await res.json()) as OptimizeSnapshot;
    },
    onSuccess: (data) => {
      client.setQueryData(["optimize"], data);
    },
  });

  const schedule = useMutation({
    mutationFn: async (body: Record<string, unknown>) => {
      const res = await fetch("/api/optimize", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ action: "schedule", ...body }),
      });
      if (!res.ok) throw new Error(`optimize schedule -> ${res.status}`);
      return (await res.json()) as OptimizeSnapshot;
    },
    onSuccess: (data) => {
      client.setQueryData(["optimize"], data);
    },
  });

  return { ...query, run, schedule };
}
