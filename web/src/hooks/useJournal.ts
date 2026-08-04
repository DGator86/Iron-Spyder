"use client";

import { useQuery } from "@tanstack/react-query";
import type { JournalSnapshot } from "@/lib/deskTypes";

async function loadJournal(): Promise<JournalSnapshot> {
  const res = await fetch("/api/journal?limit=100", { cache: "no-store" });
  if (!res.ok) throw new Error(`journal -> ${res.status}`);
  return (await res.json()) as JournalSnapshot;
}

export function useJournal() {
  return useQuery({
    queryKey: ["journal"],
    queryFn: loadJournal,
    refetchInterval: 15_000,
  });
}
