/**
 * Server-side helpers for journal + optimize BFF routes.
 * Browser never talks to the VPS — only to /api/journal and /api/optimize.
 */

import type {
  JournalEntry,
  JournalSnapshot,
  OptimizeSchedule,
  OptimizeSnapshot,
} from "./deskTypes";

export class DeskUnavailable extends Error {}

function authHeaders(): Record<string, string> {
  const token = process.env.SPYDER_API_TOKEN;
  return token ? { authorization: `Bearer ${token}` } : {};
}

function apiBase(): string | null {
  const base = process.env.SPYDER_API_BASE?.replace(/\/$/, "");
  return base || null;
}

async function deskFetch<T>(
  path: string,
  init?: RequestInit,
  timeoutMs = 8000,
): Promise<T> {
  const base = apiBase();
  if (!base) throw new DeskUnavailable("SPYDER_API_BASE is not set");
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${base}${path}`, {
      ...init,
      signal: controller.signal,
      cache: "no-store",
      headers: {
        accept: "application/json",
        ...(init?.body ? { "content-type": "application/json" } : {}),
        ...authHeaders(),
        ...(init?.headers || {}),
      },
    });
    if (res.status === 401 || res.status === 403) {
      throw new DeskUnavailable(
        `${path} -> ${res.status}: proxy rejected the token`,
      );
    }
    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      throw new DeskUnavailable(
        `${path} -> ${res.status}${detail ? `: ${detail.slice(0, 160)}` : ""}`,
      );
    }
    return (await res.json()) as T;
  } finally {
    clearTimeout(timer);
  }
}

function mapEntry(raw: Record<string, unknown>): JournalEntry {
  return {
    id: String(raw.id ?? ""),
    status: raw.status === "open" ? "open" : "closed",
    strategy: String(raw.strategy ?? "Unknown"),
    openedAt: (raw.openedAt as string | null) ?? null,
    closedAt: (raw.closedAt as string | null) ?? null,
    contracts: Number(raw.contracts ?? 0),
    entryPrice: Number(raw.entryPrice ?? 0),
    cost: Number(raw.cost ?? 0),
    currentPrice:
      raw.currentPrice === null || raw.currentPrice === undefined
        ? null
        : Number(raw.currentPrice),
    currentValue:
      raw.currentValue === null || raw.currentValue === undefined
        ? null
        : Number(raw.currentValue),
    realizedPnl:
      raw.realizedPnl === null || raw.realizedPnl === undefined
        ? null
        : Number(raw.realizedPnl),
    unrealizedPnl:
      raw.unrealizedPnl === null || raw.unrealizedPnl === undefined
        ? null
        : Number(raw.unrealizedPnl),
    result:
      raw.result === null || raw.result === undefined
        ? null
        : Number(raw.result),
    exitReason: (raw.exitReason as string | null) ?? null,
    maxLoss: Number(raw.maxLoss ?? 0),
    legs: Array.isArray(raw.legs) ? (raw.legs as JournalEntry["legs"]) : [],
  };
}

export async function fetchJournal(limit = 100): Promise<JournalSnapshot> {
  const raw = await deskFetch<Record<string, unknown>>(`/journal?limit=${limit}`);
  const entries = Array.isArray(raw.entries)
    ? raw.entries.map((e) => mapEntry(e as Record<string, unknown>))
    : [];
  return {
    source: "live",
    openCount: Number(raw.open_count ?? 0),
    closedCount: Number(raw.closed_count ?? 0),
    realizedPnl: Number(raw.realized_pnl ?? 0),
    unrealizedPnl: Number(raw.unrealized_pnl ?? 0),
    entries,
  };
}

function mapOptimize(raw: Record<string, unknown>): OptimizeSnapshot {
  const schedule = (raw.schedule || {}) as Record<string, unknown>;
  const data = (raw.data || {}) as Record<string, unknown>;
  return {
    source: "live",
    schedule: {
      enabled: Boolean(schedule.enabled),
      cadence: String(schedule.cadence ?? "daily"),
      hour_utc: Number(schedule.hour_utc ?? 6),
      session_count: Number(schedule.session_count ?? 3),
      snapshot_limit: Number(schedule.snapshot_limit ?? 120),
      next_run_at: (schedule.next_run_at as string | null) ?? null,
      last_run_at: (schedule.last_run_at as string | null) ?? null,
      last_job_id: (schedule.last_job_id as string | null) ?? null,
    },
    activeJob: (raw.active_job as OptimizeSnapshot["activeJob"]) ?? null,
    latestRun: (raw.latest_run as OptimizeSnapshot["latestRun"]) ?? null,
    priorRun: (raw.prior_run as OptimizeSnapshot["priorRun"]) ?? null,
    deltas: (raw.deltas as OptimizeSnapshot["deltas"]) ?? {},
    activeConfig: (raw.active_config as Record<string, number | string>) ?? {},
    data: {
      sessions: Number(data.sessions ?? 0),
      path: String(data.path ?? ""),
      available: Boolean(data.available),
    },
    runs: Array.isArray(raw.runs)
      ? (raw.runs as OptimizeSnapshot["runs"])
      : [],
  };
}

export async function fetchOptimizeStatus(): Promise<OptimizeSnapshot> {
  const raw = await deskFetch<Record<string, unknown>>("/optimize");
  return mapOptimize(raw);
}

export async function queueOptimizeRun(body?: {
  session_count?: number;
  snapshot_limit?: number;
}): Promise<OptimizeSnapshot> {
  const raw = await deskFetch<{ status: Record<string, unknown> }>(
    "/optimize/run",
    {
      method: "POST",
      body: JSON.stringify(body ?? {}),
    },
    15_000,
  );
  return mapOptimize(raw.status ?? raw);
}

export async function updateOptimizeSchedule(
  body: Partial<OptimizeSchedule>,
): Promise<OptimizeSnapshot> {
  const raw = await deskFetch<{ status: Record<string, unknown> }>(
    "/optimize/schedule",
    {
      method: "POST",
      body: JSON.stringify(body),
    },
  );
  return mapOptimize(raw.status ?? raw);
}

/** Empty but honest placeholders when the engine edge is unreachable. */
export function emptyJournal(reason: string): JournalSnapshot {
  return {
    source: "synthetic",
    openCount: 0,
    closedCount: 0,
    realizedPnl: 0,
    unrealizedPnl: 0,
    entries: [],
    degradedReason: reason,
  };
}

export function emptyOptimize(reason: string): OptimizeSnapshot {
  return {
    source: "synthetic",
    schedule: {
      enabled: false,
      cadence: "daily",
      hour_utc: 6,
      session_count: 3,
      snapshot_limit: 120,
      next_run_at: null,
      last_run_at: null,
      last_job_id: null,
    },
    activeJob: null,
    latestRun: null,
    priorRun: null,
    deltas: {},
    activeConfig: {},
    data: { sessions: 0, path: "", available: false },
    runs: [],
    degradedReason: reason,
  };
}
