/** Desk surfaces: trade journal + backtest optimizer. */

export interface JournalLeg {
  expiry?: string;
  strike?: number;
  right?: string;
  quantity?: number;
  entry_price?: number;
}

export interface JournalEntry {
  id: string;
  status: "open" | "closed";
  strategy: string;
  openedAt: string | null;
  closedAt: string | null;
  contracts: number;
  entryPrice: number;
  cost: number;
  currentPrice: number | null;
  currentValue: number | null;
  realizedPnl: number | null;
  unrealizedPnl: number | null;
  result: number | null;
  exitReason: string | null;
  maxLoss: number;
  legs: JournalLeg[];
}

export interface JournalSnapshot {
  source: "live" | "synthetic";
  openCount: number;
  closedCount: number;
  realizedPnl: number;
  unrealizedPnl: number;
  entries: JournalEntry[];
  degradedReason?: string;
}

export interface MetricDelta {
  prior: number | null;
  current: number | null;
  delta: number | null;
}

export interface OptimizeJob {
  id: string;
  status: string;
  reason?: string;
  created_at?: string;
  started_at?: string | null;
  finished_at?: string | null;
  session_count?: number;
  snapshot_limit?: number;
  error?: string | null;
  improved?: boolean;
}

export interface OptimizeRun {
  id: string;
  status: string;
  reason?: string;
  started_at?: string;
  finished_at?: string;
  sessions?: string[];
  snapshot_count?: number;
  metrics?: Record<string, number | string | null>;
  prior_metrics?: Record<string, number | string | null>;
  config?: Record<string, number | string>;
  prior_config?: Record<string, number | string>;
  config_changes?: Record<
    string,
    { prior: number | string | null; current: number | string | null }
  >;
  improved?: boolean;
}

export interface OptimizeSchedule {
  enabled: boolean;
  cadence: "daily" | "weekly" | string;
  hour_utc: number;
  session_count: number;
  snapshot_limit: number;
  next_run_at: string | null;
  last_run_at: string | null;
  last_job_id: string | null;
}

export interface OptimizeSnapshot {
  source: "live" | "synthetic";
  schedule: OptimizeSchedule;
  activeJob: OptimizeJob | null;
  latestRun: OptimizeRun | null;
  priorRun: OptimizeRun | null;
  deltas: Record<string, MetricDelta>;
  activeConfig: Record<string, number | string>;
  data: {
    sessions: number;
    path: string;
    available: boolean;
  };
  runs: OptimizeRun[];
  degradedReason?: string;
}
