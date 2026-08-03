import { NextResponse } from "next/server";
import { fetchLiveSnapshot } from "@/lib/adapter";
import { generateSnapshot } from "@/lib/mock";
import type { Horizon } from "@/lib/types";

/**
 * Backend-for-frontend for the radar.
 *
 * Set SPYDER_API_BASE to the engine's FastAPI origin (through a reverse proxy —
 * the engine binds loopback on the VPS and must never be exposed directly). If
 * it is unset or unreachable, the route serves the synthetic generator so the
 * deployment is inspectable without a live engine. The response always carries
 * `source`, and the UI is required to display it.
 */

export const dynamic = "force-dynamic";
export const revalidate = 0;

const VALID_HORIZONS: Horizon[] = ["5m", "15m", "30m", "60m", "eod", "1d", "expiry"];

export async function GET(request: Request) {
  const url = new URL(request.url);

  const horizonParam = url.searchParams.get("horizon") as Horizon | null;
  const horizon: Horizon =
    horizonParam && VALID_HORIZONS.includes(horizonParam) ? horizonParam : "60m";
  const expiration = url.searchParams.get("expiration") ?? "0DTE";
  const replayOffset = Number(url.searchParams.get("replayOffset") ?? "0") || 0;

  const base = process.env.SPYDER_API_BASE?.replace(/\/$/, "");

  if (base) {
    try {
      const snapshot = await fetchLiveSnapshot(base, horizon);
      return NextResponse.json(snapshot, {
        headers: { "cache-control": "no-store" },
      });
    } catch (error) {
      // Fall through to synthetic, but say so rather than pretending it's live.
      const detail = error instanceof Error ? error.message : "unknown error";
      const snapshot = generateSnapshot({ horizon, expiration, replayOffset });
      return NextResponse.json(
        { ...snapshot, degraded: true, degradedReason: `engine unreachable: ${detail}` },
        { headers: { "cache-control": "no-store" } },
      );
    }
  }

  const snapshot = generateSnapshot({ horizon, expiration, replayOffset });
  return NextResponse.json(snapshot, { headers: { "cache-control": "no-store" } });
}
