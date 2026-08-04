import { NextResponse } from "next/server";
import {
  emptyOptimize,
  fetchOptimizeStatus,
  queueOptimizeRun,
  updateOptimizeSchedule,
} from "@/lib/desk";

export const dynamic = "force-dynamic";
export const revalidate = 0;

export async function GET() {
  if (!process.env.SPYDER_API_BASE) {
    return NextResponse.json(emptyOptimize("SPYDER_API_BASE unset"), {
      headers: { "cache-control": "no-store" },
    });
  }
  try {
    const snapshot = await fetchOptimizeStatus();
    return NextResponse.json(snapshot, {
      headers: { "cache-control": "no-store" },
    });
  } catch (error) {
    const detail = error instanceof Error ? error.message : "unknown error";
    return NextResponse.json(emptyOptimize(`engine unreachable: ${detail}`), {
      headers: { "cache-control": "no-store" },
    });
  }
}

export async function POST(request: Request) {
  if (!process.env.SPYDER_API_BASE) {
    return NextResponse.json(emptyOptimize("SPYDER_API_BASE unset"), {
      status: 503,
      headers: { "cache-control": "no-store" },
    });
  }

  let body: Record<string, unknown> = {};
  try {
    body = (await request.json()) as Record<string, unknown>;
  } catch {
    body = {};
  }

  const action = String(body.action ?? "run");

  try {
    if (action === "schedule") {
      const snapshot = await updateOptimizeSchedule({
        enabled: body.enabled as boolean | undefined,
        cadence: body.cadence as string | undefined,
        hour_utc: body.hour_utc as number | undefined,
        session_count: body.session_count as number | undefined,
        snapshot_limit: body.snapshot_limit as number | undefined,
      });
      return NextResponse.json(snapshot, {
        headers: { "cache-control": "no-store" },
      });
    }

    const snapshot = await queueOptimizeRun({
      session_count:
        typeof body.session_count === "number" ? body.session_count : undefined,
      snapshot_limit:
        typeof body.snapshot_limit === "number"
          ? body.snapshot_limit
          : undefined,
    });
    return NextResponse.json(snapshot, {
      headers: { "cache-control": "no-store" },
    });
  } catch (error) {
    const detail = error instanceof Error ? error.message : "unknown error";
    return NextResponse.json(emptyOptimize(`engine unreachable: ${detail}`), {
      status: 502,
      headers: { "cache-control": "no-store" },
    });
  }
}
