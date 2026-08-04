import { NextResponse } from "next/server";
import { emptyJournal, fetchJournal } from "@/lib/desk";

export const dynamic = "force-dynamic";
export const revalidate = 0;

export async function GET(request: Request) {
  const url = new URL(request.url);
  const limit = Number(url.searchParams.get("limit") ?? "100") || 100;

  if (!process.env.SPYDER_API_BASE) {
    return NextResponse.json(emptyJournal("SPYDER_API_BASE unset"), {
      headers: { "cache-control": "no-store" },
    });
  }

  try {
    const snapshot = await fetchJournal(limit);
    return NextResponse.json(snapshot, {
      headers: { "cache-control": "no-store" },
    });
  } catch (error) {
    const detail = error instanceof Error ? error.message : "unknown error";
    return NextResponse.json(emptyJournal(`engine unreachable: ${detail}`), {
      headers: { "cache-control": "no-store" },
    });
  }
}
