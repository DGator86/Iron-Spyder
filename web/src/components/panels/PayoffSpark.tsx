"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

/**
 * Miniature expiry-payoff shape.
 *
 * Split into a profit path and a loss path against a zero rule so the shape is
 * readable at 14×6 px without a legend — the reader is matching geometry to the
 * probability field above, not reading values off it.
 */
export function PayoffSpark({
  curve,
  className,
}: {
  curve: Array<{ price: number; payoff: number }>;
  className?: string;
}) {
  const path = React.useMemo(() => {
    if (curve.length < 2) return null;

    const xs = curve.map((c) => c.price);
    const ys = curve.map((c) => c.payoff);
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    const yMin = Math.min(...ys);
    const yMax = Math.max(...ys);
    const ySpan = Math.max(1e-6, yMax - yMin);

    const toX = (price: number) => ((price - xMin) / Math.max(1e-6, xMax - xMin)) * 100;
    const toY = (payoff: number) => 24 - ((payoff - yMin) / ySpan) * 22 - 1;

    const d = curve
      .map((point, i) => `${i === 0 ? "M" : "L"}${toX(point.price).toFixed(2)},${toY(point.payoff).toFixed(2)}`)
      .join(" ");

    return { d, zeroY: toY(0), hasZero: yMin < 0 && yMax > 0 };
  }, [curve]);

  if (!path) return null;

  return (
    <svg viewBox="0 0 100 24" preserveAspectRatio="none" className={cn(className)} aria-hidden>
      {path.hasZero ? (
        <line
          x1="0"
          y1={path.zeroY}
          x2="100"
          y2={path.zeroY}
          stroke="#2A3B5C"
          strokeWidth="0.8"
          strokeDasharray="2 2"
          vectorEffect="non-scaling-stroke"
        />
      ) : null}
      <path
        d={path.d}
        fill="none"
        stroke="#22D3EE"
        strokeWidth="1.4"
        vectorEffect="non-scaling-stroke"
        strokeLinejoin="round"
      />
    </svg>
  );
}
