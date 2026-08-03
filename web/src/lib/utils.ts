import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Compact currency for stat tiles: $1.2K, $98.7K, $1.28B. */
export function compactUsd(value: number, digits = 2): string {
  const abs = Math.abs(value);
  const sign = value < 0 ? "-" : "";
  if (abs >= 1e9) return `${sign}$${(abs / 1e9).toFixed(digits)}B`;
  if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(digits)}M`;
  if (abs >= 1e3) return `${sign}$${(abs / 1e3).toFixed(1)}K`;
  return `${sign}$${abs.toFixed(0)}`;
}

export function usd(value: number, digits = 0): string {
  return `${value < 0 ? "-" : ""}$${Math.abs(value).toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}`;
}

export function signedUsd(value: number, digits = 2): string {
  return `${value >= 0 ? "+" : "-"}$${Math.abs(value).toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}`;
}

export function pct(value: number, digits = 0): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function signedPct(value: number, digits = 2): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}%`;
}

export function price(value: number): string {
  return Number.isFinite(value) ? value.toFixed(2) : "—";
}

/** Hex -> rgba() with an explicit alpha, for chart series colours. */
export function alpha(hex: string, a: number): string {
  const clean = hex.replace("#", "");
  const full =
    clean.length === 3
      ? clean
          .split("")
          .map((c) => c + c)
          .join("")
      : clean;
  const n = parseInt(full, 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

/** Em dash for a value the engine could not supply. */
export const NA = "—";

/**
 * Render a possibly-missing number, or the em dash.
 *
 * The whole point of the nullable fields on `SystemPayload`: a missing feed has
 * to *look* missing. Formatting `null` as `0` would put "$0.00" in the equity
 * tile and "+0.00 (0.00%)" beside the price, both of which read as measurements.
 */
export function orNA(
  value: number | null | undefined,
  format: (n: number) => string,
): string {
  return value === null || value === undefined || Number.isNaN(value)
    ? NA
    : format(value);
}
