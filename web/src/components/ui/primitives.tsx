"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Button
// ---------------------------------------------------------------------------

type ButtonVariant = "ghost" | "solid" | "outline" | "danger";
type ButtonSize = "xs" | "sm" | "md" | "icon";

const VARIANTS: Record<ButtonVariant, string> = {
  ghost: "text-ink-dim hover:bg-raised hover:text-ink",
  solid: "bg-signal text-void hover:bg-signal/85 font-semibold",
  outline: "border border-line-bright text-ink-dim hover:border-signal hover:text-signal",
  danger: "border border-bear/60 text-bear hover:bg-bear/10",
};

const SIZES: Record<ButtonSize, string> = {
  xs: "h-6 px-2 text-micro",
  sm: "h-7 px-2.5 text-2xs",
  md: "h-8 px-3 text-xs",
  icon: "h-7 w-7",
};

export const Button = React.forwardRef<
  HTMLButtonElement,
  React.ButtonHTMLAttributes<HTMLButtonElement> & {
    variant?: ButtonVariant;
    size?: ButtonSize;
  }
>(({ className, variant = "ghost", size = "sm", ...props }, ref) => (
  <button
    ref={ref}
    className={cn(
      "inline-flex shrink-0 items-center justify-center gap-1.5 rounded-md",
      "font-medium uppercase tracking-wider transition-colors duration-100",
      "disabled:pointer-events-none disabled:opacity-35",
      VARIANTS[variant],
      SIZES[size],
      className,
    )}
    {...props}
  />
));
Button.displayName = "Button";

// ---------------------------------------------------------------------------
// Segmented control
// ---------------------------------------------------------------------------

export function Segmented<T extends string>({
  options,
  value,
  onChange,
  className,
  size = "sm",
  ariaLabel,
}: {
  options: Array<{ value: T; label: string; title?: string }>;
  value: T;
  onChange: (value: T) => void;
  className?: string;
  size?: "xs" | "sm";
  ariaLabel?: string;
}) {
  return (
    <div
      role="radiogroup"
      aria-label={ariaLabel}
      className={cn(
        "inline-flex items-center gap-0.5 rounded-md border border-line bg-deep p-0.5",
        className,
      )}
    >
      {options.map((option) => {
        const active = option.value === value;
        return (
          <button
            key={option.value}
            role="radio"
            aria-checked={active}
            title={option.title}
            onClick={() => onChange(option.value)}
            className={cn(
              "rounded-[4px] font-semibold uppercase tracking-wider transition-colors duration-100",
              size === "xs" ? "px-1.5 py-0.5 text-[10px]" : "px-2 py-1 text-micro",
              active
                ? "bg-signal/15 text-signal shadow-[inset_0_0_0_1px_rgba(34,211,238,0.45)]"
                : "text-ink-mute hover:bg-raised hover:text-ink-dim",
            )}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Stat tile
// ---------------------------------------------------------------------------

export function Stat({
  label,
  value,
  sub,
  tone = "neutral",
  className,
  children,
}: {
  label: string;
  value: React.ReactNode;
  sub?: React.ReactNode;
  tone?: "neutral" | "bull" | "bear" | "warn" | "signal";
  className?: string;
  children?: React.ReactNode;
}) {
  const toneClass = {
    neutral: "text-ink",
    bull: "text-bull",
    bear: "text-bear",
    warn: "text-warn",
    signal: "text-signal",
  }[tone];

  return (
    <div className={cn("panel flex min-w-0 flex-col gap-1 px-3 py-2", className)}>
      <div className="panel-title truncate">{label}</div>
      <div className={cn("tnum text-xl font-semibold leading-none", toneClass)}>{value}</div>
      {sub ? <div className="text-micro text-ink-mute">{sub}</div> : null}
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Meter
// ---------------------------------------------------------------------------

export function Meter({
  value,
  tone = "signal",
  className,
  label,
}: {
  /** 0..1 */
  value: number;
  tone?: "signal" | "bull" | "bear" | "warn";
  className?: string;
  label?: string;
}) {
  const color = { signal: "#22D3EE", bull: "#34D399", bear: "#F87171", warn: "#FBBF24" }[tone];
  const clamped = Math.min(1, Math.max(0, value));
  return (
    <div
      className={cn("h-1.5 w-full overflow-hidden rounded-full bg-line", className)}
      role="meter"
      aria-valuenow={Math.round(clamped * 100)}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={label}
    >
      <div
        className="h-full rounded-full transition-[width] duration-300"
        style={{ width: `${clamped * 100}%`, backgroundColor: color }}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Badge
// ---------------------------------------------------------------------------

export function Badge({
  children,
  tone = "neutral",
  className,
}: {
  children: React.ReactNode;
  tone?: "neutral" | "bull" | "bear" | "warn" | "signal";
  className?: string;
}) {
  const tones = {
    neutral: "border-line-bright text-ink-mute",
    bull: "border-bull/50 text-bull bg-bull/10",
    bear: "border-bear/50 text-bear bg-bear/10",
    warn: "border-warn/50 text-warn bg-warn/10",
    signal: "border-signal/50 text-signal bg-signal/10",
  }[tone];

  return (
    <span
      className={cn(
        "inline-flex items-center rounded border px-1.5 py-px",
        "text-[10px] font-semibold uppercase tracking-wider",
        tones,
        className,
      )}
    >
      {children}
    </span>
  );
}
