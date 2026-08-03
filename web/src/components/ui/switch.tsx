"use client";

import * as SwitchPrimitive from "@radix-ui/react-switch";
import * as React from "react";
import { cn } from "@/lib/utils";

/**
 * Toggle that adopts the colour of the layer it controls when on, so the
 * layers panel doubles as the canvas legend. Colour is never the only cue —
 * the thumb position carries the state independently.
 */
export const Switch = React.forwardRef<
  React.ElementRef<typeof SwitchPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof SwitchPrimitive.Root> & {
    accent?: string;
  }
>(({ className, accent, checked, style, ...props }, ref) => (
  <SwitchPrimitive.Root
    ref={ref}
    checked={checked}
    className={cn(
      "peer inline-flex h-[18px] w-[32px] shrink-0 cursor-pointer items-center rounded-full",
      "border transition-colors duration-150",
      checked ? "border-transparent" : "border-line-bright bg-deep",
      "disabled:cursor-not-allowed disabled:opacity-40",
      className,
    )}
    style={{
      ...(checked ? { backgroundColor: accent ?? "#22D3EE" } : undefined),
      ...style,
    }}
    {...props}
  >
    <SwitchPrimitive.Thumb
      className={cn(
        "pointer-events-none block h-[13px] w-[13px] rounded-full shadow-sm",
        "transition-transform duration-150",
        checked ? "bg-void" : "bg-ink-mute",
        "data-[state=unchecked]:translate-x-[2px] data-[state=checked]:translate-x-[16px]",
      )}
    />
  </SwitchPrimitive.Root>
));
Switch.displayName = "Switch";
