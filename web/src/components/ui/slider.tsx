"use client";

import * as SliderPrimitive from "@radix-ui/react-slider";
import * as React from "react";
import { cn } from "@/lib/utils";

export const Slider = React.forwardRef<
  React.ElementRef<typeof SliderPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof SliderPrimitive.Root> & {
    accent?: string;
  }
>(({ className, accent, ...props }, ref) => (
  <SliderPrimitive.Root
    ref={ref}
    className={cn(
      "relative flex w-full touch-none select-none items-center py-1.5",
      className,
    )}
    {...props}
  >
    <SliderPrimitive.Track className="relative h-[3px] w-full grow overflow-hidden rounded-full bg-line">
      <SliderPrimitive.Range
        className="absolute h-full rounded-full"
        style={{ backgroundColor: accent ?? "#22D3EE" }}
      />
    </SliderPrimitive.Track>
    <SliderPrimitive.Thumb
      className={cn(
        "block h-3 w-3 rounded-full border-2 bg-void shadow transition-transform",
        "hover:scale-110 focus-visible:outline-none disabled:pointer-events-none disabled:opacity-40",
      )}
      style={{ borderColor: accent ?? "#22D3EE" }}
    />
  </SliderPrimitive.Root>
));
Slider.displayName = "Slider";
