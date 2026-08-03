"use client";

import * as DialogPrimitive from "@radix-ui/react-dialog";
import { X } from "lucide-react";
import * as React from "react";
import { cn } from "@/lib/utils";

export const Sheet = DialogPrimitive.Root;
export const SheetTrigger = DialogPrimitive.Trigger;
export const SheetClose = DialogPrimitive.Close;

/**
 * Slide-over used for the layers drawer on tablet and the bottom sheets on
 * mobile. Side is explicit rather than inferred so the same component covers
 * both without a breakpoint check at the call site.
 */
export const SheetContent = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Content> & {
    side?: "right" | "bottom";
    title: string;
  }
>(({ className, children, side = "right", title, ...props }, ref) => (
  <DialogPrimitive.Portal>
    <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-void/70 backdrop-blur-sm" />
    <DialogPrimitive.Content
      ref={ref}
      className={cn(
        "fixed z-50 flex flex-col border-line bg-panel shadow-panel",
        side === "right"
          ? "inset-y-0 right-0 w-[86vw] max-w-[340px] border-l"
          : "inset-x-0 bottom-0 max-h-[75vh] rounded-t-xl border-t",
        className,
      )}
      {...props}
    >
      <div className="flex shrink-0 items-center justify-between border-b border-line px-3 py-2.5">
        <DialogPrimitive.Title className="panel-title">{title}</DialogPrimitive.Title>
        <DialogPrimitive.Close
          className="rounded p-1 text-ink-mute transition-colors hover:bg-raised hover:text-ink"
          aria-label="Close"
        >
          <X className="h-4 w-4" />
        </DialogPrimitive.Close>
      </div>
      <div className="min-h-0 flex-1 overflow-auto">{children}</div>
    </DialogPrimitive.Content>
  </DialogPrimitive.Portal>
));
SheetContent.displayName = "SheetContent";
