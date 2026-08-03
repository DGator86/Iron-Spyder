"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useState } from "react";

export function Providers({ children }: { children: React.ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            // The radar polls; stale-while-revalidate would show a frozen field
            // during refetch, which on a trading surface reads as a stall.
            staleTime: 2_000,
            refetchOnWindowFocus: false,
            retry: 1,
          },
        },
      }),
  );

  return (
    <QueryClientProvider client={client}>
      <TooltipProvider delayDuration={200} skipDelayDuration={400}>
        {children}
      </TooltipProvider>
    </QueryClientProvider>
  );
}
