import type { Metadata, Viewport } from "next";
import "./globals.css";
import { Providers } from "./providers";

export const metadata: Metadata = {
  title: "SPY-DER — Forecast Radar",
  description:
    "SPY defined-risk options intelligence: probability field, dealer positioning, and strategy geometry on one surface.",
};

export const viewport: Viewport = {
  themeColor: "#05090F",
  width: "device-width",
  initialScale: 1,
  // The canvas relies on precise pixel mapping; pinch-zoom would desync overlays.
  maximumScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="h-full overflow-hidden">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
