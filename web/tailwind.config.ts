import type { Config } from "tailwindcss";

/**
 * Radar-terminal palette. Everything is tuned for a dark operations room:
 * near-black navy substrate, low-chroma panel chrome, and saturated accents
 * reserved exclusively for data. Chrome must never compete with the field.
 */
const config: Config = {
  darkMode: "class",
  content: ["./app/**/*.{ts,tsx}", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Substrate
        void: "#05090F",
        deep: "#070D16",
        panel: "#0B1220",
        raised: "#111B2E",
        line: "#1B2740",
        "line-bright": "#2A3B5C",

        // Type
        ink: "#E6EDF7",
        "ink-dim": "#94A6C0",
        "ink-mute": "#5A6E8C",

        // Semantic data accents
        signal: "#22D3EE", // cyan — model / forecast
        bull: "#34D399", // green — calls, profit, upside
        bear: "#F87171", // red — puts, loss, downside
        warn: "#FBBF24", // amber — VWAP, vol trigger
        magenta: "#E879F9", // violet — vanna / secondary greeks
        flip: "#67E8F9", // gamma flip

        // Status
        live: "#22C55E",
        stale: "#F59E0B",
        dead: "#EF4444",
      },
      fontFamily: {
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "0.875rem", letterSpacing: "0.04em" }],
        micro: ["0.625rem", { lineHeight: "0.8125rem", letterSpacing: "0.08em" }],
      },
      boxShadow: {
        panel: "0 1px 0 0 rgba(255,255,255,0.03) inset, 0 8px 24px -12px rgba(0,0,0,0.9)",
        glow: "0 0 24px -4px rgba(34,211,238,0.35)",
      },
      keyframes: {
        "pulse-live": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.35" },
        },
      },
      animation: {
        "pulse-live": "pulse-live 2s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};

export default config;
