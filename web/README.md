# SPY-DER Forecast Radar

Vercel-deployable Next.js frontend for the Iron-Spyder engine. The design goal
is a weather-radar instrument, not a dashboard: one large probability field that
the user layers analytics onto, rather than a grid of unrelated metric cards.

```bash
pnpm install
pnpm dev          # http://localhost:3000
pnpm build        # production build
pnpm typecheck
```

## Deploying

This app lives in `web/`, not at the repository root. In the Vercel project:

1. **Settings → Build and Deployment → Root Directory** = `web`
2. **Framework Preset** = **Next.js**
3. **Output Directory** = **empty** (clear it if it says `public`)

Leaving Output Directory as `public` makes the build fail with
`No Output Directory named "public" found` after Next.js finishes — Next.js
does not emit a `public/` folder as its build output; Vercel serves `.next`.

That Root Directory setting is the mechanism, not a convenience. Left at the
repo root, Vercel scans it, finds `pyproject.toml`, and infers a Python backend
— the build then fails asking for a FastAPI entrypoint and offers to deploy
`spy_der/app/api:app`. **Do not accept that offer.** The engine's API exposes
unauthenticated mutating routes, including `POST /risk/kill-switch?enabled=false`,
which would put a public reset for a latched kill switch on the internet. It is
also stateful by design — the pipeline carries the HMM belief, the flow and wall
trackers and the open book — so a serverless runtime would discard all of it on
every cold start, and the SQLite audit store cannot write outside `/tmp`.

The engine belongs on the dedicated CPU VPS behind loopback. See
`../deploy/CPU_VPS.md`.

## Running without the engine

`SPYDER_API_BASE` is optional. Unset, the BFF at `app/api/chart/route.ts` serves
a **seeded synthetic session** from `src/lib/mock.ts`, and the header shows a
`SYNTHETIC DATA` badge. Every figure in that mode is fabricated.

Set `SPYDER_API_BASE` to the engine origin to go live. If the engine is set but
unreachable, the route falls back to synthetic **and** returns
`degraded: true` with the transport error, which the header surfaces — it never
silently presents synthetic output as live.

The engine binds loopback on the VPS (`docker-compose.yml` publishes
`127.0.0.1:8000`), so a Vercel deployment needs a reverse proxy in front of it.
Do not publish the engine port directly.

## Architecture

| Path | Role |
|---|---|
| `app/api/chart/route.ts` | BFF: live adapter, synthetic fallback, degraded reporting |
| `src/lib/types.ts` | Wire contract (`ForecastChartPayload`, layer + view state) |
| `src/lib/density.ts` | Quantile grids → continuous probability field |
| `src/lib/adapter.ts` | Engine REST → `RadarSnapshot` |
| `src/lib/mock.ts` | Seeded synthetic session |
| `src/lib/payoff.ts` | Expiry payoff geometry from legs |
| `src/lib/fieldImage.ts` | Field → RGBA bitmap |
| `src/components/chart/ForecastCanvas.tsx` | The canvas |
| `src/store/` | Zustand layer + view stores |

### The forecast field is reconstructed, not transported

The engine emits a **nine-point quantile grid at seven discrete horizons**
(`spy_der/domain/forecast.py`), not a 2-D surface. `src/lib/density.ts` builds
the field with two interpolations:

1. **Across time, in quantile space.** Each `tau`'s price is interpolated between
   bracketing horizons. Blending densities instead would produce a spurious
   bimodal smear wherever two horizons disagree; interpolating the quantile
   function slides the distribution rather than superposing it. The
   interpolation runs in `sqrt(minutes)` because diffusive spread grows with the
   square root of time.
2. **Across price, by differencing the CDF.** Each price bin's probability is
   the difference of the inverted quantile function at its two edges — exact bin
   masses, with none of the noise of differentiating an interpolated density.

Every time column sums to 1 before display scaling.

`gexSurface` is likewise derived: `/analytics/gamma-profile` returns GEX as a
function of spot, which is replicated across the time axis. GEX is a property of
a snapshot, not of time, and replication is the honest projection onto the
canvas.

Fields the engine cannot supply are left undefined and the layer renders as
unavailable. They are never filled with a plausible-looking number.

### Rendering

Fields are rasterized to RGBA bitmaps (`fieldImage.ts`) and positioned as
ECharts `graphic` images against a plot rect with computed pixel insets. One
draw call per field regardless of lattice size; a rect-per-cell heatmap put
~14k shapes on the canvas and collapsed under two stacked layers. The browser's
bilinear filter also smooths the coarse quantile lattice into a continuous wash.

The density field uses **per-column normalization** blended against the global
peak (`columnMix`). A forecast is a spike at `t=0` that spreads with `sqrt(t)`,
so pure global scaling leaves everything past the first minutes near-black,
while pure per-column scaling implies the distant future is as certain as the
near. The cursor readout always reports the absolute bin probability.

### Payoff geometry

`strategyGeometry()` computes profit/loss zones, breakevens and max profit/loss
from the **legs**, not from the candidate's reported fields, so the overlay can
never draw a profit region that contradicts the strikes drawn beside it. Both
values are retained so a mismatch is visible rather than reconciled away.

## Notes

- **Volatility time is trading time.** Year fractions use `252 * 390` minutes.
  Using calendar minutes understates sigma by ~1.9× and roughly halves the cone.
- **Encoding is never colour alone** (brief §11): dash pattern, line weight,
  glyph shape, arrow direction and hatching all carry meaning independently. The
  layer panel's glyphs mirror the canvas encodings so it doubles as the legend.
- The price axis always contains the call and put walls. A canvas that crops
  resistance out of frame cannot answer the question it exists to answer.
