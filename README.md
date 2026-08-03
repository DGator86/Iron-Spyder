# SPY-DER

**SPY Defined-Risk Options Intelligence, Prediction, and Strategy Optimization System**

SPY-DER forecasts the distribution of future SPY price, implied volatility, and
path behaviour, then selects the defined-risk options structure with the highest
expected utility after realistic costs — or declines to trade.

```
Decision_t = argmax { U(S_1), U(S_2), …, U(S_n), U(NoTrade) }
```

The system trades one underlying (SPY), one instrument class (options), and one
risk philosophy (defined risk only). No-trade is a first-class competitor that
must be beaten by a margin, not a fallback for an empty candidate list.

---

## Governing constraints

| Constraint | Where it is enforced |
|---|---|
| Maximum loss finite and known before entry | `strategies/validation.py` — structural proof, not a numeric probe |
| No naked or uncovered short options | expiry suffix-coverage rule (below) |
| No SPY share ownership, ever | `StrategyFamily` has no ownership-dependent member |
| No reliance on assignment | `execution/manager.py` closes short ITM legs before settlement |
| Forecasting is separate from strategy selection | `ForecastBundle` has no strategy-shaped field |
| Trading disabled on degraded data | `data/validators/quality.py`, fails closed |
| Risk engine has absolute veto | `risk/engine.py` — no early return, no override parameter |

### The defined-risk proof

Every structure must satisfy one invariant, checked independently of the payoff
arithmetic and then cross-checked against it:

> For each option right, and for every expiration `T` in the structure, the sum
> of signed quantities over legs of that right with expiry `>= T` must be
> non-negative.

This single rule rejects naked shorts, ratio writes, short strangles, and — the
case a plain net-quantity check misses — **short calendars**, which net to zero
overall yet leave an uncovered far-dated short once the near leg expires.

Maximum loss for calendars and diagonals is computed with one-sided bounds
(long unexpired legs floored at intrinsic, short unexpired legs capped at `S`
or `K`), so the number handed to the risk engine is a genuine lower bound on
P&L rather than an estimate.

---

## Architecture

```
spy_der/
├── domain/        frozen dataclasses: snapshots, legs, candidates, forecasts
├── data/          providers, data-quality engine, synthetic scenarios, audit store
├── analytics/     pricing, Greeks, GEX/DEX/VEX/CEX, gamma profile, walls, surface, flow
├── features/      versioned feature construction
├── models/        structural scores, hidden-state HMM, baselines, calibration, ensemble
├── simulation/    regime-conditioned price and IV paths, exit simulation
├── strategies/    payoff engine, defined-risk validation, registry, generators
├── optimizer/     family filtering, candidate search, utility, no-trade comparison
├── risk/          trade/portfolio/session limits, sizing, kill switches
├── execution/     orders, fill model, paper broker, reconciliation, position management
├── backtest/      point-in-time replay, walk-forward, purging, cost stress
├── monitoring/    drift, calibration, and execution health
├── runtime/       market calendar, durable state, unattended supervisor
├── vps/           VPS module: state root, heartbeats, live_state, status API, deploy CLI
├── app/           FastAPI surface and Streamlit dashboard
└── pipeline.py    the end-to-end decision cycle
```

The VPS surface is its own module. `spy_der.runtime` owns the decision loop;
`spy_der.vps` owns the file layout, liveness publishing, read-only status API,
and the pull-based deploy units under `deploy/`.

The backtest drives the *same* `DecisionPipeline` as the live path. There is no
parallel backtest logic that could drift from production behaviour, and the
point-in-time guarantee `Decision_t = f(Data_{<=t})` is structural: the pipeline
only ever receives one snapshot.

---

## Quick start

```bash
pip install -e ".[dev]"

python -m scripts.demo          # one decision across every synthetic scenario
python -m scripts.paper_trade   # replay a session through the paper pipeline
python -m scripts.backtest      # the spec-37.7 cost-stress grid

uvicorn spy_der.app.api:app --reload            # API on :8000
streamlit run spy_der/app/dashboard.py          # dashboard on :8501
```

Docker (CPU-only live stack on a **dedicated** VPS — see `deploy/CPU_VPS.md`):

```bash
cp .env.example .env
docker compose up -d --build
```

### What the demo shows

```
scenario           state              conf    DQ  cands  best family          decision
strong_pin         StrongPin          0.16  0.93      5  BrokenWingPutButter… NO_TRADE
broad_range        BroadRange         0.05  0.94      0  —                    NO_TRADE
bull_breakout      BullBreakout       0.50  0.96      9  LongStraddle         LongStraddle:550
vol_expansion      VolExpansion       0.29  0.96     19  LongStraddle         LongStraddle:550
corrupt_quotes     BroadRange         0.00  0.00      0  —                    NO_TRADE
```

Most scenarios return no-trade, and that is the correct answer: the synthetic
chain is priced with the same model the simulator uses, so it is arbitrage-free
by construction and there is no edge to find. The system acts only where the
forecast genuinely disagrees with priced volatility — the expansion and breakout
regimes, where simulated realized volatility exceeds the implied level.

---

## Running unattended

```bash
python -m scripts.run --mode paper --interval 5 --state var/state.db
```

### Dedicated CPU VPS (not a GPU server)

Iron-Spyder runs on its **own** CPU machine. Do not co-tenant with SPY-DER,
0DTE, or a GPU host. The live workload is NumPy/SciPy/pandas/scikit-learn plus
FastAPI/Streamlit — a GPU would sit idle. Full sizing and the live/research
split are in [`deploy/CPU_VPS.md`](deploy/CPU_VPS.md).

| Role | Machine | Footprint |
|---|---|---|
| Live / paper | Dedicated CPU VPS | 8 vCPU, 32 GB RAM, 250–500 GB NVMe, Ubuntu 24.04, **no GPU** |
| Research | Second CPU box (on-demand) | 16–32 cores, 64–128 GB RAM, 1–2 TB NVMe |

```bash
# Live VPS
cp .env.example .env
docker compose up -d --build

# Research worker (different host)
docker compose -f docker-compose.research.yml run --rm research \
    python -m scripts.backtest
```

First-time box install (as root on the Iron-Spyder CPU VPS):

```bash
curl -fsSL https://raw.githubusercontent.com/DGator86/Iron-Spyder/main/deploy/remote-deploy.sh | bash
```

That provisions `/opt/iron-spyder`, `/var/lib/iron-spyder`, Docker Compose,
`iron-spyder.service`, and the self-update timer. Runtime units stay down until
`/etc/iron-spyder/iron-spyder.env` exists. Subsequent pushes land via the
pull-based update timer.

The VPS module (`spy_der.vps`) publishes heartbeats and `live_state.json` under
the Iron-Spyder state root; the status API serves them on loopback `:8788`.

The supervisor owns the clock and the failure handling, so nothing needs a
babysitter:

- **Sessions come from a computed NYSE calendar** (`runtime/calendar.py`) rather
  than a table that expires. Holidays are derived from their weekday rules,
  Good Friday from the Meeus/Jones/Butcher Easter algorithm, and the three early
  closes from their own conditions. Half days matter operationally: a runner
  that thinks 27 November closes at 16:00 holds positions three hours past the
  bell. Outside a session the loop sleeps until the next open instead of
  spinning.
- **Crashes are contained, not fatal.** A failing cycle is logged and retried
  with geometric backoff; consecutive failures past the configured ceiling trip
  a kill switch rather than continuing to trade blind.
- **State is durable** (`runtime/state_store.py`). Kill switches, open
  positions, and the daily loss and consecutive-loss counters are written to
  SQLite as explicit JSON — not pickle — and restored on start. This is a safety
  property, not a convenience one: without it a restart clears a latched kill
  switch, and "just restart it" becomes the way to bypass a lockout. If a stored
  position cannot be rebuilt, the `BROKER_STATE` switch trips instead of the
  daemon starting with a book it does not know about.
- **Session end forces closure** of anything still open, well before the
  assignment cutoff — the spec never relies on assignment. Positions that cannot
  be priced to close escalate to a kill switch rather than being left silently
  open.
- **Shutdown is graceful but does not liquidate.** `SIGTERM` finishes the
  current cycle, persists state, and exits with the book intact — in
  milliseconds, because every wait is interruptible rather than a plain
  `time.sleep`. That matters under a process supervisor: `docker stop`
  escalates to `SIGKILL` after ten seconds and systemd after ninety, so a loop
  that only notices the signal when its sleep expires gets killed mid-flight
  instead of saving. Dumping positions into whatever liquidity exists at
  shutdown is the worse failure, so the book is left with its stops intact.

---

## Historical data

### SPY-DER VPS recordings

Legacy session tapes staged from the old SPY-DER box live under
`/var/lib/iron-spyder/imports/spy-der/market/*.jsonl` on the dedicated CPU VPS
(never mixed into the live runtime tree). Convert and replay them with:

```bash
spyder-import-spyder inspect  --path /var/lib/iron-spyder/imports/spy-der/market
spyder-import-spyder backtest --path /var/lib/iron-spyder/imports/spy-der/market \
    --session 2026-07-31 --limit 50
```

### Vendor CSV / Parquet

The loader turns recorded chains into the same `MarketSnapshot` objects the live
path consumes, so replaying history is just a provider swap.

```bash
python -m scripts.ingest inspect  --path /data/spy --pattern 'SPY_*.csv'
python -m scripts.ingest load     --path /data/spy --bars /data/spy/SPY_bars.csv
python -m scripts.ingest backtest --path /data/spy --bars /data/spy/SPY_bars.csv
```

`inspect` reports the schema mapping, `load` reports what was ingested and its
data quality, and `backtest` streams the result through the *same*
`DecisionPipeline` the live path uses and prints the metrics — one command from
recorded files to results, with no decision logic that differs between replay
and production.

Run `inspect` first. Column naming is not standardized across vendors, so
`historical.COLUMN_ALIASES` maps a wide set of spellings onto a canonical schema
and `inspect` reports exactly which canonical fields resolved, which are
missing, and which source columns went unused — a mismatch produces a named
error rather than silently-zero open interest. Add unrecognized spellings to
that table rather than reshaping the data upstream.

CSV, gzipped CSV, and Parquet are read; reading is streaming and grouped by
timestamp, so a year of one-minute chains costs one chain of memory rather than
the whole history.

Two decisions are worth knowing about:

- **Implied volatility is optional on input.** When absent it is solved from the
  midpoint with the same model the analytics layer uses, which keeps the surface
  self-consistent instead of mixing a vendor's IV convention with ours. Chains
  that ship the underlying separately are handled: the bar close resolves the
  spot before the solve, so contracts are not all dropped for want of a price.
- **Quotes with no establishable IV are dropped, not zeroed.** Writing
  `implied_volatility=0.0` asserts that volatility *is* zero when the truth is
  that it is unknown; such a contract prices at intrinsic with zero gamma and
  zero vega and would quietly dilute every exposure aggregate it entered. The
  count is reported, and the data-quality engine independently notices if enough
  contracts go missing to thin the chain.

Timestamps: naive values are treated as UTC by default. Pass `--local-time` if
the source is wall-clock Eastern. The choice is not cosmetic — it shifts every
time to expiry.

---

## Testing

```bash
pytest -q        # 441 tests
ruff check .
mypy spy_der
```

The suite implements the spec's testing program:

- **Unit** — put-call parity, every Greek against finite differences (delta,
  gamma, vega, theta, vanna, vomma, charm, speed, colour), implied-volatility
  round trips, exact payoff geometry, breakevens, risk limits, order
  construction, reconciliation.
- **Property** — randomized over a seeded space: prices non-negative and within
  no-arbitrage bounds, monotonicity in spot and volatility, gamma and vega
  non-negative, every admitted structure has finite positive maximum loss and no
  uncovered short exposure, probabilities sum to one.
- **Synthetic** — fifteen scenarios with predefined expected states and risk
  decisions, including poor liquidity, wide spreads, stale data, corrupt quotes,
  and event lockout.
- **Backtest** — point-in-time replay, walk-forward folds, purging and embargo,
  the full cost-stress grid, parameter-stability plateau detection.
- **Runtime** — the calendar against the published NYSE schedule (holidays,
  observation rules, early closes, DST), state surviving a simulated crash with
  a latched kill switch intact, backoff and failure escalation, session-end
  flattening.
- **Ingestion** — vendor column aliasing, IV solved back to the volatility that
  generated the prices, unpriceable quotes dropped rather than zeroed, and the
  point-in-time guarantee that no bar stamped after a snapshot reaches it.
- **Integration** — the whole pipeline, kill switches, and the audit trail.

Closed-form barrier and quantile results in `models/baselines/analytic.py` are
validated against Monte Carlo and then serve as oracles for the path simulator,
so the simulator is not tested against itself.

---

## Deployment gates

`Mode` runs `research → backtest → paper → shadow → live`. Live is defined but
**cannot execute**: no live broker adapter ships, and `Settings.assert_live_allowed()`
refuses until the spec's phase-2 gates are recorded as met — six months of paper
trading, 250 completed trades, shadow execution, tested reconciliation, zero
unresolved risk defects, and explicit operator authorization. The API has no
live-order route at all, rather than one behind a flag.

---

## Known limitations

Stated plainly, because they bound what the current numbers mean:

1. **The structural weights are calibrated against the synthetic scenarios in
   this repository, not against SPY history.** Passing the scenario tests shows
   the state engine separates the regimes the generator encodes. The spec
   requires walk-forward validation on real data before those weights carry any
   authority, and that has not been done.
2. **Edge detection is far less sensitive than it should be, and is one-sided.**
   The synthetic chain is priced off `atm_iv` while paths are drawn at
   `realized_vol`, so `ScenarioSpec.vol_edge` is a known, constructed
   mispricing. Sweeping it against a fixed chain
   (`tests/synthetic/test_edge_detection.py`) shows the system admits **no
   candidate at all until realized volatility exceeds implied by roughly 0.24**
   — a twenty-four-point gap, far beyond anything a real chain offers.

   It is also directional. Twelve of the fifteen named scenarios carry
   *negative* edge, which is a short-premium opportunity, and the system takes
   none of them. The cause sits upstream of strategy selection: `dealer_agreement`
   multiplies into the single confidence score that gates all trading, and in
   quiet regimes the sign conventions disagree (≈0.37 against ≈0.86 in trending
   ones), so confidence collapses before any short-premium structure is
   considered. That edge does not depend on dealer positioning, so it is being
   vetoed by uncertainty about an unrelated input.

   Both facts are asserted in the test suite rather than left as prose, so
   neither can drift silently and an improvement shows up as a failing
   characterization test.
3. **No historical SPY data ships.** The loader exists and is tested, but every
   number in this README is from the synthetic generator. Because it prices from
   the same model the simulator samples, it is arbitrage-free by construction
   and cannot demonstrate edge — only that the machinery is correct and that the
   system abstains when there is nothing there. The alias table in
   `historical.COLUMN_ALIASES` covers the common vendor spellings but has not
   been run against a real vendor extract; `inspect` is the first thing to run
   on one.
4. **The learned baselines are unfitted.** They fall back to the analytic GBM
   priors, which are calibrated by construction but carry no learned signal.
   Nonlinear specialists are not implemented; the ensemble is wired for them and
   disabled by default per the promotion rules.
5. **Coarsening `valuation_steps` biases exit simulation optimistically** — a
   stop that would have fired between two observations is missed when price
   recovers. Defaults leave a few minutes per step; treat coarser settings as a
   speed trade that flatters results.
6. **Path simulation is not calibrated to SPY.** The regime dynamics are priors.
   What is verified is that the driftless constant-volatility case reproduces the
   closed forms.
