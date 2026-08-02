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
├── app/           FastAPI surface and Streamlit dashboard
└── pipeline.py    the end-to-end decision cycle
```

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

Docker: `docker compose up --build`.

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

## Testing

```bash
pytest -q        # 320 tests
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
2. **No historical SPY data ships.** All results are from the synthetic
   generator. Because it prices from the same model the simulator samples, it is
   arbitrage-free by construction and cannot demonstrate edge — only that the
   machinery is correct and that the system abstains when there is nothing there.
3. **The learned baselines are unfitted.** They fall back to the analytic GBM
   priors, which are calibrated by construction but carry no learned signal.
   Nonlinear specialists are not implemented; the ensemble is wired for them and
   disabled by default per the promotion rules.
4. **Coarsening `valuation_steps` biases exit simulation optimistically** — a
   stop that would have fired between two observations is missed when price
   recovers. Defaults leave a few minutes per step; treat coarser settings as a
   speed trade that flatters results.
5. **Path simulation is not calibrated to SPY.** The regime dynamics are priors.
   What is verified is that the driftless constant-volatility case reproduces the
   closed forms.
