"""Regime-conditioned price and implied-volatility path simulation (spec section 20).

Terminal payoff is not enough for structures that are exited before expiry, so
the simulator produces joint ``(S_t, IV_t)`` paths and the exit engine walks
them.

The dynamics are deliberately modest and explainable:

* log-price follows a diffusion with a regime-dependent drift and a
  stochastic volatility factor;
* log-IV follows a mean-reverting Ornstein-Uhlenbeck process, negatively
  correlated with returns (the leverage effect), so a selloff raises IV and
  long-vega structures behave the way they do in practice;
* an optional jump component captures gap risk in expansion regimes.

Nothing here is calibrated to SPY history — the parameters are priors. What
matters for correctness is that the driftless, constant-vol special case
reproduces the closed forms in :mod:`spy_der.models.baselines.analytic`, which
is asserted in the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from spy_der.domain.enums import MarketState

SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0
FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class RegimeDynamics:
    """Per-state dynamics used to condition the simulation."""

    drift_annual: float = 0.0
    vol_multiplier: float = 1.0
    vol_of_vol: float = 0.8
    iv_mean_reversion: float = 6.0
    leverage_correlation: float = -0.65
    jump_intensity_annual: float = 0.0
    jump_std: float = 0.0
    iv_drift: float = 0.0
    """Annualized drift of log implied volatility; negative means IV bleeds."""


#: Priors by market state. Pinned and contracting regimes damp volatility and
#: bleed IV; expansion and breakout regimes raise both and add jump risk.
DEFAULT_DYNAMICS: dict[MarketState, RegimeDynamics] = {
    MarketState.STRONG_PIN: RegimeDynamics(
        vol_multiplier=0.65, vol_of_vol=0.4, iv_drift=-0.55
    ),
    MarketState.BROAD_RANGE: RegimeDynamics(vol_multiplier=0.85, iv_drift=-0.25),
    MarketState.BULL_GRIND: RegimeDynamics(
        drift_annual=0.18, vol_multiplier=0.9, iv_drift=-0.20
    ),
    MarketState.BEAR_GRIND: RegimeDynamics(
        drift_annual=-0.18, vol_multiplier=1.05, iv_drift=0.15
    ),
    MarketState.BULL_BREAKOUT: RegimeDynamics(
        drift_annual=0.55,
        vol_multiplier=1.35,
        vol_of_vol=1.2,
        jump_intensity_annual=12.0,
        jump_std=0.004,
        iv_drift=0.20,
    ),
    MarketState.BEAR_BREAKDOWN: RegimeDynamics(
        drift_annual=-0.60,
        vol_multiplier=1.45,
        vol_of_vol=1.3,
        jump_intensity_annual=18.0,
        jump_std=0.005,
        iv_drift=0.45,
    ),
    MarketState.VOL_EXPANSION: RegimeDynamics(
        vol_multiplier=1.6, vol_of_vol=1.6, jump_intensity_annual=20.0, jump_std=0.005,
        iv_drift=0.50,
    ),
    MarketState.VOL_CONTRACTION: RegimeDynamics(
        vol_multiplier=0.7, vol_of_vol=0.45, iv_drift=-0.70
    ),
    MarketState.DIRECTIONAL_RANGE: RegimeDynamics(vol_multiplier=0.9, iv_drift=-0.15),
    MarketState.TRANSITION: RegimeDynamics(vol_multiplier=1.2, vol_of_vol=1.2),
    MarketState.UNSTABLE: RegimeDynamics(vol_multiplier=1.5, vol_of_vol=1.5),
}


def blend_dynamics(
    state_probabilities: dict[MarketState, float],
    table: dict[MarketState, RegimeDynamics] | None = None,
) -> RegimeDynamics:
    """Probability-weighted blend of regime dynamics.

    Mixing the parameters rather than picking the modal state matters when the
    state distribution is genuinely uncertain: a 55/45 split between pin and
    breakout should not simulate a confident pin.
    """
    source = table or DEFAULT_DYNAMICS
    total = sum(state_probabilities.values())
    if total <= 0.0:
        return RegimeDynamics()

    def weighted(attr: str) -> float:
        return sum(
            p * getattr(source.get(state, RegimeDynamics()), attr)
            for state, p in state_probabilities.items()
        ) / total

    return RegimeDynamics(
        drift_annual=weighted("drift_annual"),
        vol_multiplier=weighted("vol_multiplier"),
        vol_of_vol=weighted("vol_of_vol"),
        iv_mean_reversion=weighted("iv_mean_reversion"),
        leverage_correlation=weighted("leverage_correlation"),
        jump_intensity_annual=weighted("jump_intensity_annual"),
        jump_std=weighted("jump_std"),
        iv_drift=weighted("iv_drift"),
    )


@dataclass(frozen=True)
class PathBundle:
    """Simulated joint price and IV paths."""

    times: FloatArray
    """Elapsed time in years at each step, shape ``(steps + 1,)``."""

    spot: FloatArray
    """Shape ``(paths, steps + 1)``."""

    iv_scale: FloatArray
    """Multiplicative IV factor relative to entry, shape ``(paths, steps + 1)``."""

    horizon_years: float
    initial_spot: float

    @property
    def n_paths(self) -> int:
        return self.spot.shape[0]

    @property
    def n_steps(self) -> int:
        return self.spot.shape[1] - 1

    @property
    def terminal(self) -> FloatArray:
        return self.spot[:, -1]

    @property
    def running_max(self) -> FloatArray:
        return np.maximum.accumulate(self.spot, axis=1)

    @property
    def running_min(self) -> FloatArray:
        return np.minimum.accumulate(self.spot, axis=1)

    def touch_probability_up(self, barrier: float) -> float:
        return float(np.mean(np.max(self.spot, axis=1) >= barrier))

    def touch_probability_down(self, barrier: float) -> float:
        return float(np.mean(np.min(self.spot, axis=1) <= barrier))

    def finish_probability_above(self, level: float) -> float:
        return float(np.mean(self.terminal >= level))

    def finish_between(self, lower: float, upper: float) -> float:
        return float(np.mean((self.terminal > lower) & (self.terminal < upper)))

    def no_touch_probability(self, lower: float, upper: float) -> float:
        return float(
            np.mean((np.min(self.spot, axis=1) > lower) & (np.max(self.spot, axis=1) < upper))
        )

    def price_quantiles(self, quantiles: tuple[float, ...]) -> dict[float, float]:
        values = np.quantile(self.terminal, quantiles)
        return {q: float(v) for q, v in zip(quantiles, values)}

    def iv_quantiles(self, quantiles: tuple[float, ...], base_iv: float) -> dict[float, float]:
        values = np.quantile(self.iv_scale[:, -1] * base_iv, quantiles)
        return {q: float(v) for q, v in zip(quantiles, values)}

    def subsample(self, *, n_paths: int | None = None, max_steps: int | None = None) -> PathBundle:
        """Thin the bundle for per-candidate valuation.

        The optimizer values every candidate on the *same* thinned bundle, so
        Monte Carlo noise is common to all of them and cancels in the ranking.
        A full-resolution bundle per candidate would be far more expensive and,
        because the noise would then be independent, actually worse at ordering
        candidates whose utilities are close.

        Reducing ``max_steps`` biases exit simulation **optimistically**: a stop
        that would have triggered between two coarse observations is missed when
        price recovers before the next one, exactly as discrete monitoring
        understates barrier-touch probability. Keep the step spacing well inside
        the intended reaction time — the defaults leave a few minutes per step
        over a session — and treat any coarser setting as a speed/accuracy trade
        that flatters the results.
        """
        spot, iv, times = self.spot, self.iv_scale, self.times

        if max_steps is not None and self.n_steps > max_steps:
            stride = int(np.ceil(self.n_steps / max_steps))
            index = np.arange(0, spot.shape[1], stride)
            if index[-1] != spot.shape[1] - 1:
                index = np.append(index, spot.shape[1] - 1)
            spot, iv, times = spot[:, index], iv[:, index], times[index]

        if n_paths is not None and spot.shape[0] > n_paths:
            spot, iv = spot[:n_paths], iv[:n_paths]

        return PathBundle(
            times=times,
            spot=spot,
            iv_scale=iv,
            horizon_years=self.horizon_years,
            initial_spot=self.initial_spot,
        )

    def truncate(self, horizon_years: float) -> PathBundle:
        """Return the sub-bundle covering ``[0, horizon_years]``.

        Lets the forecast engine simulate once to the longest horizon and read
        every shorter horizon off the same paths, which keeps the horizons
        mutually consistent — a 15-minute forecast is then literally a prefix
        of the end-of-day forecast rather than an independent draw.
        """
        if horizon_years >= self.horizon_years:
            return self
        cutoff = int(np.searchsorted(self.times, horizon_years, side="right"))
        cutoff = max(2, min(cutoff, self.spot.shape[1]))
        return PathBundle(
            times=self.times[:cutoff],
            spot=self.spot[:, :cutoff],
            iv_scale=self.iv_scale[:, :cutoff],
            horizon_years=float(self.times[cutoff - 1]),
            initial_spot=self.initial_spot,
        )

    @property
    def mfe(self) -> FloatArray:
        """Maximum favourable excursion per path, in price terms."""
        return np.max(self.spot, axis=1) - self.initial_spot

    @property
    def mae(self) -> FloatArray:
        """Maximum adverse excursion per path, in price terms."""
        return np.min(self.spot, axis=1) - self.initial_spot


@dataclass
class PathSimulator:
    """Monte Carlo generator for joint price and IV paths."""

    n_paths: int = 2_000
    steps_per_hour: float = 12.0
    max_steps: int = 240
    min_steps: int = 12
    seed: int | None = 20240801
    antithetic: bool = True
    """Pair each draw with its negation to cut variance at no extra cost."""

    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def reseed(self, seed: int | None) -> None:
        self._rng = np.random.default_rng(seed)

    def step_count(self, horizon_years: float) -> int:
        hours = horizon_years * 365.0 * 24.0
        return int(np.clip(round(hours * self.steps_per_hour), self.min_steps, self.max_steps))

    def simulate(
        self,
        *,
        spot: float,
        base_vol: float,
        horizon_years: float,
        dynamics: RegimeDynamics | None = None,
        n_paths: int | None = None,
    ) -> PathBundle:
        """Simulate ``n_paths`` joint paths over ``horizon_years``."""
        if horizon_years <= 0.0:
            raise ValueError("horizon_years must be positive")

        dyn = dynamics or RegimeDynamics()
        paths = n_paths or self.n_paths
        steps = self.step_count(horizon_years)
        dt = horizon_years / steps
        sqrt_dt = float(np.sqrt(dt))

        sigma = max(1e-6, base_vol * dyn.vol_multiplier)

        z_price, z_vol = self._draw(paths, steps)
        # Correlate the IV shock with the price shock (leverage effect).
        rho = float(np.clip(dyn.leverage_correlation, -0.99, 0.99))
        z_vol = rho * z_price + float(np.sqrt(1.0 - rho * rho)) * z_vol

        # Log-IV as an OU process reverting to zero (i.e. to entry IV).
        log_iv = np.zeros((paths, steps + 1), dtype=np.float64)
        kappa = max(0.0, dyn.iv_mean_reversion)
        for t in range(steps):
            drift = (dyn.iv_drift - kappa * log_iv[:, t]) * dt
            shock = dyn.vol_of_vol * sqrt_dt * z_vol[:, t]
            log_iv[:, t + 1] = log_iv[:, t] + drift + shock
        iv_scale = np.exp(np.clip(log_iv, -2.0, 2.0))

        # Instantaneous volatility follows the simulated IV level.
        inst_vol = sigma * iv_scale[:, :-1]
        log_return = (
            (dyn.drift_annual - 0.5 * inst_vol**2) * dt + inst_vol * sqrt_dt * z_price
        )

        if dyn.jump_intensity_annual > 0.0 and dyn.jump_std > 0.0:
            counts = self._rng.poisson(dyn.jump_intensity_annual * dt, size=(paths, steps))
            jumps = self._rng.normal(0.0, dyn.jump_std, size=(paths, steps)) * counts
            log_return = log_return + jumps

        log_path = np.zeros((paths, steps + 1), dtype=np.float64)
        log_path[:, 1:] = np.cumsum(log_return, axis=1)

        return PathBundle(
            times=np.linspace(0.0, horizon_years, steps + 1),
            spot=spot * np.exp(log_path),
            iv_scale=iv_scale,
            horizon_years=horizon_years,
            initial_spot=spot,
        )

    def _draw(self, paths: int, steps: int) -> tuple[FloatArray, FloatArray]:
        """Draw shocks, optionally antithetic."""
        if not self.antithetic or paths < 2:
            return (
                self._rng.standard_normal((paths, steps)),
                self._rng.standard_normal((paths, steps)),
            )
        half = paths // 2
        base_price = self._rng.standard_normal((half, steps))
        base_vol = self._rng.standard_normal((half, steps))
        price = np.concatenate([base_price, -base_price], axis=0)
        vol = np.concatenate([base_vol, -base_vol], axis=0)
        if price.shape[0] < paths:  # odd path count
            price = np.concatenate([price, self._rng.standard_normal((1, steps))], axis=0)
            vol = np.concatenate([vol, self._rng.standard_normal((1, steps))], axis=0)
        return price, vol
