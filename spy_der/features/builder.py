"""Feature construction (spec section 34 pipeline step "calculate option analytics").

One :class:`MarketFeatures` record is produced per snapshot and is the sole
input to every model. Centralising it here means the structural model, the
baselines, the specialists and the backtest all see byte-identical inputs, and
it gives the feature set a single version string to record in the audit trail.

Temporal features (flow acceleration, IV change, wall interaction) need memory,
which lives in :class:`FeatureContext`. The context is owned by the caller and
threaded through explicitly rather than hidden in module state, so a backtest
can hold one context per walk-forward fold without leakage between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import datetime

import numpy as np

from spy_der.analytics.chain import ChainArrays, build_chain_arrays, liquidity_score
from spy_der.analytics.exposures import ExposureBundle, compute_exposures
from spy_der.analytics.flow import FlowMetrics, FlowTracker, compute_flow
from spy_der.analytics.gamma_profile import GammaProfile, build_gamma_profile
from spy_der.analytics.surface import (
    ExpectedMove,
    SkewMetrics,
    TermStructure,
    compute_expected_move,
    compute_skew,
    compute_term_structure,
    compute_volatility_edge,
)
from spy_der.analytics.walls import WallInteraction, WallSet, WallTracker, find_walls
from spy_der.domain.market import MarketSnapshot, realized_volatility

#: Bump on any change to the feature set; recorded with every decision.
FEATURE_VERSION = "1.0.0"


@dataclass
class FeatureContext:
    """Rolling state required for temporal features."""

    wall_tracker: WallTracker = field(default_factory=WallTracker)
    flow_tracker: FlowTracker = field(default_factory=FlowTracker)
    iv_history: list[float] = field(default_factory=list)
    rv_history: list[float] = field(default_factory=list)
    spot_history: list[float] = field(default_factory=list)
    max_history: int = 128

    def push(self, *, atm_iv: float, realized_vol: float, spot: float) -> None:
        for series, value in (
            (self.iv_history, atm_iv),
            (self.rv_history, realized_vol),
            (self.spot_history, spot),
        ):
            series.append(value)
            if len(series) > self.max_history:
                series.pop(0)

    @staticmethod
    def _delta(series: list[float], lag: int = 1) -> float:
        if len(series) <= lag:
            return 0.0
        return series[-1] - series[-1 - lag]

    @property
    def iv_change(self) -> float:
        return self._delta(self.iv_history)

    @property
    def rv_change(self) -> float:
        return self._delta(self.rv_history)

    @property
    def rv_acceleration(self) -> float:
        if len(self.rv_history) < 3:
            return 0.0
        return (
            self.rv_history[-1] - 2.0 * self.rv_history[-2] + self.rv_history[-3]
        )

    @property
    def spot_change(self) -> float:
        return self._delta(self.spot_history)


@dataclass(frozen=True)
class MarketFeatures:
    """Versioned feature vector for one snapshot."""

    timestamp: datetime
    spot: float
    feature_version: str = FEATURE_VERSION

    # --- dealer exposure (spec 8) ------------------------------------------
    gex_total: float = 0.0
    gex_agreement: float = 1.0
    gex_sign_consensus: float = 1.0
    dex_total: float = 0.0
    vex_total: float = 0.0
    cex_total: float = 0.0

    # --- gamma profile (spec 8.5, 8.8) -------------------------------------
    gamma_flip: float | None = None
    flip_distance_norm: float = 0.0
    gamma_slope_at_spot: float = 0.0
    in_positive_gamma: float = 0.0
    vol_trigger: float | None = None
    vol_trigger_distance_norm: float = 0.0

    # --- walls (spec 8.6, 8.7) ---------------------------------------------
    call_wall: float | None = None
    put_wall: float | None = None
    call_wall_distance_norm: float = 0.0
    put_wall_distance_norm: float = 0.0
    wall_concentration: float = 0.0
    position_in_range: float = 0.5
    wall_interaction: str = WallInteraction.AWAY.value

    # --- volatility surface (spec 9) ---------------------------------------
    atm_iv: float = 0.0
    expected_move: float = 0.0
    expected_move_pct: float = 0.0
    em_utilization: float = 0.0
    em_remaining_fraction: float = 1.0
    put_skew: float = 0.0
    call_skew: float = 0.0
    risk_reversal: float = 0.0
    butterfly: float = 0.0
    term_slope: float = 0.0
    term_curvature: float = 0.0
    term_inverted: float = 0.0
    realized_vol: float = 0.0
    iv_rv_spread: float = 0.0
    vol_edge: float = 0.0
    iv_change: float = 0.0
    rv_change: float = 0.0
    rv_acceleration: float = 0.0

    # --- flow (spec 10) -----------------------------------------------------
    flow_tilt: float = 0.0
    flow_delta: float = 0.0
    flow_gamma: float = 0.0
    flow_vega: float = 0.0
    flow_acceleration: float = 0.0
    flow_persistence: float = 0.0
    flow_reversal: float = 0.0
    flow_divergence: float = 0.0
    flow_strike_concentration: float = 0.0
    flow_informative: float = 0.0

    # --- price context ------------------------------------------------------
    return_since_open: float = 0.0
    vwap_distance_norm: float = 0.0
    minutes_to_close: float = 390.0
    session_fraction: float = 0.0

    # --- pin ----------------------------------------------------------------
    pin_strike: float | None = None
    pin_distance_norm: float = 0.0
    pin_concentration: float = 0.0

    # --- quality ------------------------------------------------------------
    data_quality: float = 1.0
    liquidity: float = 0.0
    seconds_to_nearest_expiry: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Numeric-only view for model input; non-numeric fields are dropped."""
        out: dict[str, float] = {}
        for f in fields(self):
            if f.name in {"timestamp", "feature_version", "wall_interaction"}:
                continue
            value = getattr(self, f.name)
            if value is None:
                continue
            out[f.name] = float(value)
        return out

    def vector(self, names: tuple[str, ...]) -> np.ndarray:
        data = self.to_dict()
        return np.array([data.get(name, 0.0) for name in names], dtype=np.float64)


@dataclass(frozen=True)
class AnalyticsBundle:
    """Raw analytics retained alongside features for the dashboard and audit."""

    arrays: ChainArrays
    exposures: ExposureBundle
    gamma_profile: GammaProfile
    walls: WallSet
    expected_move: ExpectedMove | None
    skew: SkewMetrics | None
    term_structure: TermStructure
    flow: FlowMetrics
    features: MarketFeatures


def build_features(
    snapshot: MarketSnapshot, context: FeatureContext | None = None
) -> AnalyticsBundle:
    """Compute every analytic and assemble the feature vector."""
    ctx = context or FeatureContext()
    spot = snapshot.spot

    arrays = build_chain_arrays(
        snapshot.option_chain,
        spot=spot,
        rate=snapshot.risk_free_rate,
        div_yield=snapshot.dividend_yield,
    )
    exposures = compute_exposures(arrays, trades=snapshot.option_trades)
    profile = build_gamma_profile(arrays)

    near_expiry = snapshot.nearest_expiration() or snapshot.expirations[0]
    expected_move = compute_expected_move(snapshot, near_expiry)

    # Walls are located on a single expiration. Aggregating across tenors lets
    # the enormous at-the-money gamma of a 0DTE contract outweigh a genuine
    # open-interest wall further out, pinning both walls to spot on every
    # snapshot.
    wall_expiry = _wall_expiration(snapshot)
    wall_quotes = snapshot.chain_for(wall_expiry) if wall_expiry else ()
    walls = find_walls(
        build_chain_arrays(
            wall_quotes,
            spot=spot,
            rate=snapshot.risk_free_rate,
            div_yield=snapshot.dividend_yield,
        )
        if wall_quotes
        else arrays
    )

    # Skew is computed on the nearest expiration with enough tenor to have a
    # meaningful delta profile; a 0DTE chain in its final minutes does not.
    skew_expiry = _skew_expiration(snapshot)
    skew = None
    if skew_expiry is not None:
        skew_quotes = snapshot.chain_for(skew_expiry)
        if skew_quotes:
            skew = compute_skew(
                build_chain_arrays(
                    skew_quotes,
                    spot=spot,
                    rate=snapshot.risk_free_rate,
                    div_yield=snapshot.dividend_yield,
                )
            )

    term = compute_term_structure(snapshot)
    flow = compute_flow(
        snapshot.option_trades,
        spot=spot,
        rate=snapshot.risk_free_rate,
        div_yield=snapshot.dividend_yield,
    )

    realized = realized_volatility(snapshot.spy_bars)
    atm_iv = expected_move.atm_iv if expected_move else term.front_iv

    ctx.flow_tracker.update(flow)
    interaction = ctx.wall_tracker.update(
        spot, walls.call_wall.strike if walls.call_wall else None
    )
    price_change = ctx.spot_change
    ctx.push(atm_iv=atm_iv, realized_vol=realized, spot=spot)

    vol_edge = compute_volatility_edge(atm_iv, realized)
    pin_strike, pin_concentration = _pin_candidate(exposures, spot)
    session_open = snapshot.spy_bars[0].open if snapshot.spy_bars else spot
    vwap = _session_vwap(snapshot)

    features = MarketFeatures(
        timestamp=snapshot.timestamp,
        spot=spot,
        gex_total=exposures.primary.gex,
        gex_agreement=exposures.agreement,
        gex_sign_consensus=exposures.sign_consensus,
        dex_total=exposures.primary.dex,
        vex_total=exposures.primary.vex,
        cex_total=exposures.primary.cex,
        gamma_flip=profile.nearest_flip,
        flip_distance_norm=profile.normalized_flip_distance or 0.0,
        gamma_slope_at_spot=profile.slope_at_spot,
        in_positive_gamma=1.0 if profile.in_positive_gamma else 0.0,
        vol_trigger=profile.vol_trigger,
        vol_trigger_distance_norm=(
            (profile.vol_trigger - spot) / spot if profile.vol_trigger else 0.0
        ),
        call_wall=walls.call_wall.strike if walls.call_wall else None,
        put_wall=walls.put_wall.strike if walls.put_wall else None,
        call_wall_distance_norm=(
            walls.call_wall.normalized_distance(spot) if walls.call_wall else 0.0
        ),
        put_wall_distance_norm=(
            walls.put_wall.normalized_distance(spot) if walls.put_wall else 0.0
        ),
        wall_concentration=walls.concentration,
        position_in_range=walls.position_in_range if walls.position_in_range is not None else 0.5,
        wall_interaction=interaction.value,
        atm_iv=atm_iv,
        expected_move=expected_move.move if expected_move else 0.0,
        expected_move_pct=expected_move.move_pct if expected_move else 0.0,
        em_utilization=expected_move.utilization if expected_move else 0.0,
        em_remaining_fraction=expected_move.remaining_fraction if expected_move else 1.0,
        put_skew=skew.put_skew if skew else 0.0,
        call_skew=skew.call_skew if skew else 0.0,
        risk_reversal=skew.risk_reversal if skew else 0.0,
        butterfly=skew.butterfly if skew else 0.0,
        term_slope=term.slope(),
        term_curvature=term.curvature,
        term_inverted=1.0 if term.is_inverted else 0.0,
        realized_vol=realized,
        iv_rv_spread=vol_edge.iv_rv_spread,
        vol_edge=vol_edge.vol_edge,
        iv_change=ctx.iv_change,
        rv_change=ctx.rv_change,
        rv_acceleration=ctx.rv_acceleration,
        flow_tilt=flow.directional_tilt,
        flow_delta=flow.net_delta_flow,
        flow_gamma=flow.net_gamma_flow,
        flow_vega=flow.net_vega_flow,
        flow_acceleration=ctx.flow_tracker.acceleration,
        flow_persistence=ctx.flow_tracker.persistence,
        flow_reversal=1.0 if ctx.flow_tracker.reversal else 0.0,
        flow_divergence=ctx.flow_tracker.divergence(price_change),
        flow_strike_concentration=flow.strike_concentration,
        flow_informative=1.0 if flow.is_informative else 0.0,
        return_since_open=(spot - session_open) / max(session_open, 1e-9),
        vwap_distance_norm=(spot - vwap) / max(spot, 1e-9),
        minutes_to_close=snapshot.context.minutes_to_close,
        session_fraction=1.0 - min(1.0, snapshot.context.minutes_to_close / 390.0),
        pin_strike=pin_strike,
        pin_distance_norm=(pin_strike - spot) / spot if pin_strike else 0.0,
        pin_concentration=pin_concentration,
        data_quality=snapshot.data_quality_score,
        liquidity=liquidity_score(arrays),
        seconds_to_nearest_expiry=max(
            0.0, (near_expiry - snapshot.timestamp).total_seconds()
        ),
    )

    return AnalyticsBundle(
        arrays=arrays,
        exposures=exposures,
        gamma_profile=profile,
        walls=walls,
        expected_move=expected_move,
        skew=skew,
        term_structure=term,
        flow=flow,
        features=features,
    )


def _skew_expiration(snapshot: MarketSnapshot) -> datetime | None:
    """Nearest expiration with at least an hour of life, else the nearest one."""
    return snapshot.nearest_expiration(min_seconds=3600.0) or snapshot.nearest_expiration()


def _wall_expiration(snapshot: MarketSnapshot) -> datetime | None:
    """Nearest expiration with at least a day of life, else the nearest one."""
    return snapshot.nearest_expiration(min_seconds=86_400.0) or snapshot.nearest_expiration()


def _pin_candidate(
    exposures: ExposureBundle,
    spot: float,
    *,
    band_pct: float = 0.005,
    neighbours: int = 4,
) -> tuple[float | None, float]:
    """Near-spot pin strike and how far it dominates the rest of the chain.

    Two things make this measurement non-obvious.

    First, concentration is a *dominance ratio* — ``1 - mean(next largest
    strikes) / largest`` — not the largest strike's share of the total. Gamma
    peaks at the money by construction, so on any smooth chain the ATM strike
    holds a double-digit share and every snapshot would look pinned.

    Second, the candidate is restricted to a narrow band around spot. A call
    wall is also a gamma concentration, and without the band a large wall eight
    points away is indistinguishable from a pin. Comparing the best *in-band*
    strike against the largest strikes *anywhere* means a chain dominated by
    distant walls correctly reports no pin at all.
    """
    by_strike = exposures.unsigned_gex_by_strike
    if not by_strike:
        return None, 0.0

    band = max(spot * band_pct, 1e-9)
    in_band = {k: abs(v) for k, v in by_strike.items() if abs(k - spot) <= band}
    if not in_band:
        return None, 0.0

    strike, largest = max(in_band.items(), key=lambda kv: kv[1])
    if largest <= 0.0:
        return None, 0.0

    competitors = sorted(
        (abs(v) for k, v in by_strike.items() if k != strike), reverse=True
    )[:neighbours]
    if not competitors:
        return strike, 0.0

    dominance = 1.0 - (sum(competitors) / len(competitors)) / largest
    return strike, float(np.clip(dominance, 0.0, 1.0))


def _session_vwap(snapshot: MarketSnapshot) -> float:
    bars = snapshot.spy_bars
    if not bars:
        return snapshot.spot
    volumes = np.array([b.volume for b in bars], dtype=np.float64)
    prices = np.array(
        [b.vwap if b.vwap is not None else b.close for b in bars], dtype=np.float64
    )
    total = float(np.sum(volumes))
    if total <= 0.0:
        return float(np.mean(prices))
    return float(np.sum(prices * volumes) / total)
