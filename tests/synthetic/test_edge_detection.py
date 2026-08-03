"""Edge detection: does the system act when genuine mispricing exists?

The rest of the synthetic suite establishes that the system *abstains* — that
it returns no-trade against an arbitrage-free chain. That is only half the
property worth having. A system that never trades also passes every one of
those tests, so abstention alone proves nothing about the decision machinery.

These tests supply the other half. The synthetic chain is priced off
``atm_iv`` while paths are drawn at ``realized_vol``, so
:attr:`~spy_der.data.synthetic.ScenarioSpec.vol_edge` is a *known, constructed*
mispricing and the only source of real edge in the synthetic world. Holding the
chain fixed and varying only ``realized_vol`` gives a controlled sweep with
ground truth: at zero edge no-trade is correct, and as the edge grows the
system should eventually act.

Two caveats, stated plainly because they bound what a pass means:

1. Edge here exists *by construction*. Detecting it validates the valuation and
   selection machinery, not the strategy. Whether comparable edge exists in real
   SPY data is a separate question these tests cannot answer.
2. The sensitivity test below is a **characterization**, not a target. It
   records the mispricing the system currently needs before it will act, which
   is far larger than the edge a real opportunity would offer. It exists so the
   number cannot drift silently, and so an improvement is visible as a change to
   a recorded constant rather than an unmeasured feeling.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from spy_der.data.providers.base import SyntheticProvider
from spy_der.data.synthetic import scenario
from spy_der.execution.paper_broker import PaperBroker
from spy_der.models.forecast_engine import ForecastEngine, ForecastEngineConfig
from spy_der.optimizer.engine import OptimizerConfig, StrategyOptimizer
from spy_der.pipeline import DecisionPipeline, PipelineConfig

SESSION = datetime(2026, 8, 3, tzinfo=UTC)
BASE_IV = 0.20

#: Measured implied-to-realized gap the system needs before it will trade.
#:
#: Recorded from the sweep in ``test_detection_sensitivity_is_recorded``: at a
#: 0.20 implied vol, nothing is admitted until realized vol reaches roughly
#: 0.44. Twenty-four vol points is an enormous mispricing — far beyond anything
#: a real chain offers — so this documents a real limitation rather than a
#: tuned parameter. Lowering it is an improvement; raising it is a regression.
DETECTION_THRESHOLD = 0.24


def decide(*, realized_vol: float, atm_iv: float = BASE_IV, name: str = "broad_range"):
    """One decision against a chain with a controlled mispricing."""
    spec = scenario(name, atm_iv=atm_iv, realized_vol=realized_vol)
    provider = SyntheticProvider(spec=spec, step=timedelta(minutes=20))
    snapshot = provider.snapshot(provider.session_timestamps(SESSION)[3])
    pipeline = DecisionPipeline(
        forecast_engine=ForecastEngine(config=ForecastEngineConfig(n_paths=150)),
        optimizer=StrategyOptimizer(OptimizerConfig(valuation_paths=100, valuation_steps=16)),
        broker=PaperBroker(),
        config=PipelineConfig(persist=False),
    )
    return spec, pipeline.step(snapshot)


def test_vol_edge_is_the_constructed_mispricing():
    spec = scenario("broad_range", atm_iv=0.20, realized_vol=0.32)
    assert spec.vol_edge == pytest.approx(0.12)
    assert scenario("broad_range", atm_iv=0.20, realized_vol=0.20).vol_edge == pytest.approx(0.0)


def test_named_scenarios_carry_the_edge_they_advertise():
    """The three long-vol scenarios are the only ones with positive edge."""
    long_vol = {"bull_breakout", "bear_breakdown", "vol_expansion"}
    for name in long_vol:
        assert scenario(name).vol_edge > 0.05, f"{name} should favour long volatility"
    for name in ("strong_pin", "vol_contraction", "bear_grind", "bull_grind"):
        assert scenario(name).vol_edge < 0.0, f"{name} should favour short volatility"


def test_no_trade_at_zero_edge():
    """The load-bearing property: a fairly priced chain must produce no trade.

    This is the one assertion here that should never change. Whatever is done
    to sensitivity, trading a chain with no mispricing is always wrong.
    """
    spec, result = decide(realized_vol=BASE_IV)
    assert spec.vol_edge == pytest.approx(0.0)
    assert result.decision.lower().startswith("no_trade")


def test_a_large_positive_edge_is_detected_and_traded():
    """Proof the machinery can act at all, not merely abstain."""
    spec, result = decide(realized_vol=BASE_IV + DETECTION_THRESHOLD)
    assert spec.vol_edge == pytest.approx(DETECTION_THRESHOLD)

    optimization = result.optimization
    assert optimization is not None
    assert optimization.candidates, "a 24-point mispricing produced no candidates at all"

    best = optimization.best
    assert best is not None
    assert best.expected_value > 0.0
    # Realized volatility far above implied is a long-volatility opportunity.
    assert "Long" in best.family.value, f"expected a long-vol structure, got {best.family.value}"


def test_detection_sensitivity_is_recorded():
    """Characterization: how much mispricing before anything is admitted.

    Not a target. The measured answer is far larger than any real opportunity,
    which is the point of recording it — the limitation is visible in the suite
    instead of being discovered later against real data.
    """
    admitted: dict[float, int] = {}
    for edge in (-0.12, -0.04, 0.0, 0.10, DETECTION_THRESHOLD):
        _spec, result = decide(realized_vol=BASE_IV + edge)
        optimization = result.optimization
        admitted[edge] = len(optimization.candidates) if optimization else 0

    assert admitted[0.0] == 0, "a fairly priced chain must admit nothing"
    assert admitted[DETECTION_THRESHOLD] > 0, "the recorded threshold no longer detects"

    below = {e: n for e, n in admitted.items() if e != DETECTION_THRESHOLD}
    assert all(n == 0 for n in below.values()), (
        f"sensitivity improved — candidates now appear below the recorded "
        f"threshold: {below}. That is good; lower DETECTION_THRESHOLD to match."
    )


def test_short_volatility_edge_is_not_monetized():
    """A known gap, asserted so it cannot be forgotten.

    Twelve of the fifteen named scenarios carry negative edge — realized
    volatility below implied — which is a short-premium opportunity. The system
    takes none of them. The cause is upstream of strategy selection: the
    dealer-agreement term multiplies into a single confidence score that gates
    all trading, and in quiet regimes the sign conventions disagree, so
    confidence collapses before any short-premium structure is considered.

    That edge does not depend on dealer positioning at all, so it is being
    vetoed by uncertainty about a different input. This test fails the day that
    is addressed, which is the intended prompt to delete it.
    """
    _spec, result = decide(realized_vol=BASE_IV - 0.12)
    optimization = result.optimization
    assert optimization is not None
    assert not optimization.candidates, (
        "short-volatility edge is now being monetized — remove this test and "
        "the note about it in the README"
    )
