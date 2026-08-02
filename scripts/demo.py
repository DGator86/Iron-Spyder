"""Run one decision cycle across every synthetic scenario.

    python -m scripts.demo
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from spy_der.data.synthetic import SCENARIOS, build_scenario
from spy_der.data.validators.quality import assess
from spy_der.features.builder import FeatureContext, build_features
from spy_der.models.forecast_engine import ForecastEngine, ForecastEngineConfig
from spy_der.optimizer.engine import OptimizerConfig, StrategyOptimizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=800, help="simulated paths per forecast")
    args = parser.parse_args()

    header = (
        f"{'scenario':<19}{'state':<18}{'conf':>6}{'DQ':>6}"
        f"{'cands':>7}  {'best family':<26}{'EV':>9}{'util':>9}  decision"
    )
    print(header)
    print("-" * len(header))

    for name in SCENARIOS:
        snapshot = build_scenario(name)
        snapshot = replace(snapshot, data_quality_score=assess(snapshot).score)

        context = FeatureContext()
        for _ in range(3):
            bundle = build_features(snapshot, context)

        engine = ForecastEngine(config=ForecastEngineConfig(n_paths=args.paths))
        for _ in range(3):
            forecast, paths = engine.forecast(bundle)

        optimizer = StrategyOptimizer(
            OptimizerConfig(valuation_paths=max(200, args.paths // 2), valuation_steps=36)
        )
        result = optimizer.optimize(
            snapshot=snapshot, features=bundle.features, forecast=forecast, paths=paths
        )

        best = result.best
        decision = result.decision if result.trade_selected else "NO_TRADE"
        print(
            f"{name:<19}{forecast.dominant_state.value:<18}"
            f"{forecast.confidence:6.2f}{snapshot.data_quality_score:6.2f}"
            f"{len(result.candidates):7d}  "
            f"{(best.family.value if best else '—'):<26}"
            f"{(best.expected_value if best else 0.0):9.2f}"
            f"{(best.utility if best else 0.0):9.2f}  {decision}"
        )


if __name__ == "__main__":
    main()
