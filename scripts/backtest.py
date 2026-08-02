"""Run the cost-stress backtest grid over a synthetic session.

    python -m scripts.backtest --scenario vol_expansion
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta

from spy_der.backtest.engine import run_cost_stress, stress_summary, survives_stress
from spy_der.data.providers.base import SyntheticProvider
from spy_der.data.synthetic import SCENARIOS, scenario
from spy_der.execution.fills import CostModel
from spy_der.execution.paper_broker import PaperBroker
from spy_der.models.forecast_engine import ForecastEngine, ForecastEngineConfig
from spy_der.optimizer.engine import OptimizerConfig, StrategyOptimizer
from spy_der.pipeline import DecisionPipeline, PipelineConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="vol_expansion", choices=sorted(SCENARIOS))
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    provider = SyntheticProvider(spec=scenario(args.scenario), step=timedelta(minutes=20))
    stamps = provider.session_timestamps(datetime(2026, 8, 3, tzinfo=UTC))[: args.steps]
    snapshots = [provider.snapshot(ts) for ts in stamps]

    def factory(costs: CostModel) -> DecisionPipeline:
        return DecisionPipeline(
            forecast_engine=ForecastEngine(config=ForecastEngineConfig(n_paths=600)),
            optimizer=StrategyOptimizer(
                OptimizerConfig(valuation_paths=300, valuation_steps=32, costs=costs)
            ),
            broker=PaperBroker(costs=costs),
            config=PipelineConfig(costs=costs, persist=False),
        )

    results = run_cost_stress(factory, snapshots)
    summary = stress_summary(results)

    header = f"{'profile':<24}{'trades':>8}{'total P&L':>12}{'expectancy':>12}{'win rate':>10}{'no-trade':>10}"
    print(header)
    print("-" * len(header))
    for name, stats in summary.items():
        print(
            f"{name:<24}{stats['trades']:8.0f}{stats['total_pnl']:12.2f}"
            f"{stats['expectancy']:12.2f}{stats['win_rate']:10.2f}{stats['no_trade_rate']:10.2f}"
        )

    print(f"\nsurvives stress through 1.50x spreads: {survives_stress(results)}")


if __name__ == "__main__":
    main()
