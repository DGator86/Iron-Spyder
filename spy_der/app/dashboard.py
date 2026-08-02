"""Streamlit dashboard (spec section 36).

Run with ``streamlit run spy_der/app/dashboard.py``.

The four panels mirror the spec's required sections: market structure,
prediction, strategy optimizer, and risk/execution plus model health. Rejected
candidates and the no-trade comparison are shown alongside the ranked list —
spec 36 asks for them explicitly, and an operator needs to see *why* the system
declined, not just that it did.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from spy_der.app.state import STATE
from spy_der.data.synthetic import SCENARIOS


def main() -> None:
    st.set_page_config(page_title="SPY-DER", layout="wide")
    st.title("SPY-DER — defined-risk SPY options intelligence")

    with st.sidebar:
        st.header("Controls")
        scenario = st.selectbox(
            "Scenario", sorted(SCENARIOS), index=sorted(SCENARIOS).index(STATE.scenario_name)
        )
        if scenario != STATE.scenario_name:
            STATE.set_scenario(scenario)
        if st.button("Run decision cycle"):
            STATE.run_once()
        if st.button("Reset pipeline"):
            STATE.reset()
        st.caption(f"Mode: {STATE.settings.mode.value}")

    result = STATE.current()

    if not result.quality.is_tradeable:
        st.error(f"Trading disabled — {result.quality.summary()}")

    _market_structure(result)
    _prediction(result)
    _optimizer(result)
    _risk_and_execution(result)


def _market_structure(result) -> None:
    st.subheader("Market structure")
    if result.analytics is None:
        st.info("Analytics unavailable under the current data-quality lockout.")
        return

    features = result.analytics.features
    columns = st.columns(6)
    for column, (label, value) in zip(
        columns,
        [
            ("SPY", f"{features.spot:,.2f}"),
            ("Expected move", f"{features.expected_move:,.2f}"),
            ("Gamma flip", _fmt(features.gamma_flip)),
            ("Call wall", _fmt(features.call_wall)),
            ("Put wall", _fmt(features.put_wall)),
            ("Vol trigger", _fmt(features.vol_trigger)),
        ],
    ):
        column.metric(label, value)

    columns = st.columns(6)
    for column, (label, value) in zip(
        columns,
        [
            ("GEX ($bn)", f"{features.gex_total / 1e9:,.2f}"),
            ("DEX ($bn)", f"{features.dex_total / 1e9:,.2f}"),
            ("ATM IV", f"{features.atm_iv:.1%}"),
            ("Realized vol", f"{features.realized_vol:.1%}"),
            ("Risk reversal", f"{features.risk_reversal:+.4f}"),
            ("Signed flow tilt", f"{features.flow_tilt:+.2f}"),
        ],
    ):
        column.metric(label, value)

    profile = result.analytics.gamma_profile
    st.line_chart(
        pd.DataFrame({"GEX": profile.gex}, index=pd.Index(profile.spots, name="SPY")),
        height=220,
    )


def _prediction(result) -> None:
    st.subheader("Prediction")
    if result.forecast is None:
        st.info("Forecast unavailable.")
        return

    forecast = result.forecast
    columns = st.columns(5)
    for column, (label, value) in zip(
        columns,
        [
            ("State", forecast.dominant_state.value),
            ("Confidence", f"{forecast.confidence:.2f}"),
            ("Entropy", f"{forecast.state_entropy:.2f}"),
            ("Disagreement", f"{forecast.model_disagreement:.2f}"),
            ("Dealer agreement", f"{forecast.dealer_agreement:.2f}"),
        ],
    ):
        column.metric(label, value)

    left, right = st.columns(2)
    left.caption("State probabilities")
    left.bar_chart(
        pd.Series(
            {state.value: p for state, p in forecast.state_probabilities.items()}
        ).sort_values(ascending=False),
        height=260,
    )

    right.caption("Price quantiles by horizon")
    right.dataframe(
        pd.DataFrame(
            {
                name: {f"q{int(q * 100):02d}": v for q, v in h.price_quantiles.items()}
                for name, h in forecast.horizons.items()
            }
        ).T,
        height=260,
    )


def _optimizer(result) -> None:
    st.subheader("Strategy optimizer")
    optimization = result.optimization
    if optimization is None:
        st.info("No optimization ran for this snapshot.")
        return

    st.caption(
        f"Decision: **{optimization.decision}** — no-trade bar is "
        f"{optimization.no_trade_utility + optimization.no_trade_margin:.2f} "
        f"({optimization.evaluated_count} structures admitted)"
    )

    if optimization.candidates:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "family": c.family.value,
                        "utility": round(c.utility, 2),
                        "EV": round(c.expected_value, 2),
                        "EROR": round(c.expected_return_on_risk, 3),
                        "max loss": round(c.max_loss, 0),
                        "CVaR": round(c.cvar, 0),
                        "liquidity": round(c.liquidity.score, 2),
                        "fill prob": round(c.fill_probability, 2),
                        "assignment": round(c.assignment_risk, 2),
                        "beats no-trade": c.utility
                        > optimization.no_trade_utility + optimization.no_trade_margin,
                    }
                    for c in optimization.top(12)
                ]
            ),
            height=320,
        )
    else:
        st.write("No candidate cleared the entry gates.")

    with st.expander("Rejected candidates and reasons"):
        rejected = [
            {"strategy": sid, "reason": "; ".join(reasons)}
            for sid, reasons in optimization.rejected[:40]
        ]
        st.dataframe(pd.DataFrame(rejected) if rejected else pd.DataFrame({"reason": []}))
        if optimization.registry_rejections:
            st.write("Defined-risk rejections:", optimization.registry_rejections)


def _risk_and_execution(result) -> None:
    st.subheader("Risk and execution")
    pipeline = STATE.pipeline
    if pipeline is None:
        return

    session = pipeline.risk_engine.session
    columns = st.columns(6)
    for column, (label, value) in zip(
        columns,
        [
            ("Equity", f"{pipeline.broker.equity:,.2f}"),
            ("Open risk", f"{pipeline.portfolio.total_open_risk:,.0f}"),
            ("Daily P&L", f"{session.realized_pnl:,.2f}"),
            ("Drawdown", f"{session.drawdown_fraction(pipeline.broker.equity):.1%}"),
            ("Data quality", f"{result.quality.score:.2f}"),
            ("Kill switch", "TRIPPED" if pipeline.risk_engine.kill_switch.is_tripped else "clear"),
        ],
    ):
        column.metric(label, value)

    if pipeline.risk_engine.kill_switch.is_tripped:
        st.error("\n".join(pipeline.risk_engine.kill_switch.reasons))

    if result.risk.veto_reasons:
        with st.expander("Risk vetoes"):
            for reason in result.risk.veto_reasons:
                st.write(f"- {reason}")

    positions = pipeline.portfolio.open_positions
    if positions:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "id": p.position_id,
                        "family": p.family.value,
                        "contracts": p.contracts,
                        "entry": round(p.entry_price, 2),
                        "mark": round(p.current_price, 2),
                        "unrealized": round(p.unrealized_pnl, 2),
                        "max loss": round(p.total_max_loss, 0),
                    }
                    for p in positions
                ]
            )
        )
    else:
        st.write("No open positions.")


def _fmt(value: float | None) -> str:
    return f"{value:,.2f}" if value is not None else "—"


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
