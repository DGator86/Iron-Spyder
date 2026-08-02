from __future__ import annotations

import streamlit as st

from app.state import STATE

st.set_page_config(page_title="SPY-DER", layout="wide")
st.title("SPY-DER Dashboard")

snap = STATE.snapshot()
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("SPY Price", f"{snap['quote'].last:.2f}")
col2.metric("Expected Move", f"{snap['expected_move']:.2f}")
col3.metric("Gamma Flip", "N/A" if snap['gamma_flip'] is None else f"{snap['gamma_flip']:.2f}")
col4.metric("Call Wall", f"{snap['call_wall']:.2f}")
col5.metric("Put Wall", f"{snap['put_wall']:.2f}")

st.subheader("Market-state probabilities")
st.json(snap["market_state"].probabilities)

st.subheader("Ranked strategies")
rows = [
    {
        "name": c.name,
        "max_loss": c.max_loss,
        "expected_return_on_risk": c.expected_return_on_risk,
        "liquidity": c.liquidity,
        "rejection_reasons": ", ".join(c.rejection_reasons),
    }
    for c in snap["ranked"]
]
st.dataframe(rows)

st.subheader("Paper positions")
st.dataframe([p.model_dump() for p in STATE.broker.positions.values()])

st.subheader("Kill switch")
enabled = st.toggle("Enable kill switch", value=STATE.broker.kill_switch.enabled)
STATE.broker.kill_switch.set(enabled)
st.write({"enabled": STATE.broker.kill_switch.enabled})
