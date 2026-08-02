from analytics.black_scholes import delta, gamma, price, vanna, vega
from data.schemas import OptionType


def test_put_call_parity() -> None:
    s, k, t, r, v = 550.0, 550.0, 30 / 365, 0.01, 0.2
    call = price(s, k, t, r, v, OptionType.CALL)
    put = price(s, k, t, r, v, OptionType.PUT)
    assert abs((call - put) - (s - k * (2.718281828459045 ** (-r * t)))) < 1.0


def test_greeks_sanity() -> None:
    s, k, t, r, v = 550.0, 550.0, 30 / 365, 0.01, 0.2
    assert 0 < delta(s, k, t, r, v, OptionType.CALL) < 1
    assert gamma(s, k, t, r, v) > 0
    assert vega(s, k, t, r, v) > 0
    assert abs(vanna(s, k, t, r, v)) < 10
