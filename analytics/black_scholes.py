from __future__ import annotations

from math import exp, log, sqrt

from scipy.stats import norm

from data.schemas import OptionType


def _d1_d2(spot: float, strike: float, tau: float, rate: float, vol: float) -> tuple[float, float]:
    if tau <= 0 or vol <= 0:
        raise ValueError("tau and vol must be positive")
    d1 = (log(spot / strike) + (rate + 0.5 * vol**2) * tau) / (vol * sqrt(tau))
    d2 = d1 - vol * sqrt(tau)
    return d1, d2


def price(spot: float, strike: float, tau: float, rate: float, vol: float, option_type: OptionType) -> float:
    d1, d2 = _d1_d2(spot, strike, tau, rate, vol)
    if option_type == OptionType.CALL:
        return float(spot * norm.cdf(d1) - strike * exp(-rate * tau) * norm.cdf(d2))
    return float(strike * exp(-rate * tau) * norm.cdf(-d2) - spot * norm.cdf(-d1))


def delta(spot: float, strike: float, tau: float, rate: float, vol: float, option_type: OptionType) -> float:
    d1, _ = _d1_d2(spot, strike, tau, rate, vol)
    return float(norm.cdf(d1) if option_type == OptionType.CALL else norm.cdf(d1) - 1)


def gamma(spot: float, strike: float, tau: float, rate: float, vol: float) -> float:
    d1, _ = _d1_d2(spot, strike, tau, rate, vol)
    return float(norm.pdf(d1) / (spot * vol * sqrt(tau)))


def theta(spot: float, strike: float, tau: float, rate: float, vol: float, option_type: OptionType) -> float:
    d1, d2 = _d1_d2(spot, strike, tau, rate, vol)
    first = -(spot * norm.pdf(d1) * vol) / (2 * sqrt(tau))
    if option_type == OptionType.CALL:
        second = -rate * strike * exp(-rate * tau) * norm.cdf(d2)
    else:
        second = rate * strike * exp(-rate * tau) * norm.cdf(-d2)
    return float(first + second)


def vega(spot: float, strike: float, tau: float, rate: float, vol: float) -> float:
    d1, _ = _d1_d2(spot, strike, tau, rate, vol)
    return float(spot * norm.pdf(d1) * sqrt(tau))


def vanna(spot: float, strike: float, tau: float, rate: float, vol: float) -> float:
    d1, d2 = _d1_d2(spot, strike, tau, rate, vol)
    return float(-norm.pdf(d1) * d2 / vol)


def charm(spot: float, strike: float, tau: float, rate: float, vol: float, option_type: OptionType) -> float:
    d1, d2 = _d1_d2(spot, strike, tau, rate, vol)
    term = norm.pdf(d1) * (2 * rate * tau - d2 * vol * sqrt(tau)) / (2 * tau * vol * sqrt(tau))
    return float(-term if option_type == OptionType.CALL else term)
