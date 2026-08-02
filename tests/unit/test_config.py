"""Configuration and deployment-mode tests (spec sections 33, 40)."""

from __future__ import annotations

import pytest

from spy_der.config.settings import (
    CONFIG_DIR,
    LiveGates,
    LiveTradingDisabled,
    Mode,
    Settings,
    load_settings,
)


@pytest.mark.parametrize("mode", list(Mode))
def test_every_mode_loads(mode):
    settings = load_settings(mode)
    assert settings.mode is mode
    assert settings.starting_equity > 0.0
    assert settings.risk.base_risk_fraction > 0.0


@pytest.mark.parametrize("mode", list(Mode))
def test_every_mode_has_a_config_file(mode):
    assert (CONFIG_DIR / f"{mode.value}.yaml").exists()


def test_only_paper_and_live_may_submit_orders():
    """Research, backtest, and shadow must never reach the broker (spec 40)."""
    assert Settings(mode=Mode.PAPER).may_submit_orders
    assert Settings(mode=Mode.LIVE).may_submit_orders
    for mode in (Mode.RESEARCH, Mode.BACKTEST, Mode.SHADOW):
        assert not Settings(mode=mode).may_submit_orders


def test_non_live_modes_never_raise_on_the_live_gate():
    for mode in (Mode.RESEARCH, Mode.BACKTEST, Mode.PAPER, Mode.SHADOW):
        Settings(mode=mode).assert_live_allowed()  # must not raise


def test_live_is_refused_with_default_gates():
    with pytest.raises(LiveTradingDisabled) as excinfo:
        Settings(mode=Mode.LIVE).assert_live_allowed()
    message = str(excinfo.value)
    assert "paper trading" in message
    assert "shadow execution" in message


def test_live_is_still_refused_once_gates_are_met():
    """Even a fully gated deployment has no live adapter to trade through."""
    gates = LiveGates(
        paper_months=12.0,
        completed_paper_trades=400,
        shadow_completed=True,
        reconciliation_tested=True,
        unresolved_risk_defects=0,
        operator_authorized=True,
    )
    assert gates.satisfied
    with pytest.raises(LiveTradingDisabled, match="no live broker adapter"):
        Settings(mode=Mode.LIVE, live_gates=gates).assert_live_allowed()


def test_gate_thresholds_match_the_spec():
    """Spec 40 phase 2: six months and 250 completed trades."""
    assert LiveGates.REQUIRED_MONTHS == 6.0
    assert LiveGates.REQUIRED_TRADES == 250


def test_partial_gates_are_reported_individually():
    gates = LiveGates(paper_months=7.0, completed_paper_trades=300)
    unmet = gates.unmet()
    assert not any("paper trading" in u for u in unmet)
    assert not any("paper trades" in u for u in unmet)
    assert any("shadow" in u for u in unmet)


def test_unknown_config_keys_are_rejected(tmp_path):
    """A typo in a config file must fail loudly, not be silently ignored."""
    path = tmp_path / "bad.yaml"
    path.write_text("risk:\n  not_a_real_setting: 1.0\n")
    with pytest.raises(ValueError, match="unknown risk settings"):
        load_settings(Mode.PAPER, path=path)


def test_config_overrides_are_applied(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text("starting_equity: 250000.0\nrisk:\n  base_risk_fraction: 0.0025\n")
    settings = load_settings(Mode.PAPER, path=path)
    assert settings.starting_equity == pytest.approx(250_000.0)
    assert settings.risk.base_risk_fraction == pytest.approx(0.0025)


def test_live_config_is_the_most_conservative():
    live = load_settings(Mode.LIVE)
    paper = load_settings(Mode.PAPER)
    assert live.risk.base_risk_fraction <= paper.risk.base_risk_fraction
