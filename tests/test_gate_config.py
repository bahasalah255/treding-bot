from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from trading.gate_config import (
    GateAccountState,
    GateConfig,
    GateMarketState,
    evaluate_entry_gate,
)


CONFIG = GateConfig(
    max_daily_loss_usd=100,
    max_total_loss_usd=500,
    max_spread=0.5,
    session_enabled=True,
    session_utc_start="09:00",
    session_utc_end="17:00",
)


def signal(**overrides):
    values = {
        "action": "BUY",
        "local_buy_score": 90.0,
        "local_sell_score": 40.0,
        "risk_reward": 2.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def market(**overrides):
    values = {
        "spread": 0.2,
        "utc_now": datetime(2026, 11, 16, 10, tzinfo=timezone.utc),
        "candle_id": "c1",
        "local_direction": "BUY",
    }
    values.update(overrides)
    return GateMarketState(**values)


def account(**overrides):
    return GateAccountState(**overrides)


def test_entry_gate_accepts_a_valid_signal():
    result = evaluate_entry_gate(signal(), market(), account(), CONFIG)

    assert result.passed is True
    assert result.failed_rules == ()


@pytest.mark.parametrize(
    ("signal_overrides", "market_overrides", "account_overrides", "expected"),
    [
        ({"local_buy_score": 60}, {}, {}, "local_score"),
        ({"local_sell_score": 70}, {}, {}, "score_gap"),
        ({"risk_reward": 1.99}, {}, {}, "risk_reward"),
        ({}, {"local_direction": "SELL"}, {}, "direction"),
        ({}, {"utc_now": datetime(2026, 11, 16, 20, tzinfo=timezone.utc)}, {}, "outside"),
        ({}, {"spread": 0.51}, {}, "spread"),
        ({}, {}, {"daily_loss_usd": 100}, "max_daily_loss_usd"),
        ({}, {}, {"consecutive_losses": 2}, "max_consecutive_losses"),
        ({}, {}, {"trades_today": 3}, "max_trades_per_day"),
        ({}, {}, {"total_loss_usd": 500}, "max_total_loss_usd"),
        ({}, {}, {"kill_switch": True}, "kill_switch_active"),
        ({}, {}, {"open_positions": 1}, "position_already_open"),
        ({}, {}, {"entered_candle": "c1"}, "candle_already_entered"),
    ],
)
def test_entry_gate_reports_each_failed_rule(
    signal_overrides, market_overrides, account_overrides, expected
):
    result = evaluate_entry_gate(
        signal(**signal_overrides),
        market(**market_overrides),
        account(**account_overrides),
        CONFIG,
    )

    assert result.passed is False
    assert any(expected in rule for rule in result.failed_rules)
