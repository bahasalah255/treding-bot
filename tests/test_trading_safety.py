from types import SimpleNamespace

import pytest

from trading.groq_signal import Signal, _validate_setup, classify_structure
from trading.mt5_demo import (
    ExecutionError,
    _validate_market_levels,
    loss_per_lot,
    require_demo_account,
)
from trading.risk import SymbolSpec, volume_for_risk

SPEC = SymbolSpec(
    point=0.01,
    volume_min=0.01,
    volume_max=100.0,
    volume_step=0.01,
)


def _symbol(**overrides):
    values = {
        "trade_tick_size": 0.01,
        "trade_tick_value": 0.1,
        "trade_tick_value_loss": 0.1,
        "trade_contract_size": 100.0,
        "currency_profit": "USD",
        "currency_margin": "USD",
        "volume_min": 0.01,
        "volume_max": 100.0,
        "volume_step": 0.01,
        "point": 0.01,
        "trade_stops_level": 0,
        "trade_freeze_level": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _account(**overrides):
    values = {
        "login": 10013615454,
        "server": "MetaQuotes-Demo",
        "currency": "USD",
        "balance": 1000.0,
        "equity": 1000.0,
        "trade_mode": 0,
        "trade_expert": True,
        "leverage": 100,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _terminal(**overrides):
    values = {
        "ACCOUNT_TRADE_MODE_DEMO": 0,
        "ORDER_TYPE_BUY": 0,
        "SYMBOL_TRADE_MODE_DISABLED": 0,
        "account_info": lambda: _account(),
        "symbol_info": lambda symbol: _symbol(),
        "order_calc_profit": lambda *args: -664.0,
        "last_error": lambda: (0, "ok"),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_sizing_uses_order_calc_profit_instead_of_stale_tick_fields():
    # The broker reports trade_tick_value=0.1, implying 66.4 per lot over a
    # 6.64 move, but its own order_calc_profit says 664.0. The larger, broker
    # authoritative value must win or the bot would size 10x too large.
    per_lot, tick_estimate = loss_per_lot(
        _terminal(), "XAUUSD", entry=3885.0, stop_loss=3878.36, spec=SPEC
    )

    assert per_lot == pytest.approx(664.0)
    assert tick_estimate == pytest.approx(66.4)
    volume = volume_for_risk(100_000.0, 0.005, per_lot, SPEC)
    assert volume == pytest.approx(0.75)


def test_sizing_falls_back_to_contract_size_when_profit_calc_is_unavailable():
    terminal = _terminal(order_calc_profit=lambda *args: None)

    per_lot, _ = loss_per_lot(terminal, "XAUUSD", 3885.0, 3878.36, SPEC)

    assert per_lot == pytest.approx(6.64 * 100)


def test_require_demo_account_refuses_a_real_account():
    terminal = _terminal(account_info=lambda: _account(trade_mode=2))

    with pytest.raises(ExecutionError, match="not a demo account"):
        require_demo_account(terminal)


def test_require_demo_account_refuses_a_disabled_expert():
    terminal = _terminal(account_info=lambda: _account(trade_expert=False))

    with pytest.raises(ExecutionError, match="algo trading is disabled"):
        require_demo_account(terminal)


def test_require_demo_account_accepts_demo_and_reports_identity():
    snapshot = require_demo_account(_terminal())

    assert snapshot.login == 10013615454
    assert snapshot.server == "MetaQuotes-Demo"
    assert snapshot.balance == pytest.approx(1000.0)


def test_structure_labels_are_standard():
    assert classify_structure([100, 110], [90, 95]) == ("HH", "HL")
    assert classify_structure([110, 100], [95, 90]) == ("LH", "LL")
    assert classify_structure([100, 90], [90, 95]) == ("LH", "HL")


def test_sell_target_below_support_requires_breakdown_retest():
    evidence = {
        "atr14": 3.0,
        "support": 3867.49,
        "resistance": 3905.86,
        "range": {"middle": False},
        "bos": {"down": False, "up": False},
    }
    assert _validate_setup("SELL", 3878.36, 3885.0, 3865.0, evidence)


def test_buy_and_sell_level_direction_is_checked():
    info = _symbol()
    assert _validate_market_levels(info, "BUY", 100, 99, 101) is None
    assert _validate_market_levels(info, "SELL", 100, 101, 99) is None
    assert _validate_market_levels(info, "SELL", 100, 99, 101) is not None


def test_freeze_level_is_honoured_as_a_minimum_distance():
    info = _symbol(trade_freeze_level=25)

    assert _validate_market_levels(info, "BUY", 100.0, 99.9, 101.0) is not None
    assert _validate_market_levels(info, "BUY", 100.0, 99.7, 100.3) is None


def test_signal_from_groq_carries_structure_context():
    signal = Signal(
        "SELL", 3878.36, 3885.0, 3865.0, 2.0, "BEARISH", "BREAKDOWN",
        0.0, 8.0, "test", (),
    )

    assert signal.action == "SELL"
    assert signal.risk_reward == pytest.approx(2.0)
    assert signal.rejection_reasons == ()
