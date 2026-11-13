from types import SimpleNamespace

import pytest

from trading.groq_signal import _validate_setup, classify_structure
from trading.mt5_demo import (
    PositionSizingError,
    calculate_volume_for_risk,
    execute_demo_signal,
)
from trading.groq_signal import Signal


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


def test_position_sizing_uses_monetary_risk_and_cap():
    volume, estimated_risk = calculate_volume_for_risk(
        _symbol(), entry=3878.36, stop_loss=3885.0, equity=99528.73,
        risk_percent=0.5, max_volume=1.0
    )
    assert volume == 1.0
    assert estimated_risk == pytest.approx(66.4)


def test_position_sizing_rejects_missing_specification():
    with pytest.raises(PositionSizingError, match="trade_tick_size"):
        calculate_volume_for_risk(
            _symbol(trade_tick_size=0), 100, 99, 1000
        )


def test_position_sizing_rejects_minimum_volume_above_risk():
    with pytest.raises(PositionSizingError, match="minimum broker volume"):
        calculate_volume_for_risk(
            _symbol(volume_min=1.0), 100, 99, 100, risk_percent=0.01, max_volume=1
        )


def test_structure_labels_are_standard():
    assert classify_structure([100, 110], [90, 95]) == ("HH", "HL")
    assert classify_structure([110, 100], [95, 90]) == ("LH", "LL")
    assert classify_structure([100, 90], [90, 95]) == ("LH", "HL")


def test_autotrading_disabled_is_blocked(monkeypatch):
    signal = Signal(
        "SELL", 3878.36, 3885.0, 3865.0, 2.0, "BEARISH", "BREAKDOWN",
        0.0, 8.0, "test", ()
    )
    monkeypatch.setenv("ENABLE_DEMO_TRADING", "1")
    monkeypatch.setattr("trading.mt5_demo.mt5.initialize", lambda: True)
    monkeypatch.setattr(
        "trading.mt5_demo.mt5.terminal_info",
        lambda: SimpleNamespace(trade_allowed=False),
    )
    monkeypatch.setattr(
        "trading.mt5_demo.mt5.account_info",
        lambda: SimpleNamespace(trade_allowed=True, trade_mode=0, equity=1000),
    )
    monkeypatch.setattr("trading.mt5_demo.mt5.shutdown", lambda: None)
    result = execute_demo_signal("XAUUSD", signal)
    assert "AutoTrading disabled" in result


def test_failed_order_check_blocks_order(monkeypatch):
    signal = Signal(
        "SELL", 3878.36, 3885.0, 3865.0, 2.0, "BEARISH", "BREAKDOWN",
        0.0, 8.0, "test", ()
    )
    monkeypatch.setenv("ENABLE_DEMO_TRADING", "1")
    monkeypatch.setattr("trading.mt5_demo.mt5.initialize", lambda: True)
    monkeypatch.setattr(
        "trading.mt5_demo.mt5.terminal_info",
        lambda: SimpleNamespace(trade_allowed=True),
    )
    monkeypatch.setattr(
        "trading.mt5_demo.mt5.account_info",
        lambda: SimpleNamespace(trade_allowed=True, trade_mode=0, equity=100000),
    )
    monkeypatch.setattr("trading.mt5_demo.mt5.positions_get", lambda symbol: [])
    monkeypatch.setattr("trading.mt5_demo.mt5.symbol_info", lambda symbol: _symbol())
    monkeypatch.setattr(
        "trading.mt5_demo.mt5.symbol_info_tick",
        lambda symbol: SimpleNamespace(bid=3878.36, ask=3878.46),
    )
    monkeypatch.setattr("trading.mt5_demo.mt5.symbol_select", lambda symbol, value: True)
    monkeypatch.setattr(
        "trading.mt5_demo.mt5.order_check",
        lambda request: SimpleNamespace(retcode=10016, comment="invalid stops"),
    )
    monkeypatch.setattr("trading.mt5_demo.mt5.shutdown", lambda: None)
    result = execute_demo_signal("XAUUSD", signal)
    assert "order_check failed" in result


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
    from trading.mt5_demo import _validate_market_levels

    info = _symbol()
    assert _validate_market_levels(info, "BUY", 100, 99, 101) is None
    assert _validate_market_levels(info, "SELL", 100, 101, 99) is None
    assert _validate_market_levels(info, "SELL", 100, 99, 101) is not None
