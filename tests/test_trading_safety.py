from types import SimpleNamespace

import MetaTrader5 as mt5
import pytest

from trading.groq_signal import _validate_setup, classify_structure
from trading.mt5_demo import (
    PositionSizingError,
    _calculate_money_amounts,
    calculate_volume_for_risk,
    execute_demo_signal,
    reconcile_pending_executions,
)
from trading.groq_signal import Signal


@pytest.fixture(autouse=True)
def clear_execution_state(monkeypatch, tmp_path):
    import trading.mt5_demo as mt5_demo
    monkeypatch.setenv(
        "DEMO_PENDING_STATE_PATH", str(tmp_path / "pending_executions.sqlite3")
    )
    mt5_demo._PENDING_EXECUTIONS.clear()
    mt5_demo._IN_FLIGHT_SYMBOLS.clear()
    yield
    mt5_demo._PENDING_EXECUTIONS.clear()
    mt5_demo._IN_FLIGHT_SYMBOLS.clear()


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
        "filling_mode": 2,
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


def _execution_signal():
    return Signal(
        "SELL", 3878.36, 3885.0, 3865.0, 2.0, "BEARISH", "BREAKDOWN",
        0.0, 8.0, "test", ()
    )


def _mock_execution_environment(monkeypatch, positions):
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
    monkeypatch.setattr("trading.mt5_demo.mt5.positions_get", lambda symbol: positions)
    monkeypatch.setattr("trading.mt5_demo.mt5.symbol_info", lambda symbol: _symbol())
    monkeypatch.setattr(
        "trading.mt5_demo.mt5.symbol_info_tick",
        lambda symbol: SimpleNamespace(bid=3878.36, ask=3878.46),
    )
    monkeypatch.setattr("trading.mt5_demo.mt5.symbol_select", lambda symbol, value: True)
    monkeypatch.setattr(
        "trading.mt5_demo.mt5.order_check",
        lambda request: SimpleNamespace(retcode=0),
    )
    monkeypatch.setattr("trading.mt5_demo.mt5.shutdown", lambda: None)


def test_spread_limit_blocks_order(monkeypatch):
    _mock_execution_environment(monkeypatch, [])
    monkeypatch.setattr(
        "trading.mt5_demo.mt5.symbol_info_tick",
        lambda symbol: SimpleNamespace(bid=3878.0, ask=3879.0),
    )
    monkeypatch.setenv("MAX_DEMO_SPREAD", "0.5")
    monkeypatch.setattr("trading.mt5_demo.mt5.order_send", lambda request: pytest.fail())
    result = execute_demo_signal("XAUUSD", _execution_signal())
    assert "spread" in result


def test_unsupported_filling_mode_blocks_order(monkeypatch):
    _mock_execution_environment(monkeypatch, [])
    monkeypatch.setattr(
        "trading.mt5_demo.mt5.symbol_info",
        lambda symbol: _symbol(filling_mode=8),
    )
    monkeypatch.setattr("trading.mt5_demo.mt5.order_send", lambda request: pytest.fail())
    result = execute_demo_signal("XAUUSD", _execution_signal())
    assert "unsupported broker filling mode" in result


def test_successful_order_requires_position_or_history_evidence(monkeypatch):
    _mock_execution_environment(monkeypatch, [])
    position = SimpleNamespace(
        ticket=77, order=10, symbol="XAUUSD", magic=26001, price_open=3878.36
    )
    calls = {"positions": 0, "send": 0, "request": None}

    def positions_get(symbol):
        calls["positions"] += 1
        return [] if calls["positions"] < 4 else [position]

    def order_send(request):
        calls["send"] += 1
        calls["request"] = request
        return SimpleNamespace(
            retcode=mt5.TRADE_RETCODE_DONE, order=10, deal=11, price=3878.36
        )

    import trading.mt5_demo as mt5_demo
    monkeypatch.setattr(mt5_demo.mt5, "positions_get", positions_get)
    monkeypatch.setattr(mt5_demo.mt5, "order_send", order_send)
    monkeypatch.setattr(mt5_demo.mt5, "history_orders_get", lambda **kwargs: [])
    monkeypatch.setattr(mt5_demo.mt5, "history_deals_get", lambda **kwargs: [])
    result = execute_demo_signal("XAUUSD", _execution_signal())
    assert result.startswith("ORDER_FILLED")
    assert calls["send"] == 1
    assert calls["request"]["volume"] == 0.02
    assert calls["request"]["type_filling"] == mt5.ORDER_FILLING_IOC
    assert calls["request"]["deviation"] == 20


def test_buy_uses_exact_fixed_volume(monkeypatch):
    _mock_execution_environment(monkeypatch, [])
    import trading.mt5_demo as mt5_demo
    captured = {}

    def order_send(request):
        captured.update(request)
        return SimpleNamespace(retcode=mt5.TRADE_RETCODE_PLACED, order=12, deal=0)

    monkeypatch.setattr(mt5_demo.mt5, "order_send", order_send)
    signal = Signal(
        "BUY", 3878.46, 3870.0, 3895.38, 2.0, "BULLISH", "BREAKOUT",
        8.0, 0.0, "test", ()
    )
    assert execute_demo_signal("XAUUSD", signal).startswith("ORDER_SENT")
    assert captured["volume"] == 0.02


def test_fixed_mode_does_not_use_dynamic_risk_or_max_volume(monkeypatch):
    _mock_execution_environment(monkeypatch, [])
    import trading.mt5_demo as mt5_demo
    monkeypatch.setenv("DEMO_VOLUME_MODE", "fixed")
    monkeypatch.setenv("DEMO_FIXED_VOLUME", "0.02")
    monkeypatch.setenv("MAX_DEMO_VOLUME", "0.01")
    monkeypatch.setenv("DEMO_MAX_RISK_AMOUNT", "1000")
    captured = {}
    monkeypatch.setattr(
        mt5_demo.mt5,
        "order_send",
        lambda request: (
            captured.update(request)
            or SimpleNamespace(retcode=mt5.TRADE_RETCODE_PLACED, order=13, deal=0)
        ),
    )
    assert execute_demo_signal("XAUUSD", _execution_signal()).startswith("ORDER_SENT")
    assert captured["volume"] == 0.02


def test_dynamic_volume_mode_is_rejected_without_silent_fallback(monkeypatch):
    _mock_execution_environment(monkeypatch, [])
    import trading.mt5_demo as mt5_demo
    monkeypatch.setenv("DEMO_VOLUME_MODE", "risk")
    monkeypatch.setattr("trading.mt5_demo.mt5.order_send", lambda request: pytest.fail())
    assert "fixed mode is required" in execute_demo_signal(
        "XAUUSD", _execution_signal()
    )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"volume_step": 0.03}, "incompatible"),
        ({"volume_min": 0.03}, "below broker minimum"),
        ({"volume_max": 0.01}, "exceeds broker maximum"),
    ],
)
def test_fixed_volume_rejects_invalid_broker_constraints(
    monkeypatch, overrides, message
):
    _mock_execution_environment(monkeypatch, [])
    monkeypatch.setattr(
        "trading.mt5_demo.mt5.symbol_info",
        lambda symbol: _symbol(**overrides),
    )
    monkeypatch.setattr("trading.mt5_demo.mt5.order_send", lambda request: pytest.fail())
    assert message in execute_demo_signal("XAUUSD", _execution_signal())


def test_fixed_volume_risk_limit_blocks_trade(monkeypatch):
    _mock_execution_environment(monkeypatch, [])
    monkeypatch.setenv("DEMO_MAX_RISK_AMOUNT", "1.00")
    monkeypatch.setattr("trading.mt5_demo.mt5.order_send", lambda request: pytest.fail())
    assert "potential loss exceeds risk limit" in execute_demo_signal(
        "XAUUSD", _execution_signal()
    )


def test_money_calculation_uses_mt5_for_buy_and_sell(monkeypatch):
    import trading.mt5_demo as mt5_demo
    calls = []

    def order_calc_profit(order_type, symbol, volume, entry, target):
        calls.append((order_type, symbol, volume, entry, target))
        is_buy = order_type == mt5.ORDER_TYPE_BUY
        is_loss = target < entry if is_buy else target > entry
        return -12.34 if is_loss else 24.68

    monkeypatch.setattr(mt5_demo.mt5, "order_calc_profit", order_calc_profit)
    buy = _calculate_money_amounts(
        "XAUUSD", "BUY", 0.02, 100.0, 90.0, 120.0, _symbol()
    )
    sell = _calculate_money_amounts(
        "XAUUSD", "SELL", 0.02, 100.0, 110.0, 80.0, _symbol()
    )
    assert buy == (12.34, 24.68)
    assert sell == (12.34, 24.68)
    assert len(calls) == 4


def test_invalid_sl_tp_is_rejected_before_order(monkeypatch):
    _mock_execution_environment(monkeypatch, [])
    monkeypatch.setattr("trading.mt5_demo.mt5.order_send", lambda request: pytest.fail())
    invalid = Signal(
        "BUY", 3878.46, 3880.0, 3890.0, 2.0, "BULLISH", "BREAKOUT",
        8.0, 0.0, "test", ()
    )
    assert "BUY requires stop_loss" in execute_demo_signal("XAUUSD", invalid)


def test_rejected_order_is_not_reported_as_sent(monkeypatch):
    _mock_execution_environment(monkeypatch, [])
    import trading.mt5_demo as mt5_demo
    monkeypatch.setattr(
        mt5_demo.mt5,
        "order_send",
        lambda request: SimpleNamespace(retcode=10016, comment="invalid stops"),
    )
    result = execute_demo_signal("XAUUSD", _execution_signal())
    assert result.startswith("ORDER_REJECTED")


def test_placed_order_is_reported_as_sent_not_filled(monkeypatch):
    _mock_execution_environment(monkeypatch, [])
    import trading.mt5_demo as mt5_demo
    monkeypatch.setattr(
        mt5_demo.mt5,
        "order_send",
        lambda request: SimpleNamespace(
            retcode=mt5.TRADE_RETCODE_PLACED, order=10, comment="placed"
        ),
    )
    result = execute_demo_signal("XAUUSD", _execution_signal())
    assert result.startswith("ORDER_SENT")
    assert "pending execution" in result


def test_missing_execution_evidence_is_unconfirmed(monkeypatch):
    _mock_execution_environment(monkeypatch, [])
    import trading.mt5_demo as mt5_demo
    monkeypatch.setattr(
        mt5_demo.mt5,
        "order_send",
        lambda request: SimpleNamespace(
            retcode=mt5.TRADE_RETCODE_DONE, order=10, deal=11, price=3878.36
        ),
    )
    monkeypatch.setattr(mt5_demo.mt5, "history_orders_get", lambda **kwargs: [])
    monkeypatch.setattr(mt5_demo.mt5, "history_deals_get", lambda **kwargs: [])
    result = execute_demo_signal("XAUUSD", _execution_signal())
    assert result.startswith("EXECUTION_UNCONFIRMED")


def test_deal_history_is_execution_evidence_without_position(monkeypatch):
    _mock_execution_environment(monkeypatch, [])
    import trading.mt5_demo as mt5_demo
    monkeypatch.setattr(
        mt5_demo.mt5,
        "order_send",
        lambda request: SimpleNamespace(
            retcode=mt5.TRADE_RETCODE_DONE, order=10, deal=11, price=3878.36
        ),
    )
    monkeypatch.setattr(mt5_demo.mt5, "history_orders_get", lambda **kwargs: [])
    monkeypatch.setattr(
        mt5_demo.mt5,
        "history_deals_get",
        lambda **kwargs: [
            SimpleNamespace(
                ticket=11, order=10, symbol="XAUUSD", magic=26001, price=3878.36
            )
        ],
    )
    result = execute_demo_signal("XAUUSD", _execution_signal())
    assert result.startswith("ORDER_FILLED")


def test_partial_execution_is_reported_with_evidence(monkeypatch):
    _mock_execution_environment(monkeypatch, [])
    import trading.mt5_demo as mt5_demo
    position = SimpleNamespace(
        ticket=88, order=10, symbol="XAUUSD", magic=26001, price_open=3878.36
    )
    calls = {"positions": 0}

    def positions_get(symbol):
        calls["positions"] += 1
        return [] if calls["positions"] < 4 else [position]

    monkeypatch.setattr(mt5_demo.mt5, "positions_get", positions_get)
    monkeypatch.setattr(
        mt5_demo.mt5,
        "order_send",
        lambda request: SimpleNamespace(
            retcode=mt5.TRADE_RETCODE_DONE_PARTIAL, order=10, deal=11, price=3878.36
        ),
    )
    monkeypatch.setattr(mt5_demo.mt5, "history_orders_get", lambda **kwargs: [])
    monkeypatch.setattr(mt5_demo.mt5, "history_deals_get", lambda **kwargs: [])
    result = execute_demo_signal("XAUUSD", _execution_signal())
    assert result.startswith("ORDER_FILLED (partial)")


def test_existing_position_prevents_duplicate_order(monkeypatch):
    _mock_execution_environment(monkeypatch, [SimpleNamespace(ticket=1)])
    monkeypatch.setattr("trading.mt5_demo.mt5.order_send", lambda request: pytest.fail())
    result = execute_demo_signal("XAUUSD", _execution_signal())
    assert "already open" in result


def test_reconnect_failure_blocks_without_submitting(monkeypatch):
    monkeypatch.setenv("ENABLE_DEMO_TRADING", "1")
    calls = {"initialize": 0, "send": 0}

    def initialize():
        calls["initialize"] += 1
        return False

    monkeypatch.setattr("trading.mt5_demo.mt5.initialize", initialize)
    monkeypatch.setattr("trading.mt5_demo.mt5.shutdown", lambda: None)
    monkeypatch.setattr("trading.mt5_demo.mt5.order_send", lambda request: calls.update(send=1))
    result = execute_demo_signal("XAUUSD", _execution_signal())
    assert result.startswith("BLOCKED: MT5 initialization failed")
    assert calls == {"initialize": 2, "send": 0}


def test_unknown_account_mode_fails_closed(monkeypatch):
    _mock_execution_environment(monkeypatch, [])
    monkeypatch.setattr(
        "trading.mt5_demo.mt5.account_info",
        lambda: SimpleNamespace(trade_allowed=True, trade_mode=99, equity=100000),
    )
    monkeypatch.setattr("trading.mt5_demo.mt5.order_send", lambda request: pytest.fail())
    assert "verified Demo" in execute_demo_signal("XAUUSD", _execution_signal())


def test_missing_account_mode_fails_closed(monkeypatch):
    _mock_execution_environment(monkeypatch, [])
    monkeypatch.setattr(
        "trading.mt5_demo.mt5.account_info",
        lambda: SimpleNamespace(trade_allowed=True, equity=100000),
    )
    monkeypatch.setattr("trading.mt5_demo.mt5.order_send", lambda request: pytest.fail())
    assert "verified Demo" in execute_demo_signal("XAUUSD", _execution_signal())


def test_zero_filling_mode_fails_closed(monkeypatch):
    _mock_execution_environment(monkeypatch, [])
    monkeypatch.setattr(
        "trading.mt5_demo.mt5.symbol_info",
        lambda symbol: _symbol(filling_mode=0),
    )
    monkeypatch.setattr("trading.mt5_demo.mt5.order_send", lambda request: pytest.fail())
    assert "filling modes" in execute_demo_signal("XAUUSD", _execution_signal())


def test_unrelated_position_after_submission_is_not_fill_evidence(monkeypatch):
    _mock_execution_environment(monkeypatch, [])
    import trading.mt5_demo as mt5_demo
    calls = {"positions": 0, "send": 0}

    def positions_get(symbol):
        calls["positions"] += 1
        if calls["positions"] <= 3:
            return []
        return [SimpleNamespace(
            ticket=77, order=999, symbol="XAUUSD", magic=26001
        )]

    monkeypatch.setattr(mt5_demo.mt5, "positions_get", positions_get)
    monkeypatch.setattr(
        mt5_demo.mt5,
        "order_send",
        lambda request: (calls.update(send=calls["send"] + 1) or SimpleNamespace(
            retcode=mt5.TRADE_RETCODE_DONE, order=10, deal=11, price=3878.36
        )),
    )
    monkeypatch.setattr(mt5_demo.mt5, "history_orders_get", lambda **kwargs: [])
    monkeypatch.setattr(mt5_demo.mt5, "history_deals_get", lambda **kwargs: [])
    result = execute_demo_signal("XAUUSD", _execution_signal())
    assert result.startswith("EXECUTION_UNCONFIRMED")
    assert calls["send"] == 1


def test_mismatching_deal_order_is_not_fill_evidence(monkeypatch):
    _mock_execution_environment(monkeypatch, [])
    import trading.mt5_demo as mt5_demo
    monkeypatch.setattr(
        mt5_demo.mt5,
        "order_send",
        lambda request: SimpleNamespace(
            retcode=mt5.TRADE_RETCODE_DONE, order=10, deal=11, price=3878.36
        ),
    )
    monkeypatch.setattr(mt5_demo.mt5, "history_orders_get", lambda **kwargs: [])
    monkeypatch.setattr(
        mt5_demo.mt5,
        "history_deals_get",
        lambda **kwargs: [SimpleNamespace(
            ticket=11, order=999, symbol="XAUUSD", magic=26001
        )],
    )
    assert execute_demo_signal("XAUUSD", _execution_signal()).startswith(
        "EXECUTION_UNCONFIRMED"
    )


def test_pending_order_is_not_resubmitted(monkeypatch):
    _mock_execution_environment(monkeypatch, [])
    import trading.mt5_demo as mt5_demo
    calls = {"send": 0}

    def order_send(request):
        calls["send"] += 1
        return SimpleNamespace(retcode=mt5.TRADE_RETCODE_PLACED, order=10, deal=0)

    monkeypatch.setattr(mt5_demo.mt5, "order_send", order_send)
    first = execute_demo_signal("XAUUSD", _execution_signal())
    second = execute_demo_signal("XAUUSD", _execution_signal())
    assert first.startswith("ORDER_SENT")
    assert "confirmation is pending" in second
    assert calls["send"] == 1


def test_pending_execution_reconciles_on_later_cycle(monkeypatch):
    _mock_execution_environment(monkeypatch, [])
    import trading.mt5_demo as mt5_demo
    monkeypatch.setattr(
        mt5_demo.mt5,
        "order_send",
        lambda request: SimpleNamespace(retcode=mt5.TRADE_RETCODE_PLACED, order=10, deal=0),
    )
    assert execute_demo_signal("XAUUSD", _execution_signal()).startswith("ORDER_SENT")
    calls = {"positions": 0}
    position = SimpleNamespace(
        ticket=77, order=10, symbol="XAUUSD", magic=26001, price_open=3878.36
    )

    def positions_get(symbol):
        calls["positions"] += 1
        return [] if calls["positions"] == 1 else [position]

    monkeypatch.setattr(mt5_demo.mt5, "positions_get", positions_get)
    monkeypatch.setattr(mt5_demo.mt5, "history_orders_get", lambda **kwargs: [])
    monkeypatch.setattr(mt5_demo.mt5, "history_deals_get", lambda **kwargs: [])
    assert reconcile_pending_executions("XAUUSD").startswith(
        "EXECUTION_UNCONFIRMED"
    )
    assert reconcile_pending_executions("XAUUSD").startswith("ORDER_FILLED")


def test_partial_pending_execution_reconciles_from_matching_deal(monkeypatch):
    _mock_execution_environment(monkeypatch, [])
    import trading.mt5_demo as mt5_demo
    monkeypatch.setattr(
        mt5_demo.mt5,
        "order_send",
        lambda request: SimpleNamespace(
            retcode=mt5.TRADE_RETCODE_DONE_PARTIAL, order=10, deal=11
        ),
    )
    monkeypatch.setattr(mt5_demo.mt5, "history_orders_get", lambda **kwargs: [])
    monkeypatch.setattr(mt5_demo.mt5, "history_deals_get", lambda **kwargs: [])
    assert execute_demo_signal("XAUUSD", _execution_signal()).startswith(
        "EXECUTION_UNCONFIRMED"
    )
    monkeypatch.setattr(
        mt5_demo.mt5,
        "history_deals_get",
        lambda **kwargs: [SimpleNamespace(
            ticket=11, order=10, symbol="XAUUSD", magic=26001, position_id=77,
            volume=0.01
        )],
    )
    assert reconcile_pending_executions("XAUUSD").startswith("ORDER_FILLED (partial)")


def test_pending_execution_survives_memory_restart(monkeypatch, tmp_path):
    _mock_execution_environment(monkeypatch, [])
    import trading.mt5_demo as mt5_demo
    monkeypatch.setattr(
        mt5_demo.mt5,
        "order_send",
        lambda request: SimpleNamespace(retcode=mt5.TRADE_RETCODE_PLACED, order=10, deal=0),
    )
    assert execute_demo_signal("XAUUSD", _execution_signal()).startswith("ORDER_SENT")
    state_path = tmp_path / "pending_executions.sqlite3"
    assert state_path.exists()
    mt5_demo._PENDING_EXECUTIONS.clear()
    monkeypatch.setattr(mt5_demo.mt5, "history_orders_get", lambda **kwargs: [])
    monkeypatch.setattr(mt5_demo.mt5, "history_deals_get", lambda **kwargs: [])
    assert reconcile_pending_executions("XAUUSD").startswith(
        "EXECUTION_UNCONFIRMED"
    )


def test_expired_pending_execution_blocks_resubmission(monkeypatch):
    _mock_execution_environment(monkeypatch, [])
    import trading.mt5_demo as mt5_demo
    monkeypatch.setenv("PENDING_EXECUTION_TIMEOUT_SECONDS", "0")
    calls = {"send": 0}

    def order_send(request):
        calls["send"] += 1
        return SimpleNamespace(retcode=mt5.TRADE_RETCODE_PLACED, order=10, deal=0)

    monkeypatch.setattr(mt5_demo.mt5, "order_send", order_send)
    assert execute_demo_signal("XAUUSD", _execution_signal()).startswith("ORDER_SENT")
    assert reconcile_pending_executions("XAUUSD").startswith("EXECUTION_EXPIRED")
    assert "manual reconciliation" in execute_demo_signal(
        "XAUUSD", _execution_signal()
    )
    assert calls["send"] == 1


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
