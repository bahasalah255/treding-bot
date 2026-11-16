"""Tests for the daemon-level safety fixes: freshness, deal accounting, levels."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from data.timestamp_validation import TimestampValidationError
from trading import daemon
from trading.groq_signal import Signal
from trading.journal import Journal, read_events
from trading.risk import SymbolSpec, validate_proposal
from trading.state import RuntimeState

SPEC = SymbolSpec(
    point=0.01,
    volume_min=0.01,
    volume_max=100.0,
    volume_step=0.01,
)


def _frame_ending(minutes_ago: int) -> pd.DataFrame:
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    index = pd.date_range(
        end=now - timedelta(minutes=minutes_ago), periods=30, freq="5min"
    )
    closes = pd.Series(range(30), index=index, dtype=float) + 3800.0
    return pd.DataFrame(
        {
            "open": closes - 0.5,
            "high": closes + 1.0,
            "low": closes - 1.0,
            "close": closes,
        },
        index=index,
    )


class _RatesTerminal:
    TIMEFRAME_M5 = 5

    def copy_rates_from_pos(self, symbol, timeframe, position, count):
        return list(range(30))

    def last_error(self):
        return (0, "ok")


def test_stale_candles_are_refused(monkeypatch):
    monkeypatch.setattr(daemon, "select_closed_m5_candles", lambda *a, **k: _frame_ending(60))
    monkeypatch.setattr(daemon.time, "sleep", lambda seconds: None)

    with pytest.raises(TimestampValidationError, match="stale"):
        daemon.read_closed_bars(_RatesTerminal(), "XAUUSD", "M5", 30)


def test_fresh_candles_are_accepted(monkeypatch):
    frame = _frame_ending(5)
    monkeypatch.setattr(daemon, "select_closed_m5_candles", lambda *a, **k: frame)

    result = daemon.read_closed_bars(_RatesTerminal(), "XAUUSD", "M5", 30)

    assert result is frame


def _deal(ticket, *, entry, minutes_ago, profit=0.0, commission=0.0, swap=0.0):
    when = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return SimpleNamespace(
        ticket=ticket,
        position_id=100 + ticket,
        symbol="XAUUSD",
        magic=26001,
        entry=entry,
        volume=0.05,
        price=3850.0,
        profit=profit,
        commission=commission,
        swap=swap,
        time=int(when.timestamp()),
    )


def test_fresh_state_skips_historical_losses_and_entry_commissions(tmp_path, monkeypatch):
    deals = [
        _deal(1, entry=1, minutes_ago=60 * 24 * 3, profit=-50.0),  # last week
        _deal(2, entry=0, minutes_ago=30, commission=-1.0),        # entry deal
        _deal(3, entry=1, minutes_ago=20, profit=-20.0),           # today's loss
    ]
    monkeypatch.setattr("trading.mt5_demo.new_deals", lambda *a, **k: deals)
    journal = Journal(tmp_path, echo=False)
    state = RuntimeState(day=datetime.now(timezone.utc).date().isoformat())
    terminal = SimpleNamespace(DEAL_ENTRY_OUT=1, DEAL_ENTRY_INOUT=2)
    config = SimpleNamespace(
        execution=SimpleNamespace(magic=26001),
        account=SimpleNamespace(symbol="XAUUSD"),
    )

    daemon.journal_new_deals(terminal, config, journal, state)

    assert state.daily_loss_usd == pytest.approx(20.0)
    assert state.total_loss_usd == pytest.approx(20.0)
    assert state.consecutive_losses == 1
    assert state.last_deal_ticket == 3
    counted = {event["ticket"]: event["counted"] for event in read_events(tmp_path)}
    assert counted == {1: False, 2: False, 3: True}


def test_profit_closes_the_consecutive_loss_streak(tmp_path, monkeypatch):
    deals = [
        _deal(1, entry=1, minutes_ago=20, profit=-20.0),
        _deal(2, entry=1, minutes_ago=10, profit=40.0),
    ]
    monkeypatch.setattr("trading.mt5_demo.new_deals", lambda *a, **k: deals)
    journal = Journal(tmp_path, echo=False)
    state = RuntimeState(day=datetime.now(timezone.utc).date().isoformat())
    terminal = SimpleNamespace(DEAL_ENTRY_OUT=1, DEAL_ENTRY_INOUT=2)
    config = SimpleNamespace(
        execution=SimpleNamespace(magic=26001),
        account=SimpleNamespace(symbol="XAUUSD"),
    )

    daemon.journal_new_deals(terminal, config, journal, state)

    assert state.consecutive_losses == 0
    assert state.daily_loss_usd == pytest.approx(20.0)


def _signal(action="BUY", entry=3853.59):
    if action == "BUY":
        return Signal(
            "BUY", entry, entry - 6.0, entry + 12.0, 2.0, "BULLISH", "PULLBACK",
            8.0, 0.0, "test", (),
        )
    return Signal(
        "SELL", entry, entry + 6.0, entry - 12.0, 2.0, "BEARISH", "PULLBACK",
        0.0, 8.0, "test", (),
    )


def test_reward_risk_at_threshold_is_not_rejected_by_float_noise():
    entry = 3853.59
    risk = 6.291200252213271
    stop_loss = entry - risk
    take_profit = entry + risk * 2.0  # exactly 2R before rounding

    # This specific pair lands a few ulps below 2.0 without a tolerance.
    assert abs(take_profit - entry) / abs(entry - stop_loss) < 2.0

    risk_distance, reward = validate_proposal(
        "BUY", entry, stop_loss, take_profit, SPEC, min_reward_risk=2.0
    )
    assert risk_distance > 0 and reward > 0


def test_apply_local_levels_rounds_to_symbol_precision():
    frame = _frame_ending(0).reset_index(drop=True)
    frame.index = pd.date_range(
        end=datetime.now(timezone.utc).replace(second=0, microsecond=0),
        periods=len(frame),
        freq="5min",
    )
    config = SimpleNamespace(
        gate=SimpleNamespace(sl_atr_mult=1.5, min_rr=2.0),
    )
    signal = _signal("BUY", entry=3853.57)

    adjusted = daemon.apply_local_levels(signal, frame, config, digits=2)

    for value in (adjusted.entry_price, adjusted.stop_loss, adjusted.take_profit):
        assert value == round(value, 2)
    assert adjusted.stop_loss < adjusted.entry_price < adjusted.take_profit
    assert adjusted.risk_reward == 2.0


def test_run_once_delegates_to_the_guarded_daemon(monkeypatch):
    """The legacy entry point must not bypass the gate or the freshness guard."""
    from trading import run_once

    captured: dict[str, list[str]] = {}

    def fake_main(argv):
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr(run_once.daemon, "main", fake_main)

    assert run_once.main([]) == 0
    assert "--once" in captured["argv"]
    assert "--dry-run" in captured["argv"]

    assert run_once.main(["--execute"]) == 0
    assert "--once" in captured["argv"]
    assert "--dry-run" not in captured["argv"]
