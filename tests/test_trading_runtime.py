import json

import pytest

from trading.config import ConfigError, load_config, parse_config
from trading.journal import Journal, read_events
from trading.report import summarize
from trading.state import RuntimeState, StateStore, single_instance


def test_journal_writes_one_json_object_per_line(tmp_path):
    journal = Journal(tmp_path, echo=False)
    journal.write("signal", action="BUY", rationale="breakout")
    journal.write("entry_blocked", reason="spread_too_wide")

    files = list(tmp_path.glob("journal-*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == "signal"


def test_journal_roundtrip_and_summary(tmp_path):
    journal = Journal(tmp_path, echo=False)
    journal.write("signal", action="BUY")
    journal.write("entry_blocked", reason="spread_too_wide")
    journal.write("order_sent", ticket=1, volume=0.1)
    journal.write("deal", position=11, profit=25.0, commission=-0.5, swap=0.0)
    journal.write("deal", position=12, profit=-10.0, commission=-0.5, swap=0.0)
    journal.write("error", stage="signal", error="GroqError")

    summary = summarize(read_events(tmp_path))

    assert summary["signals"] == 1
    assert summary["orders_sent"] == 1
    assert summary["blocked"] == {"spread_too_wide": 1}
    assert summary["errors"] == {"signal:GroqError": 1}
    assert summary["closed_positions"] == 2
    assert summary["winning_positions"] == 1
    assert summary["net_pnl"] == pytest.approx(14.0)


def test_state_roundtrip(tmp_path):
    store = StateStore(tmp_path / "state.json")
    state = RuntimeState()
    state.roll_day("2026-01-01")
    state.record_trade()
    state.begin_candle("2026-01-01T10:00:00+00:00")
    store.save(state)

    loaded = store.load()

    assert loaded.day == "2026-01-01"
    assert loaded.trades_today == 1
    assert loaded.pending_candle == "2026-01-01T10:00:00+00:00"


def test_state_missing_file_is_a_fresh_state(tmp_path):
    state = StateStore(tmp_path / "missing.json").load()

    assert state.processed_candle is None
    assert state.trades_today == 0


def test_state_roll_day_resets_trades_only_on_change():
    state = RuntimeState(day="2026-01-01", trades_today=2)

    assert state.roll_day("2026-01-01") is False
    assert state.trades_today == 2
    assert state.roll_day("2026-01-02") is True
    assert state.trades_today == 0


def test_candle_attempts_reset_for_a_new_candle():
    state = RuntimeState()

    state.begin_candle("c1")
    state.signal_attempts += 1
    state.begin_candle("c1")
    assert state.signal_attempts == 1

    state.begin_candle("c2")
    assert state.signal_attempts == 0


def test_finish_candle_prevents_reprocessing():
    state = RuntimeState()

    state.begin_candle("c1")
    state.finish_candle("c1")

    assert state.processed_candle == "c1"
    assert state.pending_candle is None
    assert state.signal_attempts == 0


def test_single_instance_refuses_a_second_holder(tmp_path):
    lock = tmp_path / "bot.lock"

    with single_instance(lock):
        with pytest.raises(RuntimeError, match="another bot instance"):
            with single_instance(lock):
                pass

    with single_instance(lock):
        pass


def test_repository_demo_config_is_valid():
    config = load_config()

    assert config.account.symbol == "XAUUSD"
    assert config.account.timeframe == "M5"
    assert config.execution.risk_fraction == pytest.approx(0.005)
    assert config.execution.max_daily_loss == pytest.approx(0.02)


def test_config_rejects_an_unsafe_risk_fraction():
    raw = {
        "account": {"symbol": "XAUUSD", "timeframe": "M5"},
        "execution": {
            "magic": 1,
            "risk_per_trade": 0.5,
            "max_daily_loss": 0.02,
            "max_trades_per_day": 3,
            "max_open_positions": 1,
            "max_spread_points": 60,
            "min_reward_risk": 1.0,
            "deviation_points": 20,
            "filling_mode": None,
        },
        "monitor": {"bars": 500, "poll_seconds": 60, "max_signal_attempts": 2, "max_backoff_seconds": 600},
        "model": {"name": "openai/gpt-oss-20b", "context_bars": 100},
    }

    with pytest.raises(ConfigError, match="risk_per_trade"):
        parse_config(raw)


def test_config_rejects_an_unknown_filling_mode():
    raw = {
        "account": {"symbol": "XAUUSD", "timeframe": "M5"},
        "execution": {
            "magic": 1,
            "risk_per_trade": 0.005,
            "max_daily_loss": 0.02,
            "max_trades_per_day": 3,
            "max_open_positions": 1,
            "max_spread_points": 60,
            "min_reward_risk": 1.0,
            "deviation_points": 20,
            "filling_mode": "magic",
        },
        "monitor": {"bars": 500, "poll_seconds": 60, "max_signal_attempts": 2, "max_backoff_seconds": 600},
        "model": {"name": "openai/gpt-oss-20b", "context_bars": 100},
    }

    with pytest.raises(ConfigError, match="filling_mode"):
        parse_config(raw)
