"""Execution tests for pending reconciliation, evidence, and volume modes."""

from types import SimpleNamespace

import pytest

from trading.config import parse_config
from trading.groq_signal import Signal
from trading.journal import Journal, read_events
from trading.mt5_demo import (
    PendingExecution,
    _persist_pending,
    load_pending,
    reconcile_pending,
    submit_signal,
    validate_fixed_volume,
)
from trading.state import RuntimeState

SYMBOL = "XAUUSD"
MAGIC = 26001
ASK = 2000.0
BID = 1999.87


class FakeTerminal:
    """Minimal MT5 double whose evidence can be switched on and off."""

    def __init__(self, *, balance=10_000.0, with_evidence=True, send_retcode=10009):
        self.balance = balance
        self.with_evidence = with_evidence
        self.send_retcode = send_retcode
        self.sent = []
        self.order_checks = 0
        self._filled = False
        self.ACCOUNT_TRADE_MODE_DEMO = 0
        self.SYMBOL_TRADE_MODE_DISABLED = 1
        self.ORDER_TYPE_BUY = 0
        self.ORDER_TYPE_SELL = 1
        self.TRADE_ACTION_DEAL = 1
        self.ORDER_TIME_GTC = 0
        self.ORDER_FILLING_IOC = 1
        self.ORDER_FILLING_FOK = 0
        self.ORDER_FILLING_RETURN = 2
        self.TRADE_RETCODE_DONE = 10009
        self.TRADE_RETCODE_DONE_PARTIAL = 10010
        self.TRADE_RETCODE_PLACED = 10008
        self.TRADE_RETCODE_INVALID_FILL = 10030

    # -- account / symbol -------------------------------------------------
    def account_info(self):
        return SimpleNamespace(
            login=10013615454,
            server="MetaQuotes-Demo",
            currency="USD",
            balance=self.balance,
            equity=self.balance,
            trade_mode=0,
            trade_expert=True,
            leverage=100,
        )

    def symbol_info(self, symbol):
        return SimpleNamespace(
            name=symbol,
            point=0.01,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            trade_stops_level=0,
            trade_freeze_level=0,
            trade_mode=0,
            trade_contract_size=100.0,
            trade_tick_size=0.01,
            trade_tick_value=0.1,
            filling_mode=3,
        )

    def symbol_info_tick(self, symbol):
        return SimpleNamespace(bid=BID, ask=ASK)

    def symbol_select(self, symbol, enable=True):
        return True

    def terminal_info(self):
        return SimpleNamespace(trade_allowed=True)

    def last_error(self):
        return (0, "ok")

    # -- market state -----------------------------------------------------
    def positions_get(self, symbol=None):
        if not self._filled:
            return []
        return [
            SimpleNamespace(
                ticket=11, identifier=11, order=11, symbol=symbol,
                magic=MAGIC, volume=0.02, price_open=ASK,
            )
        ]

    def order_calc_profit(self, order_type, symbol, volume, entry, exit_price):
        return -(abs(entry - exit_price) * 100.0 * volume)

    def history_orders_get(self, ticket=None):
        if not self.with_evidence:
            return []
        return [SimpleNamespace(symbol=SYMBOL, magic=MAGIC, ticket=ticket)]

    def history_deals_get(self, *args, **kwargs):
        if not self.with_evidence:
            return []
        ticket = kwargs.get("ticket")
        if ticket is None:
            return []
        return [
            SimpleNamespace(
                symbol=SYMBOL, magic=MAGIC, ticket=ticket, order=11,
                position_id=11, volume=0.02,
            )
        ]

    # -- ordering ---------------------------------------------------------
    def order_check(self, request):
        self.order_checks += 1
        return SimpleNamespace(retcode=0, comment="")

    def order_send(self, request):
        self.sent.append(dict(request))
        self._filled = self.send_retcode == self.TRADE_RETCODE_DONE
        return SimpleNamespace(
            retcode=self.send_retcode,
            order=11,
            deal=22,
            price=request["price"],
            comment="",
        )


def _config(**execution_overrides):
    gate_overrides = execution_overrides.pop("gate", None)
    base = {
        "account": {"symbol": SYMBOL, "timeframe": "M5"},
        "gate": {
            "min_score": 0.0,
            "min_score_gap": 0.0,
            "min_rr": 1.5,
            "fixed_volume": 0.01,
            "max_daily_loss_usd": 1_000.0,
            "max_total_loss_usd": 10_000.0,
            "max_trades_per_day": 3,
            "max_spread_points": 60,
        },
        "execution": {
            "magic": MAGIC,
            "risk_per_trade": 0.005,
            "max_daily_loss": 0.02,
            "max_open_positions": 1,
            "max_spread_points": 60,
            "deviation_points": 20,
            "max_volume": 1.0,
            "filling_mode": None,
        },
        "monitor": {
            "bars": 500,
            "poll_seconds": 60,
            "max_signal_attempts": 2,
            "max_backoff_seconds": 600,
        },
        "model": {"name": "openai/gpt-oss-20b", "context_bars": 100},
    }
    base["execution"].update(execution_overrides)
    if gate_overrides:
        base["gate"].update(gate_overrides)
    return parse_config(base)


def _signal():
    return Signal(
        "BUY", ASK, ASK - 5.0, ASK + 10.0, 2.0, "BULLISH", "PULLBACK",
        8.0, 0.0, "test", (),
    )


def _pending(pending_path, *, expires_at=None):
    import time

    now = time.time()
    return PendingExecution(
        SYMBOL, 11, 22, "BUY", MAGIC, 0.02, now,
        expires_at if expires_at is not None else now + 300,
    )


def test_missing_evidence_is_unconfirmed_and_blocks_a_second_order(tmp_path):
    terminal = FakeTerminal(with_evidence=False)
    journal = Journal(tmp_path, echo=False)
    config = _config()
    path = tmp_path / "pending.sqlite3"

    result = submit_signal(
        _signal(), config, journal, RuntimeState(), terminal,
        execute=True, pending_path=path,
    )

    assert result.status == "unconfirmed"
    assert load_pending(path, SYMBOL) is not None
    assert len(terminal.sent) == 1

    second = submit_signal(
        _signal(), config, journal, RuntimeState(), terminal,
        execute=True, pending_path=path,
    )
    assert second.status == "blocked"
    assert second.reason == "pending_execution"
    assert len(terminal.sent) == 1  # never resubmitted


def test_pending_execution_reconciles_once_evidence_appears(tmp_path):
    terminal = FakeTerminal(with_evidence=False)
    journal = Journal(tmp_path, echo=False)
    config = _config()
    path = tmp_path / "pending.sqlite3"
    submit_signal(
        _signal(), config, journal, RuntimeState(), terminal,
        execute=True, pending_path=path,
    )

    terminal.with_evidence = True
    status = reconcile_pending(terminal, config, journal, pending_path=path)

    assert status.startswith("filled")
    assert load_pending(path, SYMBOL) is None
    events = [event["event"] for event in read_events(tmp_path)]
    assert "reconciliation" in events


def test_expired_pending_blocks_new_entries(tmp_path):
    terminal = FakeTerminal(with_evidence=False)
    journal = Journal(tmp_path, echo=False)
    config = _config()
    path = tmp_path / "pending.sqlite3"
    _persist_pending(path, _pending(path, expires_at=1.0))

    status = reconcile_pending(terminal, config, journal, pending_path=path)

    assert status.startswith("expired")
    blocked = submit_signal(
        _signal(), config, journal, RuntimeState(), terminal,
        execute=True, pending_path=path,
    )
    assert blocked.status == "blocked"
    assert blocked.reason == "expired_execution"
    assert terminal.sent == []


def test_fixed_volume_mode_sends_the_exact_configured_volume(tmp_path):
    terminal = FakeTerminal()
    journal = Journal(tmp_path, echo=False)
    config = _config(volume_mode="fixed", max_risk_amount=100.0, gate={"fixed_volume": 0.02})

    result = submit_signal(
        _signal(), config, journal, RuntimeState(), terminal,
        execute=True, pending_path=tmp_path / "pending.sqlite3",
    )

    assert result.status == "filled"
    assert terminal.sent[0]["volume"] == pytest.approx(0.02)


def test_fixed_volume_risk_limit_blocks_the_order(tmp_path):
    terminal = FakeTerminal()
    journal = Journal(tmp_path, echo=False)
    config = _config(volume_mode="fixed", max_risk_amount=1.0, gate={"fixed_volume": 0.02})

    result = submit_signal(
        _signal(), config, journal, RuntimeState(), terminal,
        execute=True, pending_path=tmp_path / "pending.sqlite3",
    )

    assert result.status == "blocked"
    assert result.reason.startswith("potential_loss_exceeds_risk_limit")
    assert terminal.sent == []


def test_fixed_volume_rejects_an_incompatible_step():
    info = FakeTerminal().symbol_info(SYMBOL)

    assert validate_fixed_volume(info, 0.02) is None
    error = validate_fixed_volume(info, 0.025)
    assert error is not None and "step" in error


def test_dry_run_never_calls_order_send(tmp_path):
    terminal = FakeTerminal()
    journal = Journal(tmp_path, echo=False)
    config = _config()

    result = submit_signal(
        _signal(), config, journal, RuntimeState(), terminal,
        execute=False, pending_path=tmp_path / "pending.sqlite3",
    )

    assert result.status == "dry_run"
    assert result.volume == pytest.approx(0.1)
    assert terminal.sent == []


def test_reconcile_without_pending_returns_none(tmp_path):
    terminal = FakeTerminal()
    journal = Journal(tmp_path, echo=False)

    assert (
        reconcile_pending(
            terminal, _config(), journal, pending_path=tmp_path / "pending.sqlite3"
        )
        is None
    )
