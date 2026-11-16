"""Continuous, guarded demo monitor.

Runs one MetaTrader 5 session for its whole lifetime, trades only closed
candles, journals every decision, reconciles ambiguous submissions, and never
re-trades a candle after a restart.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from data.timestamp_validation import TimestampValidationError, select_closed_m5_candles
from trading.config import DEFAULT_PATH, DemoConfig, TIMEFRAMES, load_config
from trading.groq_signal import Signal
from trading.gate_config import (
    GateAccountState,
    GateMarketState,
    evaluate_entry_gate,
    log_gate_result,
)
from trading.journal import Journal
from trading.monitor_log import append_monitor_log
from trading.state import StateStore, RuntimeState, single_instance

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
STATE_PATH = LOG_DIR / "state.json"
LOCK_PATH = LOG_DIR / "bot.lock"
MIN_BARS = 20
# A candle older than this many timeframes means the feed is stale (weekend,
# holiday, disconnect, or a terminal that has not synced history yet).
STALE_CANDLE_FACTOR = 3
STALE_RETRIES = 3
FORCED_TEST_DISTANCE = 3.0
FORCED_TEST_REWARD = 6.6


def forced_test_signal(terminal: Any, symbol: str, action: str) -> Signal:
    """Build one deterministic test signal from the current terminal tick."""
    info = terminal.symbol_info(symbol)
    tick = terminal.symbol_info_tick(symbol)
    if info is None or tick is None:
        raise RuntimeError(f"unable to read {symbol}: {terminal.last_error()}")
    if not terminal.symbol_select(symbol, True):
        raise RuntimeError(f"unable to select {symbol}: {terminal.last_error()}")
    digits = int(getattr(info, "digits", 0))
    entry = float(tick.ask if action == "BUY" else tick.bid)
    if action == "BUY":
        stop_loss = entry - FORCED_TEST_DISTANCE
        take_profit = entry + FORCED_TEST_REWARD
    else:
        stop_loss = entry + FORCED_TEST_DISTANCE
        take_profit = entry - FORCED_TEST_REWARD
    return Signal(
        action,
        round(entry, digits),
        round(stop_loss, digits),
        round(take_profit, digits),
        2.2,
        "FORCED_TEST",
        "FORCED_TEST",
        100.0 if action == "BUY" else 0.0,
        0.0 if action == "BUY" else 100.0,
        "FORCED TEST TRADE",
        (),
    )


def atr14(frame: pd.DataFrame) -> float:
    """Calculate ATR(14) from closed M5 candles only."""
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    value = float(true_range.rolling(14).mean().iloc[-1])
    if not pd.notna(value) or value <= 0:
        raise ValueError("ATR(14) is unavailable or non-positive")
    return value


def seconds_until_next_candle(minutes: int, now: datetime | None = None) -> float:
    """Return the delay until the next UTC candle boundary."""
    current = now or datetime.now(timezone.utc)
    elapsed = current.minute % minutes
    boundary = current.replace(
        minute=current.minute - elapsed,
        second=0,
        microsecond=0,
    ) + timedelta(minutes=minutes)
    return max(0.1, (boundary - current).total_seconds())


def apply_local_levels(
    signal: Signal, frame: pd.DataFrame, config: DemoConfig, digits: int = 2
) -> Signal:
    """Replace model levels with local ATR-based levels before gating.

    Levels are rounded to the symbol's price precision, because MT5 rejects
    stops that are not aligned with the instrument's tick grid.
    """
    if signal.action not in {"BUY", "SELL"} or signal.entry_price is None:
        return signal
    risk = config.gate.sl_atr_mult * atr14(frame)
    entry = float(signal.entry_price)
    if signal.action == "BUY":
        stop_loss, take_profit = entry - risk, entry + risk * config.gate.min_rr
    else:
        stop_loss, take_profit = entry + risk, entry - risk * config.gate.min_rr
    return replace(
        signal,
        entry_price=round(entry, digits),
        stop_loss=round(stop_loss, digits),
        take_profit=round(take_profit, digits),
        risk_reward=config.gate.min_rr,
    )


def timeframe_minutes(timeframe: str) -> int:
    """Return the candle size in minutes for a configured timeframe."""
    try:
        return TIMEFRAMES[timeframe.upper()]
    except KeyError as exc:
        raise ValueError(f"unsupported timeframe {timeframe!r}") from exc


def monitor_path(config: DemoConfig) -> Path:
    """Resolve the CSV diagnostics log path against the repository root."""
    path = Path(config.monitor.csv_log)
    return path if path.is_absolute() else ROOT / path


def log_monitor(
    config: DemoConfig,
    event: str,
    status: str,
    values: dict[str, Any] | None = None,
) -> None:
    """Write a CSV diagnostics row; a logging failure must not stop trading."""
    try:
        append_monitor_log(
            monitor_path(config),
            symbol=config.account.symbol,
            timeframe=config.account.timeframe,
            event=event,
            status=status,
            values=values,
        )
    except OSError as exc:
        print(f"CSV_LOG_ERROR: {type(exc).__name__}: {exc}")


def read_closed_bars(terminal: Any, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
    """Return only fully closed, fresh candles, newest last, validated for UTC.

    MetaTrader 5 can answer with a stale cache right after the terminal wakes up
    (observed: Friday's last candle returned on Monday). A stale candle is never
    tradeable, so the read is retried briefly and then refused.
    """
    attribute = f"TIMEFRAME_{timeframe.upper()}"
    timeframe_value = getattr(terminal, attribute, None)
    if timeframe_value is None:
        raise ValueError(f"unsupported timeframe {timeframe!r}")
    minutes = timeframe_minutes(timeframe)
    last_error: Exception | None = None
    for attempt in range(STALE_RETRIES):
        rates = terminal.copy_rates_from_pos(symbol, timeframe_value, 0, count)
        if rates is None or len(rates) < MIN_BARS:
            last_error = RuntimeError(
                f"MT5 returned too few rates: {terminal.last_error()}"
            )
        else:
            # The selector drops the forming candle and rejects future/
            # duplicate/out-of-order timestamps.
            frame = select_closed_m5_candles(
                rates,
                datetime.now(timezone.utc),
                minimum_count=MIN_BARS,
                timeframe_minutes=minutes,
            )
            close_time = frame.index[-1] + timedelta(minutes=minutes)
            age_minutes = (
                datetime.now(timezone.utc) - close_time
            ).total_seconds() / 60.0
            if age_minutes <= minutes * STALE_CANDLE_FACTOR:
                return frame
            last_error = TimestampValidationError(
                f"market data is stale: latest closed candle {frame.index[-1].isoformat()}"
                f" closed {age_minutes:.0f} minutes ago"
            )
        if attempt < STALE_RETRIES - 1:
            time.sleep(2)
    raise last_error or RuntimeError("unable to read fresh candles")


def journal_new_deals(
    terminal: Any, config: DemoConfig, journal: Journal, state: RuntimeState
) -> None:
    """Record closed deals this bot produced since the last poll.

    Only closing deals are used for the loss counters: an entry deal carries no
    realized P&L and its commission must not be mistaken for a losing trade.
    A brand-new state (``last_deal_ticket == 0``) must not replay the previous
    week's losses into today's limits, so only today's closed deals are counted
    on that first pass.
    """
    from trading.mt5_demo import new_deals

    out_entries = {
        int(getattr(terminal, "DEAL_ENTRY_OUT", 1)),
        int(getattr(terminal, "DEAL_ENTRY_INOUT", 2)),
    }
    today = (
        datetime.fromisoformat(state.day).date()
        if state.day
        else datetime.now(timezone.utc).date()
    )
    backfill = state.last_deal_ticket == 0
    for deal in new_deals(terminal, config.execution.magic, state.last_deal_ticket):
        deal_time = datetime.fromtimestamp(int(deal.time), tz=timezone.utc)
        closed = int(getattr(deal, "entry", -1)) in out_entries
        counted = closed and (not backfill or deal_time.date() == today)
        net = (
            float(getattr(deal, "profit", 0.0))
            + float(getattr(deal, "commission", 0.0))
            + float(getattr(deal, "swap", 0.0))
        )
        if counted and net < 0:
            loss = abs(net)
            state.daily_loss_usd += loss
            state.total_loss_usd += loss
            state.consecutive_losses += 1
        elif counted and net > 0:
            state.consecutive_losses = 0
        journal.write(
            "deal",
            ticket=int(deal.ticket),
            position=int(getattr(deal, "position_id", 0)),
            symbol=str(getattr(deal, "symbol", config.account.symbol)),
            entry=int(getattr(deal, "entry", 0)),
            volume=float(getattr(deal, "volume", 0.0)),
            price=float(getattr(deal, "price", 0.0)),
            profit=float(getattr(deal, "profit", 0.0)),
            commission=float(getattr(deal, "commission", 0.0)),
            swap=float(getattr(deal, "swap", 0.0)),
            net=net,
            counted=counted,
            deal_time=deal_time.isoformat(),
        )
        state.last_deal_ticket = int(deal.ticket)


def process_candle(
    terminal: Any,
    config: DemoConfig,
    journal: Journal,
    state: RuntimeState,
    frame: pd.DataFrame,
    execute: bool,
    force_action: str | None = None,
) -> None:
    """Turn the newest closed candle into at most one demo order."""
    from trading.mt5_demo import ExecutionError, submit_signal

    candle = frame.index[-1].isoformat()
    close_time = (
        frame.index[-1] + timedelta(minutes=timeframe_minutes(config.account.timeframe))
    ).isoformat()
    try:
        if force_action is not None:
            signal = forced_test_signal(terminal, config.account.symbol, force_action)
        else:
            from trading.groq_signal import request_signal

            signal = request_signal(
                frame,
                model=config.model.name,
                context_bars=config.model.context_bars,
            )
    except Exception as exc:  # noqa: BLE001 - the loop must survive
        journal.write(
            "error",
            stage="signal",
            candle=candle,
            error=type(exc).__name__,
            detail=str(exc),
            attempt=state.signal_attempts,
        )
        log_monitor(
            config,
            "signal",
            "ERROR",
            {"candle_open": candle, "candle_close": close_time,
             "error": f"{type(exc).__name__}: {exc}"},
        )
        return

    journal.write(
        "signal",
        candle=candle,
        action=signal.action,
        entry_price=signal.entry_price,
        stop_loss=signal.stop_loss,
        take_profit=signal.take_profit,
        risk_reward=signal.risk_reward,
        market_trend=signal.market_trend,
        market_condition=signal.market_condition,
        buy_setup_score=signal.buy_setup_score,
        sell_setup_score=signal.sell_setup_score,
        local_buy_score=signal.local_buy_score,
        local_sell_score=signal.local_sell_score,
        local_score_breakdown=signal.local_score_breakdown,
        rejection_reasons=list(signal.rejection_reasons),
        rationale=signal.rationale,
        diagnostic=signal.diagnostic,
        timeframe=config.account.timeframe,
    )
    if signal.action in {"NO_TRADE", "DATA_ERROR"}:
        log_monitor(
            config,
            "signal",
            "COMPLETE",
            {
                "current_utc": datetime.now(timezone.utc).isoformat(),
                "candle_open": candle,
                "candle_close": close_time,
                "latest_closed_candle": candle,
                "action": signal.action,
                "market_trend": signal.market_trend,
                "market_condition": signal.market_condition,
                "buy_setup_score": signal.buy_setup_score,
                "sell_setup_score": signal.sell_setup_score,
                "local_buy_score": signal.local_buy_score,
                "local_sell_score": signal.local_sell_score,
                "local_score_breakdown": signal.local_score_breakdown,
                "rationale": signal.rationale,
                "rejection_reasons": signal.rejection_reasons,
                "error": signal.diagnostic or "",
            },
        )
        state.finish_candle(candle)
        return

    try:
        tick = terminal.symbol_info_tick(config.account.symbol)
        info = terminal.symbol_info(config.account.symbol)
        if tick is None or info is None:
            raise RuntimeError(f"unable to read tick: {terminal.last_error()}")
        entry = float(tick.ask if signal.action == "BUY" else tick.bid)
        digits = int(getattr(info, "digits", 2))
        signal = apply_local_levels(
            replace(signal, entry_price=entry), frame, config, digits
        )
        point = float(getattr(info, "point", 0.0) or 0.0)
        spread_points = (
            (float(tick.ask) - float(tick.bid)) / point if point > 0 else float("inf")
        )
        from trading.mt5_demo import open_positions

        positions = open_positions(terminal, config.account.symbol, config.execution.magic)
        local_direction = (
            "BUY" if signal.local_buy_score > signal.local_sell_score else "SELL"
        )
        gate = evaluate_entry_gate(
            signal,
            GateMarketState(
                spread_points=spread_points,
                utc_now=datetime.now(timezone.utc),
                candle_id=candle,
                local_direction=local_direction,
            ),
            GateAccountState(
                daily_loss_usd=state.daily_loss_usd,
                consecutive_losses=state.consecutive_losses,
                trades_today=state.trades_today,
                total_loss_usd=state.total_loss_usd,
                open_positions=len(positions),
                entered_candle=state.entered_candle,
                kill_switch=state.kill_switch,
            ),
            config.gate,
        )
        log_gate_result(gate)
        if not gate.passed:
            reason = "; ".join(gate.failed_rules)
            journal.write("entry_blocked", candle=candle, reason=reason)
            log_monitor(
                config,
                "gate",
                "REJECTED",
                {
                    "candle_open": candle,
                    "action": signal.action,
                    "entry_price": signal.entry_price,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "risk_reward": signal.risk_reward,
                    "local_buy_score": signal.local_buy_score,
                    "local_sell_score": signal.local_sell_score,
                    "rationale": reason,
                    "rejection_reasons": gate.failed_rules,
                },
            )
            state.finish_candle(candle)
            return
        state.entered_candle = candle
    except (RuntimeError, ValueError) as exc:
        journal.write("entry_blocked", candle=candle, reason=str(exc))
        state.finish_candle(candle)
        return

    journal.write(
        "entry_plan",
        candle=candle,
        action=signal.action,
        entry_price=signal.entry_price,
        stop_loss=signal.stop_loss,
        take_profit=signal.take_profit,
        risk_reward=signal.risk_reward,
        spread_points=round(spread_points, 2),
        score=round(signal.local_buy_score if signal.action == "BUY" else signal.local_sell_score, 2),
        timeframe=config.account.timeframe,
    )
    try:
        result = submit_signal(signal, config, journal, state, terminal, execute=execute)
    except ExecutionError as exc:
        journal.write("error", stage="execution", candle=candle, detail=str(exc))
        log_monitor(
            config, "execution", "ERROR", {"candle_open": candle, "error": str(exc)}
        )
        state.finish_candle(candle)
        return
    journal.write("result", candle=candle, status=result.status, detail=result.detail)
    log_monitor(
        config,
        "execution",
        result.status.upper(),
        {
            "candle_open": candle,
            "action": signal.action,
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "risk_reward": signal.risk_reward,
            "execution_status": result.status,
            "error": result.detail if result.status == "blocked" else "",
        },
    )
    state.finish_candle(candle)


def run(
    config: DemoConfig,
    *,
    once: bool = False,
    dry_run: bool = False,
    journal_directory: Path = LOG_DIR,
) -> int:
    """Poll the terminal until interrupted; return a process exit code."""
    from trading.mt5_demo import (
        ExecutionError,
        is_live_trading_enabled,
        reconcile_pending,
        require_demo_account,
        session,
    )

    journal = Journal(journal_directory)
    store = StateStore(journal_directory / "state.json")
    state = store.load()
    execute = is_live_trading_enabled() and not dry_run
    forced_action = os.environ.get("FORCE_TEST_TRADE", "").strip().upper()
    if forced_action not in {"BUY", "SELL"}:
        forced_action = None
    forced_test_used = False

    with single_instance(journal_directory / "bot.lock"):
        with session() as terminal:
            # Refuse a live account before doing anything else, even in dry run.
            account = require_demo_account(terminal, require_expert=execute)
            journal.write(
                "startup",
                symbol=config.account.symbol,
                timeframe=config.account.timeframe,
                magic=config.execution.magic,
                execute=execute,
                volume_mode=config.execution.volume_mode,
                login=account.login,
                server=account.server,
                balance=account.balance,
                equity=account.equity,
                leverage=account.leverage,
                restarted_at=datetime.now(timezone.utc).isoformat(),
            )
            if not execute:
                journal.write(
                    "notice",
                    detail="dry run: set ENABLE_DEMO_TRADING=1 to send demo orders",
                )
            if forced_action is not None:
                message = f"FORCE_TEST_TRADE active: {forced_action} (one order this process)"
                LOGGER.warning(message)
                journal.write("notice", detail=message)
                print(message)

            failures = 0
            while True:
                try:
                    today = datetime.now(timezone.utc).date().isoformat()
                    if state.roll_day(today):
                        store.save(state)
                    journal_new_deals(terminal, config, journal, state)
                    # Persist the deal cursor immediately: a crash right after
                    # this point must not re-count the same deals on restart.
                    store.save(state)
                    if state.kill_switch:
                        LOGGER.error(
                            "KILL SWITCH remains active; run with --reset-kill-switch "
                            "to resume after reviewing the account"
                        )
                        print(
                            "KILL SWITCH active: total_loss_usd="
                            f"{state.total_loss_usd:.2f}; exiting. "
                            "Use --reset-kill-switch to resume.",
                            file=sys.stderr,
                        )
                        return 2
                    if state.total_loss_usd >= config.gate.max_total_loss_usd:
                        state.kill_switch = True
                        journal.write(
                            "kill_switch",
                            reason="max_total_loss_usd",
                            total_loss_usd=state.total_loss_usd,
                        )
                        store.save(state)
                        LOGGER.error(
                            "KILL SWITCH active: total_loss_usd=%.2f",
                            state.total_loss_usd,
                        )
                        print(
                            "KILL SWITCH active: total_loss_usd="
                            f"{state.total_loss_usd:.2f}; exiting with code 2.",
                            file=sys.stderr,
                        )
                        return 2

                    reconciliation = reconcile_pending(terminal, config, journal)
                    if reconciliation:
                        journal.write("result", stage="reconciliation", detail=reconciliation)

                    frame = read_closed_bars(
                        terminal,
                        config.account.symbol,
                        config.account.timeframe,
                        config.monitor.bars,
                    )
                    candle = frame.index[-1].isoformat()
                    state.begin_candle(candle)

                    if candle == state.processed_candle:
                        journal.write("idle", candle=candle, processed=True)
                        log_monitor(
                            config,
                            "heartbeat",
                            "NO_NEW_CANDLE",
                            {
                                "current_utc": datetime.now(timezone.utc).isoformat(),
                                "candle_open": candle,
                                "latest_closed_candle": candle,
                            },
                        )
                    elif state.signal_attempts >= config.monitor.max_signal_attempts:
                        journal.write(
                            "abandoned",
                            candle=candle,
                            attempts=state.signal_attempts,
                        )
                        state.finish_candle(candle)
                    else:
                        state.signal_attempts += 1
                        store.save(state)
                        action_for_attempt = (
                            forced_action if not forced_test_used else None
                        )
                        if action_for_attempt is not None:
                            forced_test_used = True
                        process_candle(
                            terminal,
                            config,
                            journal,
                            state,
                            frame,
                            execute,
                            action_for_attempt,
                        )
                    failures = 0
                except (TimestampValidationError, ExecutionError) as exc:
                    failures += 1
                    journal.write(
                        "error",
                        stage="loop",
                        error=type(exc).__name__,
                        detail=str(exc),
                    )
                    log_monitor(
                        config,
                        "loop",
                        "ERROR",
                        {"current_utc": datetime.now(timezone.utc).isoformat(),
                         "error": f"{type(exc).__name__}: {exc}"},
                    )
                except Exception as exc:  # noqa: BLE001 - the loop must survive
                    failures += 1
                    journal.write(
                        "error",
                        stage="loop",
                        error=type(exc).__name__,
                        detail=str(exc),
                    )
                    log_monitor(
                        config,
                        "loop",
                        "ERROR",
                        {"current_utc": datetime.now(timezone.utc).isoformat(),
                         "error": f"{type(exc).__name__}: {exc}"},
                    )
                store.save(state)

                if once:
                    break
                delay = seconds_until_next_candle(
                    timeframe_minutes(config.account.timeframe)
                )
                if failures:
                    delay = min(delay * (2**failures), config.monitor.max_backoff_seconds)
                time.sleep(delay)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Continuous guarded XAUUSD demo monitor")
    parser.add_argument("--config", default=str(DEFAULT_PATH))
    parser.add_argument("--once", action="store_true", help="process one candle and exit")
    parser.add_argument("--dry-run", action="store_true", help="never send orders")
    parser.add_argument(
        "--reset-kill-switch",
        action="store_true",
        help=(
            "clear the kill switch and loss counters after manually reviewing "
            "the account; takes effect without connecting to MT5"
        ),
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        if args.reset_kill_switch:
            store = StateStore(STATE_PATH)
            state = store.load()
            previous = {
                "kill_switch": state.kill_switch,
                "total_loss_usd": state.total_loss_usd,
                "daily_loss_usd": state.daily_loss_usd,
                "consecutive_losses": state.consecutive_losses,
            }
            state.kill_switch = False
            state.total_loss_usd = 0.0
            state.daily_loss_usd = 0.0
            state.consecutive_losses = 0
            store.save(state)
            Journal(LOG_DIR).write("kill_switch_reset", previous=previous)
            print(
                "Kill switch reset: kill_switch=False total_loss_usd=0.0 "
                "daily_loss_usd=0.0 consecutive_losses=0"
            )
            return 0
        return run(config, once=args.once, dry_run=args.dry_run)
    except KeyboardInterrupt:
        print("Stopped by user.")
    except Exception as exc:  # noqa: BLE001 - surface a clean message
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
