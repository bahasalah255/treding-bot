"""Continuous, guarded demo monitor.

Runs one MetaTrader 5 session for its whole lifetime, trades only closed
candles, journals every decision, and never re-trades a candle after a restart.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from data.timestamp_validation import TimestampValidationError, select_closed_m5_candles
from trading.config import DEFAULT_PATH, DemoConfig, TIMEFRAMES, load_config
from trading.journal import Journal
from trading.state import StateStore, RuntimeState, single_instance

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
STATE_PATH = LOG_DIR / "state.json"
LOCK_PATH = LOG_DIR / "bot.lock"
MIN_BARS = 20


def timeframe_minutes(timeframe: str) -> int:
    """Return the candle size in minutes for a configured timeframe."""
    try:
        return TIMEFRAMES[timeframe.upper()]
    except KeyError as exc:
        raise ValueError(f"unsupported timeframe {timeframe!r}") from exc


def read_closed_bars(terminal: Any, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
    """Return only fully closed candles, newest last, validated for UTC order."""
    attribute = f"TIMEFRAME_{timeframe.upper()}"
    timeframe_value = getattr(terminal, attribute, None)
    if timeframe_value is None:
        raise ValueError(f"unsupported timeframe {timeframe!r}")
    rates = terminal.copy_rates_from_pos(symbol, timeframe_value, 0, count)
    if rates is None or len(rates) < MIN_BARS:
        raise RuntimeError(f"MT5 returned too few rates: {terminal.last_error()}")
    # The selector drops the forming candle and rejects future/duplicate/
    # out-of-order timestamps, so nothing unclosed can reach the model.
    return select_closed_m5_candles(
        rates,
        datetime.now(timezone.utc),
        minimum_count=MIN_BARS,
        timeframe_minutes=timeframe_minutes(timeframe),
    )


def journal_new_deals(
    terminal: Any, config: DemoConfig, journal: Journal, state: RuntimeState
) -> None:
    """Record deals this bot closed since the last poll."""
    from trading.mt5_demo import new_deals

    for deal in new_deals(terminal, config.execution.magic, state.last_deal_ticket):
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
            deal_time=datetime.fromtimestamp(
                int(deal.time), tz=timezone.utc
            ).isoformat(),
        )
        state.last_deal_ticket = int(deal.ticket)


def process_candle(
    terminal: Any,
    config: DemoConfig,
    journal: Journal,
    state: RuntimeState,
    frame: pd.DataFrame,
    execute: bool,
) -> None:
    """Turn the newest closed candle into at most one demo order."""
    from trading.groq_signal import request_signal
    from trading.mt5_demo import ExecutionError, submit_signal

    candle = frame.index[-1].isoformat()
    try:
        signal = request_signal(
            frame,
            model=config.model.name,
            context_bars=config.model.context_bars,
        )
    except Exception as exc:
        journal.write(
            "error", stage="signal", candle=candle,
            error=type(exc).__name__, detail=str(exc),
            attempt=state.signal_attempts,
        )
        return

    journal.write(
        "signal", candle=candle, action=signal.action,
        entry_price=signal.entry_price, stop_loss=signal.stop_loss,
        take_profit=signal.take_profit, risk_reward=signal.risk_reward,
        market_trend=signal.market_trend, market_condition=signal.market_condition,
        rejection_reasons=list(signal.rejection_reasons),
        rationale=signal.rationale,
    )
    if signal.action in {"NO_TRADE", "DATA_ERROR"}:
        state.finish_candle(candle)
        return

    try:
        result = submit_signal(signal, config, journal, state, terminal, execute=execute)
    except ExecutionError as exc:
        journal.write("error", stage="execution", candle=candle, detail=str(exc))
        state.finish_candle(candle)
        return
    journal.write("result", candle=candle, status=result.status, detail=result.detail)
    state.finish_candle(candle)


def run(
    config: DemoConfig,
    *,
    once: bool = False,
    dry_run: bool = False,
    journal_directory: Path = LOG_DIR,
) -> None:
    """Poll the terminal until interrupted."""
    from trading.mt5_demo import ExecutionError, is_live_trading_enabled, require_demo_account, session

    journal = Journal(journal_directory)
    store = StateStore(journal_directory / "state.json")
    state = store.load()
    execute = is_live_trading_enabled() and not dry_run

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

            failures = 0
            while True:
                try:
                    today = datetime.now(timezone.utc).date().isoformat()
                    if state.roll_day(today):
                        store.save(state)
                    journal_new_deals(terminal, config, journal, state)

                    frame = read_closed_bars(
                        terminal, config.account.symbol,
                        config.account.timeframe, config.monitor.bars,
                    )
                    candle = frame.index[-1].isoformat()
                    state.begin_candle(candle)

                    if candle == state.processed_candle:
                        journal.write("idle", candle=candle, processed=True)
                    elif state.signal_attempts >= config.monitor.max_signal_attempts:
                        journal.write(
                            "abandoned", candle=candle, attempts=state.signal_attempts,
                        )
                        state.finish_candle(candle)
                    else:
                        state.signal_attempts += 1
                        store.save(state)
                        process_candle(
                            terminal, config, journal, state, frame, execute
                        )
                    failures = 0
                except (TimestampValidationError, ExecutionError) as exc:
                    failures += 1
                    journal.write(
                        "error", stage="loop", error=type(exc).__name__, detail=str(exc)
                    )
                except Exception as exc:  # noqa: BLE001 - the loop must survive
                    failures += 1
                    journal.write(
                        "error", stage="loop", error=type(exc).__name__, detail=str(exc)
                    )
                store.save(state)

                if once:
                    break
                delay = config.monitor.poll_seconds
                if failures:
                    delay = min(
                        delay * (2**failures), config.monitor.max_backoff_seconds
                    )
                time.sleep(delay)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Continuous guarded XAUUSD demo monitor")
    parser.add_argument("--config", default=str(DEFAULT_PATH))
    parser.add_argument("--once", action="store_true", help="process one candle and exit")
    parser.add_argument("--dry-run", action="store_true", help="never send orders")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        run(config, once=args.once, dry_run=args.dry_run)
    except KeyboardInterrupt:
        print("Stopped by user.")
    except Exception as exc:  # noqa: BLE001 - surface a clean message
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
