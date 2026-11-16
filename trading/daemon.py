"""Continuous, guarded Demo monitor for XAUUSD M5."""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime, timezone

import MetaTrader5 as mt5
import pandas as pd

from data.timestamp_validation import (
    TimestampValidationError,
    select_closed_m5_candles,
)
from trading.groq_signal import Signal, request_signal
from trading.monitor_log import append_monitor_log
from trading.mt5_demo import execute_demo_signal, reconcile_pending_executions

LOGGER = logging.getLogger(__name__)
FORCED_TEST_DISTANCE = 3.0
FORCED_TEST_REWARD = 6.6


def _forced_test_signal(symbol: str, action: str) -> Signal:
    """Build one deterministic test signal from the current Demo tick."""
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")
    try:
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if info is None or tick is None:
            raise RuntimeError(f"unable to read {symbol}: {mt5.last_error()}")
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"unable to select {symbol}: {mt5.last_error()}")
        digits = int(getattr(info, "digits", 0))
        entry = float(tick.ask if action == "BUY" else tick.bid)
        if action == "BUY":
            stop_loss = entry - FORCED_TEST_DISTANCE
            take_profit = entry + FORCED_TEST_REWARD
        else:
            stop_loss = entry + FORCED_TEST_DISTANCE
            take_profit = entry - FORCED_TEST_REWARD
        entry = round(entry, digits)
        stop_loss = round(stop_loss, digits)
        take_profit = round(take_profit, digits)
        return Signal(
            action,
            entry,
            stop_loss,
            take_profit,
            2.2,
            "FORCED_TEST",
            "FORCED_TEST",
            0.0,
            0.0,
            "FORCED TEST TRADE",
            (),
        )
    finally:
        mt5.shutdown()


def _forced_test_error(action: str, error: Exception) -> Signal:
    return Signal(
        "DATA_ERROR",
        None,
        None,
        None,
        None,
        "UNKNOWN",
        "INVALID_DATA",
        0.0,
        0.0,
        f"FORCED TEST TRADE failed: {type(error).__name__}: {error}",
        (f"FORCED TEST TRADE failed: {type(error).__name__}: {error}",),
    )


def fetch_closed_m5(symbol: str, count: int = 500) -> pd.DataFrame:
    """Fetch only closed M5 candles from the connected MT5 terminal."""
    current_utc = datetime.now(timezone.utc)
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")
    try:
        # Position 1 starts at the last completed candle; position 0 is forming.
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 1, count)
        if rates is None or len(rates) < 20:
            raise RuntimeError(f"MT5 returned insufficient rates: {mt5.last_error()}")
        selected = select_closed_m5_candles(rates, current_utc)
        selected.attrs["mt5_start_pos"] = 1
        return selected
    finally:
        mt5.shutdown()


def _safe_log(value: object) -> str:
    return str(value).encode("ascii", "backslashreplace").decode("ascii")


def _write_monitor_log(
    log_path: str,
    symbol: str,
    event: str,
    status: str,
    values: dict[str, object] | None = None,
) -> None:
    try:
        append_monitor_log(
            log_path,
            symbol=symbol,
            event=event,
            status=status,
            values=values,
        )
    except OSError as exc:
        print(f"CSV_LOG_ERROR: {type(exc).__name__}: {exc}")


def run_monitor(
    symbol: str,
    interval_seconds: int,
    once: bool = False,
    log_path: str = r"logs\monitoring.csv",
) -> None:
    """Monitor new closed candles until interrupted."""
    last_processed: pd.Timestamp | None = None
    forced_action = os.environ.get("FORCE_TEST_TRADE", "").strip().upper()
    forced_test_used = False
    if forced_action in {"BUY", "SELL"}:
        message = f"FORCE_TEST_TRADE actif : {forced_action}"
        LOGGER.warning(message)
        print(message)
    print(f"Monitoring {symbol} M5 every {interval_seconds} seconds (Demo only).")
    while True:
        try:
            frame = fetch_closed_m5(symbol)
            candle_time = frame.index[-1]
            reconciliation_status = reconcile_pending_executions(symbol)
            if reconciliation_status:
                print(reconciliation_status)
            if candle_time != last_processed:
                candle_close = candle_time + pd.Timedelta(5, unit="min")
                print(f"MT5 start_pos: {frame.attrs['mt5_start_pos']}")
                print(f"Raw rates count: {frame.attrs['raw_rates_count']}")
                print(f"Raw latest candle: {frame.attrs['raw_latest_candle']}")
                print(f"Filtered rates count: {frame.attrs['filtered_rates_count']}")
                print(f"Latest closed candle: {frame.attrs['latest_closed_candle']}")
                print(
                    "Latest closed candle close time: "
                    f"{frame.attrs['latest_closed_close_time']}"
                )
                print(f"Current UTC: {frame.attrs['current_utc']}")
                print(f"Candle open time: {candle_time.isoformat()}")
                print(f"Candle close time: {candle_close.isoformat()}")
                print(f"Last closed candle: {candle_time.isoformat()}")
                print(f"Selected candle: {candle_time.isoformat()}")
                print("Timestamp validation: PASSED")
                print("Future candle detected: False")
                if forced_action in {"BUY", "SELL"} and not forced_test_used:
                    forced_test_used = True
                    try:
                        signal = _forced_test_signal(symbol, forced_action)
                    except Exception as exc:
                        signal = _forced_test_error(forced_action, exc)
                    print("Signal generation status: FORCED_TEST")
                else:
                    signal = request_signal(frame)
                    print("Signal generation status: COMPLETE")
                execution_status = reconciliation_status or ""
                print(
                    f"{datetime.now(timezone.utc).isoformat()} "
                    f"candle={candle_time.isoformat()} signal={_safe_log(signal)}"
                )
                if (
                    signal.action != "NO_TRADE"
                    and not (
                        reconciliation_status
                        and reconciliation_status.startswith("EXECUTION_UNCONFIRMED")
                    )
                ):
                    execution_status = execute_demo_signal(symbol, signal)
                    print(execution_status)
                _write_monitor_log(
                    log_path,
                    symbol,
                    "signal",
                    "COMPLETE",
                    {
                        "current_utc": frame.attrs["current_utc"],
                        "candle_open": candle_time.isoformat(),
                        "candle_close": candle_close.isoformat(),
                        "mt5_start_pos": frame.attrs["mt5_start_pos"],
                        "raw_rates_count": frame.attrs["raw_rates_count"],
                        "filtered_rates_count": frame.attrs["filtered_rates_count"],
                        "latest_closed_candle": frame.attrs["latest_closed_candle"],
                        "latest_closed_close_time": frame.attrs[
                            "latest_closed_close_time"
                        ],
                        "action": signal.action,
                        "entry_price": signal.entry_price,
                        "stop_loss": signal.stop_loss,
                        "take_profit": signal.take_profit,
                        "risk_reward": signal.risk_reward,
                        "market_trend": signal.market_trend,
                        "market_condition": signal.market_condition,
                        "buy_setup_score": signal.buy_setup_score,
                        "sell_setup_score": signal.sell_setup_score,
                        "local_buy_score": signal.local_buy_score,
                        "local_sell_score": signal.local_sell_score,
                        "local_score_breakdown": signal.local_score_breakdown,
                        "rationale": signal.rationale,
                        "rejection_reasons": signal.rejection_reasons,
                        "execution_status": execution_status,
                        "error": signal.diagnostic or "",
                    },
                )
                last_processed = candle_time
            else:
                print(f"No new closed candle: {candle_time.isoformat()}")
                _write_monitor_log(
                    log_path,
                    symbol,
                    "heartbeat",
                    "NO_NEW_CANDLE",
                    {
                        "current_utc": datetime.now(timezone.utc).isoformat(),
                        "candle_open": candle_time.isoformat(),
                    },
                )
        except TimestampValidationError as exc:
            current_utc = datetime.now(timezone.utc).isoformat()
            print(f"Current UTC: {current_utc}")
            print(f"Provider timestamp: {_safe_log(str(exc))}")
            print("Candle open time: unavailable")
            print("Candle close time: unavailable")
            print("Last closed candle: unavailable")
            print("Latest available candle: unavailable")
            print("Timestamp validation: FAILED")
            print(f"Future candle detected: {'future' in str(exc).lower()}")
            print(f"Signal generation status: DATA_ERROR ({exc})")
            _write_monitor_log(
                log_path,
                symbol,
                "timestamp_validation",
                "DATA_ERROR",
                {
                    "current_utc": current_utc,
                    "provider_timestamp": str(exc),
                    "error": str(exc),
                },
            )
        except Exception as exc:
            print(f"MONITOR_ERROR: {type(exc).__name__}: {exc}")
            _write_monitor_log(
                log_path,
                symbol,
                "monitor",
                "ERROR",
                {"error": f"{type(exc).__name__}: {exc}"},
            )
        if once:
            return
        time.sleep(interval_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous guarded XAUUSD Demo monitor")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--log-path", default=r"logs\monitoring.csv")
    args = parser.parse_args()
    if args.interval < 60:
        raise SystemExit("--interval must be at least 60 seconds")
    run_monitor(args.symbol, args.interval, args.once, args.log_path)


if __name__ == "__main__":
    main()
