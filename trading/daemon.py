"""Continuous, guarded Demo monitor for XAUUSD M5."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

import MetaTrader5 as mt5
import pandas as pd

from data.timestamp_validation import (
    TimestampValidationError,
    select_closed_m5_candles,
)
from trading.groq_signal import request_signal
from trading.mt5_demo import execute_demo_signal


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


def run_monitor(symbol: str, interval_seconds: int, once: bool = False) -> None:
    """Monitor new closed candles until interrupted."""
    last_processed: pd.Timestamp | None = None
    print(f"Monitoring {symbol} M5 every {interval_seconds} seconds (Demo only).")
    while True:
        try:
            frame = fetch_closed_m5(symbol)
            candle_time = frame.index[-1]
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
                signal = request_signal(frame)
                print(
                    f"{datetime.now(timezone.utc).isoformat()} "
                    f"candle={candle_time.isoformat()} signal={_safe_log(signal)}"
                )
                print("Signal generation status: COMPLETE")
                if signal.action != "NO_TRADE":
                    print(execute_demo_signal(symbol, signal))
                last_processed = candle_time
            else:
                print(f"No new closed candle: {candle_time.isoformat()}")
        except TimestampValidationError as exc:
            print(f"Current UTC: {datetime.now(timezone.utc).isoformat()}")
            print(f"Provider timestamp: {_safe_log(str(exc))}")
            print("Candle open time: unavailable")
            print("Candle close time: unavailable")
            print("Last closed candle: unavailable")
            print("Latest available candle: unavailable")
            print("Timestamp validation: FAILED")
            print(f"Future candle detected: {'future' in str(exc).lower()}")
            print(f"Signal generation status: DATA_ERROR ({exc})")
        except Exception as exc:
            print(f"MONITOR_ERROR: {type(exc).__name__}: {exc}")
        if once:
            return
        time.sleep(interval_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous guarded XAUUSD Demo monitor")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 60:
        raise SystemExit("--interval must be at least 60 seconds")
    run_monitor(args.symbol, args.interval, args.once)


if __name__ == "__main__":
    main()
