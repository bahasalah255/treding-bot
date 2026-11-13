"""Continuous, guarded Demo monitor for XAUUSD M5."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

import MetaTrader5 as mt5
import pandas as pd

from trading.groq_signal import request_signal
from trading.mt5_demo import execute_demo_signal


def fetch_closed_m5(symbol: str, count: int = 500) -> pd.DataFrame:
    """Fetch only closed M5 candles from the connected MT5 terminal."""
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")
    try:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, count)
        if rates is None or len(rates) < 20:
            raise RuntimeError(f"MT5 returned insufficient rates: {mt5.last_error()}")
        frame = pd.DataFrame(rates[:-1])
        frame["timestamp"] = pd.to_datetime(frame.pop("time"), unit="s", utc=True)
        return frame.set_index("timestamp").sort_index()
    finally:
        mt5.shutdown()


def run_monitor(symbol: str, interval_seconds: int, once: bool = False) -> None:
    """Monitor new closed candles until interrupted."""
    last_processed: pd.Timestamp | None = None
    print(f"Monitoring {symbol} M5 every {interval_seconds} seconds (Demo only).")
    while True:
        try:
            frame = fetch_closed_m5(symbol)
            candle_time = frame.index[-1]
            if candle_time != last_processed:
                signal = request_signal(frame)
                print(
                    f"{datetime.now(timezone.utc).isoformat()} "
                    f"candle={candle_time.isoformat()} signal={signal}"
                )
                if signal.action != "NO_TRADE":
                    print(execute_demo_signal(symbol, signal))
                last_processed = candle_time
            else:
                print(f"No new closed candle: {candle_time.isoformat()}")
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
