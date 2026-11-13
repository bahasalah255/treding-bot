"""Read-only historical data importer for a connected MetaTrader 5 terminal."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from data.validate import validate_frame
from data.timestamp_validation import normalize_unix_timestamps

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover - depends on the local terminal environment
    mt5 = None

GOLD_ALIASES = {"xauusd", "gold", "xau"}
TIMEFRAMES = {"M5": "TIMEFRAME_M5", "M15": "TIMEFRAME_M15"}
OUTPUT_COLUMNS = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "spread",
    "real_volume",
)


class MT5ImportError(RuntimeError):
    """Raised when MT5 data cannot be imported safely."""


def select_gold_symbol(symbols: Iterable[Any], requested: str | None = None) -> str:
    """Select an exact requested symbol or an unambiguous gold symbol."""
    names = sorted({str(symbol.name) for symbol in symbols if getattr(symbol, "name", None)})
    if requested:
        if requested not in names:
            raise MT5ImportError(f"requested symbol is not available in Market Watch: {requested}")
        return requested

    matches = [
        name for name in names
        if name.casefold() in GOLD_ALIASES or name.casefold().startswith("xauusd")
    ]
    if len(matches) != 1:
        found = ", ".join(matches) if matches else "none"
        raise MT5ImportError(
            f"gold symbol selection is ambiguous ({found}); pass --symbol with the exact Market Watch name"
        )
    return matches[0]


def rates_to_frame(rates: Any) -> pd.DataFrame:
    """Convert MT5's structured rate array to the normalized CSV schema."""
    if rates is None or len(rates) == 0:
        raise MT5ImportError("MT5 returned no historical rates for the requested range")

    frame = pd.DataFrame(rates)
    required = {"time", "open", "high", "low", "close", "tick_volume", "spread"}
    missing = required - set(frame.columns)
    if missing:
        raise MT5ImportError(f"MT5 response is missing columns: {', '.join(sorted(missing))}")

    frame["timestamp"] = normalize_unix_timestamps(frame.pop("time"))
    if "real_volume" not in frame:
        frame["real_volume"] = 0
    frame = frame[list(OUTPUT_COLUMNS)].sort_values("timestamp")
    validate_frame(frame)
    return frame


def import_history(
    symbol: str,
    start: datetime,
    end: datetime,
    output_dir: str | Path,
) -> list[Path]:
    """Export M5 and M15 history from MT5 without submitting any orders."""
    if mt5 is None:
        raise MT5ImportError(
            "MetaTrader5 is not installed; install requirements-mt5.txt in the MT5 terminal environment"
        )
    if start.tzinfo is None or end.tzinfo is None:
        raise MT5ImportError("start and end must include an explicit timezone")
    if start >= end:
        raise MT5ImportError("start must be earlier than end")

    if not mt5.initialize():
        raise MT5ImportError(f"MT5 initialization failed: {mt5.last_error()}")
    try:
        account = mt5.account_info()
        if account is None:
            raise MT5ImportError(f"unable to read MT5 account information: {mt5.last_error()}")
        if getattr(account, "trade_mode", None) == getattr(mt5, "ACCOUNT_TRADE_MODE_REAL", 2):
            raise MT5ImportError("real trading accounts are prohibited by this importer")

        symbols = mt5.symbols_get()
        if symbols is None:
            raise MT5ImportError(f"unable to inspect Market Watch: {mt5.last_error()}")
        selected = select_gold_symbol(symbols, symbol)
        if not mt5.symbol_select(selected, True):
            raise MT5ImportError(f"unable to select {selected} in Market Watch: {mt5.last_error()}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        paths = []
        for label, timeframe_name in TIMEFRAMES.items():
            rates = mt5.copy_rates_range(
                selected,
                getattr(mt5, timeframe_name),
                start.astimezone(timezone.utc),
                end.astimezone(timezone.utc),
            )
            frame = rates_to_frame(rates)
            path = output_path / f"{selected}_{label}.csv"
            frame.to_csv(path, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
            paths.append(path)
        return paths
    finally:
        mt5.shutdown()


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("datetime must include a timezone, e.g. 2025-01-01T00:00:00Z")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only MT5 XAUUSD historical CSV importer")
    parser.add_argument("--symbol", help="exact symbol name from Market Watch")
    parser.add_argument("--start", required=True, type=_parse_datetime)
    parser.add_argument("--end", required=True, type=_parse_datetime)
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()
    paths = import_history(args.symbol, args.start, args.end, args.output_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
