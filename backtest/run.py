"""Run the initial XAUUSD MTF EMA backtest."""

from __future__ import annotations

import json
from pathlib import Path

from data.validate import load_and_validate_csv

from backtest.ema_mtf import build_signals
from backtest.engine import run_backtest


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    m5 = load_and_validate_csv(root / "data" / "raw" / "XAUUSD_M5.csv")
    m15 = load_and_validate_csv(root / "data" / "raw" / "XAUUSD_M15.csv")
    signals = build_signals(m5, m15)
    report = run_backtest(signals)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
