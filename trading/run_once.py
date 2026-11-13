"""Ask Groq for one proposal and optionally execute it on MT5 Demo."""

from __future__ import annotations

import os
from pathlib import Path

from data.validate import load_and_validate_csv
from trading.groq_signal import request_signal
from trading.mt5_demo import execute_demo_signal


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    frame = load_and_validate_csv(root / "data" / "raw" / "XAUUSD_M5.csv")
    signal = request_signal(frame)
    print(signal)
    if signal.action != "NO_TRADE":
        print(execute_demo_signal("XAUUSD", signal))
    else:
        print("NO_TRADE: Groq did not find a valid setup")
    if os.environ.get("ENABLE_DEMO_TRADING") == "1":
        print("Demo execution mode was enabled explicitly.")


if __name__ == "__main__":
    main()
