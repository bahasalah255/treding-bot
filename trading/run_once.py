"""One manual pass: read closed bars, ask Groq, and optionally send the demo order."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trading.config import DEFAULT_PATH, load_config
from trading.journal import Journal
from trading.state import RuntimeState

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One guarded demo signal, optional execution")
    parser.add_argument("--config", default=str(DEFAULT_PATH))
    parser.add_argument("--execute", action="store_true", help="send the demo order")
    args = parser.parse_args(argv)

    from trading.daemon import LOG_DIR, read_closed_bars
    from trading.groq_signal import request_signal
    from trading.mt5_demo import ExecutionError, submit_signal, session

    config = load_config(args.config)
    journal = Journal(LOG_DIR)
    try:
        with session() as terminal:
            frame = read_closed_bars(
                terminal,
                config.account.symbol,
                config.account.timeframe,
                config.monitor.bars,
            )
            signal = request_signal(
                frame, model=config.model.name, context_bars=config.model.context_bars
            )
            journal.write(
                "signal",
                candle=frame.index[-1].isoformat(),
                action=signal.action,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                rationale=signal.rationale,
                source="run_once",
            )
            print(signal)
            if signal.action == "NO_TRADE":
                print("NO_TRADE: no valid setup")
                return 0
            result = submit_signal(
                signal, config, journal, RuntimeState(), terminal, execute=args.execute
            )
            print(result)
    except ExecutionError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
