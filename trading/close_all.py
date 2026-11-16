"""Close only this bot's open demo positions (panic switch).

Run without ``--yes`` first to see what would be closed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trading.config import DEFAULT_PATH, load_config
from trading.journal import Journal

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Close this bot's demo positions")
    parser.add_argument("--config", default=str(DEFAULT_PATH))
    parser.add_argument("--yes", action="store_true", help="actually send the close orders")
    args = parser.parse_args(argv)

    from trading.mt5_demo import (
        ExecutionError,
        filling_candidates,
        open_positions,
        require_demo_account,
        resolve_filling,
        session,
    )

    config = load_config(args.config)
    magic = config.execution.magic
    journal = Journal(ROOT / "logs")
    try:
        with session() as terminal:
            account = require_demo_account(terminal)
            positions = open_positions(terminal, config.account.symbol, magic)
            if not positions:
                print("No open positions for magic", magic)
                return 0
            print(f"Demo account {account.login} @ {account.server}")
            for position in positions:
                print(
                    f"  #{position.ticket} {position.symbol} "
                    f"{'BUY' if position.type == terminal.POSITION_TYPE_BUY else 'SELL'} "
                    f"{position.volume} @ {position.price_open} "
                    f"P&L {position.profit:.2f}"
                )
            if not args.yes:
                print("Dry run. Re-run with --yes to close these positions.")
                return 0

            info = terminal.symbol_info(config.account.symbol)
            tick = terminal.symbol_info_tick(config.account.symbol)
            for position in positions:
                is_buy = position.type == terminal.POSITION_TYPE_BUY
                request = {
                    "action": terminal.TRADE_ACTION_DEAL,
                    "symbol": position.symbol,
                    "volume": position.volume,
                    "type": terminal.ORDER_TYPE_SELL if is_buy else terminal.ORDER_TYPE_BUY,
                    "position": position.ticket,
                    "price": float(tick.bid if is_buy else tick.ask),
                    "deviation": config.execution.deviation_points,
                    "magic": magic,
                    "comment": "close-all",
                    "type_time": terminal.ORDER_TIME_GTC,
                }
                sent = None
                for filling in filling_candidates(
                    terminal, resolve_filling(terminal, info, config.execution.filling_mode)
                ):
                    request["type_filling"] = filling
                    sent = terminal.order_send(request)
                    if sent is not None and int(sent.retcode) == int(
                        getattr(terminal, "TRADE_RETCODE_INVALID_FILL", 10030)
                    ):
                        continue
                    break
                if sent is None or int(sent.retcode) != int(terminal.TRADE_RETCODE_DONE):
                    retcode = getattr(sent, "retcode", None)
                    print(f"  FAILED #{position.ticket} retcode={retcode}")
                    journal.write(
                        "close_failed", position=int(position.ticket), retcode=retcode
                    )
                else:
                    print(f"  CLOSED #{position.ticket} order={sent.order}")
                    journal.write("close_sent", position=int(position.ticket), order=int(sent.order))
    except ExecutionError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
