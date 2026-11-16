"""One guarded candle, kept as a thin alias for the daemon's single pass.

This used to build its own signal path that never consulted the entry gate, so
``--execute`` could place an order the daemon would have refused. It now
delegates to the exactly same code the daemon uses: freshness check, signal,
gate, risk checks, persistence, and evidence-based fill reporting.
"""

from __future__ import annotations

import argparse

from trading import daemon
from trading.config import DEFAULT_PATH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="One guarded demo candle (alias for: python -m trading.daemon --once)"
    )
    parser.add_argument("--config", default=str(DEFAULT_PATH))
    parser.add_argument("--execute", action="store_true", help="allow sending the demo order")
    args = parser.parse_args(argv)

    forwarded = ["--config", args.config, "--once"]
    if not args.execute:
        forwarded.append("--dry-run")
    return daemon.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
