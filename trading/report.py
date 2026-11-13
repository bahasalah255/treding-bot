"""Summarise the demo bot's journal: decisions, blocks, errors, and P&L."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from trading.journal import read_events

ROOT = Path(__file__).resolve().parents[1]


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate journal events into counters and realized deal P&L."""
    counts = Counter(str(event.get("event")) for event in events)
    blocks = Counter(
        str(event.get("reason"))
        for event in events
        if event.get("event") == "entry_blocked"
    )
    errors = Counter(
        f"{event.get('stage', '?')}:{event.get('error', event.get('detail', ''))}"
        for event in events
        if event.get("event") == "error"
    )

    per_position: dict[int, float] = defaultdict(float)
    for event in events:
        if event.get("event") != "deal":
            continue
        position = int(event.get("position") or 0)
        per_position[position] += (
            float(event.get("profit") or 0.0)
            + float(event.get("commission") or 0.0)
            + float(event.get("swap") or 0.0)
        )

    closed = [value for value in per_position.values()]
    wins = [value for value in closed if value > 0]
    return {
        "events": len(events),
        "counts": dict(sorted(counts.items())),
        "blocked": dict(sorted(blocks.items())),
        "errors": dict(sorted(errors.items())),
        "orders_sent": counts.get("order_sent", 0),
        "signals": counts.get("signal", 0),
        "closed_positions": len(closed),
        "winning_positions": len(wins),
        "win_rate": (len(wins) / len(closed)) if closed else 0.0,
        "net_pnl": sum(closed),
        "best_position": max(closed) if closed else 0.0,
        "worst_position": min(closed) if closed else 0.0,
        "first_event": events[0].get("time") if events else None,
        "last_event": events[-1].get("time") if events else None,
    }


def format_report(summary: dict[str, Any]) -> str:
    lines = [
        f"Journal window : {summary['first_event']} -> {summary['last_event']}",
        f"Events         : {summary['events']}",
        f"Signals        : {summary['signals']}",
        f"Orders sent    : {summary['orders_sent']}",
        f"Closed trades  : {summary['closed_positions']}",
        f"Win rate       : {summary['win_rate'] * 100:.1f}%",
        f"Net P&L        : {summary['net_pnl']:.2f}",
        f"Best / worst   : {summary['best_position']:.2f} / {summary['worst_position']:.2f}",
    ]
    if summary["blocked"]:
        lines.append("Blocked entries:")
        lines.extend(f"  {reason}: {count}" for reason, count in summary["blocked"].items())
    if summary["errors"]:
        lines.append("Errors:")
        lines.extend(f"  {reason}: {count}" for reason, count in summary["errors"].items())
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise the demo bot journal")
    parser.add_argument("--dir", default=str(ROOT / "logs"))
    parser.add_argument("--json", action="store_true", help="print raw JSON")
    args = parser.parse_args(argv)

    summary = summarize(read_events(args.dir))
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(format_report(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
