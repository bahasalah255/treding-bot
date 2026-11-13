"""Append-only CSV logging for monitor diagnostics and signal decisions."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOG_FIELDS = (
    "event_time_utc",
    "symbol",
    "timeframe",
    "event",
    "status",
    "current_utc",
    "candle_open",
    "candle_close",
    "provider_timestamp",
    "mt5_start_pos",
    "raw_rates_count",
    "filtered_rates_count",
    "latest_closed_candle",
    "latest_closed_close_time",
    "action",
    "entry_price",
    "stop_loss",
    "take_profit",
    "risk_reward",
    "market_trend",
    "market_condition",
    "buy_setup_score",
    "sell_setup_score",
    "local_buy_score",
    "local_sell_score",
    "local_score_breakdown",
    "rationale",
    "rejection_reasons",
    "execution_status",
    "error",
)


def append_monitor_log(
    path: str | Path,
    *,
    symbol: str,
    event: str,
    status: str,
    values: dict[str, Any] | None = None,
) -> None:
    """Append one structured monitor event without affecting trading decisions."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size:
        with target.open("r", newline="", encoding="utf-8-sig") as handle:
            existing_header = next(csv.reader(handle), [])
        if existing_header != list(LOG_FIELDS):
            target = target.with_name(f"{target.stem}_v2{target.suffix}")
    row = {field: "" for field in LOG_FIELDS}
    row.update(values or {})
    row["event_time_utc"] = datetime.now(timezone.utc).isoformat()
    row["symbol"] = symbol
    row["timeframe"] = "M5"
    row["event"] = event
    row["status"] = status
    with target.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS, extrasaction="ignore")
        if target.stat().st_size == 0:
            writer.writeheader()
        writer.writerow({field: _csv_value(row[field]) for field in LOG_FIELDS})


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (tuple, list)):
        return " | ".join(str(item) for item in value)
    return str(value)
