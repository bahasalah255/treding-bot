import csv

from trading.monitor_log import LOG_FIELDS, append_monitor_log


def test_append_monitor_log_creates_header_and_serializes_values(tmp_path):
    path = tmp_path / "monitoring.csv"

    append_monitor_log(
        path,
        symbol="XAUUSD",
        event="signal",
        status="COMPLETE",
        values={
            "action": "NO_TRADE",
            "rejection_reasons": ("RR below minimum", "range middle"),
            "risk_reward": 0.38,
        },
    )
    append_monitor_log(
        path,
        symbol="XAUUSD",
        event="heartbeat",
        status="NO_NEW_CANDLE",
    )

    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert list(rows[0]) == list(LOG_FIELDS)
    assert rows[0]["action"] == "NO_TRADE"
    assert rows[0]["rejection_reasons"] == "RR below minimum | range middle"
    assert rows[0]["risk_reward"] == "0.38"
    assert rows[1]["event"] == "heartbeat"
