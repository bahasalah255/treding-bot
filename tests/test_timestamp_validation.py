from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from data.timestamp_validation import (
    TimestampValidationError,
    detect_missing_candles,
    normalize_unix_timestamps,
    select_closed_m5_candles,
    validate_closed_candles,
)


UTC = timezone.utc
NOW = datetime(2026, 11, 13, 17, 0, tzinfo=UTC)


def _frame(times):
    return pd.DataFrame(
        {"timestamp": times, "open": 1.0, "high": 2.0, "low": 1.0, "close": 1.5}
    )


def _rates(times):
    return pd.DataFrame(
        {
            "time": [int(pd.Timestamp(value).timestamp()) for value in times],
            "open": 1.0,
            "high": 2.0,
            "low": 1.0,
            "close": 1.5,
            "tick_volume": 1,
            "spread": 1,
        }
    )


def test_unix_seconds_conversion():
    result = normalize_unix_timestamps([1794598200])
    assert str(result[0].tz) == "UTC"


def test_unix_milliseconds_conversion():
    result = normalize_unix_timestamps([1794598200000])
    assert result[0] == pd.Timestamp("2026-11-13T19:30:00Z")


def test_utc_normalization():
    result = validate_closed_candles(
        _frame(["2026-11-13T16:50:00+01:00"]), NOW
    )
    assert str(result.index.tz) == "UTC"
    assert result.index[0] == pd.Timestamp("2026-11-13T15:50:00Z")


def test_broker_offset_is_normalized_without_fixed_offset():
    result = validate_closed_candles(
        _frame(["2026-11-13T18:55:00+02:00"]), NOW
    )
    assert result.index[0] == pd.Timestamp("2026-11-13T16:55:00Z")


def test_naive_timestamp_is_rejected():
    with pytest.raises(TimestampValidationError, match="not naive"):
        validate_closed_candles(_frame(["2026-11-13T16:50:00"]), NOW)


def test_open_candle_is_rejected():
    with pytest.raises(TimestampValidationError, match="incomplete"):
        validate_closed_candles(_frame(["2026-11-13T16:58:00Z"]), NOW)


def test_latest_closed_candle_selection_boundary():
    result = validate_closed_candles(_frame(["2026-11-13T16:55:00Z"]), NOW)
    assert result.index[-1] + timedelta(minutes=5) == pd.Timestamp(NOW)


def test_m5_boundary_1710_closes_at_1715():
    result = validate_closed_candles(
        _frame(["2026-11-13T17:10:00Z"]),
        datetime(2026, 11, 13, 17, 15, tzinfo=UTC),
    )
    assert result.index[-1] == pd.Timestamp("2026-11-13T17:10:00Z")


def test_m5_boundary_1715_is_forming_at_1715():
    with pytest.raises(TimestampValidationError, match="incomplete"):
        validate_closed_candles(
            _frame(["2026-11-13T17:15:00Z"]),
            datetime(2026, 11, 13, 17, 15, tzinfo=UTC),
        )


def test_m5_boundary_1720_is_future_at_1715():
    with pytest.raises(TimestampValidationError, match="future"):
        validate_closed_candles(
            _frame(["2026-11-13T17:20:00Z"]),
            datetime(2026, 11, 13, 17, 15, tzinfo=UTC),
        )


def test_closed_selector_at_1713_excludes_current_1710():
    result = select_closed_m5_candles(
        _rates(["2026-11-13T17:05:00Z", "2026-11-13T17:10:00Z"]),
        datetime(2026, 11, 13, 17, 13, tzinfo=UTC),
        minimum_count=1,
    )
    assert result.index[-1] == pd.Timestamp("2026-11-13T17:05:00Z")


def test_closed_selector_at_1715_accepts_1710_boundary():
    result = select_closed_m5_candles(
        _rates(["2026-11-13T17:05:00Z", "2026-11-13T17:10:00Z"]),
        datetime(2026, 11, 13, 17, 15, tzinfo=UTC),
        minimum_count=1,
    )
    assert result.index[-1] == pd.Timestamp("2026-11-13T17:10:00Z")


def test_closed_selector_at_1720_accepts_1715():
    result = select_closed_m5_candles(
        _rates(["2026-11-13T17:10:00Z", "2026-11-13T17:15:00Z"]),
        datetime(2026, 11, 13, 17, 20, tzinfo=UTC),
        minimum_count=1,
    )
    assert result.index[-1] == pd.Timestamp("2026-11-13T17:15:00Z")


def test_future_candle_is_rejected():
    with pytest.raises(TimestampValidationError, match="future"):
        validate_closed_candles(_frame(["2026-11-13T17:01:00Z"]), NOW)


def test_duplicate_timestamp_is_rejected():
    with pytest.raises(TimestampValidationError, match="duplicate"):
        validate_closed_candles(
            _frame(["2026-11-13T16:45:00Z", "2026-11-13T16:45:00Z"]), NOW
        )


def test_out_of_order_timestamp_is_rejected():
    with pytest.raises(TimestampValidationError, match="out of order"):
        validate_closed_candles(
            _frame(["2026-11-13T16:50:00Z", "2026-11-13T16:45:00Z"]), NOW
        )


def test_missing_candle_is_reported_without_rejection():
    frame = validate_closed_candles(
        _frame(["2026-11-13T16:40:00Z", "2026-11-13T16:50:00Z"]), NOW
    )
    assert detect_missing_candles(frame) == [
        (pd.Timestamp("2026-11-13T16:40:00Z"), pd.Timestamp("2026-11-13T16:50:00Z"))
    ]
