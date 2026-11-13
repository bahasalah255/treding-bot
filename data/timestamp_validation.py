"""UTC timestamp normalization and closed-candle validation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd


class TimestampValidationError(ValueError):
    """Raised when candle timestamps cannot be trusted for signal generation."""


def normalize_unix_timestamps(values: object) -> pd.DatetimeIndex:
    """Convert homogeneous Unix seconds or milliseconds to an aware UTC index."""
    numeric = pd.to_numeric(pd.Index(values), errors="coerce")
    if numeric.isna().any():
        raise TimestampValidationError("timestamps contain non-numeric values")
    absolute = pd.Series(numeric).abs()
    if len(absolute) == 0:
        raise TimestampValidationError("timestamps are empty")
    units = "ms" if float(absolute.max()) >= 100_000_000_000 else "s"
    timestamps = pd.to_datetime(numeric, unit=units, utc=True, errors="coerce")
    if timestamps.isna().any():
        raise TimestampValidationError(f"invalid Unix {units} timestamps")
    return pd.DatetimeIndex(timestamps)


def validate_closed_candles(
    frame: pd.DataFrame,
    current_utc: datetime,
    timeframe_minutes: int = 5,
) -> pd.DataFrame:
    """Validate ordering, uniqueness, UTC, and candle closure without filtering."""
    if current_utc.tzinfo is None:
        raise TimestampValidationError("current_utc must be timezone-aware")
    current_utc = current_utc.astimezone(timezone.utc)
    if "timestamp" in frame.columns:
        parsed_without_conversion = pd.to_datetime(frame["timestamp"], errors="coerce")
        if getattr(parsed_without_conversion.dt, "tz", None) is None:
            raise TimestampValidationError("candle timestamps must be timezone-aware, not naive")
        timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        result = frame.copy()
        result["timestamp"] = timestamps
        result = result.set_index("timestamp")
    else:
        result = frame.copy()
        if not isinstance(result.index, pd.DatetimeIndex) or result.index.tz is None:
            raise TimestampValidationError("candle timestamps must be timezone-aware")
        result.index = pd.to_datetime(result.index, utc=True)
    if result.index.isna().any():
        raise TimestampValidationError("candle timestamps contain invalid values")
    if result.index.has_duplicates:
        raise TimestampValidationError("duplicate candle timestamps")
    if not result.index.is_monotonic_increasing:
        raise TimestampValidationError("candle timestamps are out of order")
    close_times = result.index + timedelta(minutes=timeframe_minutes)
    future = close_times > current_utc
    if future.any():
        position = future.argmax()
        candle_open = result.index[position]
        candle_close = close_times[position]
        delta = candle_close - pd.Timestamp(current_utc)
        raise TimestampValidationError(
            "incomplete or future candle detected: "
            f"candle_open={candle_open.isoformat()} "
            f"candle_close={candle_close.isoformat()} "
            f"current_utc={current_utc.isoformat()} "
            f"delta={delta}"
        )
    result.index.name = "timestamp"
    return result


def select_closed_m5_candles(
    rates: object,
    current_utc: datetime,
    minimum_count: int = 20,
) -> pd.DataFrame:
    """Normalize MT5 rates and pass only fully closed M5 candles downstream."""
    if current_utc.tzinfo is None:
        raise TimestampValidationError("current_utc must be timezone-aware")
    frame = pd.DataFrame(rates)
    if frame.empty or "time" not in frame.columns:
        raise TimestampValidationError("MT5 returned no usable candle data")
    raw_count = len(frame)
    raw_times = normalize_unix_timestamps(frame["time"])
    if raw_times.has_duplicates:
        raise TimestampValidationError("duplicate raw MT5 candle timestamps")
    if not raw_times.is_monotonic_increasing:
        raise TimestampValidationError("raw MT5 candle timestamps are out of order")
    frame["timestamp"] = raw_times
    close_times = raw_times + pd.Timedelta(5, unit="min")
    current_timestamp = pd.Timestamp(current_utc.astimezone(timezone.utc))
    closed_mask = close_times <= current_timestamp
    filtered = frame.loc[closed_mask].drop(columns=["time"])
    if len(filtered) < minimum_count:
        raise TimestampValidationError(
            f"insufficient closed M5 candles: raw_count={raw_count} "
            f"filtered_count={len(filtered)}"
        )
    result = validate_closed_candles(filtered, current_utc)
    result.attrs["raw_rates_count"] = raw_count
    result.attrs["filtered_rates_count"] = len(result)
    result.attrs["raw_latest_candle"] = raw_times[-1].isoformat()
    result.attrs["latest_closed_candle"] = result.index[-1].isoformat()
    result.attrs["latest_closed_close_time"] = (
        result.index[-1] + pd.Timedelta(5, unit="min")
    ).isoformat()
    result.attrs["current_utc"] = current_timestamp.isoformat()
    return result


def detect_missing_candles(
    frame: pd.DataFrame,
    timeframe_minutes: int = 5,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Report gaps without treating normal market closures as invalid candles."""
    if len(frame.index) < 2:
        return []
    expected = pd.Timedelta(timeframe_minutes, unit="min")
    timestamps = pd.DatetimeIndex(frame.index).sort_values()
    gaps: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for previous, current in zip(timestamps[:-1], timestamps[1:]):
        if current - previous > expected:
            gaps.append((previous, current))
    return gaps
