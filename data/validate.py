"""Validation for normalized historical OHLCV data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "spread",
)
PRICE_COLUMNS = ("open", "high", "low", "close")


class DataValidationError(ValueError):
    """Raised when historical market data is not suitable for backtesting."""


def load_and_validate_csv(path: str | Path) -> pd.DataFrame:
    """Load a normalized CSV and return a sorted UTC-indexed DataFrame."""
    frame = pd.read_csv(path)
    validate_frame(frame)

    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if frame["timestamp"].isna().any():
        raise DataValidationError("timestamp contains invalid values")

    frame = frame.sort_values("timestamp").set_index("timestamp")
    return frame


def validate_frame(frame: pd.DataFrame) -> None:
    """Validate schema and basic OHLCV invariants."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DataValidationError(f"missing required columns: {', '.join(missing)}")

    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise DataValidationError("timestamp contains invalid values")
    if timestamps.duplicated().any():
        raise DataValidationError("timestamp contains duplicates")

    for column in (*PRICE_COLUMNS, "tick_volume", "spread"):
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise DataValidationError(f"{column} must be numeric")
        if frame[column].isna().any():
            raise DataValidationError(f"{column} contains missing values")

    if (frame[list(PRICE_COLUMNS)] <= 0).any().any():
        raise DataValidationError("prices must be positive")
    if (frame["tick_volume"] < 0).any() or (frame["spread"] < 0).any():
        raise DataValidationError("tick_volume and spread cannot be negative")

    high_valid = frame["high"] >= frame[list(("open", "close"))].max(axis=1)
    low_valid = frame["low"] <= frame[list(("open", "close"))].min(axis=1)
    if not (high_valid & low_valid).all():
        raise DataValidationError("OHLC values violate high/low invariants")
