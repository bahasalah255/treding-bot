import pandas as pd
import pytest

from data.validate import DataValidationError, load_and_validate_csv, validate_frame


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ["2025-01-01T10:00:00Z", "2025-01-01T10:05:00Z"],
            "open": [2650.0, 2651.0],
            "high": [2652.0, 2653.0],
            "low": [2649.0, 2650.0],
            "close": [2651.0, 2652.0],
            "tick_volume": [100, 120],
            "spread": [20, 20],
        }
    )


def test_valid_csv_is_sorted_and_indexed_in_utc(tmp_path):
    frame = valid_frame().iloc[::-1]
    path = tmp_path / "xauusd.csv"
    frame.to_csv(path, index=False)

    result = load_and_validate_csv(path)

    assert result.index.tz is not None
    assert result.index.is_monotonic_increasing
    assert str(result.index.tz) == "UTC"


def test_missing_column_is_rejected():
    frame = valid_frame().drop(columns="spread")

    with pytest.raises(DataValidationError, match="missing required columns"):
        validate_frame(frame)


def test_invalid_ohlc_relationship_is_rejected():
    frame = valid_frame()
    frame.loc[0, "high"] = frame.loc[0, "close"] - 1

    with pytest.raises(DataValidationError, match="OHLC"):
        validate_frame(frame)


def test_duplicate_timestamps_are_rejected():
    frame = valid_frame()
    frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]

    with pytest.raises(DataValidationError, match="duplicates"):
        validate_frame(frame)
