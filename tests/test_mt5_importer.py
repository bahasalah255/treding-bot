from types import SimpleNamespace

import numpy as np
import pytest

from data.mt5_importer import MT5ImportError, rates_to_frame, select_gold_symbol


def test_select_gold_symbol_requires_disambiguation():
    symbols = [SimpleNamespace(name="XAUUSD"), SimpleNamespace(name="XAUUSDm")]

    with pytest.raises(MT5ImportError, match="ambiguous"):
        select_gold_symbol(symbols)


def test_select_gold_symbol_accepts_exact_market_watch_name():
    symbols = [SimpleNamespace(name="XAUUSDm")]

    assert select_gold_symbol(symbols, "XAUUSDm") == "XAUUSDm"


def test_rates_are_normalized_to_utc():
    rates = np.array(
        [(1735725600, 2650.0, 2652.0, 2649.0, 2651.0, 100, 20, 100)],
        dtype=[
            ("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"),
            ("close", "f8"), ("tick_volume", "i8"), ("spread", "i8"),
            ("real_volume", "i8"),
        ],
    )

    frame = rates_to_frame(rates)

    assert str(frame.loc[0, "timestamp"].tz) == "UTC"
    assert list(frame.columns) == [
        "timestamp", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"
    ]
