import pandas as pd

from backtest.engine import run_backtest
from backtest.ema_mtf import build_signals


def test_ema_signals_have_no_future_rows():
    index = pd.date_range("2026-01-01", periods=80, freq="5min", tz="UTC")
    m5 = pd.DataFrame(
        {"open": range(100, 180), "high": range(101, 181), "low": range(99, 179),
         "close": range(100, 180), "tick_volume": 1, "spread": 1},
        index=index,
    )
    m15 = m5.resample("15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )
    result = build_signals(m5, m15, fast_period=3, slow_period=5, trend_period=3)
    assert len(result) == len(m5)
    assert result.index.max() == index.max()


def test_backtest_closes_buy_at_take_profit():
    index = pd.date_range("2026-01-01", periods=18, freq="5min", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0] * 14 + [100.0, 100.0, 100.0, 103.0],
            "high": [101.0] * 14 + [101.0, 101.0, 104.1, 104.1],
            "low": [99.0] * 14 + [99.0, 99.0, 99.0, 103.0],
            "close": [100.0] * 14 + [100.0, 100.0, 103.0, 103.0],
            "signal": [0] * 14 + [1, 0, 0, 0],
        },
        index=index,
    )
    report = run_backtest(data, reward_risk=2, stop_distance_atr=1)
    assert report["trade_count"] == 1
    assert report["net_r"] == 2
