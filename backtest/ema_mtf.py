"""Multi-timeframe EMA crossover signals."""

from __future__ import annotations

import pandas as pd


def build_signals(
    m5: pd.DataFrame,
    m15: pd.DataFrame,
    fast_period: int = 9,
    slow_period: int = 21,
    trend_period: int = 50,
) -> pd.DataFrame:
    """Return M5 bars with non-lookahead EMA crossover signals."""
    if fast_period >= slow_period:
        raise ValueError("fast_period must be smaller than slow_period")
    if m5.empty or m15.empty:
        raise ValueError("M5 and M15 data must not be empty")

    m5 = m5.sort_index().copy()
    m15 = m15.sort_index().copy()
    m5.index.name = "timestamp"
    m15.index.name = "timestamp"
    m5["fast_ema"] = m5["close"].ewm(span=fast_period, adjust=False).mean()
    m5["slow_ema"] = m5["close"].ewm(span=slow_period, adjust=False).mean()
    m15["trend_ema"] = m15["close"].ewm(span=trend_period, adjust=False).mean()
    trend = m15[["close", "trend_ema"]].rename(
        columns={"close": "m15_close", "trend_ema": "m15_trend_ema"}
    )
    result = pd.merge_asof(
        m5.reset_index().sort_values("timestamp"),
        trend.reset_index().sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    ).set_index("timestamp")

    crossed_up = (result["fast_ema"] > result["slow_ema"]) & (
        result["fast_ema"].shift(1) <= result["slow_ema"].shift(1)
    )
    crossed_down = (result["fast_ema"] < result["slow_ema"]) & (
        result["fast_ema"].shift(1) >= result["slow_ema"].shift(1)
    )
    result["signal"] = 0
    result.loc[crossed_up & (result["m15_close"] > result["m15_trend_ema"]), "signal"] = 1
    result.loc[crossed_down & (result["m15_close"] < result["m15_trend_ema"]), "signal"] = -1
    return result
