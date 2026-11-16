"""Multi-timeframe EMA crossover signals."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

_M5_STEP = timedelta(minutes=5)
_M15_STEP = timedelta(minutes=15)


def _bar_step(index: pd.DatetimeIndex, fallback: timedelta) -> timedelta:
    """Infer how long one bar lasts from an open-time index."""
    if len(index) < 2:
        return fallback
    step = pd.Series(index).diff().median()
    if pd.isna(step) or step <= timedelta(0):
        return fallback
    return step


def build_signals(
    m5: pd.DataFrame,
    m15: pd.DataFrame,
    fast_period: int = 9,
    slow_period: int = 21,
    trend_period: int = 50,
) -> pd.DataFrame:
    """Return M5 bars with non-lookahead EMA crossover signals.

    Bars are timestamped by their open time, the way MT5 exports them. A bar
    that opens at ``T`` is not complete until ``T`` plus its duration, so the
    M15 trend filter is aligned on *close* times: an M5 bar only ever sees M15
    bars that had already closed when that M5 bar closed. Signals are therefore
    safe to act on at the open of the following M5 bar.
    """
    if fast_period >= slow_period:
        raise ValueError("fast_period must be smaller than slow_period")
    if m5.empty or m15.empty:
        raise ValueError("M5 and M15 data must not be empty")

    m5 = m5.sort_index().copy()
    m15 = m15.sort_index().copy()
    m5.index.name = "timestamp"
    m15.index.name = "timestamp"

    m15["trend_ema"] = m15["close"].ewm(span=trend_period, adjust=False).mean()

    m5_closed_at = pd.DataFrame(
        {"timestamp": m5.index, "closed_at": m5.index + _bar_step(m5.index, _M5_STEP)}
    )
    m15_closed_at = pd.DataFrame(
        {
            "closed_at": m15.index + _bar_step(m15.index, _M15_STEP),
            "m15_close": m15["close"].to_numpy(),
            "m15_trend_ema": m15["trend_ema"].to_numpy(),
        }
    ).sort_values("closed_at")

    trend = pd.merge_asof(
        m5_closed_at.sort_values("closed_at"),
        m15_closed_at,
        on="closed_at",
        direction="backward",
    ).set_index("timestamp")
    trend = trend.sort_index()[["m15_close", "m15_trend_ema"]]

    result = m5.join(trend)
    result["fast_ema"] = m5["close"].ewm(span=fast_period, adjust=False).mean()
    result["slow_ema"] = m5["close"].ewm(span=slow_period, adjust=False).mean()

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
