import pandas as pd
import pytest

from backtest.engine import run_backtest
from backtest.ema_mtf import build_signals


def _flat_bars(periods: int = 40, price: float = 100.0) -> pd.DataFrame:
    """Bars with a 1.0-wide range, so an ATR of 14 bars equals 1.0."""
    index = pd.date_range("2026-01-01", periods=periods, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": price,
            "high": price + 0.5,
            "low": price - 0.5,
            "close": price,
            "tick_volume": 1,
            "spread": 1,
            "signal": 0,
        },
        index=index,
    )


def _signal(frame: pd.DataFrame, position: int, value: int) -> None:
    frame.at[frame.index[position], "signal"] = value


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


def test_m15_trend_is_only_visible_after_its_bar_closes():
    index5 = pd.date_range("2026-01-01", periods=12, freq="5min", tz="UTC")
    m5 = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}, index=index5)
    index15 = pd.date_range("2026-01-01", periods=4, freq="15min", tz="UTC")
    # The 10.0 close belongs to the M15 bar that opens at 00:15 and closes at
    # 00:30. MT5 timestamps bars by open time, so a backward asof join on open
    # times would leak it into M5 bars at 00:15, 00:20 and 00:25.
    m15 = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": [1.0, 10.0, 1.0, 1.0]},
        index=index15,
    )
    result = build_signals(m5, m15, fast_period=3, slow_period=5, trend_period=2)

    before_close = result.loc[: pd.Timestamp("2026-01-01 00:20", tz="UTC"), "m15_close"]
    after_close = result.loc[
        pd.Timestamp("2026-01-01 00:25", tz="UTC") : pd.Timestamp("2026-01-01 00:35", tz="UTC"),
        "m15_close",
    ]
    # The first two bars close before any M15 bar has closed: no trend yet.
    assert before_close.isna().sum() == 2
    assert (before_close.dropna() == 1.0).all()
    assert (after_close == 10.0).all()
    assert result.loc[pd.Timestamp("2026-01-01 00:40", tz="UTC"), "m15_close"] == 1.0


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


def test_stop_distance_uses_the_last_closed_atr():
    data = _flat_bars(periods=30)
    _signal(data, 14, 1)
    # The entry bar has an enormous range. Sizing the stop from it would widen
    # the stop to ~5.68 instead of the 1.5 the closed ATR implies.
    data.iloc[15, [1, 2, 3]] = [120.0, 80.0, 100.0]
    data.loc[data.index[16] :, "high"] = 106.0

    report = run_backtest(data, stop_distance_atr=1.5)

    assert report["trades"][0]["stop_loss"] == pytest.approx(98.5)


def test_costs_are_deducted_from_each_trade():
    data = _flat_bars(periods=30)
    _signal(data, 14, 1)
    data.loc[data.index[16] :, ["high", "close"]] = 104.0

    free = run_backtest(data, stop_distance_atr=1.5, reward_risk=2.0)
    paid = run_backtest(
        data,
        stop_distance_atr=1.5,
        reward_risk=2.0,
        spread_price=0.15,
        slippage_price=0.05,
        commission_price=0.05,
    )

    assert free["gross_r"] == pytest.approx(2.0)
    assert free["net_r"] == pytest.approx(2.0)
    assert paid["gross_r"] == pytest.approx(2.0)
    # One spread + two slippage sides + commission = 0.30 price over 1.5 risk.
    assert paid["net_r"] == pytest.approx(2.0 - 0.30 / 1.5)
    assert paid["cost_r_total"] == pytest.approx(0.30 / 1.5)


def test_engine_rejects_negative_costs():
    with pytest.raises(ValueError):
        run_backtest(_flat_bars(periods=20), spread_price=-0.1)


def test_open_position_is_closed_at_the_last_bar():
    data = _flat_bars(periods=30)
    _signal(data, 14, 1)

    report = run_backtest(data, stop_distance_atr=1.5)

    assert report["trade_count"] == 1
    assert report["trades"][0]["exit_reason"] == "end_of_data"
    assert report["trades"][0]["exit"] == pytest.approx(100.0)


def _loss_then_second_signal() -> pd.DataFrame:
    """One stopped-out trade at bar 16, plus a fresh signal at bar 20."""
    data = _flat_bars(periods=40)
    _signal(data, 14, 1)
    data.at[data.index[16], "low"] = 98.0
    _signal(data, 20, 1)
    return data


def test_daily_loss_limit_blocks_later_entries_on_the_same_day():
    report = run_backtest(
        _loss_then_second_signal(), risk_percent=1.0, max_daily_loss=0.005
    )

    assert report["trade_count"] == 1
    assert report["trades"][0]["result_r"] == pytest.approx(-1.0)
    assert report["skipped_entries"] == 1
    assert report["halted_reason"] is None


def test_total_drawdown_limit_halts_trading():
    report = run_backtest(
        _loss_then_second_signal(), risk_percent=1.0, max_total_drawdown=0.005
    )

    assert report["trade_count"] == 1
    assert report["skipped_entries"] == 1
    assert report["halted_reason"] == "max_total_drawdown"
    assert report["final_balance"] == pytest.approx(990.0)


def test_balance_compounds_with_realised_risk():
    data = _flat_bars(periods=30)
    _signal(data, 14, 1)
    data.loc[data.index[16] :, ["high", "close"]] = 104.0
    report = run_backtest(data, risk_percent=1.0, stop_distance_atr=1.5, reward_risk=2.0)

    assert report["net_profit"] == pytest.approx(20.0)
    assert report["final_balance"] == pytest.approx(1020.0)
    assert report["return_percent"] == pytest.approx(2.0)
