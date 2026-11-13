"""Small, conservative OHLC backtesting engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Trade:
    entry_time: str
    exit_time: str
    side: str
    entry: float
    stop_loss: float
    take_profit: float
    exit: float
    result_r: float


def run_backtest(
    data: pd.DataFrame,
    risk_percent: float = 0.5,
    reward_risk: float = 2.0,
    stop_distance_atr: float = 1.5,
    atr_period: int = 14,
) -> dict[str, Any]:
    """Simulate one-position trades and return R-based performance metrics."""
    if not 0 < risk_percent <= 100:
        raise ValueError("risk_percent must be between 0 and 100")
    if reward_risk <= 0 or stop_distance_atr <= 0:
        raise ValueError("reward_risk and stop_distance_atr must be positive")
    required = {"open", "high", "low", "close", "signal"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"missing backtest columns: {', '.join(sorted(missing))}")

    frame = data.sort_index().copy()
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["atr"] = true_range.rolling(atr_period).mean()
    trades: list[Trade] = []
    position: dict[str, Any] | None = None

    for index in range(atr_period + 1, len(frame)):
        row = frame.iloc[index]
        timestamp = frame.index[index]
        if position is not None:
            exit_price = None
            if position["side"] == "BUY":
                if row["low"] <= position["stop_loss"]:
                    exit_price = position["stop_loss"]
                elif row["high"] >= position["take_profit"]:
                    exit_price = position["take_profit"]
            else:
                if row["high"] >= position["stop_loss"]:
                    exit_price = position["stop_loss"]
                elif row["low"] <= position["take_profit"]:
                    exit_price = position["take_profit"]
            if exit_price is not None:
                direction = 1 if position["side"] == "BUY" else -1
                result_r = direction * (exit_price - position["entry"]) / position["risk"]
                trades.append(
                    Trade(
                        entry_time=position["entry_time"],
                        exit_time=timestamp.isoformat(),
                        side=position["side"],
                        entry=position["entry"],
                        stop_loss=position["stop_loss"],
                        take_profit=position["take_profit"],
                        exit=exit_price,
                        result_r=result_r,
                    )
                )
                position = None

        previous_signal = frame["signal"].iloc[index - 1]
        if position is None and previous_signal in (1, -1) and pd.notna(row["atr"]):
            entry = float(row["open"])
            risk = float(row["atr"]) * stop_distance_atr
            side = "BUY" if previous_signal == 1 else "SELL"
            if side == "BUY":
                stop_loss = entry - risk
                take_profit = entry + risk * reward_risk
            else:
                stop_loss = entry + risk
                take_profit = entry - risk * reward_risk
            position = {
                "entry_time": timestamp.isoformat(),
                "side": side,
                "entry": entry,
                "risk": risk,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            }

    results = [trade.result_r for trade in trades]
    equity_r = pd.Series(results, dtype=float).cumsum() if results else pd.Series(dtype=float)
    peak = equity_r.cummax() if not equity_r.empty else equity_r
    drawdown = (equity_r - peak).min() if not equity_r.empty else 0.0
    return {
        "trades": [asdict(trade) for trade in trades],
        "trade_count": len(trades),
        "win_rate": sum(result > 0 for result in results) / len(results) if results else 0.0,
        "net_r": sum(results),
        "max_drawdown_r": abs(float(drawdown)),
        "risk_percent": risk_percent,
        "reward_risk": reward_risk,
    }
