"""Small, conservative OHLC backtesting engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
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
    cost_r: float
    exit_reason: str
    balance_after: float


class _Portfolio:
    """Balance tracking with optional daily-loss and total-drawdown limits."""

    def __init__(
        self,
        initial_balance: float,
        risk_fraction: float,
        max_daily_loss: float | None,
        max_total_drawdown: float | None,
    ) -> None:
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.peak = initial_balance
        self.risk_fraction = risk_fraction
        self.max_daily_loss = max_daily_loss
        self.max_total_drawdown = max_total_drawdown
        self.day: date | None = None
        self.day_start = initial_balance
        self.halted_reason: str | None = None

    def roll_day(self, timestamp: pd.Timestamp) -> None:
        """Reset the daily baseline when the UTC date changes."""
        current = timestamp.date()
        if current != self.day:
            self.day = current
            self.day_start = self.balance

    def block_reason(self) -> str | None:
        """Return why a new position may not be opened, if it may not."""
        if self.halted_reason is not None:
            return self.halted_reason
        if (
            self.max_total_drawdown is not None
            and self.peak - self.balance >= self.peak * self.max_total_drawdown
        ):
            self.halted_reason = "max_total_drawdown"
            return self.halted_reason
        if (
            self.max_daily_loss is not None
            and self.day_start - self.balance >= self.day_start * self.max_daily_loss
        ):
            return "max_daily_loss"
        return None

    def settle(self, result_r: float, risk_amount: float) -> float:
        """Apply a closed trade and return the new balance."""
        self.balance += result_r * risk_amount
        self.peak = max(self.peak, self.balance)
        return self.balance


def _exit_level(position: dict[str, Any], row: pd.Series) -> tuple[float, str] | None:
    """Return the first exit fill and reason inside this bar, or ``None``.

    A bar that opens beyond a level fills at the open, as a real stop or limit
    order would. When both levels are touched inside one bar and the open sits
    between them, the stop is assumed to fill first.
    """
    open_price = float(row["open"])
    stop = position["stop_loss"]
    target = position["take_profit"]
    if position["side"] == "BUY":
        if open_price <= stop:
            return open_price, "stop_loss_gap"
        if open_price >= target:
            return open_price, "take_profit_gap"
        if float(row["low"]) <= stop:
            return stop, "stop_loss"
        if float(row["high"]) >= target:
            return target, "take_profit"
    else:
        if open_price >= stop:
            return open_price, "stop_loss_gap"
        if open_price <= target:
            return open_price, "take_profit_gap"
        if float(row["high"]) >= stop:
            return stop, "stop_loss"
        if float(row["low"]) <= target:
            return target, "take_profit"
    return None


def run_backtest(
    data: pd.DataFrame,
    risk_percent: float = 0.5,
    reward_risk: float = 2.0,
    stop_distance_atr: float = 1.5,
    atr_period: int = 14,
    *,
    initial_balance: float = 1000.0,
    spread_price: float = 0.0,
    slippage_price: float = 0.0,
    commission_price: float = 0.0,
    max_daily_loss: float | None = None,
    max_total_drawdown: float | None = None,
) -> dict[str, Any]:
    """Simulate one-position trades and return R-based and balance-based metrics.

    Costs are expressed in price units because the contract specifications that
    turn points, lots, and currency into price are broker-specific. ``spread``
    is charged once per round turn, ``slippage`` once per side, and
    ``commission`` once per round turn. All of them are deducted from each
    trade's result in R, using the trade's own risk distance.

    ``max_daily_loss`` and ``max_total_drawdown`` are fractions of balance
    (``0.02`` is 2%). The daily limit blocks entries for the rest of the UTC
    day; the total limit stops the run permanently.
    """
    if not 0 < risk_percent <= 100:
        raise ValueError("risk_percent must be between 0 and 100")
    if reward_risk <= 0 or stop_distance_atr <= 0:
        raise ValueError("reward_risk and stop_distance_atr must be positive")
    if atr_period < 1:
        raise ValueError("atr_period must be at least 1")
    if initial_balance <= 0:
        raise ValueError("initial_balance must be positive")
    for name, value in (
        ("spread_price", spread_price),
        ("slippage_price", slippage_price),
        ("commission_price", commission_price),
    ):
        if value < 0:
            raise ValueError(f"{name} cannot be negative")
    for name, value in (
        ("max_daily_loss", max_daily_loss),
        ("max_total_drawdown", max_total_drawdown),
    ):
        if value is not None and not 0 < value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")

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

    cost_price = spread_price + 2 * slippage_price + commission_price
    portfolio = _Portfolio(
        initial_balance, risk_percent / 100, max_daily_loss, max_total_drawdown
    )
    trades: list[Trade] = []
    position: dict[str, Any] | None = None
    skipped_entries = 0

    for index in range(atr_period + 1, len(frame)):
        row = frame.iloc[index]
        timestamp = frame.index[index]
        portfolio.roll_day(timestamp)
        exited_this_bar = False

        if position is not None:
            level = _exit_level(position, row)
            if level is not None:
                exit_price, reason = level
                direction = 1 if position["side"] == "BUY" else -1
                gross_r = direction * (exit_price - position["entry"]) / position["risk"]
                cost_r = cost_price / position["risk"]
                result_r = gross_r - cost_r
                balance = portfolio.settle(result_r, position["risk_amount"])
                portfolio.block_reason()
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
                        cost_r=cost_r,
                        exit_reason=reason,
                        balance_after=balance,
                    )
                )
                position = None
                exited_this_bar = True

        if position is not None or exited_this_bar:
            continue

        previous_signal = frame["signal"].iloc[index - 1]
        # The stop distance must come from the last *closed* bar, otherwise the
        # bar that fills the entry leaks its own range into position sizing.
        atr = frame["atr"].iloc[index - 1]
        if previous_signal not in (1, -1) or pd.isna(atr) or atr <= 0:
            continue
        if portfolio.block_reason() is not None:
            skipped_entries += 1
            continue

        entry = float(row["open"])
        risk = float(atr) * stop_distance_atr
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
            "risk_amount": portfolio.balance * portfolio.risk_fraction,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

    if position is not None:
        exit_price = float(frame["close"].iloc[-1])
        direction = 1 if position["side"] == "BUY" else -1
        gross_r = direction * (exit_price - position["entry"]) / position["risk"]
        cost_r = cost_price / position["risk"]
        result_r = gross_r - cost_r
        balance = portfolio.settle(result_r, position["risk_amount"])
        portfolio.block_reason()
        trades.append(
            Trade(
                entry_time=position["entry_time"],
                exit_time=frame.index[-1].isoformat(),
                side=position["side"],
                entry=position["entry"],
                stop_loss=position["stop_loss"],
                take_profit=position["take_profit"],
                exit=exit_price,
                result_r=result_r,
                cost_r=cost_r,
                exit_reason="end_of_data",
                balance_after=balance,
            )
        )

    results = [trade.result_r for trade in trades]
    equity_r = pd.Series(results, dtype=float).cumsum() if results else pd.Series(dtype=float)
    peak_r = equity_r.cummax() if not equity_r.empty else equity_r
    drawdown_r = float((equity_r - peak_r).min()) if not equity_r.empty else 0.0

    balances = pd.Series(
        [initial_balance, *(trade.balance_after for trade in trades)], dtype=float
    )
    running_peak = balances.cummax()
    drawdown_percent = float(((balances - running_peak) / running_peak).min())

    exit_reasons: dict[str, int] = {}
    for trade in trades:
        exit_reasons[trade.exit_reason] = exit_reasons.get(trade.exit_reason, 0) + 1

    final_balance = float(balances.iloc[-1])
    return {
        "trades": [asdict(trade) for trade in trades],
        "trade_count": len(trades),
        "win_rate": sum(result > 0 for result in results) / len(results) if results else 0.0,
        "net_r": sum(results),
        "gross_r": sum(result + trade.cost_r for result, trade in zip(results, trades)),
        "cost_r_total": sum(trade.cost_r for trade in trades),
        "max_drawdown_r": abs(drawdown_r),
        "risk_percent": risk_percent,
        "reward_risk": reward_risk,
        "initial_balance": initial_balance,
        "final_balance": final_balance,
        "net_profit": final_balance - initial_balance,
        "return_percent": (final_balance / initial_balance - 1) * 100,
        "max_drawdown_percent": abs(drawdown_percent) * 100,
        "exit_reasons": exit_reasons,
        "skipped_entries": skipped_entries,
        "halted_reason": portfolio.halted_reason,
    }
